#include "oh-runtime.h"

#include "llama.h"
#include "llama-impl.h"
#include "llama-model.h"

#include "cpu-runtime.h"
#include "ggml-cuda/ssd/vram-slot-table.h"
#include "ggml-ssd/gpu/provider.h"
#include "ggml-ssd/gpu/transport.h"
#include "ggml-ssd/host/arbiter.h"
#if defined(_WIN32) || defined(__APPLE__)
#    include "ggml-ssd/host/ledger-native.h"
#elif defined(__linux__)
#    include "ggml-ssd/host/ledger.h"
#endif
#include "ggml-ssd/platform/file.h"
#include "ggml-ssd/runtime-services.h"
#include "ggml-ssd/settings/settings.h"
#include "ggml-ssd/store/source.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#if defined(_WIN32)
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

namespace oh_ssd {

namespace {

constexpr uint64_t MiB = 1024ull * 1024ull;

#if defined(_WIN32)
constexpr const char * PLATFORM = "windows";
#elif defined(__APPLE__)
constexpr const char * PLATFORM = "darwin";
#else
constexpr const char * PLATFORM = "linux";
#endif

// One arbiter per process: every context registers a client with it, so the
// contexts of one process share one host-memory envelope.
std::shared_ptr<ssd::host::process_arbiter> shared_arbiter() {
    static std::mutex mutex;
    static std::weak_ptr<ssd::host::process_arbiter> current;
    std::lock_guard<std::mutex> lock(mutex);
    auto owner = current.lock();
    if (!owner) {
        owner   = std::make_shared<ssd::host::process_arbiter>();
        current = owner;
    }
    return owner;
}

struct expert_range {
    int      fd;
    uint64_t offset;
    uint64_t nbytes;
    int64_t  n_expert;
};

std::string trim(const std::string & s) {
    const auto b = s.find_first_not_of(" \t\r");
    if (b == std::string::npos) {
        return {};
    }
    const auto e = s.find_last_not_of(" \t\r");
    return s.substr(b, e - b + 1);
}

uint64_t parse_mib(const std::string & key, const std::string & value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || value.size() > 12) {
        throw std::runtime_error("[oh-ssd] " + key + " needs a whole number of MiB, got '" + value + "'");
    }
    return std::stoull(value) * MiB;
}

int layer_of(const std::string & name) {
    // "blk.<N>.ffn_..._exps.weight"
    if (name.rfind("blk.", 0) != 0) {
        return -1;
    }
    const auto dot = name.find('.', 4);
    if (dot == std::string::npos || dot == 4) {
        return -1;
    }
    const std::string digits = name.substr(4, dot - 4);
    if (digits.find_first_not_of("0123456789") != std::string::npos || digits.size() > 6) {
        return -1;
    }
    return std::stoi(digits);
}

uint64_t mib_ceil(uint64_t bytes) {
    return (bytes + MiB - 1) / MiB;
}

} // namespace

class context_runtime {
  public:
    std::shared_ptr<ssd::host::process_arbiter>  arbiter;
    std::unique_ptr<ssd::host::context_client>   client;
    std::optional<ssd::settings::snapshot>       settings;
    ssd::runtime_services                        services;
    ssd::store::source_view                      source;
    std::vector<int>                             fds;
    std::unordered_map<std::string, expert_range> ranges;
    std::unique_ptr<llama_kimi_cpu_sink_runtime> sink;
    uint64_t                                     experts      = 0;
    uint64_t                                     ram_budget   = 0;
    uint64_t                                     cache_cap    = UINT64_MAX;
    uint64_t                                     tier_host    = 0; // host memory the VRAM tier takes at first decode
    std::unique_ptr<void, void (*)(void *) noexcept> gpu_transport{nullptr, [](void *) noexcept {}};

    ~context_runtime() {
        // The sink borrows the client, the settings, the services and the file
        // descriptors; it goes first. The CPU backend, which also borrows the
        // services, is already gone by now (llama_context declares this runtime
        // before its backends).
        sink.reset();
        gpu_transport.reset();
        if (client) {
            client->close();
        }
        client.reset();
        for (int fd : fds) {
            ssd::platform::close_fd(fd);
        }
    }

    static bool resolve(void * userdata, const char * name, int expert, size_t logical_nbytes, int64_t n_experts,
                        int * fd, uint64_t * offset, size_t * nbytes) noexcept {
        auto * self = static_cast<context_runtime *>(userdata);
        if (!self || !name || !fd || !offset || !nbytes) {
            return false;
        }
        const auto it = self->ranges.find(name);
        if (it == self->ranges.end()) {
            return false;
        }
        const auto & r = it->second;
        if (n_experts != r.n_expert || expert < 0 || expert >= r.n_expert || r.n_expert <= 0 ||
            r.nbytes % uint64_t(r.n_expert) != 0) {
            return false;
        }
        const uint64_t per = r.nbytes / uint64_t(r.n_expert);
        if (logical_nbytes != per) {
            return false;
        }
        *fd     = r.fd;
        *offset = r.offset + uint64_t(expert) * per;
        *nbytes = size_t(per);
        return true;
    }
};

void release_sink(context_runtime & runtime) noexcept {
    runtime.sink.reset();
}

void context_runtime_deleter::operator()(context_runtime * runtime) const noexcept {
    delete runtime;
}

context_runtime_ptr create(const llama_model & model, const char * settings_text, uint64_t epoch) {
    if (!settings_text) {
        throw std::runtime_error("[oh-ssd] no settings");
    }
    if (!model.ssd_experts) {
        throw std::runtime_error("[oh-ssd] SSD streaming needs a model loaded with ssd_experts (--ssd)");
    }
    if (model.ssd_expert_ranges.empty() || model.ssd_files.empty()) {
        throw std::runtime_error("[oh-ssd] the model has no routed expert tensors to stream");
    }

    context_runtime_ptr rt(new context_runtime());

    // Split the text into engine keys and InferSSD overrides.
    std::vector<ssd::settings::override_value> overrides;
    std::set<std::string>                      seen;
    bool                                       host_budget_set = false;
    {
        std::istringstream in(settings_text);
        std::string        line;
        while (std::getline(in, line)) {
            line = trim(line);
            if (line.empty() || line[0] == '#') {
                continue;
            }
            const auto eq = line.find('=');
            if (eq == std::string::npos) {
                throw std::runtime_error("[oh-ssd] setting '" + line + "' is not key=value");
            }
            const std::string key   = trim(line.substr(0, eq));
            const std::string value = trim(line.substr(eq + 1));
            if (!seen.insert(key).second) {
                throw std::runtime_error("[oh-ssd] setting '" + key + "' is given twice");
            }
            if (key == "engine.ram_budget_mib") {
                rt->ram_budget = parse_mib(key, value);
            } else if (key == "engine.expert_cache_mib") {
                rt->cache_cap = parse_mib(key, value);
            } else {
                if (key == "host.budget_mib") {
                    host_budget_set = true;
                }
                overrides.push_back({ key, value });
            }
        }
    }
#if defined(_WIN32) || defined(__APPLE__)
    // On Windows and macOS the arbiter also enforces the engine budget as the
    // process envelope (Linux uses the cgroup instead).
    if (rt->ram_budget && !host_budget_set) {
        overrides.push_back({ "host.budget_mib", std::to_string(rt->ram_budget / MiB) });
    }
#else
    (void) host_budget_set;
#endif
    auto parsed = ssd::settings::parse("", "", overrides, PLATFORM);
    if (!parsed) {
        throw std::runtime_error("[oh-ssd] settings: " + parsed.error);
    }
    rt->settings.emplace(std::move(parsed.value));

    // Refuse experts the sink cannot compute here, with a normal error; the sink itself only finds out inside the
    // first graph, where it can only abort.
    using supported_fn = bool (*)(ggml_type, int64_t);
    auto * cpu_reg = ggml_backend_reg_by_name("CPU");
    auto supported = cpu_reg ? reinterpret_cast<supported_fn>(
                                   ggml_backend_reg_get_proc_address(cpu_reg, "kimi_cpu_sink_expert_type_supported"))
                             : nullptr;
    if (!supported) {
        throw std::runtime_error("[oh-ssd] the CPU backend has no SSD expert sink");
    }
    for (const auto & r : model.ssd_expert_ranges) {
        const ggml_tensor * t = model.get_tensor(r.tensor.c_str());
        if (!t) {
            throw std::runtime_error("[oh-ssd] expert tensor " + r.tensor + " was not created");
        }
        if (!supported(t->type, t->ne[0])) {
            throw std::runtime_error(std::string("[oh-ssd] this quantization is not supported yet: expert tensor ") +
                                     r.tensor + " is " + ggml_type_name(t->type) +
                                     ", which the CPU expert sink cannot compute");
        }
    }

    // Original-GGUF indexing: every expert tensor is one contiguous range of
    // n_expert equal blocks in one of the model's files.
    std::vector<int> fd_of_file(model.ssd_files.size(), -1);
    for (const auto & r : model.ssd_expert_ranges) {
        if (r.file >= model.ssd_files.size() || model.ssd_files[r.file].empty()) {
            throw std::runtime_error("[oh-ssd] expert tensor " + r.tensor + " has no file path");
        }
        if (r.n_expert <= 0 || r.nbytes % uint64_t(r.n_expert) != 0) {
            throw std::runtime_error("[oh-ssd] expert tensor " + r.tensor + " is not n_expert equal blocks");
        }
        if (fd_of_file[r.file] < 0) {
            std::string error;
            const int   fd = ssd::platform::open_direct(model.ssd_files[r.file].c_str(), &error);
            if (fd < 0) {
                throw std::runtime_error("[oh-ssd] cannot open " + model.ssd_files[r.file] + " for direct reads: " +
                                         error);
            }
            fd_of_file[r.file] = fd;
            rt->fds.push_back(fd);
            uint64_t size = 0;
            if (!ssd::platform::file_size(fd, size)) {
                throw std::runtime_error("[oh-ssd] cannot size " + model.ssd_files[r.file]);
            }
        }
        rt->ranges[r.tensor] = { fd_of_file[r.file], r.offset, r.nbytes, r.n_expert };
        rt->experts += r.nbytes;
    }
    rt->source.userdata     = rt.get();
    rt->source.resolve      = &context_runtime::resolve;
    // Dense weights stay resident: this source holds none, which the GPU transport asks about through this call.
    rt->source.dense_blocks = [](void *, const char *, size_t, size_t * block) noexcept -> size_t {
        if (block) { *block = 0; }
        return 0;
    };

    rt->arbiter     = shared_arbiter();
    auto registered = rt->arbiter->make_client();
    if (!registered) {
        throw std::runtime_error("[oh-ssd] host memory client: " + registered.error);
    }
    rt->client = std::move(registered.value);

    rt->services        = {};
    rt->services.host   = rt->client.get();
    rt->services.source = &rt->source;

    std::vector<int32_t> routed;
    for (const auto & r : model.ssd_expert_ranges) {
        const int il = layer_of(r.tensor);
        if (il < 0 || il >= int(model.hparams.n_layer_all)) {
            throw std::runtime_error("[oh-ssd] cannot place expert tensor " + r.tensor + " in a layer");
        }
        // The draft head's experts have their own quantization and no adapter path yet; a load without it has no
        // tensor past n_layer, so this only fires when the head was asked for.
        if (model.arch == LLM_ARCH_GLM5_NEXT && il >= int(model.hparams.n_layer())) {
            throw std::runtime_error("[oh-ssd] GLM-5.3-Flash's draft (MTP) head is not supported with SSD experts yet: "
                                     "load the model without it");
        }
        if (std::find(routed.begin(), routed.end(), il) == routed.end()) {
            routed.push_back(il);
        }
    }
    std::sort(routed.begin(), routed.end());

    rt->sink = std::make_unique<llama_kimi_cpu_sink_runtime>(int(model.hparams.n_layer_all), epoch,
                                                             int64_t(model.hparams.n_expert), routed, *rt->settings,
                                                             rt->services, rt->source);
    // No cache until apply_budget has measured the context's buffers.
    rt->sink->cache_budget = 0;
    for (const auto & r : model.ssd_expert_ranges) {
        rt->sink->expert_shapes[r.tensor] = { size_t(r.nbytes / uint64_t(r.n_expert)), r.n_expert };
    }
    return rt;
}

llama_kimi_cpu_sink_runtime * sink_runtime(context_runtime & runtime) noexcept {
    return runtime.sink.get();
}

uint64_t vram_bookkeeping_quote(uint64_t budget, std::vector<uint64_t> layer_bytes, uint64_t layer_ids, uint64_t n_expert) {
    // Smallest first is the order that fits the most slots. The tier packs its layers in priority order at page-rounded
    // bytes, from a budget it may clamp further, so its count cannot come out above this one.
    std::sort(layer_bytes.begin(), layer_bytes.end());
    const uint64_t slots = std::min<uint64_t>(vram_layer_slot_capacity(budget, layer_bytes), layer_bytes.size() * n_expert);
    return vram_tier_bookkeeping_bytes(slots, layer_ids, n_expert);
}

uint64_t price_vram_tier(context_runtime & runtime, const std::vector<vram_tier_layer> & layers, uint64_t budget,
                         uint64_t latent, uint64_t hidden, uint64_t n_expert) {
    constexpr uint64_t hits = 16; // the routes every compute engine is created for: the tier passes it to create()
    std::set<std::pair<int32_t, int32_t>> pairs; // the tier makes one engine per up/down type pair, whatever the slot size
    std::vector<uint64_t>                 raw;
    uint64_t                              largest   = 0;
    uint64_t                              layer_ids = 0;
    for (const auto & l : layers) {
        pairs.insert({ l.up_type, l.down_type });
        raw.push_back(l.raw_bytes);
        largest   = std::max(largest, (l.raw_bytes + 4095) & ~uint64_t(4095));
        layer_ids = std::max(layer_ids, uint64_t(l.id) + 1);
    }
    // One engine's device buffers, as generic-compute-adapter.cu allocates them: the input and down rows, the up/gate
    // and hidden rows, and the quantized operand of whichever product is largest (F16 and BF16 need none).
    const auto is_float = [](int32_t type) { return type == GGML_TYPE_F16 || type == GGML_TYPE_BF16; };
    const auto q8_1     = [](uint64_t columns) -> uint64_t {
        return ggml_row_size(GGML_TYPE_Q8_1, int64_t((columns + 511) / 512 * 512)); // rows are padded to 512 columns
    };
    uint64_t device = 0;
    for (const auto & pair : pairs) {
        device += hits * (latent * sizeof(float) * 2 + hidden * sizeof(float) * 3) +
                  std::max<uint64_t>(is_float(pair.first) ? 0 : q8_1(latent), is_float(pair.second) ? 0 : hits * q8_1(hidden));
    }
    // What the tier takes on the host comes out of the host budget, at the first decode: its scratch (two slots of the
    // largest format), each engine's two pinned row buffers, the layer's input row and the rows it brings back, and the
    // bookkeeping that grows with the slots and experts.
    runtime.tier_host = 2 * largest + uint64_t(pairs.size()) * hits * latent * sizeof(float) * 2 +
                        (2 * hits + 1) * latent * sizeof(float) + vram_bookkeeping_quote(budget, raw, layer_ids, n_expert);
    return device;
}

vram_receipt compute_vram_receipt(const vram_inputs & in) {
    vram_receipt out;
    out.keep = in.reserve + in.margin + in.overhead;
    out.max_budget = in.free > out.keep ? in.free - out.keep : 0;
    for (const uint64_t b : in.layer_bytes) {
        out.min_budget += b;
    }
    out.fits = in.total != 0 && !in.layer_bytes.empty() && in.budget <= out.max_budget && in.budget >= out.min_budget;
    if (out.min_budget) {
        out.slots = std::min<uint64_t>(vram_layer_slot_capacity(in.budget, in.layer_bytes),
                                       uint64_t(in.layer_bytes.size()) * in.n_expert);
        out.slot_bytes = vram_layer_slot_bytes(size_t(out.slots), in.layer_bytes);
    }
    char line[256];
    std::snprintf(line, sizeof(line),
                  "[oh-ssd] VRAM MiB: device=%" PRIu64 " free=%" PRIu64 " reserve=%" PRIu64 " margin=%" PRIu64
                  " tier-buffers=%" PRIu64 " budget=%" PRIu64 " slots=%" PRIu64 " slot-bytes=%" PRIu64,
                  mib_ceil(in.total), mib_ceil(in.free), mib_ceil(in.reserve), mib_ceil(in.margin), mib_ceil(in.overhead),
                  mib_ceil(in.budget), out.slots, mib_ceil(out.slot_bytes));
    out.text = line;
    return out;
}

void attach_gpu_backend(context_runtime & runtime, const llama_model & model, ggml_backend_t backend) {
    const auto * budget  = runtime.settings->vram_budget();
    const auto * reserve = runtime.settings->integer("vram.reserve_mib");
    if (!budget || !reserve) {
        throw std::runtime_error("[oh-ssd] the VRAM budget and reserve settings cannot be read");
    }
    if (budget->automatic || budget->bytes == 0) {
        return;
    }
    if (!backend) {
        throw std::runtime_error("[oh-ssd] vram.budget is set but this context has no GPU device; "
                                 "remove it to keep every expert on the CPU");
    }
    auto &     services = runtime.services;
    const auto device   = ggml_backend_get_device(backend);
    const auto get      = reinterpret_cast<ssd::gpu::get_provider_v1>(
        ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(device), "ggml_backend_ssd_provider_v1"));
    if (!get) {
        throw std::runtime_error("[oh-ssd] vram.budget is set but this GPU backend has no expert tier (a CUDA build is needed)");
    }
    const auto * provider = get();
    if (!provider || provider->version != 1 || provider->size != sizeof(*provider) || !provider->device_is_unified ||
        !provider->device_ordinal || !provider->attach_services || !provider->stream) {
        throw std::runtime_error("[oh-ssd] the GPU backend has no compatible SSD provider table");
    }
    const auto * transport = provider->transport;
    if (!transport || transport->version != 1 || transport->size != sizeof(*transport) ||
        transport->runtime_services_size != sizeof(services) || !transport->create || !transport->quiesce ||
        !transport->destroy || !transport->copy_ranges || !transport->dense_has || !transport->dense_prepare ||
        !transport->dense_read_ahead || !transport->dense_copy || !transport->seed_register ||
        !transport->seed_unregister || !transport->seed_offer || !transport->seed_fence) {
        throw std::runtime_error("[oh-ssd] the GPU backend has no compatible context transport table");
    }
    const int ordinal = provider->device_ordinal(backend);
    if (ordinal < 0) {
        throw std::runtime_error("[oh-ssd] the GPU device is unavailable to the expert tier");
    }

    // One slot is an expert's three roles, page-aligned as a whole, as the tier lays them out.
    std::map<int, uint64_t> layer_slot;
    for (const auto & r : model.ssd_expert_ranges) {
        layer_slot[layer_of(r.tensor)] += r.nbytes / uint64_t(r.n_expert);
    }
    vram_inputs in;
    for (const auto & entry : layer_slot) {
        in.layer_bytes.push_back((entry.second + 4095) & ~uint64_t(4095));
    }
    // The tensors the sink declares to the tier, and each layer's expert types read from them.
    std::vector<const ggml_tensor *> tier_experts;
    std::vector<vram_tier_layer>     tier_layers;
    for (const int32_t layer : runtime.sink->layer_priority) {
        for (const char * role : { "up", "gate", "down" }) {
            const std::string name = "blk." + std::to_string(layer) + ".ffn_" + role + "_exps.weight";
            const ggml_tensor * tensor = model.get_tensor(name.c_str());
            if (!tensor) {
                throw std::runtime_error("[oh-ssd] the VRAM expert tier cannot find " + name);
            }
            tier_experts.push_back(tensor);
        }
        tier_layers.push_back({ layer, layer_slot[layer], int32_t(tier_experts[tier_experts.size() - 3]->type),
                                int32_t(tier_experts.back()->type) });
    }
    size_t device_free = 0, device_total = 0;
    ggml_backend_dev_memory(device, &device_free, &device_total);
    in.budget   = budget->bytes;
    in.reserve  = *reserve * MiB;
    in.margin   = 64 * MiB; // the tier's own driver margin
    in.overhead = price_vram_tier(runtime, tier_layers, budget->bytes, uint64_t(tier_experts[0]->ne[0]),
                                  uint64_t(tier_experts[0]->ne[1]), uint64_t(tier_experts[0]->ne[2]));
    in.free     = device_free;
    in.total    = device_total;
    in.n_expert = uint64_t(model.hparams.n_expert);
    const vram_receipt out = compute_vram_receipt(in);
    LLAMA_LOG_INFO("%s\n", out.text.c_str());
    if (!device_total) {
        throw std::runtime_error("[oh-ssd] the GPU's memory limits cannot be read, so the VRAM tier cannot be admitted");
    }
    if (!out.fits) {
        std::string msg = out.text + "; ";
        if (in.budget > out.max_budget) {
            msg += "vram.budget is more than this device can give with " + std::to_string(mib_ceil(out.keep)) +
                   " MiB kept free: the largest that fits is " + std::to_string(out.max_budget >> 20) + " MiB";
        } else {
            msg += "vram.budget is below one slot per layer: the smallest that works is " +
                   std::to_string(mib_ceil(out.min_budget)) + " MiB";
        }
        throw std::runtime_error(msg);
    }

    services.gpu             = provider;
    services.gpu_device      = ordinal;
    services.gpu_reserve_bytes = size_t(*reserve) * size_t(MiB);
    char error[512] {};
    runtime.gpu_transport = std::unique_ptr<void, void (*)(void *) noexcept>(
        transport->create(&services, ordinal, error, sizeof(error)), transport->destroy);
    if (!runtime.gpu_transport) {
        throw std::runtime_error(error[0] ? error : "[oh-ssd] cannot create the GPU expert transport");
    }
    services.gpu_transport = runtime.gpu_transport.get();
    if (!provider->attach_services(backend, &services, error, sizeof(error))) {
        throw std::runtime_error(error[0] ? error : "[oh-ssd] cannot attach the GPU expert transport");
    }
    // The sink read its budget before the GPU was attached; price_vram_tier set what the tier takes out of it.
    runtime.sink->vram_budget  = budget->bytes;
    runtime.sink->vram_experts = std::move(tier_experts);
}

bool attach_cpu_backend(context_runtime & runtime, ggml_backend_t backend_cpu) {
    if (!backend_cpu) {
        return false;
    }
    using set_fn = bool (*)(ggml_backend_t, const ssd::runtime_services *) noexcept;
    auto * reg   = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
    auto   set   = reg ? reinterpret_cast<set_fn>(
                         ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_ssd_runtime_services"))
                       : nullptr;
    if (!set || !set(backend_cpu, &runtime.services)) {
        return false;
    }
    return true;
}

uint64_t physical_memory_bytes() noexcept {
#if defined(_WIN32)
    MEMORYSTATUSEX status = {};
    status.dwLength       = sizeof(status);
    return GlobalMemoryStatusEx(&status) ? uint64_t(status.ullTotalPhys) : 0;
#else
    const long pages = sysconf(_SC_PHYS_PAGES);
    const long size  = sysconf(_SC_PAGESIZE);
    return pages > 0 && size > 0 ? uint64_t(pages) * uint64_t(size) : 0;
#endif
}

receipt compute_receipt(uint64_t budget, uint64_t headroom, uint64_t experts, uint64_t cache_cap,
                        const memory_inputs & in) {
    receipt out;
    out.budget        = budget;
    out.headroom      = headroom;
    out.experts       = experts;
    out.dense_weights = in.model_host > experts ? in.model_host - experts : 0;
    out.context       = in.context_host;
    out.compute       = in.compute_host;
    // The measured process holds the context's buffers already; the rest is the process itself. What is charged but
    // not allocated yet is not in that measurement, so taking it off would hide it from the expert cache. The dense
    // weights are memory-mapped (SSD experts require it), so the envelope does not charge them, but their pages
    // stay resident and the receipt counts them anyway. A dense weight copied out of the mapping (a repacked
    // buffer) lands in both terms, which only errs safe.
    const uint64_t allocated = out.context - std::min(in.context_unallocated, out.context) + out.compute;
    const uint64_t buffers   = in.buffers_allocated ? allocated : 0;
    out.process            = in.process_private > buffers ? in.process_private - buffers : 0;

    const uint64_t fixed = out.dense_weights + out.context + out.compute + out.process + out.headroom;
    if (budget >= fixed) {
        out.fits  = true;
        out.cache = std::min(budget - fixed, cache_cap);
    } else {
        // Hold everything but the context-linear share fixed (the rest of the
        // compute buffer only shrinks with a smaller context, so this is
        // conservative). The
        // process moves a little between loads, so a measured process term
        // keeps 1/32 of itself (at least 4 MiB) spare.
        const uint64_t linear = std::min(in.kv_linear_host, fixed);
        const uint64_t drift  = out.process ? std::max<uint64_t>(4 * MiB, out.process / 32) : 0;
        const uint64_t rest   = fixed - linear + drift;
        if (budget > rest && linear > 0 && in.n_ctx > 0) {
            const uint64_t per_token = (linear + in.n_ctx - 1) / in.n_ctx;
            // Only multiples of the smallest context survive the constructor's padding; any other size, or one
            // below the smallest, would come back as a context this receipt did not judge.
            const uint64_t step      = std::max<uint64_t>(in.n_ctx_min, 256);
            uint64_t       ctx       = (budget - rest) / per_token;
            ctx                      = ctx / step * step;
            out.max_ctx              = uint32_t(std::min<uint64_t>(ctx, in.n_ctx));
        }
    }

    char text[640];
    std::snprintf(text, sizeof(text),
                  "[oh-ssd] receipt (MiB): budget=%" PRIu64 " headroom=%" PRIu64 " dense_weights=%" PRIu64
                  " experts_on_disk=%" PRIu64 " kv_attn=%" PRIu64 " kv_swa=%" PRIu64 " kv_index=%" PRIu64 " recurrent=%" PRIu64
                  " context_host=%" PRIu64 " compute_host=%" PRIu64 " process=%" PRIu64 " device=%" PRIu64
                  " ctx=%u slots=%u expert_cache=%" PRIu64 "%s",
                  mib_ceil(budget), mib_ceil(headroom), mib_ceil(out.dense_weights), mib_ceil(experts),
                  mib_ceil(in.kv_attn_bytes), mib_ceil(in.kv_swa_bytes), mib_ceil(in.kv_index_bytes), mib_ceil(in.recurrent_bytes),
                  mib_ceil(out.context), mib_ceil(out.compute), mib_ceil(out.process), mib_ceil(in.device_bytes),
                  in.n_ctx, in.n_seq,
                  out.cache / MiB, out.fits ? "" : " (does not fit)");
    out.text = text;
    return out;
}

std::string refusal_advice(const receipt & out, const memory_inputs & in) {
    if (out.max_ctx) {
        return "the largest context that fits is " + std::to_string(out.max_ctx) + " tokens";
    }
    if (in.n_ctx <= in.n_ctx_min) {
        return std::string("no context fits these settings: this is already the smallest context, "
                           "so raise engine.ram_budget_mib or lower --batch-size or --ubatch-size") +
               (in.n_seq > 1 ? " or -np" : "");
    }
    if (!in.compute_follows_ctx) {
        return "no context fits: the dense weights and buffers alone exceed it "
               "(a smaller --batch-size or --ubatch-size shrinks the buffers)";
    }
    return "no size can be named here, because this model's compute buffers shrink with the context "
           "by an amount only a load at that size shows: try a smaller -c";
}

// The receipt for this context against the host budget; refuses, with the largest context that fits, when it is over.
static receipt checked_receipt(context_runtime & runtime, const memory_inputs & measured) {
    uint64_t budget = runtime.ram_budget ? runtime.ram_budget : physical_memory_bytes();
#if defined(__linux__)
    // The kernel enforces memory.max whatever the engine was told, so a context above it would be killed rather than
    // refused, and one whose limit exists but cannot be read cannot be admitted at all.
    kimi_host_ledger ledger;
    ledger.locate();
    uint64_t limit = UINT64_MAX;
    bool     found = false;
    if (!ledger.memory_limit(limit, &found)) {
        throw std::runtime_error("[oh-ssd] cannot tell this process's memory limit, so no context can be admitted: "
                                 "/proc/self/cgroup or a memory.max on its cgroup path cannot be read");
    }
    budget = std::min(budget, limit);
    // With no memory controller nothing here enforces a limit, so MemAvailable stands in for one. That holds only
    // before the buffers exist: once they do, MemAvailable already has them taken out.
    if (!found && !measured.buffers_allocated) {
        static std::once_flag warned;
        std::call_once(warned, [] {
            std::fprintf(stderr, "[oh-ssd] warning: no cgroup memory controller applies to this process, "
                                 "so the context is held to the RAM budget and MemAvailable\n");
        });
        uint64_t available = 0;
        if (ledger.memory_available(available)) {
            budget = std::min(budget, available);
        }
    }
#endif
    const auto *   headroom = runtime.settings->integer("host.headroom_mib");
    memory_inputs  inputs   = measured;
    inputs.context_host        += runtime.tier_host;
    inputs.context_unallocated += runtime.tier_host;
#if defined(_WIN32) || defined(__APPLE__)
    // The arbiter holds the budget against the whole process here, so the receipt counts the whole process too.
    if (!ssd::host::native::process_use(inputs.process_private)) {
        throw std::runtime_error("[oh-ssd] cannot read this process's memory use for the host budget");
    }
#endif
    receipt out = compute_receipt(budget, headroom ? *headroom * MiB : 512 * MiB, runtime.experts,
                                  runtime.cache_cap, inputs);
    if (!out.fits) {
        LLAMA_LOG_INFO("%s\n", out.text.c_str());
        std::string msg = "[oh-ssd] a context of " + std::to_string(inputs.n_ctx) +
                          " tokens does not fit the host budget of " + std::to_string(mib_ceil(budget)) + " MiB; ";
        msg += refusal_advice(out, inputs);
        throw std::runtime_error(msg);
    }
    return out;
}

receipt admit_budget(context_runtime & runtime, const memory_inputs & measured) {
    memory_inputs inputs     = measured;
    inputs.buffers_allocated = false;
    return checked_receipt(runtime, inputs);
}

receipt apply_budget(context_runtime & runtime, const memory_inputs & measured) {
    receipt out = checked_receipt(runtime, measured);
    LLAMA_LOG_INFO("%s\n", out.text.c_str());
    runtime.sink->cache_budget = out.cache;
    return out;
}

} // namespace oh_ssd
