// OpenHarness SSD expert streaming: end-to-end tests on the synthetic MoE
// fixture (tests/make_moe_fixture.py), CPU only.
//
//   ssd_stream_test equal   <model.gguf>   resident and streamed runs give the same tokens and logits
//   ssd_stream_test context <model.gguf>   the host budget reserves KV and buffers before the expert cache
//   ssd_stream_test admit   <model.gguf>   a context over budget is refused before its KV is allocated
//   ssd_stream_test huge    <model.gguf>   a context far over budget is refused without per-token bookkeeping
//   ssd_stream_test reserve <model.gguf>   sizing covers the decode graph as well as the prompt graph
//   ssd_stream_test batch   <model.gguf>   a batch size over budget is refused before anything sized by it exists
//   ssd_stream_test borrow  <model.gguf>   sizing a cache that borrows a live cache's cells leaves them alone
//   ssd_stream_test refuse  <model.gguf>   experts the sink cannot compute fail the load with a reason
//   ssd_stream_test mtp     <model.gguf>   experts of an unloaded MTP layer stay out of the source index
//   ssd_stream_test glm-fixture <base.gguf>   writes ssd-glm-iq.gguf and ssd-glm-iq-mixed.gguf next to the base
//   ssd_stream_test glm     <model.gguf>   GLM's quant mix streams identically, and its MTP head stays out or refuses
//   ssd_stream_test glm-mixed <model.gguf> up and gate experts of different types are refused with a reason
//   ssd_stream_test tier-pairs <model.gguf> the VRAM tier's price is per type pair, and admit_budget charges it
//   ssd_stream_test glm-admit <model.gguf> the indexer and its k-pool bookkeeping are priced before they exist
//
// The streamed runs read every routed expert from the GGUF file through the
// SSD runtime: with no RAM expert cache (the read ring only), with a cache too
// small for one layer's experts, and with the cache the budget leaves.

#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"

// Internal headers, for the cache that borrows another's cells: no public path reaches it on this fixture.
#include "llama-kv-cache.h"
#include "ssd/oh-runtime.h"
#include "ggml-cuda/ssd/vram-slot-table.h"
#include "llama-model.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#    include <psapi.h>
#    include <io.h>
#    define oh_dup   _dup
#    define oh_dup2  _dup2
#    define oh_close _close
#    define oh_fileno _fileno
#else
#    include <unistd.h>
#    define oh_dup   dup
#    define oh_dup2  dup2
#    define oh_close close
#    define oh_fileno fileno
#endif

namespace {

std::mutex  g_log_mutex;
std::string g_log;
bool        g_verbose = false;

void log_cb(ggml_log_level level, const char * text, void *) {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    g_log += text;
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
}

std::string take_log() {
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::string out;
    out.swap(g_log);
    return out;
}

// The sink writes its counters (reads, cache hits) to stderr, not to the llama
// log. Redirects stderr into a file while one streamed model is alive.
class stderr_capture {
  public:
    explicit stderr_capture(std::string file) : path(std::move(file)) {
        std::fflush(stderr);
        out = std::fopen(path.c_str(), "wb");
        if (out) {
            saved = oh_dup(oh_fileno(stderr));
            oh_dup2(oh_fileno(out), oh_fileno(stderr));
        }
    }
    stderr_capture(const stderr_capture &) = delete;
    stderr_capture & operator=(const stderr_capture &) = delete;
    ~stderr_capture() { finish(); }

    // Restores stderr and returns what was written to it.
    std::string finish() {
        if (!out) {
            return text;
        }
        std::fflush(stderr);
        oh_dup2(saved, oh_fileno(stderr));
        oh_close(saved);
        std::fclose(out);
        out = nullptr;
        if (FILE * in = std::fopen(path.c_str(), "rb")) {
            char   buf[4096];
            size_t n;
            while ((n = std::fread(buf, 1, sizeof(buf), in)) > 0) {
                text.append(buf, n);
            }
            std::fclose(in);
        }
        std::remove(path.c_str());
        if (g_verbose || text.find("FAIL ") != std::string::npos) { // a CHECK that failed while captured
            std::fputs(text.c_str(), stderr);
        }
        return text;
    }

  private:
    std::string path;
    std::string text;
    FILE *      out   = nullptr;
    int         saved = -1;
};

// Reads "key=<number>" from the last line in `text` that contains `marker`; empty if there is none.
std::map<std::string, uint64_t> counters(const std::string & text, const char * marker) {
    std::map<std::string, uint64_t> out;
    const auto pos = text.rfind(marker);
    if (pos == std::string::npos) {
        return out;
    }
    const auto begin = text.rfind('\n', pos);
    const auto end   = text.find('\n', pos);
    const auto line  = text.substr(begin == std::string::npos ? 0 : begin + 1,
                                   end == std::string::npos ? std::string::npos : end - (begin == std::string::npos ? 0 : begin + 1));
    size_t i = 0;
    while ((i = line.find('=', i)) != std::string::npos) {
        size_t k = i;
        while (k > 0 && line[k - 1] != ' ') {
            --k;
        }
        out[line.substr(k, i - k)] = std::strtoull(line.c_str() + i + 1, nullptr, 10);
        ++i;
    }
    return out;
}

int g_failures = 0;

#define CHECK(cond, ...)                                                      \
    do {                                                                      \
        if (!(cond)) {                                                        \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                \
            std::fprintf(stderr, "\n");                                       \
            ++g_failures;                                                     \
        }                                                                     \
    } while (0)

struct loaded {
    llama_model *   model = nullptr;
    llama_context * ctx   = nullptr;
    std::string     log;

    loaded() = default;
    loaded(const loaded &) = delete;
    loaded & operator=(const loaded &) = delete;
    loaded(loaded && o) noexcept : model(o.model), ctx(o.ctx), log(std::move(o.log)) { o.model = nullptr; o.ctx = nullptr; }
    ~loaded() {
        if (ctx) {
            llama_free(ctx);
        }
        if (model) {
            llama_model_free(model);
        }
    }
};

// ssd == false: the experts load as ordinary weights. ssd == true: they stay in
// the file and the context streams them with these settings.
loaded load(const char * path, bool ssd, const std::string & settings, uint32_t n_ctx, uint32_t n_ubatch = 16,
            uint32_t n_seq = 1, uint32_t n_batch = 0, llama_sampler * sampler = nullptr, uint32_t outputs_per_seq = 1,
            bool load_mtp = false, bool kv_unified = false, bool gpu = false, int n_gpu_layers = 0) {
    static ggml_backend_dev_t no_devices[] = { nullptr }; // CPU only, even in a CUDA build
    static ggml_backend_dev_t gpu_devices[] = { nullptr, nullptr };
    take_log();
    loaded out;
    auto mp            = llama_model_default_params();
    mp.devices         = no_devices;
    if (gpu) { // the expert tier leaves n_gpu_layers at 0, so the ordinary weights stay on the CPU
        gpu_devices[0] = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        mp.devices     = gpu_devices;
    }
    mp.n_gpu_layers    = n_gpu_layers;
    mp.use_extra_bufts = false; // the resident run uses the plain CPU kernels, as the sink does
    mp.ssd_experts     = ssd;
    mp.load_mtp        = load_mtp;
    out.model          = llama_model_load_from_file(path, mp);
    if (!out.model) {
        out.log = take_log();
        return out;
    }
    auto cp            = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_batch ? n_batch : std::max(64u, n_seq); // the whole prompt in one llama_decode, split into ubatches
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = n_seq;
    cp.n_threads       = 4;
    cp.n_threads_batch = 4;
    cp.offload_kqv     = false;
    cp.op_offload      = false;
    cp.no_perf         = true;
    cp.ssd_settings    = ssd ? settings.c_str() : nullptr;
    cp.n_outputs_max_per_seq = outputs_per_seq;
    cp.kv_unified      = kv_unified;
    llama_sampler_seq_config sampler_config = { 0, sampler };
    if (sampler) {
        cp.samplers   = &sampler_config;
        cp.n_samplers = 1;
    }
    out.ctx            = llama_init_from_model(out.model, cp);
    out.log            = take_log();
    return out;
}

struct run_result {
    bool                            ok = false;
    std::vector<llama_token>        tokens;
    std::vector<std::vector<float>> logits; // one row per generated token
};

// A 40-token prompt (three prefill ubatches of 16) then 24 greedy tokens. With forced set, the tokens recorded are still
// the greedy choices, but the forced tokens are what is decoded, so two runs stay on one sequence even if they disagree.
run_result generate(loaded & l, const std::vector<llama_token> * forced = nullptr) {
    run_result r;
    const int  n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(l.model));
    std::vector<llama_token> prompt;
    for (int i = 0; i < 40; ++i) {
        prompt.push_back(llama_token((i * 37 + 11) % n_vocab));
    }
    if (llama_decode(l.ctx, llama_batch_get_one(prompt.data(), int32_t(prompt.size()))) != 0) {
        return r;
    }
    for (int step = 0; step < 24; ++step) {
        const float * logits = llama_get_logits_ith(l.ctx, -1);
        if (!logits) {
            return r;
        }
        r.logits.emplace_back(logits, logits + n_vocab);
        llama_token best = 0;
        for (int i = 1; i < n_vocab; ++i) {
            if (logits[i] > logits[best]) {
                best = i;
            }
        }
        r.tokens.push_back(best);
        llama_token next = forced ? (*forced)[size_t(step)] : best;
        if (llama_decode(l.ctx, llama_batch_get_one(&next, 1)) != 0) {
            return r;
        }
    }
    r.ok = true;
    return r;
}

bool contains(const std::string & s, const char * what) {
    return s.find(what) != std::string::npos;
}

// Reads "key=<number>" from the last "[oh-ssd] receipt" line in a log.
std::map<std::string, uint64_t> receipt_of(const std::string & log) {
    const char * marker = "[oh-ssd] receipt (MiB):";
    auto         out    = counters(log, marker);
    if (!out.empty()) {
        const auto pos = log.rfind(marker);
        const auto end = log.find('\n', pos);
        out["fits"]    = log.substr(pos, end == std::string::npos ? std::string::npos : end - pos).find("(does not fit)") ==
                              std::string::npos ? 1 : 0;
    }
    return out;
}

// has_hint: the arch's graph feeds the sink a next-layer guess (only qwen3moe does so far).
int test_equal(const char * path, bool has_hint = true) {
    loaded resident = load(path, false, "", 1024);
    CHECK(resident.ctx, "resident load failed:\n%s", resident.log.c_str());
    if (!resident.ctx) {
        return 1;
    }
    const run_result base = generate(resident);
    CHECK(base.ok, "resident generation failed");
    // NaNs compare equal bitwise, so equal runs would prove nothing about a model whose arithmetic has broken.
    for (const auto & row : base.logits) {
        CHECK(std::all_of(row.begin(), row.end(), [](float x) { return std::isfinite(x); }), "the fixture's logits are not finite");
    }

    enum cache_kind { no_cache, tiny_cache, budget_cache };
    struct variant {
        const char * name;
        std::string  settings;
        cache_kind   cache;
    };
    // The default next-layer prefetch (prefetch.depth=2) is on in every variant but the first. It reads into cache
    // slots, so with no RAM cache it has nowhere to read to and issues nothing.
    const variant variants[] = {
        { "read ring only, no next-layer prefetch", "engine.expert_cache_mib=0\nprefetch.depth=0\n", no_cache },
        { "read ring only (no RAM cache)", "engine.expert_cache_mib=0\n", no_cache },
        { "RAM cache smaller than one layer", "engine.expert_cache_mib=1\n", tiny_cache },
        { "RAM cache from the budget", "# what a 2 GiB budget leaves\nengine.ram_budget_mib=2048\nhost.headroom_mib=16\n",
          budget_cache },
    };
    uint64_t ring_only_bytes = 0;
    for (const auto & v : variants) {
        stderr_capture capture(std::string(path) + ".stderr.txt");
        run_result     got;
        {
            loaded streamed = load(path, true, v.settings, 1024);
            CHECK(streamed.ctx, "%s: streamed load failed:\n%s", v.name, streamed.log.c_str());
            if (!streamed.ctx) {
                continue;
            }
            CHECK(contains(streamed.log, "[oh-ssd] receipt"), "%s: no receipt in the log", v.name);
            got = generate(streamed);
            take_log();
        } // freeing the context prints the sink's totals and final cache counters
        const std::string sink_log = capture.finish();
        CHECK(got.ok, "%s: streamed generation failed", v.name);
        CHECK(got.tokens == base.tokens, "%s: tokens differ", v.name);
        CHECK(got.logits.size() == base.logits.size(), "%s: %zu logit rows, expected %zu", v.name, got.logits.size(),
              base.logits.size());
        float  max_diff = 0.0f;
        size_t n_diff   = 0;
        for (size_t s = 0; s < std::min(got.logits.size(), base.logits.size()); ++s) {
            for (size_t i = 0; i < base.logits[s].size(); ++i) {
                const float d = std::fabs(got.logits[s][i] - base.logits[s][i]);
                max_diff      = std::max(max_diff, d);
                n_diff += std::memcmp(&got.logits[s][i], &base.logits[s][i], sizeof(float)) != 0; // +0 and -0 differ
            }
        }
        // The experts really came from the file, and the RAM cache did what the variant says.
        auto io    = counters(sink_log, "[kimi_cpu_sink_total]");
        auto cache = counters(sink_log, "reason=before_destroy");
        auto ahead = counters(sink_log, "[kimi_cpu_sink_prefetch] reason=");
        std::printf("%s: %zu tokens, max |logit diff| = %g, %zu differing logits; read %" PRIu64 " bytes, "
                    "cache hits %" PRIu64 ", evictions %" PRIu64 "\n",
                    v.name, got.tokens.size(), double(max_diff), n_diff, io["read_bytes"], cache["hits"], cache["evictions"]);
        CHECK(n_diff == 0, "%s: logits are not bitwise identical (max diff %g)", v.name, double(max_diff));
        CHECK(!io.empty() && io["read_bytes"] > 0 && io["read_errors"] == 0 && io["fallback_count"] == 0,
              "%s: the sink did not stream the experts from the file", v.name);
        const bool prefetch_off = contains(v.settings, "prefetch.depth=0");
        CHECK(ahead["topk"] == (prefetch_off ? 0u : 2u) && ahead["failed"] == 0,
              "%s: prefetch ran at depth %" PRIu64 " with %" PRIu64 " failed reads", v.name, ahead["topk"], ahead["failed"]);
        CHECK((ahead["issued"] > 0) == (has_hint && v.cache != no_cache),
              "%s: prefetch issued %" PRIu64 " reads, which does not match the cache this variant has", v.name, ahead["issued"]);
        switch (v.cache) {
            case no_cache:
                CHECK(cache["hits"] == 0, "%s: a disabled RAM cache reported hits", v.name);
                ring_only_bytes = std::max(ring_only_bytes, io["read_bytes"]);
                break;
            case tiny_cache:
                CHECK(cache["hits"] > 0 && cache["evictions"] > 0,
                      "%s: a 1 MiB cache should both hit and evict (hits %" PRIu64 ", evictions %" PRIu64 ")", v.name,
                      cache["hits"], cache["evictions"]);
                break;
            case budget_cache:
                CHECK(cache["hits"] > cache["misses"] && io["read_bytes"] < ring_only_bytes,
                      "%s: a cache that holds every expert should mostly hit and read less than the ring alone "
                      "(hits %" PRIu64 ", misses %" PRIu64 ", read %" PRIu64 " vs %" PRIu64 ")",
                      v.name, cache["hits"], cache["misses"], io["read_bytes"], ring_only_bytes);
                break;
        }
    }
    // Distinct tokens: the fixture's routing is not degenerate.
    std::vector<llama_token> distinct = base.tokens;
    std::sort(distinct.begin(), distinct.end());
    distinct.erase(std::unique(distinct.begin(), distinct.end()), distinct.end());
    CHECK(distinct.size() > 3, "the fixture generated only %zu distinct tokens", distinct.size());
    return 0;
}

// KV bytes the fixture needs per token: 3 layers x (K + V) x 2 KV heads x 32 dims x f16.
constexpr uint64_t FIXTURE_KV_PER_TOKEN = 3ull * 2 * 2 * 32 * 2;
constexpr uint64_t MiB                  = 1024ull * 1024;

uint64_t mib_ceil(uint64_t b) {
    return (b + MiB - 1) / MiB;
}

// The receipt's attention term for the fixture: the KV tensors and each cell's bookkeeping.
bool kv_attn_matches(uint64_t reported, uint32_t n_ctx) {
    return llama_kv_cells::host_bytes_per_cell() <= 256 &&
           reported == mib_ceil(uint64_t(n_ctx) * (FIXTURE_KV_PER_TOKEN + llama_kv_cells::host_bytes_per_cell()));
}

int test_context(const char * path) {
    // A budget far above what the fixture needs. Small headroom keeps the numbers readable.
    const std::string roomy = "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n";

    // Three context sizes: each loads, the receipt reserves exactly the KV for that size, and the expert cache gets
    // what is left after weights, KV, compute buffers and headroom.
    std::map<uint32_t, std::map<std::string, uint64_t>> receipts;
    for (const uint32_t n_ctx : { 1024u, 8192u, 32768u }) {
        loaded l = load(path, true, roomy, n_ctx);
        CHECK(l.ctx, "n_ctx=%u: load failed:\n%s", n_ctx, l.log.c_str());
        auto r = receipt_of(l.log);
        CHECK(!r.empty(), "n_ctx=%u: no receipt", n_ctx);
        if (!l.ctx || r.empty()) {
            continue;
        }
        std::printf("n_ctx=%u: %s\n", n_ctx, l.log.substr(l.log.rfind("[oh-ssd] receipt")).c_str());
        CHECK(r["ctx"] == n_ctx, "n_ctx=%u: receipt ctx=%" PRIu64, n_ctx, r["ctx"]);
        CHECK(kv_attn_matches(r["kv_attn"], n_ctx), "n_ctx=%u: kv_attn=%" PRIu64 " MiB, tensors alone are %" PRIu64,
              n_ctx, r["kv_attn"], mib_ceil(n_ctx * FIXTURE_KV_PER_TOKEN));
        CHECK(r["kv_swa"] == 0 && r["kv_index"] == 0 && r["recurrent"] == 0, "n_ctx=%u: unexpected non-attention state", n_ctx);
        CHECK(r["budget"] == 4096 && r["headroom"] == 16, "n_ctx=%u: budget or headroom not applied", n_ctx);
        // Rounding: every term is rounded up to whole MiB in the receipt, the cache down.
        // "process" is the rest of this process the host envelope charges (measured on Windows and macOS, 0 on Linux).
        const uint64_t fixed = r["dense_weights"] + r["context_host"] + r["compute_host"] + r["process"] + r["headroom"];
        CHECK(r.count("process") == 1, "n_ctx=%u: the receipt has no process term", n_ctx);
        CHECK(r["expert_cache"] + fixed <= 4096 + 5 && r["expert_cache"] + fixed + 5 >= 4096,
              "n_ctx=%u: expert_cache=%" PRIu64 " does not take what is left (fixed=%" PRIu64 ")", n_ctx, r["expert_cache"], fixed);
        CHECK(generate(l).ok, "n_ctx=%u: generation failed", n_ctx);
        receipts[n_ctx] = r;
    }
    if (receipts.size() == 3) {
        CHECK(receipts[1024]["expert_cache"] > receipts[32768]["expert_cache"],
              "a larger context did not shrink the expert cache");
    }

    // Too large: measure what 65536 tokens need, then offer a budget that is short by half the KV. The load fails
    // and names the largest context that fits; that context then loads within the same budget.
    loaded probe = load(path, true, roomy, 65536);
    auto   need  = receipt_of(probe.log);
    CHECK(probe.ctx && !need.empty(), "n_ctx=65536 probe failed:\n%s", probe.log.c_str());
    if (probe.ctx && !need.empty()) {
        const uint64_t fixed  = need["dense_weights"] + need["context_host"] + need["compute_host"] + need["process"] +
                               need["headroom"];
        const uint64_t budget = fixed - need["kv_attn"] / 2;
        llama_free(probe.ctx);
        probe.ctx = nullptr;
        llama_model_free(probe.model);
        probe.model = nullptr;
        const std::string tight =
            "engine.ram_budget_mib=" + std::to_string(budget) + "\nhost.headroom_mib=16\n";
        loaded too_big = load(path, true, tight, 65536);
        CHECK(!too_big.ctx, "n_ctx=65536 loaded in a %" PRIu64 " MiB budget it does not fit", budget);
        const char * fits_is = "the largest context that fits is ";
        const auto   at      = too_big.log.find(fits_is);
        CHECK(at != std::string::npos, "no largest-context message:\n%s", too_big.log.c_str());
        CHECK(contains(too_big.log, "(does not fit)"), "no receipt for the failed load");
        if (at != std::string::npos) {
            const uint32_t max_ctx = uint32_t(std::strtoul(too_big.log.c_str() + at + std::strlen(fits_is), nullptr, 10));
            std::printf("n_ctx=65536 in %" PRIu64 " MiB: refused, largest context that fits = %u\n", budget, max_ctx);
            CHECK(max_ctx > 1024 && max_ctx < 65536 && max_ctx % 256 == 0, "max_ctx=%u out of range", max_ctx);
            loaded fits = load(path, true, tight, max_ctx);
            CHECK(fits.ctx, "the reported largest context %u does not load in the same budget:\n%s", max_ctx,
                  fits.log.c_str());
            if (fits.ctx) {
                CHECK(generate(fits).ok, "generation at the largest context failed");
            }
        }
    }

    // Nothing fits: the dense weights and buffers alone exceed the budget.
    loaded none = load(path, true, "engine.ram_budget_mib=1\nhost.headroom_mib=16\n", 1024);
    CHECK(!none.ctx, "loaded in a 1 MiB budget");
    CHECK(contains(none.log, "no context fits"), "no 'no context fits' message:\n%s", none.log.c_str());

    // Settings mistakes fail the load with the reason.
    loaded twice = load(path, true, "engine.expert_cache_mib=1\nengine.expert_cache_mib=2\n", 1024);
    CHECK(!twice.ctx && contains(twice.log, "given twice"), "a repeated key was accepted");
    loaded junk = load(path, true, "not a setting\n", 1024);
    CHECK(!junk.ctx && contains(junk.log, "not key=value"), "a malformed line was accepted");
    return 0;
}

// The most memory this process has held so far: committed bytes on Windows, resident bytes on Linux.
uint64_t peak_bytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS counters_now = {};
    return GetProcessMemoryInfo(GetCurrentProcess(), &counters_now, sizeof(counters_now)) ? uint64_t(counters_now.PeakPagefileUsage) : 0;
#else
    uint64_t kib = 0;
    if (FILE * f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "VmHWM:", 6) == 0) {
                kib = std::strtoull(line + 6, nullptr, 10);
                break;
            }
        }
        std::fclose(f);
    }
    return kib * 1024;
#endif
}

// Runs this test binary again in a new process, and returns its exit status.
int run_self(const std::string & self, const std::string & args) {
    std::string command = "\"" + self + "\" " + args;
#ifdef _WIN32
    command = "\"" + command + "\""; // cmd.exe strips the outer pair of quotes
#endif
    std::fflush(stdout);
    std::fflush(stderr);
    return std::system(command.c_str());
}

// Admission comes before allocation. The KV of a 1,048,576-token context is 768 MiB and the budget is 640 MiB, roomy
// enough for a smaller context on any platform:
// the load must be refused with the receipt and the largest context, and the peak must not rise by anything like the
// KV. Then the same context under a roomy budget must raise the peak by most of the KV, so the check can fail.
int test_admit(const char * path, const std::string & self) {
    constexpr uint32_t n_ctx = 1u << 20;
    const uint64_t     kv    = n_ctx * FIXTURE_KV_PER_TOKEN;
    {
        loaded warm = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(warm.ctx, "the small context failed to load:\n%s", warm.log.c_str());
    }
    const uint64_t before = peak_bytes();
    CHECK(before > 0, "this platform did not report a peak");
    {
        loaded refused = load(path, true, "engine.ram_budget_mib=640\nhost.headroom_mib=16\n", n_ctx);
        CHECK(!refused.ctx, "a context over budget was admitted");
        CHECK(contains(refused.log, "the largest context that fits is "), "no largest-context message:\n%s", refused.log.c_str());
        CHECK(contains(refused.log, "(does not fit)"), "no receipt for the refused context:\n%s", refused.log.c_str());
        auto r = receipt_of(refused.log);
        CHECK(kv_attn_matches(r["kv_attn"], n_ctx), "the refusal sized the KV as %" PRIu64 " MiB, tensors alone are %" PRIu64, r["kv_attn"], mib_ceil(kv));
    }
    const uint64_t after = peak_bytes();
    std::printf("KV %" PRIu64 " MiB: peak rose %" PRIu64 " MiB while refusing\n", mib_ceil(kv), mib_ceil(after - before));
    CHECK(after - before < kv / 4, "refusing raised the peak by %" PRIu64 " MiB of a %" PRIu64 " MiB KV",
          mib_ceil(after - before), mib_ceil(kv));
    {
        loaded admitted = load(path, true, "engine.ram_budget_mib=16384\nhost.headroom_mib=16\n", n_ctx);
        CHECK(admitted.ctx, "the same context failed under a roomy budget:\n%s", admitted.log.c_str());
    }
    const uint64_t control = peak_bytes();
    std::printf("admitted: peak rose %" PRIu64 " MiB\n", mib_ceil(control - after));
    CHECK(control - after > kv / 2, "an admitted context raised the peak by only %" PRIu64 " MiB", mib_ceil(control - after));

    // A context so large that its attention mask (256 MiB here) is over the budget by itself: the mask shrinks with
    // the context, so the refusal still has to name a size, and that size has to load.
    uint64_t process = 0;
    {
        loaded baseline = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(baseline.ctx, "the small context failed to load:\n%s", baseline.log.c_str());
        process = receipt_of(baseline.log)["process"];
    }
    constexpr uint32_t mask_heavy_ctx = 1u << 21;
    const std::string  tight = "engine.ram_budget_mib=" + std::to_string(process + 192) + "\nhost.headroom_mib=16\n";
    uint32_t           named = 0;
    {
        loaded refused = load(path, true, tight, mask_heavy_ctx, 64);
        CHECK(!refused.ctx, "a context with an oversized attention mask was admitted");
        const std::string marker = "the largest context that fits is ";
        const auto        at     = refused.log.find(marker);
        CHECK(at != std::string::npos, "no largest-context message:\n%s", refused.log.c_str());
        if (at != std::string::npos) {
            named = uint32_t(std::strtoul(refused.log.c_str() + at + marker.size(), nullptr, 10));
        }
    }
    CHECK(named > 0, "the refusal named no size");
    if (named > 0) {
        // A launcher retries in a new process, with the same budget.
        const int status = run_self(self, "load-at \"" + std::string(path) + "\" " + std::to_string(process + 192) + " " +
                                              std::to_string(named) + " 64");
        CHECK(status == 0, "the named size %u did not load in a fresh process (exit status %d)", named, status);
    }
    return 0;
}

// The child side of test_admit's retry.
int test_load_at(const char * path, const char * budget_mib, uint32_t n_ctx, uint32_t n_ubatch) {
    const std::string settings = "engine.ram_budget_mib=" + std::string(budget_mib) + "\nhost.headroom_mib=16\n";
    loaded            fits     = load(path, true, settings, n_ctx, n_ubatch);
    CHECK(fits.ctx, "%u tokens did not load in %s MiB:\n%s", n_ctx, budget_mib, fits.log.c_str());
    return 0;
}

// 134,217,728 tokens: at about 50 bytes a cell the per-token bookkeeping alone is over 6 GiB, and the KV is 96 GiB.
// Run on its own so nothing earlier has raised the peak: the refusal must leave it where the small load put it.
int test_huge(const char * path) {
    constexpr uint32_t n_ctx = 1u << 27;
    const uint64_t     kv    = uint64_t(n_ctx) * FIXTURE_KV_PER_TOKEN;
    {
        loaded warm = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(warm.ctx, "the small context failed to load:\n%s", warm.log.c_str());
    }
    const uint64_t before = peak_bytes();
    CHECK(before > 0, "this platform did not report a peak");
    {
        loaded refused = load(path, true, "engine.ram_budget_mib=1024\nhost.headroom_mib=16\n", n_ctx);
        CHECK(!refused.ctx, "a context far over budget was admitted");
        CHECK(contains(refused.log, "the largest context that fits is "), "no largest-context message:\n%s", refused.log.c_str());
        auto r = receipt_of(refused.log);
        CHECK(kv_attn_matches(r["kv_attn"], n_ctx), "the refusal sized the KV as %" PRIu64 " MiB, tensors alone are %" PRIu64, r["kv_attn"], mib_ceil(kv));
    }
    const uint64_t after = peak_bytes();
    std::printf("peak rose %" PRIu64 " MiB while refusing %u tokens\n", mib_ceil(after - before), n_ctx);
    CHECK(after - before < 64ull << 20, "refusing raised the peak by %" PRIu64 " MiB", mib_ceil(after - before));
    return 0;
}

// A batch size of 1,073,741,824 tokens: output_ids alone would be 4 GiB. Nothing sized by it may exist before the
// refusal. With n_ubatch that large too, the graph reservation is refused before it is built, when there is no compute
// size yet to name a context from.
int test_batch(const char * path) {
    constexpr uint32_t n = 1u << 30;
    const std::string  budget = "engine.ram_budget_mib=1024\nhost.headroom_mib=16\n";
    {
        loaded warm = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(warm.ctx, "the small context failed to load:\n%s", warm.log.c_str());
    }
    const uint64_t before = peak_bytes();
    CHECK(before > 0, "this platform did not report a peak");
    {
        loaded refused = load(path, true, budget, n, 16, 1, n);
        CHECK(!refused.ctx, "a batch size over budget was admitted");
        CHECK(contains(refused.log, "the largest context that fits is "), "no largest-context message:\n%s", refused.log.c_str());
        CHECK(contains(refused.log, "(does not fit)"), "no receipt for the refused context:\n%s", refused.log.c_str());
    }
    {
        loaded refused = load(path, true, budget, n, n, 1, n);
        CHECK(!refused.ctx, "a micro-batch size over budget was admitted");
        CHECK(contains(refused.log, "no context fits"), "no refusal of the graph reservation:\n%s", refused.log.c_str());
    }
    const uint64_t after = peak_bytes();
    std::printf("peak rose %" PRIu64 " MiB while refusing a batch of %u tokens\n", mib_ceil(after - before), n);
    CHECK(after - before < 64ull << 20, "refusing raised the peak by %" PRIu64 " MiB", mib_ceil(after - before));
    return 0;
}

llama_sampler * greedy_chain() {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_greedy());
    return chain;
}

// A backend sampler builds a chain of nodes for every output a sequence may have, and by default that is n_batch of
// them, so the graph's bookkeeping has to be charged before it is built.
int test_sampler(const char * path) {
    constexpr uint32_t n = 1u << 24;
    {
        loaded warm = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(warm.ctx, "the small context failed to load:\n%s", warm.log.c_str());
    }
    const uint64_t before = peak_bytes();
    CHECK(before > 0, "this platform did not report a peak");
    llama_sampler * chain = greedy_chain();
    {
        loaded refused = load(path, true, "engine.ram_budget_mib=1024\nhost.headroom_mib=16\n", n, 16, 1, n, chain, 0);
        CHECK(!refused.ctx, "sampler chains over budget were admitted");
        CHECK(contains(refused.log, "no context fits"), "no refusal of the graph reservation:\n%s", refused.log.c_str());
    }
    llama_sampler_free(chain);
    const uint64_t after = peak_bytes();
    std::printf("peak rose %" PRIu64 " MiB while refusing sampler chains for %u outputs\n", mib_ceil(after - before), n);
    CHECK(after - before < 64ull << 20, "refusing raised the peak by %" PRIu64 " MiB", mib_ceil(after - before));
    return 0;
}

// Windows commits the scheduler's capacity when it is allocated, touched or not. Sampler chains for 32768 outputs ask
// for 99,328 nodes, about 2 GiB of scratch, while their graph metadata fits a 1 GiB budget easily. The context has to
// be refused before that commit, and a normal context under the same budget still has to load.
int test_sched_commit(const char * path) {
    constexpr uint32_t n = 1u << 15;
    {
        loaded warm = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(warm.ctx, "the small context failed to load:\n%s", warm.log.c_str());
    }
    const uint64_t before = peak_bytes();
    CHECK(before > 0, "this platform did not report a peak");
    llama_sampler * chain = greedy_chain();
    {
        loaded refused = load(path, true, "engine.ram_budget_mib=1024\nhost.headroom_mib=16\n", n, 16, 1, n, chain, 0);
        CHECK(!refused.ctx, "a scheduler over budget was admitted");
        CHECK(contains(refused.log, "no context fits"), "no refusal of the scheduler's capacity:\n%s", refused.log.c_str());
    }
    llama_sampler_free(chain);
    const uint64_t after = peak_bytes();
    std::printf("peak commit rose %" PRIu64 " MiB while refusing a scheduler of %u outputs\n", mib_ceil(after - before), n);
    CHECK(after - before < 64ull << 20, "refusing raised the peak by %" PRIu64 " MiB", mib_ceil(after - before));

    loaded l = load(path, true, "engine.ram_budget_mib=1024\nhost.headroom_mib=16\n", 1024);
    CHECK(l.ctx, "a normal context failed to load under the same budget:\n%s", l.log.c_str());
    const auto nodes = [&](const char * marker) -> uint64_t {
        const auto at = l.log.find(marker);
        return at == std::string::npos ? 0 : std::strtoull(l.log.c_str() + at + std::strlen(marker), nullptr, 10);
    };
    const uint64_t priced   = nodes("[oh-ssd] scheduler capacity: priced ");
    const uint64_t reserved = nodes("[oh-ssd] scheduler capacity: reserved ");
    std::printf("scheduler capacity: priced %" PRIu64 ", reserved %" PRIu64 " nodes\n", priced, reserved);
    CHECK(priced > 0 && priced == reserved, "priced %" PRIu64 " nodes for a scheduler of %" PRIu64, priced, reserved);
    return 0;
}

// The first output buffer has a row per sequence. Admission charges it by arithmetic, so that has to match what the
// buffer then takes, with and without a sampler's extra rows.
int test_outputs(const char * path) {
    constexpr uint32_t n_seq = 256;
    llama_sampler *    chain = greedy_chain();
    for (llama_sampler * sampler : { (llama_sampler *) nullptr, chain }) {
        loaded l = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1u << 16, 1, n_seq, 0, sampler);
        CHECK(l.ctx, "%u sequences failed to load:\n%s", n_seq, l.log.c_str());
        const auto bytes = [&](const char * marker) -> uint64_t {
            const auto at = l.log.find(marker);
            return at == std::string::npos ? 0 : std::strtoull(l.log.c_str() + at + std::strlen(marker), nullptr, 10);
        };
        const uint64_t charged   = bytes("[oh-ssd] first output buffer: charged ");
        const uint64_t allocated = bytes("[oh-ssd] first output buffer: allocated ");
        std::printf("first output buffer: charged %" PRIu64 ", allocated %" PRIu64 " bytes\n", charged, allocated);
        CHECK(allocated >= uint64_t(n_seq) * 256 * sizeof(float), "the buffer took only %" PRIu64 " bytes", allocated);
        CHECK(charged >= allocated && charged - allocated < 4096, "charged %" PRIu64 " bytes for a buffer of %" PRIu64, charged, allocated);
    }
    llama_sampler_free(chain);
    return 0;
}

// A unified cache can have every sequence hold every cell, each with its own tree node. A cell of a split cache belongs
// to one sequence, and one sequence costs what it always did.
int test_cells(const char * path) {
    loaded l = load(path, false, "", 1024);
    CHECK(l.ctx, "load failed:\n%s", l.log.c_str());
    if (!l.ctx) {
        return 1;
    }
    constexpr uint32_t n_cells = 1u << 14;
    const llama_model & model  = *l.model;
    const auto charged = [&](bool unified, uint32_t n_seq) {
        llama_kv_cache sizing(model, true, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, unified, n_cells, n_seq, 0,
                              LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr, "sizing");
        uint64_t host = 0;
        for (const auto & [buft, size] : sizing.memory_breakdown()) {
            if (ggml_backend_buft_is_host(buft)) {
                host += size;
            }
        }
        return host;
    };
    const uint64_t one   = charged(true, 1);
    const uint64_t many  = charged(true, 256);
    const uint64_t split = charged(false, 4);
    const uint64_t node  = llama_kv_cells::host_bytes_per_cell(2) - llama_kv_cells::host_bytes_per_cell(1);
    CHECK(llama_kv_cells::host_bytes_per_cell(1) <= 256, "one sequence now costs %" PRIu64 " bytes a cell", llama_kv_cells::host_bytes_per_cell(1));
    CHECK(many - one == uint64_t(n_cells) * 255 * node, "256 sequences on one unified cache added %" PRIu64 " bytes", many - one);
    CHECK(split + (1ull << 20) > 4 * one && split < 4 * one + (1ull << 20),
          "4 split sequences charged %" PRIu64 " bytes, one charged %" PRIu64, split, one);
    return 0;
}

// A cache that is only sized and borrows a live cache's cells (a draft model's cache measured beside its target) must
// not touch them: a target left with no cells cannot place its next batch.
int test_borrow(const char * path) {
    loaded l = load(path, false, "", 1024);
    CHECK(l.ctx, "load failed:\n%s", l.log.c_str());
    if (!l.ctx) {
        return 1;
    }
    constexpr uint32_t n_cells = 4096;
    const llama_model & model  = *l.model;
    llama_kv_cache      target(model, false, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, n_cells, 1, 0,
                               LLAMA_SWA_TYPE_NONE, nullptr, nullptr, nullptr, nullptr, "target");
    CHECK(target.get_cells(0).size() == n_cells, "the live cache has %u cells before sizing", target.get_cells(0).size());
    const auto charged = [](const llama_kv_cache & cache) {
        uint64_t host = 0;
        for (const auto & [buft, size] : cache.memory_breakdown()) {
            if (ggml_backend_buft_is_host(buft)) {
                host += size;
            }
        }
        return host;
    };
    const uint64_t target_bytes = charged(target);
    {
        llama_kv_cache sizing(model, true, model.hparams, GGML_TYPE_F16, GGML_TYPE_F16, false, false, true, n_cells, 1, 0,
                              LLAMA_SWA_TYPE_NONE, &target, nullptr, nullptr, nullptr, "sizing");
        CHECK(sizing.get_size() == n_cells, "the sizing cache reports %u cells", sizing.get_size());
        CHECK(charged(sizing) + uint64_t(n_cells) * llama_kv_cells::host_bytes_per_cell() / 2 <= target_bytes,
              "the sizing cache charged the bookkeeping of cells it only borrows");
    }
    CHECK(target.get_size() == n_cells, "sizing changed the live cache's size to %u", target.get_size());
    CHECK(target.get_cells(0).size() == n_cells, "sizing left the live cache %u cells", target.get_cells(0).size());
    return 0;
}

// One token per ubatch and 256 sequences: the prompt graph has one output, the decode graph one per sequence, so a
// budget check that measured only the prompt graph would admit less than the real reservation takes.
int test_reserve(const char * path) {
    loaded l = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1u << 16, 1, 256);
    CHECK(l.ctx, "256 sequences failed to load:\n%s", l.log.c_str());
    const auto bytes = [&](const char * marker) -> uint64_t {
        const auto at = l.log.find(marker);
        return at == std::string::npos ? 0 : std::strtoull(l.log.c_str() + at + std::strlen(marker), nullptr, 10);
    };
    const uint64_t sized     = bytes("[oh-ssd] host compute buffers: sized ");
    const uint64_t allocated = bytes("[oh-ssd] host compute buffers: allocated ");
    std::printf("compute buffers: sized %" PRIu64 ", allocated %" PRIu64 " bytes\n", sized, allocated);
    CHECK(allocated > 0, "no allocated size in the log:\n%s", l.log.c_str());
    CHECK(sized >= allocated, "sizing saw %" PRIu64 " bytes of the %" PRIu64 " the reservations took", sized, allocated);
    return 0;
}

// F32 experts: the model loads as ordinary weights, and streaming them is refused at load with the reason.
int test_refuse(const char * path) {
    {
        loaded resident = load(path, false, "", 1024);
        CHECK(resident.ctx, "resident load failed:\n%s", resident.log.c_str());
    }
    loaded streamed = load(path, true, "engine.expert_cache_mib=0\n", 1024);
    CHECK(!streamed.ctx, "unsupported experts were admitted");
    CHECK(contains(streamed.log, "not supported yet") && contains(streamed.log, "is f32"),
          "the refusal does not name the type:\n%s", streamed.log.c_str());
    return 0;
}

// MTP: the file holds a fifth layer of experts (3.75 MiB in all) that a default load leaves uncreated, so only the
// four trunk layers (3 MiB) may be indexed and the model must load and stream as usual.
int test_mtp(const char * path) {
    test_equal(path, false);
    loaded l = load(path, true, "engine.ram_budget_mib=2048\nhost.headroom_mib=16\n", 1024);
    CHECK(l.ctx, "MTP load failed:\n%s", l.log.c_str());
    auto r = receipt_of(l.log);
    CHECK(r["experts_on_disk"] == 3, "indexed %" PRIu64 " MiB of experts, expected 3 (the MTP layer must stay out)",
          r["experts_on_disk"]);
    return 0;
}

// One routed layer's expert types, and which types are written per layer. Recorded from the GGUF headers of
// GLM-5.3-Flash-UD-Q2_K_XL (tensor tables only): the first mix is 39 of its 42 trunk layers, the other two are 3.
struct glm_quant_mix {
    ggml_type up, gate, down;
};
constexpr glm_quant_mix GLM_MIXES[] = {
    { GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ3_XXS },
    { GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_XS },
    { GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ4_XS },
};
constexpr glm_quant_mix GLM_MTP_MIX = { GGML_TYPE_Q2_K, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K }; // the real file's layer 45

// floating writes every routed expert as F16 instead of the real file's types.
bool write_glm_quantized(gguf_context * in, ggml_context * meta, const std::string & out_path, bool mixed, bool floating = false) {
    std::mt19937 rng(42); // the raw output is the same on every platform, unlike the standard distributions
    std::vector<std::vector<uint8_t>> keep;
    gguf_context * out = gguf_init_empty();
    gguf_set_kv(out, in);
    keep.reserve(size_t(gguf_get_n_tensors(in)));
    for (int64_t i = 0; i < gguf_get_n_tensors(in); ++i) {
        const char * name = gguf_get_tensor_name(in, i);
        ggml_tensor * t   = ggml_get_tensor(meta, name);
        gguf_add_tensor(out, t);
        int  layer = -1;
        char role[16] = {};
        if (!std::strstr(name, "_exps.weight") || std::sscanf(name, "blk.%d.ffn_%15[a-z]", &layer, role) != 2) {
            continue;
        }
        const glm_quant_mix mix = layer == 4 ? GLM_MTP_MIX : GLM_MIXES[layer - 1];
        ggml_type type = !std::strcmp(role, "up") ? mix.up : !std::strcmp(role, "gate") ? mix.gate : mix.down;
        if (mixed && layer == 2 && !std::strcmp(role, "gate")) {
            type = GGML_TYPE_IQ2_XS; // layer 2's up experts are IQ3_XXS
        }
        const size_t n_per_row = size_t(t->ne[0]);
        const size_t n_rows    = size_t(t->ne[1] * t->ne[2]);
        const float  bound     = float(std::sqrt(3.0 / double(n_per_row)));
        std::vector<float> values(n_per_row * n_rows);
        for (float & v : values) {
            v = (float(rng()) / 4294967296.0f * 2.0f - 1.0f) * bound;
        }
        if (floating && layer != 4) {
            keep.emplace_back(values.size() * sizeof(ggml_fp16_t));
            ggml_fp32_to_fp16_row(values.data(), reinterpret_cast<ggml_fp16_t *>(keep.back().data()), values.size());
            gguf_set_tensor_type(out, name, GGML_TYPE_F16);
            gguf_set_tensor_data(out, name, keep.back().data());
            continue;
        }
        // IQ2_XS will not quantize without importance weights.
        const std::vector<float> importance(n_per_row, 1.0f);
        keep.emplace_back(ggml_row_size(type, int64_t(n_per_row)) * n_rows);
        const size_t written = ggml_quantize_chunk(type, values.data(), keep.back().data(), 0, n_rows, n_per_row,
                                                   ggml_quantize_requires_imatrix(type) ? importance.data() : nullptr);
        CHECK(written == keep.back().size(), "%s: wrote %zu bytes, sized %zu", name, written, keep.back().size());
        gguf_set_tensor_type(out, name, type);
        gguf_set_tensor_data(out, name, keep.back().data());
    }
    const bool ok = gguf_write_to_file(out, out_path.c_str(), false);
    CHECK(ok, "cannot write %s", out_path.c_str());
    gguf_free(out);
    return ok;
}

// Python cannot produce IQ types, so the base file's placeholders are filled and quantized here.
int make_glm_fixtures(const std::string & base) {
    ggml_context *   meta   = nullptr;
    gguf_init_params params = { /*.no_alloc =*/ false, /*.ctx =*/ &meta };
    gguf_context *   in     = gguf_init_from_file(base.c_str(), params);
    CHECK(in, "cannot read %s", base.c_str());
    if (!in) {
        return 1;
    }
    const auto dir = base.substr(0, base.find_last_of("/\\") + 1);
    write_glm_quantized(in, meta, dir + "ssd-glm-iq.gguf", false);
    write_glm_quantized(in, meta, dir + "ssd-glm-iq-mixed.gguf", true);
    write_glm_quantized(in, meta, dir + "ssd-glm-f16.gguf", false, true);
    ggml_quantize_free();
    gguf_free(in);
    ggml_free(meta);
    return 0;
}

// The types and row widths of the real file's routed experts (up and gate are 4096 wide, down is 2048) are all
// ones the sink can compute, and up and gate always match. The fixture's rows are narrower, so this is what ties the
// streaming test to the real file.
void check_glm_header_facts() {
    using supported_fn = bool (*)(ggml_type, int64_t);
    auto * reg = ggml_backend_reg_by_name("CPU");
    auto   supported = reg ? reinterpret_cast<supported_fn>(ggml_backend_reg_get_proc_address(reg, "kimi_cpu_sink_expert_type_supported"))
                           : nullptr;
    CHECK(supported, "the CPU backend does not report which expert types it can compute");
    if (!supported) {
        return;
    }
    for (const auto & m : GLM_MIXES) {
        CHECK(m.up == m.gate, "a recorded mix has up %s and gate %s", ggml_type_name(m.up), ggml_type_name(m.gate));
        CHECK(supported(m.up, 4096), "the sink cannot compute %s up and gate experts", ggml_type_name(m.up));
        CHECK(supported(m.down, 2048), "the sink cannot compute %s down experts", ggml_type_name(m.down));
    }
}

int test_glm(const char * path) {
    check_glm_header_facts();
    // The file also holds a fifth layer of experts, in the MTP head's own types; a default load must leave them out.
    test_equal(path, true);

    const std::string roomy = "engine.ram_budget_mib=2048\nhost.headroom_mib=16\n";
    {
        loaded l = load(path, true, roomy, 1024);
        CHECK(l.ctx, "GLM load failed:\n%s", l.log.c_str());
        CHECK(contains(l.log, "experts_on_disk="), "no receipt:\n%s", l.log.c_str());
    }
    {
        // Asking for the head with SSD experts is refused by name, with a normal load failure.
        loaded l = load(path, true, roomy, 1024, 16, 1, 0, nullptr, 1, true);
        CHECK(!l.ctx, "an SSD load with the MTP head was admitted");
        CHECK(contains(l.log, "draft (MTP) head is not supported"), "the refusal does not name the head:\n%s", l.log.c_str());
    }
    return 0;
}

int test_glm_mixed(const char * path) {
    {
        loaded resident = load(path, false, "", 1024);
        CHECK(resident.ctx, "resident load failed:\n%s", resident.log.c_str());
    }
    loaded streamed = load(path, true, "engine.expert_cache_mib=0\n", 1024);
    CHECK(!streamed.ctx, "up and gate experts of different types were admitted");
    CHECK(contains(streamed.log, "this quantization is not supported yet") && contains(streamed.log, "up=iq3_xxs") &&
              contains(streamed.log, "gate=iq2_xs"),
          "the refusal does not name the types:\n%s", streamed.log.c_str());
    return 0;
}

// Indexer bytes per cell: one indexer layer's key rows (key, gate and pooled, F16), each cell's bookkeeping, and the
// k-pool layout and state, which do not exist until the first decode: rep_gen 4, then twice (entry 8, and 4 each for
// the pool offset and is_new of one pool in 4 cells).
uint64_t glm_index_bytes_per_cell() {
    constexpr uint64_t key_row = 16 * 3 * 2;
    constexpr uint64_t kpool   = 4 + 2 * (8 + 1 + 1);
    return key_row + llama_kv_cells::host_bytes_per_cell() + kpool;
}

int test_glm_admit(const char * path, const std::string & self) {
    const std::string budget = "engine.ram_budget_mib=1024\nhost.headroom_mib=16\n";
    {
        loaded warm = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(warm.ctx, "the small context failed to load:\n%s", warm.log.c_str());
    }
    const uint64_t before = peak_bytes();
    CHECK(before > 0, "this platform did not report a peak");
    constexpr uint32_t n_ctx = 1u << 27;
    {
        loaded refused = load(path, true, budget, n_ctx);
        CHECK(!refused.ctx, "a context far over budget was admitted");
        // Here the compute buffers alone are 35 times the budget, so no share of them can be held fixed and leave room.
        CHECK(contains(refused.log, "try a smaller -c"), "the refusal does not say a smaller context may fit:\n%s", refused.log.c_str());
        auto r = receipt_of(refused.log);
        CHECK(r["kv_index"] == mib_ceil(uint64_t(n_ctx) * glm_index_bytes_per_cell()),
              "the refusal priced the indexer at %" PRIu64 " MiB, expected %" PRIu64, r["kv_index"],
              mib_ceil(uint64_t(n_ctx) * glm_index_bytes_per_cell()));
    }
    const uint64_t after = peak_bytes();
    std::printf("peak rose %" PRIu64 " MiB while refusing %u tokens\n", mib_ceil(after - before), n_ctx);
    CHECK(after - before < 64ull << 20, "refusing raised the peak by %" PRIu64 " MiB", mib_ceil(after - before));

    // A size the refusal names has to load, in a new process with the same budget.
    uint64_t process = 0;
    {
        loaded baseline = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", 1024);
        CHECK(baseline.ctx, "the small context failed to load:\n%s", baseline.log.c_str());
        process = receipt_of(baseline.log)["process"];
    }
    const std::string tight = "engine.ram_budget_mib=" + std::to_string(process + 512) + "\nhost.headroom_mib=16\n";
    uint32_t          named = 0;
    {
        loaded refused = load(path, true, tight, 1u << 20);
        CHECK(!refused.ctx, "a context over a tight budget was admitted");
        const std::string marker = "the largest context that fits is ";
        const auto        at     = refused.log.find(marker);
        CHECK(at != std::string::npos, "no largest-context message:\n%s", refused.log.c_str());
        if (at != std::string::npos) {
            named = uint32_t(std::strtoul(refused.log.c_str() + at + marker.size(), nullptr, 10));
        }
    }
    CHECK(named > 0, "the refusal named no size");
    if (named > 0) {
        const int status = run_self(self, "load-at \"" + std::string(path) + "\" " + std::to_string(process + 512) + " " +
                                              std::to_string(named) + " 16");
        CHECK(status == 0, "the named size %u did not load in a fresh process (exit status %d)", named, status);
    }

    // The state a decode will build is charged and known not to exist yet: the indexer's k-pool lists, and for the
    // attention and indexer caches the cell trees, the sequence rows and the generation.
    {
        constexpr uint32_t n_cells = 1024;
        loaded             charged = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", n_cells);
        const std::string  marker  = "[oh-ssd] context memory charged but not allocated yet: ";
        const auto         at      = charged.log.find(marker);
        CHECK(charged.ctx && at != std::string::npos, "no unallocated line:\n%s", charged.log.c_str());
        if (at != std::string::npos) {
            const uint64_t unallocated = std::strtoull(charged.log.c_str() + at + marker.size(), nullptr, 10);
            const uint64_t expected    = uint64_t(n_cells) * (4 + 2 * (8 + 1 + 1) + 2 * (48 + 56 + LLAMA_MAX_SEQ / 8 + 4));
            std::printf("not allocated yet: %" PRIu64 " bytes, at least %" PRIu64 " expected\n", unallocated, expected);
            CHECK(unallocated >= expected, "only %" PRIu64 " bytes noted as not allocated, expected %" PRIu64, unallocated, expected);
        }
    }

    // A ubatch past the model's node floor charges three graph arenas, and the two that decoding builds are in the note.
    {
        constexpr uint32_t n_cells  = 1024;
        const std::string  settings = "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n";
        const auto number = [](const std::string & log, const std::string & marker) -> uint64_t {
            const auto at = log.find(marker);
            return at == std::string::npos ? UINT64_MAX : std::strtoull(log.c_str() + at + marker.size(), nullptr, 10);
        };
        const auto read = [&](uint32_t n_ubatch, uint64_t & note, uint64_t & built) {
            loaded l = load(path, true, settings, n_cells, n_ubatch, 1, n_cells);
            CHECK(l.ctx, "ubatch %u did not load:\n%s", n_ubatch, l.log.c_str());
            note  = number(l.log, "[oh-ssd] context memory charged but not allocated yet: ");
            built = number(l.log, "[oh-ssd] graph arenas charged but not built yet: ");
        };
        uint64_t floor_note = 0, floor_arenas = 0, large_note = 0, large_arenas = 0;
        read(16, floor_note, floor_arenas);
        read(256, large_note, large_arenas);
        const uint64_t most = 2 * (40960 * ggml_tensor_overhead() + ggml_graph_overhead_custom(40960, false));
        std::printf("graph arenas not built yet: %" PRIu64 " bytes at ubatch 256\n", large_arenas);
        CHECK(floor_arenas == 0, "%" PRIu64 " bytes noted at the node floor", floor_arenas);
        CHECK(large_arenas != UINT64_MAX && large_arenas >= 2 * 1024 * ggml_tensor_overhead() && large_arenas <= most,
              "%" PRIu64 " bytes of arenas noted, at most %" PRIu64 " expected", large_arenas, most);
        CHECK(large_note == floor_note + large_arenas, "the note went from %" PRIu64 " to %" PRIu64 " for %" PRIu64 " bytes of arenas",
              floor_note, large_note, large_arenas);
    }

    // At the smallest context a smaller -c cannot help, so the refusal has to name something that can.
    {
        constexpr uint32_t smallest = 256;
        uint64_t           fixed    = 0;
        {
            loaded probe = load(path, true, "engine.ram_budget_mib=4096\nhost.headroom_mib=16\n", smallest);
            auto   need  = receipt_of(probe.log);
            CHECK(probe.ctx && !need.empty(), "the smallest context failed to load:\n%s", probe.log.c_str());
            fixed = need["dense_weights"] + need["context_host"] + need["compute_host"] + need["process"] + need["headroom"];
        }
        bool told = false;
        for (uint64_t mib = fixed; !told && mib > 17 && mib + 128 > fixed; --mib) {
            loaded refused = load(path, true, "engine.ram_budget_mib=" + std::to_string(mib) + "\nhost.headroom_mib=16\n", smallest);
            if (refused.ctx) {
                continue;
            }
            CHECK(!contains(refused.log, "try a smaller -c"), "%" PRIu64 " MiB: told to shrink the smallest context:\n%s", mib, refused.log.c_str());
            told = contains(refused.log, "no context fits these settings");
        }
        CHECK(told, "no refusal at the smallest context named another remedy (fixed = %" PRIu64 " MiB)", fixed);
    }

    // In a unified cache every sequence can list every cell, and the graph counts pools in 32 bits.
    {
        loaded refused = load(path, true, budget, 1u << 30, 16, 64, 0, nullptr, 1, false, true);
        CHECK(!refused.ctx, "more indexer pools than a 32-bit count can name were admitted");
        CHECK(contains(refused.log, "more indexer pools than this model's graph can address"),
              "the refusal does not name the pools:\n%s", refused.log.c_str());
    }
    return 0;
}

// The receipt's arithmetic and the refusal's advice, without a model.
int test_receipt() {
    constexpr uint64_t mib      = 1024 * 1024;
    constexpr uint64_t budget   = 4096 * mib;
    constexpr uint64_t headroom = 16 * mib;
    constexpr uint64_t owed     = 64 * mib;

    oh_ssd::memory_inputs base;
    base.context_host    = 100 * mib;
    base.compute_host    = 50 * mib;
    base.process_private = 400 * mib; // the buffers above and 250 MiB of everything else
    base.n_ctx           = 1024;
    base.n_seq           = 1;
    const auto compute = [&](const oh_ssd::memory_inputs & in) {
        return oh_ssd::compute_receipt(budget, headroom, 0, UINT64_MAX, in);
    };
    const auto plain = compute(base);
    CHECK(plain.fits && plain.process == 250 * mib, "process=%" PRIu64, plain.process);

    // State the first decodes will allocate is not in the measured process, so it has to come out of the cache.
    auto future = base;
    future.context_host += owed;
    future.context_unallocated = owed;
    const auto later = compute(future);
    CHECK(later.fits && later.process == plain.process, "the future state moved process to %" PRIu64, later.process);
    CHECK(later.cache + owed == plain.cache, "the future state left the cache at %" PRIu64 " of %" PRIu64, later.cache, plain.cache);

    // The same bytes already allocated are in the measurement, so the cache pays for them once, as it does above.
    auto held = base;
    held.context_host += owed;
    held.process_private += owed;
    const auto now = compute(held);
    CHECK(now.fits && now.process == plain.process && now.cache == later.cache,
          "allocated bytes moved process to %" PRIu64 " and the cache to %" PRIu64, now.process, now.cache);

    // Before the buffers exist the whole measurement is the process.
    auto sized = future;
    sized.buffers_allocated = false;
    const auto admitted = compute(sized);
    CHECK(admitted.process == sized.process_private, "admission's process was %" PRIu64, admitted.process);

    // A note larger than the charge cannot wrap the subtraction.
    auto odd = base;
    odd.context_unallocated = 2 * odd.context_host;
    CHECK(compute(odd).process == 350 * mib, "process was %" PRIu64, compute(odd).process);

    const oh_ssd::receipt none;
    oh_ssd::memory_inputs follows;
    follows.compute_follows_ctx = true;
    follows.n_seq               = 1;
    follows.n_ctx               = 512;
    follows.n_ctx_min           = 256;
    CHECK(contains(oh_ssd::refusal_advice(none, follows), "try a smaller -c"), "a context above the smallest was not told to shrink");

    follows.n_ctx            = 256;
    const std::string floor_ = oh_ssd::refusal_advice(none, follows);
    CHECK(!contains(floor_, "try a smaller -c") && contains(floor_, "no context fits these settings") &&
              contains(floor_, "engine.ram_budget_mib") && contains(floor_, "--batch-size") && contains(floor_, "--ubatch-size"),
          "the smallest context was advised: %s", floor_.c_str());
    CHECK(!contains(floor_, "-np"), "one sequence was told to lower -np: %s", floor_.c_str());

    // A model whose compute buffers stay put is told the same at the smallest context, and only there.
    follows.compute_follows_ctx = false;
    CHECK(contains(oh_ssd::refusal_advice(none, follows), "no context fits these settings"), "the smallest context was not named");
    follows.n_ctx = 512;
    CHECK(contains(oh_ssd::refusal_advice(none, follows), "alone exceed it"), "a larger context lost its advice");
    follows.compute_follows_ctx = true;

    // Split caches pad each sequence, so two sequences cannot go below 512.
    follows.n_seq            = 2;
    follows.n_ctx            = 512;
    follows.n_ctx_min        = 512;
    const std::string split  = oh_ssd::refusal_advice(none, follows);
    CHECK(!contains(split, "try a smaller -c") && contains(split, "-np"), "two split sequences were advised: %s", split.c_str());

    // A computed size is held to what the padding can produce: two split sequences come in multiples of 512.
    oh_ssd::memory_inputs tight;
    tight.buffers_allocated   = false;
    tight.compute_follows_ctx = true;
    tight.n_seq               = 2;
    tight.n_ctx_min           = 512;
    tight.n_ctx               = 512;
    tight.kv_linear_host      = 1024 * mib; // 2 MiB a token, with 600 MiB to spend: 300 tokens, which pads back to 512
    tight.context_host        = budget - headroom - 600 * mib + tight.kv_linear_host;
    const auto at_smallest    = compute(tight);
    CHECK(!at_smallest.fits && at_smallest.max_ctx == 0, "the smallest split context named %u tokens", at_smallest.max_ctx);
    const std::string computed = oh_ssd::refusal_advice(at_smallest, tight);
    CHECK(contains(computed, "no context fits these settings") && contains(computed, "-np"), "the computed receipt advised: %s", computed.c_str());

    tight.n_ctx        = 1024; // 1 MiB a token, with 800 MiB to spend: 800 tokens, of which 512 can exist
    tight.context_host = budget - headroom - 800 * mib + tight.kv_linear_host;
    const auto above   = compute(tight);
    CHECK(!above.fits && above.max_ctx == 512, "a split context of 1024 named %u tokens", above.max_ctx);

    oh_ssd::receipt named;
    named.max_ctx = 1024;
    CHECK(contains(oh_ssd::refusal_advice(named, follows), "the largest context that fits is 1024"), "a known size was not named");
    return 0;
}

} // namespace

// The bytes of a fixture's routed experts, from its tensor table.
static uint64_t expert_bytes(const char * path) {
    gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context *   gguf   = gguf_init_from_file(path, params);
    if (!gguf) {
        return 0;
    }
    uint64_t total = 0;
    for (int64_t i = 0; i < gguf_get_n_tensors(gguf); ++i) {
        if (std::strstr(gguf_get_tensor_name(gguf, i), "_exps.weight")) {
            total += gguf_get_tensor_size(gguf, i);
        }
    }
    gguf_free(gguf);
    return total;
}

// One streamed run on the GPU against the CPU's resident answer. On float experts the kernels agree to rounding, so
// tokens and every step's logits must match, and gaps is empty. Quantized experts are another matter: the GPU quantizes the
// operand of a product to Q8_1 where the CPU uses a coarser type, so plain llama.cpp on the GPU differs from the CPU on
// this fixture's random weights with no tier involved. For those, gaps holds that difference at each step, every run
// decodes the CPU's tokens so one different token cannot change every later step, and the whole run must be within
// twice the largest gap. The bound is for the run and not per step because the tier mixes the two kernels: its state
// is neither the CPU's nor the plain GPU's, so a single step can sit outside that step's gap and still be correct. The
// first step cannot hit, so the comparison only means something for the steps after it. Each routed layer of the
// fixture has its own expert type mix, so per-layer hits are hits for each mix.
//
// admit_all turns off the measured policy, so an expert's admission depends only on the routing. Only those runs can
// require hits, misses and evictions: under the production policy, a slow GPU hit makes refusing every admission and
// answering on the CPU the correct result, so those runs check the answer and that the summary is printed.
static void check_vram_run(const char * path, const run_result & base, const std::vector<float> & gaps, uint64_t budget, const char * name,
                           bool admit_all, bool short_slots) {
    const bool strict = gaps.empty();
    stderr_capture capture(std::string(path) + "." + name + ".stderr.txt");
    run_result     got;
    {
        std::string settings = "vram.budget=" + std::to_string(budget) + "\n";
        if (admit_all) {
            settings += "vram.admit_all=1\n";
        }
        loaded tier = load(path, true, settings, 1024, 16, 1, 0, nullptr, 1, false, false, true);
        CHECK(tier.ctx, "%s: VRAM tier load failed:\n%s", name, tier.log.c_str());
        if (!tier.ctx) {
            capture.finish();
            return;
        }
        CHECK(contains(tier.log, "[oh-ssd] VRAM MiB:"), "%s: no VRAM receipt in the log:\n%s", name, tier.log.c_str());
        got = generate(tier, strict ? nullptr : &base.tokens);
        take_log();
    }
    const std::string sink_log = capture.finish();
    CHECK(got.ok, "%s: generation with the VRAM tier failed", name);
    CHECK(!strict || got.tokens == base.tokens, "%s: the VRAM tier changed the tokens", name);
    CHECK(got.logits.size() == base.logits.size(), "%s: %zu logit rows, expected %zu", name, got.logits.size(), base.logits.size());
    float max_diff = 0.0f, largest = 0.0f, after_first = 0.0f;
    for (size_t s = 0; s < std::min(got.logits.size(), base.logits.size()); ++s) {
        float step_diff = 0.0f;
        for (size_t i = 0; i < base.logits[s].size(); ++i) {
            CHECK(std::isfinite(got.logits[s][i]), "%s: non-finite logit with the VRAM tier", name);
            step_diff = std::max(step_diff, std::fabs(got.logits[s][i] - base.logits[s][i]));
            largest   = std::max(largest, std::fabs(base.logits[s][i]));
        }
        max_diff = std::max(max_diff, step_diff);
        if (s > 0) {
            after_first = std::max(after_first, step_diff);
        }
    }
    float largest_gap = 0.0f;
    for (float gap : gaps) {
        largest_gap = std::max(largest_gap, gap);
    }
    const float allowed = strict ? std::max(1e-2f, 1e-3f * largest) : 2.0f * largest_gap + 1e-2f;
    CHECK(max_diff <= allowed, "%s: VRAM logits differ by %g, allowed %g (largest logit %g)", name, max_diff, allowed, largest);
    CHECK(contains(sink_log, "[oh-ssd] summary tier=vram "), "%s: no VRAM summary in the log:\n%s", name, sink_log.c_str());
    auto total = counters(sink_log, "[oh-ssd] summary tier=vram ");
    if (!short_slots) {
        CHECK(total["evictions"] == 0, "%s: evicted with room for every expert:\n%s", name, sink_log.c_str());
    }
    if (admit_all) {
        CHECK(total["hits"] > 0, "%s: the VRAM tier was never hit:\n%s", name, sink_log.c_str());
        CHECK(after_first > 0.0f, "%s: no step after the first differs from the CPU at all, so no hit ran on the GPU", name);
        if (short_slots) {
            CHECK(total["evictions"] > 0, "%s: nothing was evicted:\n%s", name, sink_log.c_str());
        }
        for (int layer = 1; layer <= int(std::size(GLM_MIXES)); ++layer) {
            const std::string marker = "[oh-ssd] summary tier=vram-layer layer=" + std::to_string(layer) + " ";
            auto in_layer = counters(sink_log, marker.c_str());
            CHECK(in_layer["hits"] > 0, "%s: layer %d was never hit:\n%s", name, layer, sink_log.c_str());
            if (short_slots) {
                CHECK(in_layer["misses"] > 0, "%s: layer %d never missed, so its slots were not short:\n%s", name, layer, sink_log.c_str());
            }
        }
    }
    std::printf("VRAM tier (%s): max |logit diff| = %g, after the first step %g, hits %" PRIu64 ", misses %" PRIu64 ", evictions %" PRIu64 "\n",
                name, max_diff, after_first, total["hits"], total["misses"], total["evictions"]);
}

// A tier whose first allocation fails steps aside: the run finishes on the CPU with exactly the CPU's answer, and the
// summary says so. Mode 1 fails the slot arena, mode 2 the compute engines' buffers.
static void check_vram_fallback(const char * path, const run_result & base, int mode, const char * name) {
    stderr_capture capture(std::string(path) + "." + name + ".stderr.txt");
    run_result     got;
    {
        const std::string settings = "vram.budget=67108864\nvram.fail_allocation=" + std::to_string(mode) + "\n";
        loaded tier = load(path, true, settings, 1024, 16, 1, 0, nullptr, 1, false, false, true);
        CHECK(tier.ctx, "%s: load failed:\n%s", name, tier.log.c_str());
        if (!tier.ctx) {
            capture.finish();
            return;
        }
        got = generate(tier);
        take_log();
    }
    const std::string sink_log = capture.finish();
    CHECK(got.ok, "%s: generation failed after the tier stepped aside", name);
    CHECK(got.tokens == base.tokens, "%s: the CPU fallback changed the tokens", name);
    CHECK(got.logits == base.logits, "%s: the CPU fallback changed the logits", name);
    CHECK(contains(sink_log, "[oh-ssd] summary tier=vram status=unavailable fallback=cpu hits=0 "), "%s: no status=unavailable line:\n%s", name,
          sink_log.c_str());
    CHECK(!contains(sink_log, "[oh-ssd] summary tier=vram hits="), "%s: a tier that stepped aside printed counters:\n%s", name, sink_log.c_str());
    std::printf("VRAM tier (%s): stepped aside, the CPU's answer unchanged\n", name);
}

// CUDA builds only (see tests/CMakeLists.txt): the VRAM tier gives the CPU's answer on a tiny fixture, both with room for
// every expert and when its slots are short and experts are evicted.
static void test_vram(const char * path) {
    loaded resident = load(path, false, "", 1024);
    CHECK(resident.ctx, "resident load failed:\n%s", resident.log.c_str());
    if (!resident.ctx) {
        return;
    }
    const run_result base = generate(resident);
    CHECK(base.ok, "resident generation failed");
    const uint64_t experts = expert_bytes(path);
    CHECK(experts > 0, "no routed experts in %s", path);
    std::vector<float> gaps;
    if (!std::strstr(path, "f16")) {
        loaded gpu = load(path, false, "", 1024, 16, 1, 0, nullptr, 1, false, false, true, 99);
        CHECK(gpu.ctx, "full GPU load failed:\n%s", gpu.log.c_str());
        if (!gpu.ctx) {
            return;
        }
        const run_result reference = generate(gpu, &base.tokens);
        CHECK(reference.ok && reference.logits.size() == base.logits.size(), "full GPU generation failed");
        if (!reference.ok || reference.logits.size() != base.logits.size()) {
            return;
        }
        float largest_gap = 0.0f;
        for (size_t s = 0; s < base.logits.size(); ++s) {
            float gap = 0.0f;
            for (size_t i = 0; i < base.logits[s].size(); ++i) {
                gap = std::max(gap, std::fabs(reference.logits[s][i] - base.logits[s][i]));
            }
            gaps.push_back(gap);
            largest_gap = std::max(largest_gap, gap);
        }
        std::printf("plain GPU against CPU on the CPU's tokens: first step %g, largest step %g\n", gaps[0], largest_gap);
    }
    const uint64_t roomy = 64 * 1024 * 1024;
    check_vram_run(path, base, gaps, roomy, "vram", false, false);
    check_vram_run(path, base, gaps, roomy, "vram-admit-all", true, false);
    if (experts > 0) {
        // About a quarter of each layer's experts fit: 8 slots for 32 experts, 4 of them selected per token.
        check_vram_run(path, base, gaps, experts / 4, "vram-tight", false, true);
        check_vram_run(path, base, gaps, experts / 4, "vram-admit-all-tight", true, true);
    }
    check_vram_fallback(path, base, 1, "vram-fallback-arena");
    check_vram_fallback(path, base, 2, "vram-fallback-engines");
}

// The VRAM tier's admission arithmetic: no device needed.
static void test_vram_receipt() {
    const uint64_t mib = 1024ull * 1024ull;
    oh_ssd::vram_inputs in;
    in.budget = 1000 * mib;
    in.reserve = 3072 * mib;
    in.margin = 64 * mib;
    in.overhead = 8 * mib;
    in.free = 8000 * mib;
    in.total = 24000 * mib;
    in.n_expert = 288;
    in.layer_bytes = {10 * mib, 20 * mib};

    auto out = oh_ssd::compute_vram_receipt(in);
    CHECK(out.fits, "a budget inside free VRAM fits: %s", out.text.c_str());
    CHECK(out.keep == (3072 + 64 + 8) * mib, "keep is the reserve, the margin and the tier's buffers");
    CHECK(out.max_budget == (8000 - 3144) * mib, "the largest budget is free less keep");
    CHECK(out.min_budget == 30 * mib, "the smallest budget is one slot per layer");
    CHECK(out.slots == 67 && out.slot_bytes == 1000 * mib, "slots %llu bytes %llu",
          (unsigned long long)out.slots, (unsigned long long)out.slot_bytes);

    auto over = in;
    over.budget = out.max_budget + 1;
    CHECK(!oh_ssd::compute_vram_receipt(over).fits, "more than free VRAM less keep is refused");
    over.budget = out.max_budget;
    CHECK(oh_ssd::compute_vram_receipt(over).fits, "exactly the largest budget fits");

    auto under = in;
    under.budget = 30 * mib - 1;
    CHECK(!oh_ssd::compute_vram_receipt(under).fits, "below one slot per layer is refused");

    auto unread = in;
    unread.total = 0;
    CHECK(!oh_ssd::compute_vram_receipt(unread).fits, "limits that cannot be read are refused");

    auto capped = in;
    capped.n_expert = 4;
    capped.free = 24000 * mib;
    capped.budget = 15000 * mib;
    const auto cap = oh_ssd::compute_vram_receipt(capped);
    CHECK(cap.slots == 8 && cap.slot_bytes == 120 * mib, "the receipt counts no more slots than experts");
}

// Many small experts: the tier's bookkeeping dwarfs its two scratch slots, so it is quoted from the slot count and the
// geometry, and a host budget only the scratch fits is refused with the largest context that does.
static void test_vram_tier_bookkeeping_receipt() {
    constexpr uint64_t mib = 1024ull * 1024ull;
    constexpr int routed = 44, experts = 288;
    constexpr uint64_t raw_expert = 3 * 32 * 32 * 2, slot_bytes = 8192; // up, gate and down at 32x32 F16, page-rounded

    std::vector<int> priority;
    for (int layer = 1; layer <= routed; ++layer) { priority.push_back(layer); }
    const std::vector<uint64_t> per_layer(routed, raw_expert);
    const uint64_t quote = oh_ssd::vram_bookkeeping_quote(uint64_t(routed) * experts * slot_bytes, per_layer, routed + 1, experts);

    // The real containers, with every keyed list at its worst: each expert listing all of its layer's slots.
    vram_slot_table table(size_t(routed) * experts, priority, false, 0, false, 2, 0, KIMI_VRAM_MAX_LAYERS - 1, experts);
    uint64_t real = table.slots.capacity() * sizeof(vram_slot_table::entry) +
                    table.layers.capacity() * sizeof(vram_slot_table::layer);
    for (auto & layer : table.layers) {
        real += layer.counts.capacity() * sizeof(double) + layer.held.capacity() * sizeof(uint16_t) +
                layer.keyed.capacity() * sizeof(std::vector<uint32_t>);
        for (auto & list : layer.keyed) {
            for (size_t i = 0; i < layer.count; ++i) { list.push_back(uint32_t(i)); }
            real += list.capacity() * sizeof(uint32_t);
        }
    }
    CHECK(real > mib && real <= quote, "the quote of %" PRIu64 " bytes does not cover the table's %" PRIu64, quote, real);

    oh_ssd::memory_inputs base;
    base.context_host    = 200 * mib;
    base.kv_linear_host  = 160 * mib;
    base.compute_host    = 50 * mib;
    base.process_private = 500 * mib;
    base.n_ctx           = 1024;
    base.n_seq           = 1;
    // Charged as checked_receipt charges the tier: owed by the budget, not allocated yet.
    const auto receipt = [&](uint64_t budget, uint64_t tier_host) {
        auto in = base;
        in.context_host += tier_host;
        in.context_unallocated += tier_host;
        return oh_ssd::compute_receipt(budget, 16 * mib, 0, UINT64_MAX, in);
    };
    const uint64_t scratch = 2 * slot_bytes + 16 * 32 * sizeof(float) * 2;
    uint64_t refused_at = 0, fits_at = 4096 * mib;
    CHECK(receipt(fits_at, scratch).fits, "the scratch alone does not fit %" PRIu64, fits_at);
    while (fits_at - refused_at > 1) {
        const uint64_t middle = refused_at + (fits_at - refused_at) / 2;
        (receipt(middle, scratch).fits ? fits_at : refused_at) = middle;
    }
    const auto refused = receipt(fits_at, scratch + quote);
    CHECK(!refused.fits, "a budget only the scratch fits admitted the tier's bookkeeping");
    CHECK(refused.max_ctx > 0 && refused.max_ctx < base.n_ctx, "the refusal names a context of %u", refused.max_ctx);
}

// The tier's price through the path a context takes: set on a real runtime, then charged by admit_budget. The nine layers
// use every up and down pairing of three types whose rows are the same size, so every slot is 6.75 MiB and only the type
// pairs tell the compute engines apart.
static void test_vram_tier_pairs(const char * path) {
    static ggml_backend_dev_t no_devices[] = { nullptr };
    constexpr uint64_t mib = 1024ull * 1024ull, latent = 16384, hidden = 256, n_expert = 2, hits = 16;
    constexpr uint64_t slot = 7077888, budget = 128 * mib;                  // 3 * latent * hidden * 18 / 32, a page multiple
    constexpr uint64_t engine_host = hits * latent * sizeof(float) * 2;     // the two pinned row buffers
    constexpr uint64_t engine_rows = hits * (latent * sizeof(float) * 2 + hidden * sizeof(float) * 3);
    constexpr uint64_t engine_device = engine_rows + 18432;                 // Q8_1 operand: 16384 columns are 512 blocks of 36

    auto mp            = llama_model_default_params();
    mp.devices         = no_devices;
    mp.n_gpu_layers    = 0;
    mp.use_extra_bufts = false;
    mp.ssd_experts     = true;
    llama_model * model = llama_model_load_from_file(path, mp);
    CHECK(model, "the fixture did not load");
    if (!model) {
        return;
    }
    {
        auto rt = oh_ssd::create(*model, "engine.ram_budget_mib=2048\nhost.headroom_mib=16\n", 1);
        const ggml_type types[] = { GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_IQ4_NL };
        std::vector<oh_ssd::vram_tier_layer> mixed, single;
        const std::vector<uint64_t>          raw(9, slot);
        for (int i = 0; i < 9; ++i) {
            mixed.push_back({ i, slot, int32_t(types[i / 3]), int32_t(types[i % 3]) });
            single.push_back({ i, slot, int32_t(GGML_TYPE_Q4_K), int32_t(GGML_TYPE_Q4_K) });
        }
        const auto inputs = [](uint64_t context) {
            oh_ssd::memory_inputs in;
            in.context_host   = context;
            in.kv_linear_host = context / 2;
            in.n_ctx          = 1024;
            in.n_seq          = 1;
            return in;
        };
        const auto refusal = [&](uint64_t context) -> std::string {
            try {
                oh_ssd::admit_budget(*rt, inputs(context));
                return "";
            } catch (const std::exception & e) {
                return e.what();
            }
        };

        // Nothing priced yet: the largest context the budget holds, to a MiB.
        uint64_t fits_at = 0, refused_at = 4096 * mib;
        while (refused_at - fits_at > mib) {
            const uint64_t middle = fits_at + (refused_at - fits_at) / 2;
            (refusal(middle).empty() ? fits_at : refused_at) = middle;
        }
        CHECK(fits_at > 256 * mib, "the unpriced runtime only holds %" PRIu64 " bytes", fits_at);

        const uint64_t bookkeeping = oh_ssd::vram_bookkeeping_quote(budget, raw, 9, n_expert);
        const auto     host_for    = [&](uint64_t engines) {
            return 2 * slot + engines * engine_host + (2 * hits + 1) * latent * sizeof(float) + bookkeeping;
        };
        const auto charged = [&] { return oh_ssd::admit_budget(*rt, inputs(mib)).context - mib; };

        // One pair is one engine however many layers share it, so a single-format model is not charged twice.
        const uint64_t single_device = oh_ssd::price_vram_tier(*rt, single, budget, latent, hidden, n_expert);
        const uint64_t single_host   = charged();
        CHECK(single_device == engine_device, "one type pair quoted %" PRIu64 " device bytes", single_device);
        CHECK(single_host == host_for(1), "one type pair quoted %" PRIu64 " host bytes, not %" PRIu64, single_host, host_for(1));

        // A layer of float experts has no quantized operand, and a small quantized one needs its rows padded to 512.
        const auto price_small = [&](int32_t up, int32_t down) {
            return oh_ssd::price_vram_tier(*rt, { { 0, 8192, up, down } }, 1024 * 1024, 256, 256, 1);
        };
        constexpr uint64_t small_rows = 16 * (256 * sizeof(float) * 2 + 256 * sizeof(float) * 3);
        CHECK(price_small(GGML_TYPE_F16, GGML_TYPE_F16) == small_rows, "float experts were quoted a quantized operand");
        CHECK(price_small(GGML_TYPE_Q4_K, GGML_TYPE_F16) == small_rows + 576, "a quantized up row was not quoted its operand");
        CHECK(price_small(GGML_TYPE_F16, GGML_TYPE_Q4_K) == small_rows + 16 * 576, "the quantized hidden rows were not quoted");

        // The nine pairs are nine engines on each side, at the same slot size as the single pair.
        const uint64_t mixed_device = oh_ssd::price_vram_tier(*rt, mixed, budget, latent, hidden, n_expert);
        const uint64_t mixed_host   = charged();
        CHECK(mixed_device == 9 * engine_device, "nine type pairs quoted %" PRIu64 " device bytes", mixed_device);
        CHECK(mixed_host == host_for(9), "nine type pairs quoted %" PRIu64 " host bytes, not %" PRIu64, mixed_host, host_for(9));

        // A context that fit before the tier was priced is refused now that the budget owes it.
        const std::string refused = refusal(fits_at - mixed_host / 2);
        CHECK(contains(refused, "does not fit the host budget"), "the priced tier was not charged: '%s'", refused.c_str());
    }
    llama_model_free(model);
}

int main(int argc, char ** argv) {
    if (argc != 3 && !(argc == 6 && !std::strcmp(argv[1], "load-at"))) {
        std::fprintf(stderr, "usage: %s equal|context|admit|huge|reserve|batch|sampler|sched|outputs|cells|borrow|refuse|mtp|glm-fixture|glm|glm-mixed|glm-admit|receipt|tier-pairs <model.gguf>\n", argv[0]);
        return 2;
    }
    g_verbose = std::getenv("OH_SSD_TEST_VERBOSE") != nullptr;
    llama_log_set(log_cb, nullptr);
    llama_backend_init();
    const std::string mode = argv[1];
    if (mode == "receipt") {
        test_receipt();
        test_vram_receipt();
        test_vram_tier_bookkeeping_receipt();
    } else if (mode == "glm-fixture") {
        make_glm_fixtures(argv[2]);
    } else if (mode == "glm") {
        test_glm(argv[2]);
    } else if (mode == "glm-mixed") {
        test_glm_mixed(argv[2]);
    } else if (mode == "glm-admit") {
        test_glm_admit(argv[2], argv[0]);
    } else if (mode == "equal") {
        test_equal(argv[2]);
    } else if (mode == "tier-pairs") {
        test_vram_tier_pairs(argv[2]);
    } else if (mode == "vram") {
        test_vram(argv[2]);
    } else if (mode == "context") {
        test_context(argv[2]);
    } else if (mode == "admit") {
        test_admit(argv[2], argv[0]);
    } else if (mode == "load-at") { // load-at <model.gguf> <budget MiB> <n_ctx> <n_ubatch>
        test_load_at(argv[2], argv[3], uint32_t(std::strtoul(argv[4], nullptr, 10)), uint32_t(std::strtoul(argv[5], nullptr, 10)));
    } else if (mode == "huge") {
        test_huge(argv[2]);
    } else if (mode == "reserve") {
        test_reserve(argv[2]);
    } else if (mode == "batch") {
        test_batch(argv[2]);
    } else if (mode == "sampler") {
        test_sampler(argv[2]);
    } else if (mode == "sched") {
        test_sched_commit(argv[2]);
    } else if (mode == "outputs") {
        test_outputs(argv[2]);
    } else if (mode == "cells") {
        test_cells(argv[2]);
    } else if (mode == "borrow") {
        test_borrow(argv[2]);
    } else if (mode == "refuse") {
        test_refuse(argv[2]);
    } else if (mode == "mtp") {
        test_mtp(argv[2]);
    } else {
        std::fprintf(stderr, "unknown mode %s\n", argv[1]);
        return 2;
    }
    llama_backend_free();
    if (g_failures) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("ok\n");
    return 0;
}
