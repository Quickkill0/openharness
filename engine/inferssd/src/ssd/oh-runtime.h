#pragma once

// OpenHarness glue between a llama_context and InferSSD's CPU expert sink.
//
// It owns everything the sink borrows for the life of a context: the parsed
// settings, the host-memory client, the source view that maps expert tensors to
// byte ranges in the original GGUF files (no repacked copy), the open file
// descriptors and the sink runtime itself. llama-context.cpp creates one when
// llama_context_params::ssd_settings is set and destroys it after its backends.

#include "ggml-backend.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct llama_model;
struct llama_kimi_cpu_sink_runtime;

namespace oh_ssd {

class context_runtime;

struct context_runtime_deleter {
    void operator()(context_runtime * runtime) const noexcept;
};

using context_runtime_ptr = std::unique_ptr<context_runtime, context_runtime_deleter>;

// Parses the settings text ("key=value" per line, '#' comments) and opens the
// model's files for direct reads. Throws std::runtime_error with the reason.
// Engine keys handled here, not by InferSSD:
//   engine.ram_budget_mib    total host memory the engine may use; 0 = physical RAM
//   engine.expert_cache_mib  upper bound for the RAM expert cache (tests force it small)
context_runtime_ptr create(const llama_model & model, const char * settings_text, uint64_t epoch);

llama_kimi_cpu_sink_runtime * sink_runtime(context_runtime & runtime) noexcept;

// Destroys the sink (its reader threads and cache pages). The context calls this first when it is destroyed; the
// rest of the runtime stays alive for the CPU backend until the backends are gone.
void release_sink(context_runtime & runtime) noexcept;

// Lets the CPU backend charge its work buffer to the same host-memory client.
// Call before the backend computes its first graph.
bool attach_cpu_backend(context_runtime & runtime, ggml_backend_t backend_cpu);

// Attaches the CUDA VRAM expert tier when vram.budget names a size; "auto" and 0 leave every expert on the CPU.
// backend_gpu is the context's GPU backend, or null when it has none. Throws std::runtime_error with the reason and a
// receipt when the tier was asked for and cannot be had. Call once the context's device buffers exist and before the
// sink is prepared.
void attach_gpu_backend(context_runtime & runtime, const llama_model & model, ggml_backend_t backend_gpu);

// What the VRAM tier's admission looks at. Bytes, except the slot sizes' per-layer meaning.
struct vram_inputs {
    uint64_t budget   = 0; // vram.budget
    uint64_t reserve  = 0; // vram.reserve_mib
    uint64_t margin   = 0; // what the tier itself always leaves free, above the reserve
    uint64_t overhead = 0; // the tier's own device buffers, per decode engine
    uint64_t free     = 0; // free on the device now
    uint64_t total    = 0; // size of the device; 0 means the limits cannot be read
    uint64_t n_expert = 0;
    std::vector<uint64_t> layer_bytes; // one expert's slot, per routed layer in layer order
};

struct vram_receipt {
    bool     fits       = false;
    uint64_t keep       = 0; // reserve + margin + overhead
    uint64_t max_budget = 0; // the largest budget that fits now
    uint64_t min_budget = 0; // one slot per layer
    uint64_t slots      = 0;
    uint64_t slot_bytes = 0; // what those slots allocate
    std::string text;        // one-line MiB receipt for the log
};

// Exposed for tests: the arithmetic behind the VRAM tier's admission.
vram_receipt compute_vram_receipt(const vram_inputs & inputs);

// Exposed for tests: the host memory the tier's slot table and job queues can take, from the same budget and per-expert
// layer bytes the slot count comes from. layer_ids is the highest routed layer id + 1.
uint64_t vram_bookkeeping_quote(uint64_t budget, std::vector<uint64_t> layer_bytes, uint64_t layer_ids, uint64_t n_expert);

// Host-memory facts the context measured after reserving its buffers. All
// bytes are host (CPU-visible) bytes unless the name says otherwise.
// One routed layer as the VRAM tier is told of it: the bytes of one expert's three roles, and the types that key its
// compute engine.
struct vram_tier_layer {
    int32_t  id        = 0;
    uint64_t raw_bytes = 0;
    int32_t  up_type   = -1;
    int32_t  down_type = -1;
};

// Prices the tier from those layers and the expert shape: sets the host bytes every later receipt charges, and returns
// the device bytes of its compute engines' buffers. attach_gpu_backend's admission takes that return value. Exposed for
// tests.
uint64_t price_vram_tier(context_runtime & runtime, const std::vector<vram_tier_layer> & layers, uint64_t budget,
                         uint64_t latent, uint64_t hidden, uint64_t n_expert);

struct memory_inputs {
    uint64_t model_host      = 0; // every model weight buffer on the host, experts included
    uint64_t context_host    = 0; // KV caches and recurrent state on the host
    // The part of context_host that has no memory behind it until decoding starts: lists and trees that fill as cells
    // are used, tables sized by the first full batch, and the graph arenas the first decodes build. The budget still
    // owes it, but process_private cannot hold it.
    uint64_t context_unallocated = 0;
    uint64_t compute_host    = 0; // compute buffers on the host
    uint64_t kv_linear_host  = 0; // the part of context_host that grows with n_ctx
    uint64_t kv_attn_bytes   = 0; // attention KV on all devices (MLA latent included)
    uint64_t kv_swa_bytes    = 0; // sliding-window KV on all devices
    uint64_t kv_index_bytes  = 0; // sparse-attention (DSA) indexer cache on all devices
    uint64_t recurrent_bytes = 0; // recurrent / linear-attention state on all devices
    uint64_t device_bytes    = 0; // model + context + compute on non-host devices
    bool compute_follows_ctx = false; // most of compute_host grows with n_ctx, by a share only a load at that size shows
    // What the host envelope charges this whole process now (Windows commit, macOS footprint), the context's
    // buffers included; apply_budget measures it. 0 where the engine does not measure it (Linux, where the
    // budget is held by the receipt and a cgroup).
    uint64_t process_private = 0;
    // False while the context has only been sized: process_private then holds none of its buffers.
    bool     buffers_allocated = true;
    uint32_t n_ctx           = 0;
    uint32_t n_seq           = 0;
    uint32_t n_ctx_min       = 256; // the smallest n_ctx this context can have: every smaller value is padded up to it
};

struct receipt {
    uint64_t budget        = 0; // the host budget the engine works within
    uint64_t headroom      = 0; // host.headroom_mib
    uint64_t dense_weights = 0; // model_host minus the streamed experts
    uint64_t experts       = 0; // expert bytes on disk (streamed, not counted against RAM)
    uint64_t context       = 0;
    uint64_t compute       = 0;
    uint64_t process       = 0; // the rest of the process the envelope charges: runtime, libraries, heap
    uint64_t cache         = 0; // what is left for the RAM expert cache
    bool     fits          = false;
    uint32_t max_ctx       = 0; // largest context that fits, when it does not
    std::string text;           // one-line MiB receipt for the log
};

// Computes the receipt and, when it fits, sets the expert cache budget before
// the sink is created. Throws std::runtime_error naming the largest context
// that fits when it does not.
// Refuses a context the host budget cannot hold, before its buffers exist. Sets nothing.
receipt admit_budget(context_runtime & runtime, const memory_inputs & inputs);

receipt apply_budget(context_runtime & runtime, const memory_inputs & inputs);

// Exposed for tests: the arithmetic behind apply_budget, without a context.
receipt compute_receipt(uint64_t budget, uint64_t headroom, uint64_t experts,
                        uint64_t cache_cap, const memory_inputs & inputs);

// Exposed for tests: the end of a refusal, saying what to change.
std::string refusal_advice(const receipt & out, const memory_inputs & inputs);

uint64_t physical_memory_bytes() noexcept;

} // namespace oh_ssd
