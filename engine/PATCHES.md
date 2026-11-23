# Patch ledger

This file lists every change we make to code we did not write: the vendored
llama.cpp snapshot in `engine/llama.cpp/` and the InferSSD imports in
`engine/inferssd/`. Two offline-checkable scripts keep it honest:

- `python engine/scripts/vendor_llama_cpp.py verify` fails when a file under
  `llama.cpp/` differs from the pristine upstream snapshot recorded in
  `UPSTREAM.json` and is not listed here, or is listed but unchanged.
- `python engine/scripts/inferssd_import.py verify` does the same for
  `inferssd/` against the hashes in `inferssd/IMPORT.json`. Files under
  `inferssd/` that are not InferSSD imports (our build glue) are "added" and
  must be listed too.

Write entries as `` `llama.cpp/<path>` `` or `` `inferssd/<path>` `` so the
verifiers can read them. Changed files also carry a short
`Modified by OpenHarness` comment near the top.

Group entries by integration boundary (model/graph, memory/allocator,
sources/transport, server/state), matching the order of the InferSSD port.

## llama.cpp

### Changed upstream files

- `llama.cpp/ggml/include/ggml-cuda.h`: declares the hooks below.
- `llama.cpp/ggml/src/ggml-cuda/common.cuh`: the context's pointer to the SSD runtime services.
- `llama.cpp/ggml/src/ggml-cuda/mmvf.cu`: moe-stream kernel entry point.
- `llama.cpp/ggml/src/ggml-cuda/mmvq.cu`: moe-stream kernel entry points.
- `llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu`: the hooks the CUDA VRAM expert tier needs, from InferSSD.
  Device allocations are charged to a reserve budget with a reclaimer, the backend exposes the
  `ggml_backend_ssd_provider_v1` table, and the moe-stream kernel entry points are added. Nothing changes until a context
  attaches the tier.

#### Model/graph

These carry InferSSD's SSD expert streaming onto b12192. Everything is off
unless a model is loaded with `ssd_experts` and a context gets
`ssd_settings`; with both off the graph and the load path are upstream's.

- `llama.cpp/include/llama.h`: new `llama_model_params::ssd_experts` (keep
  the routed experts in the GGUF files for the SSD runtime) and
  `llama_context_params::ssd_settings` (typed SSD settings, one `key=value`
  per line). Both default to off in the default-params functions.
- `llama.cpp/src/llama-model-loader.h`: declares `file_paths` (below).
- `llama.cpp/src/llama-model-loader.cpp`: the loader records the path of
  every GGUF file it opens (`file_paths`), so the SSD runtime can open the
  original files itself.
- `llama.cpp/src/llama-model.h`: declares `ssd_experts`,
  `ssd_expert_ranges` and `ssd_files` (below).
- `llama.cpp/src/llama-model.cpp`: with `ssd_experts`, the model requires mmap (refuses mlock and the lazy load
  mode), maps the files without prefetch so the expert pages are never read
  in, keeps the routed expert tensors (`ffn_{up,gate,down}_exps`) in plain
  host buffers of that mapping (no repacked copy, no device copy), and
  records where each one lies in its original file (`ssd_expert_ranges`,
  `ssd_files`). Only tensors the model actually created are recorded, so the
  experts of an MTP layer that a default load leaves in the file stay out.
- `llama.cpp/src/llama-graph.h`: declares the sink fields and
  `build_moe_prefetch_hint` (below).
- `llama.cpp/src/llama-graph.cpp`: the graph params carry the context's sink runtime. With it, `build_moe_ffn` sends the
  routed experts to an SSD sink node (decode: one token; prefill: more)
  instead of the `mul_mat_id` chain, and returns an empty tensor for a
  zero-token FFN (the last layer after output pruning), as InferSSD does.
  `build_moe_prefetch_hint` runs the next layer's router on this layer's
  input so the sink can start reading the experts it will likely need.
- `llama.cpp/src/models/qwen3moe.cpp`: builds that prefetch hint for each
  layer but the last. Qwen3-MoE is the synthetic test fixture's
  architecture; GLM-5.3-Flash follows in the next stage.
- `llama.cpp/src/llama-context.h`: declares `ssd_runtime`,
  `ssd_apply_budget`, `ssd_scale_bytes`, `ssd_admit_reservation`,
  `init_output_buffer` and the sizing state `ssd_measuring` / `no_alloc_now`
  (below).
- `llama.cpp/src/llama-context.cpp`: a context with `ssd_settings` owns an `oh_ssd::context_runtime` (declared
  before the backends so it outlives them; the sink is released first), attaches it to the CPU
  backend, sizes the context first with dummy buffers (the no_alloc paths), admits it
  against the RAM budget, and only then allocates the real buffers and
  reconciles against them
  (`ssd_apply_budget`: KV per type, recurrent state, compute buffers and the
  rest of the process before the expert cache; refuses a context that does
  not fit and names the largest one that does, counting one attention mask
  with the memory that shrinks along with the context), and opens a sink phase before
  each ubatch. A single-token, single-sequence ubatch is a decode step; b12192
  decodes with `LLM_GRAPH_TYPE_DEFAULT`, where InferSSD's older base used
  `LLM_GRAPH_TYPE_DECODER`, so both count. While sizing, `sched_reserve` keeps
  the largest buffer per backend across every reservation variant. Everything
  that scales with n_ctx or n_batch has an arithmetic size in admission and is
  allocated after it: `output_ids` and the first output buffer move behind
  admission (`init_output_buffer`), `ssd_scale_bytes` sizes the n_batch vectors,
  the first output buffer's rows (one per sequence, the sampler's extra rows
  included), the graph metadata and scheduler tables, and on their own the
  nodes of the backend samplers' chains, which `graph_max_nodes` can leave out
  of the fixed floor (`with_samplers`). `ssd_admit_reservation`
  charges the sizing graphs themselves before they are built. On Linux,
  scheduler table memory that is never touched is left out, since it takes no
  pages. Windows commits it when it is allocated, so `ssd_admit_reservation`
  charges the scheduler's whole capacity there, floor included, before
  `sched_new`. Once the scheduler exists that commit is in the measured process
  term, so the later admissions do not add it. Under `--ssd`, `sched_new` frees
  the old scheduler first, so only the one that was priced is ever held.
- `llama.cpp/src/llama-kv-cache.h`: `n_cells_stream`, the size of a stream
  without its cell arrays.
- `llama.cpp/src/llama-kv-cache.cpp`: a no_alloc cache keeps its size but
  builds no per-token cell arrays, so sizing an oversized context costs nothing
  and it can be refused by arithmetic. `get_size` reads the stored size. A
  no_alloc cache that borrows another cache's cells never resizes them.
  `memory_breakdown` adds the per-cell bookkeeping to the host share, for
  caches that own their cells, so the fit and memory prints of non-SSD contexts
  gain that share too. `host_unallocated_bytes` is the part of that share a new
  cache does not hold yet.
- `llama.cpp/src/llama-kv-cells.h`: `host_bytes_per_cell(n_seq)`, what
  admission charges for one cell that n_seq sequences hold. A unified cache is
  charged for every sequence holding every cell; a split cache, or one
  sequence, for one. `host_unallocated_bytes_per_cell` is its share that a new
  cache does not hold: the two trees, the sequence rows and the generation.

- `llama.cpp/src/models/glm5-next.cpp`: GLM-5.3-Flash gives the expert sink
  its next-layer prefetch hint, built from the next layer's router.
- `llama.cpp/src/llama-context.cpp` (GLM): refuses a context whose indexer
  pools a 32-bit count cannot name before any reservation graph is built,
  prices the indexer through `idx_memory_breakdown`, and marks that GLM's
  compute buffers follow the context.

#### Memory/allocator

- `llama.cpp/src/llama-memory-hybrid-idx.h`: declares the three functions below.
- `llama.cpp/src/llama-memory-hybrid-idx.cpp`: `idx_memory_breakdown` adds what
  the indexer's cache allocates at the first decode (the representative
  generations, and the k-pool layout and state) to its tensors, on sizes
  alone. `kpool_host_bytes` is the k-pool part of that, which the context
  reports as not allocated yet. `kpool_fits` is the pool-count check above.
- `llama.cpp/ggml/src/ggml-backend.cpp`: `ggml_backend_sched_capacity_bytes`,
  what `ggml_backend_sched_new` allocates for a node count. Both take the
  sizes from one helper, so a price cannot drift from the allocation.
- `llama.cpp/ggml/include/ggml-backend.h`: declares it.
- `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.cpp`: InferSSD's CPU backend hooks.
  The backend can be given SSD runtime services; when it has them, its work
  buffers are taken from the host budget (and reported when freed), and it
  exposes the SSD CPU API (`ssd::cpu::api_v2`: sink create/run, prefill and
  cache entry points, and a query for the expert types the sink can compute)
  through the backend registry's proc-address table.
  Without services the work-buffer path is upstream's, including its
  free-before-grow order.

#### Sources/transport

- `llama.cpp/common/common.h`: new `common_params::ssd` and `ssd_settings`.
- `llama.cpp/common/common.cpp`: passes them to the model and context, and skips `--fit` for an SSD load (its probe cannot map the experts, and SSD loads are admitted against their own budgets)
  params.
- `llama.cpp/common/arg.cpp` (SSD options): `--ssd` (`LLAMA_ARG_SSD`) loads
  the model for SSD expert streaming and keeps the routed experts on the CPU
  (the same buffer override as `--cpu-moe`); `--ssd-set KEY=VALUE`
  (repeatable) and `--ssd-config FILE` add settings and imply `--ssd`. A
  value without `=` or with a newline is refused. `common_params_to_map`
  joins repeated occurrences of these two with a newline instead of keeping
  the last, so a router does not lose them.
- `llama.cpp/common/preset.cpp`: `common_preset::to_args` splits those joined
  values back into one flag per occurrence for the router's child.

- `llama.cpp/common/arg.cpp`: with `OPENHARNESS_NO_REMOTE`, the new
  `openharness_reject_remote_sources()` throws if the model, projector or
  draft model has a URL, Hugging Face repo or Docker repo. It runs right after
  the command line and `LLAMA_ARG_*` variables are parsed (without `-m`, a URL
  alone would otherwise start router mode and be silently ignored), and again
  at the start of `common_models_handler_init` and
  `common_models_handler_apply`, which router presets reach without going
  through the parser. In every build (not tied to `OPENHARNESS_NO_REMOTE`),
  the new `openharness_reject_agent_features()` refuses the agent features at
  the same point: `--ui-mcp-proxy`, `--agent`, `--tools` (any tool name),
  `--tools-runtime`, `--mcp-servers-config`, `--mcp-servers-json`,
  `--connect` and `--connect-code`, however they arrive (flag, `LLAMA_ARG_*`
  variable, user `config.ini`, or a router preset, which reaches a child
  through its command line). OpenHarness is a model loader; agents, tools
  and MCP live in AI Shuffle. `--rpc` (and `LLAMA_ARG_RPC`, `config.ini`,
  presets) throws in its own option handler in every build, because upstream
  connects to the RPC servers right there, while arguments are still being
  parsed; `add_rpc_devices()` is deleted. `GGML_RPC` is also forced off in
  `engine/CMakeLists.txt`.
- `llama.cpp/common/http.h`: with `OPENHARNESS_NO_REMOTE`, `common_http_client`
  throws instead of creating a client. Every outbound request in llama-common
  goes through it (model and Hugging Face downloads, the router's `POST
  /models` lookup), so this is the backstop.
- `llama.cpp/tools/server/server-common.cpp`: with `OPENHARNESS_NO_REMOTE`,
  `handle_media` refuses `http(s)` media URLs in chat, completion and rerank
  requests before any request is made; base64 data is unaffected.
- `llama.cpp/tools/server/server-models.cpp`: with `OPENHARNESS_NO_REMOTE`,
  the router's `POST /models` (which only downloads from Hugging Face)
  answers 400 with the refusal instead of trying a lookup. In every build,
  router startup also refuses a preset (model section or `[*]`) that sets
  one of the agent options above or `rpc`, so no child is started with them.

#### Server/state

- `llama.cpp/scripts/ui-assets.cmake`: new `NO_UI` switch that emits an empty
  asset table before any other source is considered, so a stale `dist/` in
  the source or build tree is never embedded.
- `llama.cpp/tools/ui/CMakeLists.txt`: passes `OPENHARNESS_NO_UI` to the
  script as `NO_UI`.
- `llama.cpp/tools/server/server.cpp`: `server_setup_features` no longer
  registers `/cors-proxy` or `/tools` (they are 404s, not upstream's 403),
  never starts MCP servers or tools, and refuses to start if any agent option
  slipped through. With `OPENHARNESS_TEST_HOOKS` (engine test builds only),
  the model load waits for the file named by `OPENHARNESS_TEST_LOAD_GATE`
  (at most 120 s) so the smoke test sees the loading 503 every time.

### Known upstream behaviour left unchanged

- Crash backtraces, `ggml_print_backtrace` in ggml/src/ggml.c (not
  modified): on a crash, Linux and macOS builds print a backtrace by forking
  `gdb` or `lldb` from `PATH`. It is local and makes no network connection.
  Setting `GGML_NO_BACKTRACE` in the environment turns it off; there is no
  compile-time switch, so we have not patched it.

### Added files

None. (Our engine tests live in `engine/tests/`, outside the snapshot.)

### Removed files

None yet. (Pruning by the vendor script is not a removal; it is recorded in
`UPSTREAM.json`.)

## InferSSD

Imported from wici-ai/inferssd at the revision in `inferssd/IMPORT.json`.
Imported so far: the backend-independent core under
`inferssd/ggml/src/ggml-ssd/` (platform, io, host, place, settings, store,
telemetry, the GPU provider interface headers and `stream/copy-plan.h`) and
its unit tests under `inferssd/tests/ssd/`; the CPU expert sink under
`inferssd/ggml/src/ggml-cpu/ssd/` (host expert cache, read ring, CPU prefill
and decode, next-layer prefetch), compiled into ggml-cpu; and the llama-side
runtime under `inferssd/src/ssd/` (model source over the original GGUF files,
model adapter, options). Not imported yet: `gpu/graph-lane.*` (the
Metal/Vulkan grouped-prefill lane; we do not build those backends),
`stream/runtime.*` (it needs the scheduler stream hooks in `ggml-backend`), and
the module's
top-level `CMakeLists.txt` (replaced by ours). `cpu-sink-vram.inc` is
imported because `cpu-sink.cpp` includes it; its VRAM paths stay inactive
without a GPU tier.

### Changed InferSSD files

- `inferssd/ggml/src/ggml-cuda/ssd/vram-tier.cpp`: slots sized per layer instead of the largest format; the production tier is
  not a global elastic pool, so it takes an explicit budget; a throttled cumulative `[oh-ssd] summary` line, with
  per-layer hits at shutdown; the two device scratch slots of the optional host-compute path are never allocated,
  since they would sit outside the receipt; POSIX headers it did not use removed for MSVC; each role's bytes are exact
  and only the whole slot is page-aligned, because the original GGUF has no padding between roles; `vram.admit_all` sets
  the minimum count and half-life the policy accepts; if a tier allocation runs out of device memory on the first
  decode, the tier releases everything and the run continues on the CPU with a `status=unavailable` summary line. That
  is a deliberate exception to refusing with a receipt: the budget was already checked at load and the tier is an
  accelerator, so the answer is unaffected. The same failure after experts have been uploaded stops the run with an
  error. A prefill start prints the exact totals of the request before it.
  State the worker and upload threads write under `mutex` is now read under
  it as well: the failure text after a drain, the summary's upload bytes, the
  admission log's costs and `prefill_queued`. The `phase_release` report drains
  those threads before it reads their counters.
- `inferssd/ggml/src/ggml-cpu/ssd/cpu-sink-vram.inc`: a compile-time check that a deferred source stays within the size the host budget prices for it.
- `inferssd/ggml/src/ggml-cuda/ssd/vram-slot-table.h`: helpers for the per-layer slot capacity and offsets, and the upper bound on the tier's host bookkeeping that `oh-runtime.cpp` charges to the host budget before the context is built. `vram-tier.cpp` and `cpu-sink-vram.inc` assert their queued-job sizes stay within it.
- `inferssd/ggml/src/ggml-cuda/ssd/generic-compute-adapter.cu`: `ggml-common.h` found on the include path; an
  out-of-memory error starts with the marker from `vram-compute-api.h`.
- `inferssd/ggml/src/ggml-cuda/ssd/vram-compute-api.h`: the marker that tells the tier an engine could not be created
  because the card had no memory, as opposed to a refusal.
- `inferssd/ggml/src/ggml-ssd/testing/tier.h`: the `fail_allocation` test option behind `vram.fail_allocation`.
- `inferssd/src/ssd/cpu-runtime.h`: the RAM and SSD lines of the same summary, printed exactly at each prefill start
  so a request's totals survive the server being killed; hands the tier each layer's expert tensors
  through `vram_reserve`, so a model whose layers use different quantizations is not refused.

- `inferssd/ggml/src/ggml-cpu/ssd/cpu-prefill.cpp`: includes `quants.h` and
  `vec.h` through the include path instead of `../`, since the file lives
  outside the vendored ggml-cpu directory. The MXFP4 multi-row kernels it
  calls are InferSSD additions to ggml-cpu that b12192 does not have, so
  that branch is compiled only under `OH_HAVE_MXFP4_MULTIROW` (never set);
  otherwise MXFP4 uses the generic one-row `vec_dot` per column, which gives
  the same result more slowly.
- `inferssd/src/ssd/model-adapter.cpp`: accepts `LLM_ARCH_QWEN3MOE`, the
  synthetic fixture's architecture (plain softmax top-k routing, no shared
  expert), and `LLM_ARCH_GLM5_NEXT` on DeepSeek 4.1's path, which clamps its
  SwiGLU the same way.
- `inferssd/tests/ssd/test-kimi-cpu-sink.cpp`: `std::bitset::count` instead
  of `__builtin_popcount`, which MSVC does not have.
- `inferssd/ggml/src/ggml-ssd/host/room.h`: `room::envelope_unbounded`, set when
  no memory.max above the process is finite.
- `inferssd/ggml/src/ggml-ssd/host/ledger.h`: sets that flag at the end of the
  cgroup walk. The root's `memory.max` counts as absent only when the file
  really does not exist (ENOENT); any other read error fails closed, in the
  envelope and in the arbiter's `unbounded()`. A `memory.max` that does not
  exist below the root means the controller is not enabled there, so the walk
  carries on to the parent; with none on the path, or no cgroup v2 path at all,
  no limit applies and MemAvailable still holds the cache. `memory_limit` is
  that rule for callers that need the number: the tightest finite limit, whether
  any controller was found, or false when a limit exists but cannot be read.
  `unbounded()` and the runtime's admission both use it. A failed read of
  `/proc/self/cgroup` is an unknown limit too (`cgroup_unknown`), kept apart
  from a successful read that names no v2 path: admission refuses and the
  arbiter's envelope stays unreadable.
- `inferssd/ggml/src/ggml-cpu/ssd/cpu-sink.h`: declares
  `kimi_cpu_sink_expert_type_supported`.
- `inferssd/ggml/src/ggml-cpu/ssd/cpu-sink.cpp`: implements it from the same
  conversion traits the executor uses, so the load can refuse an expert type
  before a prompt would abort on it.
- `inferssd/ggml/src/ggml-ssd/host/expert-cache.h`: admits the RAM cache when no
  cgroup limit exists. InferSSD required a finite memory.max because its launcher
  always creates one; ours does not, and without this a Linux run would silently
  have no RAM tier. The cache stays capped by the budget receipt and the
  MemAvailable floor still applies. An unreadable envelope still refuses, and
  Windows and macOS are unchanged.

- `inferssd/tests/ssd/test-ssd-platform-threads.cpp`: on Windows, a thread
  that does not move onto its selected CPU Set within 500 ms is reported, not
  failed. CPU Sets are a soft preference and a parked E-core on a hybrid CPU
  may not take the thread; the pin read-back is still required.

### Added files (OpenHarness build glue)

- `inferssd/src/ssd/vram-group-stub.cpp`: reports grouped prefill as not built, which keeps prefill on the CPU.
  `ggml-cuda/ssd/group-compute-adapter.cu`, `group-matmul.cu` and the imported `ggml-cuda/ssd/CMakeLists.txt` are not built.
- `inferssd/tests/ssd/test-vram-layer-slots.cpp`: per-layer capacity, offsets, and admission and eviction inside a layer's slots.

- `inferssd/CMakeLists.txt`: builds the imported core modules as static
  libraries under the `ggml-ssd` interface target, compiles the CPU sink
  into `ggml-cpu` and adds the llama-side runtime to `llama`.
- `inferssd/src/ssd/oh-runtime.h`: interface of `oh-runtime.cpp`.
- `inferssd/src/ssd/oh-runtime.cpp`: the
  glue between a `llama_context` and the sink. It parses the typed settings
  (duplicate keys and malformed lines are refused; OpenHarness adds
  `engine.ram_budget_mib` and `engine.expert_cache_mib`), builds the expert
  source over the original GGUF files, and computes the RAM receipt: the
  budget minus dense weights, KV (attention, SWA, indexer), recurrent state,
  compute buffers, the rest of the process and the headroom leaves the
  expert cache. Memory the context is charged for but has not allocated yet
  (`context_unallocated`: cell trees and sequence rows, indexer lists, batch
  tables and the two graph arenas decoding builds) stays on top of the measured
  process rather than
  being subtracted from it, so it comes out of the expert cache. A context that
  does not fit is refused with the largest context that does, or, when compute
  buffers that follow the context are over budget by themselves, with a request
  for a smaller one, or at the smallest context with a larger budget or smaller
  batch sizes. GLM's draft
  (MTP) head is refused by name. `admit_budget` runs this on sizes alone, before the
  context's buffers exist; on Linux a `memory.max` on the process's cgroup
  path lowers the budget and a limit that exists but cannot be read refuses
  the context (`memory_limit` in `ledger.h`). With no memory controller it warns
  on stderr and admits against the RAM budget and MemAvailable. An expert type the sink cannot compute (F32) is
  refused here, by name, before anything is indexed.
- `inferssd/tests/ssd/test-ssd-host-cgroup-root.cpp`: an unreadable
  `memory.max` fails closed at any level, a missing one is passed over for its
  parent, no controller on the path counts as no limit, and `memory_limit` finds
  the tightest limit or reports an unknown one.
- `inferssd/ggml/src/ggml-ssd/settings/registry.def`: adds `vram.admit_all`,
  which replaces the tier's count and measured-cost admission with least
  recently used, so a test's hits and evictions do not depend on timing, and
  `vram.fail_allocation`, which fails the tier's first allocation so a test can
  check the CPU fallback.
- `inferssd/tests/ssd/test-settings-telemetry.cpp`: counts the added settings.
- `inferssd/tests/CMakeLists.txt`: registers the imported unit tests with
  CTest (labels `engine`, `ssd`); POSIX-only fixtures are skipped on Windows,
  as in InferSSD.
