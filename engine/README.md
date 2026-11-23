# OpenHarness engine

The engine is OpenHarness's own llama.cpp derivative. Its goal is to run
mixture-of-experts models that are larger than VRAM plus RAM: hot experts stay
in VRAM, warm experts in RAM, and cold experts are streamed from NVMe on
demand. The streaming design comes from
[InferSSD](https://github.com/wici-ai/inferssd) (Apache-2.0), ported onto a
current llama.cpp base. GLM-5.3-Flash is the first target model.

This revision has a pinned llama.cpp snapshot, our own build entry point
that produces `openharness-engine-server`, and SSD expert streaming on the
CPU (see [SSD expert streaming](#ssd-expert-streaming)). The server behaves
like upstream `llama-server` (same OpenAI-compatible routes) with these
product choices fixed at build time:

- no embedded web UI and no UI download during the build. A `dist/` folder
  left over from an earlier build is not embedded either;
- no remote model sources. Model, draft and projector URLs, Hugging Face and
  Docker repos (as flags or `LLAMA_ARG_*` variables) are refused at startup
  with an error before anything is resolved, the router's `POST /models`
  refuses them too, and the HTTP client itself refuses every outbound
  connection, so the server never reaches the network on its own. Media
  given as an `http(s)://` URL in a request is refused too; send it as base64
  data instead;
- no agent features. The engine is a model loader; agents, tools and MCP live
  in AI Shuffle. `--ui-mcp-proxy`, `--agent`, `--tools`, `--tools-runtime`,
  `--mcp-servers-config`, `--mcp-servers-json`, `--connect` and
  `--connect-code` are refused at startup in every build, whether they come
  as flags, `LLAMA_ARG_*` variables, the user `config.ini` or router presets,
  and the `/cors-proxy` and `/tools` routes do not exist;
- no remote RPC devices. `--rpc` (flag, variable, `config.ini` or preset) is
  refused at startup, and configuring with `GGML_RPC=ON` or `LLAMA_RPC=ON`
  fails;
- no TLS library, no `llama-connect` tunnel, and no ffmpeg helper for video
  input (upstream's `MTMD_VIDEO` is forced off, so nothing is launched from
  `PATH`).

With a CUDA build and an explicit `vram.budget`, the hottest routed experts also
sit in VRAM. One short GLM-5.3-Flash run has exercised it so far.

If the card has no memory for the tier when the first decode needs it, the run
continues on the CPU and the summary prints `tier=vram status=unavailable`. This
is a deliberate exception to refusing with a receipt: the budget was already
checked at load, and the tier only speeds the run up, so the answer is unaffected.
A failure after experts have been uploaded stops the run with an error.

## Layout

| Path | What it is |
| --- | --- |
| `llama.cpp/` | Pruned upstream snapshot. Do not edit casually: every change must be listed in `PATCHES.md`. |
| `UPSTREAM.json` | Upstream tag, commit, pruning rules and content digest of the snapshot. |
| `PATCHES.md` | Ledger of our changes to files under `llama.cpp/`. |
| `CMakeLists.txt` | Build entry point. Configure this directory, not `llama.cpp/`. |
| `server/main.cpp` | Server entry point (adds `--engine-info`, then runs upstream's server). |
| `tests/` | Engine tests, registered with CTest under the label `engine`. |
| `scripts/vendor_llama_cpp.py` | Imports, verifies and diffs the upstream snapshot. |
| `scripts/build_windows.cmd` | Windows build helper (MSVC + Ninja, optional CUDA). |
| `NOTICE`, `LICENSES/` | Notices for llama.cpp (MIT), InferSSD (Apache-2.0) and the third-party code linked into the server (`THIRD-PARTY-NOTICES.txt`). |

## Build on Windows (MSVC + CUDA)

Requirements: Visual Studio 2022 or newer with the "Desktop development with
C++" workload (it bundles CMake and Ninja), Python 3.9+ for the tests, and for
the CUDA build the CUDA Toolkit (tested with 13.3).

From any shell:

```
engine\scripts\build_windows.cmd cuda
```

The script finds Visual Studio with `vswhere`, loads `vcvars64.bat`, and
builds into `engine\build-win-cuda`. `cpu` instead of `cuda` builds without
CUDA. Knobs: `OH_CUDA_ARCH` (default `89`, RTX 40-series; use `86` for
RTX 30-series or a list like `86;89`), `OH_BUILD_JOBS` (default `12`),
`OH_BUILD_DIR`, `OH_BUILD_TESTS` (default `ON`). Extra arguments are passed to
CMake, for example `-DGGML_NATIVE=OFF`.

The CUDA runtime DLLs (`cudart64_13.dll`, `cublas64_13.dll`,
`cublasLt64_13.dll`) must be on `PATH` or next to the executable at run time.

## Download a Windows CUDA package

The **Engine package** GitHub Actions workflow builds on a free standard
`windows-latest` runner, without compiling on your PC. Run it manually from
Actions (or `gh workflow run engine-package.yml --ref main`). Its
`cuda_architectures` input defaults to `89` (RTX 40-series); use `86;89` for
RTX 30- and 40-series. Pull requests run it only when its workflow file changes.
It uploads a 14-day workflow artifact, not a release or a tag:

```
gh run download <run-id> --repo Quickkill0/openharness-public --name openharness-engine-windows-cuda-<8-character-commit> --dir engine-download
```

Extract the zip in that artifact, keeping its directories intact. `bin/`
contains `openharness-engine-server.exe`, its CUDA 13.3 runtime DLLs
(`cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`), and the app-local
Microsoft C++/OpenMP runtimes. `share/openharness-engine/` contains the installed
licenses/notices, NVIDIA runtime licenses and a separate Microsoft runtime
notice; `SHA256SUMS` covers every payload
file (not itself). The package also includes the runner's `engine-info.json`.
A compatible NVIDIA driver must already be installed on the GPU machine;
the package does not include a driver or the CUDA development toolkit.

The build uses the existing Windows helper with tests off, `BUILD_SHARED_LIBS=OFF`
and `GGML_NATIVE=OFF`: a fixed AVX2 CPU baseline, not the runner's detected ISA.
The validated CUDA virtual-memory pool and expert-tier kernels stay enabled.
CUDA 13.3's `cuda.lib` is a static driver loader, not a PE import library:
it loads `nvcuda.dll` on demand, allowing GPU-less startup without linker
delay-load flags. CI requires `GGML_CUDA_NO_VMM:BOOL=OFF` in the CMake cache
and `cuMemCreate`, `cuMemMap` and `cuMemAddressReserve` resolved from `cuda.lib`
in the final link map; missing inputs or symbols fail the job.
The fixed CPU baseline retains the SSD CPU sink's optimized kernels. MSVC C/C++
objects use sccache; nvcc is uncached because of its Windows quoted-definition
parser limitation ([sccache#2470](https://github.com/mozilla/sccache/issues/2470)).
A cold full-kernel build takes about an hour. CI runs the staged executable
without the toolkit on `PATH` and requires both `ssd_expert_streaming` and
`cuda_vram_expert_tier` to be true. Empty CUDA devices on that runner are normal:
this proves packaging and compiled capabilities, **not** GPU inference. The
binary still needs to be exercised on a real GPU, including the VRAM tier.

NVIDIA's [CUDA 13.3 EULA, Attachment A](https://docs.nvidia.com/cuda/archive/13.3.0/eula/index.html#attachment-a)
allows redistribution of `cudart.dll`, `cublas.dll` and `cublasLt.dll`, explicitly
including versioned/architecture-specific filenames such as these three DLLs.
This is conditional on section 1.1.2: distribute them only with an application
that adds material functionality, restrict their use to that application, and
use distribution terms consistent with NVIDIA's agreement. They remain under
NVIDIA's license, not the engine's open-source licenses; the package retains
the supplied license texts. No CUDA compiler, tools or driver is distributed.

## Build on Linux

Requirements: GCC 11+ or Clang 15+, CMake 3.21+, Ninja (optional), Python 3.9+
for the tests.

```
cmake -S engine -B engine/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DOH_ENGINE_BUILD_TESTS=ON
cmake --build engine/build --parallel 8
```

Add `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89` for a CUDA build.

The default build compiles only `openharness-engine-server`, what it links,
and the test tools; upstream's other tools are not built. `cmake --install
engine/build --prefix <dir>` installs the server into `bin/` and the license
and notice files into `share/openharness-engine/`, and nothing from upstream.

## Run

```
openharness-engine-server -m model.gguf --host 127.0.0.1 --port 8080
```

All `llama-server` options apply except the agent options listed above.
Readiness: `GET /health` returns 503 while
the model loads and 200 with `{"status":"ok"}` once it can serve requests.

`openharness-engine-server --engine-info` prints a single-line JSON capability
record and exits: engine version, upstream tag and commit, feature flags
(`ssd_expert_streaming`, `cuda_vram_expert_tier`, `web_ui`,
`remote_downloads`) and each backend's devices with total and free memory.
`cuda_vram_expert_tier` is true when the build compiled the tier, whether or not
this machine has a usable GPU; the backend list shows the devices.

## SSD expert streaming

```
openharness-engine-server -m model.gguf --ssd --ssd-set engine.ram_budget_mib=24576
```

With `--ssd` the routed experts of a mixture-of-experts model are not loaded
into memory. The model file is memory-mapped without read-ahead, and the
experts are read straight from the original GGUF (split files included) when
the router picks them. There is no repacked copy. Each read goes into a RAM
expert cache when one fits, or through a small read ring when it doesn't.
While one layer computes, the next layer's router is run early and its likely
experts are read ahead. Dense weights, attention, the KV cache and shared
experts stay resident as usual. Reads use IoRing on Windows (with an
overlapped `ReadFile` fallback) and io_uring or `pread` on Linux.

Settings are typed `key=value` pairs. Give them with `--ssd-set KEY=VALUE`
(repeatable) or as lines in a file with `--ssd-config FILE` (`#` starts a
comment). Either option implies `--ssd`. A key given twice, an unknown key or
an out-of-range value fails the load. The keys you will usually set:

| Key | Default | Meaning |
|---|---|---|
| `engine.ram_budget_mib` | physical RAM | Host memory the whole engine may use. The launcher should always set it. |
| `host.headroom_mib` | 512 | Part of the budget left unused, as a safety margin. |
| `engine.expert_cache_mib` | no cap | Upper bound on the RAM expert cache; `0` turns it off (read ring only). |
| `prefetch.depth` | 2 | Experts read ahead for the next layer; `0` turns it off. Only `qwen3moe` graphs supply the guess so far; other architectures stream without read-ahead. |
| `vram.budget` | `auto` | Bytes for the CUDA expert tier. `auto` and `0` keep every expert on the CPU. A size that does not fit is refused, naming the largest that does. |
| `vram.reserve_mib` | 0 | VRAM the tier leaves free for everything else on the card. |
| `vram.fail_allocation` | 0 | Test only: on the first decode, `1` fails the slot arena and `2` the compute engines, as an out-of-memory card would. |
| `vram.admit_all` | 0 | `1` admits every selected expert and evicts the least recently used slot, ignoring selection counts and measured costs. For tests and diagnosis; `0` is the production policy. |

The full list, with ranges, is in
`inferssd/ggml/src/ggml-ssd/settings/registry.def`.

The budget is spent in this order. Dense weights, the KV cache (attention,
SWA, indexer and recurrent state, per type), compute buffers, the rest of
the process and the headroom come first, and the expert cache gets what is
left. At load the engine logs one receipt line in MiB, for example:

```
[oh-ssd] receipt (MiB): budget=24576 headroom=512 dense_weights=... kv_attn=... compute_host=... process=... expert_cache=...
```

If the context does not fit, the load fails before the context's buffers are
allocated: the KV cells' bookkeeping (in a unified cache, as if every sequence
held every cell), the first output buffer and everything sized by the batch
sizes or by backend samplers are counted by arithmetic first. The error names the largest context that does
fit, in steps of 256 tokens, or says that no context fits, which is also the
answer when `--ubatch-size` alone is too large to size. On Windows and macOS the budget is
also enforced against the process's own memory (commit charge or footprint),
so the receipt counts the measured process too. On Linux the receipt is the
plan, a `memory.max` on the process's cgroup path lowers the budget, and that
limit is the hard cap. With no cgroup limit the RAM cache still runs, capped
by the receipt and by a floor on available memory. A cgroup directory with no
`memory.max` is passed over for its parent. When no memory controller applies
at all (cgroup v1 included), the engine warns on stderr and admits the context
against the RAM budget and MemAvailable. A `memory.max` that exists but cannot
be read or parsed refuses the load, and so does a failed read of
`/proc/self/cgroup`, because then the limit is unknown.

In router mode (`--models-dir`), every `--ssd-set` and `--ssd-config` reaches
the child.

Tested on synthetic fixtures: Qwen3-MoE with F16 and Q8_0 experts, and
GLM-5.3-Flash's shape with the expert types of the real file (IQ2_XS, IQ3_XXS
and IQ4_XS). GLM streams on the CPU with its shared expert and dense layers
resident. The model adapter also accepts the architectures InferSSD handled (Qwen3.5-MoE,
Qwen4exp, DeepSeek 4.1, Muse Spark) when the up and gate experts share one
type, but those have not been run here. A model it cannot stream fails the
load with the reason for the first layer it refuses. Not supported or not
tested yet:

- any GPU offload of the dense layers together with `--ssd`;
- the GPU prefill path: with `--ssd`, prefill always runs on the CPU, and only decode uses experts in VRAM;
- F32 experts. InferSSD's prefill has no F32 kernels, so the load fails and
  names the tensor. The same model loads without `--ssd`;
- GLM-5.3-Flash's draft (MTP) head with `--ssd`: the load is refused, so
  leave `--nextn` off and do not use the MTP draft speculation type. By
  default the head is not loaded. The real model has not been loaded yet;
- a named largest context for every GLM refusal. GLM's compute buffers shrink
  with the context by an amount only a load at that size shows, so when they
  alone are over budget the refusal says to try a smaller `-c`, or, at the
  smallest context, to raise the budget or lower the batch sizes;
- any real model;
- more than one sequence (`-np` > 1) with `--ssd`;
- the Linux budget beyond the receipt (no cgroup is created by the engine).

## Tests

```
ctest --test-dir engine/build -L engine --output-on-failure
```

The engine tests check the capability record, start the server on a tiny
generated model (CPU only) and call `/health`, `/v1/models`,
`/v1/completions` and `/v1/chat/completions`. They confirm that no web UI is
served, that a stale `dist/` is not embedded, and that a default launch
writes nothing to its working, home, app-data, temp or cache directories.
They also check that every remote model source, every agent option and
`--rpc` (flag, environment variable, `config.ini` and router preset) is
refused with zero requests reaching a local HTTP fixture and zero connections
reaching a raw TCP listener, that configuring with RPC on fails, that `/cors-proxy` and `/tools` are
404s, that `/health` answers 503 before 200 (test builds hold the model load
on a gate file, `OPENHARNESS_TEST_LOAD_GATE`, so this is deterministic), that `cmake --install` installs exactly the
server and its notices, and they test the vendor script offline (archive path
guards, provenance refusal, patch round-trip).

The GLM tests build a small model with GLM-5.3-Flash's shape in two steps,
because Python cannot write the IQ types: `tests/make_moe_fixture.py --glm`
writes the F32 base, and `ssd_stream_test glm-fixture` quantizes its experts
to the per-layer mix recorded from the real file's headers.

The SSD tests generate a tiny Qwen3-MoE model (`tests/make_moe_fixture.py`:
3 layers, 16 experts, top 4, a few MB, F16 and Q8_0 experts) and run it on
the CPU. `oh-ssd-stream-equal-*` compares a resident run against streamed
runs (read ring only with and without prefetch, a 1 MiB RAM cache that has
to evict, and a cache sized from a budget). It requires bitwise-identical
logits for a 40-token prompt and 24 greedy tokens, and checks from the
sink's counters that the experts were really read from the file and that
the cache hit and evicted as expected. `oh-ssd-context-budget` loads three
context sizes (1K, 8K and 32K tokens) against one budget and checks each
receipt line (KV bytes exact, cache takes the rest, cache shrinks as context
grows). It then checks that a context too large for the budget is refused
with a largest-fitting size that does load and generate, that a tiny budget
says no context fits, and that duplicate or malformed settings are refused.
InferSSD's own unit tests (`inferssd/tests/ssd`, label `ssd`) cover the
cache policy, read ring, budget arbiter and CPU sink in isolation.
Upstream's own tests are built too unless the compiler is MSVC (two upstream
test files at b12192 do not compile with MSVC 14.51; force them with
`-DOH_ENGINE_BUILD_UPSTREAM_TESTS=ON`) and run under the label `main`. The
GitHub workflow `.github/workflows/engine.yml` runs both on Linux (CPU only)
when files under `engine/` change, except changes to only Markdown files,
`NOTICE` or `LICENSES/`. It caches compiler output with ccache.

## Updating llama.cpp

```
python engine/scripts/vendor_llama_cpp.py verify             # current tree matches UPSTREAM.json + PATCHES.md
python engine/scripts/vendor_llama_cpp.py diff -o ours.patch  # save our changes
python engine/scripts/vendor_llama_cpp.py import b12345 --discard-local-changes
git apply --3way ours.patch                                  # re-apply, fix conflicts
python engine/scripts/vendor_llama_cpp.py verify
```

`verify` and `diff` download the recorded commit's tarball (about 40 MB) from
codeload.github.com every time; pass `--archive FILE` with a saved copy to run
them offline.

Run these from the repository root: the patch paths are rooted there
(`a/engine/llama.cpp/...`). `import` refuses to replace a tree with local
changes, or a nonempty tree with no `UPSTREAM.json` to check against, unless
`--discard-local-changes` is given.

`import` downloads the upstream source archive for a tag or commit, prunes it
to the paths listed in the script, writes it to `llama.cpp/` without any
`.git`, and records the tag, commit and digest in `UPSTREAM.json`. Pruning
drops backends we do not build, docs, examples, model vocab files and the web
UI sources.

## Licenses

llama.cpp and ggml are MIT (`LICENSES/llama.cpp-MIT.txt`). InferSSD-derived
code is Apache-2.0 (`LICENSES/InferSSD-Apache-2.0.txt`, with InferSSD's
`NOTICE` in `LICENSES/InferSSD-NOTICE.txt`). The server also statically links
vendored third-party code (cpp-httplib, nlohmann/json, xxHash and others);
their notices are in `LICENSES/THIRD-PARTY-NOTICES.txt`, which is installed
with the binary. See `NOTICE` for how modified files are marked.
