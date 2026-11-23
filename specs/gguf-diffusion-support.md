# Spec: ComfyUI-style GGUF diffusion support in Studio

Status: **Planning only — not implemented.** Author: research sub-agent, 2026-06-26.

## 1. Goal

Make three already-installed single-file GGUF diffusion quants actually generate in
Studio:

| Repo | Parsed arch | File on disk | Quant | Size | Studio task |
|---|---|---|---|---|---|
| `unsloth/Qwen-Image-2512-GGUF` | `qwen_image` | `…\58c7fcea….gguf` (`qwen-image-2512-Q6_K.gguf`) | Q6_K | 16.8 GB | text-to-image |
| `unsloth/LTX-2.3-GGUF` | `ltxv` | `…\191b06ba….gguf` (`ltx-2.3-22b-dev-UD-Q5_K_M.gguf`) | Q5_K_M | 18.3 GB | text-to-video |
| `QuantStack/Wan2.2-I2V-A14B-GGUF` | `wan` | `…\029c7adc….gguf` (`Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf`) | Q8_0* | 15.4 GB | image-to-video |

*The DB row records `quantization = Q5_1`; the actual installed file is the **LowNoise
Q8_0** expert. Treat the DB quant string as unreliable for these repos.

These are **ComfyUI-style single-file GGUFs**: they hold **only the quantized diffusion
transformer (DiT)** — no VAE, no text encoder(s), no tokenizer, no scheduler config. The
current worker has **no GGUF loader** and the Rust resolver routes these archs into a
Chat+Agent catch-all (the bug). This spec plans the fix.

## 2. Verified findings (read-only probes, no GPU loads)

### 2.1 Worker library versions (installed venv)

```
diffusers     0.33.1
torch         2.6.0+cu124
transformers  4.51.3
accelerate    1.6.0
gguf          0.16.0   (pinned in requirements.cuda.txt)
```

### 2.2 GGUF headers confirm DiT-only payloads

Parsed each file's header with the `gguf` lib (header only, no weight load):

- **Qwen-Image** — `general.architecture = qwen_image`, 1933 tensors, prefixes
  `img_in / txt_in / txt_norm / time_text_embed / transformer_blocks / norm_out / proj_out`.
  **No VAE, no text encoder.** Pure MMDiT transformer.
- **LTX** — `general.architecture = ltxv`, 4444 tensors, prefixes
  `patchify_proj / transformer_blocks / scale_shift_table / proj_out` **plus** a large
  block of `audio_*` / `av_ca_*` / `*_embeddings_connector` tensors → this is **LTX-2
  with audio conditioning**, but still **transformer-only** (no VAE, no T5 text encoder).
- **Wan** — `general.architecture = wan`, 1095 tensors, prefixes
  `patch_embedding / blocks / text_embedding / time_embedding / time_projection / head`.
  **No VAE, no UMT5 text encoder, no CLIP vision.** This is **one expert** (LowNoise) of
  the Wan 2.2 A14B MoE.

### 2.3 diffusers 0.33.1 class/loader support matrix

`hasattr(diffusers, …)`:

| Class | Present? | Notes |
|---|---|---|
| `GGUFQuantizationConfig` | ✅ | `GGUFQuantizationConfig(compute_dtype=…)` |
| `WanTransformer3DModel` | ✅ | in `SINGLE_FILE_LOADABLE_CLASSES`, `from_single_file` works |
| `WanPipeline` / `WanImageToVideoPipeline` | ✅ / ✅ | see Wan caveat below |
| `AutoencoderKLWan` | ✅ | in `SINGLE_FILE_LOADABLE_CLASSES` |
| `LTXVideoTransformer3DModel` | ✅ | in `SINGLE_FILE_LOADABLE_CLASSES`, `from_single_file` works |
| `LTXPipeline` / `LTXImageToVideoPipeline` / `LTXConditionPipeline` | ✅ / ✅ / ✅ | |
| `AutoencoderKLLTXVideo` | ✅ | in `SINGLE_FILE_LOADABLE_CLASSES` |
| `QwenImageTransformer2DModel` | ❌ | **absent in 0.33.1** |
| `QwenImagePipeline` | ❌ | **absent in 0.33.1** |
| `AutoencoderKLQwenImage` | ❌ | **absent in 0.33.1** |

`from_single_file(...)` accepts both `config=<base repo/dir>` (to pull the architecture
config + companion components) **and** `quantization_config=GGUFQuantizationConfig(...)`.
This is the load path for all three.

### 2.4 The Wan 2.2 dual-expert caveat (important)

`WanImageToVideoPipeline` in 0.33.1 has `__init__(tokenizer, text_encoder, image_encoder,
image_processor, transformer, vae, scheduler)` and **no `transformer_2` / `boundary_ratio`**.
That is the **Wan 2.1** orchestration. The Wan 2.2 A14B is a **two-expert MoE** (HighNoise
+ LowNoise transformers, swapped at a `boundary_ratio` timestep). diffusers added
`transformer_2`/`boundary_ratio` to the Wan pipelines in **0.34+**. Consequences:

- The installed file is only the **LowNoise** expert. Running it alone through the 0.33.1
  Wan2.1-style pipeline is **not the intended Wan2.2 dual-expert inference** — quality will
  be degraded and it is not the path upstream/ComfyUI uses.
- 0.33.1's `WanImageToVideoPipeline` also wants a **CLIP image encoder** (`image_encoder` +
  `image_processor`) that the Wan **2.1** I2V checkpoints shipped. Wan **2.2** I2V does **not
  use** a CLIP image encoder the same way — another mismatch.

→ **Wan i2v is the hardest of the three and is gated on a diffusers bump to ≥0.34 (realistically
0.35.x) to get true dual-expert support.** Planned last, behind a version-bump risk flag.

### 2.5 Worker plumbing facts

- Pipelines dispatch via `PIPELINES` dict in
  `src-tauri/worker-py/openharness_worker/pipelines/__init__.py` keyed by task name
  (`text_to_image`, `text_to_video`, …). Each module exposes `parse_params`, `_load`,
  `preload`, `run`, `unload`.
- `text_to_image.py` already has a `from_single_file` branch keyed on `arch ∈ {sdxl,sd,flux}`.
- `text_to_video.py` is **`from_pretrained`-only** (diffusers-dir) and **t2v-only** (no input
  image) — both gaps for our GGUFs.
- The Rust→worker `model` param for a GGUF is the **hashed `primary_file_path`**
  (`…\<sha256>.gguf`), because `local_path` is NULL for single-file GGUFs
  (`generation.rs::preload_pipeline`, lines 137-141; same in `submit_pipeline_job`).
- VRAM gate estimate: `generation.rs::estimate_task_vram_mb` for `text_to_image|text_to_video`
  = `path_size_mb(model) + PIPELINE_OVERHEAD_MB`. For a single-file GGUF this counts only the
  transformer file — **the companion VAE/text-encoder VRAM is NOT counted** today. Fix below.
- Frontend already passes `arch` (image) and `model` (video) in job params
  (`ImageTask.svelte` line 115, `VideoTask.svelte`); `VideoTask` has an `ltx | wan` family
  switch by name.

## 3. The crux: where the companion components come from

Each GGUF lacks VAE + text encoder(s) + tokenizer + scheduler. They must be sourced
locally (offline, zero telemetry). Normal HF sources and sizes:

| Arch | VAE | Text encoder | Tokenizer/scheduler | Approx download |
|---|---|---|---|---|
| `qwen_image` | `vae/` from `Qwen/Qwen-Image` (diffusers) ~250 MB | Qwen2.5-VL-7B text encoder (`text_encoder/`) — **~16 GB bf16**, or an fp8 community build ~8 GB | tokenizer + scheduler configs (tiny) | **~16 GB** (or ~8 GB fp8) |
| `ltxv` | `vae/` from `Lightricks/LTX-Video` ~1.5 GB (LTX-2 VAE may differ; verify) | T5-XXL (`google/t5-v1_1-xxl` or the LTX-bundled `text_encoder/`) — **~9.5 GB fp16**, fp8 ~5 GB | T5 tokenizer + scheduler (tiny) | **~11 GB** (or ~6 GB fp8) |
| `wan` (2.2 I2V) | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` `vae/` (AutoencoderKLWan) ~500 MB | UMT5-XXL (`text_encoder/`) — **~11 GB fp16**, fp8 ~5.5 GB | UMT5 tokenizer + scheduler (tiny). Wan 2.2 I2V does **not** need the 2.1 CLIP image encoder | **~11.5 GB** (or ~6 GB fp8) |

> These sizes are from architecture knowledge + the standard companion repos; **no web
> fetch was performed** (no confirmed web access in this run). Treat the exact repo
> revisions and the LTX-2-specific VAE/text-encoder pairing as **open questions to verify
> before P2/P3 implementation** (§9). The fp8 encoder option exists for all three and is the
> recommended default to keep the 24 GB budget feasible.

### 3.1 Companion-sourcing options evaluated

**(a) Auto-download a curated companion repo per arch via the existing downloader.**
On first use of a GGUF whose arch needs companions, resolve a pinned "companion pack"
(VAE + encoder + tokenizer/scheduler) and fan out a download group
(`start_repo_download` already supports groups, migration 0008). Pros: one-click, reuses
all download/verify/catalog machinery, offline thereafter. Cons: violates the **"app never
auto-downloads"** post-M16 invariant if done silently; large (6–16 GB).

**(b) A "support pack" the user installs from the Store (explicit).** Surface the companion
pack as an installable Store entry (or a one-click banner in Studio: "This GGUF needs its
support files — download (~6 GB)"). The user clicks; download is explicit. Pros: honours
the no-silent-download rule, reuses Store/downloader/catalog, fully offline after.
Cons: needs a curated, pinned manifest per arch + new "companion" catalog concept.

**(c) Require the user to point Studio at a full base diffusers repo.** User downloads e.g.
`Qwen/Qwen-Image` themselves; Studio loads the GGUF transformer + that repo's VAE/encoder
via `from_single_file(..., config=<base repo dir>)`. Pros: zero new download UX, dead simple
backend. Cons: worst UX (user must know which base repo, may double-download the fp16
transformer they don't need).

### 3.2 Recommendation

**Adopt (b), the explicit Store "support pack", with (c) as the power-user escape hatch.**

- A static, pinned **companion manifest per arch** (`qwen_image` / `ltxv` / `wan`) listing
  the VAE + (fp8-preferred) text-encoder + tokenizer/scheduler **download_files subset** and
  total size. Lives in code (mirrors the existing `store_curated_picks` static manifest +
  the qwentts `RuntimeManifest` pattern) — **not** a network manifest.
- Installed companion packs live under a new on-disk root, e.g.
  `%LOCALAPPDATA%\OpenHarness\models\companions\<arch>\` (diffusers-layout subdirs:
  `vae/`, `text_encoder/`, `tokenizer/`, `scheduler/`), registered in the catalog as a new
  lightweight kind so presence is queryable.
- Studio shows an honest **"needs companion files"** banner (reusing the `caps.reason`
  callout pattern from M16 W5) with a one-click **explicit** download when the pack is
  absent — never silent, satisfying the no-auto-download invariant.
- Escape hatch (c): an "Advanced → point at a base diffusers repo" field that sets a
  `companion_dir` job param, bypassing the managed pack.

This is the smallest change that stays offline + explicit + reuses the downloader, and it
keeps the 24 GB budget feasible by defaulting to fp8 encoders.

## 4. Feasibility verdict per model

| Model | Verdict | Why |
|---|---|---|
| **Qwen-Image (`qwen_image`)** | **Blocked on diffusers bump.** `QwenImageTransformer2DModel`/`QwenImagePipeline`/`AutoencoderKLQwenImage` are **absent in 0.33.1**. Need diffusers **≥0.35** (QwenImage landed ~0.34/0.35). Once bumped: GGUF transformer loads via `QwenImageTransformer2DModel.from_single_file(..., quantization_config=GGUFQuantizationConfig)`, assembled into `QwenImagePipeline` with companion VAE + Qwen2.5-VL text encoder. | Class availability is the hard gate. |
| **LTX (`ltxv`)** | **Most feasible.** `LTXVideoTransformer3DModel` + `LTXPipeline` + `AutoencoderKLLTXVideo` all present in 0.33.1; transformer is single-file GGUF-loadable. **Caveat:** the file is **LTX-2.3 with audio tensors**; 0.33.1's LTX pipeline targets **LTX-Video (LTX-0.9.x)** and may not accept the 2.3 transformer config or the audio path. **Verify the 2.3 transformer loads under 0.33.1; if not, bump to the diffusers release that adds LTX-2 support.** | Classes exist but a 2.3-vs-0.9.x config mismatch is a real risk. |
| **Wan 2.2 I2V A14B (`wan`)** | **Hardest; needs diffusers ≥0.34 (realistically 0.35.x) for dual-expert.** Single-expert through the 0.33.1 Wan2.1 pipeline is degraded + wrong (CLIP-image-encoder mismatch, no `boundary_ratio`). Also our t2v worker has **no i2v path** (no input-image plumbing). Even with a bump, only the LowNoise expert is installed — true A14B inference wants **both** experts (~2×15 GB) which **won't co-reside on 24 GB** without aggressive offload/expert-swap. | MoE + i2v + version bump + VRAM all compound. |

**Net:** All three need attention beyond "just add a loader." LTX t2v is the only one likely
shippable on the **currently-installed** diffusers (pending a 2.3-config verify). Qwen-Image
and Wan both require a **diffusers version bump** — a worker-requirements change that re-syncs
the venv for every user (risk/debt, flagged).

## 5. Rust changes

### 5.1 `models/capabilities.rs` resolver reroute

`resolve_other_gguf` currently sends `wan`/`ltxv`/`qwen_image` to the Chat+Agent catch-all
(they have `Capability::Other`, no card signal). Reroute by arch **before** the catch-all:

- Add a **diffusion-arch table** mapping GGUF `general.architecture` → (worker pipeline,
  workflow, needs-companion):
  - `qwen_image` → (`text_to_image`, `Image`)
  - `ltxv` → (`text_to_video`, `Video`)
  - `wan` → (`text_to_video`, `Video`) — i2v handled by an input-image param, not a separate
    pipeline name (worker decides t2v vs i2v by presence of `image`).
  - Keep a **denylist** of other known DiT archs that we explicitly do **not** support yet
    (`hunyuan-video`, `mochi`, `cogvideox`, `sd3`, `auraflow`, `lumina2`, …) → honest
    `not_runnable` reason, so they never leak into Chat+Agent either.
- Extend `ModelCaps` with fields to carry the GGUF-single-file diffusion info:
  ```rust
  /// True when this is a single-file GGUF diffusion transformer that needs
  /// companion VAE + text-encoder files to run.
  pub gguf_diffusion: bool,
  /// Companion pack key ("qwen_image" | "ltxv" | "wan") when gguf_diffusion.
  pub companion_arch: Option<String>,
  ```
  (Default `false`/`None`; `#[serde(default)]` so the TS side and existing tests are
  unaffected where unset.) `image_arch` stays for SD/SDXL/Flux single-files; the new
  `companion_arch` is the diffusion-GGUF discriminator.
- Set `pipeline = Some("text_to_image"|"text_to_video")`, the matching workflow, and
  `gguf_diffusion = true`, `companion_arch = Some(arch)`. Leave `reason = None` (it IS
  runnable once the companion pack is present) — the **companion-presence** check is a
  separate UI/command concern (§7), not a `caps.reason` (caps stay pure of filesystem state,
  per the read-time-pure constraint).

No migration needed — caps are computed in `from_row` at read time (M15 invariant).

### 5.2 Tests

- Update `gguf_diffusion_checkpoint_is_honestly_not_runnable` (currently asserts `flux` GGUF
  is not runnable — keep that, flux GGUF stays unsupported) and add:
  - `qwen_image_gguf_routes_to_image_worker` → `runtime=Worker`, `pipeline=text_to_image`,
    `workflows=[Image]`, `gguf_diffusion=true`, `companion_arch=Some("qwen_image")`.
  - `ltxv_gguf_routes_to_video_worker` → `text_to_video`, `[Video]`, `companion_arch="ltxv"`.
  - `wan_gguf_routes_to_video_worker` → `text_to_video`, `[Video]`, `companion_arch="wan"`.
  - `unsupported_diffusion_gguf_arch_is_honestly_not_runnable` (e.g. `hunyuan-video`) →
    `runtime=None`, reason set, **no** Chat/Agent.
- Keep all 228 existing tests green (the reroute only moves three archs that previously hit
  the catch-all; assert they no longer get Chat+Agent).

### 5.3 `generation.rs` VRAM estimate + param assembly

- `estimate_task_vram_mb`: when the job is a GGUF diffusion model, add the **companion pack
  size** to the estimate (transformer file + VAE + encoder + overhead). Pass the companion
  size in via a job param or look it up from the static companion manifest by `companion_arch`.
  Otherwise the gate under-counts by ~6–16 GB and over-promises co-residency.
- `preload_pipeline` / `submit_pipeline_job`: when `caps.gguf_diffusion`, add
  `params["gguf"] = true`, `params["companion_arch"] = <arch>`, and
  `params["companion_dir"] = <resolved companion pack path>` (or the user's base-repo
  override) so the worker can assemble the pipeline. For image, `arch` is **not** set to
  sd/sdxl/flux (those are the legacy single-file branches) — the worker keys on
  `companion_arch` instead.

### 5.4 New command(s) for companion packs

- `companion_status(arch)` → installed? path? size? (queries catalog/filesystem).
- `download_companion_pack(arch)` → fans out the pinned companion `download_files` as a
  download group (reuses `start_repo_download` machinery) into
  `models\companions\<arch>\`. Explicit, user-initiated only.
- A static `companion_manifest(arch)` fn (code, pinned) listing repo + files + total size,
  mirroring `store_curated_picks` / `qwentts_manifest` patterns.

## 6. Worker changes

### 6.1 Shared GGUF loader helper

Add `pipelines/_gguf.py` with a helper:

```python
def load_gguf_transformer(cls, gguf_path, *, config_dir, compute_dtype):
    from diffusers import GGUFQuantizationConfig
    return cls.from_single_file(
        gguf_path,
        quantization_config=GGUFQuantizationConfig(compute_dtype=compute_dtype),
        config=config_dir,            # base diffusers repo/dir for the arch config
        torch_dtype=compute_dtype,
        local_files_only=True,
    )
```

Companion VAE / text_encoder / tokenizer / scheduler are loaded with
`from_pretrained(companion_dir/<subdir>, local_files_only=True)` and passed into the concrete
pipeline constructor.

### 6.2 `text_to_image.py`

- In `parse_params`/`_load`: when `params["gguf"]` and `companion_arch == "qwen_image"`:
  - Load `QwenImageTransformer2DModel.from_single_file(gguf, quantization_config=…,
    config=<companion qwen-image dir>)`.
  - Build `QwenImagePipeline(transformer=…, vae=AutoencoderKLQwenImage.from_pretrained(...),
    text_encoder=…, tokenizer=…, scheduler=…)`.
  - **Requires diffusers ≥0.35** (classes absent in 0.33.1) — guard with a clear
    `RuntimeError("Qwen-Image GGUF needs diffusers ≥0.35; the worker venv has <ver>")` until
    the bump lands, so P1 ships an honest interim error rather than an ImportError traceback.
- Keep the existing sd/sdxl/flux single-file branches untouched.

### 6.3 `text_to_video.py`

- Accept single-file GGUF (`path.is_file()` + `params["gguf"]`) in addition to diffusers-dir.
- `ltxv` branch: `LTXVideoTransformer3DModel.from_single_file(gguf, quantization_config=…,
  config=<companion ltx dir>)` → `LTXPipeline(transformer=…, vae=AutoencoderKLLTXVideo…,
  text_encoder=T5…, tokenizer=…, scheduler=…)`. **Verify the LTX-2.3 transformer config loads
  under the installed diffusers**; if not, gate with an honest "needs diffusers <X>" error.
- `wan` branch: `WanTransformer3DModel.from_single_file(...)`. **i2v gap:** the worker's
  `run()` has no input-image plumbing. Add an optional `image` param (path to an input frame);
  when present route to `WanImageToVideoPipeline` (i2v), else `WanPipeline` (t2v). **Wan 2.2
  dual-expert + i2v needs diffusers ≥0.34/0.35** — gate honestly until the bump; document that
  only the LowNoise expert is installed and true A14B wants both.
- Add `image` to `parse_params` (optional; validates the file exists). For Wan i2v this is
  required.

### 6.4 VRAM / offload strategy (24 GB RTX 4090)

The text encoder is the budget killer. Strategy, in order of preference:

1. **fp8 text encoder by default** in the companion pack (halves encoder VRAM: UMT5-XXL/T5-XXL
   ~5–6 GB, Qwen2.5-VL ~8 GB). This alone makes LTX + Qwen-Image co-residence plausible.
2. **Encode-then-free:** run the text encoder, capture `prompt_embeds`/`negative_prompt_embeds`,
   then `del` the encoder + `torch.cuda.empty_cache()` before the denoise loop, so encoder and
   DiT never co-reside. diffusers pipelines accept precomputed `*_embeds` and skip the encoder.
   This is the highest-leverage trick for the big-encoder cases (Wan UMT5-XXL, Qwen2.5-VL).
3. **`enable_model_cpu_offload()`** (the existing `offload` param) when still tight — moves
   submodules to CPU between stages. Slower but safe. Already wired in both pipelines.
4. **`enable_sequential_cpu_offload()`** as the last resort for the worst case (Wan A14B with
   both experts) — much slower, layer-by-layer streaming.

Per-model feasibility on 24 GB:
- **LTX (transformer 18.3 GB Q5_K_M + fp8 T5)** — tight; **encode-then-free is essentially
  required** so the 18.3 GB transformer has the card to itself during denoise. Feasible.
- **Qwen-Image (16.8 GB Q6_K + fp8 Qwen2.5-VL ~8 GB)** — encode-then-free → transformer alone
  fits comfortably in 24 GB. Feasible.
- **Wan 2.2 A14B i2v (15.4 GB LowNoise + fp8 UMT5 ~5.5 GB, single expert)** — encode-then-free
  → single expert fits. **True dual-expert (both ~15 GB) does NOT co-reside** → requires
  expert-swap (load LowNoise, denoise its phase, unload, load HighNoise) or sequential offload;
  this is the **A14B feasibility cliff** and the main reason Wan is P4 and uncertain.

## 7. Orchestrator / VRAM ledger

- The ledger gates pipeline loads on `est_mb` (`reserve_pipeline`). The estimate must reflect
  **transformer + companion** footprint (§5.3), and ideally the **encode-then-free peak**
  (the larger of {encoder loaded, transformer loaded}, not their sum) so the gate isn't
  needlessly pessimistic. Simplest correct first cut: estimate = transformer file size +
  companion-pack size + `PIPELINE_OVERHEAD_MB`; refine to the peak-not-sum model once the
  encode-then-free path is in.
- The worker already reports **NVML-actual VRAM** in `job_done` meta, which the ledger adopts
  (`actual_mb`), self-correcting any estimate error after the first run — so a coarse estimate
  is acceptable for the gate.
- No new `ResidentKind` needed; these are ordinary `pipeline:text_to_image` /
  `pipeline:text_to_video` residents.

## 8. UI / wiring (Svelte 5 runes only)

- **ImageTask / VideoTask** already pass `model`; ImageTask passes `arch`. For GGUF diffusion
  the store/`submitJob` path picks up `gguf`/`companion_arch`/`companion_dir` from caps +
  command resolution (set in `generation.rs`, §5.3) — the task components need **no new
  user-facing arch control** for these (the model's caps drive it). VideoTask's `ltx|wan`
  family switch by name already covers fps/resolution presets; extend the `wan` branch to
  surface an **input-image picker** when the selected model is Wan i2v (`companion_arch=="wan"`
  + i2v) since i2v needs an `image` param.
- **Model pickers** (`ModelPicker` `capsFilter`/`hfTask`): these GGUFs now have
  `caps.pipeline = text_to_image|text_to_video` and the right workflow, so they appear in the
  Studio Image/Video pickers automatically once the resolver reroute lands. No filter change
  needed beyond verifying `capsFilter` keys on `workflow`/`pipeline`, not on `format`.
- **"Needs companion files" UX:** a banner (reusing the M16 `caps.reason`/callout styling)
  when `caps.gguf_diffusion` and `companion_status(arch)` reports absent → one-click **explicit**
  `download_companion_pack(arch)` with the size, plus an **Advanced** "use a base diffusers
  repo instead" field (`companion_dir` override).
- All new state via `$state`/`$derived`/`$effect`/`$props` (no `export let`, no `$:`).

### 8.1 Mock-IPC parity (`src/lib/ipc/mock.ts`, `resolveMockCaps`)

- Mirror the Rust resolver: in the `m.format === 'gguf'` branch, before the chat/agent
  fallback, add arch routing for `qwen_image`/`ltxv`/`wan` → `{ runtime:'worker',
  pipeline:'text_to_image'|'text_to_video', workflows:['image'|'video'], gguf_diffusion:true,
  companion_arch:<arch> }`, and a denylist of unsupported diffusion archs → `reason`.
- Add the three GGUF models to the mock catalog (mirroring the real DB rows) so the Studio
  pickers show them under mock IPC, plus a mock `companion_status`/`download_companion_pack`
  that simulates the explicit pack download (reuse the existing group-download simulator).
- Add `gguf_diffusion`/`companion_arch` to the mock `ModelCaps` type default (`false`/`null`).

## 9. Open questions / risks (resolve with user before P2+)

1. **diffusers version bump.** Qwen-Image and Wan2.2 dual-expert need diffusers **≥0.35**.
   Bumping `requirements.cuda.txt`/`requirements.cpu.txt` re-syncs the venv for **every user**
   (the Rust bootstrap hashes the file into the venv stamp → clean re-sync). Risk: 0.35 may
   bump `transformers`/`accelerate` mins, which VibeVoice pins to `transformers==4.51.3`,
   `accelerate==1.6.0` exactly — a conflict could break VibeVoice. **Must verify the dependency
   matrix before bumping.** This is the single biggest risk.
2. **LTX-2.3 under installed diffusers.** Does the **2.3** transformer config (with the audio
   tensors) load via `LTXVideoTransformer3DModel.from_single_file` on 0.33.1, or does it also
   need a newer diffusers? If newer, LTX collapses into the same bump as Qwen/Wan and P3
   loses its "ships today" advantage. **Verify with a header-config probe / a dry
   `from_single_file` on CPU (no GPU) before committing P3 scope.**
3. **Companion repo revisions + LTX-2 VAE/encoder pairing.** Exact pinned repos/files/sizes
   for each companion pack are reasoned from architecture, **not web-verified** in this run.
   Confirm: Qwen-Image companion = `Qwen/Qwen-Image` vae + Qwen2.5-VL encoder; LTX-2.3 VAE
   (does 2.3 use a different VAE than LTX-0.9.x?); Wan 2.2 = `Wan-AI/Wan2.2-I2V-A14B-Diffusers`
   vae + UMT5-XXL. Confirm fp8 encoder builds exist + are trustworthy (offline, no
   `trust_remote_code` surprises).
4. **No-auto-download invariant.** Confirm the recommended explicit "support pack" UX (§3.2)
   is acceptable, and where it surfaces (Studio banner vs a Store entry vs both).
5. **Wan A14B realism.** Is single-expert (LowNoise-only, degraded) acceptable as an interim,
   or is the bar full dual-expert? If full, Wan i2v is a substantial effort (expert-swap or
   sequential offload + both experts downloaded — the second expert isn't even installed).
6. **`trust_remote_code`.** Qwen-Image's text encoder (Qwen2.5-VL) and some video pipelines may
   want custom code. The deferred M15 "generic transformers/diffusers auto-pipelines with a
   `trust_remote_code` consent gate" overlaps here — decide whether to fold that gate in.

## 10. Phasing (independently-shippable PRs, smallest valuable slice first)

**P1 — Resolver reroute + honest interim state (no generation yet).**
- `capabilities.rs`: route `qwen_image`/`ltxv`/`wan` → Studio image/video with
  `gguf_diffusion`/`companion_arch`; denylist other DiT archs; new `ModelCaps` fields.
  Tests. `resolveMockCaps` parity + mock catalog entries.
- Worker pipelines gain GGUF branches **guarded** to raise an honest "needs companion files /
  needs diffusers ≥X" error (no silent failure). Studio shows the model under Image/Video with
  a "needs support files" state.
- **Value:** the three models stop being mis-routed to Chat+Agent; the bug in the prompt is
  fixed; users get an honest explanation instead of a broken chat load. **Ships on current
  diffusers.** Low risk.

**P2 — Companion pack infra + Qwen-Image t2i end-to-end** *(depends on diffusers bump)*.
- Companion manifest + `companion_status`/`download_companion_pack` + on-disk
  `models\companions\<arch>\` + Studio explicit-download banner.
- diffusers bump to ≥0.35 (after the §9.1 dependency-matrix verify) → `text_to_image.py`
  QwenImage GGUF path with encode-then-free. Real-GPU smoke (STOP-before-GPU per autonomy
  mandate).
- **Value:** first GGUF image generation; establishes all companion infra reused by P3/P4.

**P3 — LTX-Video t2v GGUF end-to-end.**
- `text_to_video.py` ltxv GGUF branch + LTX companion pack. **Gated on §9.2** — if LTX-2.3
  loads on current diffusers, P3 can precede P2's bump; if not, it rides P2's bump. Sequence
  P3 before/after P2 based on that probe.
- **Value:** first GGUF video generation.
- **PROBE RESOLVED (2026-06-26, see §12): BLOCKED.** LTX-2.3's GGUF is an
  `AVTransformer3DModel` (LTX-2), absent in diffusers 0.33.1; it needs diffusers ≥ ~0.39
  (PR #12915). No "ships today" shortcut — **P3 rides the same diffusers bump as P2/P4**, and
  the P1 honest-interim reject stays correct for the current venv. Implementation deferred
  until the bump is approved.

**P4 — Wan 2.2 I2V GGUF (hardest, most uncertain).**
- i2v input-image plumbing in `text_to_video.py` + VideoTask image picker; Wan companion pack;
  dual-expert handling (expert-swap or sequential offload) — **only after §9.1/§9.5 are
  resolved** and (likely) the second expert is downloadable. May land single-expert-degraded
  first behind a clear quality caveat, full dual-expert later.
- **Value:** completes the trio; explicitly the riskiest, planned last.

## 11. Constraints honoured

Svelte 5 runes only; single-writer orchestrator actor unchanged (new residents are ordinary
pipeline residents); `sqlx::query!` macros for any companion-catalog rows (no raw SQL);
local-first / offline / zero-telemetry (companion packs are explicit, pinned, `local_files_only`
loads); Windows-first NVIDIA CUDA (fp8 encoders + 24 GB offload strategy); worker stays
JSON-Lines-over-stdio with lazy-imported pipelines; worker auto-managed (the diffusers bump
rides the existing venv-stamp re-sync). Caps stay read-time-pure (companion **presence** is a
command/UI concern, not a `caps.reason`).

## 12. P3 feasibility probe — VERDICT (2026-06-26, resolves §9.2 + §9.3)

**P3 is BLOCKED on a diffusers bump. The installed LTX-2.3 GGUF does NOT load under
diffusers 0.33.1. Video-first does NOT dodge the bump — LTX rides the same diffusers
upgrade as Qwen-Image and Wan.** No implementation was done; per the P3 hard gate, scope
stops at this probe.

### 12.1 §9.2 — does LTX-2.3 load on diffusers 0.33.1? **NO.** Evidence (read-only, no GPU):

The installed GGUF `…\191b06ba….gguf` (`ltx-2.3-22b-dev-UD-Q5_K_M`, `general.architecture =
ltxv`, 4444 tensors) **carries its own diffusers config in a GGUF `config` metadata field.**
That config is authoritative and self-describing:

- `config.transformer._class_name = "AVTransformer3DModel"` — **not** `LTXVideoTransformer3DModel`.
  `hasattr(diffusers, "AVTransformer3DModel")` is **False** on 0.33.1. The class does not exist.
- The transformer config is a **different architecture**, not just a bigger one:
  `num_layers=48` (0.33.1's `LTXVideoTransformer3DModel` defaults to 28),
  `attention_head_dim=128` (0.33.1: 64), `cross_attention_dim=4096` (0.33.1: 2048),
  `caption_channels=3840` (0.33.1: 4096), plus an **entire audio-video cross-attention
  subsystem** with no representation in the 0.33.1 class: `use_audio_video_cross_attention`,
  `audio_num_attention_heads`, `connector_*` (learnable-register embeddings connectors),
  `av_cross_ada_norm`, `rope_type="split"`, `frequencies_precision="float64"`, etc.
- The GGUF tensors match that config, not the 0.33.1 class: prefixes include
  `audio_embeddings_connector` (129), `video_embeddings_connector` (129), `av_ca_*`,
  `audio_patchify_proj`, `audio_proj_out`, `audio_adaln_single`, `prompt_adaln_single`, …
  none of which `LTXVideoTransformer3DModel` defines.
- 0.33.1's `convert_ltx_transformer_checkpoint_to_diffusers` (in `single_file_utils`) is a
  blind LTX-0.9.x key-rename (`patchify_proj→proj_in`, `adaln_single→time_embed`) feeding a
  fixed-shape `LTXVideoTransformer3DModel`. It has no path for the 2.3 audio/connector keys;
  `from_single_file` would error on config/shape mismatch even if it got that far.

A full `from_single_file` attempt was deliberately **not** run (it would materialize the
18 GB on CPU RAM); the config-vs-class mismatch above is conclusive without it.

**Where LTX-2 support actually lands:** diffusers PR
[#12915](https://github.com/huggingface/diffusers/pull/12915) ("LTX 2.0 Video Pipelines")
merged to **main on 2026-01-08**, introducing `LTX2VideoTransformer3DModel` (the GGUF's
`AVTransformer3DModel` is the dev-branch name for the same class), `LTX2Pipeline` /
`LTX2ImageToVideoPipeline` / `LTX2ConditionPipeline` / `LTX2LatentUpsamplePipeline`, plus a
video VAE, **audio VAE decoder, and vocoder** (LTX-2 generates synced audio). Context clues
in the PR point at a ~**0.39** release. So LTX-2.3 needs diffusers **≥ ~0.39** — strictly
newer than the ≥0.35 the spec assumed for Qwen/Wan, i.e. **the same or a bigger bump**, with
all of §9.1's VibeVoice-pin risk (`transformers==4.51.3`, `accelerate==1.6.0`) and then some.

**Consequence for §10 phasing:** the spec's P3 premise ("ships on currently-installed
diffusers, can precede P2's bump") is **false**. P3 cannot precede the bump; it rides it. The
spec's planned load path (`LTXVideoTransformer3DModel` + `LTXPipeline` + `AutoencoderKLLTXVideo`
on 0.33.1) targets **LTX-Video 0.9.x**, a model nobody has installed — it would not run the
installed LTX-2.3 file. The P1 honest-interim reject in `text_to_video.py` is therefore the
**correct end state for the current venv**; it should stay until a diffusers bump lands.

### 12.2 §9.3 — LTX-2.3 companion pack (pinned for the post-bump implementation)

Verified against the diffusers LTX-2 docs + the GGUF's embedded `config`. Pin these once the
bump is approved — do NOT wire them now (no pipeline class exists to consume them on 0.33.1):

- **Pipeline / transformer / VAE:** `LTX2Pipeline` + `LTX2VideoTransformer3DModel` (load the
  GGUF DiT via `from_single_file`) + `AutoencoderKLLTXVideo` VAE. Base diffusers repo:
  **`Lightricks/LTX-2`** (or `diffusers/LTX-2.3-Diffusers` / `Lightricks/LTX-2.3`).
- **Text encoder:** **T5-XXL — `google/t5-v1_1-xxl`** (`T5EncoderModel`), confirmed by the
  LTX-2 pipeline docs. ~9.5 GB fp16; **prefer an fp8 build** to fit 24 GB (encode-then-free
  still required — see §6.4). The GGUF config's `caption_channels=3840` ≠ T5-XXL's 4096, so the
  **exact encoder/projection pairing for 2.3 must be re-confirmed against the real
  `Lightricks/LTX-2(.3)` `text_encoder/` at implementation time** — it may ship its own encoder
  rather than vanilla t5-v1_1-xxl.
- **Tokenizer + scheduler:** T5 tokenizer + the LTX-2 flow-match scheduler (the GGUF config
  declares a `RectifiedFlowScheduler`, `sampler="LinearQuadratic"`), both tiny.
- **NEW, not in the §3 sizing table — LTX-2 needs an AUDIO stack to render its synced-audio
  output:** an **audio VAE decoder + a vocoder** (the GGUF config carries full `audio_vae` and
  `vocoder` blocks; PR #12915 ships `LTX2` audio-VAE-decoder + vocoder + an audio-aware video
  export util). A text-only `LTXPipeline`-style assembly would ignore the audio path entirely.
  Plus an optional **latent upsampler** (`LTX2LatentUpsamplerModel`) + a **distilled LoRA**
  (`ltx-2-19b-distilled-lora-384.safetensors`) the reference pipeline uses. The LTX companion
  pack is therefore materially larger and more complex than the spec's ~11 GB (fp8) estimate.

### 12.3 Decisions a human must make before P3 can proceed

1. **Approve the diffusers bump** (reserved decision). LTX-2.3 needs diffusers ≥ ~0.39 (LTX-2
   landed on main 2026-01-08, PR #12915). This is the **same gate as Qwen-Image/Wan**, plus
   the §9.1 VibeVoice dependency-matrix verify (`transformers==4.51.3` / `accelerate==1.6.0`
   pins) — and 0.39 is a larger jump than the 0.35 the spec scoped, so re-run that matrix
   against 0.39, not 0.35.
2. **Sequence:** there is no longer a "video ships first" shortcut. Fold LTX into the same
   bumped-venv phase as Qwen-Image (P2). The companion-pack infra (§5.4/§7) is still worth
   building arch-agnostically, but it should land **with** the bump, not ahead of it on 0.33.1.
3. **Audio-output scope for LTX-2.** Decide whether OpenHarness renders LTX-2's synced audio
   (needs the audio VAE decoder + vocoder + audio-aware mp4 mux) or ships **video-only first**
   (drop the audio stack, smaller companion pack) with audio as a follow-up.
