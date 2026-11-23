"""Text-to-video via diffusers (LTX-Video / Wan-class), sized for 24 GB.

Two model shapes:
  * a diffusers-layout directory (`DiffusionPipeline.from_pretrained` resolves
    the concrete pipeline class from its model_index.json, so LTX-Video and Wan
    2.x dirs both load through the one legacy code path),
  * a single-file **diffusion GGUF** (the quantized DiT transformer only),
    routed here by the Rust resolver with `gguf=True` + `companion_arch`. The
    only video arch supported today is `ltxv` (LTX-2): its companion VAE +
    audio VAE + Gemma3 text encoder + tokenizer + text connectors + vocoder +
    scheduler live next to the GGUF at `<models>/companions/ltxv/` and are
    assembled into an LTX2Pipeline with model-CPU-offload so the ~18 GB DiT,
    the Gemma3 encoder and the VAEs never co-reside on the GPU (fits 24 GB).
    LTX-2 is the audio+video model — the pipeline emits a synced 48 kHz
    waveform alongside the frames, which we mux into the MP4 (see `run`).

`wan` GGUFs stay rejected (`_reject_gguf_diffusion`) — Wan 2.2's dual-expert
transformer + UMT5 companion are a later phase.

Frames are encoded to MP4 (h264) with imageio-ffmpeg; the LTX-2 audio track is
muxed in with the bundled ffmpeg binary. One model stays cached between jobs;
submitting with a different model swaps it out (VRAM is freed first).

Params:
    model            local path — diffusers dir or .gguf (required)
    prompt           text prompt (required)
    negative_prompt  optional
    gguf             bool — single-file diffusion GGUF (with companion_arch)
    companion_arch   "ltxv" when gguf — selects the companion pack + pipeline
    width / height   multiples of 32 (default 768x512)
    num_frames       default 97 (~4 s at 24 fps); LTX wants 8n+1
    fps              default 24 (mapped to LTX2's frame_rate)
    steps            inference steps (default 30; maps to num_inference_steps)
    guidance         CFG scale (default 3.0; LTX2 default is 4.0)
    seed             optional int; randomized + reported in meta when absent
    offload          bool — model CPU offload for low VRAM (default false;
                     always on for the ltxv GGUF path)
"""

from __future__ import annotations

from pathlib import Path

from . import _runtime

DEFAULTS = {
    "width": 768,
    "height": 512,
    "num_frames": 97,
    "fps": 24,
    "steps": 30,
    "guidance": 3.0,
}

# LTX-2 vocoder output sample rate. The GGUF's vocoder config declares
# `output_sampling_rate=48000` via bandwidth extension (the WithBWE variant),
# so the waveform the pipeline returns is always 48 kHz regardless of the
# transformer's internal latent rate.
_LTX_AUDIO_SR = 48000

# Single-file diffusion GGUF arch this (video) pipeline assembles. `wan` stays
# rejected; the image arch `qwen_image` routes to text_to_image instead.
_LTX_GGUF_ARCH = "ltxv"

_cache: dict = {}  # {"key": (model, offload, gguf, companion_arch), "pipe": pipeline}


def _reject_gguf_diffusion(companion_arch) -> None:
    """Honest guard for a single-file diffusion-GGUF video model we can't yet
    assemble. `ltxv` (LTX-2) is handled by `_load_ltx_gguf`; `wan` still needs
    its dual-expert companion + a loader that don't exist.

    The GGUF holds only the quantized DiT transformer — no VAE, no text encoder.
    Raise a clear, actionable error instead of attempting a load that would fail
    confusingly.
    """
    arch = str(companion_arch or "").lower()
    names = {"wan": "Wan 2.2"}
    label = names.get(arch, arch or "this")
    raise RuntimeError(
        f"{label} GGUF can't generate yet: it's a single-file DiT transformer "
        "that needs its companion VAE + text encoder, which aren't supported. "
        "Support for diffusion-GGUF Wan video is coming in a later phase."
    )


def _companion_dir(model: str, companion_arch: str) -> Path:
    """Where the companion pack lives for a single-file diffusion GGUF.

    The GGUF holds only the quantized DiT transformer; everything else (the
    video + audio VAEs, the Gemma3 text encoder + tokenizer, the text-side
    connectors, the vocoder and the scheduler) ships as a separate
    diffusers-layout folder installed next to the model weights at
    `<models>/companions/<arch>/`. This mirrors the image pipeline's helper.
    """
    return Path(model).parent / "companions" / str(companion_arch)


def parse_params(params: dict) -> dict:
    model = params.get("model")
    if not model:
        raise ValueError("text_to_video requires 'model' (local path)")
    prompt = params.get("prompt")
    if not prompt:
        raise ValueError("text_to_video requires 'prompt'")
    # Single-file diffusion GGUF (routed here by the Rust resolver with `gguf`
    # + `companion_arch`). `ltxv` parses through to the real loader; `wan` still
    # raises the honest interim error.
    gguf = bool(params.get("gguf"))
    companion_arch = None
    if gguf:
        companion_arch = str(params.get("companion_arch") or "").lower()
        if companion_arch != _LTX_GGUF_ARCH:
            _reject_gguf_diffusion(params.get("companion_arch"))
    p = {
        "model": str(model),
        "prompt": str(prompt),
        "negative_prompt": params.get("negative_prompt") or None,
        "gguf": gguf,
        "companion_arch": companion_arch,
        "width": _runtime.opt(params, "width", DEFAULTS["width"], int),
        "height": _runtime.opt(params, "height", DEFAULTS["height"], int),
        "num_frames": _runtime.opt(params, "num_frames", DEFAULTS["num_frames"], int),
        "fps": _runtime.opt(params, "fps", DEFAULTS["fps"], int),
        "steps": _runtime.opt(params, "steps", DEFAULTS["steps"], int),
        "guidance": _runtime.opt(params, "guidance", DEFAULTS["guidance"], float),
        "seed": params.get("seed"),
        "offload": bool(params.get("offload", False)),
    }
    if p["width"] < 64 or p["height"] < 64 or p["width"] % 32 or p["height"] % 32:
        raise ValueError("width and height must be multiples of 32, >= 64")
    if not 1 <= p["num_frames"] <= 257:
        raise ValueError("num_frames must be 1..257")
    if p["steps"] < 1 or p["steps"] > 100:
        raise ValueError("steps must be 1..100")
    return p


def _load_ltx_vocoder(comp: Path, dtype):
    """Load the companion vocoder, picking the bandwidth-extension variant.

    LTX-2 ships two vocoder classes — `LTX2Vocoder` and `LTX2VocoderWithBWE`
    (bandwidth extension, the one that declares the 48 kHz output rate). They
    share the same subfolder layout; the config.json's `_class_name` says which
    one to build. Read it and dispatch, falling back to a plain WithBWE→Vocoder
    try so we still load if the field is missing or unexpected.
    """
    import json

    from diffusers.pipelines.ltx2 import LTX2Vocoder, LTX2VocoderWithBWE

    sub = comp / "vocoder"
    cls = LTX2VocoderWithBWE  # the GGUF's vocoder is the BWE 48 kHz variant
    cfg_path = sub / "config.json"
    if cfg_path.is_file():
        name = json.loads(cfg_path.read_text(encoding="utf-8")).get("_class_name")
        if name == "LTX2Vocoder":
            cls = LTX2Vocoder
    try:
        return cls.from_pretrained(
            str(comp), subfolder="vocoder", torch_dtype=dtype, local_files_only=True
        )
    except Exception:
        # Config said one thing but the weights are the other variant — try the
        # sibling class before giving up (cheap insurance against a mislabeled
        # companion pack).
        other = LTX2Vocoder if cls is LTX2VocoderWithBWE else LTX2VocoderWithBWE
        return other.from_pretrained(
            str(comp), subfolder="vocoder", torch_dtype=dtype, local_files_only=True
        )


def _bnb_4bit_config(dtype):
    """A transformers `BitsAndBytesConfig` for NF4 double-quant 4-bit, or None
    when bitsandbytes isn't importable.

    The LTX-2 text encoder is Gemma-3-12B — ~42 GB in bf16, which would never
    share 24 GB VRAM with the ~18 GB GGUF transformer. Quantizing it to 4-bit
    (~6.5 GB) lets model-CPU-offload page it onto the GPU for the text-encode
    stage and back off before the denoise loop. bitsandbytes is a CUDA-only
    wheel (absent from the CPU requirements, and possibly absent while a fresh
    install is still in flight), so a missing import returns None and the
    caller falls back to a bf16 load.
    """
    try:
        import bitsandbytes  # noqa: F401 — presence check for the CUDA wheel
    except Exception:
        return None
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


def _ltx_text_encoder(comp: Path, dtype):
    """Load the LTX-2 companion text encoder (Gemma-3-12B).

    On CUDA we 4-bit-quantize it (see `_bnb_4bit_config`) so the encoder
    (~6.5 GB) and the GGUF DiT (~18 GB) never co-reside on a 24 GB card:
    `enable_model_cpu_offload` pages the 4-bit module onto the GPU for text
    encoding and off again afterwards (diffusers only refuses to *offload* an
    8-bit bnb module — a 4-bit one moves device-to-device fine). If bitsandbytes
    is unavailable, the 4-bit load fails, or we're on CPU, fall back to a plain
    bf16 load — correct, just far heavier (needs >24 GB free with the DiT, so it
    only works under offload when the DiT isn't co-resident, or on CPU).
    """
    import sys

    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    # Gemma-3 is a genuinely multimodal config: its `text_config`
    # (a `Gemma3TextConfig`, not a raw dict) is load-bearing — transformers
    # 4.51.3 reads it in `_get_torch_dtype` and `__init__`. Load it whole and
    # pass it through unmodified; stripping `text_config` here is what crashed
    # the encoder with "'Gemma3Config' object has no attribute 'text_config'".
    te_cfg = Gemma3Config.from_pretrained(
        str(comp), subfolder="text_encoder", local_files_only=True
    )
    quant_config = _bnb_4bit_config(dtype) if _runtime.cuda_available() else None
    if quant_config is not None:
        try:
            return Gemma3ForConditionalGeneration.from_pretrained(
                str(comp),
                subfolder="text_encoder",
                config=te_cfg,
                quantization_config=quant_config,
                torch_dtype=dtype,
                local_files_only=True,
            )
        except Exception as exc:  # pragma: no cover — needs a GPU + bnb to hit
            print(
                f"[text_to_video] 4-bit text-encoder load failed ({exc!r}); "
                "falling back to bf16",
                file=sys.stderr,
                flush=True,
            )
    return Gemma3ForConditionalGeneration.from_pretrained(
        str(comp),
        subfolder="text_encoder",
        config=te_cfg,
        torch_dtype=dtype,
        local_files_only=True,
    )


def _ltx_prompt_adaln_rename(checkpoint):
    """Rename the GGUF's ``prompt_adaln_single.*`` / ``audio_prompt_adaln_single.*``
    tensors to the model's ``prompt_adaln.*`` / ``audio_prompt_adaln.*`` param
    names. The ``audio_`` variant rides the same substring, so one rule covers
    both. Mutates ``checkpoint`` in place and returns it."""
    for k in list(checkpoint.keys()):
        if "prompt_adaln_single." in k:
            checkpoint[k.replace("prompt_adaln_single.", "prompt_adaln.")] = (
                checkpoint.pop(k)
            )
    return checkpoint


def _patch_ltx2_single_file_mapping():
    """Work around a diffusers 0.38 bug: ``convert_ltx2_transformer_to_diffusers``
    renames ``adaln_single``->``time_embed`` but omits the matching
    ``prompt_adaln_single``->``prompt_adaln`` rename. Without it the GGUF's 12
    prompt-AdaLN tensors never map onto the model's ``prompt_adaln.*`` /
    ``audio_prompt_adaln.*`` params; those params stay on the ``meta`` device and
    the final ``dispatch_model`` / ``.to()`` raises "Cannot copy out of meta
    tensor; no data!". Wrap the converter to apply the missing rename first, then
    patch the single-file registry entries that captured the original function at
    import time (patching the module attribute alone is not enough). Idempotent."""
    from diffusers.loaders import single_file_model as sfm
    from diffusers.loaders import single_file_utils as sfu

    orig = sfu.convert_ltx2_transformer_to_diffusers
    if getattr(orig, "_ltx_prompt_adaln_patched", False):
        return

    def patched(checkpoint, **kwargs):
        return orig(_ltx_prompt_adaln_rename(checkpoint), **kwargs)

    patched._ltx_prompt_adaln_patched = True
    sfu.convert_ltx2_transformer_to_diffusers = patched
    for spec in sfm.SINGLE_FILE_LOADABLE_CLASSES.values():
        if spec.get("checkpoint_mapping_fn") is orig:
            spec["checkpoint_mapping_fn"] = patched


def _load_ltx_gguf(model: str):
    """Assemble an LTX2Pipeline from a single-file GGUF transformer + the
    companion pack. GGUF weights stay quantized (uint8) in VRAM and dequantize
    on-the-fly per forward; model-CPU-offload keeps the ~18 GB transformer, the
    Gemma3 encoder and the two VAEs from co-residing on the GPU."""
    import torch
    from diffusers import (
        AutoencoderKLLTX2Audio,
        AutoencoderKLLTX2Video,
        FlowMatchEulerDiscreteScheduler,
        GGUFQuantizationConfig,
        LTX2VideoTransformer3DModel,
    )
    from diffusers.pipelines.ltx2 import LTX2Pipeline, LTX2TextConnectors
    from transformers import AutoTokenizer

    gguf_path = Path(model)
    if not gguf_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model}")
    comp = _companion_dir(model, _LTX_GGUF_ARCH)
    if not comp.is_dir():
        raise FileNotFoundError(
            f"LTX-2 companion pack not found at {comp}. The single-file GGUF holds "
            "only the quantized transformer; the video VAE + audio VAE + Gemma3 "
            "text encoder + tokenizer + text connectors + vocoder + scheduler must "
            "be installed alongside it (subfolders vae/ audio_vae/ text_encoder/ "
            "tokenizer/ connectors/ vocoder/ scheduler/) before generating."
        )

    dtype = torch.bfloat16
    _patch_ltx2_single_file_mapping()
    transformer = LTX2VideoTransformer3DModel.from_single_file(
        str(gguf_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        config=str(comp),
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    # The Gemma-3-12B encoder is ~42 GB in bf16; on CUDA `_ltx_text_encoder`
    # 4-bit-quantizes it (~6.5 GB) so model-CPU-offload can page it on/off the
    # 24 GB card without ever co-residing with the ~18 GB DiT (it also strips a
    # nested `text_config` the pinned transformers 4.51.3 can't parse). Falls
    # back to bf16 when bitsandbytes/CUDA is unavailable.
    text_encoder = _ltx_text_encoder(comp, dtype)
    # AutoTokenizer resolves to the Gemma tokenizer (fast/slow) the companion
    # ships — we don't hard-code GemmaTokenizerFast so a slow-only pack still
    # loads.
    tokenizer = AutoTokenizer.from_pretrained(
        str(comp), subfolder="tokenizer", local_files_only=True
    )
    connectors = LTX2TextConnectors.from_pretrained(
        str(comp), subfolder="connectors", torch_dtype=dtype, local_files_only=True
    )
    vae = AutoencoderKLLTX2Video.from_pretrained(
        str(comp), subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    audio_vae = AutoencoderKLLTX2Audio.from_pretrained(
        str(comp), subfolder="audio_vae", torch_dtype=dtype, local_files_only=True
    )
    vocoder = _load_ltx_vocoder(comp, dtype)
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(comp), subfolder="scheduler", local_files_only=True
    )
    pipe = LTX2Pipeline(
        scheduler=scheduler,
        vae=vae,
        audio_vae=audio_vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        connectors=connectors,
        transformer=transformer,
        vocoder=vocoder,
        processor=None,  # optional; the companion doesn't ship an image processor
    )
    if _runtime.cuda_available():
        # Required, not optional: the DiT (~18 GB) + Gemma3 encoder + the two
        # VAEs + vocoder exceed 24 GB if co-resident. The pipeline's offload
        # sequence (text_encoder->connectors->transformer->vae->audio_vae->
        # vocoder) runs them one stage at a time.
        pipe.enable_model_cpu_offload()
    return pipe


def _load(model: str, offload: bool, gguf: bool = False, companion_arch=None):
    import torch
    from diffusers import DiffusionPipeline

    key = (model, offload, gguf, companion_arch)
    if _cache.get("key") == key:
        return _cache["pipe"]
    unload()

    if gguf:
        pipe = _load_ltx_gguf(model)
        _cache["key"] = key
        _cache["pipe"] = pipe
        return pipe

    path = Path(model)
    if not path.is_dir():
        raise FileNotFoundError(f"model dir does not exist: {model}")
    dtype = torch.bfloat16 if _runtime.cuda_available() else torch.float32
    kwargs: dict = dict(torch_dtype=dtype, local_files_only=True)
    if _runtime.has_fp16_variant(path):
        kwargs["variant"] = "fp16"
    pipe = DiffusionPipeline.from_pretrained(str(path), **kwargs)
    if offload and _runtime.cuda_available():
        pipe.enable_model_cpu_offload()
    elif _runtime.cuda_available():
        pipe.to("cuda")

    _cache["key"] = key
    _cache["pipe"] = pipe
    return pipe


def preload(params: dict, ctx) -> dict:
    """Load the model into memory without generating (W1 explicit preload)."""
    model = params.get("model")
    if not model:
        raise ValueError("preload requires 'model' (local path)")
    gguf = bool(params.get("gguf"))
    if gguf:
        companion_arch = str(params.get("companion_arch") or "").lower()
        if companion_arch != _LTX_GGUF_ARCH:
            _reject_gguf_diffusion(params.get("companion_arch"))
        ctx.progress(pct=5.0, stage="loading model")
        # ltxv companions always offload (the working set exceeds 24 GB).
        _load(str(model), True, gguf=True, companion_arch=companion_arch)
        ctx.progress(pct=100.0, stage="loaded")
        return {"vram_mb": _runtime.vram_used_mb(), "model": str(model)}
    ctx.progress(pct=5.0, stage="loading model")
    _load(str(model), bool(params.get("offload", False)))
    ctx.progress(pct=100.0, stage="loaded")
    return {"vram_mb": _runtime.vram_used_mb(), "model": str(model)}


def _write_audio_wav(audio, wav_path: Path) -> bool:
    """Write the LTX-2 waveform tensor to a 48 kHz WAV. Returns True on success,
    False when there's no usable audio (so the caller falls back to silent
    video instead of crashing).

    The pipeline returns a torch.Tensor; soundfile wants a CPU float32 numpy
    array shaped (samples,) mono or (samples, channels). LTX-2 hands back a
    leading batch dim and channel-first layout, so we squeeze the batch and move
    channels last when present.
    """
    import numpy as np

    if audio is None:
        return False
    # torch.Tensor → CPU float32 numpy. Guard the import so a numpy-only audio
    # array (unlikely, but cheap to handle) still works.
    try:
        import torch

        if isinstance(audio, torch.Tensor):
            audio = audio.detach().to(torch.float32).cpu().numpy()
    except Exception:
        pass
    data = np.asarray(audio, dtype=np.float32)
    data = np.squeeze(data)  # drop the batch dim (and any other size-1 axes)
    if data.size == 0:
        return False
    if data.ndim == 1:
        pass  # mono (samples,)
    elif data.ndim == 2:
        # soundfile wants (samples, channels); LTX-2 is channel-first, so
        # transpose when the first axis looks like the (small) channel count.
        if data.shape[0] < data.shape[1]:
            data = data.T
    else:
        # Anything higher-rank is unexpected — flatten to mono rather than crash.
        data = data.reshape(-1)

    import soundfile as sf

    sf.write(str(wav_path), data, _LTX_AUDIO_SR)
    return True


def _mux_audio_video(video_path: Path, wav_path: Path, out_path: Path) -> None:
    """Mux a silent MP4 + a WAV into a single MP4 with an AAC audio track.

    diffusers/imageio have no audio-aware video export in 0.38.0, so we shell
    out to the ffmpeg binary bundled with imageio-ffmpeg: copy the already-
    encoded h264 video stream verbatim (`-c:v copy`), encode the audio to AAC,
    and `-shortest` so the two streams end together (the derived audio length
    and frame count won't match to the sample).
    """
    import subprocess

    import imageio_ffmpeg

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(wav_path),
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )


def run(params: dict, ctx) -> tuple[list[str], dict]:
    import numpy as np

    p = parse_params(params)

    ctx.progress(pct=0.0, stage="loading model")
    pipe = _load(p["model"], p["offload"], p["gguf"], p["companion_arch"])
    vram_mb = _runtime.vram_used_mb()
    ctx.check_cancelled()

    seed = _runtime.resolve_seed(p["seed"])
    total = p["steps"]

    def on_step(pipeline, step, timestep, callback_kwargs):
        ctx.check_cancelled()
        ctx.progress(
            pct=(step + 1) / total * 90.0,
            stage="denoising",
            step=step + 1,
            total_steps=total,
        )
        return callback_kwargs

    if p["gguf"]:
        # LTX-2 names the frame rate `frame_rate` (not `fps`) and derives the
        # audio guidance from the video guidance, so we leave every `audio_*`
        # kwarg at its default and just pass the user's fps/steps/guidance.
        result = pipe(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            width=p["width"],
            height=p["height"],
            num_frames=p["num_frames"],
            frame_rate=p["fps"],
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance"],
            generator=_runtime.make_generator(seed),
            callback_on_step_end=on_step,
        )
    else:
        result = pipe(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            width=p["width"],
            height=p["height"],
            num_frames=p["num_frames"],
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance"],
            generator=_runtime.make_generator(seed),
            callback_on_step_end=on_step,
        )

    ctx.progress(pct=92.0, stage="encoding video")
    frames = result.frames[0]  # list of PIL Images (or ndarray)
    frames_np = [np.asarray(f) for f in frames]

    import imageio

    out = ctx.output_dir / f"video-{ctx.job_id}.mp4"
    has_audio = False
    audio = getattr(result, "audio", None) if p["gguf"] else None
    if audio is not None:
        # LTX-2: write the silent video + the waveform to temp files, then mux
        # them into the final MP4. Fall back to the silent video if the audio
        # turns out to be empty/unusable so a job never fails just for muxing.
        silent = ctx.output_dir / f"video-{ctx.job_id}.silent.mp4"
        wav = ctx.output_dir / f"video-{ctx.job_id}.wav"
        imageio.mimwrite(
            str(silent),
            frames_np,
            fps=p["fps"],
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )
        try:
            if _write_audio_wav(audio, wav):
                ctx.progress(pct=96.0, stage="muxing audio")
                _mux_audio_video(silent, wav, out)
                has_audio = True
            else:
                silent.replace(out)  # no usable audio → ship the silent video
        finally:
            for tmp in (silent, wav):
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
    else:
        imageio.mimwrite(
            str(out),
            frames_np,
            fps=p["fps"],
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )

    meta = {
        "seed": seed,
        "width": p["width"],
        "height": p["height"],
        "num_frames": p["num_frames"],
        "fps": p["fps"],
        "steps": p["steps"],
        "guidance": p["guidance"],
        "has_audio": has_audio,
        "audio_sample_rate": _LTX_AUDIO_SR if has_audio else None,
        "vram_mb": vram_mb,
    }
    return [str(out)], meta


def unload() -> None:
    _cache.pop("pipe", None)
    _cache.pop("key", None)
    import gc

    gc.collect()
    _runtime.free_cuda_cache()
