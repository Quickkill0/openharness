"""text-to-image via diffusers.

Accepts three model shapes:
  * a diffusers-layout model directory (`from_pretrained`),
  * a single-file checkpoint (`from_single_file`, arch via the `arch` param —
    sdxl / sd / flux),
  * a single-file **diffusion GGUF** (the quantized DiT transformer only),
    routed here by the Rust resolver with `gguf=True` + `companion_arch`. The
    only image arch supported today is `qwen_image`: its companion VAE +
    Qwen2.5-VL text encoder + tokenizer + scheduler live next to the GGUF at
    `<models>/companions/qwen_image/` and are assembled into a QwenImagePipeline
    with model-CPU-offload so the ~16.8 GB Q6_K transformer and the ~16.6 GB
    text encoder never co-reside on the GPU (fits 24 GB).

One model stays cached between jobs; submitting with a different model swaps it
out (VRAM is freed first).

Params:
    model            local path — diffusers dir, .safetensors, or .gguf (required)
    prompt           text prompt (required)
    negative_prompt  optional
    arch             "sdxl" | "sd" | "flux" — single-file loader pick (default sdxl)
    gguf             bool — single-file diffusion GGUF (with companion_arch)
    companion_arch   "qwen_image" when gguf — selects the companion pack + pipeline
    width / height   multiples of 8 (default 1024x1024)
    steps            inference steps (default 30)
    guidance         CFG scale (default 5.0; maps to true_cfg_scale for qwen_image)
    num_images       1..8 (default 1)
    seed             optional int; randomized + reported in meta when absent
    offload          bool — enable model CPU offload for low-VRAM (default false;
                     always on for the qwen_image GGUF path)
"""

from __future__ import annotations

from pathlib import Path

from . import _runtime

DEFAULTS = {"width": 1024, "height": 1024, "steps": 30, "guidance": 5.0}

# Single-file diffusion GGUF arch supported by this (image) pipeline. Video
# archs (ltxv / wan) route to text_to_video instead.
_QWEN_GGUF_ARCH = "qwen_image"

_cache: dict = {}  # {"key": (model, arch, offload, gguf, companion_arch), "pipe": pipeline}


def _reject_gguf_diffusion(companion_arch) -> None:
    """Honest guard for an unsupported single-file diffusion GGUF reaching the
    image pipeline. `qwen_image` is handled by `_load_qwen_gguf`; anything else
    routed here lacks a loader + companion pack."""
    raise RuntimeError(
        f"This single-file diffusion GGUF (companion_arch={companion_arch!r}) "
        "isn't supported by the image pipeline yet — it needs a companion pack "
        "and a loader that don't exist."
    )


def _companion_dir(model: str, companion_arch: str) -> Path:
    """Where the companion pack lives for a single-file diffusion GGUF.

    The GGUF holds only the quantized DiT transformer; the VAE / text encoder /
    tokenizer / scheduler ship as a separate diffusers-layout folder installed
    next to the model weights at `<models>/companions/<arch>/`.
    """
    return Path(model).parent / "companions" / str(companion_arch)


def _strip_redundant_text_config(cfg):
    """Drop a nested `text_config` block from a Qwen2.5-VL config.

    The Qwen-Image text encoder ships a config.json re-saved by transformers
    4.53, which carries the newer Qwen2.5-VL `text_config` sub-block. The pinned
    transformers 4.51.3 doesn't model that field, so it survives as a raw dict
    and later crashes `GenerationConfig.from_model_config`
    ("'dict' object has no attribute 'to_dict'") during model __init__. The
    weights are in the 4.51 layout (`model.layers.*` / `lm_head`), so the
    top-level config already describes the model correctly — removing the
    redundant block lets 4.51.3 build the model unchanged. Returns `cfg`.
    """
    if hasattr(cfg, "text_config"):
        delattr(cfg, "text_config")
    return cfg


def parse_params(params: dict) -> dict:
    model = params.get("model")
    if not model:
        raise ValueError("text_to_image requires 'model' (local path)")
    prompt = params.get("prompt")
    if not prompt:
        raise ValueError("text_to_image requires 'prompt'")
    # ComfyUI-style single-file diffusion GGUF (the DiT transformer only — no
    # VAE, no text encoder). Routed here by the Rust resolver with `gguf` +
    # `companion_arch`; the companion pack supplies the missing components.
    gguf = bool(params.get("gguf"))
    companion_arch = None
    arch = "sdxl"
    if gguf:
        companion_arch = str(params.get("companion_arch") or "").lower()
        if companion_arch != _QWEN_GGUF_ARCH:
            _reject_gguf_diffusion(params.get("companion_arch"))
        arch = companion_arch  # cache-key slot; unused by the gguf loader
    else:
        arch = str(params.get("arch") or "sdxl").lower()
        if arch not in ("sdxl", "sd", "flux"):
            raise ValueError(f"unknown arch: {arch} (expected sdxl | sd | flux)")
    p = {
        "model": str(model),
        "prompt": str(prompt),
        "negative_prompt": params.get("negative_prompt") or None,
        "arch": arch,
        "gguf": gguf,
        "companion_arch": companion_arch,
        "width": _runtime.opt(params, "width", DEFAULTS["width"], int),
        "height": _runtime.opt(params, "height", DEFAULTS["height"], int),
        "steps": _runtime.opt(params, "steps", DEFAULTS["steps"], int),
        "guidance": _runtime.opt(params, "guidance", DEFAULTS["guidance"], float),
        "num_images": max(1, min(_runtime.opt(params, "num_images", 1, int), 8)),
        "seed": params.get("seed"),
        "offload": bool(params.get("offload", False)),
    }
    if p["width"] < 64 or p["height"] < 64 or p["width"] % 8 or p["height"] % 8:
        raise ValueError("width and height must be multiples of 8, >= 64")
    if p["steps"] < 1 or p["steps"] > 150:
        raise ValueError("steps must be 1..150")
    return p


def _load_qwen_gguf(model: str, companion_arch: str):
    """Assemble a QwenImagePipeline from a single-file GGUF transformer + the
    companion pack. GGUF weights stay quantized (uint8) in VRAM and dequantize
    on-the-fly per forward; model-CPU-offload keeps the transformer and the
    Qwen2.5-VL text encoder from co-residing on the GPU."""
    import torch
    from diffusers import (
        AutoencoderKLQwenImage,
        FlowMatchEulerDiscreteScheduler,
        GGUFQuantizationConfig,
        QwenImagePipeline,
        QwenImageTransformer2DModel,
    )
    from transformers import (
        Qwen2_5_VLConfig,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2Tokenizer,
    )

    gguf_path = Path(model)
    if not gguf_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model}")
    comp = _companion_dir(model, companion_arch)
    if not comp.is_dir():
        raise FileNotFoundError(
            f"Qwen-Image companion pack not found at {comp}. The single-file GGUF "
            "holds only the quantized transformer; the VAE + Qwen2.5-VL text "
            "encoder + tokenizer + scheduler must be installed alongside it before "
            "generating."
        )

    dtype = torch.bfloat16
    transformer = QwenImageTransformer2DModel.from_single_file(
        str(gguf_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        config=str(comp),
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )
    # Strip the redundant 4.53 `text_config` block so 4.51.3 can build the
    # model (see _strip_redundant_text_config for the full rationale).
    te_cfg = _strip_redundant_text_config(
        Qwen2_5_VLConfig.from_pretrained(
            str(comp), subfolder="text_encoder", local_files_only=True
        )
    )
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(comp),
        subfolder="text_encoder",
        config=te_cfg,
        torch_dtype=dtype,
        local_files_only=True,
    )
    tokenizer = Qwen2Tokenizer.from_pretrained(
        str(comp), subfolder="tokenizer", local_files_only=True
    )
    vae = AutoencoderKLQwenImage.from_pretrained(
        str(comp), subfolder="vae", torch_dtype=dtype, local_files_only=True
    )
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        str(comp), subfolder="scheduler", local_files_only=True
    )
    pipe = QwenImagePipeline(
        scheduler=scheduler,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
    )
    if _runtime.cuda_available():
        # Required, not optional: encoder (~16.6 GB) + transformer (~16.8 GB)
        # exceed 24 GB if co-resident. Offload runs them one-at-a-time.
        pipe.enable_model_cpu_offload()
    return pipe


def _load(model: str, arch: str, offload: bool, gguf: bool = False, companion_arch=None):
    import torch

    key = (model, arch, offload, gguf, companion_arch)
    if _cache.get("key") == key:
        return _cache["pipe"]
    unload()

    if gguf:
        pipe = _load_qwen_gguf(model, companion_arch)
        _cache["key"] = key
        _cache["pipe"] = pipe
        return pipe

    path = Path(model)
    if not path.exists():
        raise FileNotFoundError(f"model path does not exist: {model}")
    dtype = torch.float16 if _runtime.cuda_available() else torch.float32

    if path.is_dir():
        from diffusers import AutoPipelineForText2Image

        kwargs: dict = dict(torch_dtype=dtype, local_files_only=True)
        if _runtime.has_fp16_variant(path):
            kwargs["variant"] = "fp16"
        pipe = AutoPipelineForText2Image.from_pretrained(str(path), **kwargs)
    else:
        if arch == "sdxl":
            from diffusers import StableDiffusionXLPipeline as Cls
        elif arch == "sd":
            from diffusers import StableDiffusionPipeline as Cls
        else:  # flux
            from diffusers import FluxPipeline as Cls
        pipe = Cls.from_single_file(str(path), torch_dtype=dtype)

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
        if companion_arch != _QWEN_GGUF_ARCH:
            _reject_gguf_diffusion(params.get("companion_arch"))
        ctx.progress(pct=5.0, stage="loading model")
        _load(str(model), companion_arch, True, gguf=True, companion_arch=companion_arch)
        ctx.progress(pct=100.0, stage="loaded")
        return {"vram_mb": _runtime.vram_used_mb(), "model": str(model)}
    arch = str(params.get("arch") or "sdxl").lower()
    if arch not in ("sdxl", "sd", "flux"):
        raise ValueError(f"unknown arch: {arch} (expected sdxl | sd | flux)")
    ctx.progress(pct=5.0, stage="loading model")
    _load(str(model), arch, bool(params.get("offload", False)))
    ctx.progress(pct=100.0, stage="loaded")
    return {"vram_mb": _runtime.vram_used_mb(), "model": str(model)}


def run(params: dict, ctx) -> tuple[list[str], dict]:
    p = parse_params(params)

    ctx.progress(pct=0.0, stage="loading model")
    pipe = _load(p["model"], p["arch"], p["offload"], p["gguf"], p["companion_arch"])
    vram_mb = _runtime.vram_used_mb()
    ctx.check_cancelled()

    seed = _runtime.resolve_seed(p["seed"])
    total = p["steps"]

    def on_step(pipeline, step, timestep, callback_kwargs):
        ctx.check_cancelled()
        ctx.progress(
            pct=(step + 1) / total * 95.0,
            stage="denoising",
            step=step + 1,
            total_steps=total,
        )
        return callback_kwargs

    if p["gguf"]:
        # Qwen-Image drives CFG via true_cfg_scale (+ a negative prompt);
        # guidance_scale is inert on the current checkpoints.
        result = pipe(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"] or " ",
            width=p["width"],
            height=p["height"],
            num_inference_steps=p["steps"],
            true_cfg_scale=p["guidance"],
            num_images_per_prompt=p["num_images"],
            generator=_runtime.make_generator(seed),
            callback_on_step_end=on_step,
        )
    else:
        result = pipe(
            prompt=p["prompt"],
            negative_prompt=p["negative_prompt"],
            width=p["width"],
            height=p["height"],
            num_inference_steps=p["steps"],
            guidance_scale=p["guidance"],
            num_images_per_prompt=p["num_images"],
            generator=_runtime.make_generator(seed),
            callback_on_step_end=on_step,
        )

    ctx.progress(pct=97.0, stage="saving")
    outputs = []
    for i, image in enumerate(result.images):
        out = ctx.output_dir / f"image-{ctx.job_id}-{i:02d}.png"
        image.save(out)
        outputs.append(str(out))

    meta = {
        "seed": seed,
        "width": p["width"],
        "height": p["height"],
        "steps": p["steps"],
        "guidance": p["guidance"],
        "vram_mb": vram_mb,
    }
    return outputs, meta


def unload() -> None:
    _cache.pop("pipe", None)
    _cache.pop("key", None)
    import gc

    gc.collect()
    _runtime.free_cuda_cache()
