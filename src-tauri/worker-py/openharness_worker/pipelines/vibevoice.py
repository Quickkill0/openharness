"""Voice cloning via VibeVoice (community fork, MIT).

Microsoft pulled the original repository; the maintained fork lives at
github.com/vibevoice-community/VibeVoice and re-hosts the weights at
hf.co/vibevoice/VibeVoice-1.5B (safetensors dir, ~5.4 GB, needs ~7 GB VRAM).
The package is pinned by commit in requirements.{cuda,cpu}.txt.

The load is fully offline: VibeVoice's processor otherwise reaches out to
Hugging Face for the Qwen2.5-1.5B tokenizer (the checkpoint records it as the
bare repo id). We fetch that tokenizer at *install* time into
`<model_dir>/_qwen_tokenizer/` and point the processor there, so nothing
touches the network at inference.

Params:
    model            local dir of a VibeVoice checkpoint (required)
    text             what to say (required). Lines already formatted as
                     "Speaker 1: …" / "Speaker 2: …" are passed through as a
                     multi-speaker script; plain text is wrapped as a
                     single-speaker script.
    voice            a built-in preset voice id (resolved to its bundled clip
                     under `<model>/_builtin_voices/<id>.wav`). Used when no
                     `reference_audio` is supplied. One of `voice` /
                     `reference_audio` is required.
    reference_audio  voice sample(s) to clone — a wav/mp3 path, or a list of
                     paths (one per speaker) for multi-speaker scripts. Wins
                     over `voice` when both are given.
    num_speakers     optional; must match len(reference_audio) when given
    cfg_scale        classifier-free guidance, 1.0..3.0 (default 1.3)

Output: `speech-{job_id}.wav` (24 kHz mono). Meta reports duration, sample
rate, cfg_scale, and NVML-actual VRAM.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from . import _runtime

SAMPLE_RATE = 24_000

_SPEAKER_LINE = re.compile(r"^speaker\s+\d+\s*:", re.IGNORECASE)

# Subdir, created at install time, holding the Qwen2.5-1.5B *tokenizer* files
# (tokenizer.json/config + vocab/merges). VibeVoice's processor needs this
# tokenizer but the checkpoint records it as the bare repo id
# "Qwen/Qwen2.5-1.5B"; loading that reaches out to Hugging Face. We point the
# processor at this local dir instead so the load is 100% offline.
QWEN_TOKENIZER_SUBDIR = "_qwen_tokenizer"

# Subdir, created at install time, holding the curated built-in *preset* voice
# clips (`<voice-id>.wav`). VibeVoice has no intrinsic voices — it always clones
# from a reference clip — so a preset is just a bundled reference sample fetched
# (Rust side) when the user installs a VibeVoice repo. Must match
# BUILTIN_VOICES_SUBDIR in src-tauri/src/models/vibevoice_voices.rs. The
# pipeline only ever reads these local files; it never fetches at synth time.
BUILTIN_VOICES_SUBDIR = "_builtin_voices"

_cache: dict = {}  # {"key": model_dir, "model": ..., "processor": ...}


def parse_params(params: dict) -> dict:
    model = params.get("model")
    if not model or not str(model).strip():
        raise ValueError("vibevoice requires 'model' (local checkpoint dir)")
    text = params.get("text")
    if not text or not str(text).strip():
        raise ValueError("vibevoice requires 'text'")

    refs = params.get("reference_audio")
    if isinstance(refs, str):
        refs = [refs]
    has_refs = isinstance(refs, list) and refs and all(str(r).strip() for r in refs)

    if has_refs:
        # Custom cloning (incl. multi-speaker) — explicit reference samples win.
        refs = [str(r) for r in refs]
    else:
        # No custom reference: resolve a built-in preset voice to its bundled
        # clip. VibeVoice has no intrinsic voices, so a preset *is* a reference
        # sample shipped alongside the checkpoint. Never a network fetch.
        voice = params.get("voice")
        if not voice or not str(voice).strip():
            raise ValueError(
                "vibevoice requires either 'voice' (a built-in preset id) or "
                "'reference_audio' (path to a voice sample, or a list of paths "
                "for multi-speaker scripts)"
            )
        refs = [_resolve_builtin_voice(str(model), str(voice).strip())]

    num_speakers = params.get("num_speakers")
    if num_speakers is not None and int(num_speakers) != len(refs):
        raise ValueError(
            f"num_speakers={num_speakers} but {len(refs)} reference_audio "
            "sample(s) were given — one sample per speaker"
        )

    cfg_scale = _runtime.opt(params, "cfg_scale", 1.3, float)
    if not 1.0 <= cfg_scale <= 3.0:
        raise ValueError("cfg_scale must be 1.0..3.0")

    return {
        "model": str(model),
        "text": str(text),
        "reference_audio": refs,
        "cfg_scale": cfg_scale,
    }


def _resolve_builtin_voice(model_dir: str, voice_id: str) -> str:
    """Path to a built-in preset voice's bundled clip, or an honest error.

    Presets live at `<model_dir>/_builtin_voices/<voice-id>.wav`, fetched at
    install time (Rust side). Resolution is a pure local-file lookup — never a
    network fetch — and raises a clear, actionable error when the clip is
    absent (e.g. an older install that pre-dates the preset feature, before its
    startup heal has run) so the gap surfaces honestly rather than as a vague
    load failure.
    """
    clip = Path(model_dir) / BUILTIN_VOICES_SUBDIR / f"{voice_id}.wav"
    if not clip.exists():
        raise FileNotFoundError(
            f"Built-in voice {voice_id!r} isn't installed for this VibeVoice "
            f"model — expected its sample at {clip}. Reinstall the model from "
            "the Store (or restart the app) so its preset voices are fetched, "
            "or supply your own reference audio to clone a voice."
        )
    return str(clip)


def build_script(text: str) -> str:
    """VibeVoice consumes a podcast-style script of "Speaker N: …" lines.

    Text that already carries speaker labels passes through verbatim (blank
    lines dropped); anything else becomes a single-speaker script.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if any(_SPEAKER_LINE.match(ln) for ln in lines):
        return "\n".join(lines)
    return f"Speaker 1: {' '.join(lines)}"


def _resolve_local_tokenizer(model_dir: str) -> str:
    """Local Qwen2.5-1.5B tokenizer dir bundled with the install.

    The path check runs before the heavy import so it can never be skipped.
    Raises if either the checkpoint dir or its bundled tokenizer is absent —
    never a network fetch. The tokenizer is fetched at install time (Rust side,
    after the VibeVoice repo download finishes) into `<model_dir>/<subdir>/`.
    """
    root = Path(model_dir)
    if not root.exists():
        raise ValueError(
            f"VibeVoice model not fully installed — directory not found: "
            f"{model_dir!r}. Reinstall from the Store."
        )

    tok_dir = root / QWEN_TOKENIZER_SUBDIR
    # tokenizer.json (fast) is enough for VibeVoiceTextTokenizerFast; require it
    # so a half-finished install is caught here rather than as an HF fetch.
    if not (tok_dir / "tokenizer.json").exists() and not (
        tok_dir / "tokenizer_config.json"
    ).exists():
        raise ValueError(
            "VibeVoice needs the Qwen2.5-1.5B tokenizer installed locally at "
            f"{tok_dir} — reinstall the model from the Store so its tokenizer "
            "is fetched too (no auto-download at inference time)."
        )
    return str(tok_dir)


def _load(model_dir: str):
    # Validate the install (and its bundled tokenizer) BEFORE importing torch /
    # vibevoice, mirroring text_to_speech / transcribe offline hardening.
    tokenizer_dir = _resolve_local_tokenizer(model_dir)

    import torch
    from vibevoice.modular.modeling_vibevoice_inference import (
        VibeVoiceForConditionalGenerationInference,
    )
    from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

    if _cache.get("key") == model_dir:
        return _cache["model"], _cache["processor"]
    unload()

    # Belt-and-braces: force HF offline for the whole load so neither the
    # processor (which records the language-model tokenizer as the bare repo id
    # "Qwen/Qwen2.5-1.5B") nor any from_pretrained cache lookup can reach the
    # network. `language_model_pretrained_name` threads through **kwargs to the
    # processor and points it at the bundled local tokenizer dir.
    prev = {
        k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        processor = VibeVoiceProcessor.from_pretrained(
            model_dir,
            language_model_pretrained_name=tokenizer_dir,
        )
        if _runtime.cuda_available():
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",  # flash-attn isn't in the venv
                device_map="cuda",
            )
        else:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                model_dir,
                torch_dtype=torch.float32,
                attn_implementation="sdpa",
                device_map="cpu",
            )
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    _cache["key"] = model_dir
    _cache["model"] = model
    _cache["processor"] = processor
    return model, processor


def preload(params: dict, ctx) -> dict:
    """Load the checkpoint without synthesizing (W1 explicit preload)."""
    model = params.get("model")
    if not model or not str(model).strip():
        raise ValueError("preload requires 'model' (local checkpoint dir)")
    ctx.progress(pct=5.0, stage="loading model")
    _load(str(model))
    ctx.progress(pct=100.0, stage="loaded")
    return {"vram_mb": _runtime.vram_used_mb(), "model": str(model)}


def run(params: dict, ctx) -> tuple[list[str], dict]:
    p = parse_params(params)

    for ref in p["reference_audio"]:
        if not Path(ref).exists():
            raise FileNotFoundError(f"reference audio does not exist: {ref}")

    ctx.progress(pct=0.0, stage="loading model")
    model, processor = _load(p["model"])
    vram_mb = _runtime.vram_used_mb()
    ctx.check_cancelled()

    ctx.progress(pct=10.0, stage="preparing voices")
    script = build_script(p["text"])
    inputs = processor(
        text=[script],
        voice_samples=[p["reference_audio"]],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    device = _runtime.device()
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    ctx.check_cancelled()

    # One blocking call — VibeVoice's generate loop has no per-step callback
    # we can hook without its streamer machinery, so progress jumps from
    # "synthesizing" straight to "saving".
    ctx.progress(pct=15.0, stage="synthesizing")
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=p["cfg_scale"],
        tokenizer=processor.tokenizer,
        generation_config={"do_sample": False},
        verbose=False,
    )
    ctx.check_cancelled()

    speech = outputs.speech_outputs[0] if outputs.speech_outputs else None
    if speech is None:
        raise RuntimeError("VibeVoice produced no audio for the given script")

    ctx.progress(pct=95.0, stage="saving")
    out = ctx.output_dir / f"speech-{ctx.job_id}.wav"
    processor.save_audio(speech, output_path=str(out))

    duration = float(speech.reshape(-1).shape[0]) / SAMPLE_RATE
    meta = {
        "engine": "vibevoice",
        "sample_rate": SAMPLE_RATE,
        "duration": round(duration, 2),
        "cfg_scale": p["cfg_scale"],
        "num_speakers": len(p["reference_audio"]),
        "vram_mb": vram_mb,
    }
    return [str(out)], meta


def unload() -> None:
    model = _cache.pop("model", None)
    _cache.pop("processor", None)
    _cache.pop("key", None)
    # VibeVoice is the one pipeline loaded with device_map="cuda", i.e. through
    # accelerate's dispatch. Accelerate attaches forward hooks that keep the
    # GPU weights referenced past a plain `del`, so popping the cache and
    # calling empty_cache() frees nothing (empty_cache only returns *unrefed*
    # blocks to the driver). Detach the hooks and pull the weights back to CPU
    # so their CUDA storage actually drops to zero refs before we collect.
    if model is not None:
        try:
            from accelerate.hooks import remove_hook_from_module

            remove_hook_from_module(model, recurse=True)
        except Exception:
            pass
        try:
            model.to("cpu")
        except Exception:
            pass
        del model
    import gc

    gc.collect()
    _runtime.free_cuda_cache()
