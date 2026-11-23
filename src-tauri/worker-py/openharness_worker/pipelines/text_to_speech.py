"""Text-to-speech via Kokoro (82M, fast even on CPU).

Qwen3-TTS / XTTS-class voice-clone engines join later as additional
`engine` values; the params surface is designed to accommodate them.

Params:
    text       what to say (required)
    voice      Kokoro voice id (default "af_heart")
    speed      0.5..2.0 (default 1.0)
    lang_code  Kokoro language code; defaults to the voice's first letter
               ("a" American English, "b" British, "e" es, "f" fr-fr,
                "h" hi, "i" it, "j" ja, "p" pt-br, "z" zh)
    model      installed local Kokoro model dir (required) — no auto-download:
               the worker never reaches Hugging Face for weights.

Output: `speech-{job_id}.wav` (24 kHz mono). Meta reports voice, duration
seconds, and sample rate.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import _runtime

SAMPLE_RATE = 24_000

_cache: dict = {}  # {"key": (lang_code, repo), "pipe": KPipeline}


def parse_params(params: dict) -> dict:
    text = params.get("text")
    if not text or not str(text).strip():
        raise ValueError("text_to_speech requires 'text'")
    model = params.get("model")
    if not model or not str(model).strip():
        raise ValueError(
            "No speech model selected — install a Kokoro model first."
        )
    voice = str(params.get("voice") or "af_heart")
    lang_code = str(params.get("lang_code") or voice[0]).lower()
    speed = _runtime.opt(params, "speed", 1.0, float)
    if not 0.5 <= speed <= 2.0:
        raise ValueError("speed must be 0.5..2.0")
    return {
        "text": str(text),
        "voice": voice,
        "speed": speed,
        "lang_code": lang_code,
        "model": str(model),
    }


def _load(lang_code: str, repo: str):
    # No auto-download: `repo` must point at an installed local Kokoro model
    # dir. Reject non-local values so KPipeline never pulls from Hugging Face.
    # The path check runs before the heavy import so it can't be skipped.
    if not Path(repo).exists():
        raise FileNotFoundError(
            f"speech model not found on disk: {repo!r} — install a Kokoro "
            "model from the Store first (no auto-download)."
        )

    from kokoro import KPipeline

    key = (lang_code, repo)
    if _cache.get("key") == key:
        return _cache["pipe"]
    unload()
    # Belt-and-braces: force HF offline for the load so even a cache lookup
    # can't trigger a network fetch.
    prev = {
        k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        pipe = KPipeline(lang_code=lang_code, repo_id=repo)
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _cache["key"] = key
    _cache["pipe"] = pipe
    return pipe


def preload(params: dict, ctx) -> dict:
    """Load the Kokoro pipeline without synthesizing (W1 explicit preload)."""
    model = params.get("model")
    if not model or not str(model).strip():
        raise ValueError(
            "No speech model selected — install a Kokoro model first."
        )
    repo = str(model)
    voice = str(params.get("voice") or "af_heart")
    lang_code = str(params.get("lang_code") or voice[0]).lower()
    ctx.progress(pct=5.0, stage="loading model")
    _load(lang_code, repo)
    ctx.progress(pct=100.0, stage="loaded")
    return {"model": repo}


def run(params: dict, ctx) -> tuple[list[str], dict]:
    import numpy as np
    import soundfile as sf

    p = parse_params(params)

    ctx.progress(pct=0.0, stage="loading model")
    pipe = _load(p["lang_code"], p["model"])
    ctx.check_cancelled()

    total_chars = max(len(p["text"]), 1)
    done_chars = 0
    chunks = []
    ctx.progress(pct=2.0, stage="synthesizing")
    for result in pipe(p["text"], voice=p["voice"], speed=p["speed"]):
        ctx.check_cancelled()
        audio = result.audio
        if audio is None:
            continue
        # torch tensor → numpy float32
        chunks.append(np.asarray(audio, dtype=np.float32))
        done_chars += len(result.graphemes or "")
        ctx.progress(
            pct=min(2.0 + done_chars / total_chars * 93.0, 95.0),
            stage="synthesizing",
        )

    if not chunks:
        raise RuntimeError("Kokoro produced no audio for the given text")

    ctx.progress(pct=97.0, stage="saving")
    waveform = np.concatenate(chunks)
    out = ctx.output_dir / f"speech-{ctx.job_id}.wav"
    sf.write(str(out), waveform, SAMPLE_RATE)

    meta = {
        "voice": p["voice"],
        "speed": p["speed"],
        "sample_rate": SAMPLE_RATE,
        "duration": round(len(waveform) / SAMPLE_RATE, 2),
    }
    return [str(out)], meta


def unload() -> None:
    _cache.pop("pipe", None)
    _cache.pop("key", None)
    import gc

    gc.collect()
    _runtime.free_cuda_cache()
