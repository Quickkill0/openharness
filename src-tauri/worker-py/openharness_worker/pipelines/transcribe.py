"""Speech-to-text via faster-whisper.

Params:
    audio            local path to a media file (required)
    model            local CTranslate2 model dir (required) — an installed
                     faster-whisper model. No size-name auto-download: the
                     worker never reaches Hugging Face for weights.
    language         ISO code; autodetected when absent
    task             "transcribe" | "translate" (default transcribe)
    word_timestamps  bool (default false)

Outputs: `transcript-{job_id}.json` (segments with start/end/text [+words])
and `transcript-{job_id}.txt` (plain text). Meta reports detected language
and duration.
"""

from __future__ import annotations

import json
from pathlib import Path

_cache: dict = {}  # {"key": (model, device, compute), "model": WhisperModel}


def parse_params(params: dict) -> dict:
    audio = params.get("audio")
    if not audio:
        raise ValueError("transcribe requires 'audio' (path to a media file)")
    model = params.get("model")
    if not model or not str(model).strip():
        raise ValueError(
            "No transcription model selected — install a Whisper model first."
        )
    task = str(params.get("task") or "transcribe").lower()
    if task not in ("transcribe", "translate"):
        raise ValueError(f"unknown task: {task} (expected transcribe | translate)")
    return {
        "audio": str(audio),
        "model": str(model),
        "language": params.get("language") or None,
        "task": task,
        "word_timestamps": bool(params.get("word_timestamps", False)),
    }


def _device_and_compute() -> tuple[str, str]:
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:
        pass
    return "cpu", "int8"


def _load(model: str):
    # No auto-download: `model` must be an installed local CTranslate2 dir (or
    # a model.bin path). Reject anything that isn't on disk so faster-whisper
    # never falls through to a Hugging Face fetch for a bare size name. The
    # path check runs before the heavy import so it can't be skipped.
    if not Path(model).exists():
        raise FileNotFoundError(
            f"transcription model not found on disk: {model!r} — install a "
            "Whisper model from the Store first (no auto-download)."
        )

    from faster_whisper import WhisperModel

    device, compute = _device_and_compute()
    key = (model, device, compute)
    if _cache.get("key") == key:
        return _cache["model"]
    unload()
    # local_files_only blocks any network access even if a HF cache exists.
    wm = WhisperModel(
        model, device=device, compute_type=compute, local_files_only=True
    )
    _cache["key"] = key
    _cache["model"] = wm
    return wm


def preload(params: dict, ctx) -> dict:
    """Load the model into memory without transcribing (W1 explicit preload).

    No vram_mb: faster-whisper allocates through CTranslate2, which torch's
    allocator counters can't see — the ledger keeps its estimate.
    """
    model = params.get("model")
    if not model or not str(model).strip():
        raise ValueError(
            "No transcription model selected — install a Whisper model first."
        )
    model = str(model)
    ctx.progress(pct=5.0, stage="loading model")
    _load(model)
    ctx.progress(pct=100.0, stage="loaded")
    return {"model": model}


def run(params: dict, ctx) -> tuple[list[str], dict]:
    p = parse_params(params)
    audio = Path(p["audio"])
    if not audio.exists():
        raise FileNotFoundError(f"audio file does not exist: {audio}")

    ctx.progress(pct=0.0, stage="loading model")
    model = _load(p["model"])
    ctx.check_cancelled()

    ctx.progress(pct=2.0, stage="transcribing")
    segments_iter, info = model.transcribe(
        str(audio),
        language=p["language"],
        task=p["task"],
        word_timestamps=p["word_timestamps"],
        vad_filter=True,
    )

    duration = float(info.duration or 0.0)
    segments = []
    texts = []
    for seg in segments_iter:
        ctx.check_cancelled()
        entry = {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        }
        if p["word_timestamps"] and seg.words:
            entry["words"] = [
                {"start": round(w.start, 3), "end": round(w.end, 3), "word": w.word}
                for w in seg.words
            ]
        segments.append(entry)
        texts.append(entry["text"])
        if duration > 0:
            ctx.progress(
                pct=min(seg.end / duration * 95.0, 95.0),
                stage="transcribing",
            )

    ctx.progress(pct=97.0, stage="saving")
    json_path = ctx.output_dir / f"transcript-{ctx.job_id}.json"
    txt_path = ctx.output_dir / f"transcript-{ctx.job_id}.txt"
    json_path.write_text(
        json.dumps(
            {
                "language": info.language,
                "duration": duration,
                "segments": segments,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    txt_path.write_text("\n".join(texts), encoding="utf-8")

    meta = {
        "language": info.language,
        "language_probability": round(float(info.language_probability or 0.0), 3),
        "duration": round(duration, 2),
        "segment_count": len(segments),
    }
    return [str(json_path), str(txt_path)], meta


def unload() -> None:
    _cache.pop("model", None)
    _cache.pop("key", None)
    import gc

    gc.collect()
