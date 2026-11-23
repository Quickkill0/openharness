"""Pipeline registry.

Each pipeline is a module exposing:

    run(params: dict, ctx: JobContext) -> tuple[list[str], dict]
        Execute one job. Returns (output_paths, meta). Must call
        ctx.check_cancelled() at safe points and may call ctx.progress(...).

    unload() -> None  (optional)
        Drop any cached model/VRAM held by the module.

Registry metadata is static — importing this package must NOT import any
heavy dependency. `available()` checks use find_spec only.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Any

# task -> spec. `requires` lists top-level modules that must be importable
# for the pipeline to be available; empty means always available. `formats`
# describes what the `model` param accepts (drives store badges in M9).
PIPELINES: dict[str, dict[str, Any]] = {
    "echo": {
        "module": "openharness_worker.pipelines.echo",
        "summary": "Test pipeline — echoes params to a text file with fake progress.",
        "requires": [],
        "hidden": True,
        "formats": [],
    },
    "text_to_image": {
        "module": "openharness_worker.pipelines.text_to_image",
        "summary": "Generate images from text prompts (diffusers: SDXL, Flux).",
        "requires": ["torch", "diffusers"],
        "hidden": False,
        "formats": ["diffusers-dir", "safetensors-single-file"],
    },
    "transcribe": {
        "module": "openharness_worker.pipelines.transcribe",
        "summary": "Speech-to-text with timestamps (faster-whisper).",
        "requires": ["faster_whisper"],
        "hidden": False,
        "formats": ["ctranslate2-dir", "whisper-size-name"],
    },
    "text_to_speech": {
        "module": "openharness_worker.pipelines.text_to_speech",
        "summary": "Synthesize speech from text (Kokoro and friends).",
        "requires": ["kokoro", "soundfile"],
        "hidden": False,
        "formats": ["kokoro-repo"],
    },
    "vibevoice": {
        "module": "openharness_worker.pipelines.vibevoice",
        "summary": "Voice cloning TTS from a reference audio sample (VibeVoice).",
        "requires": ["torch", "vibevoice"],
        "hidden": False,
        "formats": ["vibevoice-dir"],
    },
    "text_to_video": {
        "module": "openharness_worker.pipelines.text_to_video",
        "summary": "Generate short video clips from text prompts (LTX-Video class).",
        "requires": ["torch", "diffusers", "imageio"],
        "hidden": False,
        "formats": ["diffusers-dir"],
    },
}

_loaded: dict[str, Any] = {}


def missing_requirements(task: str) -> list[str]:
    spec = PIPELINES[task]
    missing = []
    for mod in spec["requires"]:
        try:
            if importlib.util.find_spec(mod) is None:
                missing.append(mod)
        except (ImportError, ValueError):
            missing.append(mod)
    return missing


def capabilities() -> list[dict[str, Any]]:
    caps = []
    for task, spec in PIPELINES.items():
        missing = missing_requirements(task)
        caps.append(
            {
                "task": task,
                "summary": spec["summary"],
                "available": not missing,
                "missing": missing,
                "hidden": spec["hidden"],
                "formats": spec["formats"],
            }
        )
    return caps


def get_pipeline(task: str):
    """Lazy-import and cache the pipeline module for `task`.

    Raises KeyError for unknown tasks and RuntimeError when requirements
    are missing (with an actionable message)."""
    if task not in PIPELINES:
        raise KeyError(f"unknown task: {task}")
    if task in _loaded:
        return _loaded[task]
    missing = missing_requirements(task)
    if missing:
        raise RuntimeError(
            f"pipeline '{task}' is unavailable — missing packages: {', '.join(missing)}"
        )
    module = importlib.import_module(PIPELINES[task]["module"])
    _loaded[task] = module
    return module


def unload(task: str | None) -> list[str]:
    """Unload one loaded pipeline (or all when task is None). Returns the
    list of tasks actually unloaded."""
    targets = [task] if task else list(_loaded.keys())
    done = []
    for t in targets:
        module = _loaded.pop(t, None)
        if module is None:
            continue
        unload_fn = getattr(module, "unload", None)
        if callable(unload_fn):
            unload_fn()
        done.append(t)
    return done
