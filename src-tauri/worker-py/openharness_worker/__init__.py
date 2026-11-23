"""OpenHarness Python worker — the second (and final) engine family.

A thin JSON-Lines-over-stdio server exposing generation pipelines
(text-to-image, STT, TTS, text-to-video). Heavy dependencies are imported
lazily by the pipeline modules, so the server itself starts instantly and
runs on the standard library alone.

Versioned with the app; the Rust side pins this via WORKER_VERSION.
"""

__version__ = "0.1.0"
