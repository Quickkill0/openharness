"""Test pipeline: writes params to a text file with simulated progress.

Used by Rust integration tests and the runtime smoke check to exercise the
full submit/progress/done/cancel surface without any heavy dependency or
GPU work.
"""

from __future__ import annotations

import json
import time


def preload(params: dict, ctx) -> dict:
    """Test preload: optional delay, then report fake load meta."""
    delay = float(params.get("delay", 0.0))
    ctx.progress(pct=10.0, stage="loading model")
    if delay:
        time.sleep(delay)
    ctx.check_cancelled()
    ctx.progress(pct=100.0, stage="loaded")
    return {"vram_mb": int(params.get("vram_mb", 0)) or None}


def run(params: dict, ctx) -> tuple[list[str], dict]:
    steps = int(params.get("steps", 5))
    delay = float(params.get("delay", 0.05))
    for i in range(steps):
        ctx.check_cancelled()
        time.sleep(delay)
        ctx.progress(
            pct=(i + 1) / steps * 100.0,
            stage="echoing",
            step=i + 1,
            total_steps=steps,
        )
    out_path = ctx.output_dir / f"echo-{ctx.job_id}.txt"
    out_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return [str(out_path)], {"steps": steps}
