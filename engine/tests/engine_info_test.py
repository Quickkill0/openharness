#!/usr/bin/env python3
"""`--engine-info` prints one JSON capability record and exits 0."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--vram-tier", type=int, choices=(0, 1), required=True,
                    help="1 if this build compiled the CUDA VRAM expert tier")
    args = ap.parse_args()

    proc = subprocess.run([args.server, "--engine-info"], capture_output=True, text=True, timeout=50)
    assert proc.returncode == 0, f"exit {proc.returncode}: {proc.stderr}"
    info = json.loads(proc.stdout.strip().splitlines()[-1])

    upstream = json.loads((ENGINE_DIR / "UPSTREAM.json").read_text(encoding="utf-8"))
    assert info["engine"] == "openharness-engine", info
    assert info["llama_cpp"]["tag"] == upstream["tag"], info
    assert info["llama_cpp"]["commit"] == upstream["commit"], info
    for flag in ("ssd_expert_streaming", "cuda_vram_expert_tier", "web_ui", "remote_downloads"):
        assert isinstance(info["features"][flag], bool), flag
    assert info["features"]["ssd_expert_streaming"] is True
    assert info["features"]["cuda_vram_expert_tier"] is bool(args.vram_tier), info
    assert info["features"]["web_ui"] is False
    assert info["features"]["remote_downloads"] is False

    names = [b["name"] for b in info["backends"]]
    assert "CPU" in names, names
    for backend in info["backends"]:
        for dev in backend["devices"]:
            assert dev["type"] in ("cpu", "gpu", "igpu", "accel", "unknown"), dev
            assert dev["memory_total_mib"] >= 0
    print(f"ok: backends={names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
