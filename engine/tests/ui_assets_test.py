#!/usr/bin/env python3
"""Check that a NO_UI build embeds no web UI assets, even from a stale dist/.

Runs llama.cpp's scripts/ui-assets.cmake (as the build does) with UI
downloads and npm builds off and leftover dist/ trees in both the source and
the binary dir. With NO_UI=ON the generated ui.h must not define
LLAMA_UI_HAS_ASSETS. A control run with NO_UI=OFF shows the stale assets
would otherwise be embedded, so the check is not vacuous.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
LLAMA_DIR = ENGINE_DIR / "llama.cpp"
# A complete stale build, as upstream's asset validation expects one.
STALE_ASSETS = ["index.html", "manifest.webmanifest", "sw.js", "build.json", "version.json",
                "bundle-stale.js", "bundle-stale.css", "workbox-stale.js"]


def generate(cmake: str, root: pathlib.Path, no_ui: str) -> str:
    src, bin_dir = root / "src", root / "bin"
    for dist in (src / "dist", bin_dir / "dist"):
        dist.mkdir(parents=True, exist_ok=True)
        for name in STALE_ASSETS:
            (dist / name).write_text(f"stale {name}\n", encoding="utf-8")
    proc = subprocess.run([
        cmake,
        f"-DUI_SOURCE_DIR={src}", f"-DUI_BINARY_DIR={bin_dir}", f"-DLLAMA_SOURCE_DIR={LLAMA_DIR}",
        "-DLLAMA_BUILD_NUMBER=0", "-DHF_BUCKET=", "-DHF_VERSION=", "-DHF_ENABLED=OFF", "-DBUILD_UI=OFF",
        "-DLLAMA_UI_GZIP=ON", f"-DNO_UI={no_ui}",
        "-P", str(LLAMA_DIR / "scripts" / "ui-assets.cmake"),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode == 0, out
    return (bin_dir / "ui.h").read_text(encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmake", required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="oh-ui-assets-") as tmp:
        tmp = pathlib.Path(tmp)
        header = generate(args.cmake, tmp / "no-ui", "ON")
        assert "#define LLAMA_UI_HAS_ASSETS" not in header, header
        control = generate(args.cmake, tmp / "control", "OFF")
        assert "#define LLAMA_UI_HAS_ASSETS" in control, control
    print("ok: NO_UI=ON embeds no assets despite stale dist/ trees (control run embeds them)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
