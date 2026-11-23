#!/usr/bin/env python3
"""Install the engine into a scratch prefix and check what lands there.

`cmake --install` must succeed and install exactly the product: the server
executable plus the license and notice files. Nothing from upstream
llama.cpp (llama-server, static libraries, headers, CMake/pkg-config files)
may be installed.
"""

import argparse
import os
import pathlib
import subprocess
import sys
import tempfile

NOTICES = ["NOTICE", "llama.cpp-MIT.txt", "InferSSD-Apache-2.0.txt", "InferSSD-NOTICE.txt",
           "THIRD-PARTY-NOTICES.txt"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmake", required=True)
    ap.add_argument("--build-dir", required=True)
    ap.add_argument("--config", default="")
    args = ap.parse_args()
    exe = "openharness-engine-server" + (".exe" if os.name == "nt" else "")
    expected = {f"bin/{exe}"} | {f"share/openharness-engine/{n}" for n in NOTICES}
    with tempfile.TemporaryDirectory(prefix="oh-engine-install-") as prefix:
        cmd = [args.cmake, "--install", args.build_dir, "--prefix", prefix]
        if args.config:
            cmd += ["--config", args.config]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
        out = proc.stdout.decode("utf-8", "replace")
        assert proc.returncode == 0, out
        root = pathlib.Path(prefix)
        installed = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
        assert installed == expected, (f"missing: {sorted(expected - installed)}; "
                                       f"unexpected: {sorted(installed - expected)}")
        notices = (root / "share/openharness-engine/THIRD-PARTY-NOTICES.txt").read_text(encoding="utf-8")
        for needle in ("Copyright (c) 2017 yhirose", "Niels Lohmann", "Yann Collet"):
            assert needle in notices, needle
    print(f"ok: cmake --install installs exactly {len(expected)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
