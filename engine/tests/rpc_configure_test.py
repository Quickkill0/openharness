#!/usr/bin/env python3
"""Check that the engine refuses to configure with remote RPC devices.

ggml-rpc opens outbound TCP connections to other hosts, which the engine
never does. Configuring engine/ with GGML_RPC=ON, or with upstream's
deprecated LLAMA_RPC alias, must fail with the OpenHarness message. The check
runs before project(), so no compiler work happens.
"""

import argparse
import pathlib
import subprocess
import sys
import tempfile

ENGINE_DIR = pathlib.Path(__file__).resolve().parents[1]
MESSAGE = "is not supported. Remote RPC devices connect to other hosts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmake", required=True)
    ap.add_argument("--generator", required=True)
    args = ap.parse_args()
    with tempfile.TemporaryDirectory(prefix="oh-rpc-configure-") as tmp:
        for opt in ("GGML_RPC", "LLAMA_RPC"):
            proc = subprocess.run(
                [args.cmake, "-S", str(ENGINE_DIR), "-B", str(pathlib.Path(tmp) / opt),
                 "-G", args.generator, f"-D{opt}=ON"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120)
            out = proc.stdout.decode("utf-8", "replace")
            assert proc.returncode != 0, f"{opt}=ON configured\n{out[-3000:]}"
            # CMake word-wraps error text; compare with whitespace collapsed.
            assert f"{opt}=ON {MESSAGE}" in " ".join(out.split()),f"{opt}=ON: no OpenHarness refusal\n{out[-3000:]}"
            print(f"ok: {opt}=ON refused at configure")
    return 0


if __name__ == "__main__":
    sys.exit(main())
