"""Entry point: python -m openharness_worker --output-dir <dir>"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .server import Server


def main() -> int:
    parser = argparse.ArgumentParser(prog="openharness_worker")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where pipelines write generated files.",
    )
    args = parser.parse_args()

    # The protocol lives on stdout — force UTF-8 + line discipline so JSON
    # lines survive the Windows console layer untouched.
    sys.stdout.reconfigure(encoding="utf-8", newline="\n")  # type: ignore[union-attr]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    Server(output_dir).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
