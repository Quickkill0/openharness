#!/usr/bin/env python3
"""Track the InferSSD files imported into engine/inferssd.

engine/inferssd mirrors the layout of InferSSD's engine/ directory. Each
imported file is recorded in engine/inferssd/IMPORT.json with the SHA-256 of
its pristine InferSSD content, so the tree can be checked offline: any file
that differs from its recorded hash, and any file that is not an InferSSD
import (our own build glue), must be listed in engine/PATCHES.md as
`inferssd/<path>`.

Commands (stdlib only):

  import --source DIR PATH...   Copy PATH (files or directories, relative to
                                InferSSD's engine/ directory) from the InferSSD
                                checkout DIR and record their hashes. DIR is
                                the repository root at the pinned revision.
                                PATH must stay inside engine/ (no '..', no
                                absolute or drive paths, no backslashes), and
                                import never overwrites a local change or a
                                file that is not already an import.
  verify                        Check engine/inferssd against IMPORT.json and
                                PATCHES.md, including that every IMPORT.json
                                entry stays inside engine/inferssd. Needs no
                                network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))  # explicit, so it also works under `python -I`
from vendor_llama_cpp import contained_path, is_contained, is_safe_member  # noqa: E402

ENGINE_DIR = SCRIPTS_DIR.parent
IMPORT_DIR = ENGINE_DIR / "inferssd"
IMPORT_JSON = IMPORT_DIR / "IMPORT.json"
PATCHES_MD = ENGINE_DIR / "PATCHES.md"
UPSTREAM = "https://github.com/wici-ai/inferssd"
PINNED_COMMIT = "84bcc569ac5f41690ad38cbd820b6113c9accc45"


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_record() -> dict:
    if not IMPORT_JSON.is_file():
        return {"upstream": UPSTREAM, "commit": PINNED_COMMIT, "files": {}}
    return json.loads(IMPORT_JSON.read_text(encoding="utf-8"))


def escaping_paths(paths) -> list[str]:
    """Record entries that are not plain relative paths inside engine/inferssd
    (traversal, absolute or drive-qualified names, backslashes, symlinks out),
    plus IMPORT.json itself, which is ours, not an import."""
    return sorted(p for p in paths if p == IMPORT_JSON.name or not is_contained(IMPORT_DIR, p))


def save_record(record: dict) -> None:
    record["files"] = dict(sorted(record["files"].items()))
    IMPORT_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def local_files() -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for path in IMPORT_DIR.rglob("*"):
        if path.is_file() and path != IMPORT_JSON and "__pycache__" not in path.parts:
            out[path.relative_to(IMPORT_DIR).as_posix()] = path.read_bytes()
    return out


def ledger_paths(text: str | None = None) -> set[str]:
    """Paths listed in PATCHES.md as list items starting with `inferssd/<path>`."""
    if text is None:
        text = PATCHES_MD.read_text(encoding="utf-8") if PATCHES_MD.is_file() else ""
    return set(re.findall(r"^\s*- `inferssd/([^`\s<>]+)`", text, re.MULTILINE))


def classify(record_files: dict[str, str], files: dict[str, bytes]):
    modified = sorted(p for p in record_files.keys() & files.keys() if sha256(files[p]) != record_files[p])
    added = sorted(files.keys() - record_files.keys())
    removed = sorted(record_files.keys() - files.keys())
    return modified, added, removed


def crlf_files(files: dict[str, bytes], paths: list[str]) -> list[str]:
    """Paths among `paths` whose content has CRLF line endings."""
    return [p for p in paths if b"\r\n" in files[p]]


def source_commit(source: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def cmd_import(args) -> None:
    source = Path(args.source).resolve()
    engine_src = source / "engine"
    if not engine_src.is_dir():
        die(f"{source} does not look like an InferSSD checkout (no engine/ directory)")
    commit = source_commit(source)
    if commit and commit != PINNED_COMMIT and not args.allow_other_commit:
        die(f"checkout is at {commit}, pinned revision is {PINNED_COMMIT}")
    record = load_record()
    bad = escaping_paths(record["files"])
    if bad:
        die(f"{IMPORT_JSON} has entries outside engine/inferssd: {', '.join(bad)}")
    for rel in args.paths:
        if not is_safe_member(rel):
            die(f"{rel!r} must be a relative POSIX path inside InferSSD's engine/ directory "
                "(no '..', no absolute or drive paths, no backslashes)")
    # Collect and check everything before copying anything, so a refusal
    # leaves the tree and IMPORT.json untouched.
    plan: list[tuple[str, Path, Path]] = []
    for rel in args.paths:
        src = contained_path(engine_src, rel.rstrip("/"))
        if not src.exists():
            die(f"{rel} not found in {engine_src}")
        items = sorted(p for p in src.rglob("*") if p.is_file()) if src.is_dir() else [src]
        for item in items:
            relpath = item.relative_to(engine_src).as_posix()
            src_item = contained_path(engine_src, relpath)  # resolves symlinks out of the checkout
            dest = contained_path(IMPORT_DIR, relpath)
            if relpath == IMPORT_JSON.name:
                die(f"{relpath} would overwrite the import record")
            if dest.exists():
                if relpath not in record["files"]:
                    die(f"{relpath} exists in engine/inferssd but is not an InferSSD import; refusing to overwrite")
                if sha256(dest.read_bytes()) != record["files"][relpath]:
                    die(f"{relpath} has local changes; refusing to overwrite")
            plan.append((relpath, src_item, dest))
    for relpath, src_item, dest in plan:
        data = src_item.read_bytes()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        record["files"][relpath] = sha256(data)
    count = len(plan)
    save_record(record)
    print(f"imported {count} files into {IMPORT_DIR}")


def cmd_verify(args) -> None:  # noqa: ARG001
    record = load_record()
    if not record["files"]:
        die(f"{IMPORT_JSON} lists no files")
    bad = escaping_paths(record["files"])
    for p in bad:
        print(f"escapes   {p!r}   <-- IMPORT.json entry outside engine/inferssd")
    if bad:
        die(f"{IMPORT_JSON} has entries outside engine/inferssd")
    files = local_files()
    modified, added, removed = classify(record["files"], files)
    listed = ledger_paths()
    changed = modified + added + removed
    unlisted = [p for p in changed if p not in listed]
    stale = sorted(listed - set(changed))
    for label, paths in (("modified", modified), ("added", added), ("removed", removed)):
        for p in paths:
            mark = "" if p in listed else "   <-- not in PATCHES.md"
            print(f"{label:9} inferssd/{p}{mark}")
    for p in stale:
        print(f"stale     inferssd/{p}   <-- listed in PATCHES.md but unchanged")
    print(f"inferssd {record['commit']}: {len(record['files'])} imported files; "
          f"{len(modified)} modified, {len(added)} added, {len(removed)} removed")
    # InferSSD ships LF line endings throughout, and IMPORT.json only records
    # hashes, so a CRLF in a changed file means a tool converted it (which
    # turns a small patch into a whole-file rewrite).
    crlf = crlf_files(files, modified + added)
    for p in crlf:
        print(f"eol       inferssd/{p}   <-- CRLF line endings; InferSSD uses LF")
    if unlisted or stale:
        die("PATCHES.md does not match engine/inferssd")
    if crlf:
        die("engine/inferssd files must keep LF line endings (engine/.gitattributes marks inferssd/** -text)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import")
    p.add_argument("--source", required=True, help="InferSSD repository root")
    p.add_argument("--allow-other-commit", action="store_true")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_import)
    p = sub.add_parser("verify")
    p.set_defaults(func=cmd_verify)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
