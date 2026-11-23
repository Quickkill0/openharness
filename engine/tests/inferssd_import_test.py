#!/usr/bin/env python3
"""Offline checks for engine/scripts/inferssd_import.py, plus a verify of the
real engine/inferssd tree against IMPORT.json and PATCHES.md."""

import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
SCRIPT = ENGINE_DIR / "scripts" / "inferssd_import.py"
spec = importlib.util.spec_from_file_location("inferssd_import", SCRIPT)
assert spec and spec.loader
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_ledger_reads_list_items_only():
    text = ("Imported: `inferssd/ggml/src/ggml-ssd/` and `inferssd/IMPORT.json`.\n"
            "- `inferssd/tests/ssd/test-a.cpp`: reason\n"
            "  - `inferssd/CMakeLists.txt`: glue\n"
            "- `llama.cpp/src/llama.cpp`: other ledger\n")
    assert mod.ledger_paths(text) == {"tests/ssd/test-a.cpp", "CMakeLists.txt"}


def test_classify():
    record = {"a": mod.sha256(b"1"), "b": mod.sha256(b"2"), "c": mod.sha256(b"3")}
    files = {"a": b"1", "b": b"changed", "d": b"ours"}
    assert mod.classify(record, files) == (["b"], ["d"], ["c"])


def test_crlf_files():
    files = {"lf.h": b"a\nb\n", "crlf.h": b"a\r\nb\r\n", "skip.h": b"a\r\n"}
    assert mod.crlf_files(files, ["lf.h", "crlf.h"]) == ["crlf.h"]


def test_record_is_pinned():
    record = mod.load_record()
    assert record["commit"] == mod.PINNED_COMMIT
    assert record["files"], "IMPORT.json lists no files"
    assert all(len(h) == 64 for h in record["files"].values())


def test_tree_matches_ledger():
    out = subprocess.run([sys.executable, "-I", str(SCRIPT), "verify"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


BS = chr(92)  # backslash, the Windows path separator
ESCAPES = ["../README.md", "ggml/../../README.md", "/abs/x.h", "C:/x.h", "C:x.h",
           "ggml" + BS + ".." + BS + ".." + BS + "README.md", "ggml" + BS + "x.h", ".", "a//b"]


@contextlib.contextmanager
def sandbox():
    """A fake engine/ (inferssd/ + PATCHES.md + README.md) and a fake InferSSD
    checkout, with the module pointed at them."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        engine = root / "engine"
        (engine / "inferssd").mkdir(parents=True)
        (engine / "README.md").write_bytes(b"ours\n")
        (engine / "PATCHES.md").write_bytes(b"")
        src = root / "inferssd-checkout"
        (src / "engine" / "ggml").mkdir(parents=True)
        (src / "engine" / "ggml" / "x.h").write_bytes(b"x\n")
        (src / "README.md").write_bytes(b"upstream readme\n")
        (src / "engine" / "IMPORT.json").write_bytes(b"{}\n")
        old = mod.IMPORT_DIR, mod.IMPORT_JSON, mod.PATCHES_MD
        mod.IMPORT_DIR, mod.IMPORT_JSON, mod.PATCHES_MD = (
            engine / "inferssd", engine / "inferssd" / "IMPORT.json", engine / "PATCHES.md")
        try:
            yield root, engine, src
        finally:
            mod.IMPORT_DIR, mod.IMPORT_JSON, mod.PATCHES_MD = old


def run_import(src: Path, *paths: str) -> None:
    mod.cmd_import(types.SimpleNamespace(source=str(src), allow_other_commit=True, paths=list(paths)))


def expect_refusal(fn, *args) -> str:
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            fn(*args)
    except SystemExit as e:
        assert e.code == 1, e.code
        return err.getvalue()
    raise AssertionError(f"{fn.__name__}{args} did not refuse")


def test_import_rejects_paths_outside_engine():
    with sandbox() as (_, engine, src):
        for rel in ESCAPES:
            expect_refusal(run_import, src, rel)
            # A bad path next to a good one: nothing is copied or recorded.
            expect_refusal(run_import, src, "ggml/x.h", rel)
        assert (engine / "README.md").read_bytes() == b"ours\n"
        assert not mod.IMPORT_JSON.exists()
        assert not (engine / "inferssd" / "ggml").exists()


def test_import_copies_and_records_inside_tree():
    with sandbox() as (_, engine, src):
        run_import(src, "ggml/")
        assert (engine / "inferssd" / "ggml" / "x.h").read_bytes() == b"x\n"
        assert mod.load_record()["files"] == {"ggml/x.h": mod.sha256(b"x\n")}
        run_import(src, "ggml/x.h")  # re-import of an unchanged file is fine


def test_import_refuses_to_overwrite_local_work():
    with sandbox() as (_, engine, src):
        (engine / "inferssd" / "ggml").mkdir()
        (engine / "inferssd" / "ggml" / "x.h").write_bytes(b"our own file\n")
        assert "not an InferSSD import" in expect_refusal(run_import, src, "ggml/x.h")
        assert (engine / "inferssd" / "ggml" / "x.h").read_bytes() == b"our own file\n"
        assert "import record" in expect_refusal(run_import, src, "IMPORT.json")


def test_import_rejects_symlink_out_of_checkout():
    with sandbox() as (root, _, src):
        (root / "secret.h").write_bytes(b"secret\n")
        try:
            os.symlink(root / "secret.h", src / "engine" / "ggml" / "link.h")
        except (OSError, NotImplementedError):
            print("skip: test_import_rejects_symlink_out_of_checkout (no symlink privilege)")
            return
        assert "escapes" in expect_refusal(run_import, src, "ggml/")
        assert not mod.IMPORT_JSON.exists()


def test_escaping_record_entries():
    with sandbox():
        assert mod.escaping_paths(["ggml/x.h", "tests/ssd/a.cpp"]) == []
        assert mod.escaping_paths(ESCAPES) == sorted(ESCAPES)
        assert mod.escaping_paths(["IMPORT.json"]) == ["IMPORT.json"]


def test_verify_and_import_fail_on_escaping_record():
    with sandbox() as (_, engine, src):
        run_import(src, "ggml/x.h")
        record = mod.load_record()
        record["files"]["../README.md"] = mod.sha256(b"ours\n")
        mod.IMPORT_JSON.write_text(json.dumps(record), encoding="utf-8")
        assert "outside engine/inferssd" in expect_refusal(mod.cmd_verify, None)
        assert "outside engine/inferssd" in expect_refusal(run_import, src, "ggml/x.h")
        assert (engine / "README.md").read_bytes() == b"ours\n"


def main() -> int:
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
