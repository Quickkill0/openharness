#!/usr/bin/env python3
"""Offline checks for engine/scripts/vendor_llama_cpp.py (no network)."""

import importlib.util
import io
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("vendor", ENGINE_DIR / "scripts" / "vendor_llama_cpp.py")
assert spec and spec.loader
vendor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor)


def test_pruning_rules():
    keep = vendor.keep_path
    assert keep("CMakeLists.txt")
    assert keep("src/llama.cpp")
    assert keep("ggml/src/ggml-cuda/mmq.cu")
    assert keep("ggml/src/ggml-cpu/ggml-cpu.c")
    assert keep("tools/server/server.cpp")
    assert keep("tools/ui/ui.h.in")
    assert keep("scripts/ui-assets.cmake")
    assert not keep("scripts/sync-ggml.sh")
    assert not keep("docs/build.md")
    assert not keep("models/ggml-vocab-llama-bpe.gguf")
    assert not keep("ggml/src/ggml-vulkan/ggml-vulkan.cpp")
    assert not keep("tools/ui/src/app.svelte")
    assert not keep("tools/server/tests/unit/test_basic.py")
    assert not keep("tools/mtmd/test-3.mp4")
    assert not keep("CMakeLists.txt.bak")


def test_digest_is_order_independent_and_content_sensitive():
    a = {"x/a.c": b"1", "b.h": b"2"}
    b = {"b.h": b"2", "x/a.c": b"1"}
    assert vendor.digest(a) == vendor.digest(b)
    assert vendor.digest(a) != vendor.digest({"x/a.c": b"1", "b.h": b"3"})
    assert vendor.digest(a) != vendor.digest({"x/a.c": b"1", "c.h": b"2"})


def test_compare():
    pristine = {"a": b"1", "b": b"2", "c": b"3"}
    vendored = {"a": b"1", "b": b"changed", "d": b"new"}
    assert vendor.compare(pristine, vendored) == (["b"], ["d"], ["c"])


def test_eol_changes():
    LF, CRLF = b"\n", b"\r\n"
    assert vendor.eol_style(b"a" + LF + b"b" + LF) == "lf"
    assert vendor.eol_style(b"a" + CRLF + b"b" + CRLF) == "crlf"
    assert vendor.eol_style(b"a" + CRLF + b"b" + LF) == "mixed"
    assert vendor.eol_style(b"abc") == "none"
    pristine = {"lf.c": b"a" + LF + b"b" + LF, "crlf.c": b"a" + CRLF + b"b" + CRLF,
                "mixed.c": b"a" + CRLF + b"b" + LF, "ok.c": b"a" + LF + b"b" + LF}
    vendored = {"lf.c": b"a" + CRLF + b"c" + CRLF, "crlf.c": b"a" + LF + b"c" + LF,
                "mixed.c": b"a" + LF + b"c" + LF, "ok.c": b"a" + LF + b"c" + LF}
    modified, _, _ = vendor.compare(pristine, vendored)
    got = vendor.eol_changes(pristine, vendored, modified)
    assert got == ["crlf.c (crlf upstream, lf here)", "lf.c (lf upstream, crlf here)"], got


def test_ledger_parsing():
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "PATCHES.md"
        md.write_text("- `llama.cpp/src/llama-graph.cpp`: hook\n- `llama.cpp/ggml/src/ggml-ssd/store.cpp` added\n"
                      "Example: `` `llama.cpp/<path>` ``\n", encoding="utf-8")
        old = vendor.PATCHES_MD
        vendor.PATCHES_MD = md
        try:
            assert vendor.ledger_paths() == {"src/llama-graph.cpp", "ggml/src/ggml-ssd/store.cpp"}
        finally:
            vendor.PATCHES_MD = old


def test_upstream_record_matches_rules():
    record = json.loads((ENGINE_DIR / "UPSTREAM.json").read_text(encoding="utf-8"))
    assert record["include"] == vendor.INCLUDE_TOP
    assert record["exclude"] == vendor.EXCLUDE
    assert record["keep"] == vendor.KEEP
    assert len(record["commit"]) == 40
    assert record["tag"] == f"b{record['build_number']}"


def expect_exit(fn, *args):
    try:
        fn(*args)
    except SystemExit as e:
        assert e.code == 1, e.code
        return
    raise AssertionError(f"{fn.__name__}{args} did not refuse")


BS = chr(92)  # backslash, the Windows path separator
WIN_TRAVERSAL = "ggml/.." + BS + ".." + BS + "outside.txt"  # escapes two levels up on Windows


def test_member_guard():
    ok = ["llama.cpp-abc/", "llama.cpp-abc/ggml/src/ggml.c", "pax_global_header"]
    bad = [
        "", "/etc/passwd", "llama.cpp-abc/../outside.txt", "llama.cpp-abc/ggml/../../outside.txt",
        WIN_TRAVERSAL, "llama.cpp-abc" + BS + "ggml" + BS + "x.c", "C:/outside.txt",
        "c:outside.txt", "llama.cpp-abc//x", "llama.cpp-abc/./x", "llama.cpp-abc/x\ny",
    ]
    for name in ok:
        assert vendor.is_safe_member(name), name
    for name in bad:
        assert not vendor.is_safe_member(name), name


def make_tarball(members: dict) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_snapshot_rejects_traversal():
    good = {"root/CMakeLists.txt": b"x\n", "root/ggml/src/ggml.c": b"int x;\n", "root/docs/a.md": b"no\n"}
    assert vendor.snapshot_files(make_tarball(good)) == {"CMakeLists.txt": b"x\n", "ggml/src/ggml.c": b"int x;\n"}
    for evil in ("root/" + WIN_TRAVERSAL, "root/ggml/../../outside.txt", "/abs.txt", "C:/x.txt"):
        expect_exit(vendor.snapshot_files, make_tarball({**good, evil: b"pwned\n"}))


def test_contained_path():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "stage"
        root.mkdir()
        assert vendor.contained_path(root, "ggml/x.c") == (root / "ggml" / "x.c").resolve()
        for evil in (WIN_TRAVERSAL, "../outside.txt", "C:/outside.txt", "/outside.txt", "."):
            expect_exit(vendor.contained_path, root, evil)
            assert not vendor.is_contained(root, evil), evil
        assert vendor.is_contained(root, "ggml/x.c")


def test_pristine_files_from_local_archive():
    """--archive reads a saved tarball instead of downloading."""
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "snapshot.tar.gz"
        archive.write_bytes(make_tarball({"root/CMakeLists.txt": b"x\n", "root/docs/a.md": b"no\n"}))
        old = vendor.download
        vendor.download = lambda commit: (_ for _ in ()).throw(AssertionError("downloaded"))
        try:
            assert vendor.pristine_files("0" * 40, str(archive)) == {"CMakeLists.txt": b"x\n"}
        finally:
            vendor.download = old


def test_import_refuses_tree_without_provenance():
    with tempfile.TemporaryDirectory() as tmp:
        engine = Path(tmp)
        old = vendor.VENDOR_DIR, vendor.UPSTREAM_JSON
        vendor.VENDOR_DIR, vendor.UPSTREAM_JSON = engine / "llama.cpp", engine / "UPSTREAM.json"
        try:
            vendor.check_replaceable(False)  # no tree: fine
            vendor.VENDOR_DIR.mkdir()
            vendor.check_replaceable(False)  # empty tree: fine
            (vendor.VENDOR_DIR / "local-work.cpp").write_text("keep me\n", encoding="utf-8")
            expect_exit(vendor.check_replaceable, False)
            assert (vendor.VENDOR_DIR / "local-work.cpp").exists()
            vendor.check_replaceable(True)  # explicit discard
        finally:
            vendor.VENDOR_DIR, vendor.UPSTREAM_JSON = old


def test_diff_round_trip():
    if not shutil.which("git"):
        raise AssertionError("git is required for the patch round-trip test")
    pristine = {
        "CMakeLists.txt": b"project(x)\nadd_subdirectory(src)\n",
        "src/no-eol.c": b"int a;\nint b;",
        "src/gains-eol.c": b"int c;",
        "src/loses-eol.c": b"int d;\n",
        "src/crlf.txt": b"one\r\ntwo\r\n",
        "src/formfeed.c": b"int e;\x0c\nint f;\n",
        "src/removed.c": b"gone\n",
        "src/removed-no-eol.c": b"gone",
    }
    vendored = {
        "CMakeLists.txt": b"# Modified by OpenHarness\nproject(x)\nadd_subdirectory(src)\n",
        "src/no-eol.c": b"int a;\nint b2;",
        "src/gains-eol.c": b"int c;\n",
        "src/loses-eol.c": b"int d;",
        "src/crlf.txt": b"one\r\nTWO\r\n",
        "src/formfeed.c": b"int e;\x0c\nint f2;\n",
        "src/added.c": b"new\n",
        "src/added-no-eol.c": b"new",
    }
    patch = vendor.make_diff(pristine, vendored)
    assert "a/engine/llama.cpp/CMakeLists.txt" in patch
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp)
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        for rel, data in pristine.items():
            f = repo / "engine" / "llama.cpp" / rel
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(data)
        (repo / "local.patch").write_bytes(patch.encode("utf-8"))
        # Same invocation as the README documents: from the repository root.
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "--check", "local.patch"], cwd=repo, check=True)
        subprocess.run(["git", "-c", "core.autocrlf=false", "apply", "local.patch"], cwd=repo, check=True)
        tree = repo / "engine" / "llama.cpp"
        got = {p.relative_to(tree).as_posix(): p.read_bytes() for p in tree.rglob("*") if p.is_file()}
        assert got == vendored, sorted(set(got.items()) ^ set(vendored.items()))


def main() -> int:
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok: {t.__name__}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
