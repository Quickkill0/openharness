#!/usr/bin/env python3
"""Vendor a pinned llama.cpp snapshot into engine/llama.cpp.

The vendored tree is a pruned copy of one upstream commit, with no nested .git.
engine/UPSTREAM.json records the tag, commit, pruning rules and a content digest
of the pristine pruned tree, so anyone can rebuild the same snapshot and see
exactly which files we changed.

Commands (stdlib only, run from anywhere):

  import <tag-or-sha>   Replace engine/llama.cpp with a fresh pruned snapshot of
                        that upstream ref and rewrite UPSTREAM.json. Refuses to
                        run while the vendored tree has local changes, or while
                        a nonempty tree has no UPSTREAM.json to check it
                        against, unless --discard-local-changes is given (save
                        changes with `diff` first, then re-apply them to the
                        new tree).
  verify                Rebuild the pristine snapshot of the recorded commit and
                        compare it with engine/llama.cpp. Every modified, added
                        or removed file must be listed in engine/PATCHES.md.
                        Exits non-zero on a digest mismatch or an unlisted change.
  diff -o FILE          Write a unified diff (pristine -> vendored) of our local
                        changes, for re-applying after an upstream update.
                        Paths are rooted at the repository (a/engine/llama.cpp/
                        ...), so `git apply FILE` runs from the repository root.

verify and diff download the recorded commit's tarball (about 40 MB) from
codeload.github.com over HTTPS each time they run; pass --archive FILE with a
saved copy to run them offline. import also asks github.com for the tag's
commit. Nothing else touches the network.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import NoReturn

ENGINE_DIR = Path(__file__).resolve().parent.parent
VENDOR_DIR = ENGINE_DIR / "llama.cpp"
# Prefix of vendored paths in `diff` output, relative to the repository root.
DIFF_PREFIX = "engine/llama.cpp"
UPSTREAM_JSON = ENGINE_DIR / "UPSTREAM.json"
PATCHES_MD = ENGINE_DIR / "PATCHES.md"

UPSTREAM_REPO = "ggml-org/llama.cpp"

# Top-level upstream paths we keep. Everything else (docs, examples, model
# vocab files, conversion scripts, the web UI sources, ...) is left out.
INCLUDE_TOP = [
    "AUTHORS",
    "CMakeLists.txt",
    "LICENSE",
    "README.md",
    "cmake",
    "common",
    "ggml",
    "gguf-py",
    "grammars",
    "include",
    "licenses",
    "scripts/ui-assets.cmake",
    "src",
    "tests",
    "tools",
    "vendor",
]

# Paths (fnmatch patterns, relative to the snapshot root) pruned from the
# included directories. Backends we do not build, the web UI sources (we build
# the server without the embedded UI) and test media.
EXCLUDE = [
    "ggml/src/ggml-cann/*",
    "ggml/src/ggml-et/*",
    "ggml/src/ggml-hexagon/*",
    "ggml/src/ggml-hip/*",
    "ggml/src/ggml-metal/*",
    "ggml/src/ggml-musa/*",
    "ggml/src/ggml-opencl/*",
    "ggml/src/ggml-openvino/*",
    "ggml/src/ggml-sycl/*",
    "ggml/src/ggml-virtgpu/*",
    "ggml/src/ggml-vulkan/*",
    "ggml/src/ggml-webgpu/*",
    "ggml/src/ggml-zdnn/*",
    "ggml/src/ggml-zendnn/*",
    "tools/connect/*",
    "tools/server/bench/*",
    "tools/server/tests/*",
    "tools/ui/*",
    "*.mp4",
]

# Exceptions to EXCLUDE: the UI CMake glue the server target still needs to
# generate an empty asset table.
KEEP = [
    "tools/ui/CMakeLists.txt",
    "tools/ui/README.md",
    "tools/ui/sources.cmake",
    "tools/ui/ui.cpp.in",
    "tools/ui/ui.h.in",
]


def die(msg: str) -> NoReturn:
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def keep_path(rel: str) -> bool:
    if not any(rel == top or rel.startswith(top + "/") for top in INCLUDE_TOP):
        return False
    if rel in KEEP:
        return True
    return not any(fnmatch.fnmatchcase(rel, pat) for pat in EXCLUDE)


def resolve_ref(ref: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", ref):
        return ref
    out = subprocess.run(
        ["git", "ls-remote", f"https://github.com/{UPSTREAM_REPO}", f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    peeled = {}
    for line in out.splitlines():
        sha, name = line.split("\t")
        peeled[name] = sha
    sha = peeled.get(f"refs/tags/{ref}^{{}}") or peeled.get(f"refs/tags/{ref}")
    if not sha:
        die(f"tag {ref!r} not found in {UPSTREAM_REPO}")
    return sha


def download(commit: str) -> bytes:
    url = f"https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/{commit}"
    print(f"downloading {url}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=300) as resp:
        return resp.read()


_DRIVE = re.compile(r"^[A-Za-z]:")


def is_safe_member(name: str) -> bool:
    """True if an archive member name is a plain relative POSIX path.

    Rejects absolute paths, `..`, `.` and empty components, drive-qualified
    names (`C:x`), backslashes (a path separator on Windows, so
    `ggml/..\\..\\outside.txt` would escape) and control characters.
    """
    if not name or "\\" in name or name.startswith("/") or _DRIVE.match(name):
        return False
    if any(ord(c) < 0x20 for c in name):
        return False
    return all(part not in ("", ".", "..") for part in name.rstrip("/").split("/"))


def is_contained(root: Path, rel: str) -> bool:
    """True if `rel` is a safe relative path that resolves (symlinks
    included) to somewhere strictly inside `root`."""
    if not is_safe_member(rel):
        return False
    base = root.resolve()
    return base in (base / rel).resolve().parents


def contained_path(root: Path, rel: str) -> Path:
    """Return root/rel, refusing anything that resolves outside root."""
    if not is_safe_member(rel):
        die(f"unsafe path: {rel!r}")
    if not is_contained(root, rel):
        die(f"path escapes {root}: {rel!r}")
    return (root.resolve() / rel).resolve()


def pristine_files(commit: str, archive: str | None = None) -> dict[str, bytes]:
    """Return {relative path: bytes} of the pruned snapshot of `commit`.

    Downloads the codeload tarball unless `archive` names a local copy of it.
    Callers that have a recorded digest check the result against it, so a
    wrong archive fails the same way a tampered download would.
    """
    if archive:
        return snapshot_files(Path(archive).read_bytes())
    return snapshot_files(download(commit))


def snapshot_files(data: bytes) -> dict[str, bytes]:
    """Return {relative path: bytes} of the pruned snapshot in a codeload tarball."""
    files: dict[str, bytes] = {}
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for member in tar:
            if not is_safe_member(member.name):
                die(f"unsafe path in archive: {member.name!r}")
            parts = member.name.split("/", 1)
            if len(parts) < 2 or not parts[1].rstrip("/"):
                continue
            rel = parts[1].rstrip("/")
            if not keep_path(rel):
                continue
            if member.issym() or member.islnk():
                die(f"archive contains a link we would vendor: {rel}; add it to EXCLUDE")
            if not member.isfile():
                continue
            fh = tar.extractfile(member)
            assert fh is not None
            files[rel] = fh.read()
    if "CMakeLists.txt" not in files:
        die("snapshot is missing CMakeLists.txt; wrong archive layout?")
    return files


def digest(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for rel in sorted(files):
        h.update(rel.encode())
        h.update(b"\0")
        h.update(hashlib.sha256(files[rel]).hexdigest().encode())
        h.update(b"\n")
    return h.hexdigest()


def vendored_files() -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if not VENDOR_DIR.is_dir():
        return files
    for path in VENDOR_DIR.rglob("*"):
        if path.is_file():
            files[path.relative_to(VENDOR_DIR).as_posix()] = path.read_bytes()
    return files


def compare(pristine: dict[str, bytes], vendored: dict[str, bytes]):
    modified = sorted(p for p in pristine.keys() & vendored.keys() if pristine[p] != vendored[p])
    added = sorted(vendored.keys() - pristine.keys())
    removed = sorted(pristine.keys() - vendored.keys())
    return modified, added, removed


def eol_style(content: bytes) -> str:
    """Line-ending style of a file: "lf", "crlf", "mixed" or "none"."""
    crlf = content.count(b"\r\n")
    lf = content.count(b"\n") - crlf
    if crlf and lf:
        return "mixed"
    return "crlf" if crlf else "lf" if lf else "none"


def eol_changes(pristine: dict[str, bytes], vendored: dict[str, bytes], modified: list[str]) -> list[str]:
    """Modified files whose line endings no longer match upstream.

    A checkout that converts line endings (core.autocrlf, or a tool writing
    in text mode on Windows) rewrites every line of a file, which turns a
    small patch into a whole-file one. The byte compare above cannot tell the
    two apart once the file is in PATCHES.md, so check the style itself.
    """
    out = []
    for rel in modified:
        before, after = eol_style(pristine[rel]), eol_style(vendored[rel])
        if before in ("lf", "crlf") and after != before:
            out.append(f"{rel} ({before} upstream, {after} here)")
    return out


def load_upstream() -> dict:
    if not UPSTREAM_JSON.is_file():
        die(f"{UPSTREAM_JSON} not found; run `import` first")
    return json.loads(UPSTREAM_JSON.read_text(encoding="utf-8"))


def ledger_paths() -> set[str]:
    """Paths listed in PATCHES.md as list items starting with `llama.cpp/<path>`."""
    if not PATCHES_MD.is_file():
        return set()
    text = PATCHES_MD.read_text(encoding="utf-8")
    return set(re.findall(r"^\s*- `llama\.cpp/([^`\s<>]+)`", text, re.MULTILINE))


def check_replaceable(discard_local_changes: bool) -> None:
    """Refuse to replace a vendored tree that may hold local work."""
    if discard_local_changes or not VENDOR_DIR.exists():
        return
    if not UPSTREAM_JSON.exists():
        if any(VENDOR_DIR.iterdir()):
            die(
                "engine/llama.cpp is not empty and engine/UPSTREAM.json is missing, so its local "
                "changes cannot be checked; move the tree aside or rerun with --discard-local-changes"
            )
        return
    current = load_upstream()
    modified, added, removed = compare(pristine_files(current["commit"]), vendored_files())
    if modified or added or removed:
        die(
            f"engine/llama.cpp has {len(modified) + len(added) + len(removed)} local changes; "
            "save them with `diff -o FILE`, then rerun with --discard-local-changes"
        )


def cmd_import(args) -> None:
    check_replaceable(args.discard_local_changes)
    commit = resolve_ref(args.ref)
    files = pristine_files(commit)
    tmp = Path(tempfile.mkdtemp(prefix="llama-vendor-", dir=ENGINE_DIR))
    try:
        for rel, content in files.items():
            dest = contained_path(tmp, rel)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content)
        if VENDOR_DIR.exists():
            shutil.rmtree(VENDOR_DIR)
        os.replace(tmp, VENDOR_DIR)
    finally:
        if tmp.exists():
            shutil.rmtree(tmp)
    build = re.fullmatch(r"b(\d+)", args.ref)
    record = {
        "upstream": f"https://github.com/{UPSTREAM_REPO}",
        "tag": args.ref if not re.fullmatch(r"[0-9a-f]{40}", args.ref) else None,
        "build_number": int(build.group(1)) if build else None,
        "commit": commit,
        "vendored_path": "llama.cpp",
        "include": INCLUDE_TOP,
        "exclude": EXCLUDE,
        "keep": KEEP,
        "file_count": len(files),
        "content_sha256": digest(files),
    }
    UPSTREAM_JSON.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"vendored {UPSTREAM_REPO}@{commit} ({len(files)} files) into {VENDOR_DIR}")


def cmd_verify(args) -> None:
    record = load_upstream()
    if record.get("include") != INCLUDE_TOP or record.get("exclude") != EXCLUDE or record.get("keep") != KEEP:
        die("pruning rules in this script differ from UPSTREAM.json; re-import to apply them")
    pristine = pristine_files(record["commit"], args.archive)
    got = digest(pristine)
    if got != record["content_sha256"]:
        die(f"pristine snapshot digest {got} != recorded {record['content_sha256']}")
    vendored = vendored_files()
    modified, added, removed = compare(pristine, vendored)
    listed = ledger_paths()
    unlisted = [p for p in modified + added + removed if p not in listed]
    stale = sorted(listed - set(modified + added + removed))
    for label, paths in (("modified", modified), ("added", added), ("removed", removed)):
        for p in paths:
            mark = "" if p in listed else "   <-- not in PATCHES.md"
            print(f"{label:9} llama.cpp/{p}{mark}")
    for p in stale:
        print(f"stale     llama.cpp/{p}   <-- listed in PATCHES.md but unchanged")
    eol = eol_changes(pristine, vendored, modified)
    for p in eol:
        print(f"eol       llama.cpp/{p}   <-- line endings differ from upstream")
    print(
        f"upstream {record['commit']} digest ok; "
        f"{len(modified)} modified, {len(added)} added, {len(removed)} removed"
    )
    if unlisted or stale:
        die("PATCHES.md does not match the vendored tree")
    if eol:
        die("vendored files must keep upstream line endings (engine/.gitattributes marks llama.cpp/** -text)")


def split_lines(text: str) -> list[str]:
    """Split on "\\n" only, keeping line ends.

    str.splitlines() also breaks on form feeds, vertical tabs and Unicode
    separators, which would corrupt the patch.
    """
    lines = text.split("\n")
    out = [ln + "\n" for ln in lines[:-1]]
    if lines[-1]:
        out.append(lines[-1])
    return out


def make_diff(pristine: dict[str, bytes], vendored: dict[str, bytes], prefix: str = DIFF_PREFIX) -> str:
    """Unified diff (pristine -> vendored) that `git apply` accepts from the repository root."""
    modified, added, removed = compare(pristine, vendored)
    chunks: list[str] = []
    for rel in modified + added + removed:
        try:
            a = split_lines(pristine[rel].decode("utf-8")) if rel in pristine else []
            b = split_lines(vendored[rel].decode("utf-8")) if rel in vendored else []
        except UnicodeDecodeError:
            die(f"binary file changed: {rel}; diff it by hand")
        src = f"a/{prefix}/{rel}" if rel in pristine else "/dev/null"
        dst = f"b/{prefix}/{rel}" if rel in vendored else "/dev/null"
        for i, line in enumerate(difflib.unified_diff(a, b, src, dst)):
            # The two header lines always end in "\n". A hunk line without one
            # is the last line of a file that has no final newline.
            if i >= 2 and not line.endswith("\n"):
                line += "\n\\ No newline at end of file\n"
            chunks.append(line)
    return "".join(chunks)


def cmd_diff(args) -> None:
    record = load_upstream()
    pristine = pristine_files(record["commit"], args.archive)
    got = digest(pristine)
    if got != record["content_sha256"]:
        die(f"pristine snapshot digest {got} != recorded {record['content_sha256']}")
    vendored = vendored_files()
    modified, added, removed = compare(pristine, vendored)
    Path(args.output).write_text(make_diff(pristine, vendored), encoding="utf-8", newline="")
    print(f"wrote {len(modified) + len(added) + len(removed)} file diffs to {args.output}; "
          "apply them with `git apply FILE` from the repository root")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("import")
    p.add_argument("ref", help="upstream tag (e.g. b12192) or full commit SHA")
    p.add_argument("--discard-local-changes", action="store_true")
    p.set_defaults(func=cmd_import)
    archive_help = ("local copy of the recorded commit's codeload tarball "
                    f"(https://codeload.github.com/{UPSTREAM_REPO}/tar.gz/<commit>); skips the download")
    p = sub.add_parser("verify", help="compare engine/llama.cpp with the recorded commit (downloads it)")
    p.add_argument("--archive", metavar="FILE", help=archive_help)
    p.set_defaults(func=cmd_verify)
    p = sub.add_parser("diff", help="write our local changes as a patch (downloads the recorded commit)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--archive", metavar="FILE", help=archive_help)
    p.set_defaults(func=cmd_diff)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
