#!/usr/bin/env python3
"""Start openharness-engine-server on a tiny model and exercise its HTTP routes.

Checks that the OpenAI-compatible routes the app relies on answer, that
readiness goes through /health (a "loading" 503 must be seen before the 200;
test builds hold the load on a gate file so this is deterministic), that no
web UI is served (the root and /index.html are plain 404s with and without
Accept-Encoding: gzip), that the agent routes /cors-proxy and /tools do not
exist, and that
a default launch writes no files. The server runs with its working
directory, HOME/USERPROFILE, APPDATA/LOCALAPPDATA, XDG dirs, temp dir and
LLAMA_CACHE all pointed into one sandbox, and the sandbox must be empty after
shutdown apart from the NVIDIA driver's own cache folder. Runs on CPU only.
"""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(base: str, path: str, body=None, timeout=60, headers=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data,
                                 headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    sandbox = tempfile.mkdtemp(prefix="oh-engine-smoke-")
    dirs = {name: os.path.join(sandbox, name) for name in
            ("work", "home", "appdata", "localappdata", "xdg-cache", "xdg-config", "xdg-data", "tmp", "llama-cache")}
    for d in dirs.values():
        os.mkdir(d)
    workdir = dirs["work"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env["CUDA_VISIBLE_DEVICES"] = ""  # keep the smoke test on CPU even in CUDA builds
    env.update({
        "HOME": dirs["home"], "USERPROFILE": dirs["home"],
        "APPDATA": dirs["appdata"], "LOCALAPPDATA": dirs["localappdata"],
        "XDG_CACHE_HOME": dirs["xdg-cache"], "XDG_CONFIG_HOME": dirs["xdg-config"],
        "XDG_DATA_HOME": dirs["xdg-data"],
        "TMP": dirs["tmp"], "TEMP": dirs["tmp"], "TMPDIR": dirs["tmp"],
        "LLAMA_CACHE": dirs["llama-cache"],
    })
    cmd = [
        args.server, "-m", os.path.abspath(args.model),
        "--host", "127.0.0.1", "--port", str(port),
        "-c", "256", "-np", "1", "-t", "2", "-ngl", "0",
    ]
    # Test builds hold the model load until this file exists (see
    # OPENHARNESS_TEST_HOOKS in engine/CMakeLists.txt), so the "loading" 503
    # is observed every time, not only when loading happens to be slow. The
    # gate lives outside the sandbox so it does not count as a written file.
    gate_dir = tempfile.mkdtemp(prefix="oh-engine-gate-")
    gate = os.path.join(gate_dir, "release")
    env["OPENHARNESS_TEST_LOAD_GATE"] = gate
    proc = subprocess.Popen(cmd, cwd=workdir, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = b""
    try:
        deadline = time.time() + 120
        seen_503 = False
        while True:
            if proc.poll() is not None:
                raise AssertionError(f"server exited early with {proc.returncode}")
            try:
                status, body = request(base, "/health", timeout=2)
                if status == 200:
                    assert seen_503, "ready (200) before any loading 503 was observed"
                    assert json.loads(body).get("status") == "ok", body
                    break
                assert status == 503, (status, body)
                assert json.loads(body)["error"]["message"] == "Loading model", body
                if not seen_503:
                    seen_503 = True
                    open(gate, "w").close()  # release the load
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                pass
            if time.time() > deadline:
                raise AssertionError("server did not become ready in 120 s")
            time.sleep(0.25)

        status, body = request(base, "/v1/models")
        assert status == 200, (status, body)
        models = json.loads(body)
        assert models["data"], models

        status, body = request(base, "/v1/completions",
                               {"prompt": "hello", "max_tokens": 4, "temperature": 0, "seed": 1})
        assert status == 200, (status, body)
        comp = json.loads(body)
        assert comp["object"] == "text_completion", comp
        assert comp["usage"]["completion_tokens"] > 0, comp

        status, body = request(base, "/v1/chat/completions", {
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 4, "temperature": 0, "seed": 1,
        })
        assert status == 200, (status, body)
        chat = json.loads(body)
        assert chat["object"] == "chat.completion", chat
        assert chat["choices"][0]["message"]["role"] == "assistant", chat

        # Built without the embedded web UI: no UI route exists, so the root
        # and index.html fall through to the JSON 404 handler. A gzip UI build
        # would answer 415 without Accept-Encoding and the page with it, so
        # check both.
        for path in ("/", "/index.html"):
            for headers in ({}, {"Accept-Encoding": "gzip"}):
                status, body = request(base, path, headers=headers)
                assert status == 404, (path, headers, status, body[:200])
                assert json.loads(body)["error"]["type"] == "not_found_error", (path, headers, body[:200])

        # The agent routes (MCP CORS proxy, built-in tools) are not part of
        # this build, so they are plain 404s rather than upstream's 403.
        for path, body in (("/cors-proxy?url=http://127.0.0.1:9/x", None), ("/tools", None),
                           ("/cors-proxy?url=http://127.0.0.1:9/x", {"x": 1}),
                           ("/tools", {"tool": "get_info", "params": {}})):
            status, text = request(base, path, body)
            assert status == 404, (path, body, status, text[:200])
    except Exception:
        proc.kill()
        log = proc.communicate(timeout=30)[0]
        sys.stderr.write(log.decode("utf-8", "replace")[-6000:])
        raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                log = proc.communicate(timeout=30)[0]
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()

    # The one allowed exception: on Windows the NVIDIA driver creates its own
    # CUDA JIT cache folder (%APPDATA%\NVIDIA\ComputeCache) when the CUDA
    # backend loads, even with no visible device. That is the driver, not the
    # engine, and it holds no user data.
    driver_cache = os.path.join(dirs["appdata"], "NVIDIA")
    leftovers = [os.path.relpath(os.path.join(root, f), sandbox)
                 for root, _, files in os.walk(sandbox) if not root.startswith(driver_cache) for f in files]
    leftovers += [os.path.relpath(os.path.join(dirs[n], e), sandbox)
                  for n in dirs for e in os.listdir(dirs[n])
                  if os.path.isdir(os.path.join(dirs[n], e)) and os.path.join(dirs[n], e) != driver_cache]
    assert not leftovers, f"default launch wrote into the sandbox: {sorted(set(leftovers))}"
    shutil.rmtree(sandbox)
    shutil.rmtree(gate_dir)
    print("ok: /health 503 then 200, /v1/models, /v1/completions, /v1/chat/completions; "
          "no UI; no agent routes; no files written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
