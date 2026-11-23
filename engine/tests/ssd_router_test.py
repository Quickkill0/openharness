#!/usr/bin/env python3
"""Check that every repeated SSD option reaches a router child.

A router (--models-dir) rebuilds the child's command line from a map of
options. Two --ssd-set values must both arrive: the router's /models listing
shows the child's arguments, and the child's budget receipt shows it applied
the first value as well as the last.
"""

import argparse
import json
import os
import re
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


def get_json(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def post_json(base: str, path: str, body: dict) -> int:
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def wait(what: str, seconds: float, check):
    deadline = time.time() + seconds
    while True:
        result = check()
        if result:
            return result
        if time.time() > deadline:
            raise AssertionError(f"{what}: not reached in {seconds:.0f} s")
        time.sleep(0.25)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="the synthetic MoE fixture")
    args = ap.parse_args()

    settings = ["engine.ram_budget_mib=4096", "engine.expert_cache_mib=0"]
    env = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_ARG_")}
    env["CUDA_VISIBLE_DEVICES"] = ""

    # A child that is still on its way out can hold the directory on Windows.
    with tempfile.TemporaryDirectory(prefix="oh-ssd-router-", ignore_cleanup_errors=True) as sandbox:
        models = os.path.join(sandbox, "models")
        os.makedirs(models)
        shutil.copy(args.model, models)
        name = os.path.splitext(os.path.basename(args.model))[0]
        env["APPDATA"] = env["XDG_CONFIG_HOME"] = os.path.join(sandbox, "config")

        port = free_port()
        base = f"http://127.0.0.1:{port}"
        command = [args.server, "--host", "127.0.0.1", "--port", str(port), "-t", "1", "--models-dir", models,
                   "-c", "1024", "-ngl", "0", "-v"]
        for setting in settings:
            command += ["--ssd-set", setting]
        log_path = os.path.join(sandbox, "router.log")
        with open(log_path, "wb") as log:
            proc = subprocess.Popen(command, cwd=sandbox, env=env, stdout=log, stderr=subprocess.STDOUT)
            failed = False
            try:
                def healthy():
                    assert proc.poll() is None, f"router exited early with {proc.returncode}"
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=2) as r:
                            return r.status == 200
                    except (urllib.error.URLError, ConnectionError, socket.timeout):
                        return False

                wait("router ready", 60, healthy)
                status = post_json(base, "/models/load", {"model": name})
                assert status == 200, f"POST /models/load returned {status}"

                def loaded():
                    for model in get_json(base, "/models")["data"]:
                        if model["id"] == name and model["status"]["value"] == "loaded":
                            return model
                    return None

                model = wait("child loaded", 120, loaded)
                child = model["status"]["args"]
                print("child args:", " ".join(child))
                for setting in settings:
                    at = [i for i, a in enumerate(child) if a == setting]
                    assert len(at) == 1 and child[at[0] - 1] == "--ssd-set", f"{setting} did not reach the child: {child}"

                # Stop the child through the router: on Windows terminating the router alone leaves it to notice later.
                status = post_json(base, "/models/unload", {"model": name})
                assert status == 200, f"POST /models/unload returned {status}"

                def unloaded():
                    return any(m["id"] == name and m["status"]["value"] == "unloaded"
                               for m in get_json(base, "/models")["data"])

                wait("child unloaded", 60, unloaded)
            except Exception:
                failed = True
                raise
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                if failed:
                    with open(log_path, "rb") as f:
                        sys.stderr.write(f.read().decode("utf-8", "replace")[-6000:])

        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "replace")
        receipts = re.findall(r"\[oh-ssd\] receipt \(MiB\):.*", text)
        assert receipts, f"the child printed no receipt:\n{text[-4000:]}"
        receipt = receipts[-1]
        print(receipt)
        # The first --ssd-set sets the budget, the last empties the cache: losing either one changes this line.
        assert "budget=4096 " in receipt, f"the first --ssd-set was lost: {receipt}"
        assert "expert_cache=0" in receipt, f"the last --ssd-set was lost: {receipt}"
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
