#!/usr/bin/env python3
"""Check that openharness-engine-server never reaches the network.

Every remote model source (model URL, Hugging Face repo, Docker repo, mmproj
URL, draft repo, their LLAMA_ARG_* environment variables, and the router's
POST /models) must be refused before any request is made. The agent features
upstream's server carries (MCP CORS proxy, MCP servers, built-in tools and
their container/SSH runtime, the llama-connect tunnel) are not part of this
build: each must be refused whether it comes as a flag, an environment
variable, the user config file or a router preset, and the /cors-proxy and
/tools routes must not exist. A local HTTP fixture stands in for every
endpoint: it is the Hugging Face endpoint (HF_ENDPOINT / MODEL_ENDPOINT), the
target of every URL, and the HTTP(S) proxy, so any attempted connection shows
up as a request. Each case asserts the fixture saw zero requests and the
server gave the OpenHarness refusal. Remote RPC devices (--rpc, its variable,
config.ini and presets) dial raw TCP rather than HTTP, so those cases point at
a plain TCP listener that counts every connection, and every case asserts it
saw zero connections as well.
"""

import argparse
import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request

REFUSALS = (
    "remote downloads are disabled in openharness-engine",
    "outbound network access is disabled in openharness-engine",
)
AGENT_REFUSAL = ("is not available in openharness-engine",)


class Fixture:
    """Counts every request it receives, whatever the method."""

    def __init__(self):
        self.requests = []
        lock = threading.Lock()
        fixture = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def _any(self):
                with lock:
                    fixture.requests.append(f"{self.command} {self.path}")
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_HEAD = do_POST = do_PUT = do_CONNECT = do_OPTIONS = _any

            def log_message(self, format, *args):  # noqa: A002 - quiet the fixture
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def take(self):
        seen, self.requests = self.requests, []
        return seen


class TcpFixture:
    """Counts every TCP connection it accepts (stands in for an RPC server)."""

    def __init__(self):
        self.connections = []
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.addr = f"127.0.0.1:{self.sock.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while True:
            try:
                conn, peer = self.sock.accept()
            except OSError:
                return
            self.connections.append(f"tcp from {peer[0]}:{peer[1]}")
            conn.close()

    def take(self):
        seen, self.connections = self.connections, []
        return seen


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def base_env(fixture: Fixture, sandbox: str) -> dict:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("LLAMA_ARG_") and k.upper() not in
           ("HF_TOKEN", "HF_ENDPOINT", "MODEL_ENDPOINT", "LLAMA_CACHE", "NO_PROXY",
            "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}
    env.update({
        "CUDA_VISIBLE_DEVICES": "",
        "HF_ENDPOINT": fixture.url,
        "MODEL_ENDPOINT": fixture.url + "/",
        "HTTP_PROXY": fixture.url, "http_proxy": fixture.url,
        "HTTPS_PROXY": fixture.url, "https_proxy": fixture.url,
        "ALL_PROXY": fixture.url, "all_proxy": fixture.url,
        "LLAMA_CACHE": os.path.join(sandbox, "cache"),
        # keep a real user config.ini out of the test; the config case writes one here
        "APPDATA": os.path.join(sandbox, "config"), "XDG_CONFIG_HOME": os.path.join(sandbox, "config"),
    })
    return env


def run_refused(server: str, args: list, env: dict, cwd: str, refusals=REFUSALS) -> str:
    try:
        proc = subprocess.run([server, *args], cwd=cwd, env=env, timeout=60,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired as e:
        raise AssertionError(f"{args}: server did not exit within 60 s") from e
    out = proc.stdout.decode("utf-8", "replace")
    assert proc.returncode != 0, f"{args}: expected a nonzero exit\n{out[-3000:]}"
    assert any(r in out for r in refusals), f"{args}: no OpenHarness refusal in output\n{out[-3000:]}"
    return out


def request(base: str, method: str, path: str, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def router_case(server: str, env: dict, cwd: str, fixture_url: str) -> None:
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([server, "--host", "127.0.0.1", "--port", str(port), "-t", "1"],
                            cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + 60
        while True:
            if proc.poll() is not None:
                raise AssertionError(f"router exited early with {proc.returncode}")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except (urllib.error.URLError, ConnectionError, socket.timeout):
                pass
            if time.time() > deadline:
                raise AssertionError("router did not become ready in 60 s")
            time.sleep(0.25)
        status, body = request(base, "POST", "/models", {"model": "review/missing"})
        assert status >= 400, (status, body)
        assert any(r in body for r in REFUSALS), (status, body)
        # The agent routes are not registered at all: plain 404s, nothing proxied.
        for method, path, payload in (
            ("GET", f"/cors-proxy?url={fixture_url}/cors", None),
            ("POST", f"/cors-proxy?url={fixture_url}/cors", {"x": 1}),
            ("GET", "/tools", None),
            ("POST", "/tools", {"tool": "exec_shell_command", "params": {"command": f"curl {fixture_url}/tool"}}),
        ):
            status, text = request(base, method, path, payload)
            assert status == 404, (method, path, status, text[:200])
    except Exception:
        proc.kill()
        sys.stderr.write(proc.communicate(timeout=30)[0].decode("utf-8", "replace")[-4000:])
        raise
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.communicate()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="a valid local GGUF, so later model sources are reached")
    args = ap.parse_args()
    model = os.path.abspath(args.model)

    fixture = Fixture()
    tcp = TcpFixture()

    def take_all():
        return fixture.take() + tcp.take()
    sandbox = tempfile.mkdtemp(prefix="oh-engine-noremote-")
    env = base_env(fixture, sandbox)
    common = ["--host", "127.0.0.1", "--port", str(free_port()), "-c", "256", "-t", "1", "-ngl", "0"]
    u = fixture.url

    cases = [
        ("model URL (-mu)", ["-mu", f"{u}/x.gguf"], {}),
        ("Hugging Face repo (-hf)", ["-hf", "review/missing"], {}),
        ("Hugging Face repo with quant (-hf)", ["-hf", "review/missing:Q4_K_M"], {}),
        ("Docker repo (-dr)", ["-dr", "ai/smollm2"], {}),
        ("mmproj URL (-mmu)", ["-m", model, "-mmu", f"{u}/mmproj.gguf"], {}),
        ("draft repo (-hfd)", ["-m", model, "-hfd", "review/draft"], {}),
        ("LLAMA_ARG_MODEL_URL", [], {"LLAMA_ARG_MODEL_URL": f"{u}/env.gguf"}),
        ("LLAMA_ARG_HF_REPO", [], {"LLAMA_ARG_HF_REPO": "review/env"}),
        ("LLAMA_ARG_SPEC_DRAFT_HF_REPO", ["-m", model], {"LLAMA_ARG_SPEC_DRAFT_HF_REPO": "review/env-draft"}),
    ]
    mcp_json = json.dumps({"mcpServers": {"fixture": {"url": f"{u}/mcp"}}})
    mcp_config = os.path.join(sandbox, "mcp.json")
    with open(mcp_config, "w", encoding="utf-8") as f:
        f.write(mcp_json)
    agent_cases = [
        ("--ui-mcp-proxy", ["-m", model, "--ui-mcp-proxy"], {}),
        ("--webui-mcp-proxy", ["-m", model, "--webui-mcp-proxy"], {}),
        ("--agent", ["-m", model, "--agent"], {}),
        ("--tools all", ["-m", model, "--tools", "all"], {}),
        ("--tools exec_shell_command", ["-m", model, "--tools", "exec_shell_command"], {}),
        ("--tools read_file", ["-m", model, "--tools", "read_file"], {}),
        ("--tools-runtime", ["-m", model, "--tools-runtime", "ssh:review@127.0.0.1"], {}),
        ("--mcp-servers-json", ["-m", model, "--mcp-servers-json", mcp_json], {}),
        ("--mcp-servers-config", ["-m", model, "--mcp-servers-config", mcp_config], {}),
        ("--connect", ["-m", model, "--connect"], {}),
        ("--connect-code", ["-m", model, "--connect-code", "0" * 40], {}),
        ("router --tools", ["--tools", "all"], {}),
        ("LLAMA_ARG_UI_MCP_PROXY", ["-m", model], {"LLAMA_ARG_UI_MCP_PROXY": "1"}),
        ("LLAMA_ARG_AGENT", ["-m", model], {"LLAMA_ARG_AGENT": "1"}),
        ("LLAMA_ARG_TOOLS", ["-m", model], {"LLAMA_ARG_TOOLS": "all"}),
        ("LLAMA_ARG_TOOLS_RUNTIME", ["-m", model], {"LLAMA_ARG_TOOLS_RUNTIME": "docker:review"}),
        ("LLAMA_ARG_MCP_SERVERS_JSON", ["-m", model], {"LLAMA_ARG_MCP_SERVERS_JSON": mcp_json}),
        ("LLAMA_ARG_MCP_SERVERS_CONFIG", ["-m", model], {"LLAMA_ARG_MCP_SERVERS_CONFIG": mcp_config}),
        ("LLAMA_ARG_CONNECT", ["-m", model], {"LLAMA_ARG_CONNECT": "1"}),
        ("--rpc", ["-m", model, "--rpc", tcp.addr], {}),
        ("--rpc list", ["-m", model, "--rpc", f"{tcp.addr},localhost:{tcp.addr.split(':')[1]}"], {}),
        ("router --rpc", ["--rpc", tcp.addr], {}),
        ("LLAMA_ARG_RPC", ["-m", model], {"LLAMA_ARG_RPC": tcp.addr}),
    ]
    # Router presets: options in a model section, and in the global section.
    for i, (label, ini) in enumerate((
        ("router preset tools", f"[review]\nmodel = {model}\ntools = all\n"),
        ("router preset ui-mcp-proxy (global)", f"[*]\nui-mcp-proxy = true\n\n[review]\nmodel = {model}\n"),
        ("router preset mcp-servers-config", f"[review]\nmodel = {model}\nmcp-servers-config = {mcp_config}\n"),
        ("router preset rpc", f"[review]\nmodel = {model}\nrpc = {tcp.addr}\n"),
        ("router preset rpc (global)", f"[*]\nrpc = {tcp.addr}\n\n[review]\nmodel = {model}\n"),
    )):
        path = os.path.join(sandbox, f"preset-{i}.ini")
        with open(path, "w", encoding="utf-8") as f:
            f.write(ini)
        agent_cases.append((label, ["--models-preset", path], {}))
    failures = []
    for name, extra, extra_env in cases:
        try:
            run_refused(args.server, [*extra, *common], {**env, **extra_env}, sandbox)
            seen = take_all()
            assert not seen, f"fixture received requests: {seen}"
            print(f"ok: {name}: refused, 0 requests")
        except AssertionError as e:
            take_all()
            failures.append(f"{name}: {e}")

    for name, extra, extra_env in agent_cases:
        try:
            run_refused(args.server, [*extra, *common], {**env, **extra_env}, sandbox, AGENT_REFUSAL)
            seen = take_all()
            assert not seen, f"fixture received requests: {seen}"
            print(f"ok: {name}: refused, 0 requests")
        except AssertionError as e:
            take_all()
            failures.append(f"{name}: {e}")

    # The user config file (%APPDATA%\llama.cpp\config.ini, or
    # $XDG_CONFIG_HOME/llama.cpp/config.ini) is applied before the check too.
    config_dir = os.path.join(sandbox, "config", "llama.cpp")
    for key, value in (("tools", "all"), ("rpc", tcp.addr)):
        os.makedirs(config_dir, exist_ok=True)
        try:
            with open(os.path.join(config_dir, "config.ini"), "w", encoding="utf-8") as f:
                f.write(f"{key} = {value}\n")
            run_refused(args.server, ["-m", model, *common], env, sandbox, AGENT_REFUSAL)
            seen = take_all()
            assert not seen, f"fixture received requests: {seen}"
            print(f"ok: config.ini {key}: refused, 0 requests")
        except AssertionError as e:
            take_all()
            failures.append(f"config.ini {key}: {e}")
        finally:
            shutil.rmtree(config_dir)

    try:
        router_case(args.server, env, sandbox, fixture.url)
        seen = take_all()
        assert not seen, f"fixture received requests: {seen}"
        print("ok: router POST /models refused; /cors-proxy and /tools are 404; 0 requests")
    except AssertionError as e:
        failures.append(f"router: {e}")

    fixture.server.shutdown()
    tcp.sock.close()
    if failures:
        sys.stderr.write("\n\n".join(failures) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
