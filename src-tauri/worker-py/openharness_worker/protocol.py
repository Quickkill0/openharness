"""JSON-Lines stdio protocol.

Requests (Rust -> worker), one JSON object per line:
    {"id": "<uuid>", "op": "ping"}
    {"id": "<uuid>", "op": "capabilities"}
    {"id": "<uuid>", "op": "submit", "task": "text_to_image",
     "job_id": "<uuid>", "params": {...}}
    {"id": "<uuid>", "op": "cancel", "job_id": "<uuid>"}
    {"id": "<uuid>", "op": "unload", "task": "text_to_image" | null}
    {"id": "<uuid>", "op": "shutdown"}

Responses (worker -> Rust), correlated by "id":
    {"id": "<uuid>", "ok": true, "result": {...}}
    {"id": "<uuid>", "ok": false, "error": "..."}

Events (worker -> Rust), unsolicited, distinguished by "event":
    {"event": "job_progress", "job_id": ..., "pct": 0-100, "stage": "...",
     "step": int | null, "total_steps": int | null}
    {"event": "job_done", "job_id": ..., "outputs": ["<abs path>", ...],
     "meta": {...}}
    {"event": "job_error", "job_id": ..., "error": "..."}
    {"event": "log", "level": "info" | "warn" | "error", "msg": "..."}

Binary outputs are never sent over the pipe — pipelines write files into the
output dir and return absolute paths.
"""

from __future__ import annotations

import json
import sys
import threading
from typing import Any, Iterator

_write_lock = threading.Lock()


def write_message(obj: dict[str, Any]) -> None:
    """Serialize one message as a single line on stdout. Thread-safe."""
    line = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    with _write_lock:
        sys.stdout.write(line + "\n")
        sys.stdout.flush()


def respond_ok(req_id: str, result: dict[str, Any]) -> None:
    write_message({"id": req_id, "ok": True, "result": result})


def respond_err(req_id: str, error: str) -> None:
    write_message({"id": req_id, "ok": False, "error": error})


def emit(event: str, **fields: Any) -> None:
    write_message({"event": event, **fields})


def log(level: str, msg: str) -> None:
    emit("log", level=level, msg=msg)


def read_requests(stream) -> Iterator[dict[str, Any]]:
    """Yield parsed request objects from a line stream. Bad lines are
    reported as log events and skipped — a malformed line must never kill
    the worker."""
    for raw in stream:
        raw = raw.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            log("error", f"unparseable request line: {e}")
            continue
        if not isinstance(obj, dict) or "op" not in obj:
            log("error", "request missing 'op' field")
            continue
        yield obj
