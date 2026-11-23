"""Worker server: request dispatch + single-job executor.

Jobs run on one background thread (GPU work is serial anyway; cross-job
scheduling is the Rust orchestrator's job). The main thread reads stdin and
stays responsive for ping/cancel/shutdown while a job runs.
"""

from __future__ import annotations

import platform
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import __version__, pipelines, protocol


class JobCancelled(Exception):
    pass


def _preload_native() -> None:
    """Import BLAS/OpenMP-heavy native libraries on the main thread up front.

    On Windows, a native extension that spawns threads during its DLL
    initialization — numpy/OpenBLAS, scipy's compiled kernels, scikit-learn,
    torch's OpenMP runtime — deadlocks if its *first* import happens on a
    non-main thread: the threads it spawns block on the loader lock the
    importing thread still holds (numpy#24028) and the job hangs forever
    mid-`import`, at 0% CPU, with no error. Pipelines import these lazily on
    the single job thread, so the first job to touch a cold one would hang the
    worker. Confirmed via py-spy: any transformers model that calls
    `.generate()` pulls `transformers.generation → sklearn → scipy.stats →
    scipy.interpolate`, whose native `_fitpack` extension deadlocked on the
    job thread (this is what made VibeVoice "stick" while loading).

    Warming them here, on the main thread, loads every problematic native DLL
    process-wide before any job runs, so each later lazy pipeline import is a
    cheap sys.modules hit on the job thread — no cold native load, no
    loader-lock race. This is deliberately generic (the runtime, not any one
    model): we warm the shared numerical native stack via its umbrella
    packages, and importing one pulls its whole native subtree on the main
    thread — `import sklearn` warms scipy.stats/interpolate/special/linalg +
    sklearn's own kernels, `import scipy.stats` warms scipy.interpolate/special.
    So the four imports below cover the entire numpy/scipy/sklearn/torch
    surface every pipeline transitively touches; it fixes the deadlock *class*,
    not just the model that exposed it.

    Each import is best-effort and order-sensitive (numpy first, torch last):
    a package absent from a given venv variant (CPU vs CUDA) just surfaces
    later as a normal per-job error, never a startup crash. `import torch`
    loads its DLLs without initializing CUDA, so this grabs no VRAM. (Adding a
    pipeline that pulls a *new* thread-spawning native lib outside this stack
    means adding it here — see numpy#24028.)
    """
    warmed: list[str] = []
    skipped: list[str] = []
    for mod in ("numpy", "scipy.stats", "sklearn", "torch"):
        try:
            __import__(mod)
            warmed.append(mod)
        except Exception as e:
            # A missing/broken package surfaces as a per-job error with a real
            # traceback instead of killing the worker at startup.
            skipped.append(f"{mod} ({type(e).__name__})")
    protocol.log(
        "info",
        "native warmup: warmed=[%s]%s"
        % (", ".join(warmed), f" skipped=[{', '.join(skipped)}]" if skipped else ""),
    )


class JobContext:
    """Handed to pipeline `run` functions: progress + cancellation + paths."""

    def __init__(self, job_id: str, output_dir: Path, cancel_event: threading.Event):
        self.job_id = job_id
        self.output_dir = output_dir
        self._cancel_event = cancel_event

    def progress(
        self,
        pct: float,
        stage: str,
        step: int | None = None,
        total_steps: int | None = None,
    ) -> None:
        protocol.emit(
            "job_progress",
            job_id=self.job_id,
            pct=round(min(max(pct, 0.0), 100.0), 2),
            stage=stage,
            step=step,
            total_steps=total_steps,
        )

    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancelled():
            raise JobCancelled()


class Server:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="oh-job"
        )
        self._cancel_events: dict[str, threading.Event] = {}
        self._cancel_lock = threading.Lock()
        self._shutdown = threading.Event()

    # ── dispatch ─────────────────────────────────────────────────────────

    def handle(self, req: dict) -> None:
        req_id = str(req.get("id", ""))
        op = req["op"]
        try:
            if op == "ping":
                protocol.respond_ok(
                    req_id,
                    {
                        "pong": True,
                        "version": __version__,
                        "python": platform.python_version(),
                    },
                )
            elif op == "capabilities":
                protocol.respond_ok(req_id, {"pipelines": pipelines.capabilities()})
            elif op == "submit":
                self._submit(req_id, req)
            elif op == "preload":
                self._submit(req_id, req, preload=True)
            elif op == "cancel":
                self._cancel(req_id, str(req.get("job_id", "")))
            elif op == "unload":
                from .pipelines import _runtime

                before = _runtime.vram_used_mb()
                done = pipelines.unload(req.get("task"))
                after = _runtime.vram_used_mb()
                protocol.log(
                    "info",
                    f"unload task={req.get('task')!r}: unloaded={done} "
                    f"cuda_allocated_mb {before}->{after}",
                )
                protocol.respond_ok(req_id, {"unloaded": done})
            elif op == "shutdown":
                protocol.respond_ok(req_id, {"bye": True})
                self._shutdown.set()
            else:
                protocol.respond_err(req_id, f"unknown op: {op}")
        except Exception as e:  # dispatch must never kill the worker
            protocol.log("error", traceback.format_exc())
            protocol.respond_err(req_id, str(e))

    def _submit(self, req_id: str, req: dict, preload: bool = False) -> None:
        task = req.get("task")
        job_id = req.get("job_id")
        params = req.get("params") or {}
        op = "preload" if preload else "submit"
        if not task or not job_id:
            protocol.respond_err(req_id, f"{op} requires 'task' and 'job_id'")
            return
        if task not in pipelines.PIPELINES:
            protocol.respond_err(req_id, f"unknown task: {task}")
            return
        cancel_event = threading.Event()
        with self._cancel_lock:
            self._cancel_events[job_id] = cancel_event
        self._executor.submit(
            self._run_job, task, str(job_id), params, cancel_event, preload
        )
        protocol.respond_ok(req_id, {"job_id": job_id, "accepted": True})

    def _cancel(self, req_id: str, job_id: str) -> None:
        with self._cancel_lock:
            ev = self._cancel_events.get(job_id)
        if ev is None:
            protocol.respond_err(req_id, f"unknown job: {job_id}")
            return
        ev.set()
        protocol.respond_ok(req_id, {"job_id": job_id, "cancelling": True})

    # ── job execution (background thread) ────────────────────────────────

    def _run_job(
        self,
        task: str,
        job_id: str,
        params: dict,
        cancel_event: threading.Event,
        preload: bool = False,
    ) -> None:
        ctx = JobContext(job_id, self.output_dir, cancel_event)
        try:
            if cancel_event.is_set():
                raise JobCancelled()
            pipeline = pipelines.get_pipeline(task)
            if preload:
                fn = getattr(pipeline, "preload", None)
                if not callable(fn):
                    raise RuntimeError(
                        f"pipeline '{task}' does not support preloading"
                    )
                outputs, meta = [], dict(fn(params, ctx) or {})
                meta.setdefault("preload", True)
            else:
                outputs, meta = pipeline.run(params, ctx)
            protocol.emit(
                "job_done", job_id=job_id, outputs=outputs, meta=meta, task=task
            )
        except JobCancelled:
            protocol.emit("job_error", job_id=job_id, error="cancelled")
        except BaseException as e:
            # BaseException, not Exception: libraries that shell out can call
            # sys.exit() on failure (spacy's model auto-download does), and a
            # SystemExit would otherwise vanish into the executor's Future,
            # leaving the job stuck "running" forever.
            protocol.log("error", traceback.format_exc())
            protocol.emit(
                "job_error", job_id=job_id, error=str(e) or type(e).__name__
            )
        finally:
            with self._cancel_lock:
                self._cancel_events.pop(job_id, None)

    # ── main loop ────────────────────────────────────────────────────────

    def run(self) -> None:
        _preload_native()
        protocol.log(
            "info",
            f"openharness worker {__version__} ready "
            f"(python {platform.python_version()})",
        )
        for req in protocol.read_requests(sys.stdin):
            self.handle(req)
            if self._shutdown.is_set():
                break
        # stdin closed or shutdown requested: cancel anything in flight and
        # let the executor drain.
        with self._cancel_lock:
            for ev in self._cancel_events.values():
                ev.set()
        self._executor.shutdown(wait=True, cancel_futures=True)
        pipelines.unload(None)
