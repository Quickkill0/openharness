//! Worker process lifecycle + request/response plumbing.
//!
//! Spawns `python -m openharness_worker` from the venv, owns three tasks:
//! a stdin writer (mpsc of lines), a stdout reader (routes responses to
//! pending oneshots by id, broadcasts events), and a stderr tail into a
//! ring buffer (mirroring `LlamaServer.log_buf`).

use std::collections::{HashMap, HashSet, VecDeque};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::process::{Child, Command};
use tokio::sync::{broadcast, mpsc, oneshot, Mutex};
use uuid::Uuid;

use crate::error::AppError;
use crate::worker::protocol::{
    encode_request, parse_line, Incoming, WorkerEvent, WorkerRequest,
};

type Pending = Arc<Mutex<HashMap<String, oneshot::Sender<Result<serde_json::Value, String>>>>>;

/// Error reported for every job a worker process took and never finished.
pub const WORKER_EXITED: &str = "worker process exited";

/// Jobs sent to this worker process (submits and preloads) that have not
/// reached `job_done`/`job_error` yet. When the process dies each gets a
/// terminal `job_error`, so its DB row, its waiter and the UI's activity
/// locks clear through the normal path instead of staying "running" forever.
/// Only jobs sent to this process are here: the qwen-tts sidecar and CLI post
/// their own events on the shared bus and are never touched.
#[derive(Clone, Default)]
struct InFlightJobs(Arc<Mutex<HashSet<String>>>);

impl InFlightJobs {
    async fn add(&self, job_id: &str) {
        self.0.lock().await.insert(job_id.to_string());
    }
    async fn remove(&self, job_id: &str) {
        self.0.lock().await.remove(job_id);
    }
    /// Drop a job once the worker reported its terminal event.
    async fn observe(&self, ev: &WorkerEvent) {
        if let WorkerEvent::JobDone { job_id, .. } | WorkerEvent::JobError { job_id, .. } = ev {
            self.remove(job_id).await;
        }
    }
    /// The terminal events owed when the process exits.
    async fn drain_exit_errors(&self) -> Vec<WorkerEvent> {
        let mut ids: Vec<String> = self.0.lock().await.drain().collect();
        ids.sort();
        ids.into_iter()
            .map(|job_id| WorkerEvent::JobError { job_id, error: WORKER_EXITED.into() })
            .collect()
    }
}

/// Stdout reader: routes responses to pending requests by id and broadcasts
/// events. On EOF (the process died or was stopped) it fails every pending
/// request and reports a `job_error` for every job the process still held.
async fn read_stdout<R>(reader: R, pending: Pending, events: broadcast::Sender<WorkerEvent>, jobs: InFlightJobs)
where
    R: tokio::io::AsyncBufRead + Unpin,
{
    use tokio::io::AsyncBufReadExt;
    let mut lines = reader.lines();
    while let Ok(Some(line)) = lines.next_line().await {
        match parse_line(&line) {
            Some(Incoming::Response(resp)) => {
                let sender = pending.lock().await.remove(&resp.id);
                if let Some(tx) = sender {
                    let payload = if resp.ok {
                        Ok(resp.result.unwrap_or(serde_json::Value::Null))
                    } else {
                        Err(resp.error.unwrap_or_else(|| "unknown worker error".into()))
                    };
                    let _ = tx.send(payload);
                } else {
                    tracing::warn!(target: "worker", "response for unknown request id {}", resp.id);
                }
            }
            Some(Incoming::Event(ev)) => {
                if let WorkerEvent::Log { level, msg } = &ev {
                    match level.as_str() {
                        "error" => tracing::error!(target: "worker", "{msg}"),
                        "warn" => tracing::warn!(target: "worker", "{msg}"),
                        _ => tracing::info!(target: "worker", "{msg}"),
                    }
                }
                jobs.observe(&ev).await;
                let _ = events.send(ev);
            }
            None => {
                tracing::warn!(target: "worker", "unparseable line: {line}");
            }
        }
    }
    // stdout closed: the process died — fail everything in flight.
    for ev in jobs.drain_exit_errors().await {
        let _ = events.send(ev);
    }
    let mut p = pending.lock().await;
    for (_, tx) in p.drain() {
        let _ = tx.send(Err(WORKER_EXITED.into()));
    }
}

pub struct WorkerProcess {
    child: Child,
    stdin_tx: mpsc::Sender<String>,
    pending: Pending,
    jobs: InFlightJobs,
    events: broadcast::Sender<WorkerEvent>,
    pub log_buf: Arc<Mutex<VecDeque<String>>>,
}

impl WorkerProcess {
    /// Spawn the worker. `python` is the venv interpreter, `worker_src` the
    /// directory containing the `openharness_worker` package, `output_dir`
    /// where pipelines write generated files.
    pub async fn start(
        python: &Path,
        worker_src: &Path,
        output_dir: &Path,
    ) -> Result<Self, AppError> {
        use std::process::Stdio;

        let mut cmd = Command::new(python);
        cmd.arg("-m")
            .arg("openharness_worker")
            .arg("--output-dir")
            .arg(output_dir)
            .current_dir(worker_src)
            .env("PYTHONPATH", worker_src)
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        #[cfg(windows)]
        {
            #[allow(unused_imports)]
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| AppError::WorkerSpawnFailed(e.to_string()))?;

        // Die with the app even when Drop never runs (hard kill) — winjob.rs.
        #[cfg(windows)]
        crate::winjob::tie_to_app_lifetime(&child, "python worker");

        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (events_tx, _) = broadcast::channel::<WorkerEvent>(256);
        let log_buf: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(1000)));

        // ── stdin writer ────────────────────────────────────────────────
        let (stdin_tx, mut stdin_rx) = mpsc::channel::<String>(64);
        let mut stdin = child
            .stdin
            .take()
            .ok_or_else(|| AppError::WorkerSpawnFailed("no stdin pipe".into()))?;
        tokio::spawn(async move {
            while let Some(line) = stdin_rx.recv().await {
                if stdin.write_all(line.as_bytes()).await.is_err() {
                    break;
                }
                if stdin.write_all(b"\n").await.is_err() {
                    break;
                }
                if stdin.flush().await.is_err() {
                    break;
                }
            }
        });

        // ── stdout reader: responses + events ──────────────────────────
        let jobs = InFlightJobs::default();
        if let Some(stdout) = child.stdout.take() {
            let reader = tokio::io::BufReader::new(stdout);
            tokio::spawn(read_stdout(reader, pending.clone(), events_tx.clone(), jobs.clone()));
        }

        // ── stderr tail → ring buffer ───────────────────────────────────
        if let Some(stderr) = child.stderr.take() {
            let buf = log_buf.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "worker.err", "{}", line);
                    let mut b = buf.lock().await;
                    if b.len() == 1000 {
                        b.pop_front();
                    }
                    b.push_back(line);
                }
            });
        }

        Ok(Self {
            child,
            stdin_tx,
            pending,
            jobs,
            events: events_tx,
            log_buf,
        })
    }

    pub fn subscribe_events(&self) -> broadcast::Receiver<WorkerEvent> {
        self.events.subscribe()
    }

    /// Send one request and await its response.
    pub async fn request(
        &self,
        body: &WorkerRequest,
        timeout: Duration,
    ) -> Result<serde_json::Value, AppError> {
        let id = Uuid::new_v4().to_string();
        let line = encode_request(&id, body)?;

        let (tx, rx) = oneshot::channel();
        self.pending.lock().await.insert(id.clone(), tx);
        // Tracked before the line goes out, so no event for the job can
        // arrive ahead of its entry.
        let job_id = match body {
            WorkerRequest::Submit { job_id, .. } | WorkerRequest::Preload { job_id, .. } => {
                self.jobs.add(job_id).await;
                Some(job_id.as_str())
            }
            _ => None,
        };

        if self.stdin_tx.send(line).await.is_err() {
            self.pending.lock().await.remove(&id);
            if let Some(j) = job_id {
                self.jobs.remove(j).await;
            }
            return Err(AppError::WorkerUnavailable);
        }

        let result = match tokio::time::timeout(timeout, rx).await {
            Ok(Ok(Ok(value))) => Ok(value),
            Ok(Ok(Err(msg))) => Err(AppError::WorkerRequest(msg)),
            Ok(Err(_)) => Err(AppError::WorkerUnavailable),
            Err(_) => {
                self.pending.lock().await.remove(&id);
                Err(AppError::WorkerRequestTimeout)
            }
        };
        // A refused or unanswered submit never runs here; its caller closes
        // the job row itself.
        if let (Err(_), Some(j)) = (&result, job_id) {
            self.jobs.remove(j).await;
        }
        result
    }

    pub async fn ping(&self) -> Result<serde_json::Value, AppError> {
        self.request(&WorkerRequest::Ping, Duration::from_secs(10)).await
    }

    /// Graceful shutdown: ask politely, then wait, then kill.
    pub async fn stop(mut self) -> Result<(), AppError> {
        let _ = self
            .request(&WorkerRequest::Shutdown, Duration::from_secs(5))
            .await;
        match tokio::time::timeout(Duration::from_secs(10), self.child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                let _ = self.child.start_kill();
                let _ =
                    tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await;
            }
        }
        Ok(())
    }

    /// True while the child is still running.
    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        // Best-effort, mirroring LlamaServer: graceful stop() is preferred.
        let _ = self.child.start_kill();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::worker_src_dir;

    fn system_python() -> Option<std::path::PathBuf> {
        let name = if cfg!(windows) { "python.exe" } else { "python3" };
        std::env::var_os("PATH").and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|p| p.join(name))
                .find(|p| p.exists())
        })
    }

    /// The worker took two jobs, finished one, then its stdout closed (crash
    /// or stop). The unfinished job gets exactly one terminal `job_error`;
    /// the finished one gets none, and pending requests fail.
    #[tokio::test]
    async fn worker_exit_reports_error_for_each_unfinished_job() {
        let jobs = InFlightJobs::default();
        jobs.add("done-job").await;
        jobs.add("running-job").await;
        let pending: Pending = Arc::new(Mutex::new(HashMap::new()));
        let (req_tx, req_rx) = oneshot::channel();
        pending.lock().await.insert("r1".into(), req_tx);
        let (events, mut rx) = broadcast::channel::<WorkerEvent>(16);

        let stdout: &[u8] = b"{\"event\":\"job_progress\",\"job_id\":\"running-job\",\"pct\":40,\"stage\":\"synthesizing\"}\n\
{\"event\":\"job_done\",\"job_id\":\"done-job\",\"outputs\":[\"a.wav\"],\"task\":\"text_to_speech\"}\n";
        read_stdout(tokio::io::BufReader::new(stdout), pending, events, jobs.clone()).await;

        let mut seen = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            seen.push(ev);
        }
        assert_eq!(seen.len(), 3, "{seen:?}");
        assert!(matches!(&seen[1], WorkerEvent::JobDone { job_id, .. } if job_id == "done-job"));
        assert!(matches!(
            &seen[2],
            WorkerEvent::JobError { job_id, error } if job_id == "running-job" && error == WORKER_EXITED
        ));
        assert_eq!(req_rx.await.unwrap(), Err(WORKER_EXITED.to_string()));
        assert!(jobs.drain_exit_errors().await.is_empty(), "nothing left to report twice");
    }

    /// Full round-trip against the real Python worker using the system
    /// interpreter (the worker core is stdlib-only, so no venv is needed).
    /// Skips when python isn't on PATH.
    #[tokio::test]
    async fn live_worker_ping_echo_cancel() {
        let Some(python) = system_python() else {
            eprintln!("skipping: no python on PATH");
            return;
        };
        let src = worker_src_dir(None).expect("worker-py source dir");
        let out_dir = std::env::temp_dir().join(format!("oh-worker-it-{}", std::process::id()));
        std::fs::create_dir_all(&out_dir).unwrap();

        let proc = WorkerProcess::start(&python, &src, &out_dir)
            .await
            .expect("worker spawn");

        // ping
        let pong = proc.ping().await.expect("ping");
        assert_eq!(pong["pong"], true);
        assert_eq!(pong["version"], crate::worker::WORKER_VERSION);

        // capabilities
        let caps = proc
            .request(&WorkerRequest::Capabilities, Duration::from_secs(10))
            .await
            .expect("capabilities");
        let tasks: Vec<&str> = caps["pipelines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p["task"].as_str().unwrap())
            .collect();
        assert!(tasks.contains(&"echo"));
        assert!(tasks.contains(&"text_to_image"));

        // echo job: progress → done, file on disk
        let mut events = proc.subscribe_events();
        let accepted = proc
            .request(
                &WorkerRequest::Submit {
                    task: "echo".into(),
                    job_id: "it-job-1".into(),
                    params: serde_json::json!({"steps": 3, "delay": 0.01, "text": "live"}),
                },
                Duration::from_secs(10),
            )
            .await
            .expect("submit");
        assert_eq!(accepted["accepted"], true);

        let mut saw_progress = false;
        let outputs = loop {
            let ev = tokio::time::timeout(Duration::from_secs(15), events.recv())
                .await
                .expect("event timeout")
                .expect("event recv");
            match ev {
                WorkerEvent::JobProgress { job_id, .. } if job_id == "it-job-1" => {
                    saw_progress = true;
                }
                WorkerEvent::JobDone { job_id, outputs, .. } if job_id == "it-job-1" => {
                    break outputs;
                }
                WorkerEvent::JobError { error, .. } => panic!("job failed: {error}"),
                _ => {}
            }
        };
        assert!(saw_progress, "expected at least one progress event");
        assert!(std::path::Path::new(&outputs[0]).exists());

        // cancel a long job
        let mut events2 = proc.subscribe_events();
        proc.request(
            &WorkerRequest::Submit {
                task: "echo".into(),
                job_id: "it-job-2".into(),
                params: serde_json::json!({"steps": 500, "delay": 0.05}),
            },
            Duration::from_secs(10),
        )
        .await
        .expect("submit long");
        proc.request(
            &WorkerRequest::Cancel { job_id: "it-job-2".into() },
            Duration::from_secs(10),
        )
        .await
        .expect("cancel");
        loop {
            let ev = tokio::time::timeout(Duration::from_secs(15), events2.recv())
                .await
                .expect("event timeout")
                .expect("event recv");
            if let WorkerEvent::JobError { job_id, error } = ev {
                if job_id == "it-job-2" {
                    assert_eq!(error, "cancelled");
                    break;
                }
            }
        }

        proc.stop().await.expect("stop");
        let _ = std::fs::remove_dir_all(&out_dir);
    }
}
