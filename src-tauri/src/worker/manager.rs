//! Worker manager actor: owns bootstrap state + the worker process.
//!
//! Mirrors `engine/manager.rs`: an mpsc command loop, a broadcast status
//! channel for the UI, and oneshot request/reply. Job events (progress/
//! done/error) are re-broadcast on a manager-owned channel so subscribers
//! survive worker restarts.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use reqwest::Client;
use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, oneshot, Mutex as TokioMutex};

use crate::config::{assets_staging_dir, worker_dir};
use crate::error::AppError;
use crate::worker::protocol::{WorkerEvent, WorkerRequest};
use crate::worker::sidecar::WorkerProcess;
use crate::worker::{bootstrap, WorkerVariant};

// ---------------------------------------------------------------------------
// Status types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineCapability {
    pub task: String,
    pub summary: String,
    pub available: bool,
    pub missing: Vec<String>,
    #[serde(default)]
    pub hidden: bool,
    /// What the `model` param accepts (e.g. "diffusers-dir") — drives
    /// store "runs in OpenHarness" badges in M9.
    #[serde(default)]
    pub formats: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum WorkerStatus {
    /// No valid venv on disk — run `worker_install` first.
    NotInstalled,
    Installing {
        variant: String,
        stage: String,
        pct: f32,
        bytes: u64,
        total: u64,
    },
    /// Venv ready but the process isn't running.
    Stopped { variant: String },
    Starting,
    Ready {
        variant: String,
        version: String,
        pipelines: Vec<PipelineCapability>,
    },
    Failed { error: String },
}

// ---------------------------------------------------------------------------
// Actor commands
// ---------------------------------------------------------------------------

pub enum WorkerCmd {
    Install {
        variant: WorkerVariant,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    Start {
        reply: oneshot::Sender<Result<Vec<PipelineCapability>, AppError>>,
    },
    Stop {
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    GetStatus {
        reply: oneshot::Sender<WorkerStatus>,
    },
    Subscribe {
        reply: oneshot::Sender<broadcast::Receiver<WorkerStatus>>,
    },
    Request {
        body: WorkerRequest,
        timeout: Duration,
        reply: oneshot::Sender<Result<serde_json::Value, AppError>>,
    },
    Logs {
        reply: oneshot::Sender<Vec<String>>,
    },
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct WorkerHandle {
    tx: mpsc::Sender<WorkerCmd>,
    bcast: broadcast::Sender<WorkerStatus>,
    job_events: broadcast::Sender<WorkerEvent>,
    pub subscriber_abort: Arc<TokioMutex<Option<tokio::task::AbortHandle>>>,
    pub job_subscriber_abort: Arc<TokioMutex<Option<tokio::task::AbortHandle>>>,
}

impl WorkerHandle {
    async fn send<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> WorkerCmd,
    ) -> Result<T, AppError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(make(tx))
            .await
            .map_err(|_| AppError::WorkerUnavailable)?;
        rx.await.map_err(|_| AppError::WorkerUnavailable)
    }

    pub async fn install(&self, variant: WorkerVariant) -> Result<(), AppError> {
        self.send(|reply| WorkerCmd::Install { variant, reply }).await?
    }

    pub async fn start_worker(&self) -> Result<Vec<PipelineCapability>, AppError> {
        self.send(|reply| WorkerCmd::Start { reply }).await?
    }

    pub async fn stop_worker(&self) -> Result<(), AppError> {
        self.send(|reply| WorkerCmd::Stop { reply }).await?
    }

    pub async fn status(&self) -> WorkerStatus {
        self.send(|reply| WorkerCmd::GetStatus { reply })
            .await
            .unwrap_or(WorkerStatus::Failed {
                error: "worker manager unavailable".into(),
            })
    }

    pub async fn subscribe(&self) -> broadcast::Receiver<WorkerStatus> {
        self.send(|reply| WorkerCmd::Subscribe { reply })
            .await
            .unwrap_or_else(|_| self.bcast.subscribe())
    }

    /// Job progress/done/error events — survives worker restarts.
    pub fn subscribe_job_events(&self) -> broadcast::Receiver<WorkerEvent> {
        self.job_events.subscribe()
    }

    /// Sender side of the job-event bus. Lets non-worker engines (the
    /// Qwen3-TTS sidecar) emit job events onto the same stream the frontend
    /// and job-sync task already consume.
    pub fn job_events_sender(&self) -> broadcast::Sender<WorkerEvent> {
        self.job_events.clone()
    }

    pub async fn request(
        &self,
        body: WorkerRequest,
        timeout: Duration,
    ) -> Result<serde_json::Value, AppError> {
        self.send(|reply| WorkerCmd::Request { body, timeout, reply }).await?
    }

    pub async fn logs(&self) -> Vec<String> {
        self.send(|reply| WorkerCmd::Logs { reply })
            .await
            .unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

pub fn start(http: Client, worker_src: Option<PathBuf>) -> WorkerHandle {
    let (tx, mut rx) = mpsc::channel::<WorkerCmd>(32);
    let (bcast_tx, _) = broadcast::channel::<WorkerStatus>(32);
    let (job_events_tx, _) = broadcast::channel::<WorkerEvent>(256);

    let handle = WorkerHandle {
        tx,
        bcast: bcast_tx.clone(),
        job_events: job_events_tx.clone(),
        subscriber_abort: Arc::new(TokioMutex::new(None)),
        job_subscriber_abort: Arc::new(TokioMutex::new(None)),
    };

    tauri::async_runtime::spawn(async move {
        let mut process: Option<WorkerProcess> = None;
        let mut last_status = initial_status(worker_src.as_deref());
        tracing::info!("worker manager: initial status {:?}", phase_name(&last_status));

        while let Some(cmd) = rx.recv().await {
            match cmd {
                // ── Install ─────────────────────────────────────────────
                WorkerCmd::Install { variant, reply } => {
                    let Some(src) = worker_src.clone() else {
                        let e = AppError::Other("worker-py source dir not found".into());
                        let _ = reply.send(Err(e));
                        continue;
                    };
                    // A running worker holds the venv's python.exe open —
                    // stop it before rebuilding.
                    if let Some(p) = process.take() {
                        let _ = p.stop().await;
                    }
                    last_status = WorkerStatus::Installing {
                        variant: variant.as_slug().into(),
                        stage: "starting".into(),
                        pct: 0.0,
                        bytes: 0,
                        total: 0,
                    };
                    let _ = bcast_tx.send(last_status.clone());

                    let bcast_inner = bcast_tx.clone();
                    let variant_str = variant.as_slug().to_string();
                    let result = bootstrap::install(&http, variant, &src, move |p| {
                        let pct = if p.bytes_total > 0 {
                            (p.bytes_downloaded as f32 / p.bytes_total as f32) * 100.0
                        } else {
                            0.0
                        };
                        let _ = bcast_inner.send(WorkerStatus::Installing {
                            variant: variant_str.clone(),
                            stage: p.stage.to_string(),
                            pct,
                            bytes: p.bytes_downloaded,
                            total: p.bytes_total,
                        });
                    })
                    .await;

                    match result {
                        Ok(()) => {
                            last_status = WorkerStatus::Stopped {
                                variant: variant.as_slug().into(),
                            };
                            let _ = bcast_tx.send(last_status.clone());
                            let _ = reply.send(Ok(()));
                        }
                        Err(e) => {
                            last_status = WorkerStatus::Failed { error: e.to_string() };
                            let _ = bcast_tx.send(last_status.clone());
                            let _ = reply.send(Err(e));
                        }
                    }
                }

                // ── Start ───────────────────────────────────────────────
                WorkerCmd::Start { reply } => {
                    if let WorkerStatus::Ready { pipelines, .. } = &last_status {
                        if process.is_some() {
                            let _ = reply.send(Ok(pipelines.clone()));
                            continue;
                        }
                    }
                    let Some(src) = worker_src.clone() else {
                        let _ = reply.send(Err(AppError::Other(
                            "worker-py source dir not found".into(),
                        )));
                        continue;
                    };
                    let Some(variant) = bootstrap::ready_variant(&src) else {
                        let _ = reply.send(Err(AppError::WorkerNotInstalled));
                        continue;
                    };

                    last_status = WorkerStatus::Starting;
                    let _ = bcast_tx.send(last_status.clone());

                    let result = start_process(&src, &job_events_tx).await;
                    match result {
                        Ok((proc, version, pipelines)) => {
                            process = Some(proc);
                            last_status = WorkerStatus::Ready {
                                variant: variant.as_slug().into(),
                                version,
                                pipelines: pipelines.clone(),
                            };
                            let _ = bcast_tx.send(last_status.clone());
                            let _ = reply.send(Ok(pipelines));
                        }
                        Err(e) => {
                            last_status = WorkerStatus::Failed { error: e.to_string() };
                            let _ = bcast_tx.send(last_status.clone());
                            let _ = reply.send(Err(e));
                        }
                    }
                }

                // ── Stop ────────────────────────────────────────────────
                WorkerCmd::Stop { reply } => {
                    if let Some(p) = process.take() {
                        let _ = p.stop().await;
                    }
                    last_status = match worker_src
                        .as_deref()
                        .and_then(bootstrap::ready_variant)
                    {
                        Some(v) => WorkerStatus::Stopped { variant: v.as_slug().into() },
                        None => WorkerStatus::NotInstalled,
                    };
                    let _ = bcast_tx.send(last_status.clone());
                    let _ = reply.send(Ok(()));
                }

                // ── GetStatus / Subscribe ───────────────────────────────
                WorkerCmd::GetStatus { reply } => {
                    let _ = reply.send(last_status.clone());
                }
                WorkerCmd::Subscribe { reply } => {
                    let _ = reply.send(bcast_tx.subscribe());
                }

                // ── Request passthrough ─────────────────────────────────
                WorkerCmd::Request { body, timeout, reply } => {
                    let Some(p) = process.as_mut() else {
                        let _ = reply.send(Err(AppError::WorkerNotReady));
                        continue;
                    };
                    if !p.is_alive() {
                        // The process died out from under us.
                        process = None;
                        last_status = WorkerStatus::Failed {
                            error: "worker process exited unexpectedly".into(),
                        };
                        let _ = bcast_tx.send(last_status.clone());
                        let _ = reply.send(Err(AppError::WorkerNotReady));
                        continue;
                    }
                    let _ = reply.send(p.request(&body, timeout).await);
                }

                // ── Logs ────────────────────────────────────────────────
                WorkerCmd::Logs { reply } => {
                    let lines = match process.as_ref() {
                        Some(p) => p.log_buf.lock().await.iter().cloned().collect(),
                        None => Vec::new(),
                    };
                    let _ = reply.send(lines);
                }
            }
        }

        // Actor loop exited — stop the worker if running.
        if let Some(p) = process.take() {
            let _ = p.stop().await;
        }
    });

    handle
}

/// Torch wheel variant for a GPU reading: CUDA when an NVIDIA GPU is
/// present, CPU otherwise — same rule the engine install wizard uses.
pub fn variant_for_gpu(status: &crate::gpu::GpuStatus) -> WorkerVariant {
    match status {
        crate::gpu::GpuStatus::Available(_) => WorkerVariant::Cuda,
        crate::gpu::GpuStatus::NoGpu => WorkerVariant::Cpu,
    }
}

/// Auto-manage the worker at app startup: install the venv when missing or
/// stale (variant by GPU detection), then start the process; if it's already
/// installed but stopped, just start it.
///
/// Spawned as a background task — never blocks setup. Every step reuses the
/// normal actor code paths, so install progress and failures stream to the
/// UI through the existing status broadcast (`Installing`/`Failed`). One
/// attempt, no retry loop; lazy start-on-submit remains the fallback.
pub fn spawn_auto_manage(worker: WorkerHandle, gpu: crate::gpu::GpuHandle) {
    tauri::async_runtime::spawn(async move {
        match worker.status().await {
            WorkerStatus::NotInstalled => {
                // The GPU actor polls at 1 Hz with an immediate first tick —
                // wait for one real reading before picking the wheel variant.
                let mut rx = gpu.subscribe().await;
                let status = match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
                    Ok(Ok(s)) => s,
                    _ => gpu.status().await,
                };
                let variant = variant_for_gpu(&status);
                tracing::info!(
                    "worker auto-manage: venv missing or stale — installing the {} build",
                    variant.as_slug()
                );
                if let Err(e) = worker.install(variant).await {
                    tracing::warn!("worker auto-install failed: {e}");
                    return;
                }
                if let Err(e) = worker.start_worker().await {
                    tracing::warn!("worker auto-start failed: {e}");
                }
            }
            WorkerStatus::Stopped { .. } => {
                if let Err(e) = worker.start_worker().await {
                    tracing::warn!("worker auto-start failed: {e}");
                }
            }
            // Installing / Starting / Ready: someone beat us to it.
            // Failed: leave the honest error on the card; don't retry-loop.
            _ => {}
        }
    });
}

/// Spawn the process, verify the ping handshake, fetch capabilities, and
/// wire its event stream into the manager-owned broadcast.
async fn start_process(
    src: &std::path::Path,
    job_events: &broadcast::Sender<WorkerEvent>,
) -> Result<(WorkerProcess, String, Vec<PipelineCapability>), AppError> {
    let python = bootstrap::venv_python(&worker_dir());
    let out_dir = assets_staging_dir();
    std::fs::create_dir_all(&out_dir)?;

    let proc = WorkerProcess::start(&python, src, &out_dir).await?;

    // First handshake. The worker runs a synchronous native warmup
    // (numpy/scipy/sklearn/torch) on its main thread *before* it enters the
    // stdin request loop — see server.py::_preload_native, which orders the
    // imports that way to dodge a Windows loader-lock deadlock. A cold first
    // import of the CUDA torch wheel (~GB of DLLs, cold OS cache right after a
    // fresh venv install) routinely takes far longer than a normal ping, so
    // the startup handshake gets a generous ceiling. A genuine hang still
    // fails — just later — instead of a false "timed out" that latches the
    // worker to Failed on the one cold boot it was always going to be slow.
    let pong = proc
        .request(&WorkerRequest::Ping, Duration::from_secs(120))
        .await?;
    let version = pong["version"].as_str().unwrap_or("unknown").to_string();
    if version != crate::worker::WORKER_VERSION {
        tracing::warn!(
            "worker version mismatch: shipped {} but venv runs {}",
            crate::worker::WORKER_VERSION,
            version
        );
    }

    let caps_value = proc
        .request(&WorkerRequest::Capabilities, Duration::from_secs(15))
        .await?;
    let pipelines: Vec<PipelineCapability> =
        serde_json::from_value(caps_value["pipelines"].clone())
            .map_err(|e| AppError::WorkerRequest(format!("bad capabilities: {e}")))?;

    // Forward job events into the stable manager-owned channel.
    let mut rx = proc.subscribe_events();
    let fwd = job_events.clone();
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(ev) => {
                    let _ = fwd.send(ev);
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    });

    Ok((proc, version, pipelines))
}

fn initial_status(worker_src: Option<&std::path::Path>) -> WorkerStatus {
    match worker_src.and_then(bootstrap::ready_variant) {
        Some(v) => WorkerStatus::Stopped { variant: v.as_slug().into() },
        None => WorkerStatus::NotInstalled,
    }
}

fn phase_name(s: &WorkerStatus) -> &'static str {
    match s {
        WorkerStatus::NotInstalled => "not_installed",
        WorkerStatus::Installing { .. } => "installing",
        WorkerStatus::Stopped { .. } => "stopped",
        WorkerStatus::Starting => "starting",
        WorkerStatus::Ready { .. } => "ready",
        WorkerStatus::Failed { .. } => "failed",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{GpuMetrics, GpuStatus};

    #[test]
    fn variant_for_gpu_matches_engine_rule() {
        assert_eq!(variant_for_gpu(&GpuStatus::NoGpu), WorkerVariant::Cpu);
        let metrics = GpuMetrics {
            name: "NVIDIA GeForce RTX 4090".into(),
            driver_version: "560.94".into(),
            cuda_major: 12,
            cuda_minor: 6,
            vram_total_mb: 24_564,
            vram_used_mb: 1_200,
            vram_free_mb: 23_364,
            sampled_at: 0,
        };
        assert_eq!(
            variant_for_gpu(&GpuStatus::Available(metrics)),
            WorkerVariant::Cuda
        );
    }
}
