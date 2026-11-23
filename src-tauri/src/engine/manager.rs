use std::path::PathBuf;
use std::sync::Arc;

use reqwest::Client;
use serde::Serialize;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc, oneshot, watch, Mutex as TokioMutex};
use uuid::Uuid;

use crate::db::engines as engines_db;
use crate::engine::{
    installer,
    sidecar::{pick_free_port, LlamaServer, SidecarArgs},
    EngineVariant,
};
use crate::error::AppError;
use crate::planner::EngineKind;

// ---------------------------------------------------------------------------
// Status types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum EngineStatus {
    Idle,
    Installing {
        variant: String,
        pct: f32,
        bytes: u64,
        total: u64,
        stage: String,
    },
    Loading {
        model_id: String,
        /// Binary this load launches. Only the actor knows it for every load,
        /// UI or API.
        launcher: EngineKind,
    },
    Ready {
        model_id: String,
        port: u16,
        /// Context window (`--ctx-size`) the server was launched with.
        ctx: u32,
        launcher: EngineKind,
    },
    Failed {
        error: String,
        /// Binary that failed; absent when no load was involved (an install).
        #[serde(skip_serializing_if = "Option::is_none")]
        launcher: Option<EngineKind>,
    },
}

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EnginePhase {
    Idle,
    Installing,
    Loading,
    Ready,
    Failed,
}

/// llama-server launch arguments resolved by the command layer.
#[derive(Debug, Clone)]
pub struct ModelLoadArgs {
    pub engine_kind: crate::planner::EngineKind,
    pub inference_mode: super::sidecar::InferenceMode,
    pub ctx: u32,
    pub ngl: i32,
    pub batch_size: Option<u32>,
    pub threads: Option<u32>,
    pub flash_attn: bool,
    pub kv_cache_type: Option<String>,
    pub mmap: bool,
    pub mlock: bool,
    /// Extra llama-server flags from the load planner, appended verbatim.
    pub extra_args: Vec<String>,
}

impl Default for ModelLoadArgs {
    fn default() -> Self {
        Self {
            engine_kind: Default::default(),
            inference_mode: Default::default(),
            ctx: 4096,
            ngl: 0,
            batch_size: None,
            threads: None,
            flash_attn: false,
            kv_cache_type: None,
            mmap: true,
            mlock: false,
            extra_args: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Actor commands
// ---------------------------------------------------------------------------

pub enum EngineCmd {
    HostMemory { reply: oneshot::Sender<super::memory::HostCredit> },
    NativeInfo { retry_failed: bool, reply: oneshot::Sender<super::native::NativeStatus> },
    Install {
        variant: EngineVariant,
        reply: oneshot::Sender<Result<engines_db::EngineRow, AppError>>,
    },
    LoadModel {
        model_id: String,
        model_path: PathBuf,
        args: ModelLoadArgs,
        reply: oneshot::Sender<Result<u16, AppError>>,
    },
    UnloadModel {
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    GetStatus {
        reply: oneshot::Sender<EngineStatus>,
    },
    Subscribe {
        reply: oneshot::Sender<broadcast::Receiver<EngineStatus>>,
    },
    Logs {
        reply: oneshot::Sender<Vec<String>>,
    },
    ServerPid {
        reply: oneshot::Sender<Option<u32>>,
    },
}

// ---------------------------------------------------------------------------
// Manager handle
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct EngineManagerHandle {
    tx: mpsc::Sender<EngineCmd>,
    bcast: broadcast::Sender<EngineStatus>,
    latest_status: watch::Receiver<EngineStatus>,
    pub subscriber_abort: Arc<TokioMutex<Option<tokio::task::AbortHandle>>>,
}

impl EngineManagerHandle {
    /// Same text slot only; the running process's measured private resident
    /// bytes and private commit, never model size.
    pub async fn reclaimable_host(&self) -> super::memory::HostCredit {
        let (reply, rx) = oneshot::channel();
        if self.tx.send(EngineCmd::HostMemory { reply }).await.is_err() { return Default::default(); }
        rx.await.unwrap_or_default()
    }

    pub async fn native_info(&self) -> Result<super::native::NativeStatus, AppError> {
        self.probe_native(false).await
    }

    /// Like `native_info`, but probes again if the cached probe failed: the
    /// fix may be outside the binary (a missing DLL), so its size and mtime
    /// can't reveal it.
    pub async fn native_info_retry(&self) -> Result<super::native::NativeStatus, AppError> {
        self.probe_native(true).await
    }

    async fn probe_native(&self, retry_failed: bool) -> Result<super::native::NativeStatus, AppError> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(EngineCmd::NativeInfo { retry_failed, reply }).await.map_err(|_| AppError::Other("engine manager unavailable".into()))?;
        rx.await.map_err(|_| AppError::Other("engine manager dropped reply".into()))
    }
    pub async fn install(
        &self,
        variant: EngineVariant,
    ) -> Result<engines_db::EngineRow, AppError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineCmd::Install { variant, reply: reply_tx })
            .await
            .map_err(|_| AppError::Other("engine manager unavailable".into()))?;
        reply_rx
            .await
            .map_err(|_| AppError::Other("engine manager dropped reply".into()))?
    }

    pub async fn load_model(
        &self,
        model_id: String,
        path: PathBuf,
        args: ModelLoadArgs,
    ) -> Result<u16, AppError> {
        self.begin_load(model_id, path, args).await?.await
            .map_err(|_| AppError::Other("engine manager dropped reply".into()))?
    }

    pub(crate) async fn begin_load(&self, model_id: String, path: PathBuf, args: ModelLoadArgs) -> Result<oneshot::Receiver<Result<u16, AppError>>, AppError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineCmd::LoadModel {
                model_id,
                model_path: path,
                args,
                reply: reply_tx,
            })
            .await
            .map_err(|_| AppError::Other("engine manager unavailable".into()))?;
        Ok(reply_rx)
    }

    pub async fn unload_model(&self) -> Result<(), AppError> {
        let (reply_tx, reply_rx) = oneshot::channel();
        self.tx
            .send(EngineCmd::UnloadModel { reply: reply_tx })
            .await
            .map_err(|_| AppError::Other("engine manager unavailable".into()))?;
        reply_rx
            .await
            .map_err(|_| AppError::Other("engine manager dropped reply".into()))?
    }

    /// Read-only snapshot, including Loading, without queueing behind a model load.
    pub fn status_snapshot(&self) -> EngineStatus {
        self.latest_status.borrow().clone()
    }

    pub async fn status(&self) -> EngineStatus {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.tx.send(EngineCmd::GetStatus { reply: reply_tx }).await.is_err() {
            return EngineStatus::Failed { error: "engine manager unavailable".into(), launcher: None };
        }
        reply_rx.await.unwrap_or(EngineStatus::Failed {
            error: "engine manager dropped reply".into(),
            launcher: None,
        })
    }

    pub async fn subscribe(&self) -> broadcast::Receiver<EngineStatus> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.tx.send(EngineCmd::Subscribe { reply: reply_tx }).await.is_err() {
            return self.bcast.subscribe();
        }
        reply_rx.await.unwrap_or_else(|_| self.bcast.subscribe())
    }

    pub fn broadcast_sender(&self) -> broadcast::Sender<EngineStatus> {
        self.bcast.clone()
    }

    /// Last ~1000 stdout/stderr lines from the running llama-server
    /// (empty when no model is loaded).
    pub async fn logs(&self) -> Vec<String> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.tx.send(EngineCmd::Logs { reply: reply_tx }).await.is_err() {
            return Vec::new();
        }
        reply_rx.await.unwrap_or_default()
    }

    /// OS pid of the running llama-server, when a model is loaded.
    pub async fn server_pid(&self) -> Option<u32> {
        let (reply_tx, reply_rx) = oneshot::channel();
        if self.tx.send(EngineCmd::ServerPid { reply: reply_tx }).await.is_err() {
            return None;
        }
        reply_rx.await.unwrap_or(None)
    }
}

// ---------------------------------------------------------------------------
// Factory — starts the actor loop and returns a handle
// ---------------------------------------------------------------------------

fn publish_status(
    latest: &watch::Sender<EngineStatus>,
    events: &broadcast::Sender<EngineStatus>,
    status: &EngineStatus,
) {
    latest.send_replace(status.clone());
    let _ = events.send(status.clone());
}

pub fn start(pool: SqlitePool, http: Client) -> EngineManagerHandle {
    start_with_native(pool, http, super::native::manifest().install_dir.join(super::native::BINARY))
}

pub(crate) fn start_with_native(pool: SqlitePool, http: Client, native_binary: PathBuf) -> EngineManagerHandle {
    let (tx, mut rx) = mpsc::channel::<EngineCmd>(32);
    let (bcast_tx, _) = broadcast::channel::<EngineStatus>(16);
    let (status_tx, latest_status) = watch::channel(EngineStatus::Idle);

    let handle = EngineManagerHandle {
        tx,
        bcast: bcast_tx.clone(),
        latest_status,
        subscriber_abort: Arc::new(TokioMutex::new(None)),
    };

    let bcast = bcast_tx.clone();

    tauri::async_runtime::spawn(async move {
        let mut current_server: Option<LlamaServer> = None;
        let mut current_model: Option<String> = None;
        let mut last_status: EngineStatus = EngineStatus::Idle;

        // Log any previously-installed active engine on startup
        match engines_db::get_active(&pool).await {
            Ok(Some(active)) => {
                tracing::info!(
                    "active engine: {} ({}) at {}",
                    active.version,
                    active.variant,
                    active.path
                );
            }
            Ok(None) => tracing::info!("no active engine installed"),
            Err(e) => tracing::warn!("could not query active engine: {}", e),
        }

        // Sweep rows left inactive by past reinstalls / variant switches.
        match engines_db::prune_inactive(&pool).await {
            Ok(n) if n > 0 => tracing::info!("pruned {} orphan engine row(s)", n),
            Ok(_) => {}
            Err(e) => tracing::warn!("could not prune orphan engine rows: {}", e),
        }

        let mut native_discovery = super::native::Discovery::default();
        let mut deferred = std::collections::VecDeque::new();
        loop {
            let cmd = if let Some(cmd) = deferred.pop_front() { cmd } else {
                let Some(cmd) = rx.recv().await else { break }; cmd
            };
            // Do not lend a dead child its status or memory credit.
            if current_server.as_mut().is_some_and(|s| !s.is_running()) {
                current_server.take();
                current_model = None;
                let launcher = if let EngineStatus::Ready { launcher, .. } = &last_status { Some(*launcher) } else { None };
                last_status = EngineStatus::Failed { error: "Engine process exited".into(), launcher };
                publish_status(&status_tx, &bcast, &last_status);
            }
            match cmd {
                EngineCmd::HostMemory { reply } => {
                    let _ = reply.send(current_server.as_ref().and_then(|s| s.pid()).map(super::memory::private_memory).unwrap_or_default());
                }
                EngineCmd::NativeInfo { retry_failed, reply } => {
                    if retry_failed { native_discovery.forget_failure(); }
                    let _ = reply.send(native_discovery.at(native_binary.clone()).await);
                }
                // ── Install ──────────────────────────────────────────────────
                EngineCmd::Install { variant, reply } => {
                    last_status = EngineStatus::Installing {
                        variant: variant.as_asset_slug().into(),
                        pct: 0.0,
                        bytes: 0,
                        total: 0,
                        stage: "starting".into(),
                    };
                    publish_status(&status_tx, &bcast, &last_status);

                    let bcast_inner = bcast.clone();
                    let variant_str = variant.as_asset_slug().to_string();

                    let result: Result<PathBuf, AppError> =
                        installer::install(&http, variant, move |p| {
                            let pct = if p.bytes_total > 0 {
                                (p.bytes_downloaded as f32 / p.bytes_total as f32) * 100.0
                            } else {
                                0.0
                            };
                            let _ = bcast_inner.send(EngineStatus::Installing {
                                variant: variant_str.clone(),
                                pct,
                                bytes: p.bytes_downloaded,
                                total: p.bytes_total,
                                stage: p.stage.to_string(),
                            });
                        })
                        .await;

                    match result {
                        Ok(server_path) => {
                            // Insert as inactive, then flip via set_active(), which
                            // atomically clears any previously-active engine first.
                            // Inserting with active=1 directly would violate the
                            // `idx_engines_active` partial unique index when a prior
                            // install's active row still exists.
                            let mut row = engines_db::EngineRow {
                                id: Uuid::new_v4().to_string(),
                                version: crate::engine::ENGINE_BUILD.to_string(),
                                variant: variant.as_asset_slug().into(),
                                path: server_path.to_string_lossy().to_string(),
                                // Intentionally empty: SHA-256 manifest pinning for engine ZIPs is M4 tracked debt.
                                // TLS is the integrity story for v1.
                                sha256: String::new(),
                                active: false,
                                installed_at: chrono::Utc::now().timestamp(),
                            };
                            if let Err(e) = engines_db::upsert(&pool, &row).await {
                                last_status =
                                    EngineStatus::Failed { error: format!("db upsert: {}", e), launcher: None };
                                publish_status(&status_tx, &bcast, &last_status);
                                let _ = reply.send(Err(e));
                                continue;
                            }
                            if let Err(e) = engines_db::set_active(&pool, &row.id).await {
                                last_status = EngineStatus::Failed {
                                    error: format!("set_active: {}", e),
                                    launcher: None,
                                };
                                publish_status(&status_tx, &bcast, &last_status);
                                let _ = reply.send(Err(e));
                                continue;
                            }
                            row.active = true;
                            // The freshly-deactivated previous install is now an orphan.
                            if let Err(e) = engines_db::prune_inactive(&pool).await {
                                tracing::warn!("could not prune orphan engine rows: {}", e);
                            }
                            last_status = EngineStatus::Idle;
                            publish_status(&status_tx, &bcast, &last_status);
                            let _ = reply.send(Ok(row));
                        }
                        Err(e) => {
                            last_status = EngineStatus::Failed { error: e.to_string(), launcher: None };
                            publish_status(&status_tx, &bcast, &last_status);
                            let _ = reply.send(Err(e));
                        }
                    }
                }

                // ── LoadModel ────────────────────────────────────────────────
                EngineCmd::LoadModel { model_id, model_path, args: load_args, reply } => {
                    // Pre-flight: a GGUF whose architecture has none of the
                    // standard text-LLM keys (e.g. Qwen3-TTS talker GGUFs,
                    // whose metadata lives under `{arch}.talker.*`) cannot be
                    // served by llama-server. Refuse with a clear error before
                    // tearing down whatever is currently loaded. An unreadable
                    // header parses to all-None fields and passes through —
                    // llama-server stays the final judge.
                    if let Ok(meta) = crate::models::gguf::parse_header(&model_path) {
                        if let Some(arch) = &meta.architecture {
                            if meta.block_count.is_none() && meta.embedding_length.is_none() {
                                let e = AppError::UnsupportedModelArchitecture(arch.clone());
                                let _ = reply.send(Err(e));
                                continue;
                            }
                        }
                    }

                    // Stop any existing server
                    if let Some(s) = current_server.take() {
                        let _ = s.stop().await;
                        current_model = None;
                    }

                    let launcher = load_args.engine_kind;
                    last_status = EngineStatus::Loading { model_id: model_id.clone(), launcher };
                    publish_status(&status_tx, &bcast, &last_status);

                    let native = load_args.engine_kind == crate::planner::EngineKind::SsdStream;
                    let binary = if native {
                        let info = native_discovery.at(native_binary.clone()).await;
                        if info.support().streaming { Ok(info.path) } else {
                            Err(AppError::Other(info.error.unwrap_or_else(|| "Native engine isn't installed or does not report ssd_expert_streaming".into())))
                        }
                    } else {
                        engines_db::get_current(&pool).await.and_then(|r| r.map(|r| PathBuf::from(r.path)).ok_or(AppError::EngineNotInstalled))
                    };
                    let bin = match binary {
                        Ok(bin) => bin,
                        Err(e) => {
                            last_status = EngineStatus::Failed { error: e.to_string(), launcher: Some(launcher) };
                            publish_status(&status_tx, &bcast, &last_status);
                            let _ = reply.send(Err(e));
                            continue;
                        }
                    };

                    let port = match pick_free_port() {
                        Ok(p) => p,
                        Err(e) => {
                            let err = AppError::Other(format!("port: {}", e));
                            last_status = EngineStatus::Failed { error: err.to_string(), launcher: Some(launcher) };
                            publish_status(&status_tx, &bcast, &last_status);
                            let _ = reply.send(Err(err));
                            continue;
                        }
                    };

                    let args = SidecarArgs {
                        engine_kind: load_args.engine_kind,
                        inference_mode: load_args.inference_mode,
                        model_path: model_path.clone(),
                        ctx: load_args.ctx,
                        n_gpu_layers: load_args.ngl,
                        threads: load_args.threads,
                        batch_size: load_args.batch_size,
                        flash_attn: load_args.flash_attn,
                        kv_cache_type: load_args.kv_cache_type.clone(),
                        mmap: load_args.mmap,
                        mlock: load_args.mlock,
                        alias: Some(model_id.clone()),
                        host: "127.0.0.1".into(),
                        port,
                        extra_args: load_args.extra_args.clone(),
                    };

                    let mut server = match LlamaServer::start(&bin, args).await {
                        Ok(s) => s,
                        Err(e) => {
                            last_status = EngineStatus::Failed { error: e.to_string(), launcher: Some(launcher) };
                            publish_status(&status_tx, &bcast, &last_status);
                            let _ = reply.send(Err(e));
                            continue;
                        }
                    };

                    let ready_timeout = std::time::Duration::from_secs(if native { 1800 } else { 120 });
                    let log_buf = server.log_buf.clone();
                    let pid = server.pid();
                    let mut cancelled = None;
                    let result = {
                        let ready = server.wait_ready(&http, ready_timeout);
                        tokio::pin!(ready);
                        loop {
                            tokio::select! {
                                // A queued unload wins a simultaneous health result.
                                biased;
                                cmd = rx.recv() => match cmd {
                                    Some(EngineCmd::UnloadModel { reply }) => { cancelled = Some(reply); break Err(AppError::EngineLoadCancelled); }
                                    Some(EngineCmd::GetStatus { reply }) => { let _ = reply.send(last_status.clone()); }
                                    Some(EngineCmd::Subscribe { reply }) => { let _ = reply.send(bcast.subscribe()); }
                                    Some(EngineCmd::Logs { reply }) => { let _ = reply.send(log_buf.lock().await.iter().cloned().collect()); }
                                    Some(EngineCmd::ServerPid { reply }) => { let _ = reply.send(pid); }
                                    Some(EngineCmd::HostMemory { reply }) => { let _ = reply.send(Default::default()); }
                                    // Never run the probe here, retry or not. The UI offers the retry again once startup ends.
                                    Some(EngineCmd::NativeInfo { reply, .. }) => { let _ = reply.send(native_discovery.cached_at(&native_binary).await); }
                                    Some(cmd) => deferred.push_back(cmd),
                                    None => break Err(AppError::EngineLoadCancelled),
                                },
                                result = &mut ready => break result,
                            }
                        }
                    }; // Drop readiness's borrow before stopping the owned child.
                    if let Err(e) = result {
                        let stopped = server.stop().await;
                        last_status = if cancelled.is_some() { EngineStatus::Idle } else { EngineStatus::Failed { error: e.to_string(), launcher: Some(launcher) } };
                        publish_status(&status_tx, &bcast, &last_status);
                        let _ = reply.send(Err(e));
                        if let Some(reply) = cancelled { let _ = reply.send(stopped); }
                        continue;
                    }

                    current_server = Some(server);
                    current_model = Some(model_id.clone());
                    last_status = EngineStatus::Ready { model_id, port, ctx: load_args.ctx, launcher };
                    publish_status(&status_tx, &bcast, &last_status);
                    let _ = reply.send(Ok(port));
                }

                // ── UnloadModel ──────────────────────────────────────────────
                EngineCmd::UnloadModel { reply } => {
                    if let Some(s) = current_server.take() {
                        let _ = s.stop().await;
                    }
                    current_model = None;
                    last_status = EngineStatus::Idle;
                    publish_status(&status_tx, &bcast, &last_status);
                    let _ = reply.send(Ok(()));
                }

                // ── GetStatus ────────────────────────────────────────────────
                EngineCmd::GetStatus { reply } => {
                    let _ = reply.send(last_status.clone());
                }

                // ── Subscribe ────────────────────────────────────────────────
                EngineCmd::Subscribe { reply } => {
                    let _ = reply.send(bcast.subscribe());
                }

                // ── Logs ─────────────────────────────────────────────────────
                EngineCmd::Logs { reply } => {
                    let lines = match &current_server {
                        Some(s) => {
                            let buf = s.log_buf.lock().await;
                            buf.iter().cloned().collect()
                        }
                        None => Vec::new(),
                    };
                    let _ = reply.send(lines);
                }
                EngineCmd::ServerPid { reply } => {
                    let _ = reply.send(current_server.as_ref().and_then(|s| s.pid()));
                }
            }
        }

        // Actor loop exited — stop the server if one is running
        if let Some(s) = current_server.take() {
            let _ = s.stop().await;
        }
        // suppress unused warning on current_model
        let _ = current_model;
    });

    handle
}
