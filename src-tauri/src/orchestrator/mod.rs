//! Engine orchestrator (M8 §8.4): the single chokepoint for loading models
//! across both engine families — the llama-server text engine and the
//! Python-worker generation pipelines.
//!
//! Owns a VRAM ledger (fit estimates + NVML actuals reported by the worker),
//! lets residents coexist when they fit ("concurrent-when-fits"), and when a
//! requested load would only fully fit after freeing something else, returns
//! a typed needs-eviction outcome listing candidates instead of silently
//! evicting. Same-slot swaps stay implicit — a text model replacing the text
//! model, or a pipeline swapping its own cached model, is the explicit
//! intent of the request, so the slot's own footprint counts as already
//! reclaimed when gating.
//!
//! Absorbs §7.6 smart loading: ctx defaults to `min(context_train, 8192)`
//! and `-ngl` comes from the VRAM-fit picker unless the caller overrides.
//!
//! Startup retains actor ownership while servicing control reads, worker
//! events and cancellation. Other load/reservation writes stay FIFO deferred.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc, oneshot, Mutex as TokioMutex};

use crate::engine::fit::{estimate_vram_mb, pick_ngl, FitInputs, FIXED_OVERHEAD_MB};
use crate::chat::activity::Activity as ChatActivity;
use crate::db::generation_jobs;
use crate::engine::{EngineManagerHandle, EngineStatus, ModelLoadArgs};
use crate::error::AppError;
use crate::gpu::GpuHandle;
use crate::models::{gguf, registry};
use crate::planner::store::PlannerHandle;
use crate::planner::{self, KvType, LoadPlan, ModelAccounting, PlanOverrides, Reclaimable, VramBudget};
use crate::tts::{TtsHandle, TtsStatus};
use crate::worker::protocol::{WorkerEvent, WorkerRequest};
use crate::worker::{WorkerHandle, WorkerStatus};

pub const TEXT_RESIDENT_ID: &str = "text";
pub const TTS_RESIDENT_ID: &str = "tts";

mod accounting_cache;
mod leases;
#[cfg(test)]
pub(crate) mod native_tests;
pub use leases::{ActiveJob, JobLease};
use leases::JobLeases;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResidentKind {
    TextEngine,
    TtsEngine,
    Pipeline,
}

/// One model resident on the GPU, as tracked by the ledger.
#[derive(Debug, Clone, Serialize)]
pub struct ResidentInfo {
    /// `"text"`, `"tts"`, or `"pipeline:<task>"` — pass to `orchestrator_unload`.
    pub id: String,
    pub kind: ResidentKind,
    /// Human-readable: display name where known, else model id/path.
    pub label: String,
    /// Originating catalog model id, where known — lets the UI mark catalog
    /// rows as Loaded (W1 §16). None for loads the ledger merely adopted.
    pub model_id: Option<String>,
    pub est_mb: u64,
    /// NVML-actual VRAM after load, where known (worker reports it per job).
    pub actual_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LedgerSnapshot {
    /// False when no NVIDIA GPU is present (ledger is display-only then).
    pub gpu: bool,
    pub total_mb: u64,
    pub free_mb: u64,
    pub residents: Vec<ResidentInfo>,
}

/// Result of a text-model load request.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum LoadOutcome {
    Loaded {
        port: u16,
        model_id: String,
        ctx: u32,
        ngl: i32,
        /// The plan the model was loaded with (absent for legacy tuned loads
        /// that pass an explicit `ngl`).
        #[serde(skip_serializing_if = "Option::is_none")]
        plan: Option<Box<LoadPlan>>,
    },
    /// The load would only fully fit after freeing something — nothing was
    /// loaded or evicted; the caller picks candidates and retries.
    NeedsEviction {
        required_mb: u64,
        free_mb: u64,
        candidates: Vec<ResidentInfo>,
    },
}

/// Result of a pipeline VRAM reservation (§8.5 generation commands call this
/// before submitting a job to the worker).
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ReserveOutcome {
    Reserved,
    NeedsEviction {
        required_mb: u64,
        free_mb: u64,
        candidates: Vec<ResidentInfo>,
    },
}

/// A text-engine load request.
///
/// Without an explicit `ngl` the load planner decides placement: a supplied
/// `plan` is a preview; the planner always recomputes with its overrides or
/// `overrides` (plus `ctx` / `kv_cache_type`, which fold into them). With
/// an explicit `ngl` the legacy path runs: ctx = `min(context_train, 8192)`
/// and the caller's flags verbatim.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct TextLoadRequest {
    pub model_id: String,
    pub ctx: Option<u32>,
    pub ngl: Option<i32>,
    pub batch_size: Option<u32>,
    pub threads: Option<u32>,
    pub flash_attn: Option<bool>,
    pub kv_cache_type: Option<String>,
    pub mmap: Option<bool>,
    pub mlock: Option<bool>,
    /// Skip the eviction gate and load with whatever fits (degrading to
    /// partial offload / CPU exactly like pre-orchestrator behavior).
    #[serde(default)]
    pub force: bool,
    /// A plan from `plan_load` to realise.
    #[serde(default)]
    pub plan: Option<LoadPlan>,
    /// Planner overrides when no plan is supplied.
    #[serde(default)]
    pub overrides: Option<PlanOverrides>,
    /// Load a plan that fits as it is instead of first offering to unload
    /// residents for a faster one (`NeedsEviction` then only comes back when
    /// nothing fits). For headless callers such as API just-in-time loads.
    #[serde(default)]
    pub decline_eviction_offer: bool,
}

impl TextLoadRequest {
    /// The overrides the planner should honour: explicit overrides, with the
    /// legacy `ctx` / `kv_cache_type` fields filling unset slots.
    fn planner_overrides(&self) -> PlanOverrides {
        let mut ov = self.overrides.clone().unwrap_or_default();
        if ov.ctx.is_none() {
            ov.ctx = self.ctx;
        }
        if ov.kv_type.is_none() {
            ov.kv_type = self.kv_cache_type.as_deref().and_then(KvType::parse);
        }
        ov
    }
}

impl TextLoadRequest {
    /// A quick load with every option left at its default.
    fn is_plain(&self) -> bool {
        self.ctx.is_none()
            && self.ngl.is_none()
            && self.batch_size.is_none()
            && self.threads.is_none()
            && self.flash_attn.is_none()
            && self.kv_cache_type.is_none()
            && self.mmap.is_none()
            && self.mlock.is_none()
            && !self.force
            && self.plan.is_none()
            && self.overrides.is_none()
    }
}

// ---------------------------------------------------------------------------
// Gate logic (pure — unit-tested below)
// ---------------------------------------------------------------------------

/// Estimated VRAM at a given `-ngl`: full estimate at -1, zero at 0, and a
/// linear per-layer scale in between (matching `pick_ngl`'s model).
fn est_for_ngl(i: &FitInputs, ngl: i32) -> u64 {
    let full = estimate_vram_mb(i);
    match ngl {
        -1 => full,
        0 => 0,
        n => {
            let var = full.saturating_sub(FIXED_OVERHEAD_MB);
            let layers = (n as u64).min(i.n_layers.max(1));
            FIXED_OVERHEAD_MB + var * layers / i.n_layers.max(1)
        }
    }
}

/// Decide a text load. `free_mb` must already include the current text
/// slot's own footprint (same-slot swap reclaims it implicitly);
/// `reclaimable_mb` is the sum of OTHER residents (pipelines).
///
/// Returns `Ok(ngl)` to proceed or `Err(required_mb)` when the caller
/// should emit `NeedsEviction` — i.e. evicting would actually fix the fit.
fn text_gate(
    i: &FitInputs,
    explicit_ngl: Option<i32>,
    free_mb: u64,
    reclaimable_mb: u64,
    force: bool,
) -> Result<i32, u64> {
    match explicit_ngl {
        Some(n) => {
            let needed = est_for_ngl(i, n);
            if !force
                && reclaimable_mb > 0
                && needed > free_mb
                && needed <= free_mb + reclaimable_mb
            {
                Err(needed)
            } else {
                Ok(n)
            }
        }
        None => {
            let auto = pick_ngl(i, free_mb);
            if force || reclaimable_mb == 0 || auto == -1 {
                return Ok(auto);
            }
            // Degraded pick — would it fully fit if other residents left?
            if pick_ngl(i, free_mb + reclaimable_mb) == -1 {
                Err(estimate_vram_mb(i))
            } else {
                // Degraded either way — don't bother the user with a choice
                // that wouldn't fix anything.
                Ok(auto)
            }
        }
    }
}

/// What a planner-driven text load does with its plan.
#[derive(Debug, PartialEq)]
enum PlannedGate {
    Proceed,
    /// Offer to unload other residents first (the caller returns
    /// `NeedsEviction`).
    OfferEviction,
    /// The plan can't load; the message says why.
    Reject(String),
}

/// `force` skips only the eviction offer. An infeasible plan is rejected
/// whatever the flags: loading it would run out of memory.
fn planned_gate(
    feasible: bool,
    infeasible_reason: Option<&str>,
    has_eviction_offer: bool,
    force: bool,
    decline_eviction_offer: bool,
) -> PlannedGate {
    if !force && has_eviction_offer && !(decline_eviction_offer && feasible) {
        return PlannedGate::OfferEviction;
    }
    if !feasible {
        return PlannedGate::Reject(infeasible_reason.unwrap_or("Doesn't fit.").to_string());
    }
    PlannedGate::Proceed
}

/// Whether the active llama.cpp build can use the GPU. The CPU build
/// allocates weights and context memory in system RAM even on an NVIDIA
/// machine, so the planner must not budget graphics memory for it. An
/// unknown or missing row counts as GPU-capable (the install wizard picks
/// CUDA when a GPU is present).
fn engine_uses_gpu(active_variant: Option<&str>) -> bool {
    active_variant.and_then(crate::engine::EngineVariant::from_db_str) != Some(crate::engine::EngineVariant::Cpu)
}

/// Decide a pipeline reservation. `free_mb` already includes the same
/// task's current footprint; `reclaimable_mb` is the sum of other residents.
fn pipeline_gate(
    est_mb: u64,
    free_mb: u64,
    reclaimable_mb: u64,
    force: bool,
) -> Result<(), u64> {
    if force || est_mb <= free_mb || reclaimable_mb == 0 {
        return Ok(());
    }
    if est_mb <= free_mb + reclaimable_mb {
        Err(est_mb)
    } else {
        // Eviction wouldn't fix it — proceed; the pipeline can CPU-offload
        // or fail with an honest CUDA OOM.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PipelineResident {
    label: String,
    model_id: Option<String>,
    est_mb: u64,
    actual_mb: Option<u64>,
}

pub enum OrchestratorCmd {
    LoadText {
        // Boxed: a supplied LoadPlan makes the request large.
        req: Box<TextLoadRequest>,
        reply: oneshot::Sender<Result<LoadOutcome, AppError>>,
    },
    ReservePipeline {
        task: String,
        label: String,
        model_id: Option<String>,
        est_mb: u64,
        force: bool,
        /// Taken in the same turn the reservation is granted.
        lease: Option<JobLease>,
        reply: oneshot::Sender<Result<ReserveOutcome, AppError>>,
    },
    LoadTts {
        model_id: String,
        force: bool,
        lease: Option<JobLease>,
        reply: oneshot::Sender<Result<ReserveOutcome, AppError>>,
    },
    AbortLease {
        job_id: String,
        error: String,
        reply: oneshot::Sender<()>,
    },
    ActiveJobs {
        reply: oneshot::Sender<Vec<ActiveJob>>,
    },
    Unload {
        resident_id: String,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    UnloadIfIdle {
        resident_id: String,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    UnloadTextIfModel {
        model_id: String,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    UnloadIfModel {
        resident_id: String,
        expected_model_id: Option<String>,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    Ledger {
        reply: oneshot::Sender<LedgerSnapshot>,
    },
    Subscribe {
        reply: oneshot::Sender<broadcast::Receiver<LedgerSnapshot>>,
    },
    PlanText {
        model_id: String,
        overrides: PlanOverrides,
        reply: oneshot::Sender<Result<LoadPlan, AppError>>,
    },
    CurrentTextPlan {
        reply: oneshot::Sender<Option<(String, LoadPlan)>>,
    },
    Accounting {
        model_id: String,
        reply: oneshot::Sender<Result<Arc<ModelAccounting>, AppError>>,
    },
}

#[derive(Clone)]
pub struct OrchestratorHandle {
    tx: mpsc::Sender<OrchestratorCmd>,
    bcast: broadcast::Sender<LedgerSnapshot>,
    pub subscriber_abort: Arc<TokioMutex<Option<tokio::task::AbortHandle>>>,
}

impl OrchestratorHandle {
    async fn send<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> OrchestratorCmd,
    ) -> Result<T, AppError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(make(tx))
            .await
            .map_err(|_| AppError::Other("orchestrator unavailable".into()))?;
        rx.await
            .map_err(|_| AppError::Other("orchestrator unavailable".into()))
    }

    pub async fn load_text(&self, req: TextLoadRequest) -> Result<LoadOutcome, AppError> {
        self.send(|reply| OrchestratorCmd::LoadText { req: Box::new(req), reply }).await?
    }

    /// Called by the generation commands (§8.5) before each submit, and by
    /// the W1 explicit preload. `model_id` correlates the resident back to
    /// its catalog row where known. A job submit passes its `lease`, which is
    /// granted together with the reservation and held until the job ends.
    pub async fn reserve_pipeline(
        &self,
        task: String,
        label: String,
        model_id: Option<String>,
        est_mb: u64,
        force: bool,
        lease: Option<JobLease>,
    ) -> Result<ReserveOutcome, AppError> {
        self.send(|reply| OrchestratorCmd::ReservePipeline {
            task,
            label,
            model_id,
            est_mb,
            force,
            lease,
            reply,
        })
        .await?
    }

    /// Gate + load a Qwen3-TTS talker (the `"tts"` resident). `Reserved`
    /// means the sidecar is up and healthy. A synthesis passes its `lease`.
    pub async fn load_tts(
        &self,
        model_id: String,
        force: bool,
        lease: Option<JobLease>,
    ) -> Result<ReserveOutcome, AppError> {
        self.send(|reply| OrchestratorCmd::LoadTts { model_id, force, lease, reply }).await?
    }

    /// End a job's lease when its submit fails before the job starts. If the
    /// lease was still held this publishes `job_error` for the job, because a
    /// reloaded UI may have adopted it from `active_jobs`. A job that started
    /// ends through its own terminal event instead.
    pub async fn abort_lease(&self, job_id: String, error: String) {
        let _ = self.send(|reply| OrchestratorCmd::AbortLease { job_id, error, reply }).await;
    }

    /// Running job rows (no history cutoff) plus leased jobs whose rows
    /// aren't written yet. A reloaded UI adopts these.
    pub async fn active_jobs(&self) -> Result<Vec<ActiveJob>, AppError> {
        self.send(|reply| OrchestratorCmd::ActiveJobs { reply }).await
    }

    /// Unconditional unload, for internal releases (a reservation whose
    /// submit failed). User-requested unloads go through `unload_if_idle` or
    /// `unload_if_model`, which refuse while the resident is doing work.
    pub async fn unload(&self, resident_id: String) -> Result<(), AppError> {
        self.send(|reply| OrchestratorCmd::Unload { resident_id, reply }).await?
    }

    /// User-requested unload: refuses while a chat turn or a running
    /// generation job is using the resident (see `Actor::guarded_unload`).
    pub async fn unload_if_idle(&self, resident_id: String) -> Result<(), AppError> {
        self.send(|reply| OrchestratorCmd::UnloadIfIdle { resident_id, reply }).await?
    }

    /// Idle API cleanup must not unload a different model selected in the UI.
    pub async fn unload_text_if_model(&self, model_id: String) -> Result<(), AppError> {
        self.send(|reply| OrchestratorCmd::UnloadTextIfModel { model_id, reply }).await?
    }

    /// Unload `resident_id` only if it still holds `expected_model_id`.
    ///
    /// The check and the unload run in one actor turn, so a UI that clicked
    /// Unload on model A while B was replacing it in the same slot can't end
    /// up unloading B once B's load finishes (the actor serializes the queued
    /// unload behind the load). A slot that is already empty is a no-op; a
    /// slot that now holds another model is an error the UI can show.
    /// `expected_model_id: None` means the caller saw an uncorrelated
    /// resident; it then matches only a resident that is still uncorrelated.
    pub async fn unload_if_model(
        &self,
        resident_id: String,
        expected_model_id: Option<String>,
    ) -> Result<(), AppError> {
        self.send(|reply| OrchestratorCmd::UnloadIfModel { resident_id, expected_model_id, reply })
            .await?
    }

    pub async fn ledger(&self) -> Result<LedgerSnapshot, AppError> {
        self.send(|reply| OrchestratorCmd::Ledger { reply }).await
    }

    pub async fn subscribe(&self) -> broadcast::Receiver<LedgerSnapshot> {
        self.send(|reply| OrchestratorCmd::Subscribe { reply })
            .await
            .unwrap_or_else(|_| self.bcast.subscribe())
    }

    /// Plan a text load without loading: the planner's choice for
    /// `model_id` against the live ledger (what is resident now, which
    /// residents could be evicted), the hardware profile and the model's
    /// calibration. Empty overrides give the one-click auto plan
    /// ([`planner::auto_plan`]).
    pub async fn plan_text(
        &self,
        model_id: String,
        overrides: PlanOverrides,
    ) -> Result<LoadPlan, AppError> {
        self.send(|reply| OrchestratorCmd::PlanText { model_id, overrides, reply }).await?
    }

    /// Tensor accounting for an installed model (cached per file), for the
    /// context-to-memory function the UI slider uses.
    pub async fn model_accounting(&self, model_id: String) -> Result<Arc<ModelAccounting>, AppError> {
        self.send(|reply| OrchestratorCmd::Accounting { model_id, reply }).await?
    }

    /// The plan the loaded text model was loaded with, with its model id.
    pub async fn current_text_plan(&self) -> Option<(String, LoadPlan)> {
        self.send(|reply| OrchestratorCmd::CurrentTextPlan { reply }).await.ok().flatten()
    }
}

const MIB: u64 = 1024 * 1024;

/// Cached footprint of the loaded text model.
#[derive(Debug, Clone)]
struct TextResident {
    model_id: String,
    label: String,
    est_mb: u64,
    #[allow(dead_code)]
    ngl: i32,
    plan: Option<LoadPlan>,
}

struct Actor {
    commands: mpsc::Receiver<OrchestratorCmd>,
    deferred: std::collections::VecDeque<OrchestratorCmd>,
    job_events: broadcast::Receiver<WorkerEvent>,
    worker_status: broadcast::Receiver<WorkerStatus>,
    loading_text: Option<TextResident>,
    text_epoch: u64,
    hardware_probe: Option<Arc<dyn Fn() -> planner::HardwareProfile + Send + Sync>>,
    db: SqlitePool,
    engine: EngineManagerHandle,
    worker: WorkerHandle,
    gpu: GpuHandle,
    tts: TtsHandle,
    bcast: broadcast::Sender<LedgerSnapshot>,
    /// Estimated footprint of the loaded text model.
    text_est: Option<TextResident>,
    /// Estimated footprint of the loaded TTS talker: (model_id, label, est_mb).
    tts_est: Option<(String, String, u64)>,
    /// task -> resident; one cached model per pipeline (mirrors worker-py).
    pipelines: HashMap<String, PipelineResident>,
    planner: PlannerHandle,
    /// Parsed GGUF accounting per model file (header reads are a few MB).
    accounting: accounting_cache::AccountingCache,
    /// Native chat's engine gate: held by a chat turn from setup to its last
    /// token. A user unload of the text slot needs it exclusively.
    chat: ChatActivity,
    /// Generation jobs in flight, held from reservation to terminal event.
    leases: JobLeases,
}

pub fn start(
    db: SqlitePool,
    engine: EngineManagerHandle,
    worker: WorkerHandle,
    gpu: GpuHandle,
    tts: TtsHandle,
    planner: PlannerHandle,
    chat: ChatActivity,
) -> OrchestratorHandle {
    start_with_hardware(db, engine, worker, gpu, tts, planner, chat, None)
}

// Probe injection is internal, like GpuProbe: tests drive real admission with
// deterministic machine pressure, never a fixture-seeded final plan.
pub(crate) fn start_with_hardware(
    db: SqlitePool, engine: EngineManagerHandle, worker: WorkerHandle,
    gpu: GpuHandle, tts: TtsHandle, planner: PlannerHandle, chat: ChatActivity,
    hardware_probe: Option<Arc<dyn Fn() -> planner::HardwareProfile + Send + Sync>>,
) -> OrchestratorHandle {
    let (tx, rx) = mpsc::channel::<OrchestratorCmd>(32);
    let (bcast_tx, _) = broadcast::channel::<LedgerSnapshot>(32);

    let handle = OrchestratorHandle {
        tx,
        bcast: bcast_tx.clone(),
        subscriber_abort: Arc::new(TokioMutex::new(None)),
    };

    let job_events = worker.subscribe_job_events();
    let worker_for_status = worker.clone();

    tauri::async_runtime::spawn(async move {
        let worker_status = worker_for_status.subscribe().await;
        let mut actor = Actor {
            commands: rx,
            deferred: Default::default(),
            job_events,
            worker_status,
            loading_text: None,
            text_epoch: 0,
            hardware_probe,
            db,
            engine,
            worker,
            gpu,
            tts,
            bcast: bcast_tx,
            text_est: None,
            tts_est: None,
            pipelines: HashMap::new(),
            planner,
            accounting: accounting_cache::AccountingCache::default(),
            chat,
            leases: JobLeases::default(),
        };

        loop {
            let cmd = if let Some(cmd) = actor.deferred.pop_front() { cmd } else {
                tokio::select! {
                    cmd = actor.commands.recv() => { let Some(cmd) = cmd else { break }; cmd }
                    ev = actor.job_events.recv() => {
                        match ev { Ok(ev) => actor.on_job_event(ev).await, Err(broadcast::error::RecvError::Lagged(_)) => {}, Err(_) => break }
                        continue;
                    }
                    st = actor.worker_status.recv() => {
                        match st { Ok(st) => actor.on_worker_status(st).await, Err(broadcast::error::RecvError::Lagged(_)) => {}, Err(_) => break }
                        continue;
                    }
                }
            };
            match cmd {
                OrchestratorCmd::LoadText { req, reply } => {
                    let result = actor.load_text(*req).await;
                    let _ = reply.send(result);
                    actor.broadcast().await;
                }
                cmd => actor.handle(cmd).await,
            }
        }
    });

    handle
}

impl Actor {
    async fn handle(&mut self, cmd: OrchestratorCmd) {
        match cmd {
            cmd @ OrchestratorCmd::LoadText { .. } => self.deferred.push_back(cmd),
            OrchestratorCmd::ReservePipeline { task, label, model_id, est_mb, force, lease, reply } => {
                let result = self.reserve_pipeline(task, label, model_id, est_mb, force).await;
                self.leases.grant_if_reserved(&result, lease);
                let _ = reply.send(result);
                self.broadcast().await;
            }
            OrchestratorCmd::LoadTts { model_id, force, lease, reply } => {
                let result = self.load_tts(model_id, force).await;
                self.leases.grant_if_reserved(&result, lease);
                let _ = reply.send(result);
                self.broadcast().await;
            }
            OrchestratorCmd::AbortLease { job_id, error, reply } => {
                if let Some(ev) = self.leases.abort(&job_id, &error) {
                    let _ = self.worker.job_events_sender().send(ev);
                }
                let _ = reply.send(());
            }
            OrchestratorCmd::ActiveJobs { reply } => {
                let _ = reply.send(self.leases.active_jobs(&self.db).await);
            }
            OrchestratorCmd::Unload { resident_id, reply } => {
                let result = self.unload(&resident_id).await;
                let _ = reply.send(result);
                self.broadcast().await;
            }
            OrchestratorCmd::UnloadIfIdle { resident_id, reply } => {
                let result = self.guarded_unload(&resident_id).await;
                let _ = reply.send(result);
                self.broadcast().await;
            }
            OrchestratorCmd::UnloadTextIfModel { model_id, reply } => {
                let result = if matches!(self.engine.status().await, EngineStatus::Ready { model_id: current, .. } if current == model_id) {
                    self.unload(TEXT_RESIDENT_ID).await
                } else { Ok(()) };
                let _ = reply.send(result);
                self.broadcast().await;
            }
            OrchestratorCmd::UnloadIfModel { resident_id, expected_model_id, reply } => {
                let snap = self.snapshot().await;
                let result = match unload_guard(&snap.residents, &resident_id, expected_model_id.as_deref()) {
                    UnloadGuard::Proceed => self.guarded_unload(&resident_id).await,
                    UnloadGuard::AlreadyGone => Ok(()),
                    UnloadGuard::Replaced { now } => Err(AppError::Other(format!(
                        "Nothing was unloaded: that slot now holds {now}."
                    ))),
                };
                let _ = reply.send(result);
                self.broadcast().await;
            }
            OrchestratorCmd::Ledger { reply } => {
                let _ = reply.send(self.snapshot().await);
            }
            OrchestratorCmd::Subscribe { reply } => {
                let _ = reply.send(self.bcast.subscribe());
            }
            OrchestratorCmd::PlanText { model_id, overrides, reply } => {
                let _ = reply.send(self.plan_text(&model_id, overrides).await);
            }
            OrchestratorCmd::Accounting { model_id, reply } => {
                let _ = reply.send(self.model_accounting(&model_id).await);
            }
            OrchestratorCmd::CurrentTextPlan { reply } => {
                let cur = self
                    .refreshed_text_est()
                    .await
                    .and_then(|t| t.plan.map(|p| (t.model_id, p)));
                let _ = reply.send(cur);
            }
        }
    }

    async fn on_job_event(&mut self, ev: WorkerEvent) {
        self.leases.on_event(&ev);
        if let WorkerEvent::JobDone { task: Some(task), meta, .. } = ev {
            if let Some(mb) = meta.get("vram_mb").and_then(|v| v.as_u64()) {
                // A preload whose reservation was already released (timeout /
                // cancel race) must not resurrect a resident the UI can't
                // correlate or unload by model id — update existing only.
                let is_preload = meta.get("preload").and_then(|v| v.as_bool()).unwrap_or(false);
                if is_preload {
                    if let Some(entry) = self.pipelines.get_mut(&task) {
                        entry.actual_mb = Some(mb);
                        self.broadcast().await;
                    }
                    return;
                }
                let entry = self
                    .pipelines
                    .entry(task.clone())
                    .or_insert_with(|| PipelineResident {
                        label: task,
                        model_id: None,
                        est_mb: mb,
                        actual_mb: None,
                    });
                entry.actual_mb = Some(mb);
                self.broadcast().await;
            }
        }
    }

    async fn on_worker_status(&mut self, st: WorkerStatus) {
        match st {
            WorkerStatus::NotInstalled
            | WorkerStatus::Stopped { .. }
            | WorkerStatus::Installing { .. }
            | WorkerStatus::Failed { .. } => {
                if !self.pipelines.is_empty() {
                    self.pipelines.clear();
                    self.broadcast().await;
                }
            }
            WorkerStatus::Starting | WorkerStatus::Ready { .. } => {}
        }
    }

    // ── load / reserve / unload ─────────────────────────────────────────

    async fn load_text(&mut self, req: TextLoadRequest) -> Result<LoadOutcome, AppError> {
        // A load restarts llama-server, which would cut off a streaming chat
        // reply. Held through the load, the gate also keeps a new turn from
        // starting mid-load; it is None while a turn holds the engine.
        let chat_gate = self.chat.exclusive();
        if let EngineStatus::Ready { model_id, port, ctx, .. } = self.engine.status().await {
            match text_swap(chat_gate.is_none(), Some(&model_id), &req.model_id, req.is_plain()) {
                SwapDecision::Proceed => {}
                SwapDecision::Refuse(reason) => return Err(AppError::InUse(reason.into())),
                SwapDecision::Reuse => {
                    let live = self.text_est.as_ref().filter(|t| t.model_id == model_id);
                    let ngl = live.map_or(-1, |t| t.ngl);
                    let plan = live.and_then(|t| t.plan.clone()).map(Box::new);
                    return Ok(LoadOutcome::Loaded { port, model_id, ctx, ngl, plan });
                }
            }
        }

        let model = registry::get_by_id(&self.db, &req.model_id)
            .await?
            .ok_or_else(|| AppError::ModelNotFound(req.model_id.clone()))?;
        let path = PathBuf::from(&model.primary_file_path);

        // GGUF header → fit inputs (SWA-aware KV via FitInputs::from_meta). An
        // unparseable header skips the gate and falls back to full offload
        // (llama.cpp degrades gracefully itself).
        let meta = gguf::parse_header(&path).ok();
        let inference_mode = crate::engine::sidecar::InferenceMode::from_workflows(&model.caps.workflows);
        inference_mode.validate_pooling(meta.as_ref())?;

        if req.ngl.is_none() {
            if let Some(outcome) = self
                .load_text_planned(&req, &model.display_name, &path, inference_mode)
                .await?
            {
                return Ok(outcome);
            }
            // Unreadable tensor table: fall through to the legacy estimate.
        }

        let ctx = req
            .ctx
            .unwrap_or_else(|| (model.context_train.unwrap_or(8192) as u32).min(8192));

        let inputs = meta.map(|meta| {
            FitInputs::from_meta(&meta, model.weight_size() / (1024 * 1024), ctx, 2)
        });

        let metrics = self.gpu.latest_metrics().await;

        let ngl = match (&inputs, &metrics) {
            (Some(i), Some(m)) => {
                // Same-slot swap: the current text model's footprint is
                // reclaimed by the load itself.
                let text_reclaim = self.refreshed_text_est().await.map_or(0, |t| t.est_mb);
                let free = m.vram_free_mb + text_reclaim;
                let pipeline_reclaim: u64 =
                    self.pipelines.values().map(|p| p.actual_mb.unwrap_or(p.est_mb)).sum();
                let tts_reclaim = self.refreshed_tts_est().await.map_or(0, |(_, _, mb)| mb);

                match text_gate(i, req.ngl, free, pipeline_reclaim + tts_reclaim, req.force) {
                    Ok(ngl) => ngl,
                    Err(required_mb) => {
                        let mut candidates = self.pipeline_residents();
                        if let Some(info) = self.tts_resident().await {
                            candidates.push(info);
                        }
                        return Ok(LoadOutcome::NeedsEviction {
                            required_mb,
                            free_mb: free,
                            candidates,
                        });
                    }
                }
            }
            // No GPU: CPU inference unless explicitly overridden.
            (_, None) => req.ngl.unwrap_or(0),
            // No parseable header: explicit or full offload.
            (None, _) => req.ngl.unwrap_or(-1),
        };

        let args = ModelLoadArgs {
            engine_kind: planner::EngineKind::LlamaServer,
            inference_mode,
            ctx,
            ngl,
            batch_size: req.batch_size,
            threads: req.threads,
            flash_attn: req.flash_attn.unwrap_or(false),
            kv_cache_type: req.kv_cache_type.clone().filter(|s| !s.is_empty()),
            mmap: req.mmap.unwrap_or(true),
            mlock: req.mlock.unwrap_or(false),
            // Live tok/s for the runtime stats panel.
            extra_args: vec!["--metrics".into()],
        };

        let est = inputs.as_ref().map(|i| est_for_ngl(i, ngl)).unwrap_or(0);
        let pending = TextResident { model_id: req.model_id.clone(), label: model.display_name.clone(), est_mb: est, ngl, plan: None };
        let port = self.await_text_engine(req.model_id.clone(), path, args, pending).await?;
        self.text_est = Some(TextResident {
            model_id: req.model_id.clone(),
            label: model.display_name.clone(),
            est_mb: est,
            ngl,
            plan: None,
        });

        Ok(LoadOutcome::Loaded { port, model_id: req.model_id, ctx, ngl, plan: None })
    }

    // ── planner ─────────────────────────────────────────────────────────

    /// Tensor accounting, reused only while every shard's size and mtime match.
    async fn accounting_for(&mut self, path: &Path) -> Option<Arc<ModelAccounting>> {
        self.accounting.get(path).await
    }

    /// Graphics memory a text load may use: free now plus the current text
    /// model's footprint (same-slot swap), with pipelines and the TTS
    /// sidecar as evictable residents. Returns (budget, free_mb).
    async fn text_budget(&mut self) -> (VramBudget, u64) {
        let Some(m) = self.gpu.latest_metrics().await else {
            return (VramBudget::no_gpu(), 0);
        };
        let active = crate::db::engines::get_active(&self.db).await.ok().flatten();
        if !engine_uses_gpu(active.as_ref().map(|r| r.variant.as_str())) {
            return (VramBudget::no_gpu(), 0);
        }
        let text_reclaim = self.refreshed_text_est().await.map_or(0, |t| t.est_mb);
        let free_mb = m.vram_free_mb + text_reclaim;
        let mut reclaimable: Vec<Reclaimable> = self
            .pipeline_residents()
            .into_iter()
            .map(|r| Reclaimable {
                bytes: r.actual_mb.unwrap_or(r.est_mb) * MIB,
                id: r.id,
                label: r.label,
            })
            .collect();
        if let Some(t) = self.tts_resident().await {
            reclaimable.push(Reclaimable { id: t.id, label: t.label, bytes: t.est_mb * MIB });
        }
        reclaimable.retain(|r| r.bytes > 0);
        (
            VramBudget { total_bytes: m.vram_total_mb * MIB, free_bytes: free_mb * MIB, reclaimable },
            free_mb,
        )
    }

    /// Run the planner for a model file. `None` when its header can't be read.
    async fn plan_for(
        &mut self,
        model_id: &str,
        path: &Path,
        overrides: &PlanOverrides,
    ) -> Option<(LoadPlan, u64)> {
        let acc = self.accounting_for(path).await?;
        let (budget, free_mb) = self.text_budget().await;
        let metrics = self.gpu.latest_metrics().await;
        let ssd = self.planner.ssd_measurements().await;
        let dir = path.parent().map(Path::to_path_buf).unwrap_or_else(crate::config::models_dir);
        let mut hw = if let Some(probe) = &self.hardware_probe { probe() } else {
            tokio::task::spawn_blocking(move || planner::hardware::probe(metrics.as_ref(), &dir, &ssd)).await.ok()?
        };
        // A load replaces the running text process, so what stopping it gives
        // back counts as available: its private RAM, and its private commit
        // against the commit limit (never as RAM: commit also covers its
        // graphics-memory tier).
        let credit = if self.loading_text.is_none() { self.engine.reclaimable_host().await } else { Default::default() };
        hw.ram_available_bytes = hw.ram_available_bytes.saturating_add(credit.resident).min(hw.ram_total_bytes);
        hw.commit_available_bytes = hw.commit_available_bytes.map(|c| c.saturating_add(credit.commit));
        // And its graphics memory, which the native planner reads from this
        // profile (llama's plan gets the same credit through `budget`). Only the
        // text process: pipelines and the TTS sidecar keep their charge.
        if self.loading_text.is_none() {
            let text_vram = self.refreshed_text_est().await.map_or(0, |t| t.est_mb) * MIB;
            if let Some(g) = hw.gpu.as_mut() {
                g.vram_free_bytes = g.vram_free_bytes.saturating_add(text_vram).min(g.vram_total_bytes);
            }
        }
        let cal = self.planner.calibration(model_id).await;
        let native = self.engine.native_info().await.ok()?.support();
        let native_cal = self.planner.calibration(&format!("{model_id}:native-ssd")).await;
        let ov = overrides.clone();
        let plan = tokio::task::spawn_blocking(move || {
            planner::plan_with_native(&acc, &hw, &budget, &ov, Some(&cal), &native, Some(&native_cal))
        })
        .await
        .ok()?;
        Some((plan, free_mb))
    }

    async fn plan_text(
        &mut self,
        model_id: &str,
        overrides: PlanOverrides,
    ) -> Result<LoadPlan, AppError> {
        let model = registry::get_by_id(&self.db, model_id)
            .await?
            .ok_or_else(|| AppError::ModelNotFound(model_id.to_string()))?;
        let path = PathBuf::from(&model.primary_file_path);
        self.plan_for(model_id, &path, &overrides).await.map(|(p, _)| p).ok_or_else(|| {
            AppError::Other(format!(
                "Can't plan '{}': its GGUF tensor table couldn't be read.",
                model.display_name
            ))
        })
    }

    async fn model_accounting(&mut self, model_id: &str) -> Result<Arc<ModelAccounting>, AppError> {
        let model = registry::get_by_id(&self.db, model_id)
            .await?
            .ok_or_else(|| AppError::ModelNotFound(model_id.to_string()))?;
        let path = PathBuf::from(&model.primary_file_path);
        self.accounting_for(&path).await.ok_or_else(|| {
            AppError::Other(format!(
                "Can't size '{}': its GGUF tensor table couldn't be read.",
                model.display_name
            ))
        })
    }

    /// Planner-driven load. `Ok(None)` when the model can't be planned (the
    /// caller falls back to the legacy estimate).
    async fn load_text_planned(
        &mut self,
        req: &TextLoadRequest,
        label: &str,
        path: &Path,
        inference_mode: crate::engine::sidecar::InferenceMode,
    ) -> Result<Option<LoadOutcome>, AppError> {
        // Re-resolve the choices against live memory and binary capabilities.
        // A plan sent by the UI is a preview, not a memory reservation.
        let (plan, free_mb) = match &req.plan {
            // Recompute from choices, not a client-supplied envelope. This also
            // upgrades an old llama paging plan to native admission and refreshes
            // host RAM, which can change while the context slider is open.
            Some(p) => match self.plan_for(&req.model_id, path, &p.overrides).await {
                Some(x) => x,
                None => return Ok(None),
            },
            None => match self.plan_for(&req.model_id, path, &req.planner_overrides()).await {
                Some(x) => x,
                None => return Ok(None),
            },
        };

        let reason = plan.infeasible_reason.clone().or_else(|| plan.warnings.last().cloned());
        match planned_gate(
            plan.feasible,
            reason.as_deref(),
            plan.evict_alternative.is_some(),
            req.force,
            req.decline_eviction_offer,
        ) {
            PlannedGate::Proceed => {}
            PlannedGate::Reject(why) => return Err(AppError::Other(why)),
            PlannedGate::OfferEviction => {
                let alt = plan.evict_alternative.as_ref().expect("offer implies an alternative");
                let mut candidates = self.pipeline_residents();
                if let Some(info) = self.tts_resident().await {
                    candidates.push(info);
                }
                candidates.retain(|c| alt.evicts.iter().any(|e| e.id == c.id));
                return Ok(Some(LoadOutcome::NeedsEviction {
                    required_mb: alt.vram_bytes.div_ceil(MIB),
                    free_mb,
                    candidates,
                }));
            }
        }

        let native = plan.engine_kind == planner::EngineKind::SsdStream;
        if native && inference_mode != crate::engine::sidecar::InferenceMode::Generation {
            return Err(AppError::Other("Native expert streaming supports text generation, not embeddings.".into()));
        }
        let args = ModelLoadArgs {
            engine_kind: plan.engine_kind,
            // Embedding models keep `--embeddings` on the planned path too.
            inference_mode,
            ctx: plan.ctx,
            ngl: plan.gpu_layers,
            batch_size: if native { Some(plan.ubatch) } else { req.batch_size.or(Some(plan.ubatch.max(2048))) },
            threads: req.threads,
            flash_attn: !native,
            // `-ctk`/`-ctv` travel in `extra_args` because K and V may differ.
            kv_cache_type: None,
            // Weights beyond RAM are paged from the SSD through the mapping.
            mmap: true,
            mlock: false,
            extra_args: plan.extra_args(),
        };
        let pending = TextResident { model_id: req.model_id.clone(), label: label.into(), est_mb: plan.tiers.vram_total().div_ceil(MIB), ngl: plan.gpu_layers, plan: Some(plan.clone()) };
        let port = self.await_text_engine(req.model_id.clone(), path.to_path_buf(), args, pending).await?;

        self.text_est = Some(TextResident {
            model_id: req.model_id.clone(),
            label: label.to_string(),
            est_mb: plan.tiers.vram_total().div_ceil(MIB),
            ngl: plan.gpu_layers,
            plan: Some(plan.clone()),
        });
        Ok(Some(LoadOutcome::Loaded {
            port,
            model_id: req.model_id.clone(),
            ctx: plan.ctx,
            ngl: plan.gpu_layers,
            plan: Some(Box::new(plan)),
        }))
    }

    /// The child remains owned by the engine actor. Only its request future is
    /// polled here; no detached writer gets access to either actor's state.
    async fn await_text_engine(&mut self, id: String, path: PathBuf, args: ModelLoadArgs, pending: TextResident) -> Result<u16, AppError> {
        let load = self.engine.begin_load(id, path, args).await?;
        tokio::pin!(load);
        let epoch = self.text_epoch;
        self.loading_text = Some(pending);
        let result = loop {
            tokio::select! {
                biased;
                cmd = self.commands.recv() => match cmd {
                    Some(cmd @ (OrchestratorCmd::LoadText { .. } | OrchestratorCmd::ReservePipeline { .. } | OrchestratorCmd::LoadTts { .. })) => self.deferred.push_back(cmd),
                    Some(cmd) => self.handle(cmd).await,
                    None => { let _ = self.engine.unload_model().await; break Err(AppError::EngineLoadCancelled); }
                },
                ev = self.job_events.recv() => match ev {
                    Ok(ev) => self.on_job_event(ev).await,
                    Err(broadcast::error::RecvError::Lagged(_)) => {},
                    Err(_) => { let _ = self.engine.unload_model().await; break Err(AppError::EngineLoadCancelled); }
                },
                st = self.worker_status.recv() => match st {
                    Ok(st) => self.on_worker_status(st).await,
                    Err(broadcast::error::RecvError::Lagged(_)) => {},
                    Err(_) => { let _ = self.engine.unload_model().await; break Err(AppError::EngineLoadCancelled); }
                },
                result = &mut load => break result.unwrap_or_else(|_| Err(AppError::Other("engine manager dropped reply".into()))),
            }
        };
        self.loading_text = None;
        // An unload may have stopped a just-ready server while its successful
        // reply was buffered. Never resurrect that plan/ledger entry.
        let result = if self.text_epoch != epoch { Err(AppError::EngineLoadCancelled) } else { result };
        if result.is_err() { self.text_est = None; }
        result
    }

    async fn reserve_pipeline(
        &mut self,
        task: String,
        label: String,
        model_id: Option<String>,
        est_mb: u64,
        force: bool,
    ) -> Result<ReserveOutcome, AppError> {
        if let Some(m) = self.gpu.latest_metrics().await {
            // Same-task swap reclaims that task's current model.
            let own_reclaim = self
                .pipelines
                .get(&task)
                .map_or(0, |p| p.actual_mb.unwrap_or(p.est_mb));
            let free = m.vram_free_mb + own_reclaim;

            let text_reclaim = self.refreshed_text_est().await.map_or(0, |t| t.est_mb);
            let tts_reclaim = self.refreshed_tts_est().await.map_or(0, |(_, _, mb)| mb);
            let other_pipelines: u64 = self
                .pipelines
                .iter()
                .filter(|(t, _)| **t != task)
                .map(|(_, p)| p.actual_mb.unwrap_or(p.est_mb))
                .sum();

            if let Err(required_mb) =
                pipeline_gate(est_mb, free, text_reclaim + tts_reclaim + other_pipelines, force)
            {
                let mut candidates = Vec::new();
                if let Some(info) = self.text_resident().await {
                    candidates.push(info);
                }
                if let Some(info) = self.tts_resident().await {
                    candidates.push(info);
                }
                candidates.extend(
                    self.pipeline_residents().into_iter().filter(|r| r.id != pipeline_id(&task)),
                );
                return Ok(ReserveOutcome::NeedsEviction {
                    required_mb,
                    free_mb: free,
                    candidates,
                });
            }
        }

        // A re-reserve of the same task keeps the NVML actual only when the
        // same model stays resident (same-model repeat jobs); a swap resets it.
        let actual_mb = self
            .pipelines
            .get(&task)
            .filter(|p| p.label == label)
            .and_then(|p| p.actual_mb);
        self.pipelines
            .insert(task, PipelineResident { label, model_id, est_mb, actual_mb });
        Ok(ReserveOutcome::Reserved)
    }

    /// Gate + load a Qwen3-TTS talker via the tts sidecar actor. Mirrors
    /// `reserve_pipeline`'s gate (file-size estimate, same-slot reclaim),
    /// but the load is awaited inline like `load_text` — `Reserved` means
    /// the server answered `/health`.
    async fn load_tts(
        &mut self,
        model_id: String,
        force: bool,
    ) -> Result<ReserveOutcome, AppError> {
        // Already serving this exact model → nothing to gate. A different
        // talker stops the running server first (`TtsActor::load`), so it
        // must not replace one a synthesis holds, even before its row exists.
        let current = match self.tts.status().await {
            TtsStatus::Ready { model_id: cur, .. } => Some(cur),
            _ => None,
        };
        let leased = current.is_some() && self.leases.holds(&self.db, TTS_RESIDENT_ID).await;
        match tts_swap(current.as_deref(), &model_id, leased) {
            SwapDecision::Reuse => return Ok(ReserveOutcome::Reserved),
            SwapDecision::Refuse(reason) => return Err(AppError::InUse(reason.into())),
            SwapDecision::Proceed => {}
        }

        let model = registry::get_by_id(&self.db, &model_id)
            .await?
            .ok_or_else(|| AppError::ModelNotFound(model_id.clone()))?;
        if model.architecture.as_deref() != Some(crate::tts::TALKER_ARCH) {
            return Err(AppError::Other(format!(
                "'{}' is not a Qwen3-TTS talker model (architecture {:?})",
                model.display_name, model.architecture
            )));
        }
        let talker = PathBuf::from(&model.primary_file_path);
        let codec = crate::tts::bundled_codec_path();
        if !codec.exists() {
            // The runtime isn't provisioned (codec is one of its
            // required_files). Kick off the lazy pin-and-fetch install in the
            // background — progress streams to the settings card — and ask the
            // user to retry once it's ready. (`tts.load()` would also
            // auto-install, but doing it here keeps the gate/estimate honest:
            // we don't reserve VRAM for a load that can't happen yet.)
            self.tts.install_in_background();
            return Err(AppError::Other(
                "Qwen3-TTS engine isn't installed yet — installing it now. Watch progress in \
                 Settings → Engines, then retry. (If the install fails, the binaries are not \
                 yet hosted — see CLAUDE.md.)"
                    .into(),
            ));
        }
        let codec_mb = codec.metadata().map(|m| m.len() / (1024 * 1024)).unwrap_or(300);
        let est_mb = model.weight_size() / (1024 * 1024)
            + codec_mb
            + crate::tts::QWEN_TTS_OVERHEAD_MB;

        if let Some(m) = self.gpu.latest_metrics().await {
            // Slot swap reclaims the currently loaded talker's footprint.
            let own_reclaim = self.refreshed_tts_est().await.map_or(0, |(_, _, mb)| mb);
            let free = m.vram_free_mb + own_reclaim;

            let text_reclaim = self.refreshed_text_est().await.map_or(0, |t| t.est_mb);
            let pipeline_reclaim: u64 =
                self.pipelines.values().map(|p| p.actual_mb.unwrap_or(p.est_mb)).sum();

            if let Err(required_mb) =
                pipeline_gate(est_mb, free, text_reclaim + pipeline_reclaim, force)
            {
                let mut candidates = Vec::new();
                if let Some(info) = self.text_resident().await {
                    candidates.push(info);
                }
                candidates.extend(self.pipeline_residents());
                return Ok(ReserveOutcome::NeedsEviction {
                    required_mb,
                    free_mb: free,
                    candidates,
                });
            }
        }

        self.tts.load(model_id.clone(), talker, codec).await?;
        self.tts_est = Some((model_id, model.display_name, est_mb));
        Ok(ReserveOutcome::Reserved)
    }

    async fn unload(&mut self, resident_id: &str) -> Result<(), AppError> {
        if resident_id == TEXT_RESIDENT_ID {
            self.engine.unload_model().await?;
            self.text_epoch = self.text_epoch.wrapping_add(1);
            self.loading_text = None;
            self.text_est = None;
            return Ok(());
        }
        if resident_id == TTS_RESIDENT_ID {
            self.tts.unload().await?;
            self.tts_est = None;
            return Ok(());
        }
        if let Some(task) = resident_id.strip_prefix("pipeline:") {
            // Best-effort: a stopped worker has nothing resident anyway.
            let _ = self
                .worker
                .request(
                    WorkerRequest::Unload { task: Some(task.to_string()) },
                    Duration::from_secs(30),
                )
                .await;
            self.pipelines.remove(task);
            return Ok(());
        }
        Err(AppError::Other(format!("unknown resident id: {resident_id}")))
    }

    /// A user-requested unload. The backend is the authority on "in use": a
    /// reloaded UI has lost its own view of what is running, so this refuses
    /// while a chat turn holds the text engine or a running generation job
    /// may be using the resident. The chat gate stays held through the unload
    /// so no turn can start in the gap.
    async fn guarded_unload(&mut self, resident_id: &str) -> Result<(), AppError> {
        // During startup the load itself holds chat's exclusive gate. That is
        // not an active chat turn: allow the user to cancel their pending load.
        let chat_gate = (resident_id == TEXT_RESIDENT_ID && self.loading_text.is_none()).then(|| self.chat.exclusive());
        let chat_busy = matches!(chat_gate, Some(None));
        let running = running_job_tasks(&self.db).await;
        let leased = self.leases.holds(&self.db, resident_id).await;
        if let Some(reason) = in_use_reason(resident_id, chat_busy, leased, &running) {
            return Err(AppError::InUse(reason.into()));
        }
        self.unload(resident_id).await
    }

    // ── ledger composition ──────────────────────────────────────────────

    /// The cached text estimate, validated against live engine status (drops
    /// the cache when the engine is idle/failed or the model changed under us).
    async fn refreshed_text_est(&mut self) -> Option<TextResident> {
        match self.engine.status().await {
            EngineStatus::Ready { model_id, .. } => match &self.text_est {
                Some(t) if t.model_id == model_id => self.text_est.clone(),
                // Loaded outside the orchestrator (or before it existed) —
                // keep it visible with an unknown footprint. The ledger still
                // correlates it back to its catalog row by id.
                _ => {
                    let label = registry::get_by_id(&self.db, &model_id)
                        .await
                        .ok()
                        .flatten()
                        .map(|m| m.display_name)
                        .unwrap_or_else(|| model_id.clone());
                    self.text_est = Some(TextResident {
                        model_id,
                        label,
                        est_mb: 0,
                        ngl: -1,
                        plan: None,
                    });
                    self.text_est.clone()
                }
            },
            EngineStatus::Idle | EngineStatus::Failed { .. } => {
                self.text_est = None;
                None
            }
            EngineStatus::Loading { model_id, .. } => self.loading_text.as_ref().filter(|t| t.model_id == model_id).cloned(),
            _ => self.text_est.clone(),
        }
    }

    async fn text_resident(&mut self) -> Option<ResidentInfo> {
        self.refreshed_text_est().await.map(|t| ResidentInfo {
            id: TEXT_RESIDENT_ID.into(),
            kind: ResidentKind::TextEngine,
            label: t.label,
            model_id: Some(t.model_id),
            est_mb: t.est_mb,
            actual_mb: None,
        })
    }

    /// The cached TTS estimate, validated against the live sidecar status
    /// (drops the cache when the server is gone or swapped under us).
    async fn refreshed_tts_est(&mut self) -> Option<(String, String, u64)> {
        match self.tts.status().await {
            TtsStatus::Ready { model_id, .. } => match &self.tts_est {
                Some((cached_id, _, _)) if *cached_id == model_id => self.tts_est.clone(),
                _ => {
                    self.tts_est = Some((model_id.clone(), model_id, 0));
                    self.tts_est.clone()
                }
            },
            TtsStatus::Loading { .. } => self.tts_est.clone(),
            _ => {
                self.tts_est = None;
                None
            }
        }
    }

    async fn tts_resident(&mut self) -> Option<ResidentInfo> {
        self.refreshed_tts_est().await.map(|(model_id, label, est_mb)| ResidentInfo {
            id: TTS_RESIDENT_ID.into(),
            kind: ResidentKind::TtsEngine,
            label,
            model_id: Some(model_id),
            est_mb,
            actual_mb: None,
        })
    }

    fn pipeline_residents(&self) -> Vec<ResidentInfo> {
        let mut v: Vec<ResidentInfo> = self
            .pipelines
            .iter()
            .map(|(task, p)| ResidentInfo {
                id: pipeline_id(task),
                kind: ResidentKind::Pipeline,
                label: p.label.clone(),
                model_id: p.model_id.clone(),
                est_mb: p.est_mb,
                actual_mb: p.actual_mb,
            })
            .collect();
        v.sort_by(|a, b| a.id.cmp(&b.id));
        v
    }

    async fn snapshot(&mut self) -> LedgerSnapshot {
        let metrics = self.gpu.latest_metrics().await;
        let mut residents = Vec::new();
        if let Some(info) = self.text_resident().await {
            residents.push(info);
        }
        if let Some(info) = self.tts_resident().await {
            residents.push(info);
        }
        residents.extend(self.pipeline_residents());
        match metrics {
            Some(m) => LedgerSnapshot {
                gpu: true,
                total_mb: m.vram_total_mb,
                free_mb: m.vram_free_mb,
                residents,
            },
            None => LedgerSnapshot { gpu: false, total_mb: 0, free_mb: 0, residents },
        }
    }

    async fn broadcast(&mut self) {
        let snap = self.snapshot().await;
        let residents: Vec<(String, Option<String>)> = snap
            .residents
            .iter()
            .map(|r| (r.id.clone(), r.model_id.clone()))
            .collect();
        tracing::info!(target: "ledger", ?residents, "ledger broadcast");
        let _ = self.bcast.send(snap);
    }
}

fn pipeline_id(task: &str) -> String {
    format!("pipeline:{task}")
}

const CHAT_IN_USE: &str =
    "A chat reply is streaming from this model. Stop it in Chat before unloading.";
const JOB_IN_USE: &str =
    "A generation job is using this model. Let it finish or cancel it before unloading.";
const CHAT_SWAP_IN_USE: &str =
    "A chat reply is streaming from the loaded model. Stop it in Chat before loading another model.";
const TTS_SWAP_IN_USE: &str =
    "A speech job is using the loaded voice model. Let it finish or cancel it before loading another one.";

/// What a load into an occupied slot should do. Loading a different model
/// stops the server that is running first, so it waits while that server is
/// in use, like an unload does.
#[derive(Debug, PartialEq, Eq)]
enum SwapDecision {
    /// The requested model already serves: hand back what is running.
    Reuse,
    /// Nothing in use would be stopped.
    Proceed,
    /// The swap would stop work in flight.
    Refuse(&'static str),
}

/// The `"tts"` slot. `current`: the talker serving now. `leased`: a live
/// synthesis lease holds the slot. Same-model loads reuse the running server.
fn tts_swap(current: Option<&str>, requested: &str, leased: bool) -> SwapDecision {
    match current {
        Some(cur) if cur == requested => SwapDecision::Reuse,
        Some(_) if leased => SwapDecision::Refuse(TTS_SWAP_IN_USE),
        _ => SwapDecision::Proceed,
    }
}

/// The text slot. `chat_busy`: a native chat turn holds the engine. Every
/// llama-server load restarts the server, even for the same model, so while
/// a reply streams only a plain request for the serving model is answered
/// (from what is running); anything else is refused.
fn text_swap(chat_busy: bool, current: Option<&str>, requested: &str, plain: bool) -> SwapDecision {
    match current {
        Some(_) if !chat_busy => SwapDecision::Proceed,
        Some(cur) if cur == requested && plain => SwapDecision::Reuse,
        Some(_) => SwapDecision::Refuse(CHAT_SWAP_IN_USE),
        None => SwapDecision::Proceed,
    }
}

/// Every resident a speech job might be on. Speech rows are routed by engine
/// (Kokoro pipeline, Qwen3-TTS sidecar or clone CLI, VibeVoice), so a speech
/// row conservatively occupies all of them, as the UI's guard does.
const SPEECH_RESIDENTS: [&str; 4] = [
    TTS_RESIDENT_ID,
    "pipeline:text_to_speech",
    "pipeline:qwen_tts_clone",
    "pipeline:vibevoice",
];

/// Whether a running job row of `task` may be using `resident_id`.
fn job_occupies(task: &str, resident_id: &str) -> bool {
    match task {
        "text_to_speech" | "vibevoice" => SPEECH_RESIDENTS.contains(&resident_id),
        t => resident_id.strip_prefix("pipeline:") == Some(t),
    }
}

/// Tasks of the running generation jobs. Rows go `running` before the submit
/// and leave it on the job's real terminal event (worker exit reports one per
/// accepted job), and startup healing clears rows a previous process left
/// behind before any command is served. Read with no history cutoff. A
/// failed read fails open so a database hiccup can't pin a model forever;
/// job leases still hold the resident.
async fn running_job_tasks(db: &SqlitePool) -> Vec<String> {
    match generation_jobs::list_running(db).await {
        Ok(jobs) => jobs.into_iter().map(|(_, task)| task).collect(),
        Err(e) => {
            tracing::warn!(target: "unload", "active-job check failed: {e}");
            Vec::new()
        }
    }
}

/// Why a user unload of `resident_id` must wait, if it must. `chat_busy`:
/// a native chat turn holds the text engine. `leased`: a job lease holds
/// this resident. `running_tasks`: the tasks of running generation jobs.
fn in_use_reason(
    resident_id: &str,
    chat_busy: bool,
    leased: bool,
    running_tasks: &[String],
) -> Option<&'static str> {
    if resident_id == TEXT_RESIDENT_ID && chat_busy {
        return Some(CHAT_IN_USE);
    }
    if leased {
        return Some(JOB_IN_USE);
    }
    running_tasks
        .iter()
        .any(|t| job_occupies(t, resident_id))
        .then_some(JOB_IN_USE)
}

/// What a conditional unload should do, decided against the live residents.
#[derive(Debug, PartialEq, Eq)]
enum UnloadGuard {
    /// The slot still holds the model the caller meant.
    Proceed,
    /// The slot is empty; nothing to do.
    AlreadyGone,
    /// The slot holds a different model now (label of what's there).
    Replaced { now: String },
}

fn unload_guard(residents: &[ResidentInfo], resident_id: &str, expected: Option<&str>) -> UnloadGuard {
    match residents.iter().find(|r| r.id == resident_id) {
        None => UnloadGuard::AlreadyGone,
        Some(r) if r.model_id.as_deref() == expected => UnloadGuard::Proceed,
        Some(r) => UnloadGuard::Replaced { now: r.label.clone() },
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[test]
    fn force_never_loads_an_infeasible_plan() {
        // Force skips the eviction offer...
        assert_eq!(super::planned_gate(true, None, true, true, false), super::PlannedGate::Proceed);
        assert_eq!(super::planned_gate(true, None, true, false, false), super::PlannedGate::OfferEviction);
        // ...but never the memory check.
        let why = "A 131072 token context doesn't fit.";
        assert_eq!(
            super::planned_gate(false, Some(why), false, true, false),
            super::PlannedGate::Reject(why.into())
        );
        assert_eq!(
            super::planned_gate(false, Some(why), true, true, true),
            super::PlannedGate::Reject(why.into())
        );
        // Without force an infeasible plan with a fix on offer asks first.
        assert_eq!(super::planned_gate(false, Some(why), true, false, true), super::PlannedGate::OfferEviction);
        assert_eq!(super::planned_gate(true, None, true, false, true), super::PlannedGate::Proceed);
    }

    #[test]
    fn cpu_engine_build_gets_no_graphics_memory() {
        assert!(!super::engine_uses_gpu(Some("cpu")));
        assert!(super::engine_uses_gpu(Some("cuda-12.4")));
        assert!(super::engine_uses_gpu(None));
    }

    use super::*;

    fn resident(id: &str, model_id: Option<&str>, label: &str) -> ResidentInfo {
        ResidentInfo {
            id: id.into(),
            kind: ResidentKind::TextEngine,
            label: label.into(),
            model_id: model_id.map(Into::into),
            est_mb: 1000,
            actual_mb: None,
        }
    }

    #[test]
    fn unload_guard_matches_expected_model() {
        let rs = vec![resident("text", Some("a"), "Model A")];
        assert_eq!(unload_guard(&rs, "text", Some("a")), UnloadGuard::Proceed);
    }

    #[test]
    fn unload_guard_refuses_when_slot_was_replaced() {
        // A's chip was clicked while B replaced it; B has since loaded.
        let rs = vec![resident("text", Some("b"), "Model B")];
        assert_eq!(
            unload_guard(&rs, "text", Some("a")),
            UnloadGuard::Replaced { now: "Model B".into() }
        );
    }

    #[test]
    fn unload_guard_is_noop_for_empty_slot() {
        let rs = vec![resident("pipeline:text_to_image", Some("x"), "SDXL")];
        assert_eq!(unload_guard(&rs, "text", Some("a")), UnloadGuard::AlreadyGone);
    }

    #[test]
    fn unload_guard_uncorrelated_matches_only_uncorrelated() {
        let rs = vec![resident("text", None, "adopted.gguf")];
        assert_eq!(unload_guard(&rs, "text", None), UnloadGuard::Proceed);
        let rs = vec![resident("text", Some("b"), "Model B")];
        assert_eq!(
            unload_guard(&rs, "text", None),
            UnloadGuard::Replaced { now: "Model B".into() }
        );
    }

    #[test]
    fn speech_jobs_occupy_every_speech_slot_only() {
        for slot in SPEECH_RESIDENTS {
            assert!(job_occupies("text_to_speech", slot), "{slot}");
            assert!(job_occupies("vibevoice", slot), "{slot}");
        }
        assert!(!job_occupies("text_to_speech", "pipeline:transcribe"));
        assert!(!job_occupies("text_to_speech", TEXT_RESIDENT_ID));
        assert!(job_occupies("transcribe", "pipeline:transcribe"));
        assert!(!job_occupies("transcribe", TTS_RESIDENT_ID));
    }

    async fn job_pool() -> SqlitePool {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    /// A reloaded UI has no record of a job it didn't submit, so the backend
    /// must refuse on its own: a running Qwen3-TTS row keeps the "tts" slot
    /// in use until the row reaches a terminal state.
    #[tokio::test]
    async fn running_job_row_refuses_unload_until_terminal() {
        let pool = job_pool().await;
        let params = serde_json::json!({"engine": "qwen3-tts", "text": "hi"});
        generation_jobs::insert(&pool, "j1", "text_to_speech", &params).await.unwrap();

        let running = running_job_tasks(&pool).await;
        assert_eq!(in_use_reason(TTS_RESIDENT_ID, false, false, &running), Some(JOB_IN_USE));
        assert_eq!(in_use_reason("pipeline:text_to_image", false, false, &running), None);

        // Worker exit and real failures both land as mark_error via job sync.
        generation_jobs::mark_error(&pool, "j1", "worker process exited").await.unwrap();
        let running = running_job_tasks(&pool).await;
        assert_eq!(in_use_reason(TTS_RESIDENT_ID, false, false, &running), None);
    }

    /// An old long-running job stays visible however many newer rows exist
    /// (the check used to read only the newest 200). The id sorts last, so a
    /// newest-first window would cut it.
    #[tokio::test]
    async fn old_running_job_is_seen_past_any_history_window() {
        let pool = job_pool().await;
        let params = serde_json::json!({});
        generation_jobs::insert(&pool, "a-old-video", "text_to_video", &params).await.unwrap();
        for i in 0..250 {
            let id = format!("z-{i:03}");
            generation_jobs::insert(&pool, &id, "text_to_image", &params).await.unwrap();
            generation_jobs::mark_done(&pool, &id, &[], &params).await.unwrap();
        }
        let running = running_job_tasks(&pool).await;
        assert_eq!(running, vec!["text_to_video".to_string()]);
        assert_eq!(in_use_reason("pipeline:text_to_video", false, false, &running), Some(JOB_IN_USE));
    }

    /// A lease alone refuses the unload, with no job row yet (the submit race).
    #[test]
    fn lease_refuses_unload_without_a_job_row() {
        assert_eq!(in_use_reason(TTS_RESIDENT_ID, false, true, &[]), Some(JOB_IN_USE));
        assert_eq!(in_use_reason(TTS_RESIDENT_ID, false, false, &[]), None);
    }

    /// The chat gate is held by a native turn from setup to its last token;
    /// a text unload is refused while it is held and allowed once released.
    #[tokio::test]
    async fn chat_turn_refuses_text_unload() {
        let (writer, activity) = crate::chat::activity::channel();
        let turn = writer.reserve().await;
        assert!(activity.exclusive().is_none());
        assert_eq!(in_use_reason(TEXT_RESIDENT_ID, true, false, &[]), Some(CHAT_IN_USE));
        // A chat turn says nothing about the other slots.
        assert_eq!(in_use_reason(TTS_RESIDENT_ID, true, false, &[]), None);
        drop(turn);
        assert!(activity.exclusive().is_some());
    }

    /// A synthesis lease on talker A refuses loading talker B, whose load
    /// would stop A's server, even before A's job row exists. Loading A again
    /// reuses the running server. Once A's job ends, B may load.
    #[tokio::test]
    async fn tts_swap_waits_for_a_live_lease() {
        let pool = job_pool().await;
        let mut leases = JobLeases::default();
        leases.grant(JobLease::sidecar("a1"));
        let leased = leases.holds(&pool, TTS_RESIDENT_ID).await;
        assert!(leased);
        assert_eq!(tts_swap(Some("talker-a"), "talker-b", leased), SwapDecision::Refuse(TTS_SWAP_IN_USE));
        assert_eq!(tts_swap(Some("talker-a"), "talker-a", leased), SwapDecision::Reuse);

        leases.on_event(&WorkerEvent::JobDone {
            job_id: "a1".into(),
            outputs: vec![],
            meta: serde_json::json!({}),
            task: None,
        });
        let leased = leases.holds(&pool, TTS_RESIDENT_ID).await;
        assert!(!leased);
        assert_eq!(tts_swap(Some("talker-a"), "talker-b", leased), SwapDecision::Proceed);

        // A Kokoro job runs in the worker, not on the talker's server.
        leases.grant(JobLease::worker("k1", "text_to_speech"));
        assert!(!leases.holds(&pool, TTS_RESIDENT_ID).await);
        // Nothing serving: there is no server to stop.
        assert_eq!(tts_swap(None, "talker-b", true), SwapDecision::Proceed);
    }

    /// Every llama-server load restarts the server, so while a reply streams
    /// only a plain load of the serving model is answered (from what runs).
    /// The gate, held through a load, keeps a new turn waiting until it ends.
    #[tokio::test]
    async fn text_swap_waits_for_a_streaming_reply() {
        let plain = TextLoadRequest { model_id: "a".into(), ..Default::default() };
        assert!(plain.is_plain());
        let tuned = TextLoadRequest { model_id: "a".into(), ctx: Some(16_384), ..Default::default() };
        assert!(!tuned.is_plain());

        let (writer, activity) = crate::chat::activity::channel();
        let turn = writer.reserve().await;
        let busy = activity.exclusive().is_none();
        assert!(busy);
        assert_eq!(text_swap(busy, Some("a"), "b", true), SwapDecision::Refuse(CHAT_SWAP_IN_USE));
        assert_eq!(text_swap(busy, Some("a"), "a", true), SwapDecision::Reuse);
        assert_eq!(text_swap(busy, Some("a"), "a", false), SwapDecision::Refuse(CHAT_SWAP_IN_USE));
        assert_eq!(text_swap(busy, None, "b", true), SwapDecision::Proceed);
        drop(turn);

        let gate = activity.exclusive();
        assert!(gate.is_some());
        assert_eq!(text_swap(false, Some("a"), "b", true), SwapDecision::Proceed);
        assert!(
            tokio::time::timeout(Duration::from_millis(50), writer.reserve()).await.is_err(),
            "a turn must not start while a load holds the gate"
        );
        drop(gate);
        let _next_turn = writer.reserve().await;
    }

    /// Llama-3-8B-class inputs: ~7.3 GB full estimate at 4k ctx.
    fn inputs() -> FitInputs {
        FitInputs {
            file_size_mb: 4700,
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            ctx: 4096,
            kv_dtype_bytes: 2,
            kv: None,
        }
    }

    #[test]
    fn est_for_ngl_bounds() {
        let i = inputs();
        let full = estimate_vram_mb(&i);
        assert_eq!(est_for_ngl(&i, -1), full);
        assert_eq!(est_for_ngl(&i, 0), 0);
        let half = est_for_ngl(&i, 16);
        assert!(half > FIXED_OVERHEAD_MB && half < full, "got {half}");
    }

    #[test]
    fn text_gate_fits_outright_no_eviction() {
        let i = inputs();
        let full = estimate_vram_mb(&i);
        // Plenty free, pipelines resident — no need to bother anyone.
        assert_eq!(text_gate(&i, None, full + 4000, 8000, false), Ok(-1));
    }

    #[test]
    fn text_gate_emits_choice_when_eviction_fixes_fit() {
        let i = inputs();
        let full = estimate_vram_mb(&i);
        // Free is short by ~2 GB, but an 8 GB pipeline resident covers it.
        let free = full - 2000;
        let r = text_gate(&i, None, free, 8000, false);
        assert_eq!(r, Err(full));
    }

    #[test]
    fn text_gate_degrades_silently_when_eviction_wont_help() {
        let i = inputs();
        // Tiny GPU: even with the pipeline gone it's a partial fit → no choice.
        let r = text_gate(&i, None, 3000, 1000, false);
        assert!(matches!(r, Ok(n) if n >= 0), "got {r:?}");
    }

    #[test]
    fn text_gate_no_candidates_degrades() {
        let i = inputs();
        let full = estimate_vram_mb(&i);
        let r = text_gate(&i, None, full - 2000, 0, false);
        assert!(matches!(r, Ok(n) if n > 0), "got {r:?}");
    }

    #[test]
    fn text_gate_force_skips_choice() {
        let i = inputs();
        let full = estimate_vram_mb(&i);
        let r = text_gate(&i, None, full - 2000, 8000, true);
        assert!(matches!(r, Ok(_)), "got {r:?}");
    }

    #[test]
    fn text_gate_explicit_ngl_gated_only_when_eviction_helps() {
        let i = inputs();
        let full = estimate_vram_mb(&i);
        // Explicit full offload, short on free, pipeline covers it → choice.
        assert_eq!(text_gate(&i, Some(-1), full - 2000, 8000, false), Err(full));
        // Explicit full offload, nothing else resident → respect verbatim.
        assert_eq!(text_gate(&i, Some(-1), full - 2000, 0, false), Ok(-1));
        // Explicit CPU never needs VRAM.
        assert_eq!(text_gate(&i, Some(0), 100, 8000, false), Ok(0));
    }

    #[test]
    fn pipeline_gate_basics() {
        // Fits → reserved.
        assert_eq!(pipeline_gate(6000, 10_000, 0, false), Ok(()));
        // Doesn't fit, text model reclaim covers it → choice.
        assert_eq!(pipeline_gate(12_000, 10_000, 7000, false), Err(12_000));
        // Doesn't fit even after eviction → proceed (offload/honest OOM).
        assert_eq!(pipeline_gate(30_000, 10_000, 7000, false), Ok(()));
        // No candidates → proceed.
        assert_eq!(pipeline_gate(12_000, 10_000, 0, false), Ok(()));
        // Force skips the gate.
        assert_eq!(pipeline_gate(12_000, 10_000, 7000, true), Ok(()));
    }
}
