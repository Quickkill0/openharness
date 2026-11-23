use std::sync::atomic::AtomicBool;

use sqlx::SqlitePool;

use crate::chat::ChatHandle;
use crate::downloader::ManagerHandle;

/// Global application state, managed via `tauri::Manager::manage()`.
pub struct AppState {
    /// SQLite connection pool (WAL mode, foreign keys enabled).
    pub db: SqlitePool,
    /// Shared HTTP client — one instance for the process lifetime.
    /// Pre-configured with the OpenHarness user-agent and a 30 s timeout.
    pub http: reqwest::Client,
    /// Download manager handle — communicates with the background actor.
    pub downloads: ManagerHandle,
    /// Engine manager handle — controls llama-server lifecycle.
    pub engine: crate::engine::EngineManagerHandle,
    /// GPU actor handle — provides NVML telemetry.
    pub gpu: crate::gpu::GpuHandle,
    /// Chat actor handle — manages conversations and streaming.
    pub chat: ChatHandle,
    /// Python-worker manager handle — generation pipelines (M8).
    pub worker: crate::worker::WorkerHandle,
    /// Engine orchestrator handle — VRAM ledger + gated loads across both
    /// engine families (M8 §8.4).
    pub orchestrator: crate::orchestrator::OrchestratorHandle,
    /// Local OpenAI-compatible API listener, settings and request-log actor.
    pub server: crate::api::ServerHandle,
    /// Qwen3-TTS sidecar handle — qwentts.cpp tts-server lifecycle.
    pub tts: crate::tts::TtsHandle,
    /// Finalized generation job-event bus (from `generation::spawn_job_sync`):
    /// a re-broadcast of the raw worker bus where `JobDone.outputs` are the
    /// promoted asset-library paths. The frontend progress channel subscribes
    /// here so the webview never sees a staging path that promotion is about
    /// to rename away.
    pub gen_events: tokio::sync::broadcast::Sender<crate::worker::protocol::WorkerEvent>,
    /// Load-planner store (SSD measurements, per-model speed calibration).
    pub planner: crate::planner::store::PlannerHandle,
    /// Latest live actuals for the loaded text model.
    pub runtime_stats: tokio::sync::watch::Receiver<crate::planner::runtime::RuntimeStats>,
    /// Set when the one-time Hangar→OpenHarness data migration ran this launch.
    /// Consumed once by the frontend to show a one-off toast.
    pub migrated: AtomicBool,
}
