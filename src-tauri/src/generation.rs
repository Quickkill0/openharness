//! Generation primitives (M8 §8.5): the bridge between Tauri commands and
//! the Python worker's pipeline jobs.
//!
//! A submit (1) reserves VRAM with the orchestrator (typed needs-eviction
//! outcome instead of silent eviction), (2) persists a `generation_jobs` row,
//! then (3) hands the job to the worker. Terminal job events are mirrored
//! back into the row by [`spawn_job_sync`], which is also the single owner of
//! asset promotion: it re-broadcasts every job event on a *finalized* bus
//! whose `JobDone` carries the promoted (library) paths. The frontend's
//! `subscribe_generation_progress` consumes the finalized bus — never the raw
//! worker bus — so the webview can't observe a staging path that a concurrent
//! promotion renames away.

use std::path::Path;
use std::time::Duration;

use serde::Serialize;
use serde_json::Value;
use sqlx::SqlitePool;
use tokio::sync::broadcast;

use crate::db::generation_jobs;
use crate::error::AppError;
use crate::models::registry::{self, InstalledModel};
use crate::orchestrator::{JobLease, OrchestratorHandle, ReserveOutcome, ResidentInfo};
use crate::worker::protocol::{WorkerEvent, WorkerRequest};
use crate::worker::{WorkerHandle, WorkerStatus};

/// Working VRAM a pipeline needs beyond its weights (CUDA context,
/// activations, temp buffers). Deliberately coarse — the ledger replaces the
/// estimate with NVML actuals after the first job completes.
const PIPELINE_OVERHEAD_MB: u64 = 1500;

/// Result of a generation submit, mirrored from the orchestrator's gate.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum SubmitOutcome {
    Submitted {
        job_id: String,
    },
    /// Nothing was submitted; the UI presents the choice (unload a candidate
    /// and retry, or re-send with `force: true`).
    NeedsEviction {
        required_mb: u64,
        free_mb: u64,
        candidates: Vec<ResidentInfo>,
    },
}

/// Reserve VRAM, persist the job row, submit to the worker.
pub async fn submit(
    db: &SqlitePool,
    worker: &WorkerHandle,
    orchestrator: &OrchestratorHandle,
    task: &str,
    mut params: Value,
    force: bool,
) -> Result<SubmitOutcome, AppError> {
    // Correlate the resident back to its catalog row so the UI can mark the
    // model Loaded (lazy submits carry a model *path*, not a catalog id), and
    // — for single-file diffusion GGUFs — tag the worker request so it reaches
    // the GGUF assembly branch instead of treating the .gguf as a single-file
    // SD/SDXL/Flux checkpoint.
    let model = match params.get("model").and_then(|v| v.as_str()) {
        Some(p) => registry::find_by_source_path(db, p).await?,
        None => None,
    };
    let model_id = model.as_ref().map(|m| m.id.clone());
    if let Some(m) = &model {
        if m.caps.gguf_diffusion {
            // The worker's GGUF branch keys on these; without them it would try
            // the legacy single-file SD/SDXL/Flux path and fail confusingly.
            params["gguf"] = Value::Bool(true);
            if let Some(arch) = &m.caps.companion_arch {
                params["companion_arch"] = Value::String(arch.clone());
            }
        }
    }

    let est_mb = estimate_task_vram_mb(task, &params);
    let label = model_label(task, &params);

    // The job id comes first so its lease is granted with the reservation:
    // an unload can't land between "reserved" and the job starting.
    let job_id = uuid::Uuid::new_v4().to_string();
    let lease = JobLease::worker(&job_id, task);
    match orchestrator
        .reserve_pipeline(task.to_string(), label, model_id, est_mb, force, Some(lease))
        .await?
    {
        ReserveOutcome::Reserved => {}
        ReserveOutcome::NeedsEviction { required_mb, free_mb, candidates } => {
            return Ok(SubmitOutcome::NeedsEviction { required_mb, free_mb, candidates });
        }
    }

    if let Err(e) = generation_jobs::insert(db, &job_id, task, &params).await {
        orchestrator.abort_lease(job_id, e.to_string()).await;
        return Err(e);
    }

    let req = WorkerRequest::Submit {
        task: task.to_string(),
        job_id: job_id.clone(),
        params,
    };
    if let Err(e) = worker.request(req, Duration::from_secs(30)).await {
        // The submit never reached the job queue: the abort reports the job
        // ended (DB sync closes the row; a reloaded UI that adopted it
        // unlocks), then the reservation goes (nothing actually loaded).
        orchestrator.abort_lease(job_id, e.to_string()).await;
        let _ = orchestrator.unload(format!("pipeline:{task}")).await;
        return Err(e);
    }

    Ok(SubmitOutcome::Submitted { job_id })
}

/// Result of a W1 explicit pipeline preload.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum PreloadOutcome {
    Loaded {
        /// NVML-actual VRAM where the pipeline could measure it.
        vram_mb: Option<u64>,
    },
    NeedsEviction {
        required_mb: u64,
        free_mb: u64,
        candidates: Vec<ResidentInfo>,
    },
}

/// How long an explicit preload may take end-to-end. Multi-GB checkpoints
/// load from disk in well under this; the generous bound only catches a
/// wedged worker.
const PRELOAD_TIMEOUT: Duration = Duration::from_secs(15 * 60);

/// Explicitly load a worker pipeline's model into VRAM without running a job
/// (W1 §16 load-first flow). Reserves through the orchestrator's gate, makes
/// sure the worker process is up (auto-starting a Stopped one), then issues a
/// `preload` request and waits for its terminal job event. Progress streams
/// to the frontend over the same `subscribe_generation_progress` channel jobs
/// use, keyed by the returned synthetic job id.
pub async fn preload_pipeline(
    worker: &WorkerHandle,
    orchestrator: &OrchestratorHandle,
    model: &InstalledModel,
    force: bool,
) -> Result<PreloadOutcome, AppError> {
    let task = model.caps.pipeline.clone().ok_or_else(|| {
        AppError::Other(format!(
            "'{}' has no worker pipeline to preload",
            model.display_name
        ))
    })?;

    let model_path = model
        .local_path
        .clone()
        .filter(|p| !p.is_empty())
        .unwrap_or_else(|| model.primary_file_path.clone());
    let mut params = serde_json::json!({ "model": model_path });
    if task == "text_to_image" {
        if let Some(arch) = &model.caps.image_arch {
            params["arch"] = Value::String(arch.clone());
        }
    }
    // Single-file diffusion GGUFs must reach the worker's GGUF assembly branch
    // (today: the honest interim reject) rather than the legacy single-file
    // SD/SDXL/Flux loader. Mirror the tagging that `submit()` applies so an
    // explicit Load surfaces the same outcome as a generate. See `submit`.
    if model.caps.gguf_diffusion {
        params["gguf"] = Value::Bool(true);
        if let Some(arch) = &model.caps.companion_arch {
            params["companion_arch"] = Value::String(arch.clone());
        }
    }

    // Make sure the worker process is up before reserving — an install-less
    // setup gets an honest, actionable error instead of a timeout.
    match worker.status().await {
        WorkerStatus::Ready { .. } => {}
        WorkerStatus::Stopped { .. } | WorkerStatus::Starting => {
            worker.start_worker().await?;
        }
        WorkerStatus::NotInstalled => {
            return Err(AppError::Other(
                "The Python runtime isn't set up yet — it installs automatically on first run; \
                 check Settings → Engines for progress"
                    .into(),
            ));
        }
        WorkerStatus::Installing { .. } => {
            return Err(AppError::Other(
                "The Python runtime is still installing — try again when it finishes".into(),
            ));
        }
        WorkerStatus::Failed { error } => {
            // A prior failure isn't terminal: the venv is still valid, so the
            // common case (a cold-start handshake that timed out once) heals on
            // the next submit. Try a fresh start before surfacing an error —
            // otherwise the worker is stranded in Failed until an app restart.
            tracing::warn!(
                "worker was Failed ({error}); attempting a fresh start on submit"
            );
            worker.start_worker().await.map_err(|e| {
                AppError::Other(format!("The Python runtime failed to start: {e}"))
            })?;
        }
    }

    let est_mb = estimate_task_vram_mb(&task, &params);
    match orchestrator
        .reserve_pipeline(
            task.clone(),
            model.display_name.clone(),
            Some(model.id.clone()),
            est_mb,
            force,
            None,
        )
        .await?
    {
        ReserveOutcome::Reserved => {}
        ReserveOutcome::NeedsEviction { required_mb, free_mb, candidates } => {
            return Ok(PreloadOutcome::NeedsEviction { required_mb, free_mb, candidates });
        }
    }

    // Subscribe before submitting so the terminal event can't slip past.
    let mut events = worker.subscribe_job_events();
    let job_id = uuid::Uuid::new_v4().to_string();
    let req = WorkerRequest::Preload {
        task: task.clone(),
        job_id: job_id.clone(),
        params,
    };

    let release = || async {
        let _ = orchestrator.unload(format!("pipeline:{task}")).await;
    };

    if let Err(e) = worker.request(req, Duration::from_secs(30)).await {
        release().await;
        return Err(e);
    }

    let wait = async {
        loop {
            match events.recv().await {
                Ok(WorkerEvent::JobDone { job_id: id, meta, .. }) if id == job_id => {
                    return Ok(meta.get("vram_mb").and_then(|v| v.as_u64()));
                }
                Ok(WorkerEvent::JobError { job_id: id, error }) if id == job_id => {
                    return Err(AppError::Other(error));
                }
                Ok(_) => continue,
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(broadcast::error::RecvError::Closed) => {
                    return Err(AppError::Other("worker event stream closed".into()));
                }
            }
        }
    };
    match tokio::time::timeout(PRELOAD_TIMEOUT, wait).await {
        Ok(Ok(vram_mb)) => Ok(PreloadOutcome::Loaded { vram_mb }),
        Ok(Err(e)) => {
            release().await;
            Err(e)
        }
        Err(_) => {
            // Best-effort cancel so a preload still queued behind a long job
            // doesn't run to completion after we've given up (the worker
            // executor is single-threaded).
            let _ = worker
                .request(
                    WorkerRequest::Cancel { job_id: job_id.clone() },
                    Duration::from_secs(10),
                )
                .await;
            release().await;
            Err(AppError::Other(format!(
                "preloading '{}' timed out",
                model.display_name
            )))
        }
    }
}

/// Submit a Qwen3-TTS synthesis job (the `engine: "qwen3-tts"` branch of
/// `synthesize_speech`). Loads the talker via the orchestrator's gated
/// `"tts"` slot, persists a `generation_jobs` row, then synthesizes over
/// HTTP in a background task that reports through the same job-event bus
/// the worker pipelines use — so progress streaming, DB sync, and asset
/// promotion all work unchanged. `JobDone` carries `task: None` so the
/// orchestrator doesn't misattribute the VRAM to a worker pipeline.
pub async fn submit_qwen_tts(
    db: &SqlitePool,
    worker: &WorkerHandle,
    orchestrator: &OrchestratorHandle,
    tts: &crate::tts::TtsHandle,
    params: Value,
    force: bool,
) -> Result<SubmitOutcome, AppError> {
    let model_id = params
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            AppError::Other("Qwen3-TTS needs a talker model — pick one first".into())
        })?
        .to_string();
    let text = params
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .to_string();
    if text.trim().is_empty() {
        return Err(AppError::Other("nothing to speak — text is empty".into()));
    }

    // Lease granted with the reservation, held until the synthesis ends.
    let job_id = uuid::Uuid::new_v4().to_string();
    match orchestrator.load_tts(model_id, force, Some(JobLease::sidecar(&job_id))).await? {
        ReserveOutcome::Reserved => {}
        ReserveOutcome::NeedsEviction { required_mb, free_mb, candidates } => {
            return Ok(SubmitOutcome::NeedsEviction { required_mb, free_mb, candidates });
        }
    }

    let port = match tts.status().await {
        crate::tts::TtsStatus::Ready { port, .. } => port,
        other => {
            let e = AppError::Other(format!(
                "Qwen3-TTS engine is not ready: {}",
                serde_json::to_value(&other)
                    .ok()
                    .and_then(|v| v.get("phase").and_then(|p| p.as_str()).map(String::from))
                    .unwrap_or_else(|| "unknown".into())
            ));
            orchestrator.abort_lease(job_id, e.to_string()).await;
            return Err(e);
        }
    };

    let voice = params.get("voice").and_then(|v| v.as_str()).map(str::to_string);
    let instructions =
        params.get("instructions").and_then(|v| v.as_str()).map(str::to_string);

    if let Err(e) = generation_jobs::insert(db, &job_id, "text_to_speech", &params).await {
        orchestrator.abort_lease(job_id, e.to_string()).await;
        return Err(e);
    }

    let events = worker.job_events_sender();
    let jid = job_id.clone();
    tauri::async_runtime::spawn(async move {
        let _ = events.send(WorkerEvent::JobProgress {
            job_id: jid.clone(),
            pct: 10.0,
            stage: "synthesizing".into(),
            step: None,
            total_steps: None,
        });
        let staged = crate::config::assets_staging_dir().join(format!("speech-{jid}.wav"));
        let result = async {
            let bytes =
                crate::tts::synthesize_wav(port, &text, voice.as_deref(), instructions.as_deref())
                    .await?;
            if let Some(parent) = staged.parent() {
                tokio::fs::create_dir_all(parent).await?;
            }
            tokio::fs::write(&staged, &bytes).await?;
            Ok::<_, AppError>(())
        }
        .await;
        match result {
            Ok(()) => {
                let _ = events.send(WorkerEvent::JobDone {
                    job_id: jid,
                    outputs: vec![staged.to_string_lossy().into_owned()],
                    meta: serde_json::json!({
                        "engine": "qwen3-tts",
                        "sample_rate": 24000,
                    }),
                    task: None,
                });
            }
            Err(e) => {
                let _ = events.send(WorkerEvent::JobError { job_id: jid, error: e.to_string() });
            }
        }
    });

    Ok(SubmitOutcome::Submitted { job_id })
}

/// Mirror terminal worker job events into `generation_jobs` rows and own
/// asset promotion (files are renamed out of staging exactly once, here).
/// Returns the sender of the **finalized** job-event bus: a re-broadcast of
/// the raw worker bus where `JobDone.outputs` are the promoted library paths
/// (DB row already updated when the event lands). User-facing consumers — the
/// `subscribe_generation_progress` channel — must subscribe to this bus;
/// VRAM/ledger bookkeeping (orchestrator, preload waiter) keys off `meta` and
/// stays on the raw bus.
///
/// Runs for the process lifetime; survives worker restarts (the manager
/// re-broadcasts on its own channel). Rows orphaned by the previous process
/// are healed earlier, by [`heal_interrupted_jobs`].
pub fn spawn_job_sync(db: SqlitePool, worker: WorkerHandle) -> broadcast::Sender<WorkerEvent> {
    let rx = worker.subscribe_job_events();
    let (finalized_tx, _) = broadcast::channel::<WorkerEvent>(256);
    let tx = finalized_tx.clone();
    tauri::async_runtime::spawn(run_job_sync(db, rx, tx));
    finalized_tx
}

/// Close every `running` row the previous process left behind. Startup awaits
/// this before any command can be served: a reloaded UI or an unload check
/// that read an orphaned row first would adopt a job that then closes with
/// no terminal event, and lock its model for good.
pub async fn heal_interrupted_jobs(db: &SqlitePool) {
    match generation_jobs::heal_interrupted(db).await {
        Ok(n) if n > 0 => tracing::info!("healed {n} interrupted generation job(s)"),
        Ok(_) => {}
        Err(e) => tracing::warn!("generation-job heal failed: {e}"),
    }
}

/// The sync loop body: consume raw worker job events, promote + persist
/// terminal ones, then re-broadcast on the finalized bus. Progress events
/// pass through untouched; `Log` events never reach the finalized bus.
/// Extracted from [`spawn_job_sync`] so tests can drive it with bare
/// broadcast channels.
async fn run_job_sync(
    db: SqlitePool,
    mut rx: broadcast::Receiver<WorkerEvent>,
    finalized: broadcast::Sender<WorkerEvent>,
) {
    loop {
        match rx.recv().await {
            Ok(WorkerEvent::JobDone { job_id, outputs, meta, task }) => {
                // M10: promote outputs out of staging into the asset
                // library, then record the final paths on the job row.
                // The job row is the task source of truth; the event's
                // `task` field is a fallback (and is forwarded verbatim —
                // qwen-tts JobDone carries `task: None` on purpose).
                let (row_task, params) = match generation_jobs::get(&db, &job_id).await {
                    Ok(Some(job)) => (job.task, job.params),
                    _ => (task.clone().unwrap_or_default(), Value::Null),
                };
                let final_paths = crate::assets::promote_job_outputs(
                    &db, &job_id, &row_task, &params, &outputs, &meta,
                )
                .await;
                if let Err(e) =
                    generation_jobs::mark_done(&db, &job_id, &final_paths, &meta).await
                {
                    tracing::warn!("mark_done({job_id}) failed: {e}");
                }
                // Only now — promotion + DB mirror done — may the frontend
                // see the terminal event, carrying the final paths.
                let _ = finalized.send(WorkerEvent::JobDone {
                    job_id,
                    outputs: final_paths,
                    meta,
                    task,
                });
            }
            Ok(WorkerEvent::JobError { job_id, error }) => {
                if let Err(e) = generation_jobs::mark_error(&db, &job_id, &error).await {
                    tracing::warn!("mark_error({job_id}) failed: {e}");
                }
                let _ = finalized.send(WorkerEvent::JobError { job_id, error });
            }
            Ok(ev @ WorkerEvent::JobProgress { .. }) => {
                let _ = finalized.send(ev);
            }
            // Log events never reach the finalized bus.
            Ok(WorkerEvent::Log { .. }) => {}
            Err(broadcast::error::RecvError::Lagged(_)) => continue,
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }
}

// ---------------------------------------------------------------------------
// VRAM estimates (pure — unit-tested below)
// ---------------------------------------------------------------------------

/// Coarse VRAM estimate for a pipeline job, used only to gate the submit.
/// Diffusion models: weights-on-disk ≈ weights-in-VRAM at fp16/bf16, plus
/// the working overhead. Whisper/Kokoro: small static tables.
pub fn estimate_task_vram_mb(task: &str, params: &Value) -> u64 {
    let model = params.get("model").and_then(|v| v.as_str());
    match task {
        "text_to_image" | "text_to_video" => {
            let weights = model.map_or(0, |m| path_size_mb(Path::new(m)));
            weights + PIPELINE_OVERHEAD_MB
        }
        "transcribe" => match model {
            // Local ctranslate2 dir — size it like the diffusers case.
            Some(m) if Path::new(m).exists() => {
                path_size_mb(Path::new(m)) + PIPELINE_OVERHEAD_MB
            }
            other => whisper_est_mb(other.unwrap_or("base")),
        },
        // Kokoro-82M: tiny, fixed.
        "text_to_speech" => 1200,
        // VibeVoice: bf16 weights ≈ on-disk size; flat fallback ≈ the 1.5B
        // checkpoint when the model dir is missing.
        "vibevoice" => match model {
            Some(m) if Path::new(m).exists() => {
                path_size_mb(Path::new(m)) + PIPELINE_OVERHEAD_MB
            }
            _ => 7_000,
        },
        _ => PIPELINE_OVERHEAD_MB,
    }
}

/// Human-readable resident label for the ledger.
pub fn model_label(task: &str, params: &Value) -> String {
    let model = params.get("model").and_then(|v| v.as_str());
    match task {
        "text_to_speech" => {
            if params.get("engine").and_then(|v| v.as_str()) == Some("qwen3-tts") {
                let voice = params.get("voice").and_then(|v| v.as_str()).unwrap_or("default");
                return format!("Qwen3-TTS ({voice})");
            }
            let voice = params.get("voice").and_then(|v| v.as_str()).unwrap_or("af_heart");
            match model {
                Some(m) => format!("{} ({voice})", basename(m)),
                None => format!("Kokoro-82M ({voice})"),
            }
        }
        "transcribe" => match model {
            Some(m) if Path::new(m).exists() => basename(m),
            other => format!("whisper-{}", other.unwrap_or("base")),
        },
        "vibevoice" => match model {
            Some(m) => format!("{} (voice clone)", basename(m)),
            None => "VibeVoice".into(),
        },
        _ => model.map(basename).unwrap_or_else(|| task.to_string()),
    }
}

/// Whisper checkpoint sizes (fp16 CUDA) + working overhead, by size name.
fn whisper_est_mb(name: &str) -> u64 {
    let n = name.to_ascii_lowercase();
    if n.contains("large") || n.contains("turbo") {
        11_000
    } else if n.contains("medium") {
        6_000
    } else if n.contains("small") {
        3_000
    } else if n.contains("base") {
        2_000
    } else if n.contains("tiny") {
        1_500
    } else {
        3_000
    }
}

/// Last path component, splitting on both separators — job params carry
/// Windows paths, but tests (and a future non-Windows build) may not.
/// Trailing separators are ignored, matching `Path::file_name`.
fn basename(p: &str) -> String {
    let trimmed = p.trim_end_matches(['/', '\\']);
    trimmed
        .rsplit(['/', '\\'])
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(p)
        .to_string()
}

/// Size of a file, or the recursive size of a directory, in MiB.
/// Missing paths size to 0 — the worker reports the honest error.
fn path_size_mb(p: &Path) -> u64 {
    let bytes: u64 = if p.is_file() {
        let files = if p.extension().is_some_and(|e| e == "gguf") {
            crate::models::gguf::split_files(p).unwrap_or_default()
        } else { vec![p.to_path_buf()] };
        files.iter().map(|f| f.metadata().map(|m| m.len()).unwrap_or(0)).sum()
    } else if p.is_dir() {
        walkdir::WalkDir::new(p)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .filter(|m| m.is_file())
            .map(|m| m.len())
            .sum()
    } else {
        0
    };
    bytes / (1024 * 1024)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "generation_split_tests.rs"]
mod split_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn whisper_table_by_size_name() {
        assert_eq!(whisper_est_mb("tiny"), 1500);
        assert_eq!(whisper_est_mb("base.en"), 2000);
        assert_eq!(whisper_est_mb("distil-small.en"), 3000);
        assert_eq!(whisper_est_mb("medium"), 6000);
        assert_eq!(whisper_est_mb("large-v3"), 11_000);
        assert_eq!(whisper_est_mb("large-v3-turbo"), 11_000);
        assert_eq!(whisper_est_mb("mystery"), 3000);
    }

    #[test]
    fn estimates_by_task() {
        // Missing model path → overhead only (worker errors honestly).
        let p = json!({"model": "Z:\\does\\not\\exist", "prompt": "x"});
        assert_eq!(estimate_task_vram_mb("text_to_image", &p), PIPELINE_OVERHEAD_MB);
        // Whisper by size name.
        let p = json!({"audio": "a.wav", "model": "medium"});
        assert_eq!(estimate_task_vram_mb("transcribe", &p), 6000);
        // Kokoro fixed.
        assert_eq!(estimate_task_vram_mb("text_to_speech", &json!({"text": "hi"})), 1200);
        // VibeVoice: flat fallback when the model dir is missing…
        let p = json!({"model": "Z:\\missing\\vibevoice", "text": "hi", "reference_audio": "v.wav"});
        assert_eq!(estimate_task_vram_mb("vibevoice", &p), 7_000);
        // …and weights + overhead when it exists (this file ≈ 0 MB).
        let p = json!({"model": file!(), "text": "hi"});
        assert_eq!(estimate_task_vram_mb("vibevoice", &p), PIPELINE_OVERHEAD_MB);
    }

    #[test]
    fn estimate_includes_real_file_size() {
        // This source file exists; estimate = size (0 MB) + overhead.
        let me = json!({"model": file!(), "prompt": "x"});
        assert_eq!(estimate_task_vram_mb("text_to_image", &me), PIPELINE_OVERHEAD_MB);
    }

    #[test]
    fn labels() {
        let p = json!({"model": "C:\\models\\sdxl\\juggernaut.safetensors"});
        assert_eq!(model_label("text_to_image", &p), "juggernaut.safetensors");
        assert_eq!(
            model_label("text_to_speech", &json!({"voice": "bf_emma"})),
            "Kokoro-82M (bf_emma)"
        );
        assert_eq!(model_label("transcribe", &json!({"model": "base"})), "whisper-base");
        assert_eq!(model_label("text_to_video", &json!({})), "text_to_video");
        assert_eq!(
            model_label("vibevoice", &json!({"model": "C:\\models\\VibeVoice-1.5B"})),
            "VibeVoice-1.5B (voice clone)"
        );
        assert_eq!(model_label("vibevoice", &json!({})), "VibeVoice");
    }

    #[test]
    fn basename_ignores_trailing_separators() {
        assert_eq!(basename("C:\\models\\sdxl\\"), "sdxl");
        assert_eq!(basename("/models/sdxl/"), "sdxl");
        assert_eq!(basename("C:\\models\\sdxl"), "sdxl");
        assert_eq!(basename("plain-name"), "plain-name");
    }

    async fn test_pool() -> SqlitePool {
        let pool = sqlx::sqlite::SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    /// Startup healing must finish before the active-job snapshot is served:
    /// a row orphaned by a crash is closed by the time anything can read it,
    /// so a reloaded UI never adopts a job that will close without a terminal
    /// event. Healing is awaited (not spawned), so it has landed on return.
    #[tokio::test]
    async fn orphaned_rows_are_closed_before_the_snapshot_is_read() {
        let pool = test_pool().await;
        generation_jobs::insert(&pool, "orphan", "text_to_speech", &json!({})).await.unwrap();
        assert_eq!(generation_jobs::list_running(&pool).await.unwrap().len(), 1);

        heal_interrupted_jobs(&pool).await;

        assert!(generation_jobs::list_running(&pool).await.unwrap().is_empty());
        let row = generation_jobs::get(&pool, "orphan").await.unwrap().unwrap();
        assert_eq!(row.status, "error");
    }

    /// The promotion-race regression (frontend received staging paths that a
    /// concurrent promotion renamed away): the finalized bus must emit
    /// `JobDone` only *after* promotion + DB mirror, carrying the promoted
    /// library paths — never the staging path.
    #[tokio::test]
    async fn finalized_job_done_carries_promoted_paths() {
        let pool = test_pool().await;

        // A real staged output, like the qwen-tts bridge (or a worker
        // pipeline) writes before emitting JobDone.
        let staging = crate::config::assets_staging_dir();
        std::fs::create_dir_all(&staging).unwrap();
        let job_id = uuid::Uuid::new_v4().to_string();
        let staged = staging.join(format!("speech-{job_id}.wav"));
        std::fs::write(&staged, b"RIFF-mock").unwrap();

        generation_jobs::insert(&pool, &job_id, "text_to_speech", &json!({"text": "hi"}))
            .await
            .unwrap();

        let (raw_tx, raw_rx) = broadcast::channel::<WorkerEvent>(16);
        let (fin_tx, mut fin_rx) = broadcast::channel::<WorkerEvent>(16);
        tokio::spawn(run_job_sync(pool.clone(), raw_rx, fin_tx));

        // Progress passes through untouched…
        raw_tx
            .send(WorkerEvent::JobProgress {
                job_id: job_id.clone(),
                pct: 50.0,
                stage: "synthesizing".into(),
                step: None,
                total_steps: None,
            })
            .unwrap();
        // …then the raw terminal event with the STAGING path (and the
        // qwen-tts `task: None` shape, which must be preserved).
        raw_tx
            .send(WorkerEvent::JobDone {
                job_id: job_id.clone(),
                outputs: vec![staged.to_string_lossy().into_owned()],
                meta: json!({"engine": "qwen3-tts", "sample_rate": 24000}),
                task: None,
            })
            .unwrap();

        let first = tokio::time::timeout(Duration::from_secs(10), fin_rx.recv())
            .await
            .expect("timed out waiting for progress")
            .unwrap();
        assert!(matches!(first, WorkerEvent::JobProgress { pct, .. } if pct == 50.0));

        let done = tokio::time::timeout(Duration::from_secs(10), fin_rx.recv())
            .await
            .expect("timed out waiting for job_done")
            .unwrap();
        match done {
            WorkerEvent::JobDone { job_id: id, outputs, task, meta } => {
                assert_eq!(id, job_id);
                assert_eq!(task, None, "qwen-tts task:None must be preserved");
                assert_eq!(meta["sample_rate"], 24000);
                assert_eq!(outputs.len(), 1);
                let final_path = Path::new(&outputs[0]);
                assert!(
                    !final_path.starts_with(&staging),
                    "finalized event still carries a staging path: {}",
                    outputs[0]
                );
                assert!(final_path.exists(), "promoted file missing on disk");
                assert!(!staged.exists(), "staged file should have been renamed");
                // DB row mirrors the same final paths — promotion happened
                // exactly once, before the event was forwarded.
                let row = generation_jobs::get(&pool, &job_id).await.unwrap().unwrap();
                assert_eq!(row.outputs, outputs);
                assert_eq!(row.status, "done");
                std::fs::remove_file(final_path).ok();
            }
            other => panic!("expected JobDone, got {other:?}"),
        }
    }

    /// JobError passes through to the finalized bus after the DB mirror.
    #[tokio::test]
    async fn finalized_job_error_passes_through() {
        let pool = test_pool().await;
        let job_id = uuid::Uuid::new_v4().to_string();
        generation_jobs::insert(&pool, &job_id, "text_to_image", &json!({"prompt": "x"}))
            .await
            .unwrap();

        let (raw_tx, raw_rx) = broadcast::channel::<WorkerEvent>(16);
        let (fin_tx, mut fin_rx) = broadcast::channel::<WorkerEvent>(16);
        tokio::spawn(run_job_sync(pool.clone(), raw_rx, fin_tx));

        raw_tx
            .send(WorkerEvent::JobError { job_id: job_id.clone(), error: "CUDA OOM".into() })
            .unwrap();

        let ev = tokio::time::timeout(Duration::from_secs(10), fin_rx.recv())
            .await
            .expect("timed out waiting for job_error")
            .unwrap();
        match ev {
            WorkerEvent::JobError { job_id: id, error } => {
                assert_eq!(id, job_id);
                assert_eq!(error, "CUDA OOM");
                let row = generation_jobs::get(&pool, &job_id).await.unwrap().unwrap();
                assert_eq!(row.status, "error");
            }
            other => panic!("expected JobError, got {other:?}"),
        }
    }

    /// An aborted lease's job may have no row (its insert failed), but a
    /// reloaded UI can still hold it. Its job_error must reach the finalized
    /// bus the UI and waiters listen on.
    #[tokio::test]
    async fn job_error_without_a_row_still_reaches_the_ui() {
        let pool = test_pool().await;
        let (raw_tx, raw_rx) = broadcast::channel::<WorkerEvent>(16);
        let (fin_tx, mut fin_rx) = broadcast::channel::<WorkerEvent>(16);
        tokio::spawn(run_job_sync(pool.clone(), raw_rx, fin_tx));

        raw_tx
            .send(WorkerEvent::JobError { job_id: "unwritten".into(), error: "insert failed".into() })
            .unwrap();
        let ev = tokio::time::timeout(Duration::from_secs(10), fin_rx.recv())
            .await
            .expect("timed out waiting for job_error")
            .unwrap();
        assert!(matches!(ev, WorkerEvent::JobError { job_id, .. } if job_id == "unwritten"));
        assert!(generation_jobs::get(&pool, "unwritten").await.unwrap().is_none());
    }
}
