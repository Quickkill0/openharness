//! Generation commands (M8 §8.5): submit jobs to the Python worker's
//! pipelines, stream progress, and manage the persisted job history.

use serde_json::Value;
use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::broadcast;

use crate::db::generation_jobs::{self, GenerationJob};
use crate::error::AppError;
use crate::generation::{self, SubmitOutcome};
use crate::state::AppState;
use crate::worker::protocol::{WorkerEvent, WorkerRequest};

async fn submit(
    state: &AppState,
    task: &str,
    params: Value,
    force: Option<bool>,
) -> Result<SubmitOutcome, AppError> {
    generation::submit(
        &state.db,
        &state.worker,
        &state.orchestrator,
        task,
        params,
        force.unwrap_or(false),
    )
    .await
}

/// Submit a text-to-image job. `params` go to the pipeline verbatim
/// (model/prompt/width/height/steps/…); see `worker-py` pipeline docstrings.
/// `force: true` skips the VRAM eviction gate.
#[tauri::command]
pub async fn generate_image(
    state: State<'_, AppState>,
    params: Value,
    force: Option<bool>,
) -> Result<SubmitOutcome, AppError> {
    submit(&state, "text_to_image", params, force).await
}

/// Submit a text-to-video job (LTX-Video / Wan-class diffusers dirs).
#[tauri::command]
pub async fn generate_video(
    state: State<'_, AppState>,
    params: Value,
    force: Option<bool>,
) -> Result<SubmitOutcome, AppError> {
    submit(&state, "text_to_video", params, force).await
}

/// Submit a text-to-speech job. Routes by `params.engine`: `"qwen3-tts"`
/// with `reference_audio` is a voice clone via the one-shot `qwen-tts` CLI
/// (base talkers only); plain `"qwen3-tts"` goes to the qwentts.cpp sidecar
/// (auto-loading the talker through the orchestrator's gated `"tts"` slot);
/// `"vibevoice"` — or any other request carrying `reference_audio` — is the
/// voice-cloning worker pipeline; anything else is the Kokoro worker
/// pipeline.
#[tauri::command]
pub async fn synthesize_speech(
    state: State<'_, AppState>,
    params: Value,
    force: Option<bool>,
) -> Result<SubmitOutcome, AppError> {
    let engine = params.get("engine").and_then(|v| v.as_str());
    if engine == Some("qwen3-tts") && params.get("reference_audio").is_some() {
        return crate::tts::cli::submit_clone(
            &state.db,
            &state.worker,
            &state.orchestrator,
            &state.tts,
            params,
            force.unwrap_or(false),
        )
        .await;
    }
    if engine == Some("qwen3-tts") {
        return generation::submit_qwen_tts(
            &state.db,
            &state.worker,
            &state.orchestrator,
            &state.tts,
            params,
            force.unwrap_or(false),
        )
        .await;
    }
    if engine == Some("vibevoice") || params.get("reference_audio").is_some() {
        return submit(&state, "vibevoice", params, force).await;
    }
    submit(&state, "text_to_speech", params, force).await
}

/// Submit a transcription job (faster-whisper).
#[tauri::command]
pub async fn transcribe_audio(
    state: State<'_, AppState>,
    params: Value,
    force: Option<bool>,
) -> Result<SubmitOutcome, AppError> {
    submit(&state, "transcribe", params, force).await
}

/// Job history, newest first. `task` filters to one pipeline.
#[tauri::command]
pub async fn list_generation_jobs(
    state: State<'_, AppState>,
    task: Option<String>,
    limit: Option<u32>,
) -> Result<Vec<GenerationJob>, AppError> {
    generation_jobs::list(&state.db, task.as_deref(), limit.unwrap_or(100) as i64).await
}

/// Jobs still running, with no history cutoff: running rows plus jobs whose
/// reservation is granted but whose row isn't written yet. A (re)loaded UI
/// adopts these so its unload locks match the backend.
#[tauri::command]
pub async fn active_generation_jobs(
    state: State<'_, AppState>,
) -> Result<Vec<crate::orchestrator::ActiveJob>, AppError> {
    state.orchestrator.active_jobs().await
}

/// Request cancellation of a queued/running job. The terminal `cancelled`
/// status lands via the job-event sync once the worker confirms.
#[tauri::command]
pub async fn cancel_generation_job(
    state: State<'_, AppState>,
    job_id: String,
) -> Result<(), AppError> {
    // Voice-clone jobs run as a local one-shot process, not in the worker —
    // if the id is registered there, killing it is the whole cancellation.
    if crate::tts::cli::cancel(&job_id) {
        return Ok(());
    }
    state
        .worker
        .request(
            WorkerRequest::Cancel { job_id },
            std::time::Duration::from_secs(10),
        )
        .await?;
    Ok(())
}

/// Delete a job row; with `delete_files` its outputs are removed from disk
/// too (only paths inside the app's assets dir — never anything else).
#[tauri::command]
pub async fn delete_generation_job(
    state: State<'_, AppState>,
    job_id: String,
    delete_files: Option<bool>,
) -> Result<(), AppError> {
    if delete_files.unwrap_or(false) {
        if let Some(job) = generation_jobs::get(&state.db, &job_id).await? {
            let assets = crate::config::assets_dir();
            for out in &job.outputs {
                let p = std::path::Path::new(out);
                if p.starts_with(&assets) {
                    if let Err(e) = std::fs::remove_file(p) {
                        if e.kind() != std::io::ErrorKind::NotFound {
                            tracing::warn!("could not delete output {out}: {e}");
                        }
                    }
                }
            }
        }
    }
    generation_jobs::delete(&state.db, &job_id).await
}

/// Persist a microphone capture from the webview so the transcribe pipeline
/// can read it (M11 §11.3 voice input). Takes base64 audio (webm/opus from
/// MediaRecorder), writes it under the assets dir, returns the absolute path.
#[tauri::command]
pub async fn save_voice_capture(
    data_b64: String,
    ext: Option<String>,
) -> Result<String, AppError> {
    use base64::Engine as _;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(data_b64.as_bytes())
        .map_err(|e| AppError::Other(format!("invalid base64 audio payload: {e}")))?;
    if bytes.is_empty() {
        return Err(AppError::Other("empty audio payload".into()));
    }

    let ext = ext.unwrap_or_else(|| "webm".into());
    if !ext.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(AppError::Other(format!("invalid extension: {ext}")));
    }
    let dir = crate::config::assets_dir().join("captures");
    tokio::fs::create_dir_all(&dir).await?;
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let path = dir.join(format!("voice-{ts}.{ext}"));
    tokio::fs::write(&path, &bytes).await?;
    Ok(path.to_string_lossy().into_owned())
}

/// Subscribe to the live job-event stream (progress/done/error for ALL jobs;
/// the frontend correlates by `job_id`). Single-subscriber semantics.
///
/// Subscribes to the *finalized* bus from `generation::spawn_job_sync`, not
/// the raw worker bus: `JobDone.outputs` are the promoted asset-library paths
/// (the raw event carries staging paths that promotion renames away — the
/// webview would 404 loading them through the asset protocol).
#[tauri::command]
pub async fn subscribe_generation_progress(
    state: State<'_, AppState>,
    channel: Channel<WorkerEvent>,
) -> Result<(), AppError> {
    let mut rx = state.gen_events.subscribe();
    let join_handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                // The job-event channel never carries Log events, but filter
                // defensively so the frontend only sees job_* shapes.
                Ok(WorkerEvent::Log { .. }) => continue,
                Ok(ev) => {
                    if channel.send(ev).is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    });

    let mut abort_guard = state.worker.job_subscriber_abort.lock().await;
    if let Some(h) = abort_guard.take() {
        h.abort();
    }
    *abort_guard = Some(join_handle.abort_handle());
    Ok(())
}
