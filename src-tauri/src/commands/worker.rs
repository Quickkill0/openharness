//! Python-worker commands (M8): install/start/stop/status, log tail, and
//! the status subscription channel.

use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::broadcast;

use crate::error::AppError;
use crate::state::AppState;
use crate::worker::{WorkerStatus, WorkerVariant};

#[tauri::command]
pub async fn worker_status(state: State<'_, AppState>) -> Result<WorkerStatus, AppError> {
    Ok(state.worker.status().await)
}

/// Install (or rebuild) the worker venv. `variant` is "cuda" | "cpu" —
/// the frontend picks it the same way the engine wizard does, from
/// `gpu_status`.
#[tauri::command]
pub async fn worker_install(
    state: State<'_, AppState>,
    variant: String,
) -> Result<(), AppError> {
    let variant = WorkerVariant::from_slug(&variant)
        .ok_or_else(|| AppError::Other(format!("unknown worker variant: {variant}")))?;
    state.worker.install(variant).await
}

#[tauri::command]
pub async fn worker_start(
    state: State<'_, AppState>,
) -> Result<Vec<crate::worker::PipelineCapability>, AppError> {
    state.worker.start_worker().await
}

#[tauri::command]
pub async fn worker_stop(state: State<'_, AppState>) -> Result<(), AppError> {
    state.worker.stop_worker().await
}

/// Last ~1000 stderr lines from the running worker (empty when stopped).
#[tauri::command]
pub async fn worker_logs(state: State<'_, AppState>) -> Result<Vec<String>, AppError> {
    Ok(state.worker.logs().await)
}

#[tauri::command]
pub async fn subscribe_worker_status(
    state: State<'_, AppState>,
    channel: Channel<WorkerStatus>,
) -> Result<(), AppError> {
    // Snapshot first so the UI renders immediately.
    let snapshot = state.worker.status().await;
    let _ = channel.send(snapshot);

    let mut rx = state.worker.subscribe().await;
    let join_handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(status) => {
                    if channel.send(status).is_err() {
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    });

    let mut abort_guard = state.worker.subscriber_abort.lock().await;
    if let Some(h) = abort_guard.take() {
        h.abort();
    }
    *abort_guard = Some(join_handle.abort_handle());
    Ok(())
}
