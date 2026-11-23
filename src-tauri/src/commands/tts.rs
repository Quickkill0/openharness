//! Qwen3-TTS engine commands: status (with voices), logs, unload, install,
//! and the status subscription channel.
//! Loading happens implicitly through `synthesize_speech` — there is no
//! standalone load command; the orchestrator gates it per job.

use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::broadcast;

use crate::error::AppError;
use crate::state::AppState;
use crate::tts::TtsStatus;

/// Current sidecar status. `ready` carries the port and the talker's
/// speaker names (empty for base/cloning talkers).
#[tauri::command]
pub async fn tts_status(state: State<'_, AppState>) -> Result<TtsStatus, AppError> {
    Ok(state.tts.status().await)
}

/// Whether the qwentts engine files are present on disk
/// (`engine\qwentts\tts-server.exe` + bundled codec GGUF).
#[tauri::command]
pub async fn tts_installed() -> Result<bool, AppError> {
    Ok(crate::tts::is_installed())
}

/// Whether the one-shot `qwen-tts.exe` CLI (used for voice cloning with
/// base talkers) is present next to tts-server.exe.
#[tauri::command]
pub async fn tts_cli_installed() -> Result<bool, AppError> {
    Ok(crate::tts::cli::cli_installed())
}

/// Stop the sidecar and release its VRAM. Routed through the orchestrator
/// so the ledger updates and broadcasts.
#[tauri::command]
pub async fn tts_unload(state: State<'_, AppState>) -> Result<(), AppError> {
    state
        .orchestrator
        .unload_if_idle(crate::orchestrator::TTS_RESIDENT_ID.into())
        .await
}

/// Recent tts-server log lines (live buffer, or the last run's tail after
/// an unload/crash).
#[tauri::command]
pub async fn tts_logs(state: State<'_, AppState>) -> Result<Vec<String>, AppError> {
    Ok(state.tts.logs().await)
}

/// Provision the qwentts runtime via the generic pin-and-fetch installer.
/// Progress streams on `subscribe_tts_status` (`installing` → `idle`/`failed`).
/// No-op when already installed. **Today the binaries aren't hosted**, so a
/// real call returns an honest "binaries not hosted yet" error (see CLAUDE.md).
#[tauri::command]
pub async fn tts_install(state: State<'_, AppState>) -> Result<(), AppError> {
    state.tts.install().await
}

/// Subscribe to qwentts sidecar status (incl. install progress). Snapshots the
/// current status immediately, then streams updates. Mirrors
/// `subscribe_worker_status`.
#[tauri::command]
pub async fn subscribe_tts_status(
    state: State<'_, AppState>,
    channel: Channel<TtsStatus>,
) -> Result<(), AppError> {
    let snapshot = state.tts.status().await;
    let _ = channel.send(snapshot);

    let mut rx = state.tts.subscribe().await;
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

    let mut abort_guard = state.tts.subscriber_abort.lock().await;
    if let Some(h) = abort_guard.take() {
        h.abort();
    }
    *abort_guard = Some(join_handle.abort_handle());
    Ok(())
}
