use tauri::{ipc::Channel, State};
use uuid::Uuid;

use crate::{
    db::downloads::GroupRow,
    downloader::{job::Job, progress::DownloadProgress},
    error::AppError,
    models::hf::HfFile,
    state::AppState,
};

/// Queue a new download for a HuggingFace file.
/// Returns the new job UUID.
#[tauri::command]
pub async fn start_download(
    state: State<'_, AppState>,
    repo_id: String,
    revision: String,
    file: HfFile,
) -> Result<Uuid, AppError> {
    state.downloads.start(repo_id, revision, file).await
}

/// Queue a multi-file repo download (M9 §9.2): one job per file under a
/// shared download group. Returns the group UUID.
#[tauri::command]
pub async fn start_repo_download(
    state: State<'_, AppState>,
    repo_id: String,
    revision: String,
    files: Vec<HfFile>,
    format: String,
    task: Option<String>,
) -> Result<Uuid, AppError> {
    state
        .downloads
        .start_repo(repo_id, revision, files, format, task)
        .await
}

/// Pause every member job of a download group.
#[tauri::command]
pub async fn pause_download_group(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.pause_group(id).await;
    Ok(())
}

/// Resume every paused/failed member job of a download group.
#[tauri::command]
pub async fn resume_download_group(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.resume_group(id).await;
    Ok(())
}

/// Cancel every member job of a download group.
#[tauri::command]
pub async fn cancel_download_group(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.cancel_group(id).await;
    Ok(())
}

/// Delete a download group and all of its member jobs.
#[tauri::command]
pub async fn delete_download_group(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.delete_group(id).await;
    Ok(())
}

/// List all download groups. Group rows are immutable after creation, so this
/// reads the DB directly rather than round-tripping through the actor.
#[tauri::command]
pub async fn list_download_groups(state: State<'_, AppState>) -> Result<Vec<GroupRow>, AppError> {
    crate::db::downloads::list_groups(&state.db).await
}

/// Pause an in-progress download.
#[tauri::command]
pub async fn pause_download(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.pause(id).await;
    Ok(())
}

/// Resume a paused or failed download.
#[tauri::command]
pub async fn resume_download(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.resume(id).await;
    Ok(())
}

/// Cancel a download (stops transfer, keeps partial data in staging).
#[tauri::command]
pub async fn cancel_download(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.cancel(id).await;
    Ok(())
}

/// Delete a download job and remove its staging data.
#[tauri::command]
pub async fn delete_download(state: State<'_, AppState>, id: Uuid) -> Result<(), AppError> {
    state.downloads.delete(id).await;
    Ok(())
}

/// List all download jobs (current session + persisted).
#[tauri::command]
pub async fn list_downloads(state: State<'_, AppState>) -> Result<Vec<Job>, AppError> {
    Ok(state.downloads.list().await)
}

/// Subscribe to live download progress via a Tauri Channel.
///
/// At most one forwarder task runs at a time (v1 constraint). If called again
/// (e.g. after Vite HMR restart), the previous forwarder is aborted before a
/// new one is spawned, preventing duplicate event delivery.
#[tauri::command]
pub async fn subscribe_download_progress(
    state: State<'_, AppState>,
    channel: Channel<DownloadProgress>,
) -> Result<(), AppError> {
    let mut rx = state.downloads.subscribe_progress();

    // Bridge broadcast → Channel in a spawned task; capture the abort handle.
    // tokio::spawn is valid here — this function runs inside the Tauri async
    // runtime (tokio), so the reactor is already up.
    let join_handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(progress) => {
                    if channel.send(progress).is_err() {
                        // Frontend closed the channel
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // Fell behind — continue; some frames will be dropped
                    continue;
                }
            }
        }
    });

    // Register the abort handle so a second call aborts this task first.
    state.downloads.set_subscriber_abort(join_handle.abort_handle());

    Ok(())
}
