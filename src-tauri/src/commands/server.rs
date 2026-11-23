use crate::{
    api::{ConnectionInfo, RequestLog, ServerSettings, ServerStatus},
    state::AppState,
};
use tauri::{ipc::Channel, State};

#[tauri::command]
pub async fn server_status(state: State<'_, AppState>) -> Result<ServerStatus, String> {
    state.server.status().await.map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_start(state: State<'_, AppState>) -> Result<ServerStatus, String> {
    state
        .server
        .set_enabled(true)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_stop(state: State<'_, AppState>) -> Result<ServerStatus, String> {
    state
        .server
        .set_enabled(false)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_update_settings(
    state: State<'_, AppState>,
    settings: ServerSettings,
) -> Result<ServerStatus, String> {
    state
        .server
        .update(settings)
        .await
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_regenerate_key(state: State<'_, AppState>) -> Result<(), String> {
    state
        .server
        .key(true)
        .await
        .map(|_| ())
        .map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_get_key(state: State<'_, AppState>) -> Result<Option<String>, String> {
    state.server.key(false).await.map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_request_log(state: State<'_, AppState>) -> Result<Vec<RequestLog>, String> {
    state.server.logs().await.map_err(|e| e.to_string())
}
#[tauri::command]
pub async fn server_subscribe_request_log(
    state: State<'_, AppState>,
    channel: Channel<RequestLog>,
) -> Result<(), String> {
    let mut rx = state.server.subscribe();
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(log) => {
                    if channel.send(log).is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }
    });
    Ok(())
}
#[tauri::command]
pub async fn server_connection_info(state: State<'_, AppState>) -> Result<ConnectionInfo, String> {
    state.server.connections().await.map_err(|e| e.to_string())
}
