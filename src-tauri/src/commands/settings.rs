use serde::{Deserialize, Serialize};
use tauri::State;

use crate::{
    auth,
    config,
    db::settings as db_settings,
    error::AppError,
    state::AppState,
};

// ---------------------------------------------------------------------------
// Key/value settings
// ---------------------------------------------------------------------------

#[tauri::command]
pub async fn get_setting(
    state: State<'_, AppState>,
    key: String,
) -> Result<Option<String>, AppError> {
    db_settings::get_setting(&state.db, &key).await
}

#[tauri::command]
pub async fn set_setting(
    state: State<'_, AppState>,
    key: String,
    value: String,
) -> Result<(), AppError> {
    db_settings::set_setting(&state.db, &key, &value).await
}

// ---------------------------------------------------------------------------
// HF Token (keyring)
// ---------------------------------------------------------------------------

#[tauri::command]
pub fn get_hf_token() -> Result<Option<String>, AppError> {
    auth::get_hf_token()
}

#[tauri::command]
pub fn set_hf_token(token: String) -> Result<(), AppError> {
    auth::set_hf_token(&token)
}

#[tauri::command]
pub fn clear_hf_token() -> Result<(), AppError> {
    auth::clear_hf_token()
}

/// HuggingFace whoami response (partial — we only surface what the UI needs).
#[derive(Debug, Serialize, Deserialize)]
pub struct HfWhoami {
    pub name: String,
    pub fullname: Option<String>,
    pub email: Option<String>,
}

#[tauri::command]
pub async fn validate_hf_token(
    state: State<'_, AppState>,
) -> Result<HfWhoami, AppError> {
    // Use the shared HTTP client from AppState (replaces the per-call allocation from M1).
    let json = auth::validate_hf_token(&state.http).await?;

    let name = json["name"]
        .as_str()
        .ok_or_else(|| AppError::Other("Unexpected whoami response: missing 'name'".into()))?
        .to_string();

    let fullname = json["fullname"].as_str().map(|s| s.to_string());
    let email = json["email"].as_str().map(|s| s.to_string());

    Ok(HfWhoami { name, fullname, email })
}

// ---------------------------------------------------------------------------
// App paths
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
pub struct AppPaths {
    pub app_data_dir: String,
    pub db_path: String,
    pub models_dir: String,
    pub transcripts_dir: String,
    pub engine_dir: String,
}

#[tauri::command]
pub fn app_paths() -> Result<AppPaths, AppError> {
    Ok(AppPaths {
        app_data_dir: config::app_data_dir().to_string_lossy().into_owned(),
        db_path: config::db_path().to_string_lossy().into_owned(),
        models_dir: config::models_dir().to_string_lossy().into_owned(),
        transcripts_dir: config::transcripts_dir().to_string_lossy().into_owned(),
        engine_dir: config::engine_dir().to_string_lossy().into_owned(),
    })
}

// ---------------------------------------------------------------------------
// Rebrand migration notice
// ---------------------------------------------------------------------------

/// Returns `true` exactly once if the Hangar→OpenHarness data migration ran
/// this launch. The frontend calls this on boot to show a one-off toast;
/// subsequent calls return `false`.
#[tauri::command]
pub fn consume_migration_notice(state: State<'_, AppState>) -> bool {
    state
        .migrated
        .swap(false, std::sync::atomic::Ordering::SeqCst)
}
