//! Asset library commands (M10): list/search/delete generated artifacts and
//! reveal them in Explorer.

use tauri::State;

use crate::db::assets::{self, Asset};
use crate::error::AppError;
use crate::state::AppState;

/// Library contents, newest first. `kind` filters to image/video/audio/text.
#[tauri::command]
pub async fn list_assets(
    state: State<'_, AppState>,
    kind: Option<String>,
    limit: Option<u32>,
    offset: Option<u32>,
) -> Result<Vec<Asset>, AppError> {
    assets::list(
        &state.db,
        kind.as_deref(),
        limit.unwrap_or(100) as i64,
        offset.unwrap_or(0) as i64,
    )
    .await
}

/// Full-text search over prompt/model/task (prefix matching).
#[tauri::command]
pub async fn search_assets(
    state: State<'_, AppState>,
    query: String,
    limit: Option<u32>,
) -> Result<Vec<Asset>, AppError> {
    assets::search(&state.db, &query, limit.unwrap_or(100) as i64).await
}

/// Delete an asset row; with `delete_file` the artifact is removed from disk
/// too (only when it lives inside the app's assets dir — never anything else).
#[tauri::command]
pub async fn delete_asset(
    state: State<'_, AppState>,
    asset_id: String,
    delete_file: Option<bool>,
) -> Result<(), AppError> {
    let Some(asset) = assets::delete(&state.db, &asset_id).await? else {
        return Ok(());
    };
    if delete_file.unwrap_or(false) {
        let p = std::path::Path::new(&asset.path);
        if p.starts_with(crate::config::assets_dir()) {
            if let Err(e) = std::fs::remove_file(p) {
                if e.kind() != std::io::ErrorKind::NotFound {
                    tracing::warn!("could not delete asset file {}: {e}", asset.path);
                }
            }
        }
    }
    Ok(())
}

/// Open Windows Explorer with the asset's file selected.
#[tauri::command]
pub async fn reveal_asset(state: State<'_, AppState>, asset_id: String) -> Result<(), AppError> {
    let Some(asset) = assets::get(&state.db, &asset_id).await? else {
        return Err(AppError::Other(format!("unknown asset: {asset_id}")));
    };
    if !std::path::Path::new(&asset.path).exists() {
        return Err(AppError::Other("file no longer exists on disk".into()));
    }
    // `explorer /select,<path>` — exit status is unreliable (explorer often
    // returns 1 on success), so only spawn failures are errors.
    std::process::Command::new("explorer")
        .arg(format!("/select,{}", asset.path))
        .spawn()
        .map_err(|e| AppError::Other(format!("could not open Explorer: {e}")))?;
    Ok(())
}
