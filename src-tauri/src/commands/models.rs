use tauri::State;

use crate::{
    auth,
    error::AppError,
    models::{
        hf::{self, HfFile, HfModelDetails, HfSearchParams, HfSearchResponse},
        registry::{self, InstalledModel},
    },
    state::AppState,
};

/// Search the HuggingFace model catalog.
///
/// Uses the HF token from the OS keyring if one is stored; falls back to
/// anonymous (rate-limited) access when no token is present.
#[tauri::command]
pub async fn hf_search(
    state: State<'_, AppState>,
    params: HfSearchParams,
) -> Result<HfSearchResponse, AppError> {
    let token = auth::get_hf_token()?;
    hf::search_models(&state.http, params, token.as_deref()).await
}

/// Fetch full details (card metadata, README, file siblings) for a single model.
#[tauri::command]
pub async fn hf_model_details(
    state: State<'_, AppState>,
    repo_id: String,
) -> Result<HfModelDetails, AppError> {
    let token = auth::get_hf_token()?;
    hf::get_model_details(&state.http, &repo_id, token.as_deref()).await
}

/// Fetch the file tree for a model at a given revision (typically `"main"`).
#[tauri::command]
pub async fn hf_model_tree(
    state: State<'_, AppState>,
    repo_id: String,
    revision: String,
) -> Result<Vec<HfFile>, AppError> {
    let token = auth::get_hf_token()?;
    hf::get_model_tree(&state.http, &repo_id, &revision, token.as_deref()).await
}

// ---------------------------------------------------------------------------
// Local model catalog (M3)
// ---------------------------------------------------------------------------

/// List all locally installed models.
#[tauri::command]
pub async fn list_local_models(state: State<'_, AppState>) -> Result<Vec<InstalledModel>, AppError> {
    registry::list(&state.db).await
}

/// Delete an installed model from the catalog and remove its GGUF blob.
#[tauri::command]
pub async fn delete_local_model(state: State<'_, AppState>, id: String) -> Result<(), AppError> {
    registry::delete(&state.db, &id).await
}
