//! Store commands (M9 §9.4): batch runnability + curated picks.

use tauri::State;

use crate::error::AppError;
use crate::gpu::GpuStatus;
use crate::models::hf::HfFile;
use crate::state::AppState;
use crate::store::{self, CuratedPick, QuantFit, Runnability, RunnabilityQuery};
use crate::worker::WorkerStatus;

/// Batch runnability resolution for search pages — one entry per query, in
/// order. Uses live worker pipeline capabilities when the worker is running,
/// falling back to the static pipeline table otherwise.
#[tauri::command]
pub async fn store_runnability(
    state: State<'_, AppState>,
    items: Vec<RunnabilityQuery>,
) -> Result<Vec<Runnability>, AppError> {
    let pipelines = match state.worker.status().await {
        WorkerStatus::Ready { pipelines, .. } => pipelines,
        _ => store::static_pipelines(),
    };
    Ok(items.iter().map(|q| store::resolve(q, &pipelines)).collect())
}

/// Curated picks, optionally filtered to one store tab's task, with `fits`
/// resolved against the detected GPU's total VRAM.
#[tauri::command]
pub async fn store_curated_picks(
    state: State<'_, AppState>,
    task: Option<String>,
) -> Result<Vec<CuratedPick>, AppError> {
    let vram_total_mb = match state.gpu.status().await {
        GpuStatus::Available(m) => Some(m.vram_total_mb),
        GpuStatus::NoGpu => None,
    };
    Ok(store::curated_picks(task.as_deref(), vram_total_mb))
}

/// Per-file GGUF quant fit verdicts for a repo tree (M9 §9.6), with the
/// largest fitting single-file quant flagged `best`. Empty when no NVIDIA
/// GPU is detected — the feature is VRAM-aware by definition.
#[tauri::command]
pub async fn store_quant_fits(
    state: State<'_, AppState>,
    files: Vec<HfFile>,
) -> Result<Vec<QuantFit>, AppError> {
    let (free_mb, used_mb) = match state.gpu.status().await {
        GpuStatus::Available(m) => (m.vram_free_mb, m.vram_used_mb),
        GpuStatus::NoGpu => return Ok(vec![]),
    };
    // Loading a GGUF replaces the current engine model (same-slot swap), so
    // VRAM held by the running llama-server counts as available.
    let reclaimable_mb = crate::commands::engine::engine_reclaimable_mb(&state, used_mb).await;
    Ok(store::quant_fits(&files, free_mb.saturating_add(reclaimable_mb)))
}
