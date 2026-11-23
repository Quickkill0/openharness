//! Load planner commands: hardware profile, the on-demand SSD benchmark,
//! load plans, and the live runtime-stats stream.

use std::path::PathBuf;

use tauri::ipc::Channel;
use tauri::State;

use crate::error::AppError;
use crate::models::registry;
use crate::planner::hardware::{self, HardwareProfile, SsdMeasurement};
use crate::planner::runtime::RuntimeStats;
use crate::planner::{context_memory as context_memory_for, ContextMemory, ContextParams, LoadPlan, PlanOverrides};
use crate::state::AppState;

/// The machine the planner places models onto: GPU, RAM, the drive holding
/// the models folder and the bandwidths the speed estimate uses.
#[tauri::command]
pub async fn hardware_profile(state: State<'_, AppState>) -> Result<HardwareProfile, AppError> {
    let metrics = state.gpu.latest_metrics().await;
    let ssd = state.planner.ssd_measurements().await;
    let dir = crate::config::models_dir();
    tokio::task::spawn_blocking(move || hardware::probe(metrics.as_ref(), &dir, &ssd))
        .await
        .map_err(|e| AppError::Other(format!("hardware probe failed: {e}")))
}

/// Measure sequential read speed of the drive holding a model (or, without
/// `model_id`, the drive holding the models folder). Reads at most 4 GiB of
/// one existing model file with unbuffered I/O, for at most 15 s. Explicit
/// user action only; never runs on its own.
#[tauri::command]
pub async fn benchmark_ssd(
    state: State<'_, AppState>,
    model_id: Option<String>,
) -> Result<SsdMeasurement, AppError> {
    let file: PathBuf = match model_id {
        Some(id) => {
            let m = registry::get_by_id(&state.db, &id)
                .await?
                .ok_or_else(|| AppError::ModelNotFound(id.clone()))?;
            PathBuf::from(m.primary_file_path)
        }
        None => {
            let dir = crate::config::models_dir();
            tokio::task::spawn_blocking(move || hardware::largest_file(&dir))
                .await
                .map_err(|e| AppError::Other(e.to_string()))?
                .ok_or_else(|| {
                    AppError::Other(
                        "No model files to read yet: download a model, then run the SSD test.".into(),
                    )
                })?
        }
    };
    let measurement = tokio::task::spawn_blocking(move || -> Result<SsdMeasurement, AppError> {
        let dir = file.parent().map(PathBuf::from).unwrap_or_default();
        let mount = hardware::storage_for(&dir)
            .map(|s| s.mount)
            .ok_or_else(|| AppError::Other("Couldn't identify the drive holding the model.".into()))?;
        let r = hardware::benchmark_ssd(&file, hardware::BENCH_MAX_BYTES, hardware::BENCH_MAX_SECS)?;
        if !r.unbuffered {
            tracing::warn!("SSD benchmark ran buffered; the figure may include cached reads");
        }
        if r.bytes_read < 64 * 1024 * 1024 {
            return Err(AppError::Other(
                "The model file is too small for a meaningful SSD test (needs at least 64 MB).".into(),
            ));
        }
        Ok(SsdMeasurement {
            mount,
            gbps: r.gbps,
            bytes_read: r.bytes_read,
            seconds: r.seconds,
            measured_at: chrono_now(),
            file: r.file,
        })
    })
    .await
    .map_err(|e| AppError::Other(e.to_string()))??;
    state.planner.set_ssd_measurement(measurement.clone()).await;
    Ok(measurement)
}

fn chrono_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64)
}

/// Plan a text-model load without loading it. Empty overrides give the
/// one-click auto plan.
#[tauri::command]
pub async fn plan_load(
    state: State<'_, AppState>,
    model_id: String,
    overrides: Option<PlanOverrides>,
) -> Result<LoadPlan, AppError> {
    state.orchestrator.plan_text(model_id, overrides.unwrap_or_default()).await
}

/// What a context costs for a model: KV, sliding-window, indexer and
/// recurrent-state memory plus compute buffers, at `params` (whose `ctx` is
/// used when `contexts` is empty) or at each length in `contexts`. The
/// planner uses the same function, so a context slider and a load agree.
#[tauri::command]
pub async fn context_memory(
    state: State<'_, AppState>,
    model_id: String,
    params: Option<ContextParams>,
    contexts: Option<Vec<u32>>,
) -> Result<Vec<ContextMemory>, AppError> {
    let acc = state.orchestrator.model_accounting(model_id).await?;
    if let Some(why) = &acc.unsupported {
        return Err(AppError::Other(format!(
            "Can't estimate context memory for this {} model: {why}.",
            acc.architecture
        )));
    }
    let params = params.unwrap_or_default();
    let ctxs = contexts.filter(|c| !c.is_empty()).unwrap_or_else(|| vec![params.ctx]);
    Ok(ctxs
        .into_iter()
        .map(|ctx| context_memory_for(&acc, &ContextParams { ctx, ..params }))
        .collect())
}

/// Stream live actuals for the loaded text model (about once a second while
/// something changes). Sends the current snapshot first.
#[tauri::command]
pub async fn subscribe_runtime_stats(
    state: State<'_, AppState>,
    channel: Channel<RuntimeStats>,
) -> Result<(), AppError> {
    let mut rx = state.runtime_stats.clone();
    let first = rx.borrow_and_update().clone();
    let _ = channel.send(first);
    tokio::spawn(async move {
        while rx.changed().await.is_ok() {
            let snap = rx.borrow_and_update().clone();
            if channel.send(snap).is_err() {
                break;
            }
        }
    });
    Ok(())
}
