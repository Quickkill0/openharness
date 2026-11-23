use std::path::PathBuf;

use tauri::ipc::Channel;
use tauri::State;

use crate::engine::{
    fit::{estimate_vram_mb, kv_cache_mb, verdict, FitEstimate, FitInputs, FIXED_OVERHEAD_MB},
    detect_variant, EngineStatus, EngineVariant,
};
use crate::error::AppError;
use crate::gpu::GpuStatus;
use crate::models::{gguf, registry};
use crate::state::AppState;

/// VRAM the running llama-server would release if replaced (loading a chat
/// model is a same-slot swap). Prefers the NVML per-process figure; under
/// WDDM (consumer-GPU Windows) per-process memory reads N/A, so fall back to
/// our own footprint estimate for the loaded model at its launch context,
/// capped by total used VRAM.
pub(crate) async fn engine_reclaimable_mb(state: &AppState, used_mb: u64) -> u64 {
    let EngineStatus::Ready { model_id, ctx, .. } = state.engine.status().await else {
        return 0;
    };
    if let Some(pid) = state.engine.server_pid().await {
        if let Some(mb) = state.gpu.process_used_mb(pid).await {
            return mb;
        }
    }
    // Native expert tiers are not llama's full-weight allocation. Under WDDM
    // NVML cannot report per-process use, so prefer the orchestrator's plan.
    if let Some((id, plan)) = state.orchestrator.current_text_plan().await {
        if id == model_id { return plan.tiers.vram_total().div_ceil(1024 * 1024).min(used_mb); }
    }
    let Ok(Some(model)) = registry::get_by_id(&state.db, &model_id).await else {
        return 0;
    };
    let meta = gguf::parse_header(&PathBuf::from(&model.primary_file_path)).unwrap_or_default();
    let inputs = FitInputs::from_meta(&meta, model.weight_size() / (1024 * 1024), ctx, 2);
    estimate_vram_mb(&inputs).min(used_mb)
}

/// Discovery honours a hand-placed binary even without a provision stamp.
#[tauri::command]
pub async fn native_engine_info(state: State<'_, AppState>, retry: Option<bool>) -> Result<crate::engine::native::NativeStatus, AppError> {
    if retry.unwrap_or(false) { state.engine.native_info_retry().await } else { state.engine.native_info().await }
}

#[tauri::command]
pub async fn native_engine_install() -> Result<(), AppError> {
    crate::engine::native::install().await
}

/// Return the current engine status without subscribing.
#[tauri::command]
pub async fn engine_status(state: State<'_, AppState>) -> Result<EngineStatus, AppError> {
    Ok(state.engine.status().await)
}

/// Return the persisted active engine row, if one is installed.
///
/// The live `EngineStatus::Idle` covers BOTH "no engine installed" and "engine
/// installed but no model loaded". This command lets the UI tell them apart by
/// reading the persisted catalog directly.
#[tauri::command]
pub async fn engine_active(
    state: State<'_, AppState>,
) -> Result<Option<serde_json::Value>, AppError> {
    // Stale builds read as not installed so the wizard re-provisions them.
    let row = crate::db::engines::get_current(&state.db).await?;
    Ok(row.map(|r| {
        serde_json::json!({
            "id":           r.id,
            "version":      r.version,
            "variant":      r.variant,
            "path":         r.path,
            "active":       r.active,
            "installed_at": r.installed_at,
        })
    }))
}

/// Install the engine, optionally specifying a variant.
///
/// When `variant` is `None` the GPU actor is queried and the appropriate
/// variant (CUDA 12.4 or CPU) is auto-detected.
#[tauri::command]
pub async fn engine_install(
    variant: Option<EngineVariant>,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, AppError> {
    let v = match variant {
        Some(v) => v,
        None => {
            let status = state.gpu.status().await;
            // CUDA-12-built binaries are forward-compatible with CUDA 13+ drivers.
            // We pin the build at 12.4 in v1; M4 tracked debt covers a multi-CUDA picker.
            let has_cuda_12 =
                matches!(&status, GpuStatus::Available(m) if m.cuda_major >= 12);
            detect_variant(has_cuda_12)
        }
    };
    let row = state.engine.install(v).await?;
    Ok(serde_json::json!({
        "id":           row.id,
        "version":      row.version,
        "variant":      row.variant,
        "path":         row.path,
        "installed_at": row.installed_at,
    }))
}

/// Re-install the engine with a different variant, deactivating the current one.
#[tauri::command]
pub async fn engine_switch_variant(
    variant: EngineVariant,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    let _ = state.engine.install(variant).await?;
    Ok(())
}

/// Last ~1000 stdout/stderr lines from the running llama-server (empty when
/// no model is loaded).
#[tauri::command]
pub async fn engine_logs(state: State<'_, AppState>) -> Result<Vec<String>, AppError> {
    Ok(state.engine.logs().await)
}

/// Unload the currently loaded model (stops llama-server). Routed through
/// the orchestrator so the VRAM ledger drops the text resident and
/// re-broadcasts — every "Loaded" surface reads that channel (M16 W1).
#[tauri::command]
pub async fn engine_unload_model(state: State<'_, AppState>) -> Result<(), AppError> {
    state
        .orchestrator
        .unload_if_idle(crate::orchestrator::TEXT_RESIDENT_ID.to_string())
        .await
}

/// Subscribe to engine status updates via a Tauri Channel.
///
/// Sends the current status immediately. At most one forwarder task runs at a
/// time — a second call aborts the previous one (single-subscriber semantics,
/// identical to `subscribe_download_progress`).
#[tauri::command]
pub async fn engine_subscribe_status(
    channel: Channel<EngineStatus>,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    let mut rx = state.engine.subscribe().await;

    // Send the current status immediately so the frontend doesn't wait for the
    // next broadcast tick.
    let _ = channel.send(state.engine.status().await);

    // tokio::spawn used (not tauri::async_runtime::spawn) because JoinHandle::abort_handle()
    // is required and tauri's JoinHandle wrapper does not expose that method.
    let join_handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(status) => {
                    if channel.send(status).is_err() {
                        // Frontend closed the channel.
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => {
                    // Fell behind — drop skipped frames, keep going.
                    continue;
                }
            }
        }
    });

    let mut guard = state.engine.subscriber_abort.lock().await;
    if let Some(prev) = guard.take() {
        prev.abort();
    }
    *guard = Some(join_handle.abort_handle());

    Ok(())
}

/// Return the current GPU status (single snapshot).
#[tauri::command]
pub async fn gpu_status(state: State<'_, AppState>) -> Result<GpuStatus, AppError> {
    Ok(state.gpu.status().await)
}

/// Subscribe to GPU status updates (~1 Hz) via a Tauri Channel.
///
/// Sends the current status immediately. This subscription has no
/// single-subscriber enforcement (GPU metrics are read-only; multiple
/// subscribers are harmless).
#[tauri::command]
pub async fn gpu_subscribe_metrics(
    channel: Channel<GpuStatus>,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    let mut rx = state.gpu.subscribe().await;

    // Snapshot immediately so the frontend doesn't wait up to 1 second.
    let _ = channel.send(state.gpu.status().await);

    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(status) => {
                    if channel.send(status).is_err() {
                        break;
                    }
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
            }
        }
    });

    Ok(())
}

/// Estimate how much VRAM a model will consume and whether it fits.
///
/// Installed text models use the same admission/placement as Load (including
/// CPU RAM limits and native availability). Non-text models retain their
/// legacy graphics-only estimate. Pre-download store_quant_fits is separate.
#[tauri::command]
pub async fn estimate_model_fit(
    model_id: String,
    ctx: Option<u32>,
    state: State<'_, AppState>,
) -> Result<FitEstimate, AppError> {
    let model = registry::get_by_id(&state.db, &model_id)
        .await?
        .ok_or_else(|| AppError::ModelNotFound(model_id.clone()))?;

    if model.caps.runtime == Some(crate::models::capabilities::RuntimeKind::LlamaServer) {
        let plan = state.orchestrator.plan_text(model_id, crate::planner::PlanOverrides { ctx, ..Default::default() }).await?;
        let metrics = state.gpu.latest_metrics().await;
        let free = metrics.as_ref().map_or(0, |m| m.vram_free_mb);
        let reclaim = if let Some(m) = &metrics { engine_reclaimable_mb(&state, m.vram_used_mb).await } else { 0 };
        return Ok(FitEstimate::from_plan(plan, metrics.is_some(), free, reclaim));
    }
    let path = PathBuf::from(&model.primary_file_path);

    let meta =
        gguf::parse_header(&path).map_err(|e| AppError::GgufParseFailed(e.to_string()))?;

    let file_size_mb = model.weight_size() / (1024 * 1024);

    let ctx_final = match ctx {
        Some(c) => c,
        None => model
            .context_train
            .map(|c| (c as u32).min(8192))
            .unwrap_or(4096),
    };

    // SWA-aware KV profile is derived inside `from_meta` (fp16 KV cache).
    let inputs = FitInputs::from_meta(&meta, file_size_mb, ctx_final, 2);

    // The fit estimate has two modes, keyed on whether the caller passed a
    // context:
    //   * No ctx (the /models list badge) → the **floor**: model weights +
    //     fixed backend/safety overhead, KV EXCLUDED. Answers "can this model
    //     load at all?" without a default-context KV scaring the verdict
    //     (folding KV in unconditionally produced false "Won't fit" — a 15.8 GB
    //     Q4_K_M reading ~25 GB at 8k ctx, the original bug).
    //   * A ctx (the load dialog slider) → floor + KV at that context, so the
    //     estimate and verdict TRACK the slider as the user raises context and
    //     watches VRAM grow. Freezing it at the floor here was the regression:
    //     the badge stayed "Fits • 17.8 GB" at 256k even as the hint showed KV.
    //
    // KV is SWA-aware (see fit::KvProfile): sliding-window models bill local
    // layers only up to their window, so the figure is accurate rather than the
    // old ~6–20× over-count that treated every layer as full attention.
    let kv_at_ctx_mb = if ctx.is_some() { kv_cache_mb(&inputs) } else { 0 };
    let estimated_mb = file_size_mb
        .saturating_add(FIXED_OVERHEAD_MB)
        .saturating_add(kv_at_ctx_mb);

    let (free_mb, used_mb, gpu) = match state.gpu.latest_metrics().await {
        Some(m) => (m.vram_free_mb, m.vram_used_mb, true),
        // No GPU present — treat as unlimited so verdict = Fits.
        None => (u64::MAX / 2, 0, false),
    };

    // Loading a chat model is a same-slot swap: the running llama-server is
    // replaced, releasing its VRAM first. Count that as available, or the
    // currently-loaded model judges itself against the memory it occupies.
    let reclaimable_mb = if gpu { engine_reclaimable_mb(&state, used_mb).await } else { 0 };

    let v = verdict(estimated_mb, free_mb.saturating_add(reclaimable_mb));

    Ok(FitEstimate {
        plan: None,
        estimated_mb,
        kv_at_ctx_mb,
        // Report 0 free_mb when there is no GPU (caller should check `gpu`
        // field to interpret this correctly).
        free_mb: if gpu { free_mb } else { 0 },
        reclaimable_mb,
        verdict: v,
        gpu,
    })
}
