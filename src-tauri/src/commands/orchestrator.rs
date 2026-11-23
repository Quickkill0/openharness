//! Orchestrator commands (M8 §8.4): VRAM ledger queries and the gated
//! text-model load path, plus the unified load-first commands (M16 W1) that
//! dispatch on a model's capability record across all three engine families.

use serde::Serialize;
use tauri::ipc::Channel;
use tauri::State;
use tokio::sync::broadcast;

use crate::error::AppError;
use crate::generation::{self, PreloadOutcome};
use crate::models::capabilities::RuntimeKind;
use crate::models::registry;
use crate::orchestrator::{
    LedgerSnapshot, LoadOutcome, ResidentInfo, TextLoadRequest, TEXT_RESIDENT_ID,
    TTS_RESIDENT_ID,
};
use crate::planner::{LoadPlan, PlanOverrides};
use crate::state::AppState;

/// Result of the unified `load_model` command.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum ModelLoadOutcome {
    Loaded {
        model_id: String,
        /// Ledger resident id (`"text"`, `"tts"`, `"pipeline:<task>"`).
        resident_id: String,
        /// The load plan used (llama-server models only).
        #[serde(skip_serializing_if = "Option::is_none")]
        plan: Option<Box<LoadPlan>>,
    },
    NeedsEviction {
        required_mb: u64,
        free_mb: u64,
        candidates: Vec<ResidentInfo>,
    },
}

/// Load any runnable model into VRAM, routed by its capability record:
/// llama-server GGUFs through the orchestrator's gated text slot, Qwen3-TTS
/// talkers through the tts sidecar, worker-pipeline models through the new
/// explicit preload (M16 W1). `NeedsEviction` means nothing was loaded — the
/// UI presents the eviction choice and retries (or re-sends `force: true`).
///
/// For llama-server models the load planner places the weights: `plan` (from
/// `plan_load`) is realised as-is when it still fits, otherwise the planner
/// runs with `overrides`; neither gives the one-click auto plan.
#[tauri::command]
pub async fn load_model(
    state: State<'_, AppState>,
    model_id: String,
    force: Option<bool>,
    plan: Option<LoadPlan>,
    overrides: Option<PlanOverrides>,
) -> Result<ModelLoadOutcome, AppError> {
    let force = force.unwrap_or(false);
    let model = registry::get_by_id(&state.db, &model_id)
        .await?
        .ok_or_else(|| AppError::ModelNotFound(model_id.clone()))?;

    match model.caps.runtime {
        Some(RuntimeKind::LlamaServer) => {
            let req = TextLoadRequest {
                model_id: model_id.clone(),
                force,
                plan,
                overrides,
                ..Default::default()
            };
            match state.orchestrator.load_text(req).await? {
                LoadOutcome::Loaded { model_id, plan, .. } => Ok(ModelLoadOutcome::Loaded {
                    model_id,
                    resident_id: TEXT_RESIDENT_ID.into(),
                    plan,
                }),
                LoadOutcome::NeedsEviction { required_mb, free_mb, candidates } => {
                    Ok(ModelLoadOutcome::NeedsEviction { required_mb, free_mb, candidates })
                }
            }
        }
        Some(RuntimeKind::QwenTts) => {
            match state.orchestrator.load_tts(model_id.clone(), force, None).await? {
                crate::orchestrator::ReserveOutcome::Reserved => Ok(ModelLoadOutcome::Loaded {
                    model_id,
                    resident_id: TTS_RESIDENT_ID.into(),
                    plan: None,
                }),
                crate::orchestrator::ReserveOutcome::NeedsEviction {
                    required_mb,
                    free_mb,
                    candidates,
                } => Ok(ModelLoadOutcome::NeedsEviction { required_mb, free_mb, candidates }),
            }
        }
        Some(RuntimeKind::Worker) => {
            let outcome =
                generation::preload_pipeline(&state.worker, &state.orchestrator, &model, force)
                    .await?;
            match outcome {
                PreloadOutcome::Loaded { .. } => {
                    let task = model.caps.pipeline.as_deref().unwrap_or_default();
                    Ok(ModelLoadOutcome::Loaded {
                        model_id,
                        resident_id: format!("pipeline:{task}"),
                        plan: None,
                    })
                }
                PreloadOutcome::NeedsEviction { required_mb, free_mb, candidates } => {
                    Ok(ModelLoadOutcome::NeedsEviction { required_mb, free_mb, candidates })
                }
            }
        }
        None => Err(AppError::Other(
            model
                .caps
                .reason
                .unwrap_or_else(|| "No local runtime recognizes this model".into()),
        )),
    }
}

/// Unload a model by catalog id: finds its ledger resident and frees it.
/// A model that isn't resident unloads as a no-op.
#[tauri::command]
pub async fn unload_model(
    state: State<'_, AppState>,
    model_id: String,
) -> Result<(), AppError> {
    let snapshot = state.orchestrator.ledger().await?;
    if let Some(resident) = snapshot
        .residents
        .iter()
        .find(|r| r.model_id.as_deref() == Some(model_id.as_str()))
    {
        // Re-checked inside the actor: if another load swapped this slot
        // between the snapshot and now, the unload refuses instead of freeing
        // the newcomer.
        state
            .orchestrator
            .unload_if_model(resident.id.clone(), Some(model_id.clone()))
            .await?;
    } else {
        // The catalog id isn't resident — nothing to free. A stale "Loaded"
        // chip that triggers an unload on an already-unloaded model lands here
        // harmlessly; logging it makes that visible rather than silent.
        tracing::warn!(
            target: "unload",
            requested = %model_id,
            "unload_model: no resident matches this catalog id — nothing to unload"
        );
    }
    Ok(())
}

/// Load a text model through the orchestrator's VRAM gate.
///
/// Returns `Loaded { port, .. }` on success, or `NeedsEviction { candidates }`
/// when the model would only fully fit after freeing another resident —
/// nothing is loaded or evicted in that case; the UI presents the choice and
/// retries (after `orchestrator_unload`) or re-sends with `force: true`.
#[tauri::command]
pub async fn orchestrator_load_model(
    state: State<'_, AppState>,
    req: TextLoadRequest,
) -> Result<LoadOutcome, AppError> {
    state.orchestrator.load_text(req).await
}

/// Unload a resident by ledger id: `"text"` or `"pipeline:<task>"`.
#[tauri::command]
pub async fn orchestrator_unload(
    state: State<'_, AppState>,
    resident_id: String,
) -> Result<(), AppError> {
    state.orchestrator.unload_if_idle(resident_id).await
}

/// Unload a resident only if it still holds the model the caller saw
/// (`expected_model_id`, None for an uncorrelated resident). An empty slot is
/// a no-op; a slot that now holds another model returns an error.
#[tauri::command]
pub async fn orchestrator_unload_if(
    state: State<'_, AppState>,
    resident_id: String,
    expected_model_id: Option<String>,
) -> Result<(), AppError> {
    state.orchestrator.unload_if_model(resident_id, expected_model_id).await
}

/// One-shot ledger snapshot: GPU totals + everything resident.
#[tauri::command]
pub async fn orchestrator_ledger(
    state: State<'_, AppState>,
) -> Result<LedgerSnapshot, AppError> {
    state.orchestrator.ledger().await
}

/// Subscribe to ledger updates (broadcast on every load/reserve/unload and
/// on worker job completions). Single-subscriber semantics, identical to
/// `subscribe_worker_status`.
#[tauri::command]
pub async fn subscribe_orchestrator_ledger(
    state: State<'_, AppState>,
    channel: Channel<LedgerSnapshot>,
) -> Result<(), AppError> {
    // Snapshot first so the UI renders immediately.
    let snapshot = state.orchestrator.ledger().await?;
    let _ = channel.send(snapshot);

    let mut rx = state.orchestrator.subscribe().await;
    let join_handle = tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(snap) => {
                    if channel.send(snap).is_err() {
                        // The JS Channel was dropped/GC'd — the forwarder is now
                        // dead and the UI will go stale until it re-subscribes.
                        // This is the signature of the "still shows loaded after
                        // unload" bug; the FE now retains channels to prevent it.
                        tracing::warn!(target: "ledger", "ledger forwarder: JS channel closed (send failed) — stream severed until re-subscribe");
                        break;
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    tracing::warn!(target: "ledger", skipped = n, "ledger forwarder: receiver lagged, dropped frames");
                    continue;
                }
                Err(broadcast::error::RecvError::Closed) => {
                    tracing::info!(target: "ledger", "ledger forwarder: broadcast source closed — stopping");
                    break;
                }
            }
        }
    });

    let mut abort_guard = state.orchestrator.subscriber_abort.lock().await;
    if let Some(h) = abort_guard.take() {
        h.abort();
    }
    *abort_guard = Some(join_handle.abort_handle());
    Ok(())
}
