use super::{ApiError, ServerSettings};
use crate::{
    engine::{EngineManagerHandle, EngineStatus},
    models::{capabilities::RuntimeKind, registry},
    orchestrator::{LoadOutcome, OrchestratorHandle, TextLoadRequest},
};
use futures::future::BoxFuture;
use std::sync::Arc;

#[derive(Clone)]
pub struct Model {
    pub id: String,
    pub name: String,
}
#[derive(Clone)]
pub struct Endpoint {
    pub model_id: String,
    pub port: u16,
}

// The API actor owns policy; model loads and idle unloads run off-actor. Tests never load a GPU model.
pub trait Backend: Send + Sync {
    fn models(&self) -> BoxFuture<'_, Result<Vec<Model>, ApiError>>;
    fn loaded(&self) -> BoxFuture<'_, Result<Option<Endpoint>, ApiError>>;
    fn load(&self, id: String) -> BoxFuture<'_, Result<Endpoint, ApiError>>;
    fn unload(
        &self,
        id: String,
        idle: std::time::Duration,
    ) -> BoxFuture<'_, Result<bool, ApiError>>;
    fn native_last_used(&self, id: &str) -> Option<tokio::time::Instant>;
    fn save(&self, settings: ServerSettings) -> BoxFuture<'_, Result<(), ApiError>>;
}

pub struct LocalBackend {
    pub native_activity: crate::chat::activity::Activity,
    pub db: sqlx::SqlitePool,
    pub engine: EngineManagerHandle,
    pub orchestrator: OrchestratorHandle,
}
impl Backend for LocalBackend {
    fn models(&self) -> BoxFuture<'_, Result<Vec<Model>, ApiError>> {
        Box::pin(async move {
            let models = registry::list(&self.db)
                .await
                .map_err(|_| ApiError::unavailable("Could not read model catalog"))?;
            Ok(models
                .into_iter()
                .filter(|m| m.caps.runtime == Some(RuntimeKind::LlamaServer))
                .map(|m| Model {
                    id: m.id,
                    name: m.display_name,
                })
                .collect())
        })
    }
    fn loaded(&self) -> BoxFuture<'_, Result<Option<Endpoint>, ApiError>> {
        Box::pin(async move {
            Ok(match self.engine.status_snapshot() {
                EngineStatus::Ready { model_id, port, .. } => Some(Endpoint { model_id, port }),
                _ => None,
            })
        })
    }
    fn load(&self, id: String) -> BoxFuture<'_, Result<Endpoint, ApiError>> {
        Box::pin(load_for_api(&self.orchestrator, id))
    }
    fn unload(
        &self,
        id: String,
        idle: std::time::Duration,
    ) -> BoxFuture<'_, Result<bool, ApiError>> {
        Box::pin(async move {
            let Some(_reservation) = self.native_activity.idle_reservation(&id, idle) else {
                return Ok(false);
            };
            self.orchestrator
                .unload_text_if_model(id)
                .await
                .map_err(|_| ApiError::unavailable("Idle model unload failed"))?;
            Ok(true)
        })
    }
    fn native_last_used(&self, id: &str) -> Option<tokio::time::Instant> {
        self.native_activity.last_used(id)
    }
    fn save(&self, settings: ServerSettings) -> BoxFuture<'_, Result<(), ApiError>> {
        Box::pin(async move {
            let value = serde_json::to_string(&settings)
                .map_err(|_| ApiError::unavailable("Invalid settings"))?;
            crate::db::settings::set_setting(&self.db, "api_server", &value)
                .await
                .map_err(|_| ApiError::unavailable("Could not save server settings"))
        })
    }
}

// No explicit -ngl, so the load planner picks the auto best settings
// (planner::auto_plan) through the same gated load_text path as the UI. It
// never forces an eviction: a plan that fits loads as it is, and
// NeedsEviction only comes back when nothing fits.
async fn load_for_api(orchestrator: &OrchestratorHandle, id: String) -> Result<Endpoint, ApiError> {
    match orchestrator
        .load_text(TextLoadRequest {
            model_id: id,
            decline_eviction_offer: true,
            ..Default::default()
        })
        .await
        .map_err(|error| match error {
            crate::error::AppError::EmbeddingPoolingUnsupported => ApiError::new(
                axum::http::StatusCode::BAD_REQUEST,
                error.to_string(),
                "unsupported_embedding_pooling",
            ),
            crate::error::AppError::NativeEngineRefused(reason) => ApiError::unavailable(reason),
            crate::error::AppError::EngineLoadCancelled => ApiError::unavailable("Model load cancelled"),
            _ => ApiError::unavailable("Model load failed; check Models and engine logs"),
        })? {
        LoadOutcome::Loaded { model_id, port, .. } => Ok(Endpoint { model_id, port }),
        LoadOutcome::NeedsEviction { .. } => Err(ApiError::unavailable(
            "Not enough free VRAM. Unload a model in OpenHarness first.",
        )),
    }
}

pub trait Credentials: Send + Sync {
    fn get(&self) -> Result<Option<String>, ApiError>;
    fn set(&self, key: &str) -> Result<(), ApiError>;
}
pub struct CredentialStore;
impl Credentials for CredentialStore {
    fn get(&self) -> Result<Option<String>, ApiError> {
        let entry = keyring::Entry::new("ai.openharness.app", "api_access_key")
            .map_err(|_| ApiError::unavailable("Credential Manager unavailable"))?;
        match entry.get_password() {
            Ok(key) => Ok(Some(key)),
            Err(keyring::Error::NoEntry) => Ok(None),
            Err(_) => Err(ApiError::unavailable(
                "Could not read API key from Credential Manager",
            )),
        }
    }
    fn set(&self, key: &str) -> Result<(), ApiError> {
        keyring::Entry::new("ai.openharness.app", "api_access_key")
            .and_then(|e| e.set_password(key))
            .map_err(|_| ApiError::unavailable("Could not store API key in Credential Manager"))
    }
}
pub type SharedBackend = Arc<dyn Backend>;
