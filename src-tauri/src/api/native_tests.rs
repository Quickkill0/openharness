//! Real native stub process behind the existing API policy/proxy and text manager.
//! No weights, GPU or Windows-specific inference are exercised here.
use super::{backend::{Backend, Credentials, Endpoint, Model}, *};
use futures::future::BoxFuture;
use std::sync::Arc;

struct Key;
impl Credentials for Key {
    fn get(&self) -> Result<Option<String>, ApiError> { Ok(Some("native-test-key".into())) }
    fn set(&self, _: &str) -> Result<(), ApiError> { Ok(()) }
}
struct NativeBackend {
    engine: crate::engine::EngineManagerHandle,
    path: std::path::PathBuf,
    args: crate::engine::ModelLoadArgs,
}
impl Backend for NativeBackend {
    fn models(&self) -> BoxFuture<'_, Result<Vec<Model>, ApiError>> { Box::pin(async { Ok(vec![Model { id: "glm-native".into(), name: "GLM native".into() }]) }) }
    fn loaded(&self) -> BoxFuture<'_, Result<Option<Endpoint>, ApiError>> { Box::pin(async { Ok(match self.engine.status().await { crate::engine::EngineStatus::Ready { model_id, port, .. } => Some(Endpoint { model_id, port }), _ => None }) }) }
    fn load(&self, id: String) -> BoxFuture<'_, Result<Endpoint, ApiError>> { Box::pin(async move {
        let port = self.engine.load_model(id.clone(), self.path.clone(), self.args.clone()).await.map_err(|e| ApiError::unavailable(e.to_string()))?;
        Ok(Endpoint { model_id: id, port })
    }) }
    fn unload(&self, _: String, _: std::time::Duration) -> BoxFuture<'_, Result<bool, ApiError>> { Box::pin(async { self.engine.unload_model().await.map_err(|e| ApiError::unavailable(e.to_string()))?; Ok(true) }) }
    fn native_last_used(&self, _: &str) -> Option<tokio::time::Instant> { None }
    fn save(&self, _: ServerSettings) -> BoxFuture<'_, Result<(), ApiError>> { Box::pin(async { Ok(()) }) }
}

#[tokio::test]
async fn native_api_jit_models_chat_both_modes_and_clean_unsupported_routes() {
    let (_dir, engine, path, args) = crate::engine::native_tests::rig().await;
    let port = crate::engine::sidecar::pick_free_port().unwrap();
    let api = start(Arc::new(NativeBackend { engine: engine.clone(), path, args }), Arc::new(Key), Ok(ServerSettings { enabled: true, auto_load: true, port, ..Default::default() }));
    assert!(api.status().await.unwrap().running);
    let client = reqwest::Client::builder().no_proxy().timeout(std::time::Duration::from_secs(10)).build().unwrap();
    let url = format!("http://127.0.0.1:{port}");
    let models: serde_json::Value = client.get(format!("{url}/v1/models")).bearer_auth("native-test-key").send().await.unwrap().json().await.unwrap();
    assert_eq!(models["data"][0]["id"], "glm-native");
    for stream in [false, true] {
        let response = client.post(format!("{url}/v1/chat/completions")).bearer_auth("native-test-key").json(&serde_json::json!({"model":"glm-native", "messages":[{"role":"user","content":"hello"}], "stream":stream})).send().await.unwrap();
        assert!(response.status().is_success());
        let body = response.text().await.unwrap();
        assert!(body.contains("native stub reply"));
        assert_eq!(body.contains("[DONE]"), stream);
    }
    assert!(matches!(engine.status().await, crate::engine::EngineStatus::Ready { .. }));
    let embeddings = client.post(format!("{url}/v1/embeddings")).bearer_auth("native-test-key").json(&serde_json::json!({"model":"glm-native", "input":"hello"})).send().await.unwrap();
    assert_eq!(embeddings.status().as_u16(), 400);
    assert!(embeddings.json::<serde_json::Value>().await.unwrap()["error"].is_object());
    let other = client.post(format!("{url}/v1/images/generations")).bearer_auth("native-test-key").body("{}").send().await.unwrap();
    assert_eq!(other.status().as_u16(), 404);
    api.set_enabled(false).await.unwrap();
    engine.unload_model().await.unwrap();
}
