//! Producing path: real registry + production planner/admission/ledger actors,
//! not a Backend that launches the engine manager behind the orchestrator.
use super::*;
use std::sync::Mutex;
struct Keys(Mutex<Option<String>>);
impl backend::Credentials for Keys {
    fn get(&self) -> Result<Option<String>, ApiError> { Ok(self.0.lock().unwrap().clone()) }
    fn set(&self, value: &str) -> Result<(), ApiError> { *self.0.lock().unwrap() = Some(value.into()); Ok(()) }
}
#[tokio::test]
async fn ui_cancel_fails_every_coalesced_jit_waiter_then_real_api_reloads_and_serves() {
    let rig = crate::orchestrator::native_tests::rig().await;
    let backend = Arc::new(backend::LocalBackend { db: rig.db.clone(), engine: rig.engine.clone(), orchestrator: rig.orchestrator.clone(), native_activity: rig.activity.clone() });
    let port = crate::engine::sidecar::pick_free_port().unwrap();
    let api = start(backend, Arc::new(Keys(Mutex::new(None))), Ok(ServerSettings { enabled: true, port, auto_load: true, idle_minutes: 1, ..Default::default() }));
    assert!(api.status().await.unwrap().running);
    let one = api.clone();
    let first = tokio::spawn(async move { one.resolve("slow".into()).await });
    crate::orchestrator::native_tests::loading(&rig.engine).await;
    let two = api.clone();
    let second = tokio::spawn(async move { two.resolve("slow".into()).await });
    api.status().await.unwrap();
    assert!(rig.orchestrator.ledger().await.unwrap().residents.iter().any(|r| r.model_id.as_deref() == Some("slow")));
    rig.orchestrator.unload_if_idle("text".into()).await.unwrap();
    for waiter in [first, second] {
        let error = waiter.await.unwrap().err().expect("cancelled waiter");
        assert_eq!(error.status, axum::http::StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(error.message, "Model load cancelled");
    }
    assert!(matches!(rig.engine.status().await, crate::engine::manager::EngineStatus::Idle));
    assert!(rig.orchestrator.ledger().await.unwrap().residents.is_empty());
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    let url = format!("http://127.0.0.1:{port}");
    let models: serde_json::Value = client.get(format!("{url}/v1/models")).send().await.unwrap().json().await.unwrap();
    let warm = api.models().await.unwrap().into_iter().find(|m| m.id == "warm").unwrap();
    assert!(models["data"].as_array().unwrap().iter().any(|m| m["id"] == warm.alias));
    for stream in [false, true] {
        let response = client.post(format!("{url}/v1/chat/completions")).json(&serde_json::json!({"model": warm.alias, "messages": [{"role": "user", "content": "hello"}], "stream": stream})).send().await.unwrap();
        assert_eq!(response.status(), 200);
        let body = response.text().await.unwrap();
        assert!(body.contains("native stub"));
        if stream { assert!(body.contains("[DONE]")); }
    }
    assert!(rig.orchestrator.current_text_plan().await.unwrap().1.native.is_some());
    let response = client.post(format!("{url}/v1/embeddings")).json(&serde_json::json!({"model": warm.alias, "input":"hello"})).send().await.unwrap();
    assert!(response.status().is_client_error());
    let mut settings = api.status().await.unwrap().settings;
    settings.enabled = false;
    api.update(settings).await.unwrap();
    rig.orchestrator.unload("text".into()).await.unwrap();
}
