use super::backend::{Backend, Credentials, Endpoint, Model};
use super::*;
use axum::{body::Body, http::header, response::IntoResponse, routing::any, Json, Router};
use futures::{future::BoxFuture, stream};
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex,
};

const KEY: &str = "test-key-not-a-credential";
const SSE: &str = "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"Read\",\"arguments\":\"{\\\"path\\\":\"}}]}}]}\r\n\r\ndata: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"秘密\\\"}\"}}]}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":12,\"completion_tokens\":7}}\n\ndata: [DONE]\n\n";
struct FakeCredentials(Mutex<Option<String>>);
impl Credentials for FakeCredentials {
    fn get(&self) -> Result<Option<String>, ApiError> {
        Ok(self.0.lock().unwrap().clone())
    }
    fn set(&self, key: &str) -> Result<(), ApiError> {
        *self.0.lock().unwrap() = Some(key.into());
        Ok(())
    }
}
struct FakeBackend {
    port: u16,
    loaded: Arc<Mutex<Option<String>>>,
    loads: Arc<AtomicUsize>,
    engine_work: Arc<tokio::sync::Mutex<()>>,
    unload_started: AtomicUsize,
    unload_finished: AtomicUsize,
    unload_pause: Mutex<Option<(oneshot::Sender<()>, oneshot::Receiver<()>)>>,
    unloads: AtomicUsize,
    saved: Mutex<Option<ServerSettings>>,
    load_gate: Mutex<Option<Arc<tokio::sync::Notify>>>,
    fail_save: std::sync::atomic::AtomicBool,
    native_activity: crate::chat::activity::Activity,
    native_writer: Mutex<Option<crate::chat::activity::Writer>>,
}
impl FakeBackend {
    fn new(port: u16, loaded: bool) -> Arc<Self> {
        let (native_writer, native_activity) = crate::chat::activity::channel();
        Arc::new(Self {
            load_gate: Mutex::new(None),
            fail_save: std::sync::atomic::AtomicBool::new(false),
            native_activity,
            native_writer: Mutex::new(Some(native_writer)),
            port,
            loaded: Arc::new(Mutex::new(loaded.then(|| "id-a".into()))),
            loads: Arc::new(AtomicUsize::new(0)),
            engine_work: Arc::new(tokio::sync::Mutex::new(())),
            unload_started: AtomicUsize::new(0),
            unload_finished: AtomicUsize::new(0),
            unload_pause: Mutex::new(None),
            unloads: AtomicUsize::new(0),
            saved: Mutex::new(None),
        })
    }
}
impl Backend for FakeBackend {
    fn models(&self) -> BoxFuture<'_, Result<Vec<Model>, ApiError>> {
        Box::pin(async {
            Ok(vec![
                Model {
                    id: "id-a".into(),
                    name: "Qwen Coder 7B".into(),
                },
                Model {
                    id: "id-b".into(),
                    name: "Other Model".into(),
                },
            ])
        })
    }
    fn loaded(&self) -> BoxFuture<'_, Result<Option<Endpoint>, ApiError>> {
        Box::pin(async {
            Ok(self
                .loaded
                .lock()
                .unwrap()
                .clone()
                .map(|model_id| Endpoint {
                    model_id,
                    port: self.port,
                }))
        })
    }
    fn load(&self, id: String) -> BoxFuture<'_, Result<Endpoint, ApiError>> {
        Box::pin(async move {
            let gate = self.load_gate.lock().unwrap().clone();
            let work = self.engine_work.clone();
            let loaded = self.loaded.clone();
            let loads = self.loads.clone();
            let port = self.port;
            let (reply, receive) = oneshot::channel();
            // Mimic the real engine actor: its work survives a cancelled API waiter.
            tokio::spawn(async move {
                let _work = work.lock().await;
                loads.fetch_add(1, Ordering::SeqCst);
                if let Some(gate) = gate {
                    gate.notified().await;
                }
                *loaded.lock().unwrap() = Some(id.clone());
                let _ = reply.send(Ok(Endpoint { model_id: id, port }));
            });
            receive
                .await
                .map_err(|_| ApiError::unavailable("fake engine stopped"))?
        })
    }
    fn unload(&self, id: String, idle: Duration) -> BoxFuture<'_, Result<bool, ApiError>> {
        Box::pin(async move {
            self.unload_started.fetch_add(1, Ordering::SeqCst);
            let pause = self.unload_pause.lock().unwrap().take();
            if let Some((chosen, proceed)) = pause {
                let _ = chosen.send(());
                let _ = proceed.await;
            }
            let Some(_reservation) = self.native_activity.idle_reservation(&id, idle) else {
                self.unload_finished.fetch_add(1, Ordering::SeqCst);
                return Ok(false);
            };
            let _work = self.engine_work.lock().await;
            let mut loaded = self.loaded.lock().unwrap();
            if loaded.as_ref() == Some(&id) {
                self.unloads.fetch_add(1, Ordering::SeqCst);
                *loaded = None;
            }
            self.unload_finished.fetch_add(1, Ordering::SeqCst);
            Ok(true)
        })
    }
    fn native_last_used(&self, id: &str) -> Option<Instant> {
        self.native_activity.last_used(id)
    }
    fn save(&self, settings: ServerSettings) -> BoxFuture<'_, Result<(), ApiError>> {
        Box::pin(async move {
            if self.fail_save.load(Ordering::SeqCst) {
                return Err(ApiError::unavailable("fake settings write failure"));
            }
            *self.saved.lock().unwrap() = Some(settings);
            Ok(())
        })
    }
}
struct Rig {
    api: ServerHandle,
    backend: Arc<FakeBackend>,
    url: String,
    upstream: tokio::task::JoinHandle<()>,
    seen: mpsc::UnboundedReceiver<Value>,
}
impl Rig {
    async fn new(loaded: bool, jit: bool, bind: BindAddress) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let (seen_tx, seen) = mpsc::unbounded_channel();
        let app = Router::new().fallback(any(move |headers: axum::http::HeaderMap, Json(body): Json<Value>| {
            let seen_tx = seen_tx.clone();
            async move {
                assert!(!headers.contains_key(header::AUTHORIZATION));
                seen_tx.send(body.clone()).unwrap();
                if body["stream"] == true {
                    let chunks: Vec<_> = SSE.as_bytes().chunks(7).map(|chunk| Ok::<_, std::io::Error>(bytes::Bytes::copy_from_slice(chunk))).collect();
                    ([(header::CONTENT_TYPE, "text/event-stream")], Body::from_stream(stream::iter(chunks))).into_response()
                } else if body.get("input").is_some() {
                    Json(json!({"object":"list", "data":[{"object":"embedding", "index":0, "embedding":[0.1,0.2]}], "usage":{"prompt_tokens":2,"total_tokens":2}})).into_response()
                } else {
                    Json(json!({"id":"fake", "choices":[{"message":{"content":"private completion", "tool_calls":[{"function":{"name":"Read","arguments":"{}"}}]}}], "usage":{"prompt_tokens":12,"completion_tokens":7}})).into_response()
                }
            }
        }));
        let upstream = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let backend = FakeBackend::new(port, loaded);
        let reservation = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = reservation.local_addr().unwrap().port();
        drop(reservation);
        let settings = ServerSettings {
            enabled: true,
            port,
            bind,
            auto_load: jit,
            ..Default::default()
        };
        let api = start(
            backend.clone(),
            Arc::new(FakeCredentials(Mutex::new(Some(KEY.into())))),
            Ok(settings),
        );
        assert!(api.status().await.unwrap().running);
        Self {
            api,
            backend,
            url: format!("http://127.0.0.1:{port}"),
            upstream,
            seen,
        }
    }
    fn client(&self) -> reqwest::Client {
        reqwest::Client::builder()
            .no_proxy()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap()
    }
    async fn stop(self) {
        self.api.set_enabled(false).await.unwrap();
        self.upstream.abort();
    }
}

#[tokio::test]
async fn home_network_auth_required_and_rotation_revokes_old_key() {
    let rig = Rig::new(true, false, BindAddress::HomeNetwork).await;
    for path in ["/health", "/v1/models", "/missing"] {
        let response = rig
            .client()
            .get(format!("{}{path}", rig.url))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 401);
        assert_eq!(
            response.json::<Value>().await.unwrap()["error"]["code"],
            "invalid_api_key"
        );
    }
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .bearer_auth("wrong")
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    assert!(rig.api.key(true).await.unwrap().is_none());
    let key = rig.api.key(false).await.unwrap().unwrap();
    assert_ne!(key, KEY);
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .bearer_auth(key)
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    let status = serde_json::to_string(&rig.api.status().await.unwrap()).unwrap();
    assert!(!status.contains(KEY));
    rig.stop().await;
}

#[tokio::test]
async fn alias_routes_chat_completions_embeddings_and_preserves_tools() {
    let mut rig = Rig::new(true, false, BindAddress::ThisPc).await;
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.aliases.insert("id-a".into(), "my-coder".into());
    rig.api.update(settings).await.unwrap();
    let models: Value = rig
        .client()
        .get(format!("{}/v1/models", rig.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"].as_array().unwrap().len(), 1);
    assert_eq!(models["data"][0]["id"], "my-coder");
    let request = json!({"model":"my-coder", "messages":[{"role":"user","content":"private prompt"}], "tools":[{"type":"function","function":{"name":"Read","parameters":{"type":"object"}}}], "tool_choice":"auto", "functions":[{"name":"legacy"}], "function_call":"auto", "temperature":0.2});
    let response = rig
        .client()
        .post(format!("{}/v1/chat/completions", rig.url))
        .json(&request)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 200);
    assert_eq!(
        response.json::<Value>().await.unwrap()["choices"][0]["message"]["tool_calls"][0]
            ["function"]["name"],
        "Read"
    );
    let sent = rig.seen.recv().await.unwrap();
    assert_eq!(sent["model"], "id-a");
    for key in [
        "messages",
        "tools",
        "tool_choice",
        "functions",
        "function_call",
        "temperature",
    ] {
        assert_eq!(sent[key], request[key]);
    }
    for (path, body) in [
        (
            "/v1/completions",
            json!({"model":"id-a","prompt":"private prompt"}),
        ),
        (
            "/v1/embeddings",
            json!({"model":"my-coder","input":"private embedding"}),
        ),
    ] {
        assert_eq!(
            rig.client()
                .post(format!("{}{path}", rig.url))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
    }
    let logs = rig.api.logs().await.unwrap();
    let text = serde_json::to_string(&logs).unwrap();
    assert!(!text.contains("private"));
    assert!(!text.contains("messages"));
    let log = logs
        .iter()
        .find(|l| l.completion_tokens == Some(7))
        .unwrap();
    assert_eq!(log.prompt_tokens, Some(12));
    assert!(log.tokens_per_second.unwrap() > 0.0);
    rig.stop().await;
}

#[tokio::test]
async fn sse_is_byte_for_byte_including_fragmented_tool_call_deltas() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    let mut events = rig.api.subscribe();
    for path in ["/v1/chat/completions", "/v1/completions"] {
        let response = rig
            .client()
            .post(format!("{}{path}", rig.url))
            .json(&json!({"model":"qwen-coder-7b", "stream":true, "messages":[]}))
            .send()
            .await
            .unwrap();
        assert_eq!(
            response.headers()[header::CONTENT_TYPE],
            "text/event-stream"
        );
        assert_eq!(response.bytes().await.unwrap().as_ref(), SSE.as_bytes());
        let entry = tokio::time::timeout(Duration::from_secs(2), events.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(entry.status, 200);
        assert_eq!(entry.completion_tokens, Some(7));
        assert!(!serde_json::to_string(&entry).unwrap().contains("秘密"));
    }
    rig.stop().await;
}

#[tokio::test]
async fn jit_off_404_then_jit_on_loads_once_and_lists_installed_models() {
    let rig = Rig::new(false, false, BindAddress::ThisPc).await;
    let body = json!({"model":"qwen-coder-7b","prompt":"test"});
    let response = rig
        .client()
        .post(format!("{}/v1/completions", rig.url))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), 404);
    assert_eq!(rig.backend.loads.load(Ordering::SeqCst), 0);
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.auto_load = true;
    rig.api.update(settings).await.unwrap();
    let models: Value = rig
        .client()
        .get(format!("{}/v1/models", rig.url))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(models["data"].as_array().unwrap().len(), 2);
    for _ in 0..2 {
        assert_eq!(
            rig.client()
                .post(format!("{}/v1/completions", rig.url))
                .json(&body)
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
    }
    assert_eq!(rig.backend.loads.load(Ordering::SeqCst), 1);
    assert!(
        rig.backend
            .saved
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .auto_load
    );
    rig.stop().await;
}

#[tokio::test]
async fn port_in_use_keeps_old_listener_and_settings() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    let occupied = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.port = occupied.local_addr().unwrap().port();
    let error = rig.api.update(settings).await.err().unwrap();
    assert!(error.message.contains("already in use"));
    assert!(rig.api.status().await.unwrap().running);
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    rig.stop().await;
}

#[tokio::test(start_paused = true)]
async fn jit_coalesces_past_150_seconds_and_idle_waits_for_startup_and_leases() {
    let rig = Rig::new(false, true, BindAddress::ThisPc).await;
    let gate = Arc::new(tokio::sync::Notify::new());
    *rig.backend.load_gate.lock().unwrap() = Some(gate.clone());
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.idle_minutes = 1;
    rig.api.update(settings).await.unwrap();
    let api = rig.api.clone();
    let first = tokio::spawn(async move { api.resolve("id-a".into()).await });
    eventually("load started", || rig.backend.loads.load(Ordering::SeqCst) == 1).await;
    let api = rig.api.clone();
    let second = tokio::spawn(async move { api.resolve("id-a".into()).await });
    rig.api.status().await.unwrap();
    tokio::time::advance(Duration::from_secs(200)).await;
    rig.api.status().await.unwrap();
    assert!(!first.is_finished() && !second.is_finished());
    assert_eq!(rig.backend.loads.load(Ordering::SeqCst), 1);
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
    gate.notify_one();
    let first = first.await.unwrap().unwrap();
    let second = second.await.unwrap().unwrap();
    tokio::time::advance(Duration::from_secs(61)).await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
    drop((first, second));
    rig.stop().await;
}

#[tokio::test(start_paused = true)]
async fn fake_clock_idle_waits_for_lease_and_never_unloads_a_ui_replacement() {
    let rig = Rig::new(true, true, BindAddress::ThisPc).await;
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.idle_minutes = 1;
    rig.api.update(settings).await.unwrap();
    let lease = rig.api.resolve("id-a".into()).await.unwrap();
    assert_eq!(
        rig.api.resolve("id-b".into()).await.err().unwrap().status,
        StatusCode::CONFLICT
    );
    tokio::time::advance(Duration::from_secs(120)).await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
    drop(lease);
    rig.api.status().await.unwrap();
    tokio::time::advance(Duration::from_secs(61)).await;
    eventually("released API lease permits idle unload", || {
        rig.backend.unload_finished.load(Ordering::SeqCst) == 1
    })
    .await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 1);
    let lease = rig.api.resolve("id-a".into()).await.unwrap();
    drop(lease);
    rig.api.status().await.unwrap();
    *rig.backend.loaded.lock().unwrap() = Some("id-b".into());
    tokio::time::advance(Duration::from_secs(61)).await;
    eventually("conditional idle unload observes UI replacement", || {
        rig.backend.unload_finished.load(Ordering::SeqCst) == 2
    })
    .await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.loaded.lock().unwrap().as_deref(), Some("id-b"));
    rig.stop().await;
}

#[tokio::test]
async fn request_limits_no_cors_and_openai_errors() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    for (path, body, status) in [
        ("/v1/completions", "not-json".into(), 400),
        ("/v1/completions", "{}".into(), 400),
        ("/v1/completions", "x".repeat(8 * 1024 * 1024 + 1), 413),
        ("/unknown", "{}".into(), 404),
    ] {
        let response = rig
            .client()
            .post(format!("{}{path}", rig.url))
            .body(body)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status().as_u16(), status);
        assert!(!response
            .headers()
            .contains_key(header::ACCESS_CONTROL_ALLOW_ORIGIN));
        assert!(response.json::<Value>().await.unwrap()["error"].is_object());
    }
    assert_eq!(
        rig.client()
            .get(format!("{}/v1/completions", rig.url))
            .send()
            .await
            .unwrap()
            .status(),
        405
    );
    assert_eq!(
        rig.client()
            .post(format!("{}/v1/completions", rig.url))
            .header(header::ORIGIN, "https://example.com")
            .body("{}")
            .send()
            .await
            .unwrap()
            .status(),
        403
    );
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .bearer_auth("wrong")
            .send()
            .await
            .unwrap()
            .status(),
        401
    );
    rig.stop().await;
}

#[tokio::test]
async fn log_ring_is_bounded_and_key_never_in_status_or_saved_settings() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    for n in 0..510 {
        rig.api.record(RequestLog {
            time: n.to_string(),
            client_ip: "127.0.0.1".into(),
            model: None,
            prompt_tokens: None,
            completion_tokens: None,
            tokens_per_second: None,
            status: 200,
            duration_ms: 1,
        });
    }
    let logs = rig.api.logs().await.unwrap();
    assert_eq!(logs.len(), 500);
    assert_eq!(logs[0].time, "10");
    assert!(!serde_json::to_string(&rig.api.status().await.unwrap())
        .unwrap()
        .contains(KEY));
    rig.stop().await;
}

// S1: exercise the workflow-derived mode and the exact argument producer used by LlamaServer.
#[test]
fn embedding_workflow_emits_embedding_mode_without_forcing_pooling() {
    use crate::{
        engine::{sidecar::InferenceMode, ModelLoadArgs},
        models::capabilities::Workflow,
    };
    for (workflows, expected) in [
        (vec![Workflow::Embeddings], vec!["--embeddings"]),
        (vec![Workflow::Chat], vec![]),
    ] {
        let args = ModelLoadArgs {
            inference_mode: InferenceMode::from_workflows(&workflows),
            ..Default::default()
        };
        let mut command = tokio::process::Command::new("never-spawn-a-model");
        args.inference_mode.apply(&mut command);
        let actual: Vec<_> = command
            .as_std()
            .get_args()
            .map(|s| s.to_str().unwrap())
            .collect();
        assert_eq!(actual, expected);
    }
}

// S2: a deliberately stalled load must not hold the API control actor.
#[tokio::test]
async fn stalled_jit_coalesces_waiters_but_health_auth_rotation_and_stop_remain_responsive() {
    let rig = Rig::new(false, true, BindAddress::HomeNetwork).await;
    let gate = Arc::new(tokio::sync::Notify::new());
    *rig.backend.load_gate.lock().unwrap() = Some(gate.clone());
    let first = tokio::spawn({
        let api = rig.api.clone();
        async move { api.resolve("id-a".into()).await }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        while rig.backend.loads.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let second = tokio::spawn({
        let api = rig.api.clone();
        async move { api.resolve("qwen-coder-7b".into()).await }
    });
    tokio::time::timeout(Duration::from_secs(1), async {
        assert!(rig.api.status().await.unwrap().running);
        assert_eq!(
            rig.client()
                .get(format!("{}/health", rig.url))
                .bearer_auth(KEY)
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        rig.api.key(true).await.unwrap();
        assert_eq!(
            rig.client()
                .get(format!("{}/health", rig.url))
                .bearer_auth(KEY)
                .send()
                .await
                .unwrap()
                .status(),
            401
        );
        let key = rig.api.key(false).await.unwrap().unwrap();
        assert_eq!(
            rig.client()
                .get(format!("{}/health", rig.url))
                .bearer_auth(key)
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        rig.api.set_enabled(false).await.unwrap();
    })
    .await
    .expect("control plane blocked behind JIT");
    assert!(first.await.unwrap().is_err());
    assert!(second.await.unwrap().is_err());
    assert_eq!(rig.backend.loads.load(Ordering::SeqCst), 1);
    gate.notify_one();
    assert!(!rig.api.status().await.unwrap().running);
    rig.upstream.abort();
}

// S3: drive the same activity producer used around native Chat's stream_turn.
#[tokio::test(start_paused = true)]
async fn fake_clock_idle_preserves_native_chat_and_waits_since_its_last_activity() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.idle_minutes = 1;
    rig.api.update(settings).await.unwrap();
    drop(rig.api.resolve("id-a".into()).await.unwrap());
    rig.api.status().await.unwrap();
    let writer = rig.backend.native_writer.lock().unwrap().take().unwrap();
    let (started, ready) = oneshot::channel();
    let (finish, done) = oneshot::channel::<()>();
    let chat = tokio::spawn(async move {
        writer
            .during("id-a", async {
                started.send(()).unwrap();
                let _ = done.await;
            })
            .await;
    });
    ready.await.unwrap();
    tokio::time::advance(Duration::from_secs(120)).await;
    tokio::task::yield_now().await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
    finish.send(()).unwrap();
    chat.await.unwrap();
    tokio::time::advance(Duration::from_secs(59)).await;
    tokio::task::yield_now().await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
    tokio::time::advance(Duration::from_secs(2)).await;
    eventually("native stream idle timeout expires", || {
        rig.backend.unload_finished.load(Ordering::SeqCst) == 1
    })
    .await;
    rig.api.status().await.unwrap();
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 1);
    rig.stop().await;
}

// S4: both overlapping bind transitions plus rollback after a persisted-settings failure.
#[tokio::test]
async fn running_listener_switches_both_bind_directions_on_same_port_and_rolls_back() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    let mut settings = rig.api.status().await.unwrap().settings;
    let port = settings.port;
    let client = rig.client();
    // Keep a pooled connection open across each rebind, as real clients do.
    for bind in [BindAddress::HomeNetwork, BindAddress::ThisPc] {
        client
            .get(format!("{}/health", rig.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        settings.bind = bind;
        rig.api.update(settings.clone()).await.unwrap();
        let response = client
            .get(format!("{}/health", rig.url))
            .bearer_auth(KEY)
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), 200);
        response.bytes().await.unwrap();
        assert_eq!(rig.api.status().await.unwrap().settings.port, port);
    }
    // Windows permits overlapping specific/wildcard binds; an identical
    // wildcard bind conflicts there. Unix requires a second specific address.
    // In either case the old loopback listener can be restored after failure.
    let blocker_host = if cfg!(windows) {
        "0.0.0.0"
    } else {
        "127.0.0.2"
    };
    let blocker = TcpListener::bind((blocker_host, port)).await.unwrap();
    settings.bind = BindAddress::HomeNetwork;
    assert!(rig.api.update(settings.clone()).await.is_err());
    assert_eq!(
        rig.api.status().await.unwrap().settings.bind,
        BindAddress::ThisPc
    );
    assert_eq!(
        client
            .get(format!("{}/health", rig.url))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    drop(blocker);
    rig.backend.fail_save.store(true, Ordering::SeqCst);
    settings.bind = BindAddress::HomeNetwork;
    assert!(rig.api.update(settings).await.is_err());
    let restored = rig.api.status().await.unwrap();
    assert!(restored.running);
    assert_eq!(restored.settings.bind, BindAddress::ThisPc);
    assert_eq!(
        rig.client()
            .get(format!("{}/health", rig.url))
            .send()
            .await
            .unwrap()
            .status(),
        200
    );
    rig.backend.fail_save.store(false, Ordering::SeqCst);
    rig.stop().await;
}

async fn eventually(label: &str, mut condition: impl FnMut() -> bool) {
    for _ in 0..10_000 {
        if condition() {
            return;
        }
        tokio::task::yield_now().await;
    }
    panic!("background action did not settle: {label}");
}

// N1: parse real GGUF header bytes rather than seed the field the validator reads.
#[test]
fn embedding_pooling_preserves_gguf_metadata_and_rejects_missing_or_nonpooled_metadata() {
    use crate::{engine::sidecar::InferenceMode, models::gguf};
    use std::io::Write;
    fn string(bytes: &mut Vec<u8>, text: &str) {
        bytes.extend_from_slice(&(text.len() as u64).to_le_bytes());
        bytes.extend_from_slice(text.as_bytes());
    }
    for pooling in [None, Some(0u32), Some(1), Some(2), Some(3), Some(4)] {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&(1 + u64::from(pooling.is_some())).to_le_bytes());
        string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8u32.to_le_bytes()); // GGUF string
        string(&mut bytes, "qwen3");
        if let Some(pooling) = pooling {
            string(&mut bytes, "qwen3.pooling_type");
            bytes.extend_from_slice(&4u32.to_le_bytes()); // GGUF uint32
            bytes.extend_from_slice(&pooling.to_le_bytes());
        }
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        let meta = gguf::parse_header(file.path()).unwrap();
        assert_eq!(meta.pooling_type, pooling.map(u64::from));
        let validation = InferenceMode::Embeddings.validate_pooling(Some(&meta));
        assert_eq!(validation.is_ok(), matches!(pooling, Some(1..=3)));
        if let Err(error) = validation {
            assert!(error.to_string().contains("pooling metadata"));
        }
        assert!(InferenceMode::Generation
            .validate_pooling(Some(&meta))
            .is_ok());
    }
    assert!(InferenceMode::Embeddings.validate_pooling(None).is_err());
}

// S3: idle has selected A, but native setup reserves before resolving its endpoint.
#[tokio::test(start_paused = true)]
async fn native_setup_and_idle_unload_share_one_atomic_reservation_in_both_orders() {
    let rig = Rig::new(true, false, BindAddress::ThisPc).await;
    let mut settings = rig.api.status().await.unwrap().settings;
    settings.idle_minutes = 1;
    rig.api.update(settings).await.unwrap();
    drop(rig.api.resolve("id-a".into()).await.unwrap());
    rig.api.status().await.unwrap();
    let (chosen, selected) = oneshot::channel();
    let (proceed, resume) = oneshot::channel();
    *rig.backend.unload_pause.lock().unwrap() = Some((chosen, resume));
    tokio::time::advance(Duration::from_secs(61)).await;
    tokio::time::timeout(Duration::from_secs(1), selected)
        .await
        .unwrap()
        .unwrap();
    let writer = rig.backend.native_writer.lock().unwrap().take().unwrap();
    let native_setup = writer.reserve().await;
    assert_eq!(
        rig.backend.loaded().await.unwrap().unwrap().model_id,
        "id-a"
    );
    proceed.send(()).unwrap();
    eventually("native setup blocks idle", || {
        rig.backend.unload_finished.load(Ordering::SeqCst) >= 1
    })
    .await;
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
    assert_eq!(
        rig.backend.loaded().await.unwrap().unwrap().model_id,
        "id-a"
    );
    // Native setup/stream retains the reservation; no queued idle action can kill it.
    drop(native_setup);
    rig.api.status().await.unwrap();

    // Reverse order: an idle action already queued on a busy engine holds the
    // exclusive side. A new native request must wait before it can read Ready.
    let engine_busy = rig.backend.engine_work.clone().lock_owned().await;
    tokio::time::advance(Duration::from_secs(2)).await;
    eventually("idle holds exclusive reservation", || {
        rig.backend.unload_started.load(Ordering::SeqCst) >= 2
    })
    .await;
    let native_wait = writer.reserve();
    tokio::pin!(native_wait);
    assert!(
        tokio::time::timeout(Duration::from_millis(1), &mut native_wait)
            .await
            .is_err()
    );
    drop(engine_busy);
    let _native_setup = native_wait.await;
    assert!(rig.backend.loaded().await.unwrap().is_none());
    assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 1);
    rig.stop().await;
}

// N2: engine-owned B survives cancellation of API waiters; idle A queues behind
// it. That queue must never run on the API control actor after stop or rebind.
#[tokio::test]
async fn cancelled_jit_with_expired_model_and_queued_idle_keeps_controls_responsive() {
    for rebind in [false, true] {
        tokio::time::pause();
        let rig = Rig::new(true, true, BindAddress::HomeNetwork).await;
        let mut settings = rig.api.status().await.unwrap().settings;
        settings.idle_minutes = 1;
        settings.require_key = true;
        rig.api.update(settings).await.unwrap();
        drop(rig.api.resolve("id-a".into()).await.unwrap());
        rig.api.status().await.unwrap();
        let gate = Arc::new(tokio::sync::Notify::new());
        *rig.backend.load_gate.lock().unwrap() = Some(gate.clone());
        let waiting = tokio::spawn({
            let api = rig.api.clone();
            async move { api.resolve("id-b".into()).await }
        });
        eventually("engine-owned B load begins", || {
            rig.backend.loads.load(Ordering::SeqCst) == 1
        })
        .await;
        tokio::time::advance(Duration::from_secs(61)).await;
        if rebind {
            let mut settings = rig.api.status().await.unwrap().settings;
            settings.bind = BindAddress::ThisPc;
            rig.api.update(settings).await.unwrap();
        } else {
            rig.api.set_enabled(false).await.unwrap();
        }
        assert!(waiting.await.unwrap().is_err());
        // Advance beyond the next interval boundary (including timer-wheel rounding).
        tokio::time::advance(Duration::from_secs(2)).await;
        eventually("idle A queues behind B", || {
            rig.backend.unload_started.load(Ordering::SeqCst) > 0
        })
        .await;
        assert_eq!(rig.backend.unload_finished.load(Ordering::SeqCst), 0);
        tokio::time::resume();
        tokio::time::timeout(Duration::from_secs(1), async {
            if !rebind {
                rig.api.set_enabled(true).await.unwrap();
            }
            assert!(rig.api.status().await.unwrap().running);
            rig.api.key(true).await.unwrap();
            let key = rig.api.key(false).await.unwrap().unwrap();
            assert_eq!(
                rig.client()
                    .get(format!("{}/health", rig.url))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                401
            );
            assert_eq!(
                rig.client()
                    .get(format!("{}/health", rig.url))
                    .bearer_auth(key)
                    .send()
                    .await
                    .unwrap()
                    .status(),
                200
            );
            assert!(rig.api.resolve("id-a".into()).await.is_err());
        })
        .await
        .expect("queued idle unload blocked the control actor");
        gate.notify_one();
        eventually("queued idle A completes after B", || {
            rig.backend.unload_finished.load(Ordering::SeqCst) > 0
        })
        .await;
        assert_eq!(
            rig.backend.loaded().await.unwrap().unwrap().model_id,
            "id-b"
        );
        assert_eq!(rig.backend.unloads.load(Ordering::SeqCst), 0);
        rig.stop().await;
    }
}

#[test]
fn defaults_are_private_and_disabled() {
    let settings = ServerSettings::default();
    assert!(!settings.enabled);
    assert_eq!(settings.port, 1234);
    assert_eq!(settings.bind.ip(), "127.0.0.1");
    assert!(!settings.auth_required());
    assert_eq!(slug("Qwen 2.5 / Coder GGUF"), "qwen-2-5-coder-gguf");
}
