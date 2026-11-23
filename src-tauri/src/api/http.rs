use super::{ApiError, Lease, RequestLog, ServerHandle};
use axum::{
    body::{to_bytes, Body},
    extract::{ConnectInfo, State},
    http::{header, Method, Request, StatusCode},
    response::{IntoResponse, Response},
    routing::any,
    Json, Router,
};
use bytes::Bytes;
use futures::{stream, StreamExt};
use serde_json::{json, Value};
use std::{net::SocketAddr, sync::Arc, time::Duration};
use tokio::{net::TcpListener, sync::Semaphore, time::Instant};
use tokio_util::sync::CancellationToken;

const BODY_LIMIT: usize = 8 * 1024 * 1024;
const RESPONSE_LIMIT: usize = 32 * 1024 * 1024;
#[derive(Clone)]
struct HttpState {
    handle: ServerHandle,
    cancel: CancellationToken,
    slots: Arc<Semaphore>,
    client: reqwest::Client,
}

pub(super) fn serve(
    listener: TcpListener,
    handle: ServerHandle,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    let state = HttpState {
        handle,
        cancel: cancel.clone(),
        slots: Arc::new(Semaphore::new(64)),
        // Do not forward caller Authorization, allow redirects, or inherit proxy env vars.
        client: reqwest::Client::builder()
            .no_proxy()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(5))
            .build()
            .expect("local HTTP client"),
    };
    let app = Router::new().fallback(any(request)).with_state(state);
    tokio::spawn(async move {
        let _ = axum::serve(
            listener,
            app.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .with_graceful_shutdown(cancel.cancelled_owned())
        .await;
    })
}

struct Record {
    log: RequestLog,
    started: Instant,
    handle: ServerHandle,
    lease: Option<Lease>,
    _slot: Option<tokio::sync::OwnedSemaphorePermit>,
}
impl Record {
    fn new(handle: ServerHandle, ip: String) -> Self {
        Self {
            log: RequestLog {
                time: chrono::Utc::now().to_rfc3339(),
                client_ip: ip,
                model: None,
                prompt_tokens: None,
                completion_tokens: None,
                tokens_per_second: None,
                status: 499,
                duration_ms: 0,
            },
            started: Instant::now(),
            handle,
            lease: None,
            _slot: None,
        }
    }
    fn usage(&mut self, value: &Value) {
        if let Some(usage) = value.get("usage") {
            self.log.prompt_tokens = usage
                .get("prompt_tokens")
                .and_then(Value::as_u64)
                .or(self.log.prompt_tokens);
            self.log.completion_tokens = usage
                .get("completion_tokens")
                .and_then(Value::as_u64)
                .or(self.log.completion_tokens);
        }
    }
}
impl Drop for Record {
    fn drop(&mut self) {
        let elapsed = self.started.elapsed();
        self.log.duration_ms = elapsed.as_millis().min(u64::MAX as u128) as u64;
        self.log.tokens_per_second = self
            .log
            .completion_tokens
            .filter(|_| elapsed.as_secs_f64() > 0.0)
            .map(|n| n as f64 / elapsed.as_secs_f64());
        self.handle.record(self.log.clone());
    }
}

async fn request(
    State(state): State<HttpState>,
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    req: Request<Body>,
) -> Response {
    let mut record = Record::new(state.handle.clone(), peer.ip().to_string());
    record._slot = match state.slots.clone().try_acquire_owned() {
        Ok(slot) => Some(slot),
        Err(_) => {
            record.log.status = 429;
            return ApiError::new(
                StatusCode::TOO_MANY_REQUESTS,
                "Too many concurrent requests",
                "rate_limit_exceeded",
            )
            .into_response();
        }
    };
    match handle_request(state, req, record).await {
        Ok(response) => response,
        Err((error, mut record)) => {
            record.log.status = error.status.as_u16();
            error.into_response()
        }
    }
}

type Failure = (ApiError, Record);
async fn handle_request(
    state: HttpState,
    req: Request<Body>,
    mut record: Record,
) -> Result<Response, Failure> {
    let token = req.headers().get(header::AUTHORIZATION).map(|value| {
        value
            .to_str()
            .ok()
            .and_then(|v| v.strip_prefix("Bearer "))
            .filter(|v| v.len() <= 4096)
            .unwrap_or("")
            .to_owned()
    });
    if let Err(e) = state.handle.authorize(token).await {
        return Err((e, record));
    }
    if state.cancel.is_cancelled() {
        return Err((ApiError::unavailable("Server stopped"), record));
    }
    // No browser integration/CORS. Also reject simple cross-origin POSTs, which
    // otherwise bypass preflight and could trigger a local JIT load without a key.
    if req.headers().contains_key(header::ORIGIN) {
        return Err((
            ApiError::new(
                StatusCode::FORBIDDEN,
                "Browser-origin requests are not supported",
                "origin_not_allowed",
            ),
            record,
        ));
    }
    let path = req.uri().path().to_owned();
    let method = req.method().clone();
    if path == "/health" || path == "/v1/models" {
        if method != Method::GET {
            return Err((
                ApiError::new(
                    StatusCode::METHOD_NOT_ALLOWED,
                    "Use GET for this endpoint",
                    "method_not_allowed",
                ),
                record,
            ));
        }
        let response = if path == "/health" {
            Json(json!({"status": "ok"})).into_response()
        } else {
            match state.handle.models().await {
                Ok(models) => Json(json!({"object": "list", "data": models.into_iter().map(|m| json!({"id": m.alias, "object": "model", "created": 0, "owned_by": "openharness"})).collect::<Vec<_>>()})).into_response(),
                Err(e) => return Err((e, record)),
            }
        };
        record.log.status = 200;
        return Ok(response);
    }
    if !matches!(
        path.as_str(),
        "/v1/chat/completions" | "/v1/completions" | "/v1/embeddings"
    ) {
        return Err((
            ApiError::new(StatusCode::NOT_FOUND, "Unknown API endpoint", "not_found"),
            record,
        ));
    }
    if method != Method::POST {
        return Err((
            ApiError::new(
                StatusCode::METHOD_NOT_ALLOWED,
                "Use POST for this endpoint",
                "method_not_allowed",
            ),
            record,
        ));
    }
    let body = tokio::select! {
        _ = state.cancel.cancelled() => return Err((ApiError::unavailable("Server stopped"), record)),
        result = tokio::time::timeout(Duration::from_secs(30), to_bytes(req.into_body(), BODY_LIMIT)) => match result {
            Ok(Ok(body)) => body,
            Ok(Err(_)) => return Err((ApiError::new(StatusCode::PAYLOAD_TOO_LARGE, "Request body exceeds 8 MiB or is incomplete", "request_too_large"), record)),
            Err(_) => return Err((ApiError::new(StatusCode::REQUEST_TIMEOUT, "Request body timed out", "request_timeout"), record)),
        },
    };
    let mut value: Value = match serde_json::from_slice(&body) {
        Ok(Value::Object(map)) => Value::Object(map),
        _ => {
            return Err((
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "Expected a JSON object",
                    "invalid_json",
                ),
                record,
            ))
        }
    };
    let model = match value
        .get("model")
        .and_then(Value::as_str)
        .filter(|m| !m.is_empty() && m.len() <= 256)
    {
        Some(model) => model.to_owned(),
        None => {
            return Err((
                ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "A model ID or API alias is required",
                    "invalid_model",
                ),
                record,
            ))
        }
    };
    let lease = match state.handle.resolve(model).await {
        Ok(lease) => lease,
        Err(e) => return Err((e, record)),
    };
    // Logs use the resolved catalog ID, never untrusted request strings or message contents.
    record.log.model = Some(lease.endpoint.model_id.clone());
    value["model"] = Value::String(lease.endpoint.model_id.clone());
    let url = format!("http://127.0.0.1:{}{path}", lease.endpoint.port);
    record.lease = Some(lease);
    let response = tokio::select! {
        _ = state.cancel.cancelled() => return Err((ApiError::unavailable("Server stopped"), record)),
        result = tokio::time::timeout(Duration::from_secs(600), state.client.post(url).json(&value).send()) => match result {
            Ok(Ok(response)) => response,
            _ => return Err((ApiError::new(StatusCode::BAD_GATEWAY, "Model server did not respond", "upstream_error"), record)),
        },
    };
    let status = response.status();
    if !status.is_success() {
        // Never echo upstream messages: they may contain prompt fragments or local paths.
        return Err((
            ApiError::new(
                status,
                format!("Model server returned HTTP {}", status.as_u16()),
                "upstream_error",
            ),
            record,
        ));
    }
    let sse = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("text/event-stream"));
    if sse {
        let upstream = response.bytes_stream().boxed();
        let output = stream::unfold(
            (
                upstream,
                Some(record),
                UsageScanner::default(),
                state.cancel,
            ),
            move |(mut upstream, mut record, mut scanner, cancel)| async move {
                let next = tokio::select! { _ = cancel.cancelled() => Ok(None), next = tokio::time::timeout(Duration::from_secs(600), upstream.next()) => next };
                match next {
                    Ok(Some(Ok(bytes))) => {
                        if let Some(r) = &mut record {
                            scanner.feed(&bytes, r);
                        }
                        Some((
                            Ok::<Bytes, std::io::Error>(bytes),
                            (upstream, record, scanner, cancel),
                        ))
                    }
                    Ok(Some(Err(_))) | Err(_) => {
                        if let Some(r) = &mut record {
                            r.log.status = 502;
                        }
                        record.take();
                        let error = ApiError::new(
                            StatusCode::BAD_GATEWAY,
                            "Model stream interrupted",
                            "upstream_error",
                        );
                        let done = CancellationToken::new();
                        done.cancel();
                        Some((
                            Ok(Bytes::from(format!("data: {}\n\n", error.value()))),
                            (upstream, record, scanner, done),
                        ))
                    }
                    Ok(None) => {
                        if let Some(r) = &mut record {
                            if !cancel.is_cancelled() {
                                r.log.status = status.as_u16();
                            }
                        }
                        None
                    }
                }
            },
        );
        Ok((
            status,
            [
                (header::CONTENT_TYPE, "text/event-stream"),
                (header::CACHE_CONTROL, "no-cache"),
            ],
            Body::from_stream(output),
        )
            .into_response())
    } else {
        let mut stream = response.bytes_stream();
        let mut bytes = Vec::new();
        loop {
            let next = tokio::select! {
                _ = state.cancel.cancelled() => return Err((ApiError::unavailable("Server stopped"), record)),
                result = tokio::time::timeout(Duration::from_secs(600), stream.next()) => match result {
                    Ok(next) => next,
                    Err(_) => return Err((ApiError::new(StatusCode::GATEWAY_TIMEOUT, "Model response timed out", "upstream_timeout"), record)),
                },
            };
            match next {
                Some(Ok(chunk)) if bytes.len() + chunk.len() <= RESPONSE_LIMIT => {
                    bytes.extend_from_slice(&chunk)
                }
                Some(_) => {
                    return Err((
                        ApiError::new(
                            StatusCode::BAD_GATEWAY,
                            "Model response failed or exceeds 32 MiB",
                            "upstream_error",
                        ),
                        record,
                    ))
                }
                None => break,
            }
        }
        if let Ok(value) = serde_json::from_slice::<Value>(&bytes) {
            record.usage(&value);
        } else {
            return Err((
                ApiError::new(
                    StatusCode::BAD_GATEWAY,
                    "Model server returned invalid JSON",
                    "upstream_error",
                ),
                record,
            ));
        }
        record.log.status = status.as_u16();
        Ok((status, [(header::CONTENT_TYPE, "application/json")], bytes).into_response())
    }
}

// Only a bounded SSE line is inspected for usage. The original chunks are never rewritten.
#[derive(Default)]
struct UsageScanner {
    line: Vec<u8>,
    overflow: bool,
}
impl UsageScanner {
    fn feed(&mut self, bytes: &[u8], record: &mut Record) {
        for &byte in bytes {
            if byte == b'\n' {
                if !self.overflow {
                    if let Some(data) = self.line.strip_prefix(b"data:") {
                        if let Ok(value) = serde_json::from_slice::<Value>(data) {
                            record.usage(&value);
                        }
                    }
                }
                self.line.clear();
                self.overflow = false;
            } else if self.line.len() < 65536 && !self.overflow {
                self.line.push(byte);
            } else {
                self.line.clear();
                self.overflow = true;
            }
        }
    }
}
