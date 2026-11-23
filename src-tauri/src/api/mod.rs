//! Local OpenAI-compatible server. One actor owns settings, listener, leases and logs.
mod backend;
mod http;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod native_orchestrator_tests;
#[cfg(test)]
mod native_tests;
mod types;

use axum::http::StatusCode;
pub use backend::{CredentialStore, LocalBackend};
use backend::{Credentials, Endpoint, SharedBackend};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::Arc,
    time::Duration,
};
use subtle::ConstantTimeEq;
use tokio::{
    net::TcpListener,
    sync::{broadcast, mpsc, oneshot},
    time::Instant,
};
use tokio_util::sync::CancellationToken;
pub use types::*;

type Reply<T> = oneshot::Sender<Result<T, ApiError>>;
enum Cmd {
    Status(Reply<ServerStatus>),
    Configure(ServerSettings, Reply<ServerStatus>),
    Enabled(bool, Reply<ServerStatus>),
    Authorize(Option<String>, Reply<()>),
    Models(Reply<Vec<ApiModel>>),
    Resolve(String, Reply<Lease>),
    Loaded(u64, Result<Endpoint, ApiError>),
    Unloaded(u64, String, Result<bool, ApiError>),
    Release(String),
    Record(RequestLog),
    Logs(Reply<Vec<RequestLog>>),
    Key(bool, Reply<Option<String>>),
    Connections(Reply<ConnectionInfo>),
}
#[derive(Clone)]
pub struct ServerHandle {
    tx: mpsc::UnboundedSender<Cmd>,
    events: broadcast::Sender<RequestLog>,
}
impl ServerHandle {
    async fn call<T>(&self, cmd: impl FnOnce(Reply<T>) -> Cmd) -> Result<T, ApiError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(cmd(tx))
            .map_err(|_| ApiError::unavailable("API server actor stopped"))?;
        rx.await
            .map_err(|_| ApiError::unavailable("API server actor stopped"))?
    }
    pub async fn status(&self) -> Result<ServerStatus, ApiError> {
        self.call(Cmd::Status).await
    }
    pub async fn update(&self, settings: ServerSettings) -> Result<ServerStatus, ApiError> {
        self.call(|r| Cmd::Configure(settings, r)).await
    }
    pub async fn set_enabled(&self, enabled: bool) -> Result<ServerStatus, ApiError> {
        self.call(|r| Cmd::Enabled(enabled, r)).await
    }
    pub async fn logs(&self) -> Result<Vec<RequestLog>, ApiError> {
        self.call(Cmd::Logs).await
    }
    pub fn subscribe(&self) -> broadcast::Receiver<RequestLog> {
        self.events.subscribe()
    }
    pub async fn key(&self, regenerate: bool) -> Result<Option<String>, ApiError> {
        self.call(|r| Cmd::Key(regenerate, r)).await
    }
    pub async fn connections(&self) -> Result<ConnectionInfo, ApiError> {
        self.call(Cmd::Connections).await
    }
    async fn authorize(&self, token: Option<String>) -> Result<(), ApiError> {
        self.call(|r| Cmd::Authorize(token, r)).await
    }
    async fn models(&self) -> Result<Vec<ApiModel>, ApiError> {
        self.call(Cmd::Models).await
    }
    async fn resolve(&self, model: String) -> Result<Lease, ApiError> {
        self.call(|r| Cmd::Resolve(model, r)).await
    }
    fn record(&self, entry: RequestLog) {
        let _ = self.tx.send(Cmd::Record(entry));
    }
}

struct Lease {
    endpoint: Endpoint,
    handle: ServerHandle,
}
impl Drop for Lease {
    fn drop(&mut self) {
        let _ = self
            .handle
            .tx
            .send(Cmd::Release(self.endpoint.model_id.clone()));
    }
}
struct Listener {
    cancel: CancellationToken,
    task: tokio::task::JoinHandle<()>,
}
impl Listener {
    async fn stop(mut self) {
        self.cancel.cancel();
        // Let axum close idle keep-alive connections before rebinding this port.
        // Bound the drain: a handler may be queued on this very control actor.
        if tokio::time::timeout(Duration::from_millis(250), &mut self.task)
            .await
            .is_err()
        {
            self.task.abort();
            let _ = (&mut self.task).await;
        }
    }
}
impl Drop for Listener {
    fn drop(&mut self) {
        self.cancel.cancel();
    }
}
struct PendingLoad {
    ticket: u64,
    model_id: String,
    waiters: Vec<Reply<Lease>>,
    task: tokio::task::JoinHandle<()>,
}
struct Actor {
    pending: Option<PendingLoad>,
    idle_unloads: BTreeMap<String, u64>,
    next_ticket: u64,
    backend: SharedBackend,
    credentials: Arc<dyn Credentials>,
    settings: ServerSettings,
    key: Option<String>,
    error: Option<String>,
    listener: Option<Listener>,
    active: BTreeMap<String, usize>,
    touched: BTreeMap<String, Instant>,
    logs: VecDeque<RequestLog>,
    handle: ServerHandle,
}

pub fn start(
    backend: SharedBackend,
    credentials: Arc<dyn Credentials>,
    initial: Result<ServerSettings, ApiError>,
) -> ServerHandle {
    let (tx, mut rx) = mpsc::unbounded_channel();
    let (events, _) = broadcast::channel(512);
    let handle = ServerHandle { tx, events };
    let handle_task = handle.clone();
    let run = async move {
        let (settings, error) = match initial {
            Ok(s) => (s, None),
            Err(e) => (ServerSettings::default(), Some(e.to_string())),
        };
        let mut actor = Actor {
            pending: None,
            idle_unloads: BTreeMap::new(),
            next_ticket: 0,
            backend,
            credentials,
            settings: settings.clone(),
            key: None,
            error,
            listener: None,
            active: BTreeMap::new(),
            touched: BTreeMap::new(),
            logs: VecDeque::new(),
            handle: handle_task,
        };
        if actor.error.is_none() {
            if let Err(e) = actor.configure(settings, false).await {
                actor.error = Some(e.to_string());
            }
        }
        let mut tick = tokio::time::interval(Duration::from_secs(1));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                cmd = rx.recv() => match cmd { Some(cmd) => actor.command(cmd).await, None => break },
                _ = tick.tick() => actor.idle().await,
            }
        }
    };
    #[cfg(test)]
    tokio::spawn(run);
    #[cfg(not(test))]
    tauri::async_runtime::spawn(run);
    handle
}

impl Actor {
    fn ensure_key(&mut self) -> Result<(), ApiError> {
        if self.key.is_none() {
            self.key = self.credentials.get()?;
            if self.key.is_none() {
                self.regenerate_key()?;
            }
        }
        Ok(())
    }
    fn regenerate_key(&mut self) -> Result<(), ApiError> {
        let key = format!(
            "oh-{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        self.credentials.set(&key)?;
        self.key = Some(key);
        Ok(())
    }
    async fn catalog(&self) -> Result<Vec<ApiModel>, ApiError> {
        let mut models = self.backend.models().await?;
        let loaded = self.backend.loaded().await?;
        if let Some(e) = &loaded {
            if !models.iter().any(|m| m.id == e.model_id) {
                models.push(backend::Model {
                    id: e.model_id.clone(),
                    name: e.model_id.clone(),
                });
            }
        }
        models.sort_by(|a, b| a.id.cmp(&b.id));
        let mut aliases = BTreeSet::new();
        let ids: BTreeSet<_> = models.iter().map(|m| m.id.clone()).collect();
        // Reserve explicit aliases first; generated aliases get a stable suffix on collision.
        for alias in self.settings.aliases.values() {
            aliases.insert(alias.clone());
        }
        let mut result = Vec::new();
        for m in models {
            let alias = if let Some(alias) = self.settings.aliases.get(&m.id) {
                alias.clone()
            } else {
                let base = slug(&m.name);
                let mut alias = base.clone();
                let mut n = 2;
                while aliases.contains(&alias) || (ids.contains(&alias) && alias != m.id) {
                    alias = format!("{base}-{n}");
                    n += 1;
                }
                aliases.insert(alias.clone());
                alias
            };
            result.push(ApiModel {
                loaded: loaded.as_ref().is_some_and(|e| e.model_id == m.id),
                id: m.id,
                name: m.name,
                alias,
            });
        }
        Ok(result)
    }
    async fn status(&self) -> Result<ServerStatus, ApiError> {
        Ok(ServerStatus {
            running: self
                .listener
                .as_ref()
                .is_some_and(|l| !l.task.is_finished()),
            settings: self.settings.clone(),
            key_configured: self.key.is_some(),
            error: self.error.clone(),
            models: self.catalog().await?,
        })
    }
    async fn configure(&mut self, settings: ServerSettings, persist: bool) -> Result<(), ApiError> {
        if settings.port == 0 || settings.idle_minutes > 525600 {
            return Err(ApiError::new(
                StatusCode::BAD_REQUEST,
                "Use a port from 1 to 65535 and an idle timeout up to one year",
                "invalid_settings",
            ));
        }
        let models = self.backend.models().await?;
        let mut aliases = BTreeSet::new();
        for (id, alias) in &settings.aliases {
            if alias.len() > 128
                || slug(alias) != *alias
                || !aliases.insert(alias)
                || models.iter().any(|m| m.id == *alias && m.id != *id)
            {
                return Err(ApiError::new(
                    StatusCode::BAD_REQUEST,
                    "Aliases must be unique, lowercase model slugs, not another model's ID",
                    "invalid_alias",
                ));
            }
        }
        // Credential failure must fail closed before a listener is exposed.
        if settings.enabled || self.key.is_none() {
            self.ensure_key()?;
        }
        let rebind = settings.enabled
            && (self.listener.is_none()
                || self.settings.bind != settings.bind
                || self.settings.port != settings.port);
        let overlapping = rebind && self.listener.is_some() && self.settings.port == settings.port;
        if overlapping {
            self.stop_listener().await;
        }
        let prepared = if rebind {
            match TcpListener::bind((settings.bind.ip(), settings.port)).await {
                Ok(listener) => Some(listener),
                Err(error) => {
                    if overlapping {
                        self.restore_listener().await?;
                    }
                    return Err(ApiError::unavailable(
                        if error.kind() == std::io::ErrorKind::AddrInUse {
                            format!(
                                "Port {} is already in use. Choose another port.",
                                settings.port
                            )
                        } else {
                            format!(
                                "Could not bind {}:{}: {error}",
                                settings.bind.ip(),
                                settings.port
                            )
                        },
                    ));
                }
            }
        } else {
            None
        };
        if persist {
            if let Err(error) = self.backend.save(settings.clone()).await {
                drop(prepared);
                if overlapping {
                    self.restore_listener().await?;
                }
                return Err(error);
            }
        }
        if !settings.enabled || rebind {
            self.stop_listener().await;
        }
        self.settings = settings;
        if let Some(listener) = prepared {
            self.serve(listener);
        }
        self.error = None;
        Ok(())
    }
    fn serve(&mut self, listener: TcpListener) {
        let cancel = CancellationToken::new();
        let task = http::serve(listener, self.handle.clone(), cancel.clone());
        self.listener = Some(Listener { cancel, task });
    }
    async fn stop_listener(&mut self) {
        self.cancel_load();
        if let Some(listener) = self.listener.take() {
            listener.stop().await;
        }
    }
    async fn restore_listener(&mut self) -> Result<(), ApiError> {
        match TcpListener::bind((self.settings.bind.ip(), self.settings.port)).await {
            Ok(listener) => {
                self.serve(listener);
                Ok(())
            }
            Err(error) => {
                let error =
                    ApiError::unavailable(format!("Could not restore previous listener: {error}"));
                self.error = Some(error.to_string());
                Err(error)
            }
        }
    }
    fn lease(&mut self, endpoint: Endpoint) -> Lease {
        *self.active.entry(endpoint.model_id.clone()).or_default() += 1;
        self.touched
            .insert(endpoint.model_id.clone(), Instant::now());
        Lease {
            endpoint,
            handle: self.handle.clone(),
        }
    }
    fn cancel_load(&mut self) {
        if let Some(pending) = self.pending.take() {
            // Cancel the API wait, not the engine actor's already-enqueued load.
            // Its gate/cleanup still owns the sidecar; late replies cannot revive API leases.
            pending.task.abort();
            for reply in pending.waiters {
                let _ = reply.send(Err(ApiError::unavailable("API load wait cancelled")));
            }
        }
    }
    async fn route(&self, name: &str) -> Result<(String, Option<Endpoint>), ApiError> {
        if self.listener.is_none() {
            return Err(ApiError::unavailable("Server stopped"));
        }
        let models = self.catalog().await?;
        let model = models
            .iter()
            .find(|m| m.id == name)
            .or_else(|| models.iter().find(|m| m.alias == name))
            .ok_or_else(ApiError::missing)?;
        let id = model.id.clone();
        let loaded = self.backend.loaded().await?.filter(|e| e.model_id == id);
        Ok((id, loaded))
    }
    async fn resolve(&mut self, name: String, reply: Reply<Lease>) {
        let (id, loaded) = match self.route(&name).await {
            Ok(route) => route,
            Err(error) => {
                let _ = reply.send(Err(error));
                return;
            }
        };
        if self.idle_unloads.contains_key(&id) {
            let _ = reply.send(Err(ApiError::unavailable(
                "Model is unloading; retry shortly",
            )));
            return;
        }
        if let Some(pending) = &mut self.pending {
            if pending.model_id == id {
                pending.waiters.push(reply);
            } else {
                let _ = reply.send(Err(ApiError::new(
                    StatusCode::CONFLICT,
                    "A different model is loading",
                    "model_busy",
                )));
            }
            return;
        }
        if let Some(endpoint) = loaded {
            let _ = reply.send(Ok(self.lease(endpoint)));
            return;
        }
        if !self.settings.auto_load {
            let _ = reply.send(Err(ApiError::missing()));
            return;
        }
        if self.active.values().any(|count| *count > 0) {
            let _ = reply.send(Err(ApiError::new(
                StatusCode::CONFLICT,
                "A different model is serving a request. Retry when it finishes.",
                "model_busy",
            )));
            return;
        }
        self.next_ticket += 1;
        let ticket = self.next_ticket;
        let backend = self.backend.clone();
        let tx = self.handle.tx.clone();
        let model_id = id.clone();
        let task = tokio::spawn(async move {
            // The engine owns the startup deadline. Keep this coalesced request
            // pending through planning/probing and native startup, not just 150s.
            let result = backend.load(model_id).await;
            let _ = tx.send(Cmd::Loaded(ticket, result));
        });
        self.pending = Some(PendingLoad {
            ticket,
            model_id: id,
            waiters: vec![reply],
            task,
        });
    }
    async fn idle(&mut self) {
        if let Some(l) = &self.listener {
            if l.task.is_finished() {
                self.listener.take();
                self.error = Some("HTTP listener stopped unexpectedly; restart the server".into());
            }
        }
        if self.settings.idle_minutes == 0 || self.pending.is_some() {
            return;
        }
        let limit = Duration::from_secs(self.settings.idle_minutes as u64 * 60);
        let ids: Vec<_> = self
            .touched
            .iter()
            .filter(|(id, t)| {
                t.elapsed() >= limit
                    && !self.idle_unloads.contains_key(*id)
                    && self.active.get(*id).copied().unwrap_or(0) == 0
                    && self
                        .backend
                        .native_last_used(id)
                        .map_or(true, |used| used.elapsed() >= limit)
            })
            .map(|(id, _)| id.clone())
            .collect();
        for id in ids {
            self.next_ticket += 1;
            let ticket = self.next_ticket;
            self.idle_unloads.insert(id.clone(), ticket);
            let backend = self.backend.clone();
            let tx = self.handle.tx.clone();
            // An engine-owned JIT load can outlive its cancelled API waiters.
            // Never await a queued unload on the control actor. Do not abort this
            // task: its native reservation must outlive the queued engine action.
            tokio::spawn(async move {
                let result = backend.unload(id.clone(), limit).await;
                let _ = tx.send(Cmd::Unloaded(ticket, id, result));
            });
        }
    }
    async fn command(&mut self, cmd: Cmd) {
        match cmd {
            Cmd::Unloaded(ticket, id, result) => {
                if self.idle_unloads.get(&id) == Some(&ticket) {
                    self.idle_unloads.remove(&id);
                    match result {
                        Ok(true) => {
                            self.touched.remove(&id);
                        }
                        Ok(false) => {} // Native setup/activity won the reservation; retry next tick.
                        Err(error) => {
                            self.error = Some(error.to_string());
                            self.touched.insert(id, Instant::now());
                        }
                    }
                }
            }
            Cmd::Status(r) => {
                let _ = r.send(self.status().await);
            }
            Cmd::Configure(s, r) => {
                let result = self.configure(s, true).await;
                let _ = r.send(match result {
                    Ok(()) => self.status().await,
                    Err(e) => Err(e),
                });
            }
            Cmd::Enabled(enabled, r) => {
                let mut s = self.settings.clone();
                s.enabled = enabled;
                let result = self.configure(s, true).await;
                let _ = r.send(match result {
                    Ok(()) => self.status().await,
                    Err(e) => Err(e),
                });
            }
            Cmd::Authorize(token, r) => {
                let valid = match (token, &self.key) {
                    (None, _) => !self.settings.auth_required(),
                    (Some(token), Some(key)) => bool::from(
                        Sha256::digest(token.as_bytes()).ct_eq(&Sha256::digest(key.as_bytes())),
                    ),
                    _ => false,
                };
                let _ = r.send(if valid {
                    Ok(())
                } else {
                    Err(ApiError::new(
                        StatusCode::UNAUTHORIZED,
                        "A valid Bearer API key is required",
                        "invalid_api_key",
                    ))
                });
            }
            Cmd::Models(r) => {
                let result = self.catalog().await.map(|models| {
                    models
                        .into_iter()
                        .filter(|m| m.loaded || self.settings.auto_load)
                        .collect()
                });
                let _ = r.send(result);
            }
            Cmd::Resolve(name, r) => self.resolve(name, r).await,
            Cmd::Loaded(ticket, result) => {
                if self.pending.as_ref().is_some_and(|p| p.ticket == ticket) {
                    let pending = self.pending.take().unwrap();
                    for reply in pending.waiters {
                        if !reply.is_closed() {
                            let result = result.clone().map(|endpoint| self.lease(endpoint));
                            let _ = reply.send(result);
                        }
                    }
                }
            }
            Cmd::Release(id) => {
                if let Some(n) = self.active.get_mut(&id) {
                    *n = n.saturating_sub(1);
                }
                self.touched.insert(id, Instant::now());
            }
            Cmd::Record(log) => {
                if self.logs.len() == 500 {
                    self.logs.pop_front();
                }
                self.logs.push_back(log.clone());
                let _ = self.handle.events.send(log);
            }
            Cmd::Logs(r) => {
                let _ = r.send(Ok(self.logs.iter().cloned().collect()));
            }
            Cmd::Key(regenerate, r) => {
                let result = if regenerate {
                    self.regenerate_key().map(|()| None)
                } else {
                    self.ensure_key().map(|()| self.key.clone())
                };
                let _ = r.send(result);
            }
            Cmd::Connections(r) => {
                let mut lan = BTreeSet::new();
                if self.settings.bind == BindAddress::HomeNetwork {
                    if let Ok(interfaces) = if_addrs::get_if_addrs() {
                        for interface in interfaces {
                            if let std::net::IpAddr::V4(ip) = interface.ip() {
                                if !ip.is_loopback() && !ip.is_unspecified() {
                                    lan.insert(format!("http://{ip}:{}/v1", self.settings.port));
                                }
                            }
                        }
                    }
                }
                let _ = r.send(Ok(ConnectionInfo {
                    this_pc: format!("http://127.0.0.1:{}/v1", self.settings.port),
                    lan: lan.into_iter().collect(),
                    auth_required: self.settings.auth_required(),
                }));
            }
        }
    }
}
