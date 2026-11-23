//! Qwen3-TTS engine (qwentts.cpp `tts-server` sidecar).
//!
//! A third engine family next to llama-server and the Python worker: a
//! C++/GGML port of Qwen3-TTS that loads a talker GGUF (architecture
//! `qwen3-tts`) plus a codec GGUF (`qwen3-tts-tokenizer`) GPU-resident and
//! serves synthesis over an OpenAI-compatible HTTP API
//! (`POST /v1/audio/speech`, `GET /v1/voices`).
//!
//! Lifecycle is owned by a single-writer actor ([`TtsHandle`]) mirroring the
//! engine manager: loads are awaited inline (serialized), the orchestrator
//! gates them against the VRAM ledger as the `"tts"` resident. Synthesis
//! itself bypasses the actor — it is a plain HTTP call against the running
//! sidecar, serialized by the server's own mutex.
//!
//! Installation is data-driven pin-and-fetch via the generic `provision`
//! module: the [`manifest::qwentts_manifest`] lists the download URLs +
//! checksums + the [`manifest::REQUIRED_FILES`] that must end up in
//! `engine\qwentts\` for **synthesis** (`tts-server.exe`, the ggml + CUDA 13
//! runtime DLLs, and the bundled codec GGUF). The optional voice-cloning CLI
//! ([`manifest::CLONE_CLI_FILE`], `qwen-tts.exe`) is gated separately by
//! `cli::cli_installed`, so a synthesis-only install is never falsely reported
//! as "not installed". No DB state — `is_installed()` is
//! `provision::is_provisioned(&qwentts_manifest())`. The actor exposes an
//! `Install` command (broadcast `Installing`/`Idle`/`Failed`) mirroring the
//! engine installer, and load/clone trigger a lazy auto-install when the
//! binaries are absent. **Hosting blocker:** upstream publishes no Windows
//! builds, so the manifest URLs are placeholders today and an install attempt
//! degrades to an honest "binaries not hosted yet" error (see
//! `manifest.rs` + CLAUDE.md).

pub mod cli;
pub mod manifest;
pub mod sidecar;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::{broadcast, mpsc, oneshot};

use crate::engine::sidecar::pick_free_port;
use crate::error::AppError;
use sidecar::TtsServer;

/// Catalog architecture string of a loadable talker GGUF.
pub const TALKER_ARCH: &str = "qwen3-tts";

/// Working VRAM beyond the two GGUF files (CUDA context, KV, activations).
pub const QWEN_TTS_OVERHEAD_MB: u64 = 1024;

/// F32 talkers are 7+ GB; cold-disk loads can take minutes.
const LOAD_TIMEOUT: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Install layout
// ---------------------------------------------------------------------------

/// `%LOCALAPPDATA%\OpenHarness\engine\qwentts\`
pub fn install_dir() -> PathBuf {
    crate::config::engine_dir().join("qwentts")
}

pub fn binary_path() -> PathBuf {
    install_dir().join("tts-server.exe")
}

/// The codec/tokenizer GGUF bundled with the engine install. A talker from
/// any repo pairs with it — the codec is shared across all Qwen3-TTS models.
pub fn bundled_codec_path() -> PathBuf {
    install_dir().join("qwen-tokenizer-12hz-Q8_0.gguf")
}

/// Whether the **synthesis** runtime is provisioned — delegates to the generic
/// provisioner over the synthesis `required_files` (tts-server + codec + DLLs).
/// Voice cloning needs one extra binary on top; that capability is reported by
/// `cli::cli_installed`, which checks [`manifest::CLONE_CLI_FILE`] explicitly.
pub fn is_installed() -> bool {
    crate::provision::is_provisioned(&manifest::qwentts_manifest())
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum TtsStatus {
    NotInstalled,
    /// Pin-and-fetch install in flight (mirrors the engine/worker installers).
    Installing {
        stage: String,
        pct: f32,
        bytes: u64,
        total: u64,
    },
    Idle,
    Loading {
        model_id: String,
    },
    Ready {
        model_id: String,
        port: u16,
        /// Speaker names reported by the server (empty for base/cloning
        /// talkers that have no named speakers).
        voices: Vec<String>,
    },
    Failed {
        error: String,
    },
}

// ---------------------------------------------------------------------------
// Actor
// ---------------------------------------------------------------------------

enum TtsCmd {
    Install {
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    Load {
        model_id: String,
        talker: PathBuf,
        codec: PathBuf,
        reply: oneshot::Sender<Result<(u16, Vec<String>), AppError>>,
    },
    Unload {
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    Status {
        reply: oneshot::Sender<TtsStatus>,
    },
    Subscribe {
        reply: oneshot::Sender<broadcast::Receiver<TtsStatus>>,
    },
    Logs {
        reply: oneshot::Sender<Vec<String>>,
    },
}

#[derive(Clone)]
pub struct TtsHandle {
    tx: mpsc::Sender<TtsCmd>,
    /// Status broadcast for the settings card — the install flow streams
    /// `Installing`/`Idle`/`Failed` here (mirrors the worker/engine cards).
    bcast: broadcast::Sender<TtsStatus>,
    /// Abort-handle slot for the single status-channel forwarder, matching
    /// the worker's `subscribe_worker_status` pattern.
    pub subscriber_abort: Arc<tokio::sync::Mutex<Option<tokio::task::AbortHandle>>>,
}

impl TtsHandle {
    async fn send<T>(
        &self,
        make: impl FnOnce(oneshot::Sender<T>) -> TtsCmd,
    ) -> Result<T, AppError> {
        let (tx, rx) = oneshot::channel();
        self.tx
            .send(make(tx))
            .await
            .map_err(|_| AppError::Other("TTS engine unavailable".into()))?;
        rx.await
            .map_err(|_| AppError::Other("TTS engine unavailable".into()))
    }

    /// Provision the qwentts runtime via the generic pin-and-fetch installer
    /// (no-op when already installed). Streams `Installing` on the broadcast
    /// channel, ending `Idle` on success / `Failed` on error. Serialized by
    /// the actor — mirrors `EngineHandle::install`.
    pub async fn install(&self) -> Result<(), AppError> {
        self.send(|reply| TtsCmd::Install { reply }).await?
    }

    /// Fire-and-forget install kicked off from a lazy auto-install trigger
    /// (load/clone hit missing binaries). Never blocks the caller; the
    /// broadcast carries progress to the settings card.
    pub fn install_in_background(&self) {
        let this = self.clone();
        tauri::async_runtime::spawn(async move {
            if let Err(e) = this.install().await {
                tracing::warn!("qwentts auto-install failed: {e}");
            }
        });
    }

    /// Subscribe to status updates (snapshot is sent by the command layer).
    pub async fn subscribe(&self) -> broadcast::Receiver<TtsStatus> {
        self.send(|reply| TtsCmd::Subscribe { reply })
            .await
            .unwrap_or_else(|_| self.bcast.subscribe())
    }

    /// Load a talker + codec pair (same-model reload is a no-op). Returns
    /// the server port and its speaker names. Serialized by the actor.
    pub async fn load(
        &self,
        model_id: String,
        talker: PathBuf,
        codec: PathBuf,
    ) -> Result<(u16, Vec<String>), AppError> {
        self.send(|reply| TtsCmd::Load { model_id, talker, codec, reply }).await?
    }

    pub async fn unload(&self) -> Result<(), AppError> {
        self.send(|reply| TtsCmd::Unload { reply }).await?
    }

    pub async fn status(&self) -> TtsStatus {
        self.send(|reply| TtsCmd::Status { reply })
            .await
            .unwrap_or(TtsStatus::Failed { error: "TTS engine unavailable".into() })
    }

    pub async fn logs(&self) -> Vec<String> {
        self.send(|reply| TtsCmd::Logs { reply }).await.unwrap_or_default()
    }
}

struct Actor {
    http: reqwest::Client,
    server: Option<TtsServer>,
    model_id: Option<String>,
    voices: Vec<String>,
    last_error: Option<String>,
    /// Retained across unloads so `tts_logs` stays useful post-mortem.
    last_logs: Arc<tokio::sync::Mutex<std::collections::VecDeque<String>>>,
    bcast: broadcast::Sender<TtsStatus>,
}

pub fn start(http: reqwest::Client) -> TtsHandle {
    let (tx, mut rx) = mpsc::channel::<TtsCmd>(16);
    let (bcast, _) = broadcast::channel::<TtsStatus>(32);
    let mut actor = Actor {
        http,
        server: None,
        model_id: None,
        voices: Vec::new(),
        last_error: None,
        last_logs: Arc::new(tokio::sync::Mutex::new(std::collections::VecDeque::new())),
        bcast: bcast.clone(),
    };

    tauri::async_runtime::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            match cmd {
                TtsCmd::Install { reply } => {
                    let _ = reply.send(actor.install().await);
                }
                TtsCmd::Load { model_id, talker, codec, reply } => {
                    let result = actor.load(model_id, talker, codec).await;
                    let _ = reply.send(result);
                }
                TtsCmd::Unload { reply } => {
                    let _ = reply.send(actor.unload().await);
                }
                TtsCmd::Status { reply } => {
                    let _ = reply.send(actor.status().await);
                }
                TtsCmd::Subscribe { reply } => {
                    let _ = reply.send(actor.bcast.subscribe());
                }
                TtsCmd::Logs { reply } => {
                    let logs = match &actor.server {
                        Some(s) => s.log_buf.lock().await.iter().cloned().collect(),
                        None => actor.last_logs.lock().await.iter().cloned().collect(),
                    };
                    let _ = reply.send(logs);
                }
            }
        }
    });

    TtsHandle {
        tx,
        bcast,
        subscriber_abort: Arc::new(tokio::sync::Mutex::new(None)),
    }
}

impl Actor {
    /// Provision the qwentts runtime via the generic pin-and-fetch installer,
    /// streaming progress on the status broadcast. Idempotent: a fully
    /// provisioned install returns `Ok(())` immediately. Mirrors
    /// `EngineCmd::Install`.
    async fn install(&mut self) -> Result<(), AppError> {
        let m = manifest::qwentts_manifest();
        if crate::provision::is_provisioned(&m) {
            return Ok(());
        }

        self.last_error = None;
        let _ = self.bcast.send(TtsStatus::Installing {
            stage: "starting".into(),
            pct: 0.0,
            bytes: 0,
            total: 0,
        });

        let bcast = self.bcast.clone();
        let result = crate::provision::install(&self.http, &m, move |p| {
            let pct = if p.bytes_total > 0 {
                (p.bytes_downloaded as f32 / p.bytes_total as f32) * 100.0
            } else {
                0.0
            };
            let _ = bcast.send(TtsStatus::Installing {
                stage: format!("{} ({}/{})", p.stage, p.asset_index, p.asset_count),
                pct,
                bytes: p.bytes_downloaded,
                total: p.bytes_total,
            });
        })
        .await;

        match result {
            Ok(()) => {
                let _ = self.bcast.send(TtsStatus::Idle);
                Ok(())
            }
            Err(e) => {
                self.last_error = Some(e.to_string());
                let _ = self.bcast.send(TtsStatus::Failed { error: e.to_string() });
                Err(e)
            }
        }
    }

    async fn load(
        &mut self,
        model_id: String,
        talker: PathBuf,
        codec: PathBuf,
    ) -> Result<(u16, Vec<String>), AppError> {
        // Same model and still healthy → reuse.
        if self.model_id.as_deref() == Some(model_id.as_str()) {
            if let Some(s) = &self.server {
                if s.health_check(&self.http).await {
                    return Ok((s.port, self.voices.clone()));
                }
            }
        }

        // Lazy auto-install: if the runtime isn't provisioned, fetch it now
        // (same actor, serialized) and proceed once the binaries land. The
        // installer degrades to an honest error when the binaries aren't
        // hosted yet, surfaced verbatim to the caller.
        let bin = binary_path();
        if !bin.exists() {
            self.install().await?;
            if !bin.exists() {
                return Err(AppError::Other(
                    "Qwen3-TTS engine is not installed — tts-server.exe missing from \
                     engine\\qwentts\\ (auto-install did not place it)"
                        .into(),
                ));
            }
        }

        // Slot swap: stop whatever is running first.
        self.stop_server().await;
        self.last_error = None;

        let port = pick_free_port()?;
        let mut server = TtsServer::start(&bin, &talker, &codec, port).await?;
        if let Err(e) = server.wait_ready(&self.http, LOAD_TIMEOUT).await {
            *self.last_logs.lock().await = server.log_buf.lock().await.clone();
            let _ = server.stop().await;
            self.last_error = Some(e.to_string());
            return Err(e);
        }

        let voices = fetch_voices(&self.http, port).await.unwrap_or_else(|e| {
            tracing::warn!("tts voices fetch failed: {e}");
            Vec::new()
        });

        self.server = Some(server);
        self.model_id = Some(model_id);
        self.voices = voices.clone();
        Ok((port, voices))
    }

    async fn unload(&mut self) -> Result<(), AppError> {
        self.stop_server().await;
        Ok(())
    }

    async fn stop_server(&mut self) {
        if let Some(s) = self.server.take() {
            *self.last_logs.lock().await = s.log_buf.lock().await.clone();
            let _ = s.stop().await;
        }
        self.model_id = None;
        self.voices.clear();
    }

    async fn status(&self) -> TtsStatus {
        if let (Some(s), Some(id)) = (&self.server, &self.model_id) {
            return TtsStatus::Ready {
                model_id: id.clone(),
                port: s.port,
                voices: self.voices.clone(),
            };
        }
        if let Some(e) = &self.last_error {
            return TtsStatus::Failed { error: e.clone() };
        }
        if !is_installed() {
            return TtsStatus::NotInstalled;
        }
        TtsStatus::Idle
    }
}

// ---------------------------------------------------------------------------
// HTTP helpers (used outside the actor)
// ---------------------------------------------------------------------------

async fn fetch_voices(http: &reqwest::Client, port: u16) -> Result<Vec<String>, AppError> {
    #[derive(Deserialize)]
    struct Voice {
        name: String,
    }
    #[derive(Deserialize)]
    struct VoicesResponse {
        voices: Vec<Voice>,
    }
    let r: VoicesResponse = http
        .get(format!("http://127.0.0.1:{port}/v1/voices"))
        .timeout(Duration::from_secs(5))
        .send()
        .await?
        .json()
        .await?;
    Ok(r.voices.into_iter().map(|v| v.name).collect())
}

/// One-shot WAV synthesis against a running tts-server. Uses a dedicated
/// long-timeout client — long inputs can take minutes, well past the shared
/// client's 30 s budget. Synthesis is serialized server-side by a mutex.
pub async fn synthesize_wav(
    port: u16,
    text: &str,
    voice: Option<&str>,
    instructions: Option<&str>,
) -> Result<Vec<u8>, AppError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(AppError::Http)?;

    let mut body = serde_json::json!({
        "input": text,
        "response_format": "wav",
    });
    if let Some(v) = voice {
        if !v.is_empty() {
            body["voice"] = serde_json::Value::String(v.into());
        }
    }
    if let Some(i) = instructions {
        if !i.is_empty() {
            body["instructions"] = serde_json::Value::String(i.into());
        }
    }

    let resp = client
        .post(format!("http://127.0.0.1:{port}/v1/audio/speech"))
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let detail = resp.text().await.unwrap_or_default();
        return Err(AppError::Other(format!("tts-server error {status}: {detail}")));
    }
    Ok(resp.bytes().await?.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn install_layout_paths() {
        assert!(install_dir().ends_with("engine/qwentts") || install_dir().ends_with("engine\\qwentts"));
        assert_eq!(binary_path().file_name().unwrap(), "tts-server.exe");
        assert_eq!(
            bundled_codec_path().file_name().unwrap(),
            "qwen-tokenizer-12hz-Q8_0.gguf"
        );
    }

    #[test]
    fn status_serializes_tagged() {
        let s = TtsStatus::Ready {
            model_id: "abc".into(),
            port: 18181,
            voices: vec!["serena".into()],
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["phase"], "ready");
        assert_eq!(v["port"], 18181);
        assert_eq!(v["voices"][0], "serena");
    }
}
