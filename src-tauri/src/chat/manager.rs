use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use serde::Serialize;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc, oneshot, Mutex as TokioMutex};
use tokio_util::sync::CancellationToken;

use crate::chat::persistence as persist;
use crate::chat::sse::{stream_chat, SseChunk};
use crate::chat::transcript::{TokenCounts, TranscriptWriter};
use crate::engine::{EngineManagerHandle, EngineStatus};
use crate::error::AppError;

// ---------------------------------------------------------------------------
// Public event type
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ChatEvent {
    UserMessageStored { conv_id: String, seq: u64 },
    AssistantStart { conv_id: String, seq: u64 },
    Token { conv_id: String, seq: u64, delta: String },
    /// A reasoning-model "thinking" delta (llama-server's
    /// `delta.reasoning_content` side-channel, or an inline `<think>` block
    /// extracted post-stream). The UI accumulates these into a collapsible
    /// Thinking block; thinking never enters the persisted assistant body.
    Thinking { conv_id: String, seq: u64, delta: String },
    AssistantDone {
        conv_id: String,
        seq: u64,
        prompt_tokens: u32,
        completion_tokens: u32,
        finish_reason: String,
    },
    /// Entries with `seq >= from_seq` were removed (delete-from-here or the
    /// truncation step of a regenerate). The UI prunes its local list.
    Truncated { conv_id: String, from_seq: u64 },
    Error { conv_id: String, error: String },
}

// ---------------------------------------------------------------------------
// Actor command enum
// ---------------------------------------------------------------------------

pub enum ChatCmd {
    SendMessage {
        conv_id: String,
        content: String,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    CancelStream {
        conv_id: String,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    /// Delete every entry with `seq >= from_seq` (delete-from-here).
    TruncateFrom {
        conv_id: String,
        from_seq: u64,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    /// Re-generate the assistant turn at `seq`: truncate from there, then
    /// stream a fresh reply to the preceding user message.
    Regenerate {
        conv_id: String,
        seq: u64,
        reply: oneshot::Sender<Result<(), AppError>>,
    },
    Subscribe {
        reply: oneshot::Sender<broadcast::Receiver<ChatEvent>>,
    },
}

// ---------------------------------------------------------------------------
// Handle — cloneable, cheaply shared across Tauri commands
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ChatHandle {
    pub(crate) activity: super::activity::Activity,
    tx: mpsc::Sender<ChatCmd>,
    bcast: broadcast::Sender<ChatEvent>,
    pub subscriber_abort: Arc<TokioMutex<Option<tokio::task::AbortHandle>>>,
}

impl ChatHandle {
    pub async fn send_message(
        &self,
        conv_id: String,
        content: String,
    ) -> Result<(), AppError> {
        let (reply, recv) = oneshot::channel();
        self.tx
            .send(ChatCmd::SendMessage { conv_id, content, reply })
            .await
            .map_err(|_| AppError::Other("chat actor unavailable".into()))?;
        recv.await
            .map_err(|_| AppError::Other("chat actor dropped reply".into()))?
    }

    pub async fn cancel_stream(&self, conv_id: String) -> Result<(), AppError> {
        let (reply, recv) = oneshot::channel();
        self.tx
            .send(ChatCmd::CancelStream { conv_id, reply })
            .await
            .map_err(|_| AppError::Other("chat actor unavailable".into()))?;
        recv.await
            .map_err(|_| AppError::Other("chat actor dropped reply".into()))?
    }

    pub async fn truncate_from(&self, conv_id: String, from_seq: u64) -> Result<(), AppError> {
        let (reply, recv) = oneshot::channel();
        self.tx
            .send(ChatCmd::TruncateFrom { conv_id, from_seq, reply })
            .await
            .map_err(|_| AppError::Other("chat actor unavailable".into()))?;
        recv.await
            .map_err(|_| AppError::Other("chat actor dropped reply".into()))?
    }

    pub async fn regenerate(&self, conv_id: String, seq: u64) -> Result<(), AppError> {
        let (reply, recv) = oneshot::channel();
        self.tx
            .send(ChatCmd::Regenerate { conv_id, seq, reply })
            .await
            .map_err(|_| AppError::Other("chat actor unavailable".into()))?;
        recv.await
            .map_err(|_| AppError::Other("chat actor dropped reply".into()))?
    }

    pub async fn subscribe(&self) -> broadcast::Receiver<ChatEvent> {
        let (reply, recv) = oneshot::channel();
        let _ = self.tx.send(ChatCmd::Subscribe { reply }).await;
        recv.await.unwrap_or_else(|_| self.bcast.subscribe())
    }
}

// ---------------------------------------------------------------------------
// Per-conversation state (private to actor)
// ---------------------------------------------------------------------------

struct ActiveConversation {
    writer: TranscriptWriter,
    current_cancel: Option<CancellationToken>,
}

// ---------------------------------------------------------------------------
// Actor start
// ---------------------------------------------------------------------------

pub fn start(
    pool: SqlitePool,
    http: reqwest::Client,
    engine: EngineManagerHandle,
) -> ChatHandle {
    let (tx, mut rx) = mpsc::channel::<ChatCmd>(32);
    // 256-slot broadcast — tokens are high-volume but small; lagging receivers
    // are simply dropped (broadcast::error::RecvError::Lagged).
    let (bcast_tx, _) = broadcast::channel::<ChatEvent>(256);

    let (activity_writer, activity) = super::activity::channel();
    let handle = ChatHandle {
        activity,
        tx: tx.clone(),
        bcast: bcast_tx.clone(),
        subscriber_abort: Arc::new(TokioMutex::new(None)),
    };

    let bcast_tx_loop = bcast_tx.clone();

    tauri::async_runtime::spawn(async move {
        let mut active: HashMap<String, ActiveConversation> = HashMap::new();

        while let Some(cmd) = rx.recv().await {
            match cmd {
                // -------------------------------------------------------
                ChatCmd::Subscribe { reply } => {
                    let _ = reply.send(bcast_tx_loop.subscribe());
                }

                // -------------------------------------------------------
                ChatCmd::CancelStream { conv_id, reply } => {
                    if let Some(conv) = active.get_mut(&conv_id) {
                        if let Some(tok) = conv.current_cancel.take() {
                            tok.cancel();
                        }
                    }
                    let _ = reply.send(Ok(()));
                }

                // -------------------------------------------------------
                ChatCmd::SendMessage { conv_id, content, reply } => {
                    let _reservation = activity_writer.reserve().await;
                    // 1. Engine must be in Ready state
                    let (model_id, port) = match engine.status().await {
                        EngineStatus::Ready { model_id, port, .. } => (model_id, port),
                        _ => {
                            let err = AppError::ChatNotReady;
                            let _ = bcast_tx_loop.send(ChatEvent::Error {
                                conv_id: conv_id.clone(),
                                error: err.to_string(),
                            });
                            let _ = reply.send(Err(err));
                            continue;
                        }
                    };

                    // 2. Cancel any in-flight stream on this conversation
                    if let Some(conv) = active.get_mut(&conv_id) {
                        if let Some(tok) = conv.current_cancel.take() {
                            tok.cancel();
                        }
                    }

                    // 3. Validate conversation exists in DB and compute writer path
                    let writer_path = match persist::get_conversation(&pool, &conv_id).await {
                        Ok(Some(_)) => {
                            crate::config::transcripts_dir()
                                .join(format!("{}.jsonl", conv_id))
                        }
                        Ok(None) => {
                            let err = AppError::ConversationNotFound(conv_id.clone());
                            let _ = bcast_tx_loop.send(ChatEvent::Error {
                                conv_id: conv_id.clone(),
                                error: err.to_string(),
                            });
                            let _ = reply.send(Err(err));
                            continue;
                        }
                        Err(e) => {
                            let _ = bcast_tx_loop.send(ChatEvent::Error {
                                conv_id: conv_id.clone(),
                                error: e.to_string(),
                            });
                            let _ = reply.send(Err(e));
                            continue;
                        }
                    };

                    // 4. Open writer if not already tracked
                    if !active.contains_key(&conv_id) {
                        match TranscriptWriter::open_or_create(writer_path.clone()).await {
                            Ok(w) => {
                                active.insert(
                                    conv_id.clone(),
                                    ActiveConversation { writer: w, current_cancel: None },
                                );
                            }
                            Err(e) => {
                                let _ = bcast_tx_loop.send(ChatEvent::Error {
                                    conv_id: conv_id.clone(),
                                    error: e.to_string(),
                                });
                                let _ = reply.send(Err(e));
                                continue;
                            }
                        }
                    }

                    // 5. Append user message — borrow conv_entry, then release
                    let user_seq = {
                        let conv_entry = active.get_mut(&conv_id).unwrap();
                        let now = Utc::now().timestamp();
                        match persist::append_message(
                            &pool,
                            &mut conv_entry.writer,
                            &conv_id,
                            "user",
                            &content,
                            now,
                            None,
                            None,
                        )
                        .await
                        {
                            Ok((m, _)) => m.seq,
                            Err(e) => {
                                let _ = bcast_tx_loop.send(ChatEvent::Error {
                                    conv_id: conv_id.clone(),
                                    error: e.to_string(),
                                });
                                let _ = reply.send(Err(e));
                                continue;
                            }
                        }
                    };
                    // conv_entry borrow released here

                    let _ = bcast_tx_loop.send(ChatEvent::UserMessageStored {
                        conv_id: conv_id.clone(),
                        seq: user_seq,
                    });

                    // 6. Fire-and-forget title derivation on first message
                    {
                        let pool_title = pool.clone();
                        let conv_id_t = conv_id.clone();
                        let seed = content.clone();
                        tokio::spawn(async move {
                            let _ =
                                persist::derive_and_set_title(&pool_title, &conv_id_t, &seed)
                                    .await;
                        });
                    }

                    // Reply immediately — events flow via broadcast.
                    let _ = reply.send(Ok(()));

                    // 7. Stream the assistant turn.
                    activity_writer.during(&model_id, stream_turn(
                        &pool,
                        &http,
                        &bcast_tx_loop,
                        &mut active,
                        &conv_id,
                        &model_id,
                        port,
                    ))
                    .await;
                }

                // -------------------------------------------------------
                ChatCmd::TruncateFrom { conv_id, from_seq, reply } => {
                    // Drop the open writer (its cached seq / byte cursor go
                    // stale) and cancel any in-flight stream first.
                    if let Some(mut conv) = active.remove(&conv_id) {
                        if let Some(tok) = conv.current_cancel.take() {
                            tok.cancel();
                        }
                    }
                    match persist::truncate_from(&pool, &conv_id, from_seq).await {
                        Ok(()) => {
                            let _ = bcast_tx_loop.send(ChatEvent::Truncated {
                                conv_id: conv_id.clone(),
                                from_seq,
                            });
                            let _ = reply.send(Ok(()));
                        }
                        Err(e) => {
                            let _ = reply.send(Err(e));
                        }
                    }
                }

                // -------------------------------------------------------
                ChatCmd::Regenerate { conv_id, seq, reply } => {
                    let _reservation = activity_writer.reserve().await;
                    let (model_id, port) = match engine.status().await {
                        EngineStatus::Ready { model_id, port, .. } => (model_id, port),
                        _ => {
                            let _ = reply.send(Err(AppError::ChatNotReady));
                            continue;
                        }
                    };

                    if let Some(mut conv) = active.remove(&conv_id) {
                        if let Some(tok) = conv.current_cancel.take() {
                            tok.cancel();
                        }
                    }
                    if let Err(e) = persist::truncate_from(&pool, &conv_id, seq).await {
                        let _ = reply.send(Err(e));
                        continue;
                    }
                    let _ = bcast_tx_loop.send(ChatEvent::Truncated {
                        conv_id: conv_id.clone(),
                        from_seq: seq,
                    });

                    // Reopen the writer over the truncated file.
                    let writer_path =
                        crate::config::transcripts_dir().join(format!("{conv_id}.jsonl"));
                    match TranscriptWriter::open_or_create(writer_path).await {
                        Ok(w) => {
                            active.insert(
                                conv_id.clone(),
                                ActiveConversation { writer: w, current_cancel: None },
                            );
                        }
                        Err(e) => {
                            let _ = reply.send(Err(e));
                            continue;
                        }
                    }

                    let _ = reply.send(Ok(()));
                    activity_writer.during(&model_id, stream_turn(
                        &pool,
                        &http,
                        &bcast_tx_loop,
                        &mut active,
                        &conv_id,
                        &model_id,
                        port,
                    ))
                    .await;
                }
            }
        }
    });

    handle
}

/// Stream one assistant turn for a conversation whose transcript already ends
/// with a user message — shared by `SendMessage` and `Regenerate`. The caller
/// has already replied Ok to its oneshot; pre-stream failures surface as
/// `ChatEvent::Error` broadcasts, the same path stream errors take. Requires
/// an `active` entry for `conv_id` (SendMessage opens one, Regenerate reopens
/// after truncating).
#[allow(clippy::too_many_arguments)]
async fn stream_turn(
    pool: &SqlitePool,
    http: &reqwest::Client,
    bcast: &broadcast::Sender<ChatEvent>,
    active: &mut HashMap<String, ActiveConversation>,
    conv_id: &str,
    model_id: &str,
    port: u16,
) {
    // Load full conversation history for the request.
    let history = match persist::load_conversation_messages(pool, conv_id).await {
        Ok(h) => h,
        Err(e) => {
            let _ = bcast.send(ChatEvent::Error {
                conv_id: conv_id.to_string(),
                error: e.to_string(),
            });
            return;
        }
    };
    let Some(last) = history.last() else {
        let _ = bcast.send(ChatEvent::Error {
            conv_id: conv_id.to_string(),
            error: "conversation is empty".into(),
        });
        return;
    };
    let assistant_seq = last.seq + 1;

    // Per-conversation settings (falls back to global defaults, then struct
    // defaults).
    let settings = crate::chat::settings::effective(pool, conv_id).await;

    // Prepend the system prompt (if any) as a system message.
    let mut messages_json: Vec<serde_json::Value> = Vec::with_capacity(history.len() + 1);
    if let Some(sp) = settings.system_prompt.as_deref() {
        if !sp.trim().is_empty() {
            messages_json.push(serde_json::json!({ "role": "system", "content": sp }));
        }
    }
    messages_json.extend(
        history
            .iter()
            .map(|m| serde_json::json!({ "role": m.role, "content": m.content })),
    );
    // Build request body with the full sampling parameter set.
    let mut body = serde_json::json!({
        "model": model_id,
        "messages": messages_json,
        "stream": true,
        "stream_options": { "include_usage": true },
    });
    settings.sampling.apply_to_body(&mut body);

    let _ = bcast.send(ChatEvent::AssistantStart {
        conv_id: conv_id.to_string(),
        seq: assistant_seq,
    });

    // Install cancellation token in the active map, then release the borrow.
    let cancel = CancellationToken::new();
    {
        let conv_entry = active.get_mut(conv_id).unwrap();
        conv_entry.current_cancel = Some(cancel.clone());
    }

    // Drain SSE stream — no active borrow held across awaits.
    let mut stream_rx = stream_chat(http.clone(), port, body, cancel.clone());
    let mut accumulator = String::new();
    let mut prompt_tokens: u32 = 0;
    let mut completion_tokens: u32 = 0;
    let mut finish_reason: String = "stop".into();
    let mut transport_error: Option<AppError> = None;

    while let Some(chunk_res) = stream_rx.recv().await {
        match chunk_res {
            Ok(SseChunk::Delta(d)) => {
                accumulator.push_str(&d);
                let _ = bcast.send(ChatEvent::Token {
                    conv_id: conv_id.to_string(),
                    seq: assistant_seq,
                    delta: d,
                });
            }
            Ok(SseChunk::Reasoning(d)) => {
                // Thinking streams separately and never joins the
                // accumulator — the persisted assistant body stays prose.
                let _ = bcast.send(ChatEvent::Thinking {
                    conv_id: conv_id.to_string(),
                    seq: assistant_seq,
                    delta: d,
                });
            }
            Ok(SseChunk::Usage { prompt, completion }) => {
                prompt_tokens = prompt;
                completion_tokens = completion;
            }
            Ok(SseChunk::FinishReason(fr)) => {
                finish_reason = fr;
            }
            Ok(SseChunk::Done) => break,
            Err(e) => {
                transport_error = Some(e);
                break;
            }
        }
    }

    // If the cancel token fired, record it in finish_reason.
    if cancel.is_cancelled() {
        finish_reason = "cancelled".into();
    }

    // Fallback for models/templates whose thinking arrived inline as
    // `<think>…</think>` content (llama-server only splits reasoning for
    // templates it recognizes): route it into the Thinking event stream and
    // keep the persisted assistant body prose-only.
    if accumulator.contains("<think>") {
        let (rest, thinking) = super::reasoning::extract_thinking(&accumulator);
        if let Some(t) = thinking {
            let _ = bcast.send(ChatEvent::Thinking {
                conv_id: conv_id.to_string(),
                seq: assistant_seq,
                delta: t,
            });
            accumulator = rest.trim().to_string();
        }
    }

    // Persist the assistant turn (or partial, on cancel/error with content).
    if transport_error.is_some() && accumulator.is_empty() {
        // True transport error before any tokens — surface, skip persist.
        let e = transport_error.unwrap();
        let _ = bcast.send(ChatEvent::Error {
            conv_id: conv_id.to_string(),
            error: e.to_string(),
        });
    } else {
        let conv_entry = active.get_mut(conv_id).unwrap();
        let tokens = if prompt_tokens > 0 || completion_tokens > 0 {
            Some(TokenCounts { prompt: prompt_tokens, completion: completion_tokens })
        } else {
            None
        };
        let now = Utc::now().timestamp();
        match persist::append_message(
            pool,
            &mut conv_entry.writer,
            conv_id,
            "assistant",
            &accumulator,
            now,
            tokens,
            Some(finish_reason.clone()),
        )
        .await
        {
            Ok(_) => {
                let _ = bcast.send(ChatEvent::AssistantDone {
                    conv_id: conv_id.to_string(),
                    seq: assistant_seq,
                    prompt_tokens,
                    completion_tokens,
                    finish_reason: finish_reason.clone(),
                });
            }
            Err(e) => {
                let _ = bcast.send(ChatEvent::Error {
                    conv_id: conv_id.to_string(),
                    error: e.to_string(),
                });
            }
        }

        // Clear cancel slot.
        conv_entry.current_cancel = None;
    }
}
