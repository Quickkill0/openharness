use serde::Serialize;
use tauri::ipc::Channel;
use tauri::State;

use crate::chat::ChatEvent;
use crate::chat::persistence as persist;
use crate::chat::settings::{self as conv_settings, ConversationSettings};
use crate::chat::transcript::MessageRecord;
use crate::error::AppError;
use crate::state::AppState;

/// Wire DTO for a chat message. Flattens `MessageRecord`'s nested `tokens`
/// into `prompt_tokens` / `completion_tokens` so the IPC shape matches the
/// `assistant_done` streaming event (and the frontend `MessageDto`). The
/// canonical JSONL transcript format keeps tokens nested — this is only the
/// boundary representation handed to the UI.
#[derive(Debug, Clone, Serialize)]
pub struct MessageDto {
    pub seq: u64,
    pub role: String,
    pub content: String,
    pub ts: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

impl From<MessageRecord> for MessageDto {
    fn from(m: MessageRecord) -> Self {
        let (prompt_tokens, completion_tokens) = match m.tokens {
            Some(t) => (Some(t.prompt), Some(t.completion)),
            None => (None, None),
        };
        Self {
            seq: m.seq,
            role: m.role,
            content: m.content,
            ts: m.ts,
            prompt_tokens,
            completion_tokens,
            finish_reason: m.finish_reason,
        }
    }
}

/// List all conversations (sidebar metadata).
#[tauri::command]
pub async fn list_conversations(state: State<'_, AppState>) -> Result<Vec<persist::ConversationSummary>, AppError> {
    Ok(persist::list_conversations(&state.db).await?)
}

/// Create a new conversation, optionally bound to a model.
#[tauri::command]
pub async fn create_conversation(
    state: State<'_, AppState>,
    model_id: Option<String>,
) -> Result<persist::ConversationRecord, AppError> {
    Ok(persist::create_conversation(&state.db, model_id.as_deref()).await?)
}

/// Get all messages for a conversation from the transcript file.
#[tauri::command]
pub async fn get_conversation(
    state: State<'_, AppState>,
    id: String,
) -> Result<Vec<MessageDto>, AppError> {
    let msgs = persist::load_conversation_messages(&state.db, &id).await?;
    Ok(msgs.into_iter().map(MessageDto::from).collect())
}

/// Delete a conversation and its transcript file.
#[tauri::command]
pub async fn delete_conversation(
    state: State<'_, AppState>,
    id: String,
) -> Result<(), AppError> {
    Ok(persist::delete_conversation(&state.db, &id).await?)
}

/// Send a message to the active conversation — triggers streaming.
#[tauri::command]
pub async fn send_message(
    conv_id: String,
    content: String,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    Ok(state.chat.send_message(conv_id, content).await?)
}

/// Cancel an in-flight stream for a conversation.
#[tauri::command]
pub async fn cancel_stream(
    conv_id: String,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    Ok(state.chat.cancel_stream(conv_id).await?)
}

/// Delete a message and everything after it from a conversation (transcript +
/// indexes). Cancels any in-flight stream for the conversation first.
#[tauri::command]
pub async fn chat_truncate_from(
    conv_id: String,
    from_seq: u64,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    Ok(state.chat.truncate_from(conv_id, from_seq).await?)
}

/// Regenerate an assistant message: truncate from its seq, then re-stream a
/// fresh completion from the surviving history.
#[tauri::command]
pub async fn chat_regenerate(
    conv_id: String,
    seq: u64,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    Ok(state.chat.regenerate(conv_id, seq).await?)
}

/// Get the effective settings for a conversation (per-conversation overrides
/// if present, else the global defaults).
#[tauri::command]
pub async fn get_conversation_settings(
    state: State<'_, AppState>,
    conv_id: String,
) -> Result<ConversationSettings, AppError> {
    Ok(conv_settings::effective(&state.db, &conv_id).await)
}

/// Persist per-conversation settings (system prompt, sampling, load config).
#[tauri::command]
pub async fn set_conversation_settings(
    state: State<'_, AppState>,
    conv_id: String,
    settings: ConversationSettings,
) -> Result<(), AppError> {
    conv_settings::set(&state.db, &conv_id, &settings).await
}

/// Search messages across all conversations using FTS5.
#[tauri::command]
pub async fn search_messages(
    state: State<'_, AppState>,
    query: String,
    limit: Option<i64>,
) -> Result<Vec<persist::SearchHit>, AppError> {
    Ok(persist::search_messages(&state.db, &query, limit.unwrap_or(20)).await?)
}

/// Subscribe to chat events (token stream, message lifecycle) via a Tauri Channel.
#[tauri::command]
pub async fn subscribe_chat_events(
    channel: Channel<ChatEvent>,
    state: State<'_, AppState>,
) -> Result<(), AppError> {
    let mut rx = state.chat.subscribe().await;

    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(event) => {
                    if channel.send(event).is_err() {
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
