//! Per-conversation chat configuration (M6 §6.4).
//!
//! Three concerns, persisted per conversation as JSON in `conversation_settings`:
//!  - **system prompt** — prepended to the messages array each request.
//!  - **sampling** — applied per-request (no reload needed).
//!  - **load config** — llama-server launch args; changing these requires a reload.
//!
//! Absence of a row means "use global defaults" (the `chat.sampling.*` /
//! `chat.system_prompt` keys in the `settings` table).

use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;

use crate::error::AppError;

/// Per-request sampling parameters forwarded to llama-server's OpenAI endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub min_p: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub max_tokens: Option<u32>,
    pub repeat_penalty: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub seed: Option<i64>,
    #[serde(default)]
    pub stop: Vec<String>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            max_tokens: None,
            repeat_penalty: 1.1,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            seed: None,
            stop: Vec::new(),
        }
    }
}

impl SamplingParams {
    /// Merge these params into a llama-server `/v1/chat/completions` request body.
    pub fn apply_to_body(&self, body: &mut serde_json::Value) {
        body["temperature"] = self.temperature.into();
        body["top_p"] = self.top_p.into();
        body["top_k"] = self.top_k.into();
        body["min_p"] = self.min_p.into();
        body["repeat_penalty"] = self.repeat_penalty.into();
        body["presence_penalty"] = self.presence_penalty.into();
        body["frequency_penalty"] = self.frequency_penalty.into();
        if let Some(mt) = self.max_tokens {
            body["max_tokens"] = mt.into();
        }
        if let Some(seed) = self.seed {
            body["seed"] = seed.into();
        }
        if !self.stop.is_empty() {
            body["stop"] = serde_json::Value::from(self.stop.clone());
        }
    }
}

/// llama-server launch configuration. Changing any field requires a model reload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadConfig {
    /// Context length (`-c`).
    pub ctx: u32,
    /// GPU layers (`-ngl`); -1 = all on GPU, 0 = CPU.
    pub ngl: i32,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub batch_size: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub threads: Option<u32>,
    #[serde(default)]
    pub flash_attn: bool,
    /// KV-cache type, e.g. "f16", "q8_0", "q4_0".
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub kv_cache_type: Option<String>,
    #[serde(default = "default_true")]
    pub mmap: bool,
    #[serde(default)]
    pub mlock: bool,
}

fn default_true() -> bool {
    true
}

/// Effective per-conversation settings handed to the frontend / chat actor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationSettings {
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub sampling: SamplingParams,
    #[serde(default)]
    pub load_config: Option<LoadConfig>,
}

impl Default for ConversationSettings {
    fn default() -> Self {
        Self {
            system_prompt: None,
            sampling: SamplingParams::default(),
            load_config: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------

/// Read the stored per-conversation settings, or `None` if no row exists.
pub async fn get(pool: &SqlitePool, conv_id: &str) -> Result<Option<ConversationSettings>, AppError> {
    let row = sqlx::query!(
        "SELECT system_prompt, sampling_json, load_config_json
         FROM conversation_settings WHERE conversation_id = ?",
        conv_id
    )
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else { return Ok(None) };

    let sampling = row
        .sampling_json
        .as_deref()
        .and_then(|s| serde_json::from_str::<SamplingParams>(s).ok())
        .unwrap_or_default();
    let load_config = row
        .load_config_json
        .as_deref()
        .and_then(|s| serde_json::from_str::<LoadConfig>(s).ok());

    Ok(Some(ConversationSettings {
        system_prompt: row.system_prompt,
        sampling,
        load_config,
    }))
}

/// Resolve the *effective* settings for a conversation: the per-conversation
/// row if present, otherwise the global defaults from the `settings` table.
/// Never fails hard — falls back to struct defaults on any error.
pub async fn effective(pool: &SqlitePool, conv_id: &str) -> ConversationSettings {
    match get(pool, conv_id).await {
        Ok(Some(s)) => s,
        _ => global_defaults(pool).await,
    }
}

/// Build the global default settings from the `settings` k/v table.
/// Missing keys fall back to [`SamplingParams::default`].
pub async fn global_defaults(pool: &SqlitePool) -> ConversationSettings {
    let d = SamplingParams::default();
    let sampling = SamplingParams {
        temperature: get_f32(pool, "chat.sampling.temperature", d.temperature).await,
        top_p: get_f32(pool, "chat.sampling.top_p", d.top_p).await,
        top_k: get_u32(pool, "chat.sampling.top_k", d.top_k).await,
        min_p: get_f32(pool, "chat.sampling.min_p", d.min_p).await,
        max_tokens: get_opt_u32(pool, "chat.sampling.max_tokens").await,
        repeat_penalty: get_f32(pool, "chat.sampling.repeat_penalty", d.repeat_penalty).await,
        presence_penalty: get_f32(pool, "chat.sampling.presence_penalty", d.presence_penalty).await,
        frequency_penalty: get_f32(pool, "chat.sampling.frequency_penalty", d.frequency_penalty)
            .await,
        seed: None,
        stop: Vec::new(),
    };
    let system_prompt = get_text(pool, "chat.system_prompt").await.filter(|s| !s.is_empty());
    ConversationSettings { system_prompt, sampling, load_config: None }
}

async fn get_text(pool: &SqlitePool, k: &str) -> Option<String> {
    sqlx::query!("SELECT value FROM settings WHERE key = ?", k)
        .fetch_optional(pool)
        .await
        .ok()
        .flatten()
        .map(|r| r.value)
}

async fn get_f32(pool: &SqlitePool, k: &str, default: f32) -> f32 {
    get_text(pool, k).await.and_then(|v| v.parse().ok()).unwrap_or(default)
}

async fn get_u32(pool: &SqlitePool, k: &str, default: u32) -> u32 {
    get_text(pool, k).await.and_then(|v| v.parse().ok()).unwrap_or(default)
}

async fn get_opt_u32(pool: &SqlitePool, k: &str) -> Option<u32> {
    get_text(pool, k).await.and_then(|v| v.parse().ok())
}

/// Upsert the per-conversation settings.
pub async fn set(
    pool: &SqlitePool,
    conv_id: &str,
    settings: &ConversationSettings,
) -> Result<(), AppError> {
    let now = chrono::Utc::now().timestamp();
    let sampling_json = serde_json::to_string(&settings.sampling)
        .map_err(|e| AppError::Other(format!("serialize sampling: {e}")))?;
    let load_config_json = match &settings.load_config {
        Some(lc) => Some(
            serde_json::to_string(lc)
                .map_err(|e| AppError::Other(format!("serialize load_config: {e}")))?,
        ),
        None => None,
    };

    sqlx::query!(
        "INSERT INTO conversation_settings
            (conversation_id, system_prompt, sampling_json, load_config_json, updated_at)
         VALUES (?, ?, ?, ?, ?)
         ON CONFLICT(conversation_id) DO UPDATE SET
            system_prompt = excluded.system_prompt,
            sampling_json = excluded.sampling_json,
            load_config_json = excluded.load_config_json,
            updated_at = excluded.updated_at",
        conv_id,
        settings.system_prompt,
        sampling_json,
        load_config_json,
        now
    )
    .execute(pool)
    .await?;

    Ok(())
}
