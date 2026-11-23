use serde::Serialize;
use thiserror::Error;

/// Application-level error type.
/// Implements Serialize so it can be returned from Tauri commands as a string.
#[derive(Debug, Error)]
pub enum AppError {
    #[error("Database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("Database migration error: {0}")]
    Migration(#[from] sqlx::migrate::MigrateError),

    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),

    #[error("Keyring error: {0}")]
    Keyring(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// HuggingFace API returned 429 — caller should retry after backoff.
    #[error("HuggingFace rate-limited the request — try again in a moment")]
    RateLimited,

    /// HuggingFace API returned 401 — token missing or invalid.
    #[error("HuggingFace authentication required — please set a valid API token")]
    HfAuthRequired,

    /// HuggingFace API returned 403 — token valid but model is gated and not granted.
    #[error("HuggingFace access denied — you do not have access to this gated model")]
    HfAccessDenied,

    /// Non-2xx response not covered by the specific variants above.
    #[error("HTTP {0} error: {1}")]
    HttpStatus(u16, String),

    /// SHA-256 of the downloaded file did not match the expected digest.
    #[error("Download integrity check failed — SHA-256 mismatch")]
    DownloadIntegrityMismatch,

    /// Download was cancelled by user request.
    #[error("Download cancelled")]
    DownloadCancelled,

    /// Download was paused by user request.
    #[error("Download paused")]
    DownloadPaused,

    /// Download manager channel is unavailable (actor stopped).
    #[error("Download manager unavailable")]
    DownloadManagerUnavailable,

    /// Engine is not installed — call engine_install first.
    #[error("Engine not installed — run engine_install first")]
    EngineNotInstalled,

    /// llama-server process failed to spawn.
    #[error("Engine spawn failed: {0}")]
    EngineSpawnFailed(String),

    /// No safe model-independent pooling override exists for embedding models.
    #[error("Embedding GGUF must declare mean, CLS or last-token pooling metadata; missing, non-pooled or ranking-only pooling is not supported")]
    EmbeddingPoolingUnsupported,

    /// Engine actor is in a state that cannot accept the request yet.
    #[error("Engine not ready")]
    EngineNotReady,

    #[error("{0}")]
    NativeEngineRefused(String),

    #[error("Model load cancelled")]
    EngineLoadCancelled,

    /// Engine did not become healthy within the allotted timeout.
    #[error("Engine load timed out waiting for server health")]
    EngineLoadTimeout,

    /// A step of the ZIP extraction or archive walk failed.
    #[error("Engine extract failed: {0}")]
    EngineExtractFailed(String),

    /// Not enough disk space to install the engine.
    #[error("Insufficient disk space: need {needed_mb} MB, have {available_mb} MB")]
    EngineDiskSpace { needed_mb: u64, available_mb: u64 },

    /// Engine manager actor channel is unavailable.
    #[error("Engine manager unavailable")]
    EngineManagerUnavailable,

    /// NVML / GPU subsystem is not accessible on this machine.
    #[error("NVML unavailable — no NVIDIA driver found")]
    NvmlUnavailable,

    /// The requested model ID was not found in the local catalog.
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// GGUF header parsing failed.
    #[error("GGUF parse failed: {0}")]
    GgufParseFailed(String),

    /// The GGUF declares an architecture without the standard text-LLM
    /// metadata keys (`{arch}.block_count` / `{arch}.embedding_length`) —
    /// e.g. a Qwen3-TTS "talker" or a vision-adapter GGUF. llama-server
    /// would crash on it with an opaque error, so we refuse up front.
    #[error("'{0}' is not a text-generation model — llama.cpp can't load it for chat. Speech/TTS, vision-adapter, and other non-LLM GGUFs aren't loadable here.")]
    UnsupportedModelArchitecture(String),

    /// Transcript file I/O or serialisation error.
    #[error("Transcript IO error: {0}")]
    TranscriptIoError(String),

    /// Conversation ID not found in the database.
    #[error("Conversation not found: {0}")]
    ConversationNotFound(String),

    /// Chat: engine not ready — no model loaded.
    #[error("Chat: engine not ready — load a model first")]
    ChatNotReady,

    /// Transport-level error establishing or maintaining the SSE connection.
    #[error("Chat transport error: {0}")]
    ChatTransport(String),

    /// Error parsing or receiving the SSE stream from llama-server.
    #[error("Chat stream error: {0}")]
    ChatStream(String),

    /// Worker venv is missing or stale — run worker_install first.
    #[error("Python worker not installed — run worker_install first")]
    WorkerNotInstalled,

    /// Worker process failed to spawn.
    #[error("Worker spawn failed: {0}")]
    WorkerSpawnFailed(String),

    /// Worker process is not running (or died) — start it first.
    #[error("Worker not running — start it first")]
    WorkerNotReady,

    /// Worker manager actor channel is unavailable.
    #[error("Worker manager unavailable")]
    WorkerUnavailable,

    /// The worker rejected or failed a request.
    #[error("Worker error: {0}")]
    WorkerRequest(String),

    /// The worker did not answer within the request timeout.
    #[error("Worker request timed out")]
    WorkerRequestTimeout,

    /// A user unload or model swap refused because the resident is busy
    /// with a chat reply or a generation job. Shown to the user as-is.
    #[error("{0}")]
    InUse(String),

    #[error("{0}")]
    Other(String),
}

impl From<keyring::Error> for AppError {
    fn from(e: keyring::Error) -> Self {
        AppError::Keyring(e.to_string())
    }
}

/// Tauri commands must return `Result<T, String>` for the error to be
/// forwarded to the JS side. We implement Serialize so commands can return
/// `Result<T, AppError>` and Tauri serializes the error automatically.
impl Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.to_string())
    }
}
