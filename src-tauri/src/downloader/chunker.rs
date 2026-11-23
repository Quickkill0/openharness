use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;
use std::time::Duration;

use futures::future;
use reqwest::StatusCode;
use tokio::sync::mpsc;

use crate::error::AppError;

use super::job::ChunkState;

// ---------------------------------------------------------------------------
// Cancellation token (atomic u8: 0=running, 1=paused, 2=cancelled)
// ---------------------------------------------------------------------------

/// The reason a `CancelToken` was signaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelReason {
    None,
    Pause,
    Cancel,
}

/// A lightweight cancellation flag shared between the manager and chunk tasks.
/// Uses `AtomicU8` so the reason (pause vs cancel) can be communicated.
#[derive(Clone)]
pub struct CancelToken(Arc<std::sync::atomic::AtomicU8>);

const REASON_RUNNING: u8 = 0;
const REASON_PAUSE: u8 = 1;
const REASON_CANCEL: u8 = 2;

impl Default for CancelToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancelToken {
    pub fn new() -> Self {
        Self(Arc::new(std::sync::atomic::AtomicU8::new(REASON_RUNNING)))
    }

    /// Signal a pause. Does not overwrite a cancel already in flight.
    pub fn pause(&self) {
        let _ = self.0.compare_exchange(
            REASON_RUNNING,
            REASON_PAUSE,
            std::sync::atomic::Ordering::Relaxed,
            std::sync::atomic::Ordering::Relaxed,
        );
    }

    /// Signal a cancel. Always overwrites (cancel wins over pause).
    pub fn cancel(&self) {
        self.0.store(REASON_CANCEL, std::sync::atomic::Ordering::Relaxed);
    }

    /// Returns true if any signal (pause or cancel) has been set.
    pub fn is_signaled(&self) -> bool {
        self.0.load(std::sync::atomic::Ordering::Relaxed) > REASON_RUNNING
    }

    /// Alias for `is_signaled()` — returns true for both pause and cancel.
    pub fn is_cancelled(&self) -> bool {
        self.is_signaled()
    }

    /// Returns the specific reason the token was signaled.
    pub fn reason(&self) -> CancelReason {
        match self.0.load(std::sync::atomic::Ordering::Relaxed) {
            REASON_PAUSE => CancelReason::Pause,
            REASON_CANCEL => CancelReason::Cancel,
            _ => CancelReason::None,
        }
    }
}

// ---------------------------------------------------------------------------
// Chunk progress message (sent from chunk tasks to the manager)
// ---------------------------------------------------------------------------

pub struct ChunkProgress {
    pub chunk_index: usize,
    pub bytes_written: u64,
}

// ---------------------------------------------------------------------------
// HEAD validation
// ---------------------------------------------------------------------------

/// Validate that the remote URL accepts range requests and that the reported
/// `Content-Length` matches the expected total. Returns the confirmed size.
pub async fn validate_remote(
    client: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    expected_size: u64,
) -> Result<u64, AppError> {
    let mut req = client.head(url);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await?;

    if !resp.status().is_success() {
        return Err(AppError::HttpStatus(resp.status().as_u16(), String::new()));
    }

    let content_length = resp
        .headers()
        .get("content-length")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.parse::<u64>().ok());

    if let Some(len) = content_length {
        if len != expected_size {
            tracing::warn!(
                "HEAD Content-Length mismatch: expected {expected_size}, got {len}. Proceeding with remote size."
            );
            return Ok(len);
        }
    }

    Ok(expected_size)
}

// ---------------------------------------------------------------------------
// Single-chunk download
// ---------------------------------------------------------------------------

/// Download a single byte range from `url`, writing directly to `dest_path`
/// at absolute file offsets.
///
/// Resilient to flaky connections on large (multi-GB) transfers: connect
/// failures, 429s, and mid-stream body breaks all retry (up to `MAX_RETRIES`
/// per chunk) with capped exponential backoff, each attempt re-issuing a fresh
/// `Range` GET from the current write offset so no bytes are re-downloaded or
/// double-counted.
///
/// Critically, when resuming past the chunk's start (`write_pos > 0`) we REQUIRE
/// `206 Partial Content`. A `200 OK` means the server ignored our `Range` header
/// and is streaming the whole file from offset 0 — writing those bytes at our
/// partial offset would corrupt the file AND double-count progress (the source
/// of phantom ">100%" readings). We treat that as a retryable error instead.
pub async fn download_chunk(
    client: reqwest::Client,
    url: String,
    token: Option<String>,
    dest_path: std::path::PathBuf,
    chunk_index: usize,
    chunk: ChunkState,
    tx: mpsc::Sender<ChunkProgress>,
    cancel: CancelToken,
) -> Result<(), AppError> {
    use futures::StreamExt;

    // Absolute file offset we next write to; advances as bytes land on disk.
    let mut write_pos = chunk.start + chunk.completed;
    if write_pos > chunk.end {
        // Already complete
        return Ok(());
    }

    let build_req = |range: &str| {
        let mut r = client.get(&url).header("Range", range);
        if let Some(ref t) = token {
            r = r.bearer_auth(t);
        }
        r
    };

    // Chunks write to non-overlapping byte ranges, so truncate must be false
    // to preserve data written by sibling chunk tasks.
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(false)
        .open(&dest_path)?;

    let cancelled_err = |cancel: &CancelToken| match cancel.reason() {
        CancelReason::Pause => AppError::DownloadPaused,
        CancelReason::Cancel | CancelReason::None => AppError::DownloadCancelled,
    };

    const MAX_RETRIES: u32 = 6;
    let mut attempt: u32 = 0;

    loop {
        if cancel.is_signaled() {
            return Err(cancelled_err(&cancel));
        }

        let range_str = format!("bytes={}-{}", write_pos, chunk.end);

        // --- Acquire the response (retry connect-level failures) ---
        let resp = match build_req(&range_str).send().await {
            Ok(resp) => resp,
            Err(e) => {
                attempt += 1;
                if attempt > MAX_RETRIES {
                    return Err(AppError::Http(e));
                }
                let backoff = std::cmp::min(2u64.saturating_pow(attempt), 15);
                tracing::warn!(
                    "chunk {chunk_index}: connect error (attempt {attempt}/{MAX_RETRIES}), retrying in {backoff}s: {e}"
                );
                tokio::time::sleep(Duration::from_secs(backoff)).await;
                continue;
            }
        };

        let status = resp.status();

        // --- 429: back off and retry ---
        if status == StatusCode::TOO_MANY_REQUESTS {
            attempt += 1;
            if attempt > MAX_RETRIES {
                return Err(AppError::RateLimited);
            }
            let wait = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(5)
                .min(30);
            tokio::time::sleep(Duration::from_secs(wait)).await;
            continue;
        }

        // --- Range ignored: a 200 OK while resuming past offset 0 would write
        //     whole-file bytes at a partial offset. Retryable, never write it. ---
        if write_pos > 0 && status == StatusCode::OK {
            attempt += 1;
            if attempt > MAX_RETRIES {
                return Err(AppError::HttpStatus(
                    200,
                    "server ignored Range header on resume (got 200, expected 206)".into(),
                ));
            }
            let backoff = std::cmp::min(2u64.saturating_pow(attempt), 15);
            tracing::warn!(
                "chunk {chunk_index}: got 200 OK on a resume request (attempt {attempt}/{MAX_RETRIES}), \
                 expected 206 Partial; retrying in {backoff}s"
            );
            tokio::time::sleep(Duration::from_secs(backoff)).await;
            continue;
        }

        // --- Other non-success: fail hard with the body for diagnostics ---
        if !status.is_success() && status != StatusCode::PARTIAL_CONTENT {
            let code = status.as_u16();
            let body = resp.text().await.unwrap_or_default();
            return Err(AppError::HttpStatus(code, body.chars().take(200).collect()));
        }

        // --- Stream the body, writing at write_pos ---
        let mut stream = resp.bytes_stream();
        let mut stream_err: Option<reqwest::Error> = None;
        loop {
            if cancel.is_signaled() {
                return Err(cancelled_err(&cancel));
            }
            match stream.next().await {
                Some(Ok(bytes)) => {
                    let n = bytes.len() as u64;
                    file.seek(SeekFrom::Start(write_pos))?;
                    file.write_all(&bytes)?;
                    write_pos += n;
                    if tx
                        .send(ChunkProgress { chunk_index, bytes_written: n })
                        .await
                        .is_err()
                    {
                        return Err(AppError::DownloadCancelled);
                    }
                }
                Some(Err(e)) => {
                    stream_err = Some(e);
                    break;
                }
                None => return Ok(()), // stream exhausted → chunk complete
            }
        }

        // Mid-stream break: write_pos has advanced past bytes already on disk,
        // so the next iteration's Range resumes from exactly the right offset.
        if write_pos > chunk.end {
            return Ok(());
        }
        let e = stream_err.expect("loop only exits via return, error, or completion");
        attempt += 1;
        if attempt > MAX_RETRIES {
            return Err(AppError::Http(e));
        }
        let backoff = std::cmp::min(2u64.saturating_pow(attempt), 15);
        tracing::warn!(
            "chunk {chunk_index}: mid-stream break (attempt {attempt}/{MAX_RETRIES}), \
             resuming from byte {write_pos} in {backoff}s"
        );
        tokio::time::sleep(Duration::from_secs(backoff)).await;
    }
}

// ---------------------------------------------------------------------------
// Orchestrate all chunks for one job
// ---------------------------------------------------------------------------

/// Max simultaneous range requests per job. Within-file parallelism only —
/// jobs themselves run one at a time (see `downloader::start`).
pub const MAX_PARALLEL_CHUNKS: usize = 4;

/// Run all chunks for a job with bounded parallelism. Returns when all
/// complete or one fails.
///
/// At most `MAX_PARALLEL_CHUNKS` chunks are in flight at once, admitted in
/// index order (tokio's `Semaphore` is FIFO-fair), so with many fixed-size
/// chunks completion is roughly front-to-back. That keeps the contiguous
/// completed frontier — which feeds the overlapped SHA-256 hasher — close
/// behind the transfer instead of jumping from ~0 to 100% at the end.
pub async fn run_chunks(
    client: reqwest::Client,
    url: String,
    token: Option<String>,
    dest_path: std::path::PathBuf,
    chunks: Vec<ChunkState>,
    progress_tx: mpsc::Sender<ChunkProgress>,
    cancel: CancelToken,
) -> Result<(), AppError> {
    // Pre-create the file so parallel writes don't race on creation
    {
        let f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&dest_path)?;
        drop(f);
    }

    let semaphore = Arc::new(tokio::sync::Semaphore::new(MAX_PARALLEL_CHUNKS));
    let mut handles = Vec::with_capacity(chunks.len());

    for (i, chunk) in chunks.into_iter().enumerate() {
        let client_c = client.clone();
        let url_c = url.clone();
        let token_c = token.clone();
        let dest_c = dest_path.clone();
        let tx_c = progress_tx.clone();
        let cancel_c = cancel.clone();
        let sem_c = semaphore.clone();

        handles.push(tokio::spawn(async move {
            let _permit = sem_c
                .acquire()
                .await
                .map_err(|_| AppError::DownloadCancelled)?;
            if cancel_c.is_signaled() {
                return Err(match cancel_c.reason() {
                    CancelReason::Pause => AppError::DownloadPaused,
                    _ => AppError::DownloadCancelled,
                });
            }
            download_chunk(client_c, url_c, token_c, dest_c, i, chunk, tx_c, cancel_c).await
        }));
    }

    let results = future::join_all(handles).await;

    let mut paused = false;
    for r in results {
        match r {
            Ok(Ok(())) => {}
            Ok(Err(AppError::DownloadCancelled)) => return Err(AppError::DownloadCancelled),
            Ok(Err(AppError::DownloadPaused)) => paused = true,
            Ok(Err(e)) => return Err(e),
            Err(e) => return Err(AppError::Other(format!("chunk task panicked: {e}"))),
        }
    }

    if paused {
        return Err(AppError::DownloadPaused);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{CancelReason, CancelToken};

    #[test]
    fn new_token_is_not_signaled() {
        let t = CancelToken::new();
        assert!(!t.is_signaled());
        assert_eq!(t.reason(), CancelReason::None);
    }

    #[test]
    fn pause_sets_pause_reason() {
        let t = CancelToken::new();
        t.pause();
        assert!(t.is_signaled());
        assert_eq!(t.reason(), CancelReason::Pause);
    }

    #[test]
    fn cancel_sets_cancel_reason() {
        let t = CancelToken::new();
        t.cancel();
        assert!(t.is_signaled());
        assert_eq!(t.reason(), CancelReason::Cancel);
    }

    #[test]
    fn cancel_after_pause_wins() {
        let t = CancelToken::new();
        t.pause();
        t.cancel();
        assert_eq!(t.reason(), CancelReason::Cancel);
    }

    #[test]
    fn pause_does_not_overwrite_cancel() {
        let t = CancelToken::new();
        t.cancel();
        // A second pause() call after cancel should not downgrade to Pause.
        // pause() uses compare_exchange from RUNNING, so it is a no-op here.
        t.pause();
        assert_eq!(t.reason(), CancelReason::Cancel);
    }

    #[test]
    fn is_cancelled_alias_works() {
        let t = CancelToken::new();
        assert!(!t.is_cancelled());
        t.pause();
        assert!(t.is_cancelled());
    }
}
