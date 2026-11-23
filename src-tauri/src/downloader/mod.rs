use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use sqlx::SqlitePool;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::time::interval;
use uuid::Uuid;

use crate::{
    auth,
    config,
    db::downloads as db_dl,
    error::AppError,
    models::{
        hf::HfFile,
        registry,
        vibevoice_tokenizer,
        vibevoice_voices,
    },
};

pub mod chunker;
pub mod hasher;
pub mod job;
pub mod progress;
#[cfg(test)]
mod split_tests;

use chunker::{run_chunks, validate_remote, CancelToken, ChunkProgress};
use hasher::{contiguous_frontier, RollingHasher};
use job::{Job, JobStatus};
use progress::{DownloadPhase, DownloadProgress, SpeedTracker};

// ---------------------------------------------------------------------------
// Actor commands
// ---------------------------------------------------------------------------

pub enum Command {
    Start {
        repo_id: String,
        revision: String,
        file: HfFile,
        response: oneshot::Sender<Result<Uuid, AppError>>,
    },
    /// Multi-file repo download (M9 §9.2): one group row + one job per file.
    /// Responds with the group id.
    StartRepo {
        repo_id: String,
        revision: String,
        files: Vec<HfFile>,
        format: String,
        task: Option<String>,
        response: oneshot::Sender<Result<Uuid, AppError>>,
    },
    Pause(Uuid),
    Resume(Uuid),
    Cancel(Uuid),
    Delete(Uuid),
    /// Group-wide variants — applied to every member job of the group.
    PauseGroup(Uuid),
    ResumeGroup(Uuid),
    CancelGroup(Uuid),
    DeleteGroup(Uuid),
    List(oneshot::Sender<Vec<Job>>),
    /// Sent by the transfer task when the job reaches a terminal state.
    /// This is distinct from user-initiated Cancel so the two cannot race.
    JobFinished {
        id: Uuid,
        final_status: JobStatus,
        error: Option<String>,
    },
}

// ---------------------------------------------------------------------------
// Manager handle (cloneable, send to any thread)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct ManagerHandle {
    sender: mpsc::Sender<Command>,
    progress: broadcast::Sender<DownloadProgress>,
    /// Abort handle for the active progress-forwarder task.
    /// At most one frontend subscriber is supported at a time (v1).
    subscriber_abort: Arc<Mutex<Option<tokio::task::AbortHandle>>>,
}

impl ManagerHandle {
    pub async fn start(
        &self,
        repo_id: String,
        revision: String,
        file: HfFile,
    ) -> Result<Uuid, AppError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(Command::Start { repo_id, revision, file, response: tx })
            .await
            .map_err(|_| AppError::DownloadManagerUnavailable)?;
        rx.await.map_err(|_| AppError::DownloadManagerUnavailable)?
    }

    pub async fn start_repo(
        &self,
        repo_id: String,
        revision: String,
        files: Vec<HfFile>,
        format: String,
        task: Option<String>,
    ) -> Result<Uuid, AppError> {
        let (tx, rx) = oneshot::channel();
        self.sender
            .send(Command::StartRepo { repo_id, revision, files, format, task, response: tx })
            .await
            .map_err(|_| AppError::DownloadManagerUnavailable)?;
        rx.await.map_err(|_| AppError::DownloadManagerUnavailable)?
    }

    pub async fn pause(&self, id: Uuid) {
        let _ = self.sender.send(Command::Pause(id)).await;
    }

    pub async fn resume(&self, id: Uuid) {
        let _ = self.sender.send(Command::Resume(id)).await;
    }

    pub async fn cancel(&self, id: Uuid) {
        let _ = self.sender.send(Command::Cancel(id)).await;
    }

    pub async fn delete(&self, id: Uuid) {
        let _ = self.sender.send(Command::Delete(id)).await;
    }

    pub async fn pause_group(&self, id: Uuid) {
        let _ = self.sender.send(Command::PauseGroup(id)).await;
    }

    pub async fn resume_group(&self, id: Uuid) {
        let _ = self.sender.send(Command::ResumeGroup(id)).await;
    }

    pub async fn cancel_group(&self, id: Uuid) {
        let _ = self.sender.send(Command::CancelGroup(id)).await;
    }

    pub async fn delete_group(&self, id: Uuid) {
        let _ = self.sender.send(Command::DeleteGroup(id)).await;
    }

    pub async fn list(&self) -> Vec<Job> {
        let (tx, rx) = oneshot::channel();
        if self.sender.send(Command::List(tx)).await.is_err() {
            return vec![];
        }
        rx.await.unwrap_or_default()
    }

    pub fn subscribe_progress(&self) -> broadcast::Receiver<DownloadProgress> {
        self.progress.subscribe()
    }

    /// Register the abort handle for the active progress-forwarder task.
    /// If a previous forwarder is running, it is aborted first.
    /// This ensures at most one frontend subscriber exists at a time.
    pub fn set_subscriber_abort(&self, handle: tokio::task::AbortHandle) {
        let mut guard = self.subscriber_abort.lock().unwrap();
        if let Some(prev) = guard.take() {
            prev.abort();
        }
        *guard = Some(handle);
    }
}

// ---------------------------------------------------------------------------
// Manager actor
// ---------------------------------------------------------------------------

struct ActiveTransfer {
    cancel: CancelToken,
}

struct Manager {
    pool: SqlitePool,
    http: reqwest::Client,
    max_concurrent: usize,
    jobs: HashMap<Uuid, Job>,
    active: HashMap<Uuid, ActiveTransfer>,
    progress_tx: broadcast::Sender<DownloadProgress>,
    speed_trackers: HashMap<Uuid, SpeedTracker>,
    /// Last time chunks_json was flushed to DB per job (Unix timestamp).
    last_db_flush: HashMap<Uuid, i64>,
    /// Monotonic insertion sequence per job — `jobs` is a HashMap, so this is
    /// what makes the Queued→Running promotion FIFO (creation order; DB rows
    /// are loaded ordered by created_at on startup).
    seq: HashMap<Uuid, u64>,
    next_seq: u64,
}

impl Manager {
    fn new(
        pool: SqlitePool,
        http: reqwest::Client,
        max_concurrent: usize,
        progress_tx: broadcast::Sender<DownloadProgress>,
    ) -> Self {
        Self {
            pool,
            http,
            max_concurrent,
            jobs: HashMap::new(),
            active: HashMap::new(),
            progress_tx,
            speed_trackers: HashMap::new(),
            last_db_flush: HashMap::new(),
            seq: HashMap::new(),
            next_seq: 0,
        }
    }

    /// Track a job in memory, assigning its FIFO insertion sequence.
    fn insert_job(&mut self, job: Job) {
        self.seq.insert(job.id, self.next_seq);
        self.next_seq += 1;
        self.jobs.insert(job.id, job);
    }

    /// On startup: load persisted jobs, patch Running→Paused, re-queue Queued.
    async fn load_from_db(&mut self) {
        self.load_from_db_at(None).await;
    }

    async fn load_from_db_at(&mut self, root: Option<std::path::PathBuf>) {
        match db_dl::list_all(&self.pool).await {
            Ok(rows) => {
                for row in rows {
                    match Job::try_from(row) {
                        Ok(mut job) => {
                            if job.status == JobStatus::Running {
                                job.status = JobStatus::Paused;
                                let now = Utc::now().timestamp();
                                let _ = db_dl::update_status(
                                    &self.pool,
                                    job.id,
                                    JobStatus::Paused,
                                    None,
                                    now,
                                )
                                .await;
                            }
                            // Rows arrive ordered by created_at, so insertion
                            // sequence == persisted creation order.
                            self.insert_job(job);
                        }
                        Err(e) => tracing::warn!("load_from_db: invalid row: {e}"),
                    }
                }
            }
            Err(e) => tracing::error!("load_from_db: {e}"),
        }

        // Self-heal: a crash between last-member-done and catalog registration
        // leaves a fully-Done group unregistered. GGUF groups consume a durable
        // once-only marker in the catalog transaction; directory formats upsert.
        let group_ids: Vec<Uuid> = {
            let mut ids: Vec<Uuid> =
                self.jobs.values().filter_map(|j| j.group_id).collect();
            ids.sort_unstable();
            ids.dedup();
            ids
        };
        for gid in group_ids {
            self.register_group_at(gid, root.clone()).await;
        }
    }

    /// Promote Queued jobs up to the concurrent limit, oldest first
    /// (insertion sequence — see `insert_job`).
    async fn maybe_start_queued(&mut self, cmd_tx: mpsc::Sender<Command>) {
        let running_count = self.active.len();
        if running_count >= self.max_concurrent {
            return;
        }

        let mut queued: Vec<(u64, Uuid)> = self
            .jobs
            .values()
            .filter(|j| j.status == JobStatus::Queued)
            .map(|j| (self.seq.get(&j.id).copied().unwrap_or(u64::MAX), j.id))
            .collect();
        queued.sort_unstable();

        for (_, id) in queued {
            if self.active.len() >= self.max_concurrent { break; }
            if let Err(error) = self.check_group_write(&self.jobs[&id]).await {
                self.refuse_job(id, error).await;
                continue;
            }
            self.spawn_transfer(id, cmd_tx.clone());
        }
    }

    fn spawn_transfer(&mut self, id: Uuid, completion_tx: mpsc::Sender<Command>) {
        let job = match self.jobs.get_mut(&id) {
            Some(j) => j,
            None => return,
        };

        job.status = JobStatus::Running;

        let cancel = CancelToken::new();
        self.active.insert(id, ActiveTransfer { cancel: cancel.clone() });
        self.speed_trackers.insert(id, SpeedTracker::new());

        let pool = self.pool.clone();
        let http = self.http.clone();
        let job_clone = job.clone();
        let progress_bcast = self.progress_tx.clone();

        tokio::spawn(async move {
            let result = run_job(pool.clone(), http, job_clone.clone(), cancel, progress_bcast).await;

            let (status, error_msg) = match result {
                Ok(()) => (JobStatus::Done, None),
                Err(AppError::DownloadPaused) => (JobStatus::Paused, None),
                Err(AppError::DownloadCancelled) => (JobStatus::Cancelled, None),
                Err(e) => (JobStatus::Failed, Some(e.to_string())),
            };

            // Write final status to DB (task owns this write).
            let now = Utc::now().timestamp();
            let _ = db_dl::update_status(
                &pool,
                job_clone.id,
                status,
                error_msg.as_deref(),
                now,
            )
            .await;

            // Notify the manager via JobFinished — never collides with user Cancel.
            let _ = completion_tx
                .send(Command::JobFinished {
                    id: job_clone.id,
                    final_status: status,
                    error: error_msg,
                })
                .await;
        });
    }

    /// Member job ids of a download group.
    fn group_member_ids(&self, gid: Uuid) -> Vec<Uuid> {
        self.jobs
            .values()
            .filter(|j| j.group_id == Some(gid))
            .map(|j| j.id)
            .collect()
    }

    async fn pause_job(&mut self, id: Uuid) {
        if let Some(transfer) = self.active.remove(&id) {
            transfer.cancel.pause();
        }
        if let Some(job) = self.jobs.get_mut(&id) {
            if job.status == JobStatus::Done {
                return;
            }
            job.status = JobStatus::Paused;
            let now = Utc::now().timestamp();
            let _ = db_dl::update_status(&self.pool, id, JobStatus::Paused, None, now).await;
            self.emit_progress(id);
        }
    }

    async fn check_installed_destinations(&self, destinations: &[String]) -> Result<(), AppError> {
        if crate::db::models::owned_paths(&self.pool).await?.iter().any(|path|
            destinations.contains(&crate::models::split_gguf::destination_key(path))) {
            return Err(AppError::Other("This model is already installed. Remove it from Models before downloading it again.".into()));
        }
        Ok(())
    }

    async fn check_group_write(&self, job: &Job) -> Result<(), AppError> {
        let Some(gid) = job.group_id else { return Ok(()); };
        let Some(group) = db_dl::get_group(&self.pool, &gid.to_string()).await? else {
            return Err(AppError::Other("This download group was removed.".into()));
        };
        if group.format != "gguf" { return Ok(()); }
        if group.registration_closed {
            return Err(AppError::Other("This download was superseded. Start a new download from the Store.".into()));
        }
        self.check_installed_destinations(&[crate::models::split_gguf::destination_key(
            config::repo_root_dir(&job.repo_id).join(&job.file))]).await
    }

    async fn refuse_job(&mut self, id: Uuid, error: AppError) {
        if let Some(job) = self.jobs.get_mut(&id) {
            job.status = JobStatus::Failed;
            job.error = Some(error.to_string());
            let _ = db_dl::upsert_job(&self.pool, job).await;
        }
        self.emit_progress(id);
    }

    async fn resume_job(&mut self, id: Uuid) {
        let resumable = self
            .jobs
            .get(&id)
            .is_some_and(|j| matches!(j.status, JobStatus::Paused | JobStatus::Failed));
        if !resumable {
            return;
        }
        if let Err(error) = self.check_group_write(&self.jobs[&id]).await {
            self.refuse_job(id, error).await;
            return;
        }

        // Refresh chunk progress from the DB before requeueing: the transfer
        // task flushes chunks_json there once per second, but the manager's
        // in-memory copy is a snapshot from job creation/startup. Without
        // this, an in-session pause→resume restarts the file from byte 0.
        match db_dl::get_one(&self.pool, id).await {
            Ok(Some(row)) => match Job::try_from(row) {
                Ok(fresh) => {
                    if let Some(job) = self.jobs.get_mut(&id) {
                        job.chunks = fresh.chunks;
                        job.downloaded = fresh.downloaded;
                        job.total_bytes = fresh.total_bytes;
                    }
                }
                Err(e) => tracing::warn!("resume {id}: invalid persisted row ({e}); resuming with in-memory state"),
            },
            Ok(None) => {}
            Err(e) => tracing::warn!("resume {id}: failed to reload row ({e}); resuming with in-memory state"),
        }

        if let Some(job) = self.jobs.get_mut(&id) {
            job.status = JobStatus::Queued;
            job.error = None;
            let now = Utc::now().timestamp();
            let _ = db_dl::update_status(&self.pool, id, JobStatus::Queued, None, now).await;
            self.emit_progress(id);
        }
        // maybe_start_queued will pick it up
    }

    async fn cancel_job(&mut self, id: Uuid) {
        // User-initiated cancel only. Signal the transfer task to stop.
        if let Some(transfer) = self.active.remove(&id) {
            transfer.cancel.cancel();
        }
        if let Some(job) = self.jobs.get_mut(&id) {
            if !matches!(job.status, JobStatus::Done | JobStatus::Cancelled) {
                job.status = JobStatus::Cancelled;
                let now = Utc::now().timestamp();
                let _ =
                    db_dl::update_status(&self.pool, id, JobStatus::Cancelled, None, now).await;
                self.emit_progress(id);
            }
        }
    }

    async fn delete_job(&mut self, id: Uuid) {
        // Cancel in-flight if running
        if let Some(transfer) = self.active.remove(&id) {
            transfer.cancel.cancel();
        }
        self.jobs.remove(&id);
        self.speed_trackers.remove(&id);
        self.last_db_flush.remove(&id);
        self.seq.remove(&id);
        let _ = db_dl::delete_one(&self.pool, id).await;
    }

    /// If every member job of `gid` is Done, register the repo in the model
    /// catalog. Unconsumed GGUF groups self-heal only until their first commit;
    /// directory formats retain their existing replay/upsert behavior.
    async fn maybe_register_group(&mut self, gid: Uuid) {
        self.register_group_at(gid, None).await;
    }

    async fn register_group_at(&mut self, gid: Uuid, root: Option<std::path::PathBuf>) {
        let members: Vec<Job> = self
            .jobs
            .values()
            .filter(|j| j.group_id == Some(gid))
            .cloned()
            .collect();
        if members.is_empty() { return; }

        let gid_str = gid.to_string();
        let group = match db_dl::get_group(&self.pool, &gid_str).await {
            Ok(Some(g)) => g,
            Ok(None) => {
                tracing::error!("download group {gid} finished but its row is missing");
                return;
            }
            Err(e) => {
                tracing::error!("get_group({gid}): {e}");
                return;
            }
        };

        if group.format == "gguf" && group.registration_closed { return; }
        if !members.iter().all(|j| j.status == JobStatus::Done) { return; }

        let root = root.unwrap_or_else(|| config::repo_root_dir(&group.repo_id));
        let files: Vec<registry::RepoFile> = members
            .iter()
            .map(|j| registry::RepoFile {
                rel_path: j.file.clone(),
                path: root.join(&j.file).to_string_lossy().to_string(),
                sha256: j.expected_sha256.clone(),
                size: j.total_bytes,
            })
            .collect();

        // Done jobs outlive catalog deletion. Never resurrect a removed model
        // (or register an incomplete set) during startup self-healing.
        if !files.iter().all(|f| std::path::Path::new(&f.path).is_file()) {
            return;
        }
        match registry::insert_repo_model(
            &self.pool,
            &group.repo_id,
            &group.revision,
            &group.format,
            group.task.as_deref(),
            &root.to_string_lossy(),
            &files,
            Some(&gid_str),
        )
        .await
        {
            Ok(model_id) => {
                tracing::info!("registered repo model {model_id} ({})", group.repo_id)
            }
            Err(e) => tracing::error!("insert_repo_model({}): {e}", group.repo_id),
        }

        // VibeVoice installs need the Qwen2.5-1.5B tokenizer locally, fetched
        // here (install time) so the worker never reaches Hugging Face at
        // inference. Best-effort + idempotent: a failure leaves an honest
        // error for the load path, and the next reinstall (or a startup
        // self-heal) retries. Runs only after a successful registration.
        if vibevoice_tokenizer::is_vibevoice_repo(&group.repo_id) {
            if let Err(e) = vibevoice_tokenizer::ensure_qwen_tokenizer(&self.http, &root).await {
                tracing::error!(
                    "VibeVoice tokenizer install for {} failed: {e}",
                    group.repo_id
                );
            }
            // Curated built-in preset voices — fetched here so the user can pick
            // a named voice without supplying a reference clip. Same best-effort
            // contract: a failure just means fewer (or no) presets until a
            // reinstall or the startup heal retries.
            if let Err(e) = vibevoice_voices::ensure_builtin_voices(&self.http, &root).await {
                tracing::error!(
                    "VibeVoice preset-voice install for {} failed: {e}",
                    group.repo_id
                );
            }
        }
    }

    fn emit_progress(&self, id: Uuid) {
        let job = match self.jobs.get(&id) {
            Some(j) => j,
            None => return,
        };
        let tracker = self.speed_trackers.get(&id);
        let bps = tracker.map(|t| t.bytes_per_sec()).unwrap_or(0.0);
        let remaining = job.total_bytes.saturating_sub(job.downloaded);
        let eta_secs = tracker.and_then(|t| t.eta_secs(remaining));

        let ev = DownloadProgress {
            id,
            status: job.status,
            downloaded: job.downloaded,
            total: job.total_bytes,
            bytes_per_sec: bps,
            eta_secs,
            phase: DownloadPhase::Downloading,
            error: job.error.clone(),
        };
        let _ = self.progress_tx.send(ev);
    }
}

// ---------------------------------------------------------------------------
// Job execution (runs in a spawned task, separate from the actor)
// ---------------------------------------------------------------------------

// Extracted from run_job so fence interruption is testable with generated files.
async fn finalize_group_member(
    pool: &SqlitePool, job: &Job, gguf_group: Option<&str>,
    part_path: &std::path::Path, final_path: &std::path::Path, cancel: &CancelToken,
) -> Result<(), AppError> {
    let fence = if let Some(gid) = gguf_group {
        let fence = db_dl::gguf_write_fence(pool, gid).await?;
        match cancel.reason() {
            chunker::CancelReason::Pause => {
                // Release the write transaction before checkpointing via the pool.
                // The periodic relay may not have flushed the last completed chunk.
                drop(fence);
                db_dl::update_chunks(pool, job.id, &serde_json::to_string(&job.chunks)?,
                    job.downloaded, Utc::now().timestamp()).await?;
                return Err(AppError::DownloadPaused);
            }
            chunker::CancelReason::Cancel => return Err(AppError::DownloadCancelled),
            chunker::CancelReason::None => {}
        }
        Some(fence)
    } else { None };
    if let Some(parent) = final_path.parent() { std::fs::create_dir_all(parent)?; }
    std::fs::rename(part_path, final_path)?;
    if let Some(fence) = fence { fence.commit().await?; }
    Ok(())
}

async fn run_job(
    pool: SqlitePool,
    http: reqwest::Client,
    mut job: Job,
    cancel: CancelToken,
    progress_bcast: broadcast::Sender<DownloadProgress>,
) -> Result<(), AppError> {
    // Also guard persisted single-part jobs queued by older app versions.
    if job.group_id.is_none() { check_single_file(&job.file)?; }
    let gguf_group = if let Some(gid) = job.group_id {
        db_dl::get_group(&pool, &gid.to_string()).await?
            .filter(|group| group.format == "gguf").map(|_| gid.to_string())
    } else { None };
    let token = auth::get_hf_token().ok().flatten();

    let url = format!(
        "https://huggingface.co/{}/resolve/{}/{}",
        job.repo_id, job.revision, job.file
    );

    // --- HEAD validation ---
    let confirmed_size =
        validate_remote(&http, &url, token.as_deref(), job.total_bytes).await?;
    if confirmed_size != job.total_bytes {
        job.total_bytes = confirmed_size;
    }

    // --- Staging path ---
    let staging_dir = config::models_dir().join("staging").join(job.id.to_string());
    std::fs::create_dir_all(&staging_dir)?;
    let filename = std::path::PathBuf::from(&job.file)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf")
        .to_string();
    let part_path = staging_dir.join(format!("{filename}.part"));

    job.dest_path = part_path.clone();

    // --- Chunk setup ---
    if job.chunks.is_empty() {
        // Many fixed-size (~64 MiB) chunks downloaded in index order with
        // bounded parallelism, so completion is roughly front-to-back and the
        // rolling hasher below can chase the contiguous frontier. Small files
        // (repo configs/tokenizers) stay a single request.
        job.chunks = Job::split_chunks(job.total_bytes, job::chunk_count(job.total_bytes));
    }

    // --- Rolling SHA-256 hasher (overlaps with the transfer) ---
    // Starts at byte 0 on every (re)start — hasher state is in-memory only,
    // so a resumed job re-hashes [0..frontier) from disk while the resumed
    // network transfer proceeds in parallel.
    let rolling = RollingHasher::spawn(part_path.clone(), contiguous_frontier(&job.chunks));
    let frontier_cell = rolling.frontier_handle();

    // --- Progress relay task ---
    let (chunk_tx, mut chunk_rx) = mpsc::channel::<ChunkProgress>(256);

    // Clone job.chunks for update tracking inside the relay
    let mut local_chunks = job.chunks.clone();

    let bcast_clone = progress_bcast.clone();
    let job_id = job.id;
    let total = job.total_bytes;
    let pool_clone = pool.clone();
    let chunks_for_relay = local_chunks.clone();

    let relay_task = tokio::spawn(async move {
        let mut tracker = SpeedTracker::new();
        let mut total_downloaded: u64 = chunks_for_relay.iter().map(|c| c.completed).sum();
        let mut tick = interval(Duration::from_millis(100));
        let mut db_tick = interval(Duration::from_secs(1));
        let mut relay_chunks = chunks_for_relay;

        loop {
            tokio::select! {
                msg = chunk_rx.recv() => {
                    match msg {
                        Some(ChunkProgress { chunk_index, bytes_written }) => {
                            if chunk_index < relay_chunks.len() {
                                relay_chunks[chunk_index].completed += bytes_written;
                            }
                            total_downloaded += bytes_written;
                            tracker.record(total_downloaded);
                        }
                        None => break, // channel closed = all chunks done
                    }
                }
                _ = tick.tick() => {
                    // Publish the contiguous frontier for the rolling hasher.
                    // Bytes are on disk before chunks report progress, so the
                    // hasher never reads ahead of stable data.
                    frontier_cell.store(
                        contiguous_frontier(&relay_chunks),
                        Ordering::Release,
                    );
                    let bps = tracker.bytes_per_sec();
                    let remaining = total.saturating_sub(total_downloaded);
                    let eta = tracker.eta_secs(remaining);
                    let ev = DownloadProgress {
                        id: job_id,
                        status: JobStatus::Running,
                        downloaded: total_downloaded,
                        total,
                        bytes_per_sec: bps,
                        eta_secs: eta,
                        phase: DownloadPhase::Downloading,
                        error: None,
                    };
                    let _ = bcast_clone.send(ev);
                }
                _ = db_tick.tick() => {
                    // Flush chunk progress to DB once per second
                    let chunks_json = serde_json::to_string(&relay_chunks).unwrap_or_else(|_| "[]".into());
                    let now = Utc::now().timestamp();
                    let _ = db_dl::update_chunks(&pool_clone, job_id, &chunks_json, total_downloaded, now).await;
                }
            }
        }

        (relay_chunks, total_downloaded)
    });

    // --- Run chunks ---
    let result = run_chunks(
        http.clone(),
        url.clone(),
        token.clone(),
        part_path.clone(),
        job.chunks.clone(),
        chunk_tx,
        cancel.clone(),
    )
    .await;

    // Wait for relay to drain
    let (final_chunks, final_downloaded) = relay_task.await.unwrap_or((job.chunks.clone(), 0));
    local_chunks = final_chunks;

    if let Err(e) = result {
        // Stop the rolling hasher and wait for it to release its file handle
        // (its state is discarded — a resume re-hashes from byte 0).
        rolling.abort().await;
        // Persist current chunk state so resume can continue
        let chunks_json = serde_json::to_string(&local_chunks).unwrap_or_else(|_| "[]".into());
        let now = Utc::now().timestamp();
        let _ = db_dl::update_chunks(&pool, job_id, &chunks_json, final_downloaded, now).await;
        return Err(e);
    }

    // --- SHA256 verification ---
    // Most of the file was already hashed during the transfer by the rolling
    // hasher. Publish the final frontier (== total once every chunk is done),
    // let the hasher drain to it, then hash only the remaining tail. If the
    // hasher fell behind or failed, the tail is simply bigger — correctness
    // never depends on it having kept up.
    rolling.set_frontier(contiguous_frontier(&local_chunks));
    let rolled = rolling.finish().await;

    let _ = progress_bcast.send(DownloadProgress {
        id: job_id,
        status: JobStatus::Running,
        downloaded: rolled.pos(),
        total: job.total_bytes,
        bytes_per_sec: 0.0,
        eta_secs: None,
        phase: DownloadPhase::Verifying,
        error: None,
    });

    let part_path_for_hash = part_path.clone();
    let bcast_for_hash = progress_bcast.clone();
    let total_for_hash = job.total_bytes;
    let computed_sha256 = tokio::task::spawn_blocking(move || {
        hasher::finish_hash_with_progress(rolled, &part_path_for_hash, total_for_hash, |hashed| {
            // Hashed-bytes drive `downloaded` so the Verifying bar moves
            // instead of looking hung on whatever tail remains.
            let _ = bcast_for_hash.send(DownloadProgress {
                id: job_id,
                status: JobStatus::Running,
                downloaded: hashed,
                total: total_for_hash,
                bytes_per_sec: 0.0,
                eta_secs: None,
                phase: DownloadPhase::Verifying,
                error: None,
            });
        })
    })
    .await
    .map_err(|e| AppError::Other(format!("hash task join failed: {e}")))??;

    if let Some(expected) = &job.expected_sha256 {
        if !expected.is_empty() && computed_sha256 != *expected {
            tracing::error!(
                "SHA256 mismatch for job {}: expected {expected}, got {computed_sha256}",
                job.id
            );
            return Err(AppError::DownloadIntegrityMismatch);
        }
    }

    tracing::info!("SHA256 verified: {computed_sha256}");

    // --- Finalizing phase: rename + GGUF parse + catalog insert ---
    let _ = progress_bcast.send(DownloadProgress {
        id: job_id,
        status: JobStatus::Running,
        downloaded: job.total_bytes,
        total: job.total_bytes,
        bytes_per_sec: 0.0,
        eta_secs: None,
        phase: DownloadPhase::Finalizing,
        error: None,
    });

    if job.group_id.is_some() {
        // --- Group member: land at models\repos\<owner>__<name>\<rel_path>.
        // Catalog registration happens in the manager once the whole group
        // is Done (Command::JobFinished → maybe_register_group).
        job.chunks = local_chunks;
        job.downloaded = final_downloaded;
        finalize_group_member(&pool, &job, gguf_group.as_deref(), &part_path,
            &config::repo_root_dir(&job.repo_id).join(&job.file), &cancel).await?;
        let _ = std::fs::remove_dir_all(&staging_dir);
    } else {
        // --- Single GGUF: content-addressed blob + immediate catalog insert.
        let final_path = config::models_dir().join(format!("{computed_sha256}.gguf"));
        std::fs::rename(&part_path, &final_path)?;

        // Clean up staging dir (best effort)
        let _ = std::fs::remove_dir_all(&staging_dir);

        // --- Parse GGUF metadata ---
        let gguf_meta = crate::models::gguf::parse_header(&final_path).unwrap_or_default();

        // --- Insert into model catalog ---
        let card = crate::models::hf::HfModelCard {
            id: job.repo_id.clone(),
            author: job.repo_id.split('/').next().map(|s| s.to_string()),
            pipeline_tag: None,
            library_name: Some("gguf".to_string()),
            tags: vec![],
            downloads: 0,
            likes: 0,
            trending_score: None,
            gated: false,
            private: false,
            last_modified: None,
            created_at: None,
        };

        if let Err(e) =
            registry::insert_after_download(&pool, &job, &computed_sha256, &gguf_meta, &card).await
        {
            tracing::error!("Failed to insert model into catalog: {e}");
        }
    }

    // --- Emit final Done progress ---
    let ev = DownloadProgress {
        id: job_id,
        status: JobStatus::Done,
        downloaded: job.total_bytes,
        total: job.total_bytes,
        bytes_per_sec: 0.0,
        eta_secs: Some(0),
        phase: DownloadPhase::Finalizing,
        error: None,
    };
    let _ = progress_bcast.send(ev);

    Ok(())
}

fn check_single_file(path: &str) -> Result<(), AppError> {
    if crate::models::split_gguf::part(path).is_some() {
        return Err(AppError::Other(
            "This file is part of a split GGUF. Download the complete set instead.".into(),
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public factory — starts the actor and returns a handle
// ---------------------------------------------------------------------------

pub fn start(
    pool: SqlitePool,
    http: reqwest::Client,
    max_concurrent: usize,
) -> ManagerHandle {
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<Command>(64);
    let (progress_tx, _) = broadcast::channel::<DownloadProgress>(256);

    let handle = ManagerHandle {
        sender: cmd_tx.clone(),
        progress: progress_tx.clone(),
        subscriber_abort: Arc::new(Mutex::new(None)),
    };

    let mut mgr = Manager::new(pool, http, max_concurrent, progress_tx);

    tauri::async_runtime::spawn(async move {
        // Load persisted state from DB
        mgr.load_from_db().await;
        // Start any queued jobs
        mgr.maybe_start_queued(cmd_tx.clone()).await;

        loop {
            let Some(cmd) = cmd_rx.recv().await else { break; };
            handle_command(&mut mgr, cmd, cmd_tx.clone()).await;
            mgr.maybe_start_queued(cmd_tx.clone()).await;
        }
    });

    handle
}


async fn handle_command(mgr: &mut Manager, cmd: Command, _cmd_tx: mpsc::Sender<Command>) {
    match cmd {
        Command::Start { repo_id, revision, file, response } => {
            if let Err(e) = check_single_file(&file.path) {
                let _ = response.send(Err(e));
                return;
            }
            let now = Utc::now().timestamp();
            let id = Uuid::new_v4();
            let dest_path = config::models_dir()
                .join("staging")
                .join(id.to_string())
                .join(format!("{}.part", file.path.replace('/', "_")));

            let job = Job {
                id,
                repo_id: repo_id.clone(),
                revision: revision.clone(),
                file: file.path.clone(),
                dest_path,
                total_bytes: file.size,
                downloaded: 0,
                chunks: vec![],
                status: JobStatus::Queued,
                error: None,
                expected_sha256: file.sha256.clone(),
                group_id: None,
                created_at: now,
                updated_at: now,
            };

            // Persist to DB
            let persist_result = db_dl::upsert_job(&mgr.pool, &job).await;
            match persist_result {
                Ok(()) => {
                    mgr.insert_job(job);
                    let _ = response.send(Ok(id));
                }
                Err(e) => {
                    let _ = response.send(Err(e));
                }
            }
        }

        Command::StartRepo { repo_id, revision, files, format, task, response } => {
            // Reject path traversal before anything touches disk. HF rel_paths
            // are forward-slash relative; any ".." component is hostile.
            let unsafe_path = files.iter().find(|f| {
                std::path::Path::new(&f.path)
                    .components()
                    .any(|c| matches!(c, std::path::Component::ParentDir))
                    || std::path::Path::new(&f.path).is_absolute()
            });
            if let Some(f) = unsafe_path {
                let _ = response.send(Err(AppError::Other(format!(
                    "unsafe file path in repo download: {}",
                    f.path
                ))));
                return;
            }
            if files.is_empty() {
                let _ = response
                    .send(Err(AppError::Other("repo download has no files".into())));
                return;
            }

            let requested: Vec<_> = files.iter().map(|f|
                crate::models::split_gguf::destination_key(config::repo_root_dir(&repo_id).join(&f.path))).collect();
            if format == "gguf" {
                if requested.iter().collect::<std::collections::HashSet<_>>().len() != requested.len() {
                    let _ = response.send(Err(AppError::Other("This download contains parts that point to the same file.".into())));
                    return;
                }
                if let Err(e) = crate::models::split_gguf::validate(files.iter().map(|f| f.path.as_str())) {
                    let _ = response.send(Err(e));
                    return;
                }
                // Before any predecessor closure or new history/staging writes.
                if let Err(e) = mgr.check_installed_destinations(&requested).await {
                    let _ = response.send(Err(e));
                    return;
                }
            }
            // Closed GGUF history can never write again, even if its UI status
            // is Paused/Failed. It must not block a fresh attempt after Remove.
            let mut closed = std::collections::HashSet::new();
            if format == "gguf" {
                match db_dl::list_groups(&mgr.pool).await {
                    Ok(groups) => closed.extend(groups.into_iter().filter(|g| g.format == "gguf" && g.registration_closed)
                        .filter_map(|g| Uuid::parse_str(&g.id).ok())),
                    Err(e) => { let _ = response.send(Err(e)); return; }
                }
            }
            // Other formats retain their existing exact repo/path guard.
            if mgr.jobs.values().any(|j|
                !matches!(j.status, JobStatus::Done | JobStatus::Cancelled)
                && !(format == "gguf" && j.group_id.is_some_and(|gid| closed.contains(&gid)))
                && if format == "gguf" {
                    requested.contains(&crate::models::split_gguf::destination_key(
                        config::repo_root_dir(&j.repo_id).join(&j.file)))
                } else { j.repo_id == repo_id && files.iter().any(|f| f.path == j.file) }) {
                let _ = response.send(Err(AppError::Other("These files already have a download. Resume or remove it first.".into())));
                return;
            }
            let gid = Uuid::new_v4();
            if format == "gguf" {
                if let Err(e) = db_dl::supersede_gguf_groups(&mgr.pool, &requested).await {
                    let _ = response.send(Err(e));
                    return;
                }
            }
            let now = Utc::now().timestamp();
            let group = db_dl::GroupRow {
                id: gid.to_string(),
                repo_id: repo_id.clone(),
                revision: revision.clone(),
                format,
                task,
                created_at: now,
                registration_closed: false,
            };
            if let Err(e) = db_dl::insert_group(&mgr.pool, &group).await {
                let _ = response.send(Err(e));
                return;
            }

            for file in files {
                let id = Uuid::new_v4();
                let dest_path = config::models_dir()
                    .join("staging")
                    .join(id.to_string())
                    .join(format!("{}.part", file.path.replace('/', "_")));

                let job = Job {
                    id,
                    repo_id: repo_id.clone(),
                    revision: revision.clone(),
                    file: file.path.clone(),
                    dest_path,
                    total_bytes: file.size,
                    downloaded: 0,
                    chunks: vec![],
                    status: JobStatus::Queued,
                    error: None,
                    expected_sha256: file.sha256.clone(),
                    group_id: Some(gid),
                    created_at: now,
                    updated_at: now,
                };

                match db_dl::upsert_job(&mgr.pool, &job).await {
                    Ok(()) => {
                        mgr.insert_job(job);
                    }
                    Err(e) => {
                        // Abort the whole group on persist failure — a partial
                        // group would register an incomplete model.
                        tracing::error!("StartRepo: persisting job failed: {e}");
                        let member_ids = mgr.group_member_ids(gid);
                        for mid in member_ids {
                            mgr.jobs.remove(&mid);
                            mgr.seq.remove(&mid);
                        }
                        let _ = db_dl::delete_group(&mgr.pool, gid).await;
                        let _ = response.send(Err(e));
                        return;
                    }
                }
            }

            let _ = response.send(Ok(gid));
        }

        Command::Pause(id) => mgr.pause_job(id).await,
        Command::Resume(id) => mgr.resume_job(id).await,
        Command::Cancel(id) => mgr.cancel_job(id).await,
        Command::Delete(id) => mgr.delete_job(id).await,

        Command::PauseGroup(gid) => {
            for id in mgr.group_member_ids(gid) {
                mgr.pause_job(id).await;
            }
        }

        Command::ResumeGroup(gid) => {
            for id in mgr.group_member_ids(gid) {
                mgr.resume_job(id).await;
            }
        }

        Command::CancelGroup(gid) => {
            for id in mgr.group_member_ids(gid) {
                mgr.cancel_job(id).await;
            }
        }

        Command::DeleteGroup(gid) => {
            for id in mgr.group_member_ids(gid) {
                mgr.delete_job(id).await;
            }
            // Removes the group row and any member job rows that survived
            // (e.g. rows not loaded into memory).
            let _ = db_dl::delete_group(&mgr.pool, gid).await;
        }

        Command::JobFinished { id, final_status, error } => {
            // Sent by the transfer task when it reaches Done, Failed, Paused, or Cancelled.
            // DB has already been written by the task — manager just updates memory.
            mgr.active.remove(&id);
            mgr.speed_trackers.remove(&id);
            if let Some(job) = mgr.jobs.get_mut(&id) {
                // Do not downgrade a terminal status set by a later user action.
                // Example: user pauses then immediately cancels before the task
                // observes the pause signal; the in-memory status is Cancelled and
                // must not be overwritten with Paused from the stale task result.
                if matches!(job.status, JobStatus::Cancelled | JobStatus::Done) {
                    return;
                }
                job.status = final_status;
                job.error = error;
            }
            mgr.emit_progress(id);

            // Group completion: when the last member of a group verifies,
            // register the repo in the model catalog (exactly once — the
            // actor is single-threaded, so this check cannot race).
            if final_status == JobStatus::Done {
                if let Some(gid) = mgr.jobs.get(&id).and_then(|j| j.group_id) {
                    mgr.maybe_register_group(gid).await;
                }
            }
        }

        Command::List(response) => {
            let jobs: Vec<Job> = mgr.jobs.values().cloned().collect();
            let _ = response.send(jobs);
        }
    }
}
