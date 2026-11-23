//! Frontier hashing — overlap SHA-256 with the download so the post-transfer
//! "Verifying" phase shrinks to a short tail read instead of a full-file pass.
//!
//! Design:
//! - A `RollingHasher` background task per job chases the **contiguous
//!   completed frontier**: the byte length of the prefix `[0..frontier)` that
//!   is fully on disk. As chunks complete front-to-back, the relay task keeps
//!   an `AtomicU64` frontier up to date and the hasher reads newly contiguous
//!   bytes from the `.part` file into a rolling `Sha256`.
//! - Network is the bottleneck while downloading, so this disk read overlaps
//!   for free. When the last chunk lands, only the unhashed tail remains.
//! - **Correctness never depends on the hasher keeping up.** The verify phase
//!   (`finish_hash_with_progress`) hashes whatever remains from the rolling
//!   hasher's position to the end of the file. If the rolling task failed or
//!   fell behind, the tail is just bigger.
//! - **Resume:** hasher state is in-memory only. On every job (re)start the
//!   task begins at byte 0 and re-hashes up to the current frontier — still
//!   overlapped with the resumed network transfer.
//!
//! Safety of reading while chunk tasks write: chunk tasks write each byte
//! range to disk *before* reporting it as completed, and the frontier is
//! computed from reported completion — so the hasher only ever reads bytes
//! that are final on disk.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};

use super::job::ChunkState;

/// Read buffer for hashing. 1 MiB amortises syscalls on multi-GB files.
const BUF_SIZE: usize = 1024 * 1024;

/// Max bytes hashed per loop iteration in the background task, so control
/// signals (finish/abort) are observed with bounded latency.
const STEP_BYTES: u64 = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Contiguous frontier
// ---------------------------------------------------------------------------

/// Length of the contiguous completed prefix `[0..frontier)` of the file.
///
/// Chunks are ordered and adjacent (`split_chunks` invariant) and each chunk
/// downloads sequentially from its start, so bytes `[start, start+completed)`
/// of every chunk are on disk. The frontier advances through fully-done
/// chunks and stops inside the first incomplete one.
pub fn contiguous_frontier(chunks: &[ChunkState]) -> u64 {
    let mut frontier = 0u64;
    for c in chunks {
        if c.start != frontier {
            // Gap before this chunk (defensive — split_chunks never produces one).
            break;
        }
        frontier = c.start + c.completed.min(c.len());
        if !c.is_done() {
            break;
        }
    }
    frontier
}

// ---------------------------------------------------------------------------
// FrontierHasher — rolling SHA-256 over a growing prefix
// ---------------------------------------------------------------------------

/// Rolling SHA-256 over the prefix `[0..pos)` of a file.
pub struct FrontierHasher {
    hasher: Sha256,
    pos: u64,
}

impl Default for FrontierHasher {
    fn default() -> Self {
        Self::new()
    }
}

impl FrontierHasher {
    pub fn new() -> Self {
        Self { hasher: Sha256::new(), pos: 0 }
    }

    /// Bytes hashed so far (`[0..pos)` is in the rolling digest).
    pub fn pos(&self) -> u64 {
        self.pos
    }

    /// Hash file bytes `[pos..frontier)`, advancing `pos`. No-op when
    /// `frontier <= pos`. Stops early (without error) at EOF — the caller's
    /// final length/digest comparison catches a short file.
    pub fn advance_to(&mut self, file: &mut File, frontier: u64) -> std::io::Result<()> {
        if frontier <= self.pos {
            return Ok(());
        }
        file.seek(SeekFrom::Start(self.pos))?;
        let mut buf = vec![0u8; BUF_SIZE];
        let mut remaining = frontier - self.pos;
        while remaining > 0 {
            let want = remaining.min(BUF_SIZE as u64) as usize;
            let n = file.read(&mut buf[..want])?;
            if n == 0 {
                break; // EOF before frontier — file shorter than expected
            }
            self.hasher.update(&buf[..n]);
            self.pos += n as u64;
            remaining -= n as u64;
        }
        Ok(())
    }

    /// Consume the hasher, returning the lowercase hex digest of `[0..pos)`.
    pub fn finalize(self) -> String {
        format!("{:x}", self.hasher.finalize())
    }
}

// ---------------------------------------------------------------------------
// RollingHasher — background task that chases the frontier during download
// ---------------------------------------------------------------------------

const CTRL_RUN: u8 = 0;
const CTRL_FINISH: u8 = 1;
const CTRL_ABORT: u8 = 2;

/// Handle to the per-job background hashing task.
pub struct RollingHasher {
    frontier: Arc<AtomicU64>,
    ctrl: Arc<AtomicU8>,
    handle: tokio::task::JoinHandle<std::io::Result<FrontierHasher>>,
}

impl RollingHasher {
    /// Spawn the background hasher for `path` with an initial frontier
    /// (non-zero when resuming a partially-downloaded job).
    pub fn spawn(path: PathBuf, initial_frontier: u64) -> Self {
        let frontier = Arc::new(AtomicU64::new(initial_frontier));
        let ctrl = Arc::new(AtomicU8::new(CTRL_RUN));
        let frontier_c = frontier.clone();
        let ctrl_c = ctrl.clone();

        let handle = tokio::task::spawn_blocking(move || -> std::io::Result<FrontierHasher> {
            let mut fh = FrontierHasher::new();
            // The .part file may not exist yet on a fresh job — open lazily.
            let mut file: Option<File> = None;
            loop {
                let ctrl_now = ctrl_c.load(Ordering::Acquire);
                if ctrl_now == CTRL_ABORT {
                    return Ok(fh);
                }

                let target = frontier_c.load(Ordering::Acquire);
                let before = fh.pos();
                if before < target {
                    if file.is_none() {
                        file = File::open(&path).ok();
                    }
                    if let Some(f) = file.as_mut() {
                        // Bounded step so finish/abort signals stay responsive.
                        let step_target = target.min(before + STEP_BYTES);
                        fh.advance_to(f, step_target)?;
                    }
                }

                let progressed = fh.pos() > before;
                if ctrl_now == CTRL_FINISH && (fh.pos() >= target || !progressed) {
                    // Drained to the final frontier (or can't make progress —
                    // the verify pass will hash whatever is left).
                    return Ok(fh);
                }
                if !progressed {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        });

        Self { frontier, ctrl, handle }
    }

    /// Shared frontier cell — the progress relay stores updates into this.
    pub fn frontier_handle(&self) -> Arc<AtomicU64> {
        self.frontier.clone()
    }

    /// Publish a new frontier value.
    pub fn set_frontier(&self, frontier: u64) {
        self.frontier.store(frontier, Ordering::Release);
    }

    /// Ask the task to drain up to the current frontier and return its state.
    /// Await the result; falls back to a fresh hasher on any task failure.
    pub async fn finish(self) -> FrontierHasher {
        self.ctrl.store(CTRL_FINISH, Ordering::Release);
        match self.handle.await {
            Ok(Ok(fh)) => fh,
            Ok(Err(e)) => {
                tracing::warn!("rolling hasher failed ({e}); verify will hash the full file");
                FrontierHasher::new()
            }
            Err(e) => {
                tracing::warn!("rolling hasher task join failed ({e}); verify will hash the full file");
                FrontierHasher::new()
            }
        }
    }

    /// Ask the task to stop as soon as possible (pause/cancel/error paths)
    /// and wait for it to release its file handle.
    pub async fn abort(self) {
        self.ctrl.store(CTRL_ABORT, Ordering::Release);
        let _ = self.handle.await;
    }
}

// ---------------------------------------------------------------------------
// Verify-phase tail hash
// ---------------------------------------------------------------------------

/// Hash the remaining bytes `[fh.pos()..total)` (bounded by EOF) and finalize.
///
/// `on_progress` is invoked with the cumulative hashed byte count after each
/// step so the UI can show a moving bar during the Verifying phase.
///
/// Returns the lowercase hex digest of everything hashed (`[0..end)`).
pub fn finish_hash_with_progress(
    mut fh: FrontierHasher,
    path: &Path,
    total: u64,
    mut on_progress: impl FnMut(u64),
) -> std::io::Result<String> {
    if fh.pos() < total {
        let mut file = File::open(path)?;
        while fh.pos() < total {
            let before = fh.pos();
            let target = total.min(before + STEP_BYTES);
            fh.advance_to(&mut file, target)?;
            if fh.pos() == before {
                // EOF short of `total` — digest comparison surfaces the
                // mismatch; never spin here.
                break;
            }
            on_progress(fh.pos());
        }
    }
    Ok(fh.finalize())
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::downloader::job::Job;
    use std::io::Write;

    /// One-shot reference hash (the pre-frontier implementation) used as the
    /// oracle for every staged/rolling variant.
    fn hash_file_sha256(path: &Path) -> String {
        let file = File::open(path).unwrap();
        let mut reader = std::io::BufReader::with_capacity(BUF_SIZE, file);
        let mut hasher = Sha256::new();
        std::io::copy(&mut reader, &mut hasher).unwrap();
        format!("{:x}", hasher.finalize())
    }

    fn temp_file_with(len: usize) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file.part");
        let bytes: Vec<u8> = (0..len).map(|i| ((i * 31 + 7) % 251) as u8).collect();
        let mut f = File::create(&path).unwrap();
        f.write_all(&bytes).unwrap();
        (dir, path)
    }

    // --- contiguous_frontier ------------------------------------------------

    #[test]
    fn frontier_empty_chunks_is_zero() {
        assert_eq!(contiguous_frontier(&[]), 0);
    }

    #[test]
    fn frontier_partial_first_chunk() {
        let mut chunks = Job::split_chunks(100, 4);
        chunks[0].completed = 10;
        assert_eq!(contiguous_frontier(&chunks), 10);
    }

    #[test]
    fn frontier_ignores_out_of_order_completion() {
        // Chunk 2 fully done but chunk 0 only partial: frontier must stop
        // inside chunk 0 — bytes after the gap are not contiguous.
        let mut chunks = Job::split_chunks(100, 4);
        chunks[0].completed = 5;
        chunks[2].completed = chunks[2].len();
        assert_eq!(contiguous_frontier(&chunks), 5);
    }

    #[test]
    fn frontier_advances_through_done_chunks() {
        let mut chunks = Job::split_chunks(100, 4);
        chunks[0].completed = chunks[0].len();
        chunks[1].completed = 7;
        chunks[3].completed = chunks[3].len(); // out of order — ignored
        assert_eq!(contiguous_frontier(&chunks), chunks[1].start + 7);
    }

    #[test]
    fn frontier_all_done_is_total() {
        let mut chunks = Job::split_chunks(101, 4);
        for c in chunks.iter_mut() {
            c.completed = c.len();
        }
        assert_eq!(contiguous_frontier(&chunks), 101);
    }

    #[test]
    fn frontier_caps_overreported_completion() {
        // A chunk reporting more than its own length must not push the
        // frontier past the chunk boundary unless it is actually done.
        let mut chunks = Job::split_chunks(100, 4);
        chunks[0].completed = chunks[0].len() + 50;
        assert_eq!(contiguous_frontier(&chunks), chunks[1].start);
    }

    // --- FrontierHasher -----------------------------------------------------

    #[test]
    fn staged_advance_equals_one_shot_hash() {
        // Simulates the frontier growing in stages during a download.
        let (_dir, path) = temp_file_with(3 * 1024 * 1024 + 13);
        let oracle = hash_file_sha256(&path);

        let mut fh = FrontierHasher::new();
        let mut file = File::open(&path).unwrap();
        let total = std::fs::metadata(&path).unwrap().len();
        for frontier in [10_000, 1_500_000, 1_500_000, 2_999_999, total] {
            fh.advance_to(&mut file, frontier).unwrap();
        }
        assert_eq!(fh.pos(), total);
        assert_eq!(fh.finalize(), oracle);
    }

    #[test]
    fn resume_rehash_from_zero_equals_one_shot_hash() {
        // Resume semantics: hasher state does not survive a restart, so a
        // resumed job re-hashes [0..frontier) from disk in one big advance.
        let (_dir, path) = temp_file_with(2 * 1024 * 1024);
        let oracle = hash_file_sha256(&path);

        let mut fh = FrontierHasher::new();
        let mut file = File::open(&path).unwrap();
        let total = std::fs::metadata(&path).unwrap().len();
        // One advance straight to a mid-file frontier, then the tail.
        fh.advance_to(&mut file, total / 2 + 17).unwrap();
        fh.advance_to(&mut file, total).unwrap();
        assert_eq!(fh.finalize(), oracle);
    }

    #[test]
    fn advance_stops_at_eof_without_error() {
        let (_dir, path) = temp_file_with(1000);
        let mut fh = FrontierHasher::new();
        let mut file = File::open(&path).unwrap();
        fh.advance_to(&mut file, 5000).unwrap();
        assert_eq!(fh.pos(), 1000);
    }

    // --- finish_hash_with_progress -------------------------------------------

    #[test]
    fn finish_hashes_full_file_when_hasher_fell_behind() {
        // Worst case: the rolling hasher contributed nothing (pos = 0).
        let (_dir, path) = temp_file_with(512 * 1024 + 3);
        let oracle = hash_file_sha256(&path);
        let total = std::fs::metadata(&path).unwrap().len();

        let mut last = 0u64;
        let hex = finish_hash_with_progress(FrontierHasher::new(), &path, total, |p| last = p)
            .unwrap();
        assert_eq!(hex, oracle);
        assert_eq!(last, total);
    }

    #[test]
    fn finish_hashes_only_tail_after_partial_roll() {
        let (_dir, path) = temp_file_with(1024 * 1024);
        let oracle = hash_file_sha256(&path);
        let total = std::fs::metadata(&path).unwrap().len();

        let mut fh = FrontierHasher::new();
        let mut file = File::open(&path).unwrap();
        fh.advance_to(&mut file, 700_000).unwrap();
        drop(file);

        let hex = finish_hash_with_progress(fh, &path, total, |_| {}).unwrap();
        assert_eq!(hex, oracle);
    }

    #[test]
    fn finish_no_op_when_already_complete() {
        let (_dir, path) = temp_file_with(4096);
        let oracle = hash_file_sha256(&path);
        let total = 4096u64;

        let mut fh = FrontierHasher::new();
        let mut file = File::open(&path).unwrap();
        fh.advance_to(&mut file, total).unwrap();
        drop(file);

        let mut calls = 0;
        let hex = finish_hash_with_progress(fh, &path, total, |_| calls += 1).unwrap();
        assert_eq!(hex, oracle);
        assert_eq!(calls, 0, "no tail left to hash");
    }

    // --- RollingHasher end-to-end -------------------------------------------

    /// Simulated out-of-order chunk completion driving the background task:
    /// the file grows on disk while the frontier is published in steps, the
    /// task drains on finish(), and the verify pass completes the digest.
    #[tokio::test]
    async fn rolling_hasher_chases_frontier_and_matches_oracle() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("grow.part");
        let total: u64 = 2 * 1024 * 1024 + 777;
        let bytes: Vec<u8> = (0..total as usize).map(|i| ((i * 13 + 5) % 250) as u8).collect();

        let roller = RollingHasher::spawn(path.clone(), 0);

        // Write the file in three stages, publishing the frontier after each
        // (mirrors chunk completion: bytes land on disk before being reported).
        let stages = [total / 3, 2 * total / 3, total];
        let mut written = 0u64;
        let mut f = File::create(&path).unwrap();
        for stage in stages {
            f.write_all(&bytes[written as usize..stage as usize]).unwrap();
            f.flush().unwrap();
            written = stage;
            roller.set_frontier(stage);
            tokio::time::sleep(Duration::from_millis(120)).await;
        }
        drop(f);

        let fh = roller.finish().await;
        assert!(fh.pos() <= total);

        let hex = tokio::task::spawn_blocking({
            let path = path.clone();
            move || finish_hash_with_progress(fh, &path, total, |_| {})
        })
        .await
        .unwrap()
        .unwrap();

        let oracle = {
            let mut h = Sha256::new();
            h.update(&bytes);
            format!("{:x}", h.finalize())
        };
        assert_eq!(hex, oracle);
    }

    #[tokio::test]
    async fn rolling_hasher_abort_returns_quickly() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("never-created.part");
        let roller = RollingHasher::spawn(path, 0);
        // No file, no frontier — abort must not hang.
        tokio::time::timeout(Duration::from_secs(5), roller.abort())
            .await
            .expect("abort should resolve promptly");
    }
}
