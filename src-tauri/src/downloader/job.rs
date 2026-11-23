use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::db::downloads::JobRow;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Paused,
    Done,
    Failed,
    Cancelled,
}

impl std::fmt::Display for JobStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            JobStatus::Queued => "queued",
            JobStatus::Running => "running",
            JobStatus::Paused => "paused",
            JobStatus::Done => "done",
            JobStatus::Failed => "failed",
            JobStatus::Cancelled => "cancelled",
        };
        write!(f, "{}", s)
    }
}

impl JobStatus {
    pub fn from_str(s: &str) -> Self {
        match s {
            "queued" => JobStatus::Queued,
            "running" => JobStatus::Running,
            "paused" => JobStatus::Paused,
            "done" => JobStatus::Done,
            "failed" => JobStatus::Failed,
            "cancelled" => JobStatus::Cancelled,
            _ => JobStatus::Failed,
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChunkState {
    /// Inclusive start byte offset of the chunk.
    pub start: u64,
    /// Inclusive end byte offset of the chunk.
    pub end: u64,
    /// Bytes completed within this chunk (relative offset from start).
    pub completed: u64,
}

impl ChunkState {
    /// Returns true if this chunk has been fully downloaded.
    pub fn is_done(&self) -> bool {
        self.completed >= (self.end - self.start + 1)
    }

    /// Total bytes this chunk covers.
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> u64 {
        self.end - self.start + 1
    }
}

/// Target chunk size for large files (64 MiB).
///
/// Files are split into many ~64 MiB chunks downloaded with bounded
/// parallelism in index order (see `chunker::run_chunks`), so completion is
/// roughly front-to-back. That keeps the contiguous frontier — and therefore
/// the overlapped SHA-256 in `hasher::RollingHasher` — close behind the
/// transfer, leaving only a small tail to hash in the Verifying phase.
pub const CHUNK_TARGET_BYTES: u64 = 64 * 1024 * 1024;

/// Number of chunks to split a file of `total_bytes` into.
///
/// Small files (< 8 MiB) use a single request — repo downloads include many
/// tiny configs/tokenizers where one GET is both faster and avoids
/// `split_chunks` degenerating on tiny totals.
pub fn chunk_count(total_bytes: u64) -> u64 {
    if total_bytes < 8 * 1024 * 1024 {
        1
    } else {
        total_bytes.div_ceil(CHUNK_TARGET_BYTES).max(1)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Job {
    pub id: Uuid,
    pub repo_id: String,
    pub revision: String,
    /// Relative file path within the repo (e.g. "model.Q4_K_M.gguf").
    pub file: String,
    /// Staging destination path (temp file while downloading).
    pub dest_path: PathBuf,
    pub total_bytes: u64,
    pub downloaded: u64,
    pub chunks: Vec<ChunkState>,
    pub status: JobStatus,
    pub error: Option<String>,
    /// Expected SHA-256 digest (from HF LFS metadata), hex-encoded.
    pub expected_sha256: Option<String>,
    /// Download group this job belongs to (multi-file repo downloads).
    /// None for standalone single-GGUF jobs.
    pub group_id: Option<Uuid>,
    pub created_at: i64,
    pub updated_at: i64,
}

impl Job {
    /// Split `total_bytes` into `n` equal (as possible) chunks.
    pub fn split_chunks(total_bytes: u64, n: u64) -> Vec<ChunkState> {
        if total_bytes == 0 || n == 0 {
            return vec![];
        }
        let chunk_size = total_bytes / n;
        let mut chunks = Vec::with_capacity(n as usize);
        for i in 0..n {
            let start = i * chunk_size;
            let end = if i == n - 1 {
                total_bytes - 1
            } else {
                (i + 1) * chunk_size - 1
            };
            chunks.push(ChunkState { start, end, completed: 0 });
        }
        chunks
    }

    /// Recompute `downloaded` as the sum of all chunk completed bytes.
    #[allow(dead_code)]
    pub fn recompute_downloaded(&mut self) {
        self.downloaded = self.chunks.iter().map(|c| c.completed).sum();
    }
}

// ---------------------------------------------------------------------------
// Conversion from DB row
// ---------------------------------------------------------------------------

impl TryFrom<JobRow> for Job {
    type Error = crate::error::AppError;

    fn try_from(row: JobRow) -> Result<Self, Self::Error> {
        let id = uuid::Uuid::parse_str(&row.id)
            .map_err(|e| crate::error::AppError::Other(format!("invalid job UUID: {e}")))?;
        let chunks: Vec<ChunkState> = serde_json::from_str(&row.chunks_json)?;
        let status = JobStatus::from_str(&row.status);
        let group_id = match row.group_id {
            Some(g) => Some(
                uuid::Uuid::parse_str(&g)
                    .map_err(|e| crate::error::AppError::Other(format!("invalid group UUID: {e}")))?,
            ),
            None => None,
        };

        Ok(Job {
            id,
            repo_id: row.repo_id,
            revision: row.revision,
            file: row.file,
            dest_path: PathBuf::from(row.dest_path),
            total_bytes: row.total_bytes as u64,
            downloaded: row.downloaded as u64,
            chunks,
            status,
            error: row.error,
            expected_sha256: row.expected_sha256,
            group_id,
            created_at: row.created_at,
            updated_at: row.updated_at,
        })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_chunks_even() {
        let chunks = Job::split_chunks(100, 4);
        assert_eq!(chunks.len(), 4);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 24);
        assert_eq!(chunks[1].start, 25);
        assert_eq!(chunks[1].end, 49);
        assert_eq!(chunks[3].end, 99);
    }

    #[test]
    fn split_chunks_uneven() {
        let chunks = Job::split_chunks(101, 4);
        // last chunk picks up the remainder
        assert_eq!(chunks[3].end, 100);
        // total coverage = sum of (end - start + 1)
        let total: u64 = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(total, 101);
    }

    #[test]
    fn split_chunks_single() {
        let chunks = Job::split_chunks(50, 1);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[0].end, 49);
    }

    #[test]
    fn chunk_count_small_files_single_request() {
        assert_eq!(chunk_count(0), 1);
        assert_eq!(chunk_count(1024), 1);
        assert_eq!(chunk_count(8 * 1024 * 1024 - 1), 1);
    }

    #[test]
    fn chunk_count_targets_64mib() {
        // 8 MiB..64 MiB → one chunk
        assert_eq!(chunk_count(8 * 1024 * 1024), 1);
        assert_eq!(chunk_count(CHUNK_TARGET_BYTES), 1);
        // just over a boundary rounds up
        assert_eq!(chunk_count(CHUNK_TARGET_BYTES + 1), 2);
        // 4 GiB → 64 chunks of ~64 MiB
        assert_eq!(chunk_count(4 * 1024 * 1024 * 1024), 64);
    }

    #[test]
    fn chunk_count_split_chunks_cover_total() {
        let total = 4_999_999_999u64; // ~4.7 GiB, not chunk-aligned
        let chunks = Job::split_chunks(total, chunk_count(total));
        let covered: u64 = chunks.iter().map(|c| c.len()).sum();
        assert_eq!(covered, total);
        assert_eq!(chunks[0].start, 0);
        for w in chunks.windows(2) {
            assert_eq!(w[1].start, w[0].end + 1, "chunks must be adjacent");
        }
    }

    #[test]
    fn chunk_is_done() {
        let c = ChunkState { start: 0, end: 24, completed: 25 };
        assert!(c.is_done());
        let c2 = ChunkState { start: 0, end: 24, completed: 10 };
        assert!(!c2.is_done());
    }

    /// Verify that expected_sha256 survives a JobRow round-trip through TryFrom.
    /// This guards against the field being silently dropped on crash+resume.
    #[test]
    fn expected_sha256_survives_row_conversion() {
        use crate::db::downloads::JobRow;

        let expected_hash = "a".repeat(64); // 64-char hex string
        let row = JobRow {
            id: "550e8400-e29b-41d4-a716-446655440000".to_string(),
            repo_id: "org/repo".to_string(),
            revision: "main".to_string(),
            file: "model.gguf".to_string(),
            dest_path: "/tmp/model.gguf.part".to_string(),
            total_bytes: 1024,
            downloaded: 512,
            status: "paused".to_string(),
            error: None,
            chunks_json: "[]".to_string(),
            expected_sha256: Some(expected_hash.clone()),
            group_id: None,
            created_at: 0,
            updated_at: 0,
        };

        let job = Job::try_from(row).expect("conversion must succeed");
        assert_eq!(
            job.expected_sha256.as_deref(),
            Some(expected_hash.as_str()),
            "expected_sha256 must not be dropped during JobRow → Job conversion"
        );
    }

    /// Verify that None expected_sha256 also survives (no false positive).
    #[test]
    fn expected_sha256_none_survives_row_conversion() {
        use crate::db::downloads::JobRow;

        let row = JobRow {
            id: "550e8400-e29b-41d4-a716-446655440001".to_string(),
            repo_id: "org/repo".to_string(),
            revision: "main".to_string(),
            file: "model.gguf".to_string(),
            dest_path: "/tmp/model.gguf.part".to_string(),
            total_bytes: 1024,
            downloaded: 0,
            status: "queued".to_string(),
            error: None,
            chunks_json: "[]".to_string(),
            expected_sha256: None,
            group_id: None,
            created_at: 0,
            updated_at: 0,
        };

        let job = Job::try_from(row).expect("conversion must succeed");
        assert!(job.expected_sha256.is_none());
    }

    /// group_id must survive the JobRow → Job round-trip (repo downloads
    /// resume after restart by re-associating member jobs with their group).
    #[test]
    fn group_id_survives_row_conversion() {
        use crate::db::downloads::JobRow;

        let gid = "7c9e6679-7425-40de-944b-e07fc1f90ae7";
        let row = JobRow {
            id: "550e8400-e29b-41d4-a716-446655440002".to_string(),
            repo_id: "org/diffusers-repo".to_string(),
            revision: "main".to_string(),
            file: "unet/diffusion_pytorch_model.safetensors".to_string(),
            dest_path: "/tmp/unet/diffusion_pytorch_model.safetensors.part".to_string(),
            total_bytes: 1024,
            downloaded: 0,
            status: "queued".to_string(),
            error: None,
            chunks_json: "[]".to_string(),
            expected_sha256: None,
            group_id: Some(gid.to_string()),
            created_at: 0,
            updated_at: 0,
        };

        let job = Job::try_from(row).expect("conversion must succeed");
        assert_eq!(job.group_id, Some(uuid::Uuid::parse_str(gid).unwrap()));
    }

    /// A malformed group UUID must fail conversion loudly, not silently drop.
    #[test]
    fn invalid_group_id_errors() {
        use crate::db::downloads::JobRow;

        let row = JobRow {
            id: "550e8400-e29b-41d4-a716-446655440003".to_string(),
            repo_id: "org/repo".to_string(),
            revision: "main".to_string(),
            file: "config.json".to_string(),
            dest_path: "/tmp/config.json.part".to_string(),
            total_bytes: 10,
            downloaded: 0,
            status: "queued".to_string(),
            error: None,
            chunks_json: "[]".to_string(),
            expected_sha256: None,
            group_id: Some("not-a-uuid".to_string()),
            created_at: 0,
            updated_at: 0,
        };

        assert!(Job::try_from(row).is_err());
    }
}
