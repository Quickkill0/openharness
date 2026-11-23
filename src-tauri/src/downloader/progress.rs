use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;
use uuid::Uuid;

use super::job::JobStatus;

/// Phase of a running job. Lets the UI distinguish bytes-flowing from the
/// post-transfer SHA256 / GGUF-parse / DB-insert work, which has zero bytes
/// per second but is not actually frozen.
#[derive(Debug, Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DownloadPhase {
    Downloading,
    Verifying,
    Finalizing,
}

impl Default for DownloadPhase {
    fn default() -> Self {
        DownloadPhase::Downloading
    }
}

/// Progress snapshot broadcast to the frontend for a single download job.
#[derive(Debug, Serialize, Clone)]
pub struct DownloadProgress {
    pub id: Uuid,
    pub status: JobStatus,
    /// Current failure reason; None also clears an error from a previous attempt.
    pub error: Option<String>,
    pub downloaded: u64,
    pub total: u64,
    /// Current transfer speed in bytes per second.
    pub bytes_per_sec: f64,
    /// Estimated seconds until completion. `None` when speed is zero or unknown.
    pub eta_secs: Option<u64>,
    /// Which stage the job is in. Defaults to Downloading; flips to Verifying
    /// during SHA256 hashing and Finalizing during catalog insert.
    #[serde(default)]
    pub phase: DownloadPhase,
}

// ---------------------------------------------------------------------------
// Sliding-window speed tracker
// ---------------------------------------------------------------------------

/// Tracks bytes-per-second over a rolling 3-second window.
pub struct SpeedTracker {
    /// (timestamp, cumulative_bytes) samples within the window.
    samples: VecDeque<(Instant, u64)>,
    window: Duration,
}

impl SpeedTracker {
    pub fn new() -> Self {
        Self {
            samples: VecDeque::new(),
            window: Duration::from_secs(3),
        }
    }

    /// Record a new cumulative byte count.
    pub fn record(&mut self, cumulative: u64) {
        let now = Instant::now();
        self.samples.push_back((now, cumulative));
        // Evict samples older than the window.
        let cutoff = now - self.window;
        while self.samples.front().map(|(t, _)| *t < cutoff).unwrap_or(false) {
            self.samples.pop_front();
        }
    }

    /// Compute current speed in bytes/sec. Returns 0.0 when fewer than two samples.
    pub fn bytes_per_sec(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let (t0, b0) = self.samples.front().unwrap();
        let (t1, b1) = self.samples.back().unwrap();
        let dt = t1.duration_since(*t0).as_secs_f64();
        if dt < 1e-6 {
            return 0.0;
        }
        (b1.saturating_sub(*b0)) as f64 / dt
    }

    /// Compute ETA given remaining bytes.
    pub fn eta_secs(&self, remaining: u64) -> Option<u64> {
        let speed = self.bytes_per_sec();
        if speed < 1.0 {
            return None;
        }
        Some((remaining as f64 / speed) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn speed_tracker_zero_on_single_sample() {
        let mut t = SpeedTracker::new();
        t.record(1000);
        assert_eq!(t.bytes_per_sec(), 0.0);
    }

    #[test]
    fn speed_tracker_eta_none_when_zero() {
        let t = SpeedTracker::new();
        assert!(t.eta_secs(1000).is_none());
    }
}
