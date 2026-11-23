pub mod nvml;

use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::{broadcast, mpsc, oneshot};

#[allow(unused_imports)]
pub use nvml::{GpuError, GpuProbe, GpuSample, NvmlProbe};

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// Top-level GPU state broadcast on every 1 Hz tick.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GpuStatus {
    /// No NVIDIA driver found — AMD/Intel/headless machines land here.
    NoGpu,
    /// NVML responded successfully; carries the latest readings.
    Available(GpuMetrics),
}

/// Point-in-time GPU readings extracted from an `NvmlProbe` sample.
#[derive(Debug, Clone, Serialize)]
pub struct GpuMetrics {
    pub name: String,
    pub driver_version: String,
    pub cuda_major: u32,
    pub cuda_minor: u32,
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
    /// Unix epoch seconds at sample time.
    pub sampled_at: i64,
}

// ---------------------------------------------------------------------------
// Actor commands
// ---------------------------------------------------------------------------

enum GpuCmd {
    Status(oneshot::Sender<GpuStatus>),
    Subscribe(oneshot::Sender<broadcast::Receiver<GpuStatus>>),
    LatestMetrics(oneshot::Sender<Option<GpuMetrics>>),
    ProcessUsedMb(u32, oneshot::Sender<Option<u64>>),
}

// ---------------------------------------------------------------------------
// GpuHandle — cheap clone, send to any thread
// ---------------------------------------------------------------------------

/// Cloneable handle to the background GPU actor.
#[derive(Clone)]
pub struct GpuHandle {
    tx: mpsc::Sender<GpuCmd>,
}

impl GpuHandle {
    /// Returns the current GPU status (single-shot read, no subscription).
    pub async fn status(&self) -> GpuStatus {
        let (reply_tx, reply_rx) = oneshot::channel();
        // If the actor has stopped, fall back gracefully.
        let _ = self.tx.send(GpuCmd::Status(reply_tx)).await;
        reply_rx.await.unwrap_or(GpuStatus::NoGpu)
    }

    /// Returns the latest `GpuMetrics` if a GPU is present, else `None`.
    pub async fn latest_metrics(&self) -> Option<GpuMetrics> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self.tx.send(GpuCmd::LatestMetrics(reply_tx)).await;
        reply_rx.await.unwrap_or(None)
    }

    /// VRAM currently held by one process in MiB (NVML per-process query),
    /// or `None` when unknown.
    pub async fn process_used_mb(&self, pid: u32) -> Option<u64> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self.tx.send(GpuCmd::ProcessUsedMb(pid, reply_tx)).await;
        reply_rx.await.unwrap_or(None)
    }

    /// Returns a `broadcast::Receiver` that receives a `GpuStatus` on every
    /// poll tick (approximately 1 Hz).
    pub async fn subscribe(&self) -> broadcast::Receiver<GpuStatus> {
        let (reply_tx, reply_rx) = oneshot::channel();
        let _ = self.tx.send(GpuCmd::Subscribe(reply_tx)).await;
        // If the actor has stopped, return a receiver on a throwaway channel.
        reply_rx.await.unwrap_or_else(|_| {
            let (_, rx) = broadcast::channel(1);
            rx
        })
    }
}

// ---------------------------------------------------------------------------
// Actor start
// ---------------------------------------------------------------------------

/// Start the GPU actor using the real NVML probe. Returns immediately.
pub fn start() -> GpuHandle {
    start_with_probe(Arc::new(NvmlProbe::new()))
}

/// Start the GPU actor with a custom probe — used for unit testing.
pub fn start_with_probe(probe: Arc<dyn GpuProbe>) -> GpuHandle {
    let (tx, rx) = mpsc::channel::<GpuCmd>(16);
    let (bcast_tx, _) = broadcast::channel::<GpuStatus>(8);

    // The poll-loop owns an Arc<Mutex<GpuStatus>> as the sole writer.
    // The command-loop receives a clone of that Arc for reads.
    let latest: Arc<Mutex<GpuStatus>> = Arc::new(Mutex::new(GpuStatus::NoGpu));

    spawn_poll_loop(Arc::clone(&probe), bcast_tx.clone(), Arc::clone(&latest));
    spawn_cmd_loop(rx, bcast_tx, latest, probe);

    GpuHandle { tx }
}

// ---------------------------------------------------------------------------
// Internal: poll loop (single writer)
// ---------------------------------------------------------------------------

fn spawn_poll_loop(
    probe: Arc<dyn GpuProbe>,
    bcast_tx: broadcast::Sender<GpuStatus>,
    latest: Arc<Mutex<GpuStatus>>,
) {
    tauri::async_runtime::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(1));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            interval.tick().await;

            let status = match probe.read_primary() {
                Ok(s) => GpuStatus::Available(GpuMetrics {
                    name: s.name,
                    driver_version: s.driver_version,
                    cuda_major: s.cuda_capability.0,
                    cuda_minor: s.cuda_capability.1,
                    vram_total_mb: s.vram_total_mb,
                    vram_used_mb: s.vram_used_mb,
                    vram_free_mb: s.vram_free_mb,
                    sampled_at: chrono::Utc::now().timestamp(),
                }),
                Err(GpuError::NoDriver) => GpuStatus::NoGpu,
                Err(e) => {
                    // Transient probe error — log and keep the previous status.
                    tracing::warn!("gpu probe error: {e}");
                    continue;
                }
            };

            // Update the cached value (only this task ever writes).
            *latest.lock() = status.clone();
            // Broadcast to all subscribers; ignore "no receivers" error.
            let _ = bcast_tx.send(status);
        }
    });
}

// ---------------------------------------------------------------------------
// Internal: command loop (read-only access to latest)
// ---------------------------------------------------------------------------

fn spawn_cmd_loop(
    mut rx: mpsc::Receiver<GpuCmd>,
    bcast_tx: broadcast::Sender<GpuStatus>,
    latest: Arc<Mutex<GpuStatus>>,
    probe: Arc<dyn GpuProbe>,
) {
    tauri::async_runtime::spawn(async move {
        while let Some(cmd) = rx.recv().await {
            match cmd {
                GpuCmd::Status(reply) => {
                    let _ = reply.send(latest.lock().clone());
                }
                GpuCmd::LatestMetrics(reply) => {
                    let v = match &*latest.lock() {
                        GpuStatus::Available(m) => Some(m.clone()),
                        GpuStatus::NoGpu => None,
                    };
                    let _ = reply.send(v);
                }
                GpuCmd::Subscribe(reply) => {
                    let _ = reply.send(bcast_tx.subscribe());
                }
                GpuCmd::ProcessUsedMb(pid, reply) => {
                    let _ = reply.send(probe.process_used_mb(pid));
                }
            }
        }
    });
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    // -----------------------------------------------------------------------
    // Stub probes
    // -----------------------------------------------------------------------

    /// Always returns the same fixed sample.
    struct StubProbe {
        sample: GpuSample,
    }

    impl StubProbe {
        fn new() -> Self {
            Self {
                sample: GpuSample {
                    name: "Test GPU".to_string(),
                    driver_version: "535.00".to_string(),
                    cuda_capability: (8, 6),
                    vram_total_mb: 8192,
                    vram_used_mb: 1024,
                    vram_free_mb: 7168,
                },
            }
        }
    }

    impl GpuProbe for StubProbe {
        fn read_primary(&self) -> Result<GpuSample, GpuError> {
            Ok(self.sample.clone())
        }
    }

    /// Always returns NoDriver.
    struct NoDriverProbe;

    impl GpuProbe for NoDriverProbe {
        fn read_primary(&self) -> Result<GpuSample, GpuError> {
            Err(GpuError::NoDriver)
        }
    }

    /// Returns an error on first call, then succeeds.
    struct TransientErrProbe {
        failed_once: Arc<AtomicBool>,
        sample: GpuSample,
    }

    impl TransientErrProbe {
        fn new() -> Self {
            Self {
                failed_once: Arc::new(AtomicBool::new(false)),
                sample: GpuSample {
                    name: "Transient GPU".to_string(),
                    driver_version: "535.00".to_string(),
                    cuda_capability: (7, 5),
                    vram_total_mb: 4096,
                    vram_used_mb: 512,
                    vram_free_mb: 3584,
                },
            }
        }
    }

    impl GpuProbe for TransientErrProbe {
        fn read_primary(&self) -> Result<GpuSample, GpuError> {
            if self.failed_once.swap(true, Ordering::SeqCst) {
                Ok(self.sample.clone())
            } else {
                Err(GpuError::Nvml("transient nvml error".to_string()))
            }
        }
    }

    // -----------------------------------------------------------------------
    // Tests
    // -----------------------------------------------------------------------

    /// After ~1.2 s the handle should report Available with the stub values.
    #[tokio::test]
    async fn test_available_after_one_tick() {
        let handle = start_with_probe(Arc::new(StubProbe::new()));
        tokio::time::sleep(Duration::from_millis(1200)).await;

        match handle.status().await {
            GpuStatus::Available(m) => {
                assert_eq!(m.name, "Test GPU");
                assert_eq!(m.driver_version, "535.00");
                assert_eq!(m.cuda_major, 8);
                assert_eq!(m.cuda_minor, 6);
                assert_eq!(m.vram_total_mb, 8192);
                assert_eq!(m.vram_used_mb, 1024);
                assert_eq!(m.vram_free_mb, 7168);
            }
            GpuStatus::NoGpu => panic!("expected Available, got NoGpu"),
        }
    }

    /// latest_metrics() should return Some(...) matching stub values.
    #[tokio::test]
    async fn test_latest_metrics() {
        let handle = start_with_probe(Arc::new(StubProbe::new()));
        tokio::time::sleep(Duration::from_millis(1200)).await;

        let m = handle.latest_metrics().await.expect("expected Some(GpuMetrics)");
        assert_eq!(m.name, "Test GPU");
        assert_eq!(m.vram_total_mb, 8192);
    }

    /// subscribe() returns a receiver that gets at least one value within 1.2 s.
    #[tokio::test]
    async fn test_subscribe_receives_value() {
        let handle = start_with_probe(Arc::new(StubProbe::new()));
        let mut rx = handle.subscribe().await;

        let received = tokio::time::timeout(Duration::from_millis(1200), rx.recv()).await;
        let status = received
            .expect("timeout waiting for broadcast")
            .expect("broadcast channel closed");

        assert!(
            matches!(status, GpuStatus::Available(_)),
            "expected Available, got {status:?}"
        );
    }

    /// NoDriverProbe → handle always reports NoGpu, no panic.
    #[tokio::test]
    async fn test_no_driver_fallback() {
        let handle = start_with_probe(Arc::new(NoDriverProbe));
        tokio::time::sleep(Duration::from_millis(1200)).await;

        assert!(
            matches!(handle.status().await, GpuStatus::NoGpu),
            "expected NoGpu for no-driver probe"
        );
        assert!(
            handle.latest_metrics().await.is_none(),
            "expected None metrics for no-driver probe"
        );
    }

    /// Transient error keeps the previous cached status rather than flipping
    /// to NoGpu, and eventually recovers to Available.
    #[tokio::test]
    async fn test_transient_error_keeps_last_status() {
        let handle = start_with_probe(Arc::new(TransientErrProbe::new()));
        // Wait long enough for at least two ticks: first fails (keep NoGpu),
        // second succeeds (Available).
        tokio::time::sleep(Duration::from_millis(2200)).await;

        assert!(
            matches!(handle.status().await, GpuStatus::Available(_)),
            "expected Available after transient error recovers"
        );
    }
}
