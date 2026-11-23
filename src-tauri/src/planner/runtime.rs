//! Live actuals for the loaded text model: measured decode / prompt tok/s
//! (llama-server `/metrics`), the sidecar's VRAM (NVML per-process) and RAM
//! (working set), shown next to the plan's estimate.
//!
//! One background task samples once a second while a model is ready and
//! publishes the latest [`RuntimeStats`] on a watch channel. Completed
//! generation windows are folded into the planner store's calibration, so
//! later estimates for the same model and tier mix move toward reality.

use std::time::Duration;

use serde::Serialize;
use tokio::sync::watch;

use super::speed::{SpeedLabel, TierMix};
use super::store::{CalibrationSample, PlannerHandle};
use super::LoadPlan;
use crate::engine::{EngineManagerHandle, EngineStatus};
use crate::gpu::GpuHandle;
use crate::orchestrator::OrchestratorHandle;

const POLL: Duration = Duration::from_secs(1);
/// Minimum tokens in a window before it counts as a calibration sample:
/// short replies and short prompts are dominated by fixed latency.
const CAL_MIN_DECODE_TOKENS: f64 = 64.0;
const CAL_MIN_PROMPT_TOKENS: f64 = 512.0;

/// The plan's estimate, repeated next to the actuals.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct EstimateSummary {
    pub decode_tps: f64,
    pub prompt_tps: f64,
    pub label: SpeedLabel,
    pub tier_mix: TierMix,
    pub vram_bytes: u64,
    pub ram_bytes: u64,
    pub ssd_bytes: u64,
}

impl EstimateSummary {
    fn from_plan(p: &LoadPlan) -> Self {
        Self {
            decode_tps: p.speed.decode_tps,
            prompt_tps: p.speed.prompt_tps,
            label: p.speed.label,
            tier_mix: p.tier_mix,
            vram_bytes: p.tiers.vram_total(),
            ram_bytes: p.tiers.ram,
            ssd_bytes: p.tiers.ssd,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct RuntimeStats {
    /// Loaded text model, `None` when nothing is loaded.
    pub model_id: Option<String>,
    /// Last measured decode rate (tokens/s while generating), if any request
    /// has run since load.
    pub decode_tps: Option<f64>,
    /// Last measured prompt-processing rate.
    pub prompt_tps: Option<f64>,
    pub decode_label: Option<SpeedLabel>,
    /// True while llama-server is processing a request.
    pub busy: bool,
    pub tokens_generated: u64,
    /// VRAM held by the llama-server process (NVML), MiB.
    pub vram_mb: Option<u64>,
    /// Working set of the llama-server process (includes mapped weight pages
    /// currently resident), bytes.
    pub ram_bytes: Option<u64>,
    /// The plan this model was loaded with (absent for legacy loads).
    pub estimate: Option<EstimateSummary>,
    pub sampled_at: i64,
}

/// Counters from llama-server's Prometheus endpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MetricsSnapshot {
    pub prompt_tokens: f64,
    pub prompt_seconds: f64,
    pub predicted_tokens: f64,
    pub predicted_seconds: f64,
    pub requests_processing: f64,
}

/// Parse the `llamacpp:*` counters out of a `/metrics` body.
pub fn parse_metrics(body: &str) -> Option<MetricsSnapshot> {
    let mut m = MetricsSnapshot::default();
    let mut seen = false;
    for line in body.lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (Some(name), Some(value)) = (it.next(), it.next()) else { continue };
        let Ok(v) = value.parse::<f64>() else { continue };
        let slot = match name {
            "llamacpp:prompt_tokens_total" => &mut m.prompt_tokens,
            "llamacpp:prompt_seconds_total" => &mut m.prompt_seconds,
            "llamacpp:tokens_predicted_total" => &mut m.predicted_tokens,
            "llamacpp:tokens_predicted_seconds_total" => &mut m.predicted_seconds,
            "llamacpp:requests_processing" => &mut m.requests_processing,
            _ => continue,
        };
        *slot = v;
        seen = true;
    }
    seen.then_some(m)
}

/// Accumulates counter deltas into rates and calibration windows.
#[derive(Debug, Default)]
pub struct RateTracker {
    prev: Option<MetricsSnapshot>,
    pub decode_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
    // Calibration window since the last sample.
    win_decode_tokens: f64,
    win_decode_secs: f64,
    win_prompt_tokens: f64,
    win_prompt_secs: f64,
}

/// A finished window, ready to record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowRates {
    pub decode_tps: Option<f64>,
    pub prompt_tps: Option<f64>,
}

impl RateTracker {
    /// Feed a snapshot; returns a calibration window when one completed
    /// (llama-server went idle with enough tokens accumulated).
    pub fn update(&mut self, s: MetricsSnapshot) -> Option<WindowRates> {
        let prev = self.prev.replace(s);
        let p = prev?;
        // Counter reset (server restarted): start over.
        if s.predicted_tokens < p.predicted_tokens || s.prompt_tokens < p.prompt_tokens {
            *self = RateTracker { prev: Some(s), ..Default::default() };
            return None;
        }
        let dt = s.predicted_tokens - p.predicted_tokens;
        let ds = s.predicted_seconds - p.predicted_seconds;
        if dt > 0.0 && ds > 0.0 {
            self.decode_tps = Some(dt / ds);
            self.win_decode_tokens += dt;
            self.win_decode_secs += ds;
        }
        let pt = s.prompt_tokens - p.prompt_tokens;
        let ps = s.prompt_seconds - p.prompt_seconds;
        if pt > 0.0 && ps > 0.0 {
            self.prompt_tps = Some(pt / ps);
            self.win_prompt_tokens += pt;
            self.win_prompt_secs += ps;
        }
        if s.requests_processing > 0.0 {
            return None;
        }
        let decode = (self.win_decode_tokens >= CAL_MIN_DECODE_TOKENS)
            .then(|| self.win_decode_tokens / self.win_decode_secs);
        let prompt = (self.win_prompt_tokens >= CAL_MIN_PROMPT_TOKENS)
            .then(|| self.win_prompt_tokens / self.win_prompt_secs);
        if decode.is_some() {
            self.win_decode_tokens = 0.0;
            self.win_decode_secs = 0.0;
        }
        if prompt.is_some() {
            self.win_prompt_tokens = 0.0;
            self.win_prompt_secs = 0.0;
        }
        (decode.is_some() || prompt.is_some()).then_some(WindowRates { decode_tps: decode, prompt_tps: prompt })
    }
}

fn process_ram_bytes(sys: &mut sysinfo::System, pid: u32) -> Option<u64> {
    let pid = sysinfo::Pid::from_u32(pid);
    sys.refresh_processes(sysinfo::ProcessesToUpdate::Some(&[pid]), true);
    sys.process(pid).map(|p| p.memory())
}

/// Start the collector. Returns the receiver side the
/// `subscribe_runtime_stats` command clones.
pub fn start(
    http: reqwest::Client,
    engine: EngineManagerHandle,
    gpu: GpuHandle,
    orchestrator: OrchestratorHandle,
    store: PlannerHandle,
) -> watch::Receiver<RuntimeStats> {
    let (tx, rx) = watch::channel(RuntimeStats::default());
    tauri::async_runtime::spawn(async move {
        let mut sys = sysinfo::System::new().ok();
        let mut tracker = RateTracker::default();
        let mut current: Option<(String, u16)> = None;
        let mut tokens_generated = 0u64;
        loop {
            tokio::time::sleep(POLL).await;
            let (model_id, port) = match engine.status().await {
                EngineStatus::Ready { model_id, port, .. } => (model_id, port),
                _ => {
                    if current.take().is_some() {
                        tracker = RateTracker::default();
                        let _ = tx.send(RuntimeStats { sampled_at: now(), ..Default::default() });
                    }
                    continue;
                }
            };
            if current.as_ref() != Some(&(model_id.clone(), port)) {
                current = Some((model_id.clone(), port));
                tracker = RateTracker::default();
                tokens_generated = 0;
            }
            let plan = orchestrator.current_text_plan().await.filter(|(id, _)| *id == model_id).map(|(_, p)| p);

            let mut busy = false;
            if let Ok(resp) = http
                .get(format!("http://127.0.0.1:{port}/metrics"))
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                if let Ok(body) = resp.text().await {
                    if let Some(snap) = parse_metrics(&body) {
                        busy = snap.requests_processing > 0.0;
                        tokens_generated = snap.predicted_tokens as u64;
                        if let Some(win) = tracker.update(snap) {
                            if let Some(p) = &plan {
                                store
                                    .record(CalibrationSample {
                                        model_id: if p.engine_kind == super::EngineKind::SsdStream { format!("{model_id}:native-ssd") } else { model_id.clone() },
                                        mix: p.calibration_key(),
                                        raw_decode: p.speed.raw_decode_tps,
                                        actual_decode: win.decode_tps,
                                        raw_prompt: p.speed.raw_prompt_tps,
                                        actual_prompt: win.prompt_tps,
                                    })
                                    .await;
                            }
                        }
                    }
                }
            }

            let pid = engine.server_pid().await;
            let vram_mb = match pid {
                Some(pid) => gpu.process_used_mb(pid).await,
                None => None,
            };
            let ram_bytes = match (pid, sys.as_mut()) {
                (Some(pid), Some(sys)) => process_ram_bytes(sys, pid),
                _ => None,
            };
            let stats = RuntimeStats {
                model_id: Some(model_id),
                decode_tps: tracker.decode_tps,
                prompt_tps: tracker.prompt_tps,
                decode_label: tracker.decode_tps.map(SpeedLabel::for_tps),
                busy,
                tokens_generated,
                vram_mb,
                ram_bytes,
                estimate: plan.as_ref().map(EstimateSummary::from_plan),
                sampled_at: now(),
            };
            tx.send_if_modified(|old| {
                // Timestamps always change; skip identical idle samples.
                let mut a = old.clone();
                a.sampled_at = stats.sampled_at;
                let changed = a != stats;
                if changed {
                    *old = stats;
                }
                changed
            });
        }
    });
    rx
}

fn now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as i64)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &str = "\
# HELP llamacpp:prompt_tokens_total Number of prompt tokens processed.
# TYPE llamacpp:prompt_tokens_total counter
llamacpp:prompt_tokens_total 2048
llamacpp:prompt_seconds_total 1.6
llamacpp:tokens_predicted_total 300
llamacpp:tokens_predicted_seconds_total 2.5
llamacpp:n_decode_total 310
llamacpp:requests_processing 0
";

    #[test]
    fn parses_prometheus_counters() {
        let m = parse_metrics(BODY).unwrap();
        assert_eq!(m.prompt_tokens, 2048.0);
        assert_eq!(m.predicted_seconds, 2.5);
        assert_eq!(m.requests_processing, 0.0);
        assert!(parse_metrics("not metrics").is_none());
    }

    fn snap(pt: f64, ps: f64, dt: f64, ds: f64, busy: bool) -> MetricsSnapshot {
        MetricsSnapshot {
            prompt_tokens: pt,
            prompt_seconds: ps,
            predicted_tokens: dt,
            predicted_seconds: ds,
            requests_processing: if busy { 1.0 } else { 0.0 },
        }
    }

    #[test]
    fn rates_and_calibration_windows() {
        let mut t = RateTracker::default();
        assert!(t.update(snap(0.0, 0.0, 0.0, 0.0, false)).is_none());
        // Mid-request: prompt done (1000 tok in 0.5 s), 40 tokens decoded in 1 s.
        assert!(t.update(snap(1000.0, 0.5, 40.0, 1.0, true)).is_none());
        assert_eq!(t.prompt_tps, Some(2000.0));
        assert_eq!(t.decode_tps, Some(40.0));
        // Request finishes: 60 more tokens in 1 s → window 100 tok / 2 s.
        let w = t.update(snap(1000.0, 0.5, 100.0, 2.0, false)).unwrap();
        assert_eq!(w.decode_tps, Some(50.0));
        assert_eq!(w.prompt_tps, Some(2000.0));
        // A short follow-up reply stays below the threshold.
        assert!(t.update(snap(1100.0, 0.6, 110.0, 2.2, false)).is_none());
        // Counter reset after a restart is not a negative rate.
        assert!(t.update(snap(0.0, 0.0, 0.0, 0.0, false)).is_none());
        assert_eq!(t.decode_tps, None);
    }
}
