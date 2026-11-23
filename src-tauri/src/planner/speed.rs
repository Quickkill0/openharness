//! Speed estimator: decode and prompt tokens/s from the bytes each token
//! touches in each memory tier, divided by that tier's bandwidth.
//!
//! Decode (one token, batch 1) is memory-bound, so its time is a sum of
//! transfer times:
//!
//! ```text
//! t = vram_bytes / (vram_bw·η_vram)
//!   + ram→gpu bytes / pcie_bw                       (expert-cache misses held in RAM)
//!   + ssd→gpu bytes · (1/(ssd_bw·η_ssd) + 1/pcie_bw) (misses paged in from the SSD)
//!   + max(cpu bytes from RAM / (ram_bw·η_ram) + cpu bytes from SSD / (ssd_bw·η_ssd),
//!         2·cpu_params / cpu_flops)                 (weights the CPU multiplies)
//!   + layers · per-layer launch overhead
//! decode tok/s = 1 / t
//! ```
//!
//! Routed experts are not touched uniformly: a few are hot. Tiers are filled
//! hottest-first (VRAM cache, then the OS page cache in RAM, then the SSD), and
//! the share of expert reads served by the hottest fraction `f` of the bank is
//! `H(f) = f^0.15`, fitted to the measurements in llama.cpp PR #29887
//! (16% cached → 77% hits, 28% → ~85%). Dense weights are read in full every
//! token, so their split is linear.
//!
//! Prompt processing runs in micro-batches of `ubatch` tokens: GPU compute
//! (`2·active_params·ubatch / gpu_flops`) plus, when weights live on the host,
//! streaming each host-resident bank over PCIe once per micro-batch (llama.cpp
//! offloads large batches to the GPU).
//!
//! Every estimate can be calibrated: measured tok/s per model and tier mix are
//! stored as actual/estimate ratios and blended into later estimates.

use serde::{Deserialize, Serialize};

use super::hardware::HardwareProfile;

// ---------------------------------------------------------------------------
// Labels (the one place the thresholds live)
// ---------------------------------------------------------------------------

/// Decode speed at or above which a model is labelled fast.
pub const FAST_TPS: f64 = 30.0;
/// Decode speed at or above which a model is labelled usable (below: slow).
pub const USABLE_TPS: f64 = 8.0;

/// Plain speed label for the UI. Ordered: `Slow < Usable < Fast`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeedLabel {
    Slow,
    Usable,
    Fast,
}

impl SpeedLabel {
    pub fn for_tps(decode_tps: f64) -> SpeedLabel {
        if decode_tps >= FAST_TPS {
            SpeedLabel::Fast
        } else if decode_tps >= USABLE_TPS {
            SpeedLabel::Usable
        } else {
            SpeedLabel::Slow
        }
    }
}

// ---------------------------------------------------------------------------
// Model constants
// ---------------------------------------------------------------------------

/// Share of spec-sheet VRAM bandwidth a decode step achieves.
const ETA_VRAM: f64 = 0.75;
/// Share of RAM bandwidth CPU matmuls achieve.
const ETA_RAM: f64 = 0.7;
/// Share of sequential SSD speed that page-fault-driven reads achieve.
const ETA_SSD: f64 = 0.5;
/// Kernel launch / sync overhead per decoded layer (seconds). llama.cpp
/// with CUDA graphs spends roughly 40-60 us per layer on small models.
const LAYER_OVERHEAD_S: f64 = 0.00005;
/// Effective GPU FLOP/s per byte/s of memory bandwidth for quantized prompt
/// matmuls (an RTX 4090 at 1008 GB/s → ~100 TFLOP/s).
const GPU_FLOPS_PER_BW: f64 = 100.0;
/// Effective CPU FLOP/s per hardware thread for quantized matmuls.
const CPU_FLOPS_PER_THREAD: f64 = 25e9;
/// Exponent of the expert hit-rate curve `H(f) = f^HIT_EXP`.
const HIT_EXP: f64 = 0.15;

/// Share of expert reads served by the hottest fraction `f` of the bank.
pub fn hit_rate(f: f64) -> f64 {
    if f <= 0.0 {
        0.0
    } else if f >= 1.0 {
        1.0
    } else {
        f.powf(HIT_EXP)
    }
}

// ---------------------------------------------------------------------------
// Traffic
// ---------------------------------------------------------------------------

/// Bytes moved per decoded token, by source, plus what prompt processing
/// streams per micro-batch. Built by the placement planner.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Traffic {
    /// Read from VRAM by GPU kernels.
    pub vram: f64,
    /// Uploaded RAM → GPU (expert-cache misses held in the page cache).
    pub ram_to_gpu: f64,
    /// Paged in from the SSD, then uploaded to the GPU.
    pub ssd_to_gpu: f64,
    /// Read from RAM by CPU matmuls.
    pub cpu_from_ram: f64,
    /// Paged in from the SSD for CPU matmuls.
    pub cpu_from_ssd: f64,
    /// Parameters multiplied on the CPU per token.
    pub cpu_params: f64,
    /// Parameters multiplied per token in total.
    pub active_params: f64,
    /// Decode layers (launch overhead).
    pub layers: u32,
    /// Prompt micro-batch size.
    pub ubatch: u32,
    /// Host-resident weight bytes streamed to the GPU per micro-batch, from
    /// RAM and from the SSD.
    pub stream_ram: f64,
    pub stream_ssd: f64,
    /// True when there is no GPU in the plan (CPU does all the work).
    pub cpu_only: bool,
}

/// What limits decode speed most.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Bottleneck {
    Vram,
    Pcie,
    Ram,
    Ssd,
    Cpu,
    Overhead,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeedEstimate {
    /// Decode tokens/s after calibration.
    pub decode_tps: f64,
    /// Prompt-processing tokens/s after calibration.
    pub prompt_tps: f64,
    pub label: SpeedLabel,
    /// Uncalibrated model output (calibration ratios are measured against it).
    pub raw_decode_tps: f64,
    pub raw_prompt_tps: f64,
    /// Measured samples blended in (0 = pure model).
    pub calibration_samples: u32,
    pub bottleneck: Bottleneck,
}

pub fn estimate(t: &Traffic, hw: &HardwareProfile, cal: Option<&Calibration>) -> SpeedEstimate {
    let bw = &hw.bandwidth;
    let gb = 1e9;
    let vram_bw = (bw.vram_gbps * gb * ETA_VRAM).max(1.0);
    let pcie = (bw.pcie_gbps * gb).max(1.0);
    let ram_bw = (bw.ram_gbps * gb * ETA_RAM).max(1.0);
    let ssd_bw = (bw.ssd_gbps * gb * ETA_SSD).max(1.0);
    let cpu_flops = (hw.cpu_threads.max(1) as f64 * CPU_FLOPS_PER_THREAD).max(1.0);
    let gpu_flops = (bw.vram_gbps * gb * GPU_FLOPS_PER_BW).max(1.0);

    // ── decode ──
    let t_vram = t.vram / vram_bw;
    let t_pcie = (t.ram_to_gpu + t.ssd_to_gpu) / pcie;
    let t_ssd_gpu = t.ssd_to_gpu / ssd_bw;
    let t_cpu_ram = t.cpu_from_ram / ram_bw;
    let t_cpu_ssd = t.cpu_from_ssd / ssd_bw;
    let t_cpu_compute = 2.0 * t.cpu_params / cpu_flops;
    let t_cpu = (t_cpu_ram + t_cpu_ssd).max(t_cpu_compute);
    let t_over = t.layers as f64 * LAYER_OVERHEAD_S;
    let t_decode = (t_vram + t_pcie + t_ssd_gpu + t_cpu + t_over).max(1e-6);

    let parts = [
        (Bottleneck::Vram, t_vram),
        (Bottleneck::Pcie, t_pcie),
        (Bottleneck::Ssd, t_ssd_gpu + t_cpu_ssd),
        (Bottleneck::Ram, t_cpu_ram),
        (Bottleneck::Cpu, (t_cpu_compute - t_cpu_ram - t_cpu_ssd).max(0.0)),
        (Bottleneck::Overhead, t_over),
    ];
    let bottleneck = parts
        .iter()
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map_or(Bottleneck::Vram, |p| p.0);

    // ── prompt ──
    let ub = t.ubatch.max(1) as f64;
    let t_ub = if t.cpu_only {
        2.0 * t.active_params * ub / cpu_flops
            + t.stream_ram / ram_bw
            + t.stream_ssd / ssd_bw
    } else {
        2.0 * t.active_params * ub / gpu_flops
            + t.stream_ram / pcie
            + t.stream_ssd * (1.0 / ssd_bw + 1.0 / pcie)
            + t.layers as f64 * LAYER_OVERHEAD_S
    };
    let raw_prompt = ub / t_ub.max(1e-6);
    let raw_decode = 1.0 / t_decode;

    let (dr, pr, n) = cal.map_or((1.0, 1.0, 0), |c| (c.decode_factor(), c.prompt_factor(), c.samples));
    let decode_tps = raw_decode * dr;
    SpeedEstimate {
        decode_tps,
        prompt_tps: raw_prompt * pr,
        label: SpeedLabel::for_tps(decode_tps),
        raw_decode_tps: raw_decode,
        raw_prompt_tps: raw_prompt,
        calibration_samples: n,
        bottleneck,
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Which tiers a plan touches; calibration is kept per mix because the error
/// of the model differs between them. Ordered fastest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierMix {
    /// Everything in VRAM.
    Vram,
    /// Some weights in RAM (experts on the CPU, partial offload).
    Ram,
    /// Some weights paged from the SSD.
    Ssd,
}

/// Measured actual/estimate ratios for one model and tier mix.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Calibration {
    /// Running actual/estimate ratio for decode (1.0 = model was right).
    pub decode_ratio: f64,
    pub decode_samples: u32,
    pub prompt_ratio: f64,
    pub prompt_samples: u32,
    /// Total samples, for display.
    pub samples: u32,
}

/// Weight of each new measurement in the running ratio.
const CAL_EMA: f64 = 0.3;
/// Prior strength: with `n` samples the measured ratio gets weight n/(n+K).
const CAL_PRIOR: f64 = 2.0;
/// Ratios outside this band are treated as broken measurements.
const CAL_MIN_RATIO: f64 = 0.05;
const CAL_MAX_RATIO: f64 = 20.0;

fn blend(ratio: f64, n: u32) -> f64 {
    if n == 0 || !ratio.is_finite() || ratio <= 0.0 {
        return 1.0;
    }
    let w = n as f64 / (n as f64 + CAL_PRIOR);
    1.0 + (ratio - 1.0) * w
}

fn update(ratio: &mut f64, n: &mut u32, sample: f64) {
    let sample = sample.clamp(CAL_MIN_RATIO, CAL_MAX_RATIO);
    *ratio = if *n == 0 { sample } else { *ratio + (sample - *ratio) * CAL_EMA };
    *n = n.saturating_add(1);
}

impl Calibration {
    /// Multiplier applied to the raw decode estimate.
    pub fn decode_factor(&self) -> f64 {
        blend(self.decode_ratio, self.decode_samples)
    }

    pub fn prompt_factor(&self) -> f64 {
        blend(self.prompt_ratio, self.prompt_samples)
    }

    /// Fold in a measurement: `raw_*` is the uncalibrated estimate the plan
    /// carried, `actual_*` the measured rate. Either side may be absent.
    pub fn record(
        &mut self,
        raw_decode: f64,
        actual_decode: Option<f64>,
        raw_prompt: f64,
        actual_prompt: Option<f64>,
    ) {
        let mut any = false;
        if let Some(a) = actual_decode.filter(|a| *a > 0.0 && raw_decode > 0.0) {
            update(&mut self.decode_ratio, &mut self.decode_samples, a / raw_decode);
            any = true;
        }
        if let Some(a) = actual_prompt.filter(|a| *a > 0.0 && raw_prompt > 0.0) {
            update(&mut self.prompt_ratio, &mut self.prompt_samples, a / raw_prompt);
            any = true;
        }
        if any {
            self.samples = self.samples.saturating_add(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planner::hardware::{Bandwidths, HardwareProfile};

    pub(crate) fn hw_4090() -> HardwareProfile {
        HardwareProfile {
            gpu: None,
            ram_total_bytes: 64 << 30,
            ram_available_bytes: 56 << 30,
            commit_available_bytes: None,
            storage: None,
            bandwidth: Bandwidths {
                vram_gbps: 1008.0,
                pcie_gbps: 25.0,
                ram_gbps: 60.0,
                ssd_gbps: 6.8,
                ssd_measured: true,
            },
            cpu_threads: 32,
        }
    }

    #[test]
    fn label_thresholds() {
        assert_eq!(SpeedLabel::for_tps(30.0), SpeedLabel::Fast);
        assert_eq!(SpeedLabel::for_tps(29.99), SpeedLabel::Usable);
        assert_eq!(SpeedLabel::for_tps(8.0), SpeedLabel::Usable);
        assert_eq!(SpeedLabel::for_tps(7.99), SpeedLabel::Slow);
        assert_eq!(SpeedLabel::for_tps(0.0), SpeedLabel::Slow);
        assert!(SpeedLabel::Fast > SpeedLabel::Usable && SpeedLabel::Usable > SpeedLabel::Slow);
    }

    #[test]
    fn hit_rate_curve_matches_moe_cache_measurements() {
        assert_eq!(hit_rate(0.0), 0.0);
        assert_eq!(hit_rate(1.0), 1.0);
        assert!((hit_rate(0.16) - 0.76).abs() < 0.02);
        assert!((hit_rate(0.28) - 0.83).abs() < 0.02);
        // Concave: a small cache earns far more than its byte share.
        assert!(hit_rate(0.1) > 0.6);
    }

    #[test]
    fn vram_only_decode_is_bandwidth_over_bytes() {
        let t = Traffic {
            vram: 1.0e9,
            active_params: 2.0e9,
            layers: 0,
            ubatch: 512,
            ..Default::default()
        };
        let e = estimate(&t, &hw_4090(), None);
        let expect = 1008.0 * 0.75; // GB/s ÷ 1 GB per token
        assert!((e.decode_tps - expect).abs() < 1.0, "{}", e.decode_tps);
        assert_eq!(e.bottleneck, Bottleneck::Vram);
        assert_eq!(e.label, SpeedLabel::Fast);
    }

    #[test]
    fn ssd_misses_dominate_when_present() {
        let t = Traffic {
            vram: 8e9,
            ram_to_gpu: 0.5e9,
            ssd_to_gpu: 0.25e9,
            layers: 45,
            ubatch: 2048,
            ..Default::default()
        };
        let e = estimate(&t, &hw_4090(), None);
        assert_eq!(e.bottleneck, Bottleneck::Ssd);
        assert!(e.decode_tps > 5.0 && e.decode_tps < 15.0, "{}", e.decode_tps);
    }

    #[test]
    fn calibration_blends_toward_measurements() {
        let mut c = Calibration::default();
        assert_eq!(c.decode_factor(), 1.0);
        // Model said 100 tok/s, reality 50.
        c.record(100.0, Some(50.0), 1000.0, None);
        assert_eq!(c.decode_samples, 1);
        assert_eq!(c.prompt_samples, 0);
        // One sample, prior 2: 1 + (0.5 - 1)·1/3.
        assert!((c.decode_factor() - (1.0 - 0.5 / 3.0)).abs() < 1e-9);
        assert_eq!(c.prompt_factor(), 1.0);
        for _ in 0..30 {
            c.record(100.0, Some(50.0), 1000.0, Some(2000.0));
        }
        assert!((c.decode_factor() - 0.5).abs() < 0.05, "{}", c.decode_factor());
        assert!((c.prompt_factor() - 2.0).abs() < 0.1, "{}", c.prompt_factor());
        // Garbage measurements are clamped, not trusted verbatim.
        let mut g = Calibration::default();
        g.record(100.0, Some(1e9), 1.0, None);
        assert!(g.decode_ratio <= CAL_MAX_RATIO);

        let t = Traffic { vram: 1e9, layers: 0, ubatch: 512, ..Default::default() };
        let raw = estimate(&t, &hw_4090(), None);
        let cal = estimate(&t, &hw_4090(), Some(&c));
        assert!((cal.decode_tps / raw.decode_tps - c.decode_factor()).abs() < 1e-9);
        assert_eq!(cal.raw_decode_tps, raw.raw_decode_tps);
        assert_eq!(cal.calibration_samples, c.samples);
    }
}
