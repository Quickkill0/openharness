use serde::Serialize;

use crate::models::gguf::GgufMeta;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// All inputs required by the oobabooga VRAM formula.
///
/// Every field is a plain integer — no I/O, no async.  Slice C is responsible
/// for collecting these from `GgufMeta`, the GPU registry, and the user's
/// context-length preference before calling [`estimate_vram_mb`].
#[derive(Debug, Clone, Copy)]
pub struct FitInputs {
    /// File size of the GGUF on disk in mebibytes.
    pub file_size_mb: u64,
    /// Number of transformer blocks (`block_count` in GGUF metadata).
    pub n_layers: u64,
    /// Number of KV attention heads.  For GQA/MQA models use
    /// `attention_head_count_kv`; for standard MHA fall back to
    /// `attention_head_count`. Only consulted on the scalar fallback KV path
    /// (`kv` is `None`).
    pub n_kv_heads: u64,
    /// Per-head dimension.  Derive via `GgufMeta::head_dim()`. Only consulted on
    /// the scalar fallback KV path (`kv` is `None`).
    pub head_dim: u64,
    /// Target context length in tokens (user preference, not model max).
    pub ctx: u32,
    /// Bytes per KV-cache element: 2 = fp16 (default), 1 = q8_0 / q4_0.
    pub kv_dtype_bytes: u8,
    /// Sliding-window-aware KV profile, when the GGUF header describes per-layer
    /// attention (Gemma family and any future mixed-attention arch). `None` →
    /// the KV estimate uses the uniform full-attention scalar formula built from
    /// `n_layers`/`n_kv_heads`/`head_dim`.
    pub kv: Option<KvProfile>,
}

/// Per-layer KV-cache cost, reduced to two context coefficients so the whole
/// thing stays `Copy`. Derived once from the GGUF header (see
/// [`KvProfile::from_meta`]); KV cache at context `C` is then
/// `global_bytes_per_tok * C + swa_bytes_per_tok * min(C, swa_window)`.
#[derive(Debug, Clone, Copy)]
pub struct KvProfile {
    /// KV bytes consumed per context token by full-attention ("global") layers,
    /// summed across all such layers. Scales with the full context.
    pub global_bytes_per_tok: u64,
    /// KV bytes per context token from sliding-window ("local") layers, summed
    /// across them. Billed only up to `swa_window` tokens, so it stops growing
    /// once the context exceeds the window.
    pub swa_bytes_per_tok: u64,
    /// Sliding-window size in tokens; SWA-layer KV caps here. `u32::MAX` when the
    /// window size is unknown (then SWA layers are billed at full context — a
    /// safe upper bound).
    pub swa_window: u32,
}

impl KvProfile {
    /// Build a per-layer KV profile from a parsed GGUF header. Returns `None`
    /// when the header lacks the per-layer detail to do better than the uniform
    /// scalar formula (no `sliding_window_pattern` AND no per-layer
    /// `head_count_kv` array), in which case the caller falls back to it.
    ///
    /// Fully generic: it reads the standard `attention.sliding_window_pattern`,
    /// `attention.head_count_kv` (array form), `attention.sliding_window`, and
    /// the `key_length`/`value_length` (+ `_swa`) keys — no architecture is
    /// named here.
    pub fn from_meta(meta: &GgufMeta, kv_dtype_bytes: u8) -> Option<KvProfile> {
        let n = meta.block_count? as usize;
        let pattern = meta.sliding_window_pattern.as_deref();
        let per_layer_heads = meta.attention_head_count_kv_layers.as_deref();

        // Without either per-layer signal we can't beat the scalar formula.
        if pattern.is_none() && per_layer_heads.is_none() {
            return None;
        }

        let k_global = meta.attention_key_length.or_else(|| meta.head_dim())?;
        let v_global = meta
            .attention_value_length
            .or(meta.attention_key_length)
            .or_else(|| meta.head_dim())?;
        let k_swa = meta.attention_key_length_swa.unwrap_or(k_global);
        let v_swa = meta.attention_value_length_swa.unwrap_or(v_global);
        // No window size known → don't cap (treat SWA layers as full attention).
        let window = meta
            .attention_sliding_window
            .map_or(u32::MAX, |w| w.min(u32::MAX as u64) as u32);
        // Per-layer head count falls back to the scalar (max) when the array is
        // absent or short.
        let scalar_heads = meta
            .attention_head_count_kv
            .or(meta.attention_head_count)
            .unwrap_or(0);
        let dt = kv_dtype_bytes as u64;

        let mut global_bytes_per_tok: u64 = 0;
        let mut swa_bytes_per_tok: u64 = 0;
        for layer in 0..n {
            let is_swa = pattern.map_or(false, |p| p.get(layer).copied().unwrap_or(false));
            let heads = per_layer_heads
                .and_then(|a| a.get(layer).copied())
                .unwrap_or(scalar_heads);
            if is_swa {
                swa_bytes_per_tok =
                    swa_bytes_per_tok.saturating_add(heads.saturating_mul((k_swa + v_swa) * dt));
            } else {
                global_bytes_per_tok = global_bytes_per_tok
                    .saturating_add(heads.saturating_mul((k_global + v_global) * dt));
            }
        }

        Some(KvProfile {
            global_bytes_per_tok,
            swa_bytes_per_tok,
            swa_window: window,
        })
    }
}

impl FitInputs {
    /// Assemble fit inputs from a parsed GGUF header, deriving the SWA-aware KV
    /// profile when the header supports it. Centralizes the metadata-to-inputs
    /// mapping so every call site (estimate command, reclaim estimate,
    /// orchestrator load gate) agrees.
    pub fn from_meta(meta: &GgufMeta, file_size_mb: u64, ctx: u32, kv_dtype_bytes: u8) -> Self {
        FitInputs {
            file_size_mb,
            n_layers: meta.block_count.unwrap_or(32),
            n_kv_heads: meta
                .attention_head_count_kv
                .or(meta.attention_head_count)
                .unwrap_or(32),
            head_dim: meta.head_dim().unwrap_or(128),
            ctx,
            kv_dtype_bytes,
            kv: KvProfile::from_meta(meta, kv_dtype_bytes),
        }
    }
}

// ---------------------------------------------------------------------------
// Formula
// ---------------------------------------------------------------------------

/// Estimate total VRAM required in MiB.
///
/// Based on the oobabooga formula validated against 19 000+ measurements
/// (median absolute error ≈ 365 MiB):
///
/// ```text
/// VRAM = file_size_mb
///      + (2 × L × H_kv × D × C × dt) / 1024²   -- KV cache
///      + 1517                                    -- backend / CUDA overhead
///      + 577                                     -- safety margin
/// ```
///
/// `u128` is used for the KV-cache intermediate to avoid overflow at
/// 70 B+ parameter models with large context windows (e.g. 70 B × 128 k ctx
/// overflows u64).
const BACKEND_OVERHEAD_MB: u64 = 1517;
const SAFETY_MARGIN_MB: u64 = 577;

/// Fixed VRAM reserve (backend overhead + safety margin) — the part of the
/// estimate that does not scale with offloaded layers. The orchestrator uses
/// this to project partial-offload footprints.
pub const FIXED_OVERHEAD_MB: u64 = BACKEND_OVERHEAD_MB + SAFETY_MARGIN_MB;

/// KV-cache size in MiB for these inputs at the requested context length.
///
/// Two paths, both with a `u128` intermediate to avoid overflow at 70 B+ models
/// with large context windows:
///
///   * **SWA-aware** (`i.kv` is `Some`): sum of the per-layer profile —
///     `global_bytes_per_tok × C + swa_bytes_per_tok × min(C, swa_window)`.
///     Sliding-window (local) layers stop growing once the context exceeds the
///     window, so mixed-attention models (the Gemma family) are sized
///     accurately instead of billing every layer at full context.
///   * **Scalar fallback** (`i.kv` is `None`): the uniform full-attention
///     formula `2 × L × H_kv × D × C × dt` for models whose header doesn't
///     describe per-layer attention.
pub fn kv_cache_mb(i: &FitInputs) -> u64 {
    let kv_bytes_total: u128 = match i.kv {
        Some(p) => {
            let global = p.global_bytes_per_tok as u128 * i.ctx as u128;
            let swa_ctx = (i.ctx as u128).min(p.swa_window as u128);
            let swa = p.swa_bytes_per_tok as u128 * swa_ctx;
            global + swa
        }
        None => {
            2u128
                * i.n_layers as u128
                * i.n_kv_heads as u128
                * i.head_dim as u128
                * i.ctx as u128
                * i.kv_dtype_bytes as u128
        }
    };

    (kv_bytes_total / (1024 * 1024)) as u64
}

pub fn estimate_vram_mb(i: &FitInputs) -> u64 {
    i.file_size_mb
        .saturating_add(kv_cache_mb(i))
        .saturating_add(BACKEND_OVERHEAD_MB)
        .saturating_add(SAFETY_MARGIN_MB)
}

// ---------------------------------------------------------------------------
// Smart -ngl (M7 §7.6)
// ---------------------------------------------------------------------------

/// Choose `-ngl` for a model given free VRAM: `-1` (all layers on GPU) when
/// the full estimate fits, otherwise the largest partial offload that does.
/// The fixed backend overhead + safety margin are reserved up front; weights
/// and KV cache scale ~linearly with offloaded layers.
pub fn pick_ngl(i: &FitInputs, free_mb: u64) -> i32 {
    let total = estimate_vram_mb(i);
    if total <= free_mb {
        return -1;
    }
    let fixed = BACKEND_OVERHEAD_MB + SAFETY_MARGIN_MB;
    if i.n_layers == 0 || free_mb <= fixed {
        return 0;
    }
    let per_layer = ((total - fixed) / i.n_layers).max(1);
    let layers = (free_mb - fixed) / per_layer;
    layers.min(i.n_layers) as i32
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

/// Qualitative assessment of whether a model fits in available VRAM.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FitVerdict {
    /// Estimated VRAM ≤ free VRAM with more than 1 GiB headroom.
    Fits,
    /// Estimated VRAM ≤ free VRAM but within 1 GiB (1024 MiB) of the limit.
    /// The model may load but could OOM during inference depending on batch size.
    Tight,
    /// Estimated VRAM exceeds free VRAM — likely to OOM.
    WontFit,
}

/// Classify a `(estimated_mb, free_mb)` pair.
///
/// The `Tight` band is defined as: estimated fits but `free_mb − estimated_mb < 1024`.
pub fn verdict(estimated_mb: u64, free_mb: u64) -> FitVerdict {
    if estimated_mb > free_mb {
        FitVerdict::WontFit
    } else if free_mb - estimated_mb < 1024 {
        FitVerdict::Tight
    } else {
        FitVerdict::Fits
    }
}

// ---------------------------------------------------------------------------
// Composite result
// ---------------------------------------------------------------------------

/// Full fit estimate returned by the Tauri command (Slice C).
#[derive(Debug, Clone, Serialize)]
pub struct FitEstimate {
    /// Installed text admission receipt; absent for legacy non-text estimates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan: Option<Box<crate::planner::LoadPlan>>,
    /// Planned graphics footprint (or host footprint without a GPU), MiB.
    /// Legacy non-text estimates retain their weights/overhead formula.
    pub estimated_mb: u64,
    /// Context cache on the estimated tier, MiB (native GPU KV is zero).
    /// Native host context is detailed in `plan.context`.
    pub kv_at_ctx_mb: u64,
    /// Free VRAM reported by the GPU at query time, in MiB.
    pub free_mb: u64,
    /// VRAM currently held by the loaded text engine in MiB. Loading a model
    /// replaces the one in the engine slot, so this much frees first — the
    /// verdict compares against `free_mb + reclaimable_mb`. Without this, the
    /// already-loaded model reads as "won't fit" against the VRAM it itself
    /// occupies.
    pub reclaimable_mb: u64,
    /// Qualitative verdict.
    pub verdict: FitVerdict,
    /// `true` when the estimate is for a discrete GPU; `false` for CPU/RAM path.
    pub gpu: bool,
}

impl FitEstimate {
    pub fn from_plan(plan: crate::planner::LoadPlan, gpu: bool, free_mb: u64, reclaimable_mb: u64) -> Self {
        let mib = 1024 * 1024;
        Self {
            estimated_mb: if gpu { plan.tiers.vram_total().div_ceil(mib) } else { plan.tiers.ram.div_ceil(mib) },
            kv_at_ctx_mb: if gpu { plan.tiers.vram_kv.div_ceil(mib) } else { plan.context.cache_bytes().div_ceil(mib) },
            verdict: if plan.feasible { FitVerdict::Fits } else { FitVerdict::WontFit },
            plan: Some(Box::new(plan)), free_mb, reclaimable_mb, gpu,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Llama-3-8B Q4_K_M baseline: ~4.7 GB file, 32 layers, 8 KV heads (GQA),
    /// head_dim 128, 4 k context, fp16 KV cache.
    fn llama3_8b_q4km_ctx4k() -> FitInputs {
        FitInputs {
            file_size_mb: 4700,
            n_layers: 32,
            n_kv_heads: 8,
            head_dim: 128,
            ctx: 4096,
            kv_dtype_bytes: 2,
            kv: None,
        }
    }

    #[test]
    fn estimate_llama3_8b_q4km_in_range() {
        let est = estimate_vram_mb(&llama3_8b_q4km_ctx4k());
        // Expected: ~6–8 GB range for this quant + ctx combo.
        assert!(est > 6000 && est < 8500, "got {} MB", est);
    }

    #[test]
    fn verdict_wont_fit_when_over() {
        assert_eq!(verdict(10_000, 8_000), FitVerdict::WontFit);
    }

    #[test]
    fn verdict_tight_when_close() {
        assert_eq!(verdict(7_500, 8_000), FitVerdict::Tight);
    }

    #[test]
    fn verdict_fits_when_comfortable() {
        assert_eq!(verdict(5_000, 8_000), FitVerdict::Fits);
    }

    #[test]
    fn estimate_scales_with_ctx() {
        let mut i = llama3_8b_q4km_ctx4k();
        let e_4k = estimate_vram_mb(&i);
        i.ctx = 32768;
        let e_32k = estimate_vram_mb(&i);
        assert!(
            e_32k > e_4k + 1000,
            "ctx scaling broken: 4k={}, 32k={}",
            e_4k,
            e_32k
        );
    }

    #[test]
    fn pick_ngl_full_offload_when_it_fits() {
        let i = llama3_8b_q4km_ctx4k();
        let est = estimate_vram_mb(&i);
        assert_eq!(pick_ngl(&i, est + 1000), -1);
    }

    #[test]
    fn pick_ngl_zero_when_no_headroom() {
        let i = llama3_8b_q4km_ctx4k();
        // Free VRAM below the fixed backend+safety reserve → nothing on GPU.
        assert_eq!(pick_ngl(&i, BACKEND_OVERHEAD_MB + SAFETY_MARGIN_MB), 0);
        assert_eq!(pick_ngl(&i, 500), 0);
    }

    #[test]
    fn pick_ngl_partial_in_between() {
        let i = llama3_8b_q4km_ctx4k();
        let est = estimate_vram_mb(&i);
        // Roughly half the variable cost available → roughly half the layers.
        let fixed = BACKEND_OVERHEAD_MB + SAFETY_MARGIN_MB;
        let free = fixed + (est - fixed) / 2;
        let n = pick_ngl(&i, free);
        assert!(n > 0 && (n as u64) < i.n_layers, "got {n}");
        assert!((14..=18).contains(&n), "expected ~half of 32 layers, got {n}");
    }

    #[test]
    fn pick_ngl_never_exceeds_layer_count() {
        let mut i = llama3_8b_q4km_ctx4k();
        i.n_layers = 4;
        let est = estimate_vram_mb(&i);
        // Just under the full estimate: partial pick must stay ≤ n_layers.
        let n = pick_ngl(&i, est - 1);
        assert!((0..=4).contains(&n), "got {n}");
    }

    #[test]
    fn estimate_70b_q5km_large() {
        // Llama-3-70B Q5_K_M: ~50 GB file, 80 layers, 8 KV heads, head_dim 128,
        // 8 k context, fp16 KV cache.
        let i = FitInputs {
            file_size_mb: 50_000,
            n_layers: 80,
            n_kv_heads: 8,
            head_dim: 128,
            ctx: 8192,
            kv_dtype_bytes: 2,
            kv: None,
        };
        let est = estimate_vram_mb(&i);
        assert!(est > 52_000 && est < 58_000, "got {} MB", est);
    }

    /// Gemma4 26B-class GQA model (the bug repro). With corrected metadata —
    /// 30 layers, a small representative kv-head count (8), head_dim from
    /// key_length (512) — the KV-aware estimate stays in a sane range, NOT the
    /// ~25 GB the old code reported for a 15.8 GB file.
    fn gemma4_26b_q4km_ctx8k() -> FitInputs {
        FitInputs {
            file_size_mb: 15_800,
            n_layers: 30,
            n_kv_heads: 8,
            head_dim: 512,
            ctx: 8192,
            kv_dtype_bytes: 2,
            kv: None,
        }
    }

    #[test]
    fn estimate_gemma4_kv_in_sane_range() {
        let i = gemma4_26b_q4km_ctx8k();
        let kv = kv_cache_mb(&i);
        // 2 * 30 * 8 * 512 * 8192 * 2 / 1024^2 = 3840 MiB. Bounded by the model
        // not being treated as 16 full-context heads (which gave 7680 MiB).
        assert_eq!(kv, 3840, "kv-cache math drifted: {kv} MB");
        let est = estimate_vram_mb(&i);
        // Weights (15.8 GB) + KV (3.8 GB) + 2094 overhead ≈ 21.7 GB — fits 24 GB
        // even WITH KV, and is far below the bug's ~25 GB.
        assert!(est < 22_500, "estimate too high: {est} MB");
    }

    /// PRIMARY product fix: the user-facing verdict floor is weights + fixed
    /// overhead ONLY (no KV). A 15.8 GB Q4_K_M model must read "Fits" on 24 GB
    /// free — the exact case that wrongly read "Won't fit".
    #[test]
    fn weights_floor_fits_gemma4_on_24gb() {
        let file_size_mb = 15_800;
        let floor = file_size_mb + FIXED_OVERHEAD_MB; // ~17.9 GB
        assert_eq!(floor, 17_894);
        let free_mb = 24 * 1024; // 24 GiB GPU, fully free
        assert_eq!(verdict(floor, free_mb), FitVerdict::Fits);
    }

    /// The verdict floor excludes KV: it must be strictly below the KV-aware
    /// estimate for the same model/context.
    #[test]
    fn weights_floor_below_kv_aware_estimate() {
        let i = gemma4_26b_q4km_ctx8k();
        let floor = i.file_size_mb + FIXED_OVERHEAD_MB;
        let full = estimate_vram_mb(&i);
        assert!(floor < full, "floor {floor} should be < kv-aware {full}");
        assert_eq!(full - floor, kv_cache_mb(&i));
    }

    // -- SWA-aware KV path ---------------------------------------------------

    /// The real Gemma4 26B shape (downsized only in that every layer is listed
    /// explicitly): 30 layers in a 5-local : 1-global repeat. Global layers
    /// (sliding_window_pattern == false) carry 2 KV heads and full-width
    /// (512) K/V; local layers carry 8 KV heads and half-width (256) SWA K/V,
    /// capped at a 1024-token window. These are the exact values read from the
    /// user's file via the GGUF header.
    fn gemma4_swa_meta() -> GgufMeta {
        let mut pattern = Vec::with_capacity(30);
        let mut kv_heads = Vec::with_capacity(30);
        for i in 0..30u32 {
            let is_global = i % 6 == 5; // 5 globals: indices 5,11,17,23,29
            pattern.push(!is_global);
            kv_heads.push(if is_global { 2 } else { 8 });
        }
        GgufMeta {
            architecture: Some("gemma4".into()),
            block_count: Some(30),
            attention_head_count: Some(16),
            attention_head_count_kv: Some(8), // max, fallback only
            attention_head_count_kv_layers: Some(kv_heads),
            attention_key_length: Some(512),
            attention_value_length: Some(512),
            attention_key_length_swa: Some(256),
            attention_value_length_swa: Some(256),
            attention_sliding_window: Some(1024),
            sliding_window_pattern: Some(pattern),
            ..Default::default()
        }
    }

    /// The profile reduces the per-layer arrays to two context coefficients:
    /// 5 global layers × 2 heads × (512+512) × 2 B = 20 480 B/tok, and
    /// 25 local layers × 8 heads × (256+256) × 2 B = 204 800 B/tok, capped at
    /// the 1024-token window.
    #[test]
    fn kv_profile_gemma4_coefficients() {
        let p = KvProfile::from_meta(&gemma4_swa_meta(), 2).expect("SWA metadata present");
        assert_eq!(p.global_bytes_per_tok, 20_480);
        assert_eq!(p.swa_bytes_per_tok, 204_800);
        assert_eq!(p.swa_window, 1024);
    }

    /// End-to-end through `FitInputs::from_meta`: KV at the contexts from the bug
    /// reports. These are the ACCURATE figures — the old uniform formula reported
    /// ~48 GB at 102.5k where the truth is ~2.2 GB.
    #[test]
    fn swa_kv_cache_accurate_at_real_contexts() {
        let meta = gemma4_swa_meta();
        let kv = |ctx: u32| kv_cache_mb(&FitInputs::from_meta(&meta, 15_800, ctx, 2));
        assert_eq!(kv(8_192), 360, "8k");
        assert_eq!(kv(104_960), 2_250, "102.5k (the bug screenshot)");
        assert_eq!(kv(262_144), 5_320, "256k full context");
    }

    /// The SWA estimate is a small fraction of the uniform full-attention figure
    /// — the whole point of the fix. Same model, same context, KV profile on vs
    /// off.
    #[test]
    fn swa_kv_far_below_uniform_fallback() {
        let meta = gemma4_swa_meta();
        let swa = FitInputs::from_meta(&meta, 15_800, 104_960, 2);
        let uniform = FitInputs { kv: None, ..swa };
        let swa_kv = kv_cache_mb(&swa);
        let uniform_kv = kv_cache_mb(&uniform);
        assert!(
            swa_kv * 10 < uniform_kv,
            "SWA KV {swa_kv} should be << uniform {uniform_kv}",
        );
    }

    /// Past the window, local layers stop contributing: each extra token of
    /// context costs only the global coefficient. Doubling context from 128k to
    /// 256k must grow KV by exactly `global_bytes_per_tok × 131072`.
    #[test]
    fn swa_window_caps_local_layers() {
        let meta = gemma4_swa_meta();
        let kv = |ctx: u32| kv_cache_mb(&FitInputs::from_meta(&meta, 15_800, ctx, 2));
        let grow = kv(262_144) - kv(131_072);
        let expected = (20_480u64 * 131_072) / (1024 * 1024); // global only
        assert_eq!(grow, expected, "beyond the window only global layers grow");
    }

    /// With the accurate KV, the user's 15.8 GB Gemma4 FITS at 102.5k context on
    /// a 24 GB GPU — the case the screenshot wrongly showed as "Won't fit • 65.8
    /// GB". (Floor 17.9 GB + KV 2.2 GB ≈ 20.1 GB.)
    #[test]
    fn gemma4_fits_at_102k_with_swa() {
        let i = FitInputs::from_meta(&gemma4_swa_meta(), 15_800, 104_960, 2);
        let est = estimate_vram_mb(&i);
        assert!(est < 21_000, "estimate with SWA KV should be ~20 GB, got {est}");
        assert_eq!(verdict(est, 24 * 1024), FitVerdict::Fits);
    }

    /// A model whose header carries no per-layer attention detail (plain GQA:
    /// scalar `head_count_kv`, no pattern, no array) yields `None` and falls back
    /// to the uniform scalar formula — no regression for non-SWA models.
    #[test]
    fn kv_profile_none_without_swa_metadata() {
        let meta = GgufMeta {
            architecture: Some("llama".into()),
            block_count: Some(32),
            attention_head_count: Some(32),
            attention_head_count_kv: Some(8),
            attention_key_length: Some(128),
            ..Default::default()
        };
        assert!(KvProfile::from_meta(&meta, 2).is_none());
        // …and from_meta wires that through as the scalar path.
        assert!(FitInputs::from_meta(&meta, 4700, 4096, 2).kv.is_none());
    }
}
