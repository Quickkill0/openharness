//! What a context window costs, per architecture, the way llama.cpp b12182
//! allocates it. One function, [`context_memory`], serves the planner, the
//! load dialog's context slider and the planned-versus-actual comparison.
//!
//! Per layer kind (see [`LayerKind`]):
//! - **Full attention (dense / GQA)**: K and V rows of the projection widths
//!   (`head_count_kv × key/value length`) for every cell, K and V in their
//!   own cache types.
//! - **Sliding window**: the same rows, but only
//!   `min(cells, window × seqs + ubatch)` cells (padded to 256), per sequence
//!   stream (`llama_kv_cache_iswa`).
//! - **MLA** (DeepSeek / GLM): the compressed latent (`kv_lora_rank + rope
//!   dims`) as K only; with flash attention there is no V cache. llama.cpp
//!   refuses different K and V types for MLA models, so V follows K.
//!   Sparse-MLA (DSA) layers add a lightning-indexer cache in the K type.
//! - **Linear / recurrent** (KDA, gated delta-net, Mamba): a fixed f32 state
//!   per parallel sequence (`llama_memory_recurrent`), independent of context.
//!
//! Hybrids sum their layers. The context is padded to 256 cells; with
//! `n_parallel` sequences (explicit `-np`, non-unified KV) each sequence owns
//! `ctx / n_parallel` cells of every full-attention cache, a full SWA window,
//! and its own recurrent state.
//!
//! The compute buffer is llama.cpp's worst-case graph reservation:
//! micro-batch logits (`ubatch × n_vocab × 4`), the f16 attention masks
//! (`cells × ubatch × 2` per cache kind), activations
//! (`ubatch × n_embd × 16`, plus `ubatch × n_embd × 4` for recurrent
//! layers). When routed experts live in host memory, prompt
//! batches run on the GPU by copying one expert tensor at a time into the
//! compute buffer, so the reservation is at least that tensor plus its
//! activations. Calibrated against `llama-fit-params` dry runs (CPU and CUDA
//! builds); see the tests for the recorded numbers and tolerances.

use serde::{Deserialize, Serialize};

use super::accounting::{KvType, LayerAccount, LayerKind, ModelAccounting};

/// llama.cpp pads the context (and the SWA window) to this many cells.
pub const CTX_PAD: u32 = 256;
/// Micro-batch at or above which llama.cpp runs host-resident expert
/// matmuls on the GPU (op offload) instead of the CPU.
const OP_OFFLOAD_MIN_BATCH: u32 = 32;
/// Activation bytes per micro-batch token per `n_embd` (residual stream,
/// norms, attention output and FFN intermediates the allocator keeps live).
const ACT_BYTES_PER_EMBD: u64 = 16;
/// f16 attention-mask bytes per cell per micro-batch token.
const MASK_BYTES: u64 = 2;
/// Sparse-MLA (DSA indexer) models peak at about half the mask growth: the
/// indexer's top-k selection replaces the full mask at the peak node
/// (fitted on GLM-5.3-Flash, CPU and CUDA).
const INDEXER_MASK_BYTES: u64 = 1;
/// Activation bytes per micro-batch token of a host-expert prompt batch on
/// the GPU, per active expert and expert width (gate, up, their product and
/// the down-projection input in f32), fitted on GLM-5.3-Flash.
const EXPERT_ACT_BYTES_PER_FFN: u64 = 24;

/// Inputs that size a context.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ContextParams {
    /// Context length in tokens (padded to 256 internally).
    pub ctx: u32,
    /// K cache type (`-ctk`).
    pub kv_k: KvType,
    /// V cache type (`-ctv`). Ignored for MLA models (V follows K).
    pub kv_v: KvType,
    /// Parallel sequences (`-np`).
    pub n_parallel: u32,
    /// Micro-batch (`-ub`).
    pub ubatch: u32,
    /// Some routed experts are in host memory (`--n-cpu-moe` / `--cpu-moe`).
    pub experts_on_host: bool,
}

impl Default for ContextParams {
    fn default() -> Self {
        Self {
            ctx: 4096,
            kv_k: KvType::F16,
            kv_v: KvType::F16,
            n_parallel: 1,
            ubatch: 512,
            experts_on_host: false,
        }
    }
}

/// What a context costs. Byte counts are for the whole process (all
/// parallel sequences).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ContextMemory {
    /// The context llama.cpp will allocate (padded to 256).
    pub ctx: u32,
    /// The V type actually used (MLA models force it to the K type).
    pub kv_v: KvType,
    /// Full-attention K+V cache (MLA: the latent).
    pub kv_bytes: u64,
    /// Sliding-window K+V cache.
    pub swa_bytes: u64,
    /// DSA lightning-indexer key cache.
    pub indexer_bytes: u64,
    /// Fixed recurrent state of linear-attention / SSM layers.
    pub state_bytes: u64,
    /// Worst-case compute buffer on the device running the model (graphics
    /// memory with a GPU, else system memory).
    pub compute_bytes: u64,
    /// Host-side compute buffer next to a GPU (inputs, masks); system memory.
    pub host_compute_bytes: u64,
    /// Bytes added per extra token of context (the slider's slope).
    pub bytes_per_token: f64,
}

impl ContextMemory {
    /// Every cache that holds context: KV, SWA, indexer and recurrent state.
    pub fn cache_bytes(&self) -> u64 {
        self.kv_bytes + self.swa_bytes + self.indexer_bytes + self.state_bytes
    }

    /// Caches plus the device compute buffer.
    pub fn total_bytes(&self) -> u64 {
        self.cache_bytes() + self.compute_bytes
    }
}

fn pad(n: u64, to: u64) -> u64 {
    n.div_ceil(to) * to
}

/// Cells per sequence stream and sequence count for a context.
fn cells(p: &ContextParams) -> (u64, u64) {
    let n_seq = p.n_parallel.max(1) as u64;
    let ctx = pad(p.ctx.max(CTX_PAD) as u64, CTX_PAD as u64 * n_seq);
    (ctx / n_seq, n_seq)
}

/// Context memory one layer owns (KV or SWA cache, indexer cache and
/// recurrent state, all sequences). llama.cpp allocates each layer's share on
/// that layer's device, so the planner sums this over the layers it offloads.
pub fn layer_cache_bytes(m: &ModelAccounting, p: &ContextParams, l: &LayerAccount) -> u64 {
    if l.kind == LayerKind::Mtp {
        return 0;
    }
    let (cells, n_seq) = cells(p);
    let kv_k = p.kv_k;
    let kv_v = if m.mla_layers > 0 { kv_k } else { p.kv_v };
    let row = kv_k.row_bytes(l.k_elems_per_token) + kv_v.row_bytes(l.v_elems_per_token);
    let kv_cells = match l.swa_window {
        Some(w) => cells.min(pad(w as u64 + p.ubatch.max(1) as u64, CTX_PAD as u64)),
        None => cells,
    };
    let idx = kv_k.row_bytes(l.idx_elems_per_token) * cells;
    (row * kv_cells + idx) * n_seq + l.state_elems * 4 * n_seq
}

/// The memory a context of `p.ctx` tokens takes for model `m`.
pub fn context_memory(m: &ModelAccounting, p: &ContextParams) -> ContextMemory {
    let (cells, n_seq) = cells(p);
    let ub = p.ubatch.max(1) as u64;
    let ctx = cells * n_seq;
    let kv_k = p.kv_k;
    let kv_v = if m.mla_layers > 0 { kv_k } else { p.kv_v };

    let mut kv_row = 0u64; // bytes per cell, full-attention layers
    let mut idx_row = 0u64;
    let mut swa_bytes = 0u64;
    let mut swa_cells_max = 0u64;
    for l in m.cached_layers() {
        let row = kv_k.row_bytes(l.k_elems_per_token) + kv_v.row_bytes(l.v_elems_per_token);
        match l.swa_window {
            Some(w) => {
                let c = cells.min(pad(w as u64 + ub, CTX_PAD as u64));
                swa_cells_max = swa_cells_max.max(c);
                swa_bytes += row * c * n_seq;
            }
            None => kv_row += row,
        }
        idx_row += kv_k.row_bytes(l.idx_elems_per_token);
    }
    let kv_bytes = kv_row * cells * n_seq;
    let indexer_bytes = idx_row * cells * n_seq;
    let state_bytes = m.state_bytes(n_seq as u32);

    // Compute buffer.
    let mask_b = if m.has_indexer() { INDEXER_MASK_BYTES } else { MASK_BYTES };
    let mask = |c: u64| c * ub * mask_b;
    let masks = if kv_row > 0 { mask(cells) } else { 0 } + if swa_bytes > 0 { mask(swa_cells_max) } else { 0 };
    let act = ub * m.n_embd * ACT_BYTES_PER_EMBD;
    let recurrent = if m.recurrent_layers > 0 { ub * m.n_embd * 4 } else { 0 };
    let base = ub * m.n_vocab * 4 + masks + act + recurrent;
    let expert_split = if p.experts_on_host && p.ubatch >= OP_OFFLOAD_MIN_BATCH {
        m.max_expert_tensor_bytes
            + ub * m.expert_used.max(1) as u64 * m.expert_ffn.max(1) * EXPERT_ACT_BYTES_PER_FFN
            + masks
    } else {
        0
    };
    let compute_bytes = base.max(expert_split);
    let host_compute_bytes = masks + act;

    let growth = (kv_row + idx_row) as f64 + if kv_row > 0 { (ub * mask_b) as f64 / n_seq as f64 } else { 0.0 };

    ContextMemory {
        ctx: ctx as u32,
        kv_v,
        kv_bytes,
        swa_bytes,
        indexer_bytes,
        state_bytes,
        compute_bytes,
        host_compute_bytes,
        bytes_per_token: growth,
    }
}

/// The largest context (a multiple of 256, at most `ctx_max`) for which
/// `fits` holds, by binary search; `None` when even 256 tokens don't fit.
/// `fits` must only get harder as the context grows, which holds for every
/// memory budget because context memory never shrinks with more tokens.
pub fn largest_ctx(ctx_max: u32, fits: impl Fn(u32) -> bool) -> Option<u32> {
    let step = CTX_PAD;
    let (mut lo, mut hi) = (1u32, ctx_max / step);
    if hi == 0 || !fits(lo * step) {
        return None;
    }
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if fits(mid * step) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo * step)
}

#[cfg(test)]
#[path = "context_tests.rs"]
mod tests;
