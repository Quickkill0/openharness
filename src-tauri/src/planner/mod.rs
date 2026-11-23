//! Load planner: decides where a model's weights go across three memory
//! tiers (graphics memory, system memory, SSD), which llama-server flags
//! realise that, and how fast the result will run.
//!
//! - [`accounting`] reads the GGUF tensor table: always-used weights vs routed
//!   experts per layer, KV bytes per token, recurrent state.
//! - [`hardware`] profiles the GPU, RAM, the models drive and the bandwidths.
//! - [`speed`] turns bytes touched per token per tier into tok/s and labels.
//! - [`store`] persists SSD measurements and per-model calibration.
//! - [`runtime`] samples live tok/s, VRAM and RAM of the loaded model.
//!
//! The entry points are pure: [`plan`] (with user overrides) and
//! [`auto_plan`] (the one-click Load). The orchestrator calls them with the
//! VRAM it may use; `OrchestratorHandle::plan_text` / `load_text` are the
//! async wrappers that gather the inputs.
//!
//! # Placement
//!
//! llama.cpp offers these knobs, which the planner searches over:
//! - `-ngl N`: the last N layers' always-used weights (and their KV) go to the
//!   GPU; the rest run on the CPU from RAM.
//! - `--n-cpu-moe N` / `--cpu-moe`: routed experts of the first N blocks stay in
//!   host memory. MoE layers whose experts stay on the GPU are the last ones.
//! - `--moe-cache-mib N`: a GPU cache of hot experts for those host-resident
//!   banks (misses are uploaded over PCIe).
//! - host weights are memory-mapped; whatever does not fit in free RAM is paged
//!   from the SSD by the OS (`mmap` overflow).
//!
//! For each candidate the planner sums the bytes per tier and runs the speed
//! model; the fastest feasible layout wins.

pub mod accounting;
pub mod context;
pub mod hardware;
pub mod native;
pub mod runtime;
pub mod speed;
pub mod store;

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub use accounting::{KvType, ModelAccounting};
pub use context::{context_memory, ContextMemory, ContextParams};
pub use hardware::{DriveKind, HardwareProfile};
pub use speed::{Calibration, SpeedEstimate, SpeedLabel, TierMix};

use accounting::LayerKind;
use speed::{hit_rate, Traffic};

const MIB: u64 = 1024 * 1024;
const GIB: u64 = 1024 * MIB;

/// VRAM a fresh llama-server process takes before any tensor: CUDA context,
/// cuBLAS workspace, graph allocator slack.
const CUDA_CONTEXT_BYTES: u64 = 450 * MIB;
/// VRAM kept free for the desktop, the driver and allocation slack:
/// the larger of this and [`VRAM_HEADROOM_FRAC`] of the card.
const VRAM_HEADROOM_MIN: u64 = 512 * MIB;
const VRAM_HEADROOM_FRAC: f64 = 0.03;
/// RAM left for Windows and other apps when sizing the page cache.
const RAM_HEADROOM: u64 = 4 * GIB;
/// Smallest worthwhile `--moe-cache-mib`.
const MIN_EXPERT_CACHE: u64 = 512 * MIB;
/// KV positions read per decoded token in the speed estimate (a typical
/// conversation fill, not the full window).
const KV_TYPICAL_FILL: u32 = 8192;
/// Micro-batch sizes: default, and when host-resident weights are streamed to
/// the GPU for prompt processing (bigger batches amortise the upload).
const UBATCH_DEFAULT: u32 = 512;
const UBATCH_STREAMING: u32 = 2048;
/// Parallel sequences the planner asks for (`-np`). llama-server's `auto`
/// default is 4, which would allocate four recurrent states and SWA windows;
/// one desktop user needs one.
pub const N_PARALLEL: u32 = 1;
/// Safety margin on the predicted compute buffer. The formula is within ±6 %
/// of llama.cpp's reservation on every recorded case (see
/// `context_tests.rs`); the margin keeps an under-prediction from running out
/// of graphics memory.
pub(crate) const COMPUTE_MARGIN_FRAC: f64 = 0.08;
/// Default context cap for the one-click Load, and the cap for small models.
const AUTO_CTX_CAP: u32 = 32 * 1024;
const AUTO_CTX_CAP_SMALL: u32 = 128 * 1024;
/// Models at or under this size count as small for the context cap.
const SMALL_MODEL_BYTES: u64 = 8 * GIB;
const MIN_CTX: u32 = 4096;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Which binary realises the plan in the single text slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum EngineKind {
    #[default]
    LlamaServer,
    SsdStream,
}

/// User choices that pin parts of the plan. Unset fields are chosen by the
/// planner.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanOverrides {
    pub engine: native::EnginePreference,
    /// Context length in tokens.
    pub ctx: Option<u32>,
    /// K cache element type (and the V type unless `kv_type_v` is set).
    pub kv_type: Option<KvType>,
    /// V cache element type. MLA models keep no V cache and always use the K
    /// type.
    pub kv_type_v: Option<KvType>,
    /// MoE layers whose routed experts stay entirely in graphics memory.
    pub gpu_expert_layers: Option<u32>,
    /// GPU cache for hot experts of host-resident layers, in MiB (0 = off).
    pub expert_cache_mib: Option<u64>,
    /// Allow weights beyond free RAM to be paged from the SSD (default true).
    pub allow_ssd: Option<bool>,
    /// Most system memory the model may use, in MiB. Never raises the
    /// planner's own budget; whatever no longer fits goes to the SSD.
    pub ram_limit_mib: Option<u64>,
    /// Most graphics memory the model may use, in MiB, overhead included
    /// (0 keeps the model off the card).
    pub vram_limit_mib: Option<u64>,
}

/// Another resident the load could evict.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reclaimable {
    pub id: String,
    pub label: String,
    pub bytes: u64,
}

/// Graphics memory the plan may use.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VramBudget {
    /// Card total; 0 when there is no usable GPU.
    pub total_bytes: u64,
    /// Free now, plus the footprint of the model this load replaces.
    pub free_bytes: u64,
    /// Other residents that could be unloaded to make room.
    pub reclaimable: Vec<Reclaimable>,
}

impl VramBudget {
    pub fn no_gpu() -> Self {
        Self::default()
    }

    fn usable(&self) -> u64 {
        let headroom = VRAM_HEADROOM_MIN.max((self.total_bytes as f64 * VRAM_HEADROOM_FRAC) as u64);
        self.free_bytes.saturating_sub(headroom)
    }

    fn with_evictions(&self) -> VramBudget {
        VramBudget {
            total_bytes: self.total_bytes,
            free_bytes: self.free_bytes + self.reclaimable.iter().map(|r| r.bytes).sum::<u64>(),
            reclaimable: Vec::new(),
        }
    }
}

/// Bytes per tier. VRAM parts add up to the process's expected footprint.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TierBytes {
    /// Always-used weights on the GPU (attention, norms, shared experts,
    /// dense FFN, output) plus routed experts of GPU-resident layers.
    pub vram_weights: u64,
    /// `--moe-cache-mib` hot-expert cache.
    pub vram_expert_cache: u64,
    /// Context memory held on the GPU: KV, SWA and indexer caches plus
    /// recurrent state.
    pub vram_kv: u64,
    /// CUDA context and compute buffer (with its safety margin).
    pub vram_overhead: u64,
    /// Host-resident weights, context memory and compute buffers in RAM.
    pub ram: u64,
    /// Weights beyond free RAM, paged from the SSD on demand.
    pub ssd: u64,
}

impl TierBytes {
    pub fn vram_total(&self) -> u64 {
        self.vram_weights + self.vram_expert_cache + self.vram_kv + self.vram_overhead
    }
}

/// The better plan available if other residents were unloaded first.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvictAlternative {
    pub evicts: Vec<Reclaimable>,
    pub vram_bytes: u64,
    pub tier_mix: TierMix,
    pub decode_tps: f64,
    pub label: SpeedLabel,
}

/// A complete load decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoadPlan {
    pub engine_kind: EngineKind,
    #[serde(default)]
    pub native: Option<native::NativePlan>,
    pub architecture: String,
    pub ctx: u32,
    /// The model's trained context (0 when unknown).
    pub ctx_train: u32,
    /// K cache type (`-ctk`).
    pub kv_type: KvType,
    /// V cache type (`-ctv`).
    pub kv_type_v: KvType,
    /// Parallel sequences (`-np`).
    pub n_parallel: u32,
    /// What the context costs at these settings (whole process).
    pub context: ContextMemory,
    /// When the requested context doesn't fit: the largest one that does
    /// with the same cache types (`None` when even the weights don't fit).
    pub max_ctx: Option<u32>,
    /// Why the plan can't be loaded (set exactly when `feasible` is false).
    pub infeasible_reason: Option<String>,
    /// `-ngl`: -1 = all layers.
    pub gpu_layers: i32,
    /// MoE layers whose experts are fully on the GPU / in host memory.
    pub gpu_expert_layers: u32,
    pub cpu_expert_layers: u32,
    /// `--n-cpu-moe` value (block count), or `None` when unused / `--cpu-moe`.
    pub n_cpu_moe: Option<u32>,
    pub expert_cache_mib: u64,
    pub ubatch: u32,
    pub tiers: TierBytes,
    pub tier_mix: TierMix,
    /// False when the model cannot be placed (not enough RAM and SSD paging is
    /// off, or the KV cache alone overflows memory).
    pub feasible: bool,
    pub speed: SpeedEstimate,
    /// Planner-chosen llama-server flags beyond model/host/port (for display
    /// and for the sidecar, see [`LoadPlan::extra_args`]).
    pub args: Vec<String>,
    pub warnings: Vec<String>,
    pub evict_alternative: Option<EvictAlternative>,
    /// The overrides the plan honoured.
    pub overrides: PlanOverrides,
}

impl LoadPlan {
    /// Flags not covered by the core `ModelLoadArgs` fields (ctx, ngl,
    /// batch, flash attention, mmap). The cache types travel here because
    /// K and V may differ.
    pub fn extra_args(&self) -> Vec<String> {
        if let Some(native) = &self.native { return native.extra_args(self); }
        let mut v: Vec<String> = vec!["-np".into(), self.n_parallel.to_string()];
        if self.kv_type != KvType::F16 || self.kv_type_v != KvType::F16 {
            v.extend(["-ctk".into(), self.kv_type.as_arg().into(), "-ctv".into(), self.kv_type_v.as_arg().into()]);
        }
        if self.cpu_expert_layers > 0 {
            match self.n_cpu_moe {
                Some(n) => {
                    v.push("--n-cpu-moe".into());
                    v.push(n.to_string());
                }
                None => v.push("--cpu-moe".into()),
            }
        }
        if self.expert_cache_mib > 0 {
            v.push("--moe-cache-mib".into());
            v.push(self.expert_cache_mib.to_string());
        }
        v.push("-ub".into());
        v.push(self.ubatch.to_string());
        // A graphics memory limit that leaves nothing on the card: a CUDA
        // build would still create its context there with `-ngl 0`.
        if self.overrides.vram_limit_mib.is_some() && self.tiers.vram_total() == 0 {
            v.extend(["--device".into(), "none".into()]);
        }
        // The planner already fitted the model; llama.cpp's own fitter would
        // second-guess the layout.
        v.push("--fit".into());
        v.push("off".into());
        // Live tok/s for the runtime stats panel and calibration.
        v.push("--metrics".into());
        v
    }

    /// The tier mix label used for calibration.
    pub fn calibration_key(&self) -> TierMix {
        self.tier_mix
    }
}

/// Per-model calibration, keyed by tier mix.
pub type ModelCalibration = BTreeMap<TierMix, Calibration>;

// ---------------------------------------------------------------------------
// Layout evaluation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct Layout {
    /// Decode layers (from the end) whose always-used weights are on the GPU.
    gpu_layers: u32,
    /// MoE layers (from the end) whose experts are on the GPU.
    gpu_expert_layers: u32,
    cache_bytes: u64,
    ubatch: u32,
}

struct Evaluated {
    layout: Layout,
    context: ContextMemory,
    tiers: TierBytes,
    feasible: bool,
    fits_vram: bool,
    speed: SpeedEstimate,
    tier_mix: TierMix,
}

#[derive(Clone, Copy)]
struct Inputs<'a> {
    m: &'a ModelAccounting,
    hw: &'a HardwareProfile,
    cal: Option<&'a ModelCalibration>,
    ctx: u32,
    kv_k: KvType,
    kv_v: KvType,
    vram_usable: u64,
    /// User cap on system memory, in bytes.
    ram_limit: Option<u64>,
    /// The graphics memory limit is too small for llama-server to use the
    /// card at all, so this plan runs on the CPU (`gpu` is false).
    off_card: bool,
    gpu: bool,
    allow_ssd: bool,
}

/// Indices (into `m.layers`) of decode layers, in order.
fn decode_layers(m: &ModelAccounting) -> Vec<usize> {
    m.layers
        .iter()
        .enumerate()
        .filter(|(_, l)| l.kind != LayerKind::Mtp)
        .map(|(i, _)| i)
        .collect()
}

impl Inputs<'_> {
    fn params(&self, ctx: u32, ubatch: u32, experts_on_host: bool) -> ContextParams {
        ContextParams { ctx, kv_k: self.kv_k, kv_v: self.kv_v, n_parallel: N_PARALLEL, ubatch, experts_on_host }
    }

    fn context(&self, ctx: u32, ubatch: u32, experts_on_host: bool) -> ContextMemory {
        context_memory(self.m, &self.params(ctx, ubatch, experts_on_host))
    }

    /// System memory the plan may use: what is free after headroom, capped by
    /// the user's limit.
    fn ram_budget(&self) -> u64 {
        let free = self.hw.ram_available_bytes.saturating_sub(RAM_HEADROOM);
        self.ram_limit.map_or(free, |l| l.min(free))
    }

    /// The user's system memory limit, when it is what holds the budget down.
    fn binding_ram_limit(&self) -> Option<u64> {
        self.ram_limit.filter(|&l| l < self.hw.ram_available_bytes.saturating_sub(RAM_HEADROOM))
    }

    /// The binding system memory limit, when the context memory that must
    /// stay in RAM is larger than it. `evaluate` puts no weights in RAM in
    /// that case, so the RAM tier is that context memory alone.
    fn ram_limit_short(&self, p: &LoadPlan) -> Option<u64> {
        self.binding_ram_limit().filter(|_| p.tiers.ram > self.ram_budget())
    }
}

/// Decode position (into [`decode_layers`]) from which routed experts sit in
/// graphics memory, `usize::MAX` when none do: the last `gpu_expert_layers`
/// MoE layers, and only those whose block is on the GPU itself. Experts of
/// every other MoE layer stay in host memory through `--cpu-moe` /
/// `--n-cpu-moe`.
fn gpu_expert_from(m: &ModelAccounting, dec: &[usize], layout: Layout, gpu: bool) -> usize {
    if !gpu {
        return usize::MAX;
    }
    let n = dec.len();
    let first_gpu = n - (layout.gpu_layers as usize).min(n);
    let moe_pos: Vec<usize> =
        dec.iter().enumerate().filter(|(_, &li)| m.layers[li].expert_bytes > 0).map(|(p, _)| p).collect();
    let e = (layout.gpu_expert_layers as usize).min(moe_pos.len());
    moe_pos.get(moe_pos.len() - e).map_or(usize::MAX, |&p| p.max(first_gpu))
}

/// The `-ngl` value that puts the last `g` decode layers on the GPU.
/// llama.cpp b12182 numbers offloadable layers as every block (MTP blocks
/// included, they come last) plus the output head: `-ngl N` offloads blocks
/// `n_layer_all + 1 - N ..` and the output, so the output head and the MTP
/// blocks take the first slots.
fn ngl_for(m: &ModelAccounting, g: u32) -> i32 {
    if g == 0 {
        0
    } else if g >= m.n_layers {
        -1
    } else {
        (g + 1 + m.mtp_layers) as i32
    }
}

fn evaluate(inp: &Inputs, layout: Layout) -> Evaluated {
    let m = inp.m;
    let dec = decode_layers(m);
    let n = dec.len() as u32;
    let g = if inp.gpu { layout.gpu_layers.min(n) } else { 0 };
    let first_gpu = (n - g) as usize; // decode-layer position of the first GPU layer
    let expert_from = gpu_expert_from(m, &dec, layout, inp.gpu);

    let layer_always: u64 = dec.iter().map(|&i| m.layers[i].always_bytes).sum();
    let global = m.always_bytes.saturating_sub(layer_always);
    // Tied embeddings: the output head reuses token_embd, and llama.cpp loads
    // a copy of it on the output device (on the host it is the same mapping).
    let output_extra = if m.output_tied { m.token_embd_bytes } else { 0 };
    // `-ngl` offloads the output head before any block (see [`ngl_for`]).
    let output_on_gpu = g > 0;

    let mut gpu_weights = 0u64;
    let mut host_weights = 0u64;
    let mut gpu_read = 0f64; // bytes GPU kernels read per token
    let mut host_dense = 0u64; // host always-used weights (CPU)
    let mut host_read_extra = 0u64; // host bytes read per token beyond them
    let mut cpu_expert_bank = 0u64;
    let mut cpu_expert_active = 0f64;
    let mut gpu_expert_active = 0f64;
    for (p, &li) in dec.iter().enumerate() {
        let l = &m.layers[li];
        if p >= first_gpu {
            gpu_weights += l.always_bytes;
            gpu_read += l.always_bytes as f64;
        } else {
            host_weights += l.always_bytes;
            host_dense += l.always_bytes;
        }
        if l.expert_bytes > 0 {
            let k = (m.expert_used.max(1) as f64).min(l.n_expert.max(1) as f64);
            let active = l.expert_bytes as f64 * k / l.n_expert.max(1) as f64;
            if p >= expert_from {
                gpu_weights += l.expert_bytes;
                gpu_expert_active += active;
            } else {
                host_weights += l.expert_bytes;
                cpu_expert_bank += l.expert_bytes;
                cpu_expert_active += active;
            }
        }
    }
    if output_on_gpu {
        gpu_weights += global + output_extra;
        gpu_read += (global + output_extra) as f64;
    } else {
        host_weights += global;
        host_dense += global;
        host_read_extra += output_extra;
    }
    // token_embd stays on the host (one row read per token).
    let host_fixed = m.token_embd_bytes;

    // Context memory and compute buffers claim graphics memory before any
    // expert cache: the cache only gets what `place` finds spare after them.
    // llama.cpp allocates each layer's caches on that layer's device, so the
    // GPU share is summed over the offloaded layers.
    let params = inp.params(inp.ctx, layout.ubatch, inp.gpu && cpu_expert_bank > 0);
    let ctxm = context_memory(m, &params);
    let kv_full = ctxm.cache_bytes();
    let kv_gpu: u64 =
        dec[first_gpu..].iter().map(|&li| context::layer_cache_bytes(m, &params, &m.layers[li])).sum();
    let frac_kv = if kv_full == 0 { 0.0 } else { kv_gpu as f64 / kv_full as f64 };
    let kv_read = inp.context(inp.ctx.min(KV_TYPICAL_FILL), layout.ubatch, false).cache_bytes() as f64;
    let compute = (ctxm.compute_bytes as f64 * (1.0 + COMPUTE_MARGIN_FRAC)) as u64;
    // Host-side context memory: caches of CPU layers plus the compute
    // buffer that lives in RAM (all of it without a GPU).
    let kv_host = kv_full - kv_gpu + if inp.gpu { ctxm.host_compute_bytes } else { compute };
    let cache = if inp.gpu { layout.cache_bytes.min(cpu_expert_bank) } else { 0 };
    let overhead = if inp.gpu { CUDA_CONTEXT_BYTES + compute } else { 0 };
    let vram_total = gpu_weights + cache + kv_gpu + overhead;
    let fits_vram = !inp.gpu || vram_total <= inp.vram_usable;

    // Host side: context memory must be in RAM; weights fill the rest,
    // overflow → SSD.
    let ram_budget = inp.ram_budget();
    let host_weight_total = host_weights + host_fixed;
    let ram_for_weights = ram_budget.saturating_sub(kv_host);
    let in_ram = host_weight_total.min(ram_for_weights);
    let on_ssd = host_weight_total - in_ram;
    let feasible = fits_vram && kv_host <= ram_budget && (on_ssd == 0 || inp.allow_ssd);

    // Which host weights land on the SSD: experts are colder than dense
    // weights (read every token), so dense/fixed weights claim RAM first.
    let dense_host = host_dense + host_fixed;
    let dense_in_ram = dense_host.min(in_ram);
    let dense_on_ssd = dense_host - dense_in_ram;
    let experts_in_ram = in_ram - dense_in_ram;
    let r_dense = if dense_host == 0 { 1.0 } else { dense_in_ram as f64 / dense_host as f64 };

    // ── per-token traffic ──
    let mut t = Traffic {
        layers: n,
        ubatch: layout.ubatch,
        active_params: m.active_params as f64,
        cpu_only: !inp.gpu,
        ..Default::default()
    };
    t.vram = gpu_read + gpu_expert_active + kv_read * frac_kv;
    // Host dense weights are multiplied on the CPU every token.
    let host_dense_read = (host_dense + host_read_extra) as f64;
    t.cpu_from_ram += host_dense_read * r_dense + kv_read * (1.0 - frac_kv);
    t.cpu_from_ssd += host_dense_read * (1.0 - r_dense);
    let bpp_dense = if m.active_params > 0 {
        ((m.always_bytes + m.active_expert_bytes_per_token) as f64 / m.active_params as f64).max(0.1)
    } else {
        0.6
    };
    t.cpu_params += host_dense_read / bpp_dense;

    if cpu_expert_bank > 0 {
        let bank = cpu_expert_bank as f64;
        let f_v = cache as f64 / bank;
        let f_r = experts_in_ram as f64 / bank;
        if cache > 0 {
            let h_v = hit_rate(f_v);
            let h_vr = hit_rate(f_v + f_r);
            t.vram += cpu_expert_active * h_v;
            t.ram_to_gpu += cpu_expert_active * (h_vr - h_v);
            t.ssd_to_gpu += cpu_expert_active * (1.0 - h_vr);
        } else {
            let h_r = hit_rate(f_r);
            t.cpu_from_ram += cpu_expert_active * h_r;
            t.cpu_from_ssd += cpu_expert_active * (1.0 - h_r);
            let bpp = if m.expert_bytes_per_param > 0.0 { m.expert_bytes_per_param } else { bpp_dense };
            t.cpu_params += cpu_expert_active / bpp;
        }
    }
    // Prompt processing streams host-resident weights to the GPU per batch.
    t.stream_ram = (in_ram.saturating_sub(host_fixed)) as f64;
    t.stream_ssd = on_ssd as f64;

    let tier_mix = if on_ssd > 0 {
        TierMix::Ssd
    } else if host_weights > 0 || !inp.gpu {
        TierMix::Ram
    } else {
        TierMix::Vram
    };
    let cal = inp.cal.and_then(|c| c.get(&tier_mix));
    let speed = speed::estimate(&t, inp.hw, cal);

    let _ = (dense_on_ssd, experts_in_ram);
    Evaluated {
        layout,
        context: ctxm,
        tiers: TierBytes {
            vram_weights: if inp.gpu { gpu_weights } else { 0 },
            vram_expert_cache: cache,
            vram_kv: kv_gpu,
            vram_overhead: overhead,
            ram: in_ram + kv_host,
            ssd: on_ssd,
        },
        feasible,
        fits_vram,
        speed,
        tier_mix,
    }
}

/// Keep `e` if it beats the current best: feasible first, then decode speed,
/// then prompt speed.
fn consider(e: Evaluated, best: &mut Option<Evaluated>) {
    let better = match best {
        None => true,
        Some(b) => {
            let ka = (e.feasible, e.speed.decode_tps, e.speed.prompt_tps);
            let kb = (b.feasible, b.speed.decode_tps, b.speed.prompt_tps);
            ka.partial_cmp(&kb) == Some(std::cmp::Ordering::Greater)
        }
    };
    if better {
        *best = Some(e);
    }
}

/// Best layout for a fixed context and KV type.
fn place(inp: &Inputs, ov: &PlanOverrides) -> Evaluated {
    let m = inp.m;
    let n = decode_layers(m).len() as u32;
    let n_moe = m.moe_layers;

    if !inp.gpu {
        return evaluate(
            inp,
            Layout { gpu_layers: 0, gpu_expert_layers: 0, cache_bytes: 0, ubatch: UBATCH_DEFAULT },
        );
    }

    let pinned_experts = ov.gpu_expert_layers.map(|e| e.min(n_moe));
    let pinned_cache = ov.expert_cache_mib.map(|c| c * MIB);

    // 1. Everything on the GPU.
    let full = Layout {
        gpu_layers: n,
        gpu_expert_layers: pinned_experts.unwrap_or(n_moe),
        cache_bytes: 0,
        ubatch: UBATCH_DEFAULT,
    };
    if pinned_experts.is_none() && pinned_cache.is_none() {
        let e = evaluate(inp, full);
        if e.fits_vram {
            return e;
        }
    }

    // 2. MoE: all layers on the GPU, experts split between GPU layers, a hot
    //    cache, and host memory. Search the split for the fastest feasible one.
    let mut best: Option<Evaluated> = None;
    if n_moe > 0 {
        let expert_range: Vec<u32> = match pinned_experts {
            Some(e) => vec![e],
            None => (0..=n_moe).collect(),
        };
        for e_gpu in expert_range {
            let base = Layout { gpu_layers: n, gpu_expert_layers: e_gpu, cache_bytes: 0, ubatch: UBATCH_STREAMING };
            let probe = evaluate(inp, base);
            let spare = inp.vram_usable.saturating_sub(probe.tiers.vram_total());
            let caches: Vec<u64> = match pinned_cache {
                Some(c) => vec![c],
                None if e_gpu < n_moe && spare >= MIN_EXPERT_CACHE => {
                    // Whole MiB, 256 MiB granularity keeps args readable.
                    vec![0, (spare / (256 * MIB)) * 256 * MIB]
                }
                None => vec![0],
            };
            for c in caches {
                let ev = evaluate(inp, Layout { cache_bytes: c, ..base });
                if pinned_experts.is_some() || pinned_cache.is_some() || ev.fits_vram {
                    consider(ev, &mut best);
                }
            }
        }
        // Pinned expert placement is honoured even when it doesn't fit (the
        // plan is then marked infeasible with a warning).
        if best.as_ref().is_some_and(|b| b.fits_vram)
            || (pinned_experts.is_some() || pinned_cache.is_some()) && best.is_some()
        {
            return best.unwrap();
        }
    }

    // 3. Partial offload: fewer layers on the GPU (experts on the host).
    for g in (0..n).rev() {
        let l = Layout { gpu_layers: g, gpu_expert_layers: 0, cache_bytes: 0, ubatch: UBATCH_STREAMING };
        let e = evaluate(inp, l);
        if e.fits_vram {
            consider(e, &mut best);
            break;
        }
    }
    best.unwrap_or_else(|| {
        evaluate(inp, Layout { gpu_layers: 0, gpu_expert_layers: 0, cache_bytes: 0, ubatch: UBATCH_STREAMING })
    })
}

fn build_plan(inp: &Inputs, ev: Evaluated, ov: &PlanOverrides) -> LoadPlan {
    let m = inp.m;
    let dec = decode_layers(m);
    let n = dec.len() as u32;
    let layout = ev.layout;
    let gpu_layers = if inp.gpu { ngl_for(m, layout.gpu_layers.min(n)) } else { 0 };
    // Which MoE blocks keep their experts on the GPU, exactly as `evaluate`
    // counted them; every other MoE block's experts are pinned to host
    // memory. Without the flag llama.cpp loads the experts of every
    // offloaded block into graphics memory, partial offload included.
    let expert_from = gpu_expert_from(m, &dec, layout, inp.gpu);
    let moe: Vec<(usize, u32)> = dec
        .iter()
        .enumerate()
        .filter(|(_, &i)| m.layers[i].expert_bytes > 0)
        .map(|(p, &i)| (p, m.layers[i].index))
        .collect();
    let n_moe = moe.len() as u32;
    let gpu_expert_layers = moe.iter().filter(|(p, _)| *p >= expert_from).count() as u32;
    let cpu_expert_layers = if inp.gpu { n_moe - gpu_expert_layers } else { 0 };
    // `--n-cpu-moe N` pins the experts of blocks 0..N; `--cpu-moe` all of them.
    let n_cpu_moe = if cpu_expert_layers == 0 || gpu_expert_layers == 0 {
        None
    } else {
        moe.iter().filter(|(p, _)| *p < expert_from).map(|(_, idx)| idx + 1).max()
    };

    let mut plan = LoadPlan {
        engine_kind: EngineKind::LlamaServer,
        native: None,
        architecture: m.architecture.clone(),
        ctx: inp.ctx,
        ctx_train: m.ctx_train.min(u32::MAX as u64) as u32,
        kv_type: inp.kv_k,
        kv_type_v: inp.kv_v,
        n_parallel: N_PARALLEL,
        context: ev.context.clone(),
        max_ctx: None,
        infeasible_reason: None,
        gpu_layers,
        gpu_expert_layers,
        cpu_expert_layers,
        n_cpu_moe,
        expert_cache_mib: ev.tiers.vram_expert_cache / MIB,
        ubatch: layout.ubatch,
        tiers: ev.tiers,
        tier_mix: ev.tier_mix,
        feasible: ev.feasible,
        speed: ev.speed,
        args: Vec::new(),
        warnings: Vec::new(),
        evict_alternative: None,
        overrides: ov.clone(),
    };
    plan.args = display_args(&plan);
    plan.warnings = warnings(inp, &plan);
    plan
}

/// The planner-controlled llama-server flags, as shown to the user.
fn display_args(p: &LoadPlan) -> Vec<String> {
    let mut v: Vec<String> = vec!["-c".into(), p.ctx.to_string(), "-ngl".into()];
    v.push(if p.gpu_layers < 0 { "all".into() } else { p.gpu_layers.to_string() });
    v.extend(["-fa".into(), "on".into()]);
    v.extend(["-b".into(), p.ubatch.max(2048).to_string()]);
    v.extend(p.extra_args());
    v
}

fn gb(b: u64) -> String {
    format!("{:.1} GB", b as f64 / 1e9)
}

fn warnings(inp: &Inputs, p: &LoadPlan) -> Vec<String> {
    let mut w = Vec::new();
    if inp.off_card {
        w.push("The graphics memory limit is below what llama-server needs to use the card at all, so the model runs on the CPU and the card is left alone.".into());
    } else if !inp.gpu {
        w.push("No NVIDIA GPU detected: the model runs on the CPU.".into());
    }
    if p.tiers.ssd > 0 {
        let drive = inp.hw.storage.as_ref();
        if drive.is_some_and(|s| s.kind == DriveKind::Hdd) {
            w.push(format!(
                "{} of weights will be read from a hard disk while generating; expect it to be very slow.",
                gb(p.tiers.ssd)
            ));
        } else {
            w.push(format!(
                "SSD streaming active: {} of weights don't fit in memory and are read from the SSD as needed.",
                gb(p.tiers.ssd)
            ));
        }
        if !inp.hw.bandwidth.ssd_measured {
            w.push("SSD speed is a default guess; run the SSD benchmark for a better estimate.".into());
        }
        if inp.binding_ram_limit().is_some() {
            w.push("The system memory limit sets what is planned to stay in memory. llama-server maps the whole model file, so Windows may keep more of it cached while RAM is free.".into());
        }
    }
    if !p.feasible {
        if inp.gpu && p.tiers.vram_total() > inp.vram_usable {
            let limited = p.overrides.vram_limit_mib.is_some_and(|l| l.saturating_mul(MIB) <= inp.vram_usable);
            w.push(format!(
                "These settings need {} of graphics memory but {} {}.",
                gb(p.tiers.vram_total()),
                if limited { "the graphics memory limit allows" } else { "only" },
                if limited { gb(inp.vram_usable) } else { format!("{} is free", gb(inp.vram_usable)) }
            ));
        } else if let Some(limit) = inp.ram_limit_short(p) {
            w.push(format!(
                "These settings need {} of system memory for the context window but the system memory limit allows {}.",
                gb(p.tiers.ram),
                gb(limit)
            ));
        } else if !inp.allow_ssd && p.tiers.ssd > 0 {
            w.push("Doesn't fit in graphics and system memory, and SSD streaming is off.".into());
        } else {
            w.push("Doesn't fit: not enough memory for the context window. Lower the context length.".into());
        }
    }
    if p.ctx_train > 0 && p.ctx < p.ctx_train && p.overrides.ctx.is_none() {
        w.push(format!(
            "Context capped at {} tokens (the model supports {}).",
            p.ctx, p.ctx_train
        ));
    }
    if p.gpu_layers >= 0 && inp.gpu {
        let n = inp.m.n_layers;
        let on_gpu = (p.gpu_layers as u32).saturating_sub(1 + inp.m.mtp_layers);
        w.push(format!("Only {on_gpu} of {n} layers fit in graphics memory."));
    }
    w
}

/// (K, V) cache type pairs to try. MLA keeps no V cache, and llama.cpp
/// refuses differing types there, so V always follows K for MLA models.
fn kv_candidates(m: &ModelAccounting, ov: &PlanOverrides) -> Vec<(KvType, KvType)> {
    let ks = match ov.kv_type {
        Some(k) => vec![k],
        None => vec![KvType::F16, KvType::Q8_0],
    };
    ks.into_iter()
        .map(|k| (k, if m.mla_layers > 0 { k } else { ov.kv_type_v.unwrap_or(k) }))
        .collect()
}

fn ctx_candidates(m: &ModelAccounting, ov: &PlanOverrides) -> Vec<u32> {
    if let Some(c) = ov.ctx {
        return vec![c.max(256)];
    }
    let train = if m.ctx_train == 0 { MIN_CTX as u64 } else { m.ctx_train };
    let cap = if m.total_bytes <= SMALL_MODEL_BYTES { AUTO_CTX_CAP_SMALL } else { AUTO_CTX_CAP };
    let top = (train.min(cap as u64)) as u32;
    let mut v = vec![top];
    let mut c = top;
    while c / 2 >= MIN_CTX.min(top) && c / 2 >= 1 && c > MIN_CTX {
        c /= 2;
        v.push(c.max(MIN_CTX.min(top)));
    }
    v.dedup();
    v
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

/// Plan a load honouring `overrides`. Unset choices are made the way
/// [`auto_plan`] makes them: the largest context (up to the trained max,
/// capped at 32K unless the model is small) that keeps decode speed in the
/// best label achievable, f16 KV unless q8_0 buys a faster memory tier, and
/// experts placed for the highest decode speed.
pub fn plan(
    model: &ModelAccounting, hw: &HardwareProfile, budget: &VramBudget,
    overrides: &PlanOverrides, calibration: Option<&ModelCalibration>,
) -> LoadPlan {
    plan_with_native(model, hw, budget, overrides, calibration, &native::NativeSupport::default(), None)
}

pub fn auto_plan(model: &ModelAccounting, hw: &HardwareProfile, budget: &VramBudget, calibration: Option<&ModelCalibration>) -> LoadPlan {
    plan(model, hw, budget, &PlanOverrides::default(), calibration)
}

pub fn plan_with_native(
    model: &ModelAccounting, hw: &HardwareProfile, budget: &VramBudget,
    overrides: &PlanOverrides, calibration: Option<&ModelCalibration>,
    native: &native::NativeSupport, native_calibration: Option<&ModelCalibration>,
) -> LoadPlan {
    let llama = llama_plan(model, hw, budget, overrides, calibration);
    native::select(model, hw, budget, overrides, native, native_calibration, llama)
}

fn llama_plan(
    model: &ModelAccounting,
    hw: &HardwareProfile,
    budget: &VramBudget,
    overrides: &PlanOverrides,
    calibration: Option<&ModelCalibration>,
) -> LoadPlan {
    let mut p = plan_inner(model, hw, budget, overrides, calibration);
    if !budget.reclaimable.is_empty() && budget.total_bytes > 0 {
        let alt = plan_inner(model, hw, &budget.with_evictions(), overrides, calibration);
        let better = alt.feasible
            && (!p.feasible || alt.tier_mix < p.tier_mix || alt.speed.label > p.speed.label);
        if better {
            let names: Vec<&str> = budget.reclaimable.iter().map(|r| r.label.as_str()).collect();
            p.warnings.push(format!(
                "Unloading {} first would make this {} ({:.0} tok/s).",
                names.join(", "),
                match alt.tier_mix {
                    TierMix::Vram => "run entirely in graphics memory",
                    TierMix::Ram => "use less system memory",
                    TierMix::Ssd => "faster",
                },
                alt.speed.decode_tps
            ));
            p.evict_alternative = Some(EvictAlternative {
                evicts: budget.reclaimable.clone(),
                vram_bytes: alt.tiers.vram_total(),
                tier_mix: alt.tier_mix,
                decode_tps: alt.speed.decode_tps,
                label: alt.speed.label,
            });
        }
    }
    p
}

/// The one-click Load plan: [`plan`] with no overrides. This is the function
/// the OpenAI-compatible API calls for just-in-time loads (through
/// `OrchestratorHandle::load_text`, which gathers the budget from the ledger).
#[cfg(test)]
fn llama_auto_plan(
    model: &ModelAccounting,
    hw: &HardwareProfile,
    budget: &VramBudget,
    calibration: Option<&ModelCalibration>,
) -> LoadPlan {
    llama_plan(model, hw, budget, &PlanOverrides::default(), calibration)
}

fn plan_inner(
    m: &ModelAccounting,
    hw: &HardwareProfile,
    budget: &VramBudget,
    ov: &PlanOverrides,
    cal: Option<&ModelCalibration>,
) -> LoadPlan {
    let gpu = budget.total_bytes > 0;
    let allow_ssd = ov.allow_ssd.unwrap_or(true);
    let vram_limit = ov.vram_limit_mib.map(|l| l.saturating_mul(MIB));
    let vram_usable = vram_limit.map_or(budget.usable(), |l| l.min(budget.usable()));
    let ram_limit = ov.ram_limit_mib.map(|l| l.saturating_mul(MIB));
    let inputs = |ctx, kv_k, kv_v, vram_usable, off_card| Inputs {
        m, hw, cal, ctx, kv_k, kv_v, vram_usable, ram_limit, off_card, gpu: gpu && !off_card, allow_ssd,
    };
    // A graphics memory limit no on-card layout fits (llama-server's CUDA
    // context and compute buffers alone can exceed it): plan on the CPU with
    // the card left out, rather than refuse. The limit alone decides this,
    // by placing against it as if the card were otherwise empty. When free
    // graphics memory is what falls short the plan is refused as it is
    // without a limit, and a pinned GPU expert placement is never dropped
    // this way: it stays pinned and is refused.
    let pins_gpu = ov.gpu_expert_layers.is_some_and(|e| e > 0) || ov.expert_cache_mib.is_some_and(|c| c > 0);
    let off_card = gpu
        && !pins_gpu
        && vram_limit.is_some_and(|limit| {
            ctx_candidates(m, ov).into_iter().all(|ctx| {
                kv_candidates(m, ov)
                    .into_iter()
                    .all(|(kv_k, kv_v)| !place(&inputs(ctx, kv_k, kv_v, limit, false), ov).fits_vram)
            })
        });
    let mut candidates: Vec<(Inputs, Evaluated)> = Vec::new();
    for ctx in ctx_candidates(m, ov) {
        for (kv_k, kv_v) in kv_candidates(m, ov) {
            let inp = inputs(ctx, kv_k, kv_v, vram_usable, off_card);
            let ev = place(&inp, ov);
            candidates.push((inp, ev));
        }
    }
    // Best label among feasible candidates (or among all if none is).
    let any_feasible = candidates.iter().any(|(_, e)| e.feasible);
    let pool = |e: &Evaluated| e.feasible || !any_feasible;
    let best_label = candidates
        .iter()
        .filter(|(_, e)| pool(e))
        .map(|(_, e)| e.speed.label)
        .max()
        .unwrap_or(SpeedLabel::Slow);
    // Largest context in that label; at equal context prefer f16 unless q8_0
    // reaches a faster tier mix.
    let mut chosen: Option<usize> = None;
    for (i, (inp, e)) in candidates.iter().enumerate() {
        if !pool(e) || e.speed.label != best_label {
            continue;
        }
        chosen = match chosen {
            None => Some(i),
            Some(j) => {
                let (cinp, ce) = &candidates[j];
                let take = inp.ctx > cinp.ctx
                    || (inp.ctx == cinp.ctx
                        && (e.tier_mix < ce.tier_mix
                            || (e.tier_mix == ce.tier_mix && inp.kv_k == KvType::F16 && cinp.kv_k != KvType::F16)));
                if take { Some(i) } else { Some(j) }
            }
        };
    }
    let idx = chosen.unwrap_or(0);
    let (inp, ev) = candidates.swap_remove(idx);
    let mut plan = build_plan(&inp, ev, ov);
    if let Some(why) = &m.unsupported {
        // A layer whose context memory can't be sized: refuse rather than
        // plan with zero for it.
        let reason = format!(
            "Can't estimate memory for this {} model ({why}), so it can't be planned. Load it with a manual GPU layer count instead.",
            m.architecture
        );
        plan.feasible = false;
        plan.max_ctx = None;
        plan.warnings.push(reason.clone());
        plan.infeasible_reason = Some(reason);
        return plan;
    }
    if !plan.feasible {
        plan.max_ctx = largest_feasible_ctx(&inp, ov);
        let reason = infeasible_reason(&inp, &plan);
        plan.warnings.push(reason.clone());
        plan.infeasible_reason = Some(reason);
    }
    plan
}

/// Largest context (a multiple of 256, strictly below the requested one)
/// that `place` can fit with the same cache types, or `None` when even the
/// smallest context doesn't fit.
fn largest_feasible_ctx(inp: &Inputs, ov: &PlanOverrides) -> Option<u32> {
    context::largest_ctx(inp.ctx.saturating_sub(1), |ctx| place(&Inputs { ctx, ..*inp }, ov).feasible)
}

/// One plain sentence saying why the plan can't load and what would.
fn infeasible_reason(inp: &Inputs, p: &LoadPlan) -> String {
    // A system memory limit smaller than the context window is named: no
    // context length or expert cache setting gets around the user's own cap.
    let within = |limit| format!(" within the {} system memory limit", gb(limit));
    match (p.max_ctx, inp.ram_limit_short(p)) {
        (Some(max), limit) => format!(
            "A {} token context doesn't fit{}, even with no expert cache: its context memory and compute buffers take {}. The largest context that fits is {} tokens.",
            p.ctx,
            limit.map(within).unwrap_or_default(),
            gb(p.context.total_bytes()),
            max
        ),
        (None, Some(limit)) => format!(
            "No context length fits{}: the smallest context window needs more system memory than that. Raise the limit.",
            within(limit)
        ),
        (None, None) if !inp.allow_ssd && p.tiers.ssd > 0 => match inp.binding_ram_limit() {
            Some(limit) => format!(
                "The weights don't fit in graphics memory and the {} system memory limit, and SSD streaming is off.",
                gb(limit)
            ),
            None => "The weights don't fit in graphics and system memory, and SSD streaming is off.".into(),
        },
        (None, None) => "The model's weights don't fit in the available memory at any context length.".into(),
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod native_tests;
