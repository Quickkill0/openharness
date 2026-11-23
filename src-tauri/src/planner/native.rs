//! Admission and CPU/SSD placement for the native engine, not llama's MoE cache.
use serde::{Deserialize, Serialize};
use super::*;

pub const RAM_RESERVE: u64 = 12 * GIB;
pub const VRAM_RESERVE: u64 = 5 * GIB;
pub(super) const HOST_MARGIN: u64 = 1536 * MIB; // 512 MiB headroom + process/allocator slack
/// The same margin once the dense layers run on the card: the CUDA runtime
/// makes the process itself larger. GLM-5.3-Flash measured 2.5 GiB on an RTX
/// 4090, beside the 512 MiB headroom.
pub(super) const CARD_HOST_MARGIN: u64 = 3584 * MIB;
/// Commit (RAM plus pagefile) left for Windows and other apps. Windows charges
/// the engine's private memory and its graphics-memory tier against commit, so
/// free RAM alone does not say how large the RAM expert cache may be.
pub const COMMIT_RESERVE: u64 = 8 * GIB;
/// Commit the engine process takes beyond the planned context, cache and
/// tier (CUDA runtime, read-ahead and upload buffers, heap). GLM-5.3-Flash
/// at a 32K context measured 2.3 GiB on an RTX 4090.
pub const COMMIT_PROCESS_OVERHEAD: u64 = 3 * GIB;
/// Next-layer experts read ahead per layer. GLM-5.3-Flash measured 13-16 % more
/// tokens per second at 4 than at 0 on an RTX 4090 with a 980 PRO.
const GLM_PREFETCH_DEPTH: u32 = 4;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnginePreference { #[default] Auto, LlamaServer, SsdStream }

#[derive(Debug, Clone, Default)]
pub struct NativeSupport {
    pub installed: bool,
    pub streaming: bool,
    pub cuda_tier: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativePlan {
    /// Whole host envelope, including host context, compute, cache and headroom,
    /// plus what the dense layers and context hold on the card when they run
    /// there: the engine's receipt counts those bytes in the process.
    pub ram_budget_mib: u64,
    pub host_headroom_mib: u64,
    pub expert_cache_mib: u64,
    pub vram_budget_bytes: u64,
    pub vram_reserve_mib: u64,
    pub prefetch_depth: u32,
    /// Windows process envelope (`host.budget_mib`): the host budget plus the
    /// graphics-memory tier, which Windows also charges to the process. Unset,
    /// the engine uses the host budget alone and shrinks the RAM cache to pay
    /// for the tier. `None` where the commit limit is not read.
    #[serde(default)]
    pub host_budget_mib: Option<u64>,
    pub estimate_basis: String,
}
impl NativePlan {
    pub fn extra_args(&self, p: &LoadPlan) -> Vec<String> {
        let mut args: Vec<String> = ["--ssd", "-np", "1", "--metrics", "--no-warmup"].into_iter().map(String::from).collect();
        // Dense layers on the card keep every routed expert on the host for the
        // engine to stream; otherwise only the expert tier touches the card.
        if p.gpu_layers != 0 { args.push("--cpu-moe".into()); } else { args.extend(["--no-op-offload".into(), "--no-kv-offload".into()]); }
        args.extend(["-ctk".into(), p.kv_type.as_arg().into(), "-ctv".into(), p.kv_type_v.as_arg().into(), "-ub".into(), p.ubatch.to_string()]);
        for (key, value) in [
            ("engine.ram_budget_mib", self.ram_budget_mib),
            ("host.headroom_mib", self.host_headroom_mib),
            ("engine.expert_cache_mib", self.expert_cache_mib),
            ("prefetch.depth", self.prefetch_depth as u64),
            ("vram.budget", self.vram_budget_bytes),
        ] { args.extend(["--ssd-set".into(), format!("{key}={value}")]); }
        // CUDA-only setting: a CPU build refuses it even when the value is zero.
        if self.vram_budget_bytes > 0 {
            args.extend(["--ssd-set".into(), format!("vram.reserve_mib={}", self.vram_reserve_mib)]);
        }
        // Windows/macOS-only setting: the Linux engine refuses it.
        if let Some(budget) = self.host_budget_mib {
            args.extend(["--ssd-set".into(), format!("host.budget_mib={budget}")]);
        }
        // A graphics memory limit that leaves nothing on the card: a CUDA build
        // would still create its context there (the llama-server rule).
        if p.overrides.vram_limit_mib.is_some() && p.tiers.vram_total() == 0 {
            args.extend(["--device".into(), "none".into()]);
        }
        args
    }
}

fn refuse(mut p: LoadPlan, why: String) -> LoadPlan {
    p.feasible = false;
    p.infeasible_reason = Some(why.clone());
    p.warnings.push(why);
    p.evict_alternative = None;
    p
}

/// Match oh-runtime.cpp: combine all roles of each layer's expert, then page-align
/// that slot as a whole. Quant types may vary by layer; use its actual bytes.
pub(super) fn minimum_vram_tier(m: &ModelAccounting) -> u64 {
    m.layers.iter().filter(|l| l.kind != super::accounting::LayerKind::Mtp && l.n_expert > 0)
        .map(|l| (l.expert_bytes / u64::from(l.n_expert)).div_ceil(4096) * 4096).sum()
}

/// Device-only quote from oh-runtime.cpp::price_vram_tier. Missing separate
/// expert tensors cannot be priced safely, so leave their tier disabled.
pub(super) fn tier_device_bytes(m: &ModelAccounting) -> Option<u64> {
    let layouts: Vec<_> = m.layers.iter().filter(|l| l.kind != super::accounting::LayerKind::Mtp && l.n_expert > 0)
        .map(|l| l.expert_layout.as_ref()).collect::<Option<_>>()?;
    let first = layouts.first()?;
    let (latent, hidden) = (first.latent, first.hidden);
    let pairs: std::collections::BTreeSet<_> = layouts.iter().map(|l| (l.up_type, l.down_type)).collect();
    // GGML_TYPE_Q8_1 has 36-byte blocks of 32 elements; pad rows to 512.
    let q8_1 = |columns: u64| columns.div_ceil(512) * 512 / 32 * 36;
    let is_float = |ty| matches!(ty, 1 | 30); // F16 / BF16
    Some(pairs.into_iter().map(|(up, down)| {
        16 * (latent * 4 * 2 + hidden * 4 * 3) +
            (if is_float(up) { 0 } else { q8_1(latent) }).max(if is_float(down) { 0 } else { 16 * q8_1(hidden) })
    }).sum())
}

pub(super) fn select(m: &ModelAccounting, hw: &HardwareProfile, _budget: &VramBudget, ov: &PlanOverrides, support: &NativeSupport, cal: Option<&ModelCalibration>, llama: LoadPlan) -> LoadPlan {
    let routed = m.expert_bytes > 0 && m.moe_layers > 0;
    let needed = routed && (llama.tiers.ssd > 0 || !llama.feasible);
    if ov.engine == EnginePreference::LlamaServer {
        // A plan llama-server already refuses keeps its own reason (a memory
        // limit the context cannot fit is not fixed by changing engines).
        return if routed && llama.tiers.ssd > 0 && llama.feasible {
            refuse(llama, "This llama-server plan overcommits RAM. Choose auto or the native streaming engine.".into())
        } else { llama };
    }
    if ov.engine == EnginePreference::Auto && !needed { return llama; }
    if !routed { return refuse(llama, "Native streaming requires routed experts; this is not a routed-expert model.".into()); }
    let mut p = llama;
    p.engine_kind = EngineKind::SsdStream;
    if !support.installed { return refuse(p, "The native engine isn't installed. Install it in engine/native to stream this model without overcommitting RAM.".into()); }
    if !support.streaming {
        return refuse(p, support.error.clone().unwrap_or_else(|| "The native engine does not report ssd_expert_streaming.".into()));
    }
    if ov.allow_ssd == Some(false) { return refuse(p, "SSD streaming is disabled in this plan.".into()); }
    if let Some(why) = &m.unsupported { return refuse(p, format!("Can't estimate native context memory: {why}")); }
    // No draft/MTP loading. The engine decides which expert tensor layouts can
    // stream and its refusal is preserved by the sidecar.
    p.gpu_layers = 0;
    p.gpu_expert_layers = 0;
    p.cpu_expert_layers = m.moe_layers;
    p.n_cpu_moe = None;
    p.expert_cache_mib = 0; // llama-only cache; native cache has a separate field
    p.ubatch = UBATCH_DEFAULT;
    p.n_parallel = 1;
    p.evict_alternative = None;
    p.warnings.clear();
    p.kv_type = ov.kv_type.unwrap_or(KvType::F16);
    p.kv_type_v = if m.mla_layers > 0 { p.kv_type } else { ov.kv_type_v.unwrap_or(p.kv_type) };
    let ram_free = hw.ram_available_bytes.saturating_sub(RAM_RESERVE);
    let ram_limit = ov.ram_limit_mib.map(|l| l.saturating_mul(MIB));
    let ram = ram_limit.map_or(ram_free, |l| l.min(ram_free)) / MIB * MIB;
    // NVML is fresh. The capability cache identifies a usable CUDA backend,
    // never supplies a stale free-memory number. The orchestrator has already
    // added back the text model this load replaces; no other resident is reclaimed.
    let device = tier_device_bytes(m);
    // Keep the CUDA context allowance in addition to the runtime's own driver
    // margin and per-type compute buffers, then give the remainder to slots.
    let vram_overhead = CUDA_CONTEXT_BYTES + 64 * MIB + device.unwrap_or(0);
    let vram_free = hw.gpu.as_ref().map_or(0, |g| g.vram_free_bytes.saturating_sub(VRAM_RESERVE));
    // The limit covers everything the model puts on the card, overhead included.
    let vram_allowed = ov.vram_limit_mib.map_or(vram_free, |l| l.saturating_mul(MIB).min(vram_free));
    let tier_capable = support.cuda_tier && device.is_some();
    // oh-runtime.cpp compute_vram_receipt: one slot in EVERY routed decode
    // layer. Each slot is that layer's gate/up/down bytes per expert, not
    // the largest tensor or one average slot for the whole model.
    let min_tier = minimum_vram_tier(m);
    // The expert tier once `spent` bytes of the allowance hold something else.
    let tier_after = |spent: u64| {
        let tier = if tier_capable { vram_allowed.saturating_sub(vram_overhead.saturating_add(spent)) } else { 0 }.min(64 * GIB) / MIB * MIB;
        if tier >= min_tier && min_tier > 0 { tier } else { 0 }
    };
    let host_tier = tier_after(0);
    let limit_drops_tier = tier_capable && min_tier > 0 && host_tier == 0 && vram_free.saturating_sub(vram_overhead) >= min_tier;
    let dense = m.always_bytes + m.token_embd_bytes;
    // On the card a tied output head is a second copy of the token embeddings.
    let card_weights = m.always_bytes + if m.output_tied { m.token_embd_bytes } else { 0 };
    let context = |ctx| context_memory(m, &ContextParams { ctx, kv_k: p.kv_type, kv_v: p.kv_type_v, n_parallel: 1, ubatch: p.ubatch, experts_on_host: false });
    let margined = |bytes: u64| (bytes as f64 * (1.0 + COMPUTE_MARGIN_FRAC)) as u64;
    // Measured GLM runs on an RTX 4090: 2.25 tok/s with a 12 GiB tier and the
    // dense layers on the CPU, 3.5 tok/s with a 10 GiB tier and them on the
    // card. Never scale above the seed on unmeasured hardware, and never use
    // llama's hit curve.
    let seed = |on_card: bool| if on_card { (3.5, 10 * GIB) } else { (2.25, 12 * GIB) };
    let seeded_tps = |on_card: bool, tier: u64| {
        let (tps, full) = seed(on_card);
        tps * if tier == 0 { 0.25 } else { (tier as f64 / full as f64).min(1.0) }
    };
    // (dense layers and context on the card, expert tier). The dense layers go
    // on the card when the tier left beside them still seeds at least the
    // speed of the CPU layout's larger tier; a smaller tier there is unmeasured.
    // (0, tier) leaves them on the CPU, which a model with almost no dense
    // weights also needs when RAM is too short for the larger process.
    let card_cost = |c: &ContextMemory| m.token_embd_bytes.saturating_add(margined(c.host_compute_bytes)).saturating_add(CARD_HOST_MARGIN);
    let layout = |c: &ContextMemory| {
        let on_card = card_weights.saturating_add(c.cache_bytes()).saturating_add(margined(c.compute_bytes));
        match tier_after(on_card) {
            tier if tier > 0 && card_cost(c) <= ram && seeded_tps(true, tier) >= seeded_tps(false, host_tier) => (on_card, tier),
            _ => (0, host_tier),
        }
    };
    // RAM the engine holds besides the expert cache.
    let cost = |c: &ContextMemory| if layout(c).0 > 0 { card_cost(c) } else {
        dense.saturating_add(c.cache_bytes()).saturating_add(margined(c.compute_bytes)).saturating_add(HOST_MARGIN)
    };
    // Its commit: the mapped model file takes RAM but none, and the process
    // overhead is charged once, in the commit room.
    let private = |c: &ContextMemory| if layout(c).0 > 0 { margined(c.host_compute_bytes).saturating_add(HOST_MARGIN) } else { cost(c).saturating_sub(dense) };
    // Windows charges everything the engine puts on the card to its commit.
    let vram_commit = |c: &ContextMemory| match layout(c) { (_, 0) => 0, (on_card, tier) => tier + vram_overhead + on_card };
    let commit_room = |c: &ContextMemory| hw.commit_available_bytes.map(|a| a.saturating_sub(COMMIT_RESERVE + COMMIT_PROCESS_OVERHEAD).saturating_sub(vram_commit(c)));
    let fits_ram = |ctx| cost(&context(ctx)) <= ram;
    let fits_commit = |ctx| { let c = context(ctx); commit_room(&c).map_or(true, |room| private(&c) <= room) };
    let fits = |ctx| fits_ram(ctx) && fits_commit(ctx);
    let cap = ctx_candidates(m, ov).into_iter().max().unwrap_or(MIN_CTX);
    // Arithmetic only, in the same 256-token cells as context_memory.
    let max_ctx = context::largest_ctx(cap, fits);
    p.ctx = if ov.ctx.is_some() { cap } else { max_ctx.unwrap_or(256).min(cap).max(256) };
    p.context = context(p.ctx);
    p.max_ctx = if !fits(p.ctx) { max_ctx } else { None };
    p.feasible = fits(p.ctx);
    p.infeasible_reason = None;
    let (on_card, vram) = layout(&p.context);
    p.gpu_layers = if on_card > 0 { -1 } else { 0 };
    let cache_by_ram = ram.saturating_sub(cost(&p.context)).min(m.expert_bytes) / MIB * MIB;
    let cache = commit_room(&p.context).map_or(cache_by_ram, |room| cache_by_ram.min(room.saturating_sub(private(&p.context)) / MIB * MIB));
    p.tiers = TierBytes { vram_expert_cache: vram, vram_overhead: if vram > 0 { vram_overhead } else { 0 }, ram: ram.min(cost(&p.context) + cache), ssd: m.expert_bytes.saturating_sub(cache + vram), ..Default::default() };
    if on_card > 0 {
        p.tiers.vram_weights = card_weights;
        p.tiers.vram_kv = p.context.cache_bytes();
        p.tiers.vram_overhead += margined(p.context.compute_bytes);
    }
    p.tier_mix = TierMix::Ssd;
    let (seed_tps, seed_tier) = seed(on_card > 0);
    let raw = seeded_tps(on_card > 0, vram) * (hw.bandwidth.ssd_gbps / 6.8).clamp(0.05, 1.0) * (hw.bandwidth.ram_gbps / 60.0).clamp(0.1, 1.0);
    let c = cal.and_then(|c| c.get(&TierMix::Ssd));
    let tps = raw * c.map_or(1.0, Calibration::decode_factor);
    p.speed = SpeedEstimate { decode_tps: tps, raw_decode_tps: raw, prompt_tps: 0.0, raw_prompt_tps: 0.0, calibration_samples: c.map_or(0, |c| c.samples), label: SpeedLabel::for_tps(tps), bottleneck: speed::Bottleneck::Ssd };
    let basis = format!("Estimate seeded from GLM-5.3-Flash at {seed_tps} tok/s (RTX 4090, {} GiB expert tier); conservatively scaled. Prefill speed is unknown. Live /metrics calibrates decode when available.", seed_tier / GIB);
    p.warnings.push(basis.clone());
    p.warnings.push(if on_card > 0 {
        "Dense layers and the context cache run in graphics memory; routed experts are prefilled on the CPU. Expert layouts (including F32) are validated by the engine; MTP/draft is not enabled."
    } else {
        "Dense layers, prefill and KV stay on the CPU. Expert layouts (including F32) are validated by the engine; MTP/draft is not enabled."
    }.into());
    if limit_drops_tier {
        p.warnings.push(format!("The graphics memory limit is below the {} this model's expert tier needs at minimum, so no experts are kept in graphics memory.", gb(min_tier + vram_overhead)));
    }
    if cache_by_ram.saturating_sub(cache) >= GIB {
        p.warnings.push(format!("Windows' commit limit (RAM plus pagefile) holds the system memory expert cache to {}. A larger pagefile would let it use {} more of the free RAM.", gb(cache), gb(cache_by_ram - cache)));
    }
    let prefetch_depth = match m.architecture.as_str() { "qwen3moe" => 2, "glm5-next" => GLM_PREFETCH_DEPTH, _ => 0 };
    // The engine's receipt measures the process, and Windows counts what the
    // process holds on the card in that, so the envelope carries those bytes.
    let envelope = ram + on_card;
    let host_budget_mib = hw.commit_available_bytes.map(|_| ram / MIB + (vram_commit(&p.context) + COMMIT_PROCESS_OVERHEAD).div_ceil(MIB));
    p.native = Some(NativePlan { ram_budget_mib: envelope / MIB, host_headroom_mib: 512, expert_cache_mib: cache / MIB, vram_budget_bytes: vram, vram_reserve_mib: VRAM_RESERVE / MIB, prefetch_depth, host_budget_mib, estimate_basis: basis });
    p.args = vec!["-c".into(), p.ctx.to_string(), "-ngl".into(), if on_card > 0 { "all" } else { "0" }.into(), "-b".into(), p.ubatch.to_string()];
    p.args.extend(p.extra_args());
    if !p.feasible {
        let within = if !fits_ram(p.ctx) {
            match ram_limit { Some(l) if l < ram_free => format!("within the {} system memory limit", gb(l)), _ => "while leaving 12 GiB RAM free".to_string() }
        } else { "within Windows' commit limit (RAM plus pagefile) while leaving 8 GiB of it free".to_string() };
        let why = match p.max_ctx { Some(n) => format!("Native host context does not fit {within}. Largest fitting context: {n} tokens."), None => format!("No native context fits {within}; dense weights, context, compute and process headroom exceed the budget.") };
        return refuse(p, why);
    }
    p
}
