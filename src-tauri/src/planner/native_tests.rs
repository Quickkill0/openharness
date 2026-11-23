use super::*;
use super::native::{minimum_vram_tier, EnginePreference, NativeSupport, CARD_HOST_MARGIN, COMMIT_PROCESS_OVERHEAD, COMMIT_RESERVE, HOST_MARGIN, RAM_RESERVE, VRAM_RESERVE};
use super::hardware::{Bandwidths, GpuInfo};

fn model(name: &str) -> ModelAccounting {
    accounting::account_model(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf").join(name)).unwrap()
}
fn glm() -> ModelAccounting { model("glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf") }
fn qwen_experts(latent: u64, hidden: u64, pairs: &[(u32, u32)]) -> ModelAccounting {
    use crate::models::gguf::{ggml_type_block, GgufTensorHeader, TensorInfo};
    let mut tensors = Vec::new();
    for (layer, &(up, down)) in pairs.iter().enumerate() {
        for (role, ty) in [("up", up), ("gate", up), ("down", down)] {
            let dims = if role == "down" { vec![hidden, latent, 8] } else { vec![latent, hidden, 8] };
            let (block, bytes) = ggml_type_block(ty).unwrap();
            tensors.push(TensorInfo { name: format!("blk.{layer}.ffn_{role}_exps.weight"),
                bytes: dims.iter().product::<u64>() / block * bytes, dims, ggml_type: ty, offset: 0 });
        }
        tensors.push(TensorInfo { name: format!("blk.{layer}.attn_k.weight"), dims: vec![latent, 64],
            ggml_type: 1, offset: 0, bytes: latent * 64 * 2 });
    }
    let header = GgufTensorHeader::synthetic(&[("general.architecture", "qwen3moe")], &[
        ("qwen3moe.block_count", pairs.len() as u64), ("qwen3moe.embedding_length", latent),
        ("qwen3moe.expert_feed_forward_length", hidden), ("qwen3moe.expert_count", 8),
        ("qwen3moe.expert_used_count", 2), ("qwen3moe.attention.head_count", 4),
        ("qwen3moe.attention.head_count_kv", 4),
    ], tensors);
    accounting::account_headers(&[header]).unwrap()
}
fn tier_keep(m: &ModelAccounting) -> u64 {
    VRAM_RESERVE + CUDA_CONTEXT_BYTES + 64 * MIB + super::native::tier_device_bytes(m).unwrap()
}
/// What the dense layers, the context cache and its compute hold on the card.
fn card(m: &ModelAccounting, p: &LoadPlan) -> u64 {
    p.tiers.vram_total().saturating_sub(p.tiers.vram_expert_cache + tier_keep(m) - VRAM_RESERVE)
}
fn has(args: &[String], flag: &str) -> bool { args.iter().any(|a| a == flag) }
fn hardware() -> HardwareProfile {
    HardwareProfile { gpu: Some(GpuInfo { name: "RTX 4090".into(), vram_total_bytes: 24 * GIB, vram_free_bytes: 23 * GIB }), ram_total_bytes: 64 * GIB, ram_available_bytes: 40 * GIB, commit_available_bytes: None, storage: None,
        bandwidth: Bandwidths { vram_gbps: 1008.0, pcie_gbps: 25.0, ram_gbps: 60.0, ssd_gbps: 6.8, ssd_measured: true }, cpu_threads: 32 }
}
fn support() -> NativeSupport { NativeSupport { installed: true, streaming: true, cuda_tier: true, error: None } }
fn native(m: &ModelAccounting, h: &HardwareProfile, ov: &PlanOverrides, s: &NativeSupport) -> LoadPlan {
    let b = h.gpu.as_ref().map_or_else(VramBudget::no_gpu, |g| VramBudget { total_bytes: g.vram_total_bytes, free_bytes: g.vram_free_bytes, reclaimable: vec![] });
    plan_with_native(m, h, &b, ov, None, s, None)
}

#[test]
fn native_selection_requires_installed_capability_and_keeps_fitting_llama() {
    let h = hardware();
    let m = glm();
    let missing = native(&m, &h, &Default::default(), &NativeSupport::default());
    assert!(!missing.feasible);
    assert!(missing.infeasible_reason.unwrap().contains("isn't installed"));
    let unsupported = native(&m, &h, &Default::default(), &NativeSupport { installed: true, ..Default::default() });
    assert!(!unsupported.feasible);
    assert!(unsupported.infeasible_reason.unwrap().contains("ssd_expert_streaming"));
    let p = native(&m, &h, &Default::default(), &support());
    assert!(p.feasible, "{p:#?}");
    assert_eq!(p.engine_kind, EngineKind::SsdStream);
    assert!(p.native.is_some());
    // Mutation guard: the old admission/placement is observably not this result.
    assert_eq!(llama_plan(&m, &h, &VramBudget::no_gpu(), &Default::default(), None).engine_kind, EngineKind::LlamaServer);
    let small = model("Qwen3-8B-Q4_K_M.header.gguf");
    let p = native(&small, &h, &Default::default(), &support());
    assert!(p.feasible);
    assert_eq!(p.engine_kind, EngineKind::LlamaServer);
    assert!(p.native.is_none());
}

#[test]
fn native_overrides_never_silently_overcommit() {
    let m = glm(); let h = hardware();
    let forced_llama = native(&m, &h, &PlanOverrides { engine: EnginePreference::LlamaServer, ..Default::default() }, &support());
    assert!(!forced_llama.feasible);
    assert!(forced_llama.infeasible_reason.unwrap().contains("overcommits"));
    let dense = model("Qwen3-8B-Q4_K_M.header.gguf");
    let forced_native = native(&dense, &h, &PlanOverrides { engine: EnginePreference::SsdStream, ..Default::default() }, &support());
    assert!(!forced_native.feasible);
    assert!(forced_native.infeasible_reason.unwrap().contains("routed"));
    let disabled = native(&m, &h, &PlanOverrides { allow_ssd: Some(false), ..Default::default() }, &support());
    assert!(!disabled.feasible);
    assert!(disabled.infeasible_reason.unwrap().contains("disabled"));
}

#[test]
fn forced_llama_keeps_a_memory_limit_refusal_and_still_rejects_paging() {
    let m = model("Qwen3.6-35B-A3B-UD-Q4_K_M.header.gguf"); let h = hardware();
    let forced = |ov: PlanOverrides| native(&m, &h, &PlanOverrides { engine: EnginePreference::LlamaServer, ctx: Some(4096), ..ov }, &support());
    // The context cannot fit the user's own limit: changing engines would not
    // help, so the refusal names the limit.
    let limited = forced(PlanOverrides { ram_limit_mib: Some(0), ..Default::default() });
    assert!(!limited.feasible);
    assert_eq!(limited.engine_kind, EngineKind::LlamaServer);
    let why = limited.infeasible_reason.as_deref().unwrap();
    assert!(why.contains("0.0 GB system memory limit") && !why.contains("overcommits"), "{why}");
    assert!(!limited.warnings.iter().any(|w| w.contains("overcommits")), "{:?}", limited.warnings);
    // The context fits and only the weights spill: llama-server would page
    // them, which is still refused in favour of the streaming engine.
    let paging = forced(PlanOverrides { ram_limit_mib: Some(2048), vram_limit_mib: Some(6 * 1024), ..Default::default() });
    assert!(paging.tiers.ssd > 0, "{:?}", paging.tiers);
    assert!(!paging.feasible);
    assert!(paging.infeasible_reason.as_deref().unwrap().contains("overcommits"), "{:?}", paging.infeasible_reason);
}

#[test]
fn native_budgets_leave_machine_floors_and_dense_layers_run_on_the_card() {
    let h = hardware(); let m = glm();
    let p = native(&m, &h, &Default::default(), &support());
    let n = p.native.as_ref().unwrap();
    let ram = h.ram_available_bytes - RAM_RESERVE;
    let vram = h.gpu.as_ref().unwrap().vram_free_bytes - VRAM_RESERVE;
    assert!(p.tiers.ram + RAM_RESERVE <= h.ram_available_bytes);
    assert!(p.tiers.vram_total() <= vram);
    assert_eq!(n.vram_reserve_mib, 5120);
    assert_eq!(p.context, context_memory(&m, &ContextParams { ctx: p.ctx, kv_k: p.kv_type, kv_v: p.kv_type_v, ubatch: p.ubatch, n_parallel: 1, experts_on_host: false }));
    // Dense layers, the context cache and its compute sit on the card...
    assert_eq!(p.gpu_layers, -1);
    assert!(p.args.windows(2).any(|w| w[0] == "-ngl" && w[1] == "all"), "{:?}", p.args);
    assert_eq!(p.tiers.vram_weights, m.always_bytes); // token embeddings stay in RAM
    assert_eq!(p.tiers.vram_kv, p.context.cache_bytes());
    let on_card = card(&m, &p);
    assert_eq!(on_card, m.always_bytes + p.context.cache_bytes() + (p.context.compute_bytes as f64 * 1.08) as u64);
    // ...the expert tier takes the rest of it...
    assert!(n.vram_budget_bytes >= minimum_vram_tier(&m));
    assert!(vram - p.tiers.vram_total() < MIB);
    // ...RAM beside the cache holds the token embeddings, host compute and the process...
    assert_eq!(p.tiers.ram - n.expert_cache_mib * MIB, m.token_embd_bytes + (p.context.host_compute_bytes as f64 * 1.08) as u64 + CARD_HOST_MARGIN);
    // ...and the engine's envelope carries what Windows charges the process for the card.
    assert_eq!(n.ram_budget_mib, (ram + on_card) / MIB);
    assert!(p.speed.raw_decode_tps <= 3.5);
    assert_eq!(p.speed.prompt_tps, 0.0);
    assert!(n.estimate_basis.contains("Estimate") && n.estimate_basis.contains("3.5 tok/s"), "{}", n.estimate_basis);
}

#[test]
fn native_dense_layers_stay_on_the_cpu_unless_the_tier_beside_them_seeds_as_fast() {
    let m = glm(); let mut h = hardware();
    h.commit_available_bytes = Some(200 * GIB);
    let ov = PlanOverrides { ctx: Some(8192), ..Default::default() };
    let ram = h.ram_available_bytes - RAM_RESERVE;
    let dense = card(&m, &native(&m, &h, &ov, &support()));
    assert!(minimum_vram_tier(&m) < 6 * GIB && dense > 6 * GIB);
    // Beside the dense layers the card would keep a 6 GiB tier. That seeds
    // slower than the 12 GiB or more the CPU layout gets, so experts take it all.
    h.gpu.as_mut().unwrap().vram_free_bytes = tier_keep(&m) + dense + 6 * GIB;
    let cpu = native(&m, &h, &ov, &support());
    let n = cpu.native.as_ref().unwrap();
    assert!(cpu.feasible);
    assert_eq!(cpu.gpu_layers, 0);
    assert!(cpu.args.windows(2).any(|w| w[0] == "-ngl" && w[1] == "0"), "{:?}", cpu.args);
    assert_eq!(cpu.tiers.vram_weights + cpu.tiers.vram_kv + card(&m, &cpu), 0);
    assert_eq!(n.vram_budget_bytes, (dense + 6 * GIB) / MIB * MIB);
    let a = cpu.extra_args();
    assert!(has(&a, "--no-op-offload") && has(&a, "--no-kv-offload") && !has(&a, "--cpu-moe"), "{a:?}");
    assert!(cpu.tiers.ram - n.expert_cache_mib * MIB >= m.always_bytes + m.token_embd_bytes + cpu.context.cache_bytes() + HOST_MARGIN);
    assert_eq!(n.ram_budget_mib * MIB, ram);
    assert_eq!(n.host_budget_mib, Some(ram / MIB + (cpu.tiers.vram_total() + COMMIT_PROCESS_OVERHEAD).div_ceil(MIB)));
    assert!(cpu.speed.raw_decode_tps <= 2.25);
    assert!(n.estimate_basis.contains("2.25 tok/s"), "{}", n.estimate_basis);
    // A 7 GiB tier beside them seeds faster, so the dense layers move over.
    h.gpu.as_mut().unwrap().vram_free_bytes += GIB;
    let on = native(&m, &h, &ov, &support());
    assert_eq!(on.gpu_layers, -1);
    assert_eq!(on.native.as_ref().unwrap().vram_budget_bytes, 7 * GIB);
    assert_eq!(card(&m, &on), dense);
    assert!(on.speed.raw_decode_tps > cpu.speed.raw_decode_tps);
    let a = on.extra_args();
    assert!(has(&a, "--cpu-moe") && !has(&a, "--no-op-offload") && !has(&a, "--no-kv-offload"), "{a:?}");
    assert!(on.warnings.iter().any(|w| w.contains("run in graphics memory")), "{:?}", on.warnings);
    // A graphics memory limit below the dense layers leaves them on the CPU too.
    let limited = native(&m, &hardware(), &PlanOverrides { vram_limit_mib: Some(6 * 1024), ..ov }, &support());
    assert_eq!(limited.gpu_layers, 0);
    assert!(limited.tiers.vram_expert_cache > 5 * GIB && limited.tiers.vram_total() <= 6 * GIB, "{:?}", limited.tiers);
}

#[test]
fn native_small_dense_model_keeps_the_cpu_layout_when_ram_is_short_for_the_card_process() {
    // Almost no dense weights: the CPU layout needs less RAM than the larger
    // process the card layout brings.
    let m = qwen_experts(128, 64, &[(8, 8); 40]);
    let ov = PlanOverrides { engine: EnginePreference::SsdStream, ctx: Some(256), ..Default::default() };
    let mut h = hardware();
    h.ram_available_bytes = RAM_RESERVE + 2 * GIB;
    let cpu = native(&m, &h, &ov, &support());
    assert!(cpu.feasible, "{:?}", cpu.infeasible_reason);
    assert_eq!(cpu.gpu_layers, 0);
    assert!(cpu.native.as_ref().unwrap().vram_budget_bytes > 0);
    h.ram_available_bytes = RAM_RESERVE + 4 * GIB;
    let on = native(&m, &h, &ov, &support());
    assert!(on.feasible, "{:?}", on.infeasible_reason);
    assert_eq!(on.gpu_layers, -1);
    assert_eq!(on.native.as_ref().unwrap().ram_budget_mib, (4 * GIB + card(&m, &on)) / MIB);
}

#[test]
fn native_context_slider_consumes_the_tier_on_the_card_and_the_ram_cache_on_the_cpu() {
    let h = hardware(); let m = glm();
    let at = |ctx, vram_limit_mib| native(&m, &h, &PlanOverrides { ctx: Some(ctx), vram_limit_mib, ..Default::default() }, &support());
    let (small, big) = (at(4096, None), at(65536, None));
    let (sn, bn) = (small.native.as_ref().unwrap(), big.native.as_ref().unwrap());
    assert!(small.feasible && big.feasible);
    assert_eq!((small.gpu_layers, big.gpu_layers), (-1, -1));
    // A context that leaves too small a tier moves the dense layers back.
    assert_eq!(at(262144, None).gpu_layers, 0);
    assert!(big.context.total_bytes() > small.context.total_bytes());
    assert!(big.tiers.vram_kv > small.tiers.vram_kv + GIB);
    assert!(bn.vram_budget_bytes + GIB < sn.vram_budget_bytes);
    assert!(big.tiers.ssd > small.tiers.ssd);
    // The card is as full either way, and RAM gives up only the host compute.
    assert!(big.tiers.vram_total().abs_diff(small.tiers.vram_total()) < MIB);
    assert!(bn.expert_cache_mib <= sn.expert_cache_mib && sn.expert_cache_mib - bn.expert_cache_mib < 1024);
    // Nothing on the card: the context lives in RAM and the cache pays for it.
    let (small, big) = (at(4096, Some(0)), at(262144, Some(0)));
    assert!(small.feasible && big.feasible);
    assert!(big.native.as_ref().unwrap().expert_cache_mib + 1024 < small.native.as_ref().unwrap().expert_cache_mib);
    assert!(big.tiers.ssd > small.tiers.ssd);
    assert_eq!(big.tiers.vram_total() + small.tiers.vram_total(), 0);
}

#[test]
fn native_max_context_refusal_and_no_context_when_below_floor() {
    let mut h = hardware(); let m = glm();
    h.ram_available_bytes = 24 * GIB;
    let p = native(&m, &h, &PlanOverrides { ctx: Some(1048576), ..Default::default() }, &support());
    assert!(!p.feasible);
    let max = p.max_ctx.unwrap();
    assert!(p.infeasible_reason.as_ref().unwrap().contains(&max.to_string()));
    let ok = native(&m, &h, &PlanOverrides { ctx: Some(max), ..Default::default() }, &support());
    assert!(ok.feasible);
    h.ram_available_bytes = RAM_RESERVE - 1;
    let p = native(&m, &h, &Default::default(), &support());
    assert!(!p.feasible);
    assert_eq!(p.max_ctx, None);
    assert_eq!(p.native.unwrap().ram_budget_mib, 0);
}

#[test]
fn native_cpu_build_no_gpu_or_low_free_vram_has_no_tier_setting() {
    let m = glm();
    for mode in 0..3 {
        let mut h = hardware(); let mut s = support();
        if mode == 0 { h.gpu = None; }
        if mode == 1 { s.cuda_tier = false; }
        if mode == 2 { h.gpu.as_mut().unwrap().vram_free_bytes = VRAM_RESERVE; }
        let p = native(&m, &h, &Default::default(), &s);
        assert!(p.feasible);
        assert_eq!(p.tiers.vram_total(), 0);
        let n = p.native.as_ref().unwrap();
        assert_eq!(n.vram_budget_bytes, 0);
        assert!(p.extra_args().contains(&"vram.budget=0".to_string()));
        assert!(!p.extra_args().iter().any(|a| a.starts_with("vram.reserve_mib=")));
        // No limit was set, so the device choice stays the engine's.
        assert!(!p.extra_args().iter().any(|a| a == "--device"));
    }
}

#[test]
fn native_tier_requires_one_real_slot_in_every_routed_layer() {
    let m = glm();
    let minimum = super::native::minimum_vram_tier(&m);
    assert!(minimum > MIB);
    let mut h = hardware();
    for bytes in [MIB, minimum.saturating_sub(MIB)] {
        h.gpu.as_mut().unwrap().vram_free_bytes = VRAM_RESERVE + CUDA_CONTEXT_BYTES + bytes;
        let p = native(&m, &h, &Default::default(), &support());
        assert!(p.feasible);
        assert_eq!(p.native.unwrap().vram_budget_bytes, 0);
        assert_eq!(p.tiers.vram_total(), 0);
    }
    h.gpu.as_mut().unwrap().vram_free_bytes = tier_keep(&m) + minimum.div_ceil(MIB) * MIB;
    let p = native(&m, &h, &Default::default(), &support());
    assert!(p.native.unwrap().vram_budget_bytes >= minimum);
}

#[test]
fn native_qwen_non_page_aligned_slots_fall_back_to_cpu_below_runtime_minimum() {
    // Q8_0: raw three-role slot 26112 -> 28672 page-aligned bytes per layer.
    let m = qwen_experts(128, 64, &[(8, 8); 40]);
    assert_eq!(m.layers[0].expert_bytes / 8, 26_112);
    assert_eq!(super::native::minimum_vram_tier(&m), 1_146_880);
    let mut h = hardware();
    let ov = PlanOverrides { engine: EnginePreference::SsdStream, ..Default::default() };
    h.gpu.as_mut().unwrap().vram_free_bytes = tier_keep(&m) + 1_048_576;
    let p = native(&m, &h, &ov, &support());
    assert!(p.feasible);
    assert_eq!(p.native.as_ref().unwrap().vram_budget_bytes, 0);
    assert!(p.extra_args().contains(&"vram.budget=0".to_string()));
    h.gpu.as_mut().unwrap().vram_free_bytes += MIB;
    assert_eq!(native(&m, &h, &ov, &support()).native.unwrap().vram_budget_bytes, 2_097_152);
}

#[test]
fn native_wide_mixed_quant_device_quote_limits_tier_and_matches_receipts() {
    // Nine supported types, one float and eight quantized, in all 81 ordered
    // up/down pairs. Gate follows up. Quote must deduplicate pairs, not types.
    let types = [1, 2, 3, 6, 7, 8, 10, 11, 12];
    let pairs: Vec<_> = types.iter().flat_map(|&up| types.iter().map(move |&down| (up, down))).collect();
    let mut m = qwen_experts(16_384, 16_384, &pairs);
    // Dimensions must come from the actual up tensor, NOT these metadata fields.
    m.n_embd = 128;
    m.expert_ffn = 64;
    assert_eq!(super::native::tier_device_bytes(&m), Some(446_054_400));
    let mut h = hardware();
    h.ram_total_bytes = 1024 * GIB;
    h.ram_available_bytes = 512 * GIB;
    h.gpu.as_mut().unwrap().vram_total_bytes = 64 * GIB;
    h.gpu.as_mut().unwrap().vram_free_bytes = 54_927 * MIB;
    let p = native(&m, &h, &PlanOverrides { engine: EnginePreference::SsdStream, ctx: Some(256), ..Default::default() }, &support());
    assert!(p.feasible, "{:?}", p.infeasible_reason);
    // 54,927 - 5,120 reserve - 450 CUDA - 64 driver - 425.390625 buffers, less
    // the dense layers and context this model also puts on the card.
    let on_card = card(&m, &p);
    assert!(p.gpu_layers == -1 && on_card > 0 && on_card < GIB, "{on_card}");
    let tier = (49_293 * MIB - 446_054_400 - on_card) / MIB * MIB;
    assert_eq!(p.native.as_ref().unwrap().vram_budget_bytes, tier);
    assert_eq!(p.tiers.vram_expert_cache, tier);
    assert_eq!(p.tiers.vram_overhead - (p.context.compute_bytes as f64 * 1.08) as u64, 985_022_464);
    assert!(p.extra_args().contains(&format!("vram.budget={tier}")));
    assert!(p.extra_args().contains(&"vram.reserve_mib=5120".to_string()));
    assert!(p.tiers.vram_total() + VRAM_RESERVE <= h.gpu.as_ref().unwrap().vram_free_bytes);
}

#[test]
fn native_driver_buffers_can_remove_an_otherwise_admissible_tier() {
    let m = qwen_experts(128, 64, &[(8, 8); 40]);
    // One pair despite 40 layers, with Q8_1 rows padded from 128/64 to 512.
    assert_eq!(super::native::tier_device_bytes(&m), Some(37_888));
    let float = qwen_experts(128, 64, &[(30, 30)]);
    assert_eq!(super::native::tier_device_bytes(&float), Some(28_672)); // BF16 has no Q8_1 operand
    let mut h = hardware();
    h.gpu.as_mut().unwrap().vram_free_bytes = VRAM_RESERVE + CUDA_CONTEXT_BYTES + 2_097_152;
    assert!(2_097_152 >= super::native::minimum_vram_tier(&m)); // old admission enabled it
    let p = native(&m, &h, &PlanOverrides { engine: EnginePreference::SsdStream, ..Default::default() }, &support());
    assert!(p.feasible);
    assert_eq!(p.native.as_ref().unwrap().vram_budget_bytes, 0);
    assert_eq!(p.tiers.vram_total(), 0);
    assert!(p.extra_args().contains(&"vram.budget=0".to_string()));
    assert!(!p.extra_args().iter().any(|a| a.starts_with("vram.reserve_mib=")));
}

#[test]
fn native_glm_device_allowance_only_reduces_tier_by_small_expected_amount() {
    let m = glm();
    let h = hardware();
    assert!(super::native::tier_device_bytes(&m).unwrap() <= 25 * MIB);
    let p = native(&m, &h, &Default::default(), &support());
    let old = h.gpu.as_ref().unwrap().vram_free_bytes - VRAM_RESERVE - CUDA_CONTEXT_BYTES - card(&m, &p);
    let delta = old - p.native.as_ref().unwrap().vram_budget_bytes;
    assert!((64 * MIB..=90 * MIB).contains(&delta), "delta {delta}");
    assert_eq!(p.tiers.vram_expert_cache, p.native.as_ref().unwrap().vram_budget_bytes);
}

#[test]
fn native_flags_pair_dense_offload_with_host_experts_and_prefetch_is_arch_specific() {
    let mut m = glm(); let h = hardware();
    let p = native(&m, &h, &Default::default(), &support());
    let a = p.extra_args();
    assert!(a.contains(&"--ssd".into()));
    for flag in ["--fit", "--moe-cache-mib", "--n-cpu-moe", "--nextn", "--draft", "--spec-type"] { assert!(!has(&a, flag)); }
    // Dense layers on the card always come with every expert kept on the host.
    assert_eq!(p.gpu_layers, -1);
    assert!(has(&a, "--cpu-moe") && !has(&a, "--no-op-offload") && !has(&a, "--no-kv-offload"), "{a:?}");
    let cpu = native(&m, &h, &PlanOverrides { vram_limit_mib: Some(0), ..Default::default() }, &support());
    let c = cpu.extra_args();
    assert_eq!(cpu.gpu_layers, 0);
    assert!(!has(&c, "--cpu-moe") && has(&c, "--no-op-offload") && has(&c, "--no-kv-offload"), "{c:?}");
    assert!(a.contains(&"prefetch.depth=4".into()));
    m.architecture = "qwen3moe".into();
    assert!(native(&m, &h, &Default::default(), &support()).extra_args().contains(&"prefetch.depth=2".into()));
    m.architecture = "other-moe".into();
    assert!(native(&m, &h, &Default::default(), &support()).extra_args().contains(&"prefetch.depth=0".into()));
    m.architecture = "qwen3moe".into();
    let zero = native(&m, &h, &PlanOverrides { ctx: Some(0), ..Default::default() }, &support());
    assert_eq!(zero.ctx, 256); // -c 0 would ask the engine for trained context.
}

#[test]
fn native_ram_limit_caps_the_host_budget_and_sends_the_rest_to_ssd() {
    let m = glm(); let h = hardware();
    let auto = native(&m, &h, &PlanOverrides { ctx: Some(8192), ..Default::default() }, &support());
    let at = |mib| native(&m, &h, &PlanOverrides { ram_limit_mib: Some(mib), ctx: Some(8192), ..Default::default() }, &support());
    let free = at(1 << 30); // above what is free: never raises the budget
    assert_eq!(free.native.as_ref().unwrap().ram_budget_mib, auto.native.as_ref().unwrap().ram_budget_mib);
    let capped = at(20 * 1024);
    assert!(capped.feasible, "{:?}", capped.infeasible_reason);
    let n = capped.native.as_ref().unwrap();
    // The limit is RAM; the engine's envelope adds what sits on the card.
    assert_eq!(n.ram_budget_mib, (20 * GIB + card(&m, &capped)) / MIB);
    assert!(capped.extra_args().contains(&format!("engine.ram_budget_mib={}", n.ram_budget_mib)));
    assert!(capped.tiers.ram <= 20 * GIB);
    assert!(n.expert_cache_mib < free.native.as_ref().unwrap().expert_cache_mib);
    assert!(capped.tiers.ssd > free.tiers.ssd);
    assert_eq!(capped.tiers.vram_total(), free.tiers.vram_total());
    // Experts are in exactly one tier.
    assert_eq!(capped.tiers.ssd + capped.tiers.vram_expert_cache + n.expert_cache_mib * MIB, m.expert_bytes);
    let tiny = at(2 * 1024);
    assert!(!tiny.feasible);
    assert!(tiny.infeasible_reason.unwrap().contains("system memory limit"));
}

#[test]
fn native_vram_limit_covers_tier_and_overhead_and_zero_keeps_the_card_empty() {
    let m = glm(); let h = hardware();
    let at = |mib| native(&m, &h, &PlanOverrides { vram_limit_mib: Some(mib), ctx: Some(8192), ..Default::default() }, &support());
    let free = at(1 << 30);
    let auto = native(&m, &h, &PlanOverrides { ctx: Some(8192), ..Default::default() }, &support());
    assert_eq!(free.tiers.vram_total(), auto.tiers.vram_total());
    let capped = at(8 * 1024);
    assert!(capped.feasible);
    assert!(capped.tiers.vram_total() <= 8 * GIB);
    assert!(capped.tiers.vram_total() > 7 * GIB);
    assert!(capped.native.as_ref().unwrap().vram_budget_bytes < free.native.as_ref().unwrap().vram_budget_bytes);
    assert!(capped.tiers.ssd > free.tiers.ssd);
    // Too little for a useful tier beside the dense layers: they return to RAM.
    assert_eq!((free.gpu_layers, capped.gpu_layers), (-1, 0));
    assert!(capped.native.as_ref().unwrap().expert_cache_mib < free.native.as_ref().unwrap().expert_cache_mib);
    for mib in [0, 64] {
        let off = at(mib);
        assert!(off.feasible);
        assert_eq!(off.tiers.vram_total(), 0);
        assert!(off.extra_args().contains(&"vram.budget=0".to_string()));
        assert!(off.warnings.iter().any(|w| w.contains("graphics memory limit")), "{:?}", off.warnings);
        // A CUDA build would still open a context on the card without this.
        assert!(off.args.windows(2).any(|w| w == ["--device", "none"]), "{:?}", off.args);
    }
    assert!(!free.warnings.iter().any(|w| w.contains("graphics memory limit")));
    for kept in [&free, &capped, &auto] { assert!(!kept.args.iter().any(|a| a == "--device"), "{:?}", kept.args); }
}

#[test]
fn native_windows_commit_limit_sizes_the_cache_and_the_process_envelope() {
    let m = glm(); let mut h = hardware();
    let ov = PlanOverrides { ctx: Some(8192), ..Default::default() };
    let by_ram = native(&m, &h, &ov, &support());
    assert_eq!(by_ram.native.as_ref().unwrap().host_budget_mib, None);
    assert!(!by_ram.extra_args().iter().any(|a| a.starts_with("host.budget_mib=")));

    // Plenty of commit: same cache, and the envelope adds everything on the
    // card (tier, dense layers, context) to the RAM budget, once.
    h.commit_available_bytes = Some(200 * GIB);
    let roomy = native(&m, &h, &ov, &support());
    let n = roomy.native.as_ref().unwrap();
    assert_eq!(n.expert_cache_mib, by_ram.native.as_ref().unwrap().expert_cache_mib);
    let tier = roomy.tiers.vram_total();
    assert!(roomy.gpu_layers == -1 && tier > roomy.tiers.vram_expert_cache + m.always_bytes);
    let envelope = (h.ram_available_bytes - RAM_RESERVE) / MIB + (tier + COMMIT_PROCESS_OVERHEAD).div_ceil(MIB);
    assert_eq!(n.host_budget_mib, Some(envelope));
    assert!(envelope > n.ram_budget_mib + roomy.tiers.vram_expert_cache / MIB);
    assert!(roomy.extra_args().contains(&format!("host.budget_mib={envelope}")));
    assert!(!roomy.warnings.iter().any(|w| w.contains("pagefile")));

    // Tight commit: the card and the private host memory are charged first.
    // The mapped token embeddings take RAM but no commit, and the process's
    // own share is in the overhead already.
    let fixed = roomy.tiers.ram - n.expert_cache_mib * MIB - m.token_embd_bytes - (CARD_HOST_MARGIN - HOST_MARGIN);
    let held = COMMIT_RESERVE + COMMIT_PROCESS_OVERHEAD;
    h.commit_available_bytes = Some(held + tier + fixed + 3 * GIB);
    let tight = native(&m, &h, &ov, &support());
    assert!(tight.feasible, "{:?}", tight.infeasible_reason);
    let cache = tight.native.as_ref().unwrap().expert_cache_mib;
    assert!(cache <= 3 * 1024 && cache >= 3 * 1024 - 2, "cache {cache}");
    assert!(cache < n.expert_cache_mib);
    assert!(tight.tiers.ssd > roomy.tiers.ssd);
    assert!(tight.warnings.iter().any(|w| w.contains("pagefile")), "{:?}", tight.warnings);

    // No commit left for the fixed host memory: refused with the reason.
    h.commit_available_bytes = Some(held + tier);
    let none = native(&m, &h, &ov, &support());
    assert!(!none.feasible);
    assert!(none.infeasible_reason.unwrap().contains("commit limit"));
}

#[test]
fn native_envelope_has_no_tier_share_without_a_tier() {
    let m = glm(); let mut h = hardware();
    h.commit_available_bytes = Some(200 * GIB);
    let p = native(&m, &h, &PlanOverrides { vram_limit_mib: Some(0), ctx: Some(8192), ..Default::default() }, &support());
    let n = p.native.as_ref().unwrap();
    assert_eq!(n.host_budget_mib, Some(n.ram_budget_mib + COMMIT_PROCESS_OVERHEAD / MIB));
}
