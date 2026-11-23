//! Planner tests against real GGUF header fixtures (see
//! `tests/fixtures/gguf/README.md`).

use std::path::PathBuf;

use super::accounting::account_model;
use super::hardware::{Bandwidths, DriveKind, GpuInfo, StorageInfo};
use super::*;
// Reference llama placement arithmetic and flags, not backend admission.
use super::{llama_plan as plan, llama_auto_plan as auto_plan};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf").join(rel)
}
fn glm53() -> ModelAccounting {
    account_model(&fixture("glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf")).unwrap()
}
fn qwen36() -> ModelAccounting {
    account_model(&fixture("Qwen3.6-35B-A3B-UD-Q4_K_M.header.gguf")).unwrap()
}
fn qwen3_8b() -> ModelAccounting {
    account_model(&fixture("Qwen3-8B-Q4_K_M.header.gguf")).unwrap()
}

const GB: f64 = 1e9;

/// The reference machine from the brief: RTX 4090, 64 GB RAM, 990 Pro.
fn hw() -> HardwareProfile {
    HardwareProfile {
        gpu: Some(GpuInfo {
            name: "NVIDIA GeForce RTX 4090".into(),
            vram_total_bytes: 24 * GIB,
            vram_free_bytes: 23 * GIB,
        }),
        ram_total_bytes: 64 * GIB,
        ram_available_bytes: 52 * GIB,
        commit_available_bytes: None,
        storage: Some(StorageInfo {
            mount: "E:\\".into(),
            model: Some("Samsung SSD 990 PRO 2TB".into()),
            kind: DriveKind::Ssd,
            free_bytes: 800 * GIB,
            total_bytes: 2000 * GIB,
        }),
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

fn budget() -> VramBudget {
    VramBudget { total_bytes: 24 * GIB, free_bytes: 23 * GIB, reclaimable: Vec::new() }
}

fn arg_after<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1)).map(String::as_str)
}

#[test]
fn glm53_spreads_across_all_three_tiers() {
    let p = auto_plan(&glm53(), &hw(), &budget(), None);
    assert!(p.feasible, "{p:#?}");
    assert_eq!(p.engine_kind, EngineKind::LlamaServer);
    assert_eq!(p.tier_mix, TierMix::Ssd);
    let t = &p.tiers;
    // All always-used weights on the GPU, the rest of the card as expert cache.
    assert_eq!(p.gpu_layers, -1);
    assert!(t.vram_weights as f64 > 4.0 * GB && (t.vram_weights as f64) < 9.0 * GB, "{t:?}");
    assert!(t.vram_expert_cache as f64 > 10.0 * GB, "{t:?}");
    assert!(t.vram_total() <= 24 * GIB - VRAM_HEADROOM_MIN, "{t:?}");
    // About 48 GB of page cache, the remainder streamed from the SSD.
    assert!(t.ram as f64 > 40.0 * GB && (t.ram as f64) < 56.0 * GB, "{t:?}");
    assert!(t.ssd as f64 > 35.0 * GB && (t.ssd as f64) < 60.0 * GB, "{t:?}");
    // Roughly the brief's ~9 tok/s: slow-to-usable, nowhere near fast.
    assert!(p.speed.decode_tps > 3.0 && p.speed.decode_tps < 20.0, "{:?}", p.speed);
    assert!(p.speed.label < SpeedLabel::Fast);
    assert!(p.warnings.iter().any(|w| w.contains("SSD streaming")), "{:?}", p.warnings);
    // Flags that realise the plan.
    let x = p.extra_args();
    assert!(arg_after(&x, "--moe-cache-mib").is_some(), "{x:?}");
    assert!(x.contains(&"--cpu-moe".to_string()) || arg_after(&x, "--n-cpu-moe").is_some(), "{x:?}");
    assert_eq!(arg_after(&x, "--fit"), Some("off"));
    assert!(x.contains(&"--metrics".to_string()));
    assert_eq!(arg_after(&p.args, "-ngl"), Some("all"));
    assert_eq!(arg_after(&p.args, "-c"), Some(&*p.ctx.to_string()));
    // KV is tiny for GLM (KDA + MLA), so the context reaches the 32K cap.
    assert_eq!(p.ctx, AUTO_CTX_CAP);
}

#[test]
fn glm53_without_ssd_is_infeasible() {
    let ov = PlanOverrides { allow_ssd: Some(false), ..Default::default() };
    let p = plan(&glm53(), &hw(), &budget(), &ov, None);
    assert!(!p.feasible);
    assert!(p.warnings.iter().any(|w| w.contains("SSD streaming is off")), "{:?}", p.warnings);
}

#[test]
fn n_cpu_moe_counts_blocks_not_moe_layers() {
    // Pin 4 GPU expert layers: GLM has leading dense blocks, so the first
    // (n_moe - 4) MoE layers end at block index first_moe + n_moe - 4 - 1.
    let m = glm53();
    let ov = PlanOverrides { gpu_expert_layers: Some(4), expert_cache_mib: Some(0), ..Default::default() };
    let p = plan(&m, &hw(), &budget(), &ov, None);
    assert!(p.feasible, "{p:#?}");
    assert_eq!(p.gpu_expert_layers, 4);
    let cpu = m.moe_layers - 4;
    assert_eq!(p.cpu_expert_layers, cpu);
    let first_moe = m.layers.iter().find(|l| l.expert_bytes > 0).unwrap().index;
    assert_eq!(p.n_cpu_moe, Some(cpu + first_moe));
    assert_eq!(arg_after(&p.extra_args(), "--n-cpu-moe"), Some(&*(cpu + first_moe).to_string()));
    assert!(arg_after(&p.extra_args(), "--moe-cache-mib").is_none());
}

#[test]
fn pinned_experts_that_dont_fit_are_flagged() {
    let m = glm53();
    let ov = PlanOverrides { gpu_expert_layers: Some(m.moe_layers), ..Default::default() };
    let p = plan(&m, &hw(), &budget(), &ov, None);
    assert!(!p.feasible);
    assert_eq!(p.gpu_expert_layers, m.moe_layers);
    assert!(p.warnings.iter().any(|w| w.contains("graphics memory but only")), "{:?}", p.warnings);
}

#[test]
fn qwen36_fits_in_vram_and_is_fast() {
    let p = auto_plan(&qwen36(), &hw(), &budget(), None);
    assert!(p.feasible);
    assert_eq!(p.tier_mix, TierMix::Vram, "{:#?}", p.tiers);
    // Only the input embedding table (llama.cpp keeps token_embd on the CPU
    // and reads one row per token) and the host compute buffer stay in
    // system memory.
    assert_eq!(p.tiers.ram, qwen36().token_embd_bytes + p.context.host_compute_bytes);
    assert_eq!(p.tiers.ssd, 0);
    assert_eq!(p.cpu_expert_layers, 0);
    assert_eq!(p.speed.label, SpeedLabel::Fast);
    assert!(p.speed.decode_tps > 80.0 && p.speed.decode_tps < 300.0, "{:?}", p.speed);
    let x = p.extra_args();
    assert!(!x.iter().any(|a| a.contains("cpu-moe") || a == "--moe-cache-mib"), "{x:?}");
}

#[test]
fn dense_8b_fits_with_large_context() {
    let m = qwen3_8b();
    let p = auto_plan(&m, &hw(), &budget(), None);
    assert!(p.feasible);
    assert_eq!(p.tier_mix, TierMix::Vram);
    // Small model: capped at its trained max (40K), above the 32K default cap.
    assert_eq!(p.ctx as u64, m.ctx_train.min(AUTO_CTX_CAP_SMALL as u64));
    assert_eq!(p.speed.label, SpeedLabel::Fast);
}

#[test]
fn dense_partial_offload_on_small_card() {
    let m = qwen3_8b();
    let small = VramBudget { total_bytes: 4 * GIB, free_bytes: 4 * GIB, reclaimable: Vec::new() };
    let ov = PlanOverrides { ctx: Some(8192), ..Default::default() };
    let p = plan(&m, &hw(), &small, &ov, None);
    assert!(p.feasible);
    assert!(p.gpu_layers > 0 && (p.gpu_layers as u32) < m.n_layers, "{p:#?}");
    assert_eq!(p.tier_mix, TierMix::Ram);
    assert!(p.tiers.vram_total() <= small.usable());
    assert!(p.warnings.iter().any(|w| w.contains("layers fit")), "{:?}", p.warnings);
    assert_eq!(arg_after(&p.args, "-ngl"), Some(&*p.gpu_layers.to_string()));
}

#[test]
fn q8_kv_chosen_when_it_buys_a_tier() {
    // A budget where f16 KV at 32K spills, q8_0 KV does not.
    let m = qwen3_8b();
    let ov = PlanOverrides { ctx: Some(32768), ..Default::default() };
    let f16 = plan(&m, &hw(), &budget(), &PlanOverrides { kv_type: Some(KvType::F16), ..ov.clone() }, None);
    let q8 = plan(&m, &hw(), &budget(), &PlanOverrides { kv_type: Some(KvType::Q8_0), ..ov.clone() }, None);
    let need_f16 = f16.tiers.vram_total();
    let need_q8 = q8.tiers.vram_total();
    assert!(need_q8 < need_f16);
    let headroom = VRAM_HEADROOM_MIN;
    let free = (need_f16 + need_q8) / 2 + headroom;
    let tight = VramBudget { total_bytes: 24 * GIB, free_bytes: free, reclaimable: Vec::new() };
    let p = plan(&m, &hw(), &tight, &ov, None);
    assert_eq!(p.kv_type, KvType::Q8_0, "{p:#?}");
    assert_eq!(p.tier_mix, TierMix::Vram);
    assert_eq!(arg_after(&p.args, "-ctk"), Some("q8_0"));
    // With plenty of room f16 stays.
    let roomy = plan(&m, &hw(), &budget(), &ov, None);
    assert_eq!(roomy.kv_type, KvType::F16);
}

#[test]
fn no_gpu_runs_on_cpu() {
    let m = qwen3_8b();
    let mut h = hw();
    h.gpu = None;
    let p = auto_plan(&m, &h, &VramBudget::no_gpu(), None);
    assert!(p.feasible);
    assert_eq!(p.gpu_layers, 0);
    assert_eq!(p.tiers.vram_total(), 0);
    assert_eq!(p.tier_mix, TierMix::Ram);
    assert!(p.warnings.iter().any(|w| w.contains("No NVIDIA GPU")));
    assert!(p.speed.decode_tps > 1.0 && p.speed.decode_tps < 30.0, "{:?}", p.speed);
}

#[test]
fn eviction_alternative_reported() {
    let m = qwen36();
    let b = VramBudget {
        total_bytes: 24 * GIB,
        free_bytes: 10 * GIB,
        reclaimable: vec![Reclaimable { id: "pipeline:text_to_image".into(), label: "SDXL".into(), bytes: 13 * GIB }],
    };
    let p = auto_plan(&m, &hw(), &b, None);
    assert!(p.feasible);
    assert_ne!(p.tier_mix, TierMix::Vram);
    let alt = p.evict_alternative.as_ref().expect("alternative");
    assert_eq!(alt.tier_mix, TierMix::Vram);
    assert!(alt.decode_tps > p.speed.decode_tps);
    assert!(p.warnings.iter().any(|w| w.contains("Unloading SDXL")), "{:?}", p.warnings);
}

#[test]
fn calibration_scales_the_estimate() {
    let m = qwen36();
    let raw = auto_plan(&m, &hw(), &budget(), None);
    let mut c = Calibration::default();
    for _ in 0..8 {
        c.record(raw.speed.raw_decode_tps, Some(raw.speed.raw_decode_tps * 0.5), 0.0, None);
    }
    let mut cal = ModelCalibration::new();
    cal.insert(TierMix::Vram, c);
    let p = auto_plan(&m, &hw(), &budget(), Some(&cal));
    let ratio = p.speed.decode_tps / raw.speed.decode_tps;
    assert!(ratio > 0.5 && ratio < 0.7, "ratio {ratio}");
    assert_eq!(p.speed.raw_decode_tps, raw.speed.raw_decode_tps);
    // Calibration for another tier mix does not apply.
    let mut other = ModelCalibration::new();
    other.insert(TierMix::Ssd, cal[&TierMix::Vram].clone());
    let q = auto_plan(&m, &hw(), &budget(), Some(&other));
    assert_eq!(q.speed.decode_tps, raw.speed.decode_tps);
}

#[test]
fn overrides_pin_context_and_kv() {
    let ov = PlanOverrides { ctx: Some(16384), kv_type: Some(KvType::Q4_0), ..Default::default() };
    let p = plan(&qwen36(), &hw(), &budget(), &ov, None);
    assert_eq!(p.ctx, 16384);
    assert_eq!(p.kv_type, KvType::Q4_0);
    assert_eq!(p.overrides, ov);
    assert!(!p.warnings.iter().any(|w| w.contains("Context capped")));
}

// ---------------------------------------------------------------------------
// Context memory inside the plan
// ---------------------------------------------------------------------------

/// GLM-5.3-Flash at fixed f16 cache: raising the context takes its memory
/// out of graphics memory first, so the hot-expert cache shrinks, more
/// expert traffic goes to RAM and SSD, and decode gets slower.
#[test]
fn glm53_larger_context_shrinks_expert_cache_and_speed() {
    let m = glm53();
    let at = |ctx: u32| {
        let ov = PlanOverrides { ctx: Some(ctx), kv_type: Some(KvType::F16), ..Default::default() };
        plan(&m, &hw(), &budget(), &ov, None)
    };
    let p32 = at(32 * 1024);
    let p128 = at(128 * 1024);
    let p512 = at(512 * 1024);
    for p in [at(4096), p32.clone(), p128.clone(), p512.clone(), at(1 << 20)] {
        let t = &p.tiers;
        println!(
            "GLM-5.3 plan ctx {:>8}: feasible {} ngl {} context {:.2} GB, overhead {:.2} GB, expert cache {:.2} GB, RAM {:.1} GB, SSD {:.1} GB, decode {:.1} tok/s",
            p.ctx,
            p.feasible,
            p.gpu_layers,
            p.context.cache_bytes() as f64 / GB,
            t.vram_overhead as f64 / GB,
            t.vram_expert_cache as f64 / GB,
            t.ram as f64 / GB,
            t.ssd as f64 / GB,
            p.speed.decode_tps
        );
    }
    for p in [&p32, &p128, &p512] {
        assert!(p.feasible, "{p:#?}");
        assert!(p.tiers.vram_total() <= budget().usable(), "{:?}", p.tiers);
        // The plan reserves its own context numbers.
        assert_eq!(p.tiers.vram_kv, p.context.cache_bytes());
    }
    assert!(p128.context.cache_bytes() > p32.context.cache_bytes());
    assert!(p128.tiers.vram_expert_cache < p32.tiers.vram_expert_cache, "{:?} vs {:?}", p128.tiers, p32.tiers);
    assert!(p512.tiers.vram_expert_cache < p128.tiers.vram_expert_cache, "{:?} vs {:?}", p512.tiers, p128.tiers);
    assert!(p128.speed.decode_tps < p32.speed.decode_tps, "{:?} vs {:?}", p128.speed, p32.speed);
    assert!(p512.speed.decode_tps < p128.speed.decode_tps, "{:?} vs {:?}", p512.speed, p128.speed);
}

/// A context whose memory exceeds what the machine has can't be planned;
/// the plan says why and names the largest context that does fit, and that
/// context really plans feasibly.
#[test]
fn oversized_context_reports_largest_that_fits() {
    let m = qwen3_8b();
    let card = VramBudget { total_bytes: 12 * GIB, free_bytes: 12 * GIB, reclaimable: Vec::new() };
    let mut h = hw();
    h.ram_available_bytes = 8 * GIB; // little room to spill the KV to RAM
    let ov = PlanOverrides { ctx: Some(1 << 20), kv_type: Some(KvType::F16), ..Default::default() };
    let p = plan(&m, &h, &card, &ov, None);
    assert!(!p.feasible);
    let max = p.max_ctx.expect("a smaller context fits");
    assert!((4096..1 << 20).contains(&max) && max % 256 == 0, "{max}");
    let reason = p.infeasible_reason.as_deref().unwrap();
    assert!(reason.contains(&max.to_string()), "{reason}");
    assert_eq!(p.warnings.last().map(String::as_str), Some(reason));
    let ok = plan(&m, &h, &card, &PlanOverrides { ctx: Some(max), ..ov.clone() }, None);
    assert!(ok.feasible, "{ok:#?}");
    let over = plan(&m, &h, &card, &PlanOverrides { ctx: Some(max + 256), ..ov }, None);
    assert!(!over.feasible);
}

/// Weights that can't fit at any context get no `max_ctx`, and a reason that
/// blames the weights.
#[test]
fn weights_that_never_fit_have_no_max_ctx() {
    let ov = PlanOverrides { allow_ssd: Some(false), ..Default::default() };
    let p = plan(&glm53(), &hw(), &budget(), &ov, None);
    assert!(!p.feasible);
    assert_eq!(p.max_ctx, None);
    assert!(p.infeasible_reason.as_deref().unwrap().contains("SSD streaming is off"));
}

/// The planner pins one sequence and passes K and V types separately (V
/// follows K for MLA models).
#[test]
fn args_carry_parallel_and_separate_cache_types() {
    let m = qwen3_8b();
    let ov = PlanOverrides {
        ctx: Some(16384),
        kv_type: Some(KvType::Q8_0),
        kv_type_v: Some(KvType::Q4_0),
        ..Default::default()
    };
    let p = plan(&m, &hw(), &budget(), &ov, None);
    assert_eq!((p.kv_type, p.kv_type_v), (KvType::Q8_0, KvType::Q4_0));
    let x = p.extra_args();
    assert_eq!(arg_after(&x, "-np"), Some("1"));
    assert_eq!(arg_after(&x, "-ctk"), Some("q8_0"));
    assert_eq!(arg_after(&x, "-ctv"), Some("q4_0"));
    assert_eq!(p.args.iter().filter(|a| *a == "-ctk").count(), 1, "{:?}", p.args);
    // q8_0 K + q4_0 V: 36 layers × 1024 elements, 34 + 18 bytes per 32.
    assert_eq!(p.context.kv_bytes, 16384 * 36 * 1024 / 32 * (34 + 18));

    let f16 = plan(&m, &hw(), &budget(), &PlanOverrides { ctx: Some(16384), ..Default::default() }, None);
    let x = f16.extra_args();
    assert_eq!(arg_after(&x, "-np"), Some("1"));
    assert!(!x.contains(&"-ctk".to_string()), "{x:?}");

    let ov = PlanOverrides { kv_type: Some(KvType::Q8_0), kv_type_v: Some(KvType::Q4_0), ..Default::default() };
    let mla = plan(&glm53(), &hw(), &budget(), &ov, None);
    assert_eq!(mla.kv_type_v, KvType::Q8_0);
    assert_eq!(arg_after(&mla.extra_args(), "-ctv"), Some("q8_0"));
}

// ---------------------------------------------------------------------------
// Device placement as llama.cpp b12182 realises the plan's arguments.

/// Graphics-memory weights and context memory implied by a plan's flags,
/// worked out the way llama.cpp b12182 assigns them (independent of the
/// planner's own bookkeeping): `-ngl N` offloads the output head plus blocks
/// `n_layer_all + 1 - N ..`; `--cpu-moe` / `--n-cpu-moe N` keep routed
/// experts of all / the first N blocks in host memory; each block's caches
/// live on its device.
fn realised_gpu(m: &ModelAccounting, p: &LoadPlan) -> (u64, u64) {
    let x = p.extra_args();
    let n_all = (m.n_layers + m.mtp_layers) as i64;
    let ngl = match p.gpu_layers {
        -1 => n_all + 1,
        n => n as i64,
    };
    let first = n_all + 1 - ngl;
    let cpu_moe_all = x.iter().any(|a| a == "--cpu-moe");
    let n_cpu_moe: i64 = arg_after(&x, "--n-cpu-moe").map_or(0, |v| v.parse().unwrap());
    let params = ContextParams {
        ctx: p.ctx,
        kv_k: p.kv_type,
        kv_v: p.kv_type_v,
        n_parallel: p.n_parallel,
        ubatch: p.context.ctx.clamp(1, 512),
        experts_on_host: false,
    };
    let mut weights = 0u64;
    let mut caches = 0u64;
    let mut block_always = 0u64;
    for l in m.layers.iter().filter(|l| l.kind != LayerKind::Mtp) {
        block_always += l.always_bytes;
        if (l.index as i64) < first {
            continue;
        }
        weights += l.always_bytes;
        if !(cpu_moe_all || (l.index as i64) < n_cpu_moe) {
            weights += l.expert_bytes;
        }
        caches += context::layer_cache_bytes(m, &params, l);
    }
    if ngl > 0 {
        weights += m.always_bytes - block_always;
        if m.output_tied {
            weights += m.token_embd_bytes;
        }
    }
    (weights, caches)
}

/// Partial offload keeps every non-GPU expert in host memory through an
/// explicit flag, and every expert tensor the flags put on the GPU is in the
/// plan's graphics-memory total (review S1: GLM-5.3 at 1M context, f16).
#[test]
fn partial_offload_pins_host_experts_and_counts_gpu_ones() {
    let m = glm53();
    let ov = PlanOverrides { ctx: Some(1 << 20), kv_type: Some(KvType::F16), ..Default::default() };
    let p = plan(&m, &hw(), &budget(), &ov, None);
    assert!(p.feasible, "{p:#?}");
    assert!(p.gpu_layers > 0, "expected partial offload: {p:#?}");
    let x = p.extra_args();
    assert!(x.iter().any(|a| a == "--cpu-moe" || a == "--n-cpu-moe"), "{x:?}");
    let (weights, caches) = realised_gpu(&m, &p);
    assert_eq!(weights, p.tiers.vram_weights, "{x:?} {:?}", p.tiers);
    assert_eq!(caches, p.tiers.vram_kv);
    assert!(p.tiers.vram_total() <= budget().usable(), "{:?}", p.tiers);

    // Same check across contexts, cache types and a pinned GPU-expert count.
    let cases = [
        PlanOverrides { ctx: Some(4096), ..Default::default() },
        PlanOverrides { ctx: Some(512 * 1024), kv_type: Some(KvType::F16), ..Default::default() },
        PlanOverrides { gpu_expert_layers: Some(4), expert_cache_mib: Some(0), ..Default::default() },
    ];
    for ov in cases {
        let p = plan(&m, &hw(), &budget(), &ov, None);
        assert_eq!(realised_gpu(&m, &p), (p.tiers.vram_weights, p.tiers.vram_kv), "{ov:?} {:?}", p.extra_args());
    }
}

/// Hybrid models: the GPU share of context memory is the caches of the
/// blocks `-ngl` actually offloads, not a layer-count fraction (review S2:
/// Qwen3.6 at 262144 f16 on a card that takes part of the model).
#[test]
fn hybrid_partial_offload_reserves_the_offloaded_blocks_caches() {
    let m = qwen36();
    let ov = PlanOverrides { ctx: Some(262_144), kv_type: Some(KvType::F16), ..Default::default() };
    let mut partial = 0;
    for gib in 4..=20u64 {
        let card = VramBudget { total_bytes: gib * GIB, free_bytes: gib * GIB, reclaimable: Vec::new() };
        let p = plan(&m, &hw(), &card, &ov, None);
        if !p.feasible || p.gpu_layers <= 0 {
            continue;
        }
        let (weights, caches) = realised_gpu(&m, &p);
        assert_eq!(caches, p.tiers.vram_kv, "{gib} GiB {:?}", p.extra_args());
        assert_eq!(weights, p.tiers.vram_weights, "{gib} GiB {:?}", p.extra_args());
        assert!(p.tiers.vram_total() <= card.usable(), "{:?}", p.tiers);
        if caches > 0 && caches < p.context.cache_bytes() {
            partial += 1; // only some full-attention caches on the card
        }
    }
    assert!(partial > 0, "no card size gave a partial offload");
}

/// The `-ngl` number counts the output head and MTP blocks first, so `g`
/// decode blocks on the GPU need `g + 1 + mtp` (b12182 `i_gpu_start`).
#[test]
fn ngl_counts_output_and_mtp_slots() {
    let glm = glm53();
    assert_eq!(glm.mtp_layers, 1);
    assert_eq!(ngl_for(&glm, 0), 0);
    assert_eq!(ngl_for(&glm, 10), 12);
    assert_eq!(ngl_for(&glm, glm.n_layers), -1);
    let q = qwen3_8b();
    assert_eq!(q.mtp_layers, 0);
    assert_eq!(ngl_for(&q, 10), 11);
}

/// Gemma 3 1B ties its output head to token_embd (it has `output_norm` but
/// no `output.weight`): the copy on the output device counts as graphics
/// memory (review S3).
#[test]
fn tied_output_embeddings_count_on_the_gpu() {
    let m = account_model(&fixture("gemma-3-1b-it-Q4_K_M.header.gguf")).unwrap();
    assert!(m.output_tied);
    assert!(m.output_bytes > 0, "output_norm is still an output tensor");
    assert!(!qwen3_8b().output_tied);
    let p = auto_plan(&m, &hw(), &budget(), None);
    assert_eq!(p.gpu_layers, -1);
    let blocks: u64 = m.layers.iter().filter(|l| l.kind != LayerKind::Mtp).map(|l| l.always_bytes).sum();
    let global = m.always_bytes - blocks;
    assert_eq!(p.tiers.vram_weights, blocks + global + m.token_embd_bytes);
    assert_eq!(realised_gpu(&m, &p).0, p.tiers.vram_weights);
}

/// A model whose context memory can't be sized is refused with a reason,
/// never planned with zero for it (review S4).
#[test]
fn unsized_layers_are_refused() {
    let mut m = qwen36();
    m.unsupported = Some("block 0 is a recurrent layer of a kind this planner can't size".into());
    let p = auto_plan(&m, &hw(), &budget(), None);
    assert!(!p.feasible);
    assert_eq!(p.max_ctx, None);
    let why = p.infeasible_reason.as_deref().unwrap();
    assert!(why.contains("Can't estimate memory"), "{why}");
}

/// The largest fitting context is searched strictly below the request on the
/// 256-token grid, so an unaligned request keeps the aligned value just
/// under it (review S8: a request of 1000 must report 768 when 768 fits).
#[test]
fn max_ctx_search_keeps_the_boundary_below_unaligned_requests() {
    assert_eq!(context::largest_ctx(999, |c| c <= 768), Some(768));
    assert_eq!(context::largest_ctx(1023, |_| true), Some(768));
    assert_eq!(context::largest_ctx(1024 - 1, |c| c <= 512), Some(512));
    assert_eq!(context::largest_ctx(255, |_| true), None);
}

#[test]
fn memory_limits_cap_llama_tiers_and_push_the_rest_down() {
    let m = glm53();
    let free = plan(&m, &hw(), &budget(), &PlanOverrides { ctx: Some(8192), ..Default::default() }, None);
    let above = plan(&m, &hw(), &budget(), &PlanOverrides { ctx: Some(8192), ram_limit_mib: Some(1 << 30), vram_limit_mib: Some(1 << 30), ..Default::default() }, None);
    // Limits never raise the planner's own budgets.
    assert_eq!((above.tiers.vram_total(), above.tiers.ram, above.tiers.ssd), (free.tiers.vram_total(), free.tiers.ram, free.tiers.ssd));

    let ram = plan(&m, &hw(), &budget(), &PlanOverrides { ctx: Some(8192), ram_limit_mib: Some(16 * 1024), ..Default::default() }, None);
    assert!(ram.feasible, "{:?}", ram.infeasible_reason);
    assert!(ram.tiers.ram <= 16 * GIB, "{:?}", ram.tiers);
    assert!(ram.tiers.ssd > free.tiers.ssd);
    assert!(ram.warnings.iter().any(|w| w.contains("system memory limit")), "{:?}", ram.warnings);
    assert!(!free.warnings.iter().any(|w| w.contains("system memory limit")));

    let vram = plan(&m, &hw(), &budget(), &PlanOverrides { ctx: Some(8192), vram_limit_mib: Some(12 * 1024), ..Default::default() }, None);
    assert!(vram.feasible, "{:?}", vram.infeasible_reason);
    assert!(vram.tiers.vram_total() <= 12 * GIB, "{:?}", vram.tiers);
    assert!(vram.tiers.vram_total() < free.tiers.vram_total());

    // A dense model that fits the card moves to system memory under a limit.
    let d = qwen3_8b();
    let all = auto_plan(&d, &hw(), &budget(), None);
    assert_eq!(all.tier_mix, TierMix::Vram);
    let off = plan(&d, &hw(), &budget(), &PlanOverrides { vram_limit_mib: Some(0), ..Default::default() }, None);
    assert!(off.feasible, "{:?}", off.infeasible_reason);
    assert_eq!(off.tiers.vram_total(), 0);
    assert!(off.tiers.ram > 0);
    assert_eq!(arg_after(&off.args, "--device"), Some("none"));
    assert_eq!(arg_after(&off.args, "-ngl"), Some("0"));
    assert!(off.warnings.iter().any(|w| w.contains("graphics memory limit")), "{:?}", off.warnings);
    assert!(!off.warnings.iter().any(|w| w.contains("No NVIDIA GPU")));
    assert!(arg_after(&all.args, "--device").is_none());
    // A limit the card can serve keeps the model on it, partly offloaded.
    let part = plan(&d, &hw(), &budget(), &PlanOverrides { vram_limit_mib: Some(3 * 1024), ..Default::default() }, None);
    assert!(part.feasible, "{:?}", part.infeasible_reason);
    assert!(part.tiers.vram_total() > 0 && part.tiers.vram_total() <= 3 * GIB, "{:?}", part.tiers);
    assert!(arg_after(&part.args, "--device").is_none());
}

#[test]
fn graphics_limit_alone_decides_off_card_when_free_graphics_memory_is_short() {
    // Something else holds the card: nothing is usable after desktop headroom.
    let d = qwen3_8b();
    let tight = VramBudget { total_bytes: 24 * GIB, free_bytes: 256 * MIB, reclaimable: Vec::new() };
    let zero = PlanOverrides { vram_limit_mib: Some(0), ..Default::default() };
    let off = llama_plan(&d, &hw(), &tight, &zero, None);
    assert!(off.feasible, "{:?}", off.infeasible_reason);
    assert_eq!(off.tiers.vram_total(), 0);
    assert_eq!(arg_after(&off.args, "--device"), Some("none"));
    assert!(off.warnings.iter().any(|w| w.contains("graphics memory limit")), "{:?}", off.warnings);

    // Unloading another model cannot improve a plan that stays off the card.
    let held = VramBudget {
        reclaimable: vec![Reclaimable { id: "pipeline:text_to_image".into(), label: "SDXL".into(), bytes: 13 * GIB }],
        ..tight.clone()
    };
    let off = llama_plan(&d, &hw(), &held, &zero, None);
    assert!(off.feasible, "{:?}", off.infeasible_reason);
    assert_eq!(arg_after(&off.args, "--device"), Some("none"));
    assert!(off.evict_alternative.is_none(), "{:?}", off.evict_alternative);
    assert!(!off.warnings.iter().any(|w| w.contains("Unloading")), "{:?}", off.warnings);

    // No limit, or one the card could serve: short free memory is refused on
    // the card as before, and unloading is still offered.
    for ov in [PlanOverrides::default(), PlanOverrides { vram_limit_mib: Some(3 * 1024), ..Default::default() }] {
        let p = llama_plan(&d, &hw(), &held, &ov, None);
        assert!(!p.feasible, "{ov:?}");
        assert!(arg_after(&p.args, "--device").is_none(), "{:?}", p.args);
        assert!(p.warnings.iter().any(|w| w.contains("is free")), "{:?}", p.warnings);
        assert!(p.evict_alternative.is_some(), "{ov:?}");
    }
}

#[test]
fn pinned_gpu_experts_stay_pinned_and_are_refused_under_a_graphics_limit() {
    let m = glm53();
    for limit in [8 * 1024, 0] {
        let ov = PlanOverrides { gpu_expert_layers: Some(m.moe_layers), vram_limit_mib: Some(limit), ..Default::default() };
        let p = llama_plan(&m, &hw(), &budget(), &ov, None);
        assert!(!p.feasible, "limit {limit}");
        assert_eq!(p.gpu_expert_layers, m.moe_layers, "limit {limit}");
        assert!(arg_after(&p.args, "--device").is_none(), "{:?}", p.args);
        assert!(p.warnings.iter().any(|w| w.contains("the graphics memory limit allows")), "{:?}", p.warnings);
    }
    // A pinned hot-expert cache is honoured the same way.
    let ov = PlanOverrides { expert_cache_mib: Some(4096), vram_limit_mib: Some(0), ..Default::default() };
    let p = llama_plan(&m, &hw(), &budget(), &ov, None);
    assert!(!p.feasible);
    assert!(arg_after(&p.args, "--device").is_none(), "{:?}", p.args);
    // Without a pin the same zero limit plans off the card.
    let p = llama_plan(&m, &hw(), &budget(), &PlanOverrides { vram_limit_mib: Some(0), ..Default::default() }, None);
    assert_eq!(arg_after(&p.args, "--device"), Some("none"));
    assert_eq!(p.tiers.vram_total(), 0);
}

#[test]
fn a_system_memory_limit_below_the_context_window_is_named_in_the_refusal() {
    let d = qwen3_8b();
    let cpu = VramBudget::no_gpu();
    // No context fits a zero limit: say so, never "lower the context".
    let none = llama_plan(&d, &hw(), &cpu, &PlanOverrides { ram_limit_mib: Some(0), ..Default::default() }, None);
    assert!(!none.feasible);
    assert_eq!(none.max_ctx, None);
    let why = none.infeasible_reason.as_deref().unwrap();
    assert!(why.contains("0.0 GB system memory limit") && why.contains("Raise the limit"), "{why}");
    assert!(none.warnings.iter().any(|w| w.contains("but the system memory limit allows 0.0 GB")), "{:?}", none.warnings);
    assert!(!none.warnings.iter().any(|w| w.contains("Lower the context length")), "{:?}", none.warnings);

    // A limit a smaller context fits names both the limit and that context.
    let free = llama_plan(&d, &hw(), &cpu, &PlanOverrides { ctx: Some(32768), ..Default::default() }, None);
    assert!(free.feasible, "{:?}", free.infeasible_reason);
    let limit_mib = free.context.total_bytes() / 2 / MIB;
    let some = llama_plan(&d, &hw(), &cpu, &PlanOverrides { ctx: Some(32768), ram_limit_mib: Some(limit_mib), ..Default::default() }, None);
    assert!(!some.feasible);
    let max = some.max_ctx.expect("a smaller context fits the limit");
    let why = some.infeasible_reason.as_deref().unwrap();
    assert!(why.contains("system memory limit") && why.contains(&format!("{max} tokens")), "{why}");

    // A limit that only pushes weights to the SSD is not a refusal.
    let spill = llama_plan(&d, &hw(), &cpu, &PlanOverrides { ctx: Some(4096), ram_limit_mib: Some(2048), ..Default::default() }, None);
    assert!(spill.feasible, "{:?}", spill.infeasible_reason);
    assert!(spill.tiers.ssd > 0 && spill.tiers.ram <= 2 * GIB, "{:?}", spill.tiers);
    // With SSD streaming off, the refusal names the limit that caused it.
    let no_ssd = llama_plan(&d, &hw(), &cpu, &PlanOverrides { ctx: Some(4096), ram_limit_mib: Some(2048), allow_ssd: Some(false), ..Default::default() }, None);
    assert!(!no_ssd.feasible);
    assert!(no_ssd.infeasible_reason.as_deref().unwrap().contains("system memory limit"), "{:?}", no_ssd.infeasible_reason);
}

/// Whether a limit leaves the card usable is asked of every on-card layout,
/// not of one fixed layout: a small model fully on the card at the default
/// batch is smaller than an empty card at the streaming batch.
#[test]
fn a_graphics_limit_the_full_gpu_plan_fits_keeps_the_model_on_the_card() {
    let m = account_model(&fixture("gemma-3-1b-it-Q4_K_M.header.gguf")).unwrap();
    // Context and cache type pinned, so the layout is the only free choice.
    let pinned = PlanOverrides { ctx: Some(4096), kv_type: Some(KvType::F16), ..Default::default() };
    let at = |limit_mib| {
        let ov = PlanOverrides { vram_limit_mib: Some(limit_mib), ..pinned.clone() };
        llama_plan(&m, &hw(), &budget(), &ov, None)
    };
    let p = at(2048);
    assert!(p.feasible, "{:?}", p.infeasible_reason);
    assert_eq!(p.gpu_layers, -1, "{:?}", p.args);
    assert_eq!(p.ubatch, UBATCH_DEFAULT);
    assert!(p.tiers.vram_total() > 0 && p.tiers.vram_total() <= 2 * GIB, "{:?}", p.tiers);
    assert!(arg_after(&p.args, "--device").is_none(), "{:?}", p.args);
    assert!(!p.warnings.iter().any(|w| w.contains("graphics memory limit is below")), "{:?}", p.warnings);

    // The limit changes nothing about the plan chosen without one.
    let free = llama_plan(&m, &hw(), &budget(), &pinned, None);
    assert_eq!(free.tiers, p.tiers);
    // No layer on the card at the streaming batch needs more than the full
    // plan, so just under the full plan nothing on the card fits and the CPU
    // takes the model.
    let off = at((p.tiers.vram_total() - 1) / MIB);
    assert!(off.feasible, "{:?}", off.infeasible_reason);
    assert_eq!(off.tiers.vram_total(), 0);
    assert_eq!(arg_after(&off.args, "--device"), Some("none"));
}
