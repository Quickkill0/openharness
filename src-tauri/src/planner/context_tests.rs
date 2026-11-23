//! Calibration of [`context_memory`] against llama.cpp b12182's allocator.
//!
//! Every record below is a `llama-fit-params --fit-print on -v` dry run
//! (no allocation, no model data read) on a header-only stub of the real
//! GGUF, with the CPU build (`-ngl 0`) and the CUDA 12.4 build (`-ngl 999`,
//! RTX 4090), flash attention on, `-b max(ub, 2048)`. Numbers are MiB as
//! llama.cpp prints them: the per-cache `llama_kv_cache: size = …` lines
//! (full / MLA cache first, then the SWA or indexer cache), the
//! `llama_memory_recurrent` total, and the compute buffer of the main device
//! plus the CUDA host buffer. Families: dense GQA (Qwen3-0.6B), sliding
//! window (Gemma 3 1B), MLA (GLM-4.7-Flash), hybrid linear (Qwen3.6-35B-A3B
//! gated delta-net; GLM-5.3-Flash KDA + sparse MLA).

use std::path::PathBuf;

use super::*;
use crate::planner::accounting::account_model;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Backend {
    Cpu,
    Cuda,
}
use Backend::*;

struct Rec {
    backend: Backend,
    model: &'static str,
    ctx: u32,
    ub: u32,
    ctk: &'static str,
    ctv: &'static str,
    np: u32,
    host_experts: bool,
    caches: &'static [f64],
    state: f64,
    compute: f64,
    host_compute: f64,
}

#[allow(clippy::too_many_arguments)]
const fn rec(
    backend: Backend,
    model: &'static str,
    ctx: u32,
    ub: u32,
    ctk: &'static str,
    ctv: &'static str,
    np: u32,
    host_experts: bool,
    caches: &'static [f64],
    state: f64,
    compute: f64,
    host_compute: f64,
) -> Rec {
    Rec { backend, model, ctx, ub, ctk, ctv, np, host_experts, caches, state, compute, host_compute }
}

const RECORDS: &[Rec] = &[
    rec(Cpu, "qwen3-0.6b", 4096, 512, "f16", "f16", 1, false, &[448.0], 0.0, 312.77, 0.0),
    rec(Cpu, "qwen3-0.6b", 32768, 512, "f16", "f16", 1, false, &[3584.0], 0.0, 338.77, 0.0),
    rec(Cpu, "qwen3-0.6b", 40960, 512, "f16", "f16", 1, false, &[4480.0], 0.0, 346.77, 0.0),
    rec(Cpu, "qwen3-0.6b", 32768, 2048, "f16", "f16", 1, false, &[3584.0], 0.0, 1355.08, 0.0),
    rec(Cpu, "qwen3-0.6b", 32768, 512, "q8_0", "q8_0", 1, false, &[1904.0], 0.0, 338.77, 0.0),
    rec(Cpu, "qwen3-0.6b", 32768, 512, "q8_0", "q4_0", 1, false, &[1456.0], 0.0, 338.77, 0.0),
    rec(Cpu, "qwen3-0.6b", 32768, 512, "f16", "f16", 4, false, &[3584.0], 0.0, 316.77, 0.0),
    rec(Cpu, "gemma3-1b", 4096, 512, "f16", "f16", 1, false, &[16.0, 22.0], 0.0, 528.28, 0.0),
    rec(Cpu, "gemma3-1b", 32768, 512, "f16", "f16", 1, false, &[128.0, 22.0], 0.0, 556.28, 0.0),
    rec(Cpu, "gemma3-1b", 32768, 2048, "f16", "f16", 1, false, &[128.0, 55.0], 0.0, 2231.12, 0.0),
    rec(Cpu, "gemma3-1b", 32768, 512, "q8_0", "q8_0", 1, false, &[68.0, 11.69], 0.0, 556.28, 0.0),
    rec(Cpu, "gemma3-1b", 32768, 512, "q8_0", "q4_0", 1, false, &[52.0, 8.94], 0.0, 556.28, 0.0),
    rec(Cpu, "gemma3-1b", 32768, 512, "f16", "f16", 4, false, &[128.0, 88.0], 0.0, 534.53, 0.0),
    rec(Cpu, "glm47-flash", 4096, 512, "f16", "f16", 1, false, &[211.5], 0.0, 326.52, 0.0),
    rec(Cpu, "glm47-flash", 32768, 512, "f16", "f16", 1, false, &[1692.0], 0.0, 354.52, 0.0),
    rec(Cpu, "glm47-flash", 131072, 512, "f16", "f16", 1, false, &[6768.0], 0.0, 450.52, 0.0),
    rec(Cpu, "glm47-flash", 202752, 512, "f16", "f16", 1, false, &[10469.25], 0.0, 520.52, 0.0),
    rec(Cpu, "glm47-flash", 32768, 2048, "f16", "f16", 1, false, &[1692.0], 0.0, 1418.06, 0.0),
    rec(Cpu, "glm47-flash", 32768, 512, "q8_0", "q8_0", 1, false, &[898.88], 0.0, 354.52, 0.0),
    rec(Cpu, "glm47-flash", 32768, 512, "f16", "f16", 4, false, &[1692.0], 0.0, 330.52, 0.0),
    rec(Cpu, "glm47-flash", 131072, 2048, "q8_0", "q8_0", 1, false, &[3595.5], 0.0, 1802.06, 0.0),
    rec(Cpu, "qwen36-35b", 4096, 512, "f16", "f16", 1, false, &[80.0], 62.81, 509.03, 0.0),
    rec(Cpu, "qwen36-35b", 32768, 512, "f16", "f16", 1, false, &[640.0], 62.81, 537.03, 0.0),
    rec(Cpu, "qwen36-35b", 131072, 512, "f16", "f16", 1, false, &[2560.0], 62.81, 633.03, 0.0),
    rec(Cpu, "qwen36-35b", 262144, 512, "f16", "f16", 1, false, &[5120.0], 62.81, 761.03, 0.0),
    rec(Cpu, "qwen36-35b", 32768, 2048, "f16", "f16", 1, false, &[640.0], 62.81, 2148.1, 0.0),
    rec(Cpu, "qwen36-35b", 32768, 512, "q8_0", "q8_0", 1, false, &[340.0], 62.81, 537.03, 0.0),
    rec(Cpu, "qwen36-35b", 32768, 512, "q8_0", "q4_0", 1, false, &[260.0], 62.81, 537.03, 0.0),
    rec(Cpu, "qwen36-35b", 32768, 512, "f16", "f16", 4, false, &[640.0], 251.25, 513.03, 0.0),
    rec(Cpu, "qwen36-35b", 131072, 2048, "q8_0", "q8_0", 1, false, &[1360.0], 62.81, 2532.1, 0.0),
    rec(Cpu, "glm53-flash", 4096, 512, "f16", "f16", 1, false, &[44.0, 33.0], 145.56, 346.47, 0.0),
    rec(Cpu, "glm53-flash", 32768, 512, "f16", "f16", 1, false, &[352.0, 264.0], 145.56, 362.37, 0.0),
    rec(Cpu, "glm53-flash", 131072, 512, "f16", "f16", 1, false, &[1408.0, 1056.0], 145.56, 399.09, 0.0),
    rec(Cpu, "glm53-flash", 32768, 2048, "f16", "f16", 1, false, &[352.0, 264.0], 145.56, 1446.38, 0.0),
    rec(Cpu, "glm53-flash", 32768, 512, "q8_0", "q8_0", 1, false, &[187.0, 140.25], 145.56, 362.37, 0.0),
    rec(Cpu, "glm53-flash", 32768, 512, "f16", "f16", 4, false, &[352.0, 264.0], 582.25, 361.85, 0.0),
    rec(Cpu, "glm53-flash", 131072, 2048, "q8_0", "q8_0", 1, false, &[748.0, 561.0], 145.56, 1583.97, 0.0),
    rec(Cuda, "qwen3-0.6b", 4096, 512, "f16", "f16", 1, false, &[448.0], 0.0, 308.77, 12.02),
    rec(Cuda, "qwen3-0.6b", 32768, 512, "f16", "f16", 1, false, &[3584.0], 0.0, 336.77, 40.02),
    rec(Cuda, "qwen3-0.6b", 32768, 2048, "f16", "f16", 1, false, &[3584.0], 0.0, 1347.06, 160.08),
    rec(Cuda, "qwen3-0.6b", 32768, 512, "q8_0", "q8_0", 1, false, &[1904.0], 0.0, 336.77, 40.02),
    rec(Cuda, "qwen3-0.6b", 32768, 512, "f16", "f16", 4, false, &[3584.0], 0.0, 312.77, 16.02),
    rec(Cuda, "gemma3-1b", 4096, 512, "f16", "f16", 1, false, &[16.0, 22.0], 0.0, 526.03, 14.03),
    rec(Cuda, "gemma3-1b", 32768, 512, "f16", "f16", 1, false, &[128.0, 22.0], 0.0, 554.03, 42.03),
    rec(Cuda, "gemma3-1b", 32768, 2048, "f16", "f16", 1, false, &[128.0, 55.0], 0.0, 2222.1, 174.12),
    rec(Cuda, "gemma3-1b", 32768, 512, "q8_0", "q8_0", 1, false, &[68.0, 11.69], 0.0, 554.03, 42.03),
    rec(Cuda, "gemma3-1b", 32768, 512, "f16", "f16", 4, false, &[128.0, 88.0], 0.0, 530.03, 18.03),
    rec(Cuda, "glm47-flash", 4096, 512, "f16", "f16", 1, false, &[211.5], 0.0, 322.51, 20.02),
    rec(Cuda, "glm47-flash", 32768, 512, "f16", "f16", 1, false, &[1692.0], 0.0, 350.51, 48.02),
    rec(Cuda, "glm47-flash", 131072, 512, "f16", "f16", 1, false, &[6768.0], 0.0, 446.51, 144.02),
    rec(Cuda, "glm47-flash", 32768, 2048, "f16", "f16", 1, false, &[1692.0], 0.0, 1402.05, 192.06),
    rec(Cuda, "glm47-flash", 32768, 512, "q8_0", "q8_0", 1, false, &[898.88], 0.0, 350.51, 48.02),
    rec(Cuda, "glm47-flash", 32768, 512, "f16", "f16", 4, false, &[1692.0], 0.0, 326.51, 24.02),
    rec(Cuda, "glm47-flash", 32768, 2048, "f16", "f16", 1, true, &[1692.0], 0.0, 1402.04, 192.06),
    rec(Cuda, "glm47-flash", 131072, 2048, "f16", "f16", 1, true, &[6768.0], 0.0, 1786.04, 576.06),
    rec(Cuda, "glm47-flash", 32768, 512, "f16", "f16", 1, true, &[1692.0], 0.0, 350.51, 48.02),
    rec(Cuda, "qwen36-35b", 4096, 512, "f16", "f16", 1, false, &[80.0], 62.81, 505.02, 20.03),
    rec(Cuda, "qwen36-35b", 32768, 512, "f16", "f16", 1, false, &[640.0], 62.81, 569.02, 48.03),
    rec(Cuda, "qwen36-35b", 131072, 512, "f16", "f16", 1, false, &[2560.0], 62.81, 629.02, 144.03),
    rec(Cuda, "qwen36-35b", 32768, 2048, "f16", "f16", 1, false, &[640.0], 62.81, 2276.09, 192.1),
    rec(Cuda, "qwen36-35b", 32768, 512, "q8_0", "q8_0", 1, false, &[340.0], 62.81, 569.02, 48.03),
    rec(Cuda, "qwen36-35b", 32768, 512, "f16", "f16", 4, false, &[640.0], 251.25, 509.02, 24.03),
    rec(Cuda, "qwen36-35b", 32768, 2048, "f16", "f16", 1, true, &[640.0], 62.81, 2148.02, 192.1),
    rec(Cuda, "qwen36-35b", 131072, 2048, "f16", "f16", 1, true, &[2560.0], 62.81, 2148.02, 576.1),
    rec(Cuda, "qwen36-35b", 32768, 512, "f16", "f16", 1, true, &[640.0], 62.81, 537.0, 48.03),
    rec(Cuda, "glm53-flash", 4096, 512, "f16", "f16", 1, false, &[44.0, 33.0], 145.56, 338.46, 33.13),
    rec(Cuda, "glm53-flash", 32768, 512, "f16", "f16", 1, false, &[352.0, 264.0], 145.56, 346.36, 40.43),
    rec(Cuda, "glm53-flash", 131072, 512, "f16", "f16", 1, false, &[1408.0, 1056.0], 145.56, 391.08, 65.46),
    rec(Cuda, "glm53-flash", 32768, 2048, "f16", "f16", 1, false, &[352.0, 264.0], 145.56, 1382.36, 160.69),
    rec(Cuda, "glm53-flash", 32768, 512, "q8_0", "q8_0", 1, false, &[187.0, 140.25], 145.56, 346.36, 40.43),
    rec(Cuda, "glm53-flash", 32768, 512, "f16", "f16", 4, false, &[352.0, 264.0], 582.25, 345.85, 40.43),
    rec(Cuda, "glm53-flash", 32768, 2048, "f16", "f16", 1, true, &[352.0, 264.0], 145.56, 2036.35, 160.69),
    rec(Cuda, "glm53-flash", 131072, 2048, "f16", "f16", 1, true, &[1408.0, 1056.0], 145.56, 2135.45, 257.72),
    rec(Cuda, "glm53-flash", 32768, 512, "f16", "f16", 1, true, &[352.0, 264.0], 145.56, 1427.86, 40.43),
];

fn fixture(model: &str) -> ModelAccounting {
    let file = match model {
        "qwen3-0.6b" => "Qwen3-0.6B-Q8_0.header.gguf",
        "gemma3-1b" => "gemma-3-1b-it-Q4_K_M.header.gguf",
        "glm47-flash" => "GLM-4.7-Flash-Q4_K_M.header.gguf",
        "qwen36-35b" => "Qwen3.6-35B-A3B-UD-Q4_K_M.header.gguf",
        "glm53-flash" => "glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
        other => panic!("no fixture for {other}"),
    };
    let p: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests", "fixtures", "gguf", file].iter().collect();
    account_model(&p).unwrap()
}

fn params(r: &Rec) -> ContextParams {
    ContextParams {
        ctx: r.ctx,
        kv_k: KvType::parse(r.ctk).unwrap(),
        kv_v: KvType::parse(r.ctv).unwrap(),
        n_parallel: r.np,
        ubatch: r.ub,
        experts_on_host: r.host_experts,
    }
}

const MIB: f64 = 1024.0 * 1024.0;

fn mib(b: u64) -> f64 {
    b as f64 / MIB
}

fn rel(got: f64, want: f64) -> f64 {
    got / want - 1.0
}

/// KV, sliding-window and indexer caches plus recurrent state match the
/// allocator to within 0.5 % (in practice exactly) for every family, cache
/// type, context, micro-batch and slot count recorded.
#[test]
fn caches_and_state_match_the_allocator() {
    for r in RECORDS {
        let c = context_memory(&fixture(r.model), &params(r));
        let want: f64 = r.caches.iter().sum();
        let got = mib(c.kv_bytes + c.swa_bytes + c.indexer_bytes);
        assert!(rel(got, want).abs() < 0.005, "{:?} {} ctx {}: caches {got} vs {want}", r.backend, r.model, r.ctx);
        if r.caches.len() == 2 {
            // Second cache: SWA (Gemma) or the DSA indexer (GLM-5.3).
            let second = mib(c.swa_bytes + c.indexer_bytes);
            assert!(rel(second, r.caches[1]).abs() < 0.005, "{} ctx {}: {second} vs {}", r.model, r.ctx, r.caches[1]);
        }
        let state = mib(c.state_bytes);
        assert!((state - r.state).abs() < 0.01, "{} np {}: state {state} vs {}", r.model, r.np, r.state);
    }
}

/// Compute buffers: within ±6 % with experts on the device, and between
/// −2 % and +20 % with routed experts in host memory (the estimate errs high
/// there, which is the safe side). The CUDA host buffer is never more than
/// 15 % under.
#[test]
fn compute_buffer_within_tolerance() {
    for r in RECORDS {
        let c = context_memory(&fixture(r.model), &params(r));
        let e = rel(mib(c.compute_bytes), r.compute);
        let (lo, hi) = if r.host_experts { (-0.02, 0.20) } else { (-0.06, 0.06) };
        assert!(
            e >= lo && e <= hi,
            "{:?} {} ctx {} ub {} np {}: compute {:.2} vs {} ({:+.1} %)",
            r.backend,
            r.model,
            r.ctx,
            r.ub,
            r.np,
            mib(c.compute_bytes),
            r.compute,
            e * 100.0
        );
        if r.host_compute > 0.0 {
            let h = rel(mib(c.host_compute_bytes), r.host_compute);
            assert!(
                h > -0.15,
                "{} ctx {}: host compute {:.2} vs {}",
                r.model,
                r.ctx,
                mib(c.host_compute_bytes),
                r.host_compute
            );
        }
    }
}

/// What the planner reserves for compute (prediction plus its safety margin)
/// is never less than what llama.cpp allocated.
#[test]
fn planner_compute_reservation_never_short() {
    for r in RECORDS {
        let c = context_memory(&fixture(r.model), &params(r));
        let reserved = mib(c.compute_bytes) * (1.0 + crate::planner::COMPUTE_MARGIN_FRAC);
        assert!(reserved >= r.compute, "{:?} {} ctx {}: reserve {reserved:.1} < {}", r.backend, r.model, r.ctx, r.compute);
    }
}

/// GLM-5.3-Flash (UD-Q2_K_XL header), f16 cache, one sequence: the MLA
/// latent and DSA indexer grow with context, the KDA state doesn't.
#[test]
fn glm53_context_memory_by_length() {
    let m = fixture("glm53-flash");
    assert_eq!(m.ctx_train, 1 << 20);
    let rows = [(4096u32, 44.0, 33.0), (32768, 352.0, 264.0), (131072, 1408.0, 1056.0), (1 << 20, 11264.0, 8448.0)];
    for (ctx, mla, idx) in rows {
        let c = context_memory(&m, &ContextParams { ctx, ..Default::default() });
        println!(
            "GLM-5.3 ctx {ctx:>8}: MLA {:>8.2} MiB  indexer {:>8.2} MiB  state {:.2} MiB  compute {:.2} MiB  total {:.2} MiB",
            mib(c.kv_bytes),
            mib(c.indexer_bytes),
            mib(c.state_bytes),
            mib(c.compute_bytes),
            mib(c.total_bytes())
        );
        assert_eq!(mib(c.kv_bytes), mla, "ctx {ctx}");
        assert_eq!(mib(c.indexer_bytes), idx, "ctx {ctx}");
        assert_eq!(c.swa_bytes, 0);
        assert!((mib(c.state_bytes) - 145.56).abs() < 0.01);
    }
    // Four sequences: four states, the same total cells.
    let c4 = context_memory(&m, &ContextParams { ctx: 32768, n_parallel: 4, ..Default::default() });
    assert!((mib(c4.state_bytes) - 4.0 * 145.56).abs() < 0.05);
    assert_eq!(mib(c4.kv_bytes), 352.0);
}

/// Sliding-window caches stop growing at the window; full-attention ones
/// keep growing.
#[test]
fn swa_cache_is_capped_at_the_window() {
    let m = fixture("gemma3-1b");
    let at = |ctx| context_memory(&m, &ContextParams { ctx, ..Default::default() });
    let (a, b) = (at(32768), at(131072));
    assert_eq!(a.swa_bytes, b.swa_bytes);
    assert_eq!(b.kv_bytes, a.kv_bytes * 4);
}

/// The binary search finds the largest context within a memory budget.
#[test]
fn largest_ctx_within_budget() {
    let m = &fixture("qwen3-0.6b");
    let within = |budget: u64| {
        move |ctx: u32| context_memory(m, &ContextParams { ctx, ..Default::default() }).cache_bytes() <= budget
    };
    // f16 KV is 112 KiB per token (28 layers × (1024 + 1024) × 2 bytes), so
    // 1 GiB holds 9362 tokens, 9216 after rounding down to 256.
    assert_eq!(largest_ctx(40960, within(1 << 30)), Some(9216));
    assert_eq!(largest_ctx(40960, within(1)), None);
    assert_eq!(largest_ctx(4096, within(u64::MAX)), Some(4096));
    assert_eq!(largest_ctx(255, within(u64::MAX)), None);
}
