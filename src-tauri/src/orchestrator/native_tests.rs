use super::*;
use crate::{db::models::{NewModel, NewModelFile}, planner::{HardwareProfile, hardware::Bandwidths, native::{EnginePreference, RAM_RESERVE}}};
use std::sync::Mutex;

pub(crate) struct Rig {
    pub dir: tempfile::TempDir,
    pub db: SqlitePool,
    pub engine: EngineManagerHandle,
    pub orchestrator: OrchestratorHandle,
    pub activity: ChatActivity,
    pub hardware: Arc<Mutex<HardwareProfile>>,
    pub slow: PathBuf,
}
struct NoGpu;
impl crate::gpu::GpuProbe for NoGpu {
    fn read_primary(&self) -> Result<crate::gpu::GpuSample, crate::gpu::GpuError> { Err(crate::gpu::GpuError::NoDriver) }
}

pub(crate) async fn rig() -> Rig {
    let (dir, binary) = crate::engine::native_tests::stub();
    let db = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1).connect("sqlite::memory:").await.unwrap();
    sqlx::migrate!("./migrations").run(&db).await.unwrap();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf/glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf");
    let mut slow = PathBuf::new();
    for name in ["slow", "warm", "slow-legacy"] {
        let mut files = Vec::new();
        for (i, src) in gguf::split_files(&fixture).unwrap().into_iter().enumerate() {
            let rel = format!("{name}-{:05}-of-00004.gguf", i + 1);
            let path = dir.path().join(&rel);
            std::fs::copy(src, &path).unwrap();
            if name == "slow" && i == 0 { slow = path.clone(); }
            files.push(NewModelFile { role: if i == 0 { "primary" } else { "weight" }.into(), rel_path: rel, path: path.to_string_lossy().into_owned(), sha256: None, size: 109_000_000_000 / 4 });
        }
        crate::db::models::insert_model_with_files(&db, &NewModel { id: name.into(), repo_id: "test/GLM".into(), revision: "test".into(), display_name: name.into(), architecture: Some("glm5-next".into()), capability: "text-generation".into(), format: "gguf".into(), task: Some("text-generation".into()), local_path: None, params_b: None, quantization: None, context_train: Some(1048576), installed_at: 0 }, &files).await.unwrap();
    }
    // Also test the legacy explicit-ngl launch against the SAME control pump.
    crate::db::engines::upsert(&db, &crate::db::engines::EngineRow { id: "stub".into(), version: crate::engine::ENGINE_BUILD.into(), variant: "cpu".into(), path: binary.to_string_lossy().into_owned(), sha256: String::new(), active: true, installed_at: 0 }).await.unwrap();
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let engine = crate::engine::manager::start_with_native(db.clone(), http.clone(), binary);
    let hardware = Arc::new(Mutex::new(HardwareProfile { gpu: None, ram_total_bytes: 64 * MIB * 1024, ram_available_bytes: 40 * MIB * 1024, commit_available_bytes: None, storage: None, bandwidth: Bandwidths { vram_gbps: 0.0, ram_gbps: 60.0, pcie_gbps: 25.0, ssd_gbps: 3.0, ssd_measured: false }, cpu_threads: 8 }));
    let probe = hardware.clone();
    let (_writer, activity) = crate::chat::activity::channel();
    let worker = crate::worker::manager::start(http.clone(), None);
    let orchestrator = start_with_hardware(db.clone(), engine.clone(), worker, crate::gpu::start_with_probe(Arc::new(NoGpu)), crate::tts::start(http), crate::planner::store::start(dir.path().join("planner.json")), activity.clone(), Some(Arc::new(move || probe.lock().unwrap().clone())));
    Rig { dir, db, engine, orchestrator, activity, hardware, slow }
}

pub(crate) async fn loading(engine: &EngineManagerHandle) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            if matches!(engine.status().await, EngineStatus::Loading { .. }) && engine.logs().await.iter().any(|l| l.contains("GET /health")) { break; }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }).await.expect("control reads must not block behind startup");
}
fn request(id: &str) -> TextLoadRequest { TextLoadRequest { model_id: id.into(), overrides: Some(PlanOverrides { ctx: Some(4096), ..Default::default() }), ..Default::default() } }

#[tokio::test]
async fn pending_startup_keeps_controls_ledger_and_conditional_cancellation_live_and_queues_next_load() {
    let rig = rig().await;
    let actor = rig.orchestrator.clone();
    let first = tokio::spawn(async move { actor.load_text(request("slow")).await });
    loading(&rig.engine).await;
    let ledger = tokio::time::timeout(Duration::from_secs(1), rig.orchestrator.ledger()).await.unwrap().unwrap();
    assert_eq!(ledger.residents[0].model_id.as_deref(), Some("slow"));
    assert!(rig.orchestrator.unload_if_model(TEXT_RESIDENT_ID.into(), Some("warm".into())).await.is_err());
    assert!(!first.is_finished());
    let actor = rig.orchestrator.clone();
    let second = tokio::spawn(async move { actor.load_text(request("warm")).await });
    tokio::task::yield_now().await;
    rig.orchestrator.ledger().await.unwrap();
    rig.orchestrator.unload_if_model(TEXT_RESIDENT_ID.into(), Some("slow".into())).await.unwrap();
    assert!(matches!(first.await.unwrap(), Err(AppError::EngineLoadCancelled)));
    assert!(matches!(second.await.unwrap().unwrap(), LoadOutcome::Loaded { model_id, .. } if model_id == "warm"));
    assert_eq!(rig.orchestrator.ledger().await.unwrap().residents[0].model_id.as_deref(), Some("warm"));
    rig.orchestrator.unload(TEXT_RESIDENT_ID.into()).await.unwrap();
}

#[tokio::test]
async fn ready_unload_race_never_resurrects_a_plan_and_explicit_ngl_uses_same_control_path() {
    let rig = rig().await;
    let actor = rig.orchestrator.clone();
    let load = tokio::spawn(async move { actor.load_text(request("slow")).await });
    loading(&rig.engine).await;
    std::fs::write(format!("{}.ready", rig.slow.display()), []).unwrap();
    rig.orchestrator.unload_if_idle(TEXT_RESIDENT_ID.into()).await.unwrap();
    let result = load.await.unwrap();
    assert!(result.is_ok() || matches!(result, Err(AppError::EngineLoadCancelled)));
    assert!(matches!(rig.engine.status().await, EngineStatus::Idle));
    assert!(rig.orchestrator.ledger().await.unwrap().residents.is_empty());
    assert!(rig.orchestrator.current_text_plan().await.is_none());
    let actor = rig.orchestrator.clone();
    let load = tokio::spawn(async move { actor.load_text(TextLoadRequest { model_id: "slow-legacy".into(), ngl: Some(0), ..Default::default() }).await });
    loading(&rig.engine).await;
    rig.orchestrator.unload_if_idle(TEXT_RESIDENT_ID.into()).await.unwrap();
    assert!(matches!(load.await.unwrap(), Err(AppError::EngineLoadCancelled)));
    assert!(rig.orchestrator.ledger().await.unwrap().residents.is_empty());
}

#[tokio::test]
async fn check_again_during_startup_answers_from_the_cache_and_keeps_cancel_free() {
    let rig = rig().await;
    let marker = |name: &str| rig.dir.path().join(name);
    let quick = std::time::Duration::from_secs(5);

    // A failed probe is cached, then its retry would hang.
    std::fs::write(marker("probe-fails"), []).unwrap();
    let failed = rig.engine.native_info().await.unwrap();
    assert!(failed.error.is_some());
    std::fs::remove_file(marker("probe-fails")).unwrap();
    std::fs::write(marker("probe-hangs"), []).unwrap();

    let actor = rig.orchestrator.clone();
    let load = tokio::spawn(async move { actor.load_text(TextLoadRequest { model_id: "slow-legacy".into(), ngl: Some(0), ..Default::default() }).await });
    loading(&rig.engine).await;
    let answer = tokio::time::timeout(quick, rig.engine.native_info_retry()).await.expect("the startup pump must not wait on a probe").unwrap();
    assert_eq!(answer.error, failed.error);
    assert!(matches!(tokio::time::timeout(quick, rig.engine.status()).await.unwrap(), EngineStatus::Loading { .. }));
    tokio::time::timeout(quick, rig.orchestrator.unload_if_idle(TEXT_RESIDENT_ID.into())).await.expect("cancel stays immediate").unwrap();
    assert!(matches!(load.await.unwrap(), Err(AppError::EngineLoadCancelled)));

    // Once startup is over the retry really runs and finds the fix.
    std::fs::remove_file(marker("probe-hangs")).unwrap();
    let recovered = rig.engine.native_info_retry().await.unwrap();
    assert!(recovered.error.is_none() && recovered.support().streaming);
}

#[tokio::test]
async fn installed_text_fit_uses_native_admission_and_checks_cpu_ram() {
    let rig = rig().await;
    let plan = rig.orchestrator.plan_text("warm".into(), PlanOverrides::default()).await.unwrap();
    assert!(plan.feasible);
    let fit = crate::engine::fit::FitEstimate::from_plan(plan, true, 5000, 0);
    assert!(matches!(fit.verdict, crate::engine::fit::FitVerdict::Fits));
    assert_eq!(fit.estimated_mb, 0); // CPU-only native build, NOT the 109GB GGUF.
    assert_eq!(fit.kv_at_ctx_mb, 0); // Native context is on host.
    rig.hardware.lock().unwrap().ram_available_bytes = RAM_RESERVE - 1;
    let plan = rig.orchestrator.plan_text("warm".into(), PlanOverrides::default()).await.unwrap();
    let fit = crate::engine::fit::FitEstimate::from_plan(plan, false, 0, 0);
    assert!(matches!(fit.verdict, crate::engine::fit::FitVerdict::WontFit));
    assert!(fit.plan.unwrap().infeasible_reason.unwrap().contains("12 GiB"));
}

#[tokio::test]
async fn warmed_private_cache_is_credited_for_preview_and_reload_but_never_after_exit_or_refusal() {
    let rig = rig().await;
    rig.orchestrator.load_text(request("warm")).await.unwrap();
    let private = rig.engine.reclaimable_host().await.resident;
    assert!(private > 32 * MIB && private < 256 * MIB, "resident-private probe, not model size/commit: {private}");
    let ctx = PlanOverrides { ctx: Some(8192), ..Default::default() };
    let baseline = rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap();
    let n = baseline.native.as_ref().unwrap();
    let fixed = (n.ram_budget_mib - n.expert_cache_mib) * MIB;
    rig.hardware.lock().unwrap().ram_available_bytes = RAM_RESERVE + fixed - private / 2;
    let preview = rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap();
    assert!(preview.feasible, "measured private memory must be credited before subtracting reserve");
    assert!(preview.native.as_ref().unwrap().ram_budget_mib * MIB > fixed);
    let before = rig.engine.server_pid().await;
    let result = rig.orchestrator.load_text(TextLoadRequest { model_id: "warm".into(), overrides: Some(ctx), ..Default::default() }).await.unwrap();
    assert!(matches!(result, LoadOutcome::Loaded { plan: Some(p), .. } if p.feasible));
    assert_ne!(before, rig.engine.server_pid().await);
    let pid = rig.engine.server_pid().await;
    let refused = rig.orchestrator.load_text(TextLoadRequest { model_id: "warm".into(), overrides: Some(PlanOverrides { engine: EnginePreference::LlamaServer, ..Default::default() }), ..Default::default() }).await;
    assert!(refused.is_err());
    assert_eq!(pid, rig.engine.server_pid().await); // Refusal keeps the live model.
    if let EngineStatus::Ready { port, .. } = rig.engine.status().await {
        let _ = reqwest::Client::builder().no_proxy().build().unwrap().get(format!("http://127.0.0.1:{port}/test/exit")).send().await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(rig.engine.reclaimable_host().await.resident, 0);
    assert!(!rig.orchestrator.plan_text("warm".into(), PlanOverrides { ctx: Some(8192), ..Default::default() }).await.unwrap().feasible);
    assert!(rig.orchestrator.ledger().await.unwrap().residents.is_empty());
}

#[tokio::test]
async fn reload_credits_the_running_models_commit_to_the_commit_limit_and_never_to_ram() {
    let rig = rig().await;
    let ctx = PlanOverrides { ctx: Some(8192), ..Default::default() };
    let set = |c: Option<u64>| rig.hardware.lock().unwrap().commit_available_bytes = c;
    // Smallest available commit that admits this context with nothing loaded.
    let (mut lo, mut hi) = (0u64, 64 * 1024);
    while lo < hi {
        let mid = (lo + hi) / 2;
        set(Some(mid * MIB));
        if rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap().feasible { hi = mid } else { lo = mid + 1 }
    }
    let need = lo * MIB;
    assert!(need > 0 && lo < 64 * 1024, "commit must be what decides: {lo} MiB");
    set(None);
    rig.orchestrator.load_text(request("warm")).await.unwrap();
    let ram_only = rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap();
    let credit = rig.engine.reclaimable_host().await;
    assert!(credit.commit >= credit.resident && credit.commit > 32 * MIB, "{credit:?}");
    // The running model holds the commit, so what is left cannot admit a
    // reload by itself. Stopping it for the reload gives its commit back.
    let left = need - credit.commit / 2;
    set(Some(left));
    let preview = rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap();
    assert!(preview.feasible, "{:?}", preview.infeasible_reason);
    // Commit credit is not RAM: the system memory envelope is unchanged.
    assert_eq!(preview.native.as_ref().unwrap().ram_budget_mib, ram_only.native.as_ref().unwrap().ram_budget_mib);
    let before = rig.engine.server_pid().await;
    let result = rig.orchestrator.load_text(TextLoadRequest { model_id: "warm".into(), overrides: Some(ctx.clone()), ..Default::default() }).await.unwrap();
    assert!(matches!(result, LoadOutcome::Loaded { plan: Some(p), .. } if p.feasible));
    assert_ne!(before, rig.engine.server_pid().await);
    // Nothing is credited once the process is gone.
    if let EngineStatus::Ready { port, .. } = rig.engine.status().await {
        let _ = reqwest::Client::builder().no_proxy().build().unwrap().get(format!("http://127.0.0.1:{port}/test/exit")).send().await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(rig.engine.reclaimable_host().await, Default::default());
    let gone = rig.orchestrator.plan_text("warm".into(), ctx).await.unwrap();
    assert!(!gone.feasible);
    assert!(gone.infeasible_reason.unwrap().contains("commit limit"));
}

#[tokio::test]
async fn reload_credits_the_running_models_graphics_memory_so_dense_layers_stay_on_the_card() {
    let rig = rig().await;
    // A CUDA build of the stub, with a 24 GiB card that has 23 GiB free.
    std::fs::write(rig.dir.path().join("cuda"), []).unwrap();
    let card = |free| rig.hardware.lock().unwrap().gpu = Some(crate::planner::hardware::GpuInfo { name: "RTX 4090".into(), vram_total_bytes: 24 * 1024 * MIB, vram_free_bytes: free });
    card(23 * 1024 * MIB);
    let ctx = PlanOverrides { ctx: Some(8192), ..Default::default() };
    let empty = rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap();
    assert!(empty.feasible && empty.gpu_layers == -1, "{:?}", empty.infeasible_reason);
    let held = empty.tiers.vram_total();
    assert!(held > 10 * 1024 * MIB, "dense layers, context and tier on the card: {held}");
    let result = rig.orchestrator.load_text(TextLoadRequest { model_id: "warm".into(), overrides: Some(ctx.clone()), ..Default::default() }).await.unwrap();
    assert!(matches!(result, LoadOutcome::Loaded { plan: Some(p), .. } if p.gpu_layers == -1));
    // The running process now holds what its plan put on the card.
    card(23 * 1024 * MIB - held.div_ceil(MIB) * MIB);
    let preview = rig.orchestrator.plan_text("warm".into(), ctx.clone()).await.unwrap();
    assert_eq!((preview.gpu_layers, preview.tiers.vram_total()), (-1, held), "a reload gets back what stopping the old process frees");
    let before = rig.engine.server_pid().await;
    let reloaded = rig.orchestrator.load_text(TextLoadRequest { model_id: "warm".into(), overrides: Some(ctx.clone()), ..Default::default() }).await.unwrap();
    assert!(matches!(reloaded, LoadOutcome::Loaded { plan: Some(p), .. } if p.gpu_layers == -1 && p.tiers.vram_total() == held));
    assert_ne!(before, rig.engine.server_pid().await);
    // Once the process is gone nothing is credited: the same free memory plans less.
    if let EngineStatus::Ready { port, .. } = rig.engine.status().await {
        let _ = reqwest::Client::builder().no_proxy().build().unwrap().get(format!("http://127.0.0.1:{port}/test/exit")).send().await;
    }
    tokio::time::sleep(Duration::from_millis(50)).await;
    let gone = rig.orchestrator.plan_text("warm".into(), ctx).await.unwrap();
    assert!(gone.tiers.vram_total() < held, "{} vs {held}", gone.tiers.vram_total());
}
