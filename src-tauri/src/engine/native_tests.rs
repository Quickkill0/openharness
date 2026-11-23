use super::{manager::{self, ModelLoadArgs}, native, EngineManagerHandle};
use crate::planner::{self, PlanOverrides};
use std::path::PathBuf;

pub(crate) fn stub() -> (tempfile::TempDir, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let bin = dir.path().join(native::BINARY);
    let source = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/native_stub.rs");
    let output = std::process::Command::new("rustc").arg("--edition=2021").arg(source).arg("-o").arg(&bin).output().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    (dir, bin)
}

pub(crate) async fn rig() -> (tempfile::TempDir, EngineManagerHandle, PathBuf, ModelLoadArgs) {
    let (dir, bin) = stub();
    let db = sqlx::sqlite::SqlitePoolOptions::new().max_connections(1).connect("sqlite::memory:").await.unwrap();
    sqlx::migrate!("./migrations").run(&db).await.unwrap();
    let http = reqwest::Client::builder().no_proxy().build().unwrap();
    let engine = manager::start_with_native(db, http, bin);
    // No persisted llama engine: a hand-placed native install is sufficient.
    let info = engine.native_info().await.unwrap();
    assert!(info.installed && !info.hosted && info.support().streaming);
    assert!(!info.support().cuda_tier);
    let first = dir.path().join("model-00001-of-00002.gguf");
    let second = dir.path().join("model-00002-of-00002.gguf");
    std::fs::write(&first, []).unwrap();
    std::fs::write(&second, []).unwrap();
    let model = planner::accounting::account_model(&PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf/glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf")).unwrap();
    let hw = planner::HardwareProfile { gpu: None, ram_total_bytes: 64 << 30, ram_available_bytes: 40 << 30, commit_available_bytes: None, storage: None, bandwidth: planner::hardware::Bandwidths { vram_gbps: 0.0, pcie_gbps: 25.0, ram_gbps: 60.0, ssd_gbps: 3.0, ssd_measured: false }, cpu_threads: 8 };
    let plan = planner::plan_with_native(&model, &hw, &planner::VramBudget::no_gpu(), &PlanOverrides { ctx: Some(4096), ..Default::default() }, None, &info.support(), None);
    assert!(plan.feasible, "{plan:#?}");
    assert_eq!(plan.engine_kind, planner::EngineKind::SsdStream);
    let args = ModelLoadArgs { engine_kind: plan.engine_kind, ctx: plan.ctx, ngl: plan.gpu_layers, batch_size: Some(plan.ubatch), extra_args: plan.extra_args(), ..Default::default() };
    (dir, engine, second, args)
}

#[tokio::test]
async fn native_real_process_from_plan_through_manager_health_logs_stop_and_refusal() {
    let (_dir, engine, path, args) = rig().await;
    let port = engine.load_model("glm-native".into(), path.clone(), args.clone()).await.unwrap();
    let client = reqwest::Client::builder().no_proxy().build().unwrap();
    assert!(client.get(format!("http://127.0.0.1:{port}/health")).send().await.unwrap().status().is_success());
    let ready = engine.status().await;
    assert!(matches!(ready, super::EngineStatus::Ready { launcher: planner::EngineKind::SsdStream, .. }));
    assert_eq!(serde_json::to_value(&ready).unwrap()["launcher"], "ssd_stream");
    let logs = engine.logs().await;
    assert!(logs.iter().filter(|l| l.contains("GET /health")).count() >= 2); // 503 then 200
    assert!(engine.server_pid().await.is_some());
    engine.unload_model().await.unwrap();
    assert!(matches!(engine.status().await, super::EngineStatus::Idle));
    assert!(client.get(format!("http://127.0.0.1:{port}/health")).send().await.is_err());
    let refusal = path.parent().unwrap().join("refuse.gguf");
    std::fs::write(&refusal, []).unwrap();
    let err = engine.load_model("refused".into(), refusal, args).await.unwrap_err();
    assert_eq!(err.to_string(), "[oh-ssd] unsupported expert layout: F32 experts are refused");
    assert!(matches!(engine.status().await, super::EngineStatus::Failed { launcher: Some(planner::EngineKind::SsdStream), .. }));
}

#[tokio::test]
async fn failed_load_names_the_launcher_that_failed() {
    let (_dir, engine, path, args) = rig().await;
    // The rig has no persisted llama.cpp, so this fails before any process starts.
    let llama = ModelLoadArgs { engine_kind: planner::EngineKind::LlamaServer, ..args };
    engine.load_model("glm-llama".into(), path, llama).await.unwrap_err();
    let failed = engine.status().await;
    assert!(matches!(failed, super::EngineStatus::Failed { launcher: Some(planner::EngineKind::LlamaServer), .. }));
    assert_eq!(serde_json::to_value(&failed).unwrap()["launcher"], "llama_server");
}

#[tokio::test]
async fn check_again_recovers_a_failed_probe_without_losing_a_good_one() {
    let (dir, bin) = stub();
    let marker = dir.path().join("probe-fails");
    let mut discovery = native::Discovery::default();
    assert!(discovery.cached_at(&bin).await.error.unwrap().contains("a model is starting"), "no probe has run yet");

    // The binary fails its probe, then the fix lands outside it.
    std::fs::write(&marker, []).unwrap();
    assert!(discovery.at(bin.clone()).await.error.is_some());
    std::fs::remove_file(&marker).unwrap();
    assert!(discovery.at(bin.clone()).await.error.is_some(), "an unchanged binary keeps its cached result");
    discovery.forget_failure();
    let recovered = discovery.at(bin.clone()).await;
    assert!(recovered.error.is_none() && recovered.support().streaming);

    // A good result survives a retry even if a probe would fail right now.
    std::fs::write(&marker, []).unwrap();
    discovery.forget_failure();
    assert!(discovery.at(bin).await.support().streaming);
}

#[tokio::test]
async fn native_discovery_caches_per_binary_and_not_stamp() {
    let (dir, bin) = stub();
    let mut discovery = native::Discovery::default();
    let first = discovery.at(bin.clone()).await;
    assert!(first.installed && first.support().streaming);
    assert_eq!(discovery.at(bin.clone()).await.info, first.info);
    let missing = discovery.at(dir.path().join("missing")).await;
    assert!(!missing.installed);
    // Replacing the binary with non-executable content invalidates a successful probe.
    std::fs::write(&bin, b"not a native executable").unwrap();
    let changed = discovery.at(bin).await;
    assert!(changed.installed && changed.info.is_none() && changed.error.is_some());
}
