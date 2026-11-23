mod api;
mod auth;
mod commands;
mod config;
mod db;
mod error;
mod state;

// Module shells — no logic until later milestones
mod assets;
mod chat;
mod downloader;
mod engine;
mod generation;
mod gpu;
mod inference;
mod models;
mod orchestrator;
mod planner;
mod provision;
mod store;
mod tts;
#[cfg(windows)]
mod winjob;
mod worker;

use std::time::Duration;

use state::AppState;
use tauri::Manager;

/// Entry point called by `main.rs`. Builds and runs the Tauri application.
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_store::Builder::default().build())
        // window-state persists window position/size across sessions
        .plugin(tauri_plugin_window_state::Builder::new().build())
        .setup(|app| {
            // One-time rebrand migration (Hangar AI → OpenHarness) — must run
            // BEFORE ensure_dirs(), which would otherwise create an empty new
            // dir and suppress the move. Best-effort: log and continue on error.
            let dir_moved = match config::migrate_legacy_data_dir() {
                Ok(moved) => moved,
                Err(e) => {
                    tracing::warn!("rebrand data-dir migration failed: {e}");
                    false
                }
            };

            // Ensure app data directories exist
            config::ensure_dirs().expect("failed to create app data directories");

            // Migrate the HF token credential to the new service name. Done
            // after dirs exist; independent of the data-dir move above.
            let token_migrated = match auth::migrate_legacy_token() {
                Ok(moved) => moved,
                Err(e) => {
                    tracing::warn!("rebrand HF-token migration failed: {e}");
                    false
                }
            };
            let migrated = dir_moved || token_migrated;

            // Initialise the database pool synchronously by blocking inside setup
            let pool = tauri::async_runtime::block_on(db::pool::init_pool())
                .expect("failed to initialise SQLite database");

            // Close job rows orphaned by the previous process before anything
            // can read them (unload checks, a reloaded UI's active-job snapshot).
            tauri::async_runtime::block_on(generation::heal_interrupted_jobs(&pool));

            // The dir move leaves absolute paths inside the DB (engines,
            // model_files, downloads) pointing at the legacy location —
            // rewrite the prefix before any actor reads them. Runs every
            // launch (idempotent, three prefix-match UPDATEs) so installs
            // that moved before this fix existed self-heal too.
            {
                let sep = std::path::MAIN_SEPARATOR.to_string();
                let old_prefix = config::legacy_app_data_dir().display().to_string() + &sep;
                let new_prefix = config::app_data_dir().display().to_string() + &sep;
                match tauri::async_runtime::block_on(db::rebrand::rewrite_legacy_path_prefixes(
                    &pool,
                    &old_prefix,
                    &new_prefix,
                )) {
                    Ok(n) if n > 0 => {
                        tracing::info!("Rebrand migration: rewrote {n} stored DB path(s)")
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!("rebrand DB path rewrite failed: {e}"),
                }
            }

            // Shared HTTP client — single instance for the process lifetime.
            let http = reqwest::Client::builder()
                .user_agent("OpenHarness/0.1 (+https://github.com/openharness/openharness)")
                .timeout(Duration::from_secs(30))
                .build()
                .expect("failed to build reqwest client");

            // Spawn the download manager actor. One job at a time: each job
            // already runs parallel range requests within its file, and two
            // simultaneous chunk swarms from one IP trip Hugging Face rate
            // limiting (429 backoff stalls that look like a paused download).
            // Additional jobs queue FIFO and start automatically.
            let downloads = downloader::start(pool.clone(), http.clone(), 1);

            // Start GPU telemetry actor.
            let gpu = gpu::start();

            // Start engine manager actor.
            let engine = engine::start(pool.clone(), http.clone());

            // Start chat actor.
            let chat = chat::start(pool.clone(), http.clone(), engine.clone());

            // Start the Python-worker manager actor (M8). The worker-py
            // package is a bundled resource in production, the repo checkout
            // in dev.
            let resource_dir = app.path().resource_dir().ok();
            let worker_src = worker::worker_src_dir(resource_dir);
            if worker_src.is_none() {
                tracing::warn!("worker-py source dir not found — Studio pipelines unavailable");
            }
            let has_worker_src = worker_src.is_some();
            let worker = worker::start(http.clone(), worker_src);

            // Auto-manage the worker (user never installs/starts it by hand):
            // background task installs the venv on first run (variant by GPU
            // detection) and starts the process; progress + failures stream
            // through the normal status broadcast. Never blocks startup.
            if has_worker_src {
                worker::spawn_auto_manage(worker.clone(), gpu.clone());
            }

            // Start the Qwen3-TTS sidecar actor (qwentts.cpp tts-server).
            let tts = tts::start(http.clone());

            // Load-planner store: SSD measurements + per-model calibration
            // (planner.json). Nothing is measured at startup.
            let planner = planner::store::start(planner::store::store_path());

            // Start the engine orchestrator (M8 §8.4): VRAM ledger + gated
            // loads across the text engine, worker pipelines, and TTS sidecar.
            let orchestrator = orchestrator::start(
                pool.clone(),
                engine.clone(),
                worker.clone(),
                gpu.clone(),
                tts.clone(),
                planner.clone(),
                chat.activity.clone(),
            );

            // Live actuals for the loaded text model (tok/s, VRAM, RAM).
            let runtime_stats = planner::runtime::start(
                http.clone(),
                engine.clone(),
                gpu.clone(),
                orchestrator.clone(),
                planner.clone(),
            );

            let initial_server_settings = tauri::async_runtime::block_on(async {
                match db::settings::get_setting(&pool, "api_server").await {
                    Ok(Some(value)) => serde_json::from_str::<api::ServerSettings>(&value)
                        .map_err(|_| api::ApiError::unavailable("Saved API server settings are invalid")),
                    Ok(None) => Ok(api::ServerSettings::default()),
                    Err(_) => Err(api::ApiError::unavailable("Could not read API server settings")),
                }
            });
            let server = api::start(
                std::sync::Arc::new(api::LocalBackend { native_activity: chat.activity.clone(), db: pool.clone(), engine: engine.clone(), orchestrator: orchestrator.clone() }),
                std::sync::Arc::new(api::CredentialStore), initial_server_settings,
            );

            // Mirror terminal worker job events into generation_jobs rows
            // (M8 §8.5; orphaned rows were healed above). Also the
            // single owner of asset promotion: the returned finalized bus
            // carries JobDone events with promoted (library) paths, and is
            // what the frontend progress channel subscribes to.
            let gen_events = generation::spawn_job_sync(pool.clone(), worker.clone());

            // Backfill GGUF metadata for rows that predate parser fixes
            // (large-vocab models installed while the header parse failed).
            {
                let pool_bf = pool.clone();
                tauri::async_runtime::spawn(async move {
                    match models::registry::backfill_gguf_metadata(&pool_bf).await {
                        Ok(n) if n > 0 => {
                            tracing::info!("backfilled GGUF metadata for {n} model(s)")
                        }
                        Ok(_) => {}
                        Err(e) => tracing::warn!("gguf metadata backfill failed: {e}"),
                    }
                });
            }

            // Boot-time catalog reconcile: scan models dir against DB.
            {
                let pool_clone = pool.clone();
                tauri::async_runtime::spawn(async move {
                    if let Err(e) = models::registry::scan_dir(&pool_clone).await {
                        tracing::warn!("scan_dir error on startup: {e}");
                    }
                });
            }

            // Backfill VibeVoice built-in preset voices for already-installed
            // checkpoints (the feature post-dates some installs). Best-effort +
            // idempotent: a no-op when every clip is already present, never
            // blocks startup.
            {
                let pool_heal = pool.clone();
                let http_heal = http.clone();
                tauri::async_runtime::spawn(async move {
                    models::vibevoice_voices::heal_installed_voices(&pool_heal, &http_heal).await;
                });
            }

            app.manage(AppState {
                db: pool,
                http,
                downloads,
                engine,
                gpu,
                chat,
                worker,
                orchestrator,
                server,
                tts,
                gen_events,
                planner,
                runtime_stats,
                migrated: std::sync::atomic::AtomicBool::new(migrated),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Local OpenAI-compatible server
            commands::server::server_status,
            commands::server::server_start,
            commands::server::server_stop,
            commands::server::server_update_settings,
            commands::server::server_regenerate_key,
            commands::server::server_get_key,
            commands::server::server_request_log,
            commands::server::server_subscribe_request_log,
            commands::server::server_connection_info,
            // Settings
            commands::settings::get_setting,
            commands::settings::set_setting,
            commands::settings::get_hf_token,
            commands::settings::set_hf_token,
            commands::settings::clear_hf_token,
            commands::settings::validate_hf_token,
            commands::settings::app_paths,
            commands::settings::consume_migration_notice,
            // HuggingFace browser (M2)
            commands::models::hf_search,
            commands::models::hf_model_details,
            commands::models::hf_model_tree,
            // Local model catalog (M3)
            commands::models::list_local_models,
            commands::models::delete_local_model,
            // Downloads (M3)
            commands::downloads::start_download,
            commands::downloads::pause_download,
            commands::downloads::resume_download,
            commands::downloads::cancel_download,
            commands::downloads::delete_download,
            commands::downloads::list_downloads,
            commands::downloads::subscribe_download_progress,
            // Repo download groups (M9 §9.2)
            commands::downloads::start_repo_download,
            commands::downloads::pause_download_group,
            commands::downloads::resume_download_group,
            commands::downloads::cancel_download_group,
            commands::downloads::delete_download_group,
            commands::downloads::list_download_groups,
            // Store intelligence (M9 §9.4)
            commands::store::store_runnability,
            commands::store::store_curated_picks,
            commands::store::store_quant_fits,
            // Engine + GPU (M4)
            commands::engine::engine_status,
            commands::engine::native_engine_info,
            commands::engine::native_engine_install,
            commands::engine::engine_active,
            commands::engine::engine_install,
            commands::engine::engine_switch_variant,
            commands::engine::engine_unload_model,
            commands::engine::engine_logs,
            commands::engine::engine_subscribe_status,
            commands::engine::gpu_status,
            commands::engine::gpu_subscribe_metrics,
            commands::engine::estimate_model_fit,
            // Host telemetry (Workbench dock)
            commands::system::system_memory,
            commands::system::reveal_model,
            // Chat (M5)
            commands::chat::list_conversations,
            commands::chat::create_conversation,
            commands::chat::get_conversation,
            commands::chat::delete_conversation,
            commands::chat::send_message,
            commands::chat::cancel_stream,
            commands::chat::chat_truncate_from,
            commands::chat::chat_regenerate,
            commands::chat::search_messages,
            commands::chat::get_conversation_settings,
            commands::chat::set_conversation_settings,
            commands::chat::subscribe_chat_events,
            // Python worker (M8)
            commands::worker::worker_status,
            commands::worker::worker_install,
            commands::worker::worker_start,
            commands::worker::worker_stop,
            commands::worker::worker_logs,
            commands::worker::subscribe_worker_status,
            // Orchestrator (M8 §8.4)
            commands::orchestrator::orchestrator_load_model,
            commands::orchestrator::orchestrator_unload,
            commands::orchestrator::orchestrator_unload_if,
            commands::orchestrator::orchestrator_ledger,
            commands::orchestrator::subscribe_orchestrator_ledger,
            // Unified load-first model flow (M16 W1)
            commands::orchestrator::load_model,
            commands::orchestrator::unload_model,
            // Load planner
            commands::planner::hardware_profile,
            commands::planner::benchmark_ssd,
            commands::planner::plan_load,
            commands::planner::context_memory,
            commands::planner::subscribe_runtime_stats,
            // Generation (M8 §8.5)
            commands::generation::generate_image,
            commands::generation::generate_video,
            commands::generation::synthesize_speech,
            commands::generation::transcribe_audio,
            commands::generation::list_generation_jobs,
            commands::generation::active_generation_jobs,
            commands::generation::cancel_generation_job,
            commands::generation::delete_generation_job,
            commands::generation::subscribe_generation_progress,
            commands::generation::save_voice_capture,
            // Qwen3-TTS engine
            commands::tts::tts_status,
            commands::tts::tts_installed,
            commands::tts::tts_cli_installed,
            commands::tts::tts_unload,
            commands::tts::tts_logs,
            commands::tts::tts_install,
            commands::tts::subscribe_tts_status,
            // Asset library (M10)
            commands::assets::list_assets,
            commands::assets::search_assets,
            commands::assets::delete_asset,
            commands::assets::reveal_asset,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Tauri application");
}
