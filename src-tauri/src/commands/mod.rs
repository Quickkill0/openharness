//! Tauri command modules. `lib.rs::run()` registers all commands directly via
//! `tauri::generate_handler!` — no macro indirection needed here.

pub mod assets;
pub mod chat;
pub mod downloads;
pub mod engine;
pub mod generation;
pub mod models;
pub mod orchestrator;
pub mod planner;
pub mod server;
pub mod settings;
pub mod store;
pub mod system;
pub mod tts;
pub mod worker;
