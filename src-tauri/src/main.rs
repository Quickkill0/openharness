// Prevents an additional console window from appearing in release mode on Windows.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tracing_subscriber::{fmt, prelude::*, EnvFilter};

fn main() {
    // Initialise structured logging. Set RUST_LOG=debug to see SQL queries.
    let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));

    // Persistent on-disk log so runtime errors survive across restarts and can
    // be tailed live — the `tauri dev` stdout is easily orphaned, which meant
    // app errors were being lost. Daily-rolled under the app data dir; the
    // stdout layer is kept for when the dev server is in the foreground.
    let log_dir = std::env::var_os("LOCALAPPDATA")
        .map(|b| std::path::PathBuf::from(b).join("OpenHarness").join("logs"))
        .unwrap_or_else(|| std::path::PathBuf::from("logs"));
    let _ = std::fs::create_dir_all(&log_dir);
    let (file_writer, guard) = tracing_appender::non_blocking(
        tracing_appender::rolling::daily(&log_dir, "openharness.log"),
    );

    tracing_subscriber::registry()
        .with(filter)
        .with(fmt::layer())
        .with(fmt::layer().with_ansi(false).with_writer(file_writer))
        .init();

    tracing::info!("OpenHarness starting…");

    // Hold the non-blocking appender guard for the process lifetime so buffered
    // log lines flush (run() blocks until the app exits).
    let _guard = guard;
    openharness_lib::run();
}
