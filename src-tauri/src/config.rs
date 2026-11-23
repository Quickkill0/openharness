use std::path::PathBuf;

/// Base app data directory: %LOCALAPPDATA%\OpenHarness\
pub fn app_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .expect("cannot resolve %LOCALAPPDATA%")
        .join("OpenHarness")
}

/// Legacy app data directory from the "Hangar AI" era: %LOCALAPPDATA%\Hangar\
/// Used only by the one-time rebrand migration.
pub fn legacy_app_data_dir() -> PathBuf {
    dirs::data_local_dir()
        .expect("cannot resolve %LOCALAPPDATA%")
        .join("Hangar")
}

/// SQLite database path: %LOCALAPPDATA%\OpenHarness\data.db
pub fn db_path() -> PathBuf {
    app_data_dir().join("data.db")
}

/// Directory where GGUF model files are stored.
pub fn models_dir() -> PathBuf {
    app_data_dir().join("models")
}

/// Directory where multi-file repo-layout models live, one subdirectory per
/// repo: models\repos\<owner>__<name>\<rel_path> (M9 §9.2).
pub fn model_repos_dir() -> PathBuf {
    models_dir().join("repos")
}

/// Root directory for a specific repo-layout model. The single `/` in an HF
/// repo id ("owner/name") becomes `__`; HF restricts both halves to
/// `[A-Za-z0-9._-]`, so no further sanitization is needed on Windows.
pub fn repo_root_dir(repo_id: &str) -> PathBuf {
    model_repos_dir().join(repo_id.replace('/', "__"))
}

/// Directory where conversation transcript files are stored.
pub fn transcripts_dir() -> PathBuf {
    app_data_dir().join("transcripts")
}

/// Directory where llama.cpp engine binaries are installed.
pub fn engine_dir() -> PathBuf {
    app_data_dir().join("engine")
}

/// Python worker home: pinned uv, managed CPython, venv, stamp (M8).
pub fn worker_dir() -> PathBuf {
    app_data_dir().join("worker")
}

/// Generated-content library (images, audio, video) — the M10 asset library.
pub fn assets_dir() -> PathBuf {
    app_data_dir().join("assets")
}

/// Where worker pipelines write outputs before jobs register them in the DB.
pub fn assets_staging_dir() -> PathBuf {
    assets_dir().join("staging")
}

/// Ensure all required directories exist, creating them if necessary.
pub fn ensure_dirs() -> std::io::Result<()> {
    std::fs::create_dir_all(app_data_dir())?;
    std::fs::create_dir_all(models_dir())?;
    std::fs::create_dir_all(model_repos_dir())?;
    std::fs::create_dir_all(transcripts_dir())?;
    std::fs::create_dir_all(engine_dir())?;
    std::fs::create_dir_all(worker_dir())?;
    std::fs::create_dir_all(assets_staging_dir())?;
    Ok(())
}

/// One-time rebrand migration (Hangar AI → OpenHarness).
///
/// If the new OpenHarness data dir does **not** exist but the legacy Hangar dir
/// does, move the entire directory (db, models, transcripts, engine,
/// settings.json) across in a single rename. Idempotent: once the new dir
/// exists this is a no-op, so an existing OpenHarness install is never clobbered.
///
/// Must run **before** [`ensure_dirs`] — otherwise the empty new dir we create
/// would suppress the migration. Returns `Ok(true)` when a migration occurred.
pub fn migrate_legacy_data_dir() -> std::io::Result<bool> {
    let new_dir = app_data_dir();
    let old_dir = legacy_app_data_dir();

    if new_dir.exists() || !old_dir.exists() {
        return Ok(false);
    }

    // Same volume (%LOCALAPPDATA%) so a plain rename is atomic and cheap.
    std::fs::rename(&old_dir, &new_dir)?;
    tracing::info!(
        "Rebrand migration: moved {} → {}",
        old_dir.display(),
        new_dir.display()
    );
    Ok(true)
}
