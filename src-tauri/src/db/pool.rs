use sqlx::{sqlite::SqliteConnectOptions, SqlitePool};
use std::str::FromStr;

use crate::{config, error::AppError};

/// Initialise the SQLite connection pool with:
/// - WAL journal mode (concurrent readers + writer)
/// - foreign_keys = ON
/// - busy_timeout = 5 seconds
/// Then run all embedded migrations.
pub async fn init_pool() -> Result<SqlitePool, AppError> {
    let db_path = config::db_path();

    // Ensure parent directory exists
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let db_url = format!("sqlite://{}?mode=rwc", db_path.display());

    let options = SqliteConnectOptions::from_str(&db_url)?
        .journal_mode(sqlx::sqlite::SqliteJournalMode::Wal)
        .foreign_keys(true)
        .busy_timeout(std::time::Duration::from_secs(5))
        .create_if_missing(true);

    let pool = SqlitePool::connect_with(options).await?;

    // Run migrations from src-tauri/migrations/
    sqlx::migrate!("./migrations").run(&pool).await?;

    tracing::info!("Database initialised at {}", db_path.display());
    Ok(pool)
}
