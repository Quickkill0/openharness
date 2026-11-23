use sqlx::SqlitePool;

use crate::error::AppError;

/// Retrieve a setting value by key. Returns `None` if the key does not exist.
pub async fn get_setting(pool: &SqlitePool, key: &str) -> Result<Option<String>, AppError> {
    let row = sqlx::query!("SELECT value FROM settings WHERE key = ?", key)
        .fetch_optional(pool)
        .await?;
    Ok(row.map(|r| r.value))
}

/// Upsert a setting value.
pub async fn set_setting(pool: &SqlitePool, key: &str, value: &str) -> Result<(), AppError> {
    sqlx::query!(
        "INSERT INTO settings (key, value) VALUES (?, ?)
         ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        key,
        value
    )
    .execute(pool)
    .await?;
    Ok(())
}
