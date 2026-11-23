//! Persistence for the shared asset library (M10). One row per generated
//! artifact; `assets_fts` is kept in sync app-level (insert/delete touch
//! both tables), mirroring the messages_fts strategy.

use serde::Serialize;
use serde_json::Value;
use sqlx::SqlitePool;

use crate::error::AppError;

#[derive(Debug, Clone, Serialize)]
pub struct Asset {
    pub id: String,
    pub job_id: Option<String>,
    /// `image` | `video` | `audio` | `text`.
    pub kind: String,
    pub path: String,
    pub task: String,
    pub prompt: Option<String>,
    pub model: Option<String>,
    pub params: Value,
    pub meta: Value,
    pub size: i64,
    pub created_at: i64,
}

/// A new asset row ready for insertion (id is generated here).
pub struct NewAsset<'a> {
    pub job_id: &'a str,
    pub kind: &'a str,
    pub path: &'a str,
    pub task: &'a str,
    pub prompt: Option<&'a str>,
    pub model: Option<&'a str>,
    pub params: &'a Value,
    pub meta: &'a Value,
    pub size: i64,
}

/// Insert an asset + its FTS row. Returns the generated id.
pub async fn insert(pool: &SqlitePool, asset: NewAsset<'_>) -> Result<String, AppError> {
    let id = uuid::Uuid::new_v4().to_string();
    let params_json = serde_json::to_string(asset.params)?;
    let meta_json = serde_json::to_string(asset.meta)?;
    let created_at = chrono::Utc::now().timestamp();

    let mut tx = pool.begin().await?;
    sqlx::query!(
        r#"INSERT INTO assets (id, job_id, kind, path, task, prompt, model, params, meta, size, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"#,
        id,
        asset.job_id,
        asset.kind,
        asset.path,
        asset.task,
        asset.prompt,
        asset.model,
        params_json,
        meta_json,
        asset.size,
        created_at,
    )
    .execute(&mut *tx)
    .await?;
    sqlx::query!(
        r#"INSERT INTO assets_fts (asset_id, prompt, model, task) VALUES (?, ?, ?, ?)"#,
        id,
        asset.prompt,
        asset.model,
        asset.task,
    )
    .execute(&mut *tx)
    .await?;
    tx.commit().await?;
    Ok(id)
}

pub async fn get(pool: &SqlitePool, id: &str) -> Result<Option<Asset>, AppError> {
    let row = sqlx::query!(
        r#"SELECT id, job_id, kind, path, task, prompt, model, params, meta, size, created_at
           FROM assets WHERE id = ?"#,
        id,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| {
        from_row(
            r.id, r.job_id, r.kind, r.path, r.task, r.prompt, r.model, r.params, r.meta, r.size,
            r.created_at,
        )
    }))
}

/// Newest-first, optionally filtered by kind.
pub async fn list(
    pool: &SqlitePool,
    kind: Option<&str>,
    limit: i64,
    offset: i64,
) -> Result<Vec<Asset>, AppError> {
    let rows = match kind {
        Some(k) => sqlx::query!(
            r#"SELECT id, job_id, kind, path, task, prompt, model, params, meta, size, created_at
               FROM assets WHERE kind = ?
               ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"#,
            k,
            limit,
            offset,
        )
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(|r| {
            from_row(
                r.id, r.job_id, r.kind, r.path, r.task, r.prompt, r.model, r.params, r.meta,
                r.size, r.created_at,
            )
        })
        .collect(),
        None => sqlx::query!(
            r#"SELECT id, job_id, kind, path, task, prompt, model, params, meta, size, created_at
               FROM assets
               ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?"#,
            limit,
            offset,
        )
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(|r| {
            from_row(
                r.id, r.job_id, r.kind, r.path, r.task, r.prompt, r.model, r.params, r.meta,
                r.size, r.created_at,
            )
        })
        .collect(),
    };
    Ok(rows)
}

/// FTS search over prompt/model/task. The raw query is wrapped per-token with
/// a trailing `*` so partial words match ("lighth" finds "lighthouse").
pub async fn search(pool: &SqlitePool, query: &str, limit: i64) -> Result<Vec<Asset>, AppError> {
    let fts_query = build_fts_query(query);
    if fts_query.is_empty() {
        return list(pool, None, limit, 0).await;
    }
    let rows = sqlx::query!(
        r#"SELECT a.id, a.job_id, a.kind, a.path, a.task, a.prompt, a.model, a.params, a.meta,
                  a.size, a.created_at
           FROM assets_fts f
           JOIN assets a ON a.id = f.asset_id
           WHERE assets_fts MATCH ?
           ORDER BY rank, a.created_at DESC LIMIT ?"#,
        fts_query,
        limit,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| {
            from_row(
                r.id, r.job_id, r.kind, r.path, r.task, r.prompt, r.model, r.params, r.meta,
                r.size, r.created_at,
            )
        })
        .collect())
}

/// Delete the row + FTS entry. Returns the asset (for file cleanup by the caller).
pub async fn delete(pool: &SqlitePool, id: &str) -> Result<Option<Asset>, AppError> {
    let Some(asset) = get(pool, id).await? else {
        return Ok(None);
    };
    let mut tx = pool.begin().await?;
    sqlx::query!("DELETE FROM assets WHERE id = ?", id).execute(&mut *tx).await?;
    sqlx::query!("DELETE FROM assets_fts WHERE asset_id = ?", id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(Some(asset))
}

/// Sanitize free text into an FTS5 prefix query: bare alphanumeric tokens,
/// each suffixed with `*`, AND-joined. Strips FTS operators so user input
/// can never produce a syntax error.
pub fn build_fts_query(raw: &str) -> String {
    raw.split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .map(|t| format!("{t}*"))
        .collect::<Vec<_>>()
        .join(" ")
}

#[allow(clippy::too_many_arguments)]
fn from_row(
    id: Option<String>,
    job_id: Option<String>,
    kind: String,
    path: String,
    task: String,
    prompt: Option<String>,
    model: Option<String>,
    params: Option<String>,
    meta: Option<String>,
    size: i64,
    created_at: i64,
) -> Asset {
    Asset {
        id: id.unwrap_or_default(),
        job_id,
        kind,
        path,
        task,
        prompt,
        model,
        params: params
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null),
        meta: meta
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null),
        size,
        created_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_sanitizes_operators() {
        assert_eq!(build_fts_query("lighthouse dawn"), "lighthouse* dawn*");
        assert_eq!(build_fts_query("  \"AND(OR)*  "), "AND* OR*");
        assert_eq!(build_fts_query("a-b.c"), "a* b* c*");
        assert_eq!(build_fts_query("   "), "");
    }
}
