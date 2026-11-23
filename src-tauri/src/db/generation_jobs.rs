//! Persistence for generation jobs (M8 §8.5). One row per worker pipeline
//! job; the live progress stream is not persisted — only terminal states.

use serde::Serialize;
use serde_json::Value;
use sqlx::SqlitePool;

use crate::error::AppError;

#[derive(Debug, Clone, Serialize)]
pub struct GenerationJob {
    pub id: String,
    pub task: String,
    /// `running` | `done` | `error` | `cancelled`.
    pub status: String,
    pub params: Value,
    pub outputs: Vec<String>,
    pub meta: Value,
    pub error: Option<String>,
    pub created_at: i64,
    pub finished_at: Option<i64>,
}

fn now() -> i64 {
    chrono::Utc::now().timestamp()
}

/// Insert a new job row in `running` state (the worker accepted the submit).
pub async fn insert(
    pool: &SqlitePool,
    id: &str,
    task: &str,
    params: &Value,
) -> Result<(), AppError> {
    let params_json = serde_json::to_string(params)?;
    let created_at = now();
    sqlx::query!(
        r#"INSERT INTO generation_jobs (id, task, status, params, created_at)
           VALUES (?, ?, 'running', ?, ?)"#,
        id,
        task,
        params_json,
        created_at,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Terminal success: record outputs + meta.
pub async fn mark_done(
    pool: &SqlitePool,
    id: &str,
    outputs: &[String],
    meta: &Value,
) -> Result<(), AppError> {
    let outputs_json = serde_json::to_string(outputs)?;
    let meta_json = serde_json::to_string(meta)?;
    let finished_at = now();
    sqlx::query!(
        r#"UPDATE generation_jobs
           SET status = 'done', outputs = ?, meta = ?, finished_at = ?
           WHERE id = ?"#,
        outputs_json,
        meta_json,
        finished_at,
        id,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Terminal failure. The worker reports user cancellation as the literal
/// error string "cancelled" — map it to its own status.
pub async fn mark_error(pool: &SqlitePool, id: &str, error: &str) -> Result<(), AppError> {
    let status = if error == "cancelled" { "cancelled" } else { "error" };
    let finished_at = now();
    sqlx::query!(
        r#"UPDATE generation_jobs
           SET status = ?, error = ?, finished_at = ?
           WHERE id = ?"#,
        status,
        error,
        finished_at,
        id,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Mark every non-terminal row as interrupted — run once at startup, since a
/// `running` row can't have survived the previous process.
pub async fn heal_interrupted(pool: &SqlitePool) -> Result<u64, AppError> {
    let finished_at = now();
    let res = sqlx::query!(
        r#"UPDATE generation_jobs
           SET status = 'error', error = 'interrupted by app shutdown', finished_at = ?
           WHERE status = 'running'"#,
        finished_at,
    )
    .execute(pool)
    .await?;
    Ok(res.rows_affected())
}

pub async fn get(pool: &SqlitePool, id: &str) -> Result<Option<GenerationJob>, AppError> {
    let row = sqlx::query!(
        r#"SELECT id, task, status, params, outputs, meta, error, created_at, finished_at
           FROM generation_jobs WHERE id = ?"#,
        id,
    )
    .fetch_optional(pool)
    .await?;
    Ok(row.map(|r| from_row(r.id, r.task, r.status, r.params, r.outputs, r.meta, r.error, r.created_at, r.finished_at)))
}

/// Newest-first, optionally filtered by task.
pub async fn list(
    pool: &SqlitePool,
    task: Option<&str>,
    limit: i64,
) -> Result<Vec<GenerationJob>, AppError> {
    let rows = match task {
        Some(t) => {
            sqlx::query!(
                r#"SELECT id, task, status, params, outputs, meta, error, created_at, finished_at
                   FROM generation_jobs WHERE task = ?
                   ORDER BY created_at DESC, id DESC LIMIT ?"#,
                t,
                limit,
            )
            .fetch_all(pool)
            .await?
            .into_iter()
            .map(|r| from_row(r.id, r.task, r.status, r.params, r.outputs, r.meta, r.error, r.created_at, r.finished_at))
            .collect()
        }
        None => {
            sqlx::query!(
                r#"SELECT id, task, status, params, outputs, meta, error, created_at, finished_at
                   FROM generation_jobs
                   ORDER BY created_at DESC, id DESC LIMIT ?"#,
                limit,
            )
            .fetch_all(pool)
            .await?
            .into_iter()
            .map(|r| from_row(r.id, r.task, r.status, r.params, r.outputs, r.meta, r.error, r.created_at, r.finished_at))
            .collect()
        }
    };
    Ok(rows)
}

/// `(id, task)` of every running job, with no history cutoff: a long job
/// stays visible however many newer rows exist.
pub async fn list_running(pool: &SqlitePool) -> Result<Vec<(String, String)>, AppError> {
    let rows = sqlx::query!(
        r#"SELECT id AS "id!", task FROM generation_jobs
           WHERE status = 'running'
           ORDER BY created_at, id"#,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|r| (r.id, r.task)).collect())
}

pub async fn delete(pool: &SqlitePool, id: &str) -> Result<(), AppError> {
    sqlx::query!("DELETE FROM generation_jobs WHERE id = ?", id)
        .execute(pool)
        .await?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn from_row(
    id: Option<String>,
    task: String,
    status: String,
    params: String,
    outputs: Option<String>,
    meta: Option<String>,
    error: Option<String>,
    created_at: i64,
    finished_at: Option<i64>,
) -> GenerationJob {
    GenerationJob {
        id: id.unwrap_or_default(),
        task,
        status,
        params: serde_json::from_str(&params).unwrap_or(Value::Null),
        outputs: outputs
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default(),
        meta: meta
            .as_deref()
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or(Value::Null),
        error,
        created_at,
        finished_at,
    }
}
