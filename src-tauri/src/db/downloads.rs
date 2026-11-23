use sqlx::SqlitePool;

use crate::{
    downloader::job::{Job, JobStatus},
    error::AppError,
};

/// Insert or replace a download job row.
pub async fn upsert_job(pool: &SqlitePool, job: &Job) -> Result<(), AppError> {
    let id = job.id.to_string();
    let status = job.status.to_string();
    let chunks_json = serde_json::to_string(&job.chunks)?;
    let dest_path = job.dest_path.to_string_lossy().to_string();
    let total = job.total_bytes as i64;
    let downloaded = job.downloaded as i64;
    let group_id = job.group_id.map(|g| g.to_string());

    sqlx::query!(
        r#"INSERT INTO downloads
           (id, repo_id, revision, file, dest_path, total_bytes, downloaded,
            status, error, chunks_json, expected_sha256, group_id, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             downloaded      = excluded.downloaded,
             status          = excluded.status,
             error           = excluded.error,
             chunks_json     = excluded.chunks_json,
             expected_sha256 = excluded.expected_sha256,
             dest_path       = excluded.dest_path,
             group_id        = excluded.group_id,
             updated_at      = excluded.updated_at"#,
        id,
        job.repo_id,
        job.revision,
        job.file,
        dest_path,
        total,
        downloaded,
        status,
        job.error,
        chunks_json,
        job.expected_sha256,
        group_id,
        job.created_at,
        job.updated_at,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Update only the chunks_json and downloaded counter for an in-flight job.
pub async fn update_chunks(
    pool: &SqlitePool,
    id: uuid::Uuid,
    chunks_json: &str,
    downloaded: u64,
    updated_at: i64,
) -> Result<(), AppError> {
    let id_str = id.to_string();
    let dl = downloaded as i64;
    sqlx::query!(
        "UPDATE downloads SET chunks_json = ?, downloaded = ?, updated_at = ? WHERE id = ?",
        chunks_json,
        dl,
        updated_at,
        id_str,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Update the status (and optionally error message) of a job.
pub async fn update_status(
    pool: &SqlitePool,
    id: uuid::Uuid,
    status: JobStatus,
    error: Option<&str>,
    updated_at: i64,
) -> Result<(), AppError> {
    let id_str = id.to_string();
    let status_str = status.to_string();
    sqlx::query!(
        "UPDATE downloads SET status = ?, error = ?, updated_at = ? WHERE id = ?",
        status_str,
        error,
        updated_at,
        id_str,
    )
    .execute(pool)
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Raw DB row — converted to Job in the downloader module
// ---------------------------------------------------------------------------

pub struct JobRow {
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    pub file: String,
    pub dest_path: String,
    pub total_bytes: i64,
    pub downloaded: i64,
    pub status: String,
    pub error: Option<String>,
    pub chunks_json: String,
    pub expected_sha256: Option<String>,
    pub group_id: Option<String>,
    pub created_at: i64,
    pub updated_at: i64,
}

/// Load all download rows from the DB.
pub async fn list_all(pool: &SqlitePool) -> Result<Vec<JobRow>, AppError> {
    let rows = sqlx::query!(
        r#"SELECT id, repo_id, revision, file, dest_path, total_bytes, downloaded,
                  status, error, chunks_json, expected_sha256, group_id, created_at, updated_at
           FROM downloads ORDER BY created_at ASC"#
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| JobRow {
            id: r.id.unwrap_or_default(),
            repo_id: r.repo_id,
            revision: r.revision,
            file: r.file,
            dest_path: r.dest_path,
            total_bytes: r.total_bytes,
            downloaded: r.downloaded,
            status: r.status,
            error: r.error,
            chunks_json: r.chunks_json,
            expected_sha256: r.expected_sha256,
            group_id: r.group_id,
            created_at: r.created_at,
            updated_at: r.updated_at,
        })
        .collect())
}

/// Load a single download row by id.
pub async fn get_one(pool: &SqlitePool, id: uuid::Uuid) -> Result<Option<JobRow>, AppError> {
    let id_str = id.to_string();
    let row = sqlx::query!(
        r#"SELECT id, repo_id, revision, file, dest_path, total_bytes, downloaded,
                  status, error, chunks_json, expected_sha256, group_id, created_at, updated_at
           FROM downloads WHERE id = ?"#,
        id_str,
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| JobRow {
        id: r.id.unwrap_or_default(),
        repo_id: r.repo_id,
        revision: r.revision,
        file: r.file,
        dest_path: r.dest_path,
        total_bytes: r.total_bytes,
        downloaded: r.downloaded,
        status: r.status,
        error: r.error,
        chunks_json: r.chunks_json,
        expected_sha256: r.expected_sha256,
        group_id: r.group_id,
        created_at: r.created_at,
        updated_at: r.updated_at,
    }))
}

/// Delete a single download row.
pub async fn delete_one(pool: &SqlitePool, id: uuid::Uuid) -> Result<(), AppError> {
    let id_str = id.to_string();
    sqlx::query!("DELETE FROM downloads WHERE id = ?", id_str)
        .execute(pool)
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Download groups (M9 §9.2)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, serde::Serialize)]
pub struct GroupRow {
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    pub format: String,
    pub task: Option<String>,
    pub created_at: i64,
    #[serde(skip)]
    pub registration_closed: bool,
}

/// Insert a new download group row.
pub async fn insert_group(pool: &SqlitePool, group: &GroupRow) -> Result<(), AppError> {
    sqlx::query!(
        r#"INSERT INTO download_groups (id, repo_id, revision, format, task, created_at)
           VALUES (?, ?, ?, ?, ?, ?)"#,
        group.id,
        group.repo_id,
        group.revision,
        group.format,
        group.task,
        group.created_at,
    )
    .execute(pool)
    .await?;
    Ok(())
}

/// Look up a single download group by id.
pub async fn get_group(pool: &SqlitePool, id: &str) -> Result<Option<GroupRow>, AppError> {
    let row = sqlx::query!(
        r#"SELECT id, repo_id, revision, format, task, created_at, registration_closed AS "registration_closed!: bool"
           FROM download_groups WHERE id = ?"#,
        id,
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| GroupRow {
        id: r.id.unwrap_or_default(),
        repo_id: r.repo_id,
        revision: r.revision,
        format: r.format,
        task: r.task,
        created_at: r.created_at,
        registration_closed: r.registration_closed,
    }))
}

/// Load all download groups.
pub async fn list_groups(pool: &SqlitePool) -> Result<Vec<GroupRow>, AppError> {
    let rows = sqlx::query!(
        r#"SELECT id, repo_id, revision, format, task, created_at, registration_closed AS "registration_closed!: bool"
           FROM download_groups ORDER BY created_at ASC"#
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| GroupRow {
            id: r.id.unwrap_or_default(),
            repo_id: r.repo_id,
            revision: r.revision,
            format: r.format,
            task: r.task,
            created_at: r.created_at,
            registration_closed: r.registration_closed,
        })
        .collect())
}

/// Permanently end predecessor registration rights before starting any transfer.
pub async fn supersede_gguf_groups(pool: &SqlitePool, destinations: &[String]) -> Result<(), AppError> {
    let mut tx = pool.begin().await?;
    let candidates = sqlx::query!(
        "SELECT g.id, d.repo_id, d.file FROM download_groups g
         JOIN downloads d ON d.group_id = g.id WHERE g.format = 'gguf'"
    ).fetch_all(&mut *tx).await?;
    let ids: std::collections::HashSet<_> = candidates.into_iter().filter(|row| {
        destinations.contains(&crate::models::split_gguf::destination_key(
            crate::config::repo_root_dir(&row.repo_id).join(&row.file)))
    }).map(|row| row.id).collect();
    for id in ids {
        sqlx::query!("UPDATE download_groups SET registration_closed = 1 WHERE id = ?", id)
            .execute(&mut *tx).await?;
    }
    tx.commit().await?;
    Ok(())
}

/// Serialize final renames with closure, including a cancelled worker finishing
/// verification. The no-op UPDATE takes SQLite's write lock, never reopens a marker.
pub async fn gguf_write_fence<'a>(
    pool: &'a SqlitePool, gid: &str,
) -> Result<sqlx::Transaction<'a, sqlx::Sqlite>, AppError> {
    let mut tx = pool.begin().await?;
    let result = sqlx::query!(
        "UPDATE download_groups SET registration_closed = registration_closed
         WHERE id = ? AND format = 'gguf' AND registration_closed = 0", gid
    ).execute(&mut *tx).await?;
    if result.rows_affected() == 0 {
        return Err(AppError::Other("This download was superseded. Start a new download from the Store.".into()));
    }
    Ok(tx)
}

/// Delete a group row and all of its member job rows.
pub async fn delete_group(pool: &SqlitePool, id: uuid::Uuid) -> Result<(), AppError> {
    let id_str = id.to_string();
    sqlx::query!("DELETE FROM downloads WHERE group_id = ?", id_str)
        .execute(pool)
        .await?;
    sqlx::query!("DELETE FROM download_groups WHERE id = ?", id_str)
        .execute(pool)
        .await?;
    Ok(())
}
