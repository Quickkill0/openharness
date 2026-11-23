use sqlx::SqlitePool;

use crate::error::AppError;

// ---------------------------------------------------------------------------
// Raw row types
// ---------------------------------------------------------------------------

pub struct ModelRow {
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    pub display_name: String,
    pub architecture: Option<String>,
    pub capability: String,
    pub format: String,
    pub task: Option<String>,
    pub local_path: Option<String>,
    pub params_b: Option<f64>,
    pub quantization: Option<String>,
    pub context_train: Option<i64>,
    pub installed_at: i64,
    #[allow(dead_code)]
    pub last_used_at: Option<i64>,
    pub pinned: i64,
}

pub struct ModelFileRow {
    pub path: String,
    /// Path inside the source repo (the original filename for single GGUFs).
    pub rel_path: Option<String>,
    pub sha256: Option<String>,
    pub size: i64,
}

/// A model row joined with its primary file + per-model file aggregates.
pub struct InstalledRow {
    pub model: ModelRow,
    pub primary: ModelFileRow,
    pub file_count: i64,
    pub total_size: i64,
}

/// New-file payload for `insert_model_with_files`.
pub struct NewModelFile {
    /// 'primary' | 'weight' | 'config' | 'other'
    pub role: String,
    /// Path inside the repo (uniqueness key per model).
    pub rel_path: String,
    /// Absolute path on disk.
    pub path: String,
    pub sha256: Option<String>,
    pub size: u64,
}

/// New-model payload for `insert_model_with_files`.
pub struct NewModel {
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    pub display_name: String,
    pub architecture: Option<String>,
    pub capability: String,
    /// 'gguf' | 'safetensors' | 'diffusers-dir'
    pub format: String,
    pub task: Option<String>,
    /// Root directory for dir-layout models; None for single-blob GGUF.
    pub local_path: Option<String>,
    pub params_b: Option<f64>,
    pub quantization: Option<String>,
    pub context_train: Option<u64>,
    pub installed_at: i64,
}

// ---------------------------------------------------------------------------
// Insert
// ---------------------------------------------------------------------------

/// Upsert a model row and replace its file rows, all in one transaction.
pub async fn insert_model_with_files(
    pool: &SqlitePool,
    model: &NewModel,
    files: &[NewModelFile],
) -> Result<(), AppError> {
    insert_model_with_files_once(pool, model, files, None).await
}

/// GGUF group consumption and catalog writes commit (or roll back) together.
pub async fn insert_model_with_files_once(
    pool: &SqlitePool, model: &NewModel, files: &[NewModelFile], group_id: Option<&str>,
) -> Result<(), AppError> {
    let context_train_i: Option<i64> = model.context_train.map(|v| v as i64);
    let mut tx = pool.begin().await?;
    if let Some(gid) = group_id {
        let consumed = sqlx::query!(
            "UPDATE download_groups SET registration_closed = 1 WHERE id = ? AND format = 'gguf' AND registration_closed = 0",
            gid,
        ).execute(&mut *tx).await?;
        if consumed.rows_affected() == 0 { return Ok(()); }
    }

    sqlx::query!(
        r#"INSERT INTO models
           (id, repo_id, revision, display_name, architecture, capability,
            format, task, local_path, params_b, quantization, context_train, installed_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id) DO UPDATE SET
             installed_at  = excluded.installed_at,
             architecture  = excluded.architecture,
             capability    = excluded.capability,
             format        = excluded.format,
             task          = excluded.task,
             local_path    = excluded.local_path,
             params_b      = excluded.params_b,
             quantization  = excluded.quantization,
             context_train = excluded.context_train"#,
        model.id,
        model.repo_id,
        model.revision,
        model.display_name,
        model.architecture,
        model.capability,
        model.format,
        model.task,
        model.local_path,
        model.params_b,
        model.quantization,
        context_train_i,
        model.installed_at,
    )
    .execute(&mut *tx)
    .await?;

    // Re-registration replaces the file set wholesale — simplest correct
    // semantics for both single-blob upserts and repo re-downloads.
    sqlx::query!("DELETE FROM model_files WHERE model_id = ?", model.id)
        .execute(&mut *tx)
        .await?;

    for f in files {
        let size_i = f.size as i64;
        sqlx::query!(
            r#"INSERT INTO model_files (model_id, role, rel_path, path, sha256, size)
               VALUES (?, ?, ?, ?, ?, ?)"#,
            model.id,
            f.role,
            f.rel_path,
            f.path,
            f.sha256,
            size_i,
        )
        .execute(&mut *tx)
        .await?;
    }

    tx.commit().await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Query
// ---------------------------------------------------------------------------

/// List all installed models joined with their primary file row + aggregates.
pub async fn list_installed(pool: &SqlitePool) -> Result<Vec<InstalledRow>, AppError> {
    let rows = sqlx::query!(
        r#"SELECT m.id, m.repo_id, m.revision, m.display_name, m.architecture,
                  m.capability, m.format, m.task, m.local_path,
                  m.params_b, m.quantization, m.context_train,
                  m.installed_at, m.last_used_at, m.pinned,
                  mf.path AS "file_path?",
                  mf.rel_path AS "file_rel_path?",
                  mf.sha256 AS "file_sha256?",
                  mf.size AS "file_size?",
                  (SELECT COUNT(*) FROM model_files WHERE model_id = m.id) AS "file_count!: i64",
                  (SELECT COALESCE(SUM(size), 0) FROM model_files WHERE model_id = m.id) AS "total_size!: i64"
           FROM models m
           LEFT JOIN model_files mf ON mf.model_id = m.id AND mf.role = 'primary'
           ORDER BY m.installed_at DESC"#
    )
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|r| InstalledRow {
            model: ModelRow {
                id: r.id.unwrap_or_default(),
                repo_id: r.repo_id,
                revision: r.revision,
                display_name: r.display_name,
                architecture: r.architecture,
                capability: r.capability,
                format: r.format,
                task: r.task,
                local_path: r.local_path,
                params_b: r.params_b,
                quantization: r.quantization,
                context_train: r.context_train,
                installed_at: r.installed_at,
                last_used_at: r.last_used_at,
                pinned: r.pinned,
            },
            primary: ModelFileRow {
                path: r.file_path.unwrap_or_default(),
                rel_path: r.file_rel_path,
                sha256: r.file_sha256,
                size: r.file_size.unwrap_or(0),
            },
            file_count: r.file_count,
            total_size: r.total_size,
        })
        .collect())
}

/// Fetch a single model and its primary file row + aggregates by model id.
pub async fn get_by_id(pool: &SqlitePool, id: &str) -> Result<Option<InstalledRow>, AppError> {
    let row = sqlx::query!(
        r#"SELECT m.id, m.repo_id, m.revision, m.display_name, m.architecture,
                  m.capability, m.format, m.task, m.local_path,
                  m.params_b, m.quantization, m.context_train,
                  m.installed_at, m.last_used_at, m.pinned,
                  mf.path AS "file_path?",
                  mf.rel_path AS "file_rel_path?",
                  mf.sha256 AS "file_sha256?",
                  mf.size AS "file_size?",
                  (SELECT COUNT(*) FROM model_files WHERE model_id = m.id) AS "file_count!: i64",
                  (SELECT COALESCE(SUM(size), 0) FROM model_files WHERE model_id = m.id) AS "total_size!: i64"
           FROM models m
           LEFT JOIN model_files mf ON mf.model_id = m.id AND mf.role = 'primary'
           WHERE m.id = ?
           LIMIT 1"#,
        id
    )
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|r| InstalledRow {
        model: ModelRow {
            id: r.id.unwrap_or_default(),
            repo_id: r.repo_id,
            revision: r.revision,
            display_name: r.display_name,
            architecture: r.architecture,
            capability: r.capability,
            format: r.format,
            task: r.task,
            local_path: r.local_path,
            params_b: r.params_b,
            quantization: r.quantization,
            context_train: r.context_train,
            installed_at: r.installed_at,
            last_used_at: r.last_used_at,
            pinned: r.pinned,
        },
        primary: ModelFileRow {
            path: r.file_path.unwrap_or_default(),
            rel_path: r.file_rel_path,
            sha256: r.file_sha256,
            size: r.file_size.unwrap_or(0),
        },
        file_count: r.file_count,
        total_size: r.total_size,
    }))
}

// ---------------------------------------------------------------------------
// Metadata backfill (GGUF rows that predate parser fixes)
// ---------------------------------------------------------------------------

/// GGUF models whose header metadata never landed (architecture IS NULL),
/// with their primary blob path — candidates for a re-parse.
pub async fn list_gguf_missing_meta(
    pool: &SqlitePool,
) -> Result<Vec<(String, String)>, AppError> {
    let rows = sqlx::query!(
        r#"SELECT m.id AS "id!", mf.path AS "path!"
           FROM models m
           JOIN model_files mf ON mf.model_id = m.id AND mf.role = 'primary'
           WHERE m.format = 'gguf' AND m.architecture IS NULL"#
    )
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|r| (r.id, r.path)).collect())
}

/// Write re-parsed GGUF header metadata onto an existing catalog row.
/// Quantization is COALESCEd so a filename-derived value isn't clobbered
/// by a NULL re-parse.
pub async fn update_gguf_meta(
    pool: &SqlitePool,
    id: &str,
    architecture: Option<&str>,
    params_b: Option<f64>,
    quantization: Option<&str>,
    context_train: Option<i64>,
) -> Result<(), AppError> {
    sqlx::query!(
        r#"UPDATE models
           SET architecture = ?,
               params_b = ?,
               quantization = COALESCE(?, quantization),
               context_train = ?
           WHERE id = ?"#,
        architecture,
        params_b,
        quantization,
        context_train,
        id
    )
    .execute(pool)
    .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

/// Absolute catalog-owned files, including pre-split single/directory installs.
/// Compare in Rust: SQLite NOCASE does not fold non-ASCII Windows filenames.
pub async fn owned_paths(pool: &SqlitePool) -> Result<Vec<String>, AppError> {
    Ok(sqlx::query!(
        "SELECT mf.path FROM model_files mf JOIN models m ON m.id = mf.model_id"
    ).fetch_all(pool).await?.into_iter().map(|row| row.path).collect())
}

/// Delete a model and its associated file rows (cascaded by FK).
pub async fn delete_by_id(pool: &SqlitePool, id: &str) -> Result<(), AppError> {
    sqlx::query!("DELETE FROM models WHERE id = ?", id)
        .execute(pool)
        .await?;
    Ok(())
}
