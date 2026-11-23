use sqlx::SqlitePool;

use crate::error::AppError;

#[derive(Debug, Clone)]
pub struct EngineRow {
    pub id: String,
    pub version: String,
    pub variant: String,
    pub path: String,
    pub sha256: String,
    pub active: bool,
    pub installed_at: i64,
}

pub async fn upsert(pool: &SqlitePool, row: &EngineRow) -> Result<(), AppError> {
    let active_i = if row.active { 1_i64 } else { 0_i64 };
    sqlx::query!(
        "INSERT INTO engines (id, version, variant, path, sha256, active, installed_at)
         VALUES (?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(id) DO UPDATE SET
           version      = excluded.version,
           variant      = excluded.variant,
           path         = excluded.path,
           sha256       = excluded.sha256,
           active       = excluded.active,
           installed_at = excluded.installed_at",
        row.id,
        row.version,
        row.variant,
        row.path,
        row.sha256,
        active_i,
        row.installed_at
    )
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn set_active(pool: &SqlitePool, id: &str) -> Result<(), AppError> {
    let mut tx = pool.begin().await?;
    sqlx::query!("UPDATE engines SET active = 0 WHERE active = 1")
        .execute(&mut *tx)
        .await?;
    sqlx::query!("UPDATE engines SET active = 1 WHERE id = ?", id)
        .execute(&mut *tx)
        .await?;
    tx.commit().await?;
    Ok(())
}

pub async fn get_active(pool: &SqlitePool) -> Result<Option<EngineRow>, AppError> {
    let rec = sqlx::query!(
        "SELECT id, version, variant, path, sha256, active, installed_at
         FROM engines WHERE active = 1 LIMIT 1"
    )
    .fetch_optional(pool)
    .await?;
    Ok(rec.map(|r| EngineRow {
        id: r.id.unwrap_or_default(),
        version: r.version,
        variant: r.variant,
        path: r.path,
        sha256: r.sha256,
        active: r.active != 0,
        installed_at: r.installed_at,
    }))
}

/// The active engine, but only when it is the pinned build
/// ([`crate::engine::ENGINE_BUILD`]). An install of an older build reads as
/// "not installed", so the UI's install wizard re-provisions it: the new
/// build installs into its own `<build>-<variant>` folder and the stale row is
/// pruned once the new one is active. Old builds can't take the planner's
/// arguments (`-fa on`, `--n-cpu-moe`, `--moe-cache-mib`, `--fit off`).
pub async fn get_current(pool: &SqlitePool) -> Result<Option<EngineRow>, AppError> {
    Ok(get_active(pool)
        .await?
        .filter(|r| r.version == crate::engine::ENGINE_BUILD))
}

/// Delete all inactive engine rows. Reinstalls and variant switches leave the
/// previous row behind with `active = 0`; only the active row is ever read
/// back, so the orphans are pure cruft. Returns the number of rows removed.
pub async fn prune_inactive(pool: &SqlitePool) -> Result<u64, AppError> {
    let res = sqlx::query!("DELETE FROM engines WHERE active = 0")
        .execute(pool)
        .await?;
    Ok(res.rows_affected())
}

pub async fn list(pool: &SqlitePool) -> Result<Vec<EngineRow>, AppError> {
    let rows = sqlx::query!(
        "SELECT id, version, variant, path, sha256, active, installed_at
         FROM engines ORDER BY installed_at DESC"
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|r| EngineRow {
            id: r.id.unwrap_or_default(),
            version: r.version,
            variant: r.variant,
            path: r.path,
            sha256: r.sha256,
            active: r.active != 0,
            installed_at: r.installed_at,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqlx::sqlite::SqlitePoolOptions;

    async fn test_pool() -> SqlitePool {
        let pool = SqlitePoolOptions::new()
            .max_connections(1)
            .connect("sqlite::memory:")
            .await
            .unwrap();
        sqlx::migrate!("./migrations").run(&pool).await.unwrap();
        pool
    }

    fn row(id: &str, active: bool) -> EngineRow {
        EngineRow {
            id: id.into(),
            version: "b9222".into(),
            variant: "cuda-12.4".into(),
            path: format!("C:\\engine\\{id}"),
            sha256: String::new(),
            active,
            installed_at: 1,
        }
    }

    #[tokio::test]
    async fn prune_removes_only_inactive_rows() {
        let pool = test_pool().await;
        upsert(&pool, &row("old-a", false)).await.unwrap();
        upsert(&pool, &row("old-b", false)).await.unwrap();
        upsert(&pool, &row("current", false)).await.unwrap();
        set_active(&pool, "current").await.unwrap();

        let removed = prune_inactive(&pool).await.unwrap();
        assert_eq!(removed, 2);

        let remaining = list(&pool).await.unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].id, "current");
        assert!(remaining[0].active);
    }

    #[tokio::test]
    async fn prune_is_noop_with_only_active_row() {
        let pool = test_pool().await;
        upsert(&pool, &row("current", false)).await.unwrap();
        set_active(&pool, "current").await.unwrap();
        assert_eq!(prune_inactive(&pool).await.unwrap(), 0);
        assert_eq!(list(&pool).await.unwrap().len(), 1);
    }
}
