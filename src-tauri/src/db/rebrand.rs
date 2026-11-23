use sqlx::SqlitePool;

use crate::error::AppError;

/// One-time fix-up after the Hangar→OpenHarness data-dir move: rows written
/// before the rename hold absolute paths inside the legacy directory, which
/// no longer exists after `migrate_legacy_data_dir` renames it. Rewrites the
/// prefix on every stored path column (`engines.path`, `model_files.path`,
/// `downloads.dest_path`). Prefix-exact and idempotent — rows already under
/// the new prefix never match. Returns the number of rows rewritten.
pub async fn rewrite_legacy_path_prefixes(
    pool: &SqlitePool,
    old_prefix: &str,
    new_prefix: &str,
) -> Result<u64, AppError> {
    let mut rewritten = 0_u64;

    rewritten += sqlx::query!(
        "UPDATE engines
            SET path = ?2 || substr(path, length(?1) + 1)
          WHERE substr(path, 1, length(?1)) = ?1",
        old_prefix,
        new_prefix
    )
    .execute(pool)
    .await?
    .rows_affected();

    rewritten += sqlx::query!(
        "UPDATE model_files
            SET path = ?2 || substr(path, length(?1) + 1)
          WHERE substr(path, 1, length(?1)) = ?1",
        old_prefix,
        new_prefix
    )
    .execute(pool)
    .await?
    .rows_affected();

    rewritten += sqlx::query!(
        "UPDATE downloads
            SET dest_path = ?2 || substr(dest_path, length(?1) + 1)
          WHERE substr(dest_path, 1, length(?1)) = ?1",
        old_prefix,
        new_prefix
    )
    .execute(pool)
    .await?
    .rows_affected();

    Ok(rewritten)
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

    #[tokio::test]
    async fn rewrites_only_legacy_prefixed_paths() {
        let pool = test_pool().await;

        let legacy = r"C:\Users\u\AppData\Local\Hangar\engine\b9222-cpu\llama-server.exe";
        let already_new = r"C:\Users\u\AppData\Local\OpenHarness\engine\b9222-cuda\llama-server.exe";
        for (id, path, active) in [("e1", legacy, 1_i64), ("e2", already_new, 0)] {
            sqlx::query!(
                "INSERT INTO engines (id, version, variant, path, sha256, active, installed_at)
                 VALUES (?, 'b9222', 'cpu', ?, '', ?, 0)",
                id,
                path,
                active
            )
            .execute(&pool)
            .await
            .unwrap();
        }

        let n = rewrite_legacy_path_prefixes(
            &pool,
            r"C:\Users\u\AppData\Local\Hangar\",
            r"C:\Users\u\AppData\Local\OpenHarness\",
        )
        .await
        .unwrap();
        assert_eq!(n, 1);

        let active = crate::db::engines::get_active(&pool).await.unwrap().unwrap();
        assert_eq!(
            active.path,
            r"C:\Users\u\AppData\Local\OpenHarness\engine\b9222-cpu\llama-server.exe"
        );

        // Untouched row and idempotence: a second pass rewrites nothing.
        let again = rewrite_legacy_path_prefixes(
            &pool,
            r"C:\Users\u\AppData\Local\Hangar\",
            r"C:\Users\u\AppData\Local\OpenHarness\",
        )
        .await
        .unwrap();
        assert_eq!(again, 0);
    }
}
