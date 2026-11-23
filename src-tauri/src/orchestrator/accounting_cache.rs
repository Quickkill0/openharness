//! Filenames are stable across split-GGUF reinstalls; the files are not.
use std::{collections::HashMap, path::{Path, PathBuf}, sync::Arc, time::SystemTime};
use crate::{models::gguf, planner::{self, ModelAccounting}};

type Identity = Vec<(u64, SystemTime)>;

fn identity(path: &Path) -> Option<Identity> {
    gguf::split_files(path).ok()?.iter().map(|p| {
        let meta = std::fs::metadata(p).ok()?;
        if !meta.is_file() { return None; }
        Some((meta.len(), meta.modified().ok()?))
    }).collect()
}

#[derive(Default)]
pub(super) struct AccountingCache(HashMap<PathBuf, (Identity, Arc<ModelAccounting>)>);

impl AccountingCache {
    pub async fn get(&mut self, path: &Path) -> Option<Arc<ModelAccounting>> {
        let Some(before) = identity(path) else {
            self.0.remove(path);
            return None;
        };
        if let Some((stamp, accounting)) = self.0.get(path) {
            if *stamp == before { return Some(accounting.clone()); }
        }
        self.0.remove(path);
        let p = path.to_path_buf();
        let result = tokio::task::spawn_blocking(move || planner::accounting::account_model(&p))
            .await.ok()?;
        // Never publish an accounting assembled from files changed during parsing.
        if identity(path).as_ref() != Some(&before) { return None; }
        match result {
            Ok(accounting) => {
                let accounting = Arc::new(accounting);
                self.0.insert(path.into(), (before, accounting.clone()));
                Some(accounting)
            }
            Err(e) => {
                tracing::warn!("planner: can't account {}: {e}", path.display());
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::OpenOptions, io::Write};

    #[tokio::test]
    async fn split_replacement_missing_and_partial_files_never_reuse_old_accounting() {
        let dir = tempfile::tempdir().unwrap();
        let files = crate::models::registry::split_tests::fixture_files(dir.path(), "set/model");
        let primary = Path::new(&files[0].path);
        let mut cache = AccountingCache::default();
        let first = cache.get(primary).await.unwrap();
        assert!(Arc::ptr_eq(&first, &cache.get(primary).await.unwrap()));
        // Same filename, changed non-primary shard: even a size-only change invalidates.
        OpenOptions::new().append(true).open(&files[1].path).unwrap().write_all(&[0; 32]).unwrap();
        let replaced = cache.get(primary).await.unwrap();
        assert!(!Arc::ptr_eq(&first, &replaced));
        // Same-size replacement is caught by the non-primary part's mtime too.
        OpenOptions::new().write(true).open(&files[1].path).unwrap().set_times(
            std::fs::FileTimes::new().set_modified(SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(2))
        ).unwrap();
        assert!(!Arc::ptr_eq(&replaced, &cache.get(primary).await.unwrap()));
        let saved = std::fs::read(&files[2].path).unwrap();
        std::fs::remove_file(&files[2].path).unwrap();
        assert!(cache.get(primary).await.is_none());
        assert!(cache.0.is_empty());
        // A killed/incomplete replacement must not poison the next attempt.
        std::fs::write(&files[2].path, b"GGUF").unwrap();
        assert!(cache.get(primary).await.is_none());
        assert!(cache.0.is_empty());
        std::fs::write(&files[2].path, saved).unwrap();
        assert!(cache.get(primary).await.is_some());
    }
}
