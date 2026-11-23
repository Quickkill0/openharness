#[cfg(test)]
#[path = "split_tests.rs"]
pub(crate) mod split_tests;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use sqlx::SqlitePool;
use std::path::{Path, PathBuf};

use crate::{
    config,
    db::models as db_models,
    db::models::{InstalledRow, NewModel, NewModelFile},
    downloader::job::Job,
    error::AppError,
    models::{
        capabilities,
        capability::{self, Capability},
        gguf::GgufMeta,
        hf::HfModelCard,
    },
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct InstalledModel {
    /// Stable identifier: SHA-256 of the primary file for single-blob GGUFs,
    /// SHA-256 of "repo_id@revision" for dir layouts, plus the set stem for split GGUFs.
    pub id: String,
    pub repo_id: String,
    pub revision: String,
    pub display_name: String,
    pub architecture: Option<String>,
    pub capability: Capability,
    /// 'gguf' | 'safetensors' | 'diffusers-dir'
    pub format: String,
    /// HF pipeline_tag-style task string ('text-generation', 'text-to-image', …).
    pub task: Option<String>,
    /// Owned root for dir-layout models; None for file-backed GGUFs (including sets).
    pub local_path: Option<String>,
    pub params_b: Option<f64>,
    pub quantization: Option<String>,
    pub context_train: Option<u64>,
    /// Unix timestamp (seconds) when the model was installed.
    pub installed_at: i64,
    /// Full path to the primary file (GGUF blob; empty for dir-layout models
    /// without a designated primary).
    pub primary_file_path: String,
    pub primary_file_size: u64,
    /// The primary file's path inside the source repo: the original HF
    /// filename, since single GGUFs are stored on disk as `<sha256>.gguf`.
    pub primary_rel_path: Option<String>,
    /// SHA-256 of the primary file when known.
    pub primary_sha256: Option<String>,
    /// Number of files registered for this model.
    pub file_count: u32,
    /// Sum of all registered file sizes in bytes.
    pub total_size: u64,
    pub pinned: bool,
    /// Resolved capability record (M15) — computed at read time, never stored.
    pub caps: capabilities::ModelCaps,
}

impl InstalledModel {
    /// A GGUF set's weights span every registered part, not just its header shard.
    pub fn weight_size(&self) -> u64 {
        if self.format == "gguf" && super::split_gguf::part(&self.primary_file_path).is_some() {
            self.total_size
        } else {
            self.primary_file_size
        }
    }
}

fn from_row(r: InstalledRow) -> InstalledModel {
    let cap: Capability =
        serde_json::from_str(&format!("\"{}\"", r.model.capability)).unwrap_or(Capability::Other);
    let mut model = InstalledModel {
        id: r.model.id,
        repo_id: r.model.repo_id,
        revision: r.model.revision,
        display_name: r.model.display_name,
        architecture: r.model.architecture,
        capability: cap,
        format: r.model.format,
        task: r.model.task,
        local_path: r.model.local_path,
        params_b: r.model.params_b,
        quantization: r.model.quantization,
        context_train: r.model.context_train.map(|v| v as u64),
        installed_at: r.model.installed_at,
        primary_file_path: r.primary.path,
        primary_file_size: r.primary.size as u64,
        primary_rel_path: r.primary.rel_path,
        primary_sha256: r.primary.sha256,
        file_count: r.file_count as u32,
        total_size: r.total_size as u64,
        pinned: r.model.pinned != 0,
        caps: capabilities::ModelCaps::default(),
    };
    model.caps = capabilities::resolve(&model);
    model
}

/// Canonical task string for a classified capability (None when ambiguous).
pub fn task_for_capability(cap: Capability) -> Option<&'static str> {
    match cap {
        Capability::TextGeneration => Some("text-generation"),
        Capability::Embedding => Some("feature-extraction"),
        Capability::Speech => Some("automatic-speech-recognition"),
        Capability::ImageGen => Some("text-to-image"),
        Capability::VisionText => Some("image-text-to-text"),
        Capability::Other => None,
    }
}

// ---------------------------------------------------------------------------
// Insert after a successful download
// ---------------------------------------------------------------------------

/// Write a new single-GGUF model + file row to the catalog after a download
/// completes. `job.file` is used to derive the display name.
/// Returns the model `id` (SHA-256 hex string).
pub async fn insert_after_download(
    pool: &SqlitePool,
    job: &Job,
    sha256: &str,
    gguf_meta: &GgufMeta,
    // Minimal card built from the job's repo_id so classify() can run
    card: &HfModelCard,
) -> Result<String, AppError> {
    let capability = capability::classify(card, gguf_meta.architecture.as_deref());

    // Display name: last path component without extension
    let display_name = PathBuf::from(&job.file)
        .file_name()
        .and_then(|n| n.to_str())
        .map(|n| {
            n.strip_suffix(".gguf")
                .unwrap_or(n)
                .replace('.', " ")
                .replace('-', " ")
        })
        .unwrap_or_else(|| job.repo_id.clone());

    let cap_str = serde_json::to_string(&capability)
        .unwrap_or_else(|_| "\"other\"".into())
        .trim_matches('"')
        .to_string();

    let task = card
        .pipeline_tag
        .clone()
        .or_else(|| task_for_capability(capability).map(String::from));

    let installed_at = Utc::now().timestamp();

    // Final blob path is <models_dir>/<sha256>.gguf
    let blob_path = config::models_dir().join(format!("{sha256}.gguf"));
    let blob_path_str = blob_path.to_string_lossy().to_string();

    // Original repo filename (e.g. "model.Q4_K_M.gguf") as rel_path.
    let rel_path = PathBuf::from(&job.file)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model.gguf")
        .to_string();

    let model = NewModel {
        id: sha256.to_string(),
        repo_id: job.repo_id.clone(),
        revision: job.revision.clone(),
        display_name,
        architecture: gguf_meta.architecture.clone(),
        capability: cap_str,
        format: "gguf".to_string(),
        task,
        local_path: None,
        params_b: gguf_meta.params_b,
        quantization: gguf_meta.quantization.clone(),
        context_train: gguf_meta.context_length,
        installed_at,
    };
    let files = vec![NewModelFile {
        role: "primary".to_string(),
        rel_path,
        path: blob_path_str,
        sha256: Some(sha256.to_string()),
        size: gguf_meta.file_size,
    }];

    db_models::insert_model_with_files(pool, &model, &files).await?;

    Ok(sha256.to_string())
}

// ---------------------------------------------------------------------------
// Insert after a repo (multi-file) download completes — M9 §9.2
// ---------------------------------------------------------------------------

/// One downloaded repo file, as reported by the download group on completion.
pub struct RepoFile {
    /// Path inside the repo (e.g. "unet/diffusion_pytorch_model.safetensors").
    pub rel_path: String,
    /// Absolute path on disk after finalization.
    pub path: String,
    pub sha256: Option<String>,
    pub size: u64,
}

/// Weight-file extensions — used to pick roles for repo files.
const WEIGHT_EXTS: &[&str] = &["safetensors", "gguf", "bin", "pt", "onnx", "ckpt"];

fn role_for_rel_path(rel_path: &str) -> &'static str {
    let ext = std::path::Path::new(rel_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if WEIGHT_EXTS.contains(&ext.as_str()) {
        "weight"
    } else if matches!(ext.as_str(), "json" | "txt" | "yaml" | "yml" | "model") {
        "config"
    } else {
        "other"
    }
}

/// Stable catalog id for a repo-layout model: SHA-256 of "repo_id@revision".
pub fn repo_model_id(repo_id: &str, revision: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(format!("{repo_id}@{revision}").as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Split sets retain the directory in the stem: quants and subfolders cannot collide.
pub fn gguf_set_model_id(repo_id: &str, revision: &str, stem: &str) -> String {
    repo_model_id(repo_id, &format!("{revision}#gguf:{stem}"))
}

/// Register a completed multi-file repo download in the catalog.
/// Returns the model id.
pub async fn insert_repo_model(
    pool: &SqlitePool,
    repo_id: &str,
    revision: &str,
    format: &str,
    task: Option<&str>,
    root_dir: &str,
    files: &[RepoFile],
    group_id: Option<&str>,
) -> Result<String, AppError> {
    let set_stem = if format == "gguf" {
        Some(super::split_gguf::validate(files.iter().map(|f| f.rel_path.as_str()))?)
    } else { None };
    let primary = files.iter().position(|f| {
        super::split_gguf::part(&f.rel_path).is_some_and(|p| p.number == 1)
    });
    let meta = if set_stem.is_some() {
        super::gguf::parse_header(Path::new(&files[primary.unwrap()].path))?
    } else { GgufMeta::default() };
    let id = set_stem.as_ref().map_or_else(
        || repo_model_id(repo_id, revision),
        |stem| gguf_set_model_id(repo_id, revision, stem),
    );
    let capability = task
        .map(capability::capability_for_task)
        .unwrap_or(Capability::Other);
    let cap_str = serde_json::to_string(&capability)
        .unwrap_or_else(|_| "\"other\"".into())
        .trim_matches('"')
        .to_string();

    // Display name: repo basename ("stabilityai/sdxl-turbo" → "sdxl-turbo").
    let display_name = set_stem.as_deref().unwrap_or(repo_id)
        .rsplit('/').next().unwrap_or(repo_id).to_string();

    // Role assignment; a lone weight file is promoted to 'primary' so
    // single-safetensors models behave like single-GGUF ones downstream.
    let mut roles: Vec<&'static str> = files
        .iter()
        .map(|f| role_for_rel_path(&f.rel_path))
        .collect();
    let weight_count = roles.iter().filter(|r| **r == "weight").count();
    if set_stem.is_some() {
        roles[primary.unwrap()] = "primary";
    } else if weight_count == 1 {
        if let Some(slot) = roles.iter_mut().find(|r| **r == "weight") {
            *slot = "primary";
        }
    }

    let model = NewModel {
        id: id.clone(),
        repo_id: repo_id.to_string(),
        revision: revision.to_string(),
        display_name,
        architecture: meta.architecture,
        capability: cap_str,
        format: format.to_string(),
        task: task.map(String::from),
        // GGUF sets own named files, never the shared repo directory.
        local_path: if format == "gguf" { None } else { Some(root_dir.to_string()) },
        params_b: meta.params_b,
        quantization: meta.quantization,
        context_train: meta.context_length,
        installed_at: Utc::now().timestamp(),
    };

    let new_files: Vec<NewModelFile> = files
        .iter()
        .zip(roles)
        .map(|(f, role)| NewModelFile {
            role: role.to_string(),
            rel_path: f.rel_path.clone(),
            path: f.path.clone(),
            sha256: f.sha256.clone(),
            size: f.size,
        })
        .collect();

    db_models::insert_model_with_files_once(pool, &model, &new_files,
        group_id.filter(|_| format == "gguf")).await?;

    Ok(id)
}

// ---------------------------------------------------------------------------
// Get by id
// ---------------------------------------------------------------------------

/// Look up a single installed model by its id.
/// Returns `None` if no model with that id is in the catalog.
pub async fn get_by_id(pool: &SqlitePool, id: &str) -> Result<Option<InstalledModel>, AppError> {
    let row = db_models::get_by_id(pool, id).await?;
    Ok(row.map(from_row))
}

// ---------------------------------------------------------------------------
// List
// ---------------------------------------------------------------------------

pub async fn list(pool: &SqlitePool) -> Result<Vec<InstalledModel>, AppError> {
    let rows = db_models::list_installed(pool).await?;
    Ok(rows.into_iter().map(from_row).collect())
}

/// Find the catalog model whose root dir or primary file is `path` — lets
/// the VRAM ledger correlate pipeline residents (keyed by model *path* in
/// job params) back to their catalog rows (W1). Linear over the installed
/// list, which is small.
pub async fn find_by_source_path(
    pool: &SqlitePool,
    path: &str,
) -> Result<Option<InstalledModel>, AppError> {
    if path.is_empty() {
        return Ok(None);
    }
    Ok(list(pool).await?.into_iter().find(|m| {
        m.local_path.as_deref() == Some(path) || m.primary_file_path == path
    }))
}

// ---------------------------------------------------------------------------
// Metadata backfill
// ---------------------------------------------------------------------------

/// Re-parse GGUF headers for catalog rows whose metadata never landed —
/// models installed while the parser still choked on large vocab arrays.
/// Idempotent: rows gain `architecture` on success and drop out of the
/// candidate set; unparseable files are left for the next attempt.
pub async fn backfill_gguf_metadata(pool: &SqlitePool) -> Result<u32, AppError> {
    let candidates = db_models::list_gguf_missing_meta(pool).await?;
    let mut healed = 0u32;
    for (id, path) in candidates {
        let meta = crate::models::gguf::parse_header(Path::new(&path)).unwrap_or_default();
        if meta.architecture.is_none() {
            continue; // still unparseable — keep the row as-is
        }
        db_models::update_gguf_meta(
            pool,
            &id,
            meta.architecture.as_deref(),
            meta.params_b,
            meta.quantization.as_deref(),
            meta.context_length.map(|v| v as i64),
        )
        .await?;
        healed += 1;
    }
    Ok(healed)
}

// ---------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------

/// Remove a model from the catalog and delete its files from disk.
/// GGUFs remove only their blob or split siblings; dir layouts own their root dir.
pub async fn delete(pool: &SqlitePool, id: &str) -> Result<(), AppError> {
    delete_in(pool, id, &config::models_dir()).await
}

async fn delete_in(pool: &SqlitePool, id: &str, models_dir: &Path) -> Result<(), AppError> {
    // Look up format/local_path BEFORE deleting the row.
    let row = db_models::get_by_id(pool, id).await?;

    // DB delete cascades to model_files
    db_models::delete_by_id(pool, id).await?;

    match row {
        Some(r) if r.model.format == "gguf" => {
            delete_gguf_files(Path::new(&r.primary.path), models_dir)?;
        }
        Some(r) if r.model.local_path.is_some() => {
            let dir = PathBuf::from(r.model.local_path.unwrap());
            // Refuse to remove anything outside models_dir — a corrupted row
            // must never delete an arbitrary directory.
            if dir.starts_with(models_dir) && dir.exists() {
                if let Err(e) = std::fs::remove_dir_all(&dir) {
                    tracing::warn!("registry::delete: could not remove {:?}: {}", dir, e);
                }
            } else if dir.exists() {
                tracing::warn!(
                    "registry::delete: refusing to remove {:?} (outside models dir)",
                    dir
                );
            }
        }
        _ => {
            let blob_path = models_dir.join(format!("{id}.gguf"));
            if blob_path.exists() {
                if let Err(e) = std::fs::remove_file(&blob_path) {
                    tracing::warn!("registry::delete: could not remove {:?}: {}", blob_path, e);
                }
            }
        }
    }

    Ok(())
}

/// File-backed ownership: only the primary's exact split siblings, not their folder.
fn delete_gguf_files(primary: &Path, models_dir: &Path) -> Result<(), AppError> {
    let root = std::fs::canonicalize(models_dir)?;
    // Construct siblings even when one is missing (a partially removed install).
    let name = primary.to_string_lossy();
    let paths: Vec<PathBuf> = if let Some(p) = super::split_gguf::part(&name) {
        let (head, count) = name.strip_suffix(".gguf").unwrap().rsplit_once("-of-").unwrap();
        let (_, number) = head.rsplit_once('-').unwrap();
        (1..=p.count).map(|n| PathBuf::from(format!("{}-{n:0width$}-of-{count}.gguf", p.stem, width=number.len()))).collect()
    } else { vec![primary.to_path_buf()] };
    for path in paths {
        if !path.exists() { continue; }
        let real = std::fs::canonicalize(&path)?;
        if real.starts_with(&root) { std::fs::remove_file(path)?; }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Boot-time reconcile: walk models_dir and sync with DB
// ---------------------------------------------------------------------------

/// Walk `models_dir()`, ensuring each `*.gguf` blob has a catalog row,
/// and that each catalog row still has its files on disk.
///
/// Orphaned blobs (no DB row): parsed and inserted.
/// Orphaned rows (no blob / no model dir): deleted from DB.
pub async fn scan_dir(pool: &SqlitePool) -> Result<(), AppError> {
    scan_dir_in(pool, &config::models_dir()).await
}

async fn scan_dir_in(pool: &SqlitePool, models_dir: &Path) -> Result<(), AppError> {

    // --- Pass 1: collect blobs on disk ---
    let blobs: Vec<PathBuf> = {
        let mut v = Vec::new();
        match std::fs::read_dir(&models_dir) {
            Ok(entries) => {
                for entry in entries.flatten() {
                    let p = entry.path();
                    if p.extension().and_then(|e| e.to_str()) == Some("gguf") {
                        v.push(p);
                    }
                }
            }
            Err(e) => {
                tracing::warn!("scan_dir: cannot read models dir: {}", e);
                return Ok(());
            }
        }
        v
    };

    // --- Pass 2: collect known model IDs from DB ---
    let db_rows = db_models::list_installed(pool).await?;
    let known_ids: std::collections::HashSet<String> =
        db_rows.iter().map(|r| r.model.id.clone()).collect();

    // --- Pass 3: blob → sha256 from filename; insert if missing ---
    let mut disk_ids: std::collections::HashSet<String> = std::collections::HashSet::new();

    for blob in &blobs {
        let stem = blob
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_string();

        // Filename must be 64 hex chars (SHA-256)
        if stem.len() != 64 || !stem.chars().all(|c| c.is_ascii_hexdigit()) {
            tracing::warn!("scan_dir: skipping non-sha256 blob {:?}", blob);
            continue;
        }

        disk_ids.insert(stem.clone());

        if !known_ids.contains(&stem) {
            tracing::info!("scan_dir: orphaned blob {:?} — parsing and inserting", blob);
            let meta = crate::models::gguf::parse_header(blob).unwrap_or_default();

            // Minimal card; classify will use GGUF arch
            let card = HfModelCard {
                id: format!("unknown/{stem}"),
                author: None,
                pipeline_tag: None,
                library_name: Some("gguf".to_string()),
                tags: vec![],
                downloads: 0,
                likes: 0,
                trending_score: None,
                gated: false,
                private: false,
                last_modified: None,
                created_at: None,
            };

            // Build a synthetic job for insert_after_download
            let synthetic_job = Job {
                id: uuid::Uuid::new_v4(),
                repo_id: "unknown".to_string(),
                revision: "main".to_string(),
                file: blob
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("model.gguf")
                    .to_string(),
                dest_path: blob.clone(),
                total_bytes: meta.file_size,
                downloaded: meta.file_size,
                chunks: vec![],
                status: crate::downloader::job::JobStatus::Done,
                error: None,
                expected_sha256: Some(stem.clone()),
                group_id: None,
                created_at: 0,
                updated_at: 0,
            };

            if let Err(e) = insert_after_download(pool, &synthetic_job, &stem, &meta, &card).await {
                tracing::warn!("scan_dir: failed to insert {}: {}", stem, e);
            }
        }
    }

    // --- Pass 4: remove rows whose files no longer exist ---
    // File-backed GGUF rows require the primary and every split sibling.
    // Dir-layout rows require their owned root (contents are not deep-verified).
    // TODO(M4): consider gating this deletion behind a user-confirmation step,
    // especially if the models directory could be on a removable or network drive.
    for row in &db_rows {
        let missing = if row.model.format == "gguf" {
            let primary = Path::new(&row.primary.path);
            !primary.exists() || super::gguf::split_files(primary).is_err()
        } else {
            match row.model.local_path.as_deref() {
                Some(dir) => !Path::new(dir).exists(),
                None => !disk_ids.contains(&row.model.id),
            }
        };
        if missing {
            tracing::warn!(
                "scan_dir: orphaned row — model id={} repo={} files not found on disk; deleting catalog row",
                row.model.id,
                row.model.repo_id,
            );
            if let Err(e) = db_models::delete_by_id(pool, &row.model.id).await {
                tracing::warn!("scan_dir: delete failed for {}: {}", row.model.id, e);
            }
        }
    }

    Ok(())
}
