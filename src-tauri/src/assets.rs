//! Asset library promotion (M10): when a generation job completes, its
//! outputs move from `assets\staging\` into per-kind library directories and
//! get `assets` rows. The promoted (final) paths replace the staging paths on
//! the `generation_jobs` row, so the job history and the library agree.

use std::path::{Path, PathBuf};

use serde_json::Value;
use sqlx::SqlitePool;

use crate::config;
use crate::db::assets::{self, NewAsset};
use crate::generation::model_label;

/// Asset kind from a file extension. Unknown extensions count as `text` —
/// every current pipeline output is covered, and a wrong guess only affects
/// which gallery section the file lands in.
pub fn kind_for_path(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "webp" | "bmp" | "gif" => "image",
        "mp4" | "webm" | "mkv" | "mov" | "avi" => "video",
        "wav" | "mp3" | "flac" | "ogg" | "m4a" | "opus" => "audio",
        _ => "text",
    }
}

/// Library subdirectory per kind.
fn subdir_for_kind(kind: &str) -> &'static str {
    match kind {
        "image" => "images",
        "video" => "video",
        "audio" => "audio",
        _ => "transcripts",
    }
}

/// Prompt-ish text for the library row: `prompt` (image/video) or `text` (TTS).
fn prompt_from_params(params: &Value) -> Option<String> {
    params
        .get("prompt")
        .or_else(|| params.get("text"))
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
}

/// Pick a non-colliding destination for `file_name` inside `dir`:
/// `name.ext`, then `name-1.ext`, `name-2.ext`, …
fn unique_dest(dir: &Path, file_name: &str) -> PathBuf {
    let candidate = dir.join(file_name);
    if !candidate.exists() {
        return candidate;
    }
    let p = Path::new(file_name);
    let stem = p.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default();
    let ext = p.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();
    for n in 1.. {
        let candidate = dir.join(format!("{stem}-{n}{ext}"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!()
}

/// Move one staged output into the library. Files already outside staging
/// stay where they are (still registered). Returns the final path.
fn promote_file(output: &Path) -> std::io::Result<PathBuf> {
    let staging = config::assets_staging_dir();
    if !output.starts_with(&staging) {
        return Ok(output.to_path_buf());
    }
    let kind = kind_for_path(output);
    let dir = config::assets_dir().join(subdir_for_kind(kind));
    std::fs::create_dir_all(&dir)?;
    let file_name = output
        .file_name()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| "output".into());
    let dest = unique_dest(&dir, &file_name);
    std::fs::rename(output, &dest)?;
    Ok(dest)
}

/// Promote a finished job's outputs and register them in the asset library.
/// Returns the final output paths (for the `generation_jobs` row). A failed
/// move keeps the staging path — the file is never lost, just unpromoted.
pub async fn promote_job_outputs(
    db: &SqlitePool,
    job_id: &str,
    task: &str,
    params: &Value,
    outputs: &[String],
    meta: &Value,
) -> Vec<String> {
    let prompt = prompt_from_params(params);
    let label = model_label(task, params);
    let mut final_paths = Vec::with_capacity(outputs.len());

    for out in outputs {
        let src = Path::new(out);
        let dest = match promote_file(src) {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("asset promotion failed for {out}: {e}");
                src.to_path_buf()
            }
        };
        let size = dest.metadata().map(|m| m.len() as i64).unwrap_or(0);
        let dest_str = dest.to_string_lossy().into_owned();
        let new_asset = NewAsset {
            job_id,
            kind: kind_for_path(&dest),
            path: &dest_str,
            task,
            prompt: prompt.as_deref(),
            model: Some(label.as_str()),
            params,
            meta,
            size,
        };
        if let Err(e) = assets::insert(db, new_asset).await {
            tracing::warn!("asset registration failed for {dest_str}: {e}");
        }
        final_paths.push(dest_str);
    }
    final_paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn kind_by_extension() {
        assert_eq!(kind_for_path(Path::new("a/image-x-00.png")), "image");
        assert_eq!(kind_for_path(Path::new("video-x.MP4")), "video");
        assert_eq!(kind_for_path(Path::new("speech-x.wav")), "audio");
        assert_eq!(kind_for_path(Path::new("transcript-x.json")), "text");
        assert_eq!(kind_for_path(Path::new("transcript-x.txt")), "text");
        assert_eq!(kind_for_path(Path::new("noext")), "text");
    }

    #[test]
    fn prompt_falls_back_to_text() {
        assert_eq!(
            prompt_from_params(&json!({"prompt": "a cat"})).as_deref(),
            Some("a cat")
        );
        assert_eq!(
            prompt_from_params(&json!({"text": "hello world"})).as_deref(),
            Some("hello world")
        );
        assert_eq!(prompt_from_params(&json!({"audio": "x.wav"})), None);
    }

    #[test]
    fn unique_dest_appends_counter() {
        let dir = tempfile::tempdir().unwrap();
        let first = unique_dest(dir.path(), "out.png");
        assert_eq!(first, dir.path().join("out.png"));
        std::fs::write(&first, b"x").unwrap();
        let second = unique_dest(dir.path(), "out.png");
        assert_eq!(second, dir.path().join("out-1.png"));
        std::fs::write(&second, b"x").unwrap();
        assert_eq!(unique_dest(dir.path(), "out.png"), dir.path().join("out-2.png"));
    }
}
