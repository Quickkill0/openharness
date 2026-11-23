//! Install-time fetch of the Qwen2.5-1.5B tokenizer that VibeVoice needs.
//!
//! VibeVoice's processor (`VibeVoiceProcessor.from_pretrained`) loads a
//! language-model tokenizer, but the official checkpoints record it as the
//! bare repo id `"Qwen/Qwen2.5-1.5B"`. Loading that reaches out to Hugging
//! Face at *inference* time, which violates the local-first / no-auto-download
//! policy and fails outright when the tokenizer isn't in the HF cache.
//!
//! We close the gap at *install* time: when a user explicitly downloads a
//! VibeVoice repo, we also fetch the tokenizer-only files of Qwen2.5-1.5B
//! (never the weights) into `<model_dir>/_qwen_tokenizer/`. The worker pipeline
//! then points `language_model_pretrained_name` at that local dir and runs
//! fully offline. The user chose to install the model, so its tokenizer
//! dependency riding along is consistent with "no auto-download".

use std::path::Path;

use crate::error::AppError;

/// The HF repo that owns the tokenizer VibeVoice's language model uses.
pub const QWEN_TOKENIZER_REPO: &str = "Qwen/Qwen2.5-1.5B";

/// Subdir, under a VibeVoice install's root, that holds the bundled tokenizer.
/// Must match `QWEN_TOKENIZER_SUBDIR` in `pipelines/vibevoice.py`.
pub const TOKENIZER_SUBDIR: &str = "_qwen_tokenizer";

/// Tokenizer-only files to fetch — no weights. `tokenizer.json` (fast) is the
/// one the VibeVoice fast tokenizer actually needs; the rest are fetched
/// best-effort so slow-tokenizer fallbacks and special-token handling work.
const TOKENIZER_FILES: &[&str] = &[
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
];

const USER_AGENT: &str = "OpenHarness/0.1 (+https://github.com/openharness/openharness)";

/// Does this download group describe a VibeVoice repo? VibeVoice cards carry
/// the generic "text-to-speech" task (or none), so — like the capability
/// resolver's `is_vibevoice` — the repo id is the reliable signal.
pub fn is_vibevoice_repo(repo_id: &str) -> bool {
    repo_id.to_lowercase().contains("vibevoice")
}

/// Fetch the Qwen2.5-1.5B tokenizer into `<root>/_qwen_tokenizer/` unless it is
/// already present. Idempotent and best-effort per file: a missing optional
/// file (e.g. a repo that only ships `tokenizer.json`) is skipped, but if
/// neither `tokenizer.json` nor `tokenizer_config.json` lands the call errors,
/// since the pipeline would then refuse to load anyway.
pub async fn ensure_qwen_tokenizer(
    http: &reqwest::Client,
    root: &Path,
) -> Result<(), AppError> {
    let dest_dir = root.join(TOKENIZER_SUBDIR);

    // Already installed — nothing to do (cheap startup-heal short-circuit).
    if dest_dir.join("tokenizer.json").exists()
        || dest_dir.join("tokenizer_config.json").exists()
    {
        return Ok(());
    }

    std::fs::create_dir_all(&dest_dir)?;

    let token = crate::auth::get_hf_token().ok().flatten();
    let mut fetched_core = false;

    for &file in TOKENIZER_FILES {
        let url = format!(
            "https://huggingface.co/{QWEN_TOKENIZER_REPO}/resolve/main/{file}"
        );
        match download_to(http, &url, token.as_deref(), &dest_dir.join(file)).await {
            Ok(()) => {
                if file == "tokenizer.json" || file == "tokenizer_config.json" {
                    fetched_core = true;
                }
            }
            Err(e) => {
                // Optional files (vocab/merges/special_tokens) are legitimately
                // absent from the base repo — e.g. `special_tokens_map.json`
                // isn't shipped by `Qwen/Qwen2.5-1.5B`, so this fires on every
                // install. It's expected and harmless: the core-file check below
                // is what actually gates success. Debug-level, not a warning.
                tracing::debug!(
                    "VibeVoice tokenizer: optional file {file} not present at {QWEN_TOKENIZER_REPO} ({e}) — skipping"
                );
            }
        }
    }

    if !fetched_core {
        return Err(AppError::Other(format!(
            "VibeVoice tokenizer install failed — could not fetch the \
             Qwen2.5-1.5B tokenizer from {QWEN_TOKENIZER_REPO}. The model will \
             not run until it is present at {}.",
            dest_dir.display()
        )));
    }

    tracing::info!(
        "VibeVoice tokenizer installed at {} (from {QWEN_TOKENIZER_REPO})",
        dest_dir.display()
    );
    Ok(())
}

/// GET `url` and write the body to `dest` atomically (via a `.part` rename).
/// A 404 is surfaced as an error so the caller can treat the file as optional.
async fn download_to(
    http: &reqwest::Client,
    url: &str,
    token: Option<&str>,
    dest: &Path,
) -> Result<(), AppError> {
    let mut req = http.get(url).header("User-Agent", USER_AGENT);
    if let Some(t) = token {
        req = req.bearer_auth(t);
    }
    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        return Err(AppError::HttpStatus(status.as_u16(), url.to_string()));
    }
    let bytes = resp.bytes().await?;

    let part = dest.with_extension("part");
    std::fs::write(&part, &bytes)?;
    std::fs::rename(&part, dest)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_vibevoice_by_repo_id() {
        assert!(is_vibevoice_repo("vibevoice/VibeVoice-1.5B"));
        assert!(is_vibevoice_repo("vibevoice-community/VibeVoice-1.5B"));
        assert!(is_vibevoice_repo("someone/MyVibeVoiceClone"));
        assert!(!is_vibevoice_repo("hexgrad/Kokoro-82M"));
        assert!(!is_vibevoice_repo("Qwen/Qwen2.5-1.5B"));
    }

    #[test]
    fn subdir_matches_pipeline_constant() {
        // Must stay in lockstep with pipelines/vibevoice.py QWEN_TOKENIZER_SUBDIR.
        assert_eq!(TOKENIZER_SUBDIR, "_qwen_tokenizer");
    }

    #[test]
    fn ensure_is_noop_when_tokenizer_present() {
        // If tokenizer.json already exists, no network client is touched.
        let tmp = std::env::temp_dir().join(format!(
            "oh-vv-tok-test-{}",
            uuid::Uuid::new_v4()
        ));
        let dest = tmp.join(TOKENIZER_SUBDIR);
        std::fs::create_dir_all(&dest).unwrap();
        std::fs::write(dest.join("tokenizer.json"), b"{}").unwrap();

        let http = reqwest::Client::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let res = rt.block_on(ensure_qwen_tokenizer(&http, &tmp));
        assert!(res.is_ok());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
