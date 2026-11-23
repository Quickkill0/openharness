//! Install-time fetch of VibeVoice's built-in *preset* voices.
//!
//! VibeVoice has no voices baked into its weights — it is a pure voice-
//! *conditioning* model that always clones from a reference audio clip. The
//! "voices" in the upstream demos are just sample clips shipped alongside the
//! code. So "built-in voices" here means a small, curated set of reference
//! clips presented to the user as named presets: pick "Alice" and the pipeline
//! quietly uses Alice's clip as the cloning reference. This spares non-technical
//! users from having to record or supply their own sample just to hear speech.
//!
//! The clips are the MIT-licensed demo voices from the maintained community
//! fork, github.com/vibevoice-community/VibeVoice (`demo/voices/`). We fetch a
//! fixed curated subset (clean speech, no background music) at *install* time —
//! when the user explicitly downloads a VibeVoice repo — into
//! `<model_dir>/_builtin_voices/<voice-id>.wav`. The worker pipeline only ever
//! reads these local files; nothing touches the network at inference. The user
//! chose to install the model, so its preset voices riding along is consistent
//! with "no auto-download".
//!
//! Mirrors `vibevoice_tokenizer.rs` (idempotent, atomic `.part` rename,
//! best-effort per file). The id↔filename mapping is the single source of truth
//! the capability resolver also reads, so the dropdown and the on-disk files
//! can never drift.

use std::path::Path;

use crate::error::AppError;

/// Subdir, under a VibeVoice install's root, that holds the bundled preset
/// voice clips. Must match `BUILTIN_VOICES_SUBDIR` in `pipelines/vibevoice.py`.
pub const BUILTIN_VOICES_SUBDIR: &str = "_builtin_voices";

/// Raw-content base for the community fork's `demo/voices/` directory. The
/// clips are MIT-licensed; no auth/token is needed for raw.githubusercontent.
const VOICES_BASE_URL: &str =
    "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices";

const USER_AGENT: &str = "OpenHarness/0.1 (+https://github.com/openharness/openharness)";

/// Curated built-in voice set: `(id, label, source filename)`. The `id` names
/// the on-disk file (`<id>.wav`) and the job param; `label` is what the UI
/// shows; the filename is the wav as published in the community fork's
/// `demo/voices/`. All clean single-speaker speech, no background music.
///
/// Single source of truth: `capabilities::vibevoice_speech_caps` builds its
/// `VoiceInfo` list from `builtin_voice_list()` (id/label) and the pipeline
/// resolves `<id>.wav`, so this slice is the only place the set is defined.
pub const BUILTIN_VOICES: &[(&str, &str, &str)] = &[
    ("alice", "Alice — English female", "en-Alice_woman.wav"),
    ("maya", "Maya — English female", "en-Maya_woman.wav"),
    ("carter", "Carter — English male", "en-Carter_man.wav"),
    ("frank", "Frank — English male", "en-Frank_man.wav"),
    ("samuel", "Samuel — English male", "in-Samuel_man.wav"),
];

/// The `(id, label)` pairs for the capability resolver — same source slice, so
/// the dropdown and the fetched files stay in lockstep.
pub fn builtin_voice_list() -> Vec<(&'static str, &'static str)> {
    BUILTIN_VOICES.iter().map(|(id, label, _)| (*id, *label)).collect()
}

/// Fetch the curated preset voice clips into `<root>/_builtin_voices/` for any
/// that are missing. Idempotent (already-present files are skipped without a
/// network call) and best-effort per file: a single failed fetch is logged and
/// skipped — the UI simply won't offer that one — rather than failing the whole
/// install. Creates the subdir on demand.
pub async fn ensure_builtin_voices(
    http: &reqwest::Client,
    root: &Path,
) -> Result<(), AppError> {
    let dest_dir = root.join(BUILTIN_VOICES_SUBDIR);

    // Cheap startup-heal short-circuit: if every clip is already on disk there
    // is nothing to do and no client is touched.
    let all_present = BUILTIN_VOICES
        .iter()
        .all(|(id, _, _)| dest_dir.join(format!("{id}.wav")).exists());
    if all_present {
        return Ok(());
    }

    std::fs::create_dir_all(&dest_dir)?;

    let mut fetched = 0usize;
    for (id, _label, filename) in BUILTIN_VOICES {
        let dest = dest_dir.join(format!("{id}.wav"));
        if dest.exists() {
            fetched += 1;
            continue;
        }
        let url = format!("{VOICES_BASE_URL}/{filename}");
        match download_to(http, &url, &dest).await {
            Ok(()) => {
                fetched += 1;
                tracing::debug!("VibeVoice preset voice '{id}' installed from {filename}");
            }
            Err(e) => {
                // Best-effort: a single unreachable clip just won't be offered.
                tracing::warn!(
                    "VibeVoice preset voice '{id}' fetch failed ({e}) — skipping; the voice won't be offered until a reinstall or the next startup heal"
                );
            }
        }
    }

    if fetched > 0 {
        tracing::info!(
            "VibeVoice preset voices: {fetched}/{} present at {}",
            BUILTIN_VOICES.len(),
            dest_dir.display()
        );
    }
    Ok(())
}

/// Startup self-heal: backfill preset voices for every already-installed
/// VibeVoice model so existing installs gain the dropdown without a reinstall.
///
/// Resolves the install set from the catalog (the capability resolver tags
/// VibeVoice rows with `pipeline == "vibevoice"`), then ensures each one's
/// `local_path` carries the clips. Best-effort throughout — a DB error or a
/// failed fetch is logged and never propagated, so this can run as a detached
/// background task that must not block or crash startup.
pub async fn heal_installed_voices(pool: &sqlx::SqlitePool, http: &reqwest::Client) {
    let models = match crate::models::registry::list(pool).await {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("VibeVoice preset-voice heal: catalog list failed: {e}");
            return;
        }
    };
    for m in models {
        // The resolver routes VibeVoice to the `vibevoice` worker pipeline; that
        // is the data-driven signal (no model-id special-casing). `local_path`
        // is the dir-layout root the clips live under.
        if m.caps.pipeline.as_deref() != Some("vibevoice") {
            continue;
        }
        let Some(root) = m.local_path.as_deref() else { continue };
        if let Err(e) = ensure_builtin_voices(http, Path::new(root)).await {
            tracing::warn!(
                "VibeVoice preset-voice heal for {} failed: {e}",
                m.repo_id
            );
        }
    }
}

/// GET `url` and write the body to `dest` atomically (via a `.part` rename).
/// No bearer auth — the clips live on raw.githubusercontent and need no token.
async fn download_to(
    http: &reqwest::Client,
    url: &str,
    dest: &Path,
) -> Result<(), AppError> {
    let resp = http.get(url).header("User-Agent", USER_AGENT).send().await?;
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
    fn subdir_matches_pipeline_constant() {
        // Must stay in lockstep with pipelines/vibevoice.py BUILTIN_VOICES_SUBDIR.
        assert_eq!(BUILTIN_VOICES_SUBDIR, "_builtin_voices");
    }

    #[test]
    fn voice_list_is_consistent_and_non_empty() {
        assert!(!BUILTIN_VOICES.is_empty());
        // The brief curates exactly five.
        assert_eq!(BUILTIN_VOICES.len(), 5);
        // ids must be unique, non-empty, and filenames must be .wav.
        let mut ids: Vec<&str> = BUILTIN_VOICES.iter().map(|(id, _, _)| *id).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), BUILTIN_VOICES.len(), "voice ids must be unique");
        for (id, label, file) in BUILTIN_VOICES {
            assert!(!id.is_empty(), "voice id must be non-empty");
            assert!(!label.is_empty(), "voice label must be non-empty");
            assert!(file.ends_with(".wav"), "{file} should be a .wav");
        }
        // The id/label projection mirrors the slice exactly.
        let pairs = builtin_voice_list();
        assert_eq!(pairs.len(), BUILTIN_VOICES.len());
        assert!(pairs.iter().any(|(id, _)| *id == "alice"));
    }

    #[test]
    fn ensure_is_noop_when_all_voices_present() {
        // With every clip already on disk, no network client is touched.
        let tmp = std::env::temp_dir().join(format!(
            "oh-vv-voices-test-{}",
            uuid::Uuid::new_v4()
        ));
        let dest = tmp.join(BUILTIN_VOICES_SUBDIR);
        std::fs::create_dir_all(&dest).unwrap();
        for (id, _, _) in BUILTIN_VOICES {
            std::fs::write(dest.join(format!("{id}.wav")), b"RIFF").unwrap();
        }

        let http = reqwest::Client::new();
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let res = rt.block_on(ensure_builtin_voices(&http, &tmp));
        assert!(res.is_ok());

        let _ = std::fs::remove_dir_all(&tmp);
    }
}
