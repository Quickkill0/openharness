//! The qwentts.cpp runtime as a [`RuntimeManifest`].
//!
//! This is the **only** qwentts-specific provisioning data: download URLs,
//! checksums, the target dir, and the list of files the install must end up
//! with. Everything else (download, verify, extract, rollback, stamp) is the
//! generic `provision` machinery.
//!
//! ## HOSTING STATUS — placeholders, not live
//!
//! Upstream qwentts.cpp publishes **no Windows binaries**; the working set
//! was built locally and is not hosted anywhere yet. The URLs and checksums
//! below are PLACEHOLDERS. An install attempt against them fails with an
//! honest "engine binaries are not hosted yet" error (the generic installer
//! classifies the 404 / connection failure), it does NOT panic or extract
//! garbage.
//!
//! To make qwentts auto-install go live, the maintainer must:
//!   1. Build (or obtain) the Windows binaries listed in [`REQUIRED_FILES`]
//!      (the synthesis set) plus [`CLONE_CLI_FILE`] (the optional cloning CLI).
//!   2. Package + upload them to a GitHub release (one ZIP for the binaries +
//!      DLLs, one loose GGUF for the codec is the shape assumed below — adjust
//!      `assets()` if you package differently).
//!   3. Generate SHA-256 digests for each uploaded asset.
//!   4. Fill in [`QWENTTS_RELEASE`], the asset filenames, and the
//!      [`CHECKSUMS`] table below, and flip [`HOSTED`] to `true`.
//!
//! See CLAUDE.md → "Qwen3-TTS engine provisioning" for the full runbook.

use crate::provision::{AssetKind, ManifestAsset, RuntimeManifest};

/// Build pin. Bumping this re-provisions (the stamp won't match).
pub const QWENTTS_VERSION: &str = "v0.1.0-placeholder";

// TODO(host): replace once binaries are published to a GitHub release.
/// Base URL of the GitHub release holding the qwentts Windows binaries.
/// PLACEHOLDER — the repo + tag below do not host these assets yet.
const QWENTTS_RELEASE: &str =
    "https://github.com/REPLACE_ME/qwentts.cpp/releases/download/REPLACE_TAG";

/// Asset file names hosted at [`QWENTTS_RELEASE`]. The binaries + DLLs are
/// assumed to ship as one ZIP; the codec GGUF as a loose file.
const BINARIES_ZIP: &str = "qwentts-win-x64-cuda13.zip";
const CODEC_GGUF: &str = "qwen-tokenizer-12hz-Q8_0.gguf";

// TODO(host): fill these once the assets exist and digests are generated
// (`sha256sum <asset>` or `Get-FileHash -Algorithm SHA256 <asset>`).
/// Expected SHA-256 (lowercase hex) per hosted asset. `None` means "not
/// pinned yet" — TLS is the only integrity story until a digest is filled in.
const BINARIES_ZIP_SHA256: Option<&str> = None;
const CODEC_GGUF_SHA256: Option<&str> = None;

/// Flip to `true` once [`QWENTTS_RELEASE`] + the filenames + checksums above
/// point at real, uploaded assets. Purely documentary today: the installer
/// already degrades to an honest error when the URLs 404, so nothing branches
/// on this — it's the one-line signal a maintainer toggles when going live.
#[allow(dead_code)]
pub const HOSTED: bool = false;

/// Every file the **synthesis** runtime needs to be considered provisioned —
/// i.e. what `tts::is_installed()` checks and what gates the Studio Speech UI.
/// This is deliberately the synthesis set only; the voice-cloning CLI
/// ([`CLONE_CLI_FILE`]) is gated separately by `cli::cli_installed` so a
/// missing cloning binary never falsely reports the whole engine as "not
/// installed" — synthesis only needs `tts-server.exe` + codec + DLLs:
///   - `tts-server.exe`        — the OpenAI-compatible synthesis sidecar
///   - `qwen-tokenizer-12hz-Q8_0.gguf` — the shared codec/tokenizer GGUF
///   - ggml DLLs               — `ggml.dll`, `ggml-base.dll`, `ggml-cpu.dll`,
///                               `ggml-cuda.dll`
///   - CUDA 13 runtime DLLs    — `cublas64_13.dll`, `cublasLt64_13.dll`,
///                               `cudart64_13.dll` (from CUDA's `bin\x64\`)
pub const REQUIRED_FILES: &[&str] = &[
    "tts-server.exe",
    "qwen-tokenizer-12hz-Q8_0.gguf",
    "cublas64_13.dll",
    "cublasLt64_13.dll",
    "cudart64_13.dll",
    "ggml-base.dll",
    "ggml-cpu.dll",
    "ggml-cuda.dll",
    "ggml.dll",
];

/// The one-shot voice-cloning CLI. It ships in the same release ZIP as the
/// synthesis binaries, but it is **not** in [`REQUIRED_FILES`] on purpose:
/// cloning is an optional capability (only Base talkers can clone), so its
/// presence is checked solely by `cli::cli_installed` (via `cli::cli_path`).
/// A partial hand-placed install that has the synthesis set but lacks this
/// file is a perfectly usable synthesis engine — it just can't clone.
pub const CLONE_CLI_FILE: &str = "qwen-tts.exe";

/// Disk budget for the install: the talker-less runtime (binaries + DLLs +
/// codec) unpacks to well under 1 GiB; 2 GiB is a comfortable precheck.
const NEEDED_BYTES: u64 = 2 * 1024 * 1024 * 1024;

fn assets() -> Vec<ManifestAsset> {
    vec![
        ManifestAsset {
            url: format!("{QWENTTS_RELEASE}/{BINARIES_ZIP}"),
            filename: BINARIES_ZIP.to_string(),
            sha256: BINARIES_ZIP_SHA256.map(str::to_string),
            kind: AssetKind::Zip,
        },
        ManifestAsset {
            url: format!("{QWENTTS_RELEASE}/{CODEC_GGUF}"),
            filename: CODEC_GGUF.to_string(),
            sha256: CODEC_GGUF_SHA256.map(str::to_string),
            kind: AssetKind::RawFile,
        },
    ]
}

/// The qwentts provisioning manifest. `install_dir` is the same
/// `engine\qwentts\` directory the rest of the `tts` module reads from, so a
/// successful provision lights up `tts::is_installed()` / `cli::cli_installed()`
/// with zero other changes.
pub fn qwentts_manifest() -> RuntimeManifest {
    RuntimeManifest {
        id: "qwentts",
        version: QWENTTS_VERSION,
        install_dir: super::install_dir(),
        assets: assets(),
        required_files: REQUIRED_FILES.to_vec(),
        needed_bytes: NEEDED_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_targets_the_qwentts_install_dir() {
        let m = qwentts_manifest();
        assert_eq!(m.id, "qwentts");
        assert_eq!(m.install_dir, super::super::install_dir());
        // The codec required-file matches what bundled_codec_path() reads.
        assert!(m
            .required_files
            .contains(&"qwen-tokenizer-12hz-Q8_0.gguf"));
        assert!(m.required_files.contains(&"tts-server.exe"));
        // The cloning CLI is gated separately (cli::cli_installed), so it must
        // NOT be in the synthesis required_files — otherwise a synthesis-only
        // install falsely reports the whole engine as "not installed".
        assert!(!m.required_files.contains(&CLONE_CLI_FILE));
    }

    #[test]
    fn required_files_cover_the_synthesis_set_only() {
        // The synthesis runtime: tts-server.exe + codec + the seven DLLs. The
        // cloning CLI (qwen-tts.exe) is intentionally excluded — see
        // CLONE_CLI_FILE / cli::cli_installed.
        assert_eq!(REQUIRED_FILES.len(), 9);
        for must in [
            "tts-server.exe",
            "qwen-tokenizer-12hz-Q8_0.gguf",
            "cublas64_13.dll",
            "cublasLt64_13.dll",
            "cudart64_13.dll",
            "ggml-base.dll",
            "ggml-cpu.dll",
            "ggml-cuda.dll",
            "ggml.dll",
        ] {
            assert!(REQUIRED_FILES.contains(&must), "missing {must}");
        }
        // The cloning CLI is its own const, not a synthesis requirement.
        assert_eq!(CLONE_CLI_FILE, "qwen-tts.exe");
        assert!(!REQUIRED_FILES.contains(&CLONE_CLI_FILE));
    }

    #[test]
    fn assets_are_zip_then_rawfile() {
        let a = assets();
        assert_eq!(a.len(), 2);
        assert_eq!(a[0].kind, AssetKind::Zip);
        assert_eq!(a[1].kind, AssetKind::RawFile);
        // Placeholder URLs are clearly marked, not real.
        assert!(a[0].url.contains("REPLACE_ME"));
    }

    #[test]
    fn not_hosted_yet() {
        // Guards against accidentally shipping HOSTED=true with placeholder
        // URLs still in place.
        assert!(!HOSTED || !QWENTTS_RELEASE.contains("REPLACE_ME"));
    }
}
