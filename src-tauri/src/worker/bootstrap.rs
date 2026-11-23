//! Worker venv bootstrap: pin-and-fetch `uv`, then materialize a locked venv.
//!
//! Layout under `%LOCALAPPDATA%\OpenHarness\worker\`:
//!   uv\<UV_VERSION>\uv.exe   — pinned standalone uv
//!   python\                  — uv-managed CPython installs
//!   uv-cache\                — uv download/wheel cache (survives reinstalls)
//!   venv\                    — the worker venv (recreated on any pin change)
//!   venv.stamp               — JSON stamp recording what the venv was built from
//!
//! Mirrors `engine/installer.rs`: staged progress closure, disk precheck,
//! rollback of the venv (never the reusable uv/python/cache) on failure.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::config::worker_dir;
use crate::error::AppError;
use crate::worker::{WorkerVariant, UV_VERSION, WORKER_PYTHON, WORKER_VERSION};

#[derive(Debug, Clone, Serialize)]
pub struct BootstrapProgress {
    /// "downloading-uv" | "extracting-uv" | "installing-python" |
    /// "creating-venv" | "installing-deps" | "finalizing"
    pub stage: &'static str,
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
}

/// What the venv was built from. Any mismatch with the current pins or the
/// shipped requirements file invalidates the venv.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct VenvStamp {
    pub uv_version: String,
    pub python: String,
    pub variant: WorkerVariant,
    pub lock_sha256: String,
    pub worker_version: String,
}

// ── paths ────────────────────────────────────────────────────────────────

pub fn uv_exe(root: &Path) -> PathBuf {
    let name = if cfg!(windows) { "uv.exe" } else { "uv" };
    root.join("uv").join(UV_VERSION).join(name)
}

pub fn python_install_dir(root: &Path) -> PathBuf {
    root.join("python")
}

pub fn uv_cache_dir(root: &Path) -> PathBuf {
    root.join("uv-cache")
}

pub fn venv_dir(root: &Path) -> PathBuf {
    root.join("venv")
}

pub fn venv_python(root: &Path) -> PathBuf {
    if cfg!(windows) {
        venv_dir(root).join("Scripts").join("python.exe")
    } else {
        venv_dir(root).join("bin").join("python")
    }
}

pub fn stamp_path(root: &Path) -> PathBuf {
    root.join("venv.stamp")
}

pub fn uv_download_url() -> String {
    let asset = if cfg!(windows) {
        "uv-x86_64-pc-windows-msvc.zip"
    } else {
        "uv-x86_64-unknown-linux-gnu.tar.gz"
    };
    format!("https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/{asset}")
}

// ── stamp / readiness ────────────────────────────────────────────────────

pub fn file_sha256(path: &Path) -> Result<String, AppError> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

pub fn read_stamp(root: &Path) -> Option<VenvStamp> {
    let raw = std::fs::read_to_string(stamp_path(root)).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write_stamp(root: &Path, stamp: &VenvStamp) -> Result<(), AppError> {
    std::fs::write(stamp_path(root), serde_json::to_string_pretty(stamp)?)?;
    Ok(())
}

/// Build the stamp the current pins + requirements file would produce.
pub fn expected_stamp(
    variant: WorkerVariant,
    worker_src: &Path,
) -> Result<VenvStamp, AppError> {
    let lock = worker_src.join(variant.requirements_file());
    Ok(VenvStamp {
        uv_version: UV_VERSION.to_string(),
        python: WORKER_PYTHON.to_string(),
        variant,
        lock_sha256: file_sha256(&lock)?,
        worker_version: WORKER_VERSION.to_string(),
    })
}

/// If a valid venv exists under `root`, return its variant.
pub fn ready_variant_at(root: &Path, worker_src: &Path) -> Option<WorkerVariant> {
    let stamp = read_stamp(root)?;
    if !venv_python(root).exists() {
        return None;
    }
    let expected = expected_stamp(stamp.variant, worker_src).ok()?;
    (stamp == expected).then_some(stamp.variant)
}

/// Convenience wrapper over the real data dir.
pub fn ready_variant(worker_src: &Path) -> Option<WorkerVariant> {
    ready_variant_at(&worker_dir(), worker_src)
}

// ── install ──────────────────────────────────────────────────────────────

/// Disk needed: CUDA torch + deps unpack to ~8 GiB plus wheel cache; CPU is
/// far smaller. Conservative prechecks, matching the engine installer's style.
fn needed_bytes(variant: WorkerVariant) -> u64 {
    match variant {
        WorkerVariant::Cuda => 14 * 1024 * 1024 * 1024,
        WorkerVariant::Cpu => 5 * 1024 * 1024 * 1024,
    }
}

/// Fetch uv (if missing), provision CPython, build the venv from the
/// variant's requirements file, and write the stamp. On failure the venv +
/// stamp are removed; uv, the python install, and the wheel cache are kept
/// (they're reusable and expensive).
pub async fn install<F>(
    http: &Client,
    variant: WorkerVariant,
    worker_src: &Path,
    mut on_progress: F,
) -> Result<(), AppError>
where
    F: FnMut(BootstrapProgress) + Send + 'static,
{
    let root = worker_dir();
    std::fs::create_dir_all(&root)?;

    let free = fs2::available_space(&root)
        .map_err(|e| AppError::Other(format!("disk-space probe: {}", e)))?;
    if free < needed_bytes(variant) {
        return Err(AppError::EngineDiskSpace {
            needed_mb: needed_bytes(variant) / 1024 / 1024,
            available_mb: free / 1024 / 1024,
        });
    }

    let result: Result<(), AppError> = async {
        ensure_uv(http, &root, &mut on_progress).await?;

        // 1. Provision the pinned CPython (idempotent under UV_PYTHON_INSTALL_DIR).
        on_progress(BootstrapProgress {
            stage: "installing-python",
            bytes_downloaded: 0,
            bytes_total: 0,
        });
        run_uv(&root, &["python", "install", WORKER_PYTHON]).await?;

        // 2. Fresh venv — a stale one is never patched in place.
        on_progress(BootstrapProgress {
            stage: "creating-venv",
            bytes_downloaded: 0,
            bytes_total: 0,
        });
        let venv = venv_dir(&root);
        if venv.exists() {
            std::fs::remove_dir_all(&venv)?;
        }
        let _ = std::fs::remove_file(stamp_path(&root));
        run_uv(
            &root,
            &["venv", venv.to_str().unwrap_or_default(), "--python", WORKER_PYTHON],
        )
        .await?;

        // 3. Sync the locked requirements into it.
        on_progress(BootstrapProgress {
            stage: "installing-deps",
            bytes_downloaded: 0,
            bytes_total: 0,
        });
        let req = worker_src.join(variant.requirements_file());
        if !req.exists() {
            return Err(AppError::Other(format!(
                "requirements file missing: {}",
                req.display()
            )));
        }
        let py = venv_python(&root);
        run_uv(
            &root,
            &[
                "pip",
                "install",
                "-r",
                req.to_str().unwrap_or_default(),
                "--python",
                py.to_str().unwrap_or_default(),
                // The pytorch extra index ships stale copies of common packages
                // (e.g. requests 2.28.1); uv's default first-match strategy would
                // pin to those and fail resolution. Both indexes are trusted.
                "--index-strategy",
                "unsafe-best-match",
            ],
        )
        .await?;

        // 4. Stamp it.
        on_progress(BootstrapProgress {
            stage: "finalizing",
            bytes_downloaded: 0,
            bytes_total: 0,
        });
        write_stamp(&root, &expected_stamp(variant, worker_src)?)?;
        Ok(())
    }
    .await;

    if result.is_err() {
        let _ = std::fs::remove_dir_all(venv_dir(&root));
        let _ = std::fs::remove_file(stamp_path(&root));
    }
    result
}

/// Download + extract the pinned uv release if `uv_exe` is absent.
pub async fn ensure_uv<F>(http: &Client, root: &Path, on_progress: &mut F) -> Result<(), AppError>
where
    F: FnMut(BootstrapProgress) + Send,
{
    let exe = uv_exe(root);
    if exe.exists() {
        return Ok(());
    }
    let dest = exe.parent().expect("uv exe has parent").to_path_buf();
    std::fs::create_dir_all(&dest)?;

    let url = uv_download_url();
    let resp = http.get(&url).send().await?.error_for_status()?;
    let total = resp.content_length().unwrap_or(0);
    on_progress(BootstrapProgress {
        stage: "downloading-uv",
        bytes_downloaded: 0,
        bytes_total: total,
    });

    let tmp_zip = dest.join(".uv-download.zip");
    let mut file = tokio::fs::File::create(&tmp_zip).await?;
    let mut stream = resp.bytes_stream();
    let mut got: u64 = 0;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        got += chunk.len() as u64;
        file.write_all(&chunk).await?;
        on_progress(BootstrapProgress {
            stage: "downloading-uv",
            bytes_downloaded: got,
            bytes_total: total,
        });
    }
    file.flush().await?;
    drop(file);

    on_progress(BootstrapProgress {
        stage: "extracting-uv",
        bytes_downloaded: got,
        bytes_total: total,
    });
    let zip_path = tmp_zip.clone();
    let extract_to = dest.clone();
    tokio::task::spawn_blocking(move || -> Result<(), AppError> {
        let file = std::fs::File::open(&zip_path)?;
        let mut zip = zip::ZipArchive::new(file)
            .map_err(|e| AppError::EngineExtractFailed(format!("uv zip open: {}", e)))?;
        zip.extract(&extract_to)
            .map_err(|e| AppError::EngineExtractFailed(format!("uv zip extract: {}", e)))?;
        Ok(())
    })
    .await
    .map_err(|e| AppError::Other(format!("join: {}", e)))??;
    let _ = std::fs::remove_file(&tmp_zip);

    if !exe.exists() {
        // Some archives nest the binary one level down; flatten it.
        let name = exe.file_name().expect("uv exe name");
        for entry in std::fs::read_dir(&dest)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                let nested = entry.path().join(name);
                if nested.exists() {
                    std::fs::rename(&nested, &exe)?;
                    break;
                }
            }
        }
    }
    if !exe.exists() {
        return Err(AppError::Other(
            "uv.exe not found after extracting the release archive".into(),
        ));
    }
    Ok(())
}

/// Run uv with the worker's pinned environment; on a non-zero exit, surface
/// the tail of its output in the error.
pub async fn run_uv(root: &Path, args: &[&str]) -> Result<(), AppError> {
    let exe = uv_exe(root);
    let mut cmd = tokio::process::Command::new(&exe);
    cmd.args(args)
        .env("UV_PYTHON_INSTALL_DIR", python_install_dir(root))
        .env("UV_CACHE_DIR", uv_cache_dir(root))
        .env("UV_NO_PROGRESS", "1")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    #[cfg(windows)]
    {
        #[allow(unused_imports)]
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let out = cmd
        .output()
        .await
        .map_err(|e| AppError::Other(format!("uv spawn ({}): {}", args.join(" "), e)))?;

    let stderr = String::from_utf8_lossy(&out.stderr);
    for line in stderr.lines() {
        tracing::info!(target: "worker.uv", "{}", line);
    }
    if !out.status.success() {
        let tail: String = stderr
            .lines()
            .rev()
            .take(15)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n");
        return Err(AppError::Other(format!(
            "uv {} failed ({}): {}",
            args.join(" "),
            out.status,
            tail
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uv_url_pins_version() {
        let url = uv_download_url();
        assert!(url.contains(UV_VERSION));
        #[cfg(windows)]
        assert!(url.ends_with("uv-x86_64-pc-windows-msvc.zip"));
    }

    #[test]
    fn stamp_round_trip_and_mismatch() {
        let tmp = std::env::temp_dir().join(format!("oh-worker-test-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();

        // Fake worker-src with a requirements file.
        let src = tmp.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("requirements.cpu.txt"), "torch==1.0\n").unwrap();

        let stamp = expected_stamp(WorkerVariant::Cpu, &src).unwrap();
        write_stamp(&tmp, &stamp).unwrap();
        assert_eq!(read_stamp(&tmp), Some(stamp.clone()));

        // No venv python yet -> not ready even with a matching stamp.
        assert_eq!(ready_variant_at(&tmp, &src), None);

        // Materialize a fake venv python -> ready.
        let py = venv_python(&tmp);
        std::fs::create_dir_all(py.parent().unwrap()).unwrap();
        std::fs::write(&py, "").unwrap();
        assert_eq!(ready_variant_at(&tmp, &src), Some(WorkerVariant::Cpu));

        // Edit the requirements file -> hash mismatch -> stale.
        std::fs::write(src.join("requirements.cpu.txt"), "torch==2.0\n").unwrap();
        assert_eq!(ready_variant_at(&tmp, &src), None);

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn needed_bytes_cuda_exceeds_cpu() {
        assert!(needed_bytes(WorkerVariant::Cuda) > needed_bytes(WorkerVariant::Cpu));
    }
}
