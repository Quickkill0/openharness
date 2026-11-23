//! Generic pin-and-fetch runtime provisioning.
//!
//! A single, data-driven installer for **any** engine that needs binaries
//! placed on disk. The machinery here is a near-verbatim generalization of
//! `engine/installer.rs` (the canonical llama.cpp pin-and-fetch) and
//! `worker/bootstrap.rs` (stamp file + SHA verify) — instead of hard-coding
//! one engine's URLs and layout, every engine becomes a [`RuntimeManifest`]:
//! a list of download [`ManifestAsset`]s (each a ZIP to extract or a loose
//! file to drop in place), the [`required_files`](RuntimeManifest::required_files)
//! the install must end up with, a target [`install_dir`](RuntimeManifest::install_dir),
//! and a disk-space budget.
//!
//! Adding an engine = writing one `fn …_manifest() -> RuntimeManifest`
//! (see `tts/manifest.rs` for the first one), not more installer code.
//!
//! ## Properties (lifted from the engine installer)
//! - Disk-space precheck via `fs2` against the manifest's `needed_bytes`.
//! - Streaming download to a temp dir with `InstallProgress` events.
//! - Optional SHA-256 verify per asset (when `sha256` is `Some`).
//! - ZIP assets extracted via `spawn_blocking`; `RawFile` assets moved into
//!   `install_dir`.
//! - Post-install verification that every `required_file` exists (flattening
//!   one nested directory level, like `locate_llama_server`).
//! - Clean rollback: the whole `install_dir` is removed on any failure.

use std::path::{Path, PathBuf};

use futures::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::io::AsyncWriteExt;

use crate::error::AppError;

/// How a downloaded asset is materialized into the install dir.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssetKind {
    /// A ZIP archive whose contents are extracted into `install_dir`.
    Zip,
    /// A single loose file copied directly into `install_dir` (e.g. a DLL or
    /// a GGUF that isn't packaged in an archive).
    RawFile,
}

/// One downloadable artifact in a manifest.
#[derive(Debug, Clone)]
pub struct ManifestAsset {
    /// Fully-resolved download URL.
    pub url: String,
    /// File name to save the download as (and, for `RawFile`, its name in
    /// `install_dir`).
    pub filename: String,
    /// Expected SHA-256 (lowercase hex). When `Some`, the download is
    /// rejected on mismatch. `None` falls back to TLS as the integrity story
    /// (matching the llama.cpp engine ZIPs, which are TLS-only today).
    pub sha256: Option<String>,
    pub kind: AssetKind,
}

/// A complete, self-contained description of one engine's binary install.
#[derive(Debug, Clone)]
pub struct RuntimeManifest {
    /// Stable engine id (e.g. `"qwentts"`). Recorded in the provision stamp.
    pub id: &'static str,
    /// Version/build pin. A change re-provisions (the stamp won't match).
    pub version: &'static str,
    /// Absolute directory the engine installs into.
    pub install_dir: PathBuf,
    /// Assets to download (in order).
    pub assets: Vec<ManifestAsset>,
    /// Files (relative to `install_dir`) that MUST exist after a successful
    /// install. Drives both post-install verification and `is_provisioned`.
    pub required_files: Vec<&'static str>,
    /// Disk-space budget for the precheck (download + extraction peak).
    pub needed_bytes: u64,
}

/// Streaming install progress, mirroring `engine::installer::InstallProgress`.
#[derive(Debug, Clone, Serialize)]
pub struct InstallProgress {
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    /// Which asset is in flight (its `filename`).
    pub asset: String,
    /// Position in the asset list (`1`-based) and the total count, so a
    /// caller can render "downloading 2 / 7".
    pub asset_index: usize,
    pub asset_count: usize,
    /// "downloading" | "verifying" | "extracting" | "installing"
    pub stage: &'static str,
}

/// What an install was built from. A version/id mismatch re-provisions
/// (mirrors `worker/bootstrap.rs`'s `venv.stamp`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvisionStamp {
    pub id: String,
    pub version: String,
}

fn stamp_path(install_dir: &Path) -> PathBuf {
    install_dir.join("provision.stamp")
}

fn read_stamp(install_dir: &Path) -> Option<ProvisionStamp> {
    let raw = std::fs::read_to_string(stamp_path(install_dir)).ok()?;
    serde_json::from_str(&raw).ok()
}

fn write_stamp(manifest: &RuntimeManifest) -> Result<(), AppError> {
    let stamp = ProvisionStamp {
        id: manifest.id.to_string(),
        version: manifest.version.to_string(),
    };
    std::fs::write(
        stamp_path(&manifest.install_dir),
        serde_json::to_string_pretty(&stamp)?,
    )?;
    Ok(())
}

/// SHA-256 of a file as lowercase hex (lifted from `worker::bootstrap`).
pub fn file_sha256(path: &Path) -> Result<String, AppError> {
    let bytes = std::fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(&bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

/// True when every `required_file` is present in `install_dir`. The stamp is
/// advisory only — files are the source of truth — but a present-yet-stale
/// stamp (version bump) is treated as not-provisioned so a fresh install runs.
pub fn is_provisioned(manifest: &RuntimeManifest) -> bool {
    let all_present = manifest
        .required_files
        .iter()
        .all(|f| manifest.install_dir.join(f).exists());
    if !all_present {
        return false;
    }
    // A stamp from an older version means the on-disk binaries don't match
    // the current pin — re-provision. A *missing* stamp is tolerated: the
    // hand-placed installs that predate this system have no stamp but their
    // required_files are real, so we honour them.
    match read_stamp(&manifest.install_dir) {
        Some(s) => s.id == manifest.id && s.version == manifest.version,
        None => true,
    }
}

/// Download + materialize every asset, verify the required files landed, and
/// write the provision stamp. Removes `install_dir` on any failure (clean
/// rollback). Near-verbatim port of `engine::installer::install`,
/// parameterized by the manifest.
pub async fn install<F>(
    http: &Client,
    manifest: &RuntimeManifest,
    mut on_progress: F,
) -> Result<(), AppError>
where
    F: FnMut(InstallProgress) + Send + 'static,
{
    let dest = manifest.install_dir.clone();

    // Disk-space precheck against the parent (the dir may not exist yet).
    let parent = dest.parent().unwrap_or(Path::new(".")).to_path_buf();
    std::fs::create_dir_all(&parent)?;
    let free = fs2::available_space(&parent)
        .map_err(|e| AppError::Other(format!("disk-space probe: {}", e)))?;
    if free < manifest.needed_bytes {
        return Err(AppError::EngineDiskSpace {
            needed_mb: manifest.needed_bytes / 1024 / 1024,
            available_mb: free / 1024 / 1024,
        });
    }

    std::fs::create_dir_all(&dest)?;

    let asset_count = manifest.assets.len();
    let result: Result<(), AppError> = async {
        let tmp_dir = dest.join(".tmp-provision");
        std::fs::create_dir_all(&tmp_dir)?;

        for (i, asset) in manifest.assets.iter().enumerate() {
            let asset_index = i + 1;
            let tmp_file = tmp_dir.join(&asset.filename);

            // ── download ──────────────────────────────────────────────────
            let resp = http
                .get(&asset.url)
                .send()
                .await
                .map_err(|e| classify_fetch_error(asset, e))?
                .error_for_status()
                .map_err(|e| classify_fetch_error(asset, e))?;
            let total = resp.content_length().unwrap_or(0);
            on_progress(InstallProgress {
                bytes_downloaded: 0,
                bytes_total: total,
                asset: asset.filename.clone(),
                asset_index,
                asset_count,
                stage: "downloading",
            });

            let mut file = tokio::fs::File::create(&tmp_file).await?;
            let mut stream = resp.bytes_stream();
            let mut got: u64 = 0;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk.map_err(|e| classify_fetch_error(asset, e))?;
                got += chunk.len() as u64;
                file.write_all(&chunk).await?;
                on_progress(InstallProgress {
                    bytes_downloaded: got,
                    bytes_total: total,
                    asset: asset.filename.clone(),
                    asset_index,
                    asset_count,
                    stage: "downloading",
                });
            }
            file.flush().await?;
            drop(file);

            // ── verify (optional) ─────────────────────────────────────────
            if let Some(expected) = &asset.sha256 {
                on_progress(InstallProgress {
                    bytes_downloaded: got,
                    bytes_total: total,
                    asset: asset.filename.clone(),
                    asset_index,
                    asset_count,
                    stage: "verifying",
                });
                let actual = file_sha256(&tmp_file)?;
                if !actual.eq_ignore_ascii_case(expected) {
                    return Err(AppError::DownloadIntegrityMismatch);
                }
            }

            // ── materialize ───────────────────────────────────────────────
            match asset.kind {
                AssetKind::Zip => {
                    on_progress(InstallProgress {
                        bytes_downloaded: got,
                        bytes_total: total,
                        asset: asset.filename.clone(),
                        asset_index,
                        asset_count,
                        stage: "extracting",
                    });
                    let zip_path = tmp_file.clone();
                    let extract_to = dest.clone();
                    tokio::task::spawn_blocking(move || -> Result<(), AppError> {
                        let file = std::fs::File::open(&zip_path)?;
                        let mut zip = zip::ZipArchive::new(file).map_err(|e| {
                            AppError::EngineExtractFailed(format!("zip open: {}", e))
                        })?;
                        zip.extract(&extract_to).map_err(|e| {
                            AppError::EngineExtractFailed(format!("zip extract: {}", e))
                        })?;
                        Ok(())
                    })
                    .await
                    .map_err(|e| AppError::Other(format!("join: {}", e)))??;
                }
                AssetKind::RawFile => {
                    on_progress(InstallProgress {
                        bytes_downloaded: got,
                        bytes_total: total,
                        asset: asset.filename.clone(),
                        asset_index,
                        asset_count,
                        stage: "installing",
                    });
                    let target = dest.join(&asset.filename);
                    // rename within the same dir tree, falling back to copy
                    // across filesystems.
                    if std::fs::rename(&tmp_file, &target).is_err() {
                        std::fs::copy(&tmp_file, &target)?;
                    }
                }
            }

            let _ = std::fs::remove_file(&tmp_file);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);

        verify_required_files(manifest)?;
        write_stamp(manifest)?;
        Ok(())
    }
    .await;

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&dest);
    }
    result
}

/// Some archives nest their payload one directory deep. If a required file
/// isn't at the top level, look one level down and flatten the matching
/// entry up — mirroring `engine::installer::locate_llama_server`.
fn verify_required_files(manifest: &RuntimeManifest) -> Result<(), AppError> {
    for rel in &manifest.required_files {
        let direct = manifest.install_dir.join(rel);
        if direct.exists() {
            continue;
        }
        // Walk one level deep for a file of the same name and flatten it up.
        let name = Path::new(rel).file_name();
        let mut found = false;
        if let (Some(name), Ok(read)) = (name, std::fs::read_dir(&manifest.install_dir)) {
            for entry in read.flatten() {
                if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    let nested = entry.path().join(name);
                    if nested.exists() {
                        if let Some(parent) = direct.parent() {
                            std::fs::create_dir_all(parent)?;
                        }
                        std::fs::rename(&nested, &direct)?;
                        found = true;
                        break;
                    }
                }
            }
        }
        if !found {
            return Err(AppError::EngineExtractFailed(format!(
                "required file '{}' not found after installing {}",
                rel, manifest.id
            )));
        }
    }
    Ok(())
}

/// Map a download failure to an honest, specific error. The driving case:
/// the qwentts binaries aren't hosted yet, so the placeholder URL 404s /
/// fails DNS — surface that as a clear "not hosted" message rather than a
/// raw reqwest error.
fn classify_fetch_error(asset: &ManifestAsset, e: reqwest::Error) -> AppError {
    let host_problem = e.is_connect()
        || e.status() == Some(reqwest::StatusCode::NOT_FOUND)
        || e.status() == Some(reqwest::StatusCode::FORBIDDEN);
    if host_problem {
        AppError::Other(format!(
            "could not fetch '{}' from {} — the engine binaries may not be hosted yet \
             (see CLAUDE.md → \"Qwen3-TTS engine provisioning\"). Underlying error: {}",
            asset.filename, asset.url, e
        ))
    } else {
        AppError::Http(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn unique_dir(tag: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "oh-provision-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    /// Build a tiny ZIP containing `entries` (name → bytes) at `path`.
    fn make_zip(path: &Path, entries: &[(&str, &[u8])]) {
        let file = std::fs::File::create(path).unwrap();
        let mut zip = zip::ZipWriter::new(file);
        let opts: zip::write::FileOptions<()> = zip::write::FileOptions::default();
        for (name, bytes) in entries {
            zip.start_file(*name, opts).unwrap();
            zip.write_all(bytes).unwrap();
        }
        zip.finish().unwrap();
    }

    /// A local file:// URL for a path (so `install` hits the disk, never the
    /// network). reqwest with the `file` story isn't enabled here, so the
    /// tests use a tiny local HTTP server instead — see `serve_dir`.
    struct LocalServer {
        addr: std::net::SocketAddr,
        _handle: std::thread::JoinHandle<()>,
    }

    /// Minimal blocking HTTP/1.1 file server over a temp dir. Serves GET
    /// `/<name>` from `root/<name>`; 404s anything missing. Single-threaded,
    /// one request per accept — enough for these tests, never the network.
    fn serve_dir(root: PathBuf) -> LocalServer {
        use std::io::{Read, Write};
        use std::net::TcpListener;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = [0u8; 2048];
                let n = stream.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    continue;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                // "GET /name HTTP/1.1"
                let path = req
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("/")
                    .trim_start_matches('/')
                    .to_string();
                if path == "__shutdown" {
                    break;
                }
                let file = root.join(&path);
                match std::fs::read(&file) {
                    Ok(bytes) => {
                        let header = format!(
                            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            bytes.len()
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(&bytes);
                    }
                    Err(_) => {
                        let body = b"not found";
                        let header = format!(
                            "HTTP/1.1 404 Not Found\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                            body.len()
                        );
                        let _ = stream.write_all(header.as_bytes());
                        let _ = stream.write_all(body);
                    }
                }
                let _ = stream.flush();
            }
        });
        LocalServer { addr, _handle: handle }
    }

    impl LocalServer {
        fn url(&self, name: &str) -> String {
            format!("http://{}/{}", self.addr, name)
        }
        fn shutdown(&self) {
            // Best-effort: open one connection that the loop reads as
            // `__shutdown` so the thread exits cleanly.
            use std::io::Write;
            if let Ok(mut s) = std::net::TcpStream::connect(self.addr) {
                let _ = s.write_all(b"GET /__shutdown HTTP/1.1\r\n\r\n");
            }
        }
    }

    #[tokio::test]
    async fn install_zip_and_rawfile_end_to_end() {
        let src = unique_dir("src");
        // A ZIP that carries the binary + a DLL, plus a loose codec file.
        make_zip(
            &src.join("engine.zip"),
            &[("tool.exe", b"BINARY"), ("lib.dll", b"DLL")],
        );
        std::fs::write(src.join("codec.gguf"), b"CODEC-BYTES").unwrap();

        let codec_sha = file_sha256(&src.join("codec.gguf")).unwrap();
        let server = serve_dir(src.clone());

        let dest = unique_dir("dest-ok").join("install");
        let manifest = RuntimeManifest {
            id: "test-engine",
            version: "1.0.0",
            install_dir: dest.clone(),
            assets: vec![
                ManifestAsset {
                    url: server.url("engine.zip"),
                    filename: "engine.zip".into(),
                    sha256: None,
                    kind: AssetKind::Zip,
                },
                ManifestAsset {
                    url: server.url("codec.gguf"),
                    filename: "codec.gguf".into(),
                    sha256: Some(codec_sha),
                    kind: AssetKind::RawFile,
                },
            ],
            required_files: vec!["tool.exe", "lib.dll", "codec.gguf"],
            needed_bytes: 1024,
        };

        assert!(!is_provisioned(&manifest));
        install(&server_http(), &manifest, |_| {}).await.unwrap();

        assert!(dest.join("tool.exe").exists());
        assert!(dest.join("lib.dll").exists());
        assert!(dest.join("codec.gguf").exists());
        assert_eq!(std::fs::read(dest.join("codec.gguf")).unwrap(), b"CODEC-BYTES");
        assert!(is_provisioned(&manifest));
        // Stamp written.
        assert_eq!(
            read_stamp(&dest).unwrap(),
            ProvisionStamp { id: "test-engine".into(), version: "1.0.0".into() }
        );

        server.shutdown();
        let _ = std::fs::remove_dir_all(src);
    }

    #[tokio::test]
    async fn sha_mismatch_fails_and_rolls_back() {
        let src = unique_dir("src-sha");
        std::fs::write(src.join("codec.gguf"), b"REAL").unwrap();
        let server = serve_dir(src.clone());

        let dest = unique_dir("dest-sha").join("install");
        let manifest = RuntimeManifest {
            id: "test-engine",
            version: "1.0.0",
            install_dir: dest.clone(),
            assets: vec![ManifestAsset {
                url: server.url("codec.gguf"),
                filename: "codec.gguf".into(),
                // Deliberately wrong digest.
                sha256: Some("deadbeef".repeat(8)),
                kind: AssetKind::RawFile,
            }],
            required_files: vec!["codec.gguf"],
            needed_bytes: 1024,
        };

        let err = install(&server_http(), &manifest, |_| {}).await.unwrap_err();
        assert!(matches!(err, AppError::DownloadIntegrityMismatch));
        // Rollback: install_dir removed entirely.
        assert!(!dest.exists());

        server.shutdown();
        let _ = std::fs::remove_dir_all(src);
    }

    #[tokio::test]
    async fn missing_required_file_fails_and_rolls_back() {
        let src = unique_dir("src-missing");
        // ZIP lacks the required `tool.exe`.
        make_zip(&src.join("engine.zip"), &[("readme.txt", b"hi")]);
        let server = serve_dir(src.clone());

        let dest = unique_dir("dest-missing").join("install");
        let manifest = RuntimeManifest {
            id: "test-engine",
            version: "1.0.0",
            install_dir: dest.clone(),
            assets: vec![ManifestAsset {
                url: server.url("engine.zip"),
                filename: "engine.zip".into(),
                sha256: None,
                kind: AssetKind::Zip,
            }],
            required_files: vec!["tool.exe"],
            needed_bytes: 1024,
        };

        let err = install(&server_http(), &manifest, |_| {}).await.unwrap_err();
        assert!(matches!(err, AppError::EngineExtractFailed(_)));
        assert!(!dest.exists());

        server.shutdown();
        let _ = std::fs::remove_dir_all(src);
    }

    #[tokio::test]
    async fn nested_zip_payload_is_flattened() {
        let src = unique_dir("src-nested");
        // The binary is one directory deep inside the archive.
        make_zip(&src.join("engine.zip"), &[("build/tool.exe", b"X")]);
        let server = serve_dir(src.clone());

        let dest = unique_dir("dest-nested").join("install");
        let manifest = RuntimeManifest {
            id: "test-engine",
            version: "1.0.0",
            install_dir: dest.clone(),
            assets: vec![ManifestAsset {
                url: server.url("engine.zip"),
                filename: "engine.zip".into(),
                sha256: None,
                kind: AssetKind::Zip,
            }],
            required_files: vec!["tool.exe"],
            needed_bytes: 1024,
        };

        install(&server_http(), &manifest, |_| {}).await.unwrap();
        assert!(dest.join("tool.exe").exists());

        server.shutdown();
        let _ = std::fs::remove_dir_all(src);
    }

    #[tokio::test]
    async fn unreachable_host_gives_honest_error() {
        // Port 1 is reserved/unbindable — connect fails fast.
        let dest = unique_dir("dest-unreach").join("install");
        let manifest = RuntimeManifest {
            id: "test-engine",
            version: "1.0.0",
            install_dir: dest.clone(),
            assets: vec![ManifestAsset {
                url: "http://127.0.0.1:1/nope.zip".into(),
                filename: "nope.zip".into(),
                sha256: None,
                kind: AssetKind::Zip,
            }],
            required_files: vec!["nope"],
            needed_bytes: 1024,
        };
        let err = install(&server_http(), &manifest, |_| {}).await.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("not be hosted yet") || msg.contains("not hosted"),
            "expected an honest hosting message, got: {msg}"
        );
        assert!(!dest.exists());
    }

    #[test]
    fn is_provisioned_honours_present_files_without_stamp() {
        let dest = unique_dir("prov-check").join("install");
        std::fs::create_dir_all(&dest).unwrap();
        let manifest = RuntimeManifest {
            id: "x",
            version: "1.0.0",
            install_dir: dest.clone(),
            assets: vec![],
            required_files: vec!["a.bin", "b.bin"],
            needed_bytes: 1,
        };
        assert!(!is_provisioned(&manifest));
        std::fs::write(dest.join("a.bin"), b"").unwrap();
        assert!(!is_provisioned(&manifest), "one file still missing");
        std::fs::write(dest.join("b.bin"), b"").unwrap();
        // No stamp, but all required files present → honoured (hand-placed).
        assert!(is_provisioned(&manifest));

        // A stale stamp (different version) forces re-provision.
        write_stamp(&RuntimeManifest { version: "0.9.0", ..manifest.clone() }).unwrap();
        assert!(!is_provisioned(&manifest));

        let _ = std::fs::remove_dir_all(dest);
    }

    fn server_http() -> Client {
        Client::builder()
            .build()
            .expect("reqwest client")
    }
}
