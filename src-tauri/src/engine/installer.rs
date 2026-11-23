use std::path::{Path, PathBuf};

use futures::StreamExt;
use reqwest::Client;
use serde::Serialize;
use tokio::io::AsyncWriteExt;

use crate::config::engine_dir;
use crate::engine::{EngineVariant, ENGINE_BUILD};
use crate::error::AppError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    EngineZip,
    CudartZip,
}

pub struct InstallPlan {
    pub url: String,
    pub kind: AssetKind,
    pub filename: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallProgress {
    pub bytes_downloaded: u64,
    pub bytes_total: u64,
    pub asset_kind: &'static str,
    pub stage: &'static str, // "downloading" | "extracting"
}

pub fn plan(variant: EngineVariant) -> Vec<InstallPlan> {
    let build = ENGINE_BUILD;
    match variant {
        EngineVariant::CudaV12_4 => vec![
            InstallPlan {
                url: format!(
                    "https://github.com/ggml-org/llama.cpp/releases/download/{build}/llama-{build}-bin-win-cuda-12.4-x64.zip"
                ),
                kind: AssetKind::EngineZip,
                filename: format!("llama-{build}-bin-win-cuda-12.4-x64.zip"),
            },
            InstallPlan {
                url: format!(
                    "https://github.com/ggml-org/llama.cpp/releases/download/{build}/cudart-llama-bin-win-cuda-12.4-x64.zip"
                ),
                kind: AssetKind::CudartZip,
                filename: format!("cudart-llama-bin-win-cuda-12.4-x64.zip"),
            },
        ],
        EngineVariant::Cpu => vec![InstallPlan {
            url: format!(
                "https://github.com/ggml-org/llama.cpp/releases/download/{build}/llama-{build}-bin-win-cpu-x64.zip"
            ),
            kind: AssetKind::EngineZip,
            filename: format!("llama-{build}-bin-win-cpu-x64.zip"),
        }],
    }
}

pub fn install_dir(variant: EngineVariant) -> PathBuf {
    engine_dir().join(format!("{}-{}", ENGINE_BUILD, variant.as_asset_slug()))
}

/// Download all plans, verify total free space first, extract into `install_dir`,
/// return path to `llama-server.exe`.
pub async fn install<F>(
    http: &Client,
    variant: EngineVariant,
    mut on_progress: F,
) -> Result<PathBuf, AppError>
where
    F: FnMut(InstallProgress) + Send + 'static,
{
    let plans = plan(variant);
    let dest = install_dir(variant);

    // Disk-space precheck: 3 GiB covers download (~1.3 GiB) + extraction (~2.6 GiB peak).
    let parent = dest.parent().unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    let free = fs2::available_space(parent)
        .map_err(|e| AppError::Other(format!("disk-space probe: {}", e)))?;
    const NEEDED: u64 = 3_000 * 1024 * 1024;
    if free < NEEDED {
        return Err(AppError::EngineDiskSpace {
            needed_mb: NEEDED / 1024 / 1024,
            available_mb: free / 1024 / 1024,
        });
    }

    std::fs::create_dir_all(&dest)?;

    // Run the actual install; clean up dest on any failure.
    let result: Result<PathBuf, AppError> = async {
        let tmp_dir = parent.join(".tmp-install");
        std::fs::create_dir_all(&tmp_dir)?;

        for p in &plans {
            let tmp_zip = tmp_dir.join(&p.filename);
            let resp = http.get(&p.url).send().await?.error_for_status()?;
            let total = resp.content_length().unwrap_or(0);
            let kind_str: &'static str = match p.kind {
                AssetKind::EngineZip => "engine",
                AssetKind::CudartZip => "cudart",
            };
            on_progress(InstallProgress {
                bytes_downloaded: 0,
                bytes_total: total,
                asset_kind: kind_str,
                stage: "downloading",
            });

            let mut file = tokio::fs::File::create(&tmp_zip).await?;
            let mut stream = resp.bytes_stream();
            let mut got: u64 = 0;
            while let Some(chunk) = stream.next().await {
                let chunk = chunk?;
                got += chunk.len() as u64;
                file.write_all(&chunk).await?;
                on_progress(InstallProgress {
                    bytes_downloaded: got,
                    bytes_total: total,
                    asset_kind: kind_str,
                    stage: "downloading",
                });
            }
            file.flush().await?;
            drop(file);

            // Extract via spawn_blocking
            let zip_path = tmp_zip.clone();
            let extract_to = dest.clone();
            on_progress(InstallProgress {
                bytes_downloaded: got,
                bytes_total: total,
                asset_kind: kind_str,
                stage: "extracting",
            });
            tokio::task::spawn_blocking(move || -> Result<(), AppError> {
                let file = std::fs::File::open(&zip_path)?;
                let mut zip = zip::ZipArchive::new(file)
                    .map_err(|e| AppError::EngineExtractFailed(format!("zip open: {}", e)))?;
                zip.extract(&extract_to)
                    .map_err(|e| AppError::EngineExtractFailed(format!("zip extract: {}", e)))?;
                Ok(())
            })
            .await
            .map_err(|e| AppError::Other(format!("join: {}", e)))??;

            let _ = std::fs::remove_file(&tmp_zip);
        }

        let _ = std::fs::remove_dir_all(&tmp_dir);

        let server_path = locate_llama_server(&dest)?;
        Ok(server_path)
    }
    .await;

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&dest);
    }
    result
}

fn locate_llama_server(root: &Path) -> Result<PathBuf, AppError> {
    // Try root first
    let root_candidate = root.join("llama-server.exe");
    if root_candidate.exists() {
        return Ok(root_candidate);
    }
    // Walk one level deep
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            let nested = entry.path().join("llama-server.exe");
            if nested.exists() {
                return Ok(nested);
            }
            // Two levels deep (e.g. build/bin/)
            for sub in std::fs::read_dir(entry.path())? {
                let sub = sub?;
                if sub.file_type()?.is_dir() {
                    let deeper = sub.path().join("llama-server.exe");
                    if deeper.exists() {
                        return Ok(deeper);
                    }
                }
            }
        }
    }
    Err(AppError::EngineExtractFailed(
        "llama-server.exe not found in extracted archive".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_cuda_returns_two_assets() {
        let p = plan(EngineVariant::CudaV12_4);
        assert_eq!(p.len(), 2);
        assert!(p[0].url.contains("llama-b12182-bin-win-cuda-12.4-x64.zip"));
        assert_eq!(p[0].kind, AssetKind::EngineZip);
        assert!(p[1].url.contains("cudart-llama-bin-win-cuda-12.4-x64.zip"));
        assert_eq!(p[1].kind, AssetKind::CudartZip);
    }

    #[test]
    fn plan_cpu_returns_one_asset() {
        let p = plan(EngineVariant::Cpu);
        assert_eq!(p.len(), 1);
        assert!(p[0].url.contains("llama-b12182-bin-win-cpu-x64.zip"));
        assert_eq!(p[0].kind, AssetKind::EngineZip);
    }
}
