//! Native runtime provisioning and capability discovery. The engine actor is
//! the sole owner of the probe cache; replacing a hand-placed binary invalidates it.
use std::path::PathBuf;
use std::time::{Duration, SystemTime};
use serde::{Deserialize, Serialize};
use crate::provision::{AssetKind, ManifestAsset, RuntimeManifest};
use crate::error::AppError;

pub const HOSTED: bool = false;
pub const BINARY: &str = if cfg!(windows) { "openharness-engine-server.exe" } else { "openharness-engine-server" };

pub fn manifest() -> RuntimeManifest {
    RuntimeManifest {
        id: "native", version: "v0.1.0-placeholder",
        install_dir: crate::config::engine_dir().join("native"),
        // PLACEHOLDER: publish reviewed archives and SHA-256 pins before HOSTED=true.
        assets: vec![ManifestAsset {
            url: "https://github.com/Quickkill0/openharness-public/releases/download/REPLACE_TAG/native-REPLACE_PLATFORM.zip".into(),
            filename: "native-REPLACE_PLATFORM.zip".into(), sha256: None, kind: AssetKind::Zip,
        }],
        required_files: vec![BINARY], needed_bytes: 2 * 1024 * 1024 * 1024,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Features {
    #[serde(default)] pub ssd_expert_streaming: bool,
    #[serde(default)] pub cuda_vram_expert_tier: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Device {
    pub name: String,
    pub description: String,
    #[serde(rename = "type")] pub kind: String,
    pub memory_total_mib: u64,
    pub memory_free_mib: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Backend { pub name: String, pub devices: Vec<Device> }
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EngineInfo {
    pub engine: String,
    pub engine_version: String,
    pub llama_cpp: serde_json::Value,
    pub features: Features,
    pub backends: Vec<Backend>,
}
#[derive(Debug, Clone, Serialize)]
pub struct NativeStatus {
    pub installed: bool,
    pub hosted: bool,
    pub path: PathBuf,
    pub info: Option<EngineInfo>,
    pub error: Option<String>,
}
impl NativeStatus {
    pub fn support(&self) -> crate::planner::native::NativeSupport {
        crate::planner::native::NativeSupport {
            installed: self.installed,
            streaming: self.info.as_ref().is_some_and(|i| i.features.ssd_expert_streaming),
            cuda_tier: self.info.as_ref().is_some_and(|i| i.features.cuda_vram_expert_tier
                && i.backends.iter().any(|b| b.name == "CUDA" && b.devices.iter().any(|d| d.kind == "gpu"))),
            error: self.error.clone(),
        }
    }
}

#[derive(Default)]
pub struct Discovery {
    cached: Option<(PathBuf, u64, SystemTime, NativeStatus)>,
}
impl Discovery {
    /// Drop a cached failed probe so the next read runs it again. A good
    /// result stays, so one bad moment can't lose it. Failures stay cached
    /// otherwise: each one can cost a 15-second timeout, and the planner reads
    /// this for every model.
    pub(crate) fn forget_failure(&mut self) {
        if self.cached.as_ref().is_some_and(|(.., status)| status.error.is_some()) {
            self.cached = None;
        }
    }
    /// What is known without running the binary, for the engine actor's
    /// startup pump: a hung probe there would hold off cancel, status and logs.
    pub(crate) async fn cached_at(&self, path: &std::path::Path) -> NativeStatus {
        let meta = tokio::fs::metadata(path).await.ok().filter(|m| m.is_file());
        let mut status = NativeStatus { installed: meta.is_some(), hosted: HOSTED, path: path.to_path_buf(), info: None, error: None };
        let Some(meta) = meta else { return status; };
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if let Some((p, len, time, cached)) = &self.cached {
            if p == path && *len == meta.len() && *time == modified { return cached.clone(); }
        }
        status.error = Some("Not checked yet because a model is starting. Check again when it has finished.".into());
        status
    }
    pub async fn status(&mut self) -> NativeStatus {
        self.at(manifest().install_dir.join(BINARY)).await
    }
    pub(crate) async fn at(&mut self, path: PathBuf) -> NativeStatus {
        let meta = tokio::fs::metadata(&path).await.ok().filter(|m| m.is_file());
        let mut status = NativeStatus { installed: meta.is_some(), hosted: HOSTED, path: path.clone(), info: None, error: None };
        let Some(meta) = meta else { self.cached = None; return status; };
        let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        if let Some((p, len, time, cached)) = &self.cached {
            if p == &path && *len == meta.len() && *time == modified { return cached.clone(); }
        }
        let mut cmd = tokio::process::Command::new(&path);
        cmd.arg("--engine-info").kill_on_drop(true);
        #[cfg(windows)]
        cmd.creation_flags(0x08000000);
        match tokio::time::timeout(Duration::from_secs(15), cmd.output()).await {
            Ok(Ok(out)) if out.status.success() => {
                match serde_json::from_slice::<EngineInfo>(&out.stdout) {
                    Ok(info) if info.engine == "openharness-engine" => status.info = Some(info),
                    Ok(_) => status.error = Some("Native binary did not identify as openharness-engine".into()),
                    Err(e) => status.error = Some(format!("Native --engine-info is invalid: {e}")),
                }
            }
            Ok(Ok(out)) => status.error = Some(String::from_utf8_lossy(&out.stderr).into_owned()),
            Ok(Err(e)) => status.error = Some(format!("Native --engine-info failed: {e}")),
            Err(_) => status.error = Some("Native --engine-info timed out".into()),
        }
        self.cached = Some((path, meta.len(), modified, status.clone()));
        status
    }
}

pub async fn install() -> Result<(), AppError> {
    // Never delete or reprovision a hand-placed install, even without a stamp.
    if manifest().install_dir.join(BINARY).is_file() { return Ok(()); }
    if !HOSTED { return Err(AppError::Other("Native engine binaries are not hosted yet. Place a locally built native engine in the engine/native folder.".into())); }
    crate::provision::install(&reqwest::Client::new(), &manifest(), |_| {}).await?;
    Ok(())
}
