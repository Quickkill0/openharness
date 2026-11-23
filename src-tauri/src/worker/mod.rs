//! Python worker — the second (and final) engine family (M8).
//!
//! A managed sidecar speaking JSON-Lines over stdio (see
//! `worker-py/openharness_worker/protocol.py` for the wire format). The
//! venv is materialized by `bootstrap` (pin-and-fetch `uv`, locked
//! requirements, torch wheel by GPU detection), the process lifecycle by
//! `sidecar`, and the public actor surface by `manager`.

pub mod bootstrap;
pub mod manager;
pub mod protocol;
pub mod sidecar;

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Pinned standalone `uv` release used to materialize the venv.
pub const UV_VERSION: &str = "0.11.20";
/// CPython version `uv` installs and pins the venv to.
pub const WORKER_PYTHON: &str = "3.12";
/// Must match `openharness_worker.__version__`; the ping handshake verifies it.
pub const WORKER_VERSION: &str = "0.1.0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WorkerVariant {
    Cuda,
    Cpu,
}

impl WorkerVariant {
    pub fn as_slug(&self) -> &'static str {
        match self {
            Self::Cuda => "cuda",
            Self::Cpu => "cpu",
        }
    }

    pub fn from_slug(s: &str) -> Option<Self> {
        match s {
            "cuda" => Some(Self::Cuda),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }

    /// Requirements file (relative to the worker-py source dir) for this variant.
    pub fn requirements_file(&self) -> &'static str {
        match self {
            Self::Cuda => "requirements.cuda.txt",
            Self::Cpu => "requirements.cpu.txt",
        }
    }
}

/// Resolve the `worker-py` source directory (the Python package shipped with
/// the app). In production it's a bundled Tauri resource; in dev it's the
/// checked-out `src-tauri/worker-py`.
pub fn worker_src_dir(resource_dir: Option<PathBuf>) -> Option<PathBuf> {
    if let Some(res) = resource_dir {
        let bundled = res.join("worker-py");
        if bundled.join("openharness_worker").is_dir() {
            return Some(bundled);
        }
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("worker-py");
    if dev.join("openharness_worker").is_dir() {
        return Some(dev);
    }
    None
}

pub use manager::{spawn_auto_manage, start, PipelineCapability, WorkerHandle, WorkerStatus};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variant_slug_round_trip() {
        for v in [WorkerVariant::Cuda, WorkerVariant::Cpu] {
            assert_eq!(WorkerVariant::from_slug(v.as_slug()), Some(v));
        }
        assert_eq!(WorkerVariant::from_slug("metal"), None);
    }

    #[test]
    fn dev_worker_src_dir_resolves() {
        // The repo checkout always has worker-py next to this crate.
        let dir = worker_src_dir(None).expect("worker-py should resolve in dev");
        assert!(dir.join("openharness_worker").join("__main__.py").exists());
        assert!(dir.join("requirements.cuda.txt").exists());
        assert!(dir.join("requirements.cpu.txt").exists());
    }
}
