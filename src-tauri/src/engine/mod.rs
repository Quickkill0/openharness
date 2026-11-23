pub mod installer;
pub mod native;
mod memory;
#[cfg(test)]
pub(crate) mod native_tests;
pub mod sidecar;
pub mod manager;
pub mod fit; // Slice D fills this

use serde::{Deserialize, Serialize};

pub const ENGINE_BUILD: &str = "b12182";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineVariant {
    #[serde(rename = "cuda-12.4")]
    CudaV12_4,
    Cpu,
}

impl EngineVariant {
    pub fn as_asset_slug(&self) -> &'static str {
        match self {
            Self::CudaV12_4 => "cuda-12.4",
            Self::Cpu => "cpu",
        }
    }

    pub fn as_db_str(&self) -> &'static str {
        self.as_asset_slug()
    }

    pub fn from_db_str(s: &str) -> Option<Self> {
        match s {
            "cuda-12.4" => Some(Self::CudaV12_4),
            "cpu" => Some(Self::Cpu),
            _ => None,
        }
    }
}

/// Pure detection: caller probes GPU and passes whether CUDA 12+ is usable.
pub fn detect_variant(has_cuda_12_plus: bool) -> EngineVariant {
    if has_cuda_12_plus {
        EngineVariant::CudaV12_4
    } else {
        EngineVariant::Cpu
    }
}

pub use manager::{EngineManagerHandle, EngineStatus, ModelLoadArgs, start};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_cuda_when_supported() {
        assert_eq!(detect_variant(true), EngineVariant::CudaV12_4);
    }

    #[test]
    fn detect_cpu_when_no_cuda() {
        assert_eq!(detect_variant(false), EngineVariant::Cpu);
    }
}
