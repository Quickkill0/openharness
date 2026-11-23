use std::sync::OnceLock;

use nvml_wrapper::{Nvml, error::NvmlError};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A single point-in-time GPU reading from the primary NVIDIA device.
#[derive(Debug, Clone)]
pub struct GpuSample {
    pub name: String,
    pub driver_version: String,
    /// CUDA compute capability: (major, minor)
    pub cuda_capability: (u32, u32),
    pub vram_total_mb: u64,
    pub vram_used_mb: u64,
    pub vram_free_mb: u64,
}

/// Trait so the actor can be driven by a stub in tests.
pub trait GpuProbe: Send + Sync + 'static {
    fn read_primary(&self) -> Result<GpuSample, GpuError>;

    /// VRAM held by one process in MiB, or None when unknown (no GPU, pid not
    /// on the device, or the driver doesn't report per-process usage).
    fn process_used_mb(&self, _pid: u32) -> Option<u64> {
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum GpuError {
    #[error("nvml: no NVIDIA driver detected")]
    NoDriver,
    #[error("nvml: {0}")]
    Nvml(String),
}

// ---------------------------------------------------------------------------
// NvmlProbe — real NVML implementation
// ---------------------------------------------------------------------------

/// NVML-backed probe. Initialises lazily on first call so the crate can be
/// loaded on machines without an NVIDIA driver without panicking at start-up.
pub struct NvmlProbe {
    /// Holds `Ok(Nvml)` after a successful init, or `Err(())` after a failed
    /// init so we never retry (driver won't appear mid-process).
    nvml: OnceLock<Result<Nvml, ()>>,
}

impl NvmlProbe {
    pub fn new() -> Self {
        Self {
            nvml: OnceLock::new(),
        }
    }

    /// Returns a reference to the `Nvml` instance, initialising on first call.
    /// Maps any init failure to `GpuError::NoDriver`.
    fn nvml(&self) -> Result<&Nvml, GpuError> {
        let result = self.nvml.get_or_init(|| {
            match Nvml::init() {
                Ok(n) => Ok(n),
                Err(e) => {
                    tracing::debug!("nvml init failed (no NVIDIA driver?): {e}");
                    Err(())
                }
            }
        });
        result.as_ref().map_err(|_| GpuError::NoDriver)
    }
}

impl Default for NvmlProbe {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProbe for NvmlProbe {
    fn read_primary(&self) -> Result<GpuSample, GpuError> {
        let nvml = self.nvml()?;

        let device = nvml
            .device_by_index(0)
            .map_err(|e| nvml_err(e))?;

        let name = device.name().map_err(nvml_err)?;
        let driver_version = nvml.sys_driver_version().map_err(nvml_err)?;
        let cuda_capability = device
            .cuda_compute_capability()
            .map(|c| (c.major as u32, c.minor as u32))
            .map_err(nvml_err)?;
        let mem = device.memory_info().map_err(nvml_err)?;

        let bytes_to_mb = |b: u64| b / (1024 * 1024);

        Ok(GpuSample {
            name,
            driver_version,
            cuda_capability,
            vram_total_mb: bytes_to_mb(mem.total),
            vram_used_mb: bytes_to_mb(mem.used),
            vram_free_mb: bytes_to_mb(mem.free),
        })
    }

    fn process_used_mb(&self, pid: u32) -> Option<u64> {
        use nvml_wrapper::enums::device::UsedGpuMemory;

        let nvml = self.nvml().ok()?;
        let device = nvml.device_by_index(0).ok()?;
        let procs = device.running_compute_processes().ok()?;
        procs.into_iter().find(|p| p.pid == pid).and_then(|p| match p.used_gpu_memory {
            UsedGpuMemory::Used(bytes) => Some(bytes / (1024 * 1024)),
            UsedGpuMemory::Unavailable => None,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn nvml_err(e: NvmlError) -> GpuError {
    // Treat "library not found" / "driver not loaded" variants as NoDriver so
    // callers get the clean fallback path without needing to inspect strings.
    match e {
        NvmlError::DriverNotLoaded
        | NvmlError::LibloadingError(_)
        | NvmlError::NoPermission => GpuError::NoDriver,
        other => GpuError::Nvml(other.to_string()),
    }
}
