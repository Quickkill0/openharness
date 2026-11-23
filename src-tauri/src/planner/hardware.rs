//! Hardware profile for the load planner: GPU (NVML), system RAM and the
//! drive holding the models folder (sysinfo), plus the memory-tier
//! bandwidths the speed estimator divides by.
//!
//! Bandwidths default to conservative figures until measured. The SSD read
//! speed can be measured on demand with [`benchmark_ssd`] (bounded, explicit
//! command only — never at startup); the result is persisted by the planner
//! store and folded into every later profile.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::gpu::GpuMetrics;

// ---------------------------------------------------------------------------
// Default bandwidths (GB/s, decimal)
// ---------------------------------------------------------------------------

/// Host→GPU copies over PCIe 4.0 x16 (practical, not the 32 GB/s line rate).
pub const DEFAULT_PCIE_GBPS: f64 = 25.0;
/// DDR5 dual channel, practical read bandwidth.
pub const DEFAULT_RAM_GBPS: f64 = 60.0;
/// Unmeasured SSD: a mid-range PCIe 3.0/4.0 NVMe sequential read.
pub const DEFAULT_SSD_GBPS: f64 = 3.0;
/// Unmeasured spinning disk.
pub const DEFAULT_HDD_GBPS: f64 = 0.15;
/// GPU memory bandwidth when the card is not in [`GPU_BANDWIDTH_TABLE`].
pub const DEFAULT_VRAM_GBPS: f64 = 450.0;

/// Spec-sheet memory bandwidth by GPU name fragment (first match wins, so
/// longer names come first).
const GPU_BANDWIDTH_TABLE: &[(&str, f64)] = &[
    ("5090", 1792.0),
    ("5080", 960.0),
    ("5070 ti", 896.0),
    ("5070", 672.0),
    ("5060 ti", 448.0),
    ("4090", 1008.0),
    ("4080", 717.0),
    ("4070 ti super", 672.0),
    ("4070 ti", 504.0),
    ("4070 super", 504.0),
    ("4070", 504.0),
    ("4060 ti", 288.0),
    ("4060", 272.0),
    ("3090 ti", 1008.0),
    ("3090", 936.0),
    ("3080 ti", 912.0),
    ("3080", 760.0),
    ("3070", 448.0),
    ("3060", 360.0),
    ("a6000", 768.0),
    ("a100", 1555.0),
    ("h100", 3350.0),
];

pub fn vram_gbps_for(gpu_name: &str) -> f64 {
    let n = gpu_name.to_ascii_lowercase();
    GPU_BANDWIDTH_TABLE
        .iter()
        .find(|(frag, _)| n.contains(frag))
        .map_or(DEFAULT_VRAM_GBPS, |(_, bw)| *bw)
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GpuInfo {
    pub name: String,
    pub vram_total_bytes: u64,
    pub vram_free_bytes: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DriveKind {
    Ssd,
    Hdd,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StorageInfo {
    /// Mount point holding the models folder, e.g. `C:\`.
    pub mount: String,
    /// Device model, e.g. "Samsung SSD 990 PRO 2TB", when Windows reports it.
    pub model: Option<String>,
    pub kind: DriveKind,
    pub free_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bandwidths {
    pub vram_gbps: f64,
    pub pcie_gbps: f64,
    pub ram_gbps: f64,
    pub ssd_gbps: f64,
    /// True when `ssd_gbps` came from [`benchmark_ssd`] on this drive.
    pub ssd_measured: bool,
}

/// Snapshot of the machine the planner places a model onto.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HardwareProfile {
    /// None when no NVIDIA GPU is present (CPU-only planning).
    pub gpu: Option<GpuInfo>,
    pub ram_total_bytes: u64,
    pub ram_available_bytes: u64,
    /// Windows commit still available (RAM plus pagefile, less what every
    /// process has committed). `None` where it is not read; the planner
    /// then sizes by RAM alone.
    #[serde(default)]
    pub commit_available_bytes: Option<u64>,
    pub storage: Option<StorageInfo>,
    pub bandwidth: Bandwidths,
    pub cpu_threads: u32,
}

/// A persisted SSD measurement (see `planner::store`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SsdMeasurement {
    pub mount: String,
    pub gbps: f64,
    pub bytes_read: u64,
    pub seconds: f64,
    pub measured_at: i64,
    pub file: String,
}

// ---------------------------------------------------------------------------
// Probing
// ---------------------------------------------------------------------------

/// Build a profile from live readings. `dir` picks the drive (the models
/// folder, or the folder of the model being planned); `ssd` holds the
/// persisted measurements, of which only the one for that drive applies.
pub fn probe(gpu: Option<&GpuMetrics>, dir: &Path, ssd: &[SsdMeasurement]) -> HardwareProfile {
    let (ram_total, ram_avail) = match sysinfo::System::new() {
        Ok(mut s) => {
            s.refresh_memory();
            (s.total_memory(), s.available_memory())
        }
        Err(_) => (0, 0),
    };
    let storage = storage_for(dir);
    let cpu_threads = std::thread::available_parallelism().map_or(8, |n| n.get() as u32);
    let mut profile = build_profile(gpu, ram_total, ram_avail, storage, ssd, cpu_threads);
    profile.commit_available_bytes = commit_available();
    profile
}

/// Commit the system can still hand out (`ullAvailPageFile`).
#[cfg(windows)]
fn commit_available() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: MEMORYSTATUSEX is plain data; dwLength is set as the API requires.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    (unsafe { GlobalMemoryStatusEx(&mut status) } != 0).then_some(status.ullAvailPageFile)
}

#[cfg(not(windows))]
fn commit_available() -> Option<u64> {
    None
}

/// Pure assembly (unit-tested): picks bandwidths from the readings.
pub fn build_profile(
    gpu: Option<&GpuMetrics>,
    ram_total_bytes: u64,
    ram_available_bytes: u64,
    storage: Option<StorageInfo>,
    ssd: &[SsdMeasurement],
    cpu_threads: u32,
) -> HardwareProfile {
    let gpu = gpu.map(|m| GpuInfo {
        name: m.name.clone(),
        vram_total_bytes: m.vram_total_mb * MIB,
        vram_free_bytes: m.vram_free_mb * MIB,
    });
    let measured = ssd.iter().find(|s| {
        storage
            .as_ref()
            .is_some_and(|st| st.mount.eq_ignore_ascii_case(&s.mount))
            && s.gbps > 0.0
    });
    let ssd_gbps = match (measured, storage.as_ref().map(|s| s.kind)) {
        (Some(m), _) => m.gbps,
        (None, Some(DriveKind::Hdd)) => DEFAULT_HDD_GBPS,
        _ => DEFAULT_SSD_GBPS,
    };
    HardwareProfile {
        bandwidth: Bandwidths {
            vram_gbps: gpu.as_ref().map_or(0.0, |g| vram_gbps_for(&g.name)),
            pcie_gbps: DEFAULT_PCIE_GBPS,
            ram_gbps: DEFAULT_RAM_GBPS,
            ssd_gbps,
            ssd_measured: measured.is_some(),
        },
        gpu,
        ram_total_bytes,
        ram_available_bytes,
        commit_available_bytes: None,
        storage,
        cpu_threads,
    }
}

const MIB: u64 = 1024 * 1024;

/// The mounted drive whose mount point is the longest prefix of `dir`.
pub fn storage_for(dir: &Path) -> Option<StorageInfo> {
    let disks = sysinfo::Disks::new_with_refreshed_list().ok()?;
    let dir_s = dir.to_string_lossy().to_ascii_lowercase();
    let disk = disks
        .iter()
        .filter(|d| dir_s.starts_with(&d.mount_point().to_string_lossy().to_ascii_lowercase()))
        .max_by_key(|d| d.mount_point().as_os_str().len())?;
    let mount = disk.mount_point().to_string_lossy().into_owned();
    Some(StorageInfo {
        model: device_model(&mount),
        kind: match disk.kind() {
            sysinfo::DiskKind::SSD => DriveKind::Ssd,
            sysinfo::DiskKind::HDD => DriveKind::Hdd,
            _ => DriveKind::Unknown,
        },
        free_bytes: disk.available_space(),
        total_bytes: disk.total_space(),
        mount,
    })
}

/// Storage device product id for a volume (`C:\` → "Samsung SSD 990 PRO
/// 2TB") via `IOCTL_STORAGE_QUERY_PROPERTY`. Needs no elevation.
#[cfg(windows)]
fn device_model(mount: &str) -> Option<String> {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::Storage::FileSystem::{
        CreateFileW, FILE_SHARE_READ, FILE_SHARE_WRITE, OPEN_EXISTING,
    };
    use windows_sys::Win32::System::Ioctl::{
        PropertyStandardQuery, StorageDeviceProperty, IOCTL_STORAGE_QUERY_PROPERTY,
        STORAGE_DEVICE_DESCRIPTOR, STORAGE_PROPERTY_QUERY,
    };
    use windows_sys::Win32::System::IO::DeviceIoControl;

    let letter = mount.trim_end_matches(['\\', '/']);
    if letter.len() != 2 || !letter.ends_with(':') {
        return None;
    }
    let dev: Vec<u16> = std::ffi::OsStr::new(&format!(r"\\.\{letter}"))
        .encode_wide()
        .chain(std::iter::once(0))
        .collect();
    // SAFETY: plain Win32 calls on a handle we own and close; buffers are
    // sized locals and the returned offsets are bounds-checked before use.
    unsafe {
        let h = CreateFileW(
            dev.as_ptr(),
            0, // no access rights needed for a property query
            FILE_SHARE_READ | FILE_SHARE_WRITE,
            std::ptr::null(),
            OPEN_EXISTING,
            0,
            std::ptr::null_mut(),
        );
        if h == INVALID_HANDLE_VALUE || h.is_null() {
            return None;
        }
        let query = STORAGE_PROPERTY_QUERY {
            PropertyId: StorageDeviceProperty,
            QueryType: PropertyStandardQuery,
            AdditionalParameters: [0],
        };
        let mut buf = vec![0u8; 1024];
        let mut returned = 0u32;
        let ok = DeviceIoControl(
            h,
            IOCTL_STORAGE_QUERY_PROPERTY,
            &query as *const _ as *const _,
            std::mem::size_of::<STORAGE_PROPERTY_QUERY>() as u32,
            buf.as_mut_ptr() as *mut _,
            buf.len() as u32,
            &mut returned,
            std::ptr::null_mut(),
        );
        CloseHandle(h);
        if ok == 0 || (returned as usize) < std::mem::size_of::<STORAGE_DEVICE_DESCRIPTOR>() {
            return None;
        }
        let desc = std::ptr::read_unaligned(buf.as_ptr() as *const STORAGE_DEVICE_DESCRIPTOR);
        let read_str = |off: u32| -> Option<String> {
            let off = off as usize;
            if off == 0 || off >= returned as usize {
                return None;
            }
            let end = buf[off..returned as usize].iter().position(|&b| b == 0)? + off;
            let s = String::from_utf8_lossy(&buf[off..end]).trim().to_string();
            (!s.is_empty()).then_some(s)
        };
        let product = read_str(desc.ProductIdOffset)?;
        Some(match read_str(desc.VendorIdOffset) {
            Some(v) if !product.starts_with(&v) => format!("{v} {product}"),
            _ => product,
        })
    }
}

#[cfg(not(windows))]
fn device_model(_mount: &str) -> Option<String> {
    None
}

// ---------------------------------------------------------------------------
// SSD benchmark
// ---------------------------------------------------------------------------

/// Read budget for one benchmark run.
pub const BENCH_MAX_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Wall-clock budget for one benchmark run.
pub const BENCH_MAX_SECS: f64 = 15.0;
const BENCH_CHUNK: usize = 8 * 1024 * 1024;
/// Sector alignment for unbuffered reads (covers 512e and 4Kn drives).
const BENCH_ALIGN: usize = 4096;

#[derive(Debug, Clone, Serialize)]
pub struct SsdBenchmark {
    pub file: String,
    pub bytes_read: u64,
    pub seconds: f64,
    pub gbps: f64,
    /// False when the OS refused unbuffered I/O and the page cache may have
    /// inflated the figure.
    pub unbuffered: bool,
}

/// Sequential-read benchmark over an existing file: unbuffered (bypasses the
/// page cache, `FILE_FLAG_NO_BUFFERING` on Windows), aligned 8 MiB reads, at
/// most `max_bytes` and `max_secs`. Blocking — call from `spawn_blocking`.
pub fn benchmark_ssd(file: &Path, max_bytes: u64, max_secs: f64) -> std::io::Result<SsdBenchmark> {
    use std::io::Read;
    let (mut f, unbuffered) = open_unbuffered(file)?;
    let mut buf = AlignedBuf::new(BENCH_CHUNK, BENCH_ALIGN);
    let start = Instant::now();
    let deadline = Duration::from_secs_f64(max_secs.max(0.1));
    let mut total = 0u64;
    while total < max_bytes && start.elapsed() < deadline {
        let n = f.read(buf.as_mut_slice())?;
        if n == 0 {
            break;
        }
        total += n as u64;
    }
    let seconds = start.elapsed().as_secs_f64().max(1e-6);
    Ok(SsdBenchmark {
        file: file.to_string_lossy().into_owned(),
        bytes_read: total,
        seconds,
        gbps: total as f64 / seconds / 1e9,
        unbuffered,
    })
}

#[cfg(windows)]
fn open_unbuffered(file: &Path) -> std::io::Result<(std::fs::File, bool)> {
    use std::os::windows::fs::OpenOptionsExt;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_SEQUENTIAL_SCAN: u32 = 0x0800_0000;
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN)
        .open(file)
    {
        Ok(f) => Ok((f, true)),
        Err(_) => Ok((std::fs::File::open(file)?, false)),
    }
}

#[cfg(not(windows))]
fn open_unbuffered(file: &Path) -> std::io::Result<(std::fs::File, bool)> {
    Ok((std::fs::File::open(file)?, false))
}

/// Heap buffer with a guaranteed alignment (unbuffered I/O requires the
/// destination address and length to be sector-aligned).
struct AlignedBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

impl AlignedBuf {
    fn new(len: usize, align: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len, align).expect("valid layout");
        // SAFETY: non-zero size layout; null is handled below.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        if ptr.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        Self { ptr, layout }
    }
    fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: ptr is a live allocation of layout.size() bytes we own.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with the same layout.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// The largest regular file under `dir` (recursive, depth-limited) — the
/// benchmark target, so the read is long and sequential.
pub fn largest_file(dir: &Path) -> Option<PathBuf> {
    fn walk(dir: &Path, depth: u8, best: &mut Option<(u64, PathBuf)>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let Ok(ft) = e.file_type() else { continue };
            if ft.is_dir() && depth < 4 {
                walk(&e.path(), depth + 1, best);
            } else if ft.is_file() {
                let len = e.metadata().map_or(0, |m| m.len());
                if best.as_ref().map_or(true, |(b, _)| len > *b) {
                    *best = Some((len, e.path()));
                }
            }
        }
    }
    let mut best = None;
    walk(dir, 0, &mut best);
    best.map(|(_, p)| p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn metrics(name: &str, total_mb: u64, free_mb: u64) -> GpuMetrics {
        GpuMetrics {
            name: name.into(),
            driver_version: "x".into(),
            cuda_major: 13,
            cuda_minor: 0,
            vram_total_mb: total_mb,
            vram_used_mb: total_mb - free_mb,
            vram_free_mb: free_mb,
            sampled_at: 0,
        }
    }

    #[test]
    fn gpu_bandwidth_lookup() {
        assert_eq!(vram_gbps_for("NVIDIA GeForce RTX 4090"), 1008.0);
        assert_eq!(vram_gbps_for("NVIDIA GeForce RTX 4070 Ti SUPER"), 672.0);
        assert_eq!(vram_gbps_for("Mystery GPU"), DEFAULT_VRAM_GBPS);
    }

    #[test]
    fn measured_ssd_applies_only_to_its_drive() {
        let storage = StorageInfo {
            mount: "D:\\".into(),
            model: None,
            kind: DriveKind::Ssd,
            free_bytes: 1,
            total_bytes: 2,
        };
        let m = SsdMeasurement {
            mount: "d:\\".into(),
            gbps: 6.8,
            bytes_read: 1,
            seconds: 1.0,
            measured_at: 0,
            file: "x".into(),
        };
        let g = metrics("NVIDIA GeForce RTX 4090", 24564, 23000);
        let p = build_profile(Some(&g), 64 << 30, 48 << 30, Some(storage.clone()), std::slice::from_ref(&m), 32);
        assert_eq!(p.bandwidth.ssd_gbps, 6.8);
        assert!(p.bandwidth.ssd_measured);
        assert_eq!(p.gpu.as_ref().unwrap().vram_free_bytes, 23000 * MIB);

        let other = SsdMeasurement { mount: "E:\\".into(), ..m };
        let p = build_profile(None, 1, 1, Some(storage), &[other], 8);
        assert_eq!(p.bandwidth.ssd_gbps, DEFAULT_SSD_GBPS);
        assert!(!p.bandwidth.ssd_measured);
        assert_eq!(p.bandwidth.vram_gbps, 0.0);
    }

    #[test]
    fn benchmark_reads_small_file_within_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("blob.bin");
        let mut f = std::fs::File::create(&path).unwrap();
        // 3 aligned chunks + a ragged tail.
        let data = vec![7u8; BENCH_CHUNK * 3 + 12_345];
        f.write_all(&data).unwrap();
        drop(f);
        let r = benchmark_ssd(&path, BENCH_MAX_BYTES, 5.0).unwrap();
        assert_eq!(r.bytes_read, data.len() as u64);
        assert!(r.gbps > 0.0);
        // Budget cap stops early.
        let r = benchmark_ssd(&path, BENCH_CHUNK as u64, 5.0).unwrap();
        assert_eq!(r.bytes_read, BENCH_CHUNK as u64);
        assert_eq!(largest_file(dir.path()), Some(path));
    }
}
