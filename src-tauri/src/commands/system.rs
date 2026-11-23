//! Read-only host telemetry that the GPU actor doesn't cover.
//!
//! `system_memory` reports physical RAM for the Workbench dock's
//! "System memory" meter. It is a single cheap OS call, so the frontend polls
//! it (every few seconds) instead of holding a subscription channel open.

use serde::Serialize;

use crate::error::AppError;

/// Physical memory snapshot, in MiB. Mirrored by `SystemMemory` in
/// `src/lib/types/index.ts`.
#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct SystemMemory {
    pub total_mb: u64,
    pub used_mb: u64,
    pub available_mb: u64,
}

impl SystemMemory {
    /// Build a snapshot from byte counts. `used` is derived as total − available
    /// and saturates so a racy OS reading can never underflow.
    fn from_bytes(total: u64, available: u64) -> Self {
        const MIB: u64 = 1024 * 1024;
        let total_mb = total / MIB;
        let available_mb = (available / MIB).min(total_mb);
        Self {
            total_mb,
            used_mb: total_mb - available_mb,
            available_mb,
        }
    }
}

#[cfg(windows)]
fn read_system_memory() -> Option<SystemMemory> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};

    // SAFETY: MEMORYSTATUSEX is a plain C struct; zero-initialising it and
    // setting dwLength is exactly what the API contract requires.
    let mut status: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: `status` is a valid, writable MEMORYSTATUSEX with dwLength set.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 {
        return None;
    }
    Some(SystemMemory::from_bytes(status.ullTotalPhys, status.ullAvailPhys))
}

#[cfg(not(windows))]
fn read_system_memory() -> Option<SystemMemory> {
    // Desktop-only, Windows-first (see CLAUDE.md): other OSes report
    // "unknown" and the dock hides the meter rather than guessing.
    None
}

/// Physical RAM total/used/available. `None` when the OS query fails or the
/// platform isn't supported yet; the UI hides the meter in that case.
#[tauri::command]
pub async fn system_memory() -> Result<Option<SystemMemory>, AppError> {
    Ok(read_system_memory())
}

/// Open Windows Explorer at an installed model: the folder for dir-layout
/// models, the weights file selected for single-file GGUFs. Read-only — it
/// never touches the catalog.
#[tauri::command]
pub async fn reveal_model(
    state: tauri::State<'_, crate::state::AppState>,
    model_id: String,
) -> Result<(), AppError> {
    let Some(model) = crate::models::registry::get_by_id(&state.db, &model_id).await? else {
        return Err(AppError::Other(format!("unknown model: {model_id}")));
    };
    let path = model.local_path.clone().unwrap_or_else(|| model.primary_file_path.clone());
    if !std::path::Path::new(&path).exists() {
        return Err(AppError::Other("the model's files are no longer on disk".into()));
    }
    // Same `explorer /select,` contract as reveal_asset: explorer's exit code
    // is unreliable, so only a spawn failure is an error.
    std::process::Command::new("explorer")
        .arg(format!("/select,{path}"))
        .spawn()
        .map_err(|e| AppError::Other(format!("could not open Explorer: {e}")))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn derives_used_from_total_minus_available() {
        let m = SystemMemory::from_bytes(64 * GIB, 40 * GIB);
        assert_eq!(m.total_mb, 64 * 1024);
        assert_eq!(m.available_mb, 40 * 1024);
        assert_eq!(m.used_mb, 24 * 1024);
    }

    #[test]
    fn available_above_total_saturates_to_zero_used() {
        let m = SystemMemory::from_bytes(8 * GIB, 9 * GIB);
        assert_eq!(m.available_mb, 8 * 1024);
        assert_eq!(m.used_mb, 0);
    }

    #[cfg(windows)]
    #[test]
    fn reads_real_memory_on_windows() {
        let m = read_system_memory().expect("GlobalMemoryStatusEx should succeed");
        assert!(m.total_mb > 0);
        assert!(m.used_mb + m.available_mb == m.total_mb);
    }
}
