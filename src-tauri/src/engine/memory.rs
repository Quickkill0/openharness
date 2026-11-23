//! Memory genuinely returned by exiting the text process. File-backed GGUF
//! pages are already available/reclaimable to the OS; RSS/commit overcredits them.

/// What stopping the text process gives back. `resident` is private memory in
/// RAM and credits free RAM. `commit` is the process's private commit charge
/// and credits only the commit limit (RAM plus pagefile): on Windows it also
/// carries the graphics-memory allocations charged to the process, so it is
/// larger than `resident` and must never be counted as free RAM.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HostCredit {
    pub resident: u64,
    pub commit: u64,
}

#[cfg(target_os = "linux")]
pub fn private_memory(pid: u32) -> HostCredit {
    let resident = std::fs::read_to_string(format!("/proc/{pid}/status")).ok().and_then(|s| {
        s.lines().find_map(|l| l.strip_prefix("RssAnon:").and_then(|n| n.split_whitespace().next()?.parse::<u64>().ok()).map(|kb| kb * 1024))
    }).unwrap_or(0);
    // The planner has no commit limit off Windows; the resident bytes are a floor.
    HostCredit { resident, commit: resident }
}

#[cfg(windows)]
pub fn private_memory(pid: u32) -> HostCredit {
    use windows_sys::Win32::{Foundation::CloseHandle, System::{ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX}, Threading::{OpenProcess, PROCESS_QUERY_INFORMATION, PROCESS_VM_READ}}};
    // PROCESS_MEMORY_COUNTERS_EX2 (Windows 10+). Request its full size so old
    // systems fail closed instead of substituting PrivateUsage (commit) for
    // the resident bytes.
    #[repr(C)]
    struct Ex2 { base: PROCESS_MEMORY_COUNTERS_EX, private_working_set: usize, shared_commit: usize }
    unsafe {
        let process = OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, 0, pid);
        if process.is_null() { return HostCredit::default(); }
        let mut counts: Ex2 = std::mem::zeroed();
        let size = std::mem::size_of::<Ex2>() as u32;
        counts.base.cb = size;
        let ok = K32GetProcessMemoryInfo(process, &mut counts as *mut Ex2 as *mut _, size);
        CloseHandle(process);
        if ok != 0 { HostCredit { resident: counts.private_working_set as u64, commit: counts.base.PrivateUsage as u64 } } else { HostCredit::default() }
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
pub fn private_memory(_: u32) -> HostCredit { HostCredit::default() } // No trustworthy private-memory probe.
