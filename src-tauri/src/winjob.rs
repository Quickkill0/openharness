//! Tie sidecar process lifetime to the app via a Windows Job Object.
//!
//! Rust `Drop` impls and tokio's `kill_on_drop` only run when the parent
//! process shuts down cleanly. When the app is hard-killed — a `tauri dev`
//! rebuild, a crash, Task Manager — neither runs, and spawned sidecars
//! (llama-server, the Python worker, qwen tts-server) survive as orphans
//! holding multi-GB VRAM allocations. That is exactly how two llama-servers
//! ended up serving the same model side by side.
//!
//! The fix is OS-level: every sidecar is assigned to a per-app-process Job
//! Object configured with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. When this
//! process dies *for any reason*, the kernel closes the job handle and
//! terminates every process in the job. Since Windows 8 a process can belong
//! to multiple jobs, so this nests fine under whatever job a dev runner or
//! terminal already put us in.

#![cfg(windows)]

use std::os::windows::io::RawHandle;
use std::sync::OnceLock;

use windows_sys::Win32::Foundation::HANDLE;
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};

/// An owned job object configured to kill its members when the last handle
/// closes. The struct exists (rather than a bare `HANDLE`) so tests can
/// create a private job, drop it, and assert members die.
pub struct Job(HANDLE);

// HANDLE is a raw pointer, but job handles are process-global kernel objects;
// the only mutation is through Win32 calls that are themselves thread-safe.
unsafe impl Send for Job {}
unsafe impl Sync for Job {}

impl Job {
    /// Create a job whose members are terminated when the last job handle
    /// closes (i.e. when this process exits, however it exits).
    pub fn kill_on_close() -> Option<Job> {
        unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            if ok == 0 {
                windows_sys::Win32::Foundation::CloseHandle(job);
                return None;
            }
            Some(Job(job))
        }
    }

    /// Put a child process (by its raw process handle) into this job.
    pub fn assign_raw(&self, process: RawHandle) -> bool {
        unsafe { AssignProcessToJobObject(self.0, process as HANDLE) != 0 }
    }
}

impl Drop for Job {
    fn drop(&mut self) {
        // Closing the last handle terminates members (KILL_ON_JOB_CLOSE).
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}

/// The app-lifetime job. Created on first use and deliberately never dropped
/// while the process lives; the kernel closes the handle when the process
/// dies, which is the whole point.
fn session_job() -> Option<&'static Job> {
    static JOB: OnceLock<Option<Job>> = OnceLock::new();
    JOB.get_or_init(|| {
        let job = Job::kill_on_close();
        if job.is_none() {
            tracing::warn!("failed to create sidecar job object — orphan protection disabled");
        }
        job
    })
    .as_ref()
}

/// Best-effort: tie `child`'s lifetime to this process. Call right after
/// spawning a sidecar. Failure is logged, never fatal — the child still works,
/// it just won't be reaped if the app is hard-killed.
pub fn tie_to_app_lifetime(child: &tokio::process::Child, what: &str) {
    let Some(job) = session_job() else { return };
    match child.raw_handle() {
        Some(h) => {
            if !job.assign_raw(h) {
                tracing::warn!("could not assign {what} (pid {:?}) to the app job object", child.id());
            }
        }
        None => {
            tracing::warn!("{what} exited before it could be tied to the app job object");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    /// Regression test for the llama-server orphan leak: a child assigned to
    /// a kill-on-close job must die when the job handle closes — which is
    /// what happens to the app-lifetime job when the process is hard-killed.
    #[tokio::test]
    async fn job_close_kills_member_process() {
        let job = Job::kill_on_close().expect("create job");

        // A child that would outlive the test by far if not reaped.
        let mut child = tokio::process::Command::new("cmd")
            .args(["/C", "ping -n 60 127.0.0.1 >nul"])
            .creation_flags(0x08000000) // CREATE_NO_WINDOW
            .spawn()
            .expect("spawn child");
        let handle = child.raw_handle().expect("child handle");
        assert!(job.assign_raw(handle), "assign child to job");

        // Closing the job handle must terminate the member. The child would
        // otherwise run for ~60s, so completing within the timeout *is* the
        // assertion (Windows reports job-close kills with exit code 0, so
        // the status itself proves nothing).
        drop(job);

        let waited = tokio::time::timeout(Duration::from_secs(5), child.wait()).await;
        waited
            .expect("child did not die when the job closed")
            .expect("wait failed");
    }
}
