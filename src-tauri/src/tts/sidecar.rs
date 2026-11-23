//! tts-server process wrapper — the Qwen3-TTS sibling of
//! [`crate::engine::sidecar::LlamaServer`]. Spawns the qwentts.cpp
//! `tts-server.exe` with a talker + codec GGUF pair, captures its logs, and
//! polls `/health` until the (slow, multi-GB F32) load completes.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::error::AppError;

pub struct TtsServer {
    child: Child,
    pub port: u16,
    pub log_buf: Arc<Mutex<VecDeque<String>>>,
}

impl TtsServer {
    pub async fn start(
        bin: &Path,
        talker: &Path,
        codec: &Path,
        port: u16,
    ) -> Result<Self, AppError> {
        use std::process::Stdio;
        let mut cmd = Command::new(bin);
        cmd.arg("--model")
            .arg(talker)
            .arg("--codec")
            .arg(codec)
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--port")
            .arg(port.to_string());
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        cmd.kill_on_drop(true);

        // Hide console window on Windows
        #[cfg(windows)]
        {
            #[allow(unused_imports)]
            use std::os::windows::process::CommandExt;
            cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| AppError::EngineSpawnFailed(format!("tts-server: {e}")))?;

        // Die with the app even when Drop never runs (hard kill) — winjob.rs.
        #[cfg(windows)]
        crate::winjob::tie_to_app_lifetime(&child, "tts-server");

        let log_buf: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(1000)));

        if let Some(stdout) = child.stdout.take() {
            let buf = log_buf.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "tts-server.out", "{}", line);
                    let mut b = buf.lock().await;
                    if b.len() == 1000 {
                        b.pop_front();
                    }
                    b.push_back(line);
                }
            });
        }
        if let Some(stderr) = child.stderr.take() {
            let buf = log_buf.clone();
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "tts-server.err", "{}", line);
                    let mut b = buf.lock().await;
                    if b.len() == 1000 {
                        b.pop_front();
                    }
                    b.push_back(line);
                }
            });
        }

        Ok(Self { child, port, log_buf })
    }

    pub async fn health_check(&self, http: &reqwest::Client) -> bool {
        match http
            .get(format!("http://127.0.0.1:{}/health", self.port))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    pub async fn wait_ready(
        &mut self,
        http: &reqwest::Client,
        timeout: Duration,
    ) -> Result<(), AppError> {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if let Ok(Some(status)) = self.child.try_wait() {
                let tail = {
                    let b = self.log_buf.lock().await;
                    let mut lines: Vec<String> = b.iter().rev().take(20).cloned().collect();
                    lines.reverse();
                    lines.join("\n")
                };
                return Err(AppError::Other(format!(
                    "tts-server exited early ({}): {}",
                    status, tail
                )));
            }
            if self.health_check(http).await {
                return Ok(());
            }
            if tokio::time::Instant::now() >= deadline {
                return Err(AppError::EngineLoadTimeout);
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    }

    pub async fn stop(mut self) -> Result<(), AppError> {
        let _ = self.child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await;
        Ok(())
    }
}

impl Drop for TtsServer {
    fn drop(&mut self) {
        // Best-effort: synchronous start_kill, ignore errors.
        let _ = self.child.start_kill();
    }
}
