use std::collections::VecDeque;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use tokio::process::{Child, Command};
use tokio::sync::Mutex;

use crate::error::AppError;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum InferenceMode {
    #[default]
    Generation,
    Embeddings,
}
impl InferenceMode {
    pub fn from_workflows(workflows: &[crate::models::capabilities::Workflow]) -> Self {
        if workflows.contains(&crate::models::capabilities::Workflow::Embeddings) {
            Self::Embeddings
        } else {
            Self::Generation
        }
    }
    pub(crate) fn validate_pooling(
        self,
        meta: Option<&crate::models::gguf::GgufMeta>,
    ) -> Result<(), AppError> {
        if self == Self::Embeddings && !matches!(meta.and_then(|m| m.pooling_type), Some(1..=3)) {
            // There is no model-independent safe pooling override. Refuse before
            // replacing a resident rather than return plausible but wrong vectors.
            return Err(AppError::EmbeddingPoolingUnsupported);
        }
        Ok(())
    }
    pub(crate) fn apply(self, cmd: &mut Command) {
        if self == Self::Embeddings {
            // Preserve the model's own pooling (e.g. last-token for Qwen3-Embedding).
            cmd.arg("--embeddings");
        }
    }
}

pub struct SidecarArgs {
    pub engine_kind: crate::planner::EngineKind,
    pub inference_mode: InferenceMode,
    pub model_path: std::path::PathBuf,
    pub ctx: u32,
    pub n_gpu_layers: i32, // -1 = all, 0 = CPU
    pub threads: Option<u32>,
    pub batch_size: Option<u32>,
    pub flash_attn: bool,
    /// KV-cache quantization type, e.g. "f16", "q8_0", "q4_0".
    pub kv_cache_type: Option<String>,
    pub mmap: bool,
    pub mlock: bool,
    pub alias: Option<String>,
    pub host: String,
    pub port: u16,
    /// Planner-chosen flags appended verbatim (`--n-cpu-moe`, `--moe-cache-mib`,
    /// `-ub`, `--fit off`, `-np`, `--metrics`, ...).
    pub extra_args: Vec<String>,
}

/// The llama-server argument list for `args` (pinned build, see
/// `ENGINE_BUILD`): `-ngl all` for full offload, `-fa on`, and `-lm` for the
/// load mode (the build dropped `--no-mmap` / `--mlock`).
pub fn command_line(args: &SidecarArgs) -> Vec<std::ffi::OsString> {
    let mut v: Vec<std::ffi::OsString> = Vec::new();
    let mut push = |s: &str| v.push(s.into());
    push("-m");
    push(&args.model_path.to_string_lossy());
    push("--host");
    push(&args.host);
    push("--port");
    push(&args.port.to_string());
    push("-c");
    push(&args.ctx.to_string());
    push("-ngl");
    if args.n_gpu_layers < 0 {
        push("all");
    } else {
        push(&args.n_gpu_layers.to_string());
    }
    if let Some(t) = args.threads {
        push("-t");
        push(&t.to_string());
    }
    if let Some(b) = args.batch_size {
        push("-b");
        push(&b.to_string());
    }
    if args.flash_attn {
        push("-fa");
        push("on");
    }
    if let Some(kv) = &args.kv_cache_type {
        // Quantize both K and V caches to the same type.
        push("-ctk");
        push(kv);
        push("-ctv");
        push(kv);
    }
    let load_mode = match (args.mmap, args.mlock) {
        (true, false) => None, // default: auto (mmap)
        (true, true) => Some("mmap+mlock"),
        (false, true) => Some("mlock"),
        (false, false) => Some("none"),
    };
    if let Some(m) = load_mode {
        push("-lm");
        push(m);
    }
    if let Some(a) = &args.alias {
        push("-a");
        push(a);
    }
    for a in &args.extra_args {
        push(a);
    }
    v
}

pub struct LlamaServer {
    child: Child,
    pub port: u16,
    pub model_path: std::path::PathBuf,
    pub log_buf: Arc<Mutex<VecDeque<String>>>,
    engine_kind: crate::planner::EngineKind,
    log_tasks: Vec<tokio::task::JoinHandle<()>>,
}

/// What the native engine may be started with. Dense layers go on the card
/// only with every routed expert kept on the host (`--cpu-moe`) for the engine
/// to stream; llama-server's own expert cache and draft heads never apply.
fn native_args_allowed(args: &SidecarArgs) -> bool {
    let cpu_moe = args.extra_args.iter().any(|a| a == "--cpu-moe");
    (args.n_gpu_layers != 0) == cpu_moe && args.inference_mode == InferenceMode::Generation
        && !args.extra_args.iter().any(|a| matches!(a.as_str(), "--fit" | "--moe-cache-mib" | "--n-cpu-moe" | "--nextn" | "--draft" | "--spec-type"))
}

impl LlamaServer {
    /// OS process id of the running server (None once it has exited).
    pub fn is_running(&mut self) -> bool { matches!(self.child.try_wait(), Ok(None)) }

    pub fn pid(&self) -> Option<u32> {
        self.child.id()
    }

    pub async fn start(bin: &Path, mut args: SidecarArgs) -> Result<Self, AppError> {
        use std::process::Stdio;
        args.model_path = crate::models::gguf::split_files(&args.model_path)?.into_iter().next().ok_or_else(|| AppError::Other("GGUF has no split files".into()))?;
        if args.engine_kind == crate::planner::EngineKind::SsdStream && !native_args_allowed(&args) {
            return Err(AppError::Other("Native SSD mode keeps routed experts off the card and generates without draft speculation".into()));
        }
        let mut cmd = Command::new(bin);
        cmd.args(command_line(&args));
        args.inference_mode.apply(&mut cmd);
        cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
        // Reap the child if this handle is ever dropped without an explicit
        // stop() (e.g. a panic unwinding past the engine actor).
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
            .map_err(|e| AppError::EngineSpawnFailed(e.to_string()))?;

        tracing::info!(
            target: "llama-server",
            pid = child.id(),
            port = args.port,
            ctx = args.ctx,
            ngl = args.n_gpu_layers,
            model = %args.model_path.display(),
            "spawned llama-server"
        );

        // OS-level orphan protection: if the app dies without running Drop
        // (tauri dev rebuild, crash, Task Manager), the kernel kills the
        // server with us. This was the root of the duplicate-llama-server
        // VRAM leak — see winjob.rs.
        #[cfg(windows)]
        crate::winjob::tie_to_app_lifetime(&child, "llama-server");

        let log_buf: Arc<Mutex<VecDeque<String>>> =
            Arc::new(Mutex::new(VecDeque::with_capacity(1000)));

        let mut log_tasks = Vec::new();
        if let Some(stdout) = child.stdout.take() {
            let buf = log_buf.clone();
            log_tasks.push(tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stdout).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "llama-server.out", "{}", line);
                    let mut b = buf.lock().await;
                    if b.len() == 1000 {
                        b.pop_front();
                    }
                    b.push_back(line);
                }
            }));
        }
        if let Some(stderr) = child.stderr.take() {
            let buf = log_buf.clone();
            log_tasks.push(tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                let mut lines = BufReader::new(stderr).lines();
                while let Ok(Some(line)) = lines.next_line().await {
                    tracing::debug!(target: "llama-server.err", "{}", line);
                    let mut b = buf.lock().await;
                    if b.len() == 1000 {
                        b.pop_front();
                    }
                    b.push_back(line);
                }
            }));
        }

        Ok(Self {
            child,
            port: args.port,
            model_path: args.model_path,
            log_buf,
            engine_kind: args.engine_kind,
            log_tasks,
        })
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
                // Drain both pipes before reading a refusal. Otherwise an early
                // exit can race the log readers and lose the engine's reason.
                for task in self.log_tasks.drain(..) { let _ = task.await; }
                let tail = {
                    let b = self.log_buf.lock().await;
                    let mut lines: Vec<String> = b.iter().rev().take(20).cloned().collect();
                    lines.reverse();
                    lines.join("\n")
                };
                if self.engine_kind == crate::planner::EngineKind::SsdStream {
                    return Err(AppError::NativeEngineRefused(tail));
                }
                // Translate llama.cpp's most opaque rejection into something
                // actionable before the raw log tail.
                let hint = if tail.contains("unknown model architecture")
                    || tail.contains("unknown architecture")
                {
                    "this GGUF's architecture isn't supported by llama.cpp — \
                     it is likely not a text-generation model (e.g. a TTS or \
                     vision component). "
                } else {
                    ""
                };
                return Err(AppError::Other(format!(
                    "llama-server exited early ({}): {}{}",
                    status, hint, tail
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
        tracing::info!(
            target: "llama-server",
            pid = self.child.id(),
            port = self.port,
            "stopping llama-server"
        );
        let _ = self.child.start_kill();
        let _ = tokio::time::timeout(Duration::from_secs(5), self.child.wait()).await;
        Ok(())
    }
}

impl Drop for LlamaServer {
    fn drop(&mut self) {
        // Best-effort: synchronous start_kill, ignore errors.
        // The async wait() in stop() is preferred when shutdown is graceful.
        if let Some(pid) = self.child.id() {
            tracing::info!(target: "llama-server", pid, port = self.port, "reaping llama-server on drop");
        }
        let _ = self.child.start_kill();
    }
}

/// Pick a free TCP port via bind-and-drop. Race-prone but acceptable for
/// single-instance use; we retry in `wait_ready` if the bind loses the race.
/// See CLAUDE.md "Tracked debt" — M4 port-allocation race entry.
pub fn pick_free_port() -> std::io::Result<u16> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
    let port = listener.local_addr()?.port();
    drop(listener);
    Ok(port)
}

#[cfg(test)]
mod command_line_tests {
    use super::*;

    fn base() -> SidecarArgs {
        SidecarArgs {
            engine_kind: Default::default(),
            inference_mode: InferenceMode::Generation,
            model_path: "m.gguf".into(),
            ctx: 8192,
            n_gpu_layers: -1,
            threads: None,
            batch_size: None,
            flash_attn: true,
            kv_cache_type: Some("q8_0".into()),
            mmap: true,
            mlock: false,
            alias: None,
            host: "127.0.0.1".into(),
            port: 9000,
            extra_args: vec!["--n-cpu-moe".into(), "47".into()],
        }
    }

    fn joined(a: &SidecarArgs) -> String {
        command_line(a)
            .iter()
            .map(|s| s.to_string_lossy().into_owned())
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn renders_pinned_build_flags() {
        let s = joined(&base());
        assert!(s.contains("-ngl all"), "{s}");
        assert!(s.contains("-fa on"), "{s}");
        assert!(s.contains("-ctk q8_0 -ctv q8_0"), "{s}");
        assert!(!s.contains("-lm"), "mmap is the default: {s}");
        assert!(s.ends_with("--n-cpu-moe 47"), "{s}");
        assert!(!s.contains("--no-mmap") && !s.contains("--mlock"), "{s}");
    }

    #[test]
    fn native_dense_layers_reach_the_card_only_with_experts_on_the_host() {
        let with = |ngl, extra: &[&str]| SidecarArgs { n_gpu_layers: ngl, extra_args: extra.iter().map(|s| s.to_string()).collect(), ..base() };
        assert!(native_args_allowed(&with(0, &["--ssd", "--no-op-offload"])));
        assert!(native_args_allowed(&with(-1, &["--ssd", "--cpu-moe"])));
        assert!(joined(&with(-1, &["--ssd", "--cpu-moe"])).contains("-ngl all"));
        // Experts would follow the dense layers onto the card.
        assert!(!native_args_allowed(&with(-1, &["--ssd"])));
        assert!(!native_args_allowed(&with(20, &["--ssd"])));
        // Nothing is offloaded, so the flag has no layout to belong to.
        assert!(!native_args_allowed(&with(0, &["--ssd", "--cpu-moe"])));
        for flag in ["--fit", "--moe-cache-mib", "--n-cpu-moe", "--nextn", "--draft", "--spec-type"] {
            assert!(!native_args_allowed(&with(-1, &["--ssd", "--cpu-moe", flag])), "{flag}");
            assert!(!native_args_allowed(&with(0, &["--ssd", flag])), "{flag}");
        }
        let embedding = SidecarArgs { inference_mode: InferenceMode::Embeddings, ..with(0, &["--ssd"]) };
        assert!(!native_args_allowed(&embedding));
    }

    #[test]
    fn load_modes() {
        let mut a = base();
        a.n_gpu_layers = 20;
        a.mmap = false;
        let s = joined(&a);
        assert!(s.contains("-ngl 20") && s.contains("-lm none"), "{s}");
        a.mlock = true;
        assert!(joined(&a).contains("-lm mlock"));
        a.mmap = true;
        assert!(joined(&a).contains("-lm mmap+mlock"));
    }
}
