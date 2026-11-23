//! Qwen3-TTS voice cloning via the qwentts.cpp `qwen-tts` CLI.
//!
//! The `tts-server` sidecar's HTTP API only accepts
//! `input`/`voice`/`instructions`/`response_format`/`speed` — voice cloning
//! (`--ref-wav` + `--ref-text`) is exposed exclusively by the one-shot
//! `qwen-tts` CLI, and only **Base** talkers carry the speaker-encoder
//! weights it needs (the runtime rejects `--ref-wav` for custom_voice /
//! voice_design models). So a clone job here is: gate VRAM through the
//! orchestrator as a *transient* resident, run `qwen-tts.exe` once (it loads
//! talker + codec itself, synthesizes, writes a WAV, exits), then release
//! the reservation. Terminal events ride the same worker job-event bus the
//! sidecar path uses (`task: None` on JobDone), so progress streaming, DB
//! sync, and asset promotion all work unchanged.
//!
//! Note the CLI loads its *own* copy of the talker + codec — if the
//! tts-server sidecar happens to hold the same talker resident, the clone
//! job genuinely needs that VRAM *again* for its duration, which is why the
//! reservation deliberately does not try to dedupe against the `"tts"`
//! resident.
//!
//! Like `tts-server.exe`, the CLI binary isn't shipped yet (no upstream
//! Windows builds — M14 hosting debt); it's hand-copied into
//! `engine\qwentts\` and everything here degrades to honest errors when
//! it's absent.

use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use serde_json::Value;
use sqlx::SqlitePool;
use tokio::sync::oneshot;

use crate::db::generation_jobs;
use crate::error::AppError;
use crate::generation::SubmitOutcome;
use crate::models::registry;
use crate::orchestrator::{JobLease, OrchestratorHandle, ReserveOutcome};
use crate::worker::protocol::WorkerEvent;
use crate::worker::WorkerHandle;

/// Ledger task name for the transient clone reservation
/// (resident id `pipeline:qwen_tts_clone`).
const CLONE_TASK: &str = "qwen_tts_clone";

/// The CLI loads multi-GB GGUFs per invocation and long texts synthesize
/// slowly (minutes on CPU fallback) — time out generously.
const CLONE_TIMEOUT: Duration = Duration::from_secs(30 * 60);

/// `engine\qwentts\qwen-tts.exe` — sits next to tts-server.exe.
pub fn cli_path() -> PathBuf {
    super::install_dir().join(super::manifest::CLONE_CLI_FILE)
}

/// Whether voice cloning is available: the synthesis runtime is installed
/// **and** the one-shot `qwen-tts.exe` CLI is present next to it. The CLI is
/// the one qwentts file deliberately left out of the synthesis `required_files`
/// (cloning is an optional capability — only Base talkers can clone), so its
/// presence is checked here explicitly via `cli_path()` rather than through
/// `tts::is_installed`. A synthesis-only install (no `qwen-tts.exe`) is fully
/// usable for synthesis; it just reports cloning as unavailable here.
pub fn cli_installed() -> bool {
    cli_path().exists() && super::is_installed()
}

/// Cancel signals for in-flight clone jobs, keyed by job id. A tiny static
/// registry rather than AppState plumbing: clone jobs are detached one-shot
/// processes, and `cancel_generation_job` only needs a way to reach them.
static CANCELS: LazyLock<Mutex<HashMap<String, oneshot::Sender<()>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Request cancellation of a running clone job. Returns true when the job
/// was found (the kill + terminal `cancelled` event follow asynchronously).
pub fn cancel(job_id: &str) -> bool {
    let tx = CANCELS.lock().expect("cancel registry poisoned").remove(job_id);
    match tx {
        Some(tx) => tx.send(()).is_ok(),
        None => false,
    }
}

fn register_cancel(job_id: &str) -> oneshot::Receiver<()> {
    let (tx, rx) = oneshot::channel();
    CANCELS
        .lock()
        .expect("cancel registry poisoned")
        .insert(job_id.to_string(), tx);
    rx
}

fn unregister_cancel(job_id: &str) {
    CANCELS.lock().expect("cancel registry poisoned").remove(job_id);
}

/// Build the `qwen-tts` invocation for an ICL voice clone. The utterance
/// text goes to stdin (the CLI intentionally takes no text argv);
/// `ref_text` is a *file path* to the reference transcript.
fn build_clone_args(
    talker: &Path,
    codec: &Path,
    ref_wav: &Path,
    ref_text_file: &Path,
    out_wav: &Path,
) -> Vec<OsString> {
    vec![
        OsString::from("--model"),
        talker.into(),
        OsString::from("--codec"),
        codec.into(),
        OsString::from("--ref-wav"),
        ref_wav.into(),
        OsString::from("--ref-text"),
        ref_text_file.into(),
        OsString::from("-o"),
        out_wav.into(),
    ]
}

#[derive(Debug)]
struct CloneRequest {
    model_id: String,
    text: String,
    ref_wav: PathBuf,
    ref_text: String,
}

/// Validate the synthesis params into a clone request. Pure-ish (only
/// string shape checks) so it unit-tests without a filesystem.
fn parse_params(params: &Value) -> Result<CloneRequest, AppError> {
    let model_id = params
        .get("model")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            AppError::Other("Qwen3-TTS needs a talker model — pick one first".into())
        })?
        .to_string();
    let text = params
        .get("text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if text.is_empty() {
        return Err(AppError::Other("nothing to speak — text is empty".into()));
    }
    let ref_wav = params
        .get("reference_audio")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            AppError::Other("voice cloning needs a reference audio file".into())
        })?;
    let ref_text = params
        .get("ref_text")
        .and_then(|v| v.as_str())
        .unwrap_or_default()
        .trim()
        .to_string();
    if ref_text.is_empty() {
        return Err(AppError::Other(
            "voice cloning needs the reference transcript — type what the sample says".into(),
        ));
    }
    Ok(CloneRequest {
        model_id,
        text,
        ref_wav: PathBuf::from(ref_wav),
        ref_text,
    })
}

/// Submit a Qwen3-TTS voice-clone job (the `engine: "qwen3-tts"` +
/// `reference_audio` branch of `synthesize_speech`). Mirrors
/// `generation::submit_qwen_tts`'s shape: reserve → persist → run in a
/// background task that reports through the worker job-event bus.
pub async fn submit_clone(
    db: &SqlitePool,
    worker: &WorkerHandle,
    orchestrator: &OrchestratorHandle,
    tts: &super::TtsHandle,
    params: Value,
    force: bool,
) -> Result<SubmitOutcome, AppError> {
    let req = parse_params(&params)?;

    if !cli_installed() {
        // Lazy auto-install: kick off the pin-and-fetch provision in the
        // background (progress streams to the settings card) and ask the user
        // to retry once it's ready. Degrades to an honest error when the
        // binaries aren't hosted yet — the installer reports that on the card.
        tts.install_in_background();
        return Err(AppError::Other(
            "Voice cloning needs the Qwen3-TTS engine (qwen-tts.exe) — installing it now. \
             Watch progress in Settings → Engines, then retry. (If the install fails, the \
             binaries are not yet hosted — see CLAUDE.md.)"
                .into(),
        ));
    }
    if !req.ref_wav.exists() {
        return Err(AppError::Other(format!(
            "reference audio not found: {}",
            req.ref_wav.display()
        )));
    }

    let model = registry::get_by_id(db, &req.model_id)
        .await?
        .ok_or_else(|| AppError::ModelNotFound(req.model_id.clone()))?;
    if model.architecture.as_deref() != Some(super::TALKER_ARCH) {
        return Err(AppError::Other(format!(
            "'{}' is not a Qwen3-TTS talker model (architecture {:?})",
            model.display_name, model.architecture
        )));
    }

    let talker = PathBuf::from(&model.primary_file_path);
    let codec = super::bundled_codec_path();

    // Transient VRAM reservation, sized like the sidecar load (talker +
    // codec + working overhead) — reserved before the run, released after.
    let codec_mb = codec.metadata().map(|m| m.len() / (1024 * 1024)).unwrap_or(300);
    let est_mb =
        model.primary_file_size / (1024 * 1024) + codec_mb + super::QWEN_TTS_OVERHEAD_MB;
    // Lease granted with the reservation, held until the CLI run ends.
    let job_id = uuid::Uuid::new_v4().to_string();
    match orchestrator
        .reserve_pipeline(
            CLONE_TASK.to_string(),
            format!("{} (voice clone)", model.display_name),
            Some(model.id.clone()),
            est_mb,
            force,
            Some(JobLease::clone_cli(&job_id)),
        )
        .await?
    {
        ReserveOutcome::Reserved => {}
        ReserveOutcome::NeedsEviction { required_mb, free_mb, candidates } => {
            return Ok(SubmitOutcome::NeedsEviction { required_mb, free_mb, candidates });
        }
    }

    if let Err(e) = generation_jobs::insert(db, &job_id, "text_to_speech", &params).await {
        orchestrator.abort_lease(job_id, e.to_string()).await;
        return Err(e);
    }

    let events = worker.job_events_sender();
    let orchestrator = orchestrator.clone();
    let jid = job_id.clone();
    let cancel_rx = register_cancel(&job_id);
    tauri::async_runtime::spawn(async move {
        let _ = events.send(WorkerEvent::JobProgress {
            job_id: jid.clone(),
            pct: 10.0,
            stage: "cloning voice (qwen-tts loads the model per run — this can take a while)"
                .into(),
            step: None,
            total_steps: None,
        });

        let staging = crate::config::assets_staging_dir();
        let out_wav = staging.join(format!("speech-{jid}.wav"));
        let ref_text_file = staging.join(format!("ref-text-{jid}.txt"));
        let result = run_clone(
            &talker,
            &codec,
            &req.ref_wav,
            &req.ref_text,
            &ref_text_file,
            &req.text,
            &out_wav,
            cancel_rx,
        )
        .await;

        unregister_cancel(&jid);
        let _ = tokio::fs::remove_file(&ref_text_file).await;
        // Release the transient reservation whatever happened — the CLI has
        // exited and its VRAM is back.
        let _ = orchestrator.unload(format!("pipeline:{CLONE_TASK}")).await;

        match result {
            Ok(()) => {
                let _ = events.send(WorkerEvent::JobDone {
                    job_id: jid,
                    outputs: vec![out_wav.to_string_lossy().into_owned()],
                    meta: serde_json::json!({
                        "engine": "qwen3-tts",
                        "mode": "clone",
                        "sample_rate": 24000,
                    }),
                    // task: None — keeps the orchestrator's pipeline
                    // bookkeeping out of it, same as the sidecar path.
                    task: None,
                });
            }
            Err(e) => {
                let _ = events.send(WorkerEvent::JobError { job_id: jid, error: e.to_string() });
            }
        }
    });

    Ok(SubmitOutcome::Submitted { job_id })
}

/// Run the CLI to completion: write the transcript file, pipe the utterance
/// text via stdin, race completion against cancel + timeout, and verify the
/// output WAV landed. Stderr is captured into the error on failure.
#[allow(clippy::too_many_arguments)]
async fn run_clone(
    talker: &Path,
    codec: &Path,
    ref_wav: &Path,
    ref_text: &str,
    ref_text_file: &Path,
    text: &str,
    out_wav: &Path,
    mut cancel_rx: oneshot::Receiver<()>,
) -> Result<(), AppError> {
    use std::process::Stdio;
    use tokio::io::AsyncWriteExt;

    if let Some(parent) = out_wav.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(ref_text_file, ref_text).await?;

    let mut cmd = tokio::process::Command::new(cli_path());
    cmd.args(build_clone_args(talker, codec, ref_wav, ref_text_file, out_wav));
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    cmd.kill_on_drop(true);
    #[cfg(windows)]
    {
        #[allow(unused_imports)]
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    let mut child = cmd
        .spawn()
        .map_err(|e| AppError::Other(format!("could not start qwen-tts.exe: {e}")))?;
    #[cfg(windows)]
    crate::winjob::tie_to_app_lifetime(&child, "qwen-tts");

    // Feed the utterance text and close stdin so the CLI starts synthesizing.
    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(text.as_bytes())
            .await
            .map_err(|e| AppError::Other(format!("could not send text to qwen-tts: {e}")))?;
        drop(stdin);
    }

    // Drain stderr concurrently (avoids pipe-full deadlock) and keep a tail
    // for the error message.
    let stderr_tail = match child.stderr.take() {
        Some(stderr) => tokio::spawn(async move {
            use tokio::io::{AsyncBufReadExt, BufReader};
            let mut tail: Vec<String> = Vec::new();
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::debug!(target: "qwen-tts", "{}", line);
                if tail.len() == 20 {
                    tail.remove(0);
                }
                tail.push(line);
            }
            tail.join("\n")
        }),
        None => tokio::spawn(async { String::new() }),
    };

    let status = tokio::select! {
        status = child.wait() => {
            status.map_err(|e| AppError::Other(format!("qwen-tts wait failed: {e}")))?
        }
        _ = &mut cancel_rx => {
            let _ = child.kill().await;
            return Err(AppError::Other("cancelled".into()));
        }
        _ = tokio::time::sleep(CLONE_TIMEOUT) => {
            let _ = child.kill().await;
            return Err(AppError::Other("qwen-tts timed out".into()));
        }
    };

    let tail = stderr_tail.await.unwrap_or_default();
    if !status.success() {
        let detail = if tail.is_empty() { "(no stderr output)".into() } else { tail };
        return Err(AppError::Other(format!("qwen-tts failed ({status}): {detail}")));
    }
    let wav_ok = tokio::fs::metadata(out_wav).await.map(|m| m.len() > 0).unwrap_or(false);
    if !wav_ok {
        return Err(AppError::Other(format!(
            "qwen-tts exited 0 but produced no audio: {tail}"
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cli_lives_next_to_tts_server() {
        assert_eq!(cli_path().file_name().unwrap(), "qwen-tts.exe");
        assert_eq!(cli_path().parent(), super::super::binary_path().parent());
    }

    #[test]
    fn clone_args_shape() {
        let args = build_clone_args(
            Path::new("C:\\models\\talker.gguf"),
            Path::new("C:\\engine\\codec.gguf"),
            Path::new("C:\\refs\\sample.wav"),
            Path::new("C:\\staging\\ref.txt"),
            Path::new("C:\\staging\\out.wav"),
        );
        let s: Vec<String> = args.iter().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(
            s,
            vec![
                "--model",
                "C:\\models\\talker.gguf",
                "--codec",
                "C:\\engine\\codec.gguf",
                "--ref-wav",
                "C:\\refs\\sample.wav",
                "--ref-text",
                "C:\\staging\\ref.txt",
                "-o",
                "C:\\staging\\out.wav",
            ]
        );
        // The utterance text rides stdin, never argv — no positional args,
        // no --speaker/--instruct (rejected for base talkers).
        assert!(!s.iter().any(|a| a == "--speaker" || a == "--instruct"));
    }

    #[test]
    fn parse_params_happy_path() {
        let p = json!({
            "model": "abc123",
            "text": "  Hello there.  ",
            "engine": "qwen3-tts",
            "reference_audio": "C:\\refs\\me.wav",
            "ref_text": "This is what the sample says.",
        });
        let req = parse_params(&p).unwrap();
        assert_eq!(req.model_id, "abc123");
        assert_eq!(req.text, "Hello there.");
        assert_eq!(req.ref_wav, PathBuf::from("C:\\refs\\me.wav"));
        assert_eq!(req.ref_text, "This is what the sample says.");
    }

    #[test]
    fn parse_params_rejects_missing_fields() {
        let base = json!({
            "model": "abc",
            "text": "hi",
            "reference_audio": "C:\\r.wav",
            "ref_text": "words",
        });
        for (key, msg) in [
            ("model", "talker model"),
            ("text", "text is empty"),
            ("reference_audio", "reference audio"),
            ("ref_text", "reference transcript"),
        ] {
            let mut p = base.clone();
            p.as_object_mut().unwrap().remove(key);
            let err = parse_params(&p).unwrap_err().to_string();
            assert!(err.contains(msg), "removing {key}: got '{err}'");
        }
        // Whitespace-only counts as missing too.
        let mut p = base.clone();
        p["ref_text"] = json!("   ");
        assert!(parse_params(&p).is_err());
    }

    #[test]
    fn cancel_unknown_job_is_false() {
        assert!(!cancel("no-such-job"));
    }

    #[test]
    fn cancel_registered_job_fires_once() {
        let rx = register_cancel("job-x");
        assert!(cancel("job-x"));
        assert!(rx.blocking_recv().is_ok());
        assert!(!cancel("job-x"), "second cancel finds nothing");
    }
}
