//! Store intelligence (M9 §9.4): runnability resolver + curated picks.
//!
//! The resolver answers "can OpenHarness run this repo?" from card-level
//! signals (pipeline_tag, library, tags) plus the layout classification when
//! a detail pane has one. Verdicts are honest — a repo that downloads fine
//! but has no local runtime is badged "not runnable here", never hidden.

use serde::{Deserialize, Serialize};

use crate::models::layout::LayoutKind;
use crate::worker::PipelineCapability;

// ---------------------------------------------------------------------------
// Runnability
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum Runtime {
    /// llama-server sidecar (GGUF text/embedding models).
    LlamaCpp,
    /// Python worker pipeline (diffusers / faster-whisper / TTS).
    Worker,
}

#[derive(Debug, Clone, Serialize)]
pub struct Runnability {
    pub runnable: bool,
    pub runtime: Option<Runtime>,
    /// Worker pipeline task when `runtime == Worker` (e.g. "text_to_image").
    pub pipeline: Option<String>,
    /// Pipeline deps currently importable (always true for llama.cpp —
    /// the engine wizard gates that path elsewhere).
    pub available: bool,
    /// Honest badge/tooltip copy.
    pub reason: String,
}

/// One search-result row (or detail pane) to resolve. Card-level fields are
/// enough for list badges; `layout` sharpens the verdict on detail pages.
#[derive(Debug, Clone, Deserialize)]
pub struct RunnabilityQuery {
    #[serde(default)]
    pub repo_id: Option<String>,
    /// HF pipeline_tag.
    #[serde(default)]
    pub task: Option<String>,
    /// HF library_name.
    #[serde(default)]
    pub library: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    #[serde(default)]
    pub layout: Option<LayoutKind>,
}

/// Model-format hint derived from layout (exact) or card signals (guess).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FormatHint {
    Gguf,
    DiffusersDir,
    SafetensorsSingle,
    /// Safetensors present but single-vs-sharded unknown (card-level guess).
    Safetensors,
    Ct2Dir,
    KokoroRepo,
    Unknown,
}

fn format_hint(q: &RunnabilityQuery) -> FormatHint {
    if let Some(layout) = q.layout {
        match layout {
            LayoutKind::Gguf => return FormatHint::Gguf,
            LayoutKind::DiffusersDir => return FormatHint::DiffusersDir,
            LayoutKind::SafetensorsSingle => return FormatHint::SafetensorsSingle,
            LayoutKind::SafetensorsShards => return FormatHint::Safetensors,
            // Unknown is what CTranslate2 dirs (model.bin) and Kokoro repos
            // (.pth) classify as — fall through to card-level signals rather
            // than letting "no recognised weights" override them.
            LayoutKind::Unknown => {}
        }
    }

    let lib = q.library.as_deref().unwrap_or("");
    let repo = q.repo_id.as_deref().unwrap_or("").to_ascii_lowercase();
    let has_tag = |t: &str| q.tags.iter().any(|x| x.eq_ignore_ascii_case(t));

    if lib == "gguf" || has_tag("gguf") {
        FormatHint::Gguf
    } else if lib == "diffusers" {
        FormatHint::DiffusersDir
    } else if lib == "ctranslate2" || has_tag("ctranslate2") || repo.contains("faster-whisper") {
        FormatHint::Ct2Dir
    } else if repo.contains("kokoro") || has_tag("kokoro") {
        FormatHint::KokoroRepo
    } else if has_tag("safetensors") || lib == "transformers" || lib == "sentence-transformers" {
        FormatHint::Safetensors
    } else {
        FormatHint::Unknown
    }
}

/// Worker-pipeline format string for a hint, in the vocabulary the Python
/// worker declares in `PipelineCapability.formats`.
fn worker_format_str(hint: FormatHint) -> Option<&'static str> {
    match hint {
        FormatHint::DiffusersDir => Some("diffusers-dir"),
        FormatHint::SafetensorsSingle => Some("safetensors-single-file"),
        FormatHint::Ct2Dir => Some("ctranslate2-dir"),
        FormatHint::KokoroRepo => Some("kokoro-repo"),
        _ => None,
    }
}

/// Actionable verdict for a non-GGUF Qwen3-TTS repo (kept in sync with
/// `capabilities::QWEN_TTS_SAFETENSORS_REASON`). The Store can't read GGUF
/// architecture pre-download, so this keys off the repo name only.
const QWEN_TTS_SAFETENSORS_REASON: &str =
    "This is the original safetensors build — OpenHarness runs Qwen3-TTS from its GGUF build instead. Download the GGUF version of this voice from the Store.";

/// The upstream Qwen3-TTS release is safetensors (`text-to-speech`), which has
/// no local runtime — Qwen3-TTS only runs through the qwentts.cpp GGUF sidecar.
/// Detect the *family* by repo name (NOT a specific repo): collapse separators
/// so "qwen3-tts", "qwen3_tts" and "qwen-3-tts" all reduce to `qwen3tts`.
fn is_qwen_tts_repo(q: &RunnabilityQuery) -> bool {
    let repo = q.repo_id.as_deref().unwrap_or("").to_ascii_lowercase();
    let collapsed: String = repo.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    collapsed.contains("qwen3tts") || collapsed.contains("qwentts")
}

/// HF task → candidate worker pipeline.
fn pipeline_for_task(task: &str) -> Option<&'static str> {
    match task {
        "text-to-image" | "image-to-image" => Some("text_to_image"),
        "text-to-video" | "image-to-video" => Some("text_to_video"),
        "automatic-speech-recognition" => Some("transcribe"),
        "text-to-speech" => Some("text_to_speech"),
        _ => None,
    }
}

/// Static mirror of the Python worker's pipeline table — used for badges
/// when the worker isn't running. Keep in sync with
/// `worker-py/openharness_worker/pipelines/__init__.py`.
pub fn static_pipelines() -> Vec<PipelineCapability> {
    let entry = |task: &str, formats: &[&str]| PipelineCapability {
        task: task.to_string(),
        summary: String::new(),
        available: false, // unknown until the worker reports in
        missing: vec![],
        hidden: false,
        formats: formats.iter().map(|s| s.to_string()).collect(),
    };
    vec![
        entry("text_to_image", &["diffusers-dir", "safetensors-single-file"]),
        entry("transcribe", &["ctranslate2-dir", "whisper-size-name"]),
        entry("text_to_speech", &["kokoro-repo"]),
        entry("text_to_video", &["diffusers-dir"]),
    ]
}

/// Resolve a single query against the (live or static) pipeline table.
pub fn resolve(q: &RunnabilityQuery, pipelines: &[PipelineCapability]) -> Runnability {
    let hint = format_hint(q);
    let task = q.task.as_deref().unwrap_or("");

    // llama.cpp path: GGUF text generation / embeddings. A GGUF repo with no
    // pipeline_tag is overwhelmingly a text-generation quant dump.
    if hint == FormatHint::Gguf {
        return match task {
            "text-generation" | "" => Runnability {
                runnable: true,
                runtime: Some(Runtime::LlamaCpp),
                pipeline: None,
                available: true,
                reason: "Runs on llama.cpp".into(),
            },
            "feature-extraction" | "sentence-similarity" => Runnability {
                runnable: true,
                runtime: Some(Runtime::LlamaCpp),
                pipeline: None,
                available: true,
                reason: "Runs on llama.cpp (embeddings)".into(),
            },
            other => Runnability {
                runnable: false,
                runtime: None,
                pipeline: None,
                available: false,
                reason: format!("GGUF {other} models aren't supported by the local runtime"),
            },
        };
    }

    // Non-GGUF Qwen3-TTS = the original safetensors release, which has no local
    // runtime (it would otherwise hit the text-to-speech/Kokoro arm and report a
    // cryptic "format not supported"). GGUF Qwen3-TTS repos took the branch above
    // and are untouched. Refuse with a path to the runnable GGUF build.
    if is_qwen_tts_repo(q) {
        return Runnability {
            runnable: false,
            runtime: None,
            pipeline: None,
            available: false,
            reason: QWEN_TTS_SAFETENSORS_REASON.into(),
        };
    }

    // Worker path: task maps to a pipeline AND the format is accepted.
    if let Some(pipeline) = pipeline_for_task(task) {
        let cap = pipelines.iter().find(|c| c.task == pipeline);
        let format = worker_format_str(hint);
        match (cap, format) {
            (Some(cap), Some(fmt)) if cap.formats.iter().any(|f| f == fmt) => {
                return Runnability {
                    runnable: true,
                    runtime: Some(Runtime::Worker),
                    pipeline: Some(pipeline.to_string()),
                    available: cap.available,
                    reason: if cap.available {
                        format!("Runs on the {} pipeline", pipeline.replace('_', "-"))
                    } else {
                        format!(
                            "Runs on the {} pipeline (worker not running)",
                            pipeline.replace('_', "-")
                        )
                    },
                };
            }
            (Some(_), _) => {
                return Runnability {
                    runnable: false,
                    runtime: None,
                    pipeline: Some(pipeline.to_string()),
                    available: false,
                    reason: "Model format not supported by the local pipeline".into(),
                };
            }
            _ => {}
        }
    }

    Runnability {
        runnable: false,
        runtime: None,
        pipeline: None,
        available: false,
        reason: if task.is_empty() {
            "No local runtime for this model".into()
        } else {
            format!("No local runtime for {task}")
        },
    }
}

// ---------------------------------------------------------------------------
// Curated picks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct CuratedPick {
    pub repo_id: String,
    pub display_name: String,
    /// HF pipeline_tag-style task — matches the store tab filter.
    pub task: String,
    /// gguf | diffusers-dir | ctranslate2-dir | kokoro-repo
    pub format: String,
    /// Recommended file for single-file (GGUF) picks; None for dir layouts.
    pub recommended_file: Option<String>,
    pub min_vram_mb: u64,
    pub blurb: String,
    /// Whether the pick fits the detected GPU (total VRAM ≥ min tier).
    /// False when no NVIDIA GPU is present.
    pub fits: bool,
}

struct PickDef {
    repo_id: &'static str,
    display_name: &'static str,
    task: &'static str,
    format: &'static str,
    recommended_file: Option<&'static str>,
    min_vram_mb: u64,
    blurb: &'static str,
}

/// Static manifest baked into the binary. Per-modality, ordered best-first
/// within a tier. Repo ids are well-known, ungated, permissively licensed.
const PICKS: &[PickDef] = &[
    // --- text-generation (GGUF, llama.cpp) ---
    PickDef {
        repo_id: "Qwen/Qwen2.5-7B-Instruct-GGUF",
        display_name: "Qwen2.5 7B Instruct",
        task: "text-generation",
        format: "gguf",
        recommended_file: Some("qwen2.5-7b-instruct-q4_k_m.gguf"),
        min_vram_mb: 6144,
        blurb: "Strong all-round chat model; the Q4_K_M quant fits 8 GB cards.",
    },
    PickDef {
        repo_id: "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
        display_name: "Llama 3.1 8B Instruct",
        task: "text-generation",
        format: "gguf",
        recommended_file: Some("Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
        min_vram_mb: 6144,
        blurb: "Meta's general-purpose 8B chat model.",
    },
    PickDef {
        repo_id: "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        display_name: "Qwen2.5 1.5B Instruct",
        task: "text-generation",
        format: "gguf",
        recommended_file: Some("qwen2.5-1.5b-instruct-q4_k_m.gguf"),
        min_vram_mb: 2048,
        blurb: "Tiny but capable — runs on almost anything, even CPU.",
    },
    // --- text-to-image (diffusers, worker) ---
    PickDef {
        repo_id: "stabilityai/sdxl-turbo",
        display_name: "SDXL Turbo",
        task: "text-to-image",
        format: "diffusers-dir",
        recommended_file: None,
        min_vram_mb: 8192,
        blurb: "One-step SDXL — near-instant drafts at 512px.",
    },
    PickDef {
        repo_id: "stabilityai/stable-diffusion-xl-base-1.0",
        display_name: "SDXL Base 1.0",
        task: "text-to-image",
        format: "diffusers-dir",
        recommended_file: None,
        min_vram_mb: 10240,
        blurb: "The standard high-quality 1024px image model.",
    },
    PickDef {
        repo_id: "black-forest-labs/FLUX.1-schnell",
        display_name: "FLUX.1 schnell",
        task: "text-to-image",
        format: "diffusers-dir",
        recommended_file: None,
        min_vram_mb: 20480,
        blurb: "State-of-the-art open image gen; wants a 24 GB card.",
    },
    // --- text-to-video (diffusers, worker) ---
    PickDef {
        repo_id: "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        display_name: "Wan 2.1 T2V 1.3B",
        task: "text-to-video",
        format: "diffusers-dir",
        recommended_file: None,
        min_vram_mb: 8192,
        blurb: "Small text-to-video that runs on consumer cards.",
    },
    PickDef {
        repo_id: "Lightricks/LTX-Video",
        display_name: "LTX-Video",
        task: "text-to-video",
        format: "diffusers-dir",
        recommended_file: None,
        min_vram_mb: 20480,
        blurb: "Fast, high-quality clips; wants a 24 GB card.",
    },
    // --- text-to-speech (worker) ---
    PickDef {
        repo_id: "hexgrad/Kokoro-82M",
        display_name: "Kokoro 82M",
        task: "text-to-speech",
        format: "kokoro-repo",
        recommended_file: None,
        min_vram_mb: 1024,
        blurb: "Excellent lightweight TTS with multiple voices.",
    },
    // Qwen3-TTS runs only as a GGUF talker through the qwentts.cpp sidecar (the
    // bundled codec GGUF ships with the engine), so the runnable pick is this
    // GGUF build, not the upstream safetensors release. CustomVoice = preset
    // speakers + style instructions.
    PickDef {
        repo_id: "Serveurperso/Qwen3-TTS-GGUF",
        display_name: "Qwen3-TTS 1.7B (CustomVoice)",
        task: "text-to-speech",
        format: "gguf",
        recommended_file: Some("qwen-talker-1.7b-customvoice-Q8_0.gguf"),
        min_vram_mb: 4096,
        blurb: "Expressive multi-voice TTS; the GGUF build OpenHarness runs.",
    },
    // --- automatic-speech-recognition (worker) ---
    PickDef {
        repo_id: "Systran/faster-whisper-large-v3",
        display_name: "Whisper Large v3 (faster-whisper)",
        task: "automatic-speech-recognition",
        format: "ctranslate2-dir",
        recommended_file: None,
        min_vram_mb: 4096,
        blurb: "Best open transcription quality, with timestamps.",
    },
    PickDef {
        repo_id: "Systran/faster-whisper-small",
        display_name: "Whisper Small (faster-whisper)",
        task: "automatic-speech-recognition",
        format: "ctranslate2-dir",
        recommended_file: None,
        min_vram_mb: 1024,
        blurb: "Quick transcription on modest hardware.",
    },
    // --- feature-extraction (GGUF embeddings, llama.cpp) ---
    PickDef {
        repo_id: "nomic-ai/nomic-embed-text-v1.5-GGUF",
        display_name: "Nomic Embed Text v1.5",
        task: "feature-extraction",
        format: "gguf",
        recommended_file: None,
        min_vram_mb: 1024,
        blurb: "Solid long-context text embeddings.",
    },
];

/// Materialize the curated manifest, optionally filtered to one task, with
/// `fits` resolved against the detected VRAM (`None` = no NVIDIA GPU).
pub fn curated_picks(task_filter: Option<&str>, vram_total_mb: Option<u64>) -> Vec<CuratedPick> {
    PICKS
        .iter()
        .filter(|p| task_filter.is_none_or(|t| p.task == t))
        .map(|p| CuratedPick {
            repo_id: p.repo_id.to_string(),
            display_name: p.display_name.to_string(),
            task: p.task.to_string(),
            format: p.format.to_string(),
            recommended_file: p.recommended_file.map(String::from),
            min_vram_mb: p.min_vram_mb,
            blurb: p.blurb.to_string(),
            fits: vram_total_mb.is_some_and(|v| v >= p.min_vram_mb),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// VRAM-aware quant fits (M9 §9.6)
// ---------------------------------------------------------------------------

/// Per-GGUF-file fit verdict for a repo's file tree, plus the "best for my
/// GPU" pick: the largest single-file quant that still fits free VRAM.
#[derive(Debug, Clone, Serialize)]
pub struct QuantFit {
    pub path: String,
    /// Parsed quant label (e.g. "Q4_K_M", "IQ3_XS", "F16"); None when the
    /// filename has no recognizable quant token.
    pub label: Option<String>,
    pub size: u64,
    /// Estimated VRAM in MiB: file size + a flat KV allowance + the fixed
    /// backend overhead. Pre-download we have no GGUF metadata, so this is
    /// coarser than the installed-model estimator — verdicts are banded
    /// accordingly (the `Tight` band absorbs most of the error).
    pub estimated_mb: u64,
    pub verdict: crate::engine::fit::FitVerdict,
    /// True for exactly one entry (when any fits): the recommended one-click.
    pub best: bool,
    /// Part of a multi-file split GGUF (`-00001-of-00003.gguf`). Estimated
    /// over the summed group size; never selected as `best` because the
    /// one-click flow downloads a single file.
    pub split: bool,
    /// Repo-relative set stem and aggregate download size; None for single files.
    pub set_key: Option<String>,
    pub set_size: Option<u64>,
    pub part_count: Option<u32>,
}

/// Flat KV-cache allowance (MiB) for pre-download estimates — roughly an 8k
/// context on a 7–14B GQA model. Without GGUF metadata we can't do better.
const QUANT_KV_ALLOWANCE_MB: u64 = 768;

/// Parse a quant label token out of a GGUF filename.
fn quant_label(path: &str) -> Option<String> {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    for token in stem.split(['-', '.', ' ']) {
        let upper = token.to_ascii_uppercase();
        let core = upper.strip_prefix("IQ").or_else(|| upper.strip_prefix('Q'));
        let is_quant = core.is_some_and(|c| c.starts_with(|ch: char| ch.is_ascii_digit()))
            && upper.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
        if is_quant || upper == "F16" || upper == "BF16" || upper == "F32" {
            return Some(upper);
        }
    }
    None
}

fn split_group_stem(path: &str) -> Option<String> {
    crate::models::split_gguf::part(path).map(|p| p.stem.to_string())
}

/// Fit verdicts for every GGUF file in a repo tree against free VRAM, with
/// the largest fitting single-file quant flagged `best`. Split parts get a
/// verdict computed over their summed group size. Non-GGUF files are skipped.
pub fn quant_fits(files: &[crate::models::hf::HfFile], free_mb: u64) -> Vec<QuantFit> {
    use crate::engine::fit::{verdict, FitVerdict, FIXED_OVERHEAD_MB};

    // Sum split groups first so parts share one honest estimate.
    let mut group_totals: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for f in files {
        if let Some(stem) = split_group_stem(&f.path) {
            *group_totals.entry(stem).or_insert(0) += f.size;
        }
    }

    let mut fits: Vec<QuantFit> = files
        .iter()
        .filter(|f| f.path.ends_with(".gguf"))
        .map(|f| {
            let split_stem = split_group_stem(&f.path);
            let effective_size = split_stem
                .as_ref()
                .and_then(|s| group_totals.get(s).copied())
                .unwrap_or(f.size);
            let estimated_mb = (effective_size / (1024 * 1024))
                .saturating_add(QUANT_KV_ALLOWANCE_MB)
                .saturating_add(FIXED_OVERHEAD_MB);
            QuantFit {
                path: f.path.clone(),
                label: quant_label(&f.path),
                size: f.size,
                estimated_mb,
                verdict: verdict(estimated_mb, free_mb),
                best: false,
                split: split_stem.is_some(),
                set_key: split_stem.clone(),
                set_size: split_stem.as_ref().map(|_| effective_size),
                part_count: crate::models::split_gguf::part(&f.path).map(|p| p.count),
            }
        })
        .collect();

    // Best = largest single-file quant with comfortable headroom; fall back
    // to the largest Tight one only when nothing fits comfortably.
    let largest = |fits: &[QuantFit], want: FitVerdict| {
        fits.iter()
            .enumerate()
            .filter(|(_, f)| !f.split && f.verdict == want)
            .max_by_key(|(_, f)| f.size)
            .map(|(i, _)| i)
    };
    let best_idx = largest(&fits, FitVerdict::Fits).or_else(|| largest(&fits, FitVerdict::Tight));
    if let Some(i) = best_idx {
        fits[i].best = true;
    }
    fits
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(
        repo_id: &str,
        task: Option<&str>,
        library: Option<&str>,
        tags: &[&str],
        layout: Option<LayoutKind>,
    ) -> RunnabilityQuery {
        RunnabilityQuery {
            repo_id: Some(repo_id.to_string()),
            task: task.map(String::from),
            library: library.map(String::from),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            layout,
        }
    }

    #[test]
    fn gguf_text_generation_runs_on_llama_cpp() {
        let r = resolve(
            &q("Qwen/Qwen2.5-7B-Instruct-GGUF", Some("text-generation"), Some("gguf"), &[], None),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.runtime, Some(Runtime::LlamaCpp));
    }

    #[test]
    fn gguf_without_task_assumed_text_generation() {
        let r = resolve(
            &q("org/some-GGUF", None, None, &["gguf"], None),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.runtime, Some(Runtime::LlamaCpp));
    }

    #[test]
    fn diffusers_text_to_image_runs_on_worker() {
        let r = resolve(
            &q("stabilityai/sdxl-turbo", Some("text-to-image"), Some("diffusers"), &[], None),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.runtime, Some(Runtime::Worker));
        assert_eq!(r.pipeline.as_deref(), Some("text_to_image"));
        // static table: worker not confirmed running
        assert!(!r.available);
    }

    #[test]
    fn layout_sharpens_card_guess() {
        // Card says transformers, but the detail-pane layout knows it's a
        // diffusers dir → worker verdict.
        let r = resolve(
            &q("org/repo", Some("text-to-image"), Some("transformers"), &[],
               Some(LayoutKind::DiffusersDir)),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.pipeline.as_deref(), Some("text_to_image"));
    }

    #[test]
    fn faster_whisper_transcribe_runs_on_worker() {
        let r = resolve(
            &q("Systran/faster-whisper-large-v3", Some("automatic-speech-recognition"),
               Some("ctranslate2"), &[], None),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.pipeline.as_deref(), Some("transcribe"));
    }

    #[test]
    fn kokoro_tts_runs_on_worker() {
        let r = resolve(
            &q("hexgrad/Kokoro-82M", Some("text-to-speech"), None, &[], None),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.pipeline.as_deref(), Some("text_to_speech"));
    }

    #[test]
    fn qwen3_tts_safetensors_repo_is_not_runnable_with_gguf_guidance() {
        // Original upstream release: safetensors, "text-to-speech". Would
        // otherwise hit the Kokoro pipeline arm and report a cryptic
        // "format not supported" — give an actionable GGUF pointer instead.
        let r = resolve(
            &q(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                Some("text-to-speech"),
                Some("transformers"),
                &["safetensors"],
                None,
            ),
            &static_pipelines(),
        );
        assert!(!r.runnable);
        assert!(r.pipeline.is_none());
        assert_eq!(r.reason, QWEN_TTS_SAFETENSORS_REASON);
    }

    #[test]
    fn qwen3_tts_guard_does_not_over_match_kokoro() {
        // Kokoro must still route to the worker — the qwen guard fires only on
        // the qwen3tts/qwentts family token, not every text-to-speech repo.
        let r = resolve(
            &q("hexgrad/Kokoro-82M", Some("text-to-speech"), None, &[], None),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.pipeline.as_deref(), Some("text_to_speech"));
    }

    #[test]
    fn unknown_layout_falls_back_to_card_signals() {
        // CTranslate2 dirs (model.bin) classify as LayoutKind::Unknown — the
        // detail pane must not downgrade a card-level runnable verdict.
        let r = resolve(
            &q("Systran/faster-whisper-large-v3", Some("automatic-speech-recognition"),
               Some("ctranslate2"), &[], Some(LayoutKind::Unknown)),
            &static_pipelines(),
        );
        assert!(r.runnable);
        assert_eq!(r.pipeline.as_deref(), Some("transcribe"));
    }

    #[test]
    fn unsupported_format_for_pipeline_is_honest() {
        // text-to-image but plain transformers safetensors guess — the
        // pipeline only takes diffusers-dir / single-file safetensors.
        let r = resolve(
            &q("org/repo", Some("text-to-image"), Some("transformers"), &[], None),
            &static_pipelines(),
        );
        assert!(!r.runnable);
        assert_eq!(r.pipeline.as_deref(), Some("text_to_image"));
    }

    #[test]
    fn no_runtime_for_unknown_task() {
        let r = resolve(
            &q("org/repo", Some("reinforcement-learning"), None, &[], None),
            &static_pipelines(),
        );
        assert!(!r.runnable);
        assert!(r.reason.contains("reinforcement-learning"));
    }

    #[test]
    fn live_pipeline_availability_propagates() {
        let mut pipelines = static_pipelines();
        for p in &mut pipelines {
            p.available = true;
        }
        let r = resolve(
            &q("stabilityai/sdxl-turbo", Some("text-to-image"), Some("diffusers"), &[], None),
            &pipelines,
        );
        assert!(r.runnable && r.available);
    }

    #[test]
    fn curated_filter_and_vram_fit() {
        let all = curated_picks(None, Some(24576));
        assert!(all.len() >= 10);
        assert!(all.iter().all(|p| p.fits));

        let text = curated_picks(Some("text-generation"), Some(4096));
        assert!(text.iter().all(|p| p.task == "text-generation"));
        // 7B tier (6 GB) doesn't fit a 4 GB card; the 1.5B (2 GB) does.
        assert!(text.iter().any(|p| !p.fits));
        assert!(text.iter().any(|p| p.fits));

        let none = curated_picks(None, None);
        assert!(none.iter().all(|p| !p.fits));
    }

    // --- quant fits (§9.6) ---

    fn gguf(path: &str, size_gb: u64) -> crate::models::hf::HfFile {
        crate::models::hf::HfFile {
            path: path.to_string(),
            size: size_gb * 1024 * 1024 * 1024,
            sha256: None,
            is_lfs: true,
        }
    }

    #[test]
    fn quant_label_parses_common_shapes() {
        assert_eq!(quant_label("model.Q4_K_M.gguf").as_deref(), Some("Q4_K_M"));
        assert_eq!(quant_label("Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf").as_deref(), Some("Q5_K_M"));
        assert_eq!(quant_label("model-IQ3_XS.gguf").as_deref(), Some("IQ3_XS"));
        assert_eq!(quant_label("model-f16.gguf").as_deref(), Some("F16"));
        assert_eq!(quant_label("subdir/model.q8_0.gguf").as_deref(), Some("Q8_0"));
        assert_eq!(quant_label("README.md"), None);
        assert_eq!(quant_label("model.gguf"), None);
    }

    #[test]
    fn split_parts_grouped_and_never_best() {
        let files = vec![
            gguf("big-Q8_0-00001-of-00002.gguf", 30),
            gguf("big-Q8_0-00002-of-00002.gguf", 30),
            gguf("model-Q4_K_M.gguf", 5),
        ];
        let fits = quant_fits(&files, 24_000);
        let part = fits.iter().find(|f| f.path.contains("00001")).unwrap();
        assert!(part.split);
        assert_eq!(part.set_key.as_deref(), Some("big-Q8_0"));
        assert_eq!(part.set_size, Some(files[0].size + files[1].size));
        assert_eq!(part.part_count, Some(2));
        // Estimated over the 60 GB group, not the 30 GB part.
        assert!(part.estimated_mb > 55_000, "got {}", part.estimated_mb);
        assert!(!part.best);
        let single = fits.iter().find(|f| f.path.contains("Q4_K_M")).unwrap();
        assert!(single.best);
    }

    #[test]
    fn best_is_largest_comfortable_fit() {
        let files = vec![
            gguf("m-Q4_K_M.gguf", 5),
            gguf("m-Q5_K_M.gguf", 6),
            gguf("m-Q8_0.gguf", 9),
        ];
        // 10 GB free: Q8 (9 GB + overhead) won't fit, Q5 fits comfortably.
        let fits = quant_fits(&files, 10_240);
        let best: Vec<_> = fits.iter().filter(|f| f.best).collect();
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].path, "m-Q5_K_M.gguf");
    }

    #[test]
    fn no_best_when_nothing_fits() {
        let fits = quant_fits(&[gguf("m-Q8_0.gguf", 40)], 8_000);
        assert!(fits.iter().all(|f| !f.best));
        assert!(fits.iter().all(|f| f.verdict == crate::engine::fit::FitVerdict::WontFit));
    }
}
