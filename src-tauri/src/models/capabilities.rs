//! Model capability resolver (M15 §15.1).
//!
//! Derives, from a catalog row alone, everything the UI needs to route a
//! model into workflows and render its controls: which engine family runs
//! it, which workflows it can power, and task-specific descriptors (voice
//! lists, style-instruction support, diffusion architecture). Resolution is
//! pure and happens at read time — improving these rules upgrades every
//! existing install with no backfill or migration.

use serde::{Deserialize, Serialize};

use super::capability::Capability;
use super::registry::InstalledModel;

/// Which engine family runs a model.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum RuntimeKind {
    LlamaServer,
    Worker,
    QwenTts,
}

/// Workflows a model can power, in display order.
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum Workflow {
    Chat,
    Image,
    Speech,
    Transcribe,
    Video,
    Embeddings,
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct VoiceInfo {
    pub id: String,
    pub label: String,
}

/// Descriptor for TTS-capable models: which engine speaks, which voices it
/// offers, and which controls the form should render.
#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct SpeechCaps {
    /// "kokoro" | "qwen3-tts" | "vibevoice" — matches the `engine` job param.
    pub engine: String,
    /// Static voice list; the UI may override with a live list (e.g. the
    /// qwentts sidecar's /v1/voices once the engine is Ready). Empty for
    /// voice-cloning engines — the reference audio *is* the voice.
    pub voices: Vec<VoiceInfo>,
    pub supports_speed: bool,
    pub supports_instructions: bool,
    /// The engine clones a voice from a reference audio sample instead of
    /// picking from a preset list (VibeVoice, Qwen3-TTS Base talkers).
    pub supports_voice_cloning: bool,
    /// Cloning also wants a transcript of the reference audio. Qwen3-TTS
    /// Base talkers use it to run the high-quality ICL clone path
    /// (`--ref-text`); VibeVoice clones from the audio alone.
    pub cloning_requires_ref_text: bool,
}

/// Resolved capability record attached to every catalog model.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct ModelCaps {
    /// Engine family that runs this model; None when nothing here can.
    pub runtime: Option<RuntimeKind>,
    /// Worker pipeline name when `runtime == Worker`.
    pub pipeline: Option<String>,
    pub workflows: Vec<Workflow>,
    pub speech: Option<SpeechCaps>,
    /// Diffusion architecture hint ("sdxl" | "sd" | "flux") for image models.
    pub image_arch: Option<String>,
    /// True when this is a single-file GGUF diffusion transformer (DiT only)
    /// that needs companion VAE + text-encoder files to run. The new
    /// discriminator for ComfyUI-style diffusion GGUFs; `image_arch` stays the
    /// hint for SD/SDXL/Flux single-files only.
    #[serde(default)]
    pub gguf_diffusion: bool,
    /// Companion-pack key ("qwen_image" | "ltxv" | "wan") when
    /// `gguf_diffusion`; the worker keys its assembly path on it.
    #[serde(default)]
    pub companion_arch: Option<String>,
    /// Honest copy explaining why the model can't run, when it can't.
    pub reason: Option<String>,
}

fn voices(pairs: &[(&str, &str)]) -> Vec<VoiceInfo> {
    pairs
        .iter()
        .map(|(id, label)| VoiceInfo { id: (*id).into(), label: (*label).into() })
        .collect()
}

/// Kokoro's bundled voice pack (worker `text_to_speech` pipeline).
pub fn kokoro_voices() -> Vec<VoiceInfo> {
    voices(&[
        ("af_heart", "Heart — US female"),
        ("af_bella", "Bella — US female"),
        ("af_nicole", "Nicole — US female (whisper)"),
        ("af_sarah", "Sarah — US female"),
        ("af_sky", "Sky — US female"),
        ("am_adam", "Adam — US male"),
        ("am_michael", "Michael — US male"),
        ("bf_emma", "Emma — UK female"),
        ("bf_isabella", "Isabella — UK female"),
        ("bm_george", "George — UK male"),
        ("bm_lewis", "Lewis — UK male"),
    ])
}

/// Qwen3-TTS CustomVoice speakers — static fallback; the live `/v1/voices`
/// list from the tts-server replaces this once the engine is Ready.
pub fn qwen_tts_voices() -> Vec<VoiceInfo> {
    voices(&[
        ("serena", "Serena"),
        ("vivian", "Vivian"),
        ("uncle_fu", "Uncle Fu"),
        ("ryan", "Ryan"),
        ("aiden", "Aiden"),
        ("ono_anna", "Ono Anna"),
        ("sohee", "Sohee"),
        ("eric", "Eric"),
        ("dylan", "Dylan"),
    ])
}

fn kokoro_speech_caps() -> SpeechCaps {
    SpeechCaps {
        engine: "kokoro".into(),
        voices: kokoro_voices(),
        supports_speed: true,
        supports_instructions: false,
        supports_voice_cloning: false,
        cloning_requires_ref_text: false,
    }
}

/// Qwen3-TTS talker variant, as published by qwentts.cpp-compatible repos.
/// The GGUF metadata key `qwen3-tts.model_type` is authoritative
/// (`base` / `custom_voice` / `voice_design`) but isn't in the catalog, so
/// resolution matches the variant token every release embeds in the file
/// name (`qwen-talker-1.7b-base-Q8_0.gguf` → display name "qwen talker 1
/// 7b base Q8_0") and the repo id. Mirrors the `is_vibevoice` pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QwenTalkerMode {
    /// Default synthesis + voice cloning (`--ref-wav`); the speaker encoder
    /// only ships in Base checkpoints. No named speakers.
    Base,
    /// Named preset speakers (serena, vivian, …) + optional style
    /// instructions. The runtime rejects `--ref-wav` for this variant.
    CustomVoice,
    /// Voice described by a required style instruction; no named speakers,
    /// no cloning.
    VoiceDesign,
}

/// Detect the talker variant from catalog name signals. Unrecognized names
/// fall back to CustomVoice — the variant the original talker releases ship
/// and the one whose preset-speaker UI degrades most gracefully.
pub fn qwen_talker_mode(m: &InstalledModel) -> QwenTalkerMode {
    let hay = format!("{} {}", m.repo_id, m.display_name).to_lowercase();
    // Collapse separators so "custom_voice", "custom-voice" and
    // "customvoice" all match.
    let collapsed: String = hay.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if collapsed.contains("voicedesign") {
        return QwenTalkerMode::VoiceDesign;
    }
    if collapsed.contains("customvoice") {
        return QwenTalkerMode::CustomVoice;
    }
    // "base" only as a standalone token — substring would match e.g. a repo
    // named "database".
    if hay
        .split(|c: char| !c.is_ascii_alphanumeric())
        .any(|tok| tok == "base")
    {
        return QwenTalkerMode::Base;
    }
    QwenTalkerMode::CustomVoice
}

fn qwen_speech_caps(mode: QwenTalkerMode) -> SpeechCaps {
    match mode {
        QwenTalkerMode::Base => SpeechCaps {
            engine: "qwen3-tts".into(),
            voices: Vec::new(), // cloning — the reference audio is the voice
            supports_speed: false,
            // Upstream rejects --instruct for base models.
            supports_instructions: false,
            supports_voice_cloning: true,
            // --ref-text drives the ICL clone path (best quality); the UI
            // collects it alongside the reference audio.
            cloning_requires_ref_text: true,
        },
        QwenTalkerMode::CustomVoice => SpeechCaps {
            engine: "qwen3-tts".into(),
            voices: qwen_tts_voices(),
            supports_speed: false,
            supports_instructions: true,
            supports_voice_cloning: false,
            cloning_requires_ref_text: false,
        },
        QwenTalkerMode::VoiceDesign => SpeechCaps {
            engine: "qwen3-tts".into(),
            voices: Vec::new(), // the style instruction is the voice
            supports_speed: false,
            supports_instructions: true,
            supports_voice_cloning: false,
            cloning_requires_ref_text: false,
        },
    }
}

/// VibeVoice's curated *preset* voices — built from the same source slice the
/// install-time fetcher reads, so the dropdown and the on-disk clips never
/// drift. VibeVoice has no intrinsic voices; each preset is a reference clip
/// the pipeline clones from. `supports_voice_cloning` stays true so the form
/// still offers custom cloning alongside the presets.
fn vibevoice_voices_caps() -> Vec<VoiceInfo> {
    voices(&super::vibevoice_voices::builtin_voice_list())
}

fn vibevoice_speech_caps() -> SpeechCaps {
    SpeechCaps {
        engine: "vibevoice".into(),
        // Preset voices (curated reference clips) — picking one clones from its
        // bundled sample, so the user needn't supply their own.
        voices: vibevoice_voices_caps(),
        supports_speed: false,
        supports_instructions: false,
        supports_voice_cloning: true,
        cloning_requires_ref_text: false,
    }
}

/// VibeVoice checkpoints identify by name — their cards carry the generic
/// "text-to-speech" tag (or none at all on partial mirrors), so the repo id
/// / display name is the only reliable signal.
fn is_vibevoice(m: &InstalledModel) -> bool {
    let hay = format!("{} {}", m.repo_id, m.display_name).to_lowercase();
    hay.contains("vibevoice")
}

/// The upstream Qwen3-TTS release ships as safetensors (`text-to-speech`),
/// which OpenHarness can't run — Qwen3-TTS only runs through the qwentts.cpp
/// GGUF sidecar (real talkers carry `TALKER_ARCH` and are handled earlier).
/// Detect the *family* by name (NOT a specific repo): collapse separators so
/// "qwen3-tts", "qwen3_tts" and "qwen-3-tts" all reduce to `qwen3tts`. Mirrors
/// the `is_vibevoice` / `qwen_talker_mode` style.
fn is_qwen_tts(m: &InstalledModel) -> bool {
    let hay = format!("{} {}", m.repo_id, m.display_name).to_lowercase();
    let collapsed: String = hay.chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    collapsed.contains("qwen3tts") || collapsed.contains("qwentts")
}

/// Actionable verdict for a non-GGUF Qwen3-TTS repo: it's the original
/// safetensors build, which has no local runtime; the GGUF build does.
const QWEN_TTS_SAFETENSORS_REASON: &str =
    "This is the original safetensors build — OpenHarness runs Qwen3-TTS from its GGUF build instead. Download the GGUF version of this voice from the Store.";

/// Guess the diffusion architecture for the worker `text_to_image` pipeline.
/// Mirrors the heuristic the Studio form used to apply to file names, but
/// against richer catalog signals (architecture, repo id, display name).
fn detect_image_arch(m: &InstalledModel) -> &'static str {
    let hay = format!(
        "{} {} {}",
        m.architecture.as_deref().unwrap_or(""),
        m.repo_id,
        m.display_name
    )
    .to_lowercase();
    if hay.contains("flux") {
        "flux"
    } else if hay.contains("xl") || hay.contains("turbo") {
        "sdxl"
    } else {
        "sd"
    }
}

fn not_runnable(reason: &str) -> ModelCaps {
    ModelCaps { reason: Some(reason.into()), ..Default::default() }
}

fn worker(pipeline: &str, workflow: Workflow) -> ModelCaps {
    ModelCaps {
        runtime: Some(RuntimeKind::Worker),
        pipeline: Some(pipeline.into()),
        workflows: vec![workflow],
        ..Default::default()
    }
}

/// GGUF whose HF card gave no capability signal at install time. The parsed
/// GGUF architecture is authoritative where the card was silent: known
/// non-text architectures get their honest verdicts, and anything else is a
/// decoder llama-server can load — a missing `pipeline_tag` on the repo must
/// never block a chat model.
fn resolve_other_gguf(m: &InstalledModel) -> ModelCaps {
    let Some(arch) = m.architecture.as_deref() else {
        return not_runnable("No local runtime recognizes this model");
    };
    let arch_lc = arch.to_lowercase();
    // The hyphen-less 'qwen3tts' layout has its own actionable message; check
    // it before the generic speech/audio class below (which `tts` would
    // otherwise catch with a vaguer verdict).
    if arch_lc == "qwen3tts" {
        return not_runnable(
            "This talker uses the unsupported 'qwen3tts' GGUF layout — install a release with the 'qwen3-tts' architecture instead",
        );
    }
    // Speech / audio-modality architectures as a class. None of these tokens
    // is a substring of a real text-LLM arch (llama, qwen2/3, gemma, phi3,
    // mistral, mixtral, command-r, dbrx, deepseek2, starcoder2, gptneox,
    // falcon, bloom, mpt, stablelm …), so a bare substring match can't strip
    // Chat off a chat model. Speech models run through the Python worker
    // from their safetensors/transformers release (e.g. the `vibevoice`
    // pipeline) and llama-server has no graph for them, so a speech-arch GGUF
    // is not runnable here.
    const SPEECH_AUDIO_TOKENS: &[&str] =
        &["whisper", "asr", "tts", "speech", "audio", "voice", "codec"];
    if SPEECH_AUDIO_TOKENS.iter().any(|tok| arch_lc.contains(tok)) {
        return not_runnable(
            "This is a GGUF build of a speech/audio model — speech models run through the worker from their safetensors/transformers release, not as a GGUF. Install that release from the Store.",
        );
    }
    if arch_lc.contains("stable-diffusion") || arch_lc.contains("sdxl") || arch_lc.contains("flux")
    {
        return not_runnable(
            "GGUF diffusion checkpoints aren't supported — install the safetensors or diffusers release instead",
        );
    }
    if arch_lc.contains("bert") {
        return ModelCaps {
            runtime: Some(RuntimeKind::LlamaServer),
            workflows: vec![Workflow::Embeddings],
            ..Default::default()
        };
    }
    // ComfyUI-style single-file diffusion GGUFs (the DiT transformer only; no
    // VAE / text encoder). These archs are NOT decoder LLMs — llama-server has
    // no graph for them — so they must be intercepted before the optimistic
    // Chat catch-all below. Routed into the Studio worker pipelines with
    // `gguf_diffusion` set. `qwen_image` has a real worker loader now (companion
    // VAE/text-encoder pack + diffusers GGUF assembly); ltxv/wan loading are
    // later phases. `reason` stays None: the model IS runnable once its companion
    // files are present (presence is a UI/command concern, not a caps field, per
    // the read-time-pure invariant).
    if let Some(caps) = resolve_diffusion_gguf(&arch_lc) {
        return caps;
    }
    ModelCaps {
        runtime: Some(RuntimeKind::LlamaServer),
        workflows: vec![Workflow::Chat],
        ..Default::default()
    }
}

/// Route a single-file GGUF whose `general.architecture` names a diffusion
/// transformer (DiT). Supported archs go to a Studio worker pipeline with the
/// `gguf_diffusion` discriminator; a denylist of known-but-unsupported DiT
/// archs returns an honest `not_runnable` verdict so they never leak into the
/// Chat catch-all. Returns `None` for anything that isn't a recognized
/// diffusion arch (the caller then treats it as a decoder LLM).
fn resolve_diffusion_gguf(arch_lc: &str) -> Option<ModelCaps> {
    /// (worker pipeline, workflow, companion-pack key) per supported arch.
    const SUPPORTED: &[(&str, &str, Workflow, &str)] = &[
        ("qwen_image", "text_to_image", Workflow::Image, "qwen_image"),
        ("ltxv", "text_to_video", Workflow::Video, "ltxv"),
        ("wan", "text_to_video", Workflow::Video, "wan"),
    ];
    if let Some((_, pipeline, workflow, companion)) =
        SUPPORTED.iter().find(|(arch, ..)| *arch == arch_lc)
    {
        return Some(ModelCaps {
            gguf_diffusion: true,
            companion_arch: Some((*companion).into()),
            ..worker(pipeline, *workflow)
        });
    }
    // Other known DiT architectures we don't support yet. They aren't decoder
    // LLMs, so refuse them honestly rather than handing them Chat.
    const UNSUPPORTED: &[&str] = &[
        "hunyuan-video",
        "mochi",
        "cogvideox",
        "sd3",
        "auraflow",
        "lumina2",
    ];
    if UNSUPPORTED.contains(&arch_lc) {
        return Some(not_runnable(
            "This is a single-file GGUF diffusion model OpenHarness can't run yet — its architecture isn't supported. Use a supported diffusion GGUF (Qwen-Image, LTX, Wan) or install the full diffusers release.",
        ));
    }
    None
}

/// Resolve the capability record for a catalog model.
pub fn resolve(m: &InstalledModel) -> ModelCaps {
    // Qwen3-TTS talkers identify purely by GGUF architecture — the HF cards
    // carry no useful pipeline_tag, so this check precedes everything else.
    if m.architecture.as_deref() == Some(crate::tts::TALKER_ARCH) {
        return ModelCaps {
            runtime: Some(RuntimeKind::QwenTts),
            workflows: vec![Workflow::Speech],
            speech: Some(qwen_speech_caps(qwen_talker_mode(m))),
            ..Default::default()
        };
    }

    if m.format == "gguf" {
        return match m.capability {
            Capability::TextGeneration | Capability::VisionText => ModelCaps {
                runtime: Some(RuntimeKind::LlamaServer),
                workflows: vec![Workflow::Chat],
                ..Default::default()
            },
            Capability::Embedding => ModelCaps {
                runtime: Some(RuntimeKind::LlamaServer),
                workflows: vec![Workflow::Embeddings],
                ..Default::default()
            },
            Capability::ImageGen => not_runnable(
                "GGUF diffusion checkpoints aren't supported — install the safetensors or diffusers release instead",
            ),
            Capability::Speech => not_runnable(
                "GGUF speech models aren't supported — transcription runs faster-whisper (CTranslate2) repos",
            ),
            Capability::Other => resolve_other_gguf(m),
        };
    }

    // Dir-layout repos (safetensors / diffusers-dir) route by task into the
    // Python worker's pipelines.
    //
    // The original Qwen3-TTS release is safetensors with a "text-to-speech"
    // tag, which would otherwise fall into the Kokoro arm and be misclassified
    // as runnable (it's not a Kokoro checkpoint). Real GGUF talkers are handled
    // by the `TALKER_ARCH` check above, so this only catches the non-GGUF
    // build — refuse it with a path to the runnable GGUF version.
    if is_qwen_tts(m) {
        return not_runnable(QWEN_TTS_SAFETENSORS_REASON);
    }
    match m.task.as_deref() {
        Some("text-to-image") | Some("image-to-image") => ModelCaps {
            image_arch: Some(detect_image_arch(m).into()),
            ..worker("text_to_image", Workflow::Image)
        },
        Some("text-to-video") | Some("image-to-video") => worker("text_to_video", Workflow::Video),
        Some("automatic-speech-recognition") => worker("transcribe", Workflow::Transcribe),
        Some("text-to-speech") if is_vibevoice(m) => ModelCaps {
            speech: Some(vibevoice_speech_caps()),
            ..worker("vibevoice", Workflow::Speech)
        },
        Some("text-to-speech") => ModelCaps {
            speech: Some(kokoro_speech_caps()),
            ..worker("text_to_speech", Workflow::Speech)
        },
        Some("text-generation") => not_runnable(
            "Text models run from GGUF here — install the GGUF release instead",
        ),
        Some(task) => not_runnable(&format!("No local runtime for task '{task}' yet")),
        None if is_vibevoice(m) => ModelCaps {
            speech: Some(vibevoice_speech_caps()),
            ..worker("vibevoice", Workflow::Speech)
        },
        None => not_runnable("No local runtime recognizes this model"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(
        format: &str,
        capability: Capability,
        task: Option<&str>,
        architecture: Option<&str>,
    ) -> InstalledModel {
        InstalledModel {
            id: "test".into(),
            repo_id: "test/model".into(),
            revision: "main".into(),
            display_name: "Test Model".into(),
            architecture: architecture.map(Into::into),
            capability,
            format: format.into(),
            task: task.map(Into::into),
            local_path: None,
            params_b: None,
            quantization: None,
            context_train: None,
            installed_at: 0,
            primary_file_path: String::new(),
            primary_file_size: 0,
            primary_rel_path: None,
            primary_sha256: None,
            file_count: 1,
            total_size: 0,
            pinned: false,
            caps: ModelCaps::default(),
        }
    }

    #[test]
    fn qwen_talker_routes_to_tts_engine_regardless_of_capability() {
        // No variant token in the name → CustomVoice defaults (preset voices).
        let m = model("gguf", Capability::Other, None, Some("qwen3-tts"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::QwenTts));
        assert_eq!(caps.workflows, vec![Workflow::Speech]);
        let speech = caps.speech.unwrap();
        assert_eq!(speech.engine, "qwen3-tts");
        assert_eq!(speech.voices.len(), 9);
        assert!(speech.supports_instructions);
        assert!(!speech.supports_speed);
        assert!(!speech.supports_voice_cloning);
    }

    #[test]
    fn qwen_base_talker_gets_voice_cloning() {
        // Display name as derived from "qwen-talker-1.7b-base-Q8_0.gguf".
        let mut m = model("gguf", Capability::Other, None, Some("qwen3-tts"));
        m.repo_id = "Serveurperso/Qwen3-TTS-GGUF".into();
        m.display_name = "qwen talker 1 7b base Q8_0".into();
        assert_eq!(qwen_talker_mode(&m), QwenTalkerMode::Base);
        let speech = resolve(&m).speech.unwrap();
        assert!(speech.supports_voice_cloning);
        assert!(speech.cloning_requires_ref_text);
        assert!(speech.voices.is_empty()); // no named speakers in base talkers
        assert!(!speech.supports_instructions); // upstream rejects --instruct for base
    }

    #[test]
    fn qwen_customvoice_talker_keeps_preset_voices_no_cloning() {
        // The user-reported model: qwen3-tts.model_type = custom_voice. The
        // runtime rejects --ref-wav for it (speaker encoder is base-only), so
        // cloning must stay off — preset speakers + instructions instead.
        let mut m = model("gguf", Capability::Other, None, Some("qwen3-tts"));
        m.repo_id = "Serveurperso/Qwen3-TTS-GGUF".into();
        m.display_name = "qwen talker 1 7b customvoice F32".into();
        assert_eq!(qwen_talker_mode(&m), QwenTalkerMode::CustomVoice);
        let speech = resolve(&m).speech.unwrap();
        assert!(!speech.supports_voice_cloning);
        assert_eq!(speech.voices.len(), 9);
        assert!(speech.supports_instructions);
    }

    #[test]
    fn qwen_mode_detects_separator_variants_and_repo_id() {
        let mut m = model("gguf", Capability::Other, None, Some("qwen3-tts"));
        m.display_name = "Qwen3-TTS-12Hz-1.7B-custom_voice".into();
        assert_eq!(qwen_talker_mode(&m), QwenTalkerMode::CustomVoice);
        m.display_name = "qwen talker 1 7b voicedesign Q4_K_M".into();
        assert_eq!(qwen_talker_mode(&m), QwenTalkerMode::VoiceDesign);
        let speech = resolve(&m).speech.unwrap();
        assert!(speech.voices.is_empty());
        assert!(speech.supports_instructions);
        assert!(!speech.supports_voice_cloning);
        // Variant token in the repo id alone is enough.
        m.display_name = "talker 0 6b Q8_0".into();
        m.repo_id = "someone/Qwen3-TTS-0.6B-Base-GGUF".into();
        assert_eq!(qwen_talker_mode(&m), QwenTalkerMode::Base);
    }

    #[test]
    fn qwen_mode_base_token_is_word_bounded() {
        // "database" must not read as the Base variant.
        let mut m = model("gguf", Capability::Other, None, Some("qwen3-tts"));
        m.repo_id = "someone/database-talker".into();
        m.display_name = "database talker".into();
        assert_eq!(qwen_talker_mode(&m), QwenTalkerMode::CustomVoice);
    }

    #[test]
    fn text_gguf_gets_chat() {
        let m = model("gguf", Capability::TextGeneration, Some("text-generation"), Some("qwen2"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::LlamaServer));
        assert_eq!(caps.workflows, vec![Workflow::Chat]);
        assert!(caps.speech.is_none());
        assert!(caps.reason.is_none());
    }

    #[test]
    fn embedding_gguf_gets_embeddings_only() {
        let m = model("gguf", Capability::Embedding, Some("feature-extraction"), Some("bert"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::LlamaServer));
        assert_eq!(caps.workflows, vec![Workflow::Embeddings]);
    }

    #[test]
    fn diffusers_image_repo_routes_to_worker_with_arch() {
        let mut m = model("diffusers-dir", Capability::ImageGen, Some("text-to-image"), None);
        m.repo_id = "stabilityai/sdxl-turbo".into();
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_image"));
        assert_eq!(caps.workflows, vec![Workflow::Image]);
        assert_eq!(caps.image_arch.as_deref(), Some("sdxl"));
    }

    #[test]
    fn flux_arch_detected_from_repo_id() {
        let mut m = model("diffusers-dir", Capability::ImageGen, Some("text-to-image"), None);
        m.repo_id = "black-forest-labs/FLUX.1-schnell".into();
        assert_eq!(resolve(&m).image_arch.as_deref(), Some("flux"));
    }

    #[test]
    fn plain_sd_arch_is_default() {
        let mut m = model("diffusers-dir", Capability::ImageGen, Some("text-to-image"), None);
        m.repo_id = "runwayml/stable-diffusion-v1-5".into();
        m.display_name = "Stable Diffusion v1 5".into();
        assert_eq!(resolve(&m).image_arch.as_deref(), Some("sd"));
    }

    #[test]
    fn kokoro_repo_gets_speech_descriptor() {
        let m = model("safetensors", Capability::Speech, Some("text-to-speech"), None);
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_speech"));
        let speech = caps.speech.unwrap();
        assert_eq!(speech.engine, "kokoro");
        assert_eq!(speech.voices.len(), 11);
        assert!(speech.supports_speed);
        assert!(!speech.supports_instructions);
        assert!(!speech.supports_voice_cloning);
    }

    #[test]
    fn vibevoice_repo_routes_to_cloning_pipeline() {
        let mut m = model("safetensors", Capability::Speech, Some("text-to-speech"), None);
        m.repo_id = "vibevoice/VibeVoice-1.5B".into();
        m.display_name = "VibeVoice 1.5B".into();
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("vibevoice"));
        assert_eq!(caps.workflows, vec![Workflow::Speech]);
        let speech = caps.speech.unwrap();
        assert_eq!(speech.engine, "vibevoice");
        // Curated built-in preset voices — pick one without supplying a clip.
        assert_eq!(speech.voices.len(), 5);
        assert!(speech.voices.iter().any(|v| v.id == "alice"));
        // Cloning stays available for custom reference samples.
        assert!(speech.supports_voice_cloning);
        assert!(!speech.cloning_requires_ref_text); // VibeVoice clones from audio alone
        assert!(!speech.supports_speed);
        assert!(!speech.supports_instructions);
    }

    #[test]
    fn vibevoice_without_task_still_resolves_by_name() {
        let mut m = model("safetensors", Capability::Other, None, None);
        m.repo_id = "vibevoice-community/VibeVoice-1.5B".into();
        m.display_name = "VibeVoice-1.5B".into();
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("vibevoice"));
        let speech = caps.speech.unwrap();
        assert!(speech.supports_voice_cloning);
        assert_eq!(speech.voices.len(), 5); // preset voices ride along
    }

    #[test]
    fn gguf_named_vibevoice_does_not_route_to_cloning() {
        // The gguf-by-capability rules still win for GGUF files; the name
        // check only sharpens dir-layout resolution.
        let mut m = model("gguf", Capability::Other, None, None);
        m.repo_id = "someone/vibevoice-gguf".into();
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.reason.is_some());
    }

    #[test]
    fn whisper_repo_routes_to_transcribe() {
        let m = model(
            "safetensors",
            Capability::Speech,
            Some("automatic-speech-recognition"),
            None,
        );
        let caps = resolve(&m);
        assert_eq!(caps.pipeline.as_deref(), Some("transcribe"));
        assert_eq!(caps.workflows, vec![Workflow::Transcribe]);
    }

    #[test]
    fn qwen3_tts_safetensors_is_not_runnable_with_gguf_guidance() {
        // The original upstream release: safetensors, "text-to-speech" tag. It
        // would otherwise hit the Kokoro arm and be misclassified as runnable.
        let mut m = model("safetensors", Capability::Speech, Some("text-to-speech"), None);
        m.repo_id = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice".into();
        m.display_name = "Qwen3 TTS 12Hz 1.7B CustomVoice".into();
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.pipeline.is_none());
        assert_eq!(caps.reason.as_deref(), Some(QWEN_TTS_SAFETENSORS_REASON));
    }

    #[test]
    fn qwen3_tts_guard_does_not_over_match_kokoro() {
        // A genuine Kokoro repo must still resolve to the Kokoro pipeline — the
        // qwen guard fires only on the qwen3tts/qwentts family token.
        let mut m = model("safetensors", Capability::Speech, Some("text-to-speech"), None);
        m.repo_id = "hexgrad/Kokoro-82M".into();
        m.display_name = "Kokoro 82M".into();
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_speech"));
        assert_eq!(caps.speech.unwrap().engine, "kokoro");
    }

    #[test]
    fn video_repo_routes_to_worker_video() {
        let m = model("diffusers-dir", Capability::ImageGen, Some("text-to-video"), None);
        let caps = resolve(&m);
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_video"));
        assert_eq!(caps.workflows, vec![Workflow::Video]);
    }

    #[test]
    fn other_gguf_with_decoder_arch_falls_back_to_text() {
        // Regression: HF cards without a pipeline_tag classify as Other at
        // install time, but the parsed decoder arch means llama-server runs it.
        let m = model("gguf", Capability::Other, None, Some("qwen35moe"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::LlamaServer));
        assert_eq!(caps.workflows, vec![Workflow::Chat]);
        assert!(caps.reason.is_none());
    }

    #[test]
    fn other_gguf_without_arch_stays_not_runnable() {
        let m = model("gguf", Capability::Other, None, None);
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.workflows.is_empty());
        assert!(caps.reason.is_some());
    }

    #[test]
    fn other_gguf_bert_arch_routes_to_embeddings() {
        let m = model("gguf", Capability::Other, None, Some("nomic-bert"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::LlamaServer));
        assert_eq!(caps.workflows, vec![Workflow::Embeddings]);
    }

    #[test]
    fn incompatible_qwen3tts_layout_gets_specific_reason() {
        // The hyphen-less 'qwen3tts' arch (cstr conversions) fits neither
        // llama-server nor the qwentts sidecar — say so, don't offer Chat.
        let m = model("gguf", Capability::Other, None, Some("qwen3tts"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.reason.unwrap().contains("qwen3-tts"));
    }

    #[test]
    fn other_gguf_whisper_arch_stays_not_runnable() {
        let m = model("gguf", Capability::Other, None, Some("whisper"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.reason.unwrap().contains("speech/audio"));
    }

    #[test]
    fn other_gguf_vibevoice_asr_arch_is_not_chat() {
        // Regression: cstr/vibevoice-1.5b-GGUF installs as Capability::Other
        // with the parsed GGUF arch `vibevoice-asr`. The optimistic catch-all
        // used to hand it Chat even though llama-server has no
        // VibeVoice graph (it can't load). The speech/audio-arch denylist now
        // catches the `asr` token and returns an honest not_runnable verdict.
        let m = model("gguf", Capability::Other, None, Some("vibevoice-asr"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.workflows.is_empty());
        assert!(!caps.workflows.contains(&Workflow::Chat));
        let reason = caps.reason.unwrap();
        assert!(reason.contains("speech"));
        assert!(reason.contains("safetensors"));
        assert!(reason.contains("worker"));
    }

    #[test]
    fn other_gguf_generic_speech_archs_are_not_runnable() {
        // The denylist is generic — any *-tts / *-asr GGUF arch is rejected,
        // no model-specific hardcoding.
        for arch in ["some-tts", "some-asr"] {
            let m = model("gguf", Capability::Other, None, Some(arch));
            let caps = resolve(&m);
            assert_eq!(caps.runtime, None, "arch {arch} should not be runnable");
            assert!(caps.workflows.is_empty(), "arch {arch} should have no workflows");
            assert!(caps.reason.is_some(), "arch {arch} should carry a reason");
        }
    }

    #[test]
    fn gguf_diffusion_checkpoint_is_honestly_not_runnable() {
        let m = model("gguf", Capability::ImageGen, Some("text-to-image"), Some("flux"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.workflows.is_empty());
        assert!(caps.reason.unwrap().contains("diffusers release"));
    }

    #[test]
    fn qwen_image_gguf_routes_to_image_worker() {
        // Card gave no signal (Capability::Other), parsed arch = qwen_image.
        // It used to land in the Chat catch-all; now it's a Studio
        // text-to-image model carrying the gguf-diffusion discriminator.
        let m = model("gguf", Capability::Other, None, Some("qwen_image"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_image"));
        assert_eq!(caps.workflows, vec![Workflow::Image]);
        assert!(caps.gguf_diffusion);
        assert_eq!(caps.companion_arch.as_deref(), Some("qwen_image"));
        assert!(caps.reason.is_none());
        assert!(!caps.workflows.contains(&Workflow::Chat));
    }

    #[test]
    fn ltxv_gguf_routes_to_video_worker() {
        let m = model("gguf", Capability::Other, None, Some("ltxv"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_video"));
        assert_eq!(caps.workflows, vec![Workflow::Video]);
        assert!(caps.gguf_diffusion);
        assert_eq!(caps.companion_arch.as_deref(), Some("ltxv"));
        assert!(caps.reason.is_none());
        assert!(!caps.workflows.contains(&Workflow::Chat));
    }

    #[test]
    fn wan_gguf_routes_to_video_worker() {
        let m = model("gguf", Capability::Other, None, Some("wan"));
        let caps = resolve(&m);
        assert_eq!(caps.runtime, Some(RuntimeKind::Worker));
        assert_eq!(caps.pipeline.as_deref(), Some("text_to_video"));
        assert_eq!(caps.workflows, vec![Workflow::Video]);
        assert!(caps.gguf_diffusion);
        assert_eq!(caps.companion_arch.as_deref(), Some("wan"));
        assert!(caps.reason.is_none());
        assert!(!caps.workflows.contains(&Workflow::Chat));
    }

    #[test]
    fn unsupported_diffusion_gguf_arch_is_honestly_not_runnable() {
        // Known DiT archs we don't support yet must not leak into Chat.
        for arch in ["hunyuan-video", "mochi", "cogvideox", "sd3", "auraflow", "lumina2"] {
            let m = model("gguf", Capability::Other, None, Some(arch));
            let caps = resolve(&m);
            assert_eq!(caps.runtime, None, "arch {arch} should not be runnable");
            assert!(caps.workflows.is_empty(), "arch {arch} should have no workflows");
            assert!(!caps.gguf_diffusion, "arch {arch} is not a supported diffusion GGUF");
            assert!(caps.reason.is_some(), "arch {arch} should carry a reason");
            assert!(!caps.workflows.contains(&Workflow::Chat));
            }
    }

    #[test]
    fn safetensors_llm_points_at_gguf_release() {
        let m = model("safetensors", Capability::TextGeneration, Some("text-generation"), None);
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.reason.unwrap().contains("GGUF"));
    }

    #[test]
    fn unknown_task_reports_reason() {
        let m = model("safetensors", Capability::Other, Some("audio-classification"), None);
        let caps = resolve(&m);
        assert_eq!(caps.runtime, None);
        assert!(caps.reason.unwrap().contains("audio-classification"));
    }
}
