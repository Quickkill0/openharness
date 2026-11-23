use serde::{Deserialize, Serialize};

use super::hf::HfModelCard;

/// High-level capability classification for a model.
/// Available for server-side use in later milestones (e.g. auto-tagging on
/// install). The Tauri commands surface raw `pipeline_tag` to the frontend;
/// this enum provides a canonical local mapping.
#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq, Hash)]
#[serde(rename_all = "kebab-case")]
pub enum Capability {
    TextGeneration,
    Embedding,
    VisionText,
    Speech,
    ImageGen,
    Other,
}

#[allow(dead_code)]
/// Derive a `Capability` from HuggingFace metadata.
///
/// Priority order (first match wins):
/// 1. GGUF architecture heuristics (whisper → Speech, SD/SDXL/Flux → ImageGen)
/// 2. `pipeline_tag` exact matches
/// 3. `library_name` heuristics
/// 4. Tag scan for multimodal/vision
/// 5. Fallback → Other
pub fn classify(card: &HfModelCard, gguf_arch: Option<&str>) -> Capability {
    // 1. GGUF architecture overrides
    if let Some(arch) = gguf_arch {
        let arch_lc = arch.to_lowercase();
        if arch_lc.contains("whisper") {
            return Capability::Speech;
        }
        if arch_lc.contains("stable-diffusion")
            || arch_lc.contains("sdxl")
            || arch_lc.contains("flux")
        {
            return Capability::ImageGen;
        }
    }

    // 2. pipeline_tag
    if let Some(tag) = &card.pipeline_tag {
        match tag.as_str() {
            "text-generation" => return Capability::TextGeneration,
            "feature-extraction" => return Capability::Embedding,
            "automatic-speech-recognition" => return Capability::Speech,
            "text-to-image" => return Capability::ImageGen,
            "image-text-to-text" => return Capability::VisionText,
            _ => {}
        }
    }

    // 3. library_name heuristic
    if let Some(lib) = &card.library_name {
        if lib.as_str() == "sentence-transformers" {
            return Capability::Embedding;
        }
    }

    // 4. Tag scan
    let tags_lc: Vec<String> = card.tags.iter().map(|t| t.to_lowercase()).collect();
    if tags_lc.iter().any(|t| t == "multimodal" || t == "vision") {
        return Capability::VisionText;
    }

    Capability::Other
}

/// Map an HF pipeline_tag-style task string to a local `Capability`.
/// Used when registering repo downloads, where the task is known up front
/// and no GGUF header exists to classify from.
pub fn capability_for_task(task: &str) -> Capability {
    match task {
        "text-generation" => Capability::TextGeneration,
        "feature-extraction" | "sentence-similarity" => Capability::Embedding,
        "automatic-speech-recognition" | "text-to-speech" | "audio-to-audio" => Capability::Speech,
        "text-to-image" | "image-to-image" | "text-to-video" | "image-to-video" => {
            Capability::ImageGen
        }
        "image-text-to-text" | "visual-question-answering" => Capability::VisionText,
        _ => Capability::Other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::hf::HfModelCard;

    #[test]
    fn task_mapping_covers_store_tabs() {
        assert_eq!(capability_for_task("text-generation"), Capability::TextGeneration);
        assert_eq!(capability_for_task("text-to-image"), Capability::ImageGen);
        assert_eq!(capability_for_task("text-to-video"), Capability::ImageGen);
        assert_eq!(capability_for_task("text-to-speech"), Capability::Speech);
        assert_eq!(
            capability_for_task("automatic-speech-recognition"),
            Capability::Speech
        );
        assert_eq!(capability_for_task("feature-extraction"), Capability::Embedding);
        assert_eq!(capability_for_task("something-new"), Capability::Other);
    }

    /// Minimal card builder for tests. Fills all mandatory fields with
    /// sensible defaults; callers only supply the fields under test.
    fn card(
        pipeline_tag: Option<&str>,
        library_name: Option<&str>,
        tags: &[&str],
    ) -> HfModelCard {
        HfModelCard {
            id: "test/model".to_string(),
            author: None,
            pipeline_tag: pipeline_tag.map(|s| s.to_string()),
            library_name: library_name.map(|s| s.to_string()),
            tags: tags.iter().map(|s| s.to_string()).collect(),
            downloads: 0,
            likes: 0,
            trending_score: None,
            gated: false,
            private: false,
            last_modified: None,
            created_at: None,
        }
    }

    // a) GGUF arch "whisper" wins over conflicting pipeline_tag "text-generation"
    #[test]
    fn whisper_arch_overrides_pipeline_tag() {
        let c = card(Some("text-generation"), None, &[]);
        assert_eq!(classify(&c, Some("whisper")), Capability::Speech);
    }

    // b) GGUF arch "stable-diffusion" → ImageGen
    #[test]
    fn stable_diffusion_arch() {
        let c = card(None, None, &[]);
        assert_eq!(classify(&c, Some("stable-diffusion")), Capability::ImageGen);
    }

    // c) GGUF arch "sdxl" → ImageGen
    #[test]
    fn sdxl_arch() {
        let c = card(None, None, &[]);
        assert_eq!(classify(&c, Some("sdxl")), Capability::ImageGen);
    }

    // d) GGUF arch "flux" → ImageGen
    #[test]
    fn flux_arch() {
        let c = card(None, None, &[]);
        assert_eq!(classify(&c, Some("flux")), Capability::ImageGen);
    }

    // e) pipeline_tag "text-generation", no gguf_arch → TextGeneration
    #[test]
    fn pipeline_text_generation() {
        let c = card(Some("text-generation"), None, &[]);
        assert_eq!(classify(&c, None), Capability::TextGeneration);
    }

    // f) pipeline_tag "feature-extraction" → Embedding
    #[test]
    fn pipeline_feature_extraction() {
        let c = card(Some("feature-extraction"), None, &[]);
        assert_eq!(classify(&c, None), Capability::Embedding);
    }

    // g) library_name "sentence-transformers", no pipeline_tag → Embedding
    #[test]
    fn library_sentence_transformers() {
        let c = card(None, Some("sentence-transformers"), &[]);
        assert_eq!(classify(&c, None), Capability::Embedding);
    }

    // h) pipeline_tag "automatic-speech-recognition" → Speech
    #[test]
    fn pipeline_asr() {
        let c = card(Some("automatic-speech-recognition"), None, &[]);
        assert_eq!(classify(&c, None), Capability::Speech);
    }

    // i) pipeline_tag "text-to-image" → ImageGen
    #[test]
    fn pipeline_text_to_image() {
        let c = card(Some("text-to-image"), None, &[]);
        assert_eq!(classify(&c, None), Capability::ImageGen);
    }

    // j) pipeline_tag "image-text-to-text" → VisionText
    #[test]
    fn pipeline_image_text_to_text() {
        let c = card(Some("image-text-to-text"), None, &[]);
        assert_eq!(classify(&c, None), Capability::VisionText);
    }

    // k) tags=["multimodal"], no pipeline_tag → VisionText
    #[test]
    fn tag_multimodal() {
        let c = card(None, None, &["multimodal"]);
        assert_eq!(classify(&c, None), Capability::VisionText);
    }

    // l) All inputs None/empty → Other
    #[test]
    fn all_none_returns_other() {
        let c = card(None, None, &[]);
        assert_eq!(classify(&c, None), Capability::Other);
    }
}
