//! Repo layout classifier (M9 §9.3).
//!
//! Inspects a repo's file list (siblings) and decides what *shape* the model
//! is — single GGUF blobs, single/sharded safetensors, or a diffusers
//! directory — plus the file subset a repo download should actually fetch
//! (weights + configs, minus duplicate-format weights and repo cruft).

use serde::{Deserialize, Serialize};

use super::hf::HfFile;

#[derive(Debug, Serialize, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
pub enum LayoutKind {
    /// One or more standalone .gguf quant files — downloaded one at a time.
    Gguf,
    /// A single root .safetensors weight file.
    SafetensorsSingle,
    /// Sharded safetensors (model-0000N-of-0000M + index.json).
    SafetensorsShards,
    /// A diffusers pipeline directory (model_index.json present).
    DiffusersDir,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RepoLayout {
    pub kind: LayoutKind,
    /// Rel-paths a "download repo" action should fetch. Empty for `Gguf`
    /// (the user picks one quant file) and `Unknown`.
    pub download_files: Vec<String>,
}

/// File extensions that are alternate serializations of weights already
/// present as safetensors — skipped to avoid downloading every weight twice.
/// `onnx_data` is the external-tensor blob next to a .onnx graph and can be
/// as large as the weights themselves (10 GB on sdxl-turbo).
const DUPLICATE_WEIGHT_EXTS: &[&str] = &["bin", "pt", "ckpt", "h5", "msgpack", "onnx", "onnx_data"];

/// Repo cruft never needed to run a model.
const SKIP_EXTS: &[&str] = &["md", "png", "jpg", "jpeg", "gif", "webp", "mp4", "pdf"];

fn ext_of(path: &str) -> String {
    std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
}

fn file_name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

fn is_cruft(path: &str) -> bool {
    let name = file_name_of(path);
    if name.starts_with('.') {
        return true; // .gitattributes etc.
    }
    SKIP_EXTS.contains(&ext_of(path).as_str())
}

/// Is this a sharded-safetensors weight or its index?
fn is_shard_marker(path: &str) -> bool {
    let name = file_name_of(path);
    name.ends_with(".safetensors.index.json")
        || (name.ends_with(".safetensors") && name.contains("-of-"))
}

/// Classify a repo's siblings into a layout + downloadable subset.
pub fn classify(siblings: &[HfFile]) -> RepoLayout {
    let paths: Vec<&str> = siblings.iter().map(|f| f.path.as_str()).collect();

    let has_model_index = paths
        .iter()
        .any(|p| file_name_of(p) == "model_index.json");
    let has_safetensors = paths.iter().any(|p| ext_of(p) == "safetensors");
    let root_gguf: Vec<&&str> = paths.iter().filter(|p| ext_of(p) == "gguf").collect();

    if has_model_index {
        return RepoLayout {
            kind: LayoutKind::DiffusersDir,
            download_files: download_subset(&paths, has_safetensors, true),
        };
    }

    if !root_gguf.is_empty() {
        // GGUF repos: the user picks a single quant; no group subset.
        return RepoLayout { kind: LayoutKind::Gguf, download_files: vec![] };
    }

    if paths.iter().any(|p| is_shard_marker(p)) {
        return RepoLayout {
            kind: LayoutKind::SafetensorsShards,
            download_files: download_subset(&paths, has_safetensors, false),
        };
    }

    let root_safetensors: Vec<&&str> = paths
        .iter()
        .filter(|p| ext_of(p) == "safetensors" && !p.contains('/'))
        .collect();
    if root_safetensors.len() == 1 {
        return RepoLayout {
            kind: LayoutKind::SafetensorsSingle,
            download_files: download_subset(&paths, true, false),
        };
    }

    RepoLayout { kind: LayoutKind::Unknown, download_files: vec![] }
}

/// Directory part of a rel-path ("" for root files).
fn dir_of(path: &str) -> &str {
    path.rsplit_once('/').map(|(d, _)| d).unwrap_or("")
}

/// Is this a half-precision variant weight (diffusers `.fp16.` naming, e.g.
/// `diffusion_pytorch_model.fp16.safetensors` / `model.fp16-00001-of-00002.safetensors`)?
fn is_fp16_variant(path: &str) -> bool {
    let name = file_name_of(path);
    name.ends_with(".safetensors") && (name.contains(".fp16.") || name.contains(".fp16-"))
}

/// Everything needed to run the model: weights + configs/tokenizers, minus
/// cruft, minus duplicate-format weights when safetensors are present, and —
/// when both full- and half-precision safetensors live in the same directory
/// — minus the full-precision copy (we always load fp16/bf16 on consumer
/// GPUs, and the fp32 dupes double-to-triple the download: sdxl-turbo is
/// 51 GB unfiltered vs ~7 GB as the fp16 subset).
fn download_subset(paths: &[&str], has_safetensors: bool, diffusers_dir: bool) -> Vec<String> {
    // Directories that contain an fp16 variant — their non-variant
    // safetensors (and shard indexes) are redundant.
    let fp16_dirs: std::collections::HashSet<&str> = paths
        .iter()
        .filter(|p| is_fp16_variant(p))
        .map(|p| dir_of(p))
        .collect();

    paths
        .iter()
        .filter(|p| !is_cruft(p))
        .filter(|p| {
            // Drop alternate weight serializations only when a safetensors
            // copy exists; otherwise (.bin-only legacy repos) keep them.
            !(has_safetensors && DUPLICATE_WEIGHT_EXTS.contains(&ext_of(p).as_str()))
        })
        .filter(|p| {
            // Drop fp32 weights shadowed by an fp16 variant in the same dir.
            let name = file_name_of(p);
            let shadowed = fp16_dirs.contains(dir_of(p))
                && !is_fp16_variant(p)
                && (name.ends_with(".safetensors")
                    || name.ends_with(".safetensors.index.json"));
            !shadowed
        })
        .filter(|p| {
            // A diffusers dir is run from its component folders; bundled
            // root-level single-file checkpoints are a second full copy.
            !(diffusers_dir && !p.contains('/') && ext_of(p) == "safetensors")
        })
        .map(|p| p.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn files(paths: &[&str]) -> Vec<HfFile> {
        paths
            .iter()
            .map(|p| HfFile {
                path: p.to_string(),
                size: 100,
                sha256: None,
                is_lfs: false,
            })
            .collect()
    }

    #[test]
    fn diffusers_dir_detected_and_subset_filtered() {
        let s = files(&[
            "model_index.json",
            "README.md",
            ".gitattributes",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "unet/config.json",
            "vae/diffusion_pytorch_model.safetensors",
            "text_encoder/model.safetensors",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "preview.png",
        ]);
        let layout = classify(&s);
        assert_eq!(layout.kind, LayoutKind::DiffusersDir);
        assert!(layout.download_files.contains(&"model_index.json".to_string()));
        assert!(layout
            .download_files
            .contains(&"unet/diffusion_pytorch_model.safetensors".to_string()));
        assert!(layout.download_files.contains(&"tokenizer/merges.txt".to_string()));
        // duplicate .bin weight + cruft excluded
        assert!(!layout
            .download_files
            .contains(&"unet/diffusion_pytorch_model.bin".to_string()));
        assert!(!layout.download_files.contains(&"README.md".to_string()));
        assert!(!layout.download_files.contains(&".gitattributes".to_string()));
        assert!(!layout.download_files.contains(&"preview.png".to_string()));
    }

    #[test]
    fn fp16_variant_shadows_fp32_and_root_checkpoints_dropped() {
        // Shape of stabilityai/sdxl-turbo: fp32 + fp16 + onnx per component,
        // plus two bundled single-file checkpoints at the root.
        let s = files(&[
            "model_index.json",
            "sd_xl_turbo_1.0.safetensors",
            "sd_xl_turbo_1.0_fp16.safetensors",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/model.onnx",
            "unet/model.onnx_data",
            "vae/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.safetensors",
            "text_encoder/model.fp16.safetensors",
            "tokenizer/vocab.json",
        ]);
        let layout = classify(&s);
        assert_eq!(layout.kind, LayoutKind::DiffusersDir);
        let f = &layout.download_files;
        assert!(f.contains(&"model_index.json".to_string()));
        assert!(f.contains(&"unet/config.json".to_string()));
        assert!(f.contains(&"unet/diffusion_pytorch_model.fp16.safetensors".to_string()));
        assert!(f.contains(&"vae/diffusion_pytorch_model.fp16.safetensors".to_string()));
        assert!(f.contains(&"text_encoder/model.fp16.safetensors".to_string()));
        assert!(f.contains(&"tokenizer/vocab.json".to_string()));
        // fp32 copies shadowed by fp16 variants
        assert!(!f.contains(&"unet/diffusion_pytorch_model.safetensors".to_string()));
        assert!(!f.contains(&"vae/diffusion_pytorch_model.safetensors".to_string()));
        assert!(!f.contains(&"text_encoder/model.safetensors".to_string()));
        // onnx graph + external blob
        assert!(!f.contains(&"unet/model.onnx".to_string()));
        assert!(!f.contains(&"unet/model.onnx_data".to_string()));
        // bundled single-file checkpoints
        assert!(!f.contains(&"sd_xl_turbo_1.0.safetensors".to_string()));
        assert!(!f.contains(&"sd_xl_turbo_1.0_fp16.safetensors".to_string()));
    }

    #[test]
    fn fp16_shard_variant_shadows_fp32_shards_and_index() {
        let s = files(&[
            "model_index.json",
            "text_encoder/config.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.json",
            "text_encoder/model.fp16-00001-of-00002.safetensors",
            "text_encoder/model.fp16-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.fp16.json",
        ]);
        let f = classify(&s).download_files;
        assert!(f.contains(&"text_encoder/model.fp16-00001-of-00002.safetensors".to_string()));
        assert!(f.contains(&"text_encoder/model.safetensors.index.fp16.json".to_string()));
        assert!(!f.contains(&"text_encoder/model-00001-of-00002.safetensors".to_string()));
        assert!(!f.contains(&"text_encoder/model.safetensors.index.json".to_string()));
    }

    #[test]
    fn fp32_only_dirs_unaffected_by_fp16_rule() {
        // Wan-style: the text encoder only ships fp32 shards — keep them.
        let s = files(&[
            "model_index.json",
            "text_encoder/model-00001-of-00002.safetensors",
            "text_encoder/model-00002-of-00002.safetensors",
            "text_encoder/model.safetensors.index.json",
            "transformer/diffusion_pytorch_model.safetensors",
        ]);
        let f = classify(&s).download_files;
        assert!(f.contains(&"text_encoder/model-00001-of-00002.safetensors".to_string()));
        assert!(f.contains(&"text_encoder/model.safetensors.index.json".to_string()));
        assert!(f.contains(&"transformer/diffusion_pytorch_model.safetensors".to_string()));
    }

    #[test]
    fn gguf_repo_has_no_group_subset() {
        let s = files(&["model.Q4_K_M.gguf", "model.Q8_0.gguf", "README.md"]);
        let layout = classify(&s);
        assert_eq!(layout.kind, LayoutKind::Gguf);
        assert!(layout.download_files.is_empty());
    }

    #[test]
    fn sharded_safetensors_detected() {
        let s = files(&[
            "config.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
            "model.safetensors.index.json",
            "tokenizer.json",
        ]);
        let layout = classify(&s);
        assert_eq!(layout.kind, LayoutKind::SafetensorsShards);
        assert_eq!(layout.download_files.len(), 5);
    }

    #[test]
    fn single_safetensors_detected() {
        let s = files(&["config.json", "model.safetensors", "tokenizer.json"]);
        let layout = classify(&s);
        assert_eq!(layout.kind, LayoutKind::SafetensorsSingle);
        assert_eq!(layout.download_files.len(), 3);
    }

    #[test]
    fn bin_only_legacy_repo_keeps_bin_weights() {
        let s = files(&[
            "model_index.json",
            "unet/diffusion_pytorch_model.bin",
            "unet/config.json",
        ]);
        let layout = classify(&s);
        assert_eq!(layout.kind, LayoutKind::DiffusersDir);
        assert!(layout
            .download_files
            .contains(&"unet/diffusion_pytorch_model.bin".to_string()));
    }

    #[test]
    fn gguf_wins_over_stray_safetensors() {
        // Some GGUF repos ship an mmproj safetensors alongside quants;
        // the quant-picker flow should still apply.
        let s = files(&["model.Q4_K_M.gguf", "mmproj.safetensors"]);
        assert_eq!(classify(&s).kind, LayoutKind::Gguf);
    }

    #[test]
    fn empty_repo_is_unknown() {
        let layout = classify(&[]);
        assert_eq!(layout.kind, LayoutKind::Unknown);
        assert!(layout.download_files.is_empty());
    }
}
