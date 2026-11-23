//! Model accounting from GGUF tensor tables.
//!
//! Reads every tensor record of a (possibly split) GGUF and sorts the bytes
//! into what the placement planner needs:
//!
//! * **always-used weights** — attention, norms, output head, shared experts,
//!   dense FFN, router. Touched by every token.
//! * **routed experts** — `ffn_{gate,up,down,gate_up}_exps`. Each token reads
//!   only `expert_used / expert_count` of each layer's bank.
//! * **token embeddings** — llama.cpp keeps `token_embd` in host memory and
//!   reads one row per token, so it never needs VRAM.
//! * **MTP / next-token-prediction layers** — the trailing
//!   `nextn_predict_layers` blocks, not used for ordinary decoding.
//!
//! KV cache is accounted per layer from the tensor shapes themselves:
//! standard attention reads the K/V projection widths (exact GQA, no lookup
//! table), MLA layers cache the compressed latent (`kv_lora_rank + rope dim`,
//! plus the DSA indexer key when present), sliding-window layers cap at the
//! window, and linear-attention / recurrent layers (KDA, gated delta-net,
//! Mamba) keep a fixed f32 state instead of a growing cache.

use std::path::Path;

use serde::Serialize;

use crate::error::AppError;
use crate::models::gguf::{self, GgufTensorHeader, TensorInfo};

/// KV-cache element type (`-ctk`/`-ctv`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum KvType {
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

impl KvType {
    /// Bytes of one cache row of `elems` elements (`ggml_row_size`).
    pub fn row_bytes(self, elems: u64) -> u64 {
        match self {
            KvType::F16 => elems * 2,
            KvType::Q8_0 => elems.div_ceil(32) * 34,
            KvType::Q4_0 => elems.div_ceil(32) * 18,
        }
    }

    /// llama-server spelling.
    pub fn as_arg(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
            KvType::Q4_0 => "q4_0",
        }
    }

    pub fn parse(s: &str) -> Option<KvType> {
        match s.trim().to_ascii_lowercase().as_str() {
            "f16" | "fp16" => Some(KvType::F16),
            "q8_0" | "q8" => Some(KvType::Q8_0),
            "q4_0" | "q4" => Some(KvType::Q4_0),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    /// Standard (GQA/MHA) attention: KV grows with context.
    Attention,
    /// Multi-head latent attention: compressed KV latent per token.
    Mla,
    /// Linear attention / SSM: fixed-size state per sequence.
    Recurrent,
    /// Next-token-prediction (MTP) block: loaded by the file, unused for decode.
    Mtp,
}

/// The native tier prices compute buffers from actual up/down tensor types and
/// the first routed up tensor's ne[0]/ne[1], not model-wide metadata guesses.
#[derive(Debug, Clone, Serialize)]
pub struct ExpertLayout {
    pub up_type: u32,
    pub down_type: u32,
    pub latent: u64,
    pub hidden: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerAccount {
    pub index: u32,
    pub kind: LayerKind,
    /// Non-expert weights in this block.
    pub always_bytes: u64,
    /// Routed-expert weights in this block (0 for dense blocks).
    pub expert_bytes: u64,
    /// Experts in this block's bank (0 for dense blocks).
    pub n_expert: u32,
    pub expert_layout: Option<ExpertLayout>,
    /// Cached K elements per token (MLA: the compressed latent; 0 for
    /// recurrent / MTP).
    pub k_elems_per_token: u64,
    /// Cached V elements per token (0 for MLA: with flash attention llama.cpp
    /// stores K only).
    pub v_elems_per_token: u64,
    /// DSA lightning-indexer key elements per token (a separate cache stored
    /// in the K type), sparse-MLA layers only.
    pub idx_elems_per_token: u64,
    /// Sliding-window cap in tokens, when this layer is SWA.
    pub swa_window: Option<u32>,
    /// Fixed recurrent state elements per sequence (f32), recurrent layers only.
    pub state_elems: u64,
}

/// Everything the planner needs to know about a model's memory and traffic.
#[derive(Debug, Clone, Serialize)]
pub struct ModelAccounting {
    pub architecture: String,
    /// GGUF files the model spans (split GGUFs > 1).
    pub files: u32,
    /// Transformer blocks used for decoding (excludes MTP blocks).
    pub n_layers: u32,
    pub mtp_layers: u32,
    pub moe_layers: u32,
    pub ctx_train: u64,
    pub n_embd: u64,
    pub n_vocab: u64,
    pub expert_count: u32,
    pub expert_used: u32,
    /// Hidden width of one routed expert (0 when not MoE / unknown).
    pub expert_ffn: u64,
    /// Sum of every tensor in the file(s).
    pub total_bytes: u64,
    /// `token_embd` — host-resident in llama.cpp.
    pub token_embd_bytes: u64,
    /// Non-expert weights that should live on the GPU (excludes token_embd and
    /// MTP blocks; includes the output head).
    pub always_bytes: u64,
    /// Routed-expert weights in decode blocks.
    pub expert_bytes: u64,
    /// MTP block weights (mapped, never touched for plain decoding).
    pub mtp_bytes: u64,
    /// Output head (`output.weight` + norm), part of `always_bytes`.
    pub output_bytes: u64,
    /// No `output.weight`: the output head reuses `token_embd`, and llama.cpp
    /// puts a second copy of it on the output device.
    pub output_tied: bool,
    /// Set when some layer keeps context memory this calculator can't size
    /// (an unrecognised recurrent or attention layout). The planner refuses
    /// to estimate such a model instead of reporting zero.
    pub unsupported: Option<String>,
    /// Expert bytes read per decoded token: Σ_layers expert_bytes·k/n.
    pub active_expert_bytes_per_token: u64,
    /// Parameters touched per token (always-used + top-k experts), for the
    /// compute bound.
    pub active_params: u64,
    /// Bytes per parameter of the routed experts (quant density).
    pub expert_bytes_per_param: f64,
    pub attention_layers: u32,
    pub mla_layers: u32,
    pub recurrent_layers: u32,
    pub swa_layers: u32,
    /// Largest single routed-expert tensor. With experts in host memory,
    /// llama.cpp copies one such tensor into the GPU compute buffer to run a
    /// prompt batch on the GPU (op offload).
    pub max_expert_tensor_bytes: u64,
    #[serde(skip)]
    pub layers: Vec<LayerAccount>,
}

impl ModelAccounting {
    /// Decode layers that keep a growing cache.
    pub fn cached_layers(&self) -> impl Iterator<Item = &LayerAccount> {
        self.layers
            .iter()
            .filter(|l| l.kind != LayerKind::Mtp && l.k_elems_per_token + l.v_elems_per_token > 0)
    }

    /// Fixed recurrent-state bytes (f32) for `n_seq` parallel sequences.
    pub fn state_bytes(&self, n_seq: u32) -> u64 {
        let elems: u64 = self.layers.iter().filter(|l| l.kind != LayerKind::Mtp).map(|l| l.state_elems).sum();
        elems * 4 * n_seq as u64
    }

    /// True when sparse-MLA layers keep a DSA indexer cache.
    pub fn has_indexer(&self) -> bool {
        self.layers.iter().any(|l| l.idx_elems_per_token > 0)
    }
}

/// Fully recurrent architectures (`llm_arch_is_recurrent`, b12182).
const ALL_RECURRENT: &[&str] = &["mamba", "mamba2", "rwkv6", "rwkv6qwen2", "rwkv7", "arwkv7"];

/// Hybrids where llama.cpp marks a block recurrent when its KV head count is
/// zero (`is_recr_impl[il] = n_head_kv(il) == 0` in each model's hparams).
const HEAD_KV_RULE: &[&str] = &[
    "jamba",
    "plamo2",
    "granitehybrid",
    "lfm2",
    "lfm2moe",
    "lfm3",
    "nemotron_h",
    "nemotron_h_moe",
    "metis16",
    "kimi-linear",
    "bailingmoe3",
    "kimi-k3",
    "kimi-k31",
    "glm5-next",
];

/// Hybrid layouts this planner doesn't size yet, refused with a reason.
const NOT_SIZED: &[(&str, &str)] = &[
    ("falcon-h1", "every block runs attention and Mamba side by side"),
    ("deepseek4", "its compressed attention state isn't modelled"),
    ("deepseek41", "its compressed attention state isn't modelled"),
    ("qwen4exp", "its per-layer embedding conv history isn't modelled"),
];

/// Hybrids that pick recurrent blocks from `attention.recurrent_layers` or,
/// failing that, make every `full_attention_interval`-th block full attention.
/// `Some(None)`: the rule applies but the interval has no default.
fn interval_rule(arch: &str) -> Option<Option<u64>> {
    match arch {
        "minimax-01" => Some(Some(8)),
        "qwen3next" | "qwen35" | "qwen35moe" => Some(Some(4)),
        "clover_moe" => Some(None),
        _ => None,
    }
}

/// Every architecture llama.cpp runs with recurrent memory, fully or as a
/// hybrid (`llm_arch_is_recurrent` / `llm_arch_is_hybrid`, b12182).
fn is_recurrent_arch(arch: &str) -> bool {
    ALL_RECURRENT.contains(&arch)
        || HEAD_KV_RULE.contains(&arch)
        || NOT_SIZED.iter().any(|(a, _)| *a == arch)
        || interval_rule(arch).is_some()
}

/// Routed-expert bank tensor (`blk.N.ffn_{gate,up,down,gate_up}_exps.*`).
fn is_routed_expert(name: &str) -> bool {
    ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_up_exps"]
        .iter()
        .any(|p| name.contains(p))
}

fn block_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (n, _) = rest.split_once('.')?;
    n.parse().ok()
}

fn elems(t: &TensorInfo) -> u64 {
    t.dims.iter().product::<u64>().max(1)
}

/// Account a model from its primary GGUF path (split siblings are found the
/// way llama.cpp finds them).
pub fn account_model(primary: &Path) -> Result<ModelAccounting, AppError> {
    let files = gguf::split_files(primary)?;
    let mut headers = Vec::with_capacity(files.len());
    for f in &files {
        headers.push(gguf::read_tensor_header(f)?);
    }
    account_headers(&headers)
}

/// Account from already-read headers. Metadata comes from the first header
/// (split shards 2..N carry only `split.*` keys); tensors from all of them.
pub fn account_headers(headers: &[GgufTensorHeader]) -> Result<ModelAccounting, AppError> {
    let meta = headers
        .first()
        .ok_or_else(|| AppError::GgufParseFailed("no GGUF files".into()))?;
    let arch = meta
        .get_string("general.architecture")
        .ok_or_else(|| AppError::GgufParseFailed("missing general.architecture".into()))?
        .to_string();
    let key = |k: &str| format!("{arch}.{k}");
    let u = |k: &str| meta.get_u64(&key(k));

    let tensors: Vec<&TensorInfo> = headers.iter().flat_map(|h| h.tensors.iter()).collect();
    if tensors.is_empty() {
        return Err(AppError::GgufParseFailed("GGUF has no tensors".into()));
    }

    let n_blocks = u("block_count").unwrap_or_else(|| {
        tensors.iter().filter_map(|t| block_index(&t.name)).max().map_or(0, |m| m as u64 + 1)
    }) as u32;
    let nextn = u("nextn_predict_layers").unwrap_or(0).min(n_blocks as u64) as u32;
    let n_embd = u("embedding_length").unwrap_or(0);
    let n_head = u("attention.head_count").unwrap_or(0);
    let expert_count = u("expert_count").unwrap_or(0) as u32;
    let expert_used = u("expert_used_count").unwrap_or(0) as u32;
    let ctx_train = u("context_length").unwrap_or(4096);
    let head_kv_layers = meta.get_u64_array(&key("attention.head_count_kv")).map(|a| a.to_vec());
    let head_kv_scalar = u("attention.head_count_kv").or(Some(n_head).filter(|&h| h > 0));
    let key_len = u("attention.key_length")
        .or_else(|| (n_head > 0).then(|| n_embd / n_head))
        .unwrap_or(0);
    let val_len = u("attention.value_length").unwrap_or(key_len);
    let kv_lora = u("attention.kv_lora_rank").unwrap_or(0);
    let rope_dim = u("rope.dimension_count").unwrap_or(0);
    let idx_key_len = u("attention.indexer.key_length").unwrap_or(0);
    let idx_kpool = u("attention.indexer.kpool").unwrap_or(1).max(1);
    let swa_window = u("attention.sliding_window").map(|w| w.min(u32::MAX as u64) as u32);
    let swa_pattern = meta
        .get_bool_array(&key("attention.sliding_window_pattern"))
        .map(|a| a.to_vec())
        .or_else(|| {
            // Older SWA models store a period (or nothing) instead of a
            // per-layer array: every `n`-th layer is full attention, the rest
            // slide (`llama_hparams::set_swa_pattern`, dense_first = false).
            swa_window?;
            let n = u("attention.sliding_window_pattern").or(match arch.as_str() {
                "gemma2" | "gpt-oss" => Some(2),
                "cohere2" => Some(4),
                "gemma3n" => Some(5),
                "gemma3" => Some(6),
                _ => None,
            })?;
            Some((0..n_blocks as u64).map(|il| n == 0 || il % n < n - 1).collect())
        });
    let conv_kernel = u("ssm.conv_kernel").unwrap_or(4);
    let ssm_state = u("ssm.state_size").unwrap_or(0);
    let ssm_inner = u("ssm.inner_size").unwrap_or(0);
    let kda_head_dim = u("kda.head_dim").unwrap_or(0);
    // RWKV: every block keeps token-shift rows plus a WKV state per sequence
    // (`llama_hparams::n_embd_r` / `n_embd_s`).
    let wkv_head_size = u("wkv.head_size").unwrap_or(0);
    let token_shift_count = u("token_shift_count").unwrap_or(2);
    // LFM2 short-convolution blocks keep `l_cache - 1` rows of n_embd.
    let shortconv_l_cache = u("shortconv.l_cache").unwrap_or(0);
    // Gemma 4: the last `shared_kv_layers` blocks read an earlier block's
    // cache and own none (`kv_source`).
    let shared_kv = u("attention.shared_kv_layers").unwrap_or(0).min(n_blocks as u64) as u32;
    let expert_ffn = u("expert_feed_forward_length").unwrap_or(0);
    let mut unsupported: Option<String> = None;
    // Which blocks llama.cpp runs with recurrent memory: the explicit
    // `attention.recurrent_layers` flags, else every `full_attention_interval`-th
    // block is full attention for the hybrids that default to that rule.
    let recr_key = key("attention.recurrent_layers");
    let recr_mask: Option<Vec<bool>> = meta.get_layer_flags(&recr_key, n_blocks as usize).or_else(|| {
        if meta.has_key(&recr_key) {
            unsupported = Some(format!("{recr_key} is stored in a form this planner can't read"));
            return None;
        }
        let default = interval_rule(&arch)?;
        let n = u("full_attention_interval").or(default)?;
        (n > 0).then(|| (0..n_blocks as u64).map(|i| (i + 1) % n != 0).collect())
    });
    // Archs that mark a block recurrent by zero KV heads read the KV head
    // metadata as the rule, so only an explicit value identifies a block.
    let head_kv_rule = HEAD_KV_RULE.contains(&arch.as_str())
        && (head_kv_layers.is_some() || u("attention.head_count_kv").is_some());
    let hybrid = is_recurrent_arch(&arch);
    if let Some((_, why)) = NOT_SIZED.iter().find(|(a, _)| *a == arch) {
        unsupported = Some(why.to_string());
    }

    let mut layers: Vec<LayerAccount> = (0..n_blocks)
        .map(|i| LayerAccount {
            index: i,
            kind: LayerKind::Attention,
            always_bytes: 0,
            expert_bytes: 0,
            n_expert: 0,
            expert_layout: None,
            k_elems_per_token: 0,
            v_elems_per_token: 0,
            idx_elems_per_token: 0,
            swa_window: None,
            state_elems: 0,
        })
        .collect();
    let mut per_layer: Vec<Vec<&TensorInfo>> = vec![Vec::new(); n_blocks as usize];

    let mut total_bytes = 0u64;
    let mut token_embd_bytes = 0u64;
    let mut output_bytes = 0u64;
    let mut has_output_weight = false;
    let mut global_always = 0u64;
    let mut global_always_elems = 0u64;
    for t in &tensors {
        total_bytes += t.bytes;
        match block_index(&t.name) {
            Some(i) if (i as usize) < per_layer.len() => per_layer[i as usize].push(t),
            Some(_) => {} // stray block beyond block_count: counted in total only
            None => {
                if t.name.starts_with("token_embd.") {
                    token_embd_bytes += t.bytes;
                } else {
                    if t.name.starts_with("output") {
                        output_bytes += t.bytes;
                    }
                    if t.name == "output.weight" {
                        has_output_weight = true;
                    }
                    global_always += t.bytes;
                    global_always_elems += elems(t);
                }
            }
        }
    }

    let mut always_elems_active = global_always_elems;
    let mut active_expert_elems = 0f64;
    let mut active_expert_bytes = 0f64;
    let mut expert_elems_total = 0u64;
    let mut max_expert_tensor_bytes = 0u64;
    for (i, ts) in per_layer.iter().enumerate() {
        let l = &mut layers[i];
        let is_mtp = (i as u32) >= n_blocks - nextn;
        let has = |frag: &str| ts.iter().any(|t| t.name.contains(frag));
        let find = |frag: &str| ts.iter().find(|t| t.name.contains(frag)).copied();
        let mut layer_expert_elems = 0u64;
        for t in ts {
            if is_routed_expert(&t.name) {
                l.expert_bytes += t.bytes;
                if !is_mtp {
                    max_expert_tensor_bytes = max_expert_tensor_bytes.max(t.bytes);
                }
                layer_expert_elems += elems(t);
                // dims = [n_in, n_out, n_expert]
                if t.dims.len() >= 3 {
                    l.n_expert = l.n_expert.max(t.dims[2] as u32);
                }
            } else {
                l.always_bytes += t.bytes;
                if !is_mtp {
                    always_elems_active += elems(t);
                }
            }
        }
        l.expert_layout = find(".ffn_up_exps.weight").zip(find(".ffn_down_exps.weight")).and_then(|(up, down)| {
            Some(ExpertLayout { up_type: up.ggml_type, down_type: down.ggml_type,
                latent: *up.dims.first()?, hidden: *up.dims.get(1)? })
        });
        if l.expert_bytes > 0 && l.n_expert == 0 {
            l.n_expert = expert_count;
        }

        let head_kv = head_kv_layers
            .as_ref()
            .and_then(|a| a.get(i).copied())
            .or(head_kv_scalar)
            .unwrap_or(0);

        let shares_kv = (i as u32) >= n_blocks - shared_kv;
        let mla = has("attn_kv_a_mqa");
        let recurrent_tensors = wkv_head_size > 0 || has(".ssm_") || has(".shortconv");
        // What llama.cpp itself decides for this block, when the file says.
        let marked: Option<bool> = if let Some(m) = &recr_mask {
            m.get(i).copied()
        } else if head_kv_rule {
            Some(head_kv == 0)
        } else if ALL_RECURRENT.contains(&arch.as_str()) {
            Some(true)
        } else {
            None
        };
        l.kind = if is_mtp {
            LayerKind::Mtp
        } else if let Some(recurrent) = marked {
            if recurrent {
                LayerKind::Recurrent
            } else if mla {
                LayerKind::Mla
            } else {
                LayerKind::Attention
            }
        } else if recurrent_tensors {
            LayerKind::Recurrent
        } else if mla {
            LayerKind::Mla
        } else if head_kv == 0 && head_kv_layers.is_some() {
            // Explicit zero KV heads with no SSM tensors we recognise: treat as
            // a stateless/recurrent block rather than inventing a KV cache.
            LayerKind::Recurrent
        } else {
            LayerKind::Attention
        };
        // Fail closed: a block counts as softmax attention only when the file
        // says so. In a hybrid arch a fused QKV is also what a linear-attention
        // block looks like, and a block with no attention tensors at all isn't
        // attention whatever the metadata says.
        if l.kind == LayerKind::Attention && !shares_kv && unsupported.is_none() {
            if hybrid && marked.is_none() {
                unsupported = Some(format!(
                    "the file doesn't say whether block {i} of this hybrid is attention or recurrent"
                ));
            } else if !(has(".attn_k.") || has(".attn_qkv.") || has(".attn_kv.")) {
                unsupported = Some(format!("block {i} has no attention or recurrent tensors this planner recognises"));
            }
        }

        match l.kind {
            LayerKind::Attention if shares_kv => {
                // Reads another block's cache; owns no context memory.
            }
            LayerKind::Attention => {
                // Exact widths from the projection shapes ([n_embd, width]);
                // metadata (head_count_kv × key/value length) only as fallback.
                let k = find("attn_k.weight").and_then(|t| t.dims.get(1).copied());
                let v = find("attn_v.weight").and_then(|t| t.dims.get(1).copied());
                (l.k_elems_per_token, l.v_elems_per_token) = match (k, v) {
                    (Some(k), Some(v)) => (k, v),
                    (Some(k), None) => (k, k),
                    _ => (head_kv * key_len, head_kv * val_len),
                };
                if l.k_elems_per_token + l.v_elems_per_token == 0 && unsupported.is_none() {
                    unsupported = Some(format!("block {i} has attention with no K/V width in the file"));
                }
                if swa_pattern.as_ref().is_some_and(|p| p.get(i).copied().unwrap_or(false)) {
                    l.swa_window = swa_window;
                }
            }
            LayerKind::Mla => {
                // K-only latent cache (flash attention, which the planner
                // always enables).
                l.k_elems_per_token = kv_lora + rope_dim;
                if has("indexer.attn_k") {
                    // One row per token; with key pooling the row holds
                    // `indexer_kpool_row` (3) head-size slots, the last one the
                    // pooled key (llama-hparams.h, b12182).
                    let slots = if idx_kpool > 1 { 3 } else { 1 };
                    l.idx_elems_per_token = idx_key_len * slots;
                }
            }
            LayerKind::Recurrent => {
                let conv_channels: u64 = ts
                    .iter()
                    .filter(|t| t.name.contains(".ssm_conv1d"))
                    .filter_map(|t| t.dims.last().copied())
                    .sum();
                let state = if wkv_head_size > 0 {
                    token_shift_count * n_embd + n_embd * wkv_head_size
                } else if has(".shortconv") {
                    shortconv_l_cache.saturating_sub(1) * n_embd
                } else if arch == "minimax-01" {
                    // Lightning attention: one head_dim² matrix per head and no
                    // conv rows (`n_embd_head_la`, llama-hparams.cpp b12182).
                    key_len * key_len * n_head
                } else if kda_head_dim > 0 {
                    let heads = find("ssm_conv1d_q")
                        .and_then(|t| t.dims.last().copied())
                        .map(|c| c / kda_head_dim)
                        .unwrap_or(n_head);
                    kda_head_dim * kda_head_dim * heads
                } else {
                    ssm_state * ssm_inner
                };
                l.state_elems = if wkv_head_size > 0 || has(".shortconv") {
                    state
                } else {
                    conv_kernel.saturating_sub(1) * conv_channels + state
                };
                // A block with no attention or recurrent tensors at all (an
                // FFN-only block, e.g. Nemotron-H's MLP layers) owns nothing.
                let mixes = has(".attn") || has(".time_mix") || has(".ssm_") || has(".shortconv");
                if l.state_elems == 0 && mixes && unsupported.is_none() {
                    unsupported = Some(format!("block {i} is a recurrent layer of a kind this planner can't size"));
                }
            }
            LayerKind::Mtp => {}
        }

        if !is_mtp && l.expert_bytes > 0 {
            let n = l.n_expert.max(1) as f64;
            let k = (expert_used.max(1) as f64).min(n);
            active_expert_bytes += l.expert_bytes as f64 * k / n;
            active_expert_elems += layer_expert_elems as f64 * k / n;
            expert_elems_total += layer_expert_elems;
        }
    }

    let decode = || layers.iter().filter(|l| l.kind != LayerKind::Mtp);
    let always_bytes = global_always + decode().map(|l| l.always_bytes).sum::<u64>();
    let expert_bytes: u64 = decode().map(|l| l.expert_bytes).sum();
    let mtp_bytes: u64 = layers
        .iter()
        .filter(|l| l.kind == LayerKind::Mtp)
        .map(|l| l.always_bytes + l.expert_bytes)
        .sum();
    let count = |k: LayerKind| layers.iter().filter(|l| l.kind == k).count() as u32;

    let n_vocab = u("vocab_size").unwrap_or_else(|| {
        tensors
            .iter()
            .find(|t| t.name == "token_embd.weight")
            .and_then(|t| t.dims.get(1).copied())
            .unwrap_or(0)
    });

    Ok(ModelAccounting {
        architecture: arch,
        files: headers.len() as u32,
        n_layers: n_blocks - nextn,
        mtp_layers: nextn,
        moe_layers: decode().filter(|l| l.expert_bytes > 0).count() as u32,
        ctx_train,
        n_embd,
        n_vocab,
        expert_count,
        expert_used,
        expert_ffn,
        total_bytes,
        token_embd_bytes,
        always_bytes,
        expert_bytes,
        mtp_bytes,
        output_bytes,
        output_tied: !has_output_weight,
        unsupported,
        active_expert_bytes_per_token: active_expert_bytes as u64,
        active_params: always_elems_active + active_expert_elems as u64,
        expert_bytes_per_param: if expert_elems_total > 0 {
            expert_bytes as f64 / expert_elems_total as f64
        } else {
            0.0
        },
        attention_layers: count(LayerKind::Attention),
        mla_layers: count(LayerKind::Mla),
        recurrent_layers: count(LayerKind::Recurrent),
        swa_layers: layers.iter().filter(|l| l.swa_window.is_some()).count() as u32,
        max_expert_tensor_bytes,
        layers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gguf").join(rel)
    }

    pub(crate) fn glm53() -> ModelAccounting {
        account_model(&fixture("glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf")).unwrap()
    }
    pub(crate) fn qwen36() -> ModelAccounting {
        account_model(&fixture("Qwen3.6-35B-A3B-UD-Q4_K_M.header.gguf")).unwrap()
    }
    pub(crate) fn qwen3_8b() -> ModelAccounting {
        account_model(&fixture("Qwen3-8B-Q4_K_M.header.gguf")).unwrap()
    }

    const GB: f64 = 1e9;

    #[test]
    fn split_gguf_tensor_bytes_match_real_shards() {
        let a = glm53();
        assert_eq!(a.files, 4);
        assert_eq!(a.architecture, "glm5-next");
        // Real shard sizes total 108.72 GB; tensor bytes must land within the
        // header overhead of that.
        let real = 9_429_984u64 + 49_294_975_936 + 49_949_266_048 + 9_466_399_584;
        let diff = real as f64 - a.total_bytes as f64;
        assert!((0.0..64e6).contains(&diff),"tensor bytes {} vs files {real}", a.total_bytes);
    }

    #[test]
    fn glm53_groups_experts_mla_kda_and_mtp() {
        let a = glm53();
        assert_eq!(a.n_layers, 45);
        assert_eq!(a.mtp_layers, 1);
        assert_eq!(a.expert_count, 288);
        assert_eq!(a.expert_used, 8);
        // 3 leading dense blocks, the rest MoE.
        assert_eq!(a.moe_layers, 42);
        // Every 4th block is sparse MLA; the rest are KDA linear attention.
        assert_eq!(a.mla_layers, 11);
        assert_eq!(a.recurrent_layers, 34);
        assert_eq!(a.attention_layers, 0);
        // Routed experts dominate a 109 GB UD-Q2_K_XL; always-used is small.
        assert!(a.expert_bytes as f64 > 95.0 * GB, "experts {}", a.expert_bytes);
        assert!((3.0 * GB..9.0 * GB).contains(&(a.always_bytes as f64)), "always {}", a.always_bytes);
        assert!(a.mtp_bytes as f64 > 1.0 * GB);
        // top-8 of 288 → ~2.8 % of expert bytes per token.
        let frac = a.active_expert_bytes_per_token as f64 / a.expert_bytes as f64;
        assert!((frac - 8.0 / 288.0).abs() < 1e-3, "frac {frac}");
        // KDA state: 64 heads × 128² + conv, f32, per KDA layer.
        let kda = a.layers.iter().find(|l| l.kind == LayerKind::Recurrent).unwrap();
        assert_eq!(kda.state_elems, 128 * 128 * 64 + 3 * 3 * 8192);
        // MLA latent: kv_lora_rank 512 (+0 rope), K only; the DSA indexer
        // keeps 3 × 128-element slots per token (kpool row).
        let mla = a.layers.iter().find(|l| l.kind == LayerKind::Mla).unwrap();
        assert_eq!(mla.k_elems_per_token, 512);
        assert_eq!(mla.v_elems_per_token, 0);
        assert_eq!(mla.idx_elems_per_token, 384);
        assert!(a.has_indexer());
        assert!(a.max_expert_tensor_bytes > 0);
    }

    #[test]
    fn qwen36_hybrid_moe_accounting() {
        let a = qwen36();
        assert_eq!(a.architecture, "qwen35moe");
        assert_eq!(a.n_layers, 40);
        assert_eq!(a.moe_layers, 40);
        assert_eq!(a.attention_layers, 10);
        assert_eq!(a.recurrent_layers, 30);
        let real = 22_134_528_992f64;
        assert!((real - a.total_bytes as f64) < 64e6 && a.total_bytes as f64 <= real);
        // GQA: 2 KV heads × 256 dims, K and V, on the 10 full-attention layers.
        assert_eq!(a.cached_layers().count(), 10);
        assert!(a.cached_layers().all(|l| l.k_elems_per_token == 512 && l.v_elems_per_token == 512));
        // Gated delta-net state: 128 × 4096 + 3 × 8192 conv per layer.
        let rec = a.layers.iter().find(|l| l.kind == LayerKind::Recurrent).unwrap();
        assert_eq!(rec.state_elems, 128 * 4096 + 3 * 8192);
        assert!(a.active_params > 2_500_000_000 && a.active_params < 4_000_000_000, "{}", a.active_params);
    }

    #[test]
    fn dense_model_has_no_experts_and_exact_gqa() {
        let a = qwen3_8b();
        assert_eq!(a.expert_bytes, 0);
        assert_eq!(a.n_layers, 36);
        assert_eq!(a.attention_layers, 36);
        // 8 KV heads × 128 dims for K and for V on all 36 layers.
        assert_eq!(a.cached_layers().count(), 36);
        assert!(a.cached_layers().all(|l| l.k_elems_per_token == 1024 && l.v_elems_per_token == 1024));
        let real = 5_027_784_512f64;
        assert!((real - a.total_bytes as f64) < 16e6 && a.total_bytes as f64 <= real);
        assert_eq!(a.moe_layers, 0);
    }

    #[test]
    fn split_name_detection() {
        let p = fixture("Qwen3-8B-Q4_K_M.header.gguf");
        assert_eq!(gguf::split_files(&p).unwrap(), vec![p]);
        let missing = std::env::temp_dir().join("x-00001-of-00002.gguf");
        assert!(gguf::split_files(&missing).is_err());
    }

    /// An f32 tensor record (4 bytes per element).
    fn t(name: &str, dims: &[u64]) -> TensorInfo {
        let bytes = dims.iter().product::<u64>() * 4;
        TensorInfo { name: name.into(), dims: dims.to_vec(), ggml_type: 0, offset: 0, bytes }
    }

    fn synthetic(arch: &str, ints: &[(&str, u64)], tensors: Vec<TensorInfo>) -> ModelAccounting {
        let ints: Vec<(String, u64)> = ints.iter().map(|(k, v)| (format!("{arch}.{k}"), *v)).collect();
        let ints: Vec<(&str, u64)> = ints.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let h = GgufTensorHeader::synthetic(&[("general.architecture", arch)], &ints, tensors);
        account_headers(&[h]).unwrap()
    }

    /// RWKV blocks keep token-shift rows plus a WKV state per sequence,
    /// `token_shift_count * n_embd + n_embd * wkv_head_size` f32 values
    /// (llama.cpp `n_embd_r` + `n_embd_s`), and no KV cache.
    #[test]
    fn rwkv_state_is_sized() {
        let n_embd = 2048;
        let mut ts = vec![t("token_embd.weight", &[n_embd, 65536]), t("output.weight", &[n_embd, 65536])];
        for i in 0..4 {
            ts.push(t(&format!("blk.{i}.time_mix_key.weight"), &[n_embd, n_embd]));
            ts.push(t(&format!("blk.{i}.channel_mix_key.weight"), &[n_embd, 7168]));
        }
        let a = synthetic(
            "rwkv6",
            &[("block_count", 4), ("embedding_length", n_embd), ("wkv.head_size", 64), ("token_shift_count", 2)],
            ts,
        );
        assert!(a.unsupported.is_none(), "{:?}", a.unsupported);
        assert!(a.layers.iter().all(|l| l.kind == LayerKind::Recurrent));
        assert!(a.layers.iter().all(|l| l.state_elems == 2 * n_embd + n_embd * 64));
        assert_eq!(a.state_bytes(1), 4 * (2 * n_embd + n_embd * 64) * 4);
        assert_eq!(a.cached_layers().count(), 0);
    }

    /// A recurrent block of a kind the planner doesn't know gets flagged, not
    /// sized as zero; FFN-only blocks with no mixer own nothing and are fine.
    #[test]
    fn unknown_recurrent_layout_is_flagged() {
        let n_embd = 1024;
        let base = || vec![t("token_embd.weight", &[n_embd, 32000]), t("output.weight", &[n_embd, 32000])];
        // Explicit zero KV heads and an attention-like mixer the planner can't size.
        let mut ts = base();
        ts.push(t("blk.0.attn_q.weight", &[n_embd, n_embd]));
        ts.push(t("blk.0.ffn_up.weight", &[n_embd, 4096]));
        let mut ints = vec![("block_count", 1), ("embedding_length", n_embd), ("attention.head_count", 16)];
        ints.push(("attention.head_count_kv", 0));
        let a = synthetic("mystery", &ints, ts);
        let why = a.unsupported.as_deref().expect("flagged");
        assert!(why.contains("block 0"), "{why}");

        // Standard attention with no K/V width anywhere in the file.
        let mut ts = base();
        ts.push(t("blk.0.attn_q.weight", &[n_embd, n_embd]));
        let a = synthetic("mystery", &[("block_count", 1), ("embedding_length", n_embd)], ts);
        assert!(a.unsupported.is_some());
    }

    /// The real headers of every family the planner targets classify every
    /// block positively: nothing is refused.
    #[test]
    fn real_headers_are_all_identified() {
        for f in [
            "GLM-4.7-Flash-Q4_K_M.header.gguf",
            "Qwen3-0.6B-Q8_0.header.gguf",
            "Qwen3-8B-Q4_K_M.header.gguf",
            "Qwen3.6-35B-A3B-UD-Q4_K_M.header.gguf",
            "gemma-3-1b-it-Q4_K_M.header.gguf",
            "glm53/GLM-5.3-Flash-UD-Q2_K_XL-00001-of-00004.gguf",
        ] {
            let a = account_model(&fixture(f)).unwrap();
            assert!(a.unsupported.is_none(), "{f}: {:?}", a.unsupported);
        }
    }

    /// MiniMax-style lightning attention: a fused QKV plus gate and no SSM
    /// tensors, so only the recurrent-layer rule tells it from attention.
    fn lightning_blocks(n_blocks: u64, full: &[u64]) -> Vec<TensorInfo> {
        let n_embd = 1024;
        let mut ts = vec![t("token_embd.weight", &[n_embd, 32000]), t("output.weight", &[n_embd, 32000])];
        for i in 0..n_blocks {
            if full.contains(&i) {
                ts.push(t(&format!("blk.{i}.attn_q.weight"), &[n_embd, n_embd]));
                ts.push(t(&format!("blk.{i}.attn_k.weight"), &[n_embd, 256]));
                ts.push(t(&format!("blk.{i}.attn_v.weight"), &[n_embd, 256]));
            } else {
                ts.push(t(&format!("blk.{i}.attn_qkv.weight"), &[n_embd, 3 * n_embd]));
                ts.push(t(&format!("blk.{i}.attn_gate.weight"), &[n_embd, n_embd]));
            }
            ts.push(t(&format!("blk.{i}.attn_output.weight"), &[n_embd, n_embd]));
        }
        ts
    }

    const LIGHTNING_INTS: [(&str, u64); 4] = [
        ("block_count", 8),
        ("embedding_length", 1024),
        ("attention.head_count", 8),
        ("attention.head_count_kv", 2),
    ];

    /// b12182 makes every 8th MiniMax block full attention by default and the
    /// rest lightning attention with a 128² state per head; a GQA cache only
    /// on the full blocks.
    #[test]
    fn minimax_lightning_blocks_are_recurrent() {
        let a = synthetic("minimax-01", &LIGHTNING_INTS, lightning_blocks(8, &[7]));
        assert!(a.unsupported.is_none(), "{:?}", a.unsupported);
        assert_eq!(a.recurrent_layers, 7);
        assert_eq!(a.attention_layers, 1);
        let rec = a.layers.iter().filter(|l| l.kind == LayerKind::Recurrent);
        assert!(rec.clone().all(|l| l.state_elems == 128 * 128 * 8 && l.k_elems_per_token == 0));
        let cached: Vec<u32> = a.cached_layers().map(|l| l.index).collect();
        assert_eq!(cached, vec![7]);
        assert_eq!(a.state_bytes(1), 7 * 128 * 128 * 8 * 4);
    }

    /// An explicit `attention.recurrent_layers` array wins over the interval,
    /// for any architecture that carries it.
    #[test]
    fn recurrent_layers_metadata_is_honoured() {
        let ints: Vec<(String, u64)> = LIGHTNING_INTS.iter().map(|(k, v)| (format!("minimax-01.{k}"), *v)).collect();
        let ints: Vec<(&str, u64)> = ints.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let flags = [true, true, true, false, true, true, true, false];
        let h = GgufTensorHeader::synthetic(&[("general.architecture", "minimax-01")], &ints, lightning_blocks(8, &[3, 7]))
            .with_bool_array("minimax-01.attention.recurrent_layers", &flags);
        let a = account_headers(&[h]).unwrap();
        assert!(a.unsupported.is_none(), "{:?}", a.unsupported);
        let cached: Vec<u32> = a.cached_layers().map(|l| l.index).collect();
        assert_eq!(cached, vec![3, 7]);
        assert_eq!(a.recurrent_layers, 6);
    }

    /// Fail closed: a hybrid whose file doesn't say which blocks are recurrent
    /// (Clover has no default interval) is refused, not planned as attention;
    /// so are hybrid layouts the planner doesn't size.
    #[test]
    fn unidentified_hybrid_blocks_are_refused() {
        let a = synthetic("clover_moe", &LIGHTNING_INTS, lightning_blocks(8, &[7]));
        let why = a.unsupported.as_deref().expect("refused");
        assert!(why.contains("block 0"), "{why}");

        let mut ints = LIGHTNING_INTS.to_vec();
        ints.push(("full_attention_interval", 8));
        let a = synthetic("clover_moe", &ints, lightning_blocks(8, &[7]));
        assert!(a.unsupported.is_some(), "clover lightning state has no formula here");

        let a = synthetic("falcon-h1", &LIGHTNING_INTS, lightning_blocks(8, &[0, 1, 2, 3, 4, 5, 6, 7]));
        assert!(a.unsupported.as_deref().is_some_and(|w| w.contains("side by side")));
    }

    /// Gemma 4: the last `shared_kv_layers` blocks read an earlier block's
    /// cache, so only the cache-owning blocks are charged.
    #[test]
    fn shared_kv_layers_own_no_cache() {
        let n_embd = 1024;
        let mut ts = vec![t("token_embd.weight", &[n_embd, 32000])];
        for i in 0..6 {
            ts.push(t(&format!("blk.{i}.attn_k.weight"), &[n_embd, 256]));
            ts.push(t(&format!("blk.{i}.attn_v.weight"), &[n_embd, 256]));
        }
        let ints = [("block_count", 6), ("embedding_length", n_embd), ("attention.head_count", 8)];
        let own = synthetic("gemma4", &ints, ts.clone());
        assert_eq!(own.cached_layers().count(), 6);
        let mut ints = ints.to_vec();
        ints.push(("attention.shared_kv_layers", 2));
        let shared = synthetic("gemma4", &ints, ts);
        assert!(shared.unsupported.is_none());
        let owning: Vec<u32> = shared.cached_layers().map(|l| l.index).collect();
        assert_eq!(owning, vec![0, 1, 2, 3]);
        assert!(shared.output_tied, "no output.weight: tied to token_embd");
    }
}
