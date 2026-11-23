use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::AppError;

/// Key metadata extracted from a GGUF file header.
/// All fields are optional — parse failures are logged, not fatal.
#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct GgufMeta {
    /// e.g. "llama", "mistral", "whisper"
    pub architecture: Option<String>,
    /// Maximum context length from training.
    pub context_length: Option<u64>,
    /// Number of transformer blocks.
    pub block_count: Option<u64>,
    /// Embedding / hidden dimension.
    pub embedding_length: Option<u64>,
    /// llama.cpp pooling enum from `{arch}.pooling_type` (0=none, 1=mean, 2=cls, 3=last).
    #[serde(default)]
    pub pooling_type: Option<u64>,
    /// Multi-head attention query-head count (used for head_dim derivation).
    pub attention_head_count: Option<u64>,
    /// KV-head count for grouped-query / multi-query architectures. Falls back to
    /// `attention_head_count` at the call site when absent (i.e. standard MHA).
    pub attention_head_count_kv: Option<u64>,
    /// RoPE rotary embedding dimension. Used as a head_dim fallback when the
    /// canonical `attention.key_length` is absent.
    pub rope_dimension_count: Option<u64>,
    /// Canonical KV-cache head dimension (`{arch}.attention.key_length`). This
    /// is the authoritative per-head K/V width and is preferred over
    /// `rope_dimension_count` for KV-cache sizing — gemma-family models set a
    /// `key_length` (e.g. 512) that differs from the RoPE dimension.
    pub attention_key_length: Option<u64>,
    /// Per-head V width (`{arch}.attention.value_length`). Parsed for
    /// completeness/future use; K and V widths are equal on every architecture
    /// we size today, so `head_dim()` keys off `attention_key_length`.
    pub attention_value_length: Option<u64>,
    /// Full per-layer KV-head counts (`{arch}.attention.head_count_kv` when the
    /// header stores it as an ARRAY — sliding-window models like the Gemma
    /// family vary KV heads by layer, e.g. global layers use fewer heads than
    /// local ones). `None` for the usual scalar case; `attention_head_count_kv`
    /// still carries the max for fallback sizing.
    #[serde(default)]
    pub attention_head_count_kv_layers: Option<Vec<u64>>,
    /// Sliding-window size in tokens (`{arch}.attention.sliding_window`). Present
    /// on sliding-window-attention (SWA) models; KV for the local layers caps at
    /// this many tokens instead of growing with the full context.
    #[serde(default)]
    pub attention_sliding_window: Option<u64>,
    /// Per-layer attention pattern (`{arch}.attention.sliding_window_pattern`):
    /// `true` = sliding-window (local) layer, `false` = full-attention (global)
    /// layer, one entry per transformer block. Drives SWA-aware KV sizing.
    #[serde(default)]
    pub sliding_window_pattern: Option<Vec<bool>>,
    /// Per-head K width for sliding-window layers
    /// (`{arch}.attention.key_length_swa`). Falls back to `attention_key_length`
    /// when absent.
    #[serde(default)]
    pub attention_key_length_swa: Option<u64>,
    /// Per-head V width for sliding-window layers
    /// (`{arch}.attention.value_length_swa`). Falls back to
    /// `attention_value_length` when absent.
    #[serde(default)]
    pub attention_value_length_swa: Option<u64>,
    /// Quantization label (e.g. "Q4_K_M"), decoded from `general.file_type` or filename.
    pub quantization: Option<String>,
    /// Estimated parameter count in billions (from `general.parameter_count` / 1e9).
    pub params_b: Option<f64>,
    /// True when any `clip.*` or `vision.*` key is present in metadata.
    pub vision: bool,
    /// Size of the file on disk in bytes.
    pub file_size: u64,
}

impl GgufMeta {
    /// Derive head dimension for KV-cache size calculations.
    ///
    /// Preference order:
    /// 1. `attention_key_length` — the canonical KV head width. Authoritative
    ///    when present (gemma-family models set this and it differs from the
    ///    RoPE dimension).
    /// 2. `rope_dimension_count` — most other architectures set this explicitly.
    /// 3. `embedding_length / attention_head_count` — fallback for models that
    ///    omit both keys.
    ///
    /// Returns `None` when no derivation is possible (missing fields or zero
    /// head count).
    pub fn head_dim(&self) -> Option<u64> {
        if let Some(k) = self.attention_key_length {
            return Some(k);
        }
        if let Some(rope) = self.rope_dimension_count {
            return Some(rope);
        }
        let (e, h) = (self.embedding_length?, self.attention_head_count?);
        if h == 0 {
            return None;
        }
        Some(e / h)
    }
}

// ---------------------------------------------------------------------------
// file_type integer → quantization string
// GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
// ---------------------------------------------------------------------------

fn file_type_to_quant(ft: u64) -> Option<&'static str> {
    Some(match ft {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "Q8_K",
        20 => "IQ2_XXS",
        21 => "IQ2_XS",
        22 => "IQ3_XXS",
        23 => "IQ1_S",
        24 => "IQ4_NL",
        25 => "IQ3_S",
        26 => "IQ2_S",
        27 => "IQ4_XS",
        28 => "IQ1_M",
        29 => "BF16",
        30 => "Q4_0_4_4",
        31 => "Q4_0_4_8",
        32 => "Q4_0_8_8",
        33 => "TQ1_0",
        34 => "TQ2_0",
        _ => return None,
    })
}

/// Extract quantization from filename via regex pattern `\.([A-Z0-9_]+)\.gguf$`.
fn quant_from_filename(path: &Path) -> Option<String> {
    let filename = path.file_name()?.to_str()?;
    let lower = filename.to_lowercase();
    if !lower.ends_with(".gguf") {
        return None;
    }
    let stem = &filename[..filename.len() - 5]; // strip ".gguf"
    // Find the last '.' in the stem
    let dot_pos = stem.rfind('.')?;
    let token = &stem[dot_pos + 1..];
    // Validate: only uppercase letters, digits, underscores
    if token.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !token.is_empty() {
        Some(token.to_uppercase())
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Minimal header reader
//
// Hand-rolled because the gguf crates materialize every metadata value and
// hard-cap array lengths (gguf-rs-lib rejects anything over 65 536 elements,
// which fails on Qwen-class vocabularies at 150k–250k entries). The catalog
// only needs a handful of scalar keys, so this reader keeps scalars and seeks
// past arrays without ever allocating them.
// ---------------------------------------------------------------------------

const GGUF_MAGIC: u32 = 0x4655_4747; // "GGUF" little-endian
/// Sanity cap on the header's declared KV count, not on array lengths.
const MAX_KV_COUNT: u64 = 65_536;
/// Strings longer than this are skipped, not kept (chat templates etc.).
const MAX_KEPT_STRING: u64 = 64 * 1024;
/// Integer arrays up to this many elements are materialized (e.g. per-layer
/// `attention.head_count_kv`, one entry per transformer block). Anything larger
/// (vocab token-id tables etc.) is skipped — we only need small per-layer config
/// arrays, never the giant tokenizer arrays.
const MAX_KEPT_INT_ARRAY: u64 = 1024;

#[derive(Debug, Clone)]
enum MetaValue {
    U64(u64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// Array of widened unsigned integers. Kept only for short arrays of
    /// fixed-size integer elements (e.g. the per-layer `attention.head_count_kv`
    /// emitted by gemma-family architectures); larger or non-integer arrays
    /// stay `Skipped`.
    U64Array(Vec<u64>),
    /// Array of booleans. Kept only for short bool arrays (e.g. the per-layer
    /// `attention.sliding_window_pattern` emitted by sliding-window models);
    /// larger arrays stay `Skipped`.
    BoolArray(Vec<bool>),
    /// Oversized / non-integer arrays and oversized strings: key recorded,
    /// content skipped.
    Skipped,
}

struct MetaMap(HashMap<String, MetaValue>);

impl MetaMap {
    /// Read a scalar unsigned integer. For a key stored as a (kept) integer
    /// array — e.g. the per-layer `attention.head_count_kv` in gemma4 — returns
    /// the **max** element. That is a conservative upper bound for KV-cache
    /// sizing (it bounds the largest per-layer KV-head count).
    fn get_u64(&self, key: &str) -> Option<u64> {
        match self.0.get(key)? {
            MetaValue::U64(v) => Some(*v),
            MetaValue::U64Array(a) => a.iter().copied().max(),
            _ => None,
        }
    }

    /// Borrow a kept integer array verbatim (e.g. the per-layer
    /// `attention.head_count_kv`). Unlike `get_u64`, this preserves the full
    /// per-layer sequence rather than collapsing to the max.
    fn get_u64_array(&self, key: &str) -> Option<&[u64]> {
        match self.0.get(key)? {
            MetaValue::U64Array(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    /// Borrow a kept boolean array (e.g. `attention.sliding_window_pattern`).
    fn get_bool_array(&self, key: &str) -> Option<&[bool]> {
        match self.0.get(key)? {
            MetaValue::BoolArray(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    /// A per-layer flag the way llama.cpp's `get_key_or_arr` reads one: a bool
    /// or integer array (non-zero = set), or a scalar applied to all `n`
    /// layers. `None` when the key is absent or in a form that wasn't kept
    /// (`has_key` tells the two apart).
    fn get_layer_flags(&self, key: &str, n: usize) -> Option<Vec<bool>> {
        match self.0.get(key)? {
            MetaValue::BoolArray(a) => Some(a.clone()),
            MetaValue::U64Array(a) => Some(a.iter().map(|&v| v != 0).collect()),
            MetaValue::Bool(b) => Some(vec![*b; n]),
            MetaValue::U64(v) => Some(vec![*v != 0; n]),
            _ => None,
        }
    }

    fn get_string(&self, key: &str) -> Option<&str> {
        match self.0.get(key)? {
            MetaValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn keys(&self) -> impl Iterator<Item = &String> {
        self.0.keys()
    }
}

fn read_u32<R: Read>(r: &mut R) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

fn read_u64<R: Read>(r: &mut R) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

/// Skip `n` bytes. Short skips read through the BufReader (cheap, keeps the
/// buffer warm); long skips seek (discards the buffer but avoids the I/O).
fn skip<R: Read + Seek>(r: &mut R, n: u64) -> std::io::Result<()> {
    if n == 0 {
        return Ok(());
    }
    if n <= 4096 {
        let mut scratch = [0u8; 4096];
        let mut left = n as usize;
        while left > 0 {
            let take = left.min(scratch.len());
            r.read_exact(&mut scratch[..take])?;
            left -= take;
        }
        Ok(())
    } else {
        r.seek(SeekFrom::Current(n as i64)).map(|_| ())
    }
}

/// Fixed on-disk size of a scalar GGUF value type, or None for string/array.
fn scalar_size(ty: u32) -> Option<u64> {
    Some(match ty {
        0 | 1 | 7 => 1,      // u8, i8, bool
        2 | 3 => 2,          // u16, i16
        4 | 5 | 6 => 4,      // u32, i32, f32
        10 | 11 | 12 => 8,   // u64, i64, f64
        _ => return None,    // 8 = string, 9 = array
    })
}

fn bad(msg: impl Into<String>) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, msg.into())
}

/// Read a scalar value of type `ty`, widening integers to u64 (negative
/// signed values are dropped — no catalog field is meaningfully negative).
fn read_scalar<R: Read>(r: &mut R, ty: u32) -> std::io::Result<Option<MetaValue>> {
    let mut b8 = [0u8; 8];
    Ok(match ty {
        0 => {
            r.read_exact(&mut b8[..1])?;
            Some(MetaValue::U64(b8[0] as u64))
        }
        1 => {
            r.read_exact(&mut b8[..1])?;
            let v = b8[0] as i8;
            (v >= 0).then(|| MetaValue::U64(v as u64))
        }
        2 => {
            r.read_exact(&mut b8[..2])?;
            Some(MetaValue::U64(u16::from_le_bytes([b8[0], b8[1]]) as u64))
        }
        3 => {
            r.read_exact(&mut b8[..2])?;
            let v = i16::from_le_bytes([b8[0], b8[1]]);
            (v >= 0).then(|| MetaValue::U64(v as u64))
        }
        4 => Some(MetaValue::U64(read_u32(r)? as u64)),
        5 => {
            r.read_exact(&mut b8[..4])?;
            let v = i32::from_le_bytes([b8[0], b8[1], b8[2], b8[3]]);
            (v >= 0).then(|| MetaValue::U64(v as u64))
        }
        6 => {
            r.read_exact(&mut b8[..4])?;
            Some(MetaValue::F64(f32::from_le_bytes([b8[0], b8[1], b8[2], b8[3]]) as f64))
        }
        7 => {
            r.read_exact(&mut b8[..1])?;
            Some(MetaValue::Bool(b8[0] != 0))
        }
        10 => Some(MetaValue::U64(read_u64(r)?)),
        11 => {
            r.read_exact(&mut b8)?;
            let v = i64::from_le_bytes(b8);
            (v >= 0).then(|| MetaValue::U64(v as u64))
        }
        12 => {
            r.read_exact(&mut b8)?;
            Some(MetaValue::F64(f64::from_le_bytes(b8)))
        }
        other => return Err(bad(format!("unknown scalar type {other}"))),
    })
}

/// True for the GGUF scalar value types that encode an integer (signed or
/// unsigned). Used to decide whether an array's elements can be collected into
/// a `U64Array`.
fn is_int_type(ty: u32) -> bool {
    matches!(ty, 0 | 1 | 2 | 3 | 4 | 5 | 10 | 11)
}

/// Read a single value of an integer scalar type, widening to u64. Negative
/// signed values are dropped (`None`); no per-layer config field is negative.
fn read_int_elem<R: Read>(r: &mut R, ty: u32) -> std::io::Result<Option<u64>> {
    Ok(match read_scalar(r, ty)? {
        Some(MetaValue::U64(v)) => Some(v),
        _ => None,
    })
}

/// Read a GGUF array (type 9). Short arrays of fixed-size integer elements are
/// materialized into a `U64Array` (per-layer config like
/// `attention.head_count_kv`); everything else (long arrays, string/float/nested
/// arrays) is skipped exactly as before, returning `MetaValue::Skipped`.
fn read_array<R: Read + Seek>(r: &mut R) -> std::io::Result<MetaValue> {
    let elem_ty = read_u32(r)?;
    let count = read_u64(r)?;

    if is_int_type(elem_ty) && count <= MAX_KEPT_INT_ARRAY {
        let mut out = Vec::with_capacity(count as usize);
        for _ in 0..count {
            // A dropped negative element simply doesn't contribute to the max;
            // we still advance the reader past every element.
            if let Some(v) = read_int_elem(r, elem_ty)? {
                out.push(v);
            }
        }
        if out.is_empty() {
            Ok(MetaValue::Skipped)
        } else {
            Ok(MetaValue::U64Array(out))
        }
    } else if elem_ty == 7 && count <= MAX_KEPT_INT_ARRAY {
        // Short bool array (one byte per element) — e.g. the per-layer
        // `attention.sliding_window_pattern`.
        let mut out = Vec::with_capacity(count as usize);
        let mut b = [0u8; 1];
        for _ in 0..count {
            r.read_exact(&mut b)?;
            out.push(b[0] != 0);
        }
        Ok(MetaValue::BoolArray(out))
    } else {
        // Reuse the skip path for the array body (we've already consumed the
        // element type + count above).
        if let Some(sz) = scalar_size(elem_ty) {
            skip(r, count.saturating_mul(sz))?;
        } else {
            for _ in 0..count {
                skip_value(r, elem_ty, 1)?;
            }
        }
        Ok(MetaValue::Skipped)
    }
}

/// Skip one value of type `ty` without materializing it. Arrays of any
/// length — including 250k-entry vocabularies — cost a few seeks.
fn skip_value<R: Read + Seek>(r: &mut R, ty: u32, depth: u8) -> std::io::Result<()> {
    if depth > 4 {
        return Err(bad("metadata nesting too deep"));
    }
    if let Some(sz) = scalar_size(ty) {
        return skip(r, sz);
    }
    match ty {
        8 => {
            let len = read_u64(r)?;
            skip(r, len)
        }
        9 => {
            let elem_ty = read_u32(r)?;
            let count = read_u64(r)?;
            if let Some(sz) = scalar_size(elem_ty) {
                // Fixed-size elements: one seek covers the whole array.
                skip(r, count.saturating_mul(sz))
            } else {
                for _ in 0..count {
                    skip_value(r, elem_ty, depth + 1)?;
                }
                Ok(())
            }
        }
        other => Err(bad(format!("unknown value type {other}"))),
    }
}

/// Parse the GGUF header KV section into a scalar-only map.
fn read_meta_map<R: Read + Seek>(r: &mut R) -> std::io::Result<MetaMap> {
    read_meta_map_counted(r).map(|(m, _)| m)
}

/// [`read_meta_map`], also returning the header's tensor count. Leaves the
/// reader positioned at the start of the tensor-info table.
fn read_meta_map_counted<R: Read + Seek>(r: &mut R) -> std::io::Result<(MetaMap, u64)> {
    if read_u32(r)? != GGUF_MAGIC {
        return Err(bad("not a GGUF file (bad magic)"));
    }
    let version = read_u32(r)?;
    if !(2..=3).contains(&version) {
        return Err(bad(format!("unsupported GGUF version {version}")));
    }
    let tensor_count = read_u64(r)?;
    if tensor_count > MAX_TENSOR_COUNT {
        return Err(bad(format!("implausible tensor count {tensor_count}")));
    }
    let kv_count = read_u64(r)?;
    if kv_count > MAX_KV_COUNT {
        return Err(bad(format!("implausible metadata KV count {kv_count}")));
    }

    let mut map = HashMap::with_capacity(kv_count.min(256) as usize);
    for _ in 0..kv_count {
        let key_len = read_u64(r)?;
        if key_len > 4096 {
            return Err(bad(format!("implausible key length {key_len}")));
        }
        let mut key_bytes = vec![0u8; key_len as usize];
        r.read_exact(&mut key_bytes)?;
        let key = String::from_utf8_lossy(&key_bytes).into_owned();

        let ty = read_u32(r)?;
        let value = match ty {
            8 => {
                let len = read_u64(r)?;
                if len <= MAX_KEPT_STRING {
                    let mut buf = vec![0u8; len as usize];
                    r.read_exact(&mut buf)?;
                    MetaValue::Str(String::from_utf8_lossy(&buf).into_owned())
                } else {
                    skip(r, len)?;
                    MetaValue::Skipped
                }
            }
            9 => read_array(r)?,
            _ => read_scalar(r, ty)?.unwrap_or(MetaValue::Skipped),
        };
        map.insert(key, value);
    }
    Ok((MetaMap(map), tensor_count))
}

/// Upper bound on tensors per GGUF file (the largest MoE checkpoints carry a
/// few thousand); anything above is a corrupt header.
const MAX_TENSOR_COUNT: u64 = 1_000_000;

// ---------------------------------------------------------------------------
// Tensor table (planner accounting)
// ---------------------------------------------------------------------------

/// `(block size in elements, bytes per block)` for a ggml tensor type id,
/// mirroring `ggml_type_traits` in ggml.c. `None` for ids this table does not
/// know (newer quant formats) — callers fall back to the offset delta.
pub fn ggml_type_block(ty: u32) -> Option<(u64, u64)> {
    Some(match ty {
        0 => (1, 4),      // F32
        1 => (1, 2),      // F16
        2 => (32, 18),    // Q4_0
        3 => (32, 20),    // Q4_1
        6 => (32, 22),    // Q5_0
        7 => (32, 24),    // Q5_1
        8 => (32, 34),    // Q8_0
        9 => (32, 36),    // Q8_1
        10 => (256, 84),  // Q2_K
        11 => (256, 110), // Q3_K
        12 => (256, 144), // Q4_K
        13 => (256, 176), // Q5_K
        14 => (256, 210), // Q6_K
        15 => (256, 292), // Q8_K
        16 => (256, 66),  // IQ2_XXS
        17 => (256, 74),  // IQ2_XS
        18 => (256, 98),  // IQ3_XXS
        19 => (256, 50),  // IQ1_S
        20 => (32, 18),   // IQ4_NL
        21 => (256, 110), // IQ3_S
        22 => (256, 82),  // IQ2_S
        23 => (256, 136), // IQ4_XS
        24 => (1, 1),     // I8
        25 => (1, 2),     // I16
        26 => (1, 4),     // I32
        27 => (1, 8),     // I64
        28 => (1, 8),     // F64
        29 => (256, 56),  // IQ1_M
        30 => (1, 2),     // BF16
        34 => (256, 54),  // TQ1_0
        35 => (256, 66),  // TQ2_0
        39 => (32, 17),   // MXFP4
        _ => return None,
    })
}

/// One tensor-info record from a GGUF header (no tensor data is read).
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset of the tensor data relative to the file's data section.
    pub offset: u64,
    /// Bytes the tensor occupies: from the type table, else the gap to the
    /// next tensor's offset (exact for any type, padding included).
    pub bytes: u64,
}

/// A GGUF header's metadata plus its full tensor-info table.
pub struct GgufTensorHeader {
    meta: MetaMap,
    pub tensors: Vec<TensorInfo>,
}

impl GgufTensorHeader {
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.meta.get_u64(key)
    }
    pub fn get_u64_array(&self, key: &str) -> Option<&[u64]> {
        self.meta.get_u64_array(key)
    }
    pub fn get_bool_array(&self, key: &str) -> Option<&[bool]> {
        self.meta.get_bool_array(key)
    }
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.meta.get_string(key)
    }
    pub fn get_layer_flags(&self, key: &str, n: usize) -> Option<Vec<bool>> {
        self.meta.get_layer_flags(key, n)
    }
    pub fn has_key(&self, key: &str) -> bool {
        self.meta.0.contains_key(key)
    }

    /// Test helper: add a bool-array metadata key to a synthetic header.
    #[cfg(test)]
    pub fn with_bool_array(mut self, key: &str, values: &[bool]) -> Self {
        self.meta.0.insert(key.to_string(), MetaValue::BoolArray(values.to_vec()));
        self
    }

    /// A synthetic header for tests: string and integer metadata plus tensor
    /// records, no file behind it.
    #[cfg(test)]
    pub fn synthetic(strings: &[(&str, &str)], ints: &[(&str, u64)], tensors: Vec<TensorInfo>) -> Self {
        let mut map = HashMap::new();
        for (k, v) in strings {
            map.insert(k.to_string(), MetaValue::Str(v.to_string()));
        }
        for (k, v) in ints {
            map.insert(k.to_string(), MetaValue::U64(*v));
        }
        Self { meta: MetaMap(map), tensors }
    }
}

fn read_tensor_table<R: Read + Seek>(
    r: &mut R,
    count: u64,
    file_size: u64,
    alignment: u64,
) -> std::io::Result<Vec<TensorInfo>> {
    let mut out = Vec::with_capacity(count.min(16_384) as usize);
    for _ in 0..count {
        let name_len = read_u64(r)?;
        if name_len > 4096 {
            return Err(bad(format!("implausible tensor name length {name_len}")));
        }
        let mut name = vec![0u8; name_len as usize];
        r.read_exact(&mut name)?;
        let n_dims = read_u32(r)?;
        if n_dims > 8 {
            return Err(bad(format!("implausible tensor rank {n_dims}")));
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(r)?);
        }
        let ggml_type = read_u32(r)?;
        let offset = read_u64(r)?;
        out.push(TensorInfo {
            name: String::from_utf8_lossy(&name).into_owned(),
            dims,
            ggml_type,
            offset,
            bytes: 0,
        });
    }
    // Data section starts at the next `alignment` boundary after the table.
    let table_end = r.stream_position()?;
    let align = alignment.max(1);
    let data_start = table_end.div_ceil(align) * align;
    let data_len = file_size.saturating_sub(data_start);

    // Gap-to-next-offset sizes (by ascending offset) cover unknown types.
    let mut order: Vec<usize> = (0..out.len()).collect();
    order.sort_by_key(|&i| out[i].offset);
    let mut gap = vec![0u64; out.len()];
    for (pos, &i) in order.iter().enumerate() {
        let next = order.get(pos + 1).map(|&j| out[j].offset).unwrap_or(data_len);
        gap[i] = next.saturating_sub(out[i].offset);
    }
    for (i, t) in out.iter_mut().enumerate() {
        let elems: u64 = t.dims.iter().product::<u64>().max(1);
        t.bytes = match ggml_type_block(t.ggml_type) {
            Some((blck, size)) => elems.div_ceil(blck) * size,
            None => gap[i],
        };
    }
    Ok(out)
}

/// Read a GGUF file's metadata and tensor-info table (header only — a few MB
/// at most, never tensor data). Errors on unreadable / malformed files.
pub fn read_tensor_header(path: &Path) -> Result<GgufTensorHeader, AppError> {
    let file_size = std::fs::metadata(path)?.len();
    let mut reader = BufReader::new(File::open(path)?);
    let (meta, count) = read_meta_map_counted(&mut reader)
        .map_err(|e| AppError::GgufParseFailed(format!("{}: {e}", path.display())))?;
    let alignment = meta.get_u64("general.alignment").unwrap_or(32);
    let tensors = read_tensor_table(&mut reader, count, file_size, alignment)
        .map_err(|e| AppError::GgufParseFailed(format!("{}: {e}", path.display())))?;
    Ok(GgufTensorHeader { meta, tensors })
}

/// All files of a (possibly split) GGUF model, in shard order. A path named
/// `<stem>-00001-of-00004.gguf` expands to its siblings the way llama.cpp's
/// split loader finds them; any other path is returned alone. Missing
/// siblings are an error (the model cannot load without them).
pub fn split_files(primary: &Path) -> Result<Vec<std::path::PathBuf>, AppError> {
    let name = primary.file_name().and_then(|n| n.to_str()).unwrap_or_default();
    let Some(caps) = split_name_parts(name) else {
        return Ok(vec![primary.to_path_buf()]);
    };
    let (stem, count, width_no, width_count) = caps;
    let dir = primary.parent().unwrap_or_else(|| Path::new("."));
    let mut out = Vec::with_capacity(count as usize);
    for i in 1..=count {
        let p = dir.join(format!(
            "{stem}-{i:0wn$}-of-{count:0wc$}.gguf",
            wn = width_no,
            wc = width_count
        ));
        if !p.exists() {
            return Err(AppError::GgufParseFailed(format!(
                "split GGUF is missing shard {}",
                p.display()
            )));
        }
        out.push(p);
    }
    Ok(out)
}

/// `name` = `<stem>-NNNNN-of-MMMMM.gguf` → `(stem, MMMMM, width(N), width(M))`.
fn split_name_parts(name: &str) -> Option<(&str, u32, usize, usize)> {
    let base = name.strip_suffix(".gguf")?;
    let (head, count) = base.rsplit_once("-of-")?;
    let (stem, no) = head.rsplit_once('-')?;
    if no.is_empty() || count.is_empty() {
        return None;
    }
    if !no.bytes().all(|b| b.is_ascii_digit()) || !count.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let n: u32 = count.parse().ok()?;
    (n >= 1).then_some((stem, n, no.len(), count.len()))
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse only the GGUF header of `path`. Does not load tensor data.
///
/// On any parsing failure, returns a `GgufMeta` with only `file_size` populated
/// and logs a warning. The caller should not treat a parse failure as a download
/// failure.
pub fn parse_header(path: &Path) -> Result<GgufMeta, AppError> {
    let file_size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);

    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("gguf::parse_header: cannot open {:?}: {}", path, e);
            return Ok(GgufMeta { file_size, ..Default::default() });
        }
    };

    let mut reader = BufReader::new(file);
    let metadata = match read_meta_map(&mut reader) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!("gguf::parse_header: cannot parse {:?}: {}", path, e);
            return Ok(GgufMeta { file_size, ..Default::default() });
        }
    };

    // 1. Architecture
    let architecture = metadata.get_string("general.architecture").map(|s| s.to_string());

    // 2. Context length — keyed under {arch}.context_length
    let context_length = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.context_length"))
            .or_else(|| metadata.get_u64("llama.context_length"))
    });

    // 3. Block count
    let block_count = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.block_count"))
            .or_else(|| metadata.get_u64("llama.block_count"))
    });

    // 4. Embedding length
    let embedding_length = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.embedding_length"))
            .or_else(|| metadata.get_u64("llama.embedding_length"))
    });

    let pooling_type = architecture.as_deref()
        .and_then(|arch| metadata.get_u64(&format!("{arch}.pooling_type")));

    // 5. Attention head counts and RoPE dimension
    let attention_head_count = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.attention.head_count"))
            .or_else(|| metadata.get_u64("llama.attention.head_count"))
    });

    let attention_head_count_kv = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.attention.head_count_kv"))
            .or_else(|| metadata.get_u64("llama.attention.head_count_kv"))
    });

    let rope_dimension_count = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.rope.dimension_count"))
            .or_else(|| metadata.get_u64("llama.rope.dimension_count"))
    });

    // Canonical KV head widths. `key_length`/`value_length` are the
    // authoritative per-head K/V dimensions (gemma-family sets these and they
    // differ from rope.dimension_count). `head_dim()` prefers `key_length`.
    let attention_key_length = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.attention.key_length"))
            .or_else(|| metadata.get_u64("llama.attention.key_length"))
    });

    let attention_value_length = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64(&format!("{arch}.attention.value_length"))
            .or_else(|| metadata.get_u64("llama.attention.value_length"))
    });

    // Sliding-window-attention (SWA) geometry. These are standard, generic GGUF
    // keys (any arch that mixes local/global attention emits them — the whole
    // Gemma family today). When present they let the KV-cache estimate bill each
    // layer correctly: full-attention layers scale with the whole context, while
    // sliding-window layers cap at `sliding_window` tokens. Absent → the model
    // is treated as uniform full attention (the previous behaviour).
    let attention_head_count_kv_layers = architecture.as_deref().and_then(|arch| {
        metadata
            .get_u64_array(&format!("{arch}.attention.head_count_kv"))
            .or_else(|| metadata.get_u64_array("llama.attention.head_count_kv"))
            .map(|a| a.to_vec())
    });

    let attention_sliding_window = architecture
        .as_deref()
        .and_then(|arch| metadata.get_u64(&format!("{arch}.attention.sliding_window")));

    let sliding_window_pattern = architecture.as_deref().and_then(|arch| {
        metadata
            .get_bool_array(&format!("{arch}.attention.sliding_window_pattern"))
            .map(|a| a.to_vec())
    });

    let attention_key_length_swa = architecture
        .as_deref()
        .and_then(|arch| metadata.get_u64(&format!("{arch}.attention.key_length_swa")));

    let attention_value_length_swa = architecture
        .as_deref()
        .and_then(|arch| metadata.get_u64(&format!("{arch}.attention.value_length_swa")));

    // 6. Quantization: try general.file_type first, then filename regex
    let quantization = metadata
        .get_u64("general.file_type")
        .and_then(file_type_to_quant)
        .map(|s| s.to_string())
        .or_else(|| quant_from_filename(path));

    // 7. Parameter count
    let params_b = metadata
        .get_u64("general.parameter_count")
        .map(|n| n as f64 / 1e9);

    // 8. Vision detection: any key prefixed "clip." or "vision."
    let vision = metadata
        .keys()
        .any(|k| k.starts_with("clip.") || k.starts_with("vision."));

    Ok(GgufMeta {
        architecture,
        context_length,
        block_count,
        embedding_length,
        pooling_type,
        attention_head_count,
        attention_head_count_kv,
        rope_dimension_count,
        attention_key_length,
        attention_value_length,
        attention_head_count_kv_layers,
        attention_sliding_window,
        sliding_window_pattern,
        attention_key_length_swa,
        attention_value_length_swa,
        quantization,
        params_b,
        vision,
        file_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_from_filename_q4km() {
        let p = Path::new("llama-3.Q4_K_M.gguf");
        assert_eq!(quant_from_filename(p), Some("Q4_K_M".to_string()));
    }

    #[test]
    fn quant_from_filename_no_quant() {
        let p = Path::new("model.gguf");
        assert_eq!(quant_from_filename(p), None);
    }

    #[test]
    fn quant_from_filename_not_gguf() {
        let p = Path::new("model.bin");
        assert_eq!(quant_from_filename(p), None);
    }

    #[test]
    fn file_type_decode() {
        assert_eq!(file_type_to_quant(15), Some("Q4_K_M"));
        assert_eq!(file_type_to_quant(1), Some("F16"));
        assert_eq!(file_type_to_quant(999), None);
    }

    // -- synthetic-GGUF builder for header-reader tests ----------------------

    fn put_str(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn put_kv_u32(buf: &mut Vec<u8>, key: &str, v: u32) {
        put_str(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn put_kv_u64(buf: &mut Vec<u8>, key: &str, v: u64) {
        put_str(buf, key);
        buf.extend_from_slice(&10u32.to_le_bytes());
        buf.extend_from_slice(&v.to_le_bytes());
    }

    fn put_kv_string(buf: &mut Vec<u8>, key: &str, v: &str) {
        put_str(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        put_str(buf, v);
    }

    /// String array with `count` single-byte entries.
    fn put_kv_str_array(buf: &mut Vec<u8>, key: &str, count: u64) {
        put_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes()); // elem type: string
        buf.extend_from_slice(&count.to_le_bytes());
        for _ in 0..count {
            put_str(buf, "x");
        }
    }

    fn put_kv_f32_array(buf: &mut Vec<u8>, key: &str, count: u64) {
        put_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&6u32.to_le_bytes()); // elem type: f32
        buf.extend_from_slice(&count.to_le_bytes());
        buf.extend(std::iter::repeat(0u8).take((count * 4) as usize));
    }

    /// u32 array (GGUF elem type 4) — how gemma-family models encode the
    /// per-layer `attention.head_count_kv`.
    fn put_kv_u32_array(buf: &mut Vec<u8>, key: &str, vals: &[u32]) {
        put_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&4u32.to_le_bytes()); // elem type: u32
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    /// bool array (GGUF elem type 7) — how sliding-window models encode the
    /// per-layer `attention.sliding_window_pattern`.
    fn put_kv_bool_array(buf: &mut Vec<u8>, key: &str, vals: &[bool]) {
        put_str(buf, key);
        buf.extend_from_slice(&9u32.to_le_bytes());
        buf.extend_from_slice(&7u32.to_le_bytes()); // elem type: bool
        buf.extend_from_slice(&(vals.len() as u64).to_le_bytes());
        for v in vals {
            buf.push(*v as u8);
        }
    }

    fn synthetic_gguf(kvs: impl FnOnce(&mut Vec<u8>) -> u64) -> Vec<u8> {
        let mut body = Vec::new();
        let kv_count = kvs(&mut body);
        let mut buf = Vec::new();
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
        buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
        buf.extend_from_slice(&kv_count.to_le_bytes());
        buf.extend_from_slice(&body);
        buf
    }

    fn write_temp_gguf(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("oh-gguf-tests");
        std::fs::create_dir_all(&dir).unwrap();
        // Tests run in parallel in one process — the name must disambiguate.
        let path = dir.join(format!("{name}-{}.gguf", std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path
    }

    /// A Qwen-shaped header: vocab array way past the 65 536-element cap that
    /// broke the previous crate-backed parser, plus all the scalar keys the
    /// catalog cares about.
    #[test]
    fn parse_header_survives_huge_vocab_array() {
        let bytes = synthetic_gguf(|b| {
            put_kv_string(b, "general.architecture", "qwen3moe");
            put_kv_u32(b, "general.file_type", 12); // Q3_K_M
            put_kv_u64(b, "general.parameter_count", 35_000_000_000);
            put_kv_u64(b, "qwen3moe.context_length", 262_144);
            put_kv_u64(b, "qwen3moe.block_count", 48);
            put_kv_u64(b, "qwen3moe.embedding_length", 4096);
            put_kv_u32(b, "qwen3moe.attention.head_count", 32);
            put_kv_u32(b, "qwen3moe.attention.head_count_kv", 4);
            put_kv_u32(b, "qwen3moe.rope.dimension_count", 128);
            put_kv_str_array(b, "tokenizer.ggml.tokens", 248_320);
            put_kv_f32_array(b, "tokenizer.ggml.scores", 248_320);
            11
        });
        let path = write_temp_gguf("huge-vocab", &bytes);
        let meta = parse_header(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(meta.architecture.as_deref(), Some("qwen3moe"));
        assert_eq!(meta.context_length, Some(262_144));
        assert_eq!(meta.block_count, Some(48));
        assert_eq!(meta.attention_head_count, Some(32));
        assert_eq!(meta.attention_head_count_kv, Some(4));
        assert_eq!(meta.rope_dimension_count, Some(128));
        assert_eq!(meta.quantization.as_deref(), Some("Q3_K_M"));
        assert_eq!(meta.params_b, Some(35.0));
        assert!(!meta.vision);
        assert_eq!(meta.file_size, bytes.len() as u64);
    }

    /// gemma4-shaped header: `attention.head_count_kv` is a PER-LAYER ARRAY
    /// (not a scalar), and `attention.key_length` (512) is the canonical KV head
    /// dim — distinct from `rope.dimension_count` (also 512 here, but the parser
    /// must read key_length, not fall through to rope). The old scalar getter
    /// returned None for the array, forcing the call site to fall back to the
    /// query head count (16) and grossly over-count the KV cache.
    #[test]
    fn parse_header_gemma4_array_head_count_kv() {
        let bytes = synthetic_gguf(|b| {
            put_kv_string(b, "general.architecture", "gemma4");
            put_kv_u32(b, "general.file_type", 15); // Q4_K_M
            put_kv_u64(b, "gemma4.context_length", 262_144);
            put_kv_u32(b, "gemma4.block_count", 30);
            put_kv_u32(b, "gemma4.attention.head_count", 16);
            // Per-layer KV-head counts: mostly 4, two layers at 8 — the parser
            // must keep the array and surface its MAX (8) as the representative.
            let mut kv_heads = vec![4u32; 30];
            kv_heads[5] = 8;
            kv_heads[25] = 8;
            put_kv_u32_array(b, "gemma4.attention.head_count_kv", &kv_heads);
            put_kv_u32(b, "gemma4.attention.key_length", 512);
            put_kv_u32(b, "gemma4.attention.value_length", 512);
            // Distinct rope dim so head_dim() preferring key_length is observable
            // through the real parse path (the model's real rope dim is 512, but
            // a different value here pins the preference order).
            put_kv_u32(b, "gemma4.rope.dimension_count", 256);
            9
        });
        let path = write_temp_gguf("gemma4", &bytes);
        let meta = parse_header(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(meta.architecture.as_deref(), Some("gemma4"));
        assert_eq!(meta.block_count, Some(30));
        assert_eq!(meta.attention_head_count, Some(16));
        // Representative kv-head count = max of the per-layer array, not None.
        assert_eq!(
            meta.attention_head_count_kv,
            Some(8),
            "array head_count_kv must yield the max element, not None",
        );
        assert_eq!(meta.attention_key_length, Some(512));
        assert_eq!(meta.attention_value_length, Some(512));
        assert_eq!(meta.rope_dimension_count, Some(256));
        // head_dim() must come from key_length (512), NOT the rope dim (256).
        assert_eq!(meta.head_dim(), Some(512));
    }

    /// Sliding-window-attention metadata round-trips through the real parse
    /// path: the per-layer `sliding_window_pattern` (a BOOL array), the window
    /// size, and the SWA-specific key/value lengths. These are what let the KV
    /// estimate distinguish full-attention layers (scale with full context) from
    /// local layers (cap at the window), generically — no arch special-casing.
    #[test]
    fn parse_header_gemma4_swa_metadata() {
        let bytes = synthetic_gguf(|b| {
            put_kv_string(b, "general.architecture", "gemma4");
            put_kv_u32(b, "general.file_type", 15); // Q4_K_M
            put_kv_u64(b, "gemma4.context_length", 262_144);
            put_kv_u32(b, "gemma4.block_count", 6);
            put_kv_u32(b, "gemma4.attention.head_count", 16);
            // Real gemma4 shape (downsized): a 5-local : 1-global repeat. Global
            // (pattern false) layers carry fewer KV heads than local ones.
            put_kv_u32_array(
                b,
                "gemma4.attention.head_count_kv",
                &[8, 8, 8, 8, 8, 2],
            );
            put_kv_bool_array(
                b,
                "gemma4.attention.sliding_window_pattern",
                &[true, true, true, true, true, false],
            );
            put_kv_u32(b, "gemma4.attention.sliding_window", 1024);
            put_kv_u32(b, "gemma4.attention.key_length", 512);
            put_kv_u32(b, "gemma4.attention.value_length", 512);
            put_kv_u32(b, "gemma4.attention.key_length_swa", 256);
            put_kv_u32(b, "gemma4.attention.value_length_swa", 256);
            12
        });
        let path = write_temp_gguf("gemma4-swa", &bytes);
        let meta = parse_header(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(
            meta.attention_head_count_kv_layers.as_deref(),
            Some([8, 8, 8, 8, 8, 2].as_slice()),
            "per-layer kv-head array must be preserved verbatim",
        );
        // Scalar accessor still surfaces the max for the fallback path.
        assert_eq!(meta.attention_head_count_kv, Some(8));
        assert_eq!(
            meta.sliding_window_pattern.as_deref(),
            Some([true, true, true, true, true, false].as_slice()),
            "per-layer SWA bool pattern must be preserved verbatim",
        );
        assert_eq!(meta.attention_sliding_window, Some(1024));
        assert_eq!(meta.attention_key_length_swa, Some(256));
        assert_eq!(meta.attention_value_length_swa, Some(256));
    }

    /// `head_dim()` prefers `attention.key_length` over `rope.dimension_count`
    /// even when they differ (gemma key_length 256 vs rope 128, say).
    #[test]
    fn head_dim_prefers_key_length_over_rope() {
        let meta = GgufMeta {
            attention_key_length: Some(256),
            rope_dimension_count: Some(128),
            embedding_length: Some(4096),
            attention_head_count: Some(32),
            ..Default::default()
        };
        assert_eq!(meta.head_dim(), Some(256));

        // Without key_length, fall back to rope.
        let meta2 = GgufMeta { attention_key_length: None, ..meta.clone() };
        assert_eq!(meta2.head_dim(), Some(128));

        // Without either, fall back to embedding / head_count.
        let meta3 = GgufMeta {
            attention_key_length: None,
            rope_dimension_count: None,
            ..meta
        };
        assert_eq!(meta3.head_dim(), Some(128)); // 4096 / 32
    }

    /// Garbage input degrades to file-size-only meta, never an error.
    #[test]
    fn parse_header_garbage_degrades_gracefully() {
        let path = write_temp_gguf("garbage", b"this is not a gguf file at all");
        let meta = parse_header(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert!(meta.architecture.is_none());
        assert_eq!(meta.file_size, 30);
    }

    /// Oversized kept-string guard: a >64 KiB string value is skipped but
    /// parsing continues and later keys still land.
    #[test]
    fn parse_header_skips_oversized_strings() {
        let big = "y".repeat(70 * 1024);
        let bytes = synthetic_gguf(|b| {
            put_kv_string(b, "tokenizer.chat_template", &big);
            put_kv_string(b, "general.architecture", "llama");
            put_kv_u64(b, "llama.context_length", 8192);
            3
        });
        let path = write_temp_gguf("big-string", &bytes);
        let meta = parse_header(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(meta.architecture.as_deref(), Some("llama"));
        assert_eq!(meta.context_length, Some(8192));
    }
}
