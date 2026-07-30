//! ModernBERT configuration, parsed from HuggingFace `config.json`.
//!
//! Mirrors the published schema of `answerdotai/ModernBERT-{base,large}`. The
//! [`RawModernBertConfig`] serde mirror captures the on-disk shape; the clean
//! [`ModernBertConfig`] keeps only the fields the Rust backbone consumes and
//! adds a caller-chosen compute [`DType`] (defaults to bf16 on the AMD target,
//! f32 for CPU parity tests).
//!
//! Per-layer global vs local attention: every `global_attn_every_n_layers`-th
//! layer (0-indexed) attends to the full sequence; the rest use a
//! `local_attention`-wide sliding window split evenly. Global layers use
//! `global_rope_theta`; local layers use `local_rope_theta`.

use std::path::Path;

use serde::Deserialize;
use svod_dtype::DType;

use super::error::{Error, Result};

/// Clean, resolved ModernBERT backbone config.
#[derive(Clone, Debug)]
pub struct ModernBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: usize,
    pub layer_norm_eps: f64,
    /// Rotary base for global-attention layers.
    pub global_rope_theta: f64,
    /// Rotary base for local (sliding-window) layers.
    pub local_rope_theta: f64,
    /// Sliding-window width for local layers (ModernBERT splits it evenly:
    /// each query attends to `local_attention/2` keys on each side).
    pub local_attention: usize,
    /// Global attention every N layers (0-indexed: layers 0, N, 2N, … are
    /// global; the rest are local).
    pub global_attn_every_n_layers: usize,
    pub pad_token_id: usize,
    pub tie_word_embeddings: bool,
    /// Caller-chosen compute dtype (bf16 by default; f32 for CPU parity).
    pub dtype: DType,
    /// Upper bound on the symbolic batch variable in the JIT wrapper.
    pub max_batch_size: usize,
}

impl ModernBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// `(left, right)` window for a local layer. ModernBERT splits the
    /// `local_attention` width evenly; the published configs use 128 → (64, 64).
    pub fn local_window(&self) -> (usize, usize) {
        let half = self.local_attention / 2;
        (half, half)
    }

    /// `true` iff `layer_id` is a global-attention layer.
    pub fn is_global_layer(&self, layer_id: usize) -> bool {
        layer_id.is_multiple_of(self.global_attn_every_n_layers)
    }

    /// Rotary base for the given layer.
    pub fn rope_theta(&self, layer_id: usize) -> f64 {
        if self.is_global_layer(layer_id) { self.global_rope_theta } else { self.local_rope_theta }
    }
}

// ---------------------------------------------------------------------------
// Predefined configs — values from the published `config.json`.
// ---------------------------------------------------------------------------

/// `answerdotai/ModernBERT-base`: 22 layers, hidden 768, intermediate 1152,
/// 12 heads (head_dim 64), vocab 50368.
pub fn modernbert_base() -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 50368,
        hidden_size: 768,
        num_hidden_layers: 22,
        num_attention_heads: 12,
        intermediate_size: 1152,
        max_position_embeddings: 8192,
        layer_norm_eps: 1e-5,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        pad_token_id: 50283,
        tie_word_embeddings: true,
        dtype: DType::BFloat16,
        max_batch_size: 1,
    }
}

/// `answerdotai/ModernBERT-large`: 28 layers, hidden 1024, intermediate 2624,
/// 16 heads (head_dim 64), vocab 50368.
pub fn modernbert_large() -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 50368,
        hidden_size: 1024,
        num_hidden_layers: 28,
        num_attention_heads: 16,
        intermediate_size: 2624,
        max_position_embeddings: 8192,
        layer_norm_eps: 1e-5,
        global_rope_theta: 160_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 128,
        global_attn_every_n_layers: 3,
        pad_token_id: 50283,
        tie_word_embeddings: true,
        dtype: DType::BFloat16,
        max_batch_size: 1,
    }
}

// ---------------------------------------------------------------------------
// config.json parsing
// ---------------------------------------------------------------------------

impl ModernBertConfig {
    /// Parse a HuggingFace `config.json`. Unrecognized fields are ignored; any
    /// of the structural fields below that is absent falls back to the
    /// ModernBERT-base defaults.
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path)
            .map_err(|e| Error::Config { message: format!("reading config.json: {e}") })?;
        Self::from_json_str(&data)
    }

    pub fn from_json_str(data: &str) -> Result<Self> {
        let raw: RawModernBertConfig =
            serde_json::from_str(data).map_err(|e| Error::Config { message: format!("JSON parse error: {e}") })?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: RawModernBertConfig) -> Self {
        // Defaults from ModernBERT-base; the `large` checkpoint overrides every
        // structural field so the base fallback only matters for a truncated
        // or hand-written config.
        let base = modernbert_base();
        ModernBertConfig {
            vocab_size: raw.vocab_size.unwrap_or(base.vocab_size),
            hidden_size: raw.hidden_size.unwrap_or(base.hidden_size),
            num_hidden_layers: raw.num_hidden_layers.unwrap_or(base.num_hidden_layers),
            num_attention_heads: raw.num_attention_heads.unwrap_or(base.num_attention_heads),
            intermediate_size: raw.intermediate_size.unwrap_or(base.intermediate_size),
            max_position_embeddings: raw.max_position_embeddings.unwrap_or(base.max_position_embeddings),
            layer_norm_eps: raw.layer_norm_eps.or(raw.norm_eps).unwrap_or(base.layer_norm_eps),
            global_rope_theta: raw.global_rope_theta.unwrap_or(base.global_rope_theta),
            local_rope_theta: raw.local_rope_theta.unwrap_or(base.local_rope_theta),
            local_attention: raw.local_attention.unwrap_or(base.local_attention),
            global_attn_every_n_layers: raw.global_attn_every_n_layers.unwrap_or(base.global_attn_every_n_layers),
            pad_token_id: raw.pad_token_id.unwrap_or(base.pad_token_id),
            tie_word_embeddings: raw.tie_word_embeddings.unwrap_or(base.tie_word_embeddings),
            // Compute dtype is caller-chosen, not from config.json.
            dtype: base.dtype,
            max_batch_size: base.max_batch_size,
        }
    }
}

/// Serde mirror of the published `config.json`. Every field is optional so a
/// missing field falls back to the base defaults rather than failing.
#[derive(Deserialize)]
struct RawModernBertConfig {
    vocab_size: Option<usize>,
    hidden_size: Option<usize>,
    num_hidden_layers: Option<usize>,
    num_attention_heads: Option<usize>,
    intermediate_size: Option<usize>,
    max_position_embeddings: Option<usize>,
    /// ModernBERT publishes both `layer_norm_eps` and `norm_eps` (equal).
    layer_norm_eps: Option<f64>,
    norm_eps: Option<f64>,
    global_rope_theta: Option<f64>,
    local_rope_theta: Option<f64>,
    local_attention: Option<usize>,
    global_attn_every_n_layers: Option<usize>,
    pad_token_id: Option<usize>,
    tie_word_embeddings: Option<bool>,
}
