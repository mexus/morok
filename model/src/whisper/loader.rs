//! Checkpoint loading: safetensors from HuggingFace Hub or local directory.
//!
//! Handles two checkpoint formats:
//! - **Original OpenAI** (`.pt` converted): `encoder.conv1.weight`, `decoder.blocks.0.attn.query.weight`
//! - **HF Transformers** (`model.safetensors`): `model.encoder.conv1.weight`, `model.decoder.layers.0.self_attn.q_proj.weight`

use std::path::Path;

use crate::state::{self, HasStateDict, StateDict};

use super::config::{ModelDimensions, WhisperSize};
use super::error::{Error, Result};
use super::model::Whisper;

impl Whisper {
    /// Load from a safetensors state dict.
    pub fn from_state_dict(sd: &StateDict, dims: ModelDimensions) -> Result<Self> {
        let sd = remap_hf_keys(sd);
        let mut model = Self::empty(dims);
        model.load_state_dict(&sd, "").map_err(|e| Error::State { source: e })?;
        Ok(model)
    }

    /// Load from a local safetensors file or directory.
    pub fn from_dir(dir: &Path, dims: ModelDimensions) -> Result<Self> {
        let sd = state::load_safetensors_dir(dir).map_err(|e| Error::State { source: e })?;
        Self::from_state_dict(&sd, dims)
    }

    /// Load from HuggingFace Hub (`openai/whisper-{name}` or custom repo).
    pub fn from_hub(model_id: &str, revision: &str, dims: ModelDimensions) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().map_err(|e| Error::Checkpoint { msg: e.to_string() })?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let path = repo.get("model.safetensors").map_err(|e| Error::Checkpoint { msg: e.to_string() })?;
        let sd = state::load_safetensors(&path).map_err(|e| Error::State { source: e })?;
        Self::from_state_dict(&sd, dims)
    }

    /// Convenience: load a known size from `openai/whisper-{name}`.
    pub fn from_size(size: WhisperSize) -> Result<Self> {
        let dims = ModelDimensions::for_size(size);
        let repo = format!("openai/whisper-{}", size.name());
        Self::from_hub(&repo, "main", dims)
    }
}

/// Remap HuggingFace Transformers keys to the original OpenAI naming
/// convention used by our model structs. If keys already match (e.g.
/// loading from an original-format checkpoint), they pass through unchanged.
fn remap_hf_keys(sd: &StateDict) -> StateDict {
    // Detect format: if any key starts with "encoder." → already OpenAI format
    if sd.keys().any(|k| k.starts_with("encoder.")) {
        return sd.clone();
    }

    sd.iter().map(|(k, v)| (remap_key(k), v.clone())).collect()
}

/// Map a single HF Transformers key to the OpenAI original key name.
fn remap_key(key: &str) -> String {
    let k = key.strip_prefix("model.").unwrap_or(key);

    // Positional/token embeddings: HF Embedding params have `.weight` suffix;
    // OpenAI buffers don't (except token_embedding which does).
    let k = match k {
        "encoder.embed_positions.weight" => return "encoder.positional_embedding".into(),
        "decoder.embed_positions.weight" => return "decoder.positional_embedding".into(),
        _ => k,
    };

    // Token embedding: keep .weight
    let k = k.replacen("decoder.embed_tokens", "decoder.token_embedding", 1);

    // ── Encoder ──────────────────────────────────────────────────────────
    let k = k.replacen("encoder.layer_norm", "encoder.ln_post", 1);
    let k = k.replacen("encoder.layers.", "encoder.blocks.", 1);

    // ── Decoder ──────────────────────────────────────────────────────────
    let k = k.replacen("decoder.layer_norm", "decoder.ln", 1);
    let k = k.replacen("decoder.layers.", "decoder.blocks.", 1);

    // ── Per-layer projection names (applies to both encoder and decoder) ─
    let k = k.replace("self_attn_layer_norm", "attn_ln");
    let k = k.replace("encoder_attn_layer_norm", "cross_attn_ln");
    let k = k.replace("encoder_attn", "cross_attn");
    let k = k.replace("self_attn", "attn");
    // Projection names are now uniform: {attn,cross_attn}.{q,k,v,out}_proj
    let k = k.replace("q_proj", "query");
    let k = k.replace("k_proj", "key");
    let k = k.replace("v_proj", "value");
    let k = k.replace("out_proj", "out");
    let k = k.replace("fc1", "mlp.0");
    let k = k.replace("fc2", "mlp.2");

    k.replace("final_layer_norm", "mlp_ln")
}
