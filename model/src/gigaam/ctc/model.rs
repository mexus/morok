//! Top-level GigaAM CTC model: shared encoder + CTC projection head.

use std::path::Path;

use morok_dtype::DType;
use morok_tensor::{BoundVariable, Tensor};
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict};

use super::head::CTCHead;
use crate::gigaam::encoder::{Encoder, build_encoder_from_sd};
use crate::gigaam::error::{HubSnafu, StateSnafu, TensorSnafu};
use crate::gigaam::{GigaAmConfig, Result, remap};

/// GigaAM model: audio preprocessor + Conformer encoder + CTC head.
pub struct GigaAm {
    pub config: GigaAmConfig,
    pub encoder: Encoder,
    pub head: CTCHead,
}

impl GigaAm {
    /// Load from a HuggingFace Hub repository.
    pub fn from_hub(model_id: &str) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main")
    }

    /// Load from a HuggingFace Hub repository at a specific branch/revision.
    pub fn from_hub_with_revision(model_id: &str, revision: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let config_path = repo.get("config.json").context(HubSnafu)?;
        let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors(&weights_path, config)
    }

    /// Load from a safetensors file with a config.json in the same directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors(&weights_path, config)
    }

    /// Load a GigaAM model from a safetensors file.
    pub fn from_safetensors(path: &Path, config: GigaAmConfig) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a pre-loaded state dict.
    ///
    /// Auto-detects PyTorch key format (keys starting with `encoder.` or `model.encoder.`) and remaps.
    pub fn from_state_dict(sd: &StateDict, config: GigaAmConfig) -> Result<Self> {
        let is_pytorch = sd.keys().any(|k| k.starts_with("encoder.") || k.starts_with("model.encoder."));
        let sd_owned = if is_pytorch { remap::remap_pytorch(sd.clone(), &config)? } else { sd.clone() };
        let sd = &sd_owned;

        let encoder = build_encoder_from_sd(sd, &config)?;

        let mut head = CTCHead::empty(&config);
        head.load_state_dict(sd, "head").context(StateSnafu)?;

        Ok(Self { config, encoder, head })
    }

    pub fn with_random_weights(config: GigaAmConfig) -> Self {
        let encoder = Encoder::with_random_weights(&config);
        let head = CTCHead::empty(&config);
        Self { config, encoder, head }
    }

    /// Run full inference: waveform -> CTC log-probabilities.
    ///
    /// Input: raw audio samples at 16kHz, mono, float32.
    /// Output: lazy tensor `[1, vocab_size, T/4]` of log-probabilities.
    pub fn forward(&self, waveform: &[f32], mel_tensor: &mut Tensor) -> Result<Tensor> {
        {
            let mut view = mel_tensor.array_view_mut::<f32>().context(TensorSnafu)?;
            self.encoder.mel.forward_into(waveform, &mut view);
        }
        let encoded = self.encode(mel_tensor)?;
        self.head.forward(&encoded)
    }

    /// Encoder-only: mel features -> encoded representation.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.encoder.subsampling_output_length(mel_frames)
    }

    /// Batched encoder path with dynamic batch and mel-frame length.
    pub fn encode_batch(
        &self,
        mel: &Tensor,
        lengths: &Tensor,
        batch: &BoundVariable,
        mel_len: &BoundVariable,
    ) -> Result<Tensor> {
        self.encoder.forward_batch(mel, lengths, batch, mel_len)
    }

    /// dtype the encoder + heads operate in (read from the loaded weights).
    pub fn input_dtype(&self) -> DType {
        self.encoder.input_dtype()
    }
}
