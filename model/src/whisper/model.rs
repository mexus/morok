//! Whisper model composite: encoder + decoder + dimensions.

use svod_tensor::{BoundVariable, Tensor};

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::config::ModelDimensions;
use super::decoder::TextDecoder;
use super::encoder::AudioEncoder;
use super::error::Result;

/// The Whisper model: encoder + decoder + alignment heads.
#[derive(Clone)]
pub struct Whisper {
    pub dims: ModelDimensions,
    pub encoder: AudioEncoder,
    pub decoder: TextDecoder,
}

impl Whisper {
    pub fn empty(dims: ModelDimensions) -> Self {
        Self { encoder: AudioEncoder::empty(&dims), decoder: TextDecoder::empty(&dims), dims }
    }

    /// Encode mel spectrogram → audio features `[B, n_audio_ctx, D]`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    /// Decode tokens given audio features → logits `[B, L, n_vocab]`.
    pub fn decode(&self, tokens: &Tensor, audio_features: &Tensor, offset: usize) -> Result<Tensor> {
        self.decoder.forward(tokens, audio_features, offset)
    }

    /// Decode with cross-attention weights for DTW alignment.
    /// Returns `(logits, cross_attn_qk_per_layer)`.
    pub fn decode_with_alignment(
        &self,
        tokens: &Tensor,
        audio_features: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Vec<Tensor>)> {
        self.decoder.forward_with_alignment(tokens, audio_features, offset)
    }

    /// Prefill: initial tokens → logits + packed K/V caches.
    #[allow(clippy::type_complexity)]
    pub fn decode_prefill(
        &self,
        tokens: &Tensor,
        audio_features: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor)> {
        self.decoder.forward_prefill(tokens, audio_features, offset)
    }

    /// Single-token step with KV cache → (logits, new_self_k, new_self_v).
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.decoder.forward_step(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_mask)
    }

    /// Symbolic-batch single-token step for continuous batching. `b` is a
    /// JIT variable; the compiled plan is rebound to the live lane count at
    /// execute time. See [`TextDecoder::forward_step_batched`].
    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_batched(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_mask: &Tensor,
        b: &BoundVariable,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        self.decoder
            .forward_step_batched(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_mask, b)
    }

    /// Full forward: encode + decode.
    pub fn forward(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let audio_features = self.encode(mel)?;
        self.decode(tokens, &audio_features, 0)
    }

    pub fn is_multilingual(&self) -> bool {
        self.dims.is_multilingual()
    }
}

impl HasStateDict for Whisper {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.encoder.state_dict(&prefixed(prefix, "encoder")));
        sd.extend(self.decoder.state_dict(&prefixed(prefix, "decoder")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.encoder.load_state_dict(sd, &prefixed(prefix, "encoder"))?;
        self.decoder.load_state_dict(sd, &prefixed(prefix, "decoder"))?;
        Ok(())
    }
}
