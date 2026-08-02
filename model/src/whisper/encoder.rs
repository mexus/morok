//! Audio encoder: Conv1d frontend + sinusoidal positional embeddings + transformer blocks.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::attention::MultiHeadAttention;
use super::blocks::{Conv1dWeights, LayerNormWeights, sinusoids};
use super::config::ModelDimensions;
use super::error::{Result, TensorSnafu};

/// Encoder transformer block: self-attention + MLP, pre-norm.
#[derive(Clone)]
pub struct EncoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNormWeights,
    pub mlp0_w: Tensor,
    pub mlp0_b: Tensor,
    pub mlp1_w: Tensor,
    pub mlp1_b: Tensor,
    pub mlp_ln: LayerNormWeights,
    pub n_state: usize,
}

impl EncoderBlock {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        let mlp = n_state * 4;
        Self {
            attn: MultiHeadAttention::empty(n_state, n_head),
            attn_ln: LayerNormWeights::empty(n_state),
            mlp0_w: fan_in_uniform(&[mlp, n_state], n_state, DType::Float32),
            mlp0_b: fan_in_uniform(&[mlp], n_state, DType::Float32),
            mlp1_w: fan_in_uniform(&[n_state, mlp], mlp, DType::Float32),
            mlp1_b: fan_in_uniform(&[n_state], mlp, DType::Float32),
            mlp_ln: LayerNormWeights::empty(n_state),
            n_state,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Self-attention (pre-norm)
        let h = self.attn_ln.apply(x)?;
        let attn_out = self.attn.forward(&h, None, None)?;
        let x = x.try_add(&attn_out).context(TensorSnafu)?;

        // MLP (pre-norm)
        let h = self.mlp_ln.apply(&x)?;
        let h = h.linear().weight(&self.mlp0_w).bias(&self.mlp0_b).call().context(TensorSnafu)?;
        let h = h.gelu_exact().context(TensorSnafu)?;
        let h = h.linear().weight(&self.mlp1_w).bias(&self.mlp1_b).call().context(TensorSnafu)?;
        let x = x.try_add(&h).context(TensorSnafu)?;
        Ok(x)
    }
}

impl HasStateDict for EncoderBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.attn.state_dict(&prefixed(prefix, "attn")));
        sd.extend(self.attn_ln.state_dict(&prefixed(prefix, "attn_ln")));
        sd.insert(prefixed(prefix, "mlp.0.weight"), self.mlp0_w.clone());
        sd.insert(prefixed(prefix, "mlp.0.bias"), self.mlp0_b.clone());
        sd.insert(prefixed(prefix, "mlp.2.weight"), self.mlp1_w.clone());
        sd.insert(prefixed(prefix, "mlp.2.bias"), self.mlp1_b.clone());
        sd.extend(self.mlp_ln.state_dict(&prefixed(prefix, "mlp_ln")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.attn.load_state_dict(sd, &prefixed(prefix, "attn"))?;
        self.attn_ln.load_state_dict(sd, &prefixed(prefix, "attn_ln"))?;
        self.mlp0_w = get_tensor(sd, &prefixed(prefix, "mlp.0.weight"))?;
        self.mlp0_b = get_tensor(sd, &prefixed(prefix, "mlp.0.bias"))?;
        self.mlp1_w = get_tensor(sd, &prefixed(prefix, "mlp.2.weight"))?;
        self.mlp1_b = get_tensor(sd, &prefixed(prefix, "mlp.2.bias"))?;
        self.mlp_ln.load_state_dict(sd, &prefixed(prefix, "mlp_ln"))?;
        Ok(())
    }
}

/// Whisper audio encoder: Conv1d × 2 + sinusoidal pos-emb + N × EncoderBlock + LayerNorm.
#[derive(Clone)]
pub struct AudioEncoder {
    pub conv1: Conv1dWeights,
    pub conv2: Conv1dWeights,
    pub positional_embedding: Tensor,
    pub blocks: Vec<EncoderBlock>,
    pub ln_post: LayerNormWeights,
    pub n_state: usize,
    pub n_head: usize,
}

impl AudioEncoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_audio_state;
        Self {
            conv1: Conv1dWeights::empty(dims.n_mels, n_state, 3, 1, 1, true),
            conv2: Conv1dWeights::empty(n_state, n_state, 3, 2, 1, true),
            positional_embedding: sinusoids(dims.n_audio_ctx, n_state, 10_000.0).expect("sinusoidal embedding"),
            blocks: (0..dims.n_audio_layer).map(|_| EncoderBlock::empty(n_state, dims.n_audio_head)).collect(),
            ln_post: LayerNormWeights::empty(n_state),
            n_state,
            n_head: dims.n_audio_head,
        }
    }

    /// Forward: mel `[B, n_mels, T]` → encoder features `[B, T/2, D]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = self.conv1.forward(mel)?;
        let x = x.gelu_exact().context(TensorSnafu)?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu_exact().context(TensorSnafu)?;

        // [B, D, T/2] → [B, T/2, D]
        let x = x.try_permute(&[0, 2, 1]).context(TensorSnafu)?;

        // Add positional embedding [n_audio_ctx, D]
        let x = x.try_add(&self.positional_embedding).context(TensorSnafu)?;

        // Transformer blocks
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Final LayerNorm
        self.ln_post.apply(&x)
    }
}

impl HasStateDict for AudioEncoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.conv1.state_dict(&prefixed(prefix, "conv1")));
        sd.extend(self.conv2.state_dict(&prefixed(prefix, "conv2")));
        sd.insert(prefixed(prefix, "positional_embedding"), self.positional_embedding.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            sd.extend(block.state_dict(&prefixed(prefix, &format!("blocks.{i}"))));
        }
        sd.extend(self.ln_post.state_dict(&prefixed(prefix, "ln_post")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.conv1.load_state_dict(sd, &prefixed(prefix, "conv1"))?;
        self.conv2.load_state_dict(sd, &prefixed(prefix, "conv2"))?;
        self.positional_embedding = get_tensor(sd, &prefixed(prefix, "positional_embedding"))?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_state_dict(sd, &prefixed(prefix, &format!("blocks.{i}")))?;
        }
        self.ln_post.load_state_dict(sd, &prefixed(prefix, "ln_post"))?;
        Ok(())
    }
}
