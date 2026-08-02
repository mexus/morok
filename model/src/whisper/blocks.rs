//! Shared building blocks: LinearWeights, LayerNormWeights, Conv1dWeights, sinusoids.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{fan_in_uniform, ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};
use crate::{load_state_field, state_field};

use super::error::{Result, TensorSnafu};

// ─── LinearWeights ──────────────────────────────────────────────────────────

/// Linear layer weights matching PyTorch `nn.Linear(in, out, bias=...)`.
/// Weight shape `[out, in]`. Bias may be absent (e.g. Whisper's key projection).
#[derive(Clone)]
pub struct LinearWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl LinearWeights {
    pub fn empty(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        let weight = fan_in_uniform(&[out_features, in_features], in_features, DType::Float32);
        let bias = has_bias.then(|| fan_in_uniform(&[out_features], in_features, DType::Float32));
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.linear().weight(&self.weight).maybe_bias(self.bias.as_ref()).call().context(TensorSnafu)
    }
}

impl HasStateDict for LinearWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        if let Some(b) = &self.bias {
            sd.insert(prefixed(prefix, "bias"), b.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        let bias_key = prefixed(prefix, "bias");
        self.bias = sd.get(&bias_key).cloned();
        Ok(())
    }
}

// ─── LayerNormWeights ───────────────────────────────────────────────────────

/// Affine layer normalization: `layernorm(x) * weight + bias`.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize) -> Self {
        Self { weight: ones(&[size], DType::Float32), bias: zeros(&[size], DType::Float32), eps: 1e-5 }
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let normed = x.layernorm(-1, self.eps).context(TensorSnafu)?;
        normed.try_mul(&self.weight).context(TensorSnafu)?.try_add(&self.bias).context(TensorSnafu)
    }
}

impl HasStateDict for LayerNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [weight, bias]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        load_state_field!(self, sd, prefix, [weight, bias]);
        Ok(())
    }
}

// ─── Conv1dWeights ──────────────────────────────────────────────────────────

/// 1D convolution with optional bias. Weight shape `[out_ch, in_ch, kernel]`.
#[derive(Clone)]
pub struct Conv1dWeights {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub stride: usize,
    pub padding: usize,
}

impl Conv1dWeights {
    pub fn empty(in_ch: usize, out_ch: usize, kernel: usize, stride: usize, padding: usize, has_bias: bool) -> Self {
        let fan_in = in_ch * kernel;
        Self {
            weight: fan_in_uniform(&[out_ch, in_ch, kernel], fan_in, DType::Float32),
            bias: has_bias.then(|| fan_in_uniform(&[out_ch], fan_in, DType::Float32)),
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.padding as isize;
        x.conv2d()
            .weight(&self.weight)
            .maybe_bias(self.bias.as_ref())
            .stride(&[self.stride])
            .padding(&[(p, p)])
            .call()
            .context(TensorSnafu)
    }
}

impl HasStateDict for Conv1dWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        if let Some(b) = &self.bias {
            sd.insert(prefixed(prefix, "bias"), b.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        let bias_key = prefixed(prefix, "bias");
        self.bias = sd.get(&bias_key).cloned();
        Ok(())
    }
}

// ─── Sinusoidal positional embedding ────────────────────────────────────────

/// Compute sinusoidal positional embeddings matching `whisper.model.sinusoids()`.
/// Returns a `[length, channels]` f32 tensor (constant, not learned).
pub fn sinusoids(length: usize, channels: usize, max_timescale: f64) -> Result<Tensor> {
    assert!(channels.is_multiple_of(2), "sinusoids require even channel count");
    let half = channels / 2;
    let log_inc = max_timescale.ln() / (half - 1) as f64;
    let inv_data: Vec<f32> = (0..half).map(|i| (-log_inc * i as f64).exp() as f32).collect();
    let inv = Tensor::from_slice(&inv_data);
    let scaled_time = Tensor::arange(0, Some(length as i64), None)
        .context(TensorSnafu)?
        .cast(DType::Float32)
        .context(TensorSnafu)?
        .try_unsqueeze(-1)
        .context(TensorSnafu)?
        .try_mul(&inv)
        .context(TensorSnafu)?;
    let sin = scaled_time.sin().context(TensorSnafu)?;
    let cos = scaled_time.cos().context(TensorSnafu)?;
    Tensor::cat(&[&sin, &cos], -1).context(TensorSnafu)
}
