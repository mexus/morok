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
        Self::empty_dtype(in_features, out_features, has_bias, DType::Float32)
    }

    pub fn empty_dtype(in_features: usize, out_features: usize, has_bias: bool, dtype: DType) -> Self {
        let weight = fan_in_uniform(&[out_features, in_features], in_features, dtype.clone());
        let bias = has_bias.then(|| fan_in_uniform(&[out_features], in_features, dtype));
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.bias {
            Some(bias) => linear_with_bias(x, &self.weight, bias),
            None => x.linear().weight(&self.weight).call().context(TensorSnafu),
        }
    }
}

pub(super) fn linear_with_bias(x: &Tensor, weight: &Tensor, bias: &Tensor) -> Result<Tensor> {
    let output_dtype = x.uop().dtype();
    let is_low_precision = |dtype: &DType| dtype == &DType::Float16 || dtype == &DType::BFloat16;
    let low_precision = is_low_precision(&output_dtype) && is_low_precision(&weight.uop().dtype());
    if !low_precision {
        return x.linear().weight(weight).bias(bias).call().context(TensorSnafu);
    }

    x.linear()
        .weight(weight)
        .dtype(DType::Float32)
        .call()
        .context(TensorSnafu)?
        .try_add(&bias.cast(DType::Float32).context(TensorSnafu)?)
        .context(TensorSnafu)?
        .cast(output_dtype)
        .context(TensorSnafu)
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
        Self::empty_dtype(size, DType::Float32)
    }

    pub fn empty_dtype(size: usize, dtype: DType) -> Self {
        Self { weight: ones(&[size], dtype.clone()), bias: zeros(&[size], dtype), eps: 1e-5 }
    }

    pub fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let output_dtype = x.uop().dtype();
        if output_dtype == DType::Float16 || output_dtype == DType::BFloat16 {
            let x = x.cast(DType::Float32).context(TensorSnafu)?;
            let weight = self.weight.cast(DType::Float32).context(TensorSnafu)?;
            let bias = self.bias.cast(DType::Float32).context(TensorSnafu)?;
            return x
                .layernorm(-1, self.eps)
                .context(TensorSnafu)?
                .try_mul(&weight)
                .context(TensorSnafu)?
                .try_add(&bias)
                .context(TensorSnafu)?
                .cast(output_dtype)
                .context(TensorSnafu);
        }

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
        Self::empty_dtype(in_ch, out_ch, kernel, stride, padding, has_bias, DType::Float32)
    }

    pub fn empty_dtype(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        has_bias: bool,
        dtype: DType,
    ) -> Self {
        let fan_in = in_ch * kernel;
        Self {
            weight: fan_in_uniform(&[out_ch, in_ch, kernel], fan_in, dtype.clone()),
            bias: has_bias.then(|| fan_in_uniform(&[out_ch], fan_in, dtype)),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn realized_f32(tensor: Tensor) -> Vec<f32> {
        let mut tensor = tensor.cast(DType::Float32).unwrap();
        tensor.realize().unwrap();
        tensor.as_vec::<f32>().unwrap()
    }

    #[test]
    fn fp16_layernorm_keeps_affine_in_fp32_until_final_cast() {
        let x = Tensor::from_slice([0.1013f32, -0.2037, 0.3071, 1.913])
            .try_reshape([1usize, 4])
            .unwrap()
            .cast(DType::Float16)
            .unwrap();
        let layer = LayerNormWeights {
            weight: Tensor::from_slice([17.25f32, -31.5, 47.75, -63.0]).cast(DType::Float16).unwrap(),
            bias: Tensor::from_slice([0.03125f32, -0.0625, 0.09375, -0.125]).cast(DType::Float16).unwrap(),
            eps: 1e-5,
        };

        let reference = x
            .cast(DType::Float32)
            .unwrap()
            .layernorm(-1, layer.eps)
            .unwrap()
            .try_mul(&layer.weight.cast(DType::Float32).unwrap())
            .unwrap()
            .try_add(&layer.bias.cast(DType::Float32).unwrap())
            .unwrap()
            .cast(DType::Float16)
            .unwrap();
        let legacy = x.layernorm(-1, layer.eps).unwrap().try_mul(&layer.weight).unwrap().try_add(&layer.bias).unwrap();
        let actual = realized_f32(layer.apply(&x).unwrap());
        let reference = realized_f32(reference);
        let legacy = realized_f32(legacy);

        assert_eq!(actual, reference, "LayerNorm must round only after the FP32 affine epilogue");
        assert_ne!(legacy, reference, "fixture must detect rounding before the affine epilogue");
    }

    #[test]
    fn fp16_linear_keeps_bias_epilogue_in_fp32_until_final_cast() {
        let x = Tensor::from_slice([0.3333f32, -0.1428, 0.0909, 0.0769])
            .try_reshape([1usize, 4])
            .unwrap()
            .cast(DType::Float16)
            .unwrap();
        let weight = Tensor::from_slice([3.0f32, -5.0, 7.0, -11.0, -2.0, 4.0, -6.0, 8.0])
            .try_reshape([2usize, 4])
            .unwrap()
            .cast(DType::Float16)
            .unwrap();
        let bias = Tensor::from_slice([-2.5f32, 2.0]).cast(DType::Float16).unwrap();

        let matmul_f32 = x.linear().weight(&weight).dtype(DType::Float32).call().unwrap();
        let reference = matmul_f32.try_add(&bias.cast(DType::Float32).unwrap()).unwrap().cast(DType::Float16).unwrap();
        let legacy = x.linear().weight(&weight).bias(&bias).call().unwrap();
        let actual = realized_f32(linear_with_bias(&x, &weight, &bias).unwrap());
        let reference = realized_f32(reference);
        let legacy = realized_f32(legacy);

        assert_eq!(actual, reference, "linear bias must be added to the FP32 accumulator");
        assert_ne!(legacy, reference, "fixture must detect rounding before bias addition");
    }
}
