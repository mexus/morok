use morok_dtype::DType;
use morok_ir::SInt;
use morok_tensor::{BoundVariable, Tensor};
use ndarray::Array4;
use snafu::ResultExt;

use crate::state::{HasStateDict, StateDict, get_tensor, prefixed};
use crate::{load_state_field, state_field};

use super::error::{StateSnafu, TensorSnafu};
use super::{ConvNormType, GigaAmConfig, SubsamplingMode};

/// Precompute RoPE cos/sin cache tensors. Returns `(cos, sin)` each of shape
/// `[max_encoder_frames, 1, 1, d_k/2]` where `d_k = d_model / n_heads`. GigaAM
/// uses non-interleaved RoPE (first_half/second_half split), matching
/// `apply_rotary_emb(..., interleaved=false)`.
fn build_rope_cache(config: &GigaAmConfig) -> (Tensor, Tensor) {
    let d_k = config.d_model / config.n_heads;
    let half_d = d_k / 2;
    let max_len = config.max_encoder_frames;

    let inv_freq: Vec<f32> = (0..half_d).map(|i| 1.0 / 10000.0f32.powf(2.0 * i as f32 / d_k as f32)).collect();

    let mut cos_arr = Array4::<f32>::zeros((max_len, 1, 1, half_d));
    let mut sin_arr = Array4::<f32>::zeros((max_len, 1, 1, half_d));

    for pos in 0..max_len {
        for i in 0..half_d {
            let angle = pos as f32 * inv_freq[i];
            cos_arr[[pos, 0, 0, i]] = angle.cos();
            sin_arr[[pos, 0, 0, i]] = angle.sin();
        }
    }

    (Tensor::from_ndarray(&cos_arr), Tensor::from_ndarray(&sin_arr))
}

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::Float32).unwrap()
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(shape, DType::Float32).unwrap()
}

type Result<T> = super::Result<T>;

// ---------------------------------------------------------------------------
// LayerNormWeights
// ---------------------------------------------------------------------------

/// Affine layer normalization: `layernorm(x) * weight + bias`.
#[derive(Clone)]
pub struct LayerNormWeights {
    pub weight: Tensor,
    pub bias: Tensor,
    pub eps: f64,
}

impl LayerNormWeights {
    pub fn empty(size: usize) -> Self {
        Self { weight: ones(&[size]), bias: zeros(&[size]), eps: 1e-5 }
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

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        load_state_field!(self, sd, prefix, [weight, bias]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// Conformer FFN: LayerNorm -> Linear(d->4d) -> SiLU -> Linear(4d->d).
///
/// Does NOT apply residual or 0.5 scaling — caller handles that.
#[derive(Clone)]
pub struct FeedForward {
    pub norm: LayerNormWeights,
    pub linear1_weight: Tensor,
    pub linear1_bias: Tensor,
    pub linear2_weight: Tensor,
    pub linear2_bias: Tensor,
}

impl FeedForward {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, d_ff) = (config.d_model, config.d_ff);
        Self {
            norm: LayerNormWeights::empty(d),
            linear1_weight: zeros(&[d_ff, d]),
            linear1_bias: zeros(&[d_ff]),
            linear2_weight: zeros(&[d, d_ff]),
            linear2_bias: zeros(&[d]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = self.norm.apply(x)?;
        let y = y.linear().weight(&self.linear1_weight).bias(&self.linear1_bias).call().context(TensorSnafu)?;
        let y = y.silu().context(TensorSnafu)?;
        y.linear().weight(&self.linear2_weight).bias(&self.linear2_bias).call().context(TensorSnafu)
    }
}

impl HasStateDict for FeedForward {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        sd.insert(prefixed(prefix, "linear1.weight"), self.linear1_weight.clone());
        sd.insert(prefixed(prefix, "linear1.bias"), self.linear1_bias.clone());
        sd.insert(prefixed(prefix, "linear2.weight"), self.linear2_weight.clone());
        sd.insert(prefixed(prefix, "linear2.bias"), self.linear2_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        self.linear1_weight = get_tensor(sd, &prefixed(prefix, "linear1.weight"))?;
        self.linear1_bias = get_tensor(sd, &prefixed(prefix, "linear1.bias"))?;
        self.linear2_weight = get_tensor(sd, &prefixed(prefix, "linear2.weight"))?;
        self.linear2_bias = get_tensor(sd, &prefixed(prefix, "linear2.bias"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MultiHeadSelfAttention
// ---------------------------------------------------------------------------

/// Multi-head self-attention with rotary position embeddings.
#[derive(Clone)]
pub struct MultiHeadSelfAttention {
    pub norm: LayerNormWeights,
    pub q_proj: Tensor,
    pub q_bias: Tensor,
    pub k_proj: Tensor,
    pub k_bias: Tensor,
    pub v_proj: Tensor,
    pub v_bias: Tensor,
    pub out_proj: Tensor,
    pub out_bias: Tensor,
    pub n_heads: usize,
    pub d_model: usize,
}

impl MultiHeadSelfAttention {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        Self {
            norm: LayerNormWeights::empty(d),
            q_proj: zeros(&[d, d]),
            q_bias: zeros(&[d]),
            k_proj: zeros(&[d, d]),
            k_bias: zeros(&[d]),
            v_proj: zeros(&[d, d]),
            v_bias: zeros(&[d]),
            out_proj: zeros(&[d, d]),
            out_bias: zeros(&[d]),
            n_heads: config.n_heads,
            d_model: d,
        }
    }

    pub fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].clone();
        let t = shape[1].clone();
        let d_model = self.d_model;
        let d_k = d_model / self.n_heads;
        let h = self.n_heads;

        let y = self.norm.apply(x)?;

        // RoPE expects [T, B, H, d_k] (PyTorch ordering). Rotate once, then
        // materialise back as [B, T, d_model] so the Q/K projections share
        // a single rotated buffer.
        let y_heads = y
            .try_transpose(0, 1)
            .context(TensorSnafu)?
            .try_reshape([t.clone(), b.clone(), SInt::Const(h), SInt::Const(d_k)])
            .context(TensorSnafu)?;
        let rope_dtype = y_heads.uop().dtype();
        let cos = cos.cast(rope_dtype.clone()).context(TensorSnafu)?;
        let sin = sin.cast(rope_dtype).context(TensorSnafu)?;
        let qk_input = y_heads
            .apply_rotary_emb(&cos, &sin, false)
            .context(TensorSnafu)?
            .try_reshape([t.clone(), b.clone(), SInt::Const(d_model)])
            .context(TensorSnafu)?
            .try_transpose(0, 1)
            .context(TensorSnafu)?
            .contiguous();

        let q = qk_input.linear().weight(&self.q_proj).bias(&self.q_bias).call().context(TensorSnafu)?;
        let k = qk_input.linear().weight(&self.k_proj).bias(&self.k_bias).call().context(TensorSnafu)?;
        let v = y.linear().weight(&self.v_proj).bias(&self.v_bias).call().context(TensorSnafu)?;

        let q = split_heads(&q, b.clone(), t.clone(), h, d_k)?;
        let k = split_heads(&k, b.clone(), t.clone(), h, d_k)?;
        let v = split_heads(&v, b.clone(), t.clone(), h, d_k)?;

        let attn =
            q.scaled_dot_product_attention().key(&k).value(&v).maybe_attn_mask(mask).call().context(TensorSnafu)?;
        let out = merge_heads(&attn, b, t, d_model)?;
        out.linear().weight(&self.out_proj).bias(&self.out_bias).call().context(TensorSnafu)
    }
}

/// `[B, T, H*d_k] → [B, T, H, d_k] → [B, H, T, d_k]`.
fn split_heads(x: &Tensor, b: SInt, t: SInt, h: usize, d_k: usize) -> Result<Tensor> {
    x.try_reshape([b, t, SInt::Const(h), SInt::Const(d_k)])
        .context(TensorSnafu)?
        .try_transpose(1, 2)
        .context(TensorSnafu)
}

/// `[B, H, T, d_k] → [B, T, H, d_k] → [B, T, d_model]`.
fn merge_heads(x: &Tensor, b: SInt, t: SInt, d_model: usize) -> Result<Tensor> {
    x.try_transpose(1, 2).context(TensorSnafu)?.try_reshape([b, t, SInt::Const(d_model)]).context(TensorSnafu)
}

impl HasStateDict for MultiHeadSelfAttention {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        state_field!(sd, prefix, self, [q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, out_proj, out_bias]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        load_state_field!(self, sd, prefix, [q_proj, q_bias, k_proj, k_bias, v_proj, v_bias, out_proj, out_bias]);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConvModule
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum ConvNorm {
    LayerNorm(LayerNormWeights),
    BatchNorm { scale: Tensor, bias: Tensor, mean: Tensor, invstd: Tensor },
}

/// Conformer convolution module:
/// LayerNorm -> Conv1d(d,2d,k=1) -> GLU -> DepthwiseConv1d -> Norm -> SiLU -> Conv1d(d,d,k=1)
#[derive(Clone)]
pub struct ConvModule {
    pub norm: LayerNormWeights,
    pub pw1_weight: Tensor,
    pub pw1_bias: Tensor,
    pub dw_weight: Tensor,
    pub dw_bias: Tensor,
    pub conv_norm: ConvNorm,
    pub pw2_weight: Tensor,
    pub pw2_bias: Tensor,
    d_model: usize,
    conv_kernel: usize,
}

impl ConvModule {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let (d, k) = (config.d_model, config.conv_kernel);
        let conv_norm = match &config.conv_norm_type {
            ConvNormType::LayerNorm => ConvNorm::LayerNorm(LayerNormWeights::empty(d)),
            ConvNormType::BatchNorm => {
                ConvNorm::BatchNorm { scale: ones(&[d]), bias: zeros(&[d]), mean: zeros(&[d]), invstd: ones(&[d]) }
            }
        };
        Self {
            norm: LayerNormWeights::empty(d),
            pw1_weight: zeros(&[2 * d, d, 1]),
            pw1_bias: zeros(&[2 * d]),
            dw_weight: zeros(&[d, 1, k]),
            dw_bias: zeros(&[d]),
            conv_norm,
            pw2_weight: zeros(&[d, d, 1]),
            pw2_bias: zeros(&[d]),
            d_model: d,
            conv_kernel: k,
        }
    }

    pub fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let activation_dtype = x.uop().dtype();
        let y = self.norm.apply(x)?;

        let y = y.try_transpose(-1, -2).context(TensorSnafu)?;

        let y = y.conv2d().weight(&self.pw1_weight).bias(&self.pw1_bias).call().context(TensorSnafu)?;

        let mut y = y.glu(1).context(TensorSnafu)?;

        if let Some(mask) = pad_mask {
            let valid = mask.logical_not().context(TensorSnafu)?;
            let valid = valid.try_unsqueeze(1).context(TensorSnafu)?;
            let zeros = y.zero().context(TensorSnafu)?;
            y = y.where_(&valid, &zeros).context(TensorSnafu)?;
        }

        let pad = ((self.conv_kernel - 1) / 2) as isize;
        let y = y
            .conv2d()
            .weight(&self.dw_weight)
            .bias(&self.dw_bias)
            .groups(self.d_model)
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;

        let y = match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => {
                let y = y.try_transpose(-1, -2).context(TensorSnafu)?;
                let y = ln.apply(&y)?;
                y.try_transpose(-1, -2).context(TensorSnafu)?
            }
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                y.batchnorm().scale(scale).bias(bias).mean(mean).invstd(invstd).call().context(TensorSnafu)?
            }
        };
        // BN/LN params (scale, bias, mean, invstd) are stored fp32; broadcasting promotes
        // the norm output. Re-cast to the activation dtype so SiLU/pw2 stay in the right
        // precision, matching Python's BatchNorm1d dtype semantics. No-op when types match.
        let y = if y.uop().dtype() != activation_dtype { y.cast(activation_dtype).context(TensorSnafu)? } else { y };

        let y = y.silu().context(TensorSnafu)?;

        let y = y.conv2d().weight(&self.pw2_weight).bias(&self.pw2_bias).call().context(TensorSnafu)?;

        y.try_transpose(-1, -2).context(TensorSnafu)
    }
}

impl HasStateDict for ConvModule {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.norm.state_dict(&prefixed(prefix, "norm"));
        state_field!(sd, prefix, self, [pw1_weight, pw1_bias, dw_weight, dw_bias, pw2_weight, pw2_bias]);
        match &self.conv_norm {
            ConvNorm::LayerNorm(ln) => sd.extend(ln.state_dict(&prefixed(prefix, "conv_norm"))),
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                for (name, t) in [("bn_scale", scale), ("bn_bias", bias), ("bn_mean", mean), ("bn_invstd", invstd)] {
                    sd.insert(prefixed(prefix, name), t.clone());
                }
            }
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.norm.load_state_dict(sd, &prefixed(prefix, "norm"))?;
        load_state_field!(self, sd, prefix, [pw1_weight, pw1_bias, dw_weight, dw_bias, pw2_weight, pw2_bias]);
        match &mut self.conv_norm {
            ConvNorm::LayerNorm(ln) => ln.load_state_dict(sd, &prefixed(prefix, "conv_norm"))?,
            ConvNorm::BatchNorm { scale, bias, mean, invstd } => {
                *scale = get_tensor(sd, &prefixed(prefix, "bn_scale"))?;
                *bias = get_tensor(sd, &prefixed(prefix, "bn_bias"))?;
                *mean = get_tensor(sd, &prefixed(prefix, "bn_mean"))?;
                *invstd = get_tensor(sd, &prefixed(prefix, "bn_invstd"))?;
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// StridingSubsampling
// ---------------------------------------------------------------------------

/// Striding subsampling: 2x (Conv stride-2 + ReLU), optionally followed by Linear.
///
/// Supports two modes:
/// - **conv1d**: `Conv1d(n_mels→d, k, stride=2)` x2, no linear projection.
/// - **conv2d**: `Conv2d(1→d, 3x3, stride=2)` x2 + `Linear(d * n_mels/4, d)`.
///
/// Input: `[B, T, n_mels]` -> Output: `[B, T/4, d_model]`.
#[derive(Clone)]
pub struct StridingSubsampling {
    pub conv1_weight: Tensor,
    pub conv1_bias: Tensor,
    pub conv2_weight: Tensor,
    pub conv2_bias: Tensor,
    pub linear_weight: Option<Tensor>,
    pub linear_bias: Option<Tensor>,
    n_mels: usize,
    d_model: usize,
    mode: SubsamplingMode,
    kernel_size: usize,
}

impl StridingSubsampling {
    pub fn empty(config: &GigaAmConfig) -> Self {
        let d = config.d_model;
        let k = config.subs_kernel_size;
        match &config.subsampling_mode {
            SubsamplingMode::Conv1d => Self {
                conv1_weight: zeros(&[d, config.n_mels, k]),
                conv1_bias: zeros(&[d]),
                conv2_weight: zeros(&[d, d, k]),
                conv2_bias: zeros(&[d]),
                linear_weight: None,
                linear_bias: None,
                n_mels: config.n_mels,
                d_model: d,
                mode: SubsamplingMode::Conv1d,
                kernel_size: k,
            },
            SubsamplingMode::Conv2d => Self {
                conv1_weight: zeros(&[d, 1, 3, 3]),
                conv1_bias: zeros(&[d]),
                conv2_weight: zeros(&[d, d, 3, 3]),
                conv2_bias: zeros(&[d]),
                linear_weight: Some(zeros(&[d, d * (config.n_mels / 4)])),
                linear_bias: Some(zeros(&[d])),
                n_mels: config.n_mels,
                d_model: d,
                mode: SubsamplingMode::Conv2d,
                kernel_size: 3,
            },
        }
    }

    pub fn output_length(&self, input_length: usize) -> usize {
        let pad = (self.kernel_size - 1) / 2;
        let mut len = input_length;
        for _ in 0..2 {
            len = (len + 2 * pad - self.kernel_size) / 2 + 1;
        }
        len
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match &self.mode {
            SubsamplingMode::Conv1d => self.forward_conv1d(x),
            SubsamplingMode::Conv2d => self.forward_conv2d(x),
        }
    }

    fn forward_conv1d(&self, x: &Tensor) -> Result<Tensor> {
        let x = x.try_transpose(-1, -2).context(TensorSnafu)?;

        let pad = (self.kernel_size / 2) as isize;
        let x = x
            .conv2d()
            .weight(&self.conv1_weight)
            .bias(&self.conv1_bias)
            .stride(&[2])
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv2_weight)
            .bias(&self.conv2_bias)
            .stride(&[2])
            .padding(&[(pad, pad)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    fn forward_conv2d(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let b = shape[0].clone();

        let x = x.try_unsqueeze(1).context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv1_weight)
            .bias(&self.conv1_bias)
            .stride(&[2, 2])
            .padding(&[(1, 1), (1, 1)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x
            .conv2d()
            .weight(&self.conv2_weight)
            .bias(&self.conv2_bias)
            .stride(&[2, 2])
            .padding(&[(1, 1), (1, 1)])
            .call()
            .context(TensorSnafu)?;
        let x = x.relu().context(TensorSnafu)?;

        let x = x.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)?;
        let x = x.try_reshape([b, SInt::Infer, SInt::Const(self.d_model * self.n_mels / 4)]).context(TensorSnafu)?;

        let lw = self.linear_weight.as_ref().expect("conv2d mode requires linear_weight");
        let lb = self.linear_bias.as_ref().expect("conv2d mode requires linear_bias");
        x.linear().weight(lw).bias(lb).call().context(TensorSnafu)
    }
}

impl HasStateDict for StridingSubsampling {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [conv1_weight, conv1_bias, conv2_weight, conv2_bias]);
        if let (Some(lw), Some(lb)) = (&self.linear_weight, &self.linear_bias) {
            sd.insert(prefixed(prefix, "linear_weight"), lw.clone());
            sd.insert(prefixed(prefix, "linear_bias"), lb.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        load_state_field!(self, sd, prefix, [conv1_weight, conv1_bias, conv2_weight, conv2_bias]);
        if matches!(self.mode, SubsamplingMode::Conv2d) {
            self.linear_weight = Some(get_tensor(sd, &prefixed(prefix, "linear_weight"))?);
            self.linear_bias = Some(get_tensor(sd, &prefixed(prefix, "linear_bias"))?);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ConformerLayer
// ---------------------------------------------------------------------------

/// One Conformer layer (Macaron-style):
/// FFN1(x0.5) + MHSA + Conv + FFN2(x0.5) + LayerNorm
#[derive(Clone)]
pub struct ConformerLayer {
    pub ffn1: FeedForward,
    pub mhsa: MultiHeadSelfAttention,
    pub conv: ConvModule,
    pub ffn2: FeedForward,
    pub final_norm: LayerNormWeights,
}

impl ConformerLayer {
    pub fn empty(config: &GigaAmConfig) -> Self {
        Self {
            ffn1: FeedForward::empty(config),
            mhsa: MultiHeadSelfAttention::empty(config),
            conv: ConvModule::empty(config),
            ffn2: FeedForward::empty(config),
            final_norm: LayerNormWeights::empty(config.d_model),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        att_mask: Option<&Tensor>,
        pad_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let half = Tensor::from_const(0.5f64).cast(x.uop().dtype()).context(TensorSnafu)?;

        // FFN1 half-step
        let x = x.try_add(&self.ffn1.forward(x)?.try_mul(&half).context(TensorSnafu)?).context(TensorSnafu)?;

        // MHSA
        let x = x.try_add(&self.mhsa.forward(&x, cos, sin, att_mask)?).context(TensorSnafu)?;

        // Convolution
        let x = x.try_add(&self.conv.forward(&x, pad_mask)?).context(TensorSnafu)?;

        // FFN2 half-step
        let x = x.try_add(&self.ffn2.forward(&x)?.try_mul(&half).context(TensorSnafu)?).context(TensorSnafu)?;

        // Final layer norm
        self.final_norm.apply(&x)
    }
}

impl HasStateDict for ConformerLayer {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.ffn1.state_dict(&prefixed(prefix, "ffn1")));
        sd.extend(self.mhsa.state_dict(&prefixed(prefix, "mhsa")));
        sd.extend(self.conv.state_dict(&prefixed(prefix, "conv")));
        sd.extend(self.ffn2.state_dict(&prefixed(prefix, "ffn2")));
        sd.extend(self.final_norm.state_dict(&prefixed(prefix, "final_norm")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        self.ffn1.load_state_dict(sd, &prefixed(prefix, "ffn1"))?;
        self.mhsa.load_state_dict(sd, &prefixed(prefix, "mhsa"))?;
        self.conv.load_state_dict(sd, &prefixed(prefix, "conv"))?;
        self.ffn2.load_state_dict(sd, &prefixed(prefix, "ffn2"))?;
        self.final_norm.load_state_dict(sd, &prefixed(prefix, "final_norm"))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Encoder — audio preprocessor + Conformer backbone
// ---------------------------------------------------------------------------

/// Audio preprocessor + Conformer encoder. Shared by both heads of
/// [`crate::gigaam::GigaAm`] (`Head::Ctc` and `Head::Rnnt` layer different
/// heads on top of the same encoder). Encoder-only path: `forward` for
/// single-batch, `forward_batch` for batched JIT execution.
#[derive(Clone)]
pub struct Encoder {
    pub subsampling: StridingSubsampling,
    pub layers: Vec<ConformerLayer>,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
    pub d_model: usize,
    pub n_heads: usize,
    pub max_encoder_frames: usize,
}

impl Encoder {
    pub fn with_random_weights(config: &GigaAmConfig) -> Self {
        let (cos_cache, sin_cache) = build_rope_cache(config);
        let subsampling = StridingSubsampling::empty(config);
        let layers = (0..config.n_layers).map(|_| ConformerLayer::empty(config)).collect();
        Self {
            subsampling,
            layers,
            cos_cache,
            sin_cache,
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_encoder_frames: config.max_encoder_frames,
        }
    }

    /// dtype the encoder operates in. Read off the first subsampling
    /// conv weight (the model's compute dtype is determined by the
    /// weights it was loaded with). Falls back to f32 when the weight
    /// isn't itself a float type — should never happen in practice but
    /// avoids producing an integer dtype here.
    pub fn input_dtype(&self) -> DType {
        let dtype = self.subsampling.conv1_weight.uop().dtype();
        if dtype.is_float() { dtype } else { DType::Float32 }
    }

    /// Shrink the precomputed RoPE cache to `[t, 1, 1, d_k/2]` so both
    /// single-batch and batched encoder forwards consume the same shape.
    fn slice_rope(&self, t: SInt) -> Result<(Tensor, Tensor)> {
        let d_half = self.d_model / self.n_heads / 2;
        let shrink = [
            (SInt::Const(0), t),
            (SInt::Const(0), SInt::Const(1)),
            (SInt::Const(0), SInt::Const(1)),
            (SInt::Const(0), SInt::Const(d_half)),
        ];
        let cos = self.cos_cache.try_shrink(shrink.clone()).context(TensorSnafu)?;
        let sin = self.sin_cache.try_shrink(shrink).context(TensorSnafu)?;
        Ok((cos, sin))
    }

    /// Encoder pass on a single mel batch with no padding mask.
    /// Input: tensor `[B, n_mels, T]`. Output: lazy tensor `[B, d_model, T/4]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = x.cast(self.input_dtype()).context(TensorSnafu)?;
        let x = self.subsampling.forward(&x)?;

        let shape = x.shape().context(TensorSnafu)?;
        let seq_len = shape[1].clone();
        let (cos, sin) = self.slice_rope(seq_len)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, None, None)?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    /// Batched encoder path with dynamic batch and mel-frame length.
    /// Input: `mel` `[B, n_mels, T_mel]`, `lengths` `[B]` valid lengths in mel frames.
    /// Output: `[B, d_model, T_sub]`.
    pub fn forward_batch(
        &self,
        mel: &Tensor,
        lengths: &Tensor,
        batch: &BoundVariable,
        mel_len: &BoundVariable,
    ) -> Result<Tensor> {
        let b = batch.as_sint();
        let t_mel = mel_len.as_sint();

        let lengths = lengths.try_shrink([Some((SInt::Const(0), b.clone()))]).context(TensorSnafu)?;
        let lengths = lengths.cast(DType::Index).context(TensorSnafu)?;

        let two_t = Tensor::const_(2i64, DType::Index);
        let one_t = Tensor::const_(1i64, DType::Index);

        let mut lengths_sub = lengths;
        for _ in 0..2 {
            lengths_sub = lengths_sub.try_add(&one_t).context(TensorSnafu)?.try_div(&two_t).context(TensorSnafu)?;
        }

        let mel = mel
            .try_shrink([Some((SInt::Const(0), b.clone())), None, Some((SInt::Const(0), t_mel))])
            .context(TensorSnafu)?;
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = x.cast(self.input_dtype()).context(TensorSnafu)?;
        let x = self.subsampling.forward(&x)?;

        let shape = x.shape().context(TensorSnafu)?;
        let t_sub = shape[1].clone();

        let range = Tensor::arange(self.max_encoder_frames as i64, None, None).context(TensorSnafu)?;
        let range = range.cast(DType::Index).context(TensorSnafu)?;
        let range = range.try_shrink([(SInt::Const(0), t_sub.clone())]).context(TensorSnafu)?;
        let range = range.try_reshape([SInt::Const(1), t_sub.clone()]).context(TensorSnafu)?;
        let lens = lengths_sub;
        let lens = lens.try_reshape([b.clone(), SInt::Const(1)]).context(TensorSnafu)?;
        let pad_valid = range.try_lt(&lens).context(TensorSnafu)?;

        let pv1 = pad_valid.try_unsqueeze(1).context(TensorSnafu)?;
        let pv2 = pad_valid.try_unsqueeze(2).context(TensorSnafu)?;
        let att_mask = Some(
            pv1.bitwise_and(&pv2)
                .context(TensorSnafu)?
                .logical_not()
                .context(TensorSnafu)?
                .try_unsqueeze(1)
                .context(TensorSnafu)?,
        );
        let pad_mask = pad_valid.logical_not().context(TensorSnafu)?;

        let (cos, sin) = self.slice_rope(t_sub)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, att_mask.as_ref(), Some(&pad_mask))?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.subsampling.output_length(mel_frames)
    }

    /// Construct an `Encoder` from an already-remapped state dict + config.
    /// Called from the unified [`crate::gigaam::GigaAm::from_state_dict`] loader.
    pub(crate) fn from_state_dict(sd: &StateDict, config: &GigaAmConfig) -> Result<Self> {
        let (cos_cache, sin_cache) = build_rope_cache(config);

        let mut subsampling = StridingSubsampling::empty(config);
        subsampling.load_state_dict(sd, "subsampling").context(StateSnafu)?;

        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            let mut layer = ConformerLayer::empty(config);
            layer.load_state_dict(sd, &format!("layers.{i}")).context(StateSnafu)?;
            layers.push(layer);
        }

        Ok(Self {
            subsampling,
            layers,
            cos_cache,
            sin_cache,
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_encoder_frames: config.max_encoder_frames,
        })
    }
}
