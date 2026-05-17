//! Lazy weight initializers shared across `resnet`, `gigaam`, and `silero_vad`
//! `with_random_weights` constructors.
//!
//! These produce *lazy* tensor graphs (no `.realize()`); the THREEFRY seed
//! `BUFFER` reference inside the graph blocks the const-folding collapse that
//! plain `Tensor::zeros` weights would otherwise trigger. Matches Tinygrad's
//! `tinygrad/nn/__init__.py` convention of storing `Tensor.uniform(...)` in
//! the constructor without realizing.

use morok_dtype::DType;
use morok_tensor::Tensor;

/// PyTorch / Tinygrad fan-in uniform: samples from `[-1/√fan_in, +1/√fan_in)`.
/// Used for Conv/Linear weights, their biases, embeddings, and LSTM gates.
pub(crate) fn fan_in_uniform(shape: &[usize], fan_in: usize, dtype: DType) -> Tensor {
    let bound = (fan_in.max(1) as f64).powf(-0.5);
    Tensor::uniform_with_dtype(shape, -bound, bound, dtype).expect("non-empty shape with finite bounds")
}

pub(crate) fn ones(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::ones(shape, dtype).expect("non-empty shape")
}

pub(crate) fn zeros(shape: &[usize], dtype: DType) -> Tensor {
    Tensor::zeros(shape, dtype).expect("non-empty shape")
}
