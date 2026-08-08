//! Fixed-capacity decoder-step graph tests.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

/// Tiny config so the CPU JIT graph compiles in seconds. `n_text_ctx` is kept
/// small (8) to shrink the self-attention buffers — the step JIT only needs
/// one position of cache populated for this test.
fn tiny_dims() -> ModelDimensions {
    // Start from WhisperSize::Tiny's structural dims, but shrink the text
    // context and vocab so the compile graph is minimal. The step JIT's cache
    // buffers scale with n_text_ctx.
    let mut dims = ModelDimensions::for_size(WhisperSize::Tiny);
    dims.n_text_ctx = 8;
    dims.n_vocab = 64;
    dims
}

#[test]
fn forward_step_fixed_batch_keeps_batch_concrete() {
    let dims = tiny_dims();
    let model = Whisper::empty(dims.clone());
    let (batch, n_audio_ctx) = (2usize, 8usize);
    let d_head = dims.n_text_state / dims.n_text_head;
    let layer_heads = dims.n_text_layer * dims.n_text_head;
    let token = Tensor::zeros(&[batch, 1], DType::Int32).unwrap();
    let pos_emb = Tensor::zeros(&[batch, 1, dims.n_text_state], DType::Float32).unwrap();
    let self_k = Tensor::zeros(&[batch, dims.n_text_ctx, layer_heads, d_head], DType::Float32).unwrap();
    let self_v = Tensor::zeros(&[batch, dims.n_text_ctx, layer_heads, d_head], DType::Float32).unwrap();
    let cross_k = Tensor::zeros(&[batch, n_audio_ctx, layer_heads, d_head], DType::Float32).unwrap();
    let cross_v = Tensor::zeros(&[batch, n_audio_ctx, layer_heads, d_head], DType::Float32).unwrap();
    let mask = Tensor::zeros(&[batch, 1, 1, dims.n_text_ctx + 1], DType::Float32).unwrap();

    let (logits, new_k, new_v) =
        model.decode_step(&token, &pos_emb, &self_k, &self_v, &cross_k, &cross_v, &mask).unwrap();
    assert_eq!(logits.shape().unwrap()[0].as_const(), Some(batch));
    assert_eq!(new_k.shape().unwrap()[0].as_const(), Some(batch));
    assert_eq!(new_v.shape().unwrap()[0].as_const(), Some(batch));
}
