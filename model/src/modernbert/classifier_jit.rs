//! JIT wrapper for the ModernBERT **classifier**: backbone forward (with the
//! padding mask) + fused classification head (pool → dense → GELU → norm →
//! classifier linear), compiled as ONE JIT plan. `input_ids` `(B, L)` int64 +
//! `attention_mask` `(B, L)` bool → raw logits `(B, num_labels)` (f32).
//!
//! Mirrors [`super::embedder_jit::ModernBertEmbedderJit`] (which fuses
//! backbone + pool + L2-norm). The mask is numerically load-bearing for the
//! same reason — masked attention keeps pad tokens out of real-token
//! representations. Fusing the head keeps the `(B, L, D)` activations
//! on-device; only the small `(B, num_labels)` logits are read back.

extern crate self as svod_model;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_macros::jit_wrapper;

use super::classifier::ModernBertClassificationModel;
use super::error::TensorSnafu;

jit_wrapper! {
    ModernBertClassifierJit(ModernBertClassificationModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.backbone.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            let mask = attention_mask.cast(svod_dtype::DType::Bool).context(TensorSnafu)?;
            let mask = mask
                .try_shrink([Some((SInt::Const(0), b.as_sint())), None])
                .context(TensorSnafu)?;
            model.forward_batch(input_ids, Some(&mask), &b)
        }
    }
}
