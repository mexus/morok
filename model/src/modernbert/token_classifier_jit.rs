//! JIT wrapper for the ModernBERT **token classifier**: backbone forward (with
//! the padding mask) + fused token head (`prediction_head_tail` over the full
//! `(B, L, D)` state — no pooling), compiled as ONE JIT plan. `input_ids`
//! `(B, L)` int64 + `attention_mask` `(B, L)` bool → per-token logits
//! `(B, L, num_labels)` (f32).
//!
//! Mirrors [`super::classifier_jit::ModernBertClassifierJit`] (which fuses
//! backbone + pool + head). The mask is numerically load-bearing for backbone
//! attention; the head itself never pools. Fusing keeps the `(B, L, D)`
//! activations on-device; only the `(B, L, num_labels)` logits are read back.

extern crate self as svod_model;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_macros::jit_wrapper;

use super::error::TensorSnafu;
use super::token_classifier::ModernBertTokenClassificationModel;

jit_wrapper! {
    ModernBertTokenClassifierJit(ModernBertTokenClassificationModel) {
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
