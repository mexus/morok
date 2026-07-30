//! JIT wrapper for [`ModernBert`]. Bakes the input `input_ids` /
//! `attention_mask` shapes into the plan and exposes `b` as the rebindable
//! batch variable. Output is the `(B, L, D)` last-hidden-state.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::model::ModernBert;

jit_wrapper! {
    ModernBertJit(ModernBert) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.forward_batch(input_ids, Some(attention_mask), &b)
        }
    }
}
