//! JIT wrapper for [`ModernBert`]. Bakes the input `input_ids` /
//! `attention_mask` shapes into the plan. Output is the `(B, L, D)`
//! last-hidden-state.
//!
//! No rebindable batch variable: the token-embedding op requires concrete
//! index shapes (`Tensor::embedding` resolves every dim via `as_const()`),
//! so the batch dim is baked from the `input_ids` shape at `prepare()` time —
//! re-prepare to serve a different batch size. This matches the gigaAM RN-T
//! JIT (also embedding-based, also binds no vars).

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::model::ModernBert;

jit_wrapper! {
    ModernBertJit(ModernBert) {
        input_ids: Tensor,
        attention_mask: Tensor,

        build(input_ids, attention_mask) {
            model.forward(input_ids, Some(attention_mask))
        }
    }
}
