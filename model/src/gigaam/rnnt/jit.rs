//! `jit_wrapper!`-generated per-step JITs for the RN-T head: predictor and
//! joint each compiled as their own plan. The encoder JIT lives in the
//! shared [`crate::gigaam::jit`] now.
//!
//! All step JITs take a [`GigaAm`] (cheap to clone — weights are shared
//! via the underlying `Tensor` handle Arcs) and validate that the head is
//! the RN-T variant in their build closure (returning a typed `Err` via
//! `JitError::Build` if it isn't).

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use crate::gigaam::model::GigaAm;

jit_wrapper! {
    RnntPredictorStepJit(GigaAm) {
        prev_token: Tensor,
        h_in: Tensor,
        c_in: Tensor,

        build(prev_token, h_in, c_in) {
            let (rnnt_head, _) = model.head.expect_rnnt("RnntPredictorStepJit")?;
            rnnt_head.predictor.forward_concat(prev_token, h_in, c_in)
        }
    }
}

jit_wrapper! {
    RnntJointStepJit(GigaAm) {
        enc_t: Tensor,
        g: Tensor,

        build(enc_t, g) {
            let (rnnt_head, _) = model.head.expect_rnnt("RnntJointStepJit")?;
            rnnt_head.joint.forward_argmax(enc_t, g)
        }
    }
}
