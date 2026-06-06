//! `jit_wrapper!`-generated per-step JITs for the RN-T head, split for
//! label-looping decode: the predictor (LSTM stack) runs once per emitted
//! label; the joint+argmax runs every step. Both batched over decode lanes.
//!
//! Device residency: the predictor's `g` and tentative `state` outputs stay
//! device-local (the backend copies `g` device→device into the joint's input
//! and commits state rows over SDMA); the joint reads back one int per lane.

extern crate self as svod_model;

use snafu::ResultExt;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;

use crate::gigaam::error::TensorSnafu;
use crate::gigaam::model::GigaAm;

jit_wrapper! {
    RnntPredictorJit(GigaAm) {
        prev_tokens: Tensor,
        h_in: Tensor,
        c_in: Tensor,

        outputs { g, state },

        build(prev_tokens, h_in, c_in) {
            let (rnnt_head, _) = model.head.expect_rnnt("RnntPredictorJit")?;
            let (g, new_h, new_c) = rnnt_head.predictor.forward_parts(prev_tokens, h_in, c_in)?;
            // `g` stays on the device — the joint JIT copies it into its own
            // input buffer; the host never reads it.
            let state = Tensor::cat(&[&new_h, &new_c], 2).context(TensorSnafu)?;
            let out: crate::gigaam::error::Result<_> = Ok((g, state));
            out
        }
    }
}

jit_wrapper! {
    RnntJointJit(GigaAm) {
        enc_t: Tensor,
        g: Tensor,

        build(enc_t, g) {
            let (rnnt_head, _) = model.head.expect_rnnt("RnntJointJit")?;
            let out: crate::gigaam::error::Result<_> = rnnt_head.joint.forward_argmax(enc_t, g);
            out
        }
    }
}
