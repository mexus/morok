//! `jit_wrapper!`-generated fused per-step JIT for the RN-T head: the predictor
//! and joint compiled into ONE batched plan. The encoder JIT lives in the
//! shared [`crate::gigaam::jit`] now.
//!
//! One execute advances all B lanes at once, amortizing the per-step dispatch
//! cost across the independent VAD chunks. Fusing keeps the predictor output
//! `g` on-device (it feeds the joint inside the same plan); the plan returns
//! two typed outputs: `tokens` (int32 argmax per lane) and the new LSTM `state`
//! (f32, `[new_h | new_c]` per lane in ONE buffer — each readback carries a
//! fixed host cost, and the backend reads `state` only when some lane emits).
//! The build closure validates the head is the RN-T variant (typed `Err` via
//! `JitError::Build` otherwise).

extern crate self as svod_model;

use snafu::ResultExt;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;

use crate::gigaam::error::TensorSnafu;
use crate::gigaam::model::GigaAm;

jit_wrapper! {
    RnntBatchStepJit(GigaAm) {
        prev_tokens: Tensor,
        enc_t: Tensor,
        h_in: Tensor,
        c_in: Tensor,

        outputs { tokens, state },

        build(prev_tokens, enc_t, h_in, c_in) {
            let (rnnt_head, _) = model.head.expect_rnnt("RnntBatchStepJit")?;
            let (g, new_h, new_c) = rnnt_head.predictor.forward_parts(prev_tokens, h_in, c_in)?;
            let tokens = rnnt_head.joint.forward_argmax(enc_t, &g)?;
            // [B, 1, 2 * L * P]: one readback buffer per step, h then c per lane.
            let state = Tensor::cat(&[&new_h, &new_c], 2).context(TensorSnafu)?;
            // Typed tail pins the build closure's error type (the `?`s above
            // only constrain it via `From`, so a bare `Ok` would be ambiguous).
            let out: crate::gigaam::error::Result<_> = Ok((tokens, state));
            out
        }
    }
}
