//! JIT wrapper for [`WeSpeakerResNet34`]. Compiles the forward graph once at
//! `prepare()` time with the fbank time/freq dims and the weight time dim
//! baked, and replays it per call with the batch dimension bound to the live
//! value of `b` through
//! [`execute_with_vars`](WeSpeakerResNet34Jit::execute_with_vars).
//!
//! Shape contract:
//! - `prepare(InputSpec::f32(&[max_b, 1598, 80]), InputSpec::f32(&[max_b, 799]))`
//!   bakes the feat (T=1598, F=80) and weight (T_w=799) dims into the plan.
//! - `b` is rebindable on every call; values are clamped to
//!   `[1, max_batch_size]` by the macro-generated setters.
//! - Output is `[B, 256]` speaker embeddings.

extern crate self as morok_model;

use morok_macros::jit_wrapper;

use super::model::WeSpeakerResNet34;

jit_wrapper! {
    WeSpeakerResNet34Jit(WeSpeakerResNet34) {
        feats: Tensor,
        weights: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(feats, weights, b) {
            model.forward(feats, weights, &b)
        }
    }
}
