//! `jit_wrapper!`-generated JITs for the RNN-T head.
//!
//! The encoder JIT is encoder-only (no head); the head's predictor + joint
//! run as their own per-step JITs since their input shape depends on
//! `prev_token` and the LSTM state, which evolve through the search loop.
//!
//! All three JITs take an `Arc<GigaAmRnnt>` so the example can build them
//! from a single underlying model — Tensor weights are Arc-backed and shared
//! across clones, so the duplication is structural only.

extern crate self as morok_model;

use morok_macros::jit_wrapper;
use snafu::ResultExt;

use super::model::GigaAmRnnt;
use crate::gigaam::error::TensorSnafu;

jit_wrapper! {
    GigaAmRnntEncoderJit(std::sync::Arc<GigaAmRnnt>) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t: (1, model.config.max_mel_frames),
        }

        build(mel, lengths, b, t) {
            let out = model.encoder.forward_batch(mel, lengths, &b, &t)?;
            // Encoder may run in fp16 (depending on weight dtype); promote
            // to fp32 at the JIT boundary so the joint step + the host-side
            // copyout are uniform.
            out.cast(morok_dtype::DType::Float32).context(TensorSnafu)
        }
    }
}

jit_wrapper! {
    RnntPredictorStepJit(std::sync::Arc<GigaAmRnnt>) {
        prev_token: Tensor,
        h_in: Tensor,
        c_in: Tensor,

        build(prev_token, h_in, c_in) {
            model.head.predictor.forward_concat(prev_token, h_in, c_in)
        }
    }
}

jit_wrapper! {
    RnntJointStepJit(std::sync::Arc<GigaAmRnnt>) {
        enc_t: Tensor,
        g: Tensor,

        build(enc_t, g) {
            model.head.joint.forward(enc_t, g)
        }
    }
}
