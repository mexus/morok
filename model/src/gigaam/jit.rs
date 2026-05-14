//! Shared encoder-only JIT for [`GigaAm`]. Output is cast to fp32 so
//! the head-side path (CTC log-probs computed by `CtcHeadJit`, RN-T frames
//! consumed by the predictor/joint step JITs) sees a uniform dtype regardless
//! of whether the encoder ran in fp16, bf16, or fp32.
//!
//! The `jit_wrapper!` macro expands to `morok_model::jit::*` paths, so this
//! file needs the `extern crate self as morok_model;` binding in scope.

extern crate self as morok_model;

use std::sync::Arc;

use morok_macros::jit_wrapper;
use snafu::ResultExt;

use super::model::GigaAm;
use crate::gigaam::error::TensorSnafu;

jit_wrapper! {
    GigaAmEncoderJit(Arc<GigaAm>) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t: (1, model.config.max_mel_frames),
        }

        build(mel, lengths, b, t) {
            let out = model.encoder.forward_batch(mel, lengths, &b, &t)?;
            out.cast(morok_dtype::DType::Float32).context(TensorSnafu)
        }
    }
}
