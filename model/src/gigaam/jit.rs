//! Shared encoder-only JIT for [`GigaAm`]. Output is cast to fp32 so
//! the head-side path (CTC log-probs computed by `CtcHeadJit`, RN-T frames
//! consumed by the predictor/joint step JITs) sees a uniform dtype regardless
//! of whether the encoder ran in fp16, bf16, or fp32.
//!
//! The `jit_wrapper!` macro expands to `svod_model::jit::*` paths, so this
//! file needs the `extern crate self as svod_model;` binding in scope.

extern crate self as svod_model;

use snafu::ResultExt;
use svod_macros::jit_wrapper;

use super::model::GigaAm;
use crate::gigaam::error::TensorSnafu;

jit_wrapper! {
    GigaAmEncoderJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            ts: (1, model.config.max_sub_blocks),
        }

        build(mel, lengths, b, ts) {
            let out = model.encoder.forward_batch(mel, lengths, &b, &ts)?;
            // Permute [B, d_model, T_sub] → [B, T_sub, d_model] on-device: the
            // RN-T decoder consumes frame-major rows, and doing it here turns
            // the host-side strided transpose over the slow mapping into one
            // contiguous copyout.
            out.cast(svod_dtype::DType::Float32).context(TensorSnafu)?.try_permute(&[0, 2, 1]).context(TensorSnafu)
        }
    }
}
