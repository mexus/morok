//! `CtcHeadJit`: the small CTC projection (Conv1d k=1 + log-softmax) compiled
//! as its own JIT plan. Runs after the shared `GigaAmEncoderJit` on its
//! `[B, d_model, T_sub]` output and emits `[B, T_sub, vocab_size]` log-probs
//! consumed by `svod_arch::ctc` decoders.
//!
//! The `jit_wrapper!` macro expands to `svod_model::jit::*` paths, so this
//! file needs the `extern crate self as svod_model;` binding in scope.

extern crate self as svod_model;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_macros::jit_wrapper;

use crate::gigaam::error::TensorSnafu;
use crate::gigaam::model::GigaAm;

jit_wrapper! {
    CtcHeadJit(GigaAm) {
        encoded: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t_sub: (1, model.encoder.subsampling_output_length(model.config.max_mel_frames)),
        }

        build(encoded, b, t_sub) {
            let head = model.head.expect_ctc("CtcHeadJit")?;
            // encoded: [max_batch, d_model, max_t_sub] -> [b, d_model, t_sub].
            let encoded = encoded.try_shrink([
                Some((SInt::Const(0), b.as_sint())),
                None,
                Some((SInt::Const(0), t_sub.as_sint())),
            ]).context(TensorSnafu)?;
            head.forward(&encoded)
        }
    }
}
