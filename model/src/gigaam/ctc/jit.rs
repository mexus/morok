//! `jit_wrapper!`-generated JITs for the CTC head: full-pipeline (mel ->
//! log-probs) for both single-batch and dynamic-batch inference.
//!
//! The `jit_wrapper!` macro expands to `morok_model::jit::*` paths, so each
//! file that invokes it needs the `extern crate self as morok_model;` binding
//! in scope (the binding is module-local).

extern crate self as morok_model;

use morok_macros::jit_wrapper;

use super::model::GigaAm;

jit_wrapper! {
    GigaAmJit(GigaAm) {
        mel: Tensor,

        build(mel) {
            let encoded = model.encode(mel)?;
            model.head.forward(&encoded)
        }
    }
}

jit_wrapper! {
    GigaAmBatchedJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t: (1, model.config.max_mel_frames),
        }

        build(mel, lengths, b, t) {
            let encoded = model.encode_batch(mel, lengths, &b, &t)?;
            model.head.forward(&encoded)
        }
    }
}
