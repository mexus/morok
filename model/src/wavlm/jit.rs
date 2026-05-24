//! JIT wrapper for [`WavLm`]. Bakes a fixed sample count into the plan and
//! exposes `b` as the rebindable batch variable. Output is a single tensor of
//! shape `(B, T_frames, embed_dim, num_layers + 1)` — the stacked WavLM
//! intermediates that downstream segmentation heads consume via a
//! per-layer-weighted sum.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::model::WavLm;

jit_wrapper! {
    WavLmJit(WavLm) {
        waveform: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(waveform, b) {
            model.extract_features_stacked_batch(waveform, &b)
        }
    }
}
