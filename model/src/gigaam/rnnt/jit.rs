//! `jit_wrapper!`-generated K-step block JIT for RN-T device-resident decode
//! (`super::block::forward_block`): all loop state device-local; the host
//! reads three tapes + one flag per block.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use crate::gigaam::model::GigaAm;

#[allow(clippy::too_many_arguments)]
mod block_jit {
    use super::*;
    jit_wrapper! {
        RnntBlockJit(GigaAm) {
            enc: Tensor,
            time: Tensor,
            prev: Tensor,
            symbols: Tensor,
            valid: Tensor,
            h_in: Tensor,
            c_in: Tensor,

            outputs { tape, emit, frame, active_any, time_out, prev_out, symbols_out, h_out, c_out },

            build(enc, time, prev, symbols, valid, h_in, c_in) {
                let out: crate::gigaam::error::Result<_> =
                    crate::gigaam::rnnt::block::forward_block(model, enc, time, prev, symbols, valid, h_in, c_in);
                out
            }
        }
    }
}
pub(crate) use block_jit::RnntBlockJit;

jit_wrapper! {
    RnntEncProjJit(GigaAm) {
        enc: Tensor,

        build(enc) {
            // [B, T, E] -> [B, T, J] joint encoder projection, once per wave.
            let (rnnt_head, _) = model.head.expect_rnnt("RnntEncProjJit")?;
            let out: crate::gigaam::error::Result<_> = rnnt_head.joint.project_encoder(enc);
            out
        }
    }
}
