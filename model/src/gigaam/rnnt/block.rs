//! K-step device-resident RNN-T decode block (NeMo FULL_GRAPH analog).
//!
//! [`forward_block`] traces [`BLOCK_STEPS`] label-looping steps with all
//! state on-device and masked `where` everywhere (the predictor runs
//! unconditionally; non-emitting lanes keep their committed state). The host
//! reads one token tape per block. No runtime vars — everything concrete, so
//! the plan graph-captures into one submit.
//!
//! Per step (identical greedy semantics to `decode_batch_labels`):
//! tok = joint(enc[time], g) → emit = in_bounds & !blank →
//! prev/state where-commit → symbols run length, cap forces advance →
//! time += blank|cap; tapes record (tok, emit, time-before-advance).

use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::gigaam::Result;
use crate::gigaam::error::TensorSnafu;
use crate::gigaam::model::GigaAm;

/// Decode steps per block execute. Amortizes the per-block readback; bounds
/// the unrolled plan (~40 kernels/step).
pub(crate) const BLOCK_STEPS: usize = 16;

/// Tapes + carried state from one block trace; flat tuple keyed by
/// `RnntBlockJit`'s output order.
pub(crate) type BlockOutputs = (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor);

/// `enc_proj [B, T, J]` (pre-projected encoder, [`super::joint::RnntJoint::project_encoder`]), `time/prev [B,1] i64`, `symbols/valid [B,1] i32`,
/// `h/c [L, B, P]` → `(tape, emit, frame [B,K] i32, active_any [1,1] i32,
/// time, prev, symbols, h, c)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn forward_block(
    model: &GigaAm,
    enc_proj: &Tensor,
    time: &Tensor,
    prev: &Tensor,
    symbols: &Tensor,
    valid: &Tensor,
    h: &Tensor,
    c: &Tensor,
) -> Result<BlockOutputs> {
    let t = |r: std::result::Result<Tensor, svod_tensor::error::Error>| r.context(TensorSnafu);
    let (head, runtime) = model.head.expect_rnnt("RnntBlockJit")?;
    let max_symbols = runtime.max_symbols_per_step.max(1);
    let blank = head.predictor.blank_id as i64;
    let (l, p) = (head.pred_rnn_layers as isize, head.pred_hidden as isize);
    let shape = enc_proj.shape().context(TensorSnafu)?;
    let (b, j) = (shape[0].as_const().expect("B concrete") as isize, shape[2].as_const().expect("J concrete") as isize);
    let valid64 = t(valid.cast(DType::Int64))?;
    let last = t(valid64.try_sub(&Tensor::from_slice([1i64])))?;

    let (mut time, mut prev, mut symbols, mut h, mut c) =
        (time.clone(), prev.clone(), symbols.clone(), h.clone(), c.clone());
    let (mut tapes, mut emits, mut frames) = (Vec::new(), Vec::new(), Vec::new());

    for _ in 0..BLOCK_STEPS {
        let in_bounds = t(time.try_lt(&valid64))?; // [B,1] bool
        // Clamp the gather index for finished lanes (mask restores correctness).
        let safe_t = t(time.where_(&in_bounds, &last))?;
        let idx = t(t(safe_t.try_reshape([b, 1, 1]))?.try_expand([b, 1, j]))?;
        let enc_t = t(enc_proj.gather(1, &idx))?; // [B,1,J]

        let (g, new_h, new_c) = head.predictor.forward_parts(&prev, &h, &c)?;
        let tok = head.joint.argmax_preproj(&enc_t, &g)?; // [B,1] i32
        let tok64 = t(tok.cast(DType::Int64))?;

        let is_blank = t(tok64.try_eq(&Tensor::from_slice([blank])))?;
        let not_blank = t(is_blank.logical_not())?;
        let emit = t(t(in_bounds.cast(DType::Bool))?.try_bitand(&not_blank))?; // [B,1]

        prev = t(tok64.where_(&emit, &prev))?;
        // Commit state for emitting lanes: [B,1,L*P] → [L,B,P], masked.
        let emit_lbp = t(t(emit.try_reshape([1, b, 1]))?.try_expand([l, b, p]))?;
        let to_lbp = |s: Tensor| t(t(t(s.try_reshape([b, l, p]))?.try_permute(&[1, 0, 2]))?.try_reshape([l, b, p]));
        h = t(to_lbp(new_h)?.where_(&emit_lbp, &h))?;
        c = t(to_lbp(new_c)?.where_(&emit_lbp, &c))?;

        let symbols1 = t(t(symbols.try_add(&Tensor::from_slice([1i32])))?.where_(&emit, &symbols))?;
        let cap = t(symbols1.try_ge(&Tensor::from_slice([max_symbols as i32])))?;
        let blank_adv = t(t(in_bounds.cast(DType::Bool))?.try_bitand(&is_blank))?;
        let adv = t(blank_adv.try_bitor(&t(emit.try_bitand(&cap))?))?;
        time = t(time.try_add(&t(adv.cast(DType::Int64))?))?;
        let zeros = Tensor::zeros(&[b as usize, 1], DType::Int32).context(TensorSnafu)?;
        symbols = t(zeros.where_(&adv, &symbols1))?;

        tapes.push(tok);
        emits.push(t(emit.cast(DType::Int32))?);
        frames.push(t(safe_t.cast(DType::Int32))?);
    }

    let cat = |v: &[Tensor]| t(Tensor::cat(&v.iter().collect::<Vec<_>>(), 1)); // [B,K]
    let active = t(time.try_lt(&valid64))?;
    let active_any =
        t(t(t(active.cast(DType::Int32))?.sum_with().axes(0isize).keepdim(true).call())?.try_reshape([1, 1]))?;

    Ok((cat(&tapes)?, cat(&emits)?, cat(&frames)?, active_any, time, prev, symbols, h, c))
}
