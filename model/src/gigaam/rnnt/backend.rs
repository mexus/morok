//! Per-utterance RNN-T step backend implementing
//! [`morok_arch::rnnt::JointStep`]. Wraps the predictor and joint JITs +
//! committed/tentative LSTM state.
//!
//! For B=1 the search loop owns one of these and drives it through the
//! per-frame inner loop. JIT plans (the heavy ones — predictor + joint) are
//! prepared once at construction and reused; the only per-step overhead is
//! the buffer-pack / execute / read-out cycle.

use std::time::{Duration, Instant};

use morok_arch::rnnt::JointStep;
use snafu::ResultExt;

use crate::jit::{DeviceSnafu, InputSpec, JitError, JitRecurrent, LstmState, RecurrentJit};

use super::jit::{RnntJointStepJit, RnntPredictorStepJit};
use crate::gigaam::model::GigaAm;

impl RecurrentJit for RnntPredictorStepJit {
    fn pack_state(&mut self, s: &LstmState) -> crate::jit::Result<()> {
        {
            let buf = self.h_in_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous h_in").copy_from_slice(&s.h);
        }
        {
            let buf = self.c_in_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous c_in").copy_from_slice(&s.c);
        }
        Ok(())
    }

    fn execute_step(&mut self) -> crate::jit::Result<()> {
        self.execute()
    }

    fn output_buffer(&self) -> crate::jit::Result<&morok_device::Buffer> {
        self.output()
    }
}

pub struct RnntStepBackend {
    /// Predictor JIT + active (post-step) LSTM state, flat layout `[L * P]`
    /// row-major. Active state is overwritten by every step; the search loop
    /// reads it via [`commit`](JointStep::commit) on non-blank emission.
    predictor: JitRecurrent<RnntPredictorStepJit>,
    joint_jit: RnntJointStepJit,

    /// Last accepted LSTM state. Copied into the predictor's active state
    /// before every [`step`](JointStep::step) so the JIT sees the committed
    /// prefix; [`commit`](JointStep::commit) copies the post-step active state
    /// back here.
    committed: LstmState,
    /// Last predictor `g` output (`[P]`). Stashed here so we can drop the
    /// predictor's output borrow before mutably accessing the joint JIT's
    /// input buffer.
    g_tentative: Vec<f32>,

    blank_id: usize,
    enc_hidden: usize,
    total_vocab: usize,

    /// Per-step timing aggregates. Reset by [`reset_stats`]; printed by the
    /// example. Cheap (one `Instant::now()` per substage) — kept always-on so
    /// the example can profile without recompilation.
    pub stats: StepStats,
}

/// Aggregate timings for [`RnntStepBackend`]. Six sub-stages per `step` call
/// + commit/reset counters.
#[derive(Default, Clone, Debug)]
pub struct StepStats {
    pub n_steps: u64,
    pub n_commits: u64,
    pub n_resets: u64,
    pub t_pred_pack: Duration,
    pub t_pred_exec: Duration,
    pub t_pred_read: Duration,
    pub t_joint_pack: Duration,
    pub t_joint_exec: Duration,
    pub t_joint_read: Duration,
}

impl RnntStepBackend {
    /// Build the backend from a model. `GigaAm` is cheap to clone (weights
    /// are `Tensor` handles backed by shared `Arc<Buffer>`s) so the predictor
    /// and joint JITs each take their own clone. The model must carry an
    /// RN-T head; CTC models are rejected.
    pub fn from_model(model: GigaAm) -> crate::jit::Result<Self> {
        let (rnnt_head, _) =
            model.head.expect_rnnt("RnntStepBackend").map_err(|e| JitError::Build { source: Box::new(e) })?;
        let pred_hidden = rnnt_head.pred_hidden;
        let pred_rnn_layers = rnnt_head.pred_rnn_layers;
        let total_vocab = rnnt_head.num_classes;
        let blank_id = total_vocab - 1;
        let enc_hidden = model.config.d_model;
        let lp = pred_rnn_layers * pred_hidden;

        let mut predictor_jit = RnntPredictorStepJit::new(model.clone());
        predictor_jit.prepare(
            InputSpec::i64(&[1, 1]),
            InputSpec::f32(&[pred_rnn_layers, 1, pred_hidden]),
            InputSpec::f32(&[pred_rnn_layers, 1, pred_hidden]),
        )?;

        let mut joint_jit = RnntJointStepJit::new(model);
        joint_jit.prepare(InputSpec::f32(&[1, 1, enc_hidden]), InputSpec::f32(&[1, 1, pred_hidden]))?;

        Ok(Self {
            predictor: JitRecurrent::new(predictor_jit, LstmState::zeros(lp), pred_hidden)?,
            joint_jit,
            committed: LstmState::zeros(lp),
            g_tentative: vec![0.0f32; pred_hidden],
            blank_id,
            enc_hidden,
            total_vocab,
            stats: StepStats::default(),
        })
    }
}

impl JointStep for RnntStepBackend {
    type Error = JitError;

    fn step(
        &mut self,
        encoder_frame: &[f32],
        prev_token: Option<usize>,
        logits_out: &mut [f32],
    ) -> std::result::Result<(), Self::Error> {
        debug_assert_eq!(encoder_frame.len(), self.enc_hidden);
        debug_assert_eq!(logits_out.len(), self.total_vocab);

        let tok_value = prev_token.unwrap_or(self.blank_id) as i64;

        // ── Predictor phase ──────────────────────────────────────────────
        // Copy committed state → predictor's active state, run one JIT step,
        // copy the resulting `g` head into our own buffer (so the JIT output
        // borrow ends before we mutate the joint JIT).
        let t_state_copy = Instant::now();
        self.predictor.state_mut().h.copy_from_slice(&self.committed.h);
        self.predictor.state_mut().c.copy_from_slice(&self.committed.c);
        let state_copy = t_state_copy.elapsed();

        let g = self.predictor.step(|jit| {
            let buf = jit.prev_token_mut()?;
            let mut view = buf.as_array_mut::<i64>().context(DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous prev_token")[0] = tok_value;
            Ok(())
        })?;
        let t_g_copy = Instant::now();
        self.g_tentative.copy_from_slice(g);
        let g_copy = t_g_copy.elapsed();
        let pred_timing = self.predictor.last_timing.clone();

        // ── Joint phase ──────────────────────────────────────────────────
        let t0 = Instant::now();
        {
            let buf = self.joint_jit.enc_t_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous enc_t").copy_from_slice(encoder_frame);
        }
        {
            let buf = self.joint_jit.g_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
            view.as_slice_mut().expect("contiguous g").copy_from_slice(&self.g_tentative);
        }
        let t1 = Instant::now();
        self.joint_jit.execute()?;
        let t2 = Instant::now();
        {
            let out = self.joint_jit.output()?;
            let arr = out.as_array::<f32>().context(DeviceSnafu)?;
            let flat = arr.as_slice().expect("contiguous joint output");
            // `flat.len() == 1 * 1 * total_vocab` for the [1, 1, V+1] joint output.
            logits_out.copy_from_slice(&flat[..self.total_vocab]);
        }
        let t3 = Instant::now();

        self.stats.n_steps += 1;
        self.stats.t_pred_pack += pred_timing.pack + state_copy;
        self.stats.t_pred_exec += pred_timing.exec;
        self.stats.t_pred_read += pred_timing.read + g_copy;
        self.stats.t_joint_pack += t1 - t0;
        self.stats.t_joint_exec += t2 - t1;
        self.stats.t_joint_read += t3 - t2;
        Ok(())
    }

    fn commit(&mut self) {
        self.stats.n_commits += 1;
        let state = self.predictor.state_mut();
        self.committed.h.copy_from_slice(&state.h);
        self.committed.c.copy_from_slice(&state.c);
    }

    fn reset(&mut self) {
        self.stats.n_resets += 1;
        self.committed.reset();
        self.predictor.reset();
    }
}
