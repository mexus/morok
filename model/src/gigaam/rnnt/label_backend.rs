//! Label-looping RNN-T step backend ([`svod_arch::rnnt::BatchLabelStep`]).
//!
//! Splits the fused per-step plan into two: the predictor (LSTM stack) runs
//! once per emitted-label round, while the joint+argmax runs every step —
//! blank-advance steps cost only the joint, the cheap part. NeMo
//! label-looping ported to lockstep lane waves.
//!
//! Device residency: `h_in`/`c_in` (device-local) carry the committed state;
//! the tentative `state` and `g` outputs stay device-local; the joint copies
//! `g` device→device into its own input. Host traffic per joint step is the
//! active lanes' encoder rows in and one int per lane out.

use std::time::{Duration, Instant};

use snafu::ResultExt;
use svod_arch::rnnt::BatchLabelStep;

use crate::jit::{DeviceSnafu, InputSpec, JitError};

use super::jit::{RnntJointJit, RnntPredictorJit};
use crate::gigaam::model::GigaAm;

/// Position of the `state` output in `RnntPredictorJit`'s `outputs { g, state }`.
const STATE_OUT: usize = 1;
/// Position of the `g` output.
const G_OUT: usize = 0;

pub struct RnntLabelBackend {
    pred: RnntPredictorJit,
    joint: RnntJointJit,

    lanes: usize,
    layers: usize,
    pred_hidden: usize,
    enc_hidden: usize,

    /// Per-lane frame-major encoder output `[valid_frames[i] * enc_hidden]`,
    /// bound by [`bind_batch`](Self::bind_batch).
    frames: Vec<Vec<f32>>,
    /// Staging zeros for [`reset`](BatchLabelStep::reset) (`copyin` is SDMA).
    zeros: Vec<u8>,

    pub stats: StepStats,
    profile_next_step: bool,
    step_profiles: Option<Vec<svod_runtime::KernelProfile>>,
}

/// Aggregate timings: `n_steps` counts joint executes, `n_commits` predictor
/// rounds; `t_exec`/`t_read` are joint submit / sync+token-read.
#[derive(Default, Clone, Debug)]
pub struct StepStats {
    pub n_steps: u64,
    pub n_commits: u64,
    pub t_pack: Duration,
    pub t_exec: Duration,
    pub t_read: Duration,
    pub t_commit: Duration,
}

impl RnntLabelBackend {
    pub fn from_model(model: GigaAm, lanes: usize) -> crate::jit::Result<Self> {
        let (rnnt_head, _) =
            model.head.expect_rnnt("RnntLabelBackend").map_err(|e| JitError::Build { source: Box::new(e) })?;
        let pred_hidden = rnnt_head.pred_hidden;
        let layers = rnnt_head.pred_rnn_layers;
        let enc_hidden = model.config.d_model;

        let mut pred = RnntPredictorJit::new(model.clone());
        let mut pred_config = svod_tensor::PrepareConfig::from_env();
        pred_config.device_local_outputs = true; // g + tentative state never host-read
        pred.prepare_with_config(
            InputSpec::i64(&[lanes, 1]),
            InputSpec::f32(&[layers, lanes, pred_hidden]).device_local(),
            InputSpec::f32(&[layers, lanes, pred_hidden]).device_local(),
            &pred_config,
        )?;

        let mut joint = RnntJointJit::new(model);
        joint.prepare(
            InputSpec::f32(&[lanes, 1, enc_hidden]),
            InputSpec::f32(&[lanes, 1, pred_hidden]).device_local(),
        )?;

        Ok(Self {
            pred,
            joint,
            lanes,
            layers,
            pred_hidden,
            enc_hidden,
            frames: Vec::new(),
            zeros: vec![0u8; layers * lanes * pred_hidden * 4],
            stats: StepStats::default(),
            profile_next_step: false,
            step_profiles: None,
        })
    }

    pub fn profile_next_step(&mut self) {
        self.profile_next_step = true;
    }

    pub fn take_step_profiles(&mut self) -> Option<Vec<svod_runtime::KernelProfile>> {
        self.step_profiles.take()
    }

    pub fn bind_batch(&mut self, frames: Vec<Vec<f32>>) {
        debug_assert!(frames.len() <= self.lanes);
        self.frames = frames;
    }
}

impl BatchLabelStep for RnntLabelBackend {
    type Error = JitError;

    fn batch(&self) -> usize {
        self.lanes
    }

    fn predict(&mut self, prev: &[usize]) -> Result<(), Self::Error> {
        let t0 = Instant::now();
        {
            let buf = self.pred.prev_tokens_mut()?;
            let mut view = buf.as_array_mut::<i64>().context(DeviceSnafu)?;
            let flat = view.as_slice_mut().expect("contiguous prev_tokens");
            for (dst, &tok) in flat.iter_mut().zip(prev) {
                *dst = tok as i64;
            }
        }
        if self.profile_next_step {
            self.profile_next_step = false;
            self.step_profiles = Some(self.pred.execute_profiled()?);
        } else {
            self.pred.execute()?;
        }
        // g → joint input, device→device (the joint's plan reads its own buffer).
        let g_len = self.lanes * self.pred_hidden * 4;
        let g_src = self.pred.output_buffers()?[G_OUT].clone();
        self.joint.g_mut()?.copy_region_from(0, &g_src, 0, g_len).context(DeviceSnafu)?;
        self.stats.t_commit += t0.elapsed();
        Ok(())
    }

    fn commit(&mut self, lanes: &[bool]) {
        let t0 = Instant::now();
        self.stats.n_commits += 1;
        let (l, b, p) = (self.layers, self.lanes, self.pred_hidden);
        let lp = l * p;
        for (i, &c) in lanes.iter().enumerate() {
            if !c {
                continue;
            }
            // `state` row i is layer-major [h | c]; h_in/c_in are [L, B, P].
            for layer in 0..l {
                let dst = (layer * b * p + i * p) * 4;
                let src_h = (i * 2 * lp + layer * p) * 4;
                let src_c = (i * 2 * lp + lp + layer * p) * 4;
                self.pred.copy_output_to_h_in(STATE_OUT, dst, src_h, p * 4).expect("on-device h commit");
                self.pred.copy_output_to_c_in(STATE_OUT, dst, src_c, p * 4).expect("on-device c commit");
            }
        }
        self.stats.t_commit += t0.elapsed();
    }

    fn joint(&mut self, t: &[usize], active: &[bool], out: &mut [usize]) -> Result<(), Self::Error> {
        let e = self.enc_hidden;
        let t0 = Instant::now();
        {
            let buf = self.joint.enc_t_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
            let flat = view.as_slice_mut().expect("contiguous enc_t");
            for (i, frames) in self.frames.iter().enumerate() {
                if active[i] {
                    flat[i * e..(i + 1) * e].copy_from_slice(&frames[t[i] * e..(t[i] + 1) * e]);
                }
            }
        }
        let t1 = Instant::now();
        self.joint.execute()?;
        let t2 = Instant::now();
        {
            let buf = self.joint.output()?;
            let arr = buf.as_array::<i32>().context(DeviceSnafu)?;
            let flat = arr.as_slice().expect("contiguous tokens");
            for i in 0..self.lanes {
                out[i] = flat[i] as usize;
            }
        }
        let t3 = Instant::now();
        self.stats.n_steps += 1;
        self.stats.t_pack += t1 - t0;
        self.stats.t_exec += t2 - t1;
        self.stats.t_read += t3 - t2;
        Ok(())
    }

    fn reset(&mut self) {
        self.pred.h_in_mut().and_then(|b| b.copyin(&self.zeros).context(DeviceSnafu)).expect("zero h_in");
        self.pred.c_in_mut().and_then(|b| b.copyin(&self.zeros).context(DeviceSnafu)).expect("zero c_in");
    }
}
