//! Batched RNN-T step backend implementing
//! [`svod_arch::rnnt::BatchJointStep`]: one fused predictor+joint JIT advances
//! all lanes of a chunk batch per dispatch.
//!
//! The LSTM state lives ON the device: `h_in`/`c_in` are device-local (no host
//! mapping), and a commit recycles the emitting lanes' `state`-output rows back
//! into them with SDMA region copies — no host round-trip. The host only writes
//! `prev` tokens + the lanes' frame-`t` encoder rows and reads back the lane
//! argmax tokens (one int each); a blank lane's tentative state is simply never
//! copied, so the committed prefix stays intact. Resets stage zeros through the
//! copy engine (`copyin` on device-local buffers is SDMA).

use std::time::{Duration, Instant};

use snafu::ResultExt;
use svod_arch::rnnt::BatchJointStep;

use crate::jit::{DeviceSnafu, InputSpec, JitError};

use super::jit::RnntBatchStepJit;
use crate::gigaam::model::GigaAm;

/// Position of the `state` output in `RnntBatchStepJit`'s `outputs { tokens, state }`.
const STATE_OUT: usize = 1;

pub struct RnntStepBackend {
    /// Fused batched JIT: `prev_tokens [B,1]`, `enc_t [B,1,E]`, `h_in`/`c_in
    /// [L,B,P]` (device-local) → `tokens [B,1]` (int32 argmax), `state
    /// [B,1,2*L*P]` (f32, layer-major `[h|c]` per lane, never host-read).
    jit: RnntBatchStepJit,

    lanes: usize,
    layers: usize,
    pred_hidden: usize,
    enc_hidden: usize,

    /// Per-lane frame-major encoder output `[valid_frames[i] * enc_hidden]` for
    /// the current chunk batch, bound by [`bind_batch`](Self::bind_batch).
    frames: Vec<Vec<f32>>,
    /// Staging zeros for [`reset`](BatchJointStep::reset) (`copyin` is SDMA).
    zeros: Vec<u8>,

    /// Per-step timing aggregates (pack / execute / read) + commit time.
    pub stats: StepStats,

    /// One-shot: the next [`step`](BatchJointStep::step) executes profiled and
    /// parks its per-dispatch kernels here. One step is representative — the
    /// fused plan dispatches the same few kernels every step.
    profile_next_step: bool,
    step_profiles: Option<Vec<svod_runtime::KernelProfile>>,
}

/// Aggregate timings for [`RnntStepBackend`].
#[derive(Default, Clone, Debug)]
pub struct StepStats {
    pub n_steps: u64,
    pub n_commits: u64,
    pub n_resets: u64,
    pub t_pack: Duration,
    pub t_exec: Duration,
    pub t_read: Duration,
    pub t_commit: Duration,
}

impl RnntStepBackend {
    /// Build the backend with `lanes` decode lanes (the transcriber's chunk
    /// batch width). The model must carry an RN-T head; CTC models are
    /// rejected.
    pub fn from_model(model: GigaAm, lanes: usize) -> crate::jit::Result<Self> {
        let (rnnt_head, _) =
            model.head.expect_rnnt("RnntStepBackend").map_err(|e| JitError::Build { source: Box::new(e) })?;
        let pred_hidden = rnnt_head.pred_hidden;
        let layers = rnnt_head.pred_rnn_layers;
        let enc_hidden = model.config.d_model;

        let mut jit = RnntBatchStepJit::new(model);
        jit.prepare(
            InputSpec::i64(&[lanes, 1]),
            InputSpec::f32(&[lanes, 1, enc_hidden]),
            InputSpec::f32(&[layers, lanes, pred_hidden]).device_local(),
            InputSpec::f32(&[layers, lanes, pred_hidden]).device_local(),
        )?;

        Ok(Self {
            jit,
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

    /// Arm a one-shot profiled step: the next [`BatchJointStep::step`] runs
    /// through `execute_profiled` and parks its kernels for
    /// [`take_step_profiles`](Self::take_step_profiles).
    pub fn profile_next_step(&mut self) {
        self.profile_next_step = true;
    }

    /// Take the parked profiled-step kernels, if a profiled step ran.
    pub fn take_step_profiles(&mut self) -> Option<Vec<svod_runtime::KernelProfile>> {
        self.step_profiles.take()
    }

    /// Bind the chunk batch's per-lane encoder output (frame-major
    /// `[valid_frames[i], enc_hidden]` each). Lanes beyond `frames.len()` stay
    /// inactive.
    pub fn bind_batch(&mut self, frames: Vec<Vec<f32>>) {
        debug_assert!(frames.len() <= self.lanes);
        self.frames = frames;
    }
}

impl BatchJointStep for RnntStepBackend {
    type Error = JitError;

    fn batch(&self) -> usize {
        self.lanes
    }

    fn step(&mut self, t: usize, prev: &[usize], active: &[bool], out: &mut [usize]) -> Result<(), Self::Error> {
        let e = self.enc_hidden;

        let t0 = Instant::now();
        {
            let buf = self.jit.prev_tokens_mut()?;
            let mut view = buf.as_array_mut::<i64>().context(DeviceSnafu)?;
            let flat = view.as_slice_mut().expect("contiguous prev_tokens");
            for (dst, &tok) in flat.iter_mut().zip(prev) {
                *dst = tok as i64;
            }
        }
        {
            let buf = self.jit.enc_t_mut()?;
            let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
            let flat = view.as_slice_mut().expect("contiguous enc_t");
            for (i, frames) in self.frames.iter().enumerate() {
                if active[i] {
                    flat[i * e..(i + 1) * e].copy_from_slice(&frames[t * e..(t + 1) * e]);
                }
            }
        }
        let t1 = Instant::now();

        if self.profile_next_step {
            self.profile_next_step = false;
            self.step_profiles = Some(self.jit.execute_profiled()?);
        } else {
            self.jit.execute()?;
        }
        let t2 = Instant::now();

        // The only per-step readback: one int per lane. The new state stays on
        // the device until a commit recycles it.
        {
            let buf = self.jit.tokens()?;
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
                self.jit.copy_output_to_h_in(STATE_OUT, dst, src_h, p * 4).expect("on-device h commit");
                self.jit.copy_output_to_c_in(STATE_OUT, dst, src_c, p * 4).expect("on-device c commit");
            }
        }
        self.stats.t_commit += t0.elapsed();
    }

    fn reset(&mut self) {
        self.stats.n_resets += 1;
        self.jit.h_in_mut().and_then(|b| b.copyin(&self.zeros).context(DeviceSnafu)).expect("zero h_in");
        self.jit.c_in_mut().and_then(|b| b.copyin(&self.zeros).context(DeviceSnafu)).expect("zero c_in");
    }
}
