//! Device-block RNN-T backend ([`svod_arch::rnnt::BatchBlockStep`]):
//! [`super::block::forward_block`] unrolled to a single graph-captured plan;
//! per block the host recycles the five carried states with on-device copies
//! and reads back three small tapes + one flag.

use snafu::ResultExt;
use svod_arch::rnnt::{BatchBlockStep, BlockTapes};

use crate::jit::{DeviceSnafu, InputSpec, JitError};

use super::block::BLOCK_STEPS;
use super::jit::{RnntBlockJit, RnntEncProjJit};
use crate::gigaam::model::GigaAm;

/// Output order of `RnntBlockJit`.
const TAPE_OUT: usize = 0;
const EMIT_OUT: usize = 1;
const FRAME_OUT: usize = 2;
const ANY_OUT: usize = 3;
const TIME_OUT: usize = 4;
const PREV_OUT: usize = 5;
const SYMBOLS_OUT: usize = 6;
const H_OUT: usize = 7;
const C_OUT: usize = 8;

pub struct RnntBlockBackend {
    jit: RnntBlockJit,
    /// Per-wave encoder projection `[B, T, E] -> [B, T, J]` — one MFMA matmul
    /// replaces the per-step row projection inside the block.
    proj: RnntEncProjJit,
    lanes: usize,
    max_t: usize,
    enc_hidden: usize,
    state_bytes: usize,
    blank_id: usize,

    // Host-side tape staging (read once per block).
    tokens: Vec<i32>,
    emit: Vec<i32>,
    frames_tape: Vec<i32>,

    pub stats: BlockStats,
}

#[derive(Default, Clone, Debug)]
pub struct BlockStats {
    pub n_blocks: u64,
    /// Real (non-blank) emissions across all blocks. Total executed steps are
    /// `n_blocks * BLOCK_STEPS`; the gap to `steps_emitted` is the blank-advance
    /// + finished-lane overhead a wider window cuts down.
    pub steps_emitted: u64,
    pub t_exec: std::time::Duration,
    pub t_recycle: std::time::Duration,
    pub t_read: std::time::Duration,
}

impl RnntBlockBackend {
    /// `max_t` is the encoder-frame capacity (`max_t_sub`); the `enc` input is
    /// `[lanes, max_t, d_model]` and stays device-local across the wave.
    pub fn from_model(model: GigaAm, lanes: usize, max_t: usize) -> crate::jit::Result<Self> {
        let (head, _) =
            model.head.expect_rnnt("RnntBlockBackend").map_err(|e| JitError::Build { source: Box::new(e) })?;
        let (layers, p) = (head.pred_rnn_layers, head.pred_hidden);
        let joint_hidden = head.joint_hidden;
        let enc_hidden = model.config.d_model;
        let blank_id = head.predictor.blank_id;

        let mut proj = RnntEncProjJit::new(model.clone());
        let mut proj_config = svod_tensor::PrepareConfig::from_env();
        proj_config.device_local_outputs = true;
        proj.prepare_with_config(InputSpec::f32(&[lanes, max_t, enc_hidden]).device_local(), &proj_config)?;

        let mut jit = RnntBlockJit::new(model);
        let mut config = svod_tensor::PrepareConfig::from_env();
        config.device_local_outputs = true;
        jit.prepare_with_config(
            InputSpec::f32(&[lanes, max_t, joint_hidden]).device_local(),
            InputSpec::i64(&[lanes, 1]).device_local(),
            InputSpec::i64(&[lanes, 1]).device_local(),
            InputSpec::i32(&[lanes, 1]).device_local(),
            InputSpec::i32(&[lanes, 1]),
            InputSpec::f32(&[layers, lanes, p]).device_local(),
            InputSpec::f32(&[layers, lanes, p]).device_local(),
            &config,
        )?;

        Ok(Self {
            jit,
            proj,
            lanes,
            max_t,
            enc_hidden,
            state_bytes: layers * lanes * p * 4,
            blank_id,
            tokens: vec![0; lanes * BLOCK_STEPS],
            emit: vec![0; lanes * BLOCK_STEPS],
            frames_tape: vec![0; lanes * BLOCK_STEPS],
            stats: BlockStats::default(),
        })
    }

    /// Stage the wave's encoder rows + valid frame counts. `frames[i]` is the
    /// tight `[valid[i], enc_hidden]` block; unused rows stay stale (clamped
    /// gather + emit mask keep them inert).
    pub fn bind_batch(&mut self, frames: &[Vec<f32>], valid: &[usize]) -> crate::jit::Result<()> {
        let row = self.max_t * self.enc_hidden;
        let mut staged = vec![0f32; self.lanes * row];
        for (i, f) in frames.iter().enumerate() {
            staged[i * row..i * row + f.len()].copy_from_slice(f);
        }
        self.proj.enc_mut()?.copyin(bytemuck::cast_slice(&staged)).context(DeviceSnafu)?;
        self.proj.execute()?;
        // Projected rows -> block input, device->device (drains the proj exec).
        let proj_out = self.proj.output_buffers()?[0].clone();
        let bytes = proj_out.size();
        self.jit.enc_mut()?.copy_region_from(0, &proj_out, 0, bytes).context(DeviceSnafu)?;

        let mut v = vec![0i32; self.lanes];
        for (i, &n) in valid.iter().enumerate() {
            v[i] = n as i32;
        }
        let buf = self.jit.valid_mut()?;
        let mut view = buf.as_array_mut::<i32>().context(DeviceSnafu)?;
        view.as_slice_mut().expect("contiguous valid").copy_from_slice(&v);
        Ok(())
    }
}

impl BatchBlockStep for RnntBlockBackend {
    type Error = JitError;

    fn batch(&self) -> usize {
        self.lanes
    }

    fn block_steps(&self) -> usize {
        BLOCK_STEPS
    }

    fn run_block(&mut self) -> Result<BlockTapes<'_>, Self::Error> {
        let t0 = std::time::Instant::now();
        self.jit.execute()?;
        let t1 = std::time::Instant::now();

        // Recycle carried state on-device for the next block.
        self.jit.copy_output_to_time(TIME_OUT, 0, 0, self.lanes * 8)?;
        self.jit.copy_output_to_prev(PREV_OUT, 0, 0, self.lanes * 8)?;
        self.jit.copy_output_to_symbols(SYMBOLS_OUT, 0, 0, self.lanes * 4)?;
        self.jit.copy_output_to_h_in(H_OUT, 0, 0, self.state_bytes)?;
        self.jit.copy_output_to_c_in(C_OUT, 0, 0, self.state_bytes)?;
        let t2 = std::time::Instant::now();

        let outs = self.jit.output_buffers()?;
        let mut any = [0i32; 1];
        outs[TAPE_OUT].copyout_prefix(bytemuck::cast_slice_mut(&mut self.tokens)).context(DeviceSnafu)?;
        outs[EMIT_OUT].copyout_prefix(bytemuck::cast_slice_mut(&mut self.emit)).context(DeviceSnafu)?;
        outs[FRAME_OUT].copyout_prefix(bytemuck::cast_slice_mut(&mut self.frames_tape)).context(DeviceSnafu)?;
        outs[ANY_OUT].copyout_prefix(bytemuck::cast_slice_mut(&mut any)).context(DeviceSnafu)?;
        let t3 = std::time::Instant::now();

        self.stats.n_blocks += 1;
        self.stats.steps_emitted += self.emit.iter().filter(|&&e| e != 0).count() as u64;
        self.stats.t_exec += t1 - t0;
        self.stats.t_recycle += t2 - t1;
        self.stats.t_read += t3 - t2;
        Ok(BlockTapes { tokens: &self.tokens, emit: &self.emit, frames: &self.frames_tape, active_any: any[0] != 0 })
    }

    fn reset(&mut self) -> Result<(), Self::Error> {
        let zeros64 = vec![0u8; self.lanes * 8];
        let blanks: Vec<i64> = vec![self.blank_id as i64; self.lanes];
        self.jit.time_mut()?.copyin(&zeros64).context(DeviceSnafu)?;
        self.jit.prev_mut()?.copyin(bytemuck::cast_slice(&blanks)).context(DeviceSnafu)?;
        self.jit.symbols_mut()?.copyin(&zeros64[..self.lanes * 4]).context(DeviceSnafu)?;
        let zstate = vec![0u8; self.state_bytes];
        self.jit.h_in_mut()?.copyin(&zstate).context(DeviceSnafu)?;
        self.jit.c_in_mut()?.copyin(&zstate).context(DeviceSnafu)?;
        Ok(())
    }
}
