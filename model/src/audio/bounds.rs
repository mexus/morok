//! Encoder-derived sizing bounds for long-form chunking.
//!
//! [`EncoderBounds`] carries the model-config primitives (sample rate, hop,
//! subsampling, mel capacity) a splitter needs to derive its chunker config and
//! a transcriber needs to size its JIT buffers. The chunking itself lives in the
//! arch [`pipelines`](svod_arch::pipelines::audio) layer (`VadSplitter` /
//! `FixedLengthSplitter`).

pub use svod_arch::vad::AudioChunk;

/// Encoder-derived bounds for sizing chunks and JIT buffers.
///
/// Carries the model-config primitives (not derived seconds counts) so callers
/// can reason in whichever unit fits them. Helpers
/// [`max_samples`](Self::max_samples), [`align_to_samples`](Self::align_to_samples),
/// and [`encoder_capacity_secs`](Self::encoder_capacity_secs) cover the common
/// derivations.
///
/// The `2 * subsampling_factor` headroom that
/// [`encoder_capacity_secs`](Self::encoder_capacity_secs) subtracts mirrors the
/// JIT prepare loop's `subs_output_length` margin — a chunk filling
/// `max_samples()` is guaranteed to fit through the subsampling stack without
/// padding overflow.
#[derive(Clone, Copy, Debug)]
pub struct EncoderBounds {
    pub sample_rate: u32,
    pub hop_length: usize,
    pub subsampling_factor: usize,
    pub max_mel_frames: usize,
}

impl EncoderBounds {
    /// Sample-domain stride alignment: `hop_length * subsampling_factor`.
    /// Splitters that produce frame-aligned chunks snap boundaries to this
    /// multiple.
    pub fn align_to_samples(&self) -> usize {
        self.hop_length * self.subsampling_factor
    }

    /// Maximum chunk length (in samples) the encoder can ingest. Equals
    /// `(max_mel_frames - 2 * subsampling_factor) * hop_length` — the headroom
    /// subtraction matches the JIT prepare path.
    pub fn max_samples(&self) -> usize {
        self.max_mel_frames.saturating_sub(2 * self.subsampling_factor) * self.hop_length
    }

    /// Convenience for splitters that reason in wall-clock seconds.
    pub fn encoder_capacity_secs(&self) -> f32 {
        self.max_samples() as f32 / self.sample_rate as f32
    }
}
