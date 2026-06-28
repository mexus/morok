//! Encoder-derived sizing bounds for long-form chunking.
//!
//! [`EncoderBounds`] carries the model-config primitives (sample rate, hop,
//! subsampling, mel capacity) a splitter needs to derive its chunker config and
//! a transcriber needs to size its JIT buffers. The chunking itself lives in the
//! arch [`pipelines`](svod_arch::pipelines::audio) layer (`VadSplitter` /
//! `FixedLengthSplitter`).

pub use svod_arch::vad::AudioChunk;
use svod_arch::vad::ChunkerOpts;

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
    /// Model-recommended soft chunk target (seconds), set from
    /// [`GigaAm::recommended_chunk_secs`](crate::gigaam::GigaAm::recommended_chunk_secs);
    /// `None` keeps the greedy fill-to-max. Filled by the *caller* (this module
    /// must not import `GigaAm` — that would close a gigaam→audio→gigaam cycle).
    pub recommended_target_secs: Option<f32>,
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

    /// Assemble a [`ChunkerOpts`] for a VAD producing one prob per
    /// `samples_per_prob` input samples. The bounds-derived policy — sample
    /// rate, `align_to`, the `trough_threshold = threshold/2` heuristic, the
    /// `max_total_samples` sentinel (set per call in `VadSplitter::split`), and
    /// capacity-clamping the three duration knobs — lives here, so each splitter
    /// only supplies its model-tuned [`ChunkerKnobs`].
    pub(crate) fn chunker_opts(&self, samples_per_prob: usize, k: ChunkerKnobs) -> ChunkerOpts {
        let cap = self.encoder_capacity_secs();
        ChunkerOpts {
            sample_rate: self.sample_rate,
            samples_per_prob,
            threshold: k.threshold,
            // Clamp to encoder capacity: without it the chunker's MinExceedsMax
            // validator fires when capacity < the target min duration.
            min_duration: k.min_duration.min(cap),
            max_duration: k.max_duration.min(cap),
            // Precedence: splitter override (`k.target_duration`) > the model
            // recommendation; both clamped to encoder capacity.
            target_duration: k.target_duration.or(self.recommended_target_secs).map(|t| t.min(cap)),
            strict_limit_duration: k.strict_limit_duration.min(cap),
            min_speech_probs: k.min_speech_probs,
            min_silence_probs: k.min_silence_probs,
            merge_gap_probs: k.merge_gap_probs,
            trough_search_probs: k.trough_search_probs,
            trough_threshold: Some(k.threshold * 0.5),
            pad_samples: k.pad_samples,
            preroll_samples: k.preroll_samples,
            align_to: self.align_to_samples().max(1),
            max_total_samples: None,
        }
    }
}

/// Model-tuned chunker knobs a VAD splitter supplies to
/// [`EncoderBounds::chunker_opts`]; the bounds-derived fields are filled there.
pub(crate) struct ChunkerKnobs {
    pub threshold: f32,
    pub min_duration: f32,
    pub max_duration: f32,
    /// Soft target chunk duration (seconds); `None` keeps the greedy fill-to-max.
    pub target_duration: Option<f32>,
    pub strict_limit_duration: f32,
    pub min_speech_probs: usize,
    pub min_silence_probs: usize,
    pub merge_gap_probs: usize,
    pub trough_search_probs: Option<usize>,
    pub pad_samples: usize,
    pub preroll_samples: usize,
}
