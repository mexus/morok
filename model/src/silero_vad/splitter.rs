//! Silero-VAD-driven [`Splitter`](crate::audio::Splitter) implementation.
//!
//! Wraps [`VadInference`] + [`morok_arch::vad::chunks_from_probs`] so the
//! pre-refactor `Transcriber` chunking flow is reachable through the generic
//! splitter trait. Knobs forward to [`ChunkerOpts`](morok_arch::vad::ChunkerOpts)
//! except for `sample_rate`, `samples_per_prob`, and `align_to` — those come
//! from [`EncoderBounds`](crate::audio::EncoderBounds) at split time.

use bon::bon;
use snafu::{ResultExt, Snafu};

use crate::audio::{AudioChunk, EncoderBounds, Splitter};
use crate::silero_vad::{NUM_SAMPLES, SileroVad, VadInference};

/// VAD-driven splitter. Construction: [`from_hub`](Self::from_hub) loads the
/// default model from HF and pulls overrides from `MOROK_VAD_THRESHOLD`. For
/// custom knobs use [`builder`](Self::builder) and supply a pre-loaded
/// [`VadInference`].
pub struct SileroVadSplitter {
    vad: VadInference,
    threshold: f32,
    min_duration: f32,
    max_duration: f32,
    strict_limit_duration: f32,
    min_speech_probs: usize,
    min_silence_probs: usize,
    merge_gap_probs: usize,
    trough_search_probs: Option<usize>,
    pad_samples: usize,
}

#[bon]
impl SileroVadSplitter {
    /// Build from an already-loaded [`VadInference`]. All knob defaults match
    /// [`ChunkerOpts::default`](morok_arch::vad::ChunkerOpts) except
    /// `threshold`, which consults `MOROK_VAD_THRESHOLD`.
    #[builder]
    pub fn builder(
        vad: VadInference,
        #[builder(default = std::env::var("MOROK_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5))]
        threshold: f32,
        #[builder(default = 15.0)] min_duration: f32,
        #[builder(default = 22.0)] max_duration: f32,
        #[builder(default = 30.0)] strict_limit_duration: f32,
        #[builder(default = 8)] min_speech_probs: usize,
        #[builder(default = 4)] min_silence_probs: usize,
        #[builder(default = 8)] merge_gap_probs: usize,
        trough_search_probs: Option<usize>,
        #[builder(default = 0)] pad_samples: usize,
    ) -> Self {
        Self {
            vad,
            threshold,
            min_duration,
            max_duration,
            strict_limit_duration,
            min_speech_probs,
            min_silence_probs,
            merge_gap_probs,
            trough_search_probs,
            pad_samples,
        }
    }

    /// Convenience: download the default Silero model from HF Hub, wrap it in
    /// a [`VadInference`], and apply env-var-driven knob defaults.
    pub fn from_hub() -> Result<Self, SileroVadSplitterError> {
        let model = SileroVad::from_hub().context(LoadSnafu)?;
        let vad = VadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).build())
    }
}

impl Splitter for SileroVadSplitter {
    type Error = SileroVadSplitterError;

    fn split(&mut self, waveform: &[f32], bounds: &EncoderBounds) -> Result<Vec<AudioChunk>, Self::Error> {
        let probs = self.vad.probs(waveform).context(ProbsSnafu)?;
        // Clamp ALL THREE duration knobs to encoder capacity. Without the
        // `min_duration` clamp the arch chunker's `MinExceedsMax` validator
        // fires when encoder capacity < 15s.
        let cap = bounds.encoder_capacity_secs();
        let chunker_opts = morok_arch::vad::ChunkerOpts {
            sample_rate: bounds.sample_rate,
            samples_per_prob: NUM_SAMPLES,
            threshold: self.threshold,
            min_duration: self.min_duration.min(cap),
            max_duration: self.max_duration.min(cap),
            strict_limit_duration: self.strict_limit_duration.min(cap),
            min_speech_probs: self.min_speech_probs,
            min_silence_probs: self.min_silence_probs,
            merge_gap_probs: self.merge_gap_probs,
            trough_search_probs: self.trough_search_probs,
            pad_samples: self.pad_samples,
            align_to: bounds.align_to_samples().max(1),
        };
        morok_arch::vad::chunks_from_probs(&probs, &chunker_opts).context(ChunkSnafu)
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum SileroVadSplitterError {
    #[snafu(display("loading Silero VAD model: {source}"))]
    Load {
        #[snafu(source(from(crate::silero_vad::Error, Box::new)))]
        source: Box<crate::silero_vad::Error>,
    },
    #[snafu(display("building Silero VAD JIT: {source}"))]
    Inference {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("running Silero VAD: {source}"))]
    Probs {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("chunker: {source}"))]
    Chunk { source: morok_arch::vad::Error },
}
