//! Assembles a FireRedVAD-driven [`VadSplitter`] for long-form chunking.
//!
//! [`FireRedVadSplitter`] is a thin builder: it bakes a
//! [`ChunkerOpts`](svod_arch::vad::ChunkerOpts) from encoder
//! [`EncoderBounds`](crate::audio::EncoderBounds) plus FireRedVAD-tuned knobs and
//! wraps a [`FireRedVadProbs`] front-end in the arch
//! [`VadSplitter`](svod_arch::pipelines::audio::VadSplitter). Boundaries use the
//! encoder-aware arch chunker (not the upstream postprocessor's state machine);
//! only the probabilities match upstream.

use std::path::Path;

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::audio::{Vad, VadSplitter};
use svod_arch::vad::ChunkerOpts;

use crate::audio::EncoderBounds;
use crate::firered_vad::{DEFAULT_SMOOTH_WINDOW, FireRedVad, FireRedVadProbs};

/// Builder namespace for a configured `VadSplitter<FireRedVadProbs>`.
pub struct FireRedVadSplitter;

#[bon]
impl FireRedVadSplitter {
    /// Bake a [`VadSplitter`] from a [`FireRedVadProbs`] front-end and `bounds`.
    /// Duration knobs are encoder-packing oriented (target 15-22 s chunks, hard
    /// 30 s limit, clamped to encoder capacity); the prob-count knobs default to
    /// upstream FireRedVAD's frame counts (`min_speech=20`, `min_silence=20`,
    /// `merge=0`, at 10 ms per prob). `threshold` consults `SVOD_VAD_THRESHOLD`.
    #[builder]
    pub fn builder(
        vad: FireRedVadProbs,
        bounds: EncoderBounds,
        #[builder(default = std::env::var("SVOD_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.4))]
        threshold: f32,
        #[builder(default = 15.0)] min_duration: f32,
        #[builder(default = 22.0)] max_duration: f32,
        #[builder(default = 30.0)] strict_limit_duration: f32,
        #[builder(default = 20)] min_speech_probs: usize,
        #[builder(default = 20)] min_silence_probs: usize,
        #[builder(default = 0)] merge_gap_probs: usize,
        trough_search_probs: Option<usize>,
        /// Pad budget (samples) per chunk side. Default `1600` (= 100 ms at
        /// 16 kHz). The actual pad is capped at half the silence gap to the
        /// neighbouring chunk — chunks never overlap into each other's speech,
        /// but at seams with enough surrounding silence the encoder sees up to
        /// this many extra samples of context on each side.
        #[builder(default = 1600)]
        pad_samples: usize,
    ) -> VadSplitter<FireRedVadProbs> {
        let cap = bounds.encoder_capacity_secs();
        let opts = ChunkerOpts {
            sample_rate: bounds.sample_rate,
            samples_per_prob: vad.samples_per_prob(),
            threshold,
            min_duration: min_duration.min(cap),
            max_duration: max_duration.min(cap),
            strict_limit_duration: strict_limit_duration.min(cap),
            min_speech_probs,
            min_silence_probs,
            merge_gap_probs,
            trough_search_probs,
            trough_threshold: Some(threshold * 0.5),
            pad_samples,
            preroll_samples: 0,
            align_to: bounds.align_to_samples().max(1),
            // VadSplitter::split clamps to the actual waveform length per call.
            max_total_samples: None,
        };
        VadSplitter::new(vad, opts)
    }

    /// Convenience: download the converted FireRedVAD weights from HF Hub,
    /// prepare the JIT, and bake the splitter with env-var-driven knobs.
    pub fn from_hub(bounds: &EncoderBounds) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_model(FireRedVad::from_hub().context(LoadSnafu)?, bounds)
    }

    pub fn from_safetensors(
        path: &Path,
        bounds: &EncoderBounds,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_model(FireRedVad::from_safetensors(path).context(LoadSnafu)?, bounds)
    }

    fn from_model(
        model: FireRedVad,
        bounds: &EncoderBounds,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        let vad = FireRedVadProbs::new(model, DEFAULT_SMOOTH_WINDOW).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).bounds(*bounds).build())
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum FireRedVadSplitterError {
    #[snafu(display("loading FireRedVAD model: {source}"))]
    Load {
        #[snafu(source(from(crate::firered_vad::Error, Box::new)))]
        source: Box<crate::firered_vad::Error>,
    },
    #[snafu(display("building FireRedVAD JIT: {source}"))]
    Inference {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
}
