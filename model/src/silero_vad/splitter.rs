//! Assembles a Silero-VAD-driven [`VadSplitter`] for long-form chunking.
//!
//! [`SileroVadSplitter`] is a thin builder: it bakes a
//! [`ChunkerOpts`](svod_arch::vad::ChunkerOpts) from encoder
//! [`EncoderBounds`](crate::audio::EncoderBounds) plus Silero-tuned knobs and
//! wraps a [`VadInference`] front-end in the arch
//! [`VadSplitter`](svod_arch::pipelines::audio::VadSplitter).

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::audio::{Vad, VadSplitter};

use crate::audio::{ChunkerKnobs, EncoderBounds};
use crate::silero_vad::{SileroVad, VadInference};

/// Builder namespace for a configured `VadSplitter<VadInference>`.
pub struct SileroVadSplitter;

#[bon]
impl SileroVadSplitter {
    /// Bake a [`VadSplitter`] from a [`VadInference`] front-end and `bounds`.
    /// All knob defaults match [`ChunkerOpts::default`](svod_arch::vad::ChunkerOpts)
    /// except `threshold`, which consults `SVOD_VAD_THRESHOLD`.
    #[builder]
    pub fn builder(
        vad: VadInference,
        bounds: EncoderBounds,
        #[builder(default = std::env::var("SVOD_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5))]
        threshold: f32,
        #[builder(default = 15.0)] min_duration: f32,
        #[builder(default = 22.0)] max_duration: f32,
        /// Explicit soft target chunk duration (seconds), an override of the model
        /// recommendation supplied via `bounds.recommended_target_secs`. `None`
        /// (default) defers to `SVOD_VAD_TARGET_CHUNK_SECS` then that
        /// recommendation. See the FireRed splitter for the rationale.
        target_duration: Option<f32>,
        #[builder(default = 30.0)] strict_limit_duration: f32,
        #[builder(default = 8)] min_speech_probs: usize,
        #[builder(default = 4)] min_silence_probs: usize,
        #[builder(default = 8)] merge_gap_probs: usize,
        trough_search_probs: Option<usize>,
        /// Pad budget (samples) per chunk side. Default `1600` (= 100 ms at
        /// 16 kHz). The actual pad is capped at half the silence gap to the
        /// neighbouring chunk — chunks never overlap into each other's speech
        /// (no transcript duplication), but at seams with enough surrounding
        /// silence the encoder sees up to this many extra samples per side.
        #[builder(default = 1600)]
        pad_samples: usize,
        /// Max pre-roll (seconds) pulled into a chunk's core from the preceding
        /// silence, capped at half the gap. Moves the core-ownership boundary
        /// left so VAD onset lag doesn't clip the first word after a pause.
        /// Tune via `SVOD_VAD_PREROLL_SECS`.
        #[builder(default = std::env::var("SVOD_VAD_PREROLL_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5))]
        preroll_secs: f32,
    ) -> VadSplitter<VadInference> {
        // Resolve the soft target: explicit override > SVOD_VAD_TARGET_CHUNK_SECS.
        // `None` defers to the model recommendation, which `chunker_opts` ORs in
        // from `bounds.recommended_target_secs`.
        let target_duration =
            target_duration.or_else(|| std::env::var("SVOD_VAD_TARGET_CHUNK_SECS").ok().and_then(|s| s.parse().ok()));
        let opts = bounds.chunker_opts(
            vad.samples_per_prob(),
            ChunkerKnobs {
                threshold,
                min_duration,
                max_duration,
                target_duration,
                strict_limit_duration,
                min_speech_probs,
                min_silence_probs,
                merge_gap_probs,
                trough_search_probs,
                pad_samples,
                preroll_samples: (preroll_secs.max(0.0) * bounds.sample_rate as f32).round() as usize,
            },
        );
        VadSplitter::new(vad, opts)
    }

    /// Convenience: download the default Silero model from HF Hub, wrap it in a
    /// [`VadInference`], and bake the splitter with env-var-driven knobs.
    pub fn from_hub(bounds: &EncoderBounds) -> Result<VadSplitter<VadInference>, SileroVadSplitterError> {
        let model = SileroVad::from_hub().context(LoadSnafu)?;
        let vad = VadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).bounds(*bounds).build())
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
}
