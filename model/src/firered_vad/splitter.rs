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

use crate::audio::{ChunkerKnobs, EncoderBounds};
use crate::firered_vad::{DEFAULT_SMOOTH_WINDOW, FireRedVad, FireRedVadProbs};

/// Tuned soft-target chunk duration (seconds) for the GigaAM RN-T pipeline,
/// applied by [`ChunkProfile::Tuned`]. ~5.7 s is the joint WER+RTF optimum: short
/// chunks cut the autoregressive decoder's skip-deletions (the largest long-form
/// WER win on the Russian benchmark), and it keeps the encoder's `max_t_mel` in
/// the 1024-frame power-of-two bucket — a longer target spills into 2048,
/// ~doubling encoder cost. So `Tuned` beats `Greedy` on *both* axes here.
const TUNED_TARGET_SECS: f32 = 5.7;

/// Chunking profile for [`FireRedVadSplitter`].
///
/// On the GigaAM RN-T pipeline these are **not** a WER/RTF trade-off —
/// [`Tuned`](ChunkProfile::Tuned) wins on both axes (see [`TUNED_TARGET_SECS`]);
/// [`Greedy`](ChunkProfile::Greedy) is the legacy fill-to-max behavior, kept for
/// parity and for decoders without autoregressive skip-deletion. For a custom
/// target, set `target_duration` on the builder (or `SVOD_VAD_TARGET_CHUNK_SECS`),
/// which overrides the profile.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ChunkProfile {
    /// Soft target-split at the tuned ~5.7 s (recommended; best WER and RTF).
    #[default]
    Tuned,
    /// Greedy fill to `max_duration`; no soft target.
    Greedy,
}

impl ChunkProfile {
    /// The soft target (seconds) this profile applies, or `None` for greedy.
    fn target_secs(self) -> Option<f32> {
        match self {
            ChunkProfile::Tuned => Some(TUNED_TARGET_SECS),
            ChunkProfile::Greedy => None,
        }
    }
}

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
        /// Chunking profile (default [`ChunkProfile::Tuned`]) — soft target-split
        /// vs. greedy fill-to-max. Overridden by `target_duration` or
        /// `SVOD_VAD_TARGET_CHUNK_SECS` when set.
        #[builder(default)]
        profile: ChunkProfile,
        /// Explicit soft target chunk duration (seconds), overriding `profile` and
        /// the env var. `None` (default) defers to them. Chunks pack toward the
        /// target and re-split at 1.5× it (the long-form deletion fix); a value
        /// `>= max_duration` is a no-op (greedy).
        target_duration: Option<f32>,
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
        /// Max pre-roll (seconds) pulled into a chunk's core from the preceding
        /// silence, capped at half the gap. Default `0` (off) — FireRedVAD
        /// onsets are tight; set `SVOD_VAD_PREROLL_SECS` to enable. Mirrors the
        /// Silero splitter's knob for symmetry.
        #[builder(default = std::env::var("SVOD_VAD_PREROLL_SECS").ok().and_then(|s| s.parse().ok()).unwrap_or(0.0))]
        preroll_secs: f32,
    ) -> VadSplitter<FireRedVadProbs> {
        // Resolve the soft target: explicit override > SVOD_VAD_TARGET_CHUNK_SECS
        // > the profile (Tuned → ~5.7 s, Greedy → None). `None` ⇒ greedy.
        let target_duration = target_duration
            .or_else(|| std::env::var("SVOD_VAD_TARGET_CHUNK_SECS").ok().and_then(|s| s.parse().ok()))
            .or_else(|| profile.target_secs());
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

    /// Convenience: download the converted FireRedVAD weights from HF Hub,
    /// prepare the JIT, and bake the splitter with the default
    /// [`ChunkProfile::Tuned`] (env vars still apply).
    pub fn from_hub(bounds: &EncoderBounds) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_hub_with_profile(bounds, ChunkProfile::default())
    }

    /// Like [`from_hub`](Self::from_hub) but selects the chunking [`ChunkProfile`].
    pub fn from_hub_with_profile(
        bounds: &EncoderBounds,
        profile: ChunkProfile,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_model(FireRedVad::from_hub().context(LoadSnafu)?, bounds, profile)
    }

    pub fn from_safetensors(
        path: &Path,
        bounds: &EncoderBounds,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        Self::from_model(FireRedVad::from_safetensors(path).context(LoadSnafu)?, bounds, ChunkProfile::default())
    }

    fn from_model(
        model: FireRedVad,
        bounds: &EncoderBounds,
        profile: ChunkProfile,
    ) -> Result<VadSplitter<FireRedVadProbs>, FireRedVadSplitterError> {
        let vad = FireRedVadProbs::new(model, DEFAULT_SMOOTH_WINDOW).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).bounds(*bounds).profile(profile).build())
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
