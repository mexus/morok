//! FireRedVAD-driven [`Splitter`](crate::audio::Splitter) implementation.
//!
//! Pipeline: fbank → device DFSMN → FireRedVAD's trailing moving-average
//! smoothing → [`svod_arch::vad::chunks_from_probs`]. Knobs forward to
//! [`ChunkerOpts`](svod_arch::vad::ChunkerOpts) except `sample_rate`,
//! `samples_per_prob`, and `align_to`, which come from
//! [`EncoderBounds`](crate::audio::EncoderBounds) at split time. Segment
//! boundaries intentionally use the arch chunker (encoder-aware packing), not
//! the upstream reference's post-processing state machine — only the
//! probabilities match upstream.

use bon::bon;
use snafu::{ResultExt, Snafu};

use crate::audio::{AudioChunk, EncoderBounds, Splitter};
use crate::firered_vad::{FRAME_SHIFT, FireRedFbank, FireRedVad, FireRedVadInference};

pub struct FireRedVadSplitter {
    fbank: FireRedFbank,
    vad: FireRedVadInference,
    smooth_window: usize,
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
impl FireRedVadSplitter {
    /// Build from an already-prepared [`FireRedVadInference`]. Duration knobs
    /// are encoder-packing oriented (target 15-22 s chunks, hard 30 s limit,
    /// clamped to encoder capacity at split time); the prob-count knobs
    /// default to upstream FireRedVAD's frame counts (`smooth_window=5`,
    /// `min_speech=20`, `min_silence=20`, `merge=0`, at 10 ms per prob).
    /// `threshold` consults `SVOD_VAD_THRESHOLD`.
    #[builder]
    pub fn builder(
        vad: FireRedVadInference,
        #[builder(default = 5)] smooth_window: usize,
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
        /// 16 kHz). The actual pad applied is capped at half the silence gap
        /// to the neighbouring chunk — chunks never overlap into each other's
        /// speech, but at seams with enough surrounding silence the encoder
        /// sees up to this many extra samples of context on each side.
        #[builder(default = 1600)]
        pad_samples: usize,
    ) -> Self {
        Self {
            fbank: FireRedFbank::new(),
            vad,
            smooth_window,
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

    /// Convenience: download the converted FireRedVAD weights from HF Hub,
    /// prepare the JIT, and apply env-var-driven knob defaults.
    pub fn from_hub() -> Result<Self, FireRedVadSplitterError> {
        let model = crate::firered_vad::FireRedVad::from_hub().context(LoadSnafu)?;
        Self::from_model(model)
    }

    pub fn from_safetensors(path: &std::path::Path) -> Result<Self, FireRedVadSplitterError> {
        let model = FireRedVad::from_safetensors(path).context(LoadSnafu)?;
        Self::from_model(model)
    }

    fn from_model(model: FireRedVad) -> Result<Self, FireRedVadSplitterError> {
        let vad = FireRedVadInference::new(model).context(InferenceSnafu)?;
        Ok(Self::builder().vad(vad).build())
    }
}

impl Splitter for FireRedVadSplitter {
    type Error = FireRedVadSplitterError;

    fn split(&mut self, waveform: &[f32], bounds: &EncoderBounds) -> Result<Vec<AudioChunk>, Self::Error> {
        let t_fbank = std::time::Instant::now();
        let feat = self.fbank.forward(waveform);
        let n_frames = feat.len() / crate::firered_vad::N_MELS;
        let fbank_ms = t_fbank.elapsed().as_secs_f64() * 1e3;

        let t_probs = std::time::Instant::now();
        let probs = self.vad.probs(&feat, n_frames).context(ProbsSnafu)?;
        let probs_ms = t_probs.elapsed().as_secs_f64() * 1e3;
        let probs = smooth_trailing(&probs, self.smooth_window);

        tracing::info!(
            target: "svod_model::firered_vad",
            n_frames,
            fbank_ms,
            probs_ms,
            "firered vad probs breakdown (host fbank + device DFSMN)",
        );

        let cap = bounds.encoder_capacity_secs();
        let chunker_opts = svod_arch::vad::ChunkerOpts {
            sample_rate: bounds.sample_rate,
            samples_per_prob: FRAME_SHIFT,
            threshold: self.threshold,
            min_duration: self.min_duration.min(cap),
            max_duration: self.max_duration.min(cap),
            strict_limit_duration: self.strict_limit_duration.min(cap),
            min_speech_probs: self.min_speech_probs,
            min_silence_probs: self.min_silence_probs,
            merge_gap_probs: self.merge_gap_probs,
            trough_search_probs: self.trough_search_probs,
            trough_threshold: Some(self.threshold * 0.5),
            pad_samples: self.pad_samples,
            preroll_samples: 0,
            align_to: bounds.align_to_samples().max(1),
            max_total_samples: Some(waveform.len()),
        };
        svod_arch::vad::chunks_from_probs(&probs, &chunker_opts).context(ChunkSnafu)
    }

    /// Upper bound on chunk length the chunker can emit under this splitter's
    /// config, in samples — lets `Transcriber::new` size JIT buffers to the
    /// chunker's actual emission rather than the encoder's full capacity.
    /// Shared bound math lives in [`svod_arch::vad::strict_chunk_sample_bound`].
    fn max_chunk_samples(&self, bounds: &EncoderBounds) -> usize {
        let cap = bounds.encoder_capacity_secs();
        let secs = self.strict_limit_duration.min(cap);
        let probs_per_sec = bounds.sample_rate as f32 / FRAME_SHIFT as f32;
        let strict_limit_probs = (secs * probs_per_sec).ceil() as usize;
        let radius = self.trough_search_probs.unwrap_or(self.min_silence_probs);
        svod_arch::vad::strict_chunk_sample_bound(
            strict_limit_probs,
            radius,
            FRAME_SHIFT,
            self.pad_samples,
            bounds.align_to_samples(),
        )
    }
}

/// Upstream FireRedVAD probability smoothing
/// (`VadPostprocessor._smooth_prob`): a trailing moving average of the last
/// `w` probs, with the first `w - 1` entries replaced by the cumulative mean
/// of the prefix (compensating the average's ramp-up).
pub(crate) fn smooth_trailing(probs: &[f32], w: usize) -> Vec<f32> {
    if w <= 1 {
        return probs.to_vec();
    }
    (0..probs.len())
        .map(|i| {
            let lo = (i + 1).saturating_sub(w);
            probs[lo..=i].iter().sum::<f32>() / (i + 1 - lo) as f32
        })
        .collect()
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
    #[snafu(display("running FireRedVAD: {source}"))]
    Probs {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("chunker: {source}"))]
    Chunk { source: svod_arch::vad::Error },
}
