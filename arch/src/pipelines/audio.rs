//! Composable long-form ASR pipeline: split → transcribe → crop → stitch.
//!
//! The orchestration is host-side and model-agnostic. A model implements only
//! its irreducible part — a [`Vad`] produces per-frame probabilities, a
//! [`Transcriber`] turns one decode-window of audio into text + word times —
//! and the heavy machinery (chunking, decode-window geometry, core-crop, and
//! stitching) lives in trait defaults here.
//!
//! ```text
//! Vad::probs ─▶ chunks_from_probs ─▶ [AudioChunk]          (VadSplitter)
//!                                        │
//!                                        ▼
//!   Transcriber::transcribe_windows(decode windows) ─▶ [Transcript]
//!                                        │  crop to core + stitch (default)
//!                                        ▼
//!                                  Transcription
//! ```
//!
//! All audio crosses the boundary as host `&[f32]` (see the module rationale):
//! decode windows are zero-copy sub-slices of the waveform, and the crate stays
//! free of the Tensor/device stack. The model owns audio → mel → device tensor
//! internally.

use std::time::Instant;

use snafu::{ResultExt, Snafu};

pub use svod_runtime::RunProfile;
use svod_runtime::StageProfile;

pub use crate::rnnt::Word;
use crate::vad::{AudioChunk, ChunkerOpts, chunks_from_probs, strict_chunk_sample_bound};

// ─── Results ────────────────────────────────────────────────────────────────

/// One decode-window's transcription, with word times **relative to the window
/// start** (`0.0` == `decode_start`). Returned by a [`Transcriber`]; the
/// pipeline crops these to the chunk's core before emitting.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    pub text: String,
    pub words: Vec<Word>,
}

/// One speech region's final transcript. `start_sec`/`end_sec` reference the
/// original audio; `words` (when present) are core-relative — add `start_sec`
/// for an absolute timeline.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkResult {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
}

/// Aggregated pipeline output: chunk texts joined by single spaces (empties
/// dropped), the per-chunk results, and the optional per-stage [`RunProfile`]
/// the transcriber collected. Profile stages are free-form and extensible, so
/// any model or caller can add custom ones.
#[derive(Debug, Default)]
pub struct Transcription {
    pub text: String,
    pub chunks: Vec<ChunkResult>,
    pub profile: Option<RunProfile>,
}

// ─── Word crop / stitch (pure host machinery) ────────────────────────────────

/// Crop decoded words back to a chunk's core and drop the rest.
///
/// Word times are relative to the decode-window start; the core begins
/// `core_offset_sec` into the window and spans `core_duration` seconds. A word
/// is kept iff its midpoint falls inside the core — so a word produced in the
/// pad/pre-roll context (or duplicated in two adjacent decode windows) survives
/// in at most one chunk. Survivors are re-based to core-relative time.
pub fn crop_words_to_core(words: Vec<Word>, core_offset_sec: f32, core_duration: f32) -> Vec<Word> {
    words
        .into_iter()
        .filter_map(|mut w| {
            let rel_start = w.start - core_offset_sec;
            let rel_end = w.end - core_offset_sec;
            let mid = 0.5 * (rel_start + rel_end);
            if !(0.0..core_duration).contains(&mid) {
                return None;
            }
            w.start = rel_start.clamp(0.0, core_duration);
            w.end = rel_end.clamp(w.start, core_duration);
            Some(w)
        })
        .collect()
}

/// Join word texts with single spaces (empties dropped). Reconstructs a chunk's
/// transcript from its (possibly cropped) words.
pub fn words_to_text(words: &[Word]) -> String {
    words.iter().map(|w| w.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ")
}

// ─── VAD ─────────────────────────────────────────────────────────────────────

/// A frame-level voice-activity detector: audio → per-frame speech
/// probabilities. Implement [`probs`](Vad::probs) (the primary op — long-form
/// runs over a single waveform); [`probs_batch`](Vad::probs_batch) defaults to
/// looping it.
pub trait Vad {
    type Error: std::error::Error + 'static;

    /// Input samples covered by one probability (the VAD frame stride).
    fn samples_per_prob(&self) -> usize;

    /// Per-frame speech probabilities for one waveform.
    fn probs(&mut self, waveform: &[f32]) -> Result<Vec<f32>, Self::Error>;

    /// Probabilities for several waveforms. Defaults to looping [`probs`];
    /// override when a VAD can batch whole clips for throughput (e.g. tuning
    /// sweeps).
    fn probs_batch(&mut self, waveforms: &[&[f32]]) -> Result<Vec<Vec<f32>>, Self::Error> {
        waveforms.iter().map(|w| self.probs(w)).collect()
    }
}

// ─── Splitter (chunk source) ──────────────────────────────────────────────────

/// Turns a waveform into ordered, bounded [`AudioChunk`]s (core + decode
/// window). The VAD-driven [`VadSplitter`] and the no-VAD [`FixedLengthSplitter`]
/// both implement it; [`Asr`] is generic over it.
pub trait Splitter {
    type Error: std::error::Error + 'static;

    fn split(&mut self, waveform: &[f32]) -> Result<Vec<AudioChunk>, Self::Error>;

    /// The profile of the most recent [`split`](Splitter::split) — e.g. a `vad`
    /// stage timing the probability pass — consumed and cleared by [`Asr`].
    /// Defaults to `None` (split is host-cheap / unprofiled).
    fn take_profile(&mut self) -> Option<RunProfile> {
        None
    }
}

/// VAD-driven splitter: `vad.probs(wav)` → [`chunks_from_probs`] with `opts`.
/// The chunker config (sample rate, `align_to`, pad, pre-roll, durations) is
/// baked in at assembly — typically from a [`Transcriber`]'s primitive bounds.
pub struct VadSplitter<V: Vad> {
    vad: V,
    opts: ChunkerOpts,
    last_profile: Option<RunProfile>,
}

impl<V: Vad> VadSplitter<V> {
    pub fn new(vad: V, opts: ChunkerOpts) -> Self {
        Self { vad, opts, last_profile: None }
    }

    /// Upper bound (in samples) on the longest chunk this splitter can emit
    /// under its baked [`ChunkerOpts`]; sizes a downstream [`Transcriber`]'s JIT
    /// buffers. Derived purely from the chunker config — same math the chunker
    /// uses internally (see [`strict_chunk_sample_bound`]).
    pub fn max_chunk_samples(&self) -> usize {
        let o = &self.opts;
        let probs_per_sec = o.sample_rate as f32 / o.samples_per_prob.max(1) as f32;
        let strict_limit_probs = (o.strict_limit_duration * probs_per_sec).ceil() as usize;
        let radius = o.trough_search_probs.unwrap_or(o.min_silence_probs);
        strict_chunk_sample_bound(strict_limit_probs, radius, o.samples_per_prob, o.pad_samples, o.align_to)
    }
}

#[derive(Debug, Snafu)]
pub enum VadSplitError<E: std::error::Error + 'static> {
    #[snafu(display("running VAD: {source}"))]
    Probs { source: E },
    #[snafu(display("chunking: {source}"))]
    Chunk { source: crate::vad::Error },
}

impl<V: Vad> Splitter for VadSplitter<V> {
    type Error = VadSplitError<V::Error>;

    fn split(&mut self, waveform: &[f32]) -> Result<Vec<AudioChunk>, Self::Error> {
        let t = Instant::now();
        let probs = self.vad.probs(waveform).context(ProbsSnafu)?;
        // The chunker clamps chunk ends to the real audio (the final VAD window
        // is zero-padded, so the prob grid overshoots the waveform). The length
        // is only known here, so set it per call over the baked sentinel.
        let mut opts = self.opts.clone();
        opts.max_total_samples = Some(waveform.len());
        let chunks = chunks_from_probs(&probs, &opts).context(ChunkSnafu)?;
        let mut profile = RunProfile::default();
        profile.push(StageProfile::host("vad", t.elapsed()));
        self.last_profile = Some(profile);
        Ok(chunks)
    }

    fn take_profile(&mut self) -> Option<RunProfile> {
        self.last_profile.take()
    }
}

/// No-VAD splitter: fixed-length non-overlapping windows (decode == core).
/// Non-final chunks are aligned to `align_to`; the last keeps its tail.
pub struct FixedLengthSplitter {
    window_samples: usize,
    align_to: usize,
}

impl FixedLengthSplitter {
    pub fn new(window_samples: usize, align_to: usize) -> Self {
        Self { window_samples: window_samples.max(1), align_to: align_to.max(1) }
    }
}

impl Splitter for FixedLengthSplitter {
    type Error = std::convert::Infallible;

    fn split(&mut self, waveform: &[f32]) -> Result<Vec<AudioChunk>, Self::Error> {
        let mut chunks = Vec::new();
        let mut start = 0usize;
        while start < waveform.len() {
            let nominal_end = start.saturating_add(self.window_samples).min(waveform.len());
            let end = if nominal_end == waveform.len() {
                nominal_end
            } else {
                let span = ((nominal_end - start) / self.align_to) * self.align_to;
                start + span.max(self.align_to)
            };
            chunks.push(AudioChunk::new(start, end));
            start = end;
        }
        Ok(chunks)
    }
}

// ─── Transcriber (per-window model) ───────────────────────────────────────────

/// Transcribes a decode-window of audio → text + word times relative to the
/// window. Implement [`transcribe_windows`](Transcriber::transcribe_windows)
/// (batched, the model owns its batch geometry); the single-window method is a
/// sequential fallback. The pipeline machinery — decode-window slicing,
/// core-crop, and stitching — is the [`transcribe_chunks`](Transcriber::transcribe_chunks)
/// default and needs no model code.
pub trait Transcriber {
    type Error: std::error::Error + 'static;

    fn sample_rate(&self) -> u32;
    /// Whether to surface per-word timestamps on [`ChunkResult::words`]. The
    /// crop always runs internally regardless.
    fn wants_words(&self) -> bool;

    /// Transcribe every decode window (the model owns internal batching),
    /// returning uncropped per-window transcripts plus the optional per-stage
    /// [`RunProfile`]. Defaults to looping [`transcribe_window`] and **merging**
    /// its per-window profiles (via [`RunProfile::merge`]); override for a model
    /// that batches the encoder and profiles the batch as a whole.
    fn transcribe_windows(&mut self, windows: &[&[f32]]) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
        let mut transcripts = Vec::with_capacity(windows.len());
        let mut profile: Option<RunProfile> = None;
        for w in windows {
            let (transcript, stage) = self.transcribe_window(w)?;
            transcripts.push(transcript);
            if let Some(stage) = stage {
                profile.get_or_insert_with(RunProfile::default).merge(stage);
            }
        }
        Ok((transcripts, profile))
    }

    /// Transcribe one window + its optional profile (sequential fallback).
    /// Implement this OR [`transcribe_windows`].
    fn transcribe_window(&mut self, window: &[f32]) -> Result<(Transcript, Option<RunProfile>), Self::Error> {
        let (mut transcripts, profile) = self.transcribe_windows(&[window])?;
        Ok((transcripts.pop().unwrap_or_default(), profile))
    }

    /// Decode each chunk's window, crop its words back to the core, stitch, and
    /// carry the profile. Pure host machinery over [`transcribe_windows`];
    /// models don't override.
    fn transcribe_chunks(&mut self, waveform: &[f32], chunks: &[AudioChunk]) -> Result<Transcription, Self::Error> {
        let sr = self.sample_rate() as f32;
        let metas: Vec<ChunkGeom> = chunks
            .iter()
            .map(|c| {
                let decode_end = c.decode_end_sample.min(waveform.len());
                ChunkGeom {
                    decode_start: c.decode_start_sample.min(decode_end),
                    decode_end,
                    start_sec: c.start_sample as f32 / sr,
                    end_sec: c.end_sample.min(waveform.len()) as f32 / sr,
                    core_offset_sec: c.start_sample.saturating_sub(c.decode_start_sample) as f32 / sr,
                }
            })
            .collect();

        let windows: Vec<&[f32]> = metas.iter().map(|m| &waveform[m.decode_start..m.decode_end]).collect();
        let (transcripts, profile) = self.transcribe_windows(&windows)?;

        let want_words = self.wants_words();
        let chunk_results: Vec<ChunkResult> = transcripts
            .into_iter()
            .zip(&metas)
            .map(|(t, m)| {
                let cropped = crop_words_to_core(t.words, m.core_offset_sec, m.end_sec - m.start_sec);
                ChunkResult {
                    start_sec: m.start_sec,
                    end_sec: m.end_sec,
                    text: words_to_text(&cropped),
                    words: want_words.then_some(cropped),
                }
            })
            .collect();

        let text =
            chunk_results.iter().map(|c| c.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        Ok(Transcription { text, chunks: chunk_results, profile })
    }
}

/// Decode geometry for one chunk, derived from its [`AudioChunk`].
struct ChunkGeom {
    decode_start: usize,
    decode_end: usize,
    start_sec: f32,
    end_sec: f32,
    core_offset_sec: f32,
}

// ─── Asr (composer) ───────────────────────────────────────────────────────────

/// The full pipeline: a chunk source ([`Splitter`]) plus a per-window model
/// ([`Transcriber`]). `transcribe` = `splitter.split` then
/// `transcriber.transcribe_chunks`.
pub struct Asr<S: Splitter, T: Transcriber> {
    splitter: S,
    transcriber: T,
}

#[derive(Debug, Snafu)]
pub enum AsrError<SE: std::error::Error + 'static, TE: std::error::Error + 'static> {
    #[snafu(display("splitting: {source}"))]
    Split { source: SE },
    #[snafu(display("transcribing: {source}"))]
    Transcribe { source: TE },
}

impl<S: Splitter, T: Transcriber> Asr<S, T> {
    pub fn new(splitter: S, transcriber: T) -> Self {
        Self { splitter, transcriber }
    }

    pub fn transcribe(&mut self, waveform: &[f32]) -> Result<Transcription, AsrError<S::Error, T::Error>> {
        let chunks = self.splitter.split(waveform).context(SplitSnafu)?;
        let split_profile = self.splitter.take_profile();
        let mut transcription = self.transcriber.transcribe_chunks(waveform, &chunks).context(TranscribeSnafu)?;
        // Surface the splitter's profile only when the transcriber is profiling
        // too (all-or-nothing), and merge it in *front* so the VAD stage leads.
        if let Some(mut vad) = split_profile
            && let Some(rest) = transcription.profile.take()
        {
            vad.merge(rest);
            transcription.profile = Some(vad);
        }
        Ok(transcription)
    }

    pub fn splitter_mut(&mut self) -> &mut S {
        &mut self.splitter
    }

    pub fn transcriber_mut(&mut self) -> &mut T {
        &mut self.transcriber
    }
}
