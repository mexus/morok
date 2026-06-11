//! High-level `Transcriber` over a unified [`GigaAm`].
//!
//! Replaces the per-example pipeline (mel extract → Silero VAD → chunker →
//! sized JIT prepare → batched pack → encoder execute → per-head decode) with
//! one stateful wrapper. Construction builds the front-ends but does NOT
//! prepare the encoder JIT — bounds depend on the audio. The first
//! [`Transcriber::transcribe`] call sizes and prepares; subsequent calls reuse
//! the cached `(b, t_mel)` plan if the new audio's chunks fit underneath.
//!
//! Hides the CTC vs RN-T asymmetry behind one return type:
//! [`TranscribeResult`] with optional per-word timestamps for both heads.
//! SentencePiece `▁ → space` post-processing, the encoder-output transpose
//! that RN-T needs (`[d_model, T_sub] → [T_sub, d_model]`), and the CTC
//! `frames_to_words` grouping all live inside [`HeadDecoder`].

use std::time::{Duration, Instant};

use bon::bon;
use snafu::{ResultExt, Snafu};
use svod_arch::ctc::CtcDecoder;
use svod_arch::rnnt::{RnntDecoder, RnntOpts};
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

pub use svod_arch::rnnt::Word;

use crate::audio::{AudioChunk, EncoderBounds, MelConfig, MelSpectrogram, Splitter};
use crate::gigaam::SubsamplingMode;
use crate::gigaam::ctc::GigaAmCtcJit;
use crate::gigaam::jit::GigaAmEncoderJit;
use crate::gigaam::model::{GigaAm, Head};
use crate::gigaam::rnnt::RnntBlockBackend;
use crate::jit::InputSpec;

/// User-facing knobs for [`Transcriber::transcribe`].
///
/// Construct with [`TranscribeOpts::builder`] (per-field overrides) or
/// [`TranscribeOpts::from_env`] (read `SVOD_*` env vars with sensible
/// fallbacks). The two agree — `from_env()` is just `builder().build()` —
/// so `builder().word_timestamps(true).build()` still consults env for the
/// rest of the fields.
///
/// Field defaults consult these env vars:
///
/// | Field             | Env var                | Fallback |
/// |-------------------|------------------------|----------|
/// | `word_timestamps` | `SVOD_TIMESTAMPS=1`   | `false`  |
/// | `beam_decode`     | `SVOD_BEAM_DECODE=1`  | `false`  |
/// | `max_scores_mib`  | `SVOD_MAX_SCORES_MIB` | `256`    |
///
/// `profile` is builder-only (no env var): set it programmatically.
///
/// VAD-specific knobs (`threshold`, `min_duration`, …) live on
/// [`SileroVadSplitter`](super::SileroVadSplitter), not here.
#[derive(Clone, Debug)]
pub struct TranscribeOpts {
    /// Emit per-word `Word { text, start, end }` entries on
    /// [`ChunkResult::words`]. Both heads support this.
    pub word_timestamps: bool,
    /// Promote the model's config-default CTC decoder to a beam decoder
    /// (no-op for RN-T).
    pub beam_decode: bool,
    /// Per-allocation budget for the SDPA scores buffer. Caps `max_batch`
    /// so two simultaneously live `[B, H, T_sub², dtype]` scores tensors
    /// stay under `2 × max_scores_mib` MiB.
    pub max_scores_mib: usize,
    /// Collect a typed per-stage GPU profile ([`TranscribeResult::profile`]):
    /// one representative profiled execution per GPU stage plus host stage
    /// walls. Cheap (one extra device drain per profiled stage).
    pub profile: bool,
}

impl Default for TranscribeOpts {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[bon]
impl TranscribeOpts {
    /// Build via the [`bon`] builder. Each field default consults its
    /// `SVOD_*` env var (see the struct docs for the full table) before
    /// falling back to a literal — so `builder().build()` produces the same
    /// values as [`from_env`](Self::from_env), and partial overrides
    /// (`.word_timestamps(true).build()`) still env-read the rest.
    #[builder]
    pub fn builder(
        #[builder(default = std::env::var("SVOD_TIMESTAMPS").as_deref() == Ok("1"))] word_timestamps: bool,
        #[builder(default = std::env::var("SVOD_BEAM_DECODE").as_deref() == Ok("1"))] beam_decode: bool,
        #[builder(default = std::env::var("SVOD_MAX_SCORES_MIB").ok().and_then(|s| s.parse().ok()).unwrap_or(256))]
        max_scores_mib: usize,
        #[builder(default = false)] profile: bool,
    ) -> Self {
        Self { word_timestamps, beam_decode, max_scores_mib, profile }
    }

    /// Build from `SVOD_*` env vars with the same fallbacks as the
    /// builder. Equivalent to `Self::builder().build()`.
    pub fn from_env() -> Self {
        Self::builder().build()
    }
}

/// Aggregated transcription output. `text` is the chunk texts joined by a
/// single space (empty chunks dropped); [`words`](Self::words) flattens word
/// timestamps across chunks (shifted by each chunk's `start_sec`).
#[derive(Debug)]
pub struct TranscribeResult {
    pub text: String,
    pub chunks: Vec<ChunkResult>,
    /// Model-agnostic per-stage GPU profile, when [`TranscribeOpts::profile`] is set.
    pub profile: Option<RunProfile>,
}

impl TranscribeResult {
    /// Iterate per-word timestamps across all chunks, shifted into the
    /// original audio's timeline. Empty if `opts.word_timestamps` was false.
    pub fn words(&self) -> impl Iterator<Item = Word> + '_ {
        self.chunks.iter().flat_map(|c| {
            let offset = c.start_sec;
            c.words.iter().flatten().map(move |w| Word {
                text: w.text.clone(),
                start: w.start + offset,
                end: w.end + offset,
            })
        })
    }
}

/// One VAD-bound speech region's transcript. `start_sec`/`end_sec` reference
/// the original audio. `words` is `Some` iff [`TranscribeOpts::word_timestamps`]
/// was set; each entry's `start`/`end` is **chunk-relative** (add `start_sec`
/// to get audio-absolute, or use [`TranscribeResult::words`]).
#[derive(Clone, Debug)]
pub struct ChunkResult {
    pub start_sec: f32,
    pub end_sec: f32,
    pub text: String,
    pub words: Option<Vec<Word>>,
}

/// Per-head decoder + JIT state. CTC needs a bounds-tied head JIT (Conv1d
/// projection); RN-T's block JIT rides with [`RnntBlockBackend`].
/// One instance per `Transcriber`, so the variant-size disparity is
/// irrelevant — boxing would just add an allocation.
#[allow(clippy::large_enum_variant)]
pub(crate) enum HeadDecoder {
    Ctc { jit: GigaAmCtcJit, decoder: CtcDecoder },
    Rnnt { backend: RnntBlockBackend, decoder: RnntDecoder, sentencepiece: bool },
}

/// CTC equivalent of [`RnntDecoder::frames_to_words`].
///
/// Walks the decoded `text` in lockstep with `frames` (CTC's
/// `decode_with_timestamps` returns one frame index per emitted *token*; for
/// GigaAM's char-level vocab one token == one Unicode scalar, so `text.chars()`
/// is the right zip target). Splits on ASCII space — no SentencePiece on the
/// CTC side. Returns chunk-relative `[start, end)` in seconds.
pub(crate) fn ctc_frames_to_words(text: &str, frames: &[usize], frame_shift: f32) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current = String::new();
    let mut first_frame = 0usize;
    let mut last_frame = 0usize;

    let commit = |words: &mut Vec<Word>, current: &mut String, first: usize, last: usize| {
        if !current.is_empty() {
            words.push(Word {
                text: std::mem::take(current),
                start: first as f32 * frame_shift,
                end: (last + 1) as f32 * frame_shift,
            });
        }
    };

    for (ch, &frame) in text.chars().zip(frames.iter()) {
        if ch == ' ' {
            commit(&mut words, &mut current, first_frame, last_frame);
            continue;
        }
        if current.is_empty() {
            first_frame = frame;
        }
        current.push(ch);
        last_frame = frame;
    }
    commit(&mut words, &mut current, first_frame, last_frame);
    words
}

fn rnnt_decode_err<E: std::error::Error + 'static>(
    e: svod_arch::rnnt::RnntDecodeError<crate::jit::JitError>,
) -> TranscribeError<E> {
    TranscribeError::RnntDecode { source: Box::new(e) }
}

// ─── Errors ───────────────────────────────────────────────────────────────

/// Generic over the splitter error type so per-impl errors stay
/// pattern-matchable rather than being type-erased into `Box<dyn Error>`.
/// Mirrors the `svod_arch::rnnt::RnntDecodeError<JitError>` shape.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError<E: std::error::Error + 'static> {
    #[snafu(display("splitter: {source}"))]
    Splitter { source: E },
    #[snafu(display("{source}"))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    CtcDecode { source: svod_arch::ctc::DecodeError },
    #[snafu(display("{source}"))]
    RnntDecode { source: Box<svod_arch::rnnt::RnntDecodeError<crate::jit::JitError>> },
    #[snafu(display("{source}"))]
    Model {
        #[snafu(source(from(crate::gigaam::error::Error, Box::new)))]
        source: Box<crate::gigaam::error::Error>,
    },
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    Device {
        #[snafu(source(from(svod_device::error::Error, Box::new)))]
        source: Box<svod_device::error::Error>,
    },
    #[snafu(display("WAV is {wav_sr} Hz, model expects {model_sr} Hz (resample first)"))]
    SampleRateMismatch { wav_sr: u32, model_sr: u32 },
    #[snafu(display("chunk {idx} length {samples} samples exceeds encoder capacity {max_samples} samples"))]
    ChunkExceedsCapacity { idx: usize, samples: usize, max_samples: usize },
    #[snafu(display("chunk {idx} end {end_sample} exceeds waveform length {waveform_len}"))]
    ChunkOutOfRange { idx: usize, end_sample: usize, waveform_len: usize },
}

// ─── Transcriber ──────────────────────────────────────────────────────────

/// High-level transcription wrapper, generic over the chunking strategy.
/// JITs are prepared eagerly at construction; the splitter advertises its
/// max chunk length so JIT buffers can be sized tighter than the encoder's
/// hard ceiling. Use [`transcribe_chunks`](Self::transcribe_chunks) to
/// bypass the splitter for pre-segmented audio.
pub struct Transcriber<S: Splitter> {
    model: GigaAm,
    opts: TranscribeOpts,
    splitter: S,
    mel: MelSpectrogram,
    head_decoder: HeadDecoder,
    encoder_jit: Option<GigaAmEncoderJit>,
    max_batch: usize,
    max_t_mel: usize,
}

impl<S: Splitter> Transcriber<S> {
    /// Build the transcriber and prepare every JIT eagerly — subsequent
    /// `transcribe` calls just execute. `model` is cloned into each JIT
    /// (cheap: weights are shared via `Tensor` handle Arcs).
    pub fn new(model: GigaAm, splitter: S, opts: TranscribeOpts) -> Result<Self, TranscribeError<S::Error>> {
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: model.config.sample_rate,
            n_fft: model.config.n_fft,
            hop_length: model.config.hop_length,
            win_length: model.config.win_length,
            n_mels: model.config.n_mels,
            center: model.config.mel_center,
        });

        let subsampling_factor = model.config.subsampling_factor;
        let hop_length = model.config.hop_length;
        let model_bounds = EncoderBounds {
            sample_rate: model.config.sample_rate as u32,
            hop_length,
            subsampling_factor,
            max_mel_frames: model.config.max_mel_frames,
        };
        // Splitter advertises its emission ceiling; clamp to encoder
        // capacity, then round up to the next power of two so the JIT
        // codegen sees a clean factorisation.
        let chunk_samples_cap = splitter.max_chunk_samples(&model_bounds).min(model_bounds.max_samples());
        let chunk_mel = (chunk_samples_cap / hop_length).saturating_add(2 * subsampling_factor);
        let max_t_mel = chunk_mel.max(1).next_power_of_two().min(model.config.max_mel_frames).max(subsampling_factor);

        // SDPA scores `[B, H, T_sub², dtype]` are live twice during attention;
        // budget `max_batch` so they stay under `2 * max_scores_mib`.
        let t_sub_max = (max_t_mel / subsampling_factor).max(1);
        let scores_dtype_bytes = model.encoder.input_dtype().bytes();
        let bytes_per_batch = model.config.n_heads * t_sub_max * t_sub_max * scores_dtype_bytes;
        let target_scores_bytes = opts.max_scores_mib * 1024 * 1024;
        let max_batch_by_memory = (target_scores_bytes / bytes_per_batch.max(1)).max(1);
        let max_batch = max_batch_by_memory.min(model.config.max_batch_size);

        let prepare_config = PrepareConfig::from_env();
        let mel_spec = InputSpec::f32(&[max_batch, model.config.n_mels, max_t_mel]);
        let lengths_spec = InputSpec::i32(&[max_batch]);

        // The standalone encoder JIT exists only for the RN-T path (it shares
        // the encoder with the predictor/joint step JITs). CTC fuses the
        // encoder into `GigaAmCtcJit`, so `encoder_jit` stays `None` there —
        // that fusion is what keeps the encoder output on-device.
        let mut encoder_jit: Option<GigaAmEncoderJit> = None;

        let head_decoder = match &model.head {
            Head::Ctc(_) => {
                let decoder = if opts.beam_decode {
                    match &model.config.decoder {
                        CtcDecoder::Greedy(g) => CtcDecoder::Beam(Box::new(svod_arch::ctc::BeamDecoder::new(
                            g.vocabulary().to_vec(),
                            svod_arch::ctc::BeamOpts::default(),
                        ))),
                        other => other.clone(),
                    }
                } else {
                    model.config.decoder.clone()
                };
                let mut jit = GigaAmCtcJit::new(model.clone());
                jit.prepare_with_config(mel_spec, lengths_spec, &prepare_config).context(JitSnafu)?;
                HeadDecoder::Ctc { jit, decoder }
            }
            Head::Rnnt { runtime, .. } => {
                let mut enc = GigaAmEncoderJit::new(model.clone());
                // Device-local output: the [B, T_sub, d_model] readback goes
                // over the SDMA copy queue instead of the ~21 MB/s host-mapped
                // BAR (the old first-execute hang was tied to per-execute
                // schedule re-instantiation under runtime vars; the plan is
                // all-static now).
                let mut enc_config = prepare_config.clone();
                enc_config.device_local_outputs = true;
                enc.prepare_with_config(mel_spec, lengths_spec, &enc_config).context(JitSnafu)?;
                encoder_jit = Some(enc);
                // Decode lanes are independent of the encoder batch: wider
                // waves amortize the per-step launch floor over more chunks
                // (steps per wave = max frames in the wave, not the sum).
                // State per lane is tiny; 32 lanes ≈ a chunked long file.
                const DECODE_LANES: usize = 32;
                let subs_kernel = match model.config.subsampling_mode {
                    SubsamplingMode::Conv1d => model.config.subs_kernel_size,
                    SubsamplingMode::Conv2d => 3,
                };
                let max_t_sub = subs_output_length(subs_kernel, max_t_mel);
                let backend = RnntBlockBackend::from_model(model.clone(), DECODE_LANES, max_t_sub).context(JitSnafu)?;
                let decoder = RnntDecoder::new(
                    runtime.vocabulary.clone(),
                    RnntOpts { max_symbols_per_step: runtime.max_symbols_per_step },
                );
                HeadDecoder::Rnnt { backend, decoder, sentencepiece: runtime.sentencepiece }
            }
        };

        Ok(Self { model, opts, splitter, mel, head_decoder, encoder_jit, max_batch, max_t_mel })
    }

    /// Encoder bounds at the model's full capacity. Passed to splitters
    /// at split time so they can clamp chunks to the encoder's ceiling.
    pub fn encoder_bounds(&self, sample_rate: u32) -> Result<EncoderBounds, TranscribeError<S::Error>> {
        self.bounds_with(sample_rate, self.model.config.max_mel_frames)
    }

    /// Encoder bounds tightened to this transcriber's prepared JIT capacity.
    fn prepared_bounds(&self, sample_rate: u32) -> Result<EncoderBounds, TranscribeError<S::Error>> {
        self.bounds_with(sample_rate, self.max_t_mel)
    }

    fn bounds_with(&self, sample_rate: u32, max_mel_frames: usize) -> Result<EncoderBounds, TranscribeError<S::Error>> {
        if sample_rate as usize != self.model.config.sample_rate {
            return Err(TranscribeError::SampleRateMismatch {
                wav_sr: sample_rate,
                model_sr: self.model.config.sample_rate as u32,
            });
        }
        Ok(EncoderBounds {
            sample_rate,
            hop_length: self.model.config.hop_length,
            subsampling_factor: self.model.config.subsampling_factor,
            max_mel_frames,
        })
    }

    /// Transcribe a waveform end-to-end: bounds → splitter → mel → batched
    /// encoder → per-chunk head decode. `waveform` is fp32 PCM in `[-1, 1]`;
    /// the model expects `model.config.sample_rate` (returns
    /// [`TranscribeError::SampleRateMismatch`] otherwise).
    pub fn transcribe(
        &mut self,
        waveform: &[f32],
        sample_rate: u32,
    ) -> Result<TranscribeResult, TranscribeError<S::Error>> {
        let bounds = self.encoder_bounds(sample_rate)?;
        let t_split = Instant::now();
        let chunks = self.splitter.split(waveform, &bounds).context(SplitterSnafu)?;
        let vad_wall = t_split.elapsed();
        tracing::info!(
            target: "svod_model::gigaam::transcribe",
            split_ms = vad_wall.as_secs_f64() * 1e3,
            n_chunks = chunks.len(),
            "vad split",
        );
        let mut result = self.transcribe_chunks(waveform, sample_rate, &chunks)?;
        if let Some(profile) = &mut result.profile {
            profile.stages.insert(0, StageProfile::host("vad", vad_wall));
            tracing::info!("transcribe profile\n{profile}");
        }
        Ok(result)
    }

    /// Escape hatch: caller-supplied chunks. Validates each chunk against
    /// encoder capacity (`ChunkExceedsCapacity`) and the waveform's bounds
    /// (`ChunkOutOfRange`) rather than silently truncating. Misaligned
    /// boundaries are accepted — the mel/JIT pipeline pads the trailing
    /// fractional frame.
    pub fn transcribe_chunks(
        &mut self,
        waveform: &[f32],
        sample_rate: u32,
        chunks: &[AudioChunk],
    ) -> Result<TranscribeResult, TranscribeError<S::Error>> {
        // Validate against the prepared JIT capacity, not the model's
        // worst case — oversized chunks must error here, not inside the JIT.
        let max_samples = self.prepared_bounds(sample_rate)?.max_samples();
        for (idx, chunk) in chunks.iter().enumerate() {
            if chunk.end_sample > waveform.len() {
                return Err(TranscribeError::ChunkOutOfRange {
                    idx,
                    end_sample: chunk.end_sample,
                    waveform_len: waveform.len(),
                });
            }
            let samples = chunk.end_sample.saturating_sub(chunk.start_sample);
            if samples > max_samples {
                return Err(TranscribeError::ChunkExceedsCapacity { idx, samples, max_samples });
            }
        }

        let n_mels = self.mel.n_mels();
        if chunks.is_empty() {
            return Ok(TranscribeResult { text: String::new(), chunks: Vec::new(), profile: None });
        }

        let sample_rate_hz = self.model.config.sample_rate;
        let d_model = self.model.config.d_model;
        let subs_kernel_size = match self.model.config.subsampling_mode {
            SubsamplingMode::Conv1d => self.model.config.subs_kernel_size,
            SubsamplingMode::Conv2d => 3,
        };
        let max_t_mel = self.max_t_mel;
        let max_batch = self.max_batch;
        let want_words = self.opts.word_timestamps;
        // The JIT now runs at constant shape `[max_batch, *, max_t_mel]`, so the
        // encoder output buffer is always `[max_batch, max_t_sub, *]`: per-lane
        // rows are strided by this constant max, not the per-batch active max.
        let max_t_sub = subs_output_length(subs_kernel_size, max_t_mel);

        // (start_sample, end_sample, mel_len, start_sec, end_sec) per chunk.
        let chunks_meta: Vec<(usize, usize, usize, f32, f32)> = chunks
            .iter()
            .filter_map(|c| {
                let mel_len = self.mel.num_frames(c.end_sample.saturating_sub(c.start_sample));
                if mel_len == 0 {
                    return None;
                }
                let start_sec = c.start_sample as f32 / sample_rate_hz as f32;
                let end_sec = c.end_sample as f32 / sample_rate_hz as f32;
                Some((c.start_sample, c.end_sample, mel_len, start_sec, end_sec))
            })
            .collect();
        if chunks_meta.is_empty() {
            return Ok(TranscribeResult { text: String::new(), chunks: Vec::new(), profile: None });
        }

        let num_chunks = chunks_meta.len();
        let mut chunk_results: Vec<ChunkResult> = Vec::with_capacity(num_chunks);
        // RN-T: encoder frames accumulated across all encode batches, decoded
        // afterwards in backend-wide lane waves.
        let mut all_frames: Vec<Vec<f32>> = Vec::new();
        let mut all_valid: Vec<usize> = Vec::new();
        // Per-stage wall-clock, accumulated across batches. The JITs submit
        // async (wait=false); the GPU drains on the first host `as_array()`
        // read, so each stage timer is bounded by its drain point. `encoder_ms`
        // is the fused encoder+head for CTC, encoder-only for RN-T.
        let (mut t_mel, mut t_encoder, mut t_decode) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);
        // Per-call profiling: one representative encoder batch (a steady one —
        // batch 0 pays cold caches ~6x) plus, for RN-T, one decode step.
        let profile_batch = self.opts.profile.then(|| 3.min(num_chunks.div_ceil(max_batch) - 1) * max_batch);
        let mut profile = self.opts.profile.then(RunProfile::default);
        for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
            let b = (num_chunks - chunk_batch_start).min(max_batch);
            let chunk_lengths: Vec<usize> = (0..b).map(|bi| chunks_meta[chunk_batch_start + bi].2).collect();

            let t_stage = Instant::now();
            // Chunks are independent and `forward_into` is `&self` over shared
            // read-only state (plan, filterbank, window) — parallelize the
            // batch; per-chunk output is bit-identical to the serial loop.
            use rayon::prelude::*;
            let batch_mels: Vec<Vec<f32>> = (0..b)
                .into_par_iter()
                .map(|bi| {
                    let &(start_sample, end_sample, valid, _, _) = &chunks_meta[chunk_batch_start + bi];
                    let mut chunk_mel = ndarray::Array3::<f32>::zeros((1, n_mels, valid));
                    {
                        let mut view = chunk_mel.view_mut().into_dyn();
                        self.mel.forward_into(&waveform[start_sample..end_sample], &mut view);
                    }
                    chunk_mel.as_slice().expect("contiguous chunk mel").to_vec()
                })
                .collect();
            t_mel += t_stage.elapsed();

            // Each head packs mel/lengths into ITS OWN JIT and executes. CTC's
            // `GigaAmCtcJit` is the fused encoder+head (output = log-probs, no
            // host round-trip); RN-T runs the standalone encoder JIT and decodes
            // its output per item (its predictor/joint JITs ride with the backend).
            match &mut self.head_decoder {
                HeadDecoder::Ctc { jit, decoder } => {
                    let t_pack = Instant::now();
                    pack_mel_buffer(jit.mel_mut().context(JitSnafu)?, &batch_mels, &chunk_lengths, n_mels, max_t_mel)
                        .context(DeviceSnafu)?;
                    pack_lengths_buffer(jit.lengths_mut().context(JitSnafu)?, &chunk_lengths).context(DeviceSnafu)?;
                    t_mel += t_pack.elapsed();

                    let t_enc = Instant::now();
                    if profile_batch == Some(chunk_batch_start) {
                        let kernels = jit.execute_profiled().context(JitSnafu)?;
                        if let Some(p) = &mut profile {
                            p.push(StageProfile::gpu("ctc_head", Duration::ZERO, kernels));
                        }
                    } else {
                        jit.execute().context(JitSnafu)?;
                    }
                    let total_vocab = decoder.total_vocab();
                    let item_stride = max_t_sub * total_vocab;
                    let logits_buf = jit.output().context(JitSnafu)?;
                    let logits = logits_buf.as_array::<f32>().context(DeviceSnafu)?;
                    // `as_array` drains the async fused encoder+head dispatch.
                    t_encoder += t_enc.elapsed();
                    let flat = logits.as_slice().expect("contiguous logits");
                    for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                        let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
                        let &(start_sample, end_sample, _, start_sec, end_sec) = &chunks_meta[chunk_batch_start + bi];
                        let chunk_duration_sec = (end_sample - start_sample) as f32 / sample_rate_hz as f32;
                        let frame_shift = chunk_duration_sec / (actual_sub.max(1) as f32);

                        let item_slice = &flat[bi * item_stride..bi * item_stride + item_stride];

                        let t_dec = Instant::now();
                        let (text, frames) = if want_words {
                            let (text, frames) = decoder
                                .decode_with_timestamps(item_slice, max_t_sub, actual_sub)
                                .context(CtcDecodeSnafu)?;
                            (text, Some(frames))
                        } else {
                            let text = decoder.decode(item_slice, max_t_sub, actual_sub).context(CtcDecodeSnafu)?;
                            (text, None)
                        };
                        t_decode += t_dec.elapsed();
                        let words = want_words.then(|| {
                            let frames = frames.as_deref().unwrap_or(&[]);
                            ctc_frames_to_words(&text, frames, frame_shift)
                        });
                        chunk_results.push(ChunkResult { start_sec, end_sec, text, words });
                    }
                }
                HeadDecoder::Rnnt { .. } => {
                    let enc_jit = self.encoder_jit.as_mut().expect("RN-T path has a standalone encoder JIT");
                    let t_pack = Instant::now();
                    pack_mel_buffer(
                        enc_jit.mel_mut().context(JitSnafu)?,
                        &batch_mels,
                        &chunk_lengths,
                        n_mels,
                        max_t_mel,
                    )
                    .context(DeviceSnafu)?;
                    pack_lengths_buffer(enc_jit.lengths_mut().context(JitSnafu)?, &chunk_lengths)
                        .context(DeviceSnafu)?;
                    t_mel += t_pack.elapsed();

                    let t_enc = Instant::now();
                    if profile_batch == Some(chunk_batch_start) {
                        let kernels = enc_jit.execute_profiled().context(JitSnafu)?;
                        if let Some(p) = &mut profile {
                            p.push(StageProfile::gpu("encoder", Duration::ZERO, kernels));
                        }
                    } else {
                        enc_jit.execute().context(JitSnafu)?;
                    }
                    let item_stride = max_t_sub * d_model;
                    // Output is frame-major [B, max_t_sub, d_model] (permuted
                    // on-device): one contiguous prefix copyout drains the
                    // dispatch and skips the inactive lanes of a partial last
                    // batch (lanes are leading-dim-major, so the active region
                    // is exactly the first `b` items).
                    let enc_buf = enc_jit.output().context(JitSnafu)?;
                    // f32-typed allocation: guarantees alignment for the cast.
                    let mut raw = vec![0f32; b * item_stride];
                    enc_buf.copyout_prefix(bytemuck::cast_slice_mut(&mut raw)).context(DeviceSnafu)?;
                    t_encoder += t_enc.elapsed();
                    let flat: &[f32] = &raw;
                    // Decode is per-step floor-bound, so lanes decouple from
                    // the encoder batch: collect every chunk's frames here and
                    // decode them all in one wide lockstep wave after the
                    // encode loop.
                    for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                        let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
                        let base = bi * item_stride;
                        all_frames.push(flat[base..base + actual_sub * d_model].to_vec());
                        all_valid.push(actual_sub);
                    }
                }
            }
        }

        // RN-T: decode every chunk in lane waves as wide as the backend
        // (steps per wave = the wave's max frames, not the sum over batches).
        if let HeadDecoder::Rnnt { backend, decoder, sentencepiece } = &mut self.head_decoder {
            let lanes = svod_arch::rnnt::BatchBlockStep::batch(backend);
            for wave_start in (0..all_frames.len()).step_by(lanes) {
                let wave_end = (wave_start + lanes).min(all_frames.len());
                let valid = &all_valid[wave_start..wave_end];

                let t_dec = Instant::now();
                backend.bind_batch(&all_frames[wave_start..wave_end], valid).context(JitSnafu)?;
                let lane_results = decoder.decode_batch_blocks(valid, backend).map_err(rnnt_decode_err)?;
                t_decode += t_dec.elapsed();

                for (li, (raw, emissions)) in lane_results.into_iter().enumerate() {
                    let &(start_sample, end_sample, _, start_sec, end_sec) = &chunks_meta[wave_start + li];
                    let chunk_duration_sec = (end_sample - start_sample) as f32 / sample_rate_hz as f32;
                    let frame_shift = chunk_duration_sec / (valid[li].max(1) as f32);
                    let words = want_words.then(|| decoder.frames_to_words(&emissions, frame_shift));
                    // SP pieces carry `▁` (U+2581) as word-initial markers;
                    // after concatenation we restore them as spaces.
                    let text = if *sentencepiece { raw.replace('\u{2581}', " ").trim().to_string() } else { raw };
                    chunk_results.push(ChunkResult { start_sec, end_sec, text, words });
                }
            }
        }

        if let HeadDecoder::Rnnt { backend, .. } = &self.head_decoder {
            let s = &backend.stats;
            // Two scales, kept distinct: `steps_per_lane` (= n_blocks ×
            // block_steps) is the lockstep step count one lane drives — the WIND
            // lever, which drops as the window widens. `tokens_emitted` and
            // `tape_slots` are over the full lanes × steps tape, so
            // tokens_emitted / tape_slots is the useful fraction; `n_blocks` is
            // the host-sync count.
            let block_steps = svod_arch::rnnt::BatchBlockStep::block_steps(backend) as u64;
            let lanes = svod_arch::rnnt::BatchBlockStep::batch(backend) as u64;
            let frames_total: usize = all_valid.iter().sum();
            tracing::info!(
                target: "svod_model::gigaam::transcribe",
                n_blocks = s.n_blocks,
                steps_per_lane = s.n_blocks * block_steps,
                tokens_emitted = s.steps_emitted,
                tape_slots = s.n_blocks * lanes * block_steps,
                frames_total,
                exec_ms = s.t_exec.as_secs_f64() * 1e3,
                recycle_ms = s.t_recycle.as_secs_f64() * 1e3,
                read_ms = s.t_read.as_secs_f64() * 1e3,
                "rnnt block stats",
            );
        }

        // For RN-T the predictor/joint dispatches fold into `decode_ms`.
        tracing::info!(
            target: "svod_model::gigaam::transcribe",
            num_chunks,
            mel_ms = t_mel.as_secs_f64() * 1e3,
            encoder_ms = t_encoder.as_secs_f64() * 1e3,
            decode_ms = t_decode.as_secs_f64() * 1e3,
            "gigaam stage breakdown",
        );

        if let Some(p) = &mut profile {
            // GPU stages pushed so far (encoder / ctc_head) share the accumulated
            // encoder wall; prepend the host-only mel stage so display order is
            // mel → encoder (vad is prepended by the caller).
            for s in &mut p.stages {
                s.wall = t_encoder;
            }
            p.stages.insert(0, StageProfile::host("mel", t_mel));
        }

        let text =
            chunk_results.iter().map(|c| c.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        Ok(TranscribeResult { text, chunks: chunk_results, profile })
    }
}

/// Pack per-chunk mel features into a JIT mel input buffer
/// `[max_batch, n_mels, max_t_mel]`, zero-padding unused rows/columns.
/// `batch_mels[bi]` is a tight `[n_mels, chunk_lengths[bi]]` block.
fn pack_mel_buffer(
    buf: &mut svod_device::Buffer,
    batch_mels: &[Vec<f32>],
    chunk_lengths: &[usize],
    n_mels: usize,
    max_t_mel: usize,
) -> Result<(), svod_device::error::Error> {
    let mut view = buf.as_array_mut::<f32>()?;
    let slice = view.as_slice_mut().expect("contiguous mel buffer");
    slice.fill(0.0);
    for (bi, &valid) in chunk_lengths.iter().enumerate() {
        let chunk_mel = &batch_mels[bi];
        for mel_bin in 0..n_mels {
            let src = mel_bin * valid;
            let dst = ((bi * n_mels) + mel_bin) * max_t_mel;
            slice[dst..dst + valid].copy_from_slice(&chunk_mel[src..src + valid]);
        }
    }
    Ok(())
}

/// Pack per-chunk mel-frame counts into a JIT lengths buffer `[max_batch]`,
/// zero-padding unused entries.
fn pack_lengths_buffer(
    buf: &mut svod_device::Buffer,
    chunk_lengths: &[usize],
) -> Result<(), svod_device::error::Error> {
    let mut view = buf.as_array_mut::<i32>()?;
    let slice = view.as_slice_mut().expect("contiguous lengths buffer");
    slice.fill(0);
    for (i, &len) in chunk_lengths.iter().enumerate() {
        slice[i] = len as i32;
    }
    Ok(())
}

/// Compute the encoder's sub-sampled output frame count from the input
/// mel-frame count. Mirrors the two-stage 2× stride conv stack used by
/// GigaAM's subsampling (kernel `subs_kernel_size`, stride 2, applied twice).
fn subs_output_length(kernel_size: usize, mel_frames: usize) -> usize {
    let pad = (kernel_size - 1) / 2;
    let mut len = mel_frames;
    for _ in 0..2 {
        len = (len + 2 * pad - kernel_size) / 2 + 1;
    }
    len
}
