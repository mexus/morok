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

use bon::bon;
use morok_arch::ctc::CtcDecoder;
use morok_arch::rnnt::{RnntDecoder, RnntOpts};
use morok_dtype::DType;
use morok_tensor::{PrepareConfig, Tensor};
use snafu::{ResultExt, Snafu};

pub use morok_arch::rnnt::Word;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::gigaam::SubsamplingMode;
use crate::gigaam::ctc::CtcHeadJit;
use crate::gigaam::jit::GigaAmEncoderJit;
use crate::gigaam::model::{GigaAm, Head};
use crate::gigaam::rnnt::RnntStepBackend;
use crate::jit::InputSpec;
use crate::silero_vad::{NUM_SAMPLES, SileroVad, VadInference};

/// User-facing knobs for [`Transcriber::transcribe`].
///
/// Construct with [`TranscribeOpts::builder`] (per-field overrides) or
/// [`TranscribeOpts::from_env`] (read `MOROK_*` env vars with sensible
/// fallbacks). The two agree — `from_env()` is just `builder().build()` —
/// so `builder().word_timestamps(true).build()` still consults env for the
/// rest of the fields.
///
/// Field defaults consult these env vars:
///
/// | Field             | Env var                | Fallback |
/// |-------------------|------------------------|----------|
/// | `word_timestamps` | `MOROK_TIMESTAMPS=1`   | `false`  |
/// | `beam_decode`     | `MOROK_BEAM_DECODE=1`  | `false`  |
/// | `max_scores_mib`  | `MOROK_MAX_SCORES_MIB` | `256`    |
/// | `vad_threshold`   | `MOROK_VAD_THRESHOLD`  | `0.5`    |
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
    /// Speech-vs-silence threshold passed through to the chunker
    /// (`ChunkerOpts::threshold`).
    pub vad_threshold: f32,
}

impl Default for TranscribeOpts {
    fn default() -> Self {
        Self::builder().build()
    }
}

#[bon]
impl TranscribeOpts {
    /// Build via the [`bon`] builder. Each field default consults its
    /// `MOROK_*` env var (see the struct docs for the full table) before
    /// falling back to a literal — so `builder().build()` produces the same
    /// values as [`from_env`](Self::from_env), and partial overrides
    /// (`.word_timestamps(true).build()`) still env-read the rest.
    #[builder]
    pub fn builder(
        #[builder(default = std::env::var("MOROK_TIMESTAMPS").as_deref() == Ok("1"))] word_timestamps: bool,
        #[builder(default = std::env::var("MOROK_BEAM_DECODE").as_deref() == Ok("1"))] beam_decode: bool,
        #[builder(default = std::env::var("MOROK_MAX_SCORES_MIB").ok().and_then(|s| s.parse().ok()).unwrap_or(256))]
        max_scores_mib: usize,
        #[builder(default = std::env::var("MOROK_VAD_THRESHOLD").ok().and_then(|s| s.parse().ok()).unwrap_or(0.5))]
        vad_threshold: f32,
    ) -> Self {
        Self { word_timestamps, beam_decode, max_scores_mib, vad_threshold }
    }

    /// Build from `MOROK_*` env vars with the same fallbacks as the
    /// builder. Equivalent to `Self::builder().build()`.
    pub fn from_env() -> Self {
        Self::builder().build()
    }
}

/// Aggregated transcription output. `text` is the chunk texts joined by a
/// single space (empty chunks dropped); [`words`](Self::words) flattens word
/// timestamps across chunks (shifted by each chunk's `start_sec`).
#[derive(Clone, Debug)]
pub struct TranscribeResult {
    pub text: String,
    pub chunks: Vec<ChunkResult>,
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

// ─── Per-head decoder adapter ─────────────────────────────────────────────

/// Per-batch-item input to [`HeadDecoder::decode_chunk`]. Carries everything
/// `decode_chunk` needs about the *source* slab plus the chunk's wall-clock
/// duration (for `frame_shift` when timestamps are requested).
#[derive(Clone, Copy, Debug)]
pub(crate) struct ChunkLayout {
    /// Encoder hidden size. Only used by RN-T (the encoder-output transpose
    /// from `[d_model, T_sub]` to `[T_sub, d_model]`).
    pub d_model: usize,
    /// Frame stride of the per-item source slab. For CTC this is the
    /// `[t_exec_sub, V]` slab's first axis; for RN-T it's the
    /// `[d_model, t_exec_sub]` slab's last axis.
    pub t_exec_sub: usize,
    /// Valid (non-padding) sub-sampled frame count for this item. Bounds
    /// both the decoder's reads and the per-word `frame_shift` calculation.
    pub actual_sub: usize,
    /// Wall-clock duration of this chunk in seconds. `frame_shift =
    /// chunk_duration_sec / actual_sub`. Mirrors upstream GigaAM's
    /// `audio_length_samples / SAMPLE_RATE / encoder_seq_len`.
    pub chunk_duration_sec: f32,
}

/// One item's worth of decoded output. Returned by [`HeadDecoder::decode_chunk`].
#[derive(Clone, Debug)]
pub(crate) struct ChunkDecoded {
    pub text: String,
    pub words: Option<Vec<Word>>,
}

/// Per-head decoder state. Holds the arch decoder + (RN-T only) per-step
/// backend.
///
/// The CTC head JIT lives on the [`Transcriber`] (not here) so it can share
/// the bounds-cache lifecycle with the encoder JIT — one matmul per batch, not
/// per item. RN-T's predictor + joint JITs are B=1 and shape-independent, so
/// they ride with the backend.
pub(crate) enum HeadDecoder {
    Ctc { decoder: CtcDecoder },
    Rnnt { backend: Box<RnntStepBackend>, decoder: RnntDecoder, sentencepiece: bool },
}

impl HeadDecoder {
    /// Decode one batch item.
    ///
    /// `source` shape depends on the variant:
    /// - [`HeadDecoder::Ctc`]: `[layout.t_exec_sub, V]` log-probs row-major
    ///   (one item-slab from the CTC head JIT's output).
    /// - [`HeadDecoder::Rnnt`]: `[layout.d_model, layout.t_exec_sub]` encoder
    ///   output row-major (one item-slab from the encoder JIT's output). The
    ///   transpose to frame-major `[actual_sub, d_model]` happens inside.
    pub fn decode_chunk(
        &mut self,
        source: &[f32],
        layout: ChunkLayout,
        want_words: bool,
    ) -> Result<ChunkDecoded, TranscribeError> {
        let ChunkLayout { d_model, t_exec_sub, actual_sub, chunk_duration_sec } = layout;
        let frame_shift = chunk_duration_sec / (actual_sub.max(1) as f32);

        match self {
            HeadDecoder::Ctc { decoder, .. } => {
                if want_words {
                    let (text, frames) =
                        decoder.decode_with_timestamps(source, t_exec_sub, actual_sub).context(CtcDecodeSnafu)?;
                    let words = ctc_frames_to_words(&text, &frames, frame_shift);
                    Ok(ChunkDecoded { text, words: Some(words) })
                } else {
                    let text = decoder.decode(source, t_exec_sub, actual_sub).context(CtcDecodeSnafu)?;
                    Ok(ChunkDecoded { text, words: None })
                }
            }
            HeadDecoder::Rnnt { backend, decoder, sentencepiece } => {
                // Encoder output is `[d_model, t_exec_sub]` row-major; the
                // arch decoder wants frame-major `[actual_sub, d_model]`.
                let mut frames = vec![0.0_f32; actual_sub * d_model];
                for t in 0..actual_sub {
                    for d in 0..d_model {
                        frames[t * d_model + d] = source[d * t_exec_sub + t];
                    }
                }
                let backend: &mut RnntStepBackend = backend;
                let (raw, emissions, want_emissions) = if want_words {
                    let (s, e) = decoder
                        .decode_with_timestamps(&frames, actual_sub, actual_sub, d_model, backend)
                        .map_err(rnnt_decode_err)?;
                    (s, e, true)
                } else {
                    let s =
                        decoder.decode(&frames, actual_sub, actual_sub, d_model, backend).map_err(rnnt_decode_err)?;
                    (s, Vec::new(), false)
                };
                let words = want_emissions.then(|| decoder.frames_to_words(&emissions, frame_shift));
                // SP pieces carry `▁` (U+2581) as word-initial markers; after
                // concatenation we restore them as spaces. Char-level vocabs
                // (no SP) skip the replace and just trim.
                let text = if *sentencepiece { raw.replace('\u{2581}', " ").trim().to_string() } else { raw };
                Ok(ChunkDecoded { text, words })
            }
        }
    }
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

fn rnnt_decode_err(e: morok_arch::rnnt::RnntDecodeError<crate::jit::JitError>) -> TranscribeError {
    TranscribeError::RnntDecode { source: Box::new(e) }
}

// ─── Errors ───────────────────────────────────────────────────────────────

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError {
    #[snafu(display("{source}"))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    CtcDecode { source: morok_arch::ctc::DecodeError },
    #[snafu(display("{source}"))]
    RnntDecode { source: Box<morok_arch::rnnt::RnntDecodeError<crate::jit::JitError>> },
    #[snafu(display("{source}"))]
    Model {
        #[snafu(source(from(crate::gigaam::error::Error, Box::new)))]
        source: Box<crate::gigaam::error::Error>,
    },
    #[snafu(display("{source}"))]
    SileroVad {
        #[snafu(source(from(crate::silero_vad::Error, Box::new)))]
        source: Box<crate::silero_vad::Error>,
    },
    #[snafu(display("{source}"))]
    Vad { source: morok_arch::vad::Error },
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(morok_tensor::error::Error, Box::new)))]
        source: Box<morok_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    Device {
        #[snafu(source(from(morok_device::error::Error, Box::new)))]
        source: Box<morok_device::error::Error>,
    },
    #[snafu(display("WAV is {wav_sr} Hz, model expects {model_sr} Hz (resample first)"))]
    SampleRateMismatch { wav_sr: u32, model_sr: u32 },
}

// ─── Transcriber ──────────────────────────────────────────────────────────

/// High-level transcription wrapper. Owns the encoder JIT, the per-head
/// decoder (+ CTC head JIT for CTC models), the mel front-end, and the VAD.
///
/// Construction is cheap — JITs are unprepared. The first
/// [`Transcriber::transcribe`] call sizes the encoder + CTC head JITs to the
/// audio's actual VAD chunks and prepares them. Subsequent calls re-use the
/// cached `(max_batch, jit_t_mel)` plan if the new audio fits underneath;
/// otherwise they tear down the JITs and re-prepare with the larger bounds.
pub struct Transcriber {
    model: GigaAm,
    opts: TranscribeOpts,
    mel: MelSpectrogram,
    vad: VadInference,
    encoder_jit: Option<GigaAmEncoderJit>,
    /// Present iff the model has a CTC head.
    ctc_head_jit: Option<CtcHeadJit>,
    head_decoder: HeadDecoder,
    /// `Some((max_batch, jit_t_mel))` after the first prepare. The next
    /// `transcribe` reuses these bounds if its required `(b, t_mel)` fit;
    /// otherwise the JITs are rebuilt with the larger upper bounds.
    prepared_bounds: Option<(usize, usize)>,
    prepare_config: PrepareConfig,
}

impl Transcriber {
    /// Build the transcriber. Constructs the mel front-end, the Silero VAD,
    /// and the per-head decoder + step-backend JITs. The encoder JIT and the
    /// (CTC-only) head JIT are constructed lazily on the first
    /// [`transcribe`](Self::transcribe) call, since their bounds depend on
    /// the audio.
    ///
    /// `model` is consumed; the internal JITs each take their own
    /// `model.clone()` (cheap — weights are shared via the underlying
    /// `Tensor` handle Arcs).
    pub fn new(model: GigaAm, opts: TranscribeOpts) -> Result<Self, TranscribeError> {
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: model.config.sample_rate,
            n_fft: model.config.n_fft,
            hop_length: model.config.hop_length,
            win_length: model.config.win_length,
            n_mels: model.config.n_mels,
            center: model.config.mel_center,
        });
        let vad = VadInference::new(SileroVad::from_hub().context(SileroVadSnafu)?).context(JitSnafu)?;

        let head_decoder = match &model.head {
            Head::Ctc(_) => {
                let decoder = if opts.beam_decode {
                    match &model.config.decoder {
                        CtcDecoder::Greedy(g) => CtcDecoder::Beam(Box::new(morok_arch::ctc::BeamDecoder::new(
                            g.vocabulary().to_vec(),
                            morok_arch::ctc::BeamOpts::default(),
                        ))),
                        other => other.clone(),
                    }
                } else {
                    model.config.decoder.clone()
                };
                HeadDecoder::Ctc { decoder }
            }
            Head::Rnnt { runtime, .. } => {
                let backend = Box::new(RnntStepBackend::from_model(model.clone()).context(JitSnafu)?);
                let decoder = RnntDecoder::new(
                    runtime.vocabulary.clone(),
                    RnntOpts { max_symbols_per_step: runtime.max_symbols_per_step },
                );
                HeadDecoder::Rnnt { backend, decoder, sentencepiece: runtime.sentencepiece }
            }
        };

        Ok(Self {
            model,
            opts,
            mel,
            vad,
            encoder_jit: None,
            ctc_head_jit: None,
            head_decoder,
            prepared_bounds: None,
            prepare_config: PrepareConfig::from_env(),
        })
    }

    /// Transcribe a waveform end-to-end: mel → VAD → batched encoder →
    /// per-chunk head decode. `waveform` is fp32 PCM in `[-1, 1]`; the model
    /// expects `model.config.sample_rate` (returns
    /// [`TranscribeError::SampleRateMismatch`] otherwise).
    pub fn transcribe(&mut self, waveform: &[f32], sample_rate: u32) -> Result<TranscribeResult, TranscribeError> {
        if sample_rate as usize != self.model.config.sample_rate {
            return Err(TranscribeError::SampleRateMismatch {
                wav_sr: sample_rate,
                model_sr: self.model.config.sample_rate as u32,
            });
        }

        // ─── Mel features (whole audio, sliced per-chunk later) ─────────
        let n_mels = self.mel.n_mels();
        let total_mel_frames = self.mel.num_frames(waveform.len());
        if total_mel_frames == 0 {
            return Ok(TranscribeResult { text: String::new(), chunks: Vec::new() });
        }
        let mut full_mel = Tensor::full(&[1, n_mels, total_mel_frames], 0.0f32, DType::Float32).context(TensorSnafu)?;
        full_mel.realize().context(TensorSnafu)?;
        {
            let mut view = full_mel.array_view_mut::<f32>().context(TensorSnafu)?;
            self.mel.forward_into(waveform, &mut view);
        }
        let full_mel_data = full_mel.as_vec::<f32>().context(TensorSnafu)?;

        // ─── VAD chunking ───────────────────────────────────────────────
        let sample_rate_hz = self.model.config.sample_rate;
        let hop_length = self.model.config.hop_length;
        let subsampling_factor = self.model.config.subsampling_factor;
        let max_t_mel = self.model.config.max_mel_frames;
        let probs = self.vad.probs(waveform).context(JitSnafu)?;
        let mel_headroom = 2 * subsampling_factor;
        let encoder_capacity_secs =
            (max_t_mel.saturating_sub(mel_headroom) as f32 * hop_length as f32) / sample_rate_hz as f32;
        let default_opts = morok_arch::vad::ChunkerOpts::default();
        let chunker_opts = morok_arch::vad::ChunkerOpts {
            sample_rate: sample_rate_hz as u32,
            samples_per_prob: NUM_SAMPLES,
            threshold: self.opts.vad_threshold,
            max_duration: default_opts.max_duration.min(encoder_capacity_secs),
            strict_limit_duration: default_opts.strict_limit_duration.min(encoder_capacity_secs),
            align_to: hop_length * subsampling_factor,
            ..default_opts
        };
        let vad_chunks = morok_arch::vad::chunks_from_probs(&probs, &chunker_opts).context(VadSnafu)?;

        // (mel_start, mel_len, start_sec, end_sec) per chunk.
        let chunks_meta: Vec<(usize, usize, f32, f32)> = vad_chunks
            .iter()
            .filter_map(|c| {
                let mel_start = c.start_sample / hop_length;
                let mel_end = (c.end_sample / hop_length).min(total_mel_frames);
                if mel_end <= mel_start {
                    return None;
                }
                let start_sec = c.start_sample as f32 / sample_rate_hz as f32;
                let end_sec = c.end_sample as f32 / sample_rate_hz as f32;
                Some((mel_start, mel_end - mel_start, start_sec, end_sec))
            })
            .collect();
        if chunks_meta.is_empty() {
            return Ok(TranscribeResult { text: String::new(), chunks: Vec::new() });
        }

        // ─── JIT bounds: shrink to actual chunk extent ─────────────────
        let num_chunks = chunks_meta.len();
        let actual_max_chunk_mel = chunks_meta.iter().map(|(_, len, _, _)| *len).max().unwrap_or(0);
        let jit_t_mel = (actual_max_chunk_mel + 2 * subsampling_factor)
            .next_multiple_of(subsampling_factor)
            .min(max_t_mel)
            .max(subsampling_factor);

        let target_scores_bytes = self.opts.max_scores_mib * 1024 * 1024;
        let t_sub_max = (jit_t_mel / subsampling_factor).max(1);
        let scores_dtype_bytes = self.model.input_dtype().bytes();
        let bytes_per_batch = self.model.config.n_heads * t_sub_max * t_sub_max * scores_dtype_bytes;
        let max_batch_by_memory = (target_scores_bytes / bytes_per_batch.max(1)).max(1);
        let max_batch = max_batch_by_memory.min(self.model.config.max_batch_size).min(num_chunks);

        self.prepare_jits_if_needed(max_batch, jit_t_mel)?;

        let encoder_jit = self.encoder_jit.as_mut().expect("prepare_jits_if_needed leaves encoder_jit Some");
        let mut ctc_head_jit = self.ctc_head_jit.as_mut();

        // ─── Inference loop ────────────────────────────────────────────
        let subs_kernel_size = match self.model.config.subsampling_mode {
            SubsamplingMode::Conv1d => self.model.config.subs_kernel_size,
            SubsamplingMode::Conv2d => 3,
        };
        let d_model = self.model.config.d_model;
        let jit_t_sub = subs_output_length(subs_kernel_size, jit_t_mel);

        let total_vocab = match &self.head_decoder {
            HeadDecoder::Ctc { decoder } => decoder.total_vocab(),
            HeadDecoder::Rnnt { decoder, .. } => decoder.total_vocab(),
        };

        let mut chunk_results: Vec<ChunkResult> = Vec::with_capacity(num_chunks);
        for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
            let b = (num_chunks - chunk_batch_start).min(max_batch);
            let mut chunk_lengths = vec![0usize; b];

            // Pack mel + lengths.
            {
                let buf = encoder_jit.mel_mut().context(JitSnafu)?;
                let mut view = buf.as_array_mut::<f32>().context(DeviceSnafu)?;
                let slice = view.as_slice_mut().expect("contiguous mel buffer");
                slice.fill(0.0);
                for (bi, chunk_len) in chunk_lengths.iter_mut().enumerate() {
                    let &(mel_start, valid, _, _) = &chunks_meta[chunk_batch_start + bi];
                    *chunk_len = valid;
                    for mel_bin in 0..n_mels {
                        let src = mel_bin * total_mel_frames + mel_start;
                        let dst = ((bi * n_mels) + mel_bin) * jit_t_mel;
                        slice[dst..dst + valid].copy_from_slice(&full_mel_data[src..src + valid]);
                    }
                }
            }
            {
                let buf = encoder_jit.lengths_mut().context(JitSnafu)?;
                let mut view = buf.as_array_mut::<i32>().context(DeviceSnafu)?;
                let slice = view.as_slice_mut().expect("contiguous lengths buffer");
                slice.fill(0);
                for (i, len) in chunk_lengths.iter().enumerate() {
                    slice[i] = *len as i32;
                }
            }

            let t_exec = chunk_lengths.iter().copied().max().unwrap_or(1).max(1);
            let t_exec_sub = subs_output_length(subs_kernel_size, t_exec);
            encoder_jit.execute_with_vars(&[("b", b as i64), ("t", t_exec as i64)]).context(JitSnafu)?;

            // For CTC: chain encoder output into the head JIT input (cross-
            // stride 3D copy from kernel-packed [b, d_model, t_exec_sub] to
            // the head's max-stride [max_batch, d_model, jit_t_sub]) and run
            // the head once per batch. For RN-T: skip — `decode_chunk` reads
            // encoder output directly.
            let prepared_max_batch = self.prepared_bounds.expect("just prepared").0;
            if let Some(head_jit) = ctc_head_jit.as_deref_mut() {
                let n = b * d_model * t_exec_sub;
                let src_flat = encoder_jit.output().context(JitSnafu)?.as_array::<f32>().context(DeviceSnafu)?;
                let src_3d = src_flat
                    .slice(ndarray::s![0..n])
                    .into_shape_with_order((b, d_model, t_exec_sub))
                    .expect("encoder output reshape");
                let dst_flat = head_jit.encoded_mut().context(JitSnafu)?.as_array_mut::<f32>().context(DeviceSnafu)?;
                let mut dst_3d = dst_flat
                    .into_shape_with_order((prepared_max_batch, d_model, jit_t_sub))
                    .expect("head input reshape");
                dst_3d.slice_mut(ndarray::s![0..b, 0..d_model, 0..t_exec_sub]).assign(&src_3d);
                head_jit.execute_with_vars(&[("b", b as i64), ("t_sub", t_exec_sub as i64)]).context(JitSnafu)?;
            }

            // Per-item decode.
            for (bi, mel_len) in chunk_lengths.iter().enumerate() {
                let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
                let &(_, valid_mel, start_sec, end_sec) = &chunks_meta[chunk_batch_start + bi];
                let chunk_duration_sec = (valid_mel as f32) * hop_length as f32 / sample_rate_hz as f32;
                let layout = ChunkLayout { d_model, t_exec_sub, actual_sub, chunk_duration_sec };

                let decoded = match &mut self.head_decoder {
                    HeadDecoder::Ctc { .. } => {
                        let head_jit = ctc_head_jit.as_deref().expect("CTC path has head JIT");
                        let logits = head_jit.output().context(JitSnafu)?.as_array::<f32>().context(DeviceSnafu)?;
                        let flat = logits.as_slice().expect("contiguous head logits");
                        let item_stride = t_exec_sub * total_vocab;
                        let base = bi * item_stride;
                        let item_slice = flat[base..base + item_stride].to_vec();
                        self.head_decoder.decode_chunk(&item_slice, layout, self.opts.word_timestamps)?
                    }
                    HeadDecoder::Rnnt { .. } => {
                        let enc = encoder_jit.output().context(JitSnafu)?.as_array::<f32>().context(DeviceSnafu)?;
                        let flat = enc.as_slice().expect("contiguous encoder output");
                        let item_stride = d_model * t_exec_sub;
                        let base = bi * item_stride;
                        let item_slice = flat[base..base + item_stride].to_vec();
                        self.head_decoder.decode_chunk(&item_slice, layout, self.opts.word_timestamps)?
                    }
                };

                chunk_results.push(ChunkResult { start_sec, end_sec, text: decoded.text, words: decoded.words });
            }
        }

        let text =
            chunk_results.iter().map(|c| c.text.as_str()).filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" ");
        Ok(TranscribeResult { text, chunks: chunk_results })
    }

    /// Returns `true` if the next `transcribe` call would re-prepare the
    /// JITs given `(max_batch, jit_t_mel)`. Useful for tests and perf
    /// inspection.
    pub fn would_reprepare(&self, max_batch: usize, jit_t_mel: usize) -> bool {
        match self.prepared_bounds {
            None => true,
            Some((b, t)) => max_batch > b || jit_t_mel > t,
        }
    }

    fn prepare_jits_if_needed(&mut self, max_batch: usize, jit_t_mel: usize) -> Result<(), TranscribeError> {
        if !self.would_reprepare(max_batch, jit_t_mel) {
            return Ok(());
        }

        let n_mels = self.model.config.n_mels;
        let d_model = self.model.config.d_model;
        let subs_kernel_size = match self.model.config.subsampling_mode {
            SubsamplingMode::Conv1d => self.model.config.subs_kernel_size,
            SubsamplingMode::Conv2d => 3,
        };
        let jit_t_sub = subs_output_length(subs_kernel_size, jit_t_mel);

        let mut encoder_jit = GigaAmEncoderJit::new(self.model.clone()).with_b_bound(max_batch).with_t_bound(jit_t_mel);
        encoder_jit
            .prepare_with_config(
                InputSpec::f32(&[max_batch, n_mels, jit_t_mel]),
                InputSpec::i32(&[max_batch]),
                &self.prepare_config,
            )
            .context(JitSnafu)?;
        self.encoder_jit = Some(encoder_jit);

        if matches!(self.head_decoder, HeadDecoder::Ctc { .. }) {
            let mut head_jit = CtcHeadJit::new(self.model.clone()).with_b_bound(max_batch).with_t_sub_bound(jit_t_sub);
            head_jit
                .prepare_with_config(InputSpec::f32(&[max_batch, d_model, jit_t_sub]), &self.prepare_config)
                .context(JitSnafu)?;
            self.ctc_head_jit = Some(head_jit);
        }

        self.prepared_bounds = Some((max_batch, jit_t_mel));
        Ok(())
    }
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
