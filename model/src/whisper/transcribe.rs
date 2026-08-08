//! Prepared Whisper recognition and aligned-transcription stages.
//!
//! Each decode window runs through a concrete-capacity encoder, one reusable
//! cross-K/V projection, token prefill, and fixed-slot cached decoding. The
//! independent aligner replays finalized tokens through a teacher-forced graph
//! and computes word timings on the host.

use std::time::{Duration, Instant};

use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::audio::{Transcriber, Transcript};
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;

use super::aligner::{WhisperAligner, WhisperAlignmentInput};
use super::config::{N_AUDIO_CTX, N_FRAMES, N_TEXT_CTX, SAMPLE_RATE};
use super::decode::{DecodeLane, DecodeOptions, DecodeStrategy, run_batched_decode};
use super::jit::{WhisperCrossKvJit, WhisperDecoderJit, WhisperDecoderStepJit, WhisperEncoderJit, WhisperPrefillJit};
use super::mel::WhisperMel;
use super::model::Whisper;
use super::plan::WhisperPlan;
use super::tokenizer::WhisperTokenizer;

pub use svod_arch::rnnt::Word;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub(crate)))]
pub enum TranscribeError {
    #[snafu(display("{source}"))]
    Jit {
        #[snafu(source(from(crate::jit::JitError, Box::new)))]
        source: Box<crate::jit::JitError>,
    },
    #[snafu(display("{source}"))]
    Model {
        #[snafu(source(from(super::error::Error, Box::new)))]
        source: Box<super::error::Error>,
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
}

/// Prepared timestamp-enabled recognizer. It owns only recognition graphs;
/// word alignment is a separate [`WhisperAligner`] stage.
pub struct WhisperRecognizer {
    mel: WhisperMel,
    encoder_jit: WhisperEncoderJit,
    decoder_jit: WhisperDecoderJit,
    cross_kv_jit: WhisperCrossKvJit,
    prefill_jit: WhisperPrefillJit,
    step_jits: rustc_hash::FxHashMap<usize, WhisperDecoderStepJit>,
    /// Concrete fixed-capacity step graph. Requests keep stable row ownership;
    /// inactive rows execute with ignored outputs.
    batched_step_jit: WhisperDecoderStepJit,
    tokenizer: WhisperTokenizer,
    options: DecodeOptions,
    n_mels: usize,
    n_audio_state: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    max_batch: usize,
    /// Max concurrent decode lanes in the batched step JIT.
    max_lanes: usize,
    plan: WhisperPlan,
    pos_embedding: Vec<f32>,
}

struct RecognizedWindow {
    result: super::decode::DecodeResult,
    audio_features: Vec<f32>,
    audio_samples: usize,
}

impl WhisperRecognizer {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        max_chunk_samples: usize,
    ) -> Result<Self, TranscribeError> {
        let plan = WhisperPlan::for_recognizer(&model.dims);
        Self::new_with_plan(model, tokenizer, options, max_chunk_samples, plan)
    }

    pub fn new_with_plan(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        max_chunk_samples: usize,
        plan: WhisperPlan,
    ) -> Result<Self, TranscribeError> {
        plan.validate().map_err(|message| TranscribeError::Model {
            source: Box::new(super::error::Error::Decode { msg: message.to_string() }),
        })?;
        options.validate().map_err(|message| TranscribeError::Model {
            source: Box::new(super::error::Error::Decode { msg: message.to_string() }),
        })?;
        let n_mels = model.dims.n_mels;
        let n_audio_state = model.dims.n_audio_state;
        let n_vocab = model.dims.n_vocab;
        let n_text_ctx = model.dims.n_text_ctx;
        let n_text_head = model.dims.n_text_head;
        if max_chunk_samples > super::config::N_SAMPLES {
            return Err(TranscribeError::Model {
                source: Box::new(super::error::Error::Decode {
                    msg: format!(
                        "Whisper decode windows are limited to {} samples, got {max_chunk_samples}",
                        super::config::N_SAMPLES
                    ),
                }),
            });
        }
        let mel = WhisperMel::new(n_mels);

        let max_batch = plan.encoder_batch;

        let prepare_config = PrepareConfig::from_env();

        // Encoder JIT: [max_batch, n_mels, N_FRAMES], device-local output
        let mut encoder_jit = WhisperEncoderJit::new(model.clone());
        let mel_spec = InputSpec::f32(&[max_batch, n_mels, N_FRAMES]);
        let mut enc_config = prepare_config.clone();
        enc_config.device_local_outputs = true;
        encoder_jit.prepare_with_config(mel_spec, &enc_config).context(JitSnafu)?;

        let n_text_state = model.dims.n_text_state;
        let n_text_layer = model.dims.n_text_layer;
        let d_head = n_text_state / n_text_head;
        let cross_cache_spec = InputSpec::f32(&[1, N_AUDIO_CTX, n_text_layer * n_text_head, d_head]).device_local();

        // Cache-consuming decoder used for language detection.
        let mut decoder_jit = WhisperDecoderJit::new(model.clone());
        let tokens_spec = InputSpec::i32(&[1, N_TEXT_CTX]);
        decoder_jit
            .prepare_with_config(cross_cache_spec.clone(), cross_cache_spec.clone(), tokens_spec, &prepare_config)
            .context(JitSnafu)?;

        // Cross-attention K/V projection is token-independent. Compile it once
        // and execute it once per encoder window, before any fallback attempts.
        let mut cross_kv_jit = WhisperCrossKvJit::new(model.clone());
        let mut cross_config = prepare_config.clone();
        cross_config.device_local_outputs = true;
        cross_kv_jit
            .prepare_with_config(InputSpec::f32(&[1, N_AUDIO_CTX, model.dims.n_text_state]), &cross_config)
            .context(JitSnafu)?;

        // Timestamp-enabled prefill has a structural, model-specific prefix:
        // multilingual [SOT, language, task], English-only [SOT].
        // Compiled once at construction, reused every window.
        let init_len = if model.is_multilingual() { 3 } else { 1 };
        let mut prefill_jit = WhisperPrefillJit::new(model.clone());
        prefill_jit
            .prepare_with_config(
                InputSpec::i32(&[1, init_len]),
                cross_cache_spec.clone(),
                cross_cache_spec,
                &prepare_config,
            )
            .context(JitSnafu)?;

        // Prepare only the concrete row counts required by the primary strategy
        // and optional sampling fallback.
        let n_text_head_local = n_text_head;
        let mut step_jits: rustc_hash::FxHashMap<usize, WhisperDecoderStepJit> = rustc_hash::FxHashMap::default();

        let beam_sizes: std::collections::HashSet<usize> = {
            let mut s = std::collections::HashSet::new();
            match options.strategy {
                DecodeStrategy::Beam { size } => {
                    s.insert(size);
                }
                DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => {
                    s.insert(1);
                }
            }
            if options.fallback.is_some() {
                s.insert(1);
            }
            s
        };
        // Device-local cache inputs: the KV caches live on-device and are
        // recycled via SDMA copy (the RN-T block decoder / firered-vad idiom).
        // The host never reads/writes cache floats — only integer offsets and
        // the logits output. This unblocks low-precision compute: the cache
        // dtype is a device-side decision, not pinned to the host Vec type.
        for &bs in &beam_sizes {
            let mut sj = WhisperDecoderStepJit::new(model.clone());
            let token_spec = InputSpec::i32(&[bs, 1]);
            let pos_emb_spec = InputSpec::f32(&[bs, 1, n_text_state]);
            let self_cache_spec =
                InputSpec::f32(&[bs, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]).device_local();
            let cross_cache_spec =
                InputSpec::f32(&[bs, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]).device_local();
            let mask_spec = InputSpec::f32(&[bs, 1, 1, N_TEXT_CTX + 1]);
            sj.prepare_with_config(
                token_spec,
                pos_emb_spec,
                self_cache_spec.clone(),
                self_cache_spec,
                cross_cache_spec.clone(),
                cross_cache_spec,
                mask_spec,
                &prepare_config,
            )
            .context(JitSnafu)?;
            step_jits.insert(bs, sj);
        }

        // Fixed concrete batch keeps tensor-core dimensions static and avoids
        // cache movement when lanes finish.
        let max_lanes = plan.decoder_slots;
        let mut batched_step_jit = WhisperDecoderStepJit::new(model.clone());
        batched_step_jit
            .prepare_with_config(
                InputSpec::i32(&[max_lanes, 1]),
                InputSpec::f32(&[max_lanes, 1, n_text_state]),
                InputSpec::f32(&[max_lanes, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
                InputSpec::f32(&[max_lanes, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
                InputSpec::f32(&[max_lanes, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
                InputSpec::f32(&[max_lanes, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]).device_local(),
                InputSpec::f32(&[max_lanes, 1, 1, N_TEXT_CTX + 1]),
                &prepare_config,
            )
            .context(JitSnafu)?;

        // Read positional embedding eagerly (static weight, reused every window).
        // Cast to fp32 — the host decode math (pos_embedding slicing in decode.rs)
        // operates on Vec<f32>, while the weight loads at dims.dtype (fp16).
        let mut pe = model
            .decoder
            .positional_embedding
            .cast(svod_dtype::DType::Float32)
            .map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?;
        pe.realize().map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?;
        let pos_embedding = pe
            .as_ndarray::<f32>()
            .map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?
            .as_slice()
            .expect("pos emb")
            .to_vec();

        Ok(Self {
            mel,
            encoder_jit,
            decoder_jit,
            cross_kv_jit,
            prefill_jit,
            step_jits,
            batched_step_jit,
            tokenizer,
            options,
            n_mels,
            n_audio_state,
            n_vocab,
            n_text_ctx,
            max_batch,
            max_lanes,
            plan,
            pos_embedding,
        })
    }

    /// Override the decode language (`None` ⇒ auto-detect) for subsequent
    /// [`Transcriber::transcribe_windows`] calls. Lets a reusable transcriber
    /// serve requests with differing languages without rebuilding the JITs.
    pub fn set_language(&mut self, language: Option<String>) {
        self.options.language = language;
    }

    /// Max concurrent decode lanes the batched step JIT was compiled for.
    /// The scheduler treats this as the GPU concurrency bound.
    pub fn max_lanes(&self) -> usize {
        self.max_lanes
    }

    pub fn plan(&self) -> &WhisperPlan {
        &self.plan
    }

    /// Greedy throughput path that decodes windows through stable fixed slots,
    /// refilling each slot as its current request finishes.
    ///
    /// Encoder + prefill still run per-window (the encoder is already
    /// batched within `max_batch`; prefill is a single 4-token forward).
    /// Fallback is disabled, so the schedule is one greedy pass per window.
    fn recognize_windows_batched_greedy(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<RecognizedWindow>, TranscribeError> {
        if windows.is_empty() {
            return Ok(Vec::new());
        }

        let n_mels = self.n_mels;
        let d = self.n_audio_state;
        let mel_stride = n_mels * N_FRAMES;
        let item_stride = N_AUDIO_CTX * d;
        let max_batch = self.max_batch;
        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        // The fixed-slot API is explicitly greedy and has no retry attempts.
        let mut batched_opts = self.options.clone();
        batched_opts.strategy = DecodeStrategy::Greedy;
        batched_opts.fallback = None;

        let mut recognized = Vec::with_capacity(windows.len());
        let mut lanes: Vec<(DecodeLane, DecodeOptions)> = Vec::with_capacity(windows.len());
        let mut alignment_audio = Vec::with_capacity(windows.len());

        // Encode + prefill in encoder-sized batches, building one DecodeLane
        // per window. The encoder already handles up to max_batch windows per
        // dispatch; prefill runs per-window on the resulting audio features.
        for batch_start in (0..windows.len()).step_by(max_batch) {
            let b = (windows.len() - batch_start).min(max_batch);

            // ── Mel: compute + pack into [b, n_mels, N_FRAMES] ──────────
            let batch_mels: Vec<Vec<f32>> = (0..b).map(|bi| self.compute_mel(windows[batch_start + bi])).collect();
            {
                let mel_buf = self.encoder_jit.mel_mut().context(JitSnafu)?;
                let mut packed = vec![0f32; max_batch * mel_stride];
                for bi in 0..b {
                    packed[bi * mel_stride..(bi + 1) * mel_stride].copy_from_slice(&batch_mels[bi][..mel_stride]);
                }
                let dst = mel_buf.as_host_bytes_mut().context(DeviceSnafu)?;
                let src_bytes: &[u8] = bytemuck::cast_slice(&packed);
                dst[..src_bytes.len()].copy_from_slice(src_bytes);
            }

            // ── Encode: one dispatch for b windows ───────────────────────
            self.encoder_jit.execute().context(JitSnafu)?;
            let out_buf = self.encoder_jit.output().context(JitSnafu)?;
            let mut raw = vec![0f32; b * item_stride];
            out_buf.copyout_prefix(bytemuck::cast_slice_mut(&mut raw)).context(DeviceSnafu)?;

            // ── Prefill: per-window, build a DecodeLane ──────────────────
            for bi in 0..b {
                let base = bi * item_stride;
                // Project this window once, then bind the packed caches into
                // both language detection and token prefill.
                {
                    let buf = self.cross_kv_jit.audio_features_mut().context(JitSnafu)?;
                    buf.copy_region_from(
                        0,
                        out_buf,
                        base * std::mem::size_of::<f32>(),
                        item_stride * std::mem::size_of::<f32>(),
                    )
                    .context(DeviceSnafu)?;
                }
                self.cross_kv_jit.execute().context(JitSnafu)?;
                {
                    let src = self.cross_kv_jit.cross_k().context(JitSnafu)?;
                    self.prefill_jit
                        .prepared_cross_k_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                    self.decoder_jit
                        .prepared_cross_k_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                    let src = self.cross_kv_jit.cross_v().context(JitSnafu)?;
                    self.prefill_jit
                        .prepared_cross_v_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                    self.decoder_jit
                        .prepared_cross_v_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                }
                // Resolve language for this window (auto-detect if unset).
                let mut lane_opts = batched_opts.clone();
                if !self.tokenizer.multilingual {
                    lane_opts.language = Some("en".to_string());
                } else if lane_opts.language.is_none() {
                    let detection =
                        super::decode::detect_language(&mut self.decoder_jit, n_text_ctx, n_vocab, &self.tokenizer)
                            .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;
                    lane_opts.language = Some(detection.language);
                }

                let lane = DecodeLane::prefill(
                    &mut self.prefill_jit,
                    &self.tokenizer,
                    &lane_opts,
                    n_text_ctx,
                    n_vocab,
                    &self.pos_embedding,
                    self.n_audio_state,
                )
                .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;

                alignment_audio.push(raw[base..base + item_stride].to_vec());
                lanes.push((lane, lane_opts));
            }
        }

        // ── Batched step-locked decode ───────────────────────────────────
        // Run one concrete max-lane dispatch per token step. Finished rows stay
        // inactive until their stable slots are reused.
        let (mut lane_states, lane_options): (Vec<DecodeLane>, Vec<DecodeOptions>) = lanes.into_iter().unzip();
        run_batched_decode(
            &mut lane_states,
            &mut self.batched_step_jit,
            self.max_lanes,
            &self.tokenizer,
            &batched_opts,
            n_text_ctx,
            n_vocab,
        )
        .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;

        // ── Collect finalized recognition artifacts ──────────────────────
        for (index, (lane, lane_opts)) in lane_states.into_iter().zip(lane_options).enumerate() {
            let mut result =
                lane.finish(&self.tokenizer, &lane_opts).map_err(|e| TranscribeError::Model { source: Box::new(e) })?;
            if result.should_skip(&lane_opts) {
                result.clear_speech();
            }

            recognized.push(RecognizedWindow {
                result,
                audio_features: std::mem::take(&mut alignment_audio[index]),
                audio_samples: windows[index].len(),
            });
        }

        Ok(recognized)
    }

    /// Compute mel spectrogram for a window, padded/trimmed to N_FRAMES.
    fn compute_mel(&self, window: &[f32]) -> Vec<f32> {
        let mel = self.mel.compute(window);
        let total = self.n_mels * N_FRAMES;

        let mut padded = vec![0.0f32; total];
        let copy_len = mel.len().min(total);
        padded[..copy_len].copy_from_slice(&mel[..copy_len]);
        padded
    }
}

impl Transcriber for WhisperRecognizer {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
        let (recognized, profile) = self.recognize_windows(windows, profile)?;
        let transcripts = recognized
            .into_iter()
            .map(|recognized| {
                let segments = super::decode::split_into_segments(
                    &recognized.result.tokens,
                    &self.tokenizer,
                    recognized.audio_samples as f32 / SAMPLE_RATE as f32,
                );
                Transcript {
                    text: recognized.result.text,
                    words: Vec::new(),
                    segments,
                    language: recognized.result.language,
                }
            })
            .collect();
        Ok((transcripts, profile))
    }
}

/// Timestamp-enabled recognizer composed with the independent, fixed-shape
/// word aligner. Every call returns DTW-aligned words; there is no feature flag
/// that changes the prepared recognition graph.
pub struct WhisperAlignedTranscriber {
    recognizer: WhisperRecognizer,
    aligner: WhisperAligner,
}

impl WhisperAlignedTranscriber {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: super::config::WhisperSize,
        max_chunk_samples: usize,
    ) -> Result<Self, TranscribeError> {
        let plan = WhisperPlan::for_model(&model.dims, size);
        Self::new_with_plan(model, tokenizer, options, size, max_chunk_samples, plan)
    }

    pub fn new_with_plan(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: super::config::WhisperSize,
        max_chunk_samples: usize,
        plan: WhisperPlan,
    ) -> Result<Self, TranscribeError> {
        plan.validate().map_err(|message| TranscribeError::Model {
            source: Box::new(super::error::Error::Decode { msg: message.to_string() }),
        })?;
        let aligner = WhisperAligner::new(model.clone(), size, plan.alignment_batch)
            .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
        let recognizer = WhisperRecognizer::new_with_plan(model, tokenizer, options, max_chunk_samples, plan)?;
        Ok(Self { recognizer, aligner })
    }

    pub fn set_language(&mut self, language: Option<String>) {
        self.recognizer.set_language(language);
    }

    /// Decode windows with the concrete fixed-slot scheduler.
    ///
    /// This throughput-oriented path uses greedy decoding and disables
    /// temperature fallback. Use [`Transcriber::transcribe_windows`] when beam
    /// search and fallback policy are required.
    pub fn transcribe_windows_batched_greedy(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<Transcript>, TranscribeError> {
        let recognized = self.recognizer.recognize_windows_batched_greedy(windows)?;
        self.align_recognized(recognized)
    }

    fn align_recognized(&mut self, recognized: Vec<RecognizedWindow>) -> Result<Vec<Transcript>, TranscribeError> {
        let task = self.recognizer.options.task;
        let tokenizer = &self.recognizer.tokenizer;
        let mut transcripts = Vec::with_capacity(recognized.len());
        for chunk in recognized.chunks(self.recognizer.plan.alignment_batch) {
            let inputs: Vec<_> = chunk
                .iter()
                .map(|recognized| WhisperAlignmentInput {
                    audio_features: &recognized.audio_features,
                    decoded_tokens: &recognized.result.tokens,
                    token_probs: &recognized.result.token_probs,
                    language: recognized.result.language.as_deref(),
                    task,
                    audio_samples: recognized.audio_samples,
                })
                .collect();
            let words = self
                .aligner
                .align_batch(&inputs, tokenizer)
                .map_err(|error| TranscribeError::Model { source: Box::new(error) })?;
            for (recognized, words) in chunk.iter().zip(words) {
                let segments = super::decode::split_into_segments(
                    &recognized.result.tokens,
                    tokenizer,
                    recognized.audio_samples as f32 / SAMPLE_RATE as f32,
                );
                transcripts.push(Transcript {
                    text: recognized.result.text.clone(),
                    words,
                    segments,
                    language: recognized.result.language.clone(),
                });
            }
        }
        Ok(transcripts)
    }
}

impl Transcriber for WhisperAlignedTranscriber {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
        let (recognized, mut profile_result) = self.recognizer.recognize_windows(windows, profile)?;
        let started = Instant::now();
        let transcripts = self.align_recognized(recognized)?;
        if let Some(profile) = &mut profile_result {
            profile.push(StageProfile::host("alignment", started.elapsed()));
        }
        Ok((transcripts, profile_result))
    }
}

impl WhisperRecognizer {
    fn recognize_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<RecognizedWindow>, Option<RunProfile>), TranscribeError> {
        if windows.is_empty() {
            return Ok((Vec::new(), profile.then(RunProfile::default)));
        }

        let n_mels = self.n_mels;
        let d = self.n_audio_state;
        let mel_stride = n_mels * N_FRAMES;
        let item_stride = N_AUDIO_CTX * d;
        let max_batch = self.max_batch;
        let n_vocab = self.n_vocab;
        let n_text_ctx = self.n_text_ctx;

        let mut recognized = Vec::with_capacity(windows.len());
        let mut prof = profile.then(RunProfile::default);
        let mut encoder_kernels = Vec::new();
        let (mut t_mel, mut t_encoder, mut t_decode) = (Duration::ZERO, Duration::ZERO, Duration::ZERO);

        for batch_start in (0..windows.len()).step_by(max_batch) {
            let b = (windows.len() - batch_start).min(max_batch);

            // ── Mel: compute + pack into [b, n_mels, N_FRAMES] ──────────────
            let t = Instant::now();
            let batch_mels: Vec<Vec<f32>> = (0..b).map(|bi| self.compute_mel(windows[batch_start + bi])).collect();
            {
                let mel_buf = self.encoder_jit.mel_mut().context(JitSnafu)?;
                let mut packed = vec![0f32; max_batch * mel_stride];
                for bi in 0..b {
                    packed[bi * mel_stride..(bi + 1) * mel_stride].copy_from_slice(&batch_mels[bi][..mel_stride]);
                }
                let dst = mel_buf.as_host_bytes_mut().context(DeviceSnafu)?;
                let src_bytes: &[u8] = bytemuck::cast_slice(&packed);
                dst[..src_bytes.len()].copy_from_slice(src_bytes);
            }
            t_mel += t.elapsed();

            // ── Encode: one dispatch for b windows ───────────────────────────
            let t = Instant::now();
            if profile && batch_start == 0 {
                encoder_kernels = self.encoder_jit.execute_profiled().context(JitSnafu)?;
            } else {
                self.encoder_jit.execute().context(JitSnafu)?;
            }

            // Device-local output → SDMA copyout of only the b valid lanes
            let out_buf = self.encoder_jit.output().context(JitSnafu)?;
            let mut raw = vec![0f32; b * item_stride];
            out_buf.copyout_prefix(bytemuck::cast_slice_mut(&mut raw)).context(DeviceSnafu)?;
            t_encoder += t.elapsed();

            // ── Decode: per-window decode ──────────────────────────────────
            for bi in 0..b {
                let base = bi * item_stride;

                let t = Instant::now();

                // Project encoder features once for all fallback prefills.
                {
                    let buf = self.cross_kv_jit.audio_features_mut().context(JitSnafu)?;
                    buf.copy_region_from(
                        0,
                        out_buf,
                        base * std::mem::size_of::<f32>(),
                        item_stride * std::mem::size_of::<f32>(),
                    )
                    .context(DeviceSnafu)?;
                }
                self.cross_kv_jit.execute().context(JitSnafu)?;
                {
                    let src = self.cross_kv_jit.cross_k().context(JitSnafu)?;
                    self.prefill_jit
                        .prepared_cross_k_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                    self.decoder_jit
                        .prepared_cross_k_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                    let src = self.cross_kv_jit.cross_v().context(JitSnafu)?;
                    self.prefill_jit
                        .prepared_cross_v_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                    self.decoder_jit
                        .prepared_cross_v_mut()
                        .context(JitSnafu)?
                        .copy_region_from(0, src, 0, src.size())
                        .context(DeviceSnafu)?;
                }

                let mut result = {
                    super::decode::decode_with_fallback_cached(
                        &mut self.prefill_jit,
                        &mut self.step_jits,
                        &mut self.decoder_jit,
                        n_text_ctx,
                        n_vocab,
                        &self.tokenizer,
                        &self.options,
                        &self.pos_embedding,
                        self.n_audio_state,
                    )
                }
                .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;
                if result.should_skip(&self.options) {
                    result.clear_speech();
                }
                t_decode += t.elapsed();

                recognized.push(RecognizedWindow {
                    result,
                    audio_features: raw[base..base + item_stride].to_vec(),
                    audio_samples: windows[batch_start + bi].len(),
                });
            }
        }

        if let Some(p) = &mut prof {
            p.push(StageProfile::host("mel", t_mel));
            p.push(StageProfile::gpu("encoder", t_encoder, encoder_kernels));
            p.push(StageProfile::host("decode", t_decode));
        }

        Ok((recognized, prof))
    }
}
