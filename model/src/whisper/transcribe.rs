//! WhisperTranscriber: implements the arch `Transcriber` trait for long-form ASR.
//!
//! Each decode-window (≤30s) is: mel → encoder JIT → greedy decode.
//! The encoder JIT is batched `[max_batch, n_mels, N_FRAMES]` with device-local
//! output (SDMA copyout). The decoder JIT is compiled once at `[1, n_text_ctx]`;
//! each step writes the growing token sequence (EOT-padded) and reads only the
//! current position's logits — one compiled plan, zero recompilation.

use std::time::{Duration, Instant};

use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::audio::{Transcriber, Transcript};
use svod_runtime::{RunProfile, StageProfile};
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;

use super::config::{N_AUDIO_CTX, N_FRAMES, N_TEXT_CTX, SAMPLE_RATE};
use super::decode::{DecodeLane, DecodeOptions, run_batched_decode};
use super::jit::{WhisperDecoderJit, WhisperDecoderStepBatchedJit, WhisperDecoderStepJit, WhisperEncoderJit, WhisperPrefillJit};
use super::mel::WhisperMel;
use super::model::Whisper;
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

/// Per-window Whisper transcriber: mel frontend + batched encoder JIT +
/// compile-once decoder JIT + greedy decoder.  Produces word-level
/// timestamps via DTW on cross-attention weights (eager alignment pass).
pub struct WhisperTranscriber {
    model: Whisper,
    mel: WhisperMel,
    encoder_jit: WhisperEncoderJit,
    decoder_jit: WhisperDecoderJit,
    prefill_jit: WhisperPrefillJit,
    step_jits: rustc_hash::FxHashMap<usize, WhisperDecoderStepJit>,
    /// Continuous-batching step JIT: one plan compiled at `max_lanes`, rebound
    /// to the live lane count each dispatch via `execute_with_vars`. Substrate
    /// for [`Self::transcribe_windows_batched`] and the soroka scheduler.
    batched_step_jit: WhisperDecoderStepBatchedJit,
    tokenizer: WhisperTokenizer,
    options: DecodeOptions,
    alignment_heads: Vec<(usize, usize)>,
    n_mels: usize,
    n_audio_state: usize,
    n_vocab: usize,
    n_text_ctx: usize,
    n_text_head: usize,
    max_batch: usize,
    /// Max concurrent decode lanes in the batched step JIT.
    max_lanes: usize,
    pos_embedding: Vec<f32>,
}

impl WhisperTranscriber {
    pub fn new(
        model: Whisper,
        tokenizer: WhisperTokenizer,
        options: DecodeOptions,
        size: super::config::WhisperSize,
        _max_chunk_samples: usize,
    ) -> Result<Self, TranscribeError> {
        let n_mels = model.dims.n_mels;
        let n_audio_state = model.dims.n_audio_state;
        let n_vocab = model.dims.n_vocab;
        let n_text_ctx = model.dims.n_text_ctx;
        let n_text_head = model.dims.n_text_head;
        let alignment_heads = size.alignment_heads().to_vec();
        let mel = WhisperMel::new(n_mels);

        // Budget encoder SDPA scores [B, n_head, N_AUDIO_CTX, N_AUDIO_CTX].
        // Whisper's 1500×1500 attention is the binding memory constraint.
        const MAX_SCORES_BYTES: usize = 512 * 1024 * 1024;
        let n_head = model.dims.n_audio_head;
        let scores_per_window = n_head * N_AUDIO_CTX * N_AUDIO_CTX * std::mem::size_of::<f32>();
        let max_batch = (MAX_SCORES_BYTES / scores_per_window).clamp(1, 8);

        let prepare_config = PrepareConfig::from_env();

        // Encoder JIT: [max_batch, n_mels, N_FRAMES], device-local output
        let mut encoder_jit = WhisperEncoderJit::new(model.clone());
        let mel_spec = InputSpec::f32(&[max_batch, n_mels, N_FRAMES]);
        let mut enc_config = prepare_config.clone();
        enc_config.device_local_outputs = true;
        encoder_jit.prepare_with_config(mel_spec, &enc_config).context(JitSnafu)?;

        // Decoder JIT: [1, N_AUDIO_CTX, D] × [1, N_TEXT_CTX], compiled once.
        // Used for language detection (auto-detect path).
        let mut decoder_jit = WhisperDecoderJit::new(model.clone());
        let audio_spec = InputSpec::f32(&[1, N_AUDIO_CTX, n_audio_state]);
        let tokens_spec = InputSpec::i32(&[1, N_TEXT_CTX]);
        decoder_jit.prepare_with_config(audio_spec, tokens_spec, &prepare_config).context(JitSnafu)?;

        // Prefill JIT: [1, 4] tokens × [1, N_AUDIO_CTX, D] → logits + K/V caches.
        // Compiled once at construction, reused every window.
        let init_len = 4; // SOT + lang + task + notimestamps
        let n_text_state = model.dims.n_text_state;
        let mut prefill_jit = WhisperPrefillJit::new(model.clone());
        prefill_jit
            .prepare_with_config(
                InputSpec::i32(&[1, init_len]),
                InputSpec::f32(&[1, N_AUDIO_CTX, n_text_state]),
                &prepare_config,
            )
            .context(JitSnafu)?;

        // Step JITs: one per beam_size needed (beam_size from options + 1 for greedy).
        // Compiled once at construction, reused every step.
        let n_text_layer = model.dims.n_text_layer;
        let n_text_head_local = n_text_head;
        let d_head = n_text_state / n_text_head_local;
        let mut step_jits: rustc_hash::FxHashMap<usize, WhisperDecoderStepJit> = rustc_hash::FxHashMap::default();

        let beam_sizes: std::collections::HashSet<usize> = {
            let mut s = std::collections::HashSet::new();
            s.insert(1); // greedy (always needed for temperature fallback)
            if let Some(bs) = options.beam_size {
                s.insert(bs);
            }
            s
        };
        for &bs in &beam_sizes {
            let mut sj = WhisperDecoderStepJit::new(model.clone());
            let token_spec = InputSpec::i32(&[bs, 1]);
            let pos_emb_spec = InputSpec::f32(&[bs, 1, n_text_state]);
            let self_cache_spec = InputSpec::f32(&[bs, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]);
            let cross_cache_spec = InputSpec::f32(&[bs, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]);
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

        // Batched step JIT: one plan at max_lanes, rebound per dispatch.
        // Same shapes as the per-beam step JITs but with the batch dimension
        // symbolic (`b`), compiled once for continuous batching.
        let max_lanes = max_batch; // reuse the encoder's memory-budgeted cap
        let mut batched_step_jit = WhisperDecoderStepBatchedJit::new(model.clone()).with_b_bound(max_lanes);
        batched_step_jit
            .prepare_with_config(
                InputSpec::i32(&[max_lanes, 1]),
                InputSpec::f32(&[max_lanes, 1, n_text_state]),
                InputSpec::f32(&[max_lanes, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]),
                InputSpec::f32(&[max_lanes, N_TEXT_CTX, n_text_layer * n_text_head_local, d_head]),
                InputSpec::f32(&[max_lanes, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]),
                InputSpec::f32(&[max_lanes, N_AUDIO_CTX, n_text_layer * n_text_head_local, d_head]),
                InputSpec::f32(&[max_lanes, 1, 1, N_TEXT_CTX + 1]),
                &prepare_config,
            )
            .context(JitSnafu)?;

        // Read positional embedding eagerly (static weight, reused every window)
        let mut pe = model.decoder.positional_embedding.clone();
        pe.realize().map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?;
        let pos_embedding = pe
            .as_ndarray::<f32>()
            .map_err(|e| TranscribeError::Tensor { source: Box::new(e) })?
            .as_slice()
            .expect("pos emb")
            .to_vec();

        Ok(Self {
            model,
            mel,
            encoder_jit,
            decoder_jit,
            prefill_jit,
            step_jits,
            batched_step_jit,
            tokenizer,
            options,
            alignment_heads,
            n_mels,
            n_audio_state,
            n_vocab,
            n_text_ctx,
            n_text_head,
            max_batch,
            max_lanes,
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

    /// Batched decode path: all `windows` are decoded together via a
    /// step-locked loop over the single `batched_step_jit` (one dispatch per
    /// token step across all active lanes). This is the continuous-batching
    /// entry point — throughput-oriented, replacing the per-window serial
    /// decode of [`Transcriber::transcribe_windows`].
    ///
    /// Encoder + prefill still run per-window (the encoder is already
    /// batched within `max_batch`; prefill is a single 4-token forward).
    /// Temperature fallback is disabled (`temperature_inc = 0`), so the
    /// schedule collapses to a single greedy pass per window.
    pub fn transcribe_windows_batched(
        &mut self,
        windows: &[&[f32]],
    ) -> Result<Vec<Transcript>, TranscribeError> {
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

        // Batched-path options: disable temperature fallback so the schedule
        // collapses to a single greedy pass. Lane state is step-locked; no
        // async restarts.
        let mut batched_opts = self.options.clone();
        batched_opts.temperature_inc = 0.0;
        batched_opts.beam_size = None; // greedy-only in the batched path

        let mut transcripts = Vec::with_capacity(windows.len());
        let mut lanes: Vec<(DecodeLane, DecodeOptions)> = Vec::with_capacity(windows.len());

        // Encode + prefill in encoder-sized batches, building one DecodeLane
        // per window. The encoder already handles up to max_batch windows per
        // dispatch; prefill runs per-window on the resulting audio features.
        for batch_start in (0..windows.len()).step_by(max_batch) {
            let b = (windows.len() - batch_start).min(max_batch);

            // ── Mel: compute + pack into [b, n_mels, N_FRAMES] ──────────
            let batch_mels: Vec<Vec<f32>> =
                (0..b).map(|bi| self.compute_mel(windows[batch_start + bi])).collect();
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
                // Resolve language for this window (auto-detect if unset).
                let mut lane_opts = batched_opts.clone();
                if lane_opts.language.is_none() {
                    // Load this window's audio features into the uncached
                    // decoder JIT for language detection.
                    {
                        let buf = self.decoder_jit.audio_features_mut().context(JitSnafu)?;
                        let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
                        let src_bytes: &[u8] = bytemuck::cast_slice(&raw[base..base + item_stride]);
                        dst[..src_bytes.len()].copy_from_slice(src_bytes);
                    }
                    let detection = super::decode::detect_language(
                        &mut self.decoder_jit,
                        n_text_ctx,
                        n_vocab,
                        &self.tokenizer,
                    )
                    .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;
                    lane_opts.language = Some(detection.language);
                }

                // Load this window's audio features into prefill JIT.
                {
                    let buf = self.prefill_jit.audio_features_mut().context(JitSnafu)?;
                    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
                    let src_bytes: &[u8] = bytemuck::cast_slice(&raw[base..base + item_stride]);
                    dst[..src_bytes.len()].copy_from_slice(src_bytes);
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

                lanes.push((lane, lane_opts));
            }
        }

        // ── Batched step-locked decode ───────────────────────────────────
        // Run all lanes together: one JIT dispatch per token step, rebound
        // to the active lane count. Lanes drop out on EOT.
        let mut lane_states: Vec<DecodeLane> = lanes.into_iter().map(|(l, _)| l).collect();
        run_batched_decode(
            &mut lane_states,
            &mut self.batched_step_jit,
            &self.tokenizer,
            &batched_opts,
            n_text_ctx,
            n_vocab,
        )
        .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;

        // ── Collect transcripts ──────────────────────────────────────────
        for lane in lane_states {
            let result = lane
                .finish(&self.tokenizer, &batched_opts)
                .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;
            transcripts.push(Transcript { text: result.text, words: Vec::new(), language: result.language });
        }

        Ok(transcripts)
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

impl Transcriber for WhisperTranscriber {
    type Error = TranscribeError;

    fn sample_rate(&self) -> u32 {
        SAMPLE_RATE as u32
    }

    fn transcribe_windows(
        &mut self,
        windows: &[&[f32]],
        profile: bool,
    ) -> Result<(Vec<Transcript>, Option<RunProfile>), Self::Error> {
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

        let mut transcripts = Vec::with_capacity(windows.len());
        let mut prof = profile.then(RunProfile::default);
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
                let kernels = self.encoder_jit.execute_profiled().context(JitSnafu)?;
                eprintln!("Encoder kernels (first window):");
                for k in &kernels {
                    eprintln!("  {:>50} {:>8.3}ms", k.kernel.entry_point, k.wall.as_secs_f64() * 1000.0);
                }
                eprintln!(
                    "  encoder total: {:.3}ms",
                    kernels.iter().map(|k| k.wall.as_secs_f64() * 1000.0).sum::<f64>()
                );
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

                // Kernel-level profiling (first window only)
                if profile && batch_start == 0 && bi == 0 {
                    let kernels = self.prefill_jit.execute_profiled().context(JitSnafu)?;
                    eprintln!("Prefill kernels:");
                    for k in &kernels {
                        eprintln!("  {:>50} {:>8.3}ms", k.kernel.entry_point, k.wall.as_secs_f64() * 1000.0);
                    }
                    eprintln!(
                        "  prefill total: {:.3}ms",
                        kernels.iter().map(|k| k.wall.as_secs_f64() * 1000.0).sum::<f64>()
                    );

                    let bs = if self.options.beam_size.unwrap_or(0) > 0 { self.options.beam_size.unwrap() } else { 1 };
                    let kernels = self.step_jits.get_mut(&bs).unwrap().execute_profiled().context(JitSnafu)?;
                    eprintln!("Step JIT kernels (beam={bs}):");
                    for k in &kernels {
                        eprintln!("  {:>50} {:>8.3}ms", k.kernel.entry_point, k.wall.as_secs_f64() * 1000.0);
                    }
                    eprintln!(
                        "  step total: {:.3}ms",
                        kernels.iter().map(|k| k.wall.as_secs_f64() * 1000.0).sum::<f64>()
                    );
                }

                // Load audio features into prefill JIT
                {
                    let buf = self.prefill_jit.audio_features_mut().context(JitSnafu)?;
                    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
                    let src_bytes: &[u8] = bytemuck::cast_slice(&raw[base..base + item_stride]);
                    dst[..src_bytes.len()].copy_from_slice(src_bytes);
                }

                let result = {
                    super::decode::decode_with_fallback_cached(
                        &mut self.prefill_jit,
                        &mut self.step_jits,
                        &mut self.decoder_jit,
                        n_text_ctx,
                        n_vocab,
                        &self.tokenizer,
                        &self.options,
                        &raw[base..base + item_stride],
                        &self.pos_embedding,
                        self.n_audio_state,
                    )
                }
                .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;
                t_decode += t.elapsed();

                // DTW word-level alignment: eager decoder pass with cross-attention
                // weight extraction, then DTW on alignment heads.
                let words = if !result.tokens.is_empty() && !self.alignment_heads.is_empty() {
                    let audio_features = svod_tensor::Tensor::from_slice(&raw[base..base + item_stride])
                        .try_reshape([1usize, N_AUDIO_CTX, d])
                        .context(TensorSnafu)?;

                    let (qk_weights, sot_len) = super::decode::greedy_decode_with_alignment(
                        &self.model,
                        &audio_features,
                        &self.tokenizer,
                        &self.options,
                        &result.tokens,
                    )
                    .map_err(|e| TranscribeError::Model { source: Box::new(e) })?;

                    // Extract per-layer cross-attention weights as flat f32 vectors
                    let mut qk_flat: Vec<Vec<f32>> = Vec::with_capacity(qk_weights.len());
                    for qk in &qk_weights {
                        let mut qk_t = qk.clone();
                        qk_t.realize().context(TensorSnafu)?;
                        let arr = qk_t.as_ndarray::<f32>().context(TensorSnafu)?;
                        qk_flat.push(arr.as_slice().expect("contiguous qk").to_vec());
                    }

                    // DTW alignment
                    let s_text = result.tokens.len() + sot_len + 1; // +1 for EOT
                    let (text_indices, time_indices) = super::dtw::find_alignment_path(
                        &qk_flat,
                        1, // batch
                        self.n_text_head,
                        s_text,
                        N_AUDIO_CTX,
                        &self.alignment_heads,
                        N_AUDIO_CTX,
                        7, // medfilt_width
                        sot_len,
                    );

                    // Map to word timings
                    let (word_strings, word_token_lists) = self.tokenizer.split_to_word_tokens(&result.tokens);
                    let word_boundaries: Vec<usize> = {
                        let mut acc = 0usize;
                        let mut bounds = vec![0usize];
                        for tokens in &word_token_lists {
                            acc += tokens.len();
                            bounds.push(acc);
                        }
                        bounds
                    };

                    let token_probs: Vec<f32> = vec![0.5; result.tokens.len()];

                    super::dtw::path_to_word_timings(
                        &text_indices,
                        &time_indices,
                        &word_boundaries,
                        &word_strings,
                        &word_token_lists,
                        &token_probs,
                        super::config::TOKENS_PER_SECOND,
                    )
                    .into_iter()
                    .filter(|w| !w.word.trim().is_empty())
                    .map(|w| Word { text: w.word, start: w.start, end: w.end })
                    .collect()
                } else {
                    Vec::new()
                };

                transcripts.push(Transcript { text: result.text, words, language: result.language });
            }
        }

        if let Some(p) = &mut prof {
            p.push(StageProfile::host("mel", t_mel));
            p.push(StageProfile::host("encoder", t_encoder));
            p.push(StageProfile::host("decode", t_decode));
        }

        Ok((transcripts, prof))
    }
}
