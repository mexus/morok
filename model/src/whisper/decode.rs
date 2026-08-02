//! KV-cached Whisper decoder: greedy, beam search, temperature fallback,
//! and language detection — matching `whisper.decoding`.

use super::error::{DeviceSnafu, Error, JitSnafu, Result, TensorSnafu};
use super::jit::{WhisperDecoderJit, WhisperDecoderStepJit, WhisperPrefillJit};
use super::model::Whisper;
use super::tokenizer::WhisperTokenizer;
use snafu::ResultExt;

// ─── Language detection ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LanguageDetection {
    pub language: String,
    pub language_token: u32,
    pub probabilities: Vec<(String, f32)>,
}

pub fn detect_language(
    decoder_jit: &mut WhisperDecoderJit,
    n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
) -> Result<LanguageDetection> {
    let sot = tokenizer.sot() as i32;
    let eot = tokenizer.eot() as i32;
    let token_buf: Vec<i32> = (0..n_text_ctx).map(|i| if i == 0 { sot } else { eot }).collect();
    write_uncached(decoder_jit, &token_buf)?;
    decoder_jit.execute().context(JitSnafu)?;
    let flat = read_uncached(decoder_jit)?;
    let sot_logits = &flat[..n_vocab];

    let lang_tokens = tokenizer.all_language_tokens();
    let lang_codes = tokenizer.all_language_codes();
    let mut masked = vec![f32::NEG_INFINITY; n_vocab];
    for &tok in &lang_tokens {
        masked[tok as usize] = sot_logits[tok as usize];
    }
    let best_tok = argmax(&masked) as u32;
    let max_val = masked.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = masked.iter().map(|&l| (l - max_val).exp()).sum();
    let logsum = sum.ln() + max_val;

    let mut probabilities: Vec<(String, f32)> = lang_tokens
        .iter()
        .zip(&lang_codes)
        .map(|(&tok, code)| ((masked[tok as usize] - logsum).exp(), code.clone()))
        .map(|(p, c)| (c, p))
        .collect();
    probabilities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let language = tokenizer.code_for_token(best_tok).unwrap_or_else(|| "en".into());
    Ok(LanguageDetection { language, language_token: best_tok, probabilities })
}

// ─── Decode options & result ────────────────────────────────────────────────

#[derive(Clone)]
pub struct DecodeOptions {
    pub task: String,
    pub language: Option<String>,
    pub temperature: f32,
    pub sample_len: Option<usize>,
    pub without_timestamps: bool,
    pub suppress_blank: bool,
    pub suppress_tokens: Option<Vec<i32>>,
    pub max_initial_timestamp: Option<f32>,
    pub beam_size: Option<usize>,
    pub temperature_inc: f32,
    pub compression_ratio_threshold: Option<f32>,
    pub logprob_threshold: Option<f32>,
    pub no_speech_threshold: Option<f32>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            task: "transcribe".into(),
            language: None,
            temperature: 0.0,
            sample_len: None,
            without_timestamps: false,
            suppress_blank: true,
            suppress_tokens: Some(vec![-1]),
            max_initial_timestamp: Some(1.0),
            beam_size: Some(5),
            temperature_inc: 0.2,
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
            no_speech_threshold: Some(0.6),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    pub temperature: f32,
    pub compression_ratio: f32,
    pub language: Option<String>,
}

/// Greedy decode with cross-attention weight extraction for DTW alignment.
pub fn greedy_decode_with_alignment(
    model: &Whisper,
    audio_features: &svod_tensor::Tensor,
    tokenizer: &WhisperTokenizer,
    _options: &DecodeOptions,
    text_tokens: &[u32],
) -> Result<(Vec<svod_tensor::Tensor>, usize)> {
    let mut tokens = tokenizer.sot_sequence();
    tokens.push(tokenizer.no_timestamps());
    tokens.extend_from_slice(text_tokens);
    tokens.push(tokenizer.eot());

    let token_tensor = svod_tensor::Tensor::from_slice(tokens.iter().map(|&t| t as i32).collect::<Vec<_>>())
        .try_reshape([1usize, tokens.len()])
        .context(TensorSnafu)?
        .cast(svod_dtype::DType::Int32)
        .context(TensorSnafu)?;

    let (_, qk_weights) = model.decode_with_alignment(&token_tensor, audio_features, 0)?;
    let sot_len = tokenizer.sot_sequence().len() + 1;
    Ok((qk_weights, sot_len))
}

// ─── Decode with temperature fallback (cached) ──────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn decode_with_fallback_cached(
    prefill_jit: &mut WhisperPrefillJit,
    step_jits: &mut rustc_hash::FxHashMap<usize, WhisperDecoderStepJit>,
    decoder_jit: &mut WhisperDecoderJit,
    n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    audio_raw: &[f32],
    pos_embedding: &[f32],
    n_state: usize,
) -> Result<DecodeResult> {
    let resolved_lang = resolve_language(options, audio_raw, decoder_jit, n_text_ctx, n_vocab, tokenizer)?;

    let temperatures = build_temperature_schedule(options);
    let mut best: Option<DecodeResult> = None;

    for (t_idx, &temp) in temperatures.iter().enumerate() {
        let mut opts = options.clone();
        opts.temperature = temp;
        opts.language = resolved_lang.clone();
        if temp > 0.0 {
            opts.beam_size = None;
        }

        let result = if temp == 0.0 && opts.beam_size.unwrap_or(0) > 0 {
            let bs = opts.beam_size.unwrap();
            beam_decode_cached(
                prefill_jit,
                step_jits.get_mut(&bs).ok_or_else(|| decode_err("beam JIT missing"))?,
                n_text_ctx,
                n_vocab,
                tokenizer,
                &opts,
                bs,
                pos_embedding,
                n_state,
            )
        } else {
            greedy_decode_cached(
                prefill_jit,
                step_jits.get_mut(&1).ok_or_else(|| decode_err("greedy JIT missing"))?,
                n_text_ctx,
                n_vocab,
                tokenizer,
                &opts,
                pos_embedding,
                n_state,
            )
        }?;

        let needs_fallback = t_idx < temperatures.len() - 1 && check_fallback(&result, options);
        best = Some(result);
        if !needs_fallback {
            break;
        }
    }
    best.ok_or_else(|| decode_err("no result"))
}

fn resolve_language(
    options: &DecodeOptions,
    audio_raw: &[f32],
    decoder_jit: &mut WhisperDecoderJit,
    n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
) -> Result<Option<String>> {
    if options.language.is_some() {
        return Ok(options.language.clone());
    }
    // Load audio features into uncached decoder JIT for language detection
    {
        let buf = decoder_jit.audio_features_mut().context(JitSnafu)?;
        write_buf(buf, bytemuck::cast_slice(audio_raw))?;
    }
    let detection = detect_language(decoder_jit, n_text_ctx, n_vocab, tokenizer)?;
    Ok(Some(detection.language))
}

fn build_temperature_schedule(options: &DecodeOptions) -> Vec<f32> {
    if options.temperature_inc <= 0.0 {
        return vec![options.temperature];
    }
    let mut temps = Vec::new();
    let mut t = options.temperature;
    while t < 1.0 + 1e-6 {
        temps.push(t);
        t += options.temperature_inc;
    }
    temps
}

fn check_fallback(result: &DecodeResult, options: &DecodeOptions) -> bool {
    if let Some(th) = options.compression_ratio_threshold
        && result.compression_ratio > th
    {
        return true;
    }
    if let Some(th) = options.logprob_threshold
        && result.avg_logprob < th
    {
        if let Some(ns) = options.no_speech_threshold {
            return result.no_speech_prob <= ns;
        }
        return true;
    }
    false
}

// ─── Cached greedy decode ───────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub fn greedy_decode_cached(
    prefill_jit: &mut WhisperPrefillJit,
    step_jit: &mut WhisperDecoderStepJit,
    n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    pos_embedding: &[f32],
    n_state: usize,
) -> Result<DecodeResult> {
    let mut ctx = init_decode(prefill_jit, tokenizer, options, n_text_ctx, n_vocab, pos_embedding, n_state)?;
    ctx.write_caches_greedy(step_jit)?;

    // First token from prefill logits
    let last_logits = &ctx.prefill_logits[(ctx.init_len - 1) * n_vocab..ctx.init_len * n_vocab];
    let mut filtered = last_logits.to_vec();
    apply_logit_filters(
        &mut filtered,
        tokenizer,
        options,
        &ctx.initial_tokens,
        ctx.sample_begin,
        0,
        &ctx.suppress_tokens,
    );
    let mut next_token = pick_token(&filtered, options.temperature);
    let mut sum_logprob = log_softmax(&filtered, next_token as usize);
    let no_speech_prob = ctx.no_speech_prob;

    let mut tokens = Vec::new();
    if next_token != tokenizer.eot() {
        tokens.push(next_token);
    }

    let sample_len = options.sample_len.unwrap_or(n_text_ctx / 2);

    for step in 0..sample_len {
        if next_token == tokenizer.eot() {
            break;
        }
        let pos = ctx.init_len + step;

        write_token_buf(step_jit, &[next_token as i32])?;
        let off = pos * ctx.n_state;
        write_f32_input(step_jit.pos_emb_mut(), &ctx.pos_embedding[off..off + ctx.n_state])?;
        ctx.write_mask(step_jit, pos, n_text_ctx, 1)?;
        step_jit.execute().context(JitSnafu)?;
        ctx.copy_kv(step_jit, pos)?;

        ctx.read_logits_into(step_jit, n_vocab)?;
        let logits = &ctx.logits_buf;
        let all_toks: Vec<u32> = ctx.initial_tokens.iter().copied().chain(tokens.iter().copied()).collect();
        let mut filtered = logits.to_vec();
        apply_logit_filters(
            &mut filtered,
            tokenizer,
            options,
            &all_toks,
            ctx.sample_begin,
            step + 1,
            &ctx.suppress_tokens,
        );

        next_token = pick_token(&filtered, options.temperature);
        sum_logprob += log_softmax(&filtered, next_token as usize);

        if next_token != tokenizer.eot() {
            tokens.push(next_token);
        }
        if tokens.len() >= sample_len {
            break;
        }
    }

    finish_decode(&tokens, tokenizer, sum_logprob, no_speech_prob, options)
}

// ─── Cached beam search ─────────────────────────────────────────────────────

struct Beam {
    tokens: Vec<u32>,
    sum_logprob: f32,
}

#[allow(clippy::too_many_arguments)]
pub fn beam_decode_cached(
    prefill_jit: &mut WhisperPrefillJit,
    step_jit: &mut WhisperDecoderStepJit,
    n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    beam_size: usize,
    pos_embedding: &[f32],
    n_state: usize,
) -> Result<DecodeResult> {
    let eot = tokenizer.eot();
    let sample_len = options.sample_len.unwrap_or(n_text_ctx / 2);
    let n_audio_ctx = 1500; // N_AUDIO_CTX
    let mut ctx = init_decode(prefill_jit, tokenizer, options, n_text_ctx, n_vocab, pos_embedding, n_state)?;
    ctx.write_caches_beam(step_jit, beam_size, n_text_ctx, n_audio_ctx)?;

    // Seed beams from prefill logits
    let last_logits = &ctx.prefill_logits[(ctx.init_len - 1) * n_vocab..ctx.init_len * n_vocab];
    let mut filtered = last_logits.to_vec();
    apply_logit_filters(
        &mut filtered,
        tokenizer,
        options,
        &ctx.initial_tokens,
        ctx.sample_begin,
        0,
        &ctx.suppress_tokens,
    );
    let prefill_logprobs = log_softmax_vec(&filtered);

    let mut indexed: Vec<(usize, f32)> = prefill_logprobs.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut beams = Vec::new();
    let mut finished = Vec::new();
    for (tok_id, lp) in &indexed {
        if beams.len() >= beam_size {
            break;
        }
        let mut toks = ctx.initial_tokens.clone();
        toks.push(*tok_id as u32);
        if *tok_id as u32 == eot {
            finished.push(Beam { tokens: toks, sum_logprob: *lp });
        } else {
            beams.push(Beam { tokens: toks, sum_logprob: *lp });
        }
    }

    let per_beam_floats = ctx.per_beam_floats(n_text_ctx);
    let sample_begin = ctx.sample_begin;

    for step in 0..sample_len {
        if beams.is_empty() || finished.len() >= beam_size {
            break;
        }
        let pos = ctx.init_len + step;

        ctx.write_beam_inputs(step_jit, &beams, beam_size, pos, n_text_ctx, eot)?;
        step_jit.execute().context(JitSnafu)?;

        // Device-side K/V copy FIRST (SDMA, synchronized by subsequent logits read)
        let per_pos_bytes = ctx.per_pos_bytes();
        let cache_pos = pos * per_pos_bytes;
        for bi in 0..beam_size {
            let dst = bi * per_beam_floats * 4 + cache_pos;
            step_jit.copy_output_to_self_k_cache(1, dst, bi * per_pos_bytes, per_pos_bytes).context(JitSnafu)?;
            step_jit.copy_output_to_self_v_cache(2, dst, bi * per_pos_bytes, per_pos_bytes).context(JitSnafu)?;
        }

        let suppress_tokens: Vec<i32> = ctx.suppress_tokens.clone();
        ctx.read_logits_into(step_jit, n_vocab * beam_size)?;
        let all_logits = &ctx.logits_buf;

        // Generate + rank candidates
        let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
        for (bi, beam) in beams.iter().enumerate() {
            let start = bi * n_vocab;
            if start >= all_logits.len() {
                continue;
            }
            let beam_logits = &all_logits[start..(start + n_vocab).min(all_logits.len())];
            let mut filtered = beam_logits.to_vec();
            apply_logit_filters(
                &mut filtered,
                tokenizer,
                options,
                &beam.tokens,
                sample_begin,
                step + 1,
                &suppress_tokens,
            );
            let logprobs = log_softmax_vec(&filtered);
            let mut idx: Vec<(usize, f32)> = logprobs.iter().copied().enumerate().collect();
            idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (tok_id, lp) in idx.into_iter().take(beam_size + 1) {
                candidates.push((bi, tok_id, beam.sum_logprob + lp));
            }
        }
        candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        // Select survivors
        let mut new_beams = Vec::with_capacity(beam_size);
        let mut parent_map = Vec::with_capacity(beam_size);
        for (parent_idx, tok_id, sum_lp) in candidates {
            if new_beams.len() >= beam_size {
                break;
            }
            let mut toks = beams[parent_idx].tokens.clone();
            toks.push(tok_id as u32);
            if tok_id as u32 == eot {
                if finished.len() < beam_size {
                    finished.push(Beam { tokens: toks, sum_logprob: sum_lp });
                }
            } else {
                new_beams.push(Beam { tokens: toks, sum_logprob: sum_lp });
                parent_map.push(parent_idx);
            }
            if new_beams.len() >= beam_size {
                break;
            }
        }

        // Reorder cache by parent_map (host-side, copies ENTIRE cache incl new K/V)
        if !parent_map.is_empty() {
            reorder_cache_host(step_jit.self_k_cache_mut(), &parent_map, per_beam_floats)?;
            reorder_cache_host(step_jit.self_v_cache_mut(), &parent_map, per_beam_floats)?;
        }

        beams = new_beams;

        if beams.iter().all(|b| b.tokens.len() >= n_text_ctx) {
            for b in beams.drain(..) {
                let mut t = b.tokens;
                if *t.last().unwrap() != eot {
                    t.push(eot);
                }
                finished.push(Beam { tokens: t, sum_logprob: b.sum_logprob });
            }
            break;
        }
    }

    // Backfill + rank
    for b in beams {
        if finished.len() >= beam_size {
            break;
        }
        let mut t = b.tokens;
        if *t.last().unwrap() != eot {
            t.push(eot);
        }
        finished.push(Beam { tokens: t, sum_logprob: b.sum_logprob });
    }

    finished.sort_by(|a, b| {
        let la = a.sum_logprob / a.tokens.len().saturating_sub(ctx.sample_begin + 1).max(1) as f32;
        let lb = b.sum_logprob / b.tokens.len().saturating_sub(ctx.sample_begin + 1).max(1) as f32;
        lb.partial_cmp(&la).unwrap_or(std::cmp::Ordering::Equal)
    });

    let best = finished.into_iter().next().ok_or_else(|| decode_err("beam produced nothing"))?;
    let output_tokens: Vec<u32> = best.tokens[ctx.sample_begin..].iter().copied().take_while(|&t| t != eot).collect();
    let text = tokenizer.decode(&output_tokens);
    let avg_logprob = best.sum_logprob / (output_tokens.len() + 1) as f32;
    let compression_ratio = compression_ratio_text(&text);
    Ok(DecodeResult {
        tokens: output_tokens,
        text,
        avg_logprob,
        no_speech_prob: ctx.no_speech_prob,
        temperature: options.temperature,
        compression_ratio,
        language: options.language.clone(),
    })
}

// ─── Shared decode context ──────────────────────────────────────────────────

struct DecodeCtx {
    initial_tokens: Vec<u32>,
    sample_begin: usize,
    init_len: usize,
    suppress_tokens: Vec<i32>,
    prefill_logits: Vec<f32>,
    no_speech_prob: f32,
    pos_embedding: Vec<f32>,
    n_state: usize,
    self_k_cache: Vec<f32>,
    self_v_cache: Vec<f32>,
    cross_k: Vec<f32>,
    cross_v: Vec<f32>,
    // Reusable scratch buffers (avoid per-step allocation)
    logits_buf: Vec<f32>,
    mask_buf: Vec<f32>,
}

impl DecodeCtx {
    fn write_caches_greedy(&self, step_jit: &mut WhisperDecoderStepJit) -> Result<()> {
        write_f32_input(step_jit.self_k_cache_mut(), &self.self_k_cache)?;
        write_f32_input(step_jit.self_v_cache_mut(), &self.self_v_cache)?;
        write_f32_input(step_jit.cross_k_mut(), &self.cross_k)?;
        write_f32_input(step_jit.cross_v_mut(), &self.cross_v)?;
        Ok(())
    }

    fn write_caches_beam(
        &self,
        step_jit: &mut WhisperDecoderStepJit,
        beam_size: usize,
        n_text_ctx: usize,
        n_audio_ctx: usize,
    ) -> Result<()> {
        let layer_heads_dh = self.self_k_cache.len() / self.init_len;
        let self_stride = n_text_ctx * layer_heads_dh;
        let cross_stride = n_audio_ctx * layer_heads_dh;
        write_replicated_cache(step_jit.self_k_cache_mut(), &self.self_k_cache, beam_size, self_stride)?;
        write_replicated_cache(step_jit.self_v_cache_mut(), &self.self_v_cache, beam_size, self_stride)?;
        write_replicated_cache(step_jit.cross_k_mut(), &self.cross_k, beam_size, cross_stride)?;
        write_replicated_cache(step_jit.cross_v_mut(), &self.cross_v, beam_size, cross_stride)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_beam_inputs(
        &self,
        step_jit: &mut WhisperDecoderStepJit,
        beams: &[Beam],
        beam_size: usize,
        pos: usize,
        n_text_ctx: usize,
        eot: u32,
    ) -> Result<()> {
        // Tokens
        let mut tokens = vec![eot as i32; beam_size];
        for (bi, beam) in beams.iter().enumerate() {
            tokens[bi] = *beam.tokens.last().unwrap() as i32;
        }
        write_token_buf(step_jit, &tokens)?;
        // Broadcast pos_emb [n_state] to [beam_size, 1, n_state]
        {
            let off = pos * self.n_state;
            let emb = &self.pos_embedding[off..off + self.n_state];
            let mut packed = vec![0f32; beam_size * self.n_state];
            for bi in 0..beam_size {
                packed[bi * self.n_state..(bi + 1) * self.n_state].copy_from_slice(emb);
            }
            write_f32_input(step_jit.pos_emb_mut(), &packed)?;
        }
        write_self_mask(step_jit, pos, n_text_ctx, beam_size)
    }

    fn copy_kv(&self, step_jit: &mut WhisperDecoderStepJit, pos: usize) -> Result<()> {
        let b = self.per_pos_bytes();
        step_jit.copy_output_to_self_k_cache(1, pos * b, 0, b).context(JitSnafu)?;
        step_jit.copy_output_to_self_v_cache(2, pos * b, 0, b).context(JitSnafu)
    }

    fn per_beam_floats(&self, n_text_ctx: usize) -> usize {
        n_text_ctx * self.per_pos_floats()
    }

    fn per_pos_floats(&self) -> usize {
        self.self_k_cache.len() / self.init_len
    }

    fn per_pos_bytes(&self) -> usize {
        self.per_pos_floats() * std::mem::size_of::<f32>()
    }

    /// Read step logits into the reusable buffer, returns &mut [f32].
    fn read_logits_into(&mut self, step_jit: &mut WhisperDecoderStepJit, n_vocab: usize) -> Result<()> {
        let buf = step_jit.logits().context(JitSnafu)?;
        let src = buf.as_host_bytes().context(DeviceSnafu)?;
        let available = src.len() / std::mem::size_of::<f32>();
        let n = n_vocab.min(available);
        self.logits_buf.clear();
        self.logits_buf.extend_from_slice(bytemuck::cast_slice(&src[..n * std::mem::size_of::<f32>()]));
        Ok(())
    }

    /// Write the causal mask for the current position into the step JIT.
    /// Uses mask_buf to avoid allocating per step.
    fn write_mask(
        &mut self,
        step_jit: &mut WhisperDecoderStepJit,
        pos: usize,
        n_text_ctx: usize,
        batch: usize,
    ) -> Result<()> {
        let needed = (n_text_ctx + 1) * batch;
        if self.mask_buf.len() < needed {
            self.mask_buf.resize(needed, f32::NEG_INFINITY);
        }
        for bi in 0..batch {
            let off = bi * (n_text_ctx + 1);
            // Fill [0..pos) with 0.0, [pos..n_text_ctx) with -inf, [n_text_ctx] with 0.0
            for i in 0..pos {
                self.mask_buf[off + i] = 0.0;
            }
            for i in pos..n_text_ctx {
                self.mask_buf[off + i] = f32::NEG_INFINITY;
            }
            self.mask_buf[off + n_text_ctx] = 0.0;
        }
        let buf = step_jit.self_mask_mut().context(JitSnafu)?;
        let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
        let src_bytes: &[u8] = bytemuck::cast_slice(&self.mask_buf[..needed]);
        dst[..src_bytes.len()].copy_from_slice(src_bytes);
        Ok(())
    }
}

fn write_token_buf(step_jit: &mut WhisperDecoderStepJit, tokens: &[i32]) -> Result<()> {
    let buf = step_jit.token_mut().context(JitSnafu)?;
    write_buf(buf, bytemuck::cast_slice(tokens))?;
    Ok(())
}

fn init_decode(
    prefill_jit: &mut WhisperPrefillJit,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_text_ctx: usize,
    n_vocab: usize,
    pos_embedding: &[f32],
    n_state: usize,
) -> Result<DecodeCtx> {
    // Build initial tokens
    let lang = options.language.as_ref().ok_or_else(|| decode_err("language required"))?;
    let lang_tok = tokenizer.language_token_for(lang).unwrap_or_else(|| tokenizer.sot());
    let task_tok = if options.task == "transcribe" { tokenizer.transcribe() } else { tokenizer.translate() };
    let mut initial_tokens = vec![tokenizer.sot(), lang_tok, task_tok];
    if options.without_timestamps {
        initial_tokens.push(tokenizer.no_timestamps());
    }
    let sample_begin = initial_tokens.len();
    let init_len = initial_tokens.len();
    let suppress_tokens = get_suppress_tokens(tokenizer, options);

    // Write tokens to prefill JIT buffer
    {
        let token_data: Vec<i32> = initial_tokens.iter().map(|&t| t as i32).collect();
        let buf = prefill_jit.tokens_mut().context(JitSnafu)?;
        write_buf(buf, bytemuck::cast_slice(&token_data))?;
    }

    // Execute prefill JIT (plan manages all buffers, no realize)
    prefill_jit.execute().context(JitSnafu)?;

    // Read logits from output 0
    let prefill_logits = {
        let buf = prefill_jit.logits().context(JitSnafu)?;
        read_buf(buf, buf.size() / std::mem::size_of::<f32>())?
    };
    let no_speech_prob = tokenizer
        .no_speech()
        .map(|ns| softmax_prob(&prefill_logits[..n_vocab.min(prefill_logits.len())], ns as usize))
        .unwrap_or(f32::NAN);

    // Read packed K/V from outputs 1-4 via copyout (synchronizes device)
    let read_output = |jit: &WhisperPrefillJit,
                       accessor: fn(&WhisperPrefillJit) -> crate::jit::Result<&svod_device::Buffer>|
     -> Result<Vec<f32>> {
        let buf = accessor(jit).context(JitSnafu)?;
        read_buf(buf, buf.size() / std::mem::size_of::<f32>())
    };

    // Infer 4D shape from total element count: [1, seq, n_layer*H, Dh]
    // seq = init_len for self, 1500 for cross. d_head = n_state / n_head.
    // total = seq * n_layer * n_head * d_head. So layer_heads_dh = total / seq.
    let self_k_cache = read_output(prefill_jit, WhisperPrefillJit::self_k)?;
    let self_v_cache = read_output(prefill_jit, WhisperPrefillJit::self_v)?;
    let cross_k = read_output(prefill_jit, WhisperPrefillJit::cross_k)?;
    let cross_v = read_output(prefill_jit, WhisperPrefillJit::cross_v)?;

    let _ = n_text_ctx;

    Ok(DecodeCtx {
        initial_tokens,
        sample_begin,
        init_len,
        suppress_tokens,
        prefill_logits,
        no_speech_prob,
        pos_embedding: pos_embedding.to_vec(),
        n_state,
        self_k_cache,
        self_v_cache,
        cross_k,
        cross_v,
        logits_buf: Vec::new(),
        mask_buf: Vec::new(),
    })
}

// ─── Result helpers ─────────────────────────────────────────────────────────

fn finish_decode(
    tokens: &[u32],
    tokenizer: &WhisperTokenizer,
    sum_logprob: f32,
    no_speech_prob: f32,
    options: &DecodeOptions,
) -> Result<DecodeResult> {
    let text = tokenizer.decode(tokens);
    let avg_logprob = sum_logprob / (tokens.len() + 1) as f32;
    let compression_ratio = compression_ratio_text(&text);
    Ok(DecodeResult {
        tokens: tokens.to_vec(),
        text,
        avg_logprob,
        no_speech_prob,
        temperature: options.temperature,
        compression_ratio,
        language: options.language.clone(),
    })
}

fn pick_token(logits: &[f32], temperature: f32) -> u32 {
    if temperature > 0.0 { sample_from_logits(logits, temperature) } else { argmax(logits) as u32 }
}

fn decode_err(msg: &str) -> Error {
    Error::Decode { msg: msg.into() }
}

// ─── JIT buffer helpers ─────────────────────────────────────────────────────

fn write_uncached(jit: &mut WhisperDecoderJit, tokens: &[i32]) -> Result<()> {
    let buf = jit.tokens_mut().context(JitSnafu)?;
    write_buf(buf, bytemuck::cast_slice(tokens))
}

fn read_uncached(jit: &WhisperDecoderJit) -> Result<Vec<f32>> {
    let buf = jit.output().context(JitSnafu)?;
    read_buf(buf, buf.size() / std::mem::size_of::<f32>())
}

fn write_self_mask(step_jit: &mut WhisperDecoderStepJit, pos: usize, n_text_ctx: usize, batch: usize) -> Result<()> {
    let mut mask = vec![f32::NEG_INFINITY; (n_text_ctx + 1) * batch];
    for bi in 0..batch {
        let off = bi * (n_text_ctx + 1);
        mask[off..off + pos].fill(0.0);
        mask[off + n_text_ctx] = 0.0;
    }
    let buf = step_jit.self_mask_mut().context(JitSnafu)?;
    write_buf(buf, bytemuck::cast_slice(&mask))?;
    Ok(())
}

/// Write data directly into the buffer's host-visible mapping.
/// `as_host_bytes_mut` syncs pending GPU work before returning the slice.
/// Subsequent `execute()` sees our writes (unified memory / BAR).
fn write_buf(buf: &svod_device::Buffer, data: &[u8]) -> Result<()> {
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    let n = data.len().min(dst.len());
    dst[..n].copy_from_slice(&data[..n]);
    Ok(())
}

/// Read data directly from the buffer's host-visible mapping.
/// `as_host_bytes` syncs pending GPU work before returning the slice.
fn read_buf(buf: &svod_device::Buffer, n: usize) -> Result<Vec<f32>> {
    let src = buf.as_host_bytes().context(DeviceSnafu)?;
    let n = n.min(src.len() / std::mem::size_of::<f32>());
    Ok(bytemuck::cast_slice(&src[..n * std::mem::size_of::<f32>()]).to_vec())
}

fn write_f32_input(buf_result: crate::jit::Result<&mut svod_device::Buffer>, data: &[f32]) -> Result<()> {
    let buf = buf_result.context(JitSnafu)?;
    write_buf(buf, bytemuck::cast_slice(data))
}

fn write_replicated_cache(
    buf_result: crate::jit::Result<&mut svod_device::Buffer>,
    single_data: &[f32],
    beam_size: usize,
    per_beam_floats: usize,
) -> Result<()> {
    let buf = buf_result.context(JitSnafu)?;
    let n = single_data.len().min(per_beam_floats);
    let total = per_beam_floats * beam_size;
    let mut packed = vec![0f32; total];
    for bi in 0..beam_size {
        packed[bi * per_beam_floats..bi * per_beam_floats + n].copy_from_slice(&single_data[..n]);
    }
    write_buf(buf, bytemuck::cast_slice(&packed))
}

fn reorder_cache_host(
    buf_result: crate::jit::Result<&mut svod_device::Buffer>,
    parent_map: &[usize],
    per_beam_floats: usize,
) -> Result<()> {
    let buf = buf_result.context(JitSnafu)?;
    let total = per_beam_floats * parent_map.len();
    let current = read_buf(buf, total)?;

    let mut reordered = vec![0f32; total];
    for (new_idx, &parent_idx) in parent_map.iter().enumerate() {
        let (src, dst) = (parent_idx * per_beam_floats, new_idx * per_beam_floats);
        reordered[dst..dst + per_beam_floats].copy_from_slice(&current[src..src + per_beam_floats]);
    }

    write_buf(buf, bytemuck::cast_slice(&reordered))
}

// ─── Logit filter helpers ───────────────────────────────────────────────────

fn get_suppress_tokens(tokenizer: &WhisperTokenizer, options: &DecodeOptions) -> Vec<i32> {
    let mut tokens: Vec<i32> = options.suppress_tokens.clone().unwrap_or_default();
    if tokens.contains(&-1) {
        tokens.retain(|&t| t >= 0);
        for &t in &tokenizer.non_speech_tokens() {
            tokens.push(t as i32);
        }
    }
    tokens.extend([
        tokenizer.transcribe() as i32,
        tokenizer.translate() as i32,
        tokenizer.sot() as i32,
        tokenizer.sot_prev() as i32,
        tokenizer.sot_lm() as i32,
    ]);
    if let Some(ns) = tokenizer.no_speech() {
        tokens.push(ns as i32);
    }
    tokens.sort();
    tokens.dedup();
    tokens
}

fn apply_logit_filters(
    logits: &mut [f32],
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    tokens: &[u32],
    sample_begin: usize,
    step: usize,
    suppress_tokens: &[i32],
) {
    let eot = tokenizer.eot() as usize;
    if options.suppress_blank && step == 0 {
        for &t in &tokenizer.encode(" ") {
            if (t as usize) < logits.len() {
                logits[t as usize] = f32::NEG_INFINITY;
            }
        }
        if eot < logits.len() {
            logits[eot] = f32::NEG_INFINITY;
        }
    }
    for &t in suppress_tokens {
        if t >= 0 && (t as usize) < logits.len() {
            logits[t as usize] = f32::NEG_INFINITY;
        }
    }
    let specials =
        [tokenizer.transcribe(), tokenizer.translate(), tokenizer.sot(), tokenizer.sot_prev(), tokenizer.sot_lm()];
    for &t in &specials {
        if (t as usize) < logits.len() {
            logits[t as usize] = f32::NEG_INFINITY;
        }
    }
    if let Some(ns) = tokenizer.no_speech()
        && (ns as usize) < logits.len()
    {
        logits[ns as usize] = f32::NEG_INFINITY;
    }
    if !options.without_timestamps {
        apply_timestamp_rules(logits, tokenizer, tokens, sample_begin, options);
    } else {
        let ts_begin = tokenizer.timestamp_begin() as usize;
        if ts_begin < logits.len() {
            for t in &mut logits[ts_begin..] {
                *t = f32::NEG_INFINITY;
            }
        }
    }
}

fn apply_timestamp_rules(
    logits: &mut [f32],
    tokenizer: &WhisperTokenizer,
    tokens: &[u32],
    sample_begin: usize,
    options: &DecodeOptions,
) {
    let ts_begin = tokenizer.timestamp_begin() as usize;
    let eot = tokenizer.eot() as usize;
    let no_ts = tokenizer.no_timestamps() as usize;
    if no_ts < logits.len() {
        logits[no_ts] = f32::NEG_INFINITY;
    }

    let sampled = &tokens[sample_begin.min(tokens.len())..];
    let last_was_ts = sampled.last().map(|&t| (t as usize) >= ts_begin).unwrap_or(false);
    let penultimate_was_ts = sampled.len() < 2 || (sampled[sampled.len() - 2] as usize) >= ts_begin;

    if last_was_ts {
        if penultimate_was_ts {
            for t in &mut logits[ts_begin..] {
                *t = f32::NEG_INFINITY;
            }
        } else {
            for t in &mut logits[..eot] {
                *t = f32::NEG_INFINITY;
            }
        }
    }

    let ts_tokens: Vec<u32> = sampled.iter().filter(|&&t| (t as usize) >= ts_begin).copied().collect();
    if !ts_tokens.is_empty() {
        let last_ts = if last_was_ts && !penultimate_was_ts {
            ts_tokens.last().copied().unwrap_or(0) as usize
        } else {
            ts_tokens.last().copied().unwrap_or(0) as usize + 1
        };
        for (i, t) in logits[ts_begin..].iter_mut().enumerate() {
            if ts_begin + i < last_ts {
                *t = f32::NEG_INFINITY;
            }
        }
    }

    if tokens.len() == sample_begin {
        for t in &mut logits[..ts_begin] {
            *t = f32::NEG_INFINITY;
        }
        if let Some(max_init) = options.max_initial_timestamp {
            let last_allowed = ts_begin + (max_init / 0.02).round() as usize;
            if last_allowed + 1 < logits.len() {
                for t in &mut logits[last_allowed + 1..] {
                    *t = f32::NEG_INFINITY;
                }
            }
        }
    }

    let ts_logprob = logsumexp(&logits[ts_begin..]);
    let text_max = logits[..ts_begin].iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if ts_logprob > text_max {
        for t in &mut logits[..eot] {
            *t = f32::NEG_INFINITY;
        }
    }
}

// ─── Math helpers ───────────────────────────────────────────────────────────

fn argmax(arr: &[f32]) -> usize {
    arr.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn softmax_prob(logits: &[f32], idx: usize) -> f32 {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&l| (l - max_val).exp()).sum();
    if idx < logits.len() { (logits[idx] - max_val).exp() / sum.max(1e-10) } else { 0.0 }
}

fn log_softmax(logits: &[f32], idx: usize) -> f32 {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&l| (l - max_val).exp()).sum();
    let logsum = sum.ln() + max_val;
    if idx < logits.len() { logits[idx] - logsum } else { f32::NEG_INFINITY }
}

fn log_softmax_vec(logits: &[f32]) -> Vec<f32> {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f32 = logits.iter().map(|&l| (l - max_val).exp()).sum();
    let logsum = sum.ln() + max_val;
    logits.iter().map(|&l| l - logsum).collect()
}

fn logsumexp(arr: &[f32]) -> f32 {
    let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if max_val == f32::NEG_INFINITY {
        return f32::NEG_INFINITY;
    }
    (arr.iter().map(|&l| (l - max_val).exp()).sum::<f32>()).ln() + max_val
}

fn compression_ratio_text(text: &str) -> f32 {
    let raw = text.as_bytes();
    if raw.is_empty() {
        return 1.0;
    }
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    let _ = encoder.write_all(raw);
    let clen = encoder.finish().unwrap_or_default().len().max(1);
    raw.len() as f32 / clen as f32
}

fn sample_from_logits(logits: &[f32], temperature: f32) -> u32 {
    let scaled: Vec<f32> = logits.iter().map(|&l| (l / temperature).exp()).collect();
    let sum: f32 = scaled.iter().sum();
    let mut r = rand::random::<f32>() * sum;
    for (i, &p) in scaled.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (scaled.len() - 1) as u32
}
