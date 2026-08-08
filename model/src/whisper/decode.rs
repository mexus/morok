//! KV-cached Whisper decoder: greedy, beam search, temperature fallback,
//! and language detection — matching `whisper.decoding`.

use super::config::N_AUDIO_CTX;
use super::error::{DeviceSnafu, Error, JitSnafu, Result};
use super::jit::{WhisperDecoderJit, WhisperDecoderStepJit, WhisperPrefillJit};
use super::tokenizer::WhisperTokenizer;
use snafu::ResultExt;
use std::cmp::Ordering;
use std::collections::VecDeque;
use svod_arch::pipelines::audio::Segment;

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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WhisperTask {
    Transcribe,
    Translate,
}

impl std::str::FromStr for WhisperTask {
    type Err = &'static str;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        match value {
            "transcribe" => Ok(Self::Transcribe),
            "translate" => Ok(Self::Translate),
            _ => Err("expected `transcribe` or `translate`"),
        }
    }
}

/// Search algorithm for the first decode attempt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DecodeStrategy {
    /// Deterministic token-by-token argmax.
    Greedy,
    /// Beam search with a concrete number of decoder rows.
    Beam { size: usize },
    /// Multinomial sampling at a positive temperature.
    Sample { temperature: f32 },
}

impl DecodeStrategy {
    fn temperature(self) -> f32 {
        match self {
            Self::Greedy | Self::Beam { .. } => 0.0,
            Self::Sample { temperature } => temperature,
        }
    }
}

/// Quality-gated sampling attempts after the primary decode is rejected.
#[derive(Clone, Debug, PartialEq)]
pub struct FallbackPolicy {
    /// Positive sampling temperatures tried in order.
    pub sampling_temperatures: Vec<f32>,
    /// Retry when text compression exceeds this threshold.
    pub compression_ratio_threshold: Option<f32>,
    /// Retry below this average log-probability.
    pub logprob_threshold: Option<f32>,
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self {
            sampling_temperatures: vec![0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold: Some(2.4),
            logprob_threshold: Some(-1.0),
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecodeOptions {
    /// Whether to transcribe source speech or translate it to English.
    pub task: WhisperTask,
    /// Source language code, or `None` for automatic detection.
    pub language: Option<String>,
    /// Search algorithm for the first decode attempt.
    pub strategy: DecodeStrategy,
    /// Optional quality-gated sampling retries.
    pub fallback: Option<FallbackPolicy>,
    /// Maximum generated token count; defaults to half the text context.
    pub sample_len: Option<usize>,
    /// Suppress blank/space as the first generated token.
    pub suppress_blank: bool,
    /// Token IDs to suppress; `-1` expands to Whisper's non-speech set.
    pub suppress_tokens: Option<Vec<i32>>,
    /// Latest timestamp permitted at the beginning of a window, in seconds.
    pub max_initial_timestamp: Option<f32>,
    /// Skip likely silence when no-speech probability exceeds this threshold.
    pub no_speech_threshold: Option<f32>,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        Self {
            task: WhisperTask::Transcribe,
            language: None,
            strategy: DecodeStrategy::Beam { size: 5 },
            fallback: Some(FallbackPolicy::default()),
            sample_len: None,
            suppress_blank: true,
            suppress_tokens: Some(vec![-1]),
            max_initial_timestamp: Some(1.0),
            no_speech_threshold: Some(0.6),
        }
    }
}

impl DecodeOptions {
    /// Validate strategy geometry and sampling parameters before graph preparation.
    pub fn validate(&self) -> std::result::Result<(), &'static str> {
        match self.strategy {
            DecodeStrategy::Beam { size: 0 } => return Err("beam size must be non-zero"),
            DecodeStrategy::Sample { temperature } if !valid_temperature(temperature) => {
                return Err("sampling temperature must be finite and positive");
            }
            _ => {}
        }
        if let Some(fallback) = &self.fallback {
            if fallback.sampling_temperatures.is_empty() {
                return Err("fallback sampling temperatures must be non-empty");
            }
            if fallback.sampling_temperatures.iter().any(|&temperature| !valid_temperature(temperature)) {
                return Err("fallback sampling temperatures must be finite and positive");
            }
            if fallback.compression_ratio_threshold.is_some_and(|threshold| !threshold.is_finite() || threshold <= 0.0)
            {
                return Err("compression ratio threshold must be finite and positive");
            }
            if fallback.logprob_threshold.is_some_and(|threshold| !threshold.is_finite()) {
                return Err("log-probability threshold must be finite");
            }
        }
        if self.no_speech_threshold.is_some_and(|threshold| !threshold.is_finite() || !(0.0..=1.0).contains(&threshold))
        {
            return Err("no-speech threshold must be between zero and one");
        }
        Ok(())
    }
}

fn valid_temperature(temperature: f32) -> bool {
    temperature.is_finite() && temperature > 0.0
}

pub(crate) fn remaining_sample_steps(sample_len: usize) -> usize {
    sample_len.saturating_sub(1)
}

#[derive(Clone, Debug)]
pub struct DecodeResult {
    pub tokens: Vec<u32>,
    pub token_probs: Vec<f32>,
    pub text: String,
    pub avg_logprob: f32,
    pub no_speech_prob: f32,
    /// Sampling temperature of the accepted attempt; zero for greedy or beam.
    pub temperature: f32,
    pub compression_ratio: f32,
    pub language: Option<String>,
}

impl DecodeResult {
    pub fn should_skip(&self, options: &DecodeOptions) -> bool {
        let Some(no_speech_threshold) = options.no_speech_threshold else {
            return false;
        };
        if self.no_speech_prob <= no_speech_threshold {
            return false;
        }
        options
            .fallback
            .as_ref()
            .and_then(|fallback| fallback.logprob_threshold)
            .is_none_or(|threshold| self.avg_logprob <= threshold)
    }

    pub fn clear_speech(&mut self) {
        self.tokens.clear();
        self.token_probs.clear();
        self.text.clear();
    }
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
    pos_embedding: &[f32],
    n_state: usize,
) -> Result<DecodeResult> {
    options.validate().map_err(decode_err)?;
    let resolved_lang = resolve_language(options, decoder_jit, n_text_ctx, n_vocab, tokenizer)?;

    let mut strategies = vec![options.strategy];
    if let Some(fallback) = &options.fallback {
        strategies
            .extend(fallback.sampling_temperatures.iter().map(|&temperature| DecodeStrategy::Sample { temperature }));
    }
    let mut best: Option<DecodeResult> = None;

    for (attempt, &strategy) in strategies.iter().enumerate() {
        let mut opts = options.clone();
        opts.strategy = strategy;
        opts.fallback = None;
        opts.language = resolved_lang.clone();

        let result = match strategy {
            DecodeStrategy::Beam { size } => beam_decode_cached(
                prefill_jit,
                step_jits.get_mut(&size).ok_or_else(|| decode_err("beam JIT missing"))?,
                n_text_ctx,
                n_vocab,
                tokenizer,
                &opts,
                size,
                pos_embedding,
                n_state,
            )?,
            DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => greedy_decode_cached(
                prefill_jit,
                step_jits.get_mut(&1).ok_or_else(|| decode_err("greedy JIT missing"))?,
                n_text_ctx,
                n_vocab,
                tokenizer,
                &opts,
                pos_embedding,
                n_state,
            )?,
        };

        let needs_fallback = attempt < strategies.len() - 1
            && options.fallback.as_ref().is_some_and(|fallback| check_fallback(&result, fallback, options));
        best = Some(result);
        if !needs_fallback {
            break;
        }
    }
    best.ok_or_else(|| decode_err("no result"))
}

fn resolve_language(
    options: &DecodeOptions,
    decoder_jit: &mut WhisperDecoderJit,
    n_text_ctx: usize,
    n_vocab: usize,
    tokenizer: &WhisperTokenizer,
) -> Result<Option<String>> {
    if !tokenizer.multilingual {
        return Ok(Some("en".to_string()));
    }
    if options.language.is_some() {
        return Ok(options.language.clone());
    }
    let detection = detect_language(decoder_jit, n_text_ctx, n_vocab, tokenizer)?;
    Ok(Some(detection.language))
}

pub(crate) fn check_fallback(result: &DecodeResult, fallback: &FallbackPolicy, options: &DecodeOptions) -> bool {
    let repetitive = fallback.compression_ratio_threshold.is_some_and(|threshold| result.compression_ratio > threshold);
    let low_confidence = fallback.logprob_threshold.is_some_and(|threshold| result.avg_logprob < threshold);
    let silence =
        options.no_speech_threshold.is_some_and(|threshold| result.no_speech_prob > threshold) && low_confidence;
    (repetitive || low_confidence) && !silence
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
    let mut next_token = pick_token(&filtered, options.strategy.temperature());
    let mut sum_logprob = log_softmax(&filtered, next_token as usize);
    let no_speech_prob = ctx.no_speech_prob;

    let sample_len = options.sample_len.unwrap_or(n_text_ctx / 2);
    if sample_len == 0 {
        return finish_decode(&[], &[], tokenizer, 0.0, no_speech_prob, options);
    }

    let mut tokens = Vec::new();
    let mut token_probs = Vec::new();
    if next_token != tokenizer.eot() {
        tokens.push(next_token);
        token_probs.push(sum_logprob.exp());
    }

    // Prefill produced generated token 1, so only the remaining budget needs
    // decoder-step dispatches.
    for step in 0..remaining_sample_steps(sample_len) {
        if next_token == tokenizer.eot() {
            break;
        }
        let pos = ctx.init_len + step;

        write_token_buf(step_jit, &[next_token as i32])?;
        let off = pos * ctx.n_state;
        write_f32_input(step_jit.pos_emb_mut().context(JitSnafu)?, &ctx.pos_embedding[off..off + ctx.n_state])?;
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

        next_token = pick_token(&filtered, options.strategy.temperature());
        let token_logprob = log_softmax(&filtered, next_token as usize);
        sum_logprob += token_logprob;

        if next_token != tokenizer.eot() {
            tokens.push(next_token);
            token_probs.push(token_logprob.exp());
        }
        if tokens.len() >= sample_len {
            break;
        }
    }

    finish_decode(&tokens, &token_probs, tokenizer, sum_logprob, no_speech_prob, options)
}

// ─── Fixed-slot mixed-strategy scheduler ────────────────────────────────────

/// Immutable output of token prefill. Fallback attempts reuse this seed rather
/// than rerunning prefill. Cache vectors are currently host-resident and are
/// copied into a row only when that row changes request ownership.
pub(crate) struct DecodeSeed {
    ctx: DecodeCtx,
}

pub(crate) fn prefill_decode_seed(
    prefill_jit: &mut WhisperPrefillJit,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_text_ctx: usize,
    n_vocab: usize,
    pos_embedding: &[f32],
    n_state: usize,
) -> Result<DecodeSeed> {
    Ok(DecodeSeed { ctx: init_decode(prefill_jit, tokenizer, options, n_text_ctx, n_vocab, pos_embedding, n_state)? })
}

pub(crate) fn strategy_width(strategy: DecodeStrategy) -> usize {
    match strategy {
        DecodeStrategy::Beam { size } => size,
        DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => 1,
    }
}

pub(crate) fn attempt_strategies(options: &DecodeOptions) -> Vec<DecodeStrategy> {
    let mut strategies = vec![options.strategy];
    if let Some(fallback) = &options.fallback {
        strategies
            .extend(fallback.sampling_temperatures.iter().map(|&temperature| DecodeStrategy::Sample { temperature }));
    }
    strategies
}

pub(crate) fn collect_ordered<T>(results: Vec<Option<T>>) -> std::result::Result<Vec<T>, &'static str> {
    results.into_iter().map(|result| result.ok_or("missing scheduled result")).collect()
}

/// Small independently-testable allocator enforcing whole-attempt admission.
#[derive(Debug)]
pub(crate) struct SlotAllocator {
    owners: Vec<Option<usize>>,
}

impl SlotAllocator {
    pub(crate) fn new(capacity: usize) -> Self {
        Self { owners: vec![None; capacity] }
    }

    pub(crate) fn reserve(
        &mut self,
        owner: usize,
        width: usize,
    ) -> std::result::Result<Option<Vec<usize>>, &'static str> {
        if width == 0 {
            return Err("attempt width must be non-zero");
        }
        if width > self.owners.len() {
            return Err("decode attempt width exceeds decoder slots");
        }
        if self.owners.iter().filter(|slot| slot.is_none()).count() < width {
            return Ok(None);
        }
        let rows: Vec<_> = self
            .owners
            .iter()
            .enumerate()
            .filter_map(|(row, current)| current.is_none().then_some(row))
            .take(width)
            .collect();
        for &row in &rows {
            self.owners[row] = Some(owner);
        }
        Ok(Some(rows))
    }

    pub(crate) fn release(&mut self, owner: usize) {
        for slot in &mut self.owners {
            if *slot == Some(owner) {
                *slot = None;
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn owners(&self) -> &[Option<usize>] {
        &self.owners
    }
}

struct SingleAttempt {
    next_token: u32,
    tokens: Vec<u32>,
    token_probs: Vec<f32>,
    sum_logprob: f32,
}

struct BeamAttempt {
    active: Vec<BeamHypothesis>,
    rows: Vec<usize>,
    finished: Vec<BeamHypothesis>,
    next_logical_id: usize,
}

enum AttemptKind {
    Single(SingleAttempt),
    Beam(BeamAttempt),
}

struct ScheduledAttempt {
    strategy_index: usize,
    strategy: DecodeStrategy,
    reserved_rows: Vec<usize>,
    pos: usize,
    generated_tokens: usize,
    kind: AttemptKind,
}

impl ScheduledAttempt {
    fn is_done(&self, sample_len: usize, eot: u32) -> bool {
        match &self.kind {
            AttemptKind::Single(single) => {
                sample_len == 0 || single.next_token == eot || single.tokens.len() >= sample_len
            }
            AttemptKind::Beam(beam) => {
                sample_len == 0
                    || self.generated_tokens >= sample_len
                    || beam.active.is_empty()
                    || beam.finished.len() >= self.reserved_rows.len()
            }
        }
    }
}

fn start_attempt(
    strategy_index: usize,
    strategy: DecodeStrategy,
    rows: Vec<usize>,
    seed: &DecodeSeed,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
    n_vocab: usize,
) -> Result<ScheduledAttempt> {
    let ctx = &seed.ctx;
    let last = ctx
        .prefill_logits
        .get((ctx.init_len - 1) * n_vocab..ctx.init_len * n_vocab)
        .ok_or_else(|| decode_err("prefill logits are truncated"))?;
    let mut filtered = last.to_vec();
    apply_logit_filters(
        &mut filtered,
        tokenizer,
        options,
        &ctx.initial_tokens,
        ctx.sample_begin,
        0,
        &ctx.suppress_tokens,
    );
    let kind = match strategy {
        DecodeStrategy::Greedy | DecodeStrategy::Sample { .. } => {
            if options.sample_len == Some(0) {
                return Ok(ScheduledAttempt {
                    strategy_index,
                    strategy,
                    reserved_rows: rows,
                    pos: ctx.init_len,
                    generated_tokens: 0,
                    kind: AttemptKind::Single(SingleAttempt {
                        next_token: tokenizer.eot(),
                        tokens: Vec::new(),
                        token_probs: Vec::new(),
                        sum_logprob: 0.0,
                    }),
                });
            }
            let next_token = pick_token(&filtered, strategy.temperature());
            let sum_logprob = log_softmax(&filtered, next_token as usize);
            let (tokens, token_probs) = if next_token == tokenizer.eot() {
                (Vec::new(), Vec::new())
            } else {
                (vec![next_token], vec![sum_logprob.exp()])
            };
            AttemptKind::Single(SingleAttempt { next_token, tokens, token_probs, sum_logprob })
        }
        DecodeStrategy::Beam { size } => {
            if options.sample_len == Some(0) {
                let active = vec![BeamHypothesis {
                    logical_id: 0,
                    tokens: ctx.initial_tokens.clone(),
                    token_probs: Vec::new(),
                    sum_logprob: 0.0,
                }];
                return Ok(ScheduledAttempt {
                    strategy_index,
                    strategy,
                    reserved_rows: rows.clone(),
                    pos: ctx.init_len,
                    generated_tokens: 0,
                    kind: AttemptKind::Beam(BeamAttempt {
                        active,
                        rows: vec![rows[0]],
                        finished: Vec::new(),
                        next_logical_id: 1,
                    }),
                });
            }
            let mut ranked: Vec<_> = log_softmax_vec(&filtered).into_iter().enumerate().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            let mut active = Vec::with_capacity(size);
            let mut finished = Vec::new();
            let mut next_logical_id = 0;
            for (token, logprob) in ranked {
                if active.len() >= size {
                    break;
                }
                let mut tokens = ctx.initial_tokens.clone();
                tokens.push(token as u32);
                let hypothesis = BeamHypothesis {
                    logical_id: next_logical_id,
                    tokens,
                    token_probs: vec![logprob.exp()],
                    sum_logprob: logprob,
                };
                next_logical_id += 1;
                if token as u32 == tokenizer.eot() {
                    finished.push(hypothesis);
                } else {
                    active.push(hypothesis);
                }
            }
            let active_rows = rows[..active.len()].to_vec();
            AttemptKind::Beam(BeamAttempt { active, rows: active_rows, finished, next_logical_id })
        }
    };
    Ok(ScheduledAttempt { strategy_index, strategy, reserved_rows: rows, pos: ctx.init_len, generated_tokens: 1, kind })
}

fn seed_attempt_rows(
    jit: &mut WhisperDecoderStepJit,
    rows: &[usize],
    seed: &DecodeSeed,
    n_text_ctx: usize,
) -> Result<()> {
    let ctx = &seed.ctx;
    let per_pos = ctx.per_pos_floats();
    let self_stride = n_text_ctx.checked_mul(per_pos).ok_or_else(|| decode_err("self cache stride overflow"))?;
    let cross_stride = N_AUDIO_CTX.checked_mul(per_pos).ok_or_else(|| decode_err("cross cache stride overflow"))?;
    for &row in rows {
        copyin_cache_row(jit.self_k_cache_mut().context(JitSnafu)?, row, self_stride, &ctx.self_k_cache)?;
        copyin_cache_row(jit.self_v_cache_mut().context(JitSnafu)?, row, self_stride, &ctx.self_v_cache)?;
        copyin_cache_row(jit.cross_k_mut().context(JitSnafu)?, row, cross_stride, &ctx.cross_k)?;
        copyin_cache_row(jit.cross_v_mut().context(JitSnafu)?, row, cross_stride, &ctx.cross_v)?;
    }
    Ok(())
}

fn append_row_cache(
    jit: &mut WhisperDecoderStepJit,
    row: usize,
    pos: usize,
    per_pos_bytes: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let dst = row
        .checked_mul(row_stride_bytes)
        .and_then(|base| pos.checked_mul(per_pos_bytes).and_then(|offset| base.checked_add(offset)))
        .ok_or_else(|| decode_err("self cache append offset overflow"))?;
    let src = row.checked_mul(per_pos_bytes).ok_or_else(|| decode_err("step cache output offset overflow"))?;
    jit.copy_output_to_self_k_cache(1, dst, src, per_pos_bytes).context(JitSnafu)?;
    jit.copy_output_to_self_v_cache(2, dst, src, per_pos_bytes).context(JitSnafu)
}

fn clone_cache_prefix(
    jit: &mut WhisperDecoderStepJit,
    copies: &[CacheCopy],
    positions: usize,
    per_pos_bytes: usize,
    row_stride_bytes: usize,
) -> Result<()> {
    let len = positions.checked_mul(per_pos_bytes).ok_or_else(|| decode_err("cache prefix length overflow"))?;
    for copy in copies {
        let src =
            copy.src_row.checked_mul(row_stride_bytes).ok_or_else(|| decode_err("cache source offset overflow"))?;
        let dst = copy
            .dst_row
            .checked_mul(row_stride_bytes)
            .ok_or_else(|| decode_err("cache destination offset overflow"))?;
        jit.self_k_cache_mut().context(JitSnafu)?.copy_within(dst, src, len).context(DeviceSnafu)?;
        jit.self_v_cache_mut().context(JitSnafu)?.copy_within(dst, src, len).context(DeviceSnafu)?;
    }
    Ok(())
}

fn finish_attempt(
    attempt: ScheduledAttempt,
    seed: &DecodeSeed,
    tokenizer: &WhisperTokenizer,
    options: &DecodeOptions,
) -> Result<DecodeResult> {
    match attempt.kind {
        AttemptKind::Single(single) => finish_decode(
            &single.tokens,
            &single.token_probs,
            tokenizer,
            if options.sample_len == Some(0) { 0.0 } else { single.sum_logprob },
            seed.ctx.no_speech_prob,
            options,
        ),
        AttemptKind::Beam(beam) => {
            let size = attempt.reserved_rows.len();
            let best =
                finalize_beam_hypotheses(beam.active, beam.finished, size, tokenizer.eot(), seed.ctx.sample_begin)
                    .ok_or_else(|| decode_err("beam produced nothing"))?;
            let tokens: Vec<_> = best.tokens[seed.ctx.sample_begin..]
                .iter()
                .copied()
                .take_while(|&token| token != tokenizer.eot())
                .collect();
            let token_probs = best.token_probs.into_iter().take(tokens.len()).collect::<Vec<_>>();
            finish_decode(&tokens, &token_probs, tokenizer, best.sum_logprob, seed.ctx.no_speech_prob, options)
        }
    }
}

/// Decode all requests through one concrete `[decoder_slots, ...]` step graph.
/// Attempts reserve their full width atomically and retain every reserved row,
/// including inactive beam rows, until quality acceptance or fallback requeue.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_fixed_slot_decode(
    seeds: &[DecodeSeed],
    request_options: &[DecodeOptions],
    step_jit: &mut WhisperDecoderStepJit,
    capacity: usize,
    tokenizer: &WhisperTokenizer,
    n_text_ctx: usize,
    n_vocab: usize,
) -> Result<Vec<DecodeResult>> {
    if seeds.len() != request_options.len() {
        return Err(decode_err("decode seed/options count mismatch"));
    }
    for options in request_options {
        options.validate().map_err(decode_err)?;
        for strategy in attempt_strategies(options) {
            if strategy_width(strategy) > capacity {
                return Err(decode_err("decode attempt width exceeds decoder slots"));
            }
        }
    }

    let strategies: Vec<_> = request_options.iter().map(attempt_strategies).collect();
    let mut queue: VecDeque<_> = (0..seeds.len()).map(|request| (request, 0usize)).collect();
    let mut allocator = SlotAllocator::new(capacity);
    let mut attempts: Vec<Option<ScheduledAttempt>> = (0..seeds.len()).map(|_| None).collect();
    let mut results: Vec<Option<DecodeResult>> = (0..seeds.len()).map(|_| None).collect();

    while results.iter().any(Option::is_none) {
        while let Some(&(request, strategy_index)) = queue.front() {
            let strategy = strategies[request][strategy_index];
            let Some(rows) = allocator.reserve(request, strategy_width(strategy)).map_err(decode_err)? else {
                break;
            };
            queue.pop_front();
            let mut options = request_options[request].clone();
            options.strategy = strategy;
            let attempt = start_attempt(strategy_index, strategy, rows, &seeds[request], tokenizer, &options, n_vocab)?;
            seed_attempt_rows(step_jit, &attempt.reserved_rows, &seeds[request], n_text_ctx)?;
            attempts[request] = Some(attempt);
        }

        let active_requests: Vec<_> =
            attempts.iter().enumerate().filter_map(|(request, attempt)| attempt.as_ref().map(|_| request)).collect();
        if active_requests.is_empty() {
            return Err(decode_err("fixed-slot scheduler made no progress"));
        }

        // Attempts that finish from prefill (EOT or zero budget) need no graph dispatch.
        let mut dispatch = false;
        for &request in &active_requests {
            let attempt = attempts[request].as_ref().expect("active attempt");
            let sample_len = request_options[request].sample_len.unwrap_or(n_text_ctx / 2);
            if attempt.is_done(sample_len, tokenizer.eot()) {
                continue;
            }
            let seed = &seeds[request].ctx;
            match &attempt.kind {
                AttemptKind::Single(single) => {
                    let row = attempt.reserved_rows[0];
                    write_token_row(step_jit, row, single.next_token)?;
                    write_pos_emb_row(
                        step_jit,
                        row,
                        seed.pos_embedding
                            .get(attempt.pos * seed.n_state..(attempt.pos + 1) * seed.n_state)
                            .ok_or_else(|| decode_err("position embedding is out of bounds"))?,
                    )?;
                    write_self_mask_row(step_jit, row, attempt.pos, n_text_ctx)?;
                }
                AttemptKind::Beam(beam) => {
                    for (hypothesis, &row) in beam.active.iter().zip(&beam.rows) {
                        write_token_row(
                            step_jit,
                            row,
                            *hypothesis.tokens.last().ok_or_else(|| decode_err("empty beam"))?,
                        )?;
                        write_pos_emb_row(
                            step_jit,
                            row,
                            seed.pos_embedding
                                .get(attempt.pos * seed.n_state..(attempt.pos + 1) * seed.n_state)
                                .ok_or_else(|| decode_err("position embedding is out of bounds"))?,
                        )?;
                        write_self_mask_row(step_jit, row, attempt.pos, n_text_ctx)?;
                    }
                }
            }
            dispatch = true;
        }

        if dispatch {
            step_jit.execute().context(JitSnafu)?;
            for &request in &active_requests {
                let attempt = attempts[request].as_mut().expect("active attempt");
                let sample_len = request_options[request].sample_len.unwrap_or(n_text_ctx / 2);
                if attempt.is_done(sample_len, tokenizer.eot()) {
                    continue;
                }
                let seed = &seeds[request].ctx;
                let per_pos_bytes = seed.per_pos_bytes();
                let row_stride_bytes = n_text_ctx
                    .checked_mul(per_pos_bytes)
                    .ok_or_else(|| decode_err("self cache row stride overflow"))?;
                let mut options = request_options[request].clone();
                options.strategy = attempt.strategy;
                match &mut attempt.kind {
                    AttemptKind::Single(single) => {
                        let row = attempt.reserved_rows[0];
                        append_row_cache(step_jit, row, attempt.pos, per_pos_bytes, row_stride_bytes)?;
                        let mut logits = read_logits_row(step_jit, row, n_vocab)?;
                        let all_tokens: Vec<_> =
                            seed.initial_tokens.iter().copied().chain(single.tokens.iter().copied()).collect();
                        apply_logit_filters(
                            &mut logits,
                            tokenizer,
                            &options,
                            &all_tokens,
                            seed.sample_begin,
                            attempt.pos + 1 - seed.init_len,
                            &seed.suppress_tokens,
                        );
                        single.next_token = pick_token(&logits, attempt.strategy.temperature());
                        let logprob = log_softmax(&logits, single.next_token as usize);
                        single.sum_logprob += logprob;
                        if single.next_token != tokenizer.eot() {
                            single.tokens.push(single.next_token);
                            single.token_probs.push(logprob.exp());
                        }
                    }
                    AttemptKind::Beam(beam) => {
                        for &row in &beam.rows {
                            append_row_cache(step_jit, row, attempt.pos, per_pos_bytes, row_stride_bytes)?;
                        }
                        let size = attempt.reserved_rows.len();
                        let mut candidates = Vec::new();
                        for (parent_index, (hypothesis, &row)) in beam.active.iter().zip(&beam.rows).enumerate() {
                            let mut logits = read_logits_row(step_jit, row, n_vocab)?;
                            apply_logit_filters(
                                &mut logits,
                                tokenizer,
                                &options,
                                &hypothesis.tokens,
                                seed.sample_begin,
                                attempt.pos + 1 - seed.init_len,
                                &seed.suppress_tokens,
                            );
                            let mut ranked: Vec<_> = log_softmax_vec(&logits).into_iter().enumerate().collect();
                            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                            for (token, logprob) in ranked.into_iter().take(size + 1) {
                                candidates.push(BeamCandidate {
                                    parent_index,
                                    parent_logical_id: hypothesis.logical_id,
                                    parent_row: row,
                                    token_id: token as u32,
                                    token_logprob: logprob,
                                    sum_logprob: hypothesis.sum_logprob + logprob,
                                });
                            }
                        }
                        let (active, newly_finished, survivors) = select_beam_candidates(
                            &beam.active,
                            candidates,
                            size,
                            tokenizer.eot(),
                            size - beam.finished.len(),
                            &mut beam.next_logical_id,
                        );
                        beam.finished.extend(newly_finished);
                        let assignment = plan_beam_rows(&attempt.reserved_rows, &survivors).map_err(decode_err)?;
                        clone_cache_prefix(
                            step_jit,
                            &assignment.copies,
                            attempt.pos + 1,
                            per_pos_bytes,
                            row_stride_bytes,
                        )?;
                        beam.active = active;
                        beam.rows = assignment.rows;
                    }
                }
                attempt.pos += 1;
                attempt.generated_tokens += 1;
            }
        }

        for request in active_requests {
            let sample_len = request_options[request].sample_len.unwrap_or(n_text_ctx / 2);
            let done = attempts[request]
                .as_ref()
                .is_some_and(|attempt| attempt.is_done(sample_len, tokenizer.eot()) || attempt.pos >= n_text_ctx);
            if !done {
                continue;
            }
            let attempt = attempts[request].take().expect("completed attempt");
            let strategy_index = attempt.strategy_index;
            let mut options = request_options[request].clone();
            options.strategy = attempt.strategy;
            options.fallback = None;
            let result = finish_attempt(attempt, &seeds[request], tokenizer, &options)?;
            allocator.release(request);
            let retry = strategies[request].get(strategy_index + 1).is_some()
                && request_options[request]
                    .fallback
                    .as_ref()
                    .is_some_and(|fallback| check_fallback(&result, fallback, &request_options[request]));
            if retry {
                queue.push_back((request, strategy_index + 1));
            } else {
                results[request] = Some(result);
            }
        }
    }

    collect_ordered(results).map_err(decode_err)
}

// ─── Batched JIT buffer row helpers ─────────────────────────────────────────
//
// The batched step JIT owns max_lanes-sized buffers; each lane writes/reads
// its row. These wrap the per-row slicing so the main loop stays readable.

fn write_token_row(jit: &mut WhisperDecoderStepJit, row: usize, token: u32) -> Result<()> {
    let buf = jit.token_mut().context(JitSnafu)?;
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    // token is [max_lanes, 1] i32; row stride = 4 bytes
    let off = row.checked_mul(std::mem::size_of::<i32>()).ok_or_else(|| decode_err("token row offset overflow"))?;
    let tok = [token as i32];
    let bytes: &[u8] = bytemuck::cast_slice(&tok);
    let target = dst.get_mut(off..off + bytes.len()).ok_or_else(|| decode_err("token row is out of bounds"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

fn write_pos_emb_row(jit: &mut WhisperDecoderStepJit, row: usize, emb: &[f32]) -> Result<()> {
    let buf = jit.pos_emb_mut().context(JitSnafu)?;
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    // pos_emb is [max_lanes, 1, n_state] f32
    let row_bytes =
        emb.len().checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| decode_err("position row stride overflow"))?;
    let off = row.checked_mul(row_bytes).ok_or_else(|| decode_err("position row offset overflow"))?;
    let bytes: &[u8] = bytemuck::cast_slice(emb);
    let target = dst.get_mut(off..off + bytes.len()).ok_or_else(|| decode_err("position row is out of bounds"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

fn write_self_mask_row(jit: &mut WhisperDecoderStepJit, row: usize, pos: usize, n_text_ctx: usize) -> Result<()> {
    let buf = jit.self_mask_mut().context(JitSnafu)?;
    let dst = buf.as_host_bytes_mut().context(DeviceSnafu)?;
    // mask is [max_lanes, 1, 1, n_text_ctx + 1]; row stride = (n_text_ctx+1)*4
    let stride = n_text_ctx
        .checked_add(1)
        .and_then(|positions| positions.checked_mul(std::mem::size_of::<f32>()))
        .ok_or_else(|| decode_err("mask row stride overflow"))?;
    let off = row.checked_mul(stride).ok_or_else(|| decode_err("mask row offset overflow"))?;
    // [0..pos) = 0.0 (attend), [pos..n_text_ctx) = -inf (mask), [n_text_ctx] = 0.0
    let mut mask = vec![0f32; n_text_ctx + 1];
    let masked = mask.get_mut(pos..n_text_ctx).ok_or_else(|| decode_err("decoder position exceeds text context"))?;
    for v in masked {
        *v = f32::NEG_INFINITY;
    }
    let bytes: &[u8] = bytemuck::cast_slice(&mask);
    let target = dst.get_mut(off..off + bytes.len()).ok_or_else(|| decode_err("mask row is out of bounds"))?;
    target.copy_from_slice(bytes);
    Ok(())
}

/// Copy a full cache buffer (self or cross) into one row of a device-local
/// batched JIT input via the copy engine. `row_stride_floats` is the buffer's
/// per-row capacity (N_TEXT_CTX or N_AUDIO_CTX positions × per_pos_floats);
/// `data` may be shorter (e.g. only `init_len` positions populated for the self
/// cache after prefill). Used for one-time seeding; the cache then grows
/// on-device via SDMA append.
fn copyin_cache_row(buf: &mut svod_device::Buffer, row: usize, row_stride_floats: usize, data: &[f32]) -> Result<()> {
    let row_bytes = row_stride_floats
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| decode_err("cache row stride overflow"))?;
    let off = row.checked_mul(row_bytes).ok_or_else(|| decode_err("cache row offset overflow"))?;
    let bytes: &[u8] = bytemuck::cast_slice(data);
    let end = off.checked_add(bytes.len()).ok_or_else(|| decode_err("cache seed end overflow"))?;
    if end > buf.size() || bytes.len() > row_bytes {
        return Err(decode_err("cache seed row is out of bounds"));
    }
    buf.copyin_at(off, bytes).context(DeviceSnafu)
}

/// Read one lane's logits row `[n_vocab]` from the batched JIT output.
fn read_logits_row(jit: &mut WhisperDecoderStepJit, row: usize, n_vocab: usize) -> Result<Vec<f32>> {
    let buf = jit.logits().context(JitSnafu)?;
    let src = buf.as_host_bytes().context(DeviceSnafu)?;
    let row_bytes = n_vocab.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| decode_err("logits row overflow"))?;
    let off = row.checked_mul(row_bytes).ok_or_else(|| decode_err("logits offset overflow"))?;
    let end = off.checked_add(row_bytes).ok_or_else(|| decode_err("logits end overflow"))?;
    let logits = src.get(off..end).ok_or_else(|| decode_err("logits row is out of bounds"))?;
    Ok(bytemuck::cast_slice(logits).to_vec())
}

// ─── Cached beam search ─────────────────────────────────────────────────────

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BeamHypothesis {
    /// Stable search identity. Decoder rows are deliberately not part of it.
    pub(crate) logical_id: usize,
    pub(crate) tokens: Vec<u32>,
    pub(crate) token_probs: Vec<f32>,
    pub(crate) sum_logprob: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BeamCandidate {
    pub(crate) parent_index: usize,
    pub(crate) parent_logical_id: usize,
    pub(crate) parent_row: usize,
    pub(crate) token_id: u32,
    pub(crate) token_logprob: f32,
    pub(crate) sum_logprob: f32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BeamSurvivor {
    pub(crate) logical_id: usize,
    pub(crate) parent_row: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CacheCopy {
    pub(crate) src_row: usize,
    pub(crate) dst_row: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct RowAssignment {
    /// Destination row for each survivor, in survivor (logical rank) order.
    pub(crate) rows: Vec<usize>,
    pub(crate) copies: Vec<CacheCopy>,
}

/// Assign fixed decoder rows without scratch storage or copy cycles.
///
/// The first selected child of each parent retains that parent's row. Further
/// children use only reserved rows whose old hypotheses are no longer live.
pub(crate) fn plan_beam_rows(
    reserved_rows: &[usize],
    survivors: &[BeamSurvivor],
) -> std::result::Result<RowAssignment, &'static str> {
    let mut unique_reserved = reserved_rows.to_vec();
    unique_reserved.sort_unstable();
    unique_reserved.dedup();
    if unique_reserved.len() != reserved_rows.len() {
        return Err("reserved beam rows must be unique");
    }
    if survivors.len() > reserved_rows.len() {
        return Err("more survivors than reserved beam rows");
    }
    if survivors.iter().any(|survivor| !unique_reserved.contains(&survivor.parent_row)) {
        return Err("survivor parent row is not reserved");
    }

    let mut live_parent_rows = Vec::new();
    for survivor in survivors {
        if !live_parent_rows.contains(&survivor.parent_row) {
            live_parent_rows.push(survivor.parent_row);
        }
    }
    let mut dead_rows = reserved_rows.iter().copied().filter(|row| !live_parent_rows.contains(row));
    let mut retained = Vec::new();
    let mut rows = Vec::with_capacity(survivors.len());
    let mut copies = Vec::new();
    for survivor in survivors {
        if !retained.contains(&survivor.parent_row) {
            retained.push(survivor.parent_row);
            rows.push(survivor.parent_row);
        } else {
            let dst_row = dead_rows.next().ok_or("insufficient inactive rows for duplicate beam children")?;
            rows.push(dst_row);
            copies.push(CacheCopy { src_row: survivor.parent_row, dst_row });
        }
    }
    Ok(RowAssignment { rows, copies })
}

fn candidate_order(a: &BeamCandidate, b: &BeamCandidate) -> Ordering {
    b.sum_logprob
        .total_cmp(&a.sum_logprob)
        .then_with(|| a.parent_logical_id.cmp(&b.parent_logical_id))
        .then_with(|| a.token_id.cmp(&b.token_id))
        .then_with(|| a.parent_index.cmp(&b.parent_index))
}

/// Deterministically rank candidates and split completed from active children.
pub(crate) fn select_beam_candidates(
    parents: &[BeamHypothesis],
    mut candidates: Vec<BeamCandidate>,
    beam_size: usize,
    eot: u32,
    finished_capacity: usize,
    next_logical_id: &mut usize,
) -> (Vec<BeamHypothesis>, Vec<BeamHypothesis>, Vec<BeamSurvivor>) {
    candidates.sort_by(candidate_order);
    let mut active = Vec::with_capacity(beam_size);
    let mut finished = Vec::new();
    let mut survivors = Vec::with_capacity(beam_size);
    for candidate in candidates {
        if active.len() >= beam_size {
            break;
        }
        let Some(parent) = parents.get(candidate.parent_index) else {
            continue;
        };
        let logical_id = *next_logical_id;
        *next_logical_id += 1;
        let mut child = parent.clone();
        child.logical_id = logical_id;
        child.tokens.push(candidate.token_id);
        child.token_probs.push(candidate.token_logprob.exp());
        child.sum_logprob = candidate.sum_logprob;
        if candidate.token_id == eot {
            if finished.len() < finished_capacity {
                finished.push(child);
            }
        } else {
            active.push(child);
            survivors.push(BeamSurvivor { logical_id, parent_row: candidate.parent_row });
        }
    }
    (active, finished, survivors)
}

/// Backfill unfinished hypotheses with EOT and choose the normalized best.
pub(crate) fn finalize_beam_hypotheses(
    active: Vec<BeamHypothesis>,
    mut finished: Vec<BeamHypothesis>,
    beam_size: usize,
    eot: u32,
    sample_begin: usize,
) -> Option<BeamHypothesis> {
    for mut hypothesis in active {
        if finished.len() >= beam_size {
            break;
        }
        if hypothesis.tokens.last().is_none_or(|&token| token != eot) {
            hypothesis.tokens.push(eot);
        }
        finished.push(hypothesis);
    }
    finished.sort_by(|a, b| {
        let a_score = a.sum_logprob / a.tokens.len().saturating_sub(sample_begin + 1).max(1) as f32;
        let b_score = b.sum_logprob / b.tokens.len().saturating_sub(sample_begin + 1).max(1) as f32;
        b_score.total_cmp(&a_score).then_with(|| a.logical_id.cmp(&b.logical_id))
    });
    finished.into_iter().next()
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
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut beams = Vec::new();
    let mut finished = Vec::new();
    let mut next_logical_id = 0;
    if sample_len == 0 {
        beams.push(BeamHypothesis {
            logical_id: next_logical_id,
            tokens: ctx.initial_tokens.clone(),
            token_probs: Vec::new(),
            sum_logprob: 0.0,
        });
        next_logical_id += 1;
    }
    for (tok_id, lp) in &indexed {
        if sample_len == 0 {
            break;
        }
        if beams.len() >= beam_size {
            break;
        }
        let mut toks = ctx.initial_tokens.clone();
        toks.push(*tok_id as u32);
        if *tok_id as u32 == eot {
            finished.push(BeamHypothesis {
                logical_id: next_logical_id,
                tokens: toks,
                token_probs: vec![lp.exp()],
                sum_logprob: *lp,
            });
        } else {
            beams.push(BeamHypothesis {
                logical_id: next_logical_id,
                tokens: toks,
                token_probs: vec![lp.exp()],
                sum_logprob: *lp,
            });
        }
        next_logical_id += 1;
    }

    let per_beam_floats = ctx.per_beam_floats(n_text_ctx);
    let sample_begin = ctx.sample_begin;

    for step in 0..remaining_sample_steps(sample_len) {
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
        let mut candidates = Vec::new();
        for (bi, beam) in beams.iter().enumerate() {
            let start = bi * n_vocab;
            let Some(beam_logits) = all_logits.get(start..start + n_vocab) else {
                return Err(decode_err("beam logits row is truncated"));
            };
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
            idx.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
            for (tok_id, lp) in idx.into_iter().take(beam_size + 1) {
                candidates.push(BeamCandidate {
                    parent_index: bi,
                    parent_logical_id: beam.logical_id,
                    parent_row: bi,
                    token_id: tok_id as u32,
                    token_logprob: lp,
                    sum_logprob: beam.sum_logprob + lp,
                });
            }
        }

        let (new_beams, newly_finished, survivors) = select_beam_candidates(
            &beams,
            candidates,
            beam_size,
            eot,
            beam_size - finished.len(),
            &mut next_logical_id,
        );
        finished.extend(newly_finished);
        let parent_map: Vec<usize> = survivors.iter().map(|survivor| survivor.parent_row).collect();

        // Reorder cache by parent_map (host-side, copies ENTIRE cache incl new K/V)
        if !parent_map.is_empty() {
            reorder_cache_host(step_jit.self_k_cache_mut().context(JitSnafu)?, &parent_map, per_beam_floats)?;
            reorder_cache_host(step_jit.self_v_cache_mut().context(JitSnafu)?, &parent_map, per_beam_floats)?;
        }

        beams = new_beams;

        if beams.iter().all(|b| b.tokens.len() >= n_text_ctx) {
            break;
        }
    }

    let best = finalize_beam_hypotheses(beams, finished, beam_size, eot, ctx.sample_begin)
        .ok_or_else(|| decode_err("beam produced nothing"))?;
    let output_tokens: Vec<u32> = best.tokens[ctx.sample_begin..].iter().copied().take_while(|&t| t != eot).collect();
    let token_probs = best.token_probs.into_iter().take(output_tokens.len()).collect();
    let text = tokenizer.decode(&output_tokens);
    let avg_logprob = best.sum_logprob / (output_tokens.len() + 1) as f32;
    let compression_ratio = compression_ratio_text(&text);
    Ok(DecodeResult {
        tokens: output_tokens,
        token_probs,
        text,
        avg_logprob,
        no_speech_prob: ctx.no_speech_prob,
        temperature: options.strategy.temperature(),
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
        copyin_cache_full(step_jit.self_k_cache_mut().context(JitSnafu)?, &self.self_k_cache)?;
        copyin_cache_full(step_jit.self_v_cache_mut().context(JitSnafu)?, &self.self_v_cache)?;
        copyin_cache_full(step_jit.cross_k_mut().context(JitSnafu)?, &self.cross_k)?;
        copyin_cache_full(step_jit.cross_v_mut().context(JitSnafu)?, &self.cross_v)?;
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
        copyin_replicated_cache(
            step_jit.self_k_cache_mut().context(JitSnafu)?,
            &self.self_k_cache,
            beam_size,
            self_stride,
        )?;
        copyin_replicated_cache(
            step_jit.self_v_cache_mut().context(JitSnafu)?,
            &self.self_v_cache,
            beam_size,
            self_stride,
        )?;
        copyin_replicated_cache(step_jit.cross_k_mut().context(JitSnafu)?, &self.cross_k, beam_size, cross_stride)?;
        copyin_replicated_cache(step_jit.cross_v_mut().context(JitSnafu)?, &self.cross_v, beam_size, cross_stride)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn write_beam_inputs(
        &self,
        step_jit: &mut WhisperDecoderStepJit,
        beams: &[BeamHypothesis],
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
            write_f32_input(step_jit.pos_emb_mut().context(JitSnafu)?, &packed)?;
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
    let mut initial_tokens = vec![tokenizer.sot()];
    if tokenizer.multilingual {
        let lang = options.language.as_ref().ok_or_else(|| decode_err("language required"))?;
        let lang_tok = tokenizer.language_token_for(lang).unwrap_or_else(|| tokenizer.sot());
        let task_tok = match options.task {
            WhisperTask::Transcribe => tokenizer.transcribe(),
            WhisperTask::Translate => tokenizer.translate(),
        };
        initial_tokens.extend([lang_tok, task_tok]);
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
    // Cross K/V were projected once before fallback decoding and remain in
    // prefill's device-local inputs. Read them only to seed the step graph.
    let cross_k = {
        let buf = prefill_jit.prepared_cross_k_mut().context(JitSnafu)?;
        read_buf(buf, buf.size() / std::mem::size_of::<f32>())?
    };
    let cross_v = {
        let buf = prefill_jit.prepared_cross_v_mut().context(JitSnafu)?;
        read_buf(buf, buf.size() / std::mem::size_of::<f32>())?
    };

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

/// Split a decoded token stream into timestamp-bounded segments.
///
/// The decoder emits paired timestamp tokens (`<|t0|> text <|t1|> text <|t2|>...`)
/// during timestamp-enabled recognition. This function finds
/// consecutive timestamp-token pairs — the boundary between segments — and
/// returns one [`Segment`] per slice, with window-relative start/end times
/// decoded from the timestamp token values.
///
/// Ported from the OpenAI reference (`transcribe.py:339-367`). When no
/// consecutive timestamp pairs are found, returns a single segment spanning
/// the whole token stream.
pub fn split_into_segments(
    tokens: &[u32],
    tokenizer: &WhisperTokenizer,
    window_duration: f32,
) -> Vec<svod_arch::pipelines::audio::Segment> {
    let ts_begin = tokenizer.timestamp_begin();
    let is_ts = |t: u32| t >= ts_begin;

    // Find indices where two adjacent tokens are both timestamps — these are
    // segment boundaries (the closing ts of one segment + the opening ts of the
    // next, shared).
    let mut boundaries: Vec<usize> = Vec::new();
    for i in 1..tokens.len() {
        if is_ts(tokens[i - 1]) && is_ts(tokens[i]) {
            boundaries.push(i);
        }
    }

    let mut segments = Vec::new();
    let terminal_timestamp = tokens.last().is_some_and(|&token| is_ts(token))
        && tokens.get(tokens.len().saturating_sub(2)).is_none_or(|&token| !is_ts(token));

    if boundaries.is_empty() {
        // Whisper treats this as one window-relative segment. If any timestamp
        // was emitted, its last value limits the segment duration.
        let start = 0.0;
        let end = tokens
            .iter()
            .rev()
            .find(|&&token| is_ts(token))
            .filter(|&&token| token != ts_begin)
            .map(|&token| token_to_seconds(token, ts_begin))
            .unwrap_or(window_duration)
            .clamp(0.0, window_duration.max(0.0));
        let text = tokenizer.decode(tokens);
        let text = text.trim();
        if !text.is_empty() && end > start {
            segments.push(Segment { text: text.to_string(), start, end });
        }
        return segments;
    }

    let mut last_slice = 0;
    for &boundary in &boundaries {
        if boundary > last_slice {
            segments.push(segment_from_tokens(&tokens[last_slice..boundary], tokenizer, ts_begin, window_duration));
        }
        last_slice = boundary;
    }

    // An unfinished tail is excluded; it will be decoded again from the last
    // completed timestamp boundary by long-form host orchestration.
    if terminal_timestamp && tokens.len() > last_slice {
        segments.push(segment_from_tokens(&tokens[last_slice..], tokenizer, ts_begin, window_duration));
    }

    // Filter empty segments (can happen when consecutive timestamps have no text between them).
    segments.retain(|s| !s.text.is_empty() && s.end > s.start);
    segments
}

/// Decode one timestamp-bounded slice into a [`Segment`].
fn segment_from_tokens(slice: &[u32], tokenizer: &WhisperTokenizer, ts_begin: u32, window_duration: f32) -> Segment {
    let extent = window_duration.max(0.0);
    let start = slice
        .first()
        .filter(|&&t| t >= ts_begin)
        .map(|&t| token_to_seconds(t, ts_begin))
        .unwrap_or(0.0)
        .clamp(0.0, extent);
    let end = slice
        .last()
        .filter(|&&t| t >= ts_begin)
        .map(|&t| token_to_seconds(t, ts_begin))
        .unwrap_or(start)
        .clamp(start, extent);
    let text = tokenizer.decode(slice).trim().to_string();
    Segment { text, start, end }
}

/// Convert a timestamp token id to seconds: `(id - timestamp_begin) / TOKENS_PER_SECOND`.
fn token_to_seconds(token: u32, ts_begin: u32) -> f32 {
    (token - ts_begin) as f32 / super::config::TOKENS_PER_SECOND
}

fn finish_decode(
    tokens: &[u32],
    token_probs: &[f32],
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
        token_probs: token_probs.to_vec(),
        text,
        avg_logprob,
        no_speech_prob,
        temperature: options.strategy.temperature(),
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

fn write_f32_input(buf: &mut svod_device::Buffer, data: &[f32]) -> Result<()> {
    write_buf(buf, bytemuck::cast_slice(data))
}

/// Copy a host `&[f32]` into a device-local cache buffer via the copy engine.
/// The cache buffers are device-local (no host mapping); the greedy path writes
/// from byte 0 (single-row buffer, data may be shorter than the full allocation).
fn copyin_cache_full(buf: &mut svod_device::Buffer, data: &[f32]) -> Result<()> {
    let bytes: &[u8] = bytemuck::cast_slice(data);
    buf.copyin_at(0, bytes).context(DeviceSnafu)
}

fn copyin_replicated_cache(
    buf: &mut svod_device::Buffer,
    single_data: &[f32],
    beam_size: usize,
    per_beam_floats: usize,
) -> Result<()> {
    let n = single_data.len().min(per_beam_floats);
    let total = per_beam_floats * beam_size;
    let mut packed = vec![0f32; total];
    for bi in 0..beam_size {
        packed[bi * per_beam_floats..bi * per_beam_floats + n].copy_from_slice(&single_data[..n]);
    }
    buf.copyin_at(0, bytemuck::cast_slice(&packed)).context(DeviceSnafu)
}

fn reorder_cache_host(buf: &mut svod_device::Buffer, parent_map: &[usize], per_beam_floats: usize) -> Result<()> {
    let total = per_beam_floats * parent_map.len();
    // Device-local cache: read out via copyout_prefix, reorder on host, write
    // back via copyin_at. This is the beam-reorder path (runs once per step
    // when beams survive/drop), not the per-step hot loop.
    let mut staging = vec![0u8; total * std::mem::size_of::<f32>()];
    buf.copyout_prefix(&mut staging).context(DeviceSnafu)?;
    let current: &[f32] = bytemuck::cast_slice(&staging);

    let mut reordered = vec![0f32; total];
    for (new_idx, &parent_idx) in parent_map.iter().enumerate() {
        let (src, dst) = (parent_idx * per_beam_floats, new_idx * per_beam_floats);
        reordered[dst..dst + per_beam_floats].copy_from_slice(&current[src..src + per_beam_floats]);
    }

    buf.copyin_at(0, bytemuck::cast_slice(&reordered)).context(DeviceSnafu)
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
    apply_timestamp_rules(logits, tokenizer, tokens, sample_begin, options);
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

/// Multinomial sampling from logits at temperature T. Matches the OpenAI
/// reference's `Categorical(logits=logits/T).sample()` (`decoding.py:283`),
/// which PyTorch implements as a numerically stable softmax (max-subtract
/// before exp) followed by inverse-CDF sampling.
fn sample_from_logits(logits: &[f32], temperature: f32) -> u32 {
    // Max-subtract for numerical stability: exp(x - m) avoids overflow on
    // large positive logits. The max is a no-op for the sampling distribution
    // (it's a constant shift that cancels in normalization).
    let max_val = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) / temperature;
    let probs: Vec<f32> = logits.iter().map(|&l| ((l / temperature) - max_val).exp()).collect();
    let sum: f32 = probs.iter().copied().sum();
    let mut r = rand::random::<f32>() * sum;
    for (i, &p) in probs.iter().enumerate() {
        r -= p;
        if r <= 0.0 {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}
