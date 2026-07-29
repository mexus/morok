//! Encoder-only text inference (embeddings generation): tokenize → chunk →
//! embed → aggregate. Host-side and model-agnostic: a model implements only its
//! irreducible part — an [`Embed`] turns tokenized chunks into finished
//! embeddings — and the heavy machinery (truncation geometry and profile
//! assembly) lives in trait defaults here. This is the sibling of
//! [`audio`](super::audio); read that module first — the shape is deliberate.
//!
//! ```text
//! Tokenizer::encode ─▶ Encoding ─▶ [TextChunk]            (Chunker)
//!                                     │
//!                                     ▼
//!   Embed::embed_batch(tokenized chunks) ─▶ [Embedding]
//!                                     │  arch assembles the profile
//!                                     ▼
//!                                Embeddings
//! ```
//!
//! Text crosses the boundary as `&str`, token ids as `&[u32]`: this crate stays
//! free of the Tensor/device stack. The model owns ids → device internally —
//! exactly as [`audio`](super::audio)'s [`Transcriber`](super::audio::Transcriber)
//! owns audio → mel → device.

use std::convert::Infallible;
use std::path::Path;
use std::time::Instant;

use snafu::{ResultExt, Snafu};

pub use svod_runtime::RunProfile;
use svod_runtime::StageProfile;

// ─── Results ────────────────────────────────────────────────────────────────

/// One chunk's finished embedding — already pooled and normalized by the model.
/// `TruncatingChunker` yields exactly one per input; [`SlidingWindowChunker`]
/// yields one per window. Position-agnostic: the
/// [`Embed`] trait returns this, and [`EmbeddingsPipeline`] attaches each one's
/// source position (see [`ChunkEmbedding`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedding {
    pub values: Vec<f32>,
}

/// One [`Embedding`] paired with the char position where its source
/// [`TextChunk`] began in the original text — the per-chunk pipeline result,
/// mirroring how [`audio`](super::audio)'s `ChunkResult` carries `start_sec`/`end_sec` alongside
/// its decoded payload. `char_offset` lets [`SlidingWindowChunker`] (or an NER
/// pipeline) tell windows apart and re-base per-token char spans back to the
/// source — the same field the chunker already records on [`TextChunk`], now
/// threaded through to the output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkEmbedding {
    pub char_offset: usize,
    pub values: Embedding,
}

/// Aggregated pipeline output: one [`ChunkEmbedding`] per chunk, plus the
/// optional per-stage [`RunProfile`] the encoder collected. Profile stages are
/// free-form and extensible (see [`audio`](super::audio)).
#[derive(Debug, Default)]
pub struct Embeddings {
    pub chunks: Vec<ChunkEmbedding>,
    pub profile: Option<RunProfile>,
}

/// Per-call run switch, orthogonal to an [`EmbeddingsPipeline`]'s construction
/// config (sizing). Defaults to `false`, so one built pipeline serves profiled
/// and unprofiled runs without rebuilding — mirroring [`audio::RunOptions`](super::audio::RunOptions).
#[derive(Clone, Copy, Debug, Default)]
pub struct RunOptions {
    /// Collect a per-stage [`RunProfile`] on [`Embeddings::profile`].
    pub profile: bool,
}

// ─── Encoding (tokenizer output) ────────────────────────────────────────────

/// The tokenized form of one input: ids and the masks/offsets HF `tokenizers`
/// produces natively. All integer fields are `Vec<u32>` to match HF's types (no
/// casts in the adapter); the model casts to its buffer dtype at the boundary.
#[derive(Clone, Debug)]
pub struct Encoding {
    pub input_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    /// Zeros for RoBERTa / ModernBERT; carried for BERT-style pair inputs.
    pub token_type_ids: Vec<u32>,
    /// Byte offsets into the source text (HF default). Char offsets arrive with
    /// the deferred NER pipeline; v1 embeddings leave these unused.
    pub offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
}

impl Encoding {
    /// Copy the five getter slices out of an HF `Encoding`. The borrow ends
    /// here — the adapter owns its own `Vec`s, so the source can be dropped.
    pub fn from_hf(enc: &tokenizers::Encoding) -> Self {
        Self {
            input_ids: enc.get_ids().to_vec(),
            attention_mask: enc.get_attention_mask().to_vec(),
            token_type_ids: enc.get_type_ids().to_vec(),
            offsets: enc.get_offsets().to_vec(),
            special_tokens_mask: enc.get_special_tokens_mask().to_vec(),
        }
    }

    pub fn len(&self) -> usize {
        self.input_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.input_ids.is_empty()
    }
}

// ─── Tokenizer ──────────────────────────────────────────────────────────────

/// Text → [`Encoding`]. Implement [`encode`](Tokenizer::encode) (the primary
/// op); [`encode_batch`](Tokenizer::encode_batch) defaults to looping it. The
/// analog of [`audio::Vad`](super::audio::Vad).
pub trait Tokenizer {
    type Error: std::error::Error + 'static;

    /// The model's maximum sequence length (informational / for validation);
    /// the [`Chunker`] owns the truncation policy.
    fn max_seq(&self) -> usize;

    fn encode(&mut self, text: &str) -> Result<Encoding, Self::Error>;

    /// Encode several inputs. Defaults to looping [`encode`](Tokenizer::encode).
    fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Encoding>, Self::Error> {
        texts.iter().map(|t| self.encode(t)).collect()
    }
}

/// Owns a `tokenizers::Error` (which is `Box<dyn std::error::Error + Send +
/// Sync>`) behind a sized, `Error`-implementing type. The boxed trait object is
/// `!Sized`, and std's `impl<E: Error> Error for Box<E>` requires `E: Sized`, so
/// `tokenizers::Error` itself does **not** satisfy `std::error::Error + Sized`
/// and cannot serve as a trait's `type Error` or a snafu `source` field.
/// Wrapping it here — manually, since snafu can only build on types that already
/// implement `Error` — restores a concrete, nameable error type that does.
/// Mirrors how every model crate (`wespeaker::Error`, `resnet::Error`, …) wraps
/// its inner errors; here the inner is already boxed, so the wrapper is a plain
/// newtype.
#[derive(Debug)]
pub struct HfTokenizerError(tokenizers::Error);

impl std::fmt::Display for HfTokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

impl std::error::Error for HfTokenizerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&*self.0)
    }
}

impl From<tokenizers::Error> for HfTokenizerError {
    fn from(err: tokenizers::Error) -> Self {
        Self(err)
    }
}

/// The provided [`Tokenizer`] impl: wraps a HuggingFace `tokenizers::Tokenizer`.
///
/// `encode` calls `inner.encode(text, add_special_tokens = true)`. It does
/// **not** configure truncation/padding — the [`Chunker`] owns the `max_seq`
/// policy, so [`SlidingWindowChunker`] can still see the full token
/// stream. `max_seq()` reports the model's maximum (passed in at construction).
///
/// HF fetching (`from_hub(repo)`) is intentionally absent here — `hf-hub` lives
/// in `svod-model`, which fetches `tokenizer.json` and hands the bytes to
/// [`from_bytes`](HfTokenizer::from_bytes).
pub struct HfTokenizer {
    inner: tokenizers::Tokenizer,
    max_seq: usize,
}

impl HfTokenizer {
    pub fn new(inner: tokenizers::Tokenizer, max_seq: usize) -> Self {
        Self { inner, max_seq }
    }

    /// Load from a `tokenizer.json` path. The caller knows the model's
    /// `max_seq` and threads it in — the same value flows to the chunker and
    /// encoder at assembly.
    pub fn from_path<P: AsRef<Path>>(path: P, max_seq: usize) -> Result<Self, HfTokenizerError> {
        let inner = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self::new(inner, max_seq))
    }

    /// Load from `tokenizer.json` bytes (e.g. fetched via `hf-hub` in
    /// `svod-model`). See [`from_path`](Self::from_path) re: `max_seq`.
    pub fn from_bytes<P: AsRef<[u8]>>(bytes: P, max_seq: usize) -> Result<Self, HfTokenizerError> {
        let inner = tokenizers::Tokenizer::from_bytes(bytes)?;
        Ok(Self::new(inner, max_seq))
    }
}

impl Tokenizer for HfTokenizer {
    type Error = HfTokenizerError;

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn encode(&mut self, text: &str) -> Result<Encoding, HfTokenizerError> {
        let enc = self.inner.encode(text, true)?;
        Ok(Encoding::from_hf(&enc))
    }
}

// ─── Chunker (chunk source) ─────────────────────────────────────────────────

/// One chunked slice of an [`Encoding`], plus the char offset where it begins
/// in the source text. `char_offset` lets a future NER pipeline re-base
/// per-token char spans back to the original text — same role as
/// [`AudioChunk`'s](crate::vad::AudioChunk) sample offsets for word crop.
#[derive(Clone, Debug)]
pub struct TextChunk {
    pub encoding: Encoding,
    pub char_offset: usize,
}

/// Turns an [`Encoding`] into ordered [`TextChunk`]s. The analog of
/// [`audio::Splitter`](super::audio::Splitter); [`EmbeddingsPipeline`] is
/// generic over it.
pub trait Chunker {
    type Error: std::error::Error + 'static;

    /// The sequence length this chunker targets — the value threaded into the
    /// encoder's JIT at assembly (see [`EmbeddingsPipeline::assemble`]).
    fn max_seq(&self) -> usize;

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, Self::Error>;

    /// Stage name for this chunker's wall in a profiled run (e.g. `"chunk"`).
    /// [`EmbeddingsPipeline::embed`] times `chunk` and records it under this
    /// label *when that call requests a profile*. Defaults to `"chunk"`.
    fn profile_label(&self) -> &'static str {
        "chunk"
    }
}

/// Drops ids beyond `max_seq` and emits a single chunk at `char_offset = 0`. For
/// long documents that exceed `max_seq`, use [`SlidingWindowChunker`].
pub struct TruncatingChunker {
    max_seq: usize,
}

impl TruncatingChunker {
    pub fn new(max_seq: usize) -> Self {
        Self { max_seq: max_seq.max(1) }
    }
}

impl Chunker for TruncatingChunker {
    type Error = Infallible;

    fn max_seq(&self) -> usize {
        self.max_seq
    }

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, Infallible> {
        // Keep every field's length consistent: slice to the common cap so ids,
        // masks, and offsets stay aligned.
        let take = self.max_seq.min(enc.input_ids.len());
        let encoding = Encoding {
            input_ids: enc.input_ids[..take].to_vec(),
            attention_mask: enc.attention_mask[..take].to_vec(),
            token_type_ids: enc.token_type_ids[..take].to_vec(),
            offsets: enc.offsets[..take].to_vec(),
            special_tokens_mask: enc.special_tokens_mask[..take].to_vec(),
        };
        Ok(vec![TextChunk { encoding, char_offset: 0 }])
    }
}

// ─── SlidingWindowChunker ────────────────────────────────────────────────────

/// Detect boundary special-token counts: leading and trailing runs of
/// `special_tokens_mask == 1`. Interior specials (e.g. `[UNK]`) remain content.
fn boundary_specials(mask: &[u32]) -> (usize, usize) {
    let lead = mask.iter().take_while(|&&m| m == 1).count();
    let trail = mask.iter().rev().take_while(|&&m| m == 1).count().min(mask.len().saturating_sub(lead));
    (lead, trail)
}

/// Reassemble one windowed [`Encoding`]: `lead` boundary specials from the head,
/// the content body `[body_start..body_end]`, then `trail` specials from the
/// tail. All five fields are built in lockstep.
fn build_window(enc: &Encoding, lead: usize, trail: usize, content_start: usize, content_end: usize) -> Encoding {
    let body_start = lead + content_start;
    let body_end = lead + content_end;
    let tail_start = enc.input_ids.len() - trail;

    fn reassemble<T: Clone>(v: &[T], lead: usize, body_start: usize, body_end: usize, tail_start: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(lead + (body_end - body_start) + (v.len() - tail_start));
        out.extend_from_slice(&v[..lead]);
        out.extend_from_slice(&v[body_start..body_end]);
        out.extend_from_slice(&v[tail_start..]);
        out
    }

    Encoding {
        input_ids: reassemble(&enc.input_ids, lead, body_start, body_end, tail_start),
        attention_mask: reassemble(&enc.attention_mask, lead, body_start, body_end, tail_start),
        token_type_ids: reassemble(&enc.token_type_ids, lead, body_start, body_end, tail_start),
        offsets: reassemble(&enc.offsets, lead, body_start, body_end, tail_start),
        special_tokens_mask: reassemble(&enc.special_tokens_mask, lead, body_start, body_end, tail_start),
    }
}

/// Windows a long token stream into overlapping chunks of `window` total tokens
/// (specials + content), advancing `stride` tokens per step. Each window is a
/// well-formed sequence: boundary specials (`[CLS]`/`[SEP]` for BERT/ModernBERT)
/// are re-attached to every window, detected generically from
/// `special_tokens_mask` rather than hardcoded.
///
/// `window` is the total per-chunk length — `max_seq()` returns it so the
/// encoder JIT is sized for full windows. `stride` is the **step** between
/// consecutive window starts in content-token space: `stride == window` gives
/// adjacent (non-overlapping) chunks; `stride < window` gives an overlap of
/// `window - stride` tokens. Clamped to the content length so `stride == window`
/// never skips content despite specials consuming part of the budget.
///
/// The last window may be shorter than `window` when content doesn't divide
/// evenly — the embedder pads it. Each [`TextChunk`] carries the byte offset of
/// its first content token as `char_offset`.
pub struct SlidingWindowChunker {
    window: usize,
    stride: usize,
}

impl SlidingWindowChunker {
    /// `window` = total sequence length per chunk (incl. specials);
    /// `stride` = step between window starts. Panics unless `1 <= stride <= window`.
    pub fn new(window: usize, stride: usize) -> Self {
        assert!(window >= 1, "window must be >= 1");
        assert!((1..=window).contains(&stride), "stride must be in 1..=window");
        Self { window, stride }
    }
}

impl Chunker for SlidingWindowChunker {
    type Error = Infallible;

    fn max_seq(&self) -> usize {
        self.window
    }

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, Infallible> {
        let (lead, trail) = boundary_specials(&enc.special_tokens_mask);
        let content_len = enc.input_ids.len().saturating_sub(lead + trail);
        if content_len == 0 {
            return Ok(Vec::new());
        }

        let content_window = self.window.saturating_sub(lead + trail);
        assert!(content_window >= 1, "window ({}) too small for {} boundary special tokens", self.window, lead + trail);

        let step = self.stride.min(content_window);

        let mut chunks = Vec::new();
        let mut start = 0;
        loop {
            let end = (start + content_window).min(content_len);
            let char_offset = enc.offsets.get(lead + start).map_or(0, |o| o.0);
            chunks.push(TextChunk { encoding: build_window(enc, lead, trail, start, end), char_offset });
            if end >= content_len {
                break;
            }
            start += step;
        }

        Ok(chunks)
    }
}

// ─── Embed (per-chunk model) ────────────────────────────────────────────────

/// Turns tokenized chunks into finished embeddings. Implement
/// [`embed_batch`](Embed::embed_batch) (the throughput path — one padded JIT
/// execute); [`embed`](Embed::embed) defaults to a batch-of-one. The analog of
/// [`audio::Transcriber`](super::audio::Transcriber).
///
/// Flat trait (no base `Encoder`) — matches audio's `Vad`/`Splitter`/
/// `Transcriber` style. When `Classify` / `Recognize` land, factor a base
/// `Encoder` only if shared plumbing actually emerges.
pub trait Embed {
    type Error: std::error::Error + 'static;

    /// The model's hidden size — the length of each finished [`Embedding`] (the
    /// model pools the sequence dimension before returning it).
    fn hidden_size(&self) -> usize;

    /// `(max_batch, max_seq)` the JIT was prepared for. Informational; the
    /// pipeline sizes the JIT from the chunker's `max_seq` at assembly, so these
    /// are consistent by construction (mirrors how [`audio`](super::audio)'s transcriber is
    /// sized from the splitter's chunk ceiling).
    fn capacity(&self) -> (usize, usize);

    /// Embed every chunk (the model owns its batching/padding), returning one
    /// finished [`Embedding`] per input plus the per-stage [`RunProfile`] —
    /// populated only when `profile` is set (a per-call choice, so the same
    /// encoder serves profiled and unprofiled runs).
    fn embed_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Embedding>, Option<RunProfile>), Self::Error>;

    /// Embed one chunk + its optional profile (batch-of-one fallback).
    fn embed(&mut self, enc: &Encoding, profile: bool) -> Result<(Embedding, Option<RunProfile>), Self::Error> {
        let (mut values, prof) = self.embed_batch(&[enc], profile)?;
        Ok((values.pop().unwrap_or_default(), prof))
    }
}

// ─── EmbeddingsPipeline (composer) ──────────────────────────────────────────

/// The full pipeline: a [`Tokenizer`] + a [`Chunker`] + an [`Embed`]. `embed`
/// runs `tokenizer.encode` → `chunker.chunk` → `encoder.embed_batch`. Build with
/// [`assemble`](EmbeddingsPipeline::assemble) to size the (eagerly-JIT-prepared)
/// encoder from the chunker's `max_seq`, or [`new`](EmbeddingsPipeline::new) to
/// compose three already-built parts. The analog of
/// [`audio::Asr`](super::audio::Asr).
pub struct EmbeddingsPipeline<T: Tokenizer, C: Chunker, E: Embed> {
    tokenizer: T,
    chunker: C,
    encoder: E,
}

#[derive(Debug, Snafu)]
pub enum TextPipelineError<
    T: std::error::Error + 'static,
    C: std::error::Error + 'static,
    E: std::error::Error + 'static,
> {
    #[snafu(display("tokenizing: {source}"))]
    Tokenize { source: T },
    #[snafu(display("chunking: {source}"))]
    Chunk { source: C },
    #[snafu(display("embedding: {source}"))]
    Embed { source: E },
}

/// Return type of [`EmbeddingsPipeline::embed`] / [`EmbeddingsPipeline::embed_default`]:
/// the three sub-trait errors folded into [`TextPipelineError`]. The three-param
/// form trips `clippy::type_complexity`, so the impl spells it once here.
type PipelineResult<T, C, E> =
    Result<Embeddings, TextPipelineError<<T as Tokenizer>::Error, <C as Chunker>::Error, <E as Embed>::Error>>;

impl<T: Tokenizer, C: Chunker, E: Embed> EmbeddingsPipeline<T, C, E> {
    pub fn new(tokenizer: T, chunker: C, encoder: E) -> Self {
        Self { tokenizer, chunker, encoder }
    }

    /// Build the encoder eagerly, sized to the chunker's `max_seq`
    /// ([`Chunker::max_seq`]), then compose. `build` runs the model's JIT
    /// prepare up front — there is no lazy/first-call cost — so the caller never
    /// hand-threads the sequence length between chunker and encoder.
    pub fn assemble<EE>(tokenizer: T, chunker: C, build: impl FnOnce(usize) -> Result<E, EE>) -> Result<Self, EE> {
        let encoder = build(chunker.max_seq())?;
        Ok(Self::new(tokenizer, chunker, encoder))
    }

    /// Tokenize → chunk → embed → assemble profile. [`RunOptions`] is a per-call
    /// switch: the same pipeline serves profiled and unprofiled runs without
    /// rebuilding. When `opts.profile` is set, the tokenize and chunk stages are
    /// timed and recorded ahead of the encoder's stages — so the profile leads
    /// with `[tokenize, <chunker label>, <encoder …>]`, exactly as
    /// [`audio::Asr::transcribe`](super::audio::Asr::transcribe) prepends `vad`.
    pub fn embed(&mut self, text: &str, opts: RunOptions) -> PipelineResult<T, C, E> {
        let t = Instant::now();
        let encoding = self.tokenizer.encode(text).context(TokenizeSnafu)?;
        let tok_wall = t.elapsed();

        let t = Instant::now();
        let chunks = self.chunker.chunk(&encoding).context(ChunkSnafu)?;
        let chunk_wall = t.elapsed();

        let mut embeddings = if chunks.is_empty() {
            // No content (empty input / all-truncated): skip the empty
            // `embed_batch` call — some models index off the batch count and
            // would underflow on zero inputs, mirroring audio's zero-window guard.
            Embeddings { chunks: Vec::new(), profile: None }
        } else {
            let encodings: Vec<&Encoding> = chunks.iter().map(|c| &c.encoding).collect();
            let (values, prof) = self.encoder.embed_batch(&encodings, opts.profile).context(EmbedSnafu)?;
            // Attach each chunk's source char position to its embedding. The
            // embedder returns position-agnostic `Embedding`s in chunk order;
            // zipping with the chunks' `char_offset` yields `ChunkEmbedding`s
            // that carry their window location through to the caller — so a
            // future sliding-window chunker (or NER pipeline) can re-base spans
            // without re-tokenizing.
            let chunk_embeddings = chunks
                .into_iter()
                .map(|c| c.char_offset)
                .zip(values)
                .map(|(char_offset, values)| ChunkEmbedding { char_offset, values })
                .collect();
            Embeddings { chunks: chunk_embeddings, profile: prof }
        };

        if opts.profile {
            // Lead with the host stages, then the encoder's.
            let mut p = RunProfile::default();
            p.push(StageProfile::host("tokenize", tok_wall));
            p.push(StageProfile::host(self.chunker.profile_label(), chunk_wall));
            if let Some(rest) = embeddings.profile.take() {
                p.merge(rest);
            }
            embeddings.profile = Some(p);
        }

        Ok(embeddings)
    }

    /// [`embed`](Self::embed) with default [`RunOptions`] (no profile) — the
    /// common case, without spelling out the struct.
    pub fn embed_default(&mut self, text: &str) -> PipelineResult<T, C, E> {
        self.embed(text, RunOptions::default())
    }

    pub fn tokenizer_mut(&mut self) -> &mut T {
        &mut self.tokenizer
    }

    pub fn chunker_mut(&mut self) -> &mut C {
        &mut self.chunker
    }

    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }
}
