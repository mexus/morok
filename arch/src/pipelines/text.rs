//! Encoder-only text inference (embeddings + classification + token
//! classification): tokenize → chunk → model → aggregate. Host-side and
//! model-agnostic: a model implements only its irreducible part behind the
//! [`Encoder`] supertrait — [`Embed`] / [`Classify`] / [`Recognize`] are thin
//! specializations that fix the per-chunk output kind — and the heavy machinery
//! (sub-batching, truncation geometry, profile assembly, span decoding) lives in
//! trait defaults and free functions here. This is the sibling of
//! [`audio`](super::audio); read that module first — the shape is deliberate.
//!
//! ```text
//! Tokenizer::encode ─▶ Encoding ─▶ [TextChunk]            (Chunker)
//!                                     │
//!                                     ▼
//!   Encoder::run_batch(tokenized chunks) ─▶ [Encoder::Output]
//!                                     │  EncoderPipeline assembles the profile
//!                                     ▼
//!                          {Embeddings, Classifications, …}
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
/// yields one per window. Position-agnostic: the [`Embed`] trait returns this,
/// and [`EncoderPipeline`] attaches each one's source position (see
/// [`ChunkEmbedding`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Embedding {
    pub values: Vec<f32>,
}

/// One [`Embedding`] paired with the byte position where its source
/// [`TextChunk`] began in the original text — the per-chunk pipeline result,
/// mirroring how [`audio`](super::audio)'s `ChunkResult` carries `start_sec`/`end_sec` alongside
/// its decoded payload. `byte_offset` lets [`SlidingWindowChunker`] (or a token
/// classification pipeline) tell windows apart and re-base per-token byte spans
/// back to the source — the same field the chunker already records on
/// [`TextChunk`], now threaded through to the output.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkEmbedding {
    pub byte_offset: usize,
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

/// Batch embedding result from [`embed_batch`](EncoderPipeline::embed_batch):
/// one [`Embeddings`] per input text, plus a shared [`RunProfile`] covering the
/// entire batch run. The profile is batch-level (tokenize, chunk, and embed are
/// timed as a whole), not per-text — each [`Embeddings`] inside `results`
/// carries `profile: None`.
#[derive(Debug, Default)]
pub struct BatchEmbeddings {
    pub results: Vec<Embeddings>,
    pub profile: Option<RunProfile>,
}

/// Per-call run switch, orthogonal to a pipeline's construction config
/// (sizing). Defaults to `false`, so one built pipeline serves profiled and
/// unprofiled runs without rebuilding — mirroring [`audio::RunOptions`](super::audio::RunOptions).
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
    /// Byte offsets into the source text (HF `tokenizers` default). Each entry
    /// is the `(start, end)` byte span of one token; byte offsets (not char) so
    /// they index `&str` directly. Token classification re-bases per-token
    /// entity spans through these.
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

    fn encode_batch(&mut self, texts: &[&str]) -> Result<Vec<Encoding>, HfTokenizerError> {
        let inputs: Vec<String> = texts.iter().map(|s| s.to_string()).collect();
        let encs = self.inner.encode_batch(inputs, true)?;
        Ok(encs.iter().map(Encoding::from_hf).collect())
    }
}

// ─── Chunker (chunk source) ─────────────────────────────────────────────────

/// One chunked slice of an [`Encoding`], plus the byte offset where it begins
/// in the source text. `byte_offset` lets a future token-classification
/// pipeline re-base per-token byte spans back to the original text — same role
/// as [`AudioChunk`'s](crate::vad::AudioChunk) sample offsets for word crop.
#[derive(Clone, Debug)]
pub struct TextChunk {
    pub encoding: Encoding,
    pub byte_offset: usize,
}

/// Turns an [`Encoding`] into ordered [`TextChunk`]s. The analog of
/// [`audio::Splitter`](super::audio::Splitter); [`EncoderPipeline`] is
/// generic over it.
pub trait Chunker {
    type Error: std::error::Error + 'static;

    /// The sequence length this chunker targets — the value threaded into the
    /// encoder's JIT at assembly (see [`EncoderPipeline::assemble`]).
    fn max_seq(&self) -> usize;

    fn chunk(&mut self, enc: &Encoding) -> Result<Vec<TextChunk>, Self::Error>;

    /// Stage name for this chunker's wall in a profiled run (e.g. `"chunk"`).
    /// The pipeline's `embed`/`classify`/`recognize` methods time `chunk` and
    /// record it under this label *when that call requests a profile*. Defaults
    /// to `"chunk"`.
    fn profile_label(&self) -> &'static str {
        "chunk"
    }
}

/// Drops ids beyond `max_seq` and emits a single chunk at `byte_offset = 0`. For
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
        Ok(vec![TextChunk { encoding, byte_offset: 0 }])
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
/// its first content token as `byte_offset`.
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
            let byte_offset = enc.offsets.get(lead + start).map_or(0, |o| o.0);
            chunks.push(TextChunk { encoding: build_window(enc, lead, trail, start, end), byte_offset });
            if end >= content_len {
                break;
            }
            start += step;
        }

        Ok(chunks)
    }
}

// ─── Classify results ────────────────────────────────────────────────────────

/// One chunk's raw class logits — the model applies the classification head
/// (and any fused pooling / normalization) before returning. Position-agnostic:
/// the [`Classify`] trait returns this, and [`EncoderPipeline`] attaches each
/// one's source position (see [`ChunkClassification`]). The caller derives
/// predictions (argmax, softmax, thresholding) from these logits.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Classification {
    pub logits: Vec<f32>,
}

/// One chunk's logits paired with the byte position where its source
/// [`TextChunk`] began — the per-chunk pipeline result, mirroring
/// [`ChunkEmbedding`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkClassification {
    pub byte_offset: usize,
    pub logits: Vec<f32>,
}

/// Aggregated classification output: one [`ChunkClassification`] per chunk,
/// plus the optional per-stage [`RunProfile`]. Mirrors [`Embeddings`].
#[derive(Debug, Default)]
pub struct Classifications {
    pub chunks: Vec<ChunkClassification>,
    pub profile: Option<RunProfile>,
}

/// Batch classification result from
/// [`classify_batch`](EncoderPipeline::classify_batch): one
/// [`Classifications`] per input text, plus a shared [`RunProfile`] covering
/// the entire batch run. Mirrors [`BatchEmbeddings`].
#[derive(Debug, Default)]
pub struct BatchClassifications {
    pub results: Vec<Classifications>,
    pub profile: Option<RunProfile>,
}

// ─── Token-classification results ────────────────────────────────────────────

/// One chunk's raw per-token logits — the model applies the token head (HF
/// `ModernBertPredictionHead` + `classifier`) to every position before returning.
/// Position-agnostic: the [`Recognize`] trait returns this, and
/// [`EncoderPipeline`] attaches each one's source position and per-token byte
/// spans (see [`ChunkTokenClassification`]). `logits` is a flat row-major
/// `(seq_len, num_labels)` grid where `seq_len` is the chunk's live token count
/// (padding already stripped); `seq_len = logits.len() / num_labels`. The caller
/// derives predictions (argmax, softmax, thresholding) and decodes spans from
/// these.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenClassification {
    pub logits: Vec<f32>,
    pub num_labels: usize,
}

/// One chunk's per-token logits paired with the byte position where its source
/// [`TextChunk`] began, plus the per-token geometry the caller needs to decode
/// spans — the per-chunk pipeline result, mirroring [`ChunkClassification`] and
/// [`ChunkEmbedding`]. `token_offsets` and `special_tokens_mask` run in lockstep
/// over the chunk's `seq_len` tokens (one row per token in `logits`).
///
/// `token_offsets` are **source-absolute** byte spans (the HF tokenizer returns
/// source-referential offsets, preserved through chunking), so `&text[start..end]`
/// slices a token directly — no rebase by `byte_offset` is needed. `byte_offset`
/// is kept for chunk ordering and cross-chunk grouping (the byte offset of the
/// chunk's first content token), consistent with the sibling result types.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChunkTokenClassification {
    pub byte_offset: usize,
    pub logits: Vec<f32>,
    pub num_labels: usize,
    pub token_offsets: Vec<(usize, usize)>,
    pub special_tokens_mask: Vec<u32>,
}

/// Aggregated token-classification output: one [`ChunkTokenClassification`] per
/// chunk, plus the optional per-stage [`RunProfile`]. Mirrors [`Classifications`].
#[derive(Debug, Default)]
pub struct TokenClassifications {
    pub chunks: Vec<ChunkTokenClassification>,
    pub profile: Option<RunProfile>,
}

/// Batch token-classification result from
/// [`recognize_batch`](EncoderPipeline::recognize_batch): one
/// [`TokenClassifications`] per input text, plus a shared [`RunProfile`]
/// covering the entire batch run. Mirrors [`BatchClassifications`].
#[derive(Debug, Default)]
pub struct BatchTokenClassifications {
    pub results: Vec<TokenClassifications>,
    pub profile: Option<RunProfile>,
}

// ─── Encoder (per-chunk model) ──────────────────────────────────────────────

/// Any encoder-only head: turns tokenized chunks into a per-chunk
/// position-agnostic output. The shared shape behind the three task traits
/// ([`Embed`] / [`Classify`] / [`Recognize`]) — they fix [`Output`](Encoder::Output)
/// / [`ChunkOutput`](Encoder::ChunkOutput) and add one informational accessor
/// each, so the pipeline machinery (sub-batching, profile assembly) is written
/// once over `Encoder` and the verbs stay discoverable per task.
///
/// Implement [`run_batch`](Encoder::run_batch) (the throughput path — one padded
/// JIT execute) and [`attach`](Encoder::attach) (the only per-task variance:
/// pairing a model output with its chunk's source position);
/// [`run`](Encoder::run) defaults to a batch-of-one. The analog of
/// [`audio::Transcriber`](super::audio::Transcriber).
pub trait Encoder {
    /// Per-chunk model output, position-agnostic: [`Embedding`] /
    /// [`Classification`] / [`TokenClassification`].
    type Output: Default;
    /// Per-chunk *pipeline* result, position-attached: [`ChunkEmbedding`] /
    /// [`ChunkClassification`] / [`ChunkTokenClassification`].
    type ChunkOutput;
    type Error: std::error::Error + 'static;

    /// `(max_batch, max_seq)` the JIT was prepared for. Informational; the
    /// pipeline sizes the JIT from the chunker's `max_seq` at assembly, so these
    /// are consistent by construction (mirrors how [`audio`](super::audio)'s
    /// transcriber is sized from the splitter's chunk ceiling).
    fn capacity(&self) -> (usize, usize);

    /// Run every chunk through the model (the model owns its batching/padding),
    /// returning one [`Output`](Encoder::Output) per input plus the per-stage
    /// [`RunProfile`] — populated only when `profile` is set (a per-call choice,
    /// so the same encoder serves profiled and unprofiled runs).
    fn run_batch(&mut self, batch: &[&Encoding], profile: bool) -> Result<ModelOutputs<Self>, Self::Error>;

    /// Attach a chunk's source position (and, for token heads, its per-token
    /// `offsets` / `special_tokens_mask`) to a model output. The pipeline calls
    /// this after [`run_batch`](Encoder::run_batch) — the only per-task variance,
    /// isolated here so the machinery over `Encoder` is uniform.
    fn attach(chunk: &TextChunk, out: Self::Output) -> Self::ChunkOutput;

    /// Run one chunk + its optional profile (batch-of-one fallback).
    fn run(&mut self, enc: &Encoding, profile: bool) -> Result<(Self::Output, Option<RunProfile>), Self::Error> {
        let (mut values, prof) = self.run_batch(&[enc], profile)?;
        Ok((values.pop().unwrap_or_default(), prof))
    }
}

/// Finished-embeddings head. A specialization of [`Encoder`] fixing
/// [`Output`](Encoder::Output) = [`Embedding`] and
/// [`ChunkOutput`](Encoder::ChunkOutput) = [`ChunkEmbedding`]; implement
/// [`Encoder::run_batch`] / [`Encoder::attach`] plus
/// [`hidden_size`](Embed::hidden_size).
pub trait Embed: Encoder<Output = Embedding, ChunkOutput = ChunkEmbedding> {
    /// The model's hidden size — the length of each finished [`Embedding`] (the
    /// model pools the sequence dimension before returning it).
    fn hidden_size(&self) -> usize;
}

/// Sentence/text-classification head. A specialization of [`Encoder`] fixing
/// [`Output`](Encoder::Output) = [`Classification`] and
/// [`ChunkOutput`](Encoder::ChunkOutput) = [`ChunkClassification`]; implement
/// [`Encoder::run_batch`] / [`Encoder::attach`] plus
/// [`num_classes`](Classify::num_classes).
pub trait Classify: Encoder<Output = Classification, ChunkOutput = ChunkClassification> {
    /// The model's class count — the length of each [`Classification`]'s
    /// `logits` vector.
    fn num_classes(&self) -> usize;
}

/// Token-classification head (NER, POS tagging, chunking, …). A specialization
/// of [`Encoder`] fixing [`Output`](Encoder::Output) = [`TokenClassification`]
/// and [`ChunkOutput`](Encoder::ChunkOutput) = [`ChunkTokenClassification`];
/// implement [`Encoder::run_batch`] / [`Encoder::attach`] plus
/// [`num_labels`](Recognize::num_labels).
pub trait Recognize: Encoder<Output = TokenClassification, ChunkOutput = ChunkTokenClassification> {
    /// The label count — the trailing dim of each [`TokenClassification`]'s
    /// `logits` grid.
    fn num_labels(&self) -> usize;
}

// ─── EncoderPipeline (composer) ─────────────────────────────────────────────

/// The full pipeline: a [`Tokenizer`] + a [`Chunker`] + an [`Encoder`]. One
/// generic host-side composer for embeddings, classification, and token
/// classification — `embed` / `classify` / `recognize` (and their `_batch`
/// siblings) are bounded facades over the same machinery, so a caller grepping
/// for "how do I embed" still finds a named verb. Build with
/// [`assemble`](EncoderPipeline::assemble) to size the (eagerly-JIT-prepared)
/// encoder from the chunker's `max_seq`, or [`new`](EncoderPipeline::new) to
/// compose three already-built parts. The analog of
/// [`audio::Asr`](super::audio::Asr).
pub struct EncoderPipeline<T: Tokenizer, C: Chunker, M: Encoder> {
    tokenizer: T,
    chunker: C,
    model: M,
}

/// Error from an [`EncoderPipeline`] run — one of the tokenize / chunk / encode
/// stage failures. The single error folds the three sub-trait errors; with one
/// pipeline there are no selector collisions, so no `#[snafu(module)]`.
#[derive(Debug, Snafu)]
pub enum EncoderPipelineError<
    T: std::error::Error + 'static,
    C: std::error::Error + 'static,
    M: std::error::Error + 'static,
> {
    #[snafu(display("tokenizing: {source}"))]
    Tokenize { source: T },
    #[snafu(display("chunking: {source}"))]
    Chunk { source: C },
    #[snafu(display("encoding: {source}"))]
    Encode { source: M },
}

/// The three sub-trait errors folded into [`EncoderPipelineError`]. Spelled once
/// here so the verb facades' signatures stay readable (the three-param form
/// trips `clippy::type_complexity`).
type PipelineError<T, C, M> =
    EncoderPipelineError<<T as Tokenizer>::Error, <C as Chunker>::Error, <M as Encoder>::Error>;

/// `(per-input model outputs, optional profile)` — [`Encoder::run_batch`]'s
/// result shape. Aliased because the inline form trips `clippy::type_complexity`.
type ModelOutputs<M> = (Vec<<M as Encoder>::Output>, Option<RunProfile>);

/// `(per-chunk pipeline results, optional profile)` — the pipeline's flattened
/// / single-text result shape. Aliased (`clippy::type_complexity`).
type ChunkOutputs<M> = (Vec<<M as Encoder>::ChunkOutput>, Option<RunProfile>);

/// `(per-text chunk-result vecs, optional profile)` — the batch result shape.
/// Aliased (`clippy::type_complexity`).
type BatchChunkOutputs<M> = (Vec<Vec<<M as Encoder>::ChunkOutput>>, Option<RunProfile>);

impl<T: Tokenizer, C: Chunker, M: Encoder> EncoderPipeline<T, C, M> {
    pub fn new(tokenizer: T, chunker: C, model: M) -> Self {
        Self { tokenizer, chunker, model }
    }

    /// Build the encoder eagerly, sized to the chunker's `max_seq`
    /// ([`Chunker::max_seq`]), then compose. `build` runs the model's JIT
    /// prepare up front — there is no lazy/first-call cost — so the caller never
    /// hand-threads the sequence length between chunker and encoder.
    pub fn assemble<EE>(tokenizer: T, chunker: C, build: impl FnOnce(usize) -> Result<M, EE>) -> Result<Self, EE> {
        let model = build(chunker.max_seq())?;
        Ok(Self::new(tokenizer, chunker, model))
    }

    /// Run a flat list of chunks through the encoder, sub-batching to respect
    /// its `max_batch` capacity and attaching each chunk's source position via
    /// [`Encoder::attach`]. Returns chunk results in order plus a merged profile
    /// (one per sub-batch, accumulated via [`RunProfile::merge`]). Shared by the
    /// single-text and batch entry points.
    fn run_chunks_flat(&mut self, chunks: &[TextChunk], profile: bool) -> Result<ChunkOutputs<M>, M::Error> {
        if chunks.is_empty() {
            return Ok((Vec::new(), None));
        }
        let max_batch = self.model.capacity().0.max(1);
        let mut all = Vec::with_capacity(chunks.len());
        let mut merged_prof: Option<RunProfile> = None;
        for batch in chunks.chunks(max_batch) {
            let encodings: Vec<&Encoding> = batch.iter().map(|c| &c.encoding).collect();
            let (values, prof) = self.model.run_batch(&encodings, profile)?;
            for (chunk, out) in batch.iter().zip(values) {
                all.push(M::attach(chunk, out));
            }
            if let Some(p) = prof {
                merged_prof.get_or_insert_with(RunProfile::default).merge(p);
            }
        }
        Ok((all, merged_prof))
    }

    /// Tokenize → chunk → encode → assemble profile for a single text. Returns
    /// the chunk results and the assembled profile; the verb facades wrap these
    /// into the named aggregate ([`Embeddings`] / [`Classifications`] / …).
    /// [`RunOptions`] is a per-call switch: the same pipeline serves profiled
    /// and unprofiled runs without rebuilding. When `opts.profile` is set, the
    /// tokenize and chunk stages are timed and recorded ahead of the encoder's
    /// stages — so the profile leads with `[tokenize, <chunker label>, <encoder …>]`,
    /// exactly as [`audio::Asr::transcribe`](super::audio::Asr::transcribe) prepends `vad`.
    fn run_single_inner(&mut self, text: &str, opts: RunOptions) -> Result<ChunkOutputs<M>, PipelineError<T, C, M>> {
        let t = Instant::now();
        let encoding = self.tokenizer.encode(text).context(TokenizeSnafu)?;
        let tok_wall = t.elapsed();

        let t = Instant::now();
        let chunks = self.chunker.chunk(&encoding).context(ChunkSnafu)?;
        let chunk_wall = t.elapsed();

        let (chunk_outs, enc_prof) = self.run_chunks_flat(&chunks, opts.profile).context(EncodeSnafu)?;

        let profile = opts.profile.then(|| {
            let mut p = RunProfile::default();
            p.push(StageProfile::host("tokenize", tok_wall));
            p.push(StageProfile::host(self.chunker.profile_label(), chunk_wall));
            if let Some(rest) = enc_prof {
                p.merge(rest);
            }
            p
        });
        Ok((chunk_outs, profile))
    }

    /// Tokenize → chunk → encode multiple texts in one call. All chunks from all
    /// texts are flattened into a single stream and batched through the encoder
    /// (sub-batched to `model.capacity().0` when the total exceeds it),
    /// maximizing throughput. Returns one chunk-result vec per input text (moved
    /// out of the flat stream by each text's chunk count) plus a shared
    /// batch-level profile.
    fn run_batch_inner(
        &mut self,
        texts: &[&str],
        opts: RunOptions,
    ) -> Result<BatchChunkOutputs<M>, PipelineError<T, C, M>> {
        if texts.is_empty() {
            return Ok((Vec::new(), None));
        }

        let t = Instant::now();
        let encodings = self.tokenizer.encode_batch(texts).context(TokenizeSnafu)?;
        let tok_wall = t.elapsed();

        let t = Instant::now();
        let mut all_chunks: Vec<TextChunk> = Vec::new();
        let mut counts: Vec<usize> = Vec::with_capacity(texts.len());
        for enc in &encodings {
            let chunks = self.chunker.chunk(enc).context(ChunkSnafu)?;
            counts.push(chunks.len());
            all_chunks.extend(chunks);
        }
        let chunk_wall = t.elapsed();

        let (all_outs, enc_prof) = self.run_chunks_flat(&all_chunks, opts.profile).context(EncodeSnafu)?;

        // Reassemble per-text: drain the flat chunk-result vec by each text's
        // chunk count (moving, not cloning).
        let mut results = Vec::with_capacity(texts.len());
        let mut iter = all_outs.into_iter();
        for &count in &counts {
            let per_text: Vec<M::ChunkOutput> = iter.by_ref().take(count).collect();
            results.push(per_text);
        }

        let profile = opts.profile.then(|| {
            let mut p = RunProfile::default();
            p.push(StageProfile::host("tokenize", tok_wall));
            p.push(StageProfile::host(self.chunker.profile_label(), chunk_wall));
            if let Some(rest) = enc_prof {
                p.merge(rest);
            }
            p
        });
        Ok((results, profile))
    }

    pub fn tokenizer_mut(&mut self) -> &mut T {
        &mut self.tokenizer
    }

    pub fn chunker_mut(&mut self) -> &mut C {
        &mut self.chunker
    }

    /// Generic typed accessor for the encoder model. The task facades also
    /// expose `embedder_mut` / `classifier_mut` / `recognizer_mut` aliases.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

// ─── Verb facades ───────────────────────────────────────────────────────────
//
// `embed` / `classify` / `recognize` (and their `_batch` siblings) are bounded
// inherent impls over the same `EncoderPipeline`. Distinct bounds (Embed vs
// Classify vs Recognize) ⇒ no coherence conflict; a caller gets only the verbs
// their model's trait unlocks. Each wraps the generic inner into the task's
// named aggregate.

impl<T: Tokenizer, C: Chunker, M: Embed> EncoderPipeline<T, C, M> {
    /// Tokenize → chunk → embed → assemble profile. See
    /// [`run_single_inner`](Self::run_single_inner) for the [`RunOptions`] /
    /// profile semantics.
    pub fn embed(&mut self, text: &str, opts: RunOptions) -> Result<Embeddings, PipelineError<T, C, M>> {
        let (chunks, profile) = self.run_single_inner(text, opts)?;
        Ok(Embeddings { chunks, profile })
    }

    /// [`embed`](Self::embed) with default [`RunOptions`] (no profile) — the
    /// common case, without spelling out the struct.
    pub fn embed_default(&mut self, text: &str) -> Result<Embeddings, PipelineError<T, C, M>> {
        self.embed(text, RunOptions::default())
    }

    /// Tokenize → chunk → embed multiple texts in one call. Returns one
    /// [`Embeddings`] per input text — each carrying its own [`ChunkEmbedding`]s
    /// with correct `byte_offset`s — plus a shared batch-level [`RunProfile`] on
    /// [`BatchEmbeddings::profile`].
    pub fn embed_batch(&mut self, texts: &[&str], opts: RunOptions) -> Result<BatchEmbeddings, PipelineError<T, C, M>> {
        let (per_text, profile) = self.run_batch_inner(texts, opts)?;
        let results = per_text.into_iter().map(|chunks| Embeddings { chunks, profile: None }).collect();
        Ok(BatchEmbeddings { results, profile })
    }

    /// [`embed_batch`](Self::embed_batch) with default [`RunOptions`] (no
    /// profile).
    pub fn embed_batch_default(&mut self, texts: &[&str]) -> Result<BatchEmbeddings, PipelineError<T, C, M>> {
        self.embed_batch(texts, RunOptions::default())
    }

    /// Typed accessor for the embedder (the [`Embed`] model).
    pub fn embedder_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<T: Tokenizer, C: Chunker, M: Classify> EncoderPipeline<T, C, M> {
    /// Tokenize → chunk → classify → assemble profile. See
    /// [`run_single_inner`](Self::run_single_inner) for the [`RunOptions`] /
    /// profile semantics.
    pub fn classify(&mut self, text: &str, opts: RunOptions) -> Result<Classifications, PipelineError<T, C, M>> {
        let (chunks, profile) = self.run_single_inner(text, opts)?;
        Ok(Classifications { chunks, profile })
    }

    /// [`classify`](Self::classify) with default [`RunOptions`] (no profile).
    pub fn classify_default(&mut self, text: &str) -> Result<Classifications, PipelineError<T, C, M>> {
        self.classify(text, RunOptions::default())
    }

    /// Tokenize → chunk → classify multiple texts in one call. Returns one
    /// [`Classifications`] per input text — each carrying its own
    /// [`ChunkClassification`]s with correct `byte_offset`s — plus a shared
    /// batch-level [`RunProfile`] on [`BatchClassifications::profile`].
    pub fn classify_batch(
        &mut self,
        texts: &[&str],
        opts: RunOptions,
    ) -> Result<BatchClassifications, PipelineError<T, C, M>> {
        let (per_text, profile) = self.run_batch_inner(texts, opts)?;
        let results = per_text.into_iter().map(|chunks| Classifications { chunks, profile: None }).collect();
        Ok(BatchClassifications { results, profile })
    }

    /// [`classify_batch`](Self::classify_batch) with default [`RunOptions`].
    pub fn classify_batch_default(&mut self, texts: &[&str]) -> Result<BatchClassifications, PipelineError<T, C, M>> {
        self.classify_batch(texts, RunOptions::default())
    }

    /// Typed accessor for the classifier (the [`Classify`] model).
    pub fn classifier_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

impl<T: Tokenizer, C: Chunker, M: Recognize> EncoderPipeline<T, C, M> {
    /// Tokenize → chunk → recognize → assemble profile. See
    /// [`run_single_inner`](Self::run_single_inner) for the [`RunOptions`] /
    /// profile semantics. Decode spans with [`labels_for_tokens`] /
    /// [`group_spans`].
    pub fn recognize(&mut self, text: &str, opts: RunOptions) -> Result<TokenClassifications, PipelineError<T, C, M>> {
        let (chunks, profile) = self.run_single_inner(text, opts)?;
        Ok(TokenClassifications { chunks, profile })
    }

    /// [`recognize`](Self::recognize) with default [`RunOptions`] (no profile).
    pub fn recognize_default(&mut self, text: &str) -> Result<TokenClassifications, PipelineError<T, C, M>> {
        self.recognize(text, RunOptions::default())
    }

    /// Tokenize → chunk → recognize multiple texts in one call. Returns one
    /// [`TokenClassifications`] per input text — each carrying its own
    /// [`ChunkTokenClassification`]s with correct positions — plus a shared
    /// batch-level [`RunProfile`] on [`BatchTokenClassifications::profile`].
    pub fn recognize_batch(
        &mut self,
        texts: &[&str],
        opts: RunOptions,
    ) -> Result<BatchTokenClassifications, PipelineError<T, C, M>> {
        let (per_text, profile) = self.run_batch_inner(texts, opts)?;
        let results = per_text.into_iter().map(|chunks| TokenClassifications { chunks, profile: None }).collect();
        Ok(BatchTokenClassifications { results, profile })
    }

    /// [`recognize_batch`](Self::recognize_batch) with default [`RunOptions`].
    pub fn recognize_batch_default(
        &mut self,
        texts: &[&str],
    ) -> Result<BatchTokenClassifications, PipelineError<T, C, M>> {
        self.recognize_batch(texts, RunOptions::default())
    }

    /// Typed accessor for the recognizer (the [`Recognize`] model).
    pub fn recognizer_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

// ─── Span / label decoding (host free-fns) ──────────────────────────────────

/// Label scheme for [`group_spans`]. `Bio`/`Bilou`/`Iobes` reconstruct multi-
/// token entity spans from prefixed labels (`B-PER`, `I-PER`, `L-PER`, `E-PER`,
/// `S-PER`, `O`, …); `Flat` is a no-op — one span per token (POS tagging,
/// chunking without a prefix scheme).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scheme {
    Bio,
    Bilou,
    Iobes,
    Flat,
}

/// One content token's decoded prediction: the argmax label id, its display
/// name, and its source-absolute byte span. Produced by [`labels_for_tokens`];
/// consumed by [`group_spans`]. Special tokens are already filtered out, so
/// `token_index` runs contiguously over the chunk's content tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLabel {
    pub label_id: u32,
    pub label: String,
    pub start: usize,
    pub end: usize,
    /// Index within the content-token vec (contiguous from 0).
    pub token_index: usize,
}

/// One decoded span: a contiguous run of same-type tokens (NER under a prefix
/// scheme) or a single token (POS / `Scheme::Flat`). `start`/`end` are
/// source-absolute byte offsets; `token_range` indexes the content-token vec.
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    pub label: String,
    pub label_id: u32,
    pub start: usize,
    pub end: usize,
    pub token_range: std::ops::Range<usize>,
}

/// Argmax each token's logits, drop special tokens, and pair every surviving
/// content token with its source byte span + display label. `id2label` resolves
/// a label id to its name (e.g. `"B-PER"`); tokens whose `special_tokens_mask`
/// is set are skipped. This is the per-token view (POS / chunking / `none`
/// aggregation); follow with [`group_spans`] to reconstruct entity spans under a
/// prefix scheme.
pub fn labels_for_tokens<F>(chunk: &ChunkTokenClassification, id2label: F) -> Vec<TokenLabel>
where
    F: Fn(u32) -> String,
{
    let nl = chunk.num_labels.max(1);
    let mut out = Vec::new();
    let mut content_idx = 0;
    for (t, (row, special)) in chunk.logits.chunks_exact(nl).zip(&chunk.special_tokens_mask).enumerate() {
        if *special != 0 {
            continue;
        }
        let label_id = argmax_u32(row);
        let (start, end) = chunk.token_offsets.get(t).copied().unwrap_or((0, 0));
        out.push(TokenLabel { label_id, label: id2label(label_id), start, end, token_index: content_idx });
        content_idx += 1;
    }
    out
}

/// Group a content-token label sequence into [`Entity`] spans under the given
/// [`Scheme`]. Scans left-to-right, opening a span on a `B`/`U`/`S` (or a stray
/// `I`/`L`/`E` whose type mismatches the open span — treated leniently as a new
/// opener), extending on matching `I`, and closing on `O` / a type change / an
/// explicit `L`/`E` closer. `Scheme::Flat` emits one entity per token. Broken
/// transitions never panic: a stray `I-` opens a fresh span (HF's lenient
/// `group_entities` convention).
pub fn group_spans(tokens: &[TokenLabel], scheme: Scheme) -> Vec<Entity> {
    if matches!(scheme, Scheme::Flat) {
        return tokens
            .iter()
            .map(|t| Entity {
                label: t.label.clone(),
                label_id: t.label_id,
                start: t.start,
                end: t.end,
                token_range: t.token_index..t.token_index + 1,
            })
            .collect();
    }

    let mut entities: Vec<Entity> = Vec::new();
    // Open span: (type, label_id, start_byte, end_byte, start_token_index)
    let mut open: Option<(String, u32, usize, usize, usize)> = None;

    for t in tokens {
        let (prefix, typ) = parse_tag(&t.label, scheme);
        let typ = typ.unwrap_or_else(|| t.label.clone());
        match prefix {
            Prefix::O => {
                close_span(&mut open, &mut entities, t.token_index);
            }
            Prefix::B | Prefix::U | Prefix::S => {
                close_span(&mut open, &mut entities, t.token_index);
                if matches!(prefix, Prefix::U | Prefix::S) {
                    entities.push(Entity {
                        label: typ,
                        label_id: t.label_id,
                        start: t.start,
                        end: t.end,
                        token_range: t.token_index..t.token_index + 1,
                    });
                } else {
                    open = Some((typ, t.label_id, t.start, t.end, t.token_index));
                }
            }
            Prefix::I => {
                if let Some((ot, _, _, oe, _)) = open.as_mut()
                    && *ot == typ
                {
                    *oe = t.end;
                    continue;
                }
                // Stray I- (no matching open span): lenient — open a fresh span.
                close_span(&mut open, &mut entities, t.token_index);
                open = Some((typ, t.label_id, t.start, t.end, t.token_index));
            }
            Prefix::L | Prefix::E => {
                if let Some((ot, _, _, oe, _)) = open.as_mut()
                    && *ot == typ
                {
                    *oe = t.end;
                    let start_token = open.as_ref().unwrap().4;
                    let lid = open.as_ref().unwrap().1;
                    let start = open.as_ref().unwrap().2;
                    entities.push(Entity {
                        label: typ,
                        label_id: lid,
                        start,
                        end: t.end,
                        token_range: start_token..t.token_index + 1,
                    });
                    open = None;
                    continue;
                }
                // Stray L-/E-: emit a single-token span.
                close_span(&mut open, &mut entities, t.token_index);
                entities.push(Entity {
                    label: typ,
                    label_id: t.label_id,
                    start: t.start,
                    end: t.end,
                    token_range: t.token_index..t.token_index + 1,
                });
            }
        }
    }
    // Flush any span open at end-of-sequence. Its end-token index is the count
    // of content tokens (range exclusive end).
    close_span(&mut open, &mut entities, tokens.len());

    entities
}

/// Tag prefix decoded by [`parse_tag`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Prefix {
    O,
    B,
    I,
    L,
    U,
    E,
    S,
}

/// Split a label string into `(prefix, type)`. Recognizes the prefixes valid for
/// `scheme`; an unrecognized prefix (or a scheme-mismatched one) falls back to
/// `Begin` with the whole label as the type, so flat labels (`"NOUN"`, `"O"`-less)
/// degrade to one span per token rather than panicking.
fn parse_tag(label: &str, scheme: Scheme) -> (Prefix, Option<String>) {
    let (p, t) = match label.split_once('-') {
        Some((p, t)) => (p, Some(t.to_string())),
        None => (label, None),
    };
    let prefix = match p {
        "O" => Prefix::O,
        "B" => Prefix::B,
        "I" => Prefix::I,
        "L" if matches!(scheme, Scheme::Bilou) => Prefix::L,
        "U" if matches!(scheme, Scheme::Bilou) => Prefix::U,
        "E" if matches!(scheme, Scheme::Iobes) => Prefix::E,
        "S" if matches!(scheme, Scheme::Iobes) => Prefix::S,
        _ => return (Prefix::B, Some(label.to_string())),
    };
    (prefix, t)
}

/// Flush an open span (if any) into `out` spanning `[start_token .. end_token)`.
fn close_span(open: &mut Option<(String, u32, usize, usize, usize)>, out: &mut Vec<Entity>, end_token: usize) {
    if let Some((typ, lid, start, end, start_token)) = open.take() {
        out.push(Entity { label: typ, label_id: lid, start, end, token_range: start_token..end_token });
    }
}

/// Index of the maximum element, or `0` for an empty row (tokens have ≥1 label
/// in practice; the fallback keeps a degenerate row total).
fn argmax_u32(row: &[f32]) -> u32 {
    row.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
        .unwrap_or(0)
}
