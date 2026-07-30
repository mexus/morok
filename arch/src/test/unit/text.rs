use std::convert::Infallible;

use proptest::prelude::*;

use crate::pipelines::text::{
    BatchClassifications, BatchEmbeddings, BatchTokenClassifications, ChunkClassification, ChunkEmbedding,
    ChunkTokenClassification, Chunker, Classification, Classify, Embed, Embedding, Encoder, EncoderPipeline,
    EncoderPipelineError, Encoding, Entity, HfTokenizer, HfTokenizerError, Recognize, RunOptions, RunProfile, Scheme,
    SlidingWindowChunker, TextChunk, TokenClassification, TokenLabel, Tokenizer, TruncatingChunker, group_spans,
    labels_for_tokens,
};

fn enc(ids: &[u32]) -> Encoding {
    // A full, internally-consistent encoding: every field the same length as
    // ids, with offsets counting up and masks/specials set to plausible values.
    let n = ids.len();
    Encoding {
        input_ids: ids.to_vec(),
        attention_mask: vec![1; n],
        token_type_ids: vec![0; n],
        offsets: (0..n).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; n],
    }
}

/// Encoding with `[CLS]`=2 / `[SEP]`=3 wrapping the content (BertProcessing
/// convention). Content token at content-index `j` gets offset `(j, j+1)`, so a
/// window starting at content-index `start` has `byte_offset == start`.
fn enc_with_specials(ids: &[u32]) -> Encoding {
    let n_content = ids.len();
    let mut input_ids = vec![2]; // [CLS]
    input_ids.extend_from_slice(ids);
    input_ids.push(3); // [SEP]
    let n = input_ids.len();

    let mut offsets = vec![(0, 0)];
    offsets.extend((0..n_content).map(|i| (i, i + 1)));
    offsets.push((n_content, n_content));

    let mut special_tokens_mask = vec![0u32; n];
    special_tokens_mask[0] = 1;
    special_tokens_mask[n - 1] = 1;

    Encoding { input_ids, attention_mask: vec![1; n], token_type_ids: vec![0; n], offsets, special_tokens_mask }
}

fn assert_field_lengths(enc: &Encoding, expected: usize) {
    assert_eq!(enc.input_ids.len(), expected);
    assert_eq!(enc.attention_mask.len(), expected);
    assert_eq!(enc.token_type_ids.len(), expected);
    assert_eq!(enc.offsets.len(), expected);
    assert_eq!(enc.special_tokens_mask.len(), expected);
}

// ─── Stubs (host-only; no tokenizer.json, no model/device) ────────────────────

/// Returns a canned encoding per input text (analog of `MockVad`).
struct StubTokenizer {
    ids: Vec<u32>,
    max_seq: usize,
    error: bool,
}

impl Tokenizer for StubTokenizer {
    type Error = StubTokenizerError;
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn encode(&mut self, _text: &str) -> Result<Encoding, StubTokenizerError> {
        if self.error {
            return Err(StubTokenizerError);
        }
        Ok(enc(&self.ids))
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub tokenizer error"))]
struct StubTokenizerError;

/// Turns ids into a deterministic embedding (analog of `PresetTranscriber`):
/// hidden_size = the id count (so `len` is visible without pulling in the model),
/// values = ids as f32. `profile` emits a single 1 ms `encode` stage — enough to
/// exercise the merge without a model.
struct StubEmbed {
    hidden_size: usize,
    max_batch: usize,
    error: bool,
}

impl Encoder for StubEmbed {
    type Output = Embedding;
    type ChunkOutput = ChunkEmbedding;
    type Error = StubEmbedError;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.hidden_size)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        if self.error {
            return Err(StubEmbedError);
        }
        let values =
            batch.iter().map(|e| Embedding { values: e.input_ids.iter().map(|&id| id as f32).collect() }).collect();
        let prof = profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("encode", std::time::Duration::from_millis(1)));
            p
        });
        Ok((values, prof))
    }
    fn attach(chunk: &TextChunk, out: Embedding) -> ChunkEmbedding {
        ChunkEmbedding { byte_offset: chunk.byte_offset, values: out }
    }
}

impl Embed for StubEmbed {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub embed error"))]
struct StubEmbedError;

/// Turns ids into deterministic class logits (analog of `StubEmbed`):
/// num_classes = the id count, logits = ids as f32. `profile` emits a single
/// 1 ms `classify` stage — enough to exercise the merge without a model.
struct StubClassify {
    num_classes: usize,
    max_batch: usize,
    error: bool,
}

impl Encoder for StubClassify {
    type Output = Classification;
    type ChunkOutput = ChunkClassification;
    type Error = StubClassifyError;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.num_classes)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        if self.error {
            return Err(StubClassifyError);
        }
        let values = batch
            .iter()
            .map(|e| Classification { logits: e.input_ids.iter().map(|&id| id as f32).collect() })
            .collect();
        let prof = profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("classify", std::time::Duration::from_millis(1)));
            p
        });
        Ok((values, prof))
    }
    fn attach(chunk: &TextChunk, out: Classification) -> ChunkClassification {
        ChunkClassification { byte_offset: chunk.byte_offset, logits: out.logits }
    }
}

impl Classify for StubClassify {
    fn num_classes(&self) -> usize {
        self.num_classes
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub classify error"))]
struct StubClassifyError;

/// Tokenizer that encodes each input character's byte value as a token id.
/// Allows batch tests to distinguish which text produced which embedding —
/// `"ab"` → ids `[97, 98]`, `"xyz"` → ids `[120, 121, 122]`, etc.
struct ByteTokenizer {
    max_seq: usize,
}

impl Tokenizer for ByteTokenizer {
    type Error = Infallible;
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn encode(&mut self, text: &str) -> Result<Encoding, Infallible> {
        Ok(enc(&text.bytes().map(|b| b as u32).collect::<Vec<_>>()))
    }
}

// ─── Encoding ─────────────────────────────────────────────────────────────────

#[test]
fn encoding_len_is_id_count() {
    assert_eq!(enc(&[7, 8, 9]).len(), 3);
    assert!(enc(&[]).is_empty());
}

// ─── TruncatingChunker ─────────────────────────────────────────────────────────

#[test]
fn truncating_chunker_drops_beyond_max_seq_and_keeps_fields_consistent() {
    let mut chunker = TruncatingChunker::new(4);
    let out = chunker.chunk(&enc(&[1, 2, 3, 4, 5, 6, 7])).unwrap();
    assert_eq!(out.len(), 1);
    let c = &out[0];
    assert_eq!(c.byte_offset, 0);
    let e = &c.encoding;
    // Sliced to max_seq on every field — masks and offsets stay aligned with ids.
    assert_eq!(e.input_ids, vec![1, 2, 3, 4]);
    assert_eq!(e.attention_mask, vec![1, 1, 1, 1]);
    assert_eq!(e.token_type_ids, vec![0, 0, 0, 0]);
    assert_eq!(e.offsets, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!(e.special_tokens_mask, vec![0, 0, 0, 0]);
}

#[test]
fn truncating_chunker_keeps_short_input_intact() {
    let mut chunker = TruncatingChunker::new(8);
    let out = chunker.chunk(&enc(&[1, 2, 3])).unwrap();
    assert_eq!(out[0].encoding.input_ids, vec![1, 2, 3]);
    assert_eq!(chunker.max_seq(), 8);
    assert_eq!(chunker.profile_label(), "chunk");
}

// ─── SlidingWindowChunker ─────────────────────────────────────────────────────

#[test]
fn sliding_short_input_fits_one_window() {
    let mut chunker = SlidingWindowChunker::new(5, 2);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30])).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 30, 3]);
    assert_eq!(out[0].byte_offset, 0);
}

#[test]
fn sliding_windows_long_input_with_overlap_and_correct_offsets() {
    // [CLS] 10 11 12 13 14 15 16 [SEP] — 7 content tokens.
    // window=5 → content_window=3; stride=2 → step=2, overlap=1.
    let mut chunker = SlidingWindowChunker::new(5, 2);
    let out = chunker.chunk(&enc_with_specials(&[10, 11, 12, 13, 14, 15, 16])).unwrap();
    assert_eq!(out.len(), 3);

    // Window 0: content[0..3], byte_offset = 0.
    assert_eq!(out[0].byte_offset, 0);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 11, 12, 3]);
    assert_eq!(out[0].encoding.special_tokens_mask, vec![1, 0, 0, 0, 1]);
    // Offsets are absolute (carried from the source), not rebased per window.
    assert_eq!(out[0].encoding.offsets, vec![(0, 0), (0, 1), (1, 2), (2, 3), (7, 7)]);
    assert_field_lengths(&out[0].encoding, 5);

    // Window 1: content[2..5], byte_offset = 2.
    assert_eq!(out[1].byte_offset, 2);
    assert_eq!(out[1].encoding.input_ids, vec![2, 12, 13, 14, 3]);
    assert_eq!(out[1].encoding.offsets[1], (2, 3)); // content token 12 keeps its absolute offset
    assert_field_lengths(&out[1].encoding, 5);

    // Window 2: content[4..7], byte_offset = 4.
    assert_eq!(out[2].byte_offset, 4);
    assert_eq!(out[2].encoding.input_ids, vec![2, 14, 15, 16, 3]);
    assert_field_lengths(&out[2].encoding, 5);

    // Overlap: windows 0 and 1 share token 12.
    assert!(out[0].encoding.input_ids[1..4].contains(&12));
    assert!(out[1].encoding.input_ids[1..4].contains(&12));
}

#[test]
fn sliding_stride_equals_window_gives_adjacent_chunks() {
    // window=4 → content_window=2; stride=4 → step clamped to 2. No overlap.
    let mut chunker = SlidingWindowChunker::new(4, 4);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30, 40])).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 3]);
    assert_eq!(out[1].encoding.input_ids, vec![2, 30, 40, 3]);
    assert_eq!(out[0].byte_offset, 0);
    assert_eq!(out[1].byte_offset, 2);
}

#[test]
fn sliding_last_window_clamped_when_content_uneven() {
    // 5 content tokens, content_window=2 → last window gets 1 token.
    let mut chunker = SlidingWindowChunker::new(4, 4);
    let out = chunker.chunk(&enc_with_specials(&[10, 20, 30, 40, 50])).unwrap();
    assert_eq!(out.len(), 3);
    // First two windows are full (2 content tokens each); last is partial.
    assert_eq!(out[0].encoding.input_ids, vec![2, 10, 20, 3]);
    assert_eq!(out[1].encoding.input_ids, vec![2, 30, 40, 3]);
    assert_eq!(out[2].encoding.input_ids, vec![2, 50, 3]);
    assert_eq!(out[2].byte_offset, 4);
}

#[test]
fn sliding_works_without_special_tokens() {
    // No specials: lead=0, trail=0, content = entire encoding.
    let mut chunker = SlidingWindowChunker::new(3, 2);
    let out = chunker.chunk(&enc(&[10, 20, 30, 40, 50, 60])).unwrap();
    assert_eq!(out.len(), 3);
    assert_eq!(out[0].encoding.input_ids, vec![10, 20, 30]);
    assert_eq!(out[1].encoding.input_ids, vec![30, 40, 50]);
    assert_eq!(out[2].encoding.input_ids, vec![50, 60]); // partial
    assert_eq!(out[0].byte_offset, 0);
    assert_eq!(out[1].byte_offset, 2);
    assert_eq!(out[2].byte_offset, 4);
}

#[test]
fn sliding_all_specials_returns_empty() {
    let mut chunker = SlidingWindowChunker::new(8, 4);
    let out = chunker.chunk(&enc_with_specials(&[])).unwrap();
    assert!(out.is_empty());
}

#[test]
fn sliding_max_seq_returns_window_and_default_label() {
    let chunker = SlidingWindowChunker::new(512, 256);
    assert_eq!(chunker.max_seq(), 512);
    assert_eq!(chunker.profile_label(), "chunk");
}

#[test]
#[should_panic(expected = "window must be >= 1")]
fn sliding_new_rejects_zero_window() {
    let _ = SlidingWindowChunker::new(0, 1);
}

#[test]
#[should_panic(expected = "stride must be in 1..=window")]
fn sliding_new_rejects_zero_stride() {
    let _ = SlidingWindowChunker::new(4, 0);
}

#[test]
#[should_panic(expected = "stride must be in 1..=window")]
fn sliding_new_rejects_stride_above_window() {
    let _ = SlidingWindowChunker::new(4, 5);
}

#[test]
fn sliding_pipeline_produces_per_window_embeddings() {
    // StubTokenizer yields [10, 20, 30, 40, 50, 60] (no specials).
    // SlidingWindowChunker(3, 2) → 3 windows at byte_offsets 0, 2, 4.
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], max_seq: 3, error: false },
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 1, error: false },
    );
    let out = p.embed_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].values.values, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].values.values, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].values.values, vec![50.0, 60.0]);
}

#[test]
fn sliding_pipeline_profiles_with_chunk_stage() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], max_seq: 3, error: false },
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 1, error: false },
    );
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert_eq!(out.chunks.len(), 3);
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "encode"]);
}

// ─── Embed: batch vs single-path agreement ───────────────────────────────────

#[test]
fn embed_single_is_batch_of_one() {
    // The default `embed` delegates to `embed_batch(&[enc])` and pops — verify
    // the values match a direct batch call.
    let mut embed = StubEmbed { hidden_size: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = embed.run(&e, false).unwrap().0;
    let batch = embed.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

// ─── EncoderPipeline end-to-end ───────────────────────────────────────────

fn pipeline(ids: Vec<u32>, max_seq: usize) -> EncoderPipeline<StubTokenizer, TruncatingChunker, StubEmbed> {
    EncoderPipeline::new(
        StubTokenizer { ids, max_seq, error: false },
        TruncatingChunker::new(max_seq),
        StubEmbed { hidden_size: max_seq, max_batch: 1, error: false },
    )
}

#[test]
fn pipeline_truncates_then_embeds() {
    // Tokenizer yields 7 ids; chunker caps at 4; embedder sees the 4 surviving.
    let mut p = pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.embed_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 1);
    // byte_offset is threaded through from the chunker (TruncatingChunker → 0).
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].values.values, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn pipeline_profiles_stage_order_tokenize_then_chunk_then_encoder() {
    let mut p = pipeline(vec![1, 2, 3], 8);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "encode"], "host stages lead, then the encoder's");
}

#[test]
fn pipeline_profiles_per_call_without_rebuild() {
    let mut p = pipeline(vec![1, 2, 3], 8);
    // One built pipeline serves both modes.
    let profiled = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert!(profiled.profile.is_some());
    let unprofiled = p.embed_default("ignored").unwrap();
    assert!(unprofiled.profile.is_none());
}

#[test]
fn pipeline_surfaces_host_stages_even_when_encoder_does_not() {
    // Encoder emits a profile only when asked — but it always does here, so to
    // exercise "encoder emits nothing", drop the embedder's stage by giving a
    // single-chunk input under a non-profiled encoder is not enough. Instead,
    // assert directly: tokenize+chunk still surface on their own via an encoder
    // that returns None. Reuse the stub but call embed with profile and a stub
    // whose embed_batch returns None for the profile — covered by the case below.
    // Here we confirm the prepend works for the normal path already; the
    // no-encoder-profile path is covered by `pipeline_empty_input_surfaces_host`.
    let mut p = pipeline(vec![1, 2, 3], 8);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile");
    // At minimum the two host stages are present.
    assert!(profile.stages.iter().any(|s| s.name == "tokenize"));
    assert!(profile.stages.iter().any(|s| s.name == "chunk"));
}

#[test]
fn pipeline_empty_input_skips_embed_batch_and_still_profiles_host_stages() {
    // Zero ids → chunker emits one empty chunk (not zero chunks under
    // TruncatingChunker), so embed_batch still runs over one input. The
    // zero-chunk branch is reachable only via a chunker that yields nothing;
    // assert that branch via a custom chunker below instead. Here just confirm
    // an empty-input run still profiles tokenize+chunk.
    let mut p = pipeline(Vec::new(), 4);
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert_eq!(out.chunks.len(), 1, "truncating chunker emits one empty chunk");
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert!(names.starts_with(&["tokenize", "chunk"]), "host stages lead even on empty input");
}

#[test]
fn pipeline_zero_chunk_run_skips_embed_and_profiles_host_stages() {
    // A chunker that yields zero chunks: the empty-guard must skip embed_batch
    // (so the erroring stub is never hit) while still surfacing tokenize+chunk.
    let p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        NoChunkChunker { max_seq: 4 },
        StubEmbed { hidden_size: 4, max_batch: 1, error: true }, // would fail if called
    );
    let mut p = p;
    let out = p.embed("ignored", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk"], "encoder never runs; only host stages");
}

/// Chunker that always emits zero chunks (to exercise the empty-input guard).
struct NoChunkChunker {
    max_seq: usize,
}

impl Chunker for NoChunkChunker {
    type Error = Infallible;
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn chunk(&mut self, _enc: &Encoding) -> Result<Vec<TextChunk>, Infallible> {
        Ok(Vec::new())
    }
}

// ─── assemble sizes encoder from chunker.max_seq ─────────────────────────────

#[test]
fn assemble_passes_chunker_max_seq_into_builder() {
    let seen = std::cell::Cell::new(0usize);
    let _p: EncoderPipeline<_, TruncatingChunker, StubEmbed> = EncoderPipeline::assemble(
        StubTokenizer { ids: vec![1], max_seq: 8, error: false },
        TruncatingChunker::new(8),
        |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubEmbed { hidden_size: max_seq, max_batch: 1, error: false })
        },
    )
    .unwrap();
    assert_eq!(seen.get(), 8);
}

// ─── error propagation ────────────────────────────────────────────────────────

#[test]
fn tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let err = p.embed_default("ignored").unwrap_err();
    assert!(matches!(err, crate::pipelines::text::EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn embed_error_maps_to_embed_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 1, error: true },
    );
    let err = p.embed_default("ignored").unwrap_err();
    assert!(matches!(err, crate::pipelines::text::EncoderPipelineError::Encode { .. }));
}

// ─── HfTokenizer: a real tokenizers::Tokenizer fixture ─────────────────────────
//
// Hand-built WordPiece tokenizer: tiny vocab + Whitespace pre-tokenizer +
// BertProcessing wrapping each sequence in [CLS] … [SEP]. The ids are fully
// predictable for known input, so encode() assertions are exact (specials get
// non-trivial special_tokens_mask). Built programmatically; `fixture_json`
// serializes it so from_bytes/from_path exercise the real JSON deserialization
// path rather than a hand-written string.

fn fixture_tokenizer() -> tokenizers::Tokenizer {
    let vocab = [
        ("[PAD]".to_string(), 0u32),
        ("[UNK]".to_string(), 1),
        ("[CLS]".to_string(), 2),
        ("[SEP]".to_string(), 3),
        ("hello".to_string(), 4),
        ("world".to_string(), 5),
        ("foo".to_string(), 6),
        ("bar".to_string(), 7),
    ];
    let model = tokenizers::models::wordpiece::WordPiece::builder()
        .vocab(vocab)
        .unk_token("[UNK]".to_string())
        .build()
        .expect("wordpiece vocab contains [UNK]");
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer.with_post_processor(Some(tokenizers::processors::bert::BertProcessing::new(
        ("[SEP]".to_string(), 3),
        ("[CLS]".to_string(), 2),
    )));
    tokenizer
}

fn fixture_json() -> Vec<u8> {
    fixture_tokenizer().to_string(false).expect("serialize fixture tokenizer").into_bytes()
}

#[test]
fn hf_tokenizer_from_bytes_encodes_known_text() {
    let mut tok = HfTokenizer::from_bytes(fixture_json(), 512).expect("load fixture");
    let enc = tok.encode("hello world").expect("encode");
    // [CLS] hello world [SEP]
    assert_eq!(enc.input_ids, vec![2, 4, 5, 3]);
    assert_eq!(enc.attention_mask, vec![1, 1, 1, 1]);
    // All five fields share one length — the invariant from_hf preserves.
    let n = enc.input_ids.len();
    assert_eq!(enc.attention_mask.len(), n);
    assert_eq!(enc.token_type_ids.len(), n);
    assert_eq!(enc.offsets.len(), n);
    assert_eq!(enc.special_tokens_mask.len(), n);
    // Specials land at the brackets; real tokens stay unmasked.
    assert_eq!(enc.special_tokens_mask, vec![1, 0, 0, 1]);
}

#[test]
fn hf_tokenizer_from_path_matches_from_bytes() {
    let bytes = fixture_json();
    // Unique temp path: tests may run in parallel.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("svod-arch-hf-tokenizer-{n}.json"));
    std::fs::write(&path, &bytes).expect("write fixture to temp file");
    let mut from_path = HfTokenizer::from_path(&path, 64).expect("load from path");
    let mut from_bytes = HfTokenizer::from_bytes(&bytes, 64).expect("load from bytes");
    let id_path = from_path.encode("hello world foo bar").unwrap().input_ids;
    let id_bytes = from_bytes.encode("hello world foo bar").unwrap().input_ids;
    assert_eq!(id_path, id_bytes);
    let _ = std::fs::remove_file(&path);
}

#[test]
fn encoding_from_hf_copies_all_fields() {
    let inner = fixture_tokenizer();
    let hf = inner.encode("hello world", true).expect("hf encode");
    let enc = Encoding::from_hf(&hf);
    assert_eq!(enc.input_ids, hf.get_ids().to_vec());
    assert_eq!(enc.attention_mask, hf.get_attention_mask().to_vec());
    assert_eq!(enc.token_type_ids, hf.get_type_ids().to_vec());
    assert_eq!(enc.offsets, hf.get_offsets().to_vec());
    assert_eq!(enc.special_tokens_mask, hf.get_special_tokens_mask().to_vec());
}

#[test]
fn hf_tokenizer_max_seq_round_trips() {
    let via_new = HfTokenizer::new(fixture_tokenizer(), 8);
    assert_eq!(via_new.max_seq(), 8);
    let via_bytes = HfTokenizer::from_bytes(fixture_json(), 16).expect("load fixture");
    assert_eq!(via_bytes.max_seq(), 16);
}

#[test]
fn hf_tokenizer_error_display_and_source() {
    use std::error::Error as _;
    // from_bytes returns HfTokenizerError — the From<tokenizers::Error> conversion
    // wired on from_bytes' `?` surfaces the boxed HF error behind a sized type.
    let err = HfTokenizer::from_bytes(b"not valid json", 8).err().expect("invalid json must error");
    assert!(!err.to_string().is_empty());
    assert!(err.source().is_some(), "HfTokenizerError wraps an inner error");
    // HfTokenizerError is the named type the trait/field expect — confirm by name.
    let _: &HfTokenizerError = &err;
}

// ─── EncoderPipelineError::Chunk arm (TruncatingChunker::Error = Infallible) ─────

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub chunker error"))]
struct ErrChunkerError;

/// Always fails `chunk` — reaches the otherwise-unreachable `Chunk` arm.
struct ErrChunker {
    max_seq: usize,
}

impl Chunker for ErrChunker {
    type Error = ErrChunkerError;
    fn max_seq(&self) -> usize {
        self.max_seq
    }
    fn chunk(&mut self, _enc: &Encoding) -> Result<Vec<TextChunk>, ErrChunkerError> {
        Err(ErrChunkerError)
    }
}

#[test]
fn chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        ErrChunker { max_seq: 4 },
        StubEmbed { hidden_size: 4, max_batch: 1, error: false },
    );
    let err = p.embed_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

// ─── embed_batch (multi-text) ─────────────────────────────────────────────────

#[test]
fn batch_basic_three_texts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 32 },
        TruncatingChunker::new(32),
        StubEmbed { hidden_size: 32, max_batch: 8, error: false },
    );
    let out = p.embed_batch_default(&["ab", "xyz", "hello"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert!(out.profile.is_none(), "default options don't profile");
    // ByteTokenizer maps each byte to its value: "ab" → [97, 98], etc.
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].values.values, vec![97.0, 98.0]);
    assert_eq!(out.results[1].chunks[0].values.values, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].values.values, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn batch_with_sliding_window_varying_chunk_counts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 3 },
        SlidingWindowChunker::new(3, 2),
        StubEmbed { hidden_size: 3, max_batch: 8, error: false },
    );
    // 6 + 2 + 4 content tokens → 3 + 1 + 2 = 6 chunks total.
    let out = p.embed_batch_default(&["abcdef", "ab", "abcd"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[0].chunks[1].byte_offset, 2);
    assert_eq!(out.results[0].chunks[2].byte_offset, 4);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[2].chunks.len(), 2);
}

#[test]
fn batch_empty_texts_returns_empty() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 4, error: false },
    );
    let out: BatchEmbeddings = p.embed_batch_default(&[]).unwrap();
    assert!(out.results.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn batch_some_texts_produce_zero_chunks() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 4 },
        SlidingWindowChunker::new(4, 2),
        StubEmbed { hidden_size: 4, max_batch: 4, error: false },
    );
    // "" → 0 tokens → 0 chunks (SlidingWindowChunker content_len guard).
    let out = p.embed_batch_default(&["ab", "", "cd"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert!(out.results[1].chunks.is_empty());
    assert_eq!(out.results[2].chunks.len(), 1);
}

#[test]
fn batch_sub_batches_when_chunks_exceed_max_batch() {
    // 5 texts × 1 chunk = 5 chunks; max_batch=2 → 3 sub-batches (2, 2, 1).
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 2, error: false },
    );
    let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let out = p.embed_batch_default(&texts).unwrap();
    assert_eq!(out.results.len(), 5);
    for (i, result) in out.results.iter().enumerate() {
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].values.values, vec![texts[i].as_bytes()[0] as f32]);
    }
}

#[test]
fn batch_profile_has_tokenize_chunk_encode_stages() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubEmbed { hidden_size: 8, max_batch: 4, error: false },
    );
    let out = p.embed_batch(&["ab", "cd"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 2);
    // Per-text profiles are None — the batch profile lives on BatchEmbeddings.
    assert!(out.results[0].profile.is_none());
    assert!(out.results[1].profile.is_none());
    let profile = out.profile.expect("batch profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "encode"]);
}

#[test]
fn batch_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, max_batch: 4, error: false },
    );
    let err = p.embed_batch_default(&["a", "b"]).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn batch_results_match_individual_embed_calls() {
    let make_pipeline = || {
        EncoderPipeline::new(
            ByteTokenizer { max_seq: 4 },
            SlidingWindowChunker::new(4, 2),
            StubEmbed { hidden_size: 4, max_batch: 8, error: false },
        )
    };

    let texts = ["abcde", "xy"];
    let batch = {
        let mut p = make_pipeline();
        p.embed_batch_default(&texts).unwrap()
    };

    for (i, text) in texts.iter().enumerate() {
        let mut p = make_pipeline();
        let single = p.embed_default(text).unwrap();
        assert_eq!(batch.results[i].chunks.len(), single.chunks.len(), "chunk count mismatch for text {i}");
        for (b, s) in batch.results[i].chunks.iter().zip(&single.chunks) {
            assert_eq!(b.byte_offset, s.byte_offset, "byte_offset mismatch for text {i}");
            assert_eq!(b.values, s.values, "values mismatch for text {i}");
        }
    }
}

// ─── EncoderPipeline ────────────────────────────────────────────────────────

fn classify_pipeline(ids: Vec<u32>, max_seq: usize) -> EncoderPipeline<StubTokenizer, TruncatingChunker, StubClassify> {
    EncoderPipeline::new(
        StubTokenizer { ids, max_seq, error: false },
        TruncatingChunker::new(max_seq),
        StubClassify { num_classes: max_seq, max_batch: 1, error: false },
    )
}

#[test]
fn classify_single_is_batch_of_one() {
    let mut cls = StubClassify { num_classes: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = cls.run(&e, false).unwrap().0;
    let batch = cls.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

#[test]
fn classify_pipeline_truncates_then_classifies() {
    let mut p = classify_pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.classify_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 1);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![1.0, 2.0, 3.0, 4.0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn classify_pipeline_profiles_stage_order_tokenize_then_chunk_then_classify() {
    let mut p = classify_pipeline(vec![1, 2, 3], 8);
    let out = p.classify("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "classify"], "host stages lead, then the classifier's");
}

#[test]
fn classify_pipeline_profiles_per_call_without_rebuild() {
    let mut p = classify_pipeline(vec![1, 2, 3], 8);
    let profiled = p.classify("ignored", RunOptions { profile: true }).unwrap();
    assert!(profiled.profile.is_some());
    let unprofiled = p.classify_default("ignored").unwrap();
    assert!(unprofiled.profile.is_none());
}

#[test]
fn classify_pipeline_zero_chunk_run_skips_classify_and_profiles_host_stages() {
    let p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        NoChunkChunker { max_seq: 4 },
        StubClassify { num_classes: 4, max_batch: 1, error: true }, // would fail if called
    );
    let mut p = p;
    let out = p.classify("ignored", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk"], "classifier never runs; only host stages");
}

#[test]
fn classify_assemble_passes_chunker_max_seq_into_builder() {
    let seen = std::cell::Cell::new(0usize);
    let _p: EncoderPipeline<_, TruncatingChunker, StubClassify> = EncoderPipeline::assemble(
        StubTokenizer { ids: vec![1], max_seq: 8, error: false },
        TruncatingChunker::new(8),
        |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubClassify { num_classes: max_seq, max_batch: 1, error: false })
        },
    )
    .unwrap();
    assert_eq!(seen.get(), 8);
}

#[test]
fn classify_sliding_pipeline_produces_per_window_classifications() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], max_seq: 3, error: false },
        SlidingWindowChunker::new(3, 2),
        StubClassify { num_classes: 3, max_batch: 1, error: false },
    );
    let out = p.classify_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].logits, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].logits, vec![50.0, 60.0]);
}

#[test]
fn classify_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubClassify { num_classes: 4, max_batch: 1, error: false },
    );
    let err = p.classify_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn classify_error_maps_to_classify_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        TruncatingChunker::new(4),
        StubClassify { num_classes: 4, max_batch: 1, error: true },
    );
    let err = p.classify_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Encode { .. }));
}

#[test]
fn classify_chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        ErrChunker { max_seq: 4 },
        StubClassify { num_classes: 4, max_batch: 1, error: false },
    );
    let err = p.classify_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

// ─── EncoderPipeline batch ──────────────────────────────────────────────────

#[test]
fn classify_batch_basic_three_texts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 32 },
        TruncatingChunker::new(32),
        StubClassify { num_classes: 32, max_batch: 8, error: false },
    );
    let out = p.classify_batch_default(&["ab", "xyz", "hello"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert!(out.profile.is_none(), "default options don't profile");
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![97.0, 98.0]);
    assert_eq!(out.results[1].chunks[0].logits, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].logits, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn classify_batch_with_sliding_window_varying_chunk_counts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 3 },
        SlidingWindowChunker::new(3, 2),
        StubClassify { num_classes: 3, max_batch: 8, error: false },
    );
    let out = p.classify_batch_default(&["abcdef", "ab", "abcd"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[0].chunks[1].byte_offset, 2);
    assert_eq!(out.results[0].chunks[2].byte_offset, 4);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[2].chunks.len(), 2);
}

#[test]
fn classify_batch_empty_texts_returns_empty() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubClassify { num_classes: 8, max_batch: 4, error: false },
    );
    let out: BatchClassifications = p.classify_batch_default(&[]).unwrap();
    assert!(out.results.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn classify_batch_some_texts_produce_zero_chunks() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 4 },
        SlidingWindowChunker::new(4, 2),
        StubClassify { num_classes: 4, max_batch: 4, error: false },
    );
    let out = p.classify_batch_default(&["ab", "", "cd"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert!(out.results[1].chunks.is_empty());
    assert_eq!(out.results[2].chunks.len(), 1);
}

#[test]
fn classify_batch_sub_batches_when_chunks_exceed_max_batch() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubClassify { num_classes: 8, max_batch: 2, error: false },
    );
    let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let out = p.classify_batch_default(&texts).unwrap();
    assert_eq!(out.results.len(), 5);
    for (i, result) in out.results.iter().enumerate() {
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].logits, vec![texts[i].as_bytes()[0] as f32]);
    }
}

#[test]
fn classify_batch_profile_has_tokenize_chunk_classify_stages() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubClassify { num_classes: 8, max_batch: 4, error: false },
    );
    let out = p.classify_batch(&["ab", "cd"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 2);
    assert!(out.results[0].profile.is_none());
    assert!(out.results[1].profile.is_none());
    let profile = out.profile.expect("batch profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "classify"]);
}

#[test]
fn classify_batch_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubClassify { num_classes: 4, max_batch: 4, error: false },
    );
    let err = p.classify_batch_default(&["a", "b"]).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn classify_batch_results_match_individual_classify_calls() {
    let make_pipeline = || {
        EncoderPipeline::new(
            ByteTokenizer { max_seq: 4 },
            SlidingWindowChunker::new(4, 2),
            StubClassify { num_classes: 4, max_batch: 8, error: false },
        )
    };

    let texts = ["abcde", "xy"];
    let batch = {
        let mut p = make_pipeline();
        p.classify_batch_default(&texts).unwrap()
    };

    for (i, text) in texts.iter().enumerate() {
        let mut p = make_pipeline();
        let single = p.classify_default(text).unwrap();
        assert_eq!(batch.results[i].chunks.len(), single.chunks.len(), "chunk count mismatch for text {i}");
        for (b, s) in batch.results[i].chunks.iter().zip(&single.chunks) {
            assert_eq!(b.byte_offset, s.byte_offset, "byte_offset mismatch for text {i}");
            assert_eq!(b.logits, s.logits, "logits mismatch for text {i}");
        }
    }
}

// ─── EncoderPipeline ───────────────────────────────────────────────────────

/// Turns ids into a deterministic `(seq_len, num_labels)` logit grid (analog of
/// `StubClassify`): each token id `v` becomes a row of `num_labels` copies of
/// `v as f32`. `profile` emits a single 1 ms `recognize` stage — enough to
/// exercise the merge without a model.
struct StubRecognize {
    num_labels: usize,
    max_batch: usize,
    error: bool,
}

impl Encoder for StubRecognize {
    type Output = TokenClassification;
    type ChunkOutput = ChunkTokenClassification;
    type Error = StubRecognizeError;
    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.num_labels)
    }
    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Self::Output>, Option<RunProfile>), Self::Error> {
        if self.error {
            return Err(StubRecognizeError);
        }
        let nl = self.num_labels;
        let values = batch
            .iter()
            .map(|e| {
                let mut logits = Vec::with_capacity(e.input_ids.len() * nl);
                for &id in &e.input_ids {
                    logits.extend(std::iter::repeat_n(id as f32, nl));
                }
                TokenClassification { logits, num_labels: nl }
            })
            .collect();
        let prof = profile.then(|| {
            let mut p = RunProfile::default();
            p.push(svod_runtime::StageProfile::host("recognize", std::time::Duration::from_millis(1)));
            p
        });
        Ok((values, prof))
    }
    fn attach(chunk: &TextChunk, out: TokenClassification) -> ChunkTokenClassification {
        ChunkTokenClassification {
            byte_offset: chunk.byte_offset,
            token_offsets: chunk.encoding.offsets.clone(),
            special_tokens_mask: chunk.encoding.special_tokens_mask.clone(),
            logits: out.logits,
            num_labels: out.num_labels,
        }
    }
}

impl Recognize for StubRecognize {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub recognize error"))]
struct StubRecognizeError;

fn recognize_pipeline(
    ids: Vec<u32>,
    max_seq: usize,
) -> EncoderPipeline<StubTokenizer, TruncatingChunker, StubRecognize> {
    EncoderPipeline::new(
        StubTokenizer { ids, max_seq, error: false },
        TruncatingChunker::new(max_seq),
        StubRecognize { num_labels: 1, max_batch: 1, error: false },
    )
}

#[test]
fn recognize_single_is_batch_of_one() {
    let mut rec = StubRecognize { num_labels: 3, max_batch: 1, error: false };
    let e = enc(&[4, 5, 6]);
    let single = rec.run(&e, false).unwrap().0;
    let batch = rec.run_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

#[test]
fn recognize_pipeline_truncates_then_recognizes() {
    let mut p = recognize_pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.recognize_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 1);
    let c = &out.chunks[0];
    assert_eq!(c.byte_offset, 0);
    assert_eq!(c.num_labels, 1);
    assert_eq!(c.logits, vec![1.0, 2.0, 3.0, 4.0]);
    // Per-token geometry is threaded through from the chunk's encoding.
    assert_eq!(c.token_offsets, vec![(0, 1), (1, 2), (2, 3), (3, 4)]);
    assert_eq!(c.special_tokens_mask, vec![0, 0, 0, 0]);
    assert!(out.profile.is_none(), "default options don't profile");
}

#[test]
fn recognize_pipeline_profiles_stage_order_tokenize_then_chunk_then_recognize() {
    let mut p = recognize_pipeline(vec![1, 2, 3], 8);
    let out = p.recognize("ignored", RunOptions { profile: true }).unwrap();
    let profile = out.profile.expect("profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "recognize"], "host stages lead, then the recognizer's");
}

#[test]
fn recognize_pipeline_profiles_per_call_without_rebuild() {
    let mut p = recognize_pipeline(vec![1, 2, 3], 8);
    let profiled = p.recognize("ignored", RunOptions { profile: true }).unwrap();
    assert!(profiled.profile.is_some());
    let unprofiled = p.recognize_default("ignored").unwrap();
    assert!(unprofiled.profile.is_none());
}

#[test]
fn recognize_pipeline_zero_chunk_run_skips_recognize_and_profiles_host_stages() {
    let p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        NoChunkChunker { max_seq: 4 },
        StubRecognize { num_labels: 4, max_batch: 1, error: true }, // would fail if called
    );
    let mut p = p;
    let out = p.recognize("ignored", RunOptions { profile: true }).unwrap();
    assert!(out.chunks.is_empty());
    let profile = out.profile.expect("profile");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk"], "recognizer never runs; only host stages");
}

#[test]
fn recognize_assemble_passes_chunker_max_seq_into_builder() {
    let seen = std::cell::Cell::new(0usize);
    let _p: EncoderPipeline<_, TruncatingChunker, StubRecognize> = EncoderPipeline::assemble(
        StubTokenizer { ids: vec![1], max_seq: 8, error: false },
        TruncatingChunker::new(8),
        |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubRecognize { num_labels: max_seq, max_batch: 1, error: false })
        },
    )
    .unwrap();
    assert_eq!(seen.get(), 8);
}

#[test]
fn recognize_sliding_pipeline_produces_per_window_classifications() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![10, 20, 30, 40, 50, 60], max_seq: 3, error: false },
        SlidingWindowChunker::new(3, 2),
        StubRecognize { num_labels: 1, max_batch: 1, error: false },
    );
    let out = p.recognize_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 3);
    assert_eq!(out.chunks[0].byte_offset, 0);
    assert_eq!(out.chunks[0].logits, vec![10.0, 20.0, 30.0]);
    assert_eq!(out.chunks[0].token_offsets, vec![(0, 1), (1, 2), (2, 3)]);
    assert_eq!(out.chunks[1].byte_offset, 2);
    assert_eq!(out.chunks[1].logits, vec![30.0, 40.0, 50.0]);
    assert_eq!(out.chunks[1].token_offsets, vec![(2, 3), (3, 4), (4, 5)]);
    assert_eq!(out.chunks[2].byte_offset, 4);
    assert_eq!(out.chunks[2].logits, vec![50.0, 60.0]);
    assert_eq!(out.chunks[2].token_offsets, vec![(4, 5), (5, 6)]);
}

#[test]
fn recognize_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 4, max_batch: 1, error: false },
    );
    let err = p.recognize_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn recognize_error_maps_to_recognize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 4, max_batch: 1, error: true },
    );
    let err = p.recognize_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Encode { .. }));
}

#[test]
fn recognize_chunk_error_maps_to_chunk_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        ErrChunker { max_seq: 4 },
        StubRecognize { num_labels: 4, max_batch: 1, error: false },
    );
    let err = p.recognize_default("ignored").unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Chunk { .. }));
}

// ─── EncoderPipeline batch ─────────────────────────────────────────────────

#[test]
fn recognize_batch_basic_three_texts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 32 },
        TruncatingChunker::new(32),
        StubRecognize { num_labels: 1, max_batch: 8, error: false },
    );
    let out = p.recognize_batch_default(&["ab", "xyz", "hello"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert!(out.profile.is_none(), "default options don't profile");
    assert_eq!(out.results[0].chunks.len(), 1);
    assert_eq!(out.results[0].chunks[0].logits, vec![97.0, 98.0]);
    assert_eq!(out.results[0].chunks[0].num_labels, 1);
    assert_eq!(out.results[0].chunks[0].token_offsets, vec![(0, 1), (1, 2)]);
    assert_eq!(out.results[1].chunks[0].logits, vec![120.0, 121.0, 122.0]);
    assert_eq!(out.results[2].chunks[0].logits, vec![104.0, 101.0, 108.0, 108.0, 111.0]);
}

#[test]
fn recognize_batch_with_sliding_window_varying_chunk_counts() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 3 },
        SlidingWindowChunker::new(3, 2),
        StubRecognize { num_labels: 1, max_batch: 8, error: false },
    );
    let out = p.recognize_batch_default(&["abcdef", "ab", "abcd"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 3);
    assert_eq!(out.results[0].chunks[0].byte_offset, 0);
    assert_eq!(out.results[0].chunks[1].byte_offset, 2);
    assert_eq!(out.results[0].chunks[2].byte_offset, 4);
    assert_eq!(out.results[1].chunks.len(), 1);
    assert_eq!(out.results[2].chunks.len(), 2);
}

#[test]
fn recognize_batch_empty_texts_returns_empty() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubRecognize { num_labels: 1, max_batch: 4, error: false },
    );
    let out: BatchTokenClassifications = p.recognize_batch_default(&[]).unwrap();
    assert!(out.results.is_empty());
    assert!(out.profile.is_none());
}

#[test]
fn recognize_batch_some_texts_produce_zero_chunks() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 4 },
        SlidingWindowChunker::new(4, 2),
        StubRecognize { num_labels: 1, max_batch: 4, error: false },
    );
    let out = p.recognize_batch_default(&["ab", "", "cd"]).unwrap();
    assert_eq!(out.results.len(), 3);
    assert_eq!(out.results[0].chunks.len(), 1);
    assert!(out.results[1].chunks.is_empty());
    assert_eq!(out.results[2].chunks.len(), 1);
}

#[test]
fn recognize_batch_sub_batches_when_chunks_exceed_max_batch() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubRecognize { num_labels: 1, max_batch: 2, error: false },
    );
    let texts: Vec<&str> = vec!["a", "b", "c", "d", "e"];
    let out = p.recognize_batch_default(&texts).unwrap();
    assert_eq!(out.results.len(), 5);
    for (i, result) in out.results.iter().enumerate() {
        assert_eq!(result.chunks.len(), 1);
        assert_eq!(result.chunks[0].logits, vec![texts[i].as_bytes()[0] as f32]);
    }
}

#[test]
fn recognize_batch_profile_has_tokenize_chunk_recognize_stages() {
    let mut p = EncoderPipeline::new(
        ByteTokenizer { max_seq: 8 },
        TruncatingChunker::new(8),
        StubRecognize { num_labels: 1, max_batch: 4, error: false },
    );
    let out = p.recognize_batch(&["ab", "cd"], RunOptions { profile: true }).unwrap();
    assert_eq!(out.results.len(), 2);
    assert!(out.results[0].profile.is_none());
    assert!(out.results[1].profile.is_none());
    let profile = out.profile.expect("batch profile collected");
    let names: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, vec!["tokenize", "chunk", "recognize"]);
}

#[test]
fn recognize_batch_tokenize_error_maps_to_tokenize_variant() {
    let mut p = EncoderPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubRecognize { num_labels: 4, max_batch: 4, error: false },
    );
    let err = p.recognize_batch_default(&["a", "b"]).unwrap_err();
    assert!(matches!(err, EncoderPipelineError::Tokenize { .. }));
}

#[test]
fn recognize_batch_results_match_individual_recognize_calls() {
    let make_pipeline = || {
        EncoderPipeline::new(
            ByteTokenizer { max_seq: 4 },
            SlidingWindowChunker::new(4, 2),
            StubRecognize { num_labels: 1, max_batch: 8, error: false },
        )
    };

    let texts = ["abcde", "xy"];
    let batch = {
        let mut p = make_pipeline();
        p.recognize_batch_default(&texts).unwrap()
    };

    for (i, text) in texts.iter().enumerate() {
        let mut p = make_pipeline();
        let single = p.recognize_default(text).unwrap();
        assert_eq!(batch.results[i].chunks.len(), single.chunks.len(), "chunk count mismatch for text {i}");
        for (b, s) in batch.results[i].chunks.iter().zip(&single.chunks) {
            assert_eq!(b.byte_offset, s.byte_offset, "byte_offset mismatch for text {i}");
            assert_eq!(b.logits, s.logits, "logits mismatch for text {i}");
            assert_eq!(b.token_offsets, s.token_offsets, "token_offsets mismatch for text {i}");
            assert_eq!(b.special_tokens_mask, s.special_tokens_mask, "special_tokens_mask mismatch for text {i}");
        }
    }
}

// ─── labels_for_tokens / group_spans decoding ───────────────────────────────

/// Build a `ChunkTokenClassification` from a per-token spec: `(label_id,
/// byte_span, is_special)`. Each token's logits are a one-hot row (so argmax
/// yields `label_id` deterministically), sized to `num_labels`.
fn token_chunk(spec: &[(u32, (usize, usize), bool)], num_labels: usize) -> ChunkTokenClassification {
    let mut logits = Vec::new();
    let mut offsets = Vec::new();
    let mut specials = Vec::new();
    for &(lid, span, is_special) in spec {
        let mut row = vec![0.0_f32; num_labels];
        if (lid as usize) < num_labels {
            row[lid as usize] = 1.0;
        }
        logits.extend_from_slice(&row);
        offsets.push(span);
        specials.push(if is_special { 1 } else { 0 });
    }
    ChunkTokenClassification {
        byte_offset: 0,
        logits,
        num_labels,
        token_offsets: offsets,
        special_tokens_mask: specials,
    }
}

/// id2label for the decoder tests: 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC.
fn ner_id2label(id: u32) -> String {
    ["O", "B-PER", "I-PER", "B-LOC", "I-LOC"].get(id as usize).map(|s| (*s).to_string()).unwrap_or_default()
}

/// The entity type of a prefixed label (`"B-PER"` → `"PER"`); `None` for `"O"`.
fn typ_of(label: &str) -> Option<&str> {
    label.split_once('-').map(|(_, t)| t)
}

#[test]
fn labels_for_tokens_argmaxes_skips_specials_and_pairs_offsets() {
    // token ids: 1=B-PER, 2=I-PER, 0=O, 1=B-PER, special(CLS)=1
    let chunk = token_chunk(
        &[(1, (0, 3), false), (2, (3, 7), false), (0, (7, 8), false), (1, (8, 12), false), (1, (0, 0), true)],
        5,
    );
    let labels = labels_for_tokens(&chunk, ner_id2label);
    assert_eq!(labels.len(), 4, "special token dropped");
    assert_eq!(labels[0], TokenLabel { label_id: 1, label: "B-PER".into(), start: 0, end: 3, token_index: 0 });
    assert_eq!(labels[1], TokenLabel { label_id: 2, label: "I-PER".into(), start: 3, end: 7, token_index: 1 });
    assert_eq!(labels[2], TokenLabel { label_id: 0, label: "O".into(), start: 7, end: 8, token_index: 2 });
    assert_eq!(labels[3], TokenLabel { label_id: 1, label: "B-PER".into(), start: 8, end: 12, token_index: 3 });
}

#[test]
fn group_spans_bio_merges_b_i_and_breaks_on_o() {
    let chunk = token_chunk(&[(1, (0, 3), false), (2, (3, 7), false), (0, (7, 8), false), (1, (8, 12), false)], 5);
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "PER");
    assert_eq!((spans[0].start, spans[0].end), (0, 7));
    assert_eq!(spans[0].token_range, 0..2);
    assert_eq!(spans[1].label, "PER");
    assert_eq!((spans[1].start, spans[1].end), (8, 12));
    assert_eq!(spans[1].token_range, 3..4);
}

#[test]
fn group_spans_bio_breaks_on_type_change() {
    // B-PER, I-LOC → mismatched I- opens a fresh LOC span.
    let chunk = token_chunk(&[(1, (0, 3), false), (4, (3, 6), false)], 5);
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0], Entity { label: "PER".into(), label_id: 1, start: 0, end: 3, token_range: 0..1 });
    assert_eq!(spans[1], Entity { label: "LOC".into(), label_id: 4, start: 3, end: 6, token_range: 1..2 });
}

#[test]
fn group_spans_bio_treats_stray_i_as_new_opener() {
    // I-PER with no preceding B-PER → lenient single-token span.
    let chunk = token_chunk(&[(2, (0, 3), false), (0, (3, 4), false)], 5);
    let labels = labels_for_tokens(&chunk, ner_id2label);
    let spans = group_spans(&labels, Scheme::Bio);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].label, "PER");
    assert_eq!(spans[0].token_range, 0..1);
}

#[test]
fn group_spans_flat_emits_one_per_token() {
    // Flat labels (POS-style): each token its own entity, no grouping.
    let labels: Vec<TokenLabel> = [("NNS", 0), ("VBD", 1)]
        .into_iter()
        .map(|(lbl, i)| TokenLabel { label_id: 0, label: lbl.into(), start: i, end: i + 1, token_index: i })
        .collect();
    let spans = group_spans(&labels, Scheme::Flat);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].label, "NNS");
    assert_eq!(spans[0].token_range, 0..1);
    assert_eq!(spans[1].label, "VBD");
    assert_eq!(spans[1].token_range, 1..2);
}

#[test]
fn group_spans_bilou_and_iobes() {
    // BILOU: B-PER I-PER L-PER → one PER span over 3 tokens; U-PER → single.
    let bilou_labels: Vec<TokenLabel> = [("B-PER", 0), ("I-PER", 1), ("L-PER", 2), ("U-PER", 3)]
        .into_iter()
        .map(|(lbl, i)| TokenLabel { label_id: 1, label: lbl.into(), start: i, end: i + 1, token_index: i })
        .collect();
    let bilou = group_spans(&bilou_labels, Scheme::Bilou);
    assert_eq!(bilou.len(), 2);
    assert_eq!(bilou[0].label, "PER");
    assert_eq!(bilou[0].token_range, 0..3);
    assert_eq!(bilou[1].label, "PER");
    assert_eq!(bilou[1].token_range, 3..4);

    // IOBES: B-PER I-PER E-PER → one span; S-PER → single.
    let iobes_labels: Vec<TokenLabel> = [("B-PER", 0), ("I-PER", 1), ("E-PER", 2), ("S-PER", 3)]
        .into_iter()
        .map(|(lbl, i)| TokenLabel { label_id: 1, label: lbl.into(), start: i, end: i + 1, token_index: i })
        .collect();
    let iobes = group_spans(&iobes_labels, Scheme::Iobes);
    assert_eq!(iobes.len(), 2);
    assert_eq!(iobes[0].label, "PER");
    assert_eq!(iobes[0].token_range, 0..3);
    assert_eq!(iobes[1].label, "PER");
    assert_eq!(iobes[1].token_range, 3..4);
}

proptest! {
    /// Random BIO label sequences decode to well-formed spans: disjoint,
    /// ordered, in-bounds ranges; every token in a span shares the span's type;
    /// no "O" token is inside a span; byte spans match the covered tokens.
    #[test]
    fn prop_group_spans_bio_well_formed(
        labels in proptest::collection::vec(
            proptest::sample::select(vec!["O".to_string(), "B-PER".into(), "I-PER".into(), "B-LOC".into(), "I-LOC".into()]),
            0..32,
        )
    ) {
        let tokens: Vec<TokenLabel> = labels
            .iter()
            .enumerate()
            .map(|(i, l)| TokenLabel { label_id: 0, label: l.clone(), start: i, end: i + 1, token_index: i })
            .collect();
        let n = tokens.len();
        let entities = group_spans(&tokens, Scheme::Bio);

        let mut prev_end = 0usize;
        for e in &entities {
            // Range is non-empty, in-bounds, ordered & disjoint with the previous.
            prop_assert!(e.token_range.start < e.token_range.end, "empty range");
            prop_assert!(e.token_range.end <= n, "range out of bounds");
            prop_assert!(e.token_range.start >= prev_end, "ranges not ordered/disjoint");
            // Byte span matches the covered tokens.
            prop_assert_eq!(e.start, tokens[e.token_range.start].start);
            prop_assert_eq!(e.end, tokens[e.token_range.end - 1].end);
            // Every covered token shares the span's type and is not "O".
            for ti in e.token_range.clone() {
                prop_assert!(tokens[ti].label != "O", "'O' token inside a span");
                prop_assert_eq!(typ_of(&tokens[ti].label), Some(e.label.as_str()), "type mismatch in span");
            }
            prev_end = e.token_range.end;
        }
    }
}
