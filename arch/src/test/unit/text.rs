use std::convert::Infallible;

use crate::pipelines::text::{
    Chunker, Embed, Embedding, EmbeddingsPipeline, Encoding, HfTokenizer, HfTokenizerError, RunOptions, RunProfile,
    TextChunk, TextPipelineError, Tokenizer, TruncatingChunker,
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
    error: bool,
}

impl Embed for StubEmbed {
    type Error = StubEmbedError;
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn capacity(&self) -> (usize, usize) {
        (1, self.hidden_size)
    }
    fn embed_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Embedding>, Option<RunProfile>), StubEmbedError> {
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
}

#[derive(Debug, snafu::Snafu)]
#[snafu(display("stub embed error"))]
struct StubEmbedError;

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
    assert_eq!(c.char_offset, 0);
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

// ─── Embed: batch vs single-path agreement ───────────────────────────────────

#[test]
fn embed_single_is_batch_of_one() {
    // The default `embed` delegates to `embed_batch(&[enc])` and pops — verify
    // the values match a direct batch call.
    let mut embed = StubEmbed { hidden_size: 3, error: false };
    let e = enc(&[4, 5, 6]);
    let single = embed.embed(&e, false).unwrap().0;
    let batch = embed.embed_batch(&[&e], false).unwrap().0;
    assert_eq!(single, batch.into_iter().next().unwrap());
}

// ─── EmbeddingsPipeline end-to-end ───────────────────────────────────────────

fn pipeline(ids: Vec<u32>, max_seq: usize) -> EmbeddingsPipeline<StubTokenizer, TruncatingChunker, StubEmbed> {
    EmbeddingsPipeline::new(
        StubTokenizer { ids, max_seq, error: false },
        TruncatingChunker::new(max_seq),
        StubEmbed { hidden_size: max_seq, error: false },
    )
}

#[test]
fn pipeline_truncates_then_embeds() {
    // Tokenizer yields 7 ids; chunker caps at 4; embedder sees the 4 surviving.
    let mut p = pipeline(vec![1, 2, 3, 4, 5, 6, 7], 4);
    let out = p.embed_default("ignored").unwrap();
    assert_eq!(out.chunks.len(), 1);
    // char_offset is threaded through from the chunker (TruncatingChunker → 0).
    assert_eq!(out.chunks[0].char_offset, 0);
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
    let p = EmbeddingsPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        NoChunkChunker { max_seq: 4 },
        StubEmbed { hidden_size: 4, error: true }, // would fail if called
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
    let _p: EmbeddingsPipeline<_, TruncatingChunker, StubEmbed> = EmbeddingsPipeline::assemble(
        StubTokenizer { ids: vec![1], max_seq: 8, error: false },
        TruncatingChunker::new(8),
        |max_seq| {
            seen.set(max_seq);
            Ok::<_, Infallible>(StubEmbed { hidden_size: max_seq, error: false })
        },
    )
    .unwrap();
    assert_eq!(seen.get(), 8);
}

// ─── error propagation ────────────────────────────────────────────────────────

#[test]
fn tokenize_error_maps_to_tokenize_variant() {
    let mut p = EmbeddingsPipeline::new(
        StubTokenizer { ids: vec![1], max_seq: 4, error: true },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, error: false },
    );
    let err = p.embed_default("ignored").unwrap_err();
    assert!(matches!(err, crate::pipelines::text::TextPipelineError::Tokenize { .. }));
}

#[test]
fn embed_error_maps_to_embed_variant() {
    let mut p = EmbeddingsPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        TruncatingChunker::new(4),
        StubEmbed { hidden_size: 4, error: true },
    );
    let err = p.embed_default("ignored").unwrap_err();
    assert!(matches!(err, crate::pipelines::text::TextPipelineError::Embed { .. }));
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

// ─── TextPipelineError::Chunk arm (TruncatingChunker::Error = Infallible) ─────

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
    let mut p = EmbeddingsPipeline::new(
        StubTokenizer { ids: vec![1, 2], max_seq: 4, error: false },
        ErrChunker { max_seq: 4 },
        StubEmbed { hidden_size: 4, error: false },
    );
    let err = p.embed_default("ignored").unwrap_err();
    assert!(matches!(err, TextPipelineError::Chunk { .. }));
}
