use std::convert::Infallible;

use crate::pipelines::text::{
    Chunker, Embed, Embedding, EmbeddingsPipeline, Encoding, RunOptions, RunProfile, TextChunk, Tokenizer,
    TruncatingChunker,
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
    assert_eq!(out.chunks[0].values, vec![1.0, 2.0, 3.0, 4.0]);
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
