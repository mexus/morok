//! Host-only tests for [`ModernBertEmbedder`] over a tiny random-weight backbone
//! (f32, CPU). No real checkpoint, no HF Hub — the model is `ModernBert::empty`
//! with `fan_in_uniform` random token embeddings, enough to exercise the JIT
//! prepare/pack/execute/read path and the fused mask+pool+norm semantics.

use svod_arch::pipelines::text::{Embed, Embedding, Encoding};

use crate::modernbert::ModernBert;
use crate::test::unit::modernbert::model::tiny_cfg;

/// Build an embedder from a tiny random-weight backbone, prepared at
/// `[max_batch, max_seq]`.
fn embedder(max_batch: usize, max_seq: usize) -> ModernBertEmbedder {
    let cfg = tiny_cfg();
    let model = ModernBert::empty(cfg.clone());
    ModernBertEmbedder::new(model, max_batch, max_seq).expect("prepare embedder JIT")
}

/// A consistent encoding: `n` real token ids (1..) with mask all-ones, plus
/// optional trailing pad positions at mask 0.
fn encoding(real_ids: &[u32], n_pad: usize) -> Encoding {
    let mut ids = real_ids.to_vec();
    let mut mask = vec![1u32; real_ids.len()];
    ids.extend(std::iter::repeat_n(&0u32, n_pad).copied());
    mask.extend(std::iter::repeat_n(&0u32, n_pad));
    let l = ids.len();
    Encoding {
        input_ids: ids,
        attention_mask: mask,
        token_type_ids: vec![0; l],
        offsets: (0..l).map(|i| (i, i + 1)).collect(),
        special_tokens_mask: vec![0; l],
    }
}

fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// Max elementwise absolute difference between two equal-length slices — the
/// shared comparison primitive for the batch-vs-single and mask-leak checks.
fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

use crate::modernbert::ModernBertEmbedder;

/// `embed_batch` returns one embedding per input, each of length `hidden_size`,
/// and each L2-normalized (norm ≈ 1).
#[test]
fn embed_batch_shapes_and_norms() {
    let mut emb = embedder(4, 16);
    let hidden = emb.hidden_size();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5], 1);
    let (out, prof) = emb.embed_batch(&[&e1, &e2], false).expect("embed_batch");
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].values.len(), hidden);
    assert_eq!(out[1].values.len(), hidden);
    assert!(prof.is_none(), "unprofiled run yields no profile");
    for Embedding { values } in &out {
        let n = l2_norm(values);
        assert!((n - 1.0).abs() < 1e-3, "L2-normalized embedding should have norm ~1, got {n}");
        assert!(values.iter().all(|v| v.is_finite()), "non-finite embedding value");
    }
}

/// The trait-default `embed` (batch-of-one) agrees with `embed_batch` on a
/// single input — the default delegates to the batch path and pops.
#[test]
fn embed_single_matches_batch_of_one() {
    let mut emb = embedder(4, 16);
    let e = encoding(&[1, 2, 3, 4], 0);
    let single = emb.embed(&e, false).expect("embed").0;
    let batch = emb.embed_batch(&[&e], false).expect("embed_batch");
    let batch0 = batch.0.into_iter().next().unwrap();
    let max = single.values.iter().zip(&batch0.values).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert_eq!(max, 0.0, "default embed must match embed_batch exactly");
}

/// The padding mask is load-bearing: pooling must ignore pad positions. A
/// sequence `[1,2,3]` (no pad) and `[1,2,3]` + 2 pad tokens (mask 0) must yield
/// the **same** embedding — proving masked mean-pool, not raw mean-pool.
#[test]
fn pooling_ignores_pad_positions() {
    let mut emb = embedder(4, 16);
    let unpadded = encoding(&[1, 2, 3], 0);
    let padded = encoding(&[1, 2, 3], 2); // two trailing pad positions, mask 0
    let a = emb.embed(&unpadded, false).expect("unpadded").0;
    let b = emb.embed(&padded, false).expect("padded").0;
    let max = a.values.iter().zip(&b.values).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    assert!(max < 1e-4, "pad positions leaked into the pooled embedding: max |delta| = {max}");
}

/// `hidden_size`/`capacity` report the prepared sizes.
#[test]
fn capacity_reports_prepared_sizes() {
    let emb = embedder(3, 12);
    let (mb, ms) = emb.capacity();
    assert_eq!(mb, 3);
    assert_eq!(ms, 12);
    assert_eq!(emb.hidden_size(), 32, "matches tiny_cfg hidden_size");
}

/// A batch exceeding `max_batch` is rejected up front (CapacityExceeded), not
/// silently truncated or overflowed.
#[test]
fn batch_over_max_batch_is_rejected() {
    let mut emb = embedder(2, 8);
    let encs = [encoding(&[1], 0), encoding(&[2], 0), encoding(&[3], 0)];
    let refs: Vec<&Encoding> = encs.iter().collect();
    let err = emb.embed_batch(&refs, false).unwrap_err();
    assert!(matches!(err, crate::modernbert::EmbedderError::CapacityExceeded { .. }));
}

/// An empty batch is a no-op returning no embeddings (mirrors the pipeline's
/// zero-chunk guard).
#[test]
fn empty_batch_is_noop() {
    let mut emb = embedder(2, 8);
    let (out, prof) = emb.embed_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());
}

/// A profiled run yields a profile with an `embed` GPU stage; an unprofiled run
/// on the same embedder yields none (per-call, no rebuild).
#[test]
fn profiled_run_emits_embed_stage() {
    let mut emb = embedder(2, 8);
    let e = encoding(&[1, 2], 0);
    let prof = emb.embed(&e, true).expect("profiled").1.expect("profile present");
    assert!(prof.stages.iter().any(|s| s.name == "embed"), "embed stage present");
    assert!(emb.embed(&e, false).expect("unprofiled").1.is_none(), "unprofiled yields no profile");
}

/// Two **distinct** inputs through `embed_batch` must match the same inputs run
/// one-at-a-time via `embed` — a cross-row leakage guard. Row `i`'s output must
/// depend only on row `i`'s ids/mask, never on a sibling row packed into the
/// `[max_batch, max_seq]` buffers. Exact equality: same weights, same row 0
/// computation whether `b` binds to 1 or 2 (the symbolic-batch graph computes
/// only the first `b` rows), so the bits must match.
#[test]
fn embed_batch_rows_match_single_calls() {
    let mut emb = embedder(4, 16);
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[10, 20, 30, 40], 0);
    let batch = emb.embed_batch(&[&e1, &e2], false).expect("embed_batch").0;
    let s1 = emb.embed(&e1, false).expect("embed e1").0;
    let s2 = emb.embed(&e2, false).expect("embed e2").0;
    assert_eq!(max_delta(&batch[0].values, &s1.values), 0.0, "batch row 0 leaked from/into e1");
    assert_eq!(max_delta(&batch[1].values, &s2.values), 0.0, "batch row 1 leaked from/into e2");
}

/// On the **batch** path the attention mask is threaded per row: a row whose
/// real tokens are `[1,2,3]` with two trailing pads must pool to the same vector
/// as the same row run unpadded, and must agree with its standalone `embed`. The
/// batch packing must not collapse pooling to a raw mean over the padded length.
/// Tolerance 1e-4 mirrors `pooling_ignores_pad_positions`.
#[test]
fn batch_path_respects_attention_mask() {
    let mut emb = embedder(4, 16);
    let padded = encoding(&[1, 2, 3], 2);
    let unpadded = encoding(&[1, 2, 3], 0);
    let batch = emb.embed_batch(&[&padded, &unpadded], false).expect("embed_batch").0;
    // Pad positions must not move the pooled output within a row.
    let d = max_delta(&batch[0].values, &batch[1].values);
    assert!(d < 1e-4, "pad mask leaked in the batch path: max |delta| = {d}");
    // And the batch path agrees with the single-call path for the unpadded row.
    let alone = emb.embed(&unpadded, false).expect("embed unpadded").0;
    let d = max_delta(&batch[1].values, &alone.values);
    assert!(d < 1e-4, "batch row 1 disagrees with its standalone embed: max |delta| = {d}");
}

/// On the throughput path (`embed_batch`, not the batch-of-one `embed` default),
/// a profiled run emits a GPU stage named `embed`; an unprofiled run on the same
/// embedder emits none — the profile switch is per-call, no rebuild.
#[test]
fn embed_batch_profile_emits_stage() {
    let mut emb = embedder(2, 8);
    let e1 = encoding(&[1, 2], 0);
    let e2 = encoding(&[3, 4], 0);
    let prof = emb.embed_batch(&[&e1, &e2], true).expect("profiled embed_batch").1.expect("profile present");
    assert!(prof.stages.iter().any(|s| s.name == "embed"), "embed GPU stage present");
    assert!(
        emb.embed_batch(&[&e1, &e2], false).expect("unprofiled embed_batch").1.is_none(),
        "unprofiled yields no profile"
    );
}
