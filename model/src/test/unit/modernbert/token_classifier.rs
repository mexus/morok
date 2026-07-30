//! Host-only tests for [`ModernBertTokenClassifier`] over a tiny random-weight
//! model (f32, CPU). No real checkpoint, no HF Hub — the model is
//! `ModernBertTokenClassificationModel::empty` with `fan_in_uniform` random
//! weights, enough to exercise the JIT prepare/pack/execute/read path and the
//! fused mask + per-token head semantics.

use svod_arch::pipelines::text::{Encoding, Recognize, TokenClassification};

use crate::modernbert::{ModernBertTokenClassificationModel, ModernBertTokenClassifier};
use crate::test::unit::modernbert::model::tiny_cfg;

/// Build a token classifier from a tiny random-weight model, prepared at
/// `[max_batch, max_seq]`.
fn recognizer(max_batch: usize, max_seq: usize) -> ModernBertTokenClassifier {
    let cfg = tiny_cfg();
    let model = ModernBertTokenClassificationModel::empty(&cfg);
    ModernBertTokenClassifier::new(model, max_batch, max_seq).expect("prepare token-classifier JIT")
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

/// Max elementwise absolute difference between two equal-length slices.
fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

// ── shape / contract ───────────────────────────────────────────────────────

/// `recognize_batch` returns one token-classification per input, each with a
/// `(seq_len, num_labels)` logit grid (padding excluded from `seq_len`), and all
/// finite.
#[test]
fn recognize_batch_shapes_and_finite() {
    let mut rec = recognizer(4, 16);
    let nl = rec.num_labels();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5], 1);
    let (out, prof) = rec.recognize_batch(&[&e1, &e2], false).expect("recognize_batch");
    assert_eq!(out.len(), 2);
    // e1: 3 real tokens → 3*num_labels logits; e2: 2 real + 1 pad → 3*num_labels.
    assert_eq!(out[0].logits.len(), 3 * nl);
    assert_eq!(out[1].logits.len(), 3 * nl);
    for TokenClassification { logits, num_labels } in &out {
        assert_eq!(*num_labels, nl);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
    }
    assert!(prof.is_none(), "unprofiled run yields no profile");
}

/// `num_labels()` reports the config's value.
#[test]
fn num_labels_reported() {
    let rec = recognizer(4, 16);
    assert_eq!(rec.num_labels(), tiny_cfg().num_labels);
}

/// `capacity()` reports the prepared sizes.
#[test]
fn capacity_reported() {
    let rec = recognizer(7, 42);
    assert_eq!(rec.capacity(), (7, 42));
}

/// The trait-default `recognize` (batch-of-one) agrees exactly with
/// `recognize_batch(&[e])[0]`.
#[test]
fn recognize_single_matches_batch_of_one() {
    let mut rec = recognizer(4, 16);
    let e = encoding(&[1, 2, 3], 0);
    let single = rec.recognize(&e, false).unwrap().0;
    let batch = rec.recognize_batch(&[&e], false).unwrap().0;
    assert_eq!(single.logits, batch.into_iter().next().unwrap().logits);
}

/// A multi-row batch yields the same per-token logits as individual calls
/// (within fp tolerance) — the symbolic batch dim doesn't cross-contaminate.
#[test]
fn batch_rows_match_single_calls() {
    let mut rec = recognizer(4, 16);
    let nl = rec.num_labels();
    let inputs = [encoding(&[1, 2, 3], 0), encoding(&[4, 5], 1), encoding(&[6, 7, 8, 9], 0)];
    let refs: Vec<Vec<f32>> = inputs
        .iter()
        .map(|e| {
            let (mut s, _) = rec.recognize_batch(&[e], false).unwrap();
            s.pop().unwrap().logits
        })
        .collect();
    let (batch, _) = rec.recognize_batch(&inputs.iter().collect::<Vec<_>>(), false).unwrap();
    for (got, want) in batch.iter().zip(&refs) {
        assert_eq!(got.logits.len() / nl, want.len() / nl, "seq_len mismatch");
        assert!(max_delta(&got.logits, want) < 1e-4, "row differs from single call");
    }
}

/// An empty batch is a cheap no-op (no profile unless requested).
#[test]
fn empty_batch_returns_empty() {
    let mut rec = recognizer(4, 16);
    let (out, prof) = rec.recognize_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());
    let (out, prof) = rec.recognize_batch(&[], true).expect("empty batch profiled");
    assert!(out.is_empty());
    assert!(prof.is_some(), "profiled empty run still returns a default profile");
}

/// A batch larger than the prepared `max_batch` is rejected.
#[test]
fn capacity_exceeded_errors() {
    let mut rec = recognizer(2, 16);
    let e = encoding(&[1, 2, 3], 0);
    let err = rec.recognize_batch(&[&e, &e, &e], false).unwrap_err();
    assert!(err.to_string().contains("exceeds"), "{err}");
}

/// A profiled run emits a `recognize` GPU stage.
#[test]
fn profile_returned_when_requested() {
    let mut rec = recognizer(4, 16);
    let e = encoding(&[1, 2, 3], 0);
    let (_, prof) = rec.recognize_batch(&[&e], true).expect("profiled run");
    let prof = prof.expect("profile collected");
    assert!(prof.stage("recognize").is_some(), "missing 'recognize' stage");
}

/// Adding masked pad positions must not change the per-token logits of the real
/// tokens (the load-bearing mask property): same content, with vs without
/// trailing pad, agrees within fp tolerance on the real-token rows.
#[test]
fn padding_with_correct_mask_is_invariant() {
    let mut rec = recognizer(4, 16);
    let nl = rec.num_labels();
    let real = 3;

    let (out_no, _) = rec.recognize_batch(&[&encoding(&[1, 2, 3], 0)], false).unwrap();
    let (out_pad, _) = rec.recognize_batch(&[&encoding(&[1, 2, 3], 2)], false).unwrap();

    assert_eq!(out_no[0].logits.len(), real * nl);
    assert_eq!(out_pad[0].logits.len(), (real + 2) * nl, "pad positions keep their own logits");
    let content_no = &out_no[0].logits[..real * nl];
    let content_pad = &out_pad[0].logits[..real * nl];
    assert!(max_delta(content_no, content_pad) < 1e-3, "padding leaked into real-token logits");
}
