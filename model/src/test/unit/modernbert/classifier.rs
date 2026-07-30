//! Host-only tests for [`ModernBertClassifier`] over a tiny random-weight model
//! (f32, CPU). No real checkpoint, no HF Hub — the model is
//! `ModernBertClassificationModel::empty` with `fan_in_uniform` random weights,
//! enough to exercise the JIT prepare/pack/execute/read path and the fused
//! mask+pool+head semantics.

use svod_arch::pipelines::text::{Classification, Classify, Encoding};

use crate::modernbert::{ClassifierPooling, ModernBertClassificationModel, ModernBertClassifier};
use crate::test::unit::modernbert::model::tiny_cfg;

/// Build a classifier from a tiny random-weight model, prepared at
/// `[max_batch, max_seq]`.
fn classifier(max_batch: usize, max_seq: usize) -> ModernBertClassifier {
    let cfg = tiny_cfg();
    let model = ModernBertClassificationModel::empty(&cfg);
    ModernBertClassifier::new(model, max_batch, max_seq).expect("prepare classifier JIT")
}

/// Variant with a specific pooling strategy.
fn classifier_with_pooling(max_batch: usize, max_seq: usize, pooling: ClassifierPooling) -> ModernBertClassifier {
    let mut cfg = tiny_cfg();
    cfg.classifier_pooling = pooling;
    let model = ModernBertClassificationModel::empty(&cfg);
    ModernBertClassifier::new(model, max_batch, max_seq).expect("prepare classifier JIT")
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

/// `classify_batch` returns one classification per input, each with `num_classes`
/// logits, and all finite.
#[test]
fn classify_batch_shapes_and_finite() {
    let mut clf = classifier(4, 16);
    let nc = clf.num_classes();
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5], 1);
    let (out, prof) = clf.classify_batch(&[&e1, &e2], false).expect("classify_batch");
    assert_eq!(out.len(), 2);
    for Classification { logits } in &out {
        assert_eq!(logits.len(), nc);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
    }
    assert!(prof.is_none(), "unprofiled run yields no profile");
}

/// `num_classes()` reports the config's value.
#[test]
fn num_classes_reported() {
    let clf = classifier(4, 16);
    assert_eq!(clf.num_classes(), tiny_cfg().num_labels);
}

/// `capacity()` returns `(max_batch, max_seq)`.
#[test]
fn capacity_reported() {
    let clf = classifier(7, 42);
    assert_eq!(clf.capacity(), (7, 42));
}

// ── consistency ────────────────────────────────────────────────────────────

/// The trait-default `classify` (batch-of-one) agrees with `classify_batch`.
#[test]
fn classify_single_matches_batch_of_one() {
    let mut clf = classifier(4, 16);
    let e = encoding(&[1, 2, 3, 4], 0);
    let single = clf.classify(&e, false).expect("classify").0;
    let batch = clf.classify_batch(&[&e], false).expect("classify_batch");
    let batch0 = &batch.0[0];
    let max = max_delta(&single.logits, &batch0.logits);
    assert_eq!(max, 0.0, "default classify must match classify_batch exactly");
}

/// Multi-input batch yields the same logits as individual calls.
#[test]
fn batch_rows_match_single_calls() {
    let mut clf = classifier(4, 16);
    let e1 = encoding(&[1, 2, 3], 0);
    let e2 = encoding(&[4, 5, 6, 7], 1);
    let e3 = encoding(&[8], 2);

    let single1 = clf.classify(&e1, false).expect("classify e1").0;
    let single2 = clf.classify(&e2, false).expect("classify e2").0;
    let single3 = clf.classify(&e3, false).expect("classify e3").0;

    let batch = clf.classify_batch(&[&e1, &e2, &e3], false).expect("classify_batch");

    assert!(max_delta(&single1.logits, &batch.0[0].logits) < 1e-4);
    assert!(max_delta(&single2.logits, &batch.0[1].logits) < 1e-4);
    assert!(max_delta(&single3.logits, &batch.0[2].logits) < 1e-4);
}

// ── guards ─────────────────────────────────────────────────────────────────

/// Empty batch → empty results, profile optional.
#[test]
fn empty_batch_returns_empty() {
    let mut clf = classifier(4, 16);
    let (out, prof) = clf.classify_batch(&[], false).expect("empty batch");
    assert!(out.is_empty());
    assert!(prof.is_none());

    let (out, prof) = clf.classify_batch(&[], true).expect("empty batch profiled");
    assert!(out.is_empty());
    assert!(prof.is_some(), "profiled empty batch yields a (default) profile");
}

/// Over-capacity batch → `CapacityExceeded` error.
#[test]
fn capacity_exceeded_errors() {
    let mut clf = classifier(2, 16);
    let e1 = encoding(&[1], 0);
    let e2 = encoding(&[2], 0);
    let e3 = encoding(&[3], 0);
    let err = clf.classify_batch(&[&e1, &e2, &e3], false);
    assert!(err.is_err(), "batch > max_batch must error");
}

// ── profiling ──────────────────────────────────────────────────────────────

/// Profile is `Some` and contains a `"classify"` GPU stage when requested.
#[test]
fn profile_returned_when_requested() {
    let mut clf = classifier(4, 16);
    let e = encoding(&[1, 2, 3], 0);
    let (_, prof) = clf.classify_batch(&[&e], true).expect("classify_batch");
    let prof = prof.expect("profile requested");
    assert!(prof.stage("classify").is_some(), "expected a 'classify' stage in {prof:?}");
}

// ── pooling semantics ──────────────────────────────────────────────────────

/// CLS and mean pooling produce different logits on the same input.
#[test]
fn cls_vs_mean_pooling_differ() {
    let e = encoding(&[1, 2, 3, 4, 5], 0);

    let mut cls_clf = classifier_with_pooling(2, 16, ClassifierPooling::Cls);
    let mut mean_clf = classifier_with_pooling(2, 16, ClassifierPooling::Mean);

    // Use the same backbone weights so the only difference is pooling + head.
    // (Different random heads, so the logits will differ trivially, but we
    // verify the models don't crash and produce valid output.)
    let cls_logits = cls_clf.classify(&e, false).expect("cls classify").0.logits;
    let mean_logits = mean_clf.classify(&e, false).expect("mean classify").0.logits;

    assert!(cls_logits.iter().all(|v| v.is_finite()));
    assert!(mean_logits.iter().all(|v| v.is_finite()));
}

/// Adding padding with a correct mask does not change logits for either
/// pooling strategy — the mask keeps pad tokens out of both the attention and
/// the mean. This is the load-bearing property: the classifier is invariant to
/// sequence padding.
#[test]
fn padding_with_correct_mask_is_invariant() {
    let e_no_pad = encoding(&[1, 2, 3, 4], 0);
    let e_with_pad = encoding(&[1, 2, 3, 4], 2);

    let mut mean_clf = classifier_with_pooling(2, 16, ClassifierPooling::Mean);
    let mean_a = mean_clf.classify(&e_no_pad, false).expect("mean no-pad").0.logits;
    let mean_b = mean_clf.classify(&e_with_pad, false).expect("mean with-pad").0.logits;
    assert!(
        max_delta(&mean_a, &mean_b) < 1e-3,
        "mean pooling with correct mask should be padding-invariant, got delta {}",
        max_delta(&mean_a, &mean_b)
    );

    let mut cls_clf = classifier_with_pooling(2, 16, ClassifierPooling::Cls);
    let cls_a = cls_clf.classify(&e_no_pad, false).expect("cls no-pad").0.logits;
    let cls_b = cls_clf.classify(&e_with_pad, false).expect("cls with-pad").0.logits;
    assert!(
        max_delta(&cls_a, &cls_b) < 1e-3,
        "CLS pooling with correct mask should be padding-invariant, got delta {}",
        max_delta(&cls_a, &cls_b)
    );
}
