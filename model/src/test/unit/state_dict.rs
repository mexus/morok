//! State-dict round-trip tests for GigaAM.
//!
//! Cheap — no `.realize()`, no JIT. Build the model via `with_random_weights`,
//! emit its state dict via the per-component `HasStateDict` impls (Encoder
//! itself doesn't implement the trait; we compose its sub-modules the same
//! way `Encoder::from_state_dict` does at `gigaam/encoder.rs:764`), assert
//! representative keys cover the encoder + head surface, then reload into a
//! fresh model.
//!
//! Catches: a sub-module rename, a forgotten field in `state_dict()`, a
//! prefix mismatch between emit and load. Mirrors mexus's WeSpeaker round-trip
//! at `test/unit/wespeaker/model.rs`.

use crate::gigaam::{GigaAm, GigaAmConfig, Head, TransducerConfig};
use crate::state::{HasStateDict, StateDict};

use super::batch::test_config;

fn rnnt_test_config() -> GigaAmConfig {
    let mut cfg = test_config();
    // Mirror an `v3_e2e_rnnt`-style RN-T head: small predictor + joint, blank
    // token at the end of the vocabulary.
    let vocabulary: Vec<String> = (0..8).map(|i| format!("p{i}")).collect();
    cfg.transducer = Some(TransducerConfig {
        pred_hidden: 16,
        pred_rnn_layers: 1,
        joint_hidden: 16,
        num_classes: vocabulary.len() + 1,
        max_symbols_per_step: 10,
        vocabulary,
        sentencepiece: false,
    });
    cfg
}

/// Compose the model's full state dict from its component-level `HasStateDict`
/// impls. The encoder itself doesn't implement the trait (only its
/// sub-modules), so we mirror `Encoder::from_state_dict`'s key convention:
/// `subsampling.*` and `layers.{i}.*` (no `encoder.` prefix on disk — the
/// `remap::remap_pytorch` step at `gigaam/model.rs:146` strips it for
/// PyTorch checkpoints).
fn compose_state_dict(model: &GigaAm) -> StateDict {
    let mut sd = model.encoder.subsampling.state_dict("subsampling");
    for (i, layer) in model.encoder.layers.iter().enumerate() {
        sd.extend(layer.state_dict(&format!("layers.{i}")));
    }
    match &model.head {
        Head::Ctc(h) => sd.extend(h.state_dict("head")),
        Head::Rnnt { head, .. } => sd.extend(head.state_dict("head")),
    }
    sd
}

/// Reload the same model in-place from a state dict. Matches the per-component
/// `load_state_dict` flow used by `GigaAm::from_state_dict`.
fn reload(model: &mut GigaAm, sd: &StateDict) {
    model.encoder.subsampling.load_state_dict(sd, "subsampling").expect("subsampling reload");
    for (i, layer) in model.encoder.layers.iter_mut().enumerate() {
        layer.load_state_dict(sd, &format!("layers.{i}")).expect("conformer layer reload");
    }
    match &mut model.head {
        Head::Ctc(h) => h.load_state_dict(sd, "head").expect("ctc head reload"),
        Head::Rnnt { head, .. } => head.load_state_dict(sd, "head").expect("rnnt head reload"),
    }
}

#[test]
fn gigaam_state_dict_round_trip_ctc() {
    let cfg = test_config();
    let model = GigaAm::with_random_weights(cfg.clone());

    let sd = compose_state_dict(&model);

    // Representative keys covering subsampling, every conformer-layer sub-
    // module (ffn1 / mhsa / conv / ffn2 / final_norm), and the CTC head.
    for key in [
        "subsampling.conv1_weight",
        "layers.0.ffn1.norm.weight",
        "layers.0.ffn1.linear1.weight",
        "layers.0.ffn1.linear2.bias",
        "layers.0.mhsa.norm.weight",
        "layers.0.conv.norm.weight",
        "layers.0.ffn2.linear1.weight",
        "layers.0.final_norm.weight",
        "layers.1.ffn1.norm.weight",
        "head.weight",
        "head.bias",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    // RN-T-specific keys must NOT appear for a CTC model.
    assert!(!sd.contains_key("head.predictor.embed"), "CTC head must not emit predictor.embed");
    assert!(!sd.contains_key("head.joint.enc_w"), "CTC head must not emit joint.enc_w");

    let mut empty = GigaAm::with_random_weights(cfg);
    reload(&mut empty, &sd);
}

#[test]
fn gigaam_state_dict_round_trip_rnnt() {
    let cfg = rnnt_test_config();
    let model = GigaAm::with_random_weights(cfg.clone());

    let sd = compose_state_dict(&model);

    for key in [
        "subsampling.conv1_weight",
        "layers.0.ffn1.norm.weight",
        "layers.1.final_norm.weight",
        "head.predictor.embed",
        "head.predictor.lstm.0.w_ih",
        "head.predictor.lstm.0.b_hh",
        "head.joint.enc_w",
        "head.joint.enc_b",
        "head.joint.pred_w",
        "head.joint.out_w",
        "head.joint.out_b",
    ] {
        assert!(sd.contains_key(key), "missing key: {key}");
    }

    // CTC-specific direct projection keys must NOT appear.
    assert!(!sd.contains_key("head.weight"), "RN-T head must not emit a bare `head.weight`");
    assert!(!sd.contains_key("head.bias"), "RN-T head must not emit a bare `head.bias`");

    let mut empty = GigaAm::with_random_weights(cfg);
    reload(&mut empty, &sd);
}
