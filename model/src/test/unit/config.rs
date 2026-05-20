use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use svod_arch::ctc::CtcDecoder;

use crate::gigaam::{ConvNormType, GigaAmConfig, SubsamplingMode};

#[test]
fn test_config_from_json() {
    let config = GigaAmConfig::from_json(Path::new("tests/gigaam/ctc_config.json")).unwrap();

    assert_eq!(config.n_mels, 64);
    assert_eq!(config.max_batch_size, 32);
    assert_eq!(config.d_model, 768);
    assert_eq!(config.n_heads, 16);
    assert_eq!(config.n_layers, 16);
    assert_eq!(config.d_ff, 3072);
    assert_eq!(config.conv_kernel, 5);
    assert_eq!(config.vocab_size, 34);
    assert_eq!(config.sample_rate, 16000);
    assert_eq!(config.n_fft, 320);
    assert_eq!(config.hop_length, 160);
    assert_eq!(config.win_length, 320);
    assert!(!config.mel_center);
    assert!(matches!(config.subsampling_mode, SubsamplingMode::Conv1d));
    assert!(matches!(config.conv_norm_type, ConvNormType::LayerNorm));
    assert_eq!(config.subs_kernel_size, 5);
    assert_eq!(config.subsampling_factor, 4);
    assert_eq!(config.max_encoder_frames, 5000);
    // No explicit `max_mel_frames` / `max_seq_len` in this config — the loader
    // derives the pre-subsampling bound from the post-subsampling encoder bound
    // by scaling up by `subsampling_factor`, so audio approaching the encoder
    // cap isn't rejected at the JIT input stage.
    assert_eq!(config.max_mel_frames, 5000 * config.subsampling_factor);

    // CTC decoder built from the `decoding` section: 33 Russian glyphs + blank,
    // for `vocab_size = 34` total classes.
    assert!(matches!(config.decoder, CtcDecoder::Greedy(_)));
    assert_eq!(config.decoder.vocabulary().len(), 33);
    assert_eq!(config.decoder.blank_id(), 33);
    assert_eq!(config.decoder.total_vocab(), config.vocab_size);
    assert_eq!(config.decoder.vocabulary()[0], " ");
    assert_eq!(config.decoder.vocabulary()[32], "я");
}

#[test]
fn test_rnnt_config_from_json() {
    let config = GigaAmConfig::from_json(Path::new("tests/gigaam/rnnt_config.json")).unwrap();
    let transducer = config.transducer.as_ref().expect("rnnt transducer config");

    assert_eq!(config.vocab_size, 1025);
    assert_eq!(transducer.pred_hidden, 320);
    assert_eq!(transducer.pred_rnn_layers, 1);
    assert_eq!(transducer.joint_hidden, 320);
    assert_eq!(transducer.num_classes, 1025);
    assert_eq!(transducer.max_symbols_per_step, 10);
    assert!(transducer.vocabulary.is_empty());
    assert!(transducer.sentencepiece);
}

#[test]
fn test_config_rejects_unsupported_attention() {
    let err = expect_config_err(parse_temp_config(&minimal_config("rel_pos", 4, "htk", "null", 8, 8)));
    assert!(err.to_string().contains("self_attention_model"));
}

#[test]
fn test_config_rejects_unsupported_subsampling_factor() {
    let err = expect_config_err(parse_temp_config(&minimal_config("rotary", 2, "htk", "null", 8, 8)));
    assert!(err.to_string().contains("subsampling_factor"));
}

#[test]
fn test_config_rejects_unsupported_mel_frontend() {
    let err = expect_config_err(parse_temp_config(&minimal_config("rotary", 4, "slaney", "null", 8, 8)));
    assert!(err.to_string().contains("mel_scale"));

    let err = expect_config_err(parse_temp_config(&minimal_config("rotary", 4, "htk", r#""slaney""#, 8, 8)));
    assert!(err.to_string().contains("mel_norm"));

    let err = expect_config_err(parse_temp_config(&minimal_config("rotary", 4, "htk", "null", 16, 8)));
    assert!(err.to_string().contains("n_fft"));
}

#[test]
fn test_config_rejects_mel_bound_exceeding_rope_cache() {
    let json = minimal_config("rotary", 4, "htk", "null", 8, 8)
        .replace(r#""pos_emb_max_len": 8,"#, r#""pos_emb_max_len": 8, "max_mel_frames": 40,"#);
    let err = expect_config_err(parse_temp_config(&json));
    assert!(err.to_string().contains("max_mel_frames"));
}

fn expect_config_err(result: Result<GigaAmConfig, crate::gigaam::Error>) -> crate::gigaam::Error {
    match result {
        Ok(_) => panic!("expected config error"),
        Err(err) => err,
    }
}

fn parse_temp_config(json: &str) -> Result<GigaAmConfig, crate::gigaam::Error> {
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let path = std::env::temp_dir().join(format!("svod_gigaam_config_{stamp}.json"));
    std::fs::write(&path, json).unwrap();
    let out = GigaAmConfig::from_json(&path);
    let _ = std::fs::remove_file(path);
    out
}

fn minimal_config(
    self_attention_model: &str,
    subsampling_factor: usize,
    mel_scale: &str,
    mel_norm: &str,
    n_fft: usize,
    win_length: usize,
) -> String {
    format!(
        r#"{{
  "cfg": {{
    "model": {{
      "cfg": {{
        "preprocessor": {{
          "sample_rate": 16000,
          "features": 4,
          "win_length": {win_length},
          "hop_length": 4,
          "mel_scale": "{mel_scale}",
          "n_fft": {n_fft},
          "mel_norm": {mel_norm},
          "center": false
        }},
        "encoder": {{
          "d_model": 8,
          "n_layers": 1,
          "subsampling": "conv1d",
          "subs_kernel_size": 5,
          "subsampling_factor": {subsampling_factor},
          "ff_expansion_factor": 4,
          "self_attention_model": "{self_attention_model}",
          "pos_emb_max_len": 8,
          "n_heads": 2,
          "conv_kernel_size": 5,
          "conv_norm_type": "layer_norm"
        }},
        "head": {{ "num_classes": 2 }},
        "decoding": {{ "_target_": "CTCGreedyDecoding", "vocabulary": ["a"] }}
      }}
    }}
  }}
}}"#
    )
}
