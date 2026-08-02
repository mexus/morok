//! Forward shape + state-dict round-trip tests for Whisper model.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::state::HasStateDict;
use crate::whisper::{ModelDimensions, Whisper, WhisperSize};

fn make_dims() -> ModelDimensions {
    ModelDimensions::for_size(WhisperSize::Tiny)
}

#[test]
fn encoder_forward_shape() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());

    // mel: [1, n_mels, 3000]
    let mel = Tensor::zeros(&[1, dims.n_mels, 3000], DType::Float32).unwrap();

    let out = model.encode(&mel).unwrap();
    let shape = out.shape().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0].as_const(), Some(1));
    // conv stride 2: 3000/2 = 1500
    assert_eq!(shape[1].as_const(), Some(1500));
    assert_eq!(shape[2].as_const(), Some(dims.n_audio_state));
}

#[test]
fn decoder_forward_shape() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());

    // mel → encoder → features
    let mel = Tensor::zeros(&[1, dims.n_mels, 3000], DType::Float32).unwrap();
    let features = model.encode(&mel).unwrap();

    // tokens: [1, 4]
    let tokens = Tensor::from_slice([50363i32, 50364, 50359, 50363]).try_reshape([1usize, 4]).unwrap();

    let logits = model.decode(&tokens, &features, 0).unwrap();
    let shape = logits.shape().unwrap();
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0].as_const(), Some(1));
    assert_eq!(shape[1].as_const(), Some(4));
    assert_eq!(shape[2].as_const(), Some(dims.n_vocab));
}

#[test]
fn state_dict_round_trip() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());

    let sd = model.state_dict("");

    // Verify some key names
    assert!(sd.contains_key("encoder.conv1.weight"));
    assert!(sd.contains_key("encoder.conv1.bias"));
    assert!(sd.contains_key("encoder.conv2.weight"));
    assert!(sd.contains_key("encoder.positional_embedding"));
    assert!(sd.contains_key("encoder.blocks.0.attn.query.weight"));
    assert!(sd.contains_key("encoder.blocks.0.attn.query.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn.key.weight"));
    assert!(!sd.contains_key("encoder.blocks.0.attn.key.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn.value.weight"));
    assert!(sd.contains_key("encoder.blocks.0.attn.value.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn.out.weight"));
    assert!(sd.contains_key("encoder.blocks.0.attn.out.bias"));
    assert!(sd.contains_key("encoder.blocks.0.attn_ln.weight"));
    assert!(sd.contains_key("encoder.blocks.0.mlp.0.weight"));
    assert!(sd.contains_key("encoder.blocks.0.mlp.2.weight"));
    assert!(sd.contains_key("encoder.blocks.0.mlp_ln.weight"));
    assert!(sd.contains_key("encoder.ln_post.weight"));
    assert!(sd.contains_key("decoder.token_embedding.weight"));
    assert!(sd.contains_key("decoder.positional_embedding"));
    assert!(sd.contains_key("decoder.blocks.0.attn.query.weight"));
    assert!(sd.contains_key("decoder.blocks.0.cross_attn.query.weight"));
    assert!(sd.contains_key("decoder.blocks.0.cross_attn.key.weight"));
    assert!(sd.contains_key("decoder.blocks.0.mlp.0.weight"));
    assert!(sd.contains_key("decoder.ln.weight"));

    // Reload into a fresh model
    let mut model2 = Whisper::empty(dims);
    model2.load_state_dict(&sd, "").unwrap();
}

#[test]
fn dims_table() {
    let tiny = ModelDimensions::for_size(WhisperSize::Tiny);
    assert_eq!(tiny.n_audio_state, 384);
    assert_eq!(tiny.n_audio_head, 6);
    assert_eq!(tiny.n_audio_layer, 4);
    assert!(tiny.is_multilingual()); // "tiny" (non-.en) is multilingual

    let tiny_en = ModelDimensions::for_size(WhisperSize::TinyEn);
    assert!(!tiny_en.is_multilingual());

    let base = ModelDimensions::for_size(WhisperSize::Base);
    assert_eq!(base.n_audio_state, 512);
    assert_eq!(base.n_audio_head, 8);

    let large_v3 = ModelDimensions::for_size(WhisperSize::LargeV3);
    assert_eq!(large_v3.n_audio_state, 1280);
    assert_eq!(large_v3.n_mels, 128);
    assert_eq!(large_v3.n_vocab, 51866);

    let turbo = ModelDimensions::for_size(WhisperSize::Turbo);
    assert_eq!(turbo.n_audio_layer, 4);
    assert_eq!(turbo.n_text_layer, 8);
    assert_eq!(turbo.n_audio_state, 1280);
}

#[test]
fn alignment_heads_nonempty() {
    for size in [
        WhisperSize::Tiny,
        WhisperSize::Base,
        WhisperSize::Small,
        WhisperSize::Medium,
        WhisperSize::LargeV3,
        WhisperSize::Turbo,
    ] {
        let heads = size.alignment_heads();
        assert!(!heads.is_empty(), "{:?} has no alignment heads", size);
    }
}
