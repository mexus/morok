//! Forward shape + state-dict round-trip tests for Whisper model.

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::jit::InputSpec;
use crate::state::HasStateDict;
use crate::whisper::{
    DecodeOptions, DecodeResult, ModelDimensions, Whisper, WhisperCrossKvJit, WhisperDecoderJit, WhisperPlan,
    WhisperPrefillJit, WhisperSize,
};

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

fn small_decoder_dims() -> ModelDimensions {
    ModelDimensions {
        n_mels: 4,
        n_audio_ctx: 5,
        n_audio_state: 8,
        n_audio_head: 2,
        n_audio_layer: 1,
        n_vocab: 16,
        n_text_ctx: 8,
        n_text_state: 8,
        n_text_head: 2,
        n_text_layer: 2,
        dtype: DType::Float32,
    }
}

#[test]
fn projected_cross_kv_and_prefill_shapes_are_concrete() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio = Tensor::zeros(&[1, dims.n_audio_ctx, dims.n_text_state], DType::Float32).unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3]).try_reshape([1usize, 3]).unwrap();

    let (cross_k, cross_v) = model.project_cross_kv(&audio).unwrap();
    let expected_cross = [1, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    for cache in [&cross_k, &cross_v] {
        let shape = cache.shape().unwrap();
        assert_eq!(shape.iter().map(|dim| dim.as_const().unwrap()).collect::<Vec<_>>(), expected_cross);
    }

    let (logits, self_k, self_v) = model.decode_prefill(&tokens, &cross_k, &cross_v, 0).unwrap();
    assert_eq!(
        logits.shape().unwrap().iter().map(|dim| dim.as_const().unwrap()).collect::<Vec<_>>(),
        [1, 3, dims.n_vocab]
    );
    let expected_self = [1, 3, dims.n_text_layer * dims.n_text_head, 4];
    for cache in [&self_k, &self_v] {
        assert_eq!(cache.shape().unwrap().iter().map(|dim| dim.as_const().unwrap()).collect::<Vec<_>>(), expected_self);
    }
}

#[test]
fn prepared_cross_kv_prefill_matches_direct_decoder() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio_values: Vec<f32> = (0..dims.n_audio_ctx * dims.n_text_state).map(|i| i as f32 * 0.01).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3]).try_reshape([1usize, 3]).unwrap();

    let mut direct = model.decode(&tokens, &audio, 0).unwrap();
    let (cross_k, cross_v) = model.project_cross_kv(&audio).unwrap();
    let (mut prepared, _, _) = model.decode_prefill(&tokens, &cross_k, &cross_v, 0).unwrap();
    direct.realize().unwrap();
    prepared.realize().unwrap();
    let direct = direct.as_vec::<f32>().unwrap();
    let prepared = prepared.as_vec::<f32>().unwrap();
    let max_delta = direct.iter().zip(&prepared).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "prepared cross-cache logits drifted by {max_delta}");
}

#[test]
#[ignore = "heavy: prepares the cross projection and prefill graphs through the CPU backend"]
fn prepared_cross_kv_graph_reuses_device_local_outputs() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let cache_shape = [1, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    let mut config = svod_tensor::PrepareConfig::from_env();
    config.device_local_outputs = true;

    let mut cross = WhisperCrossKvJit::new(model.clone());
    cross.prepare_with_config(InputSpec::f32(&[1, dims.n_audio_ctx, dims.n_text_state]), &config).unwrap();
    cross.execute().unwrap();

    let mut prefill = WhisperPrefillJit::new(model);
    prefill
        .prepare(
            InputSpec::i32(&[1, 3]),
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::f32(&cache_shape).device_local(),
        )
        .unwrap();
    let cross_k = cross.cross_k().unwrap();
    prefill.prepared_cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    let cross_v = cross.cross_v().unwrap();
    prefill.prepared_cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    prefill.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[1i32, 2, 3])).unwrap();
    prefill.execute().unwrap();

    assert_eq!(prefill.logits().unwrap().size(), 3 * dims.n_vocab * std::mem::size_of::<f32>());
    assert_eq!(prefill.prepared_cross_k_mut().unwrap().size(), cross_k.size());
    assert_eq!(prefill.prepared_cross_v_mut().unwrap().size(), cross_v.size());

    let mut detector = WhisperDecoderJit::new(Whisper::empty(dims.clone()));
    detector
        .prepare(
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::i32(&[1, dims.n_text_ctx]),
        )
        .unwrap();
    detector.prepared_cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    detector.prepared_cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    detector.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&vec![0i32; dims.n_text_ctx])).unwrap();
    detector.execute().unwrap();
    assert_eq!(detector.output().unwrap().size(), dims.n_text_ctx * dims.n_vocab * std::mem::size_of::<f32>());
}

#[test]
fn alignment_forward_exports_only_selected_heads() {
    let dims = make_dims();
    let model = Whisper::empty(dims.clone());
    let features = Tensor::zeros(&[2, 8, dims.n_text_state], DType::Float32).unwrap();
    let tokens = Tensor::from_slice([50363i32, 50364, 50359, 50257, 50363, 50364, 50359, 50257])
        .try_reshape([2usize, 4])
        .unwrap();
    let heads = &[(2, 2), (3, 0)];

    let qk = model.align(&tokens, &features, heads).unwrap();
    let shape = qk.shape().unwrap();
    assert_eq!(shape[0].as_const(), Some(2));
    assert_eq!(shape[1].as_const(), Some(heads.len()));
    assert_eq!(shape[2].as_const(), Some(4));
    assert_eq!(shape[3].as_const(), Some(8));
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

#[test]
fn prepared_plan_has_concrete_nonzero_capacities() {
    let dims = ModelDimensions::for_size(WhisperSize::LargeV2);
    let plan = WhisperPlan::for_model(&dims, WhisperSize::LargeV2);
    assert!(plan.encoder_batch > 0);
    assert!(plan.decoder_slots > 0);
    assert!(plan.alignment_batch > 0);
    assert!(plan.alignment_batch <= plan.encoder_batch);
    plan.validate().unwrap();
}

#[test]
fn no_speech_skip_respects_logprob_override() {
    let options = DecodeOptions::default();
    let mut result = DecodeResult {
        tokens: vec![1, 2],
        token_probs: vec![0.2, 0.3],
        text: "hallucination".to_string(),
        avg_logprob: -2.0,
        no_speech_prob: 0.9,
        temperature: 0.0,
        compression_ratio: 1.0,
        language: Some("en".to_string()),
    };
    assert!(result.should_skip(&options));
    result.clear_speech();
    assert!(result.tokens.is_empty());
    assert!(result.text.is_empty());

    result.avg_logprob = -0.5;
    assert!(!result.should_skip(&options));
}
