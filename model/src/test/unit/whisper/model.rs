//! Forward shape + state-dict round-trip tests for Whisper model.

use svod_dtype::DType;
use svod_ir::{AxisType, ConstValue, Op};
use svod_tensor::Tensor;

use crate::jit::InputSpec;
use crate::state::HasStateDict;
use crate::whisper::{
    DecodeOptions, DecodeResult, DecodeStrategy, FallbackPolicy, ModelDimensions, Whisper, WhisperAlignmentJit,
    WhisperAlignmentModel, WhisperCrossKvJit, WhisperDecoderJit, WhisperPlan, WhisperPrefillJit, WhisperSize,
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

fn reference_cross_kv_projection(model: &Whisper, audio: &Tensor) -> (Tensor, Tensor) {
    let mut keys = Vec::with_capacity(model.decoder.blocks.len());
    let mut values = Vec::with_capacity(model.decoder.blocks.len());
    for block in &model.decoder.blocks {
        let key = block.cross_attn.key.forward(audio).unwrap();
        let value = block.cross_attn.value.forward(audio).unwrap();
        keys.push(block.cross_attn.split_heads(&key).unwrap().try_permute(&[0, 2, 1, 3]).unwrap());
        values.push(block.cross_attn.split_heads(&value).unwrap().try_permute(&[0, 2, 1, 3]).unwrap());
    }
    let keys = Tensor::cat(&keys.iter().collect::<Vec<_>>(), 2).unwrap().cast(DType::Float32).unwrap();
    let values = Tensor::cat(&values.iter().collect::<Vec<_>>(), 2).unwrap().cast(DType::Float32).unwrap();
    (keys, values)
}

#[test]
fn materialized_cross_kv_matches_reference_projection() {
    let mut dims = small_decoder_dims();
    dims.n_text_layer = 3;
    let seed = Whisper::empty(dims.clone());
    let model = Whisper::from_state_dict(&seed.state_dict(""), dims.clone()).unwrap();
    let audio_values: Vec<f32> =
        (0..2 * dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 31.0) * 0.017).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([2usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();

    let (mut expected_k, mut expected_v) = reference_cross_kv_projection(&model, &audio);
    let (mut actual_k, mut actual_v) = model.project_cross_kv(&audio).unwrap();
    Tensor::realize_batch([&mut expected_k, &mut expected_v, &mut actual_k, &mut actual_v]).unwrap();

    let expected_shape = [2, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    for (expected, actual) in [(&expected_k, &actual_k), (&expected_v, &actual_v)] {
        assert_eq!(actual.uop().dtype(), DType::Float32);
        assert_eq!(
            actual.shape().unwrap().iter().map(|dim| dim.as_const().unwrap()).collect::<Vec<_>>(),
            expected_shape
        );
        let expected = expected.as_vec::<f32>().unwrap();
        let actual = actual.as_vec::<f32>().unwrap();
        let max_delta = expected.iter().zip(&actual).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_delta < 1e-5, "materialized cross projection drifted by {max_delta}");
    }
}

#[test]
fn prepared_cross_kv_materializes_each_projection_before_packing() {
    let mut dims = small_decoder_dims();
    dims.n_text_layer = 3;
    let seed = Whisper::empty(dims.clone());
    let model = Whisper::from_state_dict(&seed.state_dict(""), dims.clone()).unwrap();
    let mut jit = WhisperCrossKvJit::new(model);
    jit.prepare(InputSpec::f32(&[1, dims.n_audio_ctx, dims.n_text_state])).unwrap();

    let kernels = jit.prepared_kernels().unwrap();
    let reduction_counts: Vec<_> = kernels
        .iter()
        .map(|kernel| {
            kernel
                .ast
                .toposort()
                .into_iter()
                .filter(|uop| matches!(uop.op(), Op::Range { axis_type: AxisType::Reduce, .. }))
                .count()
        })
        .collect();
    assert!(
        reduction_counts.iter().all(|&count| count <= 1),
        "no dispatch may contain repeated independent reductions: {reduction_counts:?}"
    );
    assert_eq!(
        reduction_counts.iter().filter(|&&count| count == 1).count(),
        2 * dims.n_text_layer,
        "each layer must have independent key and value projection kernels: {reduction_counts:?}"
    );
    assert!(
        reduction_counts.iter().filter(|&&count| count == 0).count() >= 2,
        "key and value packing must remain reduction-free: {reduction_counts:?}"
    );
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
fn one_token_language_logits_match_full_context_sot_logits() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let audio_values: Vec<f32> = (0..dims.n_audio_ctx * dims.n_text_state).map(|i| i as f32 * 0.013).collect();
    let audio = Tensor::from_slice(audio_values).try_reshape([1usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let (cross_k, cross_v) = model.project_cross_kv(&audio).unwrap();
    let mut padded_tokens = vec![0i32; dims.n_text_ctx];
    padded_tokens[0] = 1;
    let full_tokens = Tensor::from_slice(padded_tokens).try_reshape([1usize, dims.n_text_ctx]).unwrap();
    let one_token = Tensor::from_slice([1i32]).try_reshape([1usize, 1]).unwrap();

    let mut full = model.decode_with_cross_kv(&full_tokens, &cross_k, &cross_v).unwrap();
    let mut one = model.decode_with_cross_kv(&one_token, &cross_k, &cross_v).unwrap();
    Tensor::realize_batch([&mut full, &mut one]).unwrap();
    let full = full.as_vec::<f32>().unwrap();
    let one = one.as_vec::<f32>().unwrap();
    let max_delta = full[..dims.n_vocab].iter().zip(&one).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "one-token SOT logits drifted by {max_delta}");

    let language_tokens = [2usize, 5, 9, 12];
    let rank = |logits: &[f32]| {
        let mut ranked = language_tokens.map(|token| (token, logits[token]));
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
        ranked
    };
    assert_eq!(rank(&full[..dims.n_vocab]).map(|(token, _)| token), rank(&one).map(|(token, _)| token));
}

#[test]
fn prepared_language_detector_has_one_token_and_one_logits_row() {
    const WHISPER_TEXT_CONTEXT: i64 = 448;

    let dims = small_decoder_dims();
    let cache_shape = [1, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    let model = Whisper::empty(dims.clone());
    let audio = Tensor::zeros(&[1, dims.n_audio_ctx, dims.n_text_state], DType::Float32).unwrap();
    let (cross_k, cross_v) = model.project_cross_kv(&audio).unwrap();
    let token = Tensor::from_slice([1i32]).try_reshape([1usize, 1]).unwrap();
    let logits = model.decode_with_cross_kv(&token, &cross_k, &cross_v).unwrap();
    assert_eq!(
        logits.shape().unwrap().iter().map(|dim| dim.as_const().unwrap()).collect::<Vec<_>>(),
        [1, 1, dims.n_vocab]
    );

    let mut detector = WhisperDecoderJit::new(model);
    detector
        .prepare(
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::i32(&[1, 1]),
        )
        .unwrap();
    assert_eq!(detector.tokens_mut().unwrap().size(), std::mem::size_of::<i32>());
    assert_eq!(detector.output().unwrap().size(), dims.n_vocab * std::mem::size_of::<f32>());
    assert!(detector.prepared_kernels().unwrap().iter().all(|kernel| {
        kernel.ast.toposort().into_iter().all(|uop| {
            !matches!(uop.op(), Op::Const(value) if matches!(value.0, ConstValue::Int(WHISPER_TEXT_CONTEXT) | ConstValue::UInt(448)))
        })
    }));
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
            InputSpec::i32(&[1, 1]),
        )
        .unwrap();
    detector.prepared_cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    detector.prepared_cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    detector.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[0i32])).unwrap();
    detector.execute().unwrap();
    assert_eq!(detector.output().unwrap().size(), dims.n_vocab * std::mem::size_of::<f32>());
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

    let (cross_k, cross_v) = model.project_cross_kv(&features).unwrap();
    let qk = model.align_with_cross_kv(&tokens, &cross_k, &cross_v, heads).unwrap();
    let shape = qk.shape().unwrap();
    assert_eq!(shape[0].as_const(), Some(2));
    assert_eq!(shape[1].as_const(), Some(heads.len()));
    assert_eq!(shape[2].as_const(), Some(4));
    assert_eq!(shape[3].as_const(), Some(8));
}

fn eager_audio_feature_alignment_reference(
    model: &Whisper,
    tokens: &Tensor,
    features: &Tensor,
    alignment_heads: &[(usize, usize)],
) -> Tensor {
    let seq_len = tokens.shape().unwrap()[1].as_const().unwrap();
    let tok_emb = model.decoder.token_embedding.embedding(tokens).unwrap();
    let pos_emb = model.decoder.positional_embedding.try_shrink([Some((0isize, seq_len as isize)), None]).unwrap();
    let mut x = tok_emb.try_add(&pos_emb).unwrap().cast(features.uop().dtype()).unwrap();
    let mask = crate::whisper::causal_mask(seq_len, x.uop().dtype().clone()).unwrap();
    let mut selected: Vec<Option<Tensor>> = (0..alignment_heads.len()).map(|_| None).collect();

    for (layer, block) in model.decoder.blocks.iter().enumerate() {
        let h = block.attn_ln.apply(&x).unwrap();
        x = x.try_add(&block.attn.forward(&h, None, Some(&mask)).unwrap()).unwrap();

        let h = block.cross_attn_ln.apply(&x).unwrap();
        x = x.try_add(&block.cross_attn.forward(&h, Some(features), None).unwrap()).unwrap();
        let q = block.cross_attn.split_heads(&block.cross_attn.query.forward(&h).unwrap()).unwrap();
        let k = block.cross_attn.split_heads(&block.cross_attn.key.forward(features).unwrap()).unwrap();
        for (index, &(_, head)) in alignment_heads.iter().enumerate().filter(|&(_, &(l, _))| l == layer) {
            let q = q.try_shrink([None, Some((head as isize, head as isize + 1)), None, None]).unwrap();
            let k = k.try_shrink([None, Some((head as isize, head as isize + 1)), None, None]).unwrap();
            let scores = q.matmul(&k.try_transpose(-1, -2).unwrap()).unwrap();
            let scale = Tensor::const_(
                ConstValue::Float(1.0 / ((model.decoder.n_state / model.decoder.n_head) as f64).sqrt()),
                scores.uop().dtype().clone(),
            );
            selected[index] = Some(scores.try_mul(&scale).unwrap());
        }

        let h = block.mlp_ln.apply(&x).unwrap();
        let h = h.linear().weight(&block.mlp0_w).bias(&block.mlp0_b).call().unwrap().gelu_exact().unwrap();
        let h = h.linear().weight(&block.mlp1_w).bias(&block.mlp1_b).call().unwrap();
        x = x.try_add(&h).unwrap();
    }

    let selected: Vec<_> = selected.into_iter().map(Option::unwrap).collect();
    Tensor::cat(&selected.iter().collect::<Vec<_>>(), 1).unwrap().cast(DType::Float32).unwrap()
}

#[test]
fn cached_cross_alignment_matches_audio_feature_reference() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let values: Vec<f32> =
        (0..2 * dims.n_audio_ctx * dims.n_text_state).map(|index| (index as f32 - 17.0) * 0.013).collect();
    let features = Tensor::from_slice(values).try_reshape([2usize, dims.n_audio_ctx, dims.n_text_state]).unwrap();
    let tokens = Tensor::from_slice([1i32, 2, 3, 4, 4, 3, 2, 1]).try_reshape([2usize, 4]).unwrap();
    let heads = [(1, 1), (0, 0)];

    let mut reference = eager_audio_feature_alignment_reference(&model, &tokens, &features, &heads);
    let (cross_k, cross_v) = model.project_cross_kv(&features).unwrap();
    let mut cached = model.align_with_cross_kv(&tokens, &cross_k, &cross_v, &heads).unwrap();
    reference.realize().unwrap();
    cached.realize().unwrap();
    let reference = reference.as_vec::<f32>().unwrap();
    let cached = cached.as_vec::<f32>().unwrap();
    let max_delta = reference.iter().zip(&cached).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    assert!(max_delta < 1e-5, "cached-cross alignment drifted by {max_delta}");
}

#[test]
#[ignore = "heavy: prepares cross projection, prefill, and alignment graphs through the CPU backend"]
fn recognition_cross_kv_seeds_prefill_and_alignment_device_locally() {
    let dims = small_decoder_dims();
    let model = Whisper::empty(dims.clone());
    let cache_shape = [1, dims.n_audio_ctx, dims.n_text_layer * dims.n_text_head, 4];
    let mut config = svod_tensor::PrepareConfig::from_env();
    config.device_local_outputs = true;

    let mut cross = WhisperCrossKvJit::new(model.clone());
    cross.prepare_with_config(InputSpec::f32(&[1, dims.n_audio_ctx, dims.n_text_state]), &config).unwrap();
    cross.execute().unwrap();

    let mut prefill = WhisperPrefillJit::new(model.clone());
    prefill
        .prepare(
            InputSpec::i32(&[1, 3]),
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::f32(&cache_shape).device_local(),
        )
        .unwrap();
    let alignment_model = WhisperAlignmentModel::new(model, vec![(0, 0), (1, 1)]);
    let mut alignment = WhisperAlignmentJit::new(alignment_model);
    alignment
        .prepare(
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::f32(&cache_shape).device_local(),
            InputSpec::i32(&[1, 3]),
        )
        .unwrap();

    let cross_k = cross.cross_k().unwrap();
    let cross_v = cross.cross_v().unwrap();
    prefill.prepared_cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    prefill.prepared_cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    alignment.cross_k_mut().unwrap().copy_region_from(0, cross_k, 0, cross_k.size()).unwrap();
    alignment.cross_v_mut().unwrap().copy_region_from(0, cross_v, 0, cross_v.size()).unwrap();
    prefill.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[1i32, 2, 3])).unwrap();
    alignment.tokens_mut().unwrap().copyin(bytemuck::cast_slice(&[1i32, 2, 3])).unwrap();
    prefill.execute().unwrap();
    alignment.execute().unwrap();

    assert_eq!(prefill.logits().unwrap().size(), 3 * dims.n_vocab * std::mem::size_of::<f32>());
    assert_eq!(alignment.output().unwrap().size(), 2 * 3 * dims.n_audio_ctx * std::mem::size_of::<f32>());
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
fn default_decode_policy_is_explicit_openai_fallback() {
    let options = DecodeOptions::default();
    assert_eq!(options.strategy, DecodeStrategy::Beam { size: 5 });
    assert_eq!(options.fallback.unwrap().sampling_temperatures, [0.2, 0.4, 0.6, 0.8, 1.0]);
}

#[test]
fn decode_policy_rejects_invalid_geometry_and_temperatures() {
    let invalid_beam = DecodeOptions { strategy: DecodeStrategy::Beam { size: 0 }, ..Default::default() };
    assert!(invalid_beam.validate().is_err());

    let invalid_sample = DecodeOptions { strategy: DecodeStrategy::Sample { temperature: 0.0 }, ..Default::default() };
    assert!(invalid_sample.validate().is_err());

    let invalid_fallback = DecodeOptions {
        fallback: Some(FallbackPolicy { sampling_temperatures: vec![f32::NAN], ..FallbackPolicy::default() }),
        ..Default::default()
    };
    assert!(invalid_fallback.validate().is_err());

    let invalid_silence = DecodeOptions { no_speech_threshold: Some(1.1), ..Default::default() };
    assert!(invalid_silence.validate().is_err());
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

#[test]
fn confident_silence_cancels_quality_fallback() {
    let options = DecodeOptions::default();
    let fallback = options.fallback.as_ref().unwrap();
    let result = DecodeResult {
        tokens: Vec::new(),
        token_probs: Vec::new(),
        text: String::new(),
        avg_logprob: -2.0,
        no_speech_prob: 0.9,
        temperature: 0.0,
        compression_ratio: 3.0,
        language: Some("en".to_string()),
    };
    assert!(!crate::whisper::decode::check_fallback(&result, fallback, &options));
}
