use morok_arch::ctc::{CtcDecoder, GreedyDecoder};
use morok_dtype::DType;
use morok_tensor::{Tensor, Variable};

use crate::gigaam::{ConvNormType, GigaAm, GigaAmConfig, GigaAmEncoderJit, SubsamplingMode};

pub(super) fn test_config() -> GigaAmConfig {
    GigaAmConfig {
        max_batch_size: 8,
        n_mels: 64,
        d_model: 32,
        n_heads: 4,
        n_layers: 2,
        d_ff: 128,
        conv_kernel: 5,
        subsampling_factor: 4,
        subsampling_mode: SubsamplingMode::Conv1d,
        subs_kernel_size: 5,
        conv_norm_type: ConvNormType::LayerNorm,
        vocab_size: 34,
        sample_rate: 16000,
        n_fft: 320,
        hop_length: 160,
        win_length: 320,
        mel_center: false,
        max_mel_frames: 512,
        max_encoder_frames: 128,
        decoder: CtcDecoder::Greedy(GreedyDecoder::new(Vec::new())),
        transducer: None,
    }
}

fn model_with_random_weights() -> GigaAm {
    GigaAm::with_random_weights(test_config())
}

fn read_prefix_f32(t: &Tensor, len: usize) -> Vec<f32> {
    let buf = t.buffer().unwrap();
    buf.as_array::<f32>().unwrap().as_slice().unwrap()[..len].to_vec()
}

// ---------------------------------------------------------------------------
// Cheap default tests (no realize): config + structural invariants.
// ---------------------------------------------------------------------------

#[test]
fn test_output_length_matches_forward() {
    let model = GigaAm::with_random_weights(test_config());

    let x = Tensor::full(&[1, 100, 64], 0.0f32, DType::Float32).unwrap();
    let out = model.encoder.subsampling.forward(&x).unwrap();
    let actual_t = out.shape().unwrap()[1].as_const().unwrap();
    assert_eq!(model.encoder.subsampling_output_length(100), actual_t);

    let x2 = Tensor::full(&[1, 50, 64], 0.0f32, DType::Float32).unwrap();
    let out2 = model.encoder.subsampling.forward(&x2).unwrap();
    let actual_t2 = out2.shape().unwrap()[1].as_const().unwrap();
    assert_eq!(model.encoder.subsampling_output_length(50), actual_t2);
}

#[test]
fn test_rope_cache_uses_encoder_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();

    assert_eq!(model.encoder.cos_cache.shape().unwrap()[0].as_const().unwrap(), cfg.max_encoder_frames);
    assert_eq!(model.encoder.sin_cache.shape().unwrap()[0].as_const().unwrap(), cfg.max_encoder_frames);
    assert_ne!(cfg.max_encoder_frames, cfg.max_mel_frames);
}

#[test]
fn test_subsampled_max_mel_fits_encoder_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t_sub = model.encoder.subsampling_output_length(cfg.max_mel_frames);
    assert!(
        t_sub <= cfg.max_encoder_frames,
        "subsampled max_mel ({t_sub}) > max_encoder_frames ({})",
        cfg.max_encoder_frames
    );
}

#[test]
#[should_panic(expected = "with_b_bound(0) creates empty range")]
fn test_with_b_bound_panics_on_empty_range() {
    let model = model_with_random_weights();
    let _jit = GigaAmEncoderJit::new(model).with_b_bound(0);
}

// ---------------------------------------------------------------------------
// Heavy tests (realize / prepare): unique signal, gated behind --ignored.
// Run on demand when actively touching the GigaAM encoder, JIT plumbing, or
// the schedule pipeline. Each costs ~30s of prepare with random weights.
// ---------------------------------------------------------------------------

#[test]
#[ignore = "heavy: realize-based bounds-error JIT smoke"]
fn test_batched_jit_rejects_t_above_max_mel_frames() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let mut jit = GigaAmEncoderJit::new(model);
    jit.prepare(crate::jit::InputSpec::f32(&[1, cfg.n_mels, cfg.max_mel_frames]), crate::jit::InputSpec::i32(&[1]))
        .unwrap();
    let err = jit.execute_with_vars(&[("b", 1), ("t", cfg.max_mel_frames as i64 + 1)]).unwrap_err();
    assert_runtime_bounds_err(err);
}

#[test]
#[ignore = "heavy: full encoder forward at max_mel_frames"]
fn test_encode_batch_near_max_mel_runs() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t = cfg.max_mel_frames;

    let x = Tensor::full(&[1, cfg.n_mels, t], 0.1f32, DType::Float32).unwrap();
    let lengths = Tensor::from_slice([t as i32]);
    let b_var = Variable::new("B", 1, cfg.max_batch_size as i64);
    let t_var = Variable::new("T", 1, cfg.max_mel_frames as i64);
    let b1 = b_var.bind(1).unwrap();
    let t_bound = t_var.bind(t as i64).unwrap();

    let mut out = model.encoder.forward_batch(&x, &lengths, &b1, &t_bound).unwrap();
    out.realize().unwrap();
    assert!(out.buffer().unwrap().size() > 0);
}

#[test]
#[ignore = "heavy: batched-vs-single numerical consistency (the suite's only correctness assert)"]
fn test_single_vs_batch_consistency() {
    let model = model_with_random_weights();
    let d = test_config().d_model;
    let n_mels = test_config().n_mels;
    let t = 10;
    let t_sub = model.encoder.subsampling_output_length(t);

    let x1 = Tensor::full(&[1, n_mels, t], 0.5f32, DType::Float32).unwrap();
    let x2 = Tensor::full(&[1, n_mels, t], 0.3f32, DType::Float32).unwrap();
    let lengths_single = Tensor::from_slice([t as i32]);

    let b_var = Variable::new("B", 1, test_config().max_batch_size as i64);
    let t_var = Variable::new("T", 1, test_config().max_mel_frames as i64);
    let b1 = b_var.bind(1).unwrap();
    let t1 = t_var.bind(t as i64).unwrap();

    let mut out1 = model.encoder.forward_batch(&x1, &lengths_single, &b1, &t1).unwrap();
    out1.realize().unwrap();
    let data1 = read_prefix_f32(&out1, d * t_sub);

    let mut out2 = model.encoder.forward_batch(&x2, &lengths_single, &b1, &t1).unwrap();
    out2.realize().unwrap();
    let data2 = read_prefix_f32(&out2, d * t_sub);

    let batch = {
        let mut x1r = x1.clone();
        x1r.realize().unwrap();
        let d1 = x1r.as_vec::<f32>().unwrap();
        let mut x2r = x2.clone();
        x2r.realize().unwrap();
        let d2 = x2r.as_vec::<f32>().unwrap();
        let mut batch_data = vec![0.0f32; 2 * n_mels * t];
        batch_data[..n_mels * t].copy_from_slice(&d1);
        batch_data[n_mels * t..].copy_from_slice(&d2);
        ndarray::Array3::from_shape_vec((2, n_mels, t), batch_data).unwrap()
    };
    let batch_tensor = Tensor::from_ndarray(&batch);
    let batch_lengths = Tensor::from_slice([t as i32, t as i32]);

    let b2 = b_var.bind(2).unwrap();
    let mut batch_out = model.encoder.forward_batch(&batch_tensor, &batch_lengths, &b2, &t1).unwrap();
    batch_out.realize().unwrap();
    let batch_data = read_prefix_f32(&batch_out, 2 * d * t_sub);

    assert_eq!(data1.len() * 2, batch_data.len());

    for (i, (&b, &s)) in batch_data[..data1.len()].iter().zip(data1.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[0] mismatch at {}: batch={} single={}", i, b, s);
    }
    for (i, (&b, &s)) in batch_data[data1.len()..].iter().zip(data2.iter()).enumerate() {
        assert!((b - s).abs() < 1e-4, "batch[1] mismatch at {}: batch={} single={}", i, b, s);
    }
}

#[test]
#[ignore = "heavy: NaN/Inf detector across encoder forward"]
fn test_encode_batch_full_lengths_finite() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t = 256usize;

    let x = Tensor::full(&[2, cfg.n_mels, t], 0.1f32, DType::Float32).unwrap();
    let lengths = Tensor::from_slice([t as i32, t as i32]);

    let b_var = Variable::new("B", 1, cfg.max_batch_size as i64);
    let t_var = Variable::new("T", 1, cfg.max_mel_frames as i64);
    let b2 = b_var.bind(2).unwrap();
    let t_bound = t_var.bind(t as i64).unwrap();

    let mut out = model.encoder.forward_batch(&x, &lengths, &b2, &t_bound).unwrap();
    out.realize().unwrap();

    let buf = out.buffer().unwrap();
    let data = buf.as_array::<f32>().unwrap();
    for v in data.as_slice().unwrap() {
        assert!(v.is_finite(), "encode_batch produced non-finite value: {v}");
    }
}

#[test]
#[ignore = "heavy: symbolic seq-len threading through compiled kernels"]
fn test_encode_batch_respects_dynamic_seq_len() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let t_dynamic = 64usize;

    let mut jit = GigaAmEncoderJit::new(model);
    jit.prepare(
        crate::jit::InputSpec::f32(&[cfg.max_batch_size, cfg.n_mels, cfg.max_mel_frames]),
        crate::jit::InputSpec::i32(&[cfg.max_batch_size]),
    )
    .unwrap();
    let profiles = jit.execute_with_vars_profiled(&[("b", 1), ("t", t_dynamic as i64)]).unwrap();

    assert!(!profiles.is_empty(), "expected kernels for profiled dynamic execute");
    assert!(
        profiles.iter().any(|p| p.kernel.var_names.iter().any(|name| name == "t")),
        "expected at least one kernel to keep dynamic seq var 't'"
    );
}

fn assert_runtime_bounds_err(err: crate::jit::JitError) {
    match err {
        crate::jit::JitError::Runtime { source: morok_runtime::Error::Execution { reason } } => {
            assert!(reason.contains("outside bounds"), "unexpected runtime error: {reason}");
        }
        other => panic!("expected runtime bounds error, got {other:?}"),
    }
}

#[test]
#[ignore = "heavy: variable upper-bound enforcement at execute time"]
fn test_with_b_bound_shrinks_upper_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let mut jit = GigaAmEncoderJit::new(model).with_b_bound(2);
    jit.prepare(crate::jit::InputSpec::f32(&[2, cfg.n_mels, 64]), crate::jit::InputSpec::i32(&[2])).unwrap();
    jit.execute_with_vars(&[("b", 2), ("t", 64)]).unwrap();
    assert_runtime_bounds_err(jit.execute_with_vars(&[("b", 3), ("t", 64)]).unwrap_err());
}

#[test]
#[ignore = "heavy: with_t_fixed should fold the symbolic dim out of compiled kernels"]
fn test_with_t_fixed_specializes_kernels() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let pinned_t = 64usize;
    let mut jit = GigaAmEncoderJit::new(model).with_t_fixed(pinned_t);
    jit.prepare(crate::jit::InputSpec::f32(&[1, cfg.n_mels, pinned_t]), crate::jit::InputSpec::i32(&[1])).unwrap();
    let kernels = jit.prepared_kernels().unwrap();
    let any_kernel_keeps_t = kernels.iter().any(|k| k.kernel.var_names.iter().any(|n| n == "t"));
    assert!(!any_kernel_keeps_t, "with_t_fixed should fold `t` out of kernel var lists");
    let any_kernel_keeps_b = kernels.iter().any(|k| k.kernel.var_names.iter().any(|n| n == "b"));
    assert!(any_kernel_keeps_b, "`b` is still dynamic and should remain in some kernel's var list");

    jit.execute_with_vars(&[("b", 1), ("t", pinned_t as i64)]).unwrap();
}

#[test]
#[ignore = "heavy: variable lower-bound enforcement at execute time"]
fn test_with_b_min_bound_raises_lower_bound() {
    let model = model_with_random_weights();
    let cfg = test_config();
    let mut jit = GigaAmEncoderJit::new(model).with_b_min_bound(2);
    jit.prepare(crate::jit::InputSpec::f32(&[2, cfg.n_mels, 64]), crate::jit::InputSpec::i32(&[2])).unwrap();
    jit.execute_with_vars(&[("b", 2), ("t", 64)]).unwrap();
    assert_runtime_bounds_err(jit.execute_with_vars(&[("b", 1), ("t", 64)]).unwrap_err());
}
