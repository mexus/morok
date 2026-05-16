use morok_dtype::DType;
use morok_tensor::{Tensor, Variable};

use crate::resnet::{OutputMode, ResNet, ResNetConfig, ResNetDepth};

fn run_forward(depth: ResNetDepth, output: OutputMode, image_hw: usize) -> Tensor {
    let max_batch = 1;
    let config = ResNetConfig::new(depth, output).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);

    let images = Tensor::zeros(&[max_batch, 3, image_hw, image_hw], DType::Float32).unwrap();
    let var = Variable::new("b", 1, max_batch as i64);
    let b1 = var.bind(1).unwrap();

    let mut out = model.forward(&images, &b1).unwrap();
    out.realize().unwrap();
    out
}

/// Returns the shape, resolving symbolic dims via `vmax()` (which equals the
/// bound value here because every test binds the var to its declared upper
/// bound).
fn shape_const(t: &Tensor) -> Vec<usize> {
    t.shape()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, s)| {
            s.as_const().or_else(|| s.vmax()).unwrap_or_else(|| panic!("shape[{i}] has no concrete value: {s:?}"))
        })
        .collect()
}

#[test]
fn features_r18_returns_512_channel_map() {
    let out = run_forward(ResNetDepth::R18, OutputMode::Features, 32);
    assert_eq!(shape_const(&out), vec![1, 512, 1, 1]);
}

#[test]
fn classification_r18_returns_logits() {
    let out = run_forward(ResNetDepth::R18, OutputMode::Classification { num_classes: 10 }, 32);
    assert_eq!(shape_const(&out), vec![1, 10]);
}

/// Exercises the Bottleneck (1×1 → 3×3 → 1×1) path. Realising ResNet-50 through
/// the CPU JIT is a heavyweight operation (full graph compile), so this test
/// is gated behind `--ignored` — keep the BasicBlock smoke tests in the
/// default suite, run this one explicitly when touching layer code.
#[test]
#[ignore = "heavy: full ResNet-50 graph compile through the CPU backend"]
fn features_r50_returns_2048_channel_map() {
    let out = run_forward(ResNetDepth::R50, OutputMode::Features, 32);
    assert_eq!(shape_const(&out), vec![1, 2048, 1, 1]);
}

#[test]
fn feature_channels_matches_depth_expansion() {
    let r18 = ResNet::with_zero_weights(ResNetConfig::new(ResNetDepth::R18, OutputMode::Features));
    let r50 = ResNet::with_zero_weights(ResNetConfig::new(ResNetDepth::R50, OutputMode::Features));
    assert_eq!(r18.feature_channels(), 512);
    assert_eq!(r50.feature_channels(), 2048);
}
