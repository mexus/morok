use crate::jit::InputSpec;
use crate::resnet::{OutputMode, ResNet, ResNetConfig, ResNetDepth, ResNetJit};

fn build_classifier_jit(max_batch: usize, num_classes: usize) -> ResNetJit {
    let config =
        ResNetConfig::new(ResNetDepth::R18, OutputMode::Classification { num_classes }).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);
    ResNetJit::new(model)
}

#[test]
fn prepare_and_execute_at_max_batch() {
    let max_batch = 2;
    let mut jit = build_classifier_jit(max_batch, 5);
    jit.prepare(InputSpec::f32(&[max_batch, 3, 32, 32])).unwrap();
    jit.execute_with_vars(&[("b", max_batch as i64)]).unwrap();
    let out = jit.output().unwrap();
    // [B, num_classes] = [2, 5] = 10 f32 elements
    assert_eq!(out.size(), 2 * 5 * std::mem::size_of::<f32>());
}

/// Confirms one prepared plan can serve multiple bound values of `b`. Gated
/// behind `--ignored` because each `execute_with_vars` triggers a kernel
/// dispatch through the full ResNet-18 graph; the smoke tests above already
/// validate the single-shot path.
#[test]
#[ignore = "heavy: three executes of the full ResNet-18 graph"]
fn rebind_batch_without_reprepare() {
    let max_batch = 4;
    let mut jit = build_classifier_jit(max_batch, 3);
    jit.prepare(InputSpec::f32(&[max_batch, 3, 32, 32])).unwrap();

    for b in [1, 2, 4] {
        jit.execute_with_vars(&[("b", b as i64)]).unwrap();
        let out = jit.output().unwrap();
        assert!(out.size() > 0, "output buffer empty for b={b}");
    }
}

#[test]
fn features_mode_returns_spatial_map() {
    let max_batch = 1;
    let config = ResNetConfig::new(ResNetDepth::R18, OutputMode::Features).with_max_batch_size(max_batch);
    let model = ResNet::with_zero_weights(config);
    let mut jit = ResNetJit::new(model);
    jit.prepare(InputSpec::f32(&[max_batch, 3, 32, 32])).unwrap();
    jit.execute_with_vars(&[("b", 1)]).unwrap();
    let out = jit.output().unwrap();
    // [1, 512, 1, 1] = 512 f32 elements
    assert_eq!(out.size(), 512 * std::mem::size_of::<f32>());
}
