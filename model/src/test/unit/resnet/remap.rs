use morok_tensor::Tensor;

use crate::resnet::remap::fold_batchnorm;
use crate::state::StateDict;

fn tensor_f32(values: &[f32]) -> Tensor {
    Tensor::from_slice(values)
}

#[test]
fn fold_replaces_running_var_with_invstd() {
    let mut sd = StateDict::new();
    sd.insert("bn1.weight".into(), tensor_f32(&[1.0, 2.0]));
    sd.insert("bn1.bias".into(), tensor_f32(&[0.0, 0.0]));
    sd.insert("bn1.running_mean".into(), tensor_f32(&[0.0, 0.0]));
    sd.insert("bn1.running_var".into(), tensor_f32(&[0.25, 1.0])); // var
    sd.insert("bn1.num_batches_tracked".into(), tensor_f32(&[42.0]));

    let folded = fold_batchnorm(sd).expect("fold");
    assert!(!folded.contains_key("bn1.num_batches_tracked"), "num_batches_tracked must be dropped");
    assert!(folded.contains_key("bn1.weight"), "non-BN-stats keys preserved");

    let invstd = folded.get("bn1.running_var").expect("running_var slot still present after fold");
    let invstd_vals = invstd.as_vec::<f32>().expect("read invstd");
    // invstd = 1 / sqrt(var + 1e-5)
    let expected = vec![1.0 / (0.25_f32 + 1e-5).sqrt(), 1.0 / (1.0_f32 + 1e-5).sqrt()];
    assert!(
        invstd_vals.iter().zip(&expected).all(|(a, b)| (a - b).abs() < 1e-6),
        "got {:?}, expected {:?}",
        invstd_vals,
        expected
    );
}

#[test]
fn fold_handles_multiple_bn_layers() {
    let mut sd = StateDict::new();
    for prefix in ["bn1", "layer1.0.bn1", "layer4.2.downsample.1"] {
        sd.insert(format!("{prefix}.weight"), tensor_f32(&[1.0]));
        sd.insert(format!("{prefix}.bias"), tensor_f32(&[0.0]));
        sd.insert(format!("{prefix}.running_mean"), tensor_f32(&[0.0]));
        sd.insert(format!("{prefix}.running_var"), tensor_f32(&[1.0]));
        sd.insert(format!("{prefix}.num_batches_tracked"), tensor_f32(&[1.0]));
    }
    let folded = fold_batchnorm(sd).expect("fold");
    for prefix in ["bn1", "layer1.0.bn1", "layer4.2.downsample.1"] {
        assert!(!folded.contains_key(&format!("{prefix}.num_batches_tracked")));
        let invstd = folded.get(&format!("{prefix}.running_var")).unwrap();
        let v = invstd.as_vec::<f32>().unwrap();
        let expected = 1.0_f32 / (1.0_f32 + 1e-5).sqrt();
        assert!((v[0] - expected).abs() < 1e-6);
    }
}
