use crate::test::helpers::*;

#[test]
fn test_registry_relu() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([-2.0f32, -1.0, 0.0, 1.0, 2.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Relu", "", &[x], &node);
    let result = result.unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_sigmoid() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([-1.0f32, 0.0, 1.0]);
    let node = NodeProto::default();

    let result = registry.dispatch("Sigmoid", "", &[x], &node);
    let result = result.unwrap().realize().unwrap();
    assert!(result.buffer().is_some());
}

#[test]
fn test_registry_log_softmax() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[2, 2]).unwrap();
    let node = NodeProto::default();

    let result = registry.dispatch("LogSoftmax", "", &[x], &node);
    assert!(result.is_ok());
}

#[test]
fn test_gelu_exact() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([0.0f32, 1.0, -1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_string("approximate", "none"));

    let result = registry.dispatch("Gelu", "", &[x], &node).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert!((vals[0] - 0.0).abs() < 1e-4, "gelu(0) = {}", vals[0]);
    assert!((vals[1] - 0.8413).abs() < 1e-3, "gelu(1) = {}", vals[1]);
    assert!((vals[2] - (-0.1587)).abs() < 1e-3, "gelu(-1) = {}", vals[2]);
}

#[test]
fn test_gelu_tanh_regression() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([0.0f32, 1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_string("approximate", "tanh"));

    let result = registry.dispatch("Gelu", "", &[x], &node).unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert!((vals[0] - 0.0).abs() < 1e-4, "gelu_tanh(0) = {}", vals[0]);
    assert!((vals[1] - 0.8412).abs() < 1e-3, "gelu_tanh(1) = {}", vals[1]);
}
