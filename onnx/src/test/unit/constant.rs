use crate::test::helpers::*;

#[test]
fn test_constant_value_float() {
    let registry = OpRegistry::new();
    let mut node = NodeProto::default();
    node.op_type = "Constant".to_string();
    node.attribute.push(make_attr_float("value_float", 3.14));

    let result = registry.dispatch("Constant", "", &[], &node).unwrap();
    let realized = result.contiguous().realize().unwrap();
    assert!(realized.buffer().is_some());
}

#[test]
fn test_constant_value_floats() {
    let registry = OpRegistry::new();
    let mut node = NodeProto::default();
    node.op_type = "Constant".to_string();
    node.attribute.push(make_attr_floats("value_floats", &[1.0, 2.0, 3.0]));

    let result = registry.dispatch("Constant", "", &[], &node).unwrap();
    let realized = result.realize().unwrap();
    assert!(realized.buffer().is_some());
}

#[test]
fn test_constant_value_int() {
    let registry = OpRegistry::new();
    let mut node = NodeProto::default();
    node.op_type = "Constant".to_string();
    node.attribute.push(make_attr_int("value_int", 42));

    let result = registry.dispatch("Constant", "", &[], &node).unwrap();
    let realized = result.contiguous().realize().unwrap();
    assert!(realized.buffer().is_some());
}

#[test]
fn test_constant_value_ints() {
    let registry = OpRegistry::new();
    let mut node = NodeProto::default();
    node.op_type = "Constant".to_string();
    node.attribute.push(make_attr_ints("value_ints", &[0, 1, 2, 3]));

    let result = registry.dispatch("Constant", "", &[], &node).unwrap();
    let realized = result.realize().unwrap();
    assert!(realized.buffer().is_some());
}

#[test]
fn test_constant_value_tensor() {
    let registry = OpRegistry::new();
    let mut node = NodeProto::default();
    node.op_type = "Constant".to_string();

    let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let tp = make_tensor_proto(raw, vec![2, 2], 1); // FLOAT
    node.attribute.push(make_attr_tensor("value", tp));

    let result = registry.dispatch("Constant", "", &[], &node).unwrap();
    let realized = result.realize().unwrap();
    assert!(realized.buffer().is_some());
}
