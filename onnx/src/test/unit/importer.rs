use std::collections::HashMap;

use crate::importer::{DimValue, InputSpec, OnnxImporter};
use crate::parser::onnx::{GraphProto, ModelProto, NodeProto, TensorProto, ValueInfoProto, tensor_proto};
use crate::test::helpers::*;

#[test]
fn test_importer_creation() {
    let _importer = OnnxImporter::new();
}

#[test]
fn test_prepare_minimal_model() {
    let importer = OnnxImporter::new();
    let model = make_minimal_model();
    let graph = importer.prepare(model).unwrap();

    assert!(graph.inputs.is_empty());
    assert_eq!(graph.outputs.len(), 1);
    assert_eq!(graph.outputs[0], "output");
    assert!(graph.initializers.contains_key("input"));
}

#[test]
fn test_execute_minimal_model() {
    let importer = OnnxImporter::new();
    let model = make_minimal_model();
    let graph = importer.prepare(model).unwrap();

    let outputs = importer.execute(&graph, HashMap::new()).unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_import_model_minimal() {
    let mut importer = OnnxImporter::new();
    let model = make_minimal_model();
    let outputs = importer.import_model(model).unwrap();

    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_multi_output_model() {
    let importer = OnnxImporter::new();
    let model = make_multi_output_model();
    let graph = importer.prepare(model).unwrap();

    assert_eq!(graph.outputs.len(), 2);
    assert!(graph.outputs.contains(&"out1".to_string()));
    assert!(graph.outputs.contains(&"out2".to_string()));

    let outputs = importer.execute(&graph, HashMap::new()).unwrap();
    assert_eq!(outputs.len(), 2);
    assert!(outputs.contains_key("out1"));
    assert!(outputs.contains_key("out2"));
}

#[test]
fn test_import_empty_model() {
    let importer = OnnxImporter::new();
    let model = ModelProto::default();
    let result = importer.prepare(model);
    assert!(result.is_err());
}

#[test]
fn test_import_with_add() {
    let mut importer = OnnxImporter::new();

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "add_test".to_string();

    for (name, values) in [("a", vec![1.0f32, 2.0, 3.0]), ("b", vec![4.0f32, 5.0, 6.0])] {
        let mut input = ValueInfoProto::default();
        input.name = name.to_string();
        graph.input.push(input);

        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Float as i32;
        init.dims = vec![3];
        init.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(init);
    }

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "Add".to_string();
    node.input.push("a".to_string());
    node.input.push("b".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let outputs = importer.import_model(model).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_import_with_matmul() {
    let mut importer = OnnxImporter::new();

    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "matmul_test".to_string();

    for (name, values) in [("a", vec![1.0f32, 2.0, 3.0, 4.0]), ("b", vec![5.0f32, 6.0, 7.0, 8.0])] {
        let mut input = ValueInfoProto::default();
        input.name = name.to_string();
        graph.input.push(input);

        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Float as i32;
        init.dims = vec![2, 2];
        init.raw_data = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        graph.initializer.push(init);
    }

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    let mut node = NodeProto::default();
    node.op_type = "MatMul".to_string();
    node.input.push("a".to_string());
    node.input.push("b".to_string());
    node.output.push("output".to_string());
    graph.node.push(node);

    model.graph = Some(graph);

    let outputs = importer.import_model(model).unwrap();
    assert_eq!(outputs.len(), 1);
    assert!(outputs.contains_key("output"));
}

#[test]
fn test_input_spec_static_shape() {
    let spec = InputSpec::new(
        vec![DimValue::Static(2), DimValue::Static(3)],
        DType::Scalar(morok_dtype::ScalarDType::Float32),
        false,
    );

    assert!(spec.is_static());
    assert_eq!(spec.static_shape(), Some(vec![2, 3]));
}

#[test]
fn test_input_spec_dynamic_shape() {
    let spec = InputSpec::new(
        vec![DimValue::Dynamic("batch".to_string()), DimValue::Static(3)],
        DType::Scalar(morok_dtype::ScalarDType::Float32),
        false,
    );

    assert!(!spec.is_static());
    assert_eq!(spec.static_shape(), None);
}

/// Build a minimal model with an If node.
///
/// Graph: condition (bool scalar initializer) -> If -> output
/// The then_branch adds `x + 10.0`, the else_branch adds `x + 20.0`.
fn make_if_model(cond_value: bool, x_values: &[f32]) -> ModelProto {
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_test".to_string();

    // Initializer: condition (bool scalar)
    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);

    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = vec![cond_value as u8];
    graph.initializer.push(cond_init);

    // Initializer: x (float tensor)
    let mut x_input = ValueInfoProto::default();
    x_input.name = "x".to_string();
    graph.input.push(x_input);

    let mut x_init = TensorProto::default();
    x_init.name = "x".to_string();
    x_init.data_type = tensor_proto::DataType::Float as i32;
    x_init.dims = vec![x_values.len() as i64];
    x_init.raw_data = x_values.iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(x_init);

    // Then branch: output = x + 10
    let then_add_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("ten".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(10.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
        ));
        n
    };
    let then_add = {
        let mut n = NodeProto::default();
        n.op_type = "Add".to_string();
        n.input.push("x".to_string());
        n.input.push("ten".to_string());
        n.output.push("then_out".to_string());
        n
    };
    let then_graph = make_graph(vec![then_add_const, then_add], vec!["x"], vec!["then_out"], vec![]);

    // Else branch: output = x + 20
    let else_add_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("twenty".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(20.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
        ));
        n
    };
    let else_add = {
        let mut n = NodeProto::default();
        n.op_type = "Add".to_string();
        n.input.push("x".to_string());
        n.input.push("twenty".to_string());
        n.output.push("else_out".to_string());
        n
    };
    let else_graph = make_graph(vec![else_add_const, else_add], vec!["x"], vec!["else_out"], vec![]);

    // If node
    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);
    model
}

#[test]
fn test_if_true_condition() {
    let mut importer = OnnxImporter::new();
    let model = make_if_model(true, &[1.0, 2.0, 3.0]);
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![11.0, 12.0, 13.0]); // x + 10
}

#[test]
fn test_if_false_condition() {
    let mut importer = OnnxImporter::new();
    let model = make_if_model(false, &[1.0, 2.0, 3.0]);
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    assert_eq!(vals, vec![21.0, 22.0, 23.0]); // x + 20
}

#[test]
fn test_if_where_path() {
    // Both branches produce same shape -> lazy where_ selection
    let mut importer = OnnxImporter::new();
    let model = make_if_model(true, &[5.0, 6.0]);
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // condition=true -> then_branch (x + 10) = [15, 16]
    assert_eq!(vals, vec![15.0, 16.0]);
}

#[test]
fn test_if_shape_mismatch_fallback() {
    // Build a model where then_branch outputs shape [3] and else_branch outputs shape [2]
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_mismatch_test".to_string();

    // Condition: true
    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);
    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = vec![1u8]; // true
    graph.initializer.push(cond_init);

    // Then branch: produces [1, 2, 3] (shape [3])
    let then_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("then_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [1.0f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![3],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let then_graph = make_graph(vec![then_const], vec![], vec!["then_out"], vec![]);

    // Else branch: produces [10, 20] (shape [2])
    let else_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("else_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![2],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let else_graph = make_graph(vec![else_const], vec![], vec!["else_out"], vec![]);

    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // condition=true, shapes differ -> eager fallback selects then_branch
    assert_eq!(vals, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_if_with_parent_scope() {
    // Subgraph references a tensor ("parent_val") from the parent graph scope
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "if_parent_scope_test".to_string();

    // Condition: true
    let mut cond_input = ValueInfoProto::default();
    cond_input.name = "condition".to_string();
    graph.input.push(cond_input);
    let mut cond_init = TensorProto::default();
    cond_init.name = "condition".to_string();
    cond_init.data_type = tensor_proto::DataType::Bool as i32;
    cond_init.dims = vec![];
    cond_init.raw_data = vec![1u8];
    graph.initializer.push(cond_init);

    // Parent value: [100, 200, 300]
    let mut pv_input = ValueInfoProto::default();
    pv_input.name = "parent_val".to_string();
    graph.input.push(pv_input);
    let mut pv_init = TensorProto::default();
    pv_init.name = "parent_val".to_string();
    pv_init.data_type = tensor_proto::DataType::Float as i32;
    pv_init.dims = vec![3];
    pv_init.raw_data = [100.0f32, 200.0, 300.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(pv_init);

    // Then branch: Identity of parent_val (references parent scope)
    let then_id = {
        let mut n = NodeProto::default();
        n.op_type = "Identity".to_string();
        n.input.push("parent_val".to_string());
        n.output.push("then_out".to_string());
        n
    };
    let then_graph = make_graph(vec![then_id], vec![], vec!["then_out"], vec![]);

    // Else branch: Constant [0]
    let else_const = {
        let mut n = NodeProto::default();
        n.op_type = "Constant".to_string();
        n.output.push("else_out".to_string());
        n.attribute.push(make_attr_tensor(
            "value",
            make_tensor_proto(
                [0.0f32, 0.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                vec![3],
                tensor_proto::DataType::Float as i32,
            ),
        ));
        n
    };
    let else_graph = make_graph(vec![else_const], vec![], vec!["else_out"], vec![]);

    let mut if_node = NodeProto::default();
    if_node.op_type = "If".to_string();
    if_node.input.push("condition".to_string());
    if_node.output.push("output".to_string());
    if_node.attribute.push(make_attr_graph("then_branch", then_graph));
    if_node.attribute.push(make_attr_graph("else_branch", else_graph));
    graph.node.push(if_node);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // condition=true, then_branch uses parent_val -> [100, 200, 300]
    assert_eq!(vals, vec![100.0, 200.0, 300.0]);
}

#[test]
fn test_if_nested() {
    // Outer If (cond=true) -> then_branch contains inner If (cond=false)
    // Inner If: then_branch = x + 1, else_branch = x + 2
    // Since outer=true, inner=false => result should be x + 2
    let mut model = ModelProto::default();
    let mut graph = GraphProto::default();
    graph.name = "nested_if_test".to_string();

    // Initializers
    for (name, val) in [("outer_cond", true), ("inner_cond", false)] {
        let mut inp = ValueInfoProto::default();
        inp.name = name.to_string();
        graph.input.push(inp);
        let mut init = TensorProto::default();
        init.name = name.to_string();
        init.data_type = tensor_proto::DataType::Bool as i32;
        init.dims = vec![];
        init.raw_data = vec![val as u8];
        graph.initializer.push(init);
    }

    let mut x_input = ValueInfoProto::default();
    x_input.name = "x".to_string();
    graph.input.push(x_input);
    let mut x_init = TensorProto::default();
    x_init.name = "x".to_string();
    x_init.data_type = tensor_proto::DataType::Float as i32;
    x_init.dims = vec![2];
    x_init.raw_data = [10.0f32, 20.0].iter().flat_map(|v| v.to_le_bytes()).collect();
    graph.initializer.push(x_init);

    // Inner then: x + 1
    let inner_then = {
        let c = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("one".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(1.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
            ));
            n
        };
        let add = {
            let mut n = NodeProto::default();
            n.op_type = "Add".to_string();
            n.input.push("x".to_string());
            n.input.push("one".to_string());
            n.output.push("inner_then_out".to_string());
            n
        };
        make_graph(vec![c, add], vec!["x"], vec!["inner_then_out"], vec![])
    };

    // Inner else: x + 2
    let inner_else = {
        let c = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("two".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(2.0f32.to_le_bytes().to_vec(), vec![1], tensor_proto::DataType::Float as i32),
            ));
            n
        };
        let add = {
            let mut n = NodeProto::default();
            n.op_type = "Add".to_string();
            n.input.push("x".to_string());
            n.input.push("two".to_string());
            n.output.push("inner_else_out".to_string());
            n
        };
        make_graph(vec![c, add], vec!["x"], vec!["inner_else_out"], vec![])
    };

    // Inner If node (inside outer then_branch)
    let inner_if = {
        let mut n = NodeProto::default();
        n.op_type = "If".to_string();
        n.input.push("inner_cond".to_string()); // references parent scope
        n.output.push("outer_then_out".to_string());
        n.attribute.push(make_attr_graph("then_branch", inner_then));
        n.attribute.push(make_attr_graph("else_branch", inner_else));
        n
    };
    let outer_then = make_graph(vec![inner_if], vec![], vec!["outer_then_out"], vec![]);

    // Outer else: constant [0, 0]
    let outer_else = {
        let c = {
            let mut n = NodeProto::default();
            n.op_type = "Constant".to_string();
            n.output.push("outer_else_out".to_string());
            n.attribute.push(make_attr_tensor(
                "value",
                make_tensor_proto(
                    [0.0f32, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect(),
                    vec![2],
                    tensor_proto::DataType::Float as i32,
                ),
            ));
            n
        };
        make_graph(vec![c], vec![], vec!["outer_else_out"], vec![])
    };

    // Outer If node
    let mut outer_if = NodeProto::default();
    outer_if.op_type = "If".to_string();
    outer_if.input.push("outer_cond".to_string());
    outer_if.output.push("output".to_string());
    outer_if.attribute.push(make_attr_graph("then_branch", outer_then));
    outer_if.attribute.push(make_attr_graph("else_branch", outer_else));
    graph.node.push(outer_if);

    let mut output = ValueInfoProto::default();
    output.name = "output".to_string();
    graph.output.push(output);

    model.graph = Some(graph);

    let mut importer = OnnxImporter::new();
    let outputs = importer.import_model(model).unwrap();

    let result = outputs.get("output").unwrap();
    let arr = result.to_ndarray::<f32>().unwrap();
    let vals: Vec<f32> = arr.iter().copied().collect();
    // outer_cond=true -> outer then_branch executes
    // inner_cond=false -> inner else_branch executes (x + 2)
    // x = [10, 20], so result = [12, 22]
    assert_eq!(vals, vec![12.0, 22.0]);
}

#[test]
fn test_load_silero_vad_prepare() {
    use prost::Message;
    use std::fs::File;
    use std::io::BufReader;

    let file = File::open("audio.onnx").expect("audio.onnx not found");
    let mut reader = BufReader::new(file);
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut reader, &mut bytes).unwrap();

    let model = ModelProto::decode(&bytes[..]).expect("Failed to decode ONNX");

    let importer = OnnxImporter::new();
    let graph = importer.prepare(model).expect("Failed to prepare graph");

    println!("Inputs: {:?}", graph.input_names());
    println!("Outputs: {:?}", graph.output_names());
    println!("Num initializers: {}", graph.initializers.len());
    println!("Num nodes: {}", graph.nodes.len());

    let mut op_counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for node in &graph.nodes {
        *op_counts.entry(node.op_type.as_str()).or_insert(0) += 1;
    }

    let mut ops: Vec<_> = op_counts.into_iter().collect();
    ops.sort_by(|a, b| b.1.cmp(&a.1));

    println!("\nOperators (sorted by frequency):");
    for (op, count) in ops {
        println!("  {:4}x  {}", count, op);
    }
}

// =========================================================================
// Transformer / LLM Operator tests
// =========================================================================

use crate::registry::OpRegistry;

#[test]
fn test_rms_norm() {
    let registry = OpRegistry::new();
    // X: [1, 4], scale: [4]
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 4]).unwrap();
    let scale = Tensor::from_slice([1.0f32, 1.0, 1.0, 1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("axis", -1));
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    let inputs = vec![Some(x), Some(scale)];
    let result = registry.dispatch_multi("RMSNormalization", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 4]);
    // With scale=[1,1,1,1], output = rms_norm(x)
    let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
    for i in 0..4 {
        let expected = (i + 1) as f32 * rms_inv;
        assert!((arr[[0, i]] - expected).abs() < 1e-4);
    }
}

#[test]
fn test_rms_norm_with_scale() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]).try_reshape(&[1, 4]).unwrap();
    let scale = Tensor::from_slice([2.0f32, 0.5, 1.0, 3.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    let inputs = vec![Some(x), Some(scale)];
    let result = registry.dispatch_multi("RMSNormalization", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    let rms_inv = 1.0 / (7.5f32 + 1e-5).sqrt();
    let scales = [2.0, 0.5, 1.0, 3.0];
    for i in 0..4 {
        let expected = (i + 1) as f32 * rms_inv * scales[i];
        assert!((arr[[0, i]] - expected).abs() < 1e-4);
    }
}

#[test]
fn test_skip_layer_norm() {
    let registry = OpRegistry::new();
    // x: [1, 3], skip: [1, 3], gamma: [3], beta: [3]
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape(&[1, 3]).unwrap();
    let skip = Tensor::from_slice([0.1f32, 0.2, 0.3]).try_reshape(&[1, 3]).unwrap();
    let gamma = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let beta = Tensor::from_slice([0.0f32, 0.0, 0.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    let inputs = vec![Some(x), Some(skip), Some(gamma), Some(beta)];
    let result = registry.dispatch_multi("SkipLayerNormalization", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 4);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 3]);
    // x_sum = x + skip = [1.1, 2.2, 3.3], layernorm → mean ~0
    let vals: Vec<f32> = arr.iter().copied().collect();
    let mean: f32 = vals.iter().sum::<f32>() / 3.0;
    assert!(mean.abs() < 1e-4, "layernorm mean should be ~0, got {mean}");
    // 4th output is x_sum
    let x_sum = result[3].to_ndarray::<f32>().unwrap();
    assert!((x_sum[[0, 0]] - 1.1).abs() < 1e-4);
}

#[test]
fn test_skip_layer_norm_no_optionals() {
    let registry = OpRegistry::new();
    let x = Tensor::from_slice([1.0f32, 2.0, 3.0]).try_reshape(&[1, 3]).unwrap();
    let skip = Tensor::from_slice([4.0f32, 5.0, 6.0]).try_reshape(&[1, 3]).unwrap();
    let gamma = Tensor::from_slice([1.0f32, 1.0, 1.0]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    // No beta, no bias
    let inputs = vec![Some(x), Some(skip), Some(gamma)];
    let result = registry.dispatch_multi("SkipLayerNormalization", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 4);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 3]);
}

#[test]
fn test_embed_layer_norm() {
    let registry = OpRegistry::new();
    // input_ids: [1, 3] (batch=1, seq=3)
    let input_ids = Tensor::from_slice([0i32, 1, 2]).try_reshape(&[1, 3]).unwrap();
    // word_embedding: [3, 4] (vocab=3, embed=4)
    let word_emb_data: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let word_emb = Tensor::from_slice(&word_emb_data).try_reshape(&[3, 4]).unwrap();
    // position_embedding: [3, 4] (max_pos=3, embed=4)
    let pos_emb = Tensor::from_slice([0.1f32; 12]).try_reshape(&[3, 4]).unwrap();
    // gamma: [4], beta: [4]
    let gamma = Tensor::from_slice([1.0f32; 4]);
    let beta = Tensor::from_slice([0.0f32; 4]);
    let mut node = NodeProto::default();
    node.attribute.push(make_attr_float("epsilon", 1e-5));
    // inputs: input_ids, segment_ids(None), word_emb, pos_emb, seg_emb(None), gamma, beta
    let inputs = vec![
        Some(input_ids),
        None, // segment_ids
        Some(word_emb),
        Some(pos_emb),
        None, // segment_embedding
        Some(gamma),
        Some(beta),
    ];
    let result = registry.dispatch_multi("EmbedLayerNormalization", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 3);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 3, 4]);
    // 3rd output is the raw embedding sum (before layernorm)
    let sum_arr = result[2].to_ndarray::<f32>().unwrap();
    assert_eq!(sum_arr.shape(), &[1, 3, 4]);
}

#[test]
fn test_rotary_embedding_split() {
    let registry = OpRegistry::new();
    // x: [1, 1, 2, 4] (B=1, H=1, S=2, D=4)
    let x_data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let x = Tensor::from_slice(&x_data).try_reshape(&[1, 1, 2, 4]).unwrap();
    // cos_cache: [2, 2], sin_cache: [2, 2]
    let cos_cache = Tensor::from_slice([1.0f32, 1.0, 1.0, 1.0]).try_reshape(&[2, 2]).unwrap();
    let sin_cache = Tensor::from_slice([0.0f32, 0.0, 0.0, 0.0]).try_reshape(&[2, 2]).unwrap();
    // position_ids: [1, 2]
    let pos_ids = Tensor::from_slice([0i32, 1]).try_reshape(&[1, 2]).unwrap();

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("interleaved", 0));
    node.attribute.push(make_attr_int("num_heads", 1));

    let inputs = vec![Some(x), Some(pos_ids), Some(cos_cache), Some(sin_cache)];
    let result = registry.dispatch_multi("RotaryEmbedding", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 1);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 2, 4]);
    // With cos=1, sin=0 → identity rotation, output = input
    let flat: Vec<f32> = arr.iter().copied().collect();
    for (i, val) in flat.iter().enumerate() {
        assert!((val - (i + 1) as f32).abs() < 1e-4, "rotary identity: got {val}, expected {}", i + 1);
    }
}

#[test]
fn test_attention_contrib_basic() {
    let registry = OpRegistry::new();
    // x: [1, 2, 4] (batch=1, seq=2, hidden=4)
    let x = Tensor::from_slice([1.0f32; 8]).try_reshape(&[1, 2, 4]).unwrap();
    // weights: [4, 12] (input_hidden=4, 3*hidden = 3*4 = 12)
    // ONNX contrib Attention weight layout: [input_hidden, 3*hidden]
    // qkv_hidden_sizes default: [4, 4, 4] (each third)
    // num_heads=2, so head_dim=2 for each of Q, K, V
    let mut w_data = vec![0.0f32; 48]; // 4 * 12
    // Identity-like: for each input dim i, map to Q[i], K[i], V[i]
    for i in 0..4 {
        w_data[i * 12 + i] = 1.0; // Q block
        w_data[i * 12 + 4 + i] = 1.0; // K block
        w_data[i * 12 + 8 + i] = 1.0; // V block
    }
    let weights = Tensor::from_slice(&w_data).try_reshape(&[4, 12]).unwrap();
    let bias = Tensor::from_slice([0.0f32; 12]);

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_heads", 2));

    let inputs = vec![Some(x), Some(weights), Some(bias)];
    let result = registry.dispatch_multi("Attention", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    assert!(result.len() >= 1);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 2, 4]);
}

#[test]
fn test_attention_contrib_causal() {
    let registry = OpRegistry::new();
    // x: [1, 2, 4], weights: [4, 6] with num_heads=1, so Q,K,V each have hidden=2
    let x = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).try_reshape(&[1, 2, 4]).unwrap();
    // Weight: [4, 6] (input=4, 3*hidden=6 where hidden=2, num_heads=1, head_dim=2)
    let mut w_data = vec![0.1f32; 24]; // 4 * 6
    // Make Q,K,V project to something predictable
    for i in 0..2 {
        w_data[i * 6 + i] = 1.0; // Q
        w_data[i * 6 + 2 + i] = 1.0; // K
        w_data[i * 6 + 4 + i] = 1.0; // V
    }
    let weights = Tensor::from_slice(&w_data).try_reshape(&[4, 6]).unwrap();
    let bias = Tensor::from_slice([0.0f32; 6]);

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("num_heads", 1));
    node.attribute.push(make_attr_int("unidirectional", 1));
    node.attribute.push(make_attr_ints("qkv_hidden_sizes", &[2, 2, 2]));

    let inputs = vec![Some(x), Some(weights), Some(bias)];
    let result = registry.dispatch_multi("Attention", "com.microsoft", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 2, 2]);
    // With unidirectional, position 0 can only attend to position 0
}

#[test]
fn test_attention_onnx_basic() {
    let registry = OpRegistry::new();
    // Q, K, V: [1, 2, 2, 2] (batch=1, heads=2, seq=2, dim=2)
    let q = Tensor::from_slice([1.0f32; 8]).try_reshape(&[1, 2, 2, 2]).unwrap();
    let k = q.clone();
    let v = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0]).try_reshape(&[1, 2, 2, 2]).unwrap();

    let node = NodeProto::default();
    let inputs = vec![Some(q), Some(k), Some(v)];
    let result = registry.dispatch_multi("Attention", "", &inputs, &node, i64::MAX).unwrap();
    assert_eq!(result.len(), 4);
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 2, 2, 2]);
}

#[test]
fn test_attention_onnx_causal() {
    let registry = OpRegistry::new();
    let q = Tensor::from_slice([1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5])
        .try_reshape(&[1, 1, 3, 4])
        .unwrap();
    let k = q.clone();
    let v = Tensor::from_slice([1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        .try_reshape(&[1, 1, 3, 4])
        .unwrap();

    let mut node = NodeProto::default();
    node.attribute.push(make_attr_int("is_causal", 1));
    let inputs = vec![Some(q), Some(k), Some(v)];
    let result = registry.dispatch_multi("Attention", "", &inputs, &node, i64::MAX).unwrap();
    let arr = result[0].to_ndarray::<f32>().unwrap();
    assert_eq!(arr.shape(), &[1, 1, 3, 4]);
    // Position 0 can only attend to position 0 -> output[0] = V[0] = [1, 0, 0, 0]
    assert!((arr[[0, 0, 0, 0]] - 1.0).abs() < 1e-4);
    assert!((arr[[0, 0, 0, 1]] - 0.0).abs() < 1e-4);
}

#[test]
fn test_attention_domain_dispatch() {
    // Same op name "Attention" dispatches to different implementations based on domain
    let registry = OpRegistry::new();

    // ONNX standard: takes pre-projected Q, K, V (4D)
    let q = Tensor::from_slice([1.0f32; 4]).try_reshape(&[1, 1, 2, 2]).unwrap();
    let k = q.clone();
    let v = q.clone();
    let node = NodeProto::default();
    let inputs_onnx = vec![Some(q), Some(k), Some(v)];
    let result_onnx = registry.dispatch_multi("Attention", "", &inputs_onnx, &node, i64::MAX).unwrap();
    assert_eq!(result_onnx.len(), 4); // ONNX returns [output, present_key, present_value, qk]

    // Microsoft contrib: takes x + weights (packed QKV projection)
    // x: [1, 2, 2], weights: [2, 6] (input_hidden=2, 3*hidden=6), num_heads=1, head_dim=2
    let x = Tensor::from_slice([1.0f32; 4]).try_reshape(&[1, 2, 2]).unwrap();
    let w = Tensor::from_slice([0.1f32; 12]).try_reshape(&[2, 6]).unwrap();
    let b = Tensor::from_slice([0.0f32; 6]);
    let mut node_contrib = NodeProto::default();
    node_contrib.attribute.push(make_attr_int("num_heads", 1));
    node_contrib.attribute.push(make_attr_ints("qkv_hidden_sizes", &[2, 2, 2]));
    let inputs_contrib = vec![Some(x), Some(w), Some(b)];
    let result_contrib =
        registry.dispatch_multi("Attention", "com.microsoft", &inputs_contrib, &node_contrib, i64::MAX).unwrap();
    assert_eq!(result_contrib.len(), 2); // Contrib returns [output, present]
}
