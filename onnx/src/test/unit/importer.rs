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
