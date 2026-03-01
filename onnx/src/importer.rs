//! ONNX model importer - converts ONNX protobuf to Morok Tensors.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use morok_dtype::DType;
use morok_tensor::Tensor;
use prost::Message;
use snafu::ResultExt;

use crate::error::{EmptyModelSnafu, IoSnafu, MissingInputSnafu, ProtobufDecodeSnafu, Result};
use crate::parser::onnx::{ModelProto, NodeProto, ValueInfoProto};
use crate::registry::{OpRegistry, OpSetId, convert_onnx_dtype, onnx_opset_version, tensor_from_proto_ext};

/// Dimension value - either static (known size) or dynamic (named, e.g., batch dim).
#[derive(Debug, Clone, PartialEq)]
pub enum DimValue {
    /// Static dimension with known size
    Static(usize),
    /// Dynamic dimension with symbolic name (e.g., "batch_size")
    Dynamic(String),
}

/// Input specification extracted from ONNX ValueInfoProto.
#[derive(Debug, Clone)]
pub struct InputSpec {
    /// Shape dimensions (can be static or dynamic)
    pub shape: Vec<DimValue>,
    /// Data type
    pub dtype: DType,
    /// Whether this input is optional
    pub optional: bool,
}

impl InputSpec {
    /// Create a new input spec.
    pub fn new(shape: Vec<DimValue>, dtype: DType, optional: bool) -> Self {
        Self { shape, dtype, optional }
    }

    /// Check if all dimensions are static (no dynamic dims).
    pub fn is_static(&self) -> bool {
        self.shape.iter().all(|d| matches!(d, DimValue::Static(_)))
    }

    /// Get static shape if all dimensions are known.
    pub fn static_shape(&self) -> Option<Vec<usize>> {
        self.shape
            .iter()
            .map(|d| match d {
                DimValue::Static(s) => Some(*s),
                DimValue::Dynamic(_) => None,
            })
            .collect()
    }
}

/// Prepared ONNX graph - structure extracted, ready for execution.
pub struct OnnxGraph {
    /// Input specifications (name -> spec)
    pub inputs: HashMap<String, InputSpec>,
    /// Output names in order
    pub outputs: Vec<String>,
    /// Initializers/weights (name -> tensor)
    pub initializers: HashMap<String, Tensor>,
    /// Nodes in topological execution order
    pub nodes: Vec<NodeProto>,
    /// Opset versions
    pub opsets: Vec<OpSetId>,
}

impl OnnxGraph {
    /// Get the list of input names.
    pub fn input_names(&self) -> Vec<&str> {
        self.inputs.keys().map(|s| s.as_str()).collect()
    }

    /// Get the list of output names.
    pub fn output_names(&self) -> &[String] {
        &self.outputs
    }

    /// Check if an input is optional.
    pub fn is_input_optional(&self, name: &str) -> bool {
        self.inputs.get(name).map(|s| s.optional).unwrap_or(false)
    }
}

/// ONNX model importer.
///
/// Converts ONNX models to Morok Tensors using a two-phase approach:
/// 1. `prepare()` - Extract graph structure without executing
/// 2. `execute()` - Run graph with provided inputs
pub struct OnnxImporter {
    /// Operator registry for dispatch
    registry: OpRegistry,
    /// Directory containing the model file (for external data loading)
    model_dir: Option<std::path::PathBuf>,
}

impl OnnxImporter {
    /// Create a new ONNX importer.
    pub fn new() -> Self {
        Self { registry: OpRegistry::new(), model_dir: None }
    }

    /// Import ONNX model from file path (convenience method for all-initializer models).
    pub fn import_path<P: AsRef<Path>>(&mut self, path: P) -> Result<HashMap<String, Tensor>> {
        self.model_dir = path.as_ref().parent().map(|p| p.to_path_buf());
        let file = File::open(path.as_ref()).context(IoSnafu)?;
        let mut reader = BufReader::new(file);
        self.import_reader(&mut reader)
    }

    /// Import ONNX model from a reader.
    pub fn import_reader<R: Read>(&mut self, reader: &mut R) -> Result<HashMap<String, Tensor>> {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes).context(IoSnafu)?;
        self.import_bytes(&bytes)
    }

    /// Import from bytes.
    pub fn import_bytes(&mut self, bytes: &[u8]) -> Result<HashMap<String, Tensor>> {
        let model = ModelProto::decode(bytes).context(ProtobufDecodeSnafu)?;
        self.import_model(model)
    }

    /// Import from parsed ModelProto (convenience for all-initializer models).
    ///
    /// For models with runtime inputs, use `prepare()` + `execute()` instead.
    pub fn import_model(&mut self, model: ModelProto) -> Result<HashMap<String, Tensor>> {
        let graph = self.prepare(model)?;

        // If all inputs have initializers, we can execute directly
        let has_runtime_inputs = graph.inputs.keys().any(|name| !graph.initializers.contains_key(name));

        if has_runtime_inputs {
            // Need user inputs - return empty map, user should use execute()
            // Actually, for backward compat, let's try with empty inputs and see what happens
            // This will error on missing inputs, which is expected
            let inputs = HashMap::new();
            self.execute_with_initializers(&graph, inputs)
        } else {
            // All inputs are initializers - can execute with empty runtime inputs
            let inputs = HashMap::new();
            self.execute_with_initializers(&graph, inputs)
        }
    }

    /// Phase 1: Extract graph structure from ONNX model.
    ///
    /// Returns an `OnnxGraph` that can be executed multiple times with different inputs.
    pub fn prepare(&self, model: ModelProto) -> Result<OnnxGraph> {
        let proto_graph = model.graph.ok_or_else(|| EmptyModelSnafu.build())?;

        // Collect opsets
        let opsets: Vec<OpSetId> =
            model.opset_import.iter().map(|op| OpSetId { domain: op.domain.clone(), version: op.version }).collect();

        // Build initializer map (weights/constants)
        let mut initializers: HashMap<String, Tensor> = HashMap::new();
        let initializer_names: Vec<String> = proto_graph.initializer.iter().map(|i| i.name.clone()).collect();

        for init in &proto_graph.initializer {
            if !init.name.is_empty() {
                let tensor = tensor_from_proto_ext(init, self.model_dir.as_deref())?;
                initializers.insert(init.name.clone(), tensor);
            }
        }

        // Extract input specs (excluding initializers)
        let mut inputs: HashMap<String, InputSpec> = HashMap::new();
        for input in &proto_graph.input {
            if input.name.is_empty() {
                continue;
            }
            // Skip if this input is an initializer
            if initializer_names.contains(&input.name) {
                continue;
            }
            // Extract input spec from ValueInfoProto
            if let Some(spec) = self.extract_input_spec(input)? {
                inputs.insert(input.name.clone(), spec);
            }
        }

        // Collect output names
        let outputs: Vec<String> =
            proto_graph.output.iter().filter(|o| !o.name.is_empty()).map(|o| o.name.clone()).collect();

        // Collect nodes
        let nodes = proto_graph.node;

        Ok(OnnxGraph { inputs, outputs, initializers, nodes, opsets })
    }

    /// Extract InputSpec from ValueInfoProto.
    fn extract_input_spec(&self, input: &ValueInfoProto) -> Result<Option<InputSpec>> {
        // ValueInfoProto has a `type` field (TypeProto) - use r#type for keyword
        // TypeProto has a oneof `value` containing tensor_type, sequence_type, etc.
        use crate::parser::onnx::tensor_shape_proto::dimension::Value as DimValueProto;
        use crate::parser::onnx::type_proto::Value;

        let type_proto = match input.r#type.as_ref() {
            Some(t) => t,
            None => return Ok(None),
        };

        // Access the oneof value
        let tensor_type = match &type_proto.value {
            Some(Value::TensorType(tt)) => tt,
            _ => return Ok(None), // Not a tensor type
        };

        // Extract dtype
        let dtype = convert_onnx_dtype(tensor_type.elem_type)?;

        // Extract shape from tensor_type.shape
        let shape: Vec<DimValue> = tensor_type
            .shape
            .as_ref()
            .map(|s| {
                s.dim
                    .iter()
                    .map(|d| match &d.value {
                        Some(DimValueProto::DimValue(v)) => {
                            if *v > 0 {
                                DimValue::Static(*v as usize)
                            } else {
                                DimValue::Dynamic(String::new())
                            }
                        }
                        Some(DimValueProto::DimParam(name)) => DimValue::Dynamic(name.clone()),
                        None => DimValue::Dynamic(String::new()),
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(Some(InputSpec::new(shape, dtype, false)))
    }

    /// Phase 2: Execute the graph with provided inputs.
    ///
    /// Returns a HashMap mapping output names to their tensor values.
    pub fn execute(&self, graph: &OnnxGraph, inputs: HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
        self.execute_with_initializers(graph, inputs)
    }

    /// Execute graph with initializers merged into values.
    fn execute_with_initializers(
        &self,
        graph: &OnnxGraph,
        inputs: HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Merge initializers into values
        let mut values: HashMap<String, Tensor> = graph.initializers.clone();

        // Add user inputs
        for (name, tensor) in inputs {
            values.insert(name, tensor);
        }

        // Validate required inputs are present
        for (name, spec) in &graph.inputs {
            if !values.contains_key(name) && !spec.optional {
                return MissingInputSnafu { node: "graph_input".to_string(), input: name.clone() }.fail();
            }
        }

        // Resolve the default (ai.onnx) opset version
        let opset_version = onnx_opset_version(&graph.opsets, "");

        // Process nodes in order (ONNX guarantees topological order)
        for node in &graph.nodes {
            // Resolve opset for the node's domain (most nodes use default "")
            let node_opset = if node.domain.is_empty() || node.domain == "ai.onnx" {
                opset_version
            } else {
                onnx_opset_version(&graph.opsets, &node.domain)
            };
            self.process_node(node, &mut values, node_opset)?;
        }

        // Collect outputs by name
        let outputs: HashMap<String, Tensor> =
            graph.outputs.iter().filter_map(|name| values.get(name).cloned().map(|t| (name.clone(), t))).collect();

        Ok(outputs)
    }

    /// Process a single ONNX node.
    fn process_node(&self, node: &NodeProto, values: &mut HashMap<String, Tensor>, opset_version: i64) -> Result<()> {
        let op_type = &node.op_type;
        let domain = &node.domain;
        let node_name = if node.name.is_empty() { "unnamed" } else { &node.name };

        // Collect input Tensors, preserving positional indices for optional inputs.
        // Empty input names in ONNX mean "optional, not provided" — we represent
        // these as None to keep correct positional indexing for operators like Clip.
        let mut inputs: Vec<Option<Tensor>> = Vec::new();
        for input_name in &node.input {
            if input_name.is_empty() {
                inputs.push(None);
            } else {
                match values.get(input_name) {
                    Some(tensor) => inputs.push(Some(tensor.clone())),
                    None => {
                        return Err(crate::Error::MissingInput {
                            node: node_name.to_string(),
                            input: input_name.clone(),
                        });
                    }
                }
            }
        }

        // Dispatch to operator registry - may return multiple outputs
        let outputs = self.registry.dispatch_multi(op_type, domain, &inputs, node, opset_version)?;

        // Register outputs by name
        for (i, output_name) in node.output.iter().enumerate() {
            if let Some(output_tensor) = outputs.get(i) {
                values.insert(output_name.clone(), output_tensor.clone());
            }
        }

        Ok(())
    }
}

impl Default for OnnxImporter {
    fn default() -> Self {
        Self::new()
    }
}
