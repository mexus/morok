use svod_dtype::{AddrSpace, DType};

use crate::uop::gc_dead_refs;
use crate::{
    BinaryOp, CanonicalArg, CanonicalConst, CanonicalDType, CanonicalGraph, CanonicalShapeDim, ConstValue, Op, UOp,
};

fn graph_json() -> String {
    let lhs = UOp::const_(DType::Int32, ConstValue::Int(7));
    let rhs = UOp::const_(DType::Int32, ConstValue::Int(2));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, lhs, rhs), DType::Int32);
    CanonicalGraph::from_root("tensor", &sub).unwrap().to_pretty_json().unwrap()
}

#[test]
fn canonical_graph_is_independent_of_runtime_ids() {
    let first = graph_json();
    gc_dead_refs();
    let second = graph_json();
    assert_eq!(first, second);
    assert!(!first.contains("runtime_id"));
}

#[test]
fn canonical_graph_preserves_dag_sharing_and_source_order() {
    let shared = UOp::const_(DType::Int32, ConstValue::Int(3));
    let rhs = UOp::const_(DType::Int32, ConstValue::Int(1));
    let sub = UOp::new(Op::Binary(BinaryOp::Sub, shared.clone(), rhs), DType::Int32);
    let root = UOp::new(Op::Binary(BinaryOp::Add, sub, shared), DType::Int32);
    let graph = CanonicalGraph::from_root("tensor", &root).unwrap();

    assert_eq!(graph.nodes.len(), 4);
    assert_eq!(graph.roots, vec![3]);
    assert_eq!(graph.nodes[2].op, "SUB");
    assert_eq!(graph.nodes[2].src, vec![0, 1]);
    assert_eq!(graph.nodes[3].src, vec![2, 0]);
}

#[test]
fn canonical_graph_records_structured_param_and_shape_source() {
    let ptr = DType::Float16.ptr(Some(32), AddrSpace::Global).unwrap();
    let param = UOp::param(4, 32, ptr, None);
    let graph = CanonicalGraph::from_root("kernel_ast", &param).unwrap();

    assert_eq!(graph.nodes[1].src, vec![0]);
    assert_eq!(graph.nodes[1].shape, Some(vec![CanonicalShapeDim::Const { value: 32 }]));
    assert_eq!(graph.nodes[1].dtype, CanonicalDType::Scalar { name: "float16".to_string() });
    assert_eq!(
        graph.nodes[1].arg,
        CanonicalArg::Param {
            slot: 4,
            dtype: CanonicalDType::Scalar { name: "float16".to_string() },
            vmin_vmax: None,
            multiple_of: None,
            name: None,
            address_space: Some("global".to_string()),
            axis: None,
            device: None,
            volatile: false,
        }
    );
}

#[test]
fn canonical_float_uses_exact_bits() {
    let constant = UOp::const_(DType::Float64, ConstValue::Float(-0.0));
    let graph = CanonicalGraph::from_root("tensor", &constant).unwrap();
    assert_eq!(
        graph.nodes[0].arg,
        CanonicalArg::Const { value: CanonicalConst::Float { bits: "0x8000000000000000".to_string() } }
    );
}

#[test]
fn canonical_stack_has_scalar_dtype_shape_and_ordered_sources() {
    let stack = UOp::stack(smallvec::smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    let graph = CanonicalGraph::from_root("tensor", &stack).unwrap();

    assert_eq!(graph.nodes[2].op, "STACK");
    assert_eq!(graph.nodes[2].dtype, CanonicalDType::Scalar { name: "int32".to_string() });
    assert_eq!(graph.nodes[2].shape, Some(vec![CanonicalShapeDim::Const { value: 2 }]));
    assert_eq!(graph.nodes[2].src, vec![0, 1]);
}

#[test]
fn canonical_multiple_roots_are_ordered_and_deduplicated() {
    let shared = UOp::const_(DType::Int32, ConstValue::Int(1));
    let lhs = UOp::new(Op::Binary(BinaryOp::Add, shared.clone(), shared.clone()), DType::Int32);
    let rhs = UOp::new(Op::Binary(BinaryOp::Mul, shared.clone(), shared), DType::Int32);
    let graph = CanonicalGraph::from_roots("scheduled", &[lhs, rhs]).unwrap();

    assert_eq!(graph.roots, vec![1, 2]);
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(graph.nodes[1].src, vec![0, 0]);
    assert_eq!(graph.nodes[2].src, vec![0, 0]);
}

#[test]
fn canonical_json_round_trips_as_generic_json() {
    let serialized = graph_json();
    let parsed: serde_json::Value = serde_json::from_str(&serialized).unwrap();
    assert_eq!(parsed["schema_version"], 1);
    assert_eq!(parsed["stage"], "tensor");
    assert_eq!(parsed["nodes"][2]["op"], "SUB");
}
