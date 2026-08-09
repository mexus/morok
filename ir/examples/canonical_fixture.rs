use svod_dtype::DType;
use svod_ir::{BinaryOp, CanonicalGraph, ConstValue, Op, TernaryOp, UOp};

fn fixture(name: &str) -> std::sync::Arc<UOp> {
    match name {
        "weak_int_add" => {
            let lhs = UOp::const_(DType::WeakInt, ConstValue::Int(7));
            let rhs = UOp::const_(DType::WeakInt, ConstValue::Int(2));
            UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::WeakInt)
        }
        "weak_float_neg_zero" => UOp::const_(DType::WeakFloat, ConstValue::Float(-0.0)),
        "invalid_where" => {
            let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
            let value = UOp::const_(DType::Float16, ConstValue::Float(1.0));
            UOp::new(Op::Ternary(TernaryOp::Where, condition, value, UOp::invalid_marker()), DType::Float16)
        }
        _ => panic!("unknown fixture {name:?}"),
    }
}

fn main() {
    let name = std::env::args().nth(1).expect("usage: canonical_fixture <fixture>");
    let graph = CanonicalGraph::from_root("tensor", &fixture(&name)).expect("fixture must have a valid shape");
    println!("{}", graph.to_pretty_json().expect("canonical graph must serialize"));
}
