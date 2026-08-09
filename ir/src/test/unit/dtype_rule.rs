use crate::{BinaryOp, ConstValue, ConstValueHash, DType, Op, UOp, dtype_from_op};

#[test]
fn constants_derive_weak_dtypes() {
    let integer = Op::Const(ConstValueHash(ConstValue::Int(1)));
    let float = Op::Const(ConstValueHash(ConstValue::Float(1.0)));
    let boolean = Op::Const(ConstValueHash(ConstValue::Bool(true)));

    assert_eq!(dtype_from_op(&integer), Some(DType::WeakInt));
    assert_eq!(dtype_from_op(&float), Some(DType::WeakFloat));
    assert_eq!(dtype_from_op(&boolean), Some(DType::Bool));
}

#[test]
fn alu_dtype_uses_current_promotion_rules() {
    let weak = UOp::new(Op::Const(ConstValueHash(ConstValue::Int(1))), DType::WeakInt);
    let strong = UOp::const_(DType::Int16, ConstValue::Int(2));
    let add = Op::Binary(BinaryOp::Add, weak.clone(), strong.clone());
    let comparison = Op::Binary(BinaryOp::Lt, weak, strong);

    assert_eq!(dtype_from_op(&add), Some(DType::Int16));
    assert_eq!(dtype_from_op(&comparison), Some(DType::Bool));
}

#[test]
fn invalid_has_one_produced_dtype() {
    assert_eq!(dtype_from_op(UOp::invalid_marker().op()), Some(DType::Bool));
}

#[test]
fn source_rewrite_rederives_parent_dtype() {
    let lhs = UOp::const_(DType::Int16, ConstValue::Int(1));
    let rhs = UOp::const_(DType::Int16, ConstValue::Int(2));
    let add = UOp::new(Op::Binary(BinaryOp::Add, lhs.clone(), rhs), DType::Int16);
    let float = UOp::const_(DType::Float32, ConstValue::Float(2.0));

    let rewritten = add.with_sources(vec![lhs, float]);
    assert_eq!(rewritten.dtype(), DType::Float32);
}

#[test]
fn invalid_source_does_not_retype_parent() {
    let lhs = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let rhs = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let add = UOp::new(Op::Binary(BinaryOp::Add, lhs.clone(), rhs), DType::Float32);
    let invalid = UOp::invalid_marker();

    let rewritten = add.with_sources(vec![lhs, invalid]);
    assert_eq!(rewritten.dtype(), DType::Float32);
}

#[test]
fn invalid_is_canonical_and_polymorphic() {
    let invalid = UOp::invalid_marker();
    assert_eq!(invalid.dtype(), DType::Bool);
    assert!(std::sync::Arc::ptr_eq(&invalid, &UOp::const_(DType::Float32, ConstValue::Invalid)));

    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let add = value.try_add(&invalid).unwrap();
    let Op::Binary(_, _, rhs) = add.op() else { panic!("expected binary operation") };
    assert!(std::sync::Arc::ptr_eq(rhs, &invalid));
    assert_eq!(add.dtype(), DType::Float32);
}
