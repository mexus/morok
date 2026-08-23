use crate::{BinaryOp, ConstValue, ConstValueHash, DType, Op, TernaryOp, UOp, UnaryOp, dtype_from_op};
use svod_dtype::{AddrSpace, DeviceSpec};

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
fn control_dtypes_match_target_rules() {
    let end = UOp::index_const(8);
    let range = UOp::range_const(8, 0);
    let special = UOp::special(end, "gidx0".to_string());

    assert_eq!(dtype_from_op(range.op()), Some(DType::WeakInt));
    assert_eq!(dtype_from_op(special.op()), Some(DType::WeakInt));
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
fn alu_reconstruction_does_not_preserve_legacy_vector_result_dtype() {
    let old_float = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let old_bool = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let floats = UOp::stack(vec![old_float.clone(), old_float.clone()].into());
    let bools = UOp::stack(vec![old_bool.clone(), old_bool.clone()].into());
    let vector_float = DType::Float32.vec(2).unwrap();

    let unary =
        UOp::new(Op::Unary(UnaryOp::Sqrt, old_float.clone()), vector_float.clone()).with_sources(vec![floats.clone()]);
    let binary = UOp::new(Op::Binary(BinaryOp::Add, old_float.clone(), old_float.clone()), vector_float.clone())
        .with_sources(vec![floats.clone(), floats.clone()]);
    let comparison =
        UOp::new(Op::Binary(BinaryOp::Lt, old_float.clone(), old_float.clone()), DType::Bool.vec(2).unwrap())
            .with_sources(vec![floats.clone(), floats.clone()]);
    let where_op = UOp::new(Op::Ternary(TernaryOp::Where, old_bool, old_float.clone(), old_float), vector_float)
        .with_sources(vec![bools, floats.clone(), floats]);

    for result in [&unary, &binary, &where_op] {
        assert_eq!(result.dtype(), DType::Float32);
        assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
    }
    assert_eq!(comparison.dtype(), DType::Bool);
    assert_eq!(comparison.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
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
fn load_reconstruction_rederives_dtype_without_preserving_old_lanes() {
    let old_buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let old_index = UOp::index().buffer(old_buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let old_load = UOp::load().index(old_index).call();
    let new_buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float64);
    let new_index = UOp::index().buffer(new_buffer).indices(vec![UOp::index_const(0)]).call().unwrap();

    assert_eq!(old_load.with_sources(vec![new_index]).dtype(), DType::Float64);
}

#[test]
fn weak_equivalent_explicit_dtype_is_only_an_index_exception() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let offset = UOp::index_const(0);
    let weak_index =
        UOp::index().buffer(buffer.clone()).indices(vec![offset.clone()]).dtype(DType::WeakFloat).call().unwrap();
    assert_eq!(weak_index.dtype(), DType::WeakFloat);

    let strong_index = UOp::index().buffer(buffer).indices(vec![offset]).call().unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            UOp::load().index(strong_index).dtype(DType::WeakFloat).call()
        }))
        .is_err()
    );
}

#[test]
fn load_accepts_only_target_source_structures() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let alt = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));

    assert!(matches!(UOp::load().index(index.clone()).call().op(), Op::Load { alt: None, gate: None, .. }));
    assert!(matches!(
        UOp::load().index(index.clone()).alt(alt.clone()).gate(gate.clone()).call().op(),
        Op::Load { alt: Some(_), gate: Some(_), .. }
    ));
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

#[test]
fn index_dtype_matches_target_param_image_exception() {
    let image_shape = crate::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 3usize.into(), 4usize.into()]);
    let image_arg = crate::ParamArg::buffer(0, DType::Float16, AddrSpace::Global, None);
    let param = UOp::new(Op::Param { shape: image_shape.clone(), arg: image_arg.clone() }, DType::Float16);
    let buffer = UOp::new(Op::Buffer { shape: image_shape, arg: image_arg }, DType::Float16);
    let wrong_shape = crate::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 3usize.into(), 5usize.into()]);
    let wrong_param = UOp::new(
        Op::Param { shape: wrong_shape, arg: crate::ParamArg::buffer(1, DType::Float16, AddrSpace::Global, None) },
        DType::Float16,
    );
    let wrong_rank_shape = crate::shape::shape_to_uop(&smallvec::smallvec![3usize.into(), 4usize.into()]);
    let wrong_rank_param = UOp::new(
        Op::Param { shape: wrong_rank_shape, arg: crate::ParamArg::buffer(2, DType::Float16, AddrSpace::Global, None) },
        DType::Float16,
    );
    let offset = UOp::index_const(0);

    assert_eq!(UOp::index().buffer(param).indices(vec![offset.clone()]).call().unwrap().dtype(), DType::Float32);
    assert_eq!(UOp::index().buffer(buffer).indices(vec![offset.clone()]).call().unwrap().dtype(), DType::Float16);
    assert_eq!(UOp::index().buffer(wrong_param).indices(vec![offset.clone()]).call().unwrap().dtype(), DType::Float16);
    assert_eq!(UOp::index().buffer(wrong_rank_param).indices(vec![offset]).call().unwrap().dtype(), DType::Float16);
}
