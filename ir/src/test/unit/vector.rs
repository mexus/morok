//! Vector operation tests.
//!
//! Tests shaped STACK and late vector operations.

use smallvec::smallvec;

use svod_dtype::DType;

use crate::{ConstValue, Op, UOp};

// =========================================================================
// Stack Tests
// =========================================================================

#[test]
fn test_stack_has_scalar_dtype_and_shaped_lanes() {
    let stack = UOp::stack(smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    assert_eq!(stack.dtype(), DType::Int32);
    assert_eq!(stack.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
}

#[test]
fn test_stack_casts_weak_lane_to_promoted_strong_dtype() {
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let strong = UOp::const_(DType::Int16, ConstValue::Int(2));
    let stack = UOp::stack(smallvec![weak.clone(), strong.clone()]);

    assert_eq!(stack.dtype(), DType::Int16);
    let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
    assert_eq!(sources.iter().map(|source| source.dtype()).collect::<Vec<_>>(), vec![DType::Int16; 2]);
    assert!(
        matches!(sources[0].op(), Op::Cast { src, dtype } if std::sync::Arc::ptr_eq(src, &weak) && *dtype == DType::Int16)
    );
    assert!(std::sync::Arc::ptr_eq(&sources[1], &strong));
}

#[test]
fn test_stack_casts_mixed_integer_widths() {
    let narrow = UOp::const_(DType::Int8, ConstValue::Int(1));
    let wide = UOp::const_(DType::UInt16, ConstValue::UInt(2));
    let stack = UOp::stack(smallvec![narrow, wide]);

    assert_eq!(stack.dtype(), DType::Int32);
    let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
    assert!(sources.iter().all(|source| source.dtype() == DType::Int32));
}

#[test]
fn test_stack_casts_integer_lane_to_promoted_float_dtype() {
    let integer = UOp::const_(DType::Int16, ConstValue::Int(1));
    let float = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let stack = UOp::stack(smallvec![integer, float]);

    assert_eq!(stack.dtype(), DType::Float32);
    let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
    assert!(sources.iter().all(|source| source.dtype() == DType::Float32));
}

#[test]
fn test_stack_keeps_invalid_polymorphic_and_uncast() {
    let invalid = UOp::invalid_marker();
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let stack = UOp::stack(smallvec![invalid.clone(), value.clone()]);

    assert_eq!(stack.dtype(), DType::Float32);
    let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
    assert!(std::sync::Arc::ptr_eq(&sources[0], &invalid));
    assert!(std::sync::Arc::ptr_eq(&sources[1], &value));
}

#[test]
fn test_stack_keeps_shaped_invalid_lane_uncast() {
    let invalid = UOp::invalid_marker();
    let shaped_invalid = invalid.try_reshape(&smallvec![1usize.into()]).unwrap();
    let value = UOp::stack(smallvec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let stack = UOp::stack(smallvec![shaped_invalid.clone(), value]);

    assert_eq!(stack.dtype(), DType::Float32);
    assert_eq!(stack.shape().unwrap().unwrap().as_slice(), &[2usize.into(), 1usize.into()]);
    let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
    assert!(std::sync::Arc::ptr_eq(&sources[0], &shaped_invalid));
}

#[test]
fn test_stack_casts_shaped_lanes_without_losing_leading_axis() {
    let weak_row = UOp::stack(smallvec![
        UOp::const_(DType::WeakInt, ConstValue::Int(1)),
        UOp::const_(DType::WeakInt, ConstValue::Int(2)),
    ]);
    let strong_row = UOp::stack(smallvec![
        UOp::const_(DType::Int16, ConstValue::Int(3)),
        UOp::const_(DType::Int16, ConstValue::Int(4)),
    ]);
    let matrix = UOp::stack(smallvec![weak_row, strong_row]);

    assert_eq!(matrix.dtype(), DType::Int16);
    assert_eq!(matrix.shape().unwrap().unwrap().as_slice(), &[2usize.into(), 2usize.into()]);
    let Op::Stack { sources } = matrix.op() else { panic!("expected STACK") };
    assert!(sources.iter().all(|source| source.dtype() == DType::Int16));
}

#[test]
fn test_stack_does_not_hide_mismatched_source_shapes_with_casts() {
    let short = UOp::stack(smallvec![UOp::const_(DType::WeakInt, ConstValue::Int(1))]);
    let long = UOp::stack(smallvec![
        UOp::const_(DType::Int16, ConstValue::Int(2)),
        UOp::const_(DType::Int16, ConstValue::Int(3)),
    ]);
    let stack = UOp::stack(smallvec![short, long]);

    assert_eq!(stack.dtype(), DType::Int16);
    assert_eq!(stack.shape().unwrap(), None);
    let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
    assert!(sources.iter().all(|source| source.dtype() == DType::Int16));
}

#[test]
fn test_stack_reconstruction_recasts_rewritten_lanes() {
    let original = UOp::stack(smallvec![
        UOp::const_(DType::Int16, ConstValue::Int(1)),
        UOp::const_(DType::Int16, ConstValue::Int(2)),
    ]);
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(3));
    let float = UOp::const_(DType::Float32, ConstValue::Float(4.0));
    let rebuilt = original.with_sources(vec![weak, float]);

    assert_eq!(rebuilt.dtype(), DType::Float32);
    let Op::Stack { sources } = rebuilt.op() else { panic!("expected STACK") };
    assert!(sources.iter().all(|source| source.dtype() == DType::Float32));
}

#[test]
fn test_stack_constant_index_returns_lane() {
    let first = UOp::native_const(11i32);
    let second = UOp::native_const(22i32);
    let stack = UOp::stack(smallvec![first, second.clone()]);
    let selected = UOp::index().buffer(stack).indices(vec![UOp::index_const(1)]).call().unwrap();
    assert!(std::sync::Arc::ptr_eq(&selected, &second));
}

#[test]
fn test_stack_reconstruction_preserves_hash_cons_identity() {
    let stack = UOp::stack(smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    let rebuilt = stack.with_sources(stack.op().sources().into_vec());
    assert!(std::sync::Arc::ptr_eq(&stack, &rebuilt));
    assert!(matches!(rebuilt.op(), Op::Stack { .. }));
}

#[test]
fn test_stack_all_invalid_is_invalid_marker() {
    let stack = UOp::stack(smallvec![UOp::invalid_marker(), UOp::invalid_marker()]);
    assert!(UOp::is_invalid_marker(&stack));
}

#[test]
fn test_stack_adds_leading_axis_to_shaped_sources() {
    let row = UOp::stack(smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    let matrix = UOp::stack(smallvec![row.clone(), row]);
    assert_eq!(matrix.dtype(), DType::Int32);
    assert_eq!(matrix.shape().unwrap().unwrap().as_slice(), &[2usize.into(), 2usize.into()]);
}

#[test]
fn test_shaped_index_selects_multiple_positions() {
    let vec = UOp::stack(smallvec![
        UOp::native_const(10i32),
        UOp::native_const(20i32),
        UOp::native_const(30i32),
        UOp::native_const(40i32),
    ]);

    // A shaped index adds its target axis before the source's trailing axes.
    let result = vec.index_axes(vec![0, 2]);
    assert_eq!(result.dtype(), DType::Int32);
    assert_eq!(result.shape().unwrap().unwrap().as_slice(), &[2usize.into()]);
}

// =========================================================================
// VConst Tests
// =========================================================================

#[test]
fn test_vconst_basic() {
    let values = vec![ConstValue::Float(1.0), ConstValue::Float(2.0), ConstValue::Float(3.0), ConstValue::Float(4.0)];

    let vec = UOp::vconst(values, DType::Float64);
    assert_eq!(vec.dtype(), DType::Float64.vec(4).unwrap());
}
