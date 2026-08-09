//! Vector operation tests.
//!
//! Tests vector operations: Vectorize, Gep, VConst, Cat, PtrCat.

use smallvec::smallvec;

use svod_dtype::{AddrSpace, DType};

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

// =========================================================================
// Vectorize Tests
// =========================================================================

#[test]
fn test_vectorize_basic() {
    // Should be Float32 vector of size 4
    assert_eq!(
        UOp::vectorize(smallvec![
            UOp::native_const(1.0f32),
            UOp::native_const(2.0f32),
            UOp::native_const(3.0f32),
            UOp::native_const(4.0f32)
        ])
        .dtype(),
        DType::Float32.vec(4).unwrap()
    );
}

#[test]
fn test_vectorize_preserves_base_dtype() {
    let vec = UOp::vectorize(smallvec![UOp::native_const(1i32), UOp::native_const(2i32)]);
    assert_eq!(vec.dtype(), DType::Int32.vec(2).unwrap());
}

// =========================================================================
// Gep (Get Element Pointer) Tests
// =========================================================================

#[test]
fn test_gep_basic() {
    // Create a vector
    let vec = UOp::vectorize(smallvec![
        UOp::native_const(1.0f32),
        UOp::native_const(2.0f32),
        UOp::native_const(3.0f32),
        UOp::native_const(4.0f32)
    ]);

    // GEP operation exists (actual behavior may vary based on implementation)
    let _elem = vec.gep(vec![0]);
    // Just verify it compiles and creates a UOp
}

#[test]
fn test_gep_multiple_indices() {
    let vec = UOp::vectorize(smallvec![
        UOp::native_const(10i32),
        UOp::native_const(20i32),
        UOp::native_const(30i32),
        UOp::native_const(40i32),
    ]);

    // Extract multiple elements -> produces vector of extracted elements
    let result = vec.gep(vec![0, 2]);
    assert_eq!(result.dtype(), DType::Int32.vec(2).unwrap());
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

// =========================================================================
// Cat Tests
// =========================================================================

#[test]
fn test_cat_basic() {
    let a = UOp::vectorize(smallvec![UOp::native_const(1.0f32), UOp::native_const(2.0f32),]);
    let b = UOp::vectorize(smallvec![UOp::native_const(3.0f32), UOp::native_const(4.0f32),]);

    let result = UOp::cat().sources(vec![a, b]).call();
    // Cat concatenates vectors: <2 x f32> + <2 x f32> = <4 x f32>
    assert_eq!(result.dtype(), DType::Float32.vec(4).unwrap());
}

// =========================================================================
// PtrCat Tests
// =========================================================================

#[test]
fn test_ptrcat_basic() {
    let ptr_dtype = DType::Float32.ptr(None, AddrSpace::Global).unwrap();
    let a = UOp::const_(ptr_dtype.clone(), ConstValue::Int(0));
    let b = UOp::const_(ptr_dtype.clone(), ConstValue::Int(0));

    let result = UOp::ptrcat().sources(vec![a, b]).call();
    // PTRCAT of 2 scalar pointers → vcount=2
    assert_eq!(result.dtype(), ptr_dtype.vec(2).unwrap());
}
