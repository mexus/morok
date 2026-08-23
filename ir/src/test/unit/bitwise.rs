//! Bitwise operation tests.
//!
//! Tests all bitwise operations including and, or, xor, shifts, and dtype validation.

use std::f32::consts::PI;

use svod_dtype::DType;

use crate::{ConstValue, Op, UOp, error::Error}; // ConstValue kept for i8, i16, u8

// =========================================================================
// Basic Bitwise Operations with Int Types
// =========================================================================

#[test]
fn test_and_int32() {
    assert_eq!(UOp::native_const(0b1010i32).try_and_op(&UOp::native_const(0b1100i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_or_int32() {
    assert_eq!(UOp::native_const(0b1010i32).try_or_op(&UOp::native_const(0b1100i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_xor_int32() {
    assert_eq!(UOp::native_const(0b1010i32).try_xor_op(&UOp::native_const(0b1100i32)).unwrap().dtype(), DType::Int32);
}

#[test]
fn test_weak_int_bitwise_and_shift_graph_construction() {
    let value = UOp::index_const(0b1010);
    let mask = UOp::index_const(0b1100);
    let shift = UOp::index_const(2);

    for node in [value.try_and_op(&mask).unwrap(), value.try_xor_op(&mask).unwrap(), value.try_shr_op(&shift).unwrap()]
    {
        assert_eq!(node.dtype(), DType::WeakInt);
    }
}

// =========================================================================
// Shift Operations
// =========================================================================

#[test]
fn test_shl_int32() {
    let value = UOp::native_const(8i32);
    let shift = UOp::native_const(2i32);

    let result = value.try_shl_op(&shift).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_shr_int32() {
    let value = UOp::native_const(32i32);
    let shift = UOp::native_const(2i32);

    let result = value.try_shr_op(&shift).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_shift_preserves_lhs_dtype() {
    let value = UOp::native_const(100i64);
    let shift = UOp::native_const(3i32);

    // Shift should preserve LHS dtype (Int64), not promote
    let result = value.try_shl_op(&shift).unwrap();
    assert_eq!(result.dtype(), DType::Int64);
}

#[test]
fn test_shift_dtype_matrix_preserves_lhs_and_rhs_types() {
    let integer_dtypes = [
        DType::Int8,
        DType::UInt8,
        DType::Int16,
        DType::UInt16,
        DType::Int32,
        DType::UInt32,
        DType::Int64,
        DType::UInt64,
    ];

    for lhs_dtype in integer_dtypes {
        let lhs = UOp::const_(lhs_dtype.clone(), ConstValue::Int(1));
        for rhs in [UOp::index_const(1), UOp::native_const(1u32)] {
            for shifted in [lhs.try_shl_op(&rhs).unwrap(), lhs.try_shr_op(&rhs).unwrap()] {
                let Op::Binary(_, actual_lhs, actual_rhs) = shifted.op() else { panic!("expected shift") };
                assert_eq!(shifted.dtype(), lhs_dtype);
                assert_eq!(actual_lhs.dtype(), lhs_dtype);
                assert_eq!(actual_rhs.dtype(), rhs.dtype());
            }
        }
    }
}

#[test]
fn test_weak_shift_lhs_is_not_promoted_by_strong_count() {
    let lhs = UOp::index_const(8);
    for rhs_dtype in [DType::Int8, DType::UInt16, DType::Int64, DType::UInt32] {
        let rhs = UOp::const_(rhs_dtype.clone(), ConstValue::Int(1));
        for shifted in [lhs.try_shl_op(&rhs).unwrap(), lhs.try_shr_op(&rhs).unwrap()] {
            let Op::Binary(_, actual_lhs, actual_rhs) = shifted.op() else { panic!("expected shift") };
            assert_eq!(shifted.dtype(), DType::WeakInt);
            assert_eq!(actual_lhs.dtype(), DType::WeakInt);
            assert_eq!(actual_rhs.dtype(), rhs_dtype);
        }
    }
}

#[test]
fn test_shift_rejects_bool_sources_without_broadening_bitwise_bool() {
    let value = UOp::native_const(true);
    let shift = UOp::native_const(false);

    assert!(value.try_shl_op(&shift).is_err());
    assert!(value.try_shr_op(&shift).is_err());
    assert!(value.try_and_op(&shift).is_ok());
}

// =========================================================================
// Boolean Type Operations
// =========================================================================

#[test]
fn test_and_bool() {
    let a = UOp::native_const(true);
    let b = UOp::native_const(false);

    let result = a.try_and_op(&b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_or_bool() {
    let a = UOp::native_const(true);
    let b = UOp::native_const(false);

    let result = a.try_or_op(&b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

#[test]
fn test_xor_bool() {
    let a = UOp::native_const(true);
    let b = UOp::native_const(false);

    let result = a.try_xor_op(&b).unwrap();
    assert_eq!(result.dtype(), DType::Bool);
}

// =========================================================================
// Unsigned Integer Operations
// =========================================================================

#[test]
fn test_and_uint32() {
    let a = UOp::native_const(15u32);
    let b = UOp::native_const(7u32);

    let result = a.try_and_op(&b).unwrap();
    // Same-type operands: result is same type
    assert_eq!(result.dtype(), DType::UInt32);
}

#[test]
fn test_shl_uint64() {
    let value = UOp::native_const(1u64);
    let shift = UOp::native_const(10u32);

    let result = value.try_shl_op(&shift).unwrap();
    assert_eq!(result.dtype(), DType::UInt64);
}

// =========================================================================
// Type Promotion in Bitwise Ops
// =========================================================================

#[test]
fn test_and_type_promotion() {
    let small = UOp::const_(DType::Int8, ConstValue::Int(15));
    let large = UOp::native_const(255i32);

    let result = small.try_and_op(&large).unwrap();
    // Int8 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_mixed_int_types() {
    let i16 = UOp::const_(DType::Int16, ConstValue::Int(100));
    let i32 = UOp::native_const(200i32);

    let result = i16.try_or_op(&i32).unwrap();
    // Int16 should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

// =========================================================================
// Invalid DType Error Tests
// =========================================================================

#[test]
fn test_and_float_error() {
    let float_val = UOp::native_const(PI);
    let int_val = UOp::native_const(5i32);

    let result = float_val.try_and_op(&int_val);
    assert!(matches!(result, Err(Error::InvalidDTypeForBinaryOp { .. })));
}

#[test]
fn test_or_float_error() {
    let float_val = UOp::native_const(2.5f32);
    let int_val = UOp::native_const(10i32);

    let result = float_val.try_or_op(&int_val);
    assert!(matches!(result, Err(Error::InvalidDTypeForBinaryOp { .. })));
}

#[test]
fn test_xor_float_error() {
    let float_val = UOp::native_const(1.5f64);
    let int_val = UOp::native_const(7i64);

    let result = float_val.try_xor_op(&int_val);
    assert!(matches!(result, Err(Error::InvalidDTypeForBinaryOp { .. })));
}

#[test]
fn test_shl_float_error() {
    let float_val = UOp::native_const(8.0f32);
    let shift = UOp::native_const(2i32);

    let result = float_val.try_shl_op(&shift);
    assert!(matches!(result, Err(Error::InvalidDTypeForBinaryOp { .. })));
}

#[test]
fn test_shr_float_error() {
    let float_val = UOp::native_const(16.0f32);
    let shift = UOp::native_const(1i32);

    let result = float_val.try_shr_op(&shift);
    assert!(matches!(result, Err(Error::InvalidDTypeForBinaryOp { .. })));
}

#[test]
fn test_shift_float_rhs_error() {
    let value = UOp::native_const(16i32);
    let shift = UOp::native_const(1.0f32);

    assert!(matches!(value.try_shr_op(&shift), Err(Error::InvalidDTypeForBinaryOp { .. })));
}

#[test]
fn test_and_both_floats_error() {
    let f1 = UOp::native_const(1.0f32);
    let f2 = UOp::native_const(2.0f32);

    let result = f1.try_and_op(&f2);
    assert!(matches!(result, Err(Error::InvalidDTypeForBinaryOp { .. })));
}

// =========================================================================
// Mixed Bool and Int Operations
// =========================================================================

#[test]
fn test_and_bool_and_int() {
    let bool_val = UOp::native_const(true);
    let int_val = UOp::native_const(5i32);

    let result = bool_val.try_and_op(&int_val).unwrap();
    // Bool should promote to Int32
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_int_and_bool() {
    let int_val = UOp::const_(DType::Int8, ConstValue::Int(7));
    let bool_val = UOp::native_const(false);

    let result = int_val.try_or_op(&bool_val).unwrap();
    // Bool should promote to Int8
    assert_eq!(result.dtype(), DType::Int8);
}

// =========================================================================
// Edge Cases
// =========================================================================

#[test]
fn test_and_with_zero() {
    let val = UOp::native_const(0xFFi32);
    let zero = UOp::native_const(0i32);

    let result = val.try_and_op(&zero).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_or_with_max() {
    let val = UOp::const_(DType::UInt8, ConstValue::UInt(0x55));
    let max = UOp::const_(DType::UInt8, ConstValue::UInt(0xFF));

    let result = val.try_or_op(&max).unwrap();
    // Same-type operands: result is same type
    assert_eq!(result.dtype(), DType::UInt8);
}

#[test]
fn test_xor_with_self() {
    let val = UOp::native_const(42i32);

    let result = val.try_xor_op(&val).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}

#[test]
fn test_shl_by_zero() {
    let val = UOp::native_const(100i32);
    let zero = UOp::native_const(0i32);

    let result = val.try_shl_op(&zero).unwrap();
    assert_eq!(result.dtype(), DType::Int32);
}
