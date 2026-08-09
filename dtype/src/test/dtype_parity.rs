#![cfg(test)]

use crate::{DType, ScalarDType};

#[test]
fn tinygrad_weak_promotion_lattice() {
    assert_eq!(DType::least_upper_dtype(&[DType::Bool, DType::WeakInt]), Some(DType::WeakInt));
    assert_eq!(DType::least_upper_dtype(&[DType::WeakInt, DType::Int8]), Some(DType::Int8));
    assert_eq!(DType::least_upper_dtype(&[DType::Int64, DType::UInt64]), Some(DType::WeakFloat));
    assert_eq!(DType::least_upper_dtype(&[DType::WeakFloat, DType::Float16]), Some(DType::Float16));
    assert_eq!(DType::least_upper_float(DType::WeakInt), Some(DType::WeakFloat));
}

#[test]
fn tinygrad_strong_and_weak_dtype() {
    assert_eq!(DType::WeakInt.strong_dtype(), DType::Int32);
    assert_eq!(DType::WeakFloat.strong_dtype(), DType::Float32);
    assert_eq!(DType::Int16.weak_dtype(), DType::WeakInt);
    assert_eq!(DType::Float64.weak_dtype(), DType::WeakFloat);
    assert_eq!(ScalarDType::WeakInt.bytes(), 100);
    assert_eq!(ScalarDType::WeakFloat.bitsize(), 800);
}

#[test]
fn tinygrad_lossless_cast_table() {
    assert!(!ScalarDType::Int8.can_safe_cast(ScalarDType::UInt64));
    assert!(!ScalarDType::Int32.can_safe_cast(ScalarDType::UInt32));
    assert!(ScalarDType::UInt8.can_safe_cast(ScalarDType::Int16));
    assert!(ScalarDType::UInt32.can_safe_cast(ScalarDType::Int64));
    assert!(!ScalarDType::Int32.can_safe_cast(ScalarDType::Float32));
    assert!(!ScalarDType::Int64.can_safe_cast(ScalarDType::Float64));
    assert!(ScalarDType::Int8.can_safe_cast(ScalarDType::Float16));
    assert!(!ScalarDType::Int8.can_safe_cast(ScalarDType::BFloat16));
}
