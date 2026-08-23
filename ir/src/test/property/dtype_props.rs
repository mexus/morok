//! Property tests for DType operations and casting.

use proptest::prelude::*;

use svod_dtype::DType;

use crate::UOp;
use crate::types::{BinaryOp, ConstValue};
use crate::uop::eval::eval_binary_op_typed;
use crate::uop::range_eval::compute_sound_vmin_vmax;

use super::generators::*;

// ============================================================================
// Constant Casting Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Casting to the same dtype should be identity.
    #[test]
    fn const_cast_identity((dtype, cv) in const_pair()) {
        let casted = cv.cast(&dtype).expect("Cast to same dtype should succeed");
        prop_assert_eq!(cv, casted, "Cast to same dtype should be identity");
    }

    /// Casting bool to int should give 0 or 1.
    #[test]
    fn const_cast_bool_to_int(b: bool) {
        let cv = ConstValue::Bool(b);
        let expected = if b { 1i64 } else { 0i64 };

        let casted = cv.cast(&DType::Int32).expect("Bool to int32 should succeed");
        prop_assert_eq!(casted, ConstValue::Int(expected));

        let casted64 = cv.cast(&DType::Int64).expect("Bool to int64 should succeed");
        prop_assert_eq!(casted64, ConstValue::Int(expected));
    }

    /// Casting int 0 to bool should give false, non-zero should give true.
    #[test]
    fn const_cast_int_to_bool(i in -100i64..=100) {
        let cv = ConstValue::Int(i);
        let expected = i != 0;

        let casted = cv.cast(&DType::Bool).expect("Int to bool should succeed");
        prop_assert_eq!(casted, ConstValue::Bool(expected));
    }

    /// Widening integer casts preserve value.
    #[test]
    fn const_cast_int_widening_preserves_value(i in -100i64..=100) {
        let cv8 = ConstValue::Int(i).cast(&DType::Int8).unwrap();

        // Cast to wider types
        let cv16 = cv8.cast(&DType::Int16).unwrap();
        let cv32 = cv16.cast(&DType::Int32).unwrap();
        let cv64 = cv32.cast(&DType::Int64).unwrap();

        // Extract final value
        if let ConstValue::Int(final_val) = cv64 {
            // Should match the truncated i8 value
            let expected = i as i8 as i64;
            prop_assert_eq!(final_val, expected,
                "Widening should preserve truncated value: {} -> i8 -> i16 -> i32 -> i64",
                i);
        } else {
            panic!("Expected Int after widening chain");
        }
    }

    /// Direct wide cast equals chained narrow casts (for in-range values).
    #[test]
    fn const_cast_widening_chain_equals_direct(i in -100i64..=100) {
        // i is small enough to fit in i8
        let cv_narrow = ConstValue::Int(i).cast(&DType::Int8).unwrap();

        // Chain: i8 -> i16 -> i32
        let via_chain = cv_narrow
            .cast(&DType::Int16).unwrap()
            .cast(&DType::Int32).unwrap();

        // Direct: i8 -> i32
        let direct = cv_narrow.cast(&DType::Int32).unwrap();

        prop_assert_eq!(via_chain, direct,
            "Chained widening should equal direct widening");
    }

    /// Float to int to float preserves integer part.
    #[test]
    fn const_cast_float_to_int_to_float(f in -100.0..=100.0) {
        let cv = ConstValue::Float(f);

        // Float -> Int32 -> Float
        let via_int = cv.cast(&DType::Int32).unwrap().cast(&DType::Float32).unwrap();

        if let ConstValue::Float(result) = via_int {
            // Should match truncated integer value
            let expected = (f as i32) as f64;
            prop_assert!((result - expected).abs() < 0.1,
                "Float->Int->Float should preserve integer part: {} -> {} (expected {})",
                f, result, expected);
        } else {
            panic!("Expected Float after cast chain");
        }
    }

    /// Zero casts to zero in any dtype.
    #[test]
    fn const_cast_zero_to_any_dtype(sdtype in arithmetic_sdtype()) {
        let dtype = DType::from(sdtype);
        let zero_int = ConstValue::Int(0);
        let zero_float = ConstValue::Float(0.0);
        let zero_bool = ConstValue::Bool(false);

        if let Some(casted_int) = zero_int.cast(&dtype) {
            match casted_int {
                ConstValue::Invalid => unreachable!(),
                ConstValue::Int(v) => prop_assert_eq!(v, 0),
                ConstValue::UInt(v) => prop_assert_eq!(v, 0),
                ConstValue::Float(v) => prop_assert_eq!(v, 0.0),
                ConstValue::Bool(v) => prop_assert!(!v),
            }
        }

        if let Some(casted_float) = zero_float.cast(&dtype) {
            match casted_float {
                ConstValue::Invalid => unreachable!(),
                ConstValue::Int(v) => prop_assert_eq!(v, 0),
                ConstValue::UInt(v) => prop_assert_eq!(v, 0),
                ConstValue::Float(v) => prop_assert_eq!(v, 0.0),
                ConstValue::Bool(v) => prop_assert!(!v),
            }
        }

        if let Some(casted_bool) = zero_bool.cast(&dtype) {
            match casted_bool {
                ConstValue::Invalid => unreachable!(),
                ConstValue::Int(v) => prop_assert_eq!(v, 0),
                ConstValue::UInt(v) => prop_assert_eq!(v, 0),
                ConstValue::Float(v) => prop_assert_eq!(v, 0.0),
                ConstValue::Bool(v) => prop_assert!(!v),
            }
        }
    }

    /// One casts to one in numeric dtypes.
    #[test]
    fn const_cast_one_to_numeric_dtype(dtype in arithmetic_sdtype()) {
        let dtype = DType::from(dtype);
        let one = ConstValue::Int(1);

        if let Some(casted) = one.cast(&dtype) {
            match casted {
                ConstValue::Invalid => unreachable!(),
                ConstValue::Int(v) => prop_assert_eq!(v, 1),
                ConstValue::UInt(v) => prop_assert_eq!(v, 1),
                ConstValue::Float(v) => prop_assert_eq!(v, 1.0),
                ConstValue::Bool(v) => prop_assert!(v), // 1 -> true
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn exact_float_operation_sound_bounds_enclose_special_results(
        a_bits in any::<u32>(),
        b_bits in any::<u32>(),
        op in prop_oneof![Just(BinaryOp::Add), Just(BinaryOp::Mul), Just(BinaryOp::Fdiv)],
    ) {
        let a = f32::from_bits(a_bits);
        let b = f32::from_bits(b_bits);
        let lhs = UOp::const_(DType::Float32, ConstValue::Float(a as f64));
        let rhs = UOp::const_(DType::Float32, ConstValue::Float(b as f64));
        let expr = UOp::new(crate::Op::Binary(op, lhs, rhs), DType::Float32);
        let expected = eval_binary_op_typed(op, ConstValue::Float(a as f64), ConstValue::Float(b as f64), svod_dtype::ScalarDType::Float32);

        match expected {
            Some(ConstValue::Float(value)) if value.is_nan() => prop_assert!(compute_sound_vmin_vmax(&expr).is_none()),
            Some(value) => prop_assert_eq!(compute_sound_vmin_vmax(&expr), Some((value, value))),
            None => prop_assert!(compute_sound_vmin_vmax(&expr).is_none()),
        }
    }
}

fn value_is_enclosed(value: ConstValue, min: ConstValue, max: ConstValue) -> bool {
    match (value, min, max) {
        (ConstValue::Int(value), ConstValue::Int(min), ConstValue::Int(max)) => min <= value && value <= max,
        (ConstValue::UInt(value), ConstValue::UInt(min), ConstValue::UInt(max)) => min <= value && value <= max,
        (ConstValue::Bool(value), ConstValue::Bool(min), ConstValue::Bool(max)) => min <= value && value <= max,
        _ => false,
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    #[test]
    fn sampled_int8_binary_values_are_enclosed_by_sound_ranges(
        a in any::<(i8, i8)>(),
        b in any::<(i8, i8)>(),
        shift in 0i8..8,
    ) {
        let (amin, amax) = if a.0 <= a.1 { (a.0, a.1) } else { (a.1, a.0) };
        let (bmin, bmax) = if b.0 <= b.1 { (b.0, b.1) } else { (b.1, b.0) };
        let lhs = UOp::var("prop_i8_lhs", DType::Int8, i64::from(amin), i64::from(amax));
        let rhs = UOp::var("prop_i8_rhs", DType::Int8, i64::from(bmin), i64::from(bmax));
        let samples_a = [amin, amax, ((i16::from(amin) + i16::from(amax)) / 2) as i8];
        let samples_b = [bmin, bmax, ((i16::from(bmin) + i16::from(bmax)) / 2) as i8];

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Max,
            BinaryOp::FloorDiv,
            BinaryOp::CDiv,
            BinaryOp::FloorMod,
            BinaryOp::CMod,
            BinaryOp::And,
            BinaryOp::Or,
            BinaryOp::Xor,
        ] {
            let expr = UOp::new(crate::Op::Binary(op, lhs.clone(), rhs.clone()), DType::Int8);
            if let Some((min, max)) = compute_sound_vmin_vmax(&expr) {
                for a in samples_a {
                    for b in samples_b {
                        if let Some(value) = eval_binary_op_typed(
                            op,
                            ConstValue::Int(i64::from(a)),
                            ConstValue::Int(i64::from(b)),
                            svod_dtype::ScalarDType::Int8,
                        ) {
                            prop_assert!(value_is_enclosed(value, min, max), "{op:?}: {a}, {b} -> {value:?} not in [{min:?}, {max:?}]");
                        }
                    }
                }
            }
        }

        let shift_rhs = UOp::const_(DType::Int8, ConstValue::Int(i64::from(shift)));
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            let expr = UOp::new(crate::Op::Binary(op, lhs.clone(), shift_rhs.clone()), DType::Int8);
            if let Some((min, max)) = compute_sound_vmin_vmax(&expr) {
                for a in samples_a {
                    let value = eval_binary_op_typed(
                        op,
                        ConstValue::Int(i64::from(a)),
                        ConstValue::Int(i64::from(shift)),
                        svod_dtype::ScalarDType::Int8,
                    ).unwrap();
                    prop_assert!(value_is_enclosed(value, min, max), "{op:?}: {a}, {shift} -> {value:?} not in [{min:?}, {max:?}]");
                }
            }
        }
    }

    #[test]
    fn sampled_uint8_binary_values_are_enclosed_by_sound_ranges(
        a in any::<(u8, u8)>(),
        b in any::<(u8, u8)>(),
        shift in 0u8..8,
    ) {
        let (amin, amax) = if a.0 <= a.1 { (a.0, a.1) } else { (a.1, a.0) };
        let (bmin, bmax) = if b.0 <= b.1 { (b.0, b.1) } else { (b.1, b.0) };
        let lhs = UOp::var("prop_u8_lhs", DType::UInt8, i64::from(amin), i64::from(amax));
        let rhs = UOp::var("prop_u8_rhs", DType::UInt8, i64::from(bmin), i64::from(bmax));
        let samples_a = [amin, amax, amin + (amax - amin) / 2];
        let samples_b = [bmin, bmax, bmin + (bmax - bmin) / 2];

        for op in [
            BinaryOp::Add,
            BinaryOp::Sub,
            BinaryOp::Mul,
            BinaryOp::Max,
            BinaryOp::FloorDiv,
            BinaryOp::CDiv,
            BinaryOp::FloorMod,
            BinaryOp::CMod,
            BinaryOp::And,
            BinaryOp::Or,
            BinaryOp::Xor,
        ] {
            let expr = UOp::new(crate::Op::Binary(op, lhs.clone(), rhs.clone()), DType::UInt8);
            if let Some((min, max)) = compute_sound_vmin_vmax(&expr) {
                for a in samples_a {
                    for b in samples_b {
                        if let Some(value) = eval_binary_op_typed(
                            op,
                            ConstValue::UInt(u64::from(a)),
                            ConstValue::UInt(u64::from(b)),
                            svod_dtype::ScalarDType::UInt8,
                        ) {
                            prop_assert!(value_is_enclosed(value, min, max), "{op:?}: {a}, {b} -> {value:?} not in [{min:?}, {max:?}]");
                        }
                    }
                }
            }
        }

        let shift_rhs = UOp::const_(DType::UInt8, ConstValue::UInt(u64::from(shift)));
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            let expr = UOp::new(crate::Op::Binary(op, lhs.clone(), shift_rhs.clone()), DType::UInt8);
            if let Some((min, max)) = compute_sound_vmin_vmax(&expr) {
                for a in samples_a {
                    let value = eval_binary_op_typed(
                        op,
                        ConstValue::UInt(u64::from(a)),
                        ConstValue::UInt(u64::from(shift)),
                        svod_dtype::ScalarDType::UInt8,
                    ).unwrap();
                    prop_assert!(value_is_enclosed(value, min, max), "{op:?}: {a}, {shift} -> {value:?} not in [{min:?}, {max:?}]");
                }
            }
        }
    }

    #[test]
    fn sampled_narrow_integer_cast_values_are_enclosed(
        pair in any::<(i16, i16)>(),
        upair in any::<(u16, u16)>(),
    ) {
        let (min, max) = if pair.0 <= pair.1 { pair } else { (pair.1, pair.0) };
        let src = UOp::var("prop_cast_i16", DType::Int16, i64::from(min), i64::from(max));
        let samples = [min, max, ((i32::from(min) + i32::from(max)) / 2) as i16];
        for target in [DType::Int8, DType::UInt8] {
            let cast = src.cast(target.clone());
            if let Some((range_min, range_max)) = compute_sound_vmin_vmax(&cast) {
                for value in samples {
                    let value = ConstValue::Int(i64::from(value)).cast(&target).unwrap();
                    prop_assert!(value_is_enclosed(value, range_min, range_max));
                }
            }
        }

        let (min, max) = if upair.0 <= upair.1 { upair } else { (upair.1, upair.0) };
        let src = UOp::var("prop_cast_u16", DType::UInt16, i64::from(min), i64::from(max));
        let samples = [min, max, min + (max - min) / 2];
        for target in [DType::Int8, DType::UInt8] {
            let cast = src.cast(target.clone());
            if let Some((range_min, range_max)) = compute_sound_vmin_vmax(&cast) {
                for value in samples {
                    let value = ConstValue::UInt(u64::from(value)).cast(&target).unwrap();
                    prop_assert!(value_is_enclosed(value, range_min, range_max));
                }
            }
        }
    }
}

// ============================================================================
// DType Family Properties
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Within a dtype family, widening should preserve values for in-range constants.
    #[test]
    fn dtype_family_widening_preserves_small_values(
        family in arb_dtype_family(),
        val in -10i64..=10,
    ) {
        let dtypes = family.widening_sequence();
        let narrowest = &dtypes[0];

        // Cast to narrowest type
        let cv = ConstValue::Int(val).cast(narrowest);
        if cv.is_none() {
            // Skip if narrowest dtype can't represent this value
            return Ok(());
        }
        let cv = cv.unwrap();

        // Widen through the family
        let mut current = cv;
        for dtype in &dtypes[1..] {
            let widened = current.cast(dtype).expect("Widening should succeed");

            // Numeric value should be preserved
            let original_numeric = const_value_to_f64(&cv);
            let widened_numeric = const_value_to_f64(&widened);

            prop_assert!((original_numeric - widened_numeric).abs() < 0.1,
                "Widening from {:?} to {:?} should preserve value: {} -> {}",
                narrowest, dtype, original_numeric, widened_numeric);

            current = widened;
        }
    }

    /// Widening then narrowing back may lose precision but shouldn't change sign for small values.
    #[test]
    fn dtype_roundtrip_preserves_sign(
        family in arb_dtype_family(),
        val in -10i64..=10,
    ) {
        let dtypes = family.widening_sequence();
        let narrowest = &dtypes[0];
        let widest = &dtypes[dtypes.len() - 1];

        // Start with value in narrowest type
        let cv = ConstValue::Int(val).cast(narrowest);
        if cv.is_none() {
            return Ok(());
        }
        let cv = cv.unwrap();
        let original_sign = const_value_sign(&cv);

        // Widen to widest, then narrow back
        let widened = cv.cast(widest).expect("Widening should succeed");
        let narrowed = widened.cast(narrowest).expect("Narrowing should succeed");

        // Sign should be preserved for small values
        let final_sign = const_value_sign(&narrowed);
        prop_assert_eq!(original_sign, final_sign,
            "Round-trip should preserve sign: {:?} -> {:?} -> {:?}",
            narrowest, widest, narrowest);
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Convert ConstValue to f64 for numeric comparison.
fn const_value_to_f64(cv: &ConstValue) -> f64 {
    match cv {
        ConstValue::Invalid => panic!("Invalid has no numeric value"),
        ConstValue::Int(v) => *v as f64,
        ConstValue::UInt(v) => *v as f64,
        ConstValue::Float(v) => *v,
        ConstValue::Bool(v) => {
            if *v {
                1.0
            } else {
                0.0
            }
        }
    }
}

/// Get sign of ConstValue (-1, 0, or 1).
fn const_value_sign(cv: &ConstValue) -> i8 {
    let val = const_value_to_f64(cv);
    if val < 0.0 {
        -1
    } else if val > 0.0 {
        1
    } else {
        0
    }
}
