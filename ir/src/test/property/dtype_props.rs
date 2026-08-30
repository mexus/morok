//! Property tests for DType operations and casting.

use proptest::prelude::*;
use proptest::test_runner::TestCaseError;

use svod_dtype::{DType, ScalarDType};

use crate::UOp;
use crate::types::{BinaryOp, ConstValue};
use crate::uop::eval::eval_binary_op_typed;
use crate::uop::range_eval::compute_sound_vmin_vmax;

use super::generators::*;

// ============================================================================
// Constant casting
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2000))]

    /// Casting to the same dtype should be identity.
    #[test]
    fn const_cast_identity((dtype, cv) in const_pair()) {
        prop_assert_eq!(cv, cv.cast(&dtype).expect("cast to same dtype should succeed"));
    }

    #[test]
    fn const_cast_bool_round_trips_through_zero_and_one(b: bool, i in -100i64..=100) {
        let expected = if b { 1i64 } else { 0i64 };
        prop_assert_eq!(ConstValue::Bool(b).cast(&DType::Int32), Some(ConstValue::Int(expected)));
        prop_assert_eq!(ConstValue::Bool(b).cast(&DType::Int64), Some(ConstValue::Int(expected)));
        prop_assert_eq!(ConstValue::Int(i).cast(&DType::Bool), Some(ConstValue::Bool(i != 0)));
    }

    /// Truncating to i8 then widening preserves the truncated value, and the widening path
    /// taken (chained or direct) does not matter.
    #[test]
    fn const_cast_widening_preserves_the_narrowed_value(i in -100i64..=100) {
        let narrow = ConstValue::Int(i).cast(&DType::Int8).unwrap();
        let via_chain = narrow.cast(&DType::Int16).unwrap().cast(&DType::Int32).unwrap();
        prop_assert_eq!(via_chain, narrow.cast(&DType::Int32).unwrap(), "chained must equal direct widening");
        prop_assert_eq!(via_chain.cast(&DType::Int64).unwrap(), ConstValue::Int(i as i8 as i64));
    }

    /// Float to int to float preserves the integer part.
    #[test]
    fn const_cast_float_to_int_to_float(f in -100.0..=100.0) {
        let via_int = ConstValue::Float(f).cast(&DType::Int32).unwrap().cast(&DType::Float32).unwrap();
        let ConstValue::Float(result) = via_int else { panic!("expected Float after cast chain") };
        prop_assert!((result - (f as i32) as f64).abs() < 0.1, "{f} -> {result}");
    }

    /// Numeric zero and one survive a cast into any arithmetic dtype, whichever
    /// representation they start from.
    #[test]
    fn const_cast_preserves_zero_and_one(sdtype in arithmetic_sdtype(), one: bool) {
        let dtype = DType::from(sdtype);
        let sources = if one {
            [ConstValue::Int(1), ConstValue::Float(1.0), ConstValue::Bool(true)]
        } else {
            [ConstValue::Int(0), ConstValue::Float(0.0), ConstValue::Bool(false)]
        };
        for source in sources {
            if let Some(casted) = source.cast(&dtype) {
                prop_assert_eq!(const_value_to_f64(&casted), f64::from(u8::from(one)), "{:?} -> {:?}", source, dtype);
            }
        }
    }
}

// ============================================================================
// Sound range analysis
// ============================================================================

fn value_is_enclosed(value: ConstValue, min: ConstValue, max: ConstValue) -> bool {
    match (value, min, max) {
        (ConstValue::Int(value), ConstValue::Int(min), ConstValue::Int(max)) => min <= value && value <= max,
        (ConstValue::UInt(value), ConstValue::UInt(min), ConstValue::UInt(max)) => min <= value && value <= max,
        (ConstValue::Bool(value), ConstValue::Bool(min), ConstValue::Bool(max)) => min <= value && value <= max,
        _ => false,
    }
}

/// Endpoints and midpoint of an inclusive range; enough to catch a bound that excludes an
/// achievable value without evaluating the whole domain.
fn samples((min, max): (i64, i64)) -> [i64; 3] {
    [min, max, min + (max - min) / 2]
}

fn ordered(pair: (i64, i64)) -> (i64, i64) {
    if pair.0 <= pair.1 { pair } else { (pair.1, pair.0) }
}

/// Every value an operand range can actually take must fall inside the sound range the
/// analysis reports for the operation — for both signedness families of a narrow dtype,
/// where wrap-around is what makes the bound hard.
fn sound_ranges_enclose_sampled_binary_values(
    dtype: DType,
    scalar: ScalarDType,
    wrap: fn(i64) -> ConstValue,
    lhs_bounds: (i64, i64),
    rhs_bounds: (i64, i64),
    shift: i64,
) -> Result<(), TestCaseError> {
    let lhs = UOp::var("prop_lhs", dtype.clone(), lhs_bounds.0, lhs_bounds.1);
    let rhs = UOp::var("prop_rhs", dtype.clone(), rhs_bounds.0, rhs_bounds.1);
    let shift_rhs = UOp::const_(dtype.clone(), wrap(shift));

    let ops = [
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
    ];
    let cases = ops
        .iter()
        .map(|op| (*op, rhs.clone(), samples(rhs_bounds).to_vec()))
        .chain([BinaryOp::Shl, BinaryOp::Shr].iter().map(|op| (*op, shift_rhs.clone(), vec![shift])));

    for (op, rhs, rhs_samples) in cases {
        let expr = UOp::new(crate::Op::Binary(op, lhs.clone(), rhs), dtype.clone());
        let Some((min, max)) = compute_sound_vmin_vmax(&expr) else { continue };
        for a in samples(lhs_bounds) {
            for b in &rhs_samples {
                if let Some(value) = eval_binary_op_typed(op, wrap(a), wrap(*b), scalar) {
                    prop_assert!(
                        value_is_enclosed(value, min, max),
                        "{op:?}: {a}, {b} -> {value:?} not in [{min:?}, {max:?}]"
                    );
                }
            }
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// The sound range of a constant-folded float operation is that exact value, and is
    /// withheld entirely when the result is NaN.
    #[test]
    fn exact_float_operation_sound_bounds_enclose_special_results(
        a_bits in any::<u32>(),
        b_bits in any::<u32>(),
        op in prop_oneof![Just(BinaryOp::Add), Just(BinaryOp::Mul), Just(BinaryOp::Fdiv)],
    ) {
        let (a, b) = (f32::from_bits(a_bits) as f64, f32::from_bits(b_bits) as f64);
        let lhs = UOp::const_(DType::Float32, ConstValue::Float(a));
        let rhs = UOp::const_(DType::Float32, ConstValue::Float(b));
        let expr = UOp::new(crate::Op::Binary(op, lhs, rhs), DType::Float32);

        match eval_binary_op_typed(op, ConstValue::Float(a), ConstValue::Float(b), ScalarDType::Float32) {
            Some(ConstValue::Float(value)) if value.is_nan() => prop_assert!(compute_sound_vmin_vmax(&expr).is_none()),
            Some(value) => prop_assert_eq!(compute_sound_vmin_vmax(&expr), Some((value, value))),
            None => prop_assert!(compute_sound_vmin_vmax(&expr).is_none()),
        }
    }

    #[test]
    fn sampled_int8_binary_values_are_enclosed_by_sound_ranges(
        a in any::<(i8, i8)>(),
        b in any::<(i8, i8)>(),
        shift in 0i8..8,
    ) {
        sound_ranges_enclose_sampled_binary_values(
            DType::Int8,
            ScalarDType::Int8,
            ConstValue::Int,
            ordered((i64::from(a.0), i64::from(a.1))),
            ordered((i64::from(b.0), i64::from(b.1))),
            i64::from(shift),
        )?;
    }

    #[test]
    fn sampled_uint8_binary_values_are_enclosed_by_sound_ranges(
        a in any::<(u8, u8)>(),
        b in any::<(u8, u8)>(),
        shift in 0u8..8,
    ) {
        sound_ranges_enclose_sampled_binary_values(
            DType::UInt8,
            ScalarDType::UInt8,
            |value| ConstValue::UInt(value as u64),
            ordered((i64::from(a.0), i64::from(a.1))),
            ordered((i64::from(b.0), i64::from(b.1))),
            i64::from(shift),
        )?;
    }

    /// Narrowing casts wrap, so the reported range must still enclose every wrapped value.
    #[test]
    fn sampled_narrow_integer_cast_values_are_enclosed(
        pair in any::<(i16, i16)>(),
        upair in any::<(u16, u16)>(),
    ) {
        let signed = ordered((i64::from(pair.0), i64::from(pair.1)));
        let unsigned = ordered((i64::from(upair.0), i64::from(upair.1)));
        let sources = [
            (UOp::var("prop_cast_i16", DType::Int16, signed.0, signed.1), signed, ConstValue::Int as fn(i64) -> _),
            (UOp::var("prop_cast_u16", DType::UInt16, unsigned.0, unsigned.1), unsigned, |v| ConstValue::UInt(v as u64)),
        ];

        for (src, bounds, wrap) in sources {
            for target in [DType::Int8, DType::UInt8] {
                let cast = src.cast(target.clone());
                let Some((min, max)) = compute_sound_vmin_vmax(&cast) else { continue };
                for value in samples(bounds) {
                    prop_assert!(value_is_enclosed(wrap(value).cast(&target).unwrap(), min, max));
                }
            }
        }
    }
}

// ============================================================================
// DType families
// ============================================================================

proptest! {
    #![proptest_config(ProptestConfig::with_cases(1000))]

    /// Within a dtype family, widening should preserve values for in-range constants.
    #[test]
    fn dtype_family_widening_preserves_small_values(family in arb_dtype_family(), val in -10i64..=10) {
        let dtypes = family.widening_sequence();
        let Some(narrowest) = ConstValue::Int(val).cast(&dtypes[0]) else { return Ok(()) };
        let expected = const_value_to_f64(&narrowest);

        let mut current = narrowest;
        for dtype in &dtypes[1..] {
            current = current.cast(dtype).expect("widening should succeed");
            prop_assert!(
                (expected - const_value_to_f64(&current)).abs() < 0.1,
                "widening to {dtype:?} should preserve {expected}"
            );
        }
    }

    /// Widening then narrowing back may lose precision but must not change sign.
    #[test]
    fn dtype_roundtrip_preserves_sign(family in arb_dtype_family(), val in -10i64..=10) {
        let dtypes = family.widening_sequence();
        let (narrowest, widest) = (&dtypes[0], &dtypes[dtypes.len() - 1]);
        let Some(cv) = ConstValue::Int(val).cast(narrowest) else { return Ok(()) };

        let round_tripped = cv.cast(widest).expect("widening").cast(narrowest).expect("narrowing");
        prop_assert_eq!(const_value_sign(&cv), const_value_sign(&round_tripped), "{:?} -> {:?} -> back", narrowest, widest);
    }
}

fn const_value_to_f64(cv: &ConstValue) -> f64 {
    match cv {
        ConstValue::Invalid => panic!("Invalid has no numeric value"),
        ConstValue::Int(v) => *v as f64,
        ConstValue::UInt(v) => *v as f64,
        ConstValue::Float(v) => *v,
        ConstValue::Bool(v) => f64::from(u8::from(*v)),
    }
}

fn const_value_sign(cv: &ConstValue) -> i8 {
    const_value_to_f64(cv).partial_cmp(&0.0).map_or(0, |ordering| ordering as i8)
}
