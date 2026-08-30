//! Unit tests for vmin/vmax range analysis.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::uop::range_eval::compute_sound_vmin_vmax;
use crate::{BinaryOp, ConstValue, Op, UOp};
use svod_dtype::DType;

fn binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>, dtype: DType) -> Arc<UOp> {
    UOp::new(Op::Binary(op, lhs, rhs), dtype)
}

fn unknown_float(dtype: DType) -> Arc<UOp> {
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 1, dtype);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    UOp::load().index(index).call()
}

/// Nothing is known about the value, so the analysis must stay at the whole float line
/// and refuse to hand out a sound (NaN-free) range.
fn assert_unbounded(value: &Arc<UOp>, label: &str) {
    assert_eq!(value.vmin(), &ConstValue::Float(f64::NEG_INFINITY), "{label}");
    assert_eq!(value.vmax(), &ConstValue::Float(f64::INFINITY), "{label}");
    assert!(compute_sound_vmin_vmax(value).is_none(), "{label}");
}

// ============================================================================
// Interval arithmetic
// ============================================================================

/// One row per range rule. Constant-folded rows have `min == max`; the rest pin the
/// interval the rule widens to, including the dtype-bound fallbacks for the cases
/// (division/modulo by a range containing zero, oversized shifts, narrowing casts)
/// where no tighter interval is sound.
#[test]
fn range_analysis_intervals() {
    let ranged = |name: &str, lo, hi| UOp::var(name, DType::Int32, lo, hi);
    let int = |v: i32| UOp::native_const(v);
    let x = ranged("x", 0, 10);

    let int_rows: Vec<(&str, Arc<UOp>, i64, i64)> = vec![
        ("const", int(5), 5, 5),
        ("negative const", int(-3), -3, -3),
        ("add", int(2).try_add(&int(3)).unwrap(), 5, 5),
        ("sub", int(10).try_sub(&int(3)).unwrap(), 7, 7),
        ("mul", int(-2).try_mul(&int(3)).unwrap(), -6, -6),
        ("max", int(5).try_max(&int(10)).unwrap(), 10, 10),
        ("nested max", int(3).try_max(&int(7)).unwrap().try_max(&int(5)).unwrap(), 7, 7),
        ("idiv", int(15).try_div(&int(3)).unwrap(), 5, 5),
        ("mod", int(17).try_mod(&int(5)).unwrap(), 2, 2),
        ("neg", int(5).neg(), -5, -5),
        ("and", int(15).try_and_op(&int(7)).unwrap(), 7, 7),
        ("shl", int(3).try_shl_op(&int(2)).unwrap(), 12, 12),
        ("shr", int(12).try_shr_op(&int(2)).unwrap(), 3, 3),
        ("mulacc", UOp::try_mulacc(int(3), int(4), int(5)).unwrap(), 17, 17),
        ("cast float to int", UOp::native_const(5.7f32).cast(DType::Int32), 5, 5),
        ("where true", UOp::try_where(UOp::native_const(true), int(10), int(5)).unwrap(), 10, 10),
        ("where false", UOp::try_where(UOp::native_const(false), int(10), int(5)).unwrap(), 5, 5),
        ("define_var", UOp::define_var("v".to_string(), 5, 20), 5, 20),
        ("range", UOp::range_const(10, 0), 0, 9),
        ("neg of range", ranged("n", 0, 5).neg(), -5, 0),
        ("mul of ranges", ranged("a", 0, 3).try_mul(&ranged("b", 0, 4)).unwrap(), 0, 12),
        ("(x + 5) * 2", x.try_add(&int(5)).unwrap().try_mul(&int(2)).unwrap(), 10, 30),
        (
            "where over a range condition",
            UOp::try_where(ranged("c", 0, 1).try_cmpgt(&UOp::index_const(0)).unwrap(), int(10), int(5)).unwrap(),
            5,
            10,
        ),
        // No tighter interval is sound below this line.
        ("narrowing cast wraps", ranged("w", 0, 1000).cast(DType::Int8), -128, 127),
        (
            "idiv by a range containing zero",
            int(10).try_div(&ranged("z", 0, 1)).unwrap(),
            i32::MIN as i64,
            i32::MAX as i64,
        ),
        (
            "mod by a range containing zero",
            int(10).try_mod(&ranged("z", 0, 1)).unwrap(),
            i32::MIN as i64,
            i32::MAX as i64,
        ),
        ("oversized shift", int(1).try_shl_op(&int(64)).unwrap(), i32::MIN as i64, i32::MAX as i64),
    ];
    for (name, uop, min, max) in int_rows {
        assert_eq!((uop.vmin(), uop.vmax()), (&ConstValue::Int(min), &ConstValue::Int(max)), "{name}");
    }

    for (name, uop, expected) in [
        ("float const", UOp::native_const(PI), PI as f64),
        ("float add", UOp::native_const(2.5f32).try_add(&UOp::native_const(1.5f32)).unwrap(), 4.0),
        ("float sub", UOp::native_const(2.5f32).try_sub(&UOp::native_const(1.5f32)).unwrap(), 1.0),
        ("float mul", UOp::native_const(2.5f32).try_mul(&UOp::native_const(1.5f32)).unwrap(), 3.75),
        // The f32 operation is committed before its analysis bound is recorded.
        ("float div", UOp::native_const(2.5f32).try_div(&UOp::native_const(1.5f32)).unwrap(), (2.5f32 / 1.5f32) as f64),
    ] {
        assert_eq!((uop.vmin(), uop.vmax()), (&ConstValue::Float(expected), &ConstValue::Float(expected)), "{name}");
    }

    for (name, uop, expected) in [
        ("cmplt", UOp::native_const(5i32).try_cmplt(&UOp::native_const(10i32)).unwrap(), true),
        ("cmpeq", UOp::native_const(5i32).try_cmpeq(&UOp::native_const(5i32)).unwrap(), true),
        ("bool and", UOp::native_const(true).try_and_op(&UOp::native_const(false)).unwrap(), false),
        ("bool or", UOp::native_const(true).try_or_op(&UOp::native_const(false)).unwrap(), true),
    ] {
        assert_eq!((uop.vmin(), uop.vmax()), (&ConstValue::Bool(expected), &ConstValue::Bool(expected)), "{name}");
    }
}

// ============================================================================
// Sound float bounds
// ============================================================================

#[test]
fn unknown_floats_stay_unbounded_through_stacking_casts_and_division() {
    for dtype in [
        DType::FP8E4M3,
        DType::FP8E5M2,
        DType::FP8E4M3FNUZ,
        DType::FP8E5M2FNUZ,
        DType::Float16,
        DType::BFloat16,
        DType::Float32,
        DType::Float64,
    ] {
        assert_unbounded(&unknown_float(dtype.clone()), &format!("{dtype:?} load"));
        assert_unbounded(&unknown_float(DType::Float32).cast(dtype.clone()), &format!("cast to {dtype:?}"));
    }
    assert_unbounded(&UOp::param(0, 1, DType::Float32, Some(svod_dtype::DeviceSpec::Cpu)), "param");

    let scalar = unknown_float(DType::Float32);
    let row = UOp::stack(vec![scalar.clone(); 4].into());
    assert_unbounded(&UOp::stack(vec![row.clone(), row].into()), "shaped stack");

    assert_unbounded(&scalar.try_div(&unknown_float(DType::Float32)).unwrap(), "unknown / unknown");
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    assert_unbounded(&binary(BinaryOp::Fdiv, scalar, zero, DType::Float32), "unknown / 0");
}

#[test]
fn float_constants_specials_and_overflow_have_only_sound_exact_bounds() {
    for value in [f64::NEG_INFINITY, -0.0, 0.0, f64::INFINITY] {
        let constant = UOp::const_(DType::Float32, ConstValue::Float(value));
        assert_eq!(compute_sound_vmin_vmax(&constant), Some((ConstValue::Float(value), ConstValue::Float(value))));
    }

    let nan = UOp::const_(DType::Float32, ConstValue::Float(f64::NAN));
    assert!(nan.vmin().try_float().unwrap().is_nan());
    assert!(compute_sound_vmin_vmax(&nan).is_none());

    for dtype in [DType::Float16, DType::BFloat16, DType::Float32, DType::Float64] {
        let max = UOp::const_(dtype.clone(), ConstValue::Float(dtype.max_value()));
        let overflow = max.try_add(&max).unwrap();
        let ConstValue::Float(bound) = overflow.vmin() else { panic!("expected float bound") };
        assert!(bound.is_infinite(), "{dtype:?} finite overflow must include infinity");
        assert_eq!(overflow.vmin(), overflow.vmax());
    }
}

/// An integer range casts to an exact float range unless the target format overflows,
/// in which case the bound has to open up to infinity.
#[test]
fn integer_to_float_casts_are_exact_until_the_format_overflows() {
    let bounded = UOp::var("i", DType::Int32, -3, 7).cast(DType::Float16);
    assert_eq!(compute_sound_vmin_vmax(&bounded), Some((ConstValue::Float(-3.0), ConstValue::Float(7.0))));

    let near_overflow = UOp::var("wide", DType::Int32, 65_504, 65_520).cast(DType::Float16);
    assert_eq!(
        compute_sound_vmin_vmax(&near_overflow),
        Some((ConstValue::Float(65_504.0), ConstValue::Float(f64::INFINITY)))
    );
}

#[test]
fn float_division_by_zero_tracks_infinity_and_nan() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let one_over_zero = binary(BinaryOp::Fdiv, one, zero.clone(), DType::Float32);
    assert_eq!(
        compute_sound_vmin_vmax(&one_over_zero),
        Some((ConstValue::Float(f64::INFINITY), ConstValue::Float(f64::INFINITY)))
    );

    let zero_over_zero = binary(BinaryOp::Fdiv, zero.clone(), zero, DType::Float32);
    assert!(zero_over_zero.vmin().try_float().is_some_and(f64::is_nan));
    assert!(compute_sound_vmin_vmax(&zero_over_zero).is_none());
}

#[test]
fn sound_float_ranges_preserve_signed_zero_and_domains() {
    let index = UOp::var("i", DType::Int32, 0, 1);
    let condition = index.try_cmplt(&UOp::native_const(1i32)).unwrap();
    let negative_zero = UOp::const_(DType::Float32, ConstValue::Float(-0.0));
    let positive_zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let selected = UOp::try_where(condition, negative_zero, positive_zero).unwrap();

    let (min, max) = compute_sound_vmin_vmax(&selected).expect("NaN-free WHERE has sound bounds");
    assert!(matches!(min, ConstValue::Float(v) if v == 0.0 && v.is_sign_negative()));
    assert!(matches!(max, ConstValue::Float(v) if v == 0.0 && v.is_sign_positive()));
    assert!(compute_sound_vmin_vmax(&UOp::try_reciprocal(&selected).unwrap()).is_none());

    let negative = UOp::const_(DType::Float32, ConstValue::Float(-1.0));
    assert!(compute_sound_vmin_vmax(&UOp::try_sqrt(&negative).unwrap()).is_none());
}

// ============================================================================
// Narrow integer wrap-around
// ============================================================================

#[test]
fn narrow_integer_ranges_commit_or_fall_back_before_float_proofs() {
    let signed = UOp::var("signed", DType::Int8, 100, 127);
    let sum = signed.try_add(&signed).unwrap();
    assert_eq!(compute_sound_vmin_vmax(&sum), Some((ConstValue::Int(-128), ConstValue::Int(127))));

    let as_float = sum.cast(DType::Float32);
    assert_eq!(compute_sound_vmin_vmax(&as_float), Some((ConstValue::Float(-128.0), ConstValue::Float(127.0))));
    let positive = as_float.try_cmpgt(&UOp::const_(DType::Float32, ConstValue::Float(0.0))).unwrap();
    assert_eq!(compute_sound_vmin_vmax(&positive), Some((ConstValue::Bool(false), ConstValue::Bool(true))));

    let selected = UOp::try_where(positive, UOp::native_const(7i32), UOp::native_const(11i32)).unwrap();
    assert_eq!(compute_sound_vmin_vmax(&selected), Some((ConstValue::Int(7), ConstValue::Int(11))));

    let unsigned = UOp::var("unsigned", DType::UInt8, 200, 255);
    let increment = UOp::var("increment", DType::UInt8, 0, 100);
    let wrapped = unsigned.try_add(&increment).unwrap();
    assert_eq!(compute_sound_vmin_vmax(&wrapped), Some((ConstValue::UInt(0), ConstValue::UInt(255))));

    let minuend = UOp::var("minuend", DType::UInt8, 0, 20);
    let subtrahend = UOp::var("subtrahend", DType::UInt8, 10, 30);
    let underflow = binary(BinaryOp::Sub, minuend, subtrahend, DType::UInt8);
    assert_eq!(compute_sound_vmin_vmax(&underflow), Some((ConstValue::UInt(0), ConstValue::UInt(255))));
}

#[test]
fn narrow_integer_multiplication_shifts_negation_mulacc_and_safe_ranges_are_sound() {
    let lhs = UOp::var("lhs", DType::Int8, 20, 30);
    let rhs = UOp::var("rhs", DType::Int8, 10, 10);
    let wrapped_product = lhs.try_mul(&rhs).unwrap();
    assert_eq!(compute_sound_vmin_vmax(&wrapped_product), Some((ConstValue::Int(-128), ConstValue::Int(127))));

    let safe_lhs = UOp::var("safe_lhs", DType::Int8, 10, 20);
    let safe_rhs = UOp::var("safe_rhs", DType::Int8, 3, 4);
    let safe_sum = safe_lhs.try_add(&safe_rhs).unwrap();
    assert_eq!(compute_sound_vmin_vmax(&safe_sum), Some((ConstValue::Int(13), ConstValue::Int(24))));

    let shift_value = UOp::var("shift_value", DType::Int8, 60, 64);
    let shift = UOp::var("shift", DType::UInt8, 1, 1);
    assert_eq!(
        compute_sound_vmin_vmax(&shift_value.try_shl_op(&shift).unwrap()),
        Some((ConstValue::Int(-128), ConstValue::Int(127)))
    );
    assert_eq!(
        compute_sound_vmin_vmax(&shift_value.try_shr_op(&shift).unwrap()),
        Some((ConstValue::Int(30), ConstValue::Int(32)))
    );

    let negated = UOp::new(Op::Unary(crate::UnaryOp::Neg, UOp::var("neg", DType::Int8, -128, -127)), DType::Int8);
    assert_eq!(compute_sound_vmin_vmax(&negated), Some((ConstValue::Int(-128), ConstValue::Int(127))));

    let a = UOp::var("a", DType::Int8, 10, 12);
    let b = UOp::var("b", DType::Int8, 2, 3);
    let c = UOp::var("c", DType::Int8, 1, 2);
    let mulacc = UOp::try_mulacc(a, b, c).unwrap();
    assert!(compute_sound_vmin_vmax(&mulacc).is_none());
    assert_eq!((mulacc.vmin(), mulacc.vmax()), (&ConstValue::Int(-128), &ConstValue::Int(127)));
}

#[test]
fn integer_cast_ranges_commit_without_endpoint_only_wrap() {
    let wide = UOp::var("wide", DType::Int16, -200, 300);
    let narrow = wide.cast(DType::Int8);
    assert!(compute_sound_vmin_vmax(&narrow).is_none());
    assert_eq!((narrow.vmin(), narrow.vmax()), (&ConstValue::Int(-128), &ConstValue::Int(127)));

    let nonnegative = UOp::var("nonnegative", DType::Int16, 0, 100).cast(DType::UInt8);
    assert_eq!(compute_sound_vmin_vmax(&nonnegative), Some((ConstValue::UInt(0), ConstValue::UInt(100))));

    let crosses_zero = UOp::var("crosses_zero", DType::Int8, -1, 1).cast(DType::Bool);
    assert_eq!(compute_sound_vmin_vmax(&crosses_zero), Some((ConstValue::Bool(false), ConstValue::Bool(true))));
}

// ============================================================================
// Float comparisons
// ============================================================================

#[test]
fn unknown_and_nan_capable_float_comparisons_never_narrow_ordinary_bounds() {
    let ops = [BinaryOp::Eq, BinaryOp::Ne, BinaryOp::Lt, BinaryOp::Le, BinaryOp::Gt, BinaryOp::Ge];
    for dtype in [DType::FP8E4M3, DType::FP8E5M2FNUZ, DType::Float16, DType::BFloat16, DType::Float32] {
        let unknown = unknown_float(dtype.clone());
        let rhs = [
            UOp::const_(dtype.clone(), ConstValue::Float(1.0)),
            UOp::const_(dtype.clone(), ConstValue::Float(f64::INFINITY)),
            UOp::const_(dtype, ConstValue::Float(f64::NEG_INFINITY)),
            unknown.clone(),
        ];
        for op in ops {
            for rhs in &rhs {
                let comparison = binary(op, unknown.clone(), rhs.clone(), DType::Bool);
                assert_eq!((comparison.vmin(), comparison.vmax()), (&ConstValue::Bool(false), &ConstValue::Bool(true)));
            }
        }
    }

    let nan = UOp::const_(DType::Float32, ConstValue::Float(f64::NAN));
    for op in ops {
        let comparison = binary(op, nan.clone(), nan.clone(), DType::Bool);
        assert_eq!((comparison.vmin(), comparison.vmax()), (&ConstValue::Bool(false), &ConstValue::Bool(true)));
    }
}

#[test]
fn nan_free_float_ranges_allow_all_reflexive_comparison_proofs() {
    let value = UOp::var("finite", DType::Float32, -3, 7);
    for (op, expected) in [
        (BinaryOp::Eq, true),
        (BinaryOp::Ne, false),
        (BinaryOp::Lt, false),
        (BinaryOp::Le, true),
        (BinaryOp::Gt, false),
        (BinaryOp::Ge, true),
    ] {
        let comparison = binary(op, value.clone(), value.clone(), DType::Bool);
        assert_eq!(
            compute_sound_vmin_vmax(&comparison),
            Some((ConstValue::Bool(expected), ConstValue::Bool(expected)))
        );
        assert_eq!((comparison.vmin(), comparison.vmax()), (&ConstValue::Bool(expected), &ConstValue::Bool(expected)));
    }
}
