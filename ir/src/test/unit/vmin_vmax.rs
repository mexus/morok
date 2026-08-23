//! Unit tests for vmin/vmax range analysis.

use std::f32::consts::PI;
use std::sync::Arc;

use crate::uop::range_eval::compute_sound_vmin_vmax;
use crate::{AxisId, BinaryOp, ConstValue, Op, UOp};
use svod_dtype::DType;

// ============================================================================
// Test Constants
// ============================================================================

#[test]
fn test_vmin_vmax_const() {
    assert_eq!(UOp::native_const(5i32).vmin(), &ConstValue::Int(5));
    assert_eq!(UOp::native_const(5i32).vmax(), &ConstValue::Int(5));

    assert_eq!(UOp::native_const(-3i32).vmin(), &ConstValue::Int(-3));
    assert_eq!(UOp::native_const(-3i32).vmax(), &ConstValue::Int(-3));

    assert_eq!(UOp::native_const(PI).vmin(), &ConstValue::Float(PI as f64));
    assert_eq!(UOp::native_const(PI).vmax(), &ConstValue::Float(PI as f64));

    assert_eq!(UOp::native_const(true).vmin(), &ConstValue::Bool(true));
    assert_eq!(UOp::native_const(true).vmax(), &ConstValue::Bool(true));
}

// ============================================================================
// Test Arithmetic Operations
// ============================================================================

#[test]
fn test_vmin_vmax_add() {
    let sum = UOp::native_const(2i32).try_add(&UOp::native_const(3i32)).unwrap();

    assert_eq!(sum.vmin(), &ConstValue::Int(5));
    assert_eq!(sum.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_sub() {
    let a = UOp::native_const(10i32);
    let b = UOp::native_const(3i32);
    let diff = a.try_sub(&b).unwrap();

    assert_eq!(diff.vmin(), &ConstValue::Int(7));
    assert_eq!(diff.vmax(), &ConstValue::Int(7));
}

#[test]
fn test_vmin_vmax_mul() {
    let a = UOp::native_const(-2i32);
    let b = UOp::native_const(3i32);
    let prod = a.try_mul(&b).unwrap();

    assert_eq!(prod.vmin(), &ConstValue::Int(-6));
    assert_eq!(prod.vmax(), &ConstValue::Int(-6));
}

#[test]
fn test_vmin_vmax_mul_range() {
    // Test multiplication with ranges
    let a = UOp::define_var("a".to_string(), 0, 3);
    let b = UOp::define_var("b".to_string(), 0, 4);
    let prod = a.try_mul(&b).unwrap();

    // a ∈ [0, 3], b ∈ [0, 4]
    // Check all 4 corners: 0*0=0, 0*4=0, 3*0=0, 3*4=12
    // Min is 0, max is 12
    assert_eq!(prod.vmin(), &ConstValue::Int(0));
    assert_eq!(prod.vmax(), &ConstValue::Int(12));
}

#[test]
fn test_vmin_vmax_max() {
    let a = UOp::native_const(5i32);
    let b = UOp::native_const(10i32);
    let max_val = a.try_max(&b).unwrap();

    assert_eq!(max_val.vmin(), &ConstValue::Int(10));
    assert_eq!(max_val.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_idiv() {
    let a = UOp::native_const(15i32);
    let b = UOp::native_const(3i32);
    let div = a.try_div(&b).unwrap();

    assert_eq!(div.vmin(), &ConstValue::Int(5));
    assert_eq!(div.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_mod() {
    let a = UOp::native_const(17i32);
    let b = UOp::native_const(5i32);
    let modulo = a.try_mod(&b).unwrap();

    assert_eq!(modulo.vmin(), &ConstValue::Int(2));
    assert_eq!(modulo.vmax(), &ConstValue::Int(2));
}

// ============================================================================
// Test Unary Operations
// ============================================================================

#[test]
fn test_vmin_vmax_neg() {
    let five = UOp::native_const(5i32);
    let neg = five.neg();

    assert_eq!(neg.vmin(), &ConstValue::Int(-5));
    assert_eq!(neg.vmax(), &ConstValue::Int(-5));
}

#[test]
fn test_vmin_vmax_neg_range() {
    let var = UOp::define_var("x".to_string(), 0, 5);
    let neg = var.neg();

    // var ∈ [0, 5], so neg(var) ∈ [-5, 0]
    assert_eq!(neg.vmin(), &ConstValue::Int(-5));
    assert_eq!(neg.vmax(), &ConstValue::Int(0));
}

// ============================================================================
// Test Comparison Operations
// ============================================================================

#[test]
fn test_vmin_vmax_cmplt() {
    let a = UOp::native_const(5i32);
    let b = UOp::native_const(10i32);
    let cmp = a.try_cmplt(&b).unwrap();

    // 5 < 10 is always true, so range is [true, true]
    assert_eq!(cmp.vmin(), &ConstValue::Bool(true));
    assert_eq!(cmp.vmax(), &ConstValue::Bool(true));
}

#[test]
fn test_vmin_vmax_eq() {
    let a = UOp::native_const(5i32);
    let b = UOp::native_const(5i32);
    let eq = a.try_cmpeq(&b).unwrap();

    // 5 == 5 is always true, so range is [true, true]
    assert_eq!(eq.vmin(), &ConstValue::Bool(true));
    assert_eq!(eq.vmax(), &ConstValue::Bool(true));
}

// ============================================================================
// Test Bitwise Operations
// ============================================================================

#[test]
fn test_vmin_vmax_and_bool() {
    let and = UOp::native_const(true).try_and_op(&UOp::native_const(false)).unwrap();

    // true & false = false
    assert_eq!(and.vmin(), &ConstValue::Bool(false));
    assert_eq!(and.vmax(), &ConstValue::Bool(false));
}

#[test]
fn test_vmin_vmax_or_bool() {
    let or = UOp::native_const(true).try_or_op(&UOp::native_const(false)).unwrap();

    // true | false = true
    assert_eq!(or.vmin(), &ConstValue::Bool(true));
    assert_eq!(or.vmax(), &ConstValue::Bool(true));
}

#[test]
fn test_vmin_vmax_and_int() {
    let a = UOp::native_const(15i32); // 0b1111
    let b = UOp::native_const(7i32); // 0b0111
    let and = a.try_and_op(&b).unwrap();

    // 15 & 7 = 7
    assert_eq!(and.vmin(), &ConstValue::Int(7));
    assert_eq!(and.vmax(), &ConstValue::Int(7));
}

#[test]
fn test_vmin_vmax_shl() {
    let a = UOp::native_const(3i32);
    let b = UOp::native_const(2i32);
    let shl = a.try_shl_op(&b).unwrap();

    // 3 << 2 = 12
    assert_eq!(shl.vmin(), &ConstValue::Int(12));
    assert_eq!(shl.vmax(), &ConstValue::Int(12));
}

#[test]
fn test_vmin_vmax_shr() {
    let a = UOp::native_const(12i32);
    let b = UOp::native_const(2i32);
    let shr = a.try_shr_op(&b).unwrap();

    // 12 >> 2 = 3
    assert_eq!(shr.vmin(), &ConstValue::Int(3));
    assert_eq!(shr.vmax(), &ConstValue::Int(3));
}

// ============================================================================
// Test Special Operations
// ============================================================================

#[test]
fn test_vmin_vmax_define_var() {
    let var = UOp::define_var("x".to_string(), 0, 20);

    assert_eq!(var.vmin(), &ConstValue::Int(0));
    assert_eq!(var.vmax(), &ConstValue::Int(20));
}

#[test]
fn test_vmin_vmax_define_var_with_min() {
    // Test variable with non-zero min_val
    let var = UOp::define_var("x".to_string(), 5, 20);

    assert_eq!(var.vmin(), &ConstValue::Int(5));
    assert_eq!(var.vmax(), &ConstValue::Int(20));
}

#[test]
fn test_vmin_vmax_range() {
    let end = UOp::native_const(10i32);
    let range = UOp::new(
        Op::Range {
            end,
            axis_id: AxisId::Renumbered(0),
            axis_type: crate::types::AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Int32,
    );

    // RANGE goes from 0 to end-1
    assert_eq!(range.vmin(), &ConstValue::Int(0));
    assert_eq!(range.vmax(), &ConstValue::Int(9));
}

#[test]
fn test_vmin_vmax_cast() {
    let float_val = UOp::native_const(5.7f32);
    let int_val = float_val.cast(DType::Int32);

    // Cast from 5.7 to int = 5
    assert_eq!(int_val.vmin(), &ConstValue::Int(5));
    assert_eq!(int_val.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_cast_range() {
    let var = UOp::define_var("x".to_string(), 0, 1000);
    // Cast to Int8 which has range [-128, 127]
    let casted = var.cast(DType::Int8);

    // The interior wraps through the full Int8 domain; endpoint clamping is unsound.
    assert_eq!(casted.vmin(), &ConstValue::Int(-128));
    assert_eq!(casted.vmax(), &ConstValue::Int(127));
}

// ============================================================================
// Test Ternary Operations
// ============================================================================

#[test]
fn test_vmin_vmax_where_true() {
    let where_op = UOp::try_where(UOp::native_const(true), UOp::native_const(10i32), UOp::native_const(5i32)).unwrap();

    // Condition is always true, so result is true_val
    assert_eq!(where_op.vmin(), &ConstValue::Int(10));
    assert_eq!(where_op.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_where_false() {
    let where_op = UOp::try_where(UOp::native_const(false), UOp::native_const(10i32), UOp::native_const(5i32)).unwrap();

    // Condition is always false, so result is false_val
    assert_eq!(where_op.vmin(), &ConstValue::Int(5));
    assert_eq!(where_op.vmax(), &ConstValue::Int(5));
}

#[test]
fn test_vmin_vmax_where_range() {
    // Condition can be either true or false - use a comparison to get bool dtype
    let var = UOp::define_var("cond".to_string(), 0, 1);
    let zero = UOp::index_const(0);
    let cond = var.try_cmpgt(&zero).unwrap();
    let true_val = UOp::native_const(10i32);
    let false_val = UOp::native_const(5i32);
    let where_op = UOp::try_where(cond, true_val, false_val).unwrap();

    // Could be either branch
    assert_eq!(where_op.vmin(), &ConstValue::Int(5));
    assert_eq!(where_op.vmax(), &ConstValue::Int(10));
}

#[test]
fn test_vmin_vmax_mulacc() {
    let a = UOp::native_const(3i32);
    let b = UOp::native_const(4i32);
    let c = UOp::native_const(5i32);
    let mulacc = UOp::try_mulacc(a, b, c).unwrap();

    // 3 * 4 + 5 = 17
    assert_eq!(mulacc.vmin(), &ConstValue::Int(17));
    assert_eq!(mulacc.vmax(), &ConstValue::Int(17));
}

// ============================================================================
// Test Complex Expressions
// ============================================================================

#[test]
fn test_vmin_vmax_complex_expression() {
    // Test: (x + 5) * 2 where x in [0, 10]
    let x = UOp::var("x", DType::Int32, 0, 10);
    let five = UOp::native_const(5i32);
    let two = UOp::native_const(2i32);

    let x_plus_5 = x.try_add(&five).unwrap();
    let result = x_plus_5.try_mul(&two).unwrap();

    // x in [0, 10] => x+5 in [5, 15] => (x+5)*2 in [10, 30]
    assert_eq!(result.vmin(), &ConstValue::Int(10));
    assert_eq!(result.vmax(), &ConstValue::Int(30));
}

#[test]
fn test_vmin_vmax_nested_max() {
    // Test: max(max(a, b), c) where a=3, b=7, c=5
    let a = UOp::native_const(3i32);
    let b = UOp::native_const(7i32);
    let c = UOp::native_const(5i32);

    let max_ab = a.try_max(&b).unwrap();
    let max_abc = max_ab.try_max(&c).unwrap();

    assert_eq!(max_abc.vmin(), &ConstValue::Int(7));
    assert_eq!(max_abc.vmax(), &ConstValue::Int(7));
}

// ============================================================================
// Test Float Operations
// ============================================================================

#[test]
fn test_vmin_vmax_float_ops() {
    let a = UOp::native_const(2.5f32);
    let b = UOp::native_const(1.5f32);

    let sum = a.try_add(&b).unwrap();
    assert_eq!(sum.vmin(), &ConstValue::Float(4.0));
    assert_eq!(sum.vmax(), &ConstValue::Float(4.0));

    let diff = a.try_sub(&b).unwrap();
    assert_eq!(diff.vmin(), &ConstValue::Float(1.0));
    assert_eq!(diff.vmax(), &ConstValue::Float(1.0));

    let prod = a.try_mul(&b).unwrap();
    assert_eq!(prod.vmin(), &ConstValue::Float(3.75));
    assert_eq!(prod.vmax(), &ConstValue::Float(3.75));

    let div = a.try_div(&b).unwrap();
    // The f32 operation is committed before its analysis bound is recorded.
    if let ConstValue::Float(min_val) = div.vmin() {
        assert_eq!(*min_val, (2.5f32 / 1.5f32) as f64);
    } else {
        panic!("Expected float result");
    }
}

fn unknown_float(dtype: DType) -> Arc<UOp> {
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 1, dtype);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    UOp::load().index(index).call()
}

#[test]
fn unknown_float_formats_use_infinite_analysis_bounds() {
    let dtypes = [
        DType::FP8E4M3,
        DType::FP8E5M2,
        DType::FP8E4M3FNUZ,
        DType::FP8E5M2FNUZ,
        DType::Float16,
        DType::BFloat16,
        DType::Float32,
        DType::Float64,
    ];
    for dtype in dtypes {
        let value = unknown_float(dtype.clone());
        assert_eq!(value.vmin(), &ConstValue::Float(f64::NEG_INFINITY), "{dtype:?}");
        assert_eq!(value.vmax(), &ConstValue::Float(f64::INFINITY), "{dtype:?}");
        assert!(compute_sound_vmin_vmax(&value).is_none(), "{dtype:?}");
    }

    let param = UOp::param(0, 1, DType::Float32, Some(svod_dtype::DeviceSpec::Cpu));
    assert_eq!(param.vmin(), &ConstValue::Float(f64::NEG_INFINITY));
    assert_eq!(param.vmax(), &ConstValue::Float(f64::INFINITY));
    assert!(compute_sound_vmin_vmax(&param).is_none());
}

#[test]
fn unknown_vector_and_shaped_float_values_stay_unbounded() {
    let scalar = unknown_float(DType::Float32);
    let vector = UOp::stack(vec![scalar.clone(), scalar.clone(), scalar.clone(), scalar.clone()].into());
    let shaped = UOp::stack(vec![vector.clone(), vector].into());
    for value in [scalar, shaped] {
        assert_eq!(value.vmin(), &ConstValue::Float(f64::NEG_INFINITY));
        assert_eq!(value.vmax(), &ConstValue::Float(f64::INFINITY));
        assert!(compute_sound_vmin_vmax(&value).is_none());
    }
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

#[test]
fn unknown_float_cast_and_division_ranges_are_conservative() {
    let value = unknown_float(DType::Float32);
    let cast = value.cast(DType::Float16);
    assert_eq!(cast.vmin(), &ConstValue::Float(f64::NEG_INFINITY));
    assert_eq!(cast.vmax(), &ConstValue::Float(f64::INFINITY));
    assert!(compute_sound_vmin_vmax(&cast).is_none());

    let divisor = unknown_float(DType::Float32);
    let division = value.try_div(&divisor).unwrap();
    assert_eq!(division.vmin(), &ConstValue::Float(f64::NEG_INFINITY));
    assert_eq!(division.vmax(), &ConstValue::Float(f64::INFINITY));
    assert!(compute_sound_vmin_vmax(&division).is_none());

    let bounded_int = UOp::var("i", DType::Int32, -3, 7);
    let exact_float_cast = bounded_int.cast(DType::Float16);
    assert_eq!(compute_sound_vmin_vmax(&exact_float_cast), Some((ConstValue::Float(-3.0), ConstValue::Float(7.0))));
}

#[test]
fn float_division_by_zero_tracks_infinity_and_nan() {
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let one_over_zero = UOp::new(Op::Binary(BinaryOp::Fdiv, one, zero.clone()), DType::Float32);
    assert_eq!(
        compute_sound_vmin_vmax(&one_over_zero),
        Some((ConstValue::Float(f64::INFINITY), ConstValue::Float(f64::INFINITY)))
    );

    let zero_over_zero = UOp::new(Op::Binary(BinaryOp::Fdiv, zero.clone(), zero), DType::Float32);
    assert!(zero_over_zero.vmin().try_float().is_some_and(f64::is_nan));
    assert!(compute_sound_vmin_vmax(&zero_over_zero).is_none());

    let unknown = unknown_float(DType::Float32);
    let unknown_over_zero = UOp::new(
        Op::Binary(BinaryOp::Fdiv, unknown, UOp::const_(DType::Float32, ConstValue::Float(0.0))),
        DType::Float32,
    );
    assert_eq!(unknown_over_zero.vmin(), &ConstValue::Float(f64::NEG_INFINITY));
    assert_eq!(unknown_over_zero.vmax(), &ConstValue::Float(f64::INFINITY));
    assert!(compute_sound_vmin_vmax(&unknown_over_zero).is_none());
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

#[test]
fn reduced_float_casts_include_overflow_and_unknown_specials() {
    let near_f16_overflow = UOp::var("wide", DType::Int32, 65_504, 65_520).cast(DType::Float16);
    assert_eq!(
        compute_sound_vmin_vmax(&near_f16_overflow),
        Some((ConstValue::Float(65_504.0), ConstValue::Float(f64::INFINITY)))
    );

    let unknown = unknown_float(DType::Float32);
    for dtype in [DType::FP8E4M3, DType::FP8E5M2, DType::FP8E4M3FNUZ, DType::FP8E5M2FNUZ, DType::BFloat16] {
        let cast = unknown.cast(dtype.clone());
        assert_eq!(cast.vmin(), &ConstValue::Float(f64::NEG_INFINITY), "{dtype:?}");
        assert_eq!(cast.vmax(), &ConstValue::Float(f64::INFINITY), "{dtype:?}");
        assert!(compute_sound_vmin_vmax(&cast).is_none(), "{dtype:?}");
    }
}

fn binary(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>, dtype: DType) -> Arc<UOp> {
    UOp::new(Op::Binary(op, lhs, rhs), dtype)
}

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

#[test]
fn unknown_and_nan_capable_float_comparisons_never_narrow_ordinary_bounds() {
    let ops = [BinaryOp::Eq, BinaryOp::Ne, BinaryOp::Lt, BinaryOp::Le, BinaryOp::Gt, BinaryOp::Ge];
    for dtype in [DType::FP8E4M3, DType::FP8E5M2FNUZ, DType::Float16, DType::BFloat16, DType::Float32] {
        let unknown = unknown_float(dtype.clone());
        let finite = UOp::const_(dtype.clone(), ConstValue::Float(1.0));
        let infinity = UOp::const_(dtype.clone(), ConstValue::Float(f64::INFINITY));
        let negative_infinity = UOp::const_(dtype, ConstValue::Float(f64::NEG_INFINITY));
        for op in ops {
            for comparison in [
                binary(op, unknown.clone(), finite.clone(), DType::Bool),
                binary(op, unknown.clone(), infinity.clone(), DType::Bool),
                binary(op, unknown.clone(), negative_infinity.clone(), DType::Bool),
                binary(op, unknown.clone(), unknown.clone(), DType::Bool),
            ] {
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

// ============================================================================
// Test Edge Cases
// ============================================================================

#[test]
fn test_vmin_vmax_division_by_zero_range() {
    // Test division when divisor range includes zero
    let a = UOp::native_const(10i32);
    let b = UOp::var("b", DType::Int32, 0, 1); // Includes zero!
    let div = a.try_div(&b).unwrap();

    // Division by zero range returns dtype bounds
    assert_eq!(div.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(div.vmax(), &ConstValue::Int(i32::MAX as i64));
}

#[test]
fn test_vmin_vmax_mod_by_zero_range() {
    // Test modulo when divisor range includes zero
    let a = UOp::native_const(10i32);
    let b = UOp::var("b", DType::Int32, 0, 1); // Includes zero!
    let modulo = a.try_mod(&b).unwrap();

    // Modulo by zero range returns dtype bounds
    assert_eq!(modulo.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(modulo.vmax(), &ConstValue::Int(i32::MAX as i64));
}

#[test]
fn test_vmin_vmax_shift_overflow() {
    let a = UOp::native_const(1i32);
    let b = UOp::native_const(64i32); // Shift by 64 or more
    let shl = a.try_shl_op(&b).unwrap();

    // Shift by >= 64 returns dtype bounds
    assert_eq!(shl.vmin(), &ConstValue::Int(i32::MIN as i64));
    assert_eq!(shl.vmax(), &ConstValue::Int(i32::MAX as i64));
}
