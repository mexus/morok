use super::*;

#[test]
fn test_magic_unsigned_div_3() {
    // x / 3 for x in 0..=100
    let result = magic_unsigned(100, 3);
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    // Verify for some values
    for x in 0..=100 {
        let expected = x / 3;
        let actual = ((x as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_div_7() {
    // x / 7 for x in 0..=1000
    let result = magic_unsigned(1000, 7);
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in 0..=1000 {
        let expected = x / 7;
        let actual = ((x as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_div_10() {
    // x / 10 for x in 0..=10000
    let result = magic_unsigned(10000, 10);
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in (0..=10000).step_by(100) {
        let expected = x / 10;
        let actual = ((x as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_invalid() {
    // Zero divisor
    assert!(magic_unsigned(100, 0).is_none());

    // Negative divisor
    assert!(magic_unsigned(100, -5).is_none());
}

#[test]
fn test_magic_unsigned_div_6_factorization() {
    // x / 6 for x in 0..=1000
    // Tests power-of-two factorization: 6 = 2 * 3
    // Division by 6 should become: (x >> 1) / 3
    let result = magic_unsigned(500, 3); // After shift, max is 500
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in 0..=1000 {
        let expected = x / 6;
        // Simulate factorization: (x >> 1) then magic divide by 3
        let shifted = x >> 1;
        let actual = ((shifted as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn test_magic_unsigned_div_12_factorization() {
    // x / 12 for x in 0..=1200
    // Tests power-of-two factorization: 12 = 4 * 3
    // Division by 12 should become: (x >> 2) / 3
    let result = magic_unsigned(300, 3); // After shift by 2, max is 300
    assert!(result.is_some());
    let (m, s) = result.unwrap();

    for x in 0..=1200 {
        let expected = x / 12;
        // Simulate factorization: (x >> 2) then magic divide by 3
        let shifted = x >> 2;
        let actual = ((shifted as i128 * m as i128) >> s) as i64;
        assert_eq!(expected, actual, "Failed for x = {}", x);
    }
}

#[test]
fn fast_division_does_not_rewrite_signed_negative_range() {
    let x = UOp::variable("x".into(), -100, 100, svod_ir::DType::Int32);
    let divisor = UOp::const_(svod_ir::DType::Int32, ConstValue::Int(7));
    let div = x.cdiv(&divisor);

    assert!(matches!(
        fast_division_patterns(std::collections::HashSet::new()).rewrite(&div, &mut ()),
        svod_ir::RewriteResult::NoMatch
    ));
}

#[test]
fn fast_division_replacements_are_exhaustive_for_eight_bit_ranges() {
    use std::collections::{HashMap, HashSet};
    use svod_ir::{Op, UOpKey};

    for (dtype, vmax, wider) in [
        (svod_ir::DType::UInt8, u8::MAX as i64, ScalarDType::UInt16),
        (svod_ir::DType::Int8, i8::MAX as i64, ScalarDType::Int16),
    ] {
        let variable = UOp::variable("x".into(), 0, vmax, dtype.clone());
        let supported = HashSet::from([dtype.base(), wider]);
        for divisor in (2..=vmax).filter(|divisor| !(*divisor as u64).is_power_of_two()) {
            let Some(replacement) = fast_idiv(&variable, divisor, false, &supported) else { continue };
            for value in 0..=vmax {
                let substituted = replacement.substitute(&HashMap::from([(
                    UOpKey(variable.clone()),
                    UOp::const_(dtype.clone(), ConstValue::Int(value)),
                )]));
                let folded = svod_ir::rewrite::graph_rewrite(
                    &(crate::symbolic::symbolic() + crate::symbolic::pm_fold_cast_const()),
                    substituted,
                    &mut (),
                );
                let Op::Const(actual) = folded.op() else {
                    panic!("replacement did not fold for {dtype:?} {value}/{divisor}: {}", folded.tree())
                };
                assert_eq!(actual.0.try_int(), Some(value / divisor), "{dtype:?} {value}/{divisor}");
            }
        }
    }
}
