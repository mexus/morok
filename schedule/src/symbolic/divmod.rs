//! Conservative div/mod congruence folding.
//!
//! Candidate construction uses mathematical integer algebra, but callers must
//! reject it unless both the original and replacement arithmetic trees are
//! proven not to wrap under their concrete dtype.

use std::sync::Arc;

use svod_ir::UOp;
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::SoundVminVmaxProperty;

pub(crate) fn uop_sum(terms: &[Arc<UOp>], template: &Arc<UOp>) -> Arc<UOp> {
    terms.iter().cloned().reduce(|sum, term| sum.add(&term)).unwrap_or_else(|| template.const_like(0i64))
}

fn scaled(term: &Arc<UOp>, coefficient: i64) -> Option<Arc<UOp>> {
    match coefficient {
        0 => Some(term.const_like(0i64)),
        1 => Some(term.clone()),
        _ => term.try_mul(&term.const_like(coefficient)).ok(),
    }
}

fn try_uop_sum(terms: &[Arc<UOp>], template: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut sum: Option<Arc<UOp>> = None;
    for term in terms {
        sum = Some(if let Some(sum) = sum { sum.try_add(term).ok()? } else { term.clone() });
    }
    Some(sum.unwrap_or_else(|| template.const_like(0i64)))
}

/// Fold an affine numerator modulo a positive constant divisor.
///
/// For `x = sum(f_i*t_i) + k`, choose centered `r_i = f_i (mod c)` and
/// construct `rem = sum(r_i*t_i) + (k mod c)`. If `rem` remains in one
/// Euclidean quotient bucket, then:
///
/// * `x % c = rem - floor(rem/c)*c`
/// * `x // c = sum((f_i-r_i)/c*t_i) + (k-k%c+floor(rem/c)*c)/c`
///
/// This function only constructs the candidate. `exact_integer_rewrite` is the
/// mandatory typed no-wrap proof at the pattern call site.
pub fn fold_divmod_congruence(x: &Arc<UOp>, _c_uop: &Arc<UOp>, c_val: ConstValue, is_mod: bool) -> Option<Arc<UOp>> {
    // Hardware vectors need lane-wise constants and scaling. Keep this
    // indexing rewrite scalar rather than constructing a partial candidate.
    if x.dtype().vcount() != 1 {
        return None;
    }
    let ConstValue::Int(c) = c_val else { return None };
    if c <= 0 {
        return None;
    }
    let c128 = c as i128;

    // Keep this rule in the non-negative indexing domain. This is sufficient
    // for QR and avoids adding negative-domain normalization rules.
    let (ConstValue::Int(x_min), ConstValue::Int(_)) = SoundVminVmaxProperty::get(x).as_ref()? else { return None };
    if *x_min < 0 {
        return None;
    }

    let (without_const, constant) = x.pop_const(BinaryOp::Add);
    let ConstValue::Int(constant) = constant else { return None };
    let terms = without_const.split_uop(BinaryOp::Add);
    let decomposition: Option<Vec<_>> = terms
        .iter()
        .map(|term| {
            let factor = term.const_factor();
            (factor != 0).then(|| term.divides(factor)).flatten().map(|base| (base, factor))
        })
        .collect();
    let decomposition = decomposition?;

    let remainders: Option<Vec<i64>> = decomposition
        .iter()
        .map(|(_, factor)| {
            let positive = (*factor as i128).rem_euclid(c128);
            let negative = positive.checked_sub(c128)?;
            i64::try_from(if negative.unsigned_abs() < positive.unsigned_abs() { negative } else { positive }).ok()
        })
        .collect();
    let remainders = remainders?;
    let constant_remainder = constant.rem_euclid(c);

    let mut remainder_terms = Vec::new();
    for ((base, _), coefficient) in decomposition.iter().zip(&remainders) {
        if *coefficient != 0 {
            remainder_terms.push(scaled(base, *coefficient)?);
        }
    }
    if constant_remainder != 0 {
        remainder_terms.push(x.const_like(constant_remainder));
    }
    let remainder = try_uop_sum(&remainder_terms, x)?;
    let (ConstValue::Int(rem_min), ConstValue::Int(rem_max)) = SoundVminVmaxProperty::get(&remainder).as_ref()? else {
        return None;
    };
    let quotient_bucket = rem_min.div_euclid(c);
    if quotient_bucket != rem_max.div_euclid(c) {
        return None;
    }

    if is_mod {
        let offset = (quotient_bucket as i128).checked_mul(c128)?;
        return if offset == 0 {
            Some(remainder)
        } else {
            remainder.try_sub(&x.const_like(i64::try_from(offset).ok()?)).ok()
        };
    }

    let mut quotient_terms = Vec::new();
    for ((base, factor), remainder) in decomposition.iter().zip(&remainders) {
        let coefficient = (*factor as i128).checked_sub(*remainder as i128)?.checked_div(c128)?;
        if coefficient != 0 {
            quotient_terms.push(scaled(base, i64::try_from(coefficient).ok()?)?);
        }
    }
    let bucket_offset = (quotient_bucket as i128).checked_mul(c128)?;
    let constant_quotient =
        (constant as i128).checked_sub(constant_remainder as i128)?.checked_add(bucket_offset)?.checked_div(c128)?;
    if constant_quotient != 0 {
        quotient_terms.push(x.const_like(i64::try_from(constant_quotient).ok()?));
    }
    try_uop_sum(&quotient_terms, x)
}

/// Tinygrad's variable-denominator fallback (`uop/divandmod.py:76-96`).
///
/// Covers `divide_by_gcd` and `factor_remainder`, which is what folds
/// `(N*i + j) // N` for a symbolic `N`. Constant divisors are deliberately
/// excluded: their folds live in `advanced_division_dsl_patterns`, and the
/// constant-denominator half of upstream's `fold_divmod_general`
/// (`nested_div`, `remove_nested_mod`, `gcd_with_remainder`, `nest_by_factor`)
/// is not ported yet.
///
/// Like `fold_divmod_congruence` this only constructs a candidate; the caller
/// must still run the typed no-wrap proof.
pub(crate) fn fold_divmod_general(op: BinaryOp, x: &Arc<UOp>, y: &Arc<UOp>) -> Option<Arc<UOp>> {
    if matches!(y.op(), svod_ir::Op::Const(_)) || x.dtype().vcount() != 1 {
        return None;
    }
    let is_mod = op == BinaryOp::FloorMod;

    // The quotient is a single bucket, so the whole division is constant.
    let quotient = x.try_div(y).ok()?;
    let (ConstValue::Int(q_min), ConstValue::Int(q_max)) = SoundVminVmaxProperty::get(&quotient).as_ref()? else {
        return None;
    };
    if q_min == q_max {
        return if is_mod { x.try_sub(&scaled(y, *q_min)?).ok() } else { Some(x.const_like(*q_min)) };
    }

    let terms = x.split_uop(BinaryOp::Add);

    // divide_by_gcd: x op y -> (x/g) op (y/g), rescaled by g for the remainder.
    let mut with_divisor = terms.clone();
    with_divisor.push(y.clone());
    let divisor_gcd = UOp::symbolic_gcd(&with_divisor);
    if !matches!(divisor_gcd.op(), svod_ir::Op::Const(value) if value.0 == ConstValue::Int(1)) {
        let folded = binary(op, &x.divide_exact(&divisor_gcd)?, &y.divide_exact(&divisor_gcd)?)?;
        return if is_mod { folded.try_mul(&divisor_gcd).ok() } else { Some(folded) };
    }

    // factor_remainder: (y*a + b) op y -> a + b//y / b%y, in the non-negative domain.
    if non_negative(x).is_none() || non_negative(y).is_none() {
        return None;
    }
    let (quotient, remainder): (Vec<_>, Vec<_>) =
        terms.iter().map(|term| (term.divide_exact(y), term)).fold((vec![], vec![]), |mut acc, (q, term)| {
            match q {
                Some(q) => acc.0.push(q),
                None => acc.1.push(term.clone()),
            }
            acc
        });
    if quotient.is_empty() {
        return None;
    }
    let new_x = try_uop_sum(&remainder, x)?;
    non_negative(&new_x)?;
    let folded = binary(op, &new_x, y)?;
    if is_mod { Some(folded) } else { folded.try_add(&try_uop_sum(&quotient, x)?).ok() }
}

fn binary(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Option<Arc<UOp>> {
    match op {
        BinaryOp::FloorMod => lhs.try_mod(rhs).ok(),
        BinaryOp::FloorDiv => lhs.try_div(rhs).ok(),
        _ => None,
    }
}

fn non_negative(u: &Arc<UOp>) -> Option<()> {
    let (ConstValue::Int(vmin), _) = SoundVminVmaxProperty::get(u).as_ref()? else { return None };
    (*vmin >= 0).then_some(())
}
