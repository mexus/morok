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

#[allow(dead_code)]
pub(crate) fn fold_divmod_general(_op: BinaryOp, _x: &Arc<UOp>, _y: &Arc<UOp>) -> Option<Arc<UOp>> {
    None
}
