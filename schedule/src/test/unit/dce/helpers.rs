use std::sync::Arc;
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use crate::TypedPatternMatcher;
use crate::symbolic::{pm_fold_cast_const, symbolic_simple};

pub fn get_matcher() -> &'static TypedPatternMatcher {
    static MATCHER: std::sync::LazyLock<TypedPatternMatcher> =
        std::sync::LazyLock::new(|| symbolic_simple() + pm_fold_cast_const());
    &MATCHER
}

/// # Panics
/// Panics if `uop` is not a `Const` holding `expected`.
pub fn assert_const_value(uop: &Arc<UOp>, expected: ConstValue) {
    match uop.op() {
        Op::Const(value) => assert_eq!(value.0, expected),
        other => panic!("expected Const({expected:?}), got {other:?}"),
    }
}
