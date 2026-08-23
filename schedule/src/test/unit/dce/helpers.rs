//! Test helpers for DCE tests.

use std::sync::Arc;
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use crate::TypedPatternMatcher;
use crate::symbolic::{pm_fold_cast_const, symbolic_simple};

/// Get the symbolic_simple pattern matcher (reduces duplication).
pub fn get_matcher() -> &'static TypedPatternMatcher {
    static MATCHER: std::sync::LazyLock<TypedPatternMatcher> =
        std::sync::LazyLock::new(|| symbolic_simple() + pm_fold_cast_const());
    &MATCHER
}

/// Assert that a UOp transforms to a specific constant value.
///
/// # Panics
/// Panics if the UOp is not a Const or doesn't match the expected value.
pub fn assert_const_value(uop: &Arc<UOp>, expected: ConstValue) {
    match uop.op() {
        Op::Const(cv) => {
            assert_eq!(cv.0, expected, "Expected Const({:?}), got Const({:?})", expected, cv.0);
        }
        other => panic!("Expected Const({:?}), got {:?}", expected, other),
    }
}
