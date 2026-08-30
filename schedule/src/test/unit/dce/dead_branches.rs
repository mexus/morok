//! Dead branch elimination in WHERE operations.

use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::types::TernaryOp;
use svod_ir::{Op, UOp};
use test_case::test_case;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

fn undecided_condition() -> Arc<UOp> {
    UOp::var("x", DType::Int32, 0, 100).try_cmplt(&UOp::native_const(50i32)).unwrap()
}

fn range_decided_condition() -> Arc<UOp> {
    UOp::var("x", DType::Int32, 0, 10).try_cmplt(&UOp::native_const(20i32)).unwrap()
}

#[test_case(UOp::native_const(true), true; "constant true")]
#[test_case(UOp::native_const(false), false; "constant false")]
#[test_case(range_decided_condition(), true; "comparison the declared range decides")]
fn decided_condition_selects_its_branch(condition: Arc<UOp>, takes_true_branch: bool) {
    let (then_branch, else_branch) = (UOp::native_const(42i32), UOp::native_const(0i32));
    let where_op = UOp::try_where(condition, then_branch.clone(), else_branch.clone()).unwrap();

    let result = graph_rewrite(symbolic_simple(), where_op, &mut ());

    assert!(Arc::ptr_eq(&result, if takes_true_branch { &then_branch } else { &else_branch }));
}

#[test]
fn undecided_condition_keeps_both_branches() {
    let condition = undecided_condition();
    let (then_branch, else_branch) = (UOp::native_const(1i32), UOp::native_const(0i32));
    let where_op = UOp::try_where(condition.clone(), then_branch.clone(), else_branch.clone()).unwrap();

    let result = graph_rewrite(symbolic_simple(), where_op, &mut ());

    match result.op() {
        Op::Ternary(TernaryOp::Where, cond, then_, else_) => {
            assert!(Arc::ptr_eq(cond, &condition));
            assert!(Arc::ptr_eq(then_, &then_branch));
            assert!(Arc::ptr_eq(else_, &else_branch));
        }
        other => panic!("expected the WHERE to survive, got {other:?}"),
    }
}

/// `WHERE(cond, INVALID, INVALID)` must collapse to `INVALID` instead of
/// ping-ponging through the INVALID canonicalization
/// (`WHERE(c, INV, x) -> WHERE(NOT c, x, INV)`), which flips the gate forever.
#[test_case(false ; "plain condition")]
#[test_case(true ; "negated condition")]
fn where_with_two_invalid_branches_collapses(negate: bool) {
    let condition = undecided_condition();
    let condition = if negate { condition.not() } else { condition };
    let invalid = UOp::invalid_marker();

    let where_op = UOp::try_where(condition, invalid.clone(), invalid).unwrap();
    let result = graph_rewrite(symbolic_simple(), where_op, &mut ());

    assert!(UOp::is_invalid_marker(&result), "expected bare INVALID, got {:?}", result.op());
}
