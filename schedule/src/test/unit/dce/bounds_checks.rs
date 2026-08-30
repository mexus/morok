//! Range-driven folding of the comparisons that guard buffer accesses.

use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::{Op, UOp};
use test_case::test_case;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic;

/// `idx < size` folds exactly when the declared range of `idx` already decides it.
#[test_case(15, 32, Some(true); "range entirely below the bound")]
#[test_case(100, 0, Some(false); "range entirely above the bound")]
#[test_case(100, 50, None; "range straddles the bound")]
fn bounds_check_folds_only_when_the_range_decides_it(idx_max: i64, size: i32, decided: Option<bool>) {
    let idx = UOp::var("idx", DType::Int32, 0, idx_max);
    let size = UOp::native_const(size);
    let result = graph_rewrite(symbolic(), idx.try_cmplt(&size).unwrap(), &mut ());

    match decided {
        Some(value) => assert!(
            matches!(result.op(), Op::Const(c) if c.0 == ConstValue::Bool(value)),
            "expected Const({value}), got {:?}",
            result.op()
        ),
        None => match result.op() {
            Op::Binary(BinaryOp::Lt, lhs, rhs) => assert!(Arc::ptr_eq(lhs, &idx) && Arc::ptr_eq(rhs, &size)),
            other => panic!("undecided comparison must survive, got {other:?}"),
        },
    }
}
