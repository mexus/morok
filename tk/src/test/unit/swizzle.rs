//! Pure tests for ST swizzles.

use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::uop::eval::eval_binary_op;
use svod_ir::{ConstValue, Op, UOp};

use crate::swizzle::Swizzle;
use crate::tiles::ST_16X16;

/// Fold a pure constant `Index` expression tree (the swizzle is `Const`s under
/// `Binary` ops only) to its `i64` value.
fn eval_const(u: &Arc<UOp>) -> i64 {
    match u.op() {
        Op::Const(cv) => match cv.0 {
            ConstValue::Int(i) => i,
            other => panic!("eval_const: non-int const {other:?}"),
        },
        Op::Binary(op, a, b) => {
            let (av, bv) = (eval_const(a), eval_const(b));
            match eval_binary_op(*op, ConstValue::Int(av), ConstValue::Int(bv)) {
                Some(ConstValue::Int(r)) => r,
                other => panic!("eval_const: {op:?}({av},{bv}) folded to {other:?}"),
            }
        }
        other => panic!("eval_const: unexpected op {other:?}"),
    }
}

/// A swizzle must be a bijection over `[0,rows)×[0,cols)` (else LDS round-trips
/// corrupt), and must not move an element out of its base fragment.
fn assert_bijection(sw: Swizzle, rows: usize, cols: usize, scalar: ScalarDType) {
    let cidx = |v: usize| UOp::index_const(v as i64);
    let mut seen = vec![false; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let (srow, scol) = sw.swizzle_rc(cidx(r), cidx(c), cols, scalar);
            let (sr, sc) = (eval_const(&srow), eval_const(&scol));
            assert!((0..rows as i64).contains(&sr), "{sw:?}: row {r},{c} -> srow {sr} out of range");
            assert!((0..cols as i64).contains(&sc), "{sw:?}: row {r},{c} -> scol {sc} out of range");
            let slot = (sr as usize) * cols + sc as usize;
            assert!(!seen[slot], "{sw:?}: collision at ({sr},{sc}) — not a bijection");
            seen[slot] = true;
        }
    }
    assert!(seen.iter().all(|&b| b), "{sw:?}: not surjective");
}

/// The 16 contiguous columns a single `ds_read` row gathers (`r` fixed) must map
/// to 16 distinct LDS slots — the bank-conflict-free property the XOR buys.
#[test]
fn test_sw16x16_row_distinct_banks() {
    let cidx = |v: usize| UOp::index_const(v as i64);
    for r in 0..16usize {
        let mut cols_seen = std::collections::HashSet::new();
        for c in 0..16usize {
            let (srow, scol) = Swizzle::Sw16x16.swizzle_rc(cidx(r), cidx(c), 16, ScalarDType::BFloat16);
            assert_eq!(eval_const(&srow), r as i64, "Sw16x16 must keep the row");
            cols_seen.insert(eval_const(&scol));
        }
        assert_eq!(cols_seen.len(), 16, "row {r}: 16 cols must map to 16 distinct slots");
    }
}

#[test]
fn test_swizzle_is_bijection() {
    assert_bijection(Swizzle::Sw16x16, 16, 16, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw32x32, 32, 32, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw16x32, 16, 32, ScalarDType::BFloat16);
    assert_bijection(Swizzle::Sw32x16, 32, 16, ScalarDType::BFloat16);
}

#[test]
fn test_identity_swizzle_passthrough() {
    let row = UOp::const_(DType::Int32, ConstValue::Int(3));
    let col = UOp::const_(DType::Int32, ConstValue::Int(5));
    let (srow, scol) = ST_16X16.swizzle.swizzle_rc(row.clone(), col.clone(), 16, ScalarDType::Float32);
    assert!(Arc::ptr_eq(&srow, &row), "identity swizzle must return row unchanged");
    assert!(Arc::ptr_eq(&scol, &col), "identity swizzle must return col unchanged");
}

#[test]
fn test_base_shape_arithmetic() {
    use crate::tiles::{RT_16X16, RT_32X32};
    // 16x16 over wave64 -> 4 elements/thread; RT stride 4 -> 1 stride-group.
    assert_eq!(ST_16X16.base.elements_per_thread(), 4);
    assert_eq!(RT_16X16.elements_per_thread(), 4);
    assert_eq!(RT_16X16.num_strides(), 1);
    // 32x32 over wave64 -> 16 elements/thread; stride 4 -> 4 stride-groups.
    assert_eq!(RT_32X32.elements_per_thread(), 16);
    assert_eq!(RT_32X32.num_strides(), 4);
}
