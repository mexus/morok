//! Pure (GPU-free) tests for the M4 grid / chiplet L2 swizzle.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::uop::eval::{eval_binary_op, eval_ternary_op};
use svod_ir::{ConstValue, Op, UOp};

use crate::grid::{l2_swizzle, l2_swizzle_ref};
use crate::index::cidx;

/// Fold a pure constant `Index`/`Bool` expression tree (the swizzle is `Const`s
/// under `Binary`/`Ternary(Where)` only) to its value.
fn eval(u: &Arc<UOp>) -> ConstValue {
    match u.op() {
        Op::Const(cv) => cv.0,
        Op::Binary(op, a, b) => eval_binary_op(*op, eval(a), eval(b)).expect("fold binary"),
        Op::Ternary(op, a, b, c) => eval_ternary_op(*op, eval(a), eval(b), eval(c)).expect("fold ternary"),
        other => panic!("eval: unexpected op {other:?}"),
    }
}

fn eval_i64(u: &Arc<UOp>) -> i64 {
    match eval(u) {
        ConstValue::Int(i) => i,
        other => panic!("eval_i64: non-int {other:?}"),
    }
}

/// The UOp `l2_swizzle` folds to exactly its pure-`i64` oracle for every wgid,
/// and the oracle is a **bijection** `0..num_wgs → [0,grid_m)×[0,grid_n)` (so
/// the chiplet swizzle is a pure permutation of which workgroup owns which
/// block — the computed C is unchanged).
fn assert_bijection(grid_m: i64, grid_n: i64) {
    let num_wgs = grid_m * grid_n;
    let mut seen = vec![false; (grid_m * grid_n) as usize];
    for w in 0..num_wgs {
        let (rm, rn) = l2_swizzle_ref(w, num_wgs, grid_m, grid_n);
        // UOp path agrees with the oracle.
        let (um, un) = l2_swizzle(cidx(w), num_wgs, grid_m, grid_n);
        assert_eq!((eval_i64(&um), eval_i64(&un)), (rm, rn), "wgid {w}: UOp ≠ ref on {grid_m}×{grid_n}");
        // In range + distinct (bijection).
        assert!((0..grid_m).contains(&rm), "wgid {w} → pid_m {rm} out of range (grid_m {grid_m})");
        assert!((0..grid_n).contains(&rn), "wgid {w} → pid_n {rn} out of range (grid_n {grid_n})");
        let slot = (rm * grid_n + rn) as usize;
        assert!(!seen[slot], "{grid_m}×{grid_n}: collision at ({rm},{rn}) — not a bijection");
        seen[slot] = true;
    }
    assert!(seen.iter().all(|&b| b), "{grid_m}×{grid_n}: not surjective");
}

#[test]
fn test_l2_swizzle_is_bijection() {
    // Matmul squares: N=2048 (8×8, chiplet-identity: 64 < 128 wgs) and N=4096
    // (16×16, full chiplet reorder: 256 wgs = 2 full 128-blocks).
    assert_bijection(8, 8);
    assert_bijection(16, 16);
    // WGM-multiple square that exercises the chiplet remainder (24 not a
    // multiple of the 128-block).
    assert_bijection(24, 24);
    // Rectangular grids (separate grid_m/grid_n; group_size_m clamps the final
    // M-group) — incl. a non-WGM-multiple M side.
    assert_bijection(8, 16);
    assert_bijection(16, 8);
    assert_bijection(6, 4);
    assert_bijection(10, 7);
}

/// The UOp `l2_swizzle` is built from pure `Index` arithmetic + `Where` selects
/// (no control flow, no loads) — so it composes into the matmul's index math.
#[test]
fn test_l2_swizzle_is_pure_index_math() {
    let (m, n) = l2_swizzle(cidx(5), 256, 16, 16);
    for u in m.toposort().into_iter().chain(n.toposort()) {
        assert!(
            matches!(u.op(), Op::Const(_) | Op::Binary(..) | Op::Ternary(..)),
            "l2_swizzle node {:?} is not pure Index/Bool arithmetic",
            u.op()
        );
        let base = u.dtype().base();
        assert!(
            matches!(base, svod_dtype::ScalarDType::Index | svod_dtype::ScalarDType::Bool),
            "l2_swizzle node dtype {:?} is not Index/Bool",
            u.dtype()
        );
    }
    let _ = DType::Index;
}
