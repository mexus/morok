//! Equivalence pin for the pattern matcher's early reject on the production symbolic tiers.
//!
//! An early reject only skips entries whose fixed-position sources demand an op kind the
//! node's children do not have, so those entries could not have matched. Rewriting with the
//! rejects cleared must therefore produce the pointer-identical graph — hash consing makes
//! `Arc::ptr_eq` an exact structural check. The mechanism itself (derivation, `src_ops`,
//! wildcards) is covered by `svod_ir::test::unit::pattern::early_reject`; what is pinned
//! here is that the real tiers stay equivalent under it.
//!
//! Tinygrad equivalent: `if not early_reject.issubset(ler): continue` (uop/ops.py:1482).

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::op::pattern_derived::OpKey;
use svod_ir::pattern::TypedPatternMatcher;
use svod_ir::{BinaryOp, ConstValue, Op, UOp, UnaryOp};
use test_case::test_case;

use crate::rewrite::graph_rewrite;
use crate::symbolic::{symbolic, symbolic_simple};

fn idx(value: i64) -> Arc<UOp> {
    UOp::index_const(value)
}

fn bin(op: BinaryOp, lhs: &Arc<UOp>, rhs: &Arc<UOp>) -> Arc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(op, lhs.clone(), rhs.clone()), dtype)
}

/// A mixed pool of arithmetic, comparison, select, cast and gated-index shapes over the
/// leaves the symbolic tiers reason about, built so each rule family sees both matching
/// and non-matching nodes.
fn graphs() -> Vec<Arc<UOp>> {
    let leaves = vec![
        UOp::range(idx(64), 0),
        UOp::range(idx(16), 1),
        UOp::special(idx(32), "gidx0".to_string()),
        UOp::var("n", DType::WeakInt, 1, 128),
        idx(0),
        idx(1),
        idx(4),
        idx(-3),
    ];
    let ops = [
        BinaryOp::Add,
        BinaryOp::Mul,
        BinaryOp::Sub,
        BinaryOp::Max,
        BinaryOp::FloorDiv,
        BinaryOp::FloorMod,
        BinaryOp::Lt,
        BinaryOp::Ne,
        BinaryOp::And,
        BinaryOp::Or,
    ];

    let mut pool: Vec<Arc<UOp>> = leaves.clone();
    for (i, lhs) in leaves.iter().enumerate() {
        for (j, rhs) in leaves.iter().enumerate() {
            let op = ops[(i * leaves.len() + j) % ops.len()];
            // Division by a literal zero is not a legal node to build.
            if matches!(op, BinaryOp::FloorDiv | BinaryOp::FloorMod)
                && matches!(rhs.op(), Op::Const(c) if c.0 == ConstValue::Int(0))
            {
                continue;
            }
            pool.push(bin(op, lhs, rhs));
        }
    }

    let (bools, values): (Vec<Arc<UOp>>, Vec<Arc<UOp>>) = pool.iter().cloned().partition(|u| u.dtype() == DType::Bool);
    let mut nested = Vec::new();
    for (i, value) in values.iter().enumerate() {
        nested.push(bin(BinaryOp::Add, value, &values[(i + 1) % values.len()]));
        nested.push(bin(BinaryOp::Mul, value, &values[(i + 3) % values.len()]));
        nested.push(UOp::new(Op::Unary(UnaryOp::Neg, value.clone()), value.dtype()));
        nested.push(value.cast(DType::Int32));
        if let Some(cond) = bools.get(i % bools.len().max(1)) {
            nested.push(
                UOp::try_where(cond.clone(), value.clone(), values[(i + 2) % values.len()].clone())
                    .expect("where over matching branches"),
            );
        }
    }
    pool.extend(nested);
    pool
}

#[test_case(symbolic(), "symbolic"; "symbolic tier")]
#[test_case(symbolic_simple(), "symbolic_simple"; "symbolic simple tier")]
fn early_reject_preserves_symbolic_rewrites(matcher: &TypedPatternMatcher<()>, label: &str) {
    // Without a non-trivial requirement somewhere the equivalence below would be vacuous.
    let requirements = [OpKey::Binary(BinaryOp::Add), OpKey::Binary(BinaryOp::Mul), OpKey::Binary(BinaryOp::FloorMod)]
        .iter()
        .flat_map(|key| matcher.early_rejects(key))
        .filter(|reject| !reject.is_empty())
        .count();
    assert!(requirements > 0, "{label} derived no early rejects");

    let permissive = matcher.without_early_reject();
    for graph in graphs() {
        let with_reject = graph_rewrite(matcher, graph.clone(), &mut ());
        let without_reject = graph_rewrite(&permissive, graph.clone(), &mut ());
        assert!(Arc::ptr_eq(&with_reject, &without_reject), "{label} diverged on {:?}", graph.op());
    }
}
