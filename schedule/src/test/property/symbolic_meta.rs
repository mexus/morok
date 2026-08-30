//! Structural invariants that every `symbolic_simple` rewrite must hold, whatever it
//! rewrites: a fixpoint, dtype-preserving, acyclic, and not meaningfully larger.

use std::sync::Arc;

use proptest::prelude::*;

use svod_dtype::DType;
use svod_ir::{Op, UOp};

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;

use svod_ir::test::property::generators::*;
use svod_ir::test::property::shrinking::{uop_depth, uop_op_count};

/// Every constant must stay in the same type family as the node holding it.
fn prop_assert_constant_dtypes(uop: &Arc<UOp>) -> Result<(), TestCaseError> {
    match uop.op() {
        Op::Const(value) => {
            let held = value.0.dtype();
            if let Some(scalar) = uop.dtype().scalar() {
                let is_int = |dtype: &DType| matches!(dtype.scalar(), Some(scalar) if scalar.is_int());
                prop_assert_eq!(
                    is_int(&held),
                    is_int(&DType::Scalar(scalar)),
                    "constant dtype family mismatch: {:?} in a {:?} node",
                    held,
                    uop.dtype()
                );
            }
            Ok(())
        }
        Op::Unary(_, src) => prop_assert_constant_dtypes(src),
        Op::Binary(_, lhs, rhs) => {
            prop_assert_constant_dtypes(lhs)?;
            prop_assert_constant_dtypes(rhs)
        }
        Op::Ternary(_, a, b, c) => {
            prop_assert_constant_dtypes(a)?;
            prop_assert_constant_dtypes(b)?;
            prop_assert_constant_dtypes(c)
        }
        _ => Ok(()),
    }
}

fn rewrite_twice(graph: Arc<UOp>) -> (Arc<UOp>, Arc<UOp>) {
    let once = graph_rewrite(symbolic_simple(), graph, &mut ());
    let twice = graph_rewrite(symbolic_simple(), once.clone(), &mut ());
    (once, twice)
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    /// Distribution rules may add a couple of nodes before a later fold removes them again,
    /// hence the slack on the size bounds; the point is to catch unbounded growth.
    #[test]
    fn rewriting_is_a_structure_preserving_fixpoint(graph in arb_arithmetic_tree_up_to(DType::Int32, 4)) {
        let (ops, depth, dtype) = (uop_op_count(&graph), uop_depth(&graph), graph.dtype());

        let (once, twice) = rewrite_twice(graph);

        prop_assert!(Arc::ptr_eq(&once, &twice), "rewriting twice must equal rewriting once");
        prop_assert_eq!(once.dtype(), dtype, "rewriting must preserve the dtype");
        prop_assert!(uop_op_count(&once) <= ops + 2, "op count grew {} -> {}", ops, uop_op_count(&once));
        prop_assert!(uop_depth(&once) <= depth + 1, "depth grew {} -> {}", depth, uop_depth(&once));
        prop_assert_constant_dtypes(&once)?;
        // Panics rather than looping if a rewrite introduced a cycle.
        once.toposort();
    }

    /// The same fixpoint over graphs built from known algebraic identities, which exercise
    /// far more rules than the random arithmetic trees reach.
    #[test]
    fn known_property_graphs_rewrite_to_a_fixpoint(kpg in arb_known_property_graph()) {
        let (once, twice) = rewrite_twice(kpg.build());
        prop_assert!(Arc::ptr_eq(&once, &twice));
    }
}
