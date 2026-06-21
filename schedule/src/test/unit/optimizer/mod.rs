pub mod heuristics;
pub mod opts_to_apply;
pub mod opts_validation;
pub mod scheduler;
pub mod tc;

#[cfg(test)]
mod pipeline_composition {
    use crate::linearize::pm_split_ends;
    use crate::rewrite::graph_rewrite;
    use smallvec::smallvec;
    use svod_ir::{AxisId, AxisType, ConstValue, DType, Op, UOp};

    /// `pm_split_ends` (composed into `PM_FINAL` at
    /// `optimizer/mod.rs:391-393`) splits a multi-range END into nested
    /// single-range ENDs. Behavioral assertion against the helper so a
    /// `split_end` regression surfaces here before reaching the pipeline.
    #[test]
    fn test_pm_split_ends_unfolds_multi_range_end() {
        // Two Range ops with distinct axis ids.
        let end_a = UOp::const_(DType::Index, ConstValue::Int(4));
        let range_a = UOp::range_axis(end_a, AxisId::Renumbered(0), AxisType::Reduce);
        let end_b = UOp::const_(DType::Index, ConstValue::Int(8));
        let range_b = UOp::range_axis(end_b, AxisId::Renumbered(1), AxisType::Reduce);

        // Multi-range END wrapping a noop computation.
        let computation = UOp::noop();
        let multi_end = computation.end(smallvec![range_a, range_b]);

        let result = graph_rewrite(pm_split_ends(), multi_end, &mut ());

        // Expected: outer END wraps an inner END (nested single-range).
        let Op::End { computation: inner, ranges, .. } = result.op() else {
            panic!("Expected outer END after split, got {:?}", result.op());
        };
        assert_eq!(ranges.len(), 1, "outer END must have exactly one range, got {}", ranges.len());
        assert!(
            matches!(inner.op(), Op::End { ranges: ir, .. } if ir.len() == 1),
            "inner computation must be a single-range END, got {:?}",
            inner.op()
        );
    }
}
