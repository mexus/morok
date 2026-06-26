//! Cheap, symbolic cost estimates over a kernel AST.
//!
//! These walk the UOp graph (no execution) to approximate a kernel's compute
//! work. Used by the BEAM `least_compute_ops` bloat filter and by the runtime
//! profiler's roofline (GFLOP/s) column.

use std::collections::HashMap;
use std::sync::Arc;

use crate::op::Op;
use crate::uop::UOp;

/// Symbolic estimate of compute ops in a kernel.
///
/// Each ALU/Ternary/Reduce/WMMA node contributes `prod(enclosing-RANGE sizes)`
/// flops. Symbolic RANGE ends resolve to the midpoint of their `vmin`/`vmax`
/// bounds (matching the `(vmax+vmin)/2` choice in the BEAM timing path), so
/// dynamic-shape kernels participate in the `least_compute_ops*1000` bloat
/// filter.
pub fn compute_ops_estimate(uop: &Arc<UOp>) -> u64 {
    let topo = uop.toposort();

    // Pre-compute the size contribution of every loop-bound node — RANGE for
    // ordinary loops, SPECIAL for hardware-provided indices.
    let mut range_size: HashMap<u64, u64> = HashMap::new();
    for node in &topo {
        let end = match node.op() {
            Op::Range { end, .. } => Some(end),
            Op::Special { end, .. } => Some(end),
            _ => None,
        };
        if let Some(end) = end {
            range_size.insert(node.id, range_size_estimate(end));
        }
    }

    // Each ALU/Reduce/WMMA accumulates `prod(in-scope range sizes)`. Backward
    // slice membership tells us which RANGEs the node sits inside, mirroring
    // tinygrad's `mult_stack` discipline structurally.
    let mut flops: u64 = 0;
    for node in &topo {
        let is_alu =
            matches!(node.op(), Op::Binary(..) | Op::Unary(..) | Op::Ternary(..) | Op::Reduce { .. } | Op::Wmma { .. });
        if !is_alu {
            continue;
        }
        let bws = node.backward_slice_ids();
        let mut weight: u64 = 1;
        for (rid, sz) in &range_size {
            if bws.contains(rid) {
                weight = weight.saturating_mul(*sz);
            }
        }
        flops = flops.saturating_add(weight);
    }
    flops
}

/// Estimate a RANGE end's iteration count.
///
/// Concrete `Const(Int)` ends use the value directly; everything else falls
/// back to the midpoint of the `end` UOp's symbolic `vmin`/`vmax` bounds, so
/// dynamic-shape ranges still contribute a representative number of flops.
fn range_size_estimate(end: &Arc<UOp>) -> u64 {
    if let Op::Const(cv) = end.op()
        && let Some(v) = cv.0.try_int()
    {
        return (v.max(1)) as u64;
    }
    let vmin = end.vmin().try_int().unwrap_or(1);
    let vmax = end.vmax().try_int().unwrap_or(vmin);
    (((vmin + vmax) / 2).max(1)) as u64
}
