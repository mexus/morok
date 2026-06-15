//! Tests for the `KernelInfo.opts_to_apply` optimization-control mechanism
//! (the Svod port of tinygrad's `opts_to_apply`).

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisType, ConstValue, KernelInfo, Op, Opt, UOp};

use crate::optimizer::config::{OptStrategy, OptimizerConfig};
use crate::optimizer::{Renderer, optimize_kernel_with_config};

/// Build a hand-ranged `out[i] = in[i] + 1` SINK over PARAM buffers, marked
/// with the given `opts_to_apply`. Mirrors a `Tensor::custom_kernel` body.
fn hand_ranged_sink(n: i64, opts_to_apply: Option<Vec<Opt>>) -> std::sync::Arc<UOp> {
    let out_buf = UOp::param(0, n as usize, DType::Float32.ptr(Some(n as usize), AddrSpace::Global), None);
    let in_buf = UOp::param(1, n as usize, DType::Float32.ptr(Some(n as usize), AddrSpace::Global), None);
    let i = UOp::range_const(n, 0);
    let in_idx = UOp::index().buffer(in_buf.clone()).indices(vec![i.clone()]).ptr(true).call().unwrap();
    let loaded = UOp::load().buffer(in_buf).index(in_idx).call();
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let val = loaded.try_add(&one).unwrap();
    let out_idx = UOp::index().buffer(out_buf).indices(vec![i.clone()]).ptr(true).call().unwrap();
    let store = out_idx.store(val).end(smallvec![i]);
    UOp::sink_with_info(vec![store], KernelInfo { opts_to_apply, name: None })
}

fn count_axis_type(ast: &std::sync::Arc<UOp>, axis_type: AxisType) -> usize {
    ast.toposort().iter().filter(|u| matches!(u.op(), Op::Range { axis_type: at, .. } if *at == axis_type)).count()
}

/// `opts_to_apply = Some(vec![])` (the tinygrad `()` analog): the optimizer
/// must apply ZERO opts — no heuristic default-upcast — so the manual Loop
/// range survives and no Upcast axis is introduced.
#[test]
fn test_opts_to_apply_empty_skips_heuristic_upcast() {
    let sink = hand_ranged_sink(8, Some(vec![]));
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let optimized = optimize_kernel_with_config(sink, &Renderer::cpu(), &config);

    assert_eq!(count_axis_type(&optimized, AxisType::Upcast), 0, "opts_to_apply=() must not introduce an Upcast axis");
    assert!(count_axis_type(&optimized, AxisType::Loop) >= 1, "the manual Loop range must survive untouched");
}

/// The same `Beam` strategy is overridden by an explicit (empty) opt list at
/// the `optimize_kernel_with_config` level — no heuristic/beam upcast.
#[test]
fn test_opts_to_apply_empty_overrides_beam_strategy() {
    let sink = hand_ranged_sink(8, Some(vec![]));
    let config = OptimizerConfig { strategy: OptStrategy::Beam { width: 1 }, ..Default::default() };
    let optimized = optimize_kernel_with_config(sink, &Renderer::cpu(), &config);

    assert_eq!(count_axis_type(&optimized, AxisType::Upcast), 0, "explicit opts must win over the beam strategy");
}

/// Control: with `opts_to_apply = None` the heuristic optimizer is free to
/// upcast the divisible Loop axis (DEFAULT_UPCAST_FACTOR = 4), which post-opt
/// lowers into vectorized (vcount > 1) load/store nodes. Proves the skip above
/// is the marker's effect, not an artifact of an un-upcastable kernel.
#[test]
fn test_opts_to_apply_none_allows_heuristic_upcast() {
    let sink = hand_ranged_sink(8, None);
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let optimized = optimize_kernel_with_config(sink, &Renderer::cpu(), &config);

    let vectorized = optimized.toposort().iter().any(|u| u.dtype().vcount() > 1);
    assert!(vectorized, "with opts_to_apply=None the heuristic upcaster should vectorize the divisible loop");

    // And the empty-opts variant of the same kernel stays scalar.
    let scalar_sink = hand_ranged_sink(8, Some(vec![]));
    let scalar = optimize_kernel_with_config(scalar_sink, &Renderer::cpu(), &config);
    assert!(
        scalar.toposort().iter().all(|u| u.dtype().vcount() == 1),
        "opts_to_apply=() must leave the kernel scalar (no vectorization)"
    );
}
