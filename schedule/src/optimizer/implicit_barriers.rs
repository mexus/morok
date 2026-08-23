use std::collections::HashSet;
use std::sync::{Arc, LazyLock};

use smallvec::{SmallVec, smallvec};
use svod_dtype::AddrSpace;
use svod_ir::{AxisType, ConstValue, Op, TypedPatternMatcher, UOp, UOpKey};

fn access_buffer(uop: &Arc<UOp>) -> Option<Arc<UOp>> {
    match uop.op() {
        Op::Param { .. } | Op::Buffer { .. } | Op::MSelect { .. } | Op::MStack { .. } => Some(uop.clone()),
        Op::Index { buffer, .. } | Op::After { passthrough: buffer, .. } => access_buffer(buffer),
        Op::Cast { src, .. }
        | Op::Reshape { src, .. }
        | Op::Permute { src, .. }
        | Op::Expand { src, .. }
        | Op::Pad { src, .. }
        | Op::Shrink { src, .. }
        | Op::Flip { src, .. } => access_buffer(src),
        _ => None,
    }
}

fn is_local_store(uop: &Arc<UOp>) -> bool {
    matches!(uop.op(), Op::Store { .. }) && uop.addrspace() == Some(AddrSpace::Local)
}

fn barrier_from_sources(sources: &[Arc<UOp>]) -> Option<Arc<UOp>> {
    let (src, deps) = sources.split_first()?;
    Some(src.barrier(deps.iter().cloned().collect()))
}

fn add_raw_barrier(after: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::After { passthrough, deps } = after.op() else { return None };
    if after.addrspace() != Some(AddrSpace::Local) {
        return None;
    }

    // Match Tinygrad's single gated toposort over SINK(*after.src[1:]).
    let dependency_sink = UOp::sink(deps.iter().cloned().collect());
    let dependency_toposort = dependency_sink.toposort_filtered(|uop| !matches!(uop.op(), Op::Barrier { .. }));
    if !dependency_toposort.iter().any(is_local_store) {
        return None;
    }

    Some(passthrough.after(smallvec![barrier_from_sources(deps)?]))
}

fn add_war_barrier(end: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::End { computation, ranges } = end.op() else { return None };
    if matches!(computation.op(), Op::Barrier { .. }) {
        return None;
    }

    let loop_ranges: Vec<_> = ranges
        .iter()
        .filter(|range| {
            matches!(range.op(), Op::Range { axis_type: AxisType::Reduce | AxisType::Weak | AxisType::Loop, .. })
                && matches!(range.vmax(), ConstValue::Int(vmax) if *vmax > 0)
        })
        .cloned()
        .collect();
    if loop_ranges.is_empty() {
        return None;
    }

    let backward_slice = computation.backward_slice_with_self();
    let loop_range_ids: HashSet<_> = loop_ranges.iter().map(|range| range.id).collect();
    let store_buffers: HashSet<_> = backward_slice
        .iter()
        .filter(|uop| {
            is_local_store(uop) && uop.in_scope_ranges().iter().any(|range| loop_range_ids.contains(&range.0.id))
        })
        .filter_map(|uop| match uop.op() {
            Op::Store { index, .. } => access_buffer(index).map(UOpKey),
            _ => None,
        })
        .collect();

    let loads: SmallVec<[Arc<UOp>; 4]> = backward_slice
        .iter()
        .filter(|uop| match uop.op() {
            Op::Load { index, .. } => {
                access_buffer(index).is_some_and(|buffer| store_buffers.contains(&UOpKey(buffer)))
            }
            _ => false,
        })
        .cloned()
        .collect();
    if loads.is_empty() {
        return None;
    }

    Some(computation.barrier(loads).end(ranges.clone()))
}

pub(super) fn pm_implicit_barriers() -> &'static TypedPatternMatcher {
    static PM: LazyLock<TypedPatternMatcher> = LazyLock::new(|| {
        crate::patterns! {
            after @ After { passthrough: _, deps: _ } => |after| add_raw_barrier(after),
            end @ End { computation: _, ranges: _ } => |end| add_war_barrier(end),
        }
    });
    &PM
}

#[cfg(test)]
mod tests {
    use super::*;
    use svod_dtype::DType;
    use svod_ir::rewrite::graph_rewrite;
    use svod_ir::{AxisId, BinaryOp, ParamArg, RendererOps};

    fn buffer(slot: usize, addrspace: AddrSpace) -> Arc<UOp> {
        UOp::new(
            Op::Param {
                shape: svod_ir::shape::shape_to_uop(&smallvec::smallvec![8usize.into()]),
                arg: ParamArg::buffer(slot, DType::Float32, addrspace, None),
            },
            DType::Float32,
        )
    }

    fn index(buffer: Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
        UOp::index().buffer(buffer).indices(vec![offset]).call().unwrap()
    }

    fn rewrite(root: Arc<UOp>) -> Arc<UOp> {
        graph_rewrite(pm_implicit_barriers(), root, &mut ())
    }

    #[test]
    fn renderer_extra_matcher_local_dependency_precedes_barrier_inference() {
        let extra = crate::patterns! {
            Noop() => || {
                let local = buffer(0, AddrSpace::Local);
                let store = index(local.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
                Some(local.after(smallvec![store]))
            },
        };
        let renderer =
            crate::optimizer::Renderer::cpu().with_rewrite_capabilities(RendererOps::all(), None, Some(extra));
        let rewritten = graph_rewrite(renderer.extra_matcher().unwrap(), UOp::new(Op::Noop, DType::Void), &mut ());
        let result = crate::optimizer::finish_final_rewrite(rewritten);

        assert!(
            matches!(result.op(), Op::After { deps, .. }
            if matches!(deps.as_slice(), [barrier] if matches!(barrier.op(), Op::Barrier { .. }))),
            "{}",
            result.tree()
        );
    }

    #[test]
    fn renderer_supported_ops_control_decomposition() {
        let x = UOp::const_(DType::UInt64, ConstValue::UInt(1));
        let key = UOp::const_(DType::UInt64, ConstValue::UInt(2));
        let threefry = UOp::new(Op::Binary(BinaryOp::Threefry, x, key), DType::UInt64);

        let supported = RendererOps::all();
        let unchanged =
            graph_rewrite(&super::super::early_decomposition_patterns(&supported), threefry.clone(), &mut ());
        assert!(matches!(unchanged.op(), Op::Binary(BinaryOp::Threefry, ..)));

        let mut unsupported = RendererOps::all();
        unsupported.binary.remove(&BinaryOp::Threefry);
        let decomposed = graph_rewrite(&super::super::early_decomposition_patterns(&unsupported), threefry, &mut ());
        assert!(!decomposed.toposort().iter().any(|uop| matches!(uop.op(), Op::Binary(BinaryOp::Threefry, ..))));

        let erf = UOp::native_const(0.5f32).erf().unwrap();
        let native = graph_rewrite(&super::super::early_decomposition_patterns(&supported), erf.clone(), &mut ());
        assert!(matches!(native.op(), Op::Unary(svod_ir::UnaryOp::Erf, _)));

        unsupported.unary.remove(&svod_ir::UnaryOp::Erf);
        let decomposed = graph_rewrite(&super::super::early_decomposition_patterns(&unsupported), erf, &mut ());
        assert!(!decomposed.toposort().iter().any(|uop| matches!(uop.op(), Op::Unary(svod_ir::UnaryOp::Erf, _))));
    }

    #[test]
    fn local_after_store_gets_raw_barrier() {
        let local = buffer(0, AddrSpace::Local);
        let store = index(local.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
        let result = rewrite(local.after(smallvec![store.clone()]));

        let Op::After { passthrough, deps } = result.op() else { panic!("expected AFTER") };
        assert!(Arc::ptr_eq(passthrough, &local));
        assert!(matches!(deps.as_slice(), [barrier]
            if matches!(barrier.op(), Op::Barrier { src, deps } if Arc::ptr_eq(src, &store) && deps.is_empty())));
    }

    #[test]
    fn global_after_store_does_not_get_barrier() {
        let global = buffer(0, AddrSpace::Global);
        let store = index(global.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
        let result = rewrite(global.after(smallvec![store.clone()]));

        assert!(matches!(result.op(), Op::After { deps, .. }
            if matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &store))));
    }

    #[test]
    fn local_store_and_load_get_war_barrier_for_all_loop_axes() {
        for (slot, axis_type) in [AxisType::Reduce, AxisType::Weak, AxisType::Loop].into_iter().enumerate() {
            let local = buffer(slot, AddrSpace::Local);
            let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(slot), axis_type);
            let load = UOp::load().index(index(local.clone(), range.clone())).call();
            let store = index(local, range.clone()).store_value(load.clone());
            let result = rewrite(store.end(smallvec![range.clone()]));

            let Op::End { computation, ranges } = result.op() else { panic!("expected END") };
            assert!(matches!(computation.op(), Op::Barrier { src, deps }
                if Arc::ptr_eq(src, &store) && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &load))));
            assert!(matches!(ranges.as_slice(), [closed] if Arc::ptr_eq(closed, &range)));
        }
    }

    #[test]
    fn end_computation_load_participates_in_war_detection() {
        let local = buffer(0, AddrSpace::Local);
        let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Weak);
        let store = index(local.clone(), range.clone()).store_value(UOp::native_const(1.0f32));
        let load = UOp::load().index(index(local.after(smallvec![store]), range.clone())).call();
        let result = rewrite(load.end(smallvec![range]));

        assert!(
            matches!(result.op(), Op::End { computation, .. }
            if matches!(computation.op(), Op::Barrier { src, deps }
                if matches!(src.op(), Op::Load { .. })
                    && matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, src)))),
            "{}",
            result.tree()
        );
    }

    #[test]
    fn nonpositive_range_does_not_get_war_barrier() {
        let local = buffer(0, AddrSpace::Local);
        let range = UOp::range_axis(UOp::index_const(0), AxisId::Renumbered(0), AxisType::Weak);
        let load = UOp::load().index(index(local.clone(), range.clone())).call();
        let store = index(local, range.clone()).store_value(load);
        let result = rewrite(store.clone().end(smallvec![range]));

        assert!(matches!(result.op(), Op::End { computation, .. } if Arc::ptr_eq(computation, &store)));
    }

    #[test]
    fn global_memory_hazard_does_not_get_war_barrier() {
        let global = buffer(0, AddrSpace::Global);
        let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Weak);
        let load = UOp::load().index(index(global.clone(), range.clone())).call();
        let store = index(global, range.clone()).store_value(load);
        let result = rewrite(store.clone().end(smallvec![range]));

        assert!(matches!(result.op(), Op::End { computation, .. } if Arc::ptr_eq(computation, &store)));
    }

    #[test]
    fn unrelated_global_load_does_not_match_local_store() {
        let local = buffer(0, AddrSpace::Local);
        let global = buffer(1, AddrSpace::Global);
        let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Weak);
        let store = index(local, range.clone()).store_value(UOp::native_const(1.0f32));
        let load = UOp::load().index(index(global, range.clone())).call();
        let computation = UOp::sink(vec![store, load]);
        let result = rewrite(computation.clone().end(smallvec![range]));

        assert!(matches!(result.op(), Op::End { computation: rewritten, .. } if Arc::ptr_eq(rewritten, &computation)));
    }

    #[test]
    fn existing_barrier_is_not_reinferred() {
        let local = buffer(0, AddrSpace::Local);
        let store = index(local.clone(), UOp::index_const(0)).store_value(UOp::native_const(1.0f32));
        let explicit = store.barrier(SmallVec::new());
        let result = rewrite(local.after(smallvec![explicit.clone()]));

        assert!(matches!(result.op(), Op::After { deps, .. }
            if matches!(deps.as_slice(), [dep] if Arc::ptr_eq(dep, &explicit))));
    }
}
