//! Exact in-kernel multi-device rewrites.
//!
//! `Op::Multi` is the single-axis subset of Tinygrad's `UNSHARD`. It does not
//! carry a shard range or a tuple-valued device, so rewrites that need either
//! are deliberately not represented here.

use std::sync::Arc;

use svod_ir::{ConstValue, Op, UOp};

use crate::TypedPatternMatcher;

fn multi_axis(uop: &Arc<UOp>) -> Option<(Arc<UOp>, usize)> {
    match uop.op() {
        Op::Multi { src, axis } => Some((src.clone(), *axis)),
        _ => None,
    }
}

fn rewrite_per_shard_alu(root: &Arc<UOp>) -> Option<Arc<UOp>> {
    if !matches!(root.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..)) {
        return None;
    }

    let axis = root.op().sources().iter().find_map(|src| multi_axis(src).map(|(_, axis)| axis))?;
    let mut local_sources = Vec::with_capacity(root.op().sources().len());
    for src in root.op().sources() {
        if let Some((local, src_axis)) = multi_axis(&src) {
            if src_axis != axis {
                return None;
            }
            local_sources.push(local);
        } else if src.shape().ok().flatten().is_some_and(|shape| shape.is_empty()) {
            local_sources.push(src.clone());
        } else {
            return None;
        }
    }
    Some(UOp::multi(root.with_sources(local_sources), axis).rtag(root.tag().clone()))
}

fn passthrough_unary_wrapper(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let (local, axis) = multi_axis(multi)?;
    if !matches!(
        root.op(),
        Op::Cast { .. }
            | Op::BitCast { .. }
            | Op::Contiguous { .. }
            | Op::Detach { .. }
            | Op::ContiguousBackward { .. }
    ) {
        return None;
    }
    Some(UOp::multi(root.with_sources(vec![local]), axis).rtag(root.tag().clone()))
}

fn reduce_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Reduce { ranges, reduce_op, num_axes, .. } = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if axis < *num_axes {
        return None;
    }
    Some(UOp::multi(local.reduce_with_num_axes(ranges.clone(), *reduce_op, *num_axes), axis - num_axes))
}

fn permute_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Permute { axes, .. } = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    let new_axis = axes.iter().position(|&candidate| candidate == axis)?;
    Some(UOp::multi(root.with_sources(vec![local]), new_axis).rtag(root.tag().clone()))
}

fn flip_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Flip { axes, .. } = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if axes.get(axis).copied().unwrap_or(true) {
        return None;
    }
    Some(UOp::multi(root.with_sources(vec![local]), axis).rtag(root.tag().clone()))
}

fn const_at(uop: &Arc<UOp>, axis: usize) -> Option<ConstValue> {
    match uop.op() {
        Op::Stack { sources } => match sources.get(axis)?.op() {
            Op::Const(value) => Some(value.0),
            _ => None,
        },
        Op::VConst { values } => values.get(axis).copied(),
        _ if axis == 0 => match uop.op() {
            Op::Const(value) => Some(value.0),
            _ => None,
        },
        _ => None,
    }
}

fn pad_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Pad { begin_pads, end_pads, .. } = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if !matches!(const_at(begin_pads, axis), Some(ConstValue::Int(0) | ConstValue::UInt(0)))
        || !matches!(const_at(end_pads, axis), Some(ConstValue::Int(0) | ConstValue::UInt(0)))
    {
        return None;
    }
    Some(
        UOp::multi(root.with_sources(vec![local, begin_pads.clone(), end_pads.clone()]), axis).rtag(root.tag().clone()),
    )
}

fn move_mselect_before_movement(root: &Arc<UOp>, buffer: &Arc<UOp>, device_index: usize) -> Option<Arc<UOp>> {
    if !buffer.op().is_movement() {
        return None;
    }
    let mut sources: Vec<_> = buffer.op().sources().iter().map(|src| (*src).clone()).collect();
    sources[0] = sources[0].mselect(device_index);
    Some(buffer.with_sources(sources).rtag(root.tag().clone()))
}

/// Tinygrad `multi_pm` clauses that have an exact representation in Svod.
pub fn multi_pm() -> TypedPatternMatcher {
    crate::patterns! {
        selected @ MSelect { buffer: MStack { buffers }, device_index: _ }
            => |selected, buffers| {
                let Op::MSelect { device_index, .. } = selected.op() else { unreachable!() };
                buffers.get(*device_index).cloned()
            },
        selected @ MSelect { buffer, device_index: _ }
            if buffer.op().is_movement()
            => |selected, buffer| {
                let Op::MSelect { device_index, .. } = selected.op() else { unreachable!() };
                move_mselect_before_movement(selected, buffer, *device_index)
            },
        root @ Reduce { src: multi @ Multi { src: _ }, ranges: _, reduce_op: _, num_axes: _ }
            => |root, multi| reduce_multi(root, multi),
        root @ Permute { src: multi @ Multi { src: _ }, axes: _ }
            => |root, multi| permute_multi(root, multi),
        root @ Flip { src: multi @ Multi { src: _ }, axes: _ }
            => |root, multi| flip_multi(root, multi),
        root @ Pad { src: multi @ Multi { src: _ }, begin_pads: _, end_pads: _ }
            => |root, multi| pad_multi(root, multi),
        root @ Cast { src: multi @ Multi { src: _ }, dtype: _ }
            => |root, multi| passthrough_unary_wrapper(root, multi),
        root @ BitCast { src: multi @ Multi { src: _ }, dtype: _ }
            => |root, multi| passthrough_unary_wrapper(root, multi),
        root @ Contiguous { src: multi @ Multi { src: _ }, opts: _ }
            => |root, multi| passthrough_unary_wrapper(root, multi),
        root @ Detach { src: multi @ Multi { src: _ } }
            => |root, multi| passthrough_unary_wrapper(root, multi),
        root @ ContiguousBackward { src: multi @ Multi { src: _ } }
            => |root, multi| passthrough_unary_wrapper(root, multi),
        root if matches!(root.op(), Op::Unary(..) | Op::Binary(..) | Op::Ternary(..))
            => |root| rewrite_per_shard_alu(root),
    }
}
