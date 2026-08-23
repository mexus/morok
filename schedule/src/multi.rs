//! Exact in-kernel multi-device rewrites.
//!
//! `Op::Multi` is the single-axis subset of Tinygrad's `UNSHARD`. It does not
//! carry a shard range or a tuple-valued device, so rewrites that need either
//! are deliberately not represented here.

use std::sync::Arc;

use svod_ir::{ConstValue, Op, UOp};

use crate::TypedPatternMatcher;

/// Hardware-independent subset supported before range assignment.
///
/// `None` is an ordinary unsharded layout and is valid by itself or as a
/// scalar ALU broadcast. `Axis` is Svod's single represented shard layout.
/// Two axes, nested layouts, and operations requiring shard ranges are rejected
/// by [`validate_supported_subset`] rather than leaking into rangeification.
///
/// Supported rewrites are MSELECT(MSTACK), same-axis ALU with scalar operands,
/// non-sharded-axis reductions, PERMUTE, non-shard-axis FLIP/PAD, and the
/// dtype/contiguity wrappers listed in [`multi_pm`]. Outer MULTI, MSTACK, and
/// independent graph outputs remain structural markers at this boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiLayout {
    None,
    Axis(usize),
}

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

fn reduce_axis_multi(root: &Arc<UOp>, multi: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::ReduceAxis { reduce_op, axes, .. } = root.op() else { return None };
    let (local, axis) = multi_axis(multi)?;
    if axes.contains(&axis) {
        return None;
    }
    let output_axis = axis - axes.iter().filter(|&&reduced_axis| reduced_axis < axis).count();
    Some(
        UOp::multi(
            UOp::new(Op::ReduceAxis { src: local, reduce_op: *reduce_op, axes: axes.clone() }, root.dtype()),
            output_axis,
        )
        .rtag(root.tag().clone()),
    )
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
        root @ ReduceAxis { src: multi @ Multi { src: _ }, reduce_op: _, axes: _ }
            => |root, multi| reduce_axis_multi(root, multi),
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

fn operation_name(op: &Op) -> &'static str {
    match op {
        Op::Unary(..) => "unary ALU",
        Op::Binary(..) => "binary ALU",
        Op::Ternary(..) => "ternary ALU",
        Op::ReduceAxis { .. } | Op::Reduce { .. } => "reduction",
        Op::Reshape { .. } => "RESHAPE",
        Op::Permute { .. } => "PERMUTE",
        Op::Expand { .. } => "EXPAND",
        Op::Pad { .. } => "PAD",
        Op::Shrink { .. } => "SHRINK",
        Op::Flip { .. } => "FLIP",
        Op::MSelect { .. } => "MSELECT",
        _ => "operation",
    }
}

fn source_layout(source: &Arc<UOp>) -> MultiLayout {
    match source.op() {
        Op::Multi { axis, .. } => MultiLayout::Axis(*axis),
        _ => MultiLayout::None,
    }
}

fn classify_supported_form(node: &Arc<UOp>) -> svod_ir::Result<()> {
    if let Op::Multi { src, axis } = node.op() {
        if src.toposort().iter().any(|inner| matches!(inner.op(), Op::Multi { .. })) {
            return Err(svod_ir::Error::MultiNested { axis: *axis });
        }
        let shape = src.shape()?.ok_or(svod_ir::Error::MultiUnsupported {
            operation: "MULTI",
            reason: "source layout has no inferable shape",
        })?;
        if *axis >= shape.len() {
            return Err(svod_ir::Error::MultiUnsupported {
                operation: "MULTI",
                reason: "shard axis is outside the source shape",
            });
        }
        return Ok(());
    }

    if let Op::MSelect { .. } = node.op() {
        return Err(svod_ir::Error::MultiUnsupported {
            operation: "MSELECT",
            reason: "selection did not resolve to an in-range MSTACK shard",
        });
    }

    let layouts: Vec<_> = node.op().sources().iter().map(source_layout).collect();
    let mut axes: Vec<_> = layouts
        .iter()
        .filter_map(|layout| match layout {
            MultiLayout::Axis(axis) => Some(*axis),
            MultiLayout::None => None,
        })
        .collect();
    if axes.is_empty() {
        return Ok(());
    }

    // Graph containers do not combine their independent source layouts.
    if matches!(node.op(), Op::Sink { .. } | Op::Group { .. } | Op::Tuple { .. }) {
        return Ok(());
    }
    axes.sort_unstable();
    axes.dedup();

    let operation = operation_name(node.op());
    if axes.len() != 1 {
        return Err(svod_ir::Error::MultiAxisMismatch { operation, axes });
    }
    let axis = axes[0];

    match node.op() {
        Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) => {
            for (source, layout) in node.op().sources().iter().zip(layouts) {
                if layout == MultiLayout::None && !source.shape()?.is_some_and(|shape| shape.is_empty()) {
                    return Err(svod_ir::Error::MultiLayoutMissing { operation, axis, source_id: source.id });
                }
            }
            Err(svod_ir::Error::MultiUnsupported {
                operation,
                reason: "supported per-shard ALU did not normalize before rangeification",
            })
        }
        Op::ReduceAxis { axes, .. } => {
            if axes.contains(&axis) {
                Err(svod_ir::Error::MultiReductionAcrossShardAxis { axis })
            } else {
                Err(svod_ir::Error::MultiUnsupported {
                    operation,
                    reason: "non-sharded-axis reduction did not normalize before rangeification",
                })
            }
        }
        Op::Reduce { num_axes, .. } => {
            if axis < *num_axes {
                Err(svod_ir::Error::MultiReductionAcrossShardAxis { axis })
            } else {
                Err(svod_ir::Error::MultiUnsupported {
                    operation,
                    reason: "non-sharded-axis reduction did not normalize before rangeification",
                })
            }
        }
        Op::Reshape { .. } => Err(svod_ir::Error::MultiMovementUnsupported {
            operation,
            axis,
            reason: "the shard boundary cannot be mapped without shard-count metadata",
        }),
        op if op.is_movement() => Err(svod_ir::Error::MultiMovementUnsupported {
            operation,
            axis,
            reason: "the movement crosses or cannot prove preservation of the shard boundary",
        }),
        _ => Err(svod_ir::Error::MultiUnsupported {
            operation,
            reason: "no hardware-independent per-shard rewrite is defined",
        }),
    }
}

/// Reject every unresolved form outside the exact hardware-independent subset.
pub fn validate_supported_subset(root: &Arc<UOp>) -> svod_ir::Result<()> {
    for node in root.toposort_call_aware(true) {
        classify_supported_form(&node)?;
    }
    Ok(())
}
