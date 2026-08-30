//! Dtype production rules for UOps.
//!
//! This mirrors Tinygrad's `dtype_from_uop`. Operations whose dtype still
//! depends on legacy Svod metadata return `None` until that metadata moves into
//! the operation itself.

use svod_dtype::DType;

use crate::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};

fn promote(src: impl IntoIterator<Item = DType>) -> Option<DType> {
    let dtypes: Vec<_> = src.into_iter().collect();
    let first = dtypes.first()?;
    if dtypes.iter().all(|dtype| dtype == first) { Some(first.clone()) } else { DType::least_upper_dtype(&dtypes) }
}

fn const_dtype(value: &ConstValue) -> DType {
    match value {
        ConstValue::Invalid => DType::Bool,
        ConstValue::Bool(_) => DType::Bool,
        ConstValue::Int(_) | ConstValue::UInt(_) => DType::WeakInt,
        ConstValue::Float(_) => DType::WeakFloat,
    }
}

/// Derive an operation's result dtype from its sources and metadata.
///
/// `None` means the current operation carries result-type information outside
/// the modern Tinygrad production rule and still requires an explicit dtype.
pub fn dtype_from_op(op: &Op) -> Option<DType> {
    match op {
        Op::Sink { .. }
        | Op::Group { .. }
        | Op::If { .. }
        | Op::EndIf { .. }
        | Op::End { .. }
        | Op::Barrier { .. }
        | Op::Tuple { .. }
        | Op::Function { .. }
        | Op::Program { .. }
        | Op::Linear { .. }
        | Op::Source { .. }
        | Op::CustomFunction { .. }
        | Op::Store { .. }
        | Op::Unique(_)
        | Op::LUnique(_) => Some(DType::Void),

        Op::ProgramBinary { .. } => Some(DType::UInt8),

        Op::Const(value) => Some(const_dtype(&value.0)),
        Op::Noop | Op::Custom { .. } | Op::CustomI { .. } | Op::VConst { .. } | Op::Ins { .. } => None,

        // These operations still keep their storage dtype outside Op metadata.
        Op::Param { arg, .. } | Op::Buffer { arg, .. } => Some(arg.dtype.clone()),
        Op::Slice { .. } => None,
        Op::Index { buffer, indices } => Some(if matches!(buffer.op(), Op::Param { .. }) && is_image_shape(buffer) {
            DType::Float32
        } else if !indices.is_empty() && buffer.dtype().vcount() > 1 && !is_storage_index_source(buffer) {
            buffer.dtype().scalar_dtype()
        } else {
            buffer.dtype()
        }),
        Op::Load { index, .. } => Some(index.dtype()),
        Op::GetAddr { .. } => Some(DType::UInt64),

        Op::Cast { dtype, .. } | Op::BitCast { dtype, .. } => Some(dtype.clone()),

        Op::Unary(unary, src) => match unary {
            UnaryOp::Sqrt
            | UnaryOp::Rsqrt
            | UnaryOp::Exp
            | UnaryOp::Exp2
            | UnaryOp::Log
            | UnaryOp::Log2
            | UnaryOp::Sin
            | UnaryOp::Cos
            | UnaryOp::Tan
            | UnaryOp::Reciprocal
            | UnaryOp::Erf => DType::least_upper_float(src.dtype()),
            _ => Some(src.dtype()),
        },
        Op::Binary(binary, lhs, rhs) if binary.is_comparison() => Some(DType::Bool),
        Op::Binary(BinaryOp::Shl | BinaryOp::Shr, lhs, _) => Some(lhs.dtype()),
        Op::Binary(_, lhs, rhs) => promote([lhs.dtype(), rhs.dtype()]),
        Op::Ternary(TernaryOp::Where, condition, true_value, false_value) => {
            if !condition.dtype().is_bool() {
                return None;
            }
            if UOp::is_invalid_marker(true_value) {
                Some(false_value.dtype())
            } else if UOp::is_invalid_marker(false_value) {
                Some(true_value.dtype())
            } else {
                promote([true_value.dtype(), false_value.dtype()])
            }
        }
        Op::Ternary(TernaryOp::MulAcc, a, b, c) => promote([a.dtype(), b.dtype(), c.dtype()]),

        Op::MSelect { buffer, .. }
        | Op::Copy { src: buffer, .. }
        | Op::Stage { compute: buffer, .. }
        | Op::Reshape { src: buffer, .. }
        | Op::Permute { src: buffer, .. }
        | Op::Expand { src: buffer, .. }
        | Op::Pad { src: buffer, .. }
        | Op::Shrink { src: buffer, .. }
        | Op::Flip { src: buffer, .. }
        | Op::Multi { src: buffer, .. }
        | Op::ReduceAxis { src: buffer, .. }
        | Op::Reduce { src: buffer, .. }
        | Op::AllReduce { src: buffer, .. }
        | Op::Detach { src: buffer }
        | Op::Contiguous { src: buffer, .. }
        | Op::ContiguousBackward { src: buffer }
        | Op::After { passthrough: buffer, .. }
        | Op::Precast { src: buffer } => Some(buffer.dtype()),

        Op::Special { end, .. } => Some(end.dtype()),
        Op::Range { end, .. } => Some(end.dtype()),
        Op::Bind { var, value } if var.dtype() == value.dtype() => Some(var.dtype()),
        Op::Bind { .. } => None,
        Op::Wmma { c, .. } => Some(c.dtype()),
        Op::MStack { buffers } => buffers.first().map(|buffer| buffer.dtype()),

        Op::Stack { sources } if sources.is_empty() => Some(DType::Void),
        Op::Stack { sources } => promote(sources.iter().map(|source| source.dtype())),

        Op::DefineVar { .. } => None,
        Op::Call { body, .. } if body.dtype() == DType::Void => Some(DType::Void),
        Op::Call { .. } => None,
        Op::GetTuple { src, index } => tuple_element(src, *index).map(|element| element.dtype()),
    }
}

fn tuple_element(src: &std::sync::Arc<UOp>, index: usize) -> Option<&std::sync::Arc<UOp>> {
    let tuple = match src.op() {
        Op::Function { body, .. } => body,
        _ => src,
    };
    match tuple.op() {
        Op::Tuple { src } => src.get(index),
        _ => None,
    }
}

fn is_image_shape(u: &std::sync::Arc<UOp>) -> bool {
    u.shape().ok().flatten().is_some_and(|shape| shape.len() == 3 && shape[2].as_const() == Some(4))
}

fn is_storage_index_source(u: &std::sync::Arc<UOp>) -> bool {
    match u.op() {
        Op::Param { .. } | Op::Buffer { .. } | Op::Slice { .. } => true,
        Op::After { passthrough, .. } | Op::Precast { src: passthrough } => is_storage_index_source(passthrough),
        _ => false,
    }
}
