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
        | Op::ProgramBinary { .. }
        | Op::CustomFunction { .. }
        | Op::Store { .. }
        | Op::Unique(_)
        | Op::LUnique(_)
        | Op::Device(_) => Some(DType::Void),

        Op::Const(value) => Some(const_dtype(&value.0)),
        Op::Noop | Op::Custom { .. } | Op::CustomI { .. } | Op::VConst { .. } => None,

        // These operations still keep their storage dtype outside Op metadata.
        Op::Param { .. } | Op::Buffer { .. } | Op::DefineLocal(_) | Op::DefineReg { .. } => None,
        Op::BufferView { buffer, .. } => Some(buffer.dtype()),
        Op::Index { .. } | Op::Load { .. } | Op::PointerIndex { .. } => None,

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
        | Op::Bufferize { compute: buffer, .. }
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
        | Op::Precast { src: buffer }
        | Op::Contract { src: buffer, .. }
        | Op::Unroll { src: buffer, .. } => Some(buffer.dtype()),

        Op::Special { end, .. } | Op::Range { end, .. } => Some(end.dtype()),
        Op::Bind { var, value } if var.dtype() == value.dtype() => Some(var.dtype()),
        Op::Bind { .. } => None,
        Op::Wmma { c, .. } => Some(c.dtype()),
        Op::MStack { buffers } => buffers.first().map(|buffer| buffer.dtype()),

        Op::Vectorize { elements } => {
            let scalar = promote(elements.iter().map(|element| element.dtype()))?;
            scalar.vec(elements.len())
        }
        Op::Gep { vector, indices } => vector.dtype().scalar_dtype().vec(indices.len()),
        Op::Cat { sources } => {
            let scalar = promote(sources.iter().map(|source| source.dtype().scalar_dtype()))?;
            scalar.vec(sources.iter().map(|source| source.dtype().vcount()).sum())
        }
        Op::PtrCat { .. } => None,

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
