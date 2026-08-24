use std::collections::HashMap;
use std::sync::Arc;

use svod_device::Buffer;
use svod_dtype::{DType, ScalarDType};
use svod_ir::{CustomFunctionKind, ReduceOp, UOp};

use crate::{Error, Result};

fn unsupported(kind: &str, attrs: &[Arc<UOp>], buffers: &[Buffer], vars: &HashMap<String, i64>) -> Error {
    Error::Unsupported {
        kind: kind.to_string(),
        reason: format!(
            "runtime is reserved but not implemented (attrs={}, buffers={}, vars={})",
            attrs.len(),
            buffers.len(),
            vars.len()
        ),
    }
}

pub fn run_custom_function(
    kind: &CustomFunctionKind,
    attrs: &[Arc<UOp>],
    buffers: &mut [Buffer],
    vars: &HashMap<String, i64>,
) -> Result<()> {
    match kind {
        CustomFunctionKind::EncDec => Err(unsupported("EncDec", attrs, buffers, vars)),
        CustomFunctionKind::Graph => Err(unsupported("Graph", attrs, buffers, vars)),
        CustomFunctionKind::AllReduce { reduce_op } => run_host_allreduce(*reduce_op, buffers),
    }
}

fn run_host_allreduce(reduce_op: ReduceOp, buffers: &mut [Buffer]) -> Result<()> {
    if buffers.len() < 3 {
        return Err(Error::Execution { reason: "all-reduce requires output and at least two shard buffers".into() });
    }
    if !matches!(reduce_op, ReduceOp::Add | ReduceOp::Max) {
        return Err(Error::Unsupported {
            kind: "AllReduce".into(),
            reason: format!("host collective only supports SUM and MAX, got {reduce_op:?}"),
        });
    }
    let dtype = buffers[0].dtype();
    let byte_len = buffers[0].size();
    let shape = buffers[0].shape();
    if buffers.iter().any(|buffer| buffer.dtype() != dtype || buffer.size() != byte_len || buffer.shape() != shape) {
        return Err(Error::Execution {
            reason: "all-reduce buffers must have identical dtype, shape, and byte size".into(),
        });
    }
    let element_width = match dtype {
        DType::Scalar(ScalarDType::Float16 | ScalarDType::BFloat16 | ScalarDType::Int16 | ScalarDType::UInt16) => 2,
        DType::Scalar(ScalarDType::Float32 | ScalarDType::Int32 | ScalarDType::UInt32) => 4,
        DType::Scalar(ScalarDType::Float64 | ScalarDType::Int64 | ScalarDType::UInt64) => 8,
        DType::Scalar(ScalarDType::Int8 | ScalarDType::UInt8) => 1,
        ref other => {
            return Err(Error::Unsupported {
                kind: "AllReduce".into(),
                reason: format!("host collective dtype {other:?} is not supported"),
            });
        }
    };
    if !byte_len.is_multiple_of(element_width) {
        return Err(Error::Execution {
            reason: format!(
                "all-reduce byte size {byte_len} is not aligned to {element_width}-byte {dtype:?} elements"
            ),
        });
    }
    let mut shards = Vec::with_capacity(buffers.len() - 1);
    for buffer in &buffers[1..] {
        let mut bytes = vec![0; byte_len];
        buffer
            .copyout(&mut bytes)
            .map_err(|source| Error::Exec { source, context: "host all-reduce copyout".into() })?;
        shards.push(bytes);
    }
    let mut output = vec![0; byte_len];

    macro_rules! reduce_primitive {
        ($ty:ty) => {{
            let width = std::mem::size_of::<$ty>();
            for offset in (0..byte_len).step_by(width) {
                let read = |bytes: &[u8]| <$ty>::from_le_bytes(bytes[offset..offset + width].try_into().unwrap());
                let mut value = read(&shards[0]);
                for shard in &shards[1..] {
                    let rhs = read(shard);
                    value = match reduce_op {
                        ReduceOp::Add => value + rhs,
                        ReduceOp::Max => value.max(rhs),
                        _ => unreachable!(),
                    };
                }
                output[offset..offset + width].copy_from_slice(&value.to_le_bytes());
            }
        }};
    }

    macro_rules! reduce_integer {
        ($ty:ty) => {{
            let width = std::mem::size_of::<$ty>();
            for offset in (0..byte_len).step_by(width) {
                let read = |bytes: &[u8]| <$ty>::from_le_bytes(bytes[offset..offset + width].try_into().unwrap());
                let mut value = read(&shards[0]);
                for shard in &shards[1..] {
                    let rhs = read(shard);
                    value = match reduce_op {
                        ReduceOp::Add => value.wrapping_add(rhs),
                        ReduceOp::Max => value.max(rhs),
                        _ => unreachable!(),
                    };
                }
                output[offset..offset + width].copy_from_slice(&value.to_le_bytes());
            }
        }};
    }

    match dtype {
        DType::Scalar(ScalarDType::Float32) => reduce_primitive!(f32),
        DType::Scalar(ScalarDType::Float64) => reduce_primitive!(f64),
        DType::Scalar(ScalarDType::Int8) => reduce_integer!(i8),
        DType::Scalar(ScalarDType::Int16) => reduce_integer!(i16),
        DType::Scalar(ScalarDType::Int32) => reduce_integer!(i32),
        DType::Scalar(ScalarDType::Int64) => reduce_integer!(i64),
        DType::Scalar(ScalarDType::UInt8) => reduce_integer!(u8),
        DType::Scalar(ScalarDType::UInt16) => reduce_integer!(u16),
        DType::Scalar(ScalarDType::UInt32) => reduce_integer!(u32),
        DType::Scalar(ScalarDType::UInt64) => reduce_integer!(u64),
        DType::Scalar(ScalarDType::Float16 | ScalarDType::BFloat16) => {
            let scalar = dtype.scalar().unwrap();
            for offset in (0..byte_len).step_by(2) {
                let decode = |bytes: &[u8]| {
                    let bits = u16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap());
                    match scalar {
                        ScalarDType::Float16 => svod_dtype::cast::f16_bits_to_float(bits),
                        ScalarDType::BFloat16 => f32::from_bits((bits as u32) << 16) as f64,
                        _ => unreachable!(),
                    }
                };
                let mut value = decode(&shards[0]);
                for shard in &shards[1..] {
                    let rhs = decode(shard);
                    value = match reduce_op {
                        ReduceOp::Add => {
                            let sum = if scalar == ScalarDType::BFloat16 {
                                ((value as f32) + (rhs as f32)) as f64
                            } else {
                                value + rhs
                            };
                            svod_dtype::cast::commit_float(sum, scalar).ok_or_else(|| Error::Execution {
                                reason: format!("cannot commit {value} + {rhs} as {scalar:?} during all-reduce"),
                            })?
                        }
                        ReduceOp::Max => value.max(rhs),
                        _ => unreachable!(),
                    };
                }
                let bits = svod_dtype::cast::committed_float_bits(value, scalar).ok_or_else(|| Error::Execution {
                    reason: format!("cannot encode {value} as {scalar:?} during all-reduce"),
                })? as u16;
                output[offset..offset + 2].copy_from_slice(&bits.to_le_bytes());
            }
        }
        _ => unreachable!("dtype was validated above"),
    }

    buffers[0].copyin(&output).map_err(|source| Error::Exec { source, context: "host all-reduce copyin".into() })
}

#[cfg(test)]
#[path = "test/unit/custom_function.rs"]
mod tests;
