//! ONNX operator registry - maps ONNX ops to Morok Tensor operations.

pub mod proto;
pub use proto::*;

pub(crate) mod attr;
pub(crate) use attr::*;

mod constant;
mod indexing;
mod nn;
mod shape;
mod transformer;

use morok_dtype::{DType, ScalarDType};
use morok_ir::ConstValue;
use morok_tensor::Tensor;
use morok_tensor::reduce::AxisSpec;

use crate::error::{Error, Result, UnsupportedOpSnafu};
use crate::parser::onnx::NodeProto;

/// Operator registry for dispatching ONNX ops to Morok Tensor operations.
pub struct OpRegistry;

impl OpRegistry {
    pub fn new() -> Self {
        Self
    }

    /// Dispatch an ONNX operator (convenience for callers with non-optional inputs).
    /// Uses a default opset version (latest). For opset-aware dispatch, use `dispatch_multi`.
    pub fn dispatch(&self, op_type: &str, domain: &str, inputs: &[Tensor], node: &NodeProto) -> Result<Tensor> {
        let inputs: Vec<Option<Tensor>> = inputs.iter().cloned().map(Some).collect();
        let outputs = self.dispatch_multi(op_type, domain, &inputs, node, i64::MAX)?;
        outputs
            .into_iter()
            .next()
            .ok_or_else(|| Error::IrConstruction { details: format!("Operator {} produced no outputs", op_type) })
    }

    /// Dispatch an ONNX operator, returning a vector of output tensors.
    /// Inputs use `Option<Tensor>` to correctly handle optional ONNX inputs
    /// (empty input names become `None`, preserving positional indices).
    pub fn dispatch_multi(
        &self,
        op_type: &str,
        domain: &str,
        inputs: &[Option<Tensor>],
        node: &NodeProto,
        opset_version: i64,
    ) -> Result<Vec<Tensor>> {
        // Domain-specific ops (checked first)
        if let Some(result) = match (domain, op_type) {
            ("com.microsoft", "Attention") => Some(transformer::op_attention_contrib(inputs, node)),
            ("com.microsoft", "SkipLayerNormalization") => Some(transformer::op_skip_layer_norm(inputs, node)),
            ("com.microsoft", "EmbedLayerNormalization") => Some(transformer::op_embed_layer_norm(inputs, node)),
            ("com.microsoft", "RotaryEmbedding") => Some(transformer::op_rotary_embedding(inputs, node)),
            _ => None,
        } {
            return result;
        }

        let r = match op_type {
            // === Arithmetic ===
            "Add" => inp(inputs, 0).try_add(inp(inputs, 1))?,
            "Sub" => inp(inputs, 0).try_sub(inp(inputs, 1))?,
            "Mul" => inp(inputs, 0).try_mul(inp(inputs, 1))?,
            "Div" => {
                let x = inp(inputs, 0);
                let y = inp(inputs, 1);
                let result = x.try_div(y)?;
                if x.uop().dtype().is_int() { result.trunc()? } else { result }
            }
            "Neg" => inp(inputs, 0).try_neg()?,
            "Abs" => inp(inputs, 0).try_abs()?,
            "Pow" => inp(inputs, 0).try_pow(inp(inputs, 1))?,
            "Mod" => {
                let fmod = get_attr_int(node, "fmod", 0);
                if fmod == 1 {
                    inp(inputs, 0).try_mod(inp(inputs, 1))?
                } else {
                    let x = inp(inputs, 0);
                    let y = inp(inputs, 1);
                    let div = x.try_div(y)?;
                    let floored = div.floor()?;
                    let product = floored.try_mul(y)?;
                    x.try_sub(&product)?
                }
            }
            "Sum" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Sum requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.try_add(t)?;
                }
                acc
            }
            "Mean" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let count = valid.len();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Mean requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.try_add(t)?;
                }
                acc.try_div(&Tensor::from_slice([count as f32]))?
            }

            // === Bitwise ===
            "BitShift" => {
                let dir = get_attr_string(node, "direction", "");
                if dir == "LEFT" {
                    inp(inputs, 0).lshift(inp(inputs, 1))?
                } else {
                    inp(inputs, 0).rshift(inp(inputs, 1))?
                }
            }
            "BitwiseAnd" => inp(inputs, 0).bitwise_and(inp(inputs, 1))?,
            "BitwiseOr" => inp(inputs, 0).bitwise_or(inp(inputs, 1))?,
            "BitwiseXor" => inp(inputs, 0).bitwise_xor(inp(inputs, 1))?,
            "BitwiseNot" => inp(inputs, 0).bitwise_not()?,

            // === Math ===
            "Sqrt" => inp(inputs, 0).try_sqrt()?,
            "Exp" => inp(inputs, 0).try_exp()?,
            "Log" => inp(inputs, 0).try_log()?,
            "Ceil" => inp(inputs, 0).ceil()?,
            "Floor" => inp(inputs, 0).floor()?,
            "Round" => inp(inputs, 0).round()?,
            "Sign" => inp(inputs, 0).sign()?,
            "Reciprocal" => inp(inputs, 0).reciprocal()?,
            "Erf" => inp(inputs, 0).erf()?,
            "Sin" => inp(inputs, 0).sin()?,
            "Cos" => inp(inputs, 0).cos()?,
            "Tan" => inp(inputs, 0).tan()?,

            // === Activation ===
            "Relu" => inp(inputs, 0).relu()?,
            "Sigmoid" => inp(inputs, 0).sigmoid()?,
            "Tanh" => inp(inputs, 0).tanh()?,
            "Softmax" => {
                let default_axis = if opset_version < 13 { 1 } else { -1 };
                let axis = get_attr_int(node, "axis", default_axis) as isize;
                inp(inputs, 0).softmax(axis)?
            }
            "LogSoftmax" => {
                let default_axis = if opset_version < 13 { 1 } else { -1 };
                let axis = get_attr_int(node, "axis", default_axis) as isize;
                inp(inputs, 0).log_softmax(axis)?
            }
            "Gelu" => {
                let approximate = get_attr_string(node, "approximate", "none");
                if approximate == "tanh" {
                    inp(inputs, 0).gelu()?
                } else {
                    let x = inp(inputs, 0);
                    let dtype = x.uop().dtype();
                    let half = Tensor::const_(0.5f64, dtype.clone());
                    let one = Tensor::const_(1.0f64, dtype.clone());
                    let sqrt2 = Tensor::const_(std::f64::consts::SQRT_2, dtype);
                    half.try_mul(x)?.try_mul(&one.try_add(&x.try_div(&sqrt2)?.erf()?)?)?
                }
            }
            "HardSigmoid" => {
                let alpha = get_attr_float(node, "alpha", 0.2) as f64;
                let beta = get_attr_float(node, "beta", 0.5) as f64;
                inp(inputs, 0).hard_sigmoid(alpha, beta)?
            }
            "LeakyRelu" => {
                let alpha = get_attr_float(node, "alpha", 0.01) as f64;
                inp(inputs, 0).leaky_relu(alpha)?
            }
            "PRelu" => inp(inputs, 0).prelu(inp(inputs, 1))?,
            "ThresholdedRelu" => {
                let alpha = get_attr_float(node, "alpha", 1.0) as f64;
                inp(inputs, 0).thresholded_relu(alpha)?
            }
            "Elu" => {
                let alpha = get_attr_float(node, "alpha", 1.0) as f64;
                inp(inputs, 0).elu(alpha)?
            }
            "Selu" => {
                let alpha = get_attr_float(node, "alpha", 1.6732632) as f64;
                let gamma = get_attr_float(node, "gamma", 1.050_701) as f64;
                inp(inputs, 0).selu(alpha, gamma)?
            }

            // === Comparison ===
            "Equal" => inp(inputs, 0).try_eq(inp(inputs, 1))?,
            "Less" => inp(inputs, 0).try_lt(inp(inputs, 1))?,
            "LessOrEqual" => inp(inputs, 0).try_le(inp(inputs, 1))?,
            "Greater" => inp(inputs, 0).try_gt(inp(inputs, 1))?,
            "GreaterOrEqual" => inp(inputs, 0).try_ge(inp(inputs, 1))?,
            "Not" => inp(inputs, 0).logical_not()?,

            // === Conditional ===
            "Where" => inp(inputs, 1).where_(inp(inputs, 0), inp(inputs, 2))?,
            "Max" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Max requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.maximum(t)?;
                }
                acc
            }
            "Min" => {
                let valid: Vec<&Tensor> = inputs.iter().filter_map(Option::as_ref).collect();
                let first = valid
                    .first()
                    .ok_or_else(|| Error::IrConstruction { details: "Min requires at least one input".into() })?;
                let mut acc = (*first).clone();
                for t in &valid[1..] {
                    acc = acc.minimum(t)?;
                }
                acc
            }
            "Clip" => {
                let min = inputs.get(1).and_then(|o| o.as_ref());
                let max = inputs.get(2).and_then(|o| o.as_ref());
                inp(inputs, 0).clamp().maybe_min(min).maybe_max(max).call()?
            }

            // === Type ===
            "Cast" => {
                let to = get_attr_int(node, "to", 1);
                let dtype = convert_onnx_dtype(to as i32).unwrap_or_else(|_| {
                    tracing::warn!("ONNX dtype {to} unsupported, falling back to Float32");
                    DType::Float32
                });
                inp(inputs, 0).cast(dtype)?
            }
            "CastLike" => inp(inputs, 0).cast(inp(inputs, 1).uop().dtype())?,

            // === Shape ===
            "Reshape" => shape::op_reshape(inputs, node)?,
            "Transpose" => shape::op_transpose(inputs, node)?,
            "Squeeze" => shape::op_squeeze(inputs, node, opset_version)?,
            "Unsqueeze" => shape::op_unsqueeze(inputs, node, opset_version)?,
            "Flatten" => shape::op_flatten(inputs, node)?,
            "Concat" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let tensors: Vec<&Tensor> = inputs.iter().filter_map(|o| o.as_ref()).collect();
                Tensor::cat(&tensors, axis)?
            }
            "Shape" => {
                let start = get_attr_int(node, "start", 0) as isize;
                let end = get_attr_int(node, "end", i64::MAX) as isize;
                let shape = inp(inputs, 0).shape()?;
                let ndim = shape.len() as isize;
                let s = if start < 0 { (ndim + start).max(0) } else { start.min(ndim) } as usize;
                let e = (if end < 0 { (ndim + end).max(0) } else { end.min(ndim) } as usize).max(s);
                let dims: Vec<i64> = shape[s..e]
                    .iter()
                    .map(|d| {
                        d.as_const()
                            .map(|v| v as i64)
                            .ok_or_else(|| Error::IrConstruction { details: "Shape requires concrete dims".into() })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Tensor::from_slice(&dims)
            }
            "Expand" => shape::op_expand(inputs)?,
            "Pad" => shape::op_pad(inputs, node)?,
            "Slice" => shape::op_slice(inputs)?,
            "Split" => return shape::op_split(inputs, node),
            "Tile" => {
                let repeats: Vec<usize> = tensor_to_i64_vec(inp(inputs, 1))?.iter().map(|&v| v as usize).collect();
                inp(inputs, 0).repeat(&repeats)?
            }
            "Range" => {
                let start_t = inp(inputs, 0);
                let out_dtype = start_t.uop().dtype();
                if out_dtype.is_float() {
                    let start = tensor_to_f64_scalar(start_t)?;
                    let limit = tensor_to_f64_scalar(inp(inputs, 1))?;
                    let delta = tensor_to_f64_scalar(inp(inputs, 2))?;
                    Tensor::arange_f64(start, limit, delta, out_dtype)?
                } else {
                    let start = tensor_to_i64_vec(start_t)?[0];
                    let limit = tensor_to_i64_vec(inp(inputs, 1))?[0];
                    let delta = tensor_to_i64_vec(inp(inputs, 2))?[0];
                    Tensor::arange_with_dtype(start, Some(limit), Some(delta), out_dtype)?
                }
            }
            "ConstantOfShape" => {
                let shape_i64 = tensor_to_i64_vec(inp(inputs, 0))?;
                let value = get_attr_tensor(node, "value")
                    .map(tensor_from_proto)
                    .transpose()?
                    .unwrap_or_else(|| Tensor::from_slice([0.0f32]));
                if shape_i64.contains(&0) {
                    Tensor::empty(value.uop().dtype())
                } else {
                    let shape: Vec<isize> = shape_i64.iter().map(|&v| v as isize).collect();
                    value.try_reshape(&[1])?.try_expand(&shape)?
                }
            }
            "Size" => Tensor::from_slice([inp(inputs, 0).numel()? as i64]),
            "Dropout" => return nn::op_dropout(inputs, node, opset_version),

            // === Indexing ===
            "Gather" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let data = inp(inputs, 0);
                let idx = inp(inputs, 1);
                let data_shape = data.shape()?;
                let ndim = data_shape.len();
                let norm_axis = if axis < 0 { (ndim as isize + axis) as usize } else { axis as usize };
                let dim_size = data_shape[norm_axis].as_const().ok_or_else(|| Error::IrConstruction {
                    details: format!("Gather requires concrete dimension on axis {norm_axis}"),
                })? as i64;
                let zero = Tensor::const_(ConstValue::Int(0), idx.uop().dtype());
                let dim_t = Tensor::const_(ConstValue::Int(dim_size), idx.uop().dtype());
                let neg_mask = idx.try_lt(&zero)?;
                let normalized_idx = idx.try_add(&dim_t)?.where_(&neg_mask, idx)?;
                data.gather(axis, &normalized_idx)?
            }
            "GatherElements" => indexing::op_gather_elements(inputs, node)?,
            "GatherND" => indexing::op_gather_nd(inputs, node)?,
            "Trilu" => indexing::op_trilu(inputs, node)?,
            "OneHot" => indexing::op_one_hot(inputs, node)?,
            "CumSum" => indexing::op_cumsum(inputs, node)?,
            "ScatterElements" => indexing::op_scatter_elements(inputs, node)?,
            "ScatterND" => indexing::op_scatter_nd(inputs, node)?,

            // === Reductions ===
            "ReduceSum" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMean" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).mean_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMax" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).max_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceMin" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).min_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceProd" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).prod_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceSumSquare" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).square()?.sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceL1" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).try_abs()?.sum_with().axes(spec).keepdim(kd).call()?
            }
            "ReduceL2" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                let x = inp(inputs, 0);
                let orig_dtype = x.uop().dtype();
                let needs_upcast =
                    matches!(orig_dtype.scalar(), Some(ScalarDType::Float16) | Some(ScalarDType::BFloat16));
                let x = if needs_upcast { x.cast(DType::Float32)? } else { x.clone() };
                let result = x.square()?.sum_with().axes(spec).keepdim(kd).call()?.try_sqrt()?;
                if needs_upcast { result.cast(orig_dtype)? } else { result }
            }
            "ReduceLogSum" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).sum_with().axes(spec).keepdim(kd).call()?.try_log()?
            }
            "ReduceLogSumExp" => {
                let (spec, kd) = reduce_attrs(node, inputs, opset_version)?;
                inp(inputs, 0).try_exp()?.sum_with().axes(spec).keepdim(kd).call()?.try_log()?
            }
            "ArgMax" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let keepdims = get_attr_int(node, "keepdims", 1) == 1;
                let select_last = get_attr_int(node, "select_last_index", 0) == 1;
                let x = inp(inputs, 0);
                if select_last {
                    let shape = x.shape()?;
                    let na = if axis < 0 { shape.len() as isize + axis } else { axis } as usize;
                    let ds = shape[na].as_const().ok_or_else(|| Error::IrConstruction {
                        details: format!("ArgMax select_last_index needs concrete axis {na}"),
                    })? as i64;
                    Tensor::const_(ConstValue::Int(ds - 1), DType::Int64).try_sub(
                        &x.flip(&[axis])?
                            .argmax_with()
                            .axis(Some(axis))
                            .keepdim(keepdims)
                            .call()?
                            .cast(DType::Int64)?,
                    )?
                } else {
                    x.argmax_with().axis(Some(axis)).keepdim(keepdims).call()?.cast(DType::Int64)?
                }
            }
            "ArgMin" => {
                let axis = get_attr_int(node, "axis", 0) as isize;
                let keepdims = get_attr_int(node, "keepdims", 1) == 1;
                let select_last = get_attr_int(node, "select_last_index", 0) == 1;
                let x = inp(inputs, 0);
                let neg_x = x.try_neg()?;
                if select_last {
                    let shape = x.shape()?;
                    let na = if axis < 0 { shape.len() as isize + axis } else { axis } as usize;
                    let ds = shape[na].as_const().ok_or_else(|| Error::IrConstruction {
                        details: format!("ArgMin select_last_index needs concrete axis {na}"),
                    })? as i64;
                    Tensor::const_(ConstValue::Int(ds - 1), DType::Int64).try_sub(
                        &neg_x
                            .flip(&[axis])?
                            .argmax_with()
                            .axis(Some(axis))
                            .keepdim(keepdims)
                            .call()?
                            .cast(DType::Int64)?,
                    )?
                } else {
                    neg_x.argmax_with().axis(Some(axis)).keepdim(keepdims).call()?.cast(DType::Int64)?
                }
            }

            // === NN ===
            "MatMul" => inp(inputs, 0).matmul(inp(inputs, 1))?,
            "Gemm" => nn::op_gemm(inputs, node)?,
            "BatchNormalization" => return nn::op_batch_norm(inputs, node),
            "Conv" => nn::op_conv(inputs, node)?,
            "ConvTranspose" => nn::op_conv_transpose(inputs, node)?,
            "AveragePool" => nn::op_avg_pool(inputs, node)?,
            "MaxPool" => return nn::op_max_pool(inputs, node),
            "GlobalAveragePool" => {
                let x = inp(inputs, 0);
                let axes: Vec<isize> = (2..x.ndim()? as isize).collect();
                x.mean_with().axes(AxisSpec::Multiple(axes)).keepdim(true).call()?
            }
            "GlobalMaxPool" => {
                let x = inp(inputs, 0);
                let axes: Vec<isize> = (2..x.ndim()? as isize).collect();
                x.max_with().axes(AxisSpec::Multiple(axes)).keepdim(true).call()?
            }
            "LayerNormalization" => return nn::op_layer_norm(inputs, node),
            "GroupNormalization" => nn::op_group_norm(inputs, node)?,
            "InstanceNormalization" => nn::op_instance_norm(inputs, node)?,
            "Resize" => nn::op_resize(inputs, node)?,
            "DepthToSpace" => nn::op_depth_to_space(inputs, node)?,
            "SpaceToDepth" => nn::op_space_to_depth(inputs, node)?,
            "LpNormalization" => nn::op_lp_norm(inputs, node)?,
            "MeanVarianceNormalization" => nn::op_mean_variance_norm(inputs, node)?,
            "LRN" => nn::op_lrn(inputs, node)?,
            "AffineGrid" => nn::op_affine_grid(inputs, node)?,
            "NegativeLogLikelihoodLoss" => return nn::op_nll_loss(inputs, node),
            "SoftmaxCrossEntropyLoss" => return nn::op_softmax_ce_loss(inputs, node),
            "RMSNormalization" => transformer::op_rms_norm(inputs, node)?,
            "Attention" => return transformer::op_attention_onnx(inputs, node),

            // === NonZero ===
            "NonZero" => inp(inputs, 0).nonzero()?.try_transpose(0, 1)?.cast(DType::Int64)?,

            // === Einsum ===
            "Einsum" => {
                let equation = get_attr_string(node, "equation", "");
                let ops: Vec<&Tensor> = inputs.iter().filter_map(|o| o.as_ref()).collect();
                Tensor::einsum(&equation, &ops)?
            }

            // === TopK ===
            "TopK" => {
                let k = tensor_to_i64_vec(inp(inputs, 1))?[0] as usize;
                let axis = get_attr_int(node, "axis", -1) as isize;
                let largest = get_attr_int(node, "largest", 1) == 1;
                let (values, indices) = inp(inputs, 0).topk(k, axis, largest)?;
                return Ok(vec![values, indices.cast(DType::Int64)?]);
            }

            // === Identity / Constant ===
            "Identity" => inp(inputs, 0).clone(),
            "Constant" => return constant::op_constant(node).map(|t| vec![t]),

            _ => return UnsupportedOpSnafu { op: op_type.to_string(), domain: domain.to_string() }.fail(),
        };

        Ok(vec![r])
    }
}

impl Default for OpRegistry {
    fn default() -> Self {
        Self::new()
    }
}
