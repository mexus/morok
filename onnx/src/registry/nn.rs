use morok_tensor::Tensor;

use crate::error::Result;
use crate::parser::onnx::NodeProto;

use super::*;

pub(crate) fn op_gemm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let alpha = get_attr_float(node, "alpha", 1.0);
    let beta = get_attr_float(node, "beta", 1.0);
    let a = inp(inputs, 0);
    let b = inp(inputs, 1);
    let a = if get_attr_int(node, "transA", 0) == 1 { a.try_transpose(0, 1)? } else { a.clone() };
    let b = if get_attr_int(node, "transB", 0) == 1 { b.try_transpose(0, 1)? } else { b.clone() };
    let mut result = a.matmul(&b)?;
    if alpha != 1.0 {
        result = result.try_mul(&Tensor::from_slice([alpha]))?;
    }
    if let Some(c) = inputs.get(2).and_then(|o| o.as_ref()) {
        let c = if beta != 1.0 { c.try_mul(&Tensor::from_slice([beta]))? } else { c.clone() };
        result = result.try_add(&c)?;
    }
    Ok(result)
}

pub(crate) fn op_batch_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let (x, scale, bias, mean, var) = (inp(inputs, 0), inp(inputs, 1), inp(inputs, 2), inp(inputs, 3), inp(inputs, 4));
    let epsilon = get_attr_float(node, "epsilon", 1e-5);

    let var_plus_eps = var.try_add(&Tensor::from_slice([epsilon]))?;
    let invstd = var_plus_eps.try_rsqrt()?;

    Ok(x.batchnorm().scale(scale).bias(bias).mean(mean).invstd(&invstd).call()?)
}

pub(crate) fn op_conv(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let ks: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad = get_attr_string(node, "auto_pad", "NOTSET");
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    Ok(inp(inputs, 0)
        .conv()
        .weight(inp(inputs, 1))
        .maybe_bias(inputs.get(2).and_then(|o| o.as_ref()))
        .auto_pad(&auto_pad)
        .group(get_attr_int(node, "group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_conv_transpose(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let ks: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let op: Vec<usize> = get_attr_ints(node, "output_padding").iter().map(|&p| p as usize).collect();
    let os = get_attr_ints(node, "output_shape");
    let auto_pad = get_attr_string(node, "auto_pad", "NOTSET");
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    Ok(inp(inputs, 0)
        .conv_transpose()
        .weight(inp(inputs, 1))
        .maybe_bias(inputs.get(2).and_then(|o| o.as_ref()))
        .auto_pad(&auto_pad)
        .group(get_attr_int(node, "group", 1) as usize)
        .maybe_kernel_shape((!ks.is_empty()).then_some(ks.as_slice()))
        .maybe_pads(non_empty_i64(&pads))
        .maybe_output_shape(non_empty_i64(&os))
        .maybe_output_padding((!op.is_empty()).then_some(op.as_slice()))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_avg_pool(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let kernel: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad = get_attr_string(node, "auto_pad", "NOTSET");
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    Ok(inp(inputs, 0)
        .avg_pool()
        .kernel_shape(&kernel)
        .auto_pad(&auto_pad)
        .ceil_mode(get_attr_int(node, "ceil_mode", 0) == 1)
        .count_include_pad(get_attr_int(node, "count_include_pad", 0) == 1)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?)
}

pub(crate) fn op_max_pool(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let kernel: Vec<usize> = get_attr_ints(node, "kernel_shape").iter().map(|&k| k as usize).collect();
    let auto_pad = get_attr_string(node, "auto_pad", "NOTSET");
    let pads = get_attr_ints(node, "pads");
    let strides = get_attr_ints(node, "strides");
    let dilations = get_attr_ints(node, "dilations");
    let (values, indices) = inp(inputs, 0)
        .max_pool()
        .kernel_shape(&kernel)
        .auto_pad(&auto_pad)
        .ceil_mode(get_attr_int(node, "ceil_mode", 0) == 1)
        .storage_order(get_attr_int(node, "storage_order", 0) as usize)
        .maybe_pads(non_empty_i64(&pads))
        .maybe_strides(non_empty_i64(&strides))
        .maybe_dilations(non_empty_i64(&dilations))
        .call()?;
    Ok(vec![values, indices])
}

pub(crate) fn op_layer_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Vec<Tensor>> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inputs.get(2).and_then(|o| o.as_ref());
    let axis = get_attr_int(node, "axis", -1) as isize;
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    let (mut output, mean, inv_std_dev) = x.layernorm_with_stats(axis, epsilon)?;
    output = output.try_mul(scale)?;
    if let Some(bias) = bias {
        output = output.try_add(bias)?;
    }
    Ok(vec![output, mean, inv_std_dev])
}

pub(crate) fn op_group_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inp(inputs, 2);
    let num_groups = get_attr_int(node, "num_groups", 1) as usize;
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    Ok(x.group_norm(scale, bias, num_groups, epsilon)?)
}

pub(crate) fn op_instance_norm(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scale = inp(inputs, 1);
    let bias = inp(inputs, 2);
    let epsilon = get_attr_float(node, "epsilon", 1e-5) as f64;
    let num_channels = x.shape()?[1].as_const().unwrap();
    Ok(x.group_norm(scale, bias, num_channels, epsilon)?)
}

pub(crate) fn op_resize(inputs: &[Option<Tensor>], node: &NodeProto) -> Result<Tensor> {
    let x = inp(inputs, 0);
    let scales: Option<Vec<f64>> = inputs
        .get(2)
        .and_then(|o| o.as_ref())
        .filter(|t| t.numel().unwrap_or(0) > 0)
        .map(tensor_to_f64_vec)
        .transpose()?;
    let sizes: Option<Vec<usize>> = inputs
        .get(3)
        .and_then(|o| o.as_ref())
        .filter(|t| t.numel().unwrap_or(0) > 0)
        .map(|t| tensor_to_i64_vec(t).map(|v| v.iter().map(|&x| x as usize).collect()))
        .transpose()?;
    let mode = get_attr_string(node, "mode", "nearest");
    let coord_mode = get_attr_string(node, "coordinate_transformation_mode", "half_pixel");
    let nearest_mode = get_attr_string(node, "nearest_mode", "round_prefer_floor");
    let cubic_coeff = get_attr_float(node, "cubic_coeff_a", -0.75) as f64;
    let exclude_outside = get_attr_int(node, "exclude_outside", 0) != 0;
    let policy = get_attr_string(node, "keep_aspect_ratio_policy", "stretch");
    let axes_attr = get_attr_ints(node, "axes");
    let axes: Option<Vec<usize>> = if axes_attr.is_empty() {
        None
    } else {
        let ndim = x.ndim()?;
        Some(axes_attr.iter().map(|&a| if a < 0 { (ndim as i64 + a) as usize } else { a as usize }).collect())
    };
    Ok(x.resize()
        .maybe_scales(scales.as_deref())
        .maybe_sizes(sizes.as_deref())
        .mode(&mode)
        .coordinate_transformation_mode(&coord_mode)
        .nearest_mode(&nearest_mode)
        .cubic_coeff_a(cubic_coeff)
        .exclude_outside(exclude_outside)
        .keep_aspect_ratio_policy(&policy)
        .maybe_axes(axes.as_deref())
        .call()?)
}
