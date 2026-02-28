//! Neural network operations: convolution, pooling, normalization.

use bon::bon;
use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};

use crate::Tensor;
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

// =========================================================================
// Padding helpers (moved from ONNX crate — universal padding concepts)
// =========================================================================

/// Convert flat pads `[begin0, begin1, ..., end0, end1, ...]` to `[(begin0, end0), ...]`.
pub fn flat_pads_to_pairs(pads: &[i64]) -> Vec<(isize, isize)> {
    let n = pads.len() / 2;
    (0..n).map(|i| (pads[i] as isize, pads[i + n] as isize)).collect()
}

/// Split total padding per dimension into `[begin0, begin1, ..., end0, end1, ...]`
/// based on auto_pad mode (SAME_UPPER: more padding at end; SAME_LOWER: more at begin).
pub fn auto_pad_split(total_pads: &[isize], auto_pad: &str) -> Vec<isize> {
    let first: Vec<isize> = if auto_pad == "SAME_UPPER" {
        total_pads.iter().map(|&p| p / 2).collect()
    } else {
        total_pads.iter().map(|&p| p - p / 2).collect()
    };
    let mut result = first.clone();
    result.extend(total_pads.iter().zip(&first).map(|(p, f)| p - f));
    result
}

/// Resolve auto_pad + flat pads into `[(begin, end), ...]` pairs.
/// Handles VALID, NOTSET, SAME_UPPER, SAME_LOWER.
pub fn resolve_pool_pads(
    input_spatial: &[usize],
    pads: &[i64],
    kernel: &[usize],
    _dilations: &[usize],
    strides: &[usize],
    auto_pad: &str,
) -> Vec<(isize, isize)> {
    let n = kernel.len();
    match auto_pad {
        "VALID" => vec![(0, 0); n],
        "NOTSET" | "" => {
            if pads.is_empty() {
                vec![(0, 0); n]
            } else if pads.len() == n * 2 {
                flat_pads_to_pairs(pads)
            } else {
                pads.iter().map(|&p| (p as isize, p as isize)).collect()
            }
        }
        "SAME_UPPER" | "SAME_LOWER" => {
            let mut pairs = Vec::with_capacity(n);
            for i in 0..n {
                let o = (input_spatial[i] as isize - 1) / strides[i] as isize + 1;
                let total_pad = ((o - 1) * strides[i] as isize + kernel[i] as isize - input_spatial[i] as isize).max(0);
                let (begin, end) = if auto_pad == "SAME_UPPER" {
                    (total_pad / 2, total_pad - total_pad / 2)
                } else {
                    (total_pad - total_pad / 2, total_pad / 2)
                };
                pairs.push((begin, end));
            }
            pairs
        }
        _ => vec![(0, 0); n],
    }
}

impl Tensor {
    /// Pad with a custom fill value. Delegates to try_pad when value == 0.
    pub fn try_pad_value(&self, padding: &[(isize, isize)], value: f64) -> Result<Tensor> {
        if value == 0.0 {
            return self.try_pad(padding);
        }
        // Tinygrad approach: x.pad(0) + ones_pad.where(0, fill_value)
        // ADD-based avoids fragile nested WHERE conditions that can evaluate to -inf.
        let dtype = self.uop().dtype();
        let sdtype = dtype.scalar().expect("pad_value requires scalar dtype");
        let padded = self.try_pad(padding)?;
        let ones = Tensor::new(UOp::const_(dtype.clone(), ConstValue::one(sdtype)));
        let ones = ones.broadcast_to(&self.shape()?)?;
        let ones_padded = ones.try_pad(padding)?;
        let zero_cmp = Tensor::new(UOp::const_(dtype.clone(), ConstValue::zero(sdtype)));
        let mask = ones_padded.try_ne(&zero_cmp)?;
        let zero_val = Tensor::new(UOp::const_(dtype.clone(), ConstValue::zero(sdtype)));
        let fill_val = Tensor::new(UOp::const_(dtype, ConstValue::Float(value)));
        // mask ? zero : fill_value  →  data region gets 0, pad region gets fill_value
        let fill_term = zero_val.where_(&mask, &fill_val)?;
        padded.try_add(&fill_term)
    }

    /// Sliding window extraction via shape manipulation (Tinygrad's _pool).
    /// Input: (..., *spatial)  Output: (..., *out_spatial, *kernel)
    pub(crate) fn pool(&self, kernel: &[usize], stride: &[usize], dilation: &[usize]) -> Result<Tensor> {
        let shape = self.shape()?;
        let ndim = shape.len();
        let n_spatial = kernel.len();
        let n_batch = ndim - n_spatial;

        assert!(ndim >= n_spatial, "can't pool {ndim}D with {n_spatial}D kernel");
        assert_eq!(kernel.len(), stride.len(), "kernel/stride length mismatch");
        assert_eq!(kernel.len(), dilation.len(), "kernel/dilation length mismatch");

        let i_: Vec<usize> = (0..n_spatial)
            .map(|j| shape[n_batch + j].as_const().expect("pool requires concrete spatial dims"))
            .collect();

        for j in 0..n_spatial {
            assert!(
                dilation[j] * (kernel[j] - 1) < i_[j],
                "kernel size {} (dilated {}) > input size {}",
                kernel[j],
                dilation[j] * (kernel[j] - 1) + 1,
                i_[j]
            );
        }

        let o_: Vec<usize> =
            (0..n_spatial).map(|j| usize::div_ceil(i_[j] - dilation[j] * (kernel[j] - 1), stride[j])).collect();

        let f_: Vec<usize> =
            (0..n_spatial).map(|j| 1.max(usize::div_ceil(o_[j] * stride[j] - dilation[j], i_[j]))).collect();

        // Step 1: repeat
        let mut repeats: Vec<usize> = vec![1; n_batch];
        for j in 0..n_spatial {
            repeats.push(usize::div_ceil(kernel[j] * (i_[j] * f_[j] + dilation[j]), i_[j]));
        }
        let mut x = self.repeat(&repeats)?;

        // Step 2: shrink to exact needed size
        let batch_dims: Vec<isize> = (0..n_batch).map(|d| x.shape().unwrap()[d].as_const().unwrap() as isize).collect();
        let mut shrink_ranges: Vec<(isize, isize)> = batch_dims.iter().map(|&d| (0, d)).collect();
        for j in 0..n_spatial {
            shrink_ranges.push((0, (kernel[j] * (i_[j] * f_[j] + dilation[j])) as isize));
        }
        x = x.try_shrink(&shrink_ranges)?;

        // Step 3: reshape to interleave kernel and spatial dims
        let mut reshape_dims: Vec<isize> = batch_dims.clone();
        for j in 0..n_spatial {
            reshape_dims.push(kernel[j] as isize);
            reshape_dims.push((i_[j] * f_[j] + dilation[j]) as isize);
        }
        x = x.try_reshape(&reshape_dims)?;

        // Step 4: shrink for stride
        let mut shrink_ranges: Vec<(isize, isize)> = batch_dims.iter().map(|&d| (0, d)).collect();
        for j in 0..n_spatial {
            shrink_ranges.push((0, kernel[j] as isize));
            shrink_ranges.push((0, (o_[j] * stride[j]) as isize));
        }
        x = x.try_shrink(&shrink_ranges)?;

        // Step 5: reshape to separate stride: K_j, o_j, S_j
        let mut reshape_dims: Vec<isize> = batch_dims.clone();
        for j in 0..n_spatial {
            reshape_dims.push(kernel[j] as isize);
            reshape_dims.push(o_[j] as isize);
            reshape_dims.push(stride[j] as isize);
        }
        x = x.try_reshape(&reshape_dims)?;

        // Step 6: shrink stride dim to 1
        let mut shrink_ranges: Vec<(isize, isize)> = batch_dims.iter().map(|&d| (0, d)).collect();
        for j in 0..n_spatial {
            shrink_ranges.push((0, kernel[j] as isize));
            shrink_ranges.push((0, o_[j] as isize));
            shrink_ranges.push((0, 1));
        }
        x = x.try_shrink(&shrink_ranges)?;

        // Step 7: reshape to collapse stride dim
        let mut reshape_dims: Vec<isize> = batch_dims.clone();
        for j in 0..n_spatial {
            reshape_dims.push(kernel[j] as isize);
            reshape_dims.push(o_[j] as isize);
        }
        x = x.try_reshape(&reshape_dims)?;

        // Step 8: permute to move kernel dims to end
        let mut perm: Vec<isize> = (0..n_batch as isize).collect();
        for j in 0..n_spatial {
            perm.push(n_batch as isize + j as isize * 2 + 1); // output spatial
        }
        for j in 0..n_spatial {
            perm.push(n_batch as isize + j as isize * 2); // kernel
        }
        x = x.try_permute(&perm)?;

        Ok(x)
    }
}

#[bon]
impl Tensor {
    /// N-d convolution. Input (N,Cin,*spatial), Weight (Cout,Cin/groups,*kernel).
    #[builder]
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default = 1)] groups: usize,
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
    ) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let w_shape = weight.shape()?;

        let bs = x_shape[0].as_const().expect("batch dim must be concrete");
        let cin_ = x_shape[1].as_const().expect("channel dim must be concrete");
        let cout = w_shape[0].as_const().expect("cout must be concrete");
        let cin = w_shape[1].as_const().expect("cin/g must be concrete");

        let hw: Vec<usize> = w_shape[2..].iter().map(|s| s.as_const().expect("kernel dim must be concrete")).collect();
        let n_spatial = hw.len();

        assert_eq!(x_shape.len(), w_shape.len(), "input and weight must have same ndim");
        assert_eq!(groups * cin, cin_, "groups*cin/g ({}) != input channels ({cin_})", groups * cin);

        let default_ones: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(&default_ones);
        let dilation = dilation.unwrap_or(&default_ones);
        let no_padding: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_padding);

        let mut x = self.clone();
        if padding.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); 2];
            full_pad.extend_from_slice(padding);
            x = x.try_pad(&full_pad)?;
        }

        x = x.pool(&hw, stride, dilation)?;

        let oyx: Vec<usize> = {
            let xs = x.shape()?;
            (0..n_spatial).map(|j| xs[2 + j].as_const().unwrap()).collect()
        };

        let rcout = cout / groups;

        // Reshape: (bs, groups, cin, 1, *oyx, *hw)
        let mut reshape_dims: Vec<isize> = vec![bs as isize, groups as isize, cin as isize, 1];
        reshape_dims.extend(oyx.iter().map(|&o| o as isize));
        reshape_dims.extend(hw.iter().map(|&k| k as isize));
        x = x.try_reshape(&reshape_dims)?;

        // Expand: (bs, groups, cin, rcout, *oyx, *hw)
        let mut expand_dims: Vec<isize> = vec![bs as isize, groups as isize, cin as isize, rcout as isize];
        expand_dims.extend(oyx.iter().map(|&o| o as isize));
        expand_dims.extend(hw.iter().map(|&k| k as isize));
        x = x.try_expand(&expand_dims)?;

        // Permute: (bs, groups, rcout, *oyx, cin, *hw)
        let mut perm: Vec<isize> = vec![0, 1, 3];
        for j in 0..n_spatial {
            perm.push(4 + j as isize);
        }
        perm.push(2);
        for j in 0..n_spatial {
            perm.push((4 + n_spatial + j) as isize);
        }
        x = x.try_permute(&perm)?;

        // Reshape weight: (1, groups, rcout, *[1]*n_spatial, cin, *hw)
        let mut w_reshape: Vec<isize> = vec![1, groups as isize, rcout as isize];
        w_reshape.extend(std::iter::repeat_n(1isize, n_spatial));
        w_reshape.push(cin as isize);
        w_reshape.extend(hw.iter().map(|&k| k as isize));
        let w = weight.try_reshape(&w_reshape)?;

        x = x.try_mul(&w)?;

        // Sum over last (1 + n_spatial) dims
        let total_dims = x.ndim()?;
        let reduce_axes: Vec<isize> = (0..(1 + n_spatial)).map(|i| (total_dims - 1 - i) as isize).collect();
        x = x.sum_with().axes(AxisSpec::Multiple(reduce_axes)).keepdim(true).call()?;

        // Reshape to (bs, cout, *oyx)
        let mut final_shape: Vec<isize> = vec![bs as isize, cout as isize];
        final_shape.extend(oyx.iter().map(|&o| o as isize));
        x = x.try_reshape(&final_shape)?;

        if let Some(bias) = bias {
            let mut bias_shape: Vec<isize> = vec![1, cout as isize];
            bias_shape.extend(std::iter::repeat_n(1isize, n_spatial));
            let bias = bias.try_reshape(&bias_shape)?;
            x = x.try_add(&bias)?;
        }

        Ok(x)
    }

    /// Transposed convolution.
    #[builder]
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default = 1)] groups: usize,
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        output_padding: Option<&[usize]>,
    ) -> Result<Tensor> {
        let w_shape = weight.shape()?;
        let hw: Vec<usize> = w_shape[2..].iter().map(|s| s.as_const().expect("kernel dim must be concrete")).collect();
        let n_spatial = hw.len();

        let default_ones: Vec<usize> = vec![1; n_spatial];
        let default_zeros: Vec<usize> = vec![0; n_spatial];
        let default_no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let stride = stride.unwrap_or(&default_ones);
        let dilation = dilation.unwrap_or(&default_ones);
        let padding = padding.unwrap_or(&default_no_pad);
        let output_padding = output_padding.unwrap_or(&default_zeros);

        let cout_in = w_shape[0].as_const().unwrap();
        let cin_g = w_shape[1].as_const().unwrap();
        let rcout = cout_in / groups;

        // Reshape to (groups, rcout, cin_g, *HW)
        let mut unflatten_shape: Vec<isize> = vec![groups as isize, rcout as isize, cin_g as isize];
        unflatten_shape.extend(hw.iter().map(|&k| k as isize));
        let mut w = weight.try_reshape(&unflatten_shape)?;

        // Transpose dim 1 and 2: (groups, cin_g, rcout, *HW)
        w = w.try_transpose(1, 2)?;

        // Flip kernel dims
        let flip_axes: Vec<isize> = (3..(3 + n_spatial) as isize).collect();
        w = w.flip(&flip_axes)?;

        // Flatten back: (groups * cin_g, rcout, *HW)
        let mut flat_shape: Vec<isize> = vec![(groups * cin_g) as isize, rcout as isize];
        flat_shape.extend(hw.iter().map(|&k| k as isize));
        w = w.try_reshape(&flat_shape)?;

        // Handle stride > 1: interleave zeros
        let mut x = self.clone();
        if stride.iter().any(|&s| s > 1) {
            for (j, &s) in stride.iter().enumerate() {
                if s <= 1 {
                    continue;
                }
                let cur_shape = x.shape()?;
                let spatial_idx = 2 + j;
                let k = cur_shape[spatial_idx].as_const().unwrap();

                // insert dim of 1 after spatial dim
                let mut rshape: Vec<isize> = cur_shape.iter().map(|d| d.as_const().unwrap() as isize).collect();
                rshape.insert(spatial_idx + 1, 1);
                x = x.try_reshape(&rshape)?;

                // pad: (0, s-1) on the inserted dim
                let mut pad: Vec<(isize, isize)> = vec![(0, 0); rshape.len()];
                pad[spatial_idx + 1] = (0, (s - 1) as isize);
                x = x.try_pad(&pad)?;

                // merge spatial_idx and spatial_idx+1
                let cur_shape = x.shape()?;
                let mut rshape: Vec<isize> = cur_shape.iter().map(|d| d.as_const().unwrap() as isize).collect();
                let merged = rshape[spatial_idx] * rshape[spatial_idx + 1];
                rshape[spatial_idx] = merged;
                rshape.remove(spatial_idx + 1);
                x = x.try_reshape(&rshape)?;

                // shrink: remove trailing (s-1) from this dim
                let cur_shape = x.shape()?;
                let new_size = k * s - (s - 1);
                let mut ranges: Vec<(isize, isize)> =
                    cur_shape.iter().map(|d| (0, d.as_const().unwrap() as isize)).collect();
                ranges[spatial_idx] = (0, new_size as isize);
                x = x.try_shrink(&ranges)?;
            }
        }

        // Compute transposed padding
        let conv_padding: Vec<(isize, isize)> = (0..n_spatial)
            .map(|j| {
                let pb = padding[j].0;
                let pa = padding[j].1;
                let begin = (hw[j] as isize - 1) * dilation[j] as isize - pb;
                let end = (hw[j] as isize - 1) * dilation[j] as isize - pa + output_padding[j] as isize;
                (begin, end)
            })
            .collect();

        x.conv2d().weight(&w).groups(groups).maybe_bias(bias).dilation(dilation).padding(&conv_padding).call()
    }

    /// Adjust padding for ceil_mode output sizes.
    /// Per arXiv:1603.07285 section 5.1, relationship 15.
    fn apply_ceil_mode(
        padding: &[(isize, isize)],
        input_spatial: &[usize],
        kernel: &[usize],
        stride: &[usize],
        dilation: &[usize],
    ) -> Vec<(isize, isize)> {
        let n = kernel.len();
        let grouped: Vec<(isize, isize)> = padding.to_vec();
        let mut ceil_pads = grouped.clone();
        for i in 0..n {
            let padded = input_spatial[i] as isize + grouped[i].0 + grouped[i].1;
            let eff_k = (dilation[i] * (kernel[i] - 1) + 1) as isize;
            let s = stride[i] as isize;
            // Output with ceil: ceildiv(padded - eff_k, s) + 1
            let o_ceil = (padded - eff_k + s - 1) / s + 1;
            // Output without ceil: (padded - eff_k) / s + 1
            let o_floor = (padded - eff_k) / s + 1;
            if o_ceil > o_floor {
                let last_start = s * (o_ceil - 1);
                let extra = last_start + eff_k - padded;
                // Decrease when last window starts past real data + pad_before
                let correction = (last_start - (grouped[i].0 + input_spatial[i] as isize - 1)).max(0);
                ceil_pads[i].1 += extra - correction;
            }
        }
        ceil_pads
    }

    /// Average pooling.
    #[builder]
    pub fn avg_pool2d(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = true)] count_include_pad: bool,
        #[builder(default = false)] ceil_mode: bool,
    ) -> Result<Tensor> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        let shape = self.shape()?;
        let n_batch = shape.len() - n_spatial;
        let input_spatial: Vec<usize> = (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap()).collect();

        let reg_pads = padding.to_vec();
        let ceil_pads = if ceil_mode {
            Self::apply_ceil_mode(&reg_pads, &input_spatial, kernel_size, stride, dilation)
        } else {
            reg_pads.clone()
        };

        let pad_and_pool = |x: &Tensor, pads: &[(isize, isize)]| -> Result<Tensor> {
            let mut out = x.clone();
            if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
                let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
                full_pad.extend_from_slice(pads);
                out = out.try_pad(&full_pad)?;
            }
            out.pool(kernel_size, stride, dilation)
        };

        if !count_include_pad {
            // Path 1: sum(pool(x, pads)) / sum(pool(ones, pads))
            let pads = if ceil_mode { &ceil_pads } else { &reg_pads };
            let pooled = pad_and_pool(self, pads)?;
            let sum_x = pooled.sum_with().axes(axes.clone()).keepdim(false).call()?;
            let ones = Tensor::new(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
            let ones = ones.broadcast_to(&self.shape()?)?;
            let pooled_ones = pad_and_pool(&ones, pads)?;
            let sum_ones = pooled_ones.sum_with().axes(axes).keepdim(false).call()?;
            return sum_x.try_div(&sum_ones);
        }

        if !ceil_mode {
            // Path 2: count_include_pad=true, ceil_mode=false → simple mean
            let pooled = pad_and_pool(self, &reg_pads)?;
            return pooled.mean(axes);
        }

        // Path 3: count_include_pad=true, ceil_mode=true
        // Ceil-extra padding does NOT count in the average.
        // Divide by pool(ones_on_regularly_padded, extra_ceil_pads).sum
        let pooled = pad_and_pool(self, &ceil_pads)?;
        let sum_x = pooled.sum_with().axes(axes.clone()).keepdim(false).call()?;

        // Build ones on regularly-padded input, then pool with extra ceil pads
        let ones = Tensor::new(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
        let mut ones_reg = ones.broadcast_to(&self.shape()?)?;
        if reg_pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&reg_pads);
            ones_reg = ones_reg.try_pad(&full_pad)?;
        }
        let extra_pads: Vec<(isize, isize)> =
            ceil_pads.iter().zip(reg_pads.iter()).map(|(c, r)| (c.0 - r.0, c.1 - r.1)).collect();
        let mut ones_extra = ones_reg;
        if extra_pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&extra_pads);
            ones_extra = ones_extra.try_pad(&full_pad)?;
        }
        let pooled_ones = ones_extra.pool(kernel_size, stride, dilation)?;
        let sum_ones = pooled_ones.sum_with().axes(axes).keepdim(false).call()?;
        sum_x.try_div(&sum_ones)
    }

    /// Max pooling.
    #[builder]
    pub fn max_pool2d(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = false)] ceil_mode: bool,
    ) -> Result<Tensor> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let pads = if ceil_mode {
            let shape = self.shape()?;
            let n_batch = shape.len() - n_spatial;
            let input_spatial: Vec<usize> = (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap()).collect();
            Self::apply_ceil_mode(padding, &input_spatial, kernel_size, stride, dilation)
        } else {
            padding.to_vec()
        };

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        let mut x = self.clone();
        if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); self.ndim()? - n_spatial];
            full_pad.extend_from_slice(&pads);
            x = x.try_pad_value(&full_pad, f64::NEG_INFINITY)?;
        }

        let pooled = x.pool(kernel_size, stride, dilation)?;
        pooled.max(axes)
    }

    /// Max pool returning both values and indices.
    #[builder]
    pub fn max_pool2d_with_indices(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = false)] ceil_mode: bool,
    ) -> Result<(Tensor, Tensor)> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let shape = self.shape()?;
        let n_batch = shape.len() - n_spatial;

        let pads = if ceil_mode {
            let input_spatial: Vec<usize> = (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap()).collect();
            Self::apply_ceil_mode(padding, &input_spatial, kernel_size, stride, dilation)
        } else {
            padding.to_vec()
        };

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes.clone());

        // Pool the data with -inf padding
        let mut x = self.clone();
        if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&pads);
            x = x.try_pad_value(&full_pad, f64::NEG_INFINITY)?;
        }
        let pooled = x.pool(kernel_size, stride, dilation)?;
        let values = pooled.max_with().axes(axes.clone()).keepdim(false).call()?;

        // Compute indices using reverse arange trick (Tinygrad approach)
        let spatial_sz: usize = (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap()).product();

        // Create reverse arange: spatial_sz, spatial_sz-1, ..., 1
        let idx_range = Tensor::arange(spatial_sz as i64, Some(0), Some(-1))?;
        // Reshape to match spatial dims
        let spatial_dims: Vec<isize> =
            (0..n_spatial).map(|j| shape[n_batch + j].as_const().unwrap() as isize).collect();
        let mut idx_shape: Vec<isize> = vec![1; n_batch];
        idx_shape.extend_from_slice(&spatial_dims);
        let idx = idx_range.try_reshape(&idx_shape)?;

        // Pad and pool the index tensor identically
        let mut idx_padded = idx;
        if pads.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); n_batch];
            full_pad.extend_from_slice(&pads);
            idx_padded = idx_padded.try_pad(&full_pad)?;
        }
        let pooled_idx = idx_padded.pool(kernel_size, stride, dilation)?;

        // Create mask: pooled == pooled.max(keepdim=True)
        let pooled_max = pooled.max_with().axes(axes.clone()).keepdim(true).call()?;
        let mask = pooled.try_eq(&pooled_max)?;

        // Multiply mask * pooled_indices, take max → first-occurrence (via reverse index)
        let masked_idx = mask.cast(DType::Int32)?.try_mul(&pooled_idx)?;
        let max_idx = masked_idx.max_with().axes(axes).keepdim(false).call()?;

        // spatial_sz - max_idx → convert reverse index to forward index
        let sz_t = Tensor::const_(ConstValue::Int(spatial_sz as i64), DType::Int32);
        let indices = sz_t.try_sub(&max_idx)?;

        Ok((values, indices))
    }

    /// Layer normalization over axes [axis..ndim). Casts to f32 internally.
    pub fn layernorm(&self, axis: isize, eps: f64) -> Result<Tensor> {
        let (normed, _, _) = self.layernorm_with_stats(axis, eps)?;
        Ok(normed)
    }

    /// Layer normalization returning `(normalized, mean, inv_std_dev)`.
    /// Computes in f32, casts normalized output back to original dtype.
    /// Mean and inv_std_dev remain in f32 (matching ONNX stash_type=1 semantics).
    pub fn layernorm_with_stats(&self, axis: isize, eps: f64) -> Result<(Tensor, Tensor, Tensor)> {
        let ndim = self.ndim()?;
        let norm_axis = Tensor::normalize_axis(axis, ndim)?;
        let axes: Vec<isize> = (norm_axis..ndim).map(|a| a as isize).collect();
        let axes_spec = AxisSpec::Multiple(axes);

        let original_dtype = self.uop().dtype();
        let x32 = if original_dtype != DType::Float32 { self.cast(DType::Float32)? } else { self.clone() };

        let mean = x32.mean_with().axes(axes_spec.clone()).keepdim(true).call()?;
        let centered = x32.try_sub(&mean)?;
        let variance = centered.square()?.mean_with().axes(axes_spec).keepdim(true).call()?;
        let eps_t = Tensor::new(UOp::const_(DType::Float32, ConstValue::Float(eps)));
        let inv_std = variance.try_add(&eps_t)?.try_rsqrt()?;
        let normalized = centered.try_mul(&inv_std)?;

        let normalized = if original_dtype != DType::Float32 { normalized.cast(original_dtype)? } else { normalized };
        Ok((normalized, mean, inv_std))
    }

    /// Group normalization: reshape → layernorm → scale + bias.
    /// Matches Tinygrad's ONNX `GroupNormalization` pattern.
    pub fn group_norm(&self, scale: &Tensor, bias: &Tensor, num_groups: usize, eps: f64) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let ndim = x_shape.len();
        let batch = x_shape[0].as_const().unwrap();

        // Reshape to (batch, num_groups, -1), cast to f32 before layernorm
        let reshaped = self.try_reshape(&[batch as isize, num_groups as isize, -1])?;
        let reshaped = if reshaped.uop().dtype() != DType::Float32 { reshaped.cast(DType::Float32)? } else { reshaped };
        let normed = reshaped.layernorm(-1, eps)?;
        // Cast back and reshape to original
        let normed = if self.uop().dtype() != DType::Float32 { normed.cast(self.uop().dtype())? } else { normed };
        let orig_shape: Vec<isize> = x_shape.iter().map(|s| s.as_const().unwrap() as isize).collect();
        let normed = normed.try_reshape(&orig_shape)?;

        // Scale and bias: reshape to (1, C, 1, 1, ...)
        let mut sb_shape: Vec<isize> = vec![1, -1];
        sb_shape.extend(std::iter::repeat_n(1isize, ndim - 2));
        let scale = scale.try_reshape(&sb_shape)?;
        let bias = bias.try_reshape(&sb_shape)?;
        normed.try_mul(&scale)?.try_add(&bias)
    }
}

// =========================================================================
// Higher-level building blocks (Tinygrad-style wrappers)
// =========================================================================

#[bon]
impl Tensor {
    /// Convolution with ONNX-style parameters. Wraps `conv2d`.
    #[builder]
    pub fn conv(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default = "NOTSET")] auto_pad: &str,
        #[builder(default = 1)] group: usize,
        kernel_shape: Option<&[usize]>,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let w_shape = weight.shape()?;
        let kernel: Vec<usize> = kernel_shape
            .map(|ks| ks.to_vec())
            .unwrap_or_else(|| w_shape[2..].iter().map(|s| s.as_const().unwrap()).collect());
        let n = kernel.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
        let empty_pads: Vec<i64> = vec![];
        let padding =
            resolve_pool_pads(&input_spatial, pads.unwrap_or(&empty_pads), &kernel, &dilations_u, &strides_u, auto_pad);
        self.conv2d()
            .weight(weight)
            .maybe_bias(bias)
            .groups(group)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .call()
    }

    /// Transposed convolution with ONNX-style parameters. Wraps `conv_transpose2d`.
    #[builder]
    pub fn conv_transpose(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        #[builder(default = "NOTSET")] auto_pad: &str,
        #[builder(default = 1)] group: usize,
        kernel_shape: Option<&[usize]>,
        pads: Option<&[i64]>,
        output_shape: Option<&[i64]>,
        output_padding: Option<&[usize]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let w_shape = weight.shape()?;
        let kernel: Vec<usize> = kernel_shape
            .map(|ks| ks.to_vec())
            .unwrap_or_else(|| w_shape[2..].iter().map(|s| s.as_const().unwrap()).collect());
        let n = kernel.len();
        let x_shape = self.shape()?;
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let output_padding_u: Vec<usize> = output_padding.map(|op| op.to_vec()).unwrap_or_else(|| vec![0; n]);

        // 3-path padding resolution (matches Tinygrad's ConvTranspose)
        let mut pads_resolved: Option<Vec<isize>> = None;

        // Path 1: output_shape provided → derive total pads, apply auto_pad
        if let Some(os) = output_shape {
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    (strides_u[i] * (input_spatial[i] - 1) + output_padding_u[i] + (kernel[i] - 1) * dilations_u[i] + 1)
                        as isize
                        - os[i] as isize
                })
                .collect();
            pads_resolved = Some(auto_pad_split(&total_pads, auto_pad));
        }

        // Path 2: no explicit pads → derive from default output_shape
        if pads_resolved.is_none() && pads.is_none_or(|p| p.is_empty()) {
            let default_out: Vec<usize> = (0..n).map(|i| input_spatial[i] * strides_u[i]).collect();
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    (strides_u[i] * (input_spatial[i] - 1) + output_padding_u[i] + (kernel[i] - 1) * dilations_u[i] + 1)
                        as isize
                        - default_out[i] as isize
                })
                .collect();
            pads_resolved = Some(if auto_pad != "NOTSET" && !auto_pad.is_empty() {
                auto_pad_split(&total_pads, auto_pad)
            } else {
                vec![0; n * 2]
            });
        }

        // Path 3: explicit pads provided
        let padding: Vec<(isize, isize)> = if let Some(flat) = pads_resolved {
            let half = flat.len() / 2;
            (0..half).map(|i| (flat[i], flat[i + half])).collect()
        } else {
            flat_pads_to_pairs(pads.unwrap())
        };

        self.conv_transpose2d()
            .weight(weight)
            .maybe_bias(bias)
            .groups(group)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .output_padding(&output_padding_u)
            .call()
    }

    /// Average pooling with ONNX-style parameters. Wraps `avg_pool2d`.
    #[builder]
    pub fn avg_pool(
        &self,
        kernel_shape: &[usize],
        #[builder(default = "NOTSET")] auto_pad: &str,
        #[builder(default = false)] ceil_mode: bool,
        #[builder(default = false)] count_include_pad: bool,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<Tensor> {
        let n = kernel_shape.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
        let empty_pads: Vec<i64> = vec![];
        let padding = resolve_pool_pads(
            &input_spatial,
            pads.unwrap_or(&empty_pads),
            kernel_shape,
            &dilations_u,
            &strides_u,
            auto_pad,
        );
        self.avg_pool2d()
            .kernel_size(kernel_shape)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .ceil_mode(ceil_mode)
            .count_include_pad(count_include_pad)
            .call()
    }

    /// Max pooling with ONNX-style parameters. Always returns `(values, indices)`. Wraps `max_pool2d_with_indices`.
    #[builder]
    pub fn max_pool(
        &self,
        kernel_shape: &[usize],
        #[builder(default = "NOTSET")] auto_pad: &str,
        #[builder(default = false)] ceil_mode: bool,
        #[builder(default = 0)] storage_order: usize,
        pads: Option<&[i64]>,
        strides: Option<&[i64]>,
        dilations: Option<&[i64]>,
    ) -> Result<(Tensor, Tensor)> {
        let n = kernel_shape.len();
        let strides_u: Vec<usize> =
            strides.map(|s| s.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let dilations_u: Vec<usize> =
            dilations.map(|d| d.iter().map(|&v| v as usize).collect()).unwrap_or_else(|| vec![1; n]);
        let x_shape = self.shape()?;
        let input_spatial: Vec<usize> = x_shape[2..].iter().map(|s| s.as_const().unwrap()).collect();
        let empty_pads: Vec<i64> = vec![];
        let padding = resolve_pool_pads(
            &input_spatial,
            pads.unwrap_or(&empty_pads),
            kernel_shape,
            &dilations_u,
            &strides_u,
            auto_pad,
        );
        let (values, indices) = self
            .max_pool2d_with_indices()
            .kernel_size(kernel_shape)
            .stride(&strides_u)
            .dilation(&dilations_u)
            .padding(&padding)
            .ceil_mode(ceil_mode)
            .call()?;
        let indices = if storage_order == 1 {
            indices.try_transpose(-2, -1)?.cast(DType::Int64)?
        } else {
            indices.cast(DType::Int64)?
        };
        Ok((values, indices))
    }
}
