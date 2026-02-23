//! Neural network operations: convolution, pooling, normalization.

use bon::bon;
use morok_dtype::DType;
use morok_ir::{ConstValue, UOp};

use crate::Tensor;
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

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

    /// Average pooling.
    #[builder]
    pub fn avg_pool2d(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
        #[builder(default = true)] count_include_pad: bool,
    ) -> Result<Tensor> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        let mut x = self.clone();
        if padding.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); self.ndim()? - n_spatial];
            full_pad.extend_from_slice(padding);
            x = x.try_pad(&full_pad)?;
        }

        if !count_include_pad {
            let pooled = x.pool(kernel_size, stride, dilation)?;
            let sum_x = pooled.sum_with().axes(axes.clone()).keepdim(false).call()?;

            let ones = Tensor::new(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
            let mut ones = ones.broadcast_to(&self.shape()?)?;
            if padding.iter().any(|&(b, e)| b != 0 || e != 0) {
                let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); self.ndim()? - n_spatial];
                full_pad.extend_from_slice(padding);
                ones = ones.try_pad(&full_pad)?;
            }
            let pooled_ones = ones.pool(kernel_size, stride, dilation)?;
            let sum_ones = pooled_ones.sum_with().axes(axes).keepdim(false).call()?;
            return sum_x.try_div(&sum_ones);
        }

        let pooled = x.pool(kernel_size, stride, dilation)?;
        pooled.mean(axes)
    }

    /// Max pooling.
    #[builder]
    pub fn max_pool2d(
        &self,
        kernel_size: &[usize],
        stride: Option<&[usize]>,
        dilation: Option<&[usize]>,
        padding: Option<&[(isize, isize)]>,
    ) -> Result<Tensor> {
        let n_spatial = kernel_size.len();
        let default_dilation: Vec<usize> = vec![1; n_spatial];
        let stride = stride.unwrap_or(kernel_size);
        let dilation = dilation.unwrap_or(&default_dilation);
        let no_pad: Vec<(isize, isize)> = vec![(0, 0); n_spatial];
        let padding = padding.unwrap_or(&no_pad);

        let reduce_axes: Vec<isize> = (0..n_spatial).map(|j| -(1 + j as isize)).collect();
        let axes = AxisSpec::Multiple(reduce_axes);

        let mut x = self.clone();
        if padding.iter().any(|&(b, e)| b != 0 || e != 0) {
            let mut full_pad: Vec<(isize, isize)> = vec![(0, 0); self.ndim()? - n_spatial];
            full_pad.extend_from_slice(padding);
            x = x.try_pad_value(&full_pad, f64::NEG_INFINITY)?;
        }

        let pooled = x.pool(kernel_size, stride, dilation)?;
        pooled.max(axes)
    }

    /// Layer normalization over axes [axis..ndim). Casts to f32 internally.
    pub fn layernorm(&self, axis: isize, eps: f64) -> Result<Tensor> {
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

        if original_dtype != DType::Float32 { normalized.cast(original_dtype) } else { Ok(normalized) }
    }
}
