//! Convolution operations: conv2d, conv_transpose2d.

use bon::bon;
use snafu::ResultExt;

use crate::Tensor;
use crate::error::UOpSnafu;
use crate::reduce::AxisSpec;

type Result<T> = crate::Result<T>;

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
        acc_dtype: Option<morok_dtype::DType>,
    ) -> Result<Tensor> {
        let x_shape = self.shape()?;
        let w_shape = weight.shape()?;

        let bs = x_shape[0].as_const().expect("batch dim must be concrete");
        let cin_ = x_shape[1].as_const().expect("channel dim must be concrete");
        let cout = w_shape[0].as_const().expect("cout must be concrete");
        let cin = w_shape[1].as_const().expect("cin/g must be concrete");

        let hw: Vec<usize> = w_shape[2..].iter().map(|s| s.as_const().expect("kernel dim must be concrete")).collect();
        let n_spatial = hw.len();

        if x_shape.len() != w_shape.len() {
            return Err(crate::error::Error::IrConstruction {
                details: format!("input and weight must have same ndim, got {} and {}", x_shape.len(), w_shape.len()),
            });
        }
        if groups * cin != cin_ {
            return Err(crate::error::Error::IrConstruction {
                details: format!("groups*cin/g ({}) != input channels ({cin_})", groups * cin),
            });
        }

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
        x = x.sum_with().axes(AxisSpec::Multiple(reduce_axes)).keepdim(true).maybe_dtype(acc_dtype).call()?;

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
                let mut rshape = morok_ir::shape::to_vec_isize(&cur_shape).context(UOpSnafu)?;
                rshape.insert(spatial_idx + 1, 1);
                x = x.try_reshape(&rshape)?;

                // pad: (0, s-1) on the inserted dim
                let mut pad: Vec<(isize, isize)> = vec![(0, 0); rshape.len()];
                pad[spatial_idx + 1] = (0, (s - 1) as isize);
                x = x.try_pad(&pad)?;

                // merge spatial_idx and spatial_idx+1
                let cur_shape = x.shape()?;
                let mut rshape = morok_ir::shape::to_vec_isize(&cur_shape).context(UOpSnafu)?;
                let merged = rshape[spatial_idx] * rshape[spatial_idx + 1];
                rshape[spatial_idx] = merged;
                rshape.remove(spatial_idx + 1);
                x = x.try_reshape(&rshape)?;

                // shrink: remove trailing (s-1) from this dim
                let cur_shape = x.shape()?;
                let new_size = k * s - (s - 1);
                let dims = morok_ir::shape::to_vec_isize(&cur_shape).context(UOpSnafu)?;
                let mut ranges: Vec<(isize, isize)> = dims.iter().map(|&d| (0, d)).collect();
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
}
