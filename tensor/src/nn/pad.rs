//! Padding helpers: flat-to-pair conversion, auto-pad, pool pad resolution.

use morok_ir::{ConstValue, UOp};

use super::AutoPad;
use crate::Tensor;

type Result<T> = crate::Result<T>;

/// Convert flat pads `[begin0, begin1, ..., end0, end1, ...]` to `[(begin0, end0), ...]`.
pub fn flat_pads_to_pairs(pads: &[i64]) -> Vec<(isize, isize)> {
    let n = pads.len() / 2;
    (0..n).map(|i| (pads[i] as isize, pads[i + n] as isize)).collect()
}

/// Split total padding per dimension into `[begin0, begin1, ..., end0, end1, ...]`
/// based on auto_pad mode (SAME_UPPER: more padding at end; SAME_LOWER: more at begin).
pub fn auto_pad_split(total_pads: &[isize], auto_pad: AutoPad) -> Vec<isize> {
    let first: Vec<isize> = if auto_pad == AutoPad::SameUpper {
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
    dilations: &[usize],
    strides: &[usize],
    auto_pad: AutoPad,
) -> Vec<(isize, isize)> {
    let n = kernel.len();
    match auto_pad {
        AutoPad::Valid => vec![(0, 0); n],
        AutoPad::NotSet => {
            if pads.is_empty() {
                vec![(0, 0); n]
            } else {
                flat_pads_to_pairs(pads)
            }
        }
        AutoPad::SameUpper | AutoPad::SameLower => {
            let total_pads: Vec<isize> = (0..n)
                .map(|i| {
                    let out_size = usize::div_ceil(input_spatial[i], strides[i]);
                    let eff_kernel = dilations[i] * (kernel[i] - 1) + 1;
                    let needed = (out_size - 1) * strides[i] + eff_kernel;
                    needed.saturating_sub(input_spatial[i]) as isize
                })
                .collect();
            let flat = auto_pad_split(&total_pads, auto_pad);
            let half = flat.len() / 2;
            (0..half).map(|i| (flat[i], flat[i + half])).collect()
        }
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
}

/// Adjust padding for ceil_mode output sizes.
/// Per arXiv:1603.07285 section 5.1, relationship 15.
pub(super) fn apply_ceil_mode(
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
