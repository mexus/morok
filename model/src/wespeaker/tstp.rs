//! Temporal-Statistical Pooling (TSTP) head from WeSpeaker.
//!
//! Takes a 4D backbone feature map `[B, C, H, T]` plus per-frame attention
//! weights `[B, T_w]`, interpolates the weights to `T` with nearest mode (as
//! pyannote's `StatsPool` does internally), and returns the weighted mean
//! and (unbiased, Bessel-corrected) standard deviation concatenated along the
//! feature axis: `[B, 2 * C * H]`. The exact arithmetic — including the
//! `v1 - v2/v1 + 1e-8` denominator and `1e-8` epsilon on `v1` — matches
//! `pyannote.audio.models.blocks.pooling.StatsPool._pool`.

use morok_ir::SInt;
use morok_tensor::Tensor;
use snafu::ResultExt;

use super::error::{Result, TensorSnafu};

/// Numerical epsilon — same value pyannote's `_pool` uses
/// (`weights.sum + 1e-8`, denominator `+ 1e-8`).
const EPS: f64 = 1e-8;

/// Weighted statistics pooling. `features` is `[B, C, H, T]`,
/// `weights` is `[B, T_w]`. `T` must be concrete (the backbone bakes the
/// time stride sequence at `prepare()` time). Returns `[B, 2 * C * H]`
/// = `concat(mean, std)` along the feature axis.
pub fn tstp_forward(features: &Tensor, weights: &Tensor) -> Result<Tensor> {
    let shape = features.shape().context(TensorSnafu)?;
    if shape.len() != 4 {
        return Err(super::error::Error::Tensor {
            source: Box::new(morok_tensor::error::Error::IrConstruction {
                details: format!("TSTP expects 4D features, got {}D", shape.len()),
            }),
        });
    }
    let t_back = shape[3].as_const().ok_or_else(|| super::error::Error::Tensor {
        source: Box::new(morok_tensor::error::Error::IrConstruction {
            details: "TSTP requires concrete T (backbone time dim) — symbolic not supported".into(),
        }),
    })?;

    // weights: [B, T_w] → [B, T_back] via a constant one-hot nearest matrix,
    // then unsqueeze to [B, 1, 1, T_back] for 4D broadcasting against features.
    // The matrix is precomputed so weights stays a simple matmul; we can't use
    // tensor::resize() here because it requires every shape dim to be concrete
    // and our batch dim is symbolic.
    let t_w = weights.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
        source: Box::new(morok_tensor::error::Error::IrConstruction {
            details: "TSTP requires concrete T_w (weight time dim)".into(),
        }),
    })?;
    let mat = nearest_interp_matrix(t_w, t_back);
    let w = weights.linear().weight(&mat).call().context(TensorSnafu)?;
    let w = w.try_unsqueeze(1).context(TensorSnafu)?;
    let w = w.try_unsqueeze(2).context(TensorSnafu)?;

    let dtype = features.uop().dtype();
    let eps = Tensor::const_(EPS, dtype.clone());

    // v1 = weights.sum(dim=3, keepdim=True) + eps              [B, 1, 1, 1]
    let v1_raw = w.sum_with().axes(3isize).keepdim(true).call().context(TensorSnafu)?;
    let v1 = v1_raw.try_add(&eps).context(TensorSnafu)?;

    // mean = (features * w).sum(dim=3, keepdim=True) / v1      [B, C, H, 1]
    let xw = features.try_mul(&w).context(TensorSnafu)?;
    let xw_sum = xw.sum_with().axes(3isize).keepdim(true).call().context(TensorSnafu)?;
    let mean = xw_sum.try_div(&v1).context(TensorSnafu)?;

    // dx2 = (features - mean)^2
    let centered = features.try_sub(&mean).context(TensorSnafu)?;
    let dx2 = centered.square().context(TensorSnafu)?;

    // v2 = (w^2).sum(dim=3, keepdim=True)                      [B, 1, 1, 1]
    let w_sq = w.square().context(TensorSnafu)?;
    let v2 = w_sq.sum_with().axes(3isize).keepdim(true).call().context(TensorSnafu)?;

    // denom = v1 - v2/v1 + eps                                 [B, 1, 1, 1]
    let denom = v1.try_sub(&v2.try_div(&v1).context(TensorSnafu)?).context(TensorSnafu)?;
    let denom = denom.try_add(&eps).context(TensorSnafu)?;

    // var = (dx2 * w).sum(dim=3, keepdim=True) / denom         [B, C, H, 1]
    let var_num = dx2.try_mul(&w).context(TensorSnafu)?;
    let var_num = var_num.sum_with().axes(3isize).keepdim(true).call().context(TensorSnafu)?;
    let var = var_num.try_div(&denom).context(TensorSnafu)?;
    let std = var.try_sqrt().context(TensorSnafu)?;

    // Squeeze the trailing time dim and flatten (C, H) → C*H
    // mean / std are [B, C, H, 1] → flatten dims 1..4 (i.e., 1,2,3 → C*H*1 = C*H).
    let b = shape[0].clone();
    let c = shape[1].as_const().ok_or_else(|| super::error::Error::Tensor {
        source: Box::new(morok_tensor::error::Error::IrConstruction {
            details: "TSTP requires concrete C (channel dim)".into(),
        }),
    })?;
    let h = shape[2].as_const().ok_or_else(|| super::error::Error::Tensor {
        source: Box::new(morok_tensor::error::Error::IrConstruction {
            details: "TSTP requires concrete H (freq dim)".into(),
        }),
    })?;
    let stats_dim = SInt::Const(c * h);

    let mean_flat = mean.try_reshape([b.clone(), stats_dim.clone()]).context(TensorSnafu)?;
    let std_flat = std.try_reshape([b, stats_dim]).context(TensorSnafu)?;

    Tensor::cat(&[&mean_flat, &std_flat], 1).context(TensorSnafu)
}

/// One-hot interpolation matrix `[t_out, t_in]` such that `dst = src @ M.T`
/// performs PyTorch's `F.interpolate(..., mode="nearest")` along the trailing
/// axis (asymmetric coordinate transform, floor rounding).
///
/// For each output position `o ∈ [0, t_out)` the source position is
/// `floor(o * t_in / t_out)`. Integer math here matches the float-arithmetic
/// floor exactly because `o * t_in` is a non-negative integer.
fn nearest_interp_matrix(t_in: usize, t_out: usize) -> Tensor {
    let mut m = vec![0.0f32; t_out * t_in];
    for o in 0..t_out {
        let src = (o * t_in) / t_out;
        m[o * t_in + src] = 1.0;
    }
    Tensor::from_slice(&m).try_reshape([t_out as isize, t_in as isize]).expect("nearest interp matrix reshape")
}
