//! `Tensor::rand_like` / `Tensor::randn_like` — convenience wrappers that
//! inherit shape, dtype, and device from an existing tensor.
//!
//! Used by nn modules that want to sample noise matching some reference
//! tensor (dropout masks, noise injection, gaussian-init layers, etc.).

use morok_dtype::DType;
use morok_ir::shape::to_vec_usize;
use snafu::ResultExt;

use crate::{Result, Tensor, UOpSnafu};

impl Tensor {
    /// `rand_like` with a dtype override (device and shape still inherited).
    pub fn rand_like_with_dtype(&self, dtype: DType) -> Result<Tensor> {
        let shape = to_vec_usize(&self.shape()?).context(UOpSnafu)?;
        Self::rand_with(&shape, dtype, self.device())
    }

    /// Uniform `[0, 1)` random tensor with the same shape/dtype/device as `self`.
    pub fn rand_like(&self) -> Result<Tensor> {
        self.rand_like_with_dtype(self.uop().dtype())
    }

    /// `randn_like` with a dtype override.
    ///
    /// Internally generates f32 samples via Box-Muller, then casts to the
    /// target dtype. Using f32 inside Box-Muller keeps cos/log/sqrt accurate
    /// even when the caller wants low-precision output.
    pub fn randn_like_with_dtype(&self, dtype: DType) -> Result<Tensor> {
        let shape = to_vec_usize(&self.shape()?).context(UOpSnafu)?;
        Tensor::randn(&shape)?.cast(dtype)
    }

    /// Standard normal `N(0, 1)` random tensor with the same shape/dtype/device as `self`.
    pub fn randn_like(&self) -> Result<Tensor> {
        self.randn_like_with_dtype(self.uop().dtype())
    }

    /// Uniform integer `[low, high)` random tensor with the same shape/dtype/device as `self`.
    ///
    /// The underlying `Tensor::randint` returns `Int32`; if `self`'s dtype
    /// differs the result is cast to match (e.g. `Int64` template → `Int64`
    /// result). Requires `low < high`.
    pub fn randint_like(&self, low: i64, high: i64) -> Result<Tensor> {
        let shape = to_vec_usize(&self.shape()?).context(UOpSnafu)?;
        let r = Tensor::randint(&shape, low, high)?;
        if r.uop().dtype() == self.uop().dtype() { Ok(r) } else { r.cast(self.uop().dtype()) }
    }
}
