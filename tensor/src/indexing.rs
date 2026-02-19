//! Indexing operations for Tensors.

use super::*;
use crate::error::ShapeMismatchSnafu;

impl Tensor {
    /// Gather values along an axis specified by `dim`, using `index` for element selection.
    #[track_caller]
    pub fn gather(&self, dim: isize, index: &Tensor) -> Result<Self> {
        let self_shape = self.shape()?;
        let index_shape = index.shape()?;
        let ndim = self_shape.len();
        let dim = Self::normalize_axis(dim, ndim)?;

        snafu::ensure!(
            index_shape.len() == ndim,
            ShapeMismatchSnafu {
                context: "gather",
                expected: format!("{ndim}D"),
                actual: format!("{}D index", index_shape.len())
            }
        );

        let self_dims = morok_ir::shape::to_vec_usize(&self_shape).context(UOpSnafu)?;
        let index_dims = morok_ir::shape::to_vec_usize(&index_shape).context(UOpSnafu)?;

        snafu::ensure!(
            self_dims.iter().zip(&index_dims).enumerate()
                .all(|(d, (s, i))| d == dim || s >= i),
            ShapeMismatchSnafu {
                context: "gather",
                expected: "self[d] >= index[d] for d != dim".to_string(),
                actual: format!("self={self_dims:?}, index={index_dims:?}")
            }
        );

        let shrink: Vec<_> = (0..ndim)
            .map(|d| (0, (if d == dim { self_dims[d] } else { index_dims[d] }) as isize))
            .collect();
        let x = self.try_shrink(&shrink)?.try_unsqueeze(-1)?.try_transpose(-1, dim as isize)?;

        let arange = Tensor::arange(0, Some(self_dims[dim] as i64), None)?
            .cast(index.uop().dtype())?;
        let mask = index.try_unsqueeze(-1)?.try_eq(&arange)?;

        x.where_(&mask, &Self::new(x.uop().const_like(0.0)))?
            .sum_with().axes(-1).dtype(self.uop().dtype()).call()
    }
}
