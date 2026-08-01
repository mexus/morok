//! CLS pooling: take the first token's embedding.

use snafu::ResultExt;
use svod_tensor::{Tensor, s};

use super::error::{Result, TensorSnafu};

/// CLS pooling: take the first token's embedding. `hidden_states`: `(B, L, D)`
/// → `(B, D)`.
pub fn cls(hidden_states: &Tensor) -> Result<Tensor> {
    hidden_states.getitem(s![.., 0, ..]).context(TensorSnafu)
}
