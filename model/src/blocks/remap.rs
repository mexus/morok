//! Pre-process a PyTorch-format BN state dict into the layout that
//! [`BatchNormWeights`](super::batchnorm::BatchNormWeights) expects.
//!
//! PyTorch's `nn.BatchNorm2d` stores `running_var` and recomputes
//! `invstd = 1 / sqrt(var + eps)` on every forward. Folding it once at load
//! time keeps the JIT graph purely affine. We also strip
//! `num_batches_tracked`, which is metadata of no inference use.

use svod_tensor::Tensor;

use crate::state::StateDict;

use super::error::{Error, Result};

/// Default PyTorch BatchNorm eps. timm and WeSpeaker checkpoints we target do
/// not override it.
const BN_EPS: f32 = 1e-5;

/// Walk `sd` and:
/// 1. Replace every value under a `*.running_var` key with
///    `1 / sqrt(var + BN_EPS)`, computed elementwise as f32. The
///    [`BatchNormWeights::load_state_dict`](super::batchnorm::BatchNormWeights)
///    impl reads the `running_var` slot directly into its `invstd` field.
/// 2. Drop every `*.num_batches_tracked` entry (no consumer; PyTorch metadata).
pub fn fold_batchnorm(mut sd: StateDict) -> Result<StateDict> {
    sd.retain(|k, _| !k.ends_with("num_batches_tracked"));

    let var_keys: Vec<String> = sd.keys().filter(|k| k.ends_with("running_var")).cloned().collect();
    for key in var_keys {
        let var = sd.remove(&key).expect("key just enumerated");
        let var_f32 = var.as_vec::<f32>().map_err(|e| Error::Tensor { source: Box::new(e) })?;
        let invstd: Vec<f32> = var_f32.iter().map(|&v| 1.0 / (v + BN_EPS).sqrt()).collect();
        sd.insert(key, Tensor::from_slice(&invstd));
    }

    Ok(sd)
}
