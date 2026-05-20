use snafu::ResultExt;
use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::init::{ones, zeros};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::{Result, TensorSnafu};

/// BN with the running variance pre-folded into `invstd`. State-dict round-trip
/// uses the canonical timm/PyTorch keys `weight` (→ `scale`), `bias`,
/// `running_mean` (→ `mean`), `running_var` (→ `invstd` after fold).
#[derive(Clone)]
pub struct BatchNormWeights {
    pub scale: Tensor,
    pub bias: Tensor,
    pub mean: Tensor,
    pub invstd: Tensor,
}

impl BatchNormWeights {
    pub fn empty(channels: usize) -> Self {
        Self {
            scale: ones(&[channels], DType::Float32),
            bias: zeros(&[channels], DType::Float32),
            mean: zeros(&[channels], DType::Float32),
            invstd: ones(&[channels], DType::Float32),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.batchnorm()
            .scale(&self.scale)
            .bias(&self.bias)
            .mean(&self.mean)
            .invstd(&self.invstd)
            .call()
            .context(TensorSnafu)
    }
}

impl HasStateDict for BatchNormWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.scale.clone());
        sd.insert(prefixed(prefix, "bias"), self.bias.clone());
        sd.insert(prefixed(prefix, "running_mean"), self.mean.clone());
        sd.insert(prefixed(prefix, "running_var"), self.invstd.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.scale = get_tensor(sd, &prefixed(prefix, "weight"))?;
        self.bias = get_tensor(sd, &prefixed(prefix, "bias"))?;
        self.mean = get_tensor(sd, &prefixed(prefix, "running_mean"))?;
        self.invstd = get_tensor(sd, &prefixed(prefix, "running_var"))?;
        Ok(())
    }
}
