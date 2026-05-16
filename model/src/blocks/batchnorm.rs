use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::{Result, TensorSnafu};

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::Float32).expect("zeros for block placeholder must succeed")
}

fn ones(shape: &[usize]) -> Tensor {
    Tensor::ones(shape, DType::Float32).expect("ones for block placeholder must succeed")
}

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
        Self { scale: ones(&[channels]), bias: zeros(&[channels]), mean: zeros(&[channels]), invstd: ones(&[channels]) }
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
