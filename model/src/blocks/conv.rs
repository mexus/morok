use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::{Result, TensorSnafu};

fn zeros(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape, DType::Float32).expect("zeros for block placeholder must succeed")
}

/// Bias-less 2D convolution wrapper. `weight` has layout
/// `[out_ch, in_ch / groups, kH, kW]` (same as PyTorch / timm).
#[derive(Clone)]
pub struct Conv2dWeights {
    pub weight: Tensor,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}

impl Conv2dWeights {
    pub fn empty(out_ch: usize, in_ch: usize, kernel: usize, stride: usize, padding: usize) -> Self {
        Self { weight: zeros(&[out_ch, in_ch, kernel, kernel]), stride, padding, groups: 1 }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let p = self.padding as isize;
        x.conv2d()
            .weight(&self.weight)
            .groups(self.groups)
            .stride(&[self.stride, self.stride])
            .padding(&[(p, p), (p, p)])
            .call()
            .context(TensorSnafu)
    }
}

impl HasStateDict for Conv2dWeights {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "weight"), self.weight.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.weight = get_tensor(sd, &prefixed(prefix, "weight"))?;
        Ok(())
    }
}
