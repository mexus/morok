//! RNN-T joint network: encoder + predictor projections combined into per-step
//! log-probabilities.

use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict};
use crate::{load_state_field, state_field};

use crate::gigaam::Result;
use crate::gigaam::error::TensorSnafu;

/// RNN-T joint: `log_softmax(out_w · ReLU(enc_w · enc_t + enc_b + pred_w · g + pred_b) + out_b)`.
///
/// All Linear weights stored PyTorch-style `[out_features, in_features]` so
/// they plug straight into the `linear()` builder (which transposes
/// internally).
#[derive(Clone)]
pub struct RnntJoint {
    pub enc_w: Tensor,
    pub enc_b: Tensor,
    pub pred_w: Tensor,
    pub pred_b: Tensor,
    pub out_w: Tensor,
    pub out_b: Tensor,
}

impl RnntJoint {
    pub fn empty(enc_hidden: usize, pred_hidden: usize, joint_hidden: usize, num_classes: usize) -> Self {
        Self {
            enc_w: Tensor::zeros(&[joint_hidden, enc_hidden], DType::Float32).unwrap(),
            enc_b: Tensor::zeros(&[joint_hidden], DType::Float32).unwrap(),
            pred_w: Tensor::zeros(&[joint_hidden, pred_hidden], DType::Float32).unwrap(),
            pred_b: Tensor::zeros(&[joint_hidden], DType::Float32).unwrap(),
            out_w: Tensor::zeros(&[num_classes, joint_hidden], DType::Float32).unwrap(),
            out_b: Tensor::zeros(&[num_classes], DType::Float32).unwrap(),
        }
    }

    /// `enc_t [1, 1, enc_hidden]`, `g [1, 1, pred_hidden]` → log-probs
    /// `[1, 1, num_classes]`.
    pub fn forward(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        let enc_proj = enc_t.linear().weight(&self.enc_w).bias(&self.enc_b).call().context(TensorSnafu)?;
        let pred_proj = g.linear().weight(&self.pred_w).bias(&self.pred_b).call().context(TensorSnafu)?;
        let summed = enc_proj.try_add(&pred_proj).context(TensorSnafu)?;
        let activated = summed.relu().context(TensorSnafu)?;
        let logits = activated.linear().weight(&self.out_w).bias(&self.out_b).call().context(TensorSnafu)?;
        logits.log_softmax(-1isize).context(TensorSnafu)
    }
}

impl HasStateDict for RnntJoint {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [enc_w, enc_b, pred_w, pred_b, out_w, out_b]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        load_state_field!(self, sd, prefix, [enc_w, enc_b, pred_w, pred_b, out_w, out_b]);
        Ok(())
    }
}
