//! RNN-T joint network: encoder + predictor projections combined into per-step
//! log-probabilities.

use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

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

    /// Cast all weights to f32 for inference. The joint is small; matches
    /// the predictor's f32 path so cross-network types align.
    pub(crate) fn cast_to_f32(&mut self) -> Result<()> {
        for t in
            [&mut self.enc_w, &mut self.enc_b, &mut self.pred_w, &mut self.pred_b, &mut self.out_w, &mut self.out_b]
        {
            *t = t.cast(DType::Float32).context(TensorSnafu)?;
            t.realize().context(TensorSnafu)?;
        }
        Ok(())
    }

    /// `enc_t [1, 1, enc_hidden]`, `g [1, 1, pred_hidden]` → log-probs
    /// `[1, 1, num_classes]`.
    pub fn forward(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        let enc_t = if enc_t.uop().dtype() != DType::Float32 {
            enc_t.cast(DType::Float32).context(TensorSnafu)?
        } else {
            enc_t.clone()
        };
        let enc_proj = enc_t.linear().weight(&self.enc_w).bias(&self.enc_b).call().context(TensorSnafu)?;
        let pred_proj = g.linear().weight(&self.pred_w).bias(&self.pred_b).call().context(TensorSnafu)?;
        let summed = enc_proj.try_add(&pred_proj).context(TensorSnafu)?;
        let activated = summed.relu().context(TensorSnafu)?;
        let logits = activated.linear().weight(&self.out_w).bias(&self.out_b).call().context(TensorSnafu)?;
        // Promote sub-fp32 floats to fp32 for log_softmax stability — same
        // policy as the CTC head.
        let logits_dtype = logits.uop().dtype();
        let promoted =
            if logits_dtype.is_float() && logits_dtype.bytes() < 4 { DType::Float32 } else { logits_dtype.clone() };
        let logits = if promoted != logits_dtype { logits.cast(promoted).context(TensorSnafu)? } else { logits };
        logits.log_softmax(-1isize).context(TensorSnafu)
    }
}

impl HasStateDict for RnntJoint {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "enc_w"), self.enc_w.clone());
        sd.insert(prefixed(prefix, "enc_b"), self.enc_b.clone());
        sd.insert(prefixed(prefix, "pred_w"), self.pred_w.clone());
        sd.insert(prefixed(prefix, "pred_b"), self.pred_b.clone());
        sd.insert(prefixed(prefix, "out_w"), self.out_w.clone());
        sd.insert(prefixed(prefix, "out_b"), self.out_b.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.enc_w = get_tensor(sd, &prefixed(prefix, "enc_w"))?;
        self.enc_b = get_tensor(sd, &prefixed(prefix, "enc_b"))?;
        self.pred_w = get_tensor(sd, &prefixed(prefix, "pred_w"))?;
        self.pred_b = get_tensor(sd, &prefixed(prefix, "pred_b"))?;
        self.out_w = get_tensor(sd, &prefixed(prefix, "out_w"))?;
        self.out_b = get_tensor(sd, &prefixed(prefix, "out_b"))?;
        Ok(())
    }
}
