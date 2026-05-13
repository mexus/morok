//! RNN-T head = predictor + joint. Composed similarly to Python `RNNTHead`.

use crate::state::{self, HasStateDict, StateDict, prefixed};

use super::joint::RnntJoint;
use super::predictor::RnntPredictor;

pub struct RnntHead {
    pub predictor: RnntPredictor,
    pub joint: RnntJoint,
    pub pred_rnn_layers: usize,
    pub pred_hidden: usize,
    pub joint_hidden: usize,
    pub num_classes: usize,
}

impl RnntHead {
    pub fn empty(
        enc_hidden: usize,
        pred_hidden: usize,
        pred_rnn_layers: usize,
        joint_hidden: usize,
        num_classes: usize,
    ) -> Self {
        Self {
            predictor: RnntPredictor::empty(pred_hidden, pred_rnn_layers, num_classes),
            joint: RnntJoint::empty(enc_hidden, pred_hidden, joint_hidden, num_classes),
            pred_rnn_layers,
            pred_hidden,
            joint_hidden,
            num_classes,
        }
    }
}

impl HasStateDict for RnntHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = self.predictor.state_dict(&prefixed(prefix, "predictor"));
        sd.extend(self.joint.state_dict(&prefixed(prefix, "joint")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.predictor.load_state_dict(sd, &prefixed(prefix, "predictor"))?;
        self.joint.load_state_dict(sd, &prefixed(prefix, "joint"))?;
        Ok(())
    }
}
