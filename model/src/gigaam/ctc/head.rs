//! CTC projection head: `Conv1d(d_model, vocab_size, k=1)` + transpose +
//! `LogSoftmax`. Produces the `[B, T, vocab_size]` log-probabilities consumed
//! by `morok_arch::ctc` decoders — the head itself is just the final
//! projection layer, not the decoder.

use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::state::{HasStateDict, StateDict};
use crate::{load_state_field, state_field};

use crate::gigaam::error::TensorSnafu;
use crate::gigaam::{GigaAmConfig, Result};

#[derive(Clone)]
pub struct CTCHead {
    pub weight: Tensor, // [vocab_size, d_model, 1]
    pub bias: Tensor,   // [vocab_size]
}

impl CTCHead {
    pub fn empty(config: &GigaAmConfig) -> Self {
        Self {
            weight: Tensor::zeros(&[config.vocab_size, config.d_model, 1], DType::Float32).unwrap(),
            bias: Tensor::zeros(&[config.vocab_size], DType::Float32).unwrap(),
        }
    }

    /// Forward pass. Input: `[B, d_model, T]`, output: `[B, T, vocab_size]` log-probs.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let y = x.conv2d().weight(&self.weight).bias(&self.bias).call().context(TensorSnafu)?;
        let y = y.try_transpose(-1, -2).context(TensorSnafu)?;
        y.log_softmax(-1isize).context(TensorSnafu)
    }
}

impl HasStateDict for CTCHead {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        state_field!(sd, prefix, self, [weight, bias]);
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), crate::state::Error> {
        load_state_field!(self, sd, prefix, [weight, bias]);
        Ok(())
    }
}
