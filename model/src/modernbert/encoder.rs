//! ModernBERT transformer encoder: a stack of pre-norm [`EncoderLayer`]s with
//! per-layer RoPE. No internal embeddings or final norm — those live on the
//! root [`super::model::ModernBert`].
//!
//! Per-layer rotary tables are rebuilt on each forward (the sequence length is
//! concrete per call); global vs local theta is selected via
//! [`ModernBertConfig::rope_theta`].

use snafu::{OptionExt, ResultExt};
use svod_ir::SInt;
use svod_tensor::Tensor;

use crate::state::{self, HasStateDict, StateDict};

use super::config::ModernBertConfig;
use super::encoder_layer::EncoderLayer;
use super::error::{Result, SymbolicShapeSnafu, TensorSnafu};
use super::rotary::RotaryTable;

#[derive(Clone)]
pub struct Encoder {
    pub config: ModernBertConfig,
    pub layers: Vec<EncoderLayer>,
}

impl Encoder {
    pub fn empty(config: &ModernBertConfig) -> Self {
        let layers = (0..config.num_hidden_layers).map(|i| EncoderLayer::empty(config, i)).collect();
        Self { config: config.clone(), layers }
    }

    /// Run the encoder stack. `x`: `(B, L, D)` → `(B, L, D)`.
    /// `padding_mask`: optional bool `(B, L)` where `true` = real token,
    /// `false` = padding. When present it is reshaped to `(B, 1, 1, L)` and
    /// inverted to the SDPA "True = masked out" convention.
    pub fn forward(&self, x: &Tensor, padding_mask: Option<&Tensor>) -> Result<Tensor> {
        let shape = x.shape().context(TensorSnafu)?;
        let seq_len: usize = shape[1].as_const().context(SymbolicShapeSnafu { what: "encoder" })?;
        let head_dim = self.config.head_dim();

        // Build the SDPA key-axis mask once: bool (B,1,1,L), True = masked out.
        let mask_4d = match padding_mask {
            Some(m) => {
                let b_dim = shape[0].clone();
                let m2 = m.try_reshape([b_dim, SInt::from(seq_len)]).context(TensorSnafu)?;
                let inverted = m2.logical_not().context(TensorSnafu)?;
                Some(inverted.try_unsqueeze(1).context(TensorSnafu)?.try_unsqueeze(1).context(TensorSnafu)?)
            }
            None => None,
        };

        let mut h = x.clone();
        for (layer_id, layer) in self.layers.iter().enumerate() {
            let theta = self.config.rope_theta(layer_id);
            let rotary = RotaryTable::new(theta, seq_len, head_dim, self.config.dtype.clone())?;
            h = layer.forward(&h, &rotary, mask_4d.as_ref())?;
        }
        Ok(h)
    }
}

impl HasStateDict for Encoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        for (i, layer) in self.layers.iter().enumerate() {
            sd.extend(layer.state_dict(&format!("{prefix}.{i}")));
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(sd, &format!("{prefix}.{i}"))?;
        }
        Ok(())
    }
}
