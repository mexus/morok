//! RNN-T predictor: token embedding + multi-layer LSTM. Stateful per-utterance:
//! the search loop carries `(h, c)` across calls and resets to zeros at the
//! start of a new utterance.

use morok_dtype::DType;
use morok_tensor::Tensor;
use morok_tensor::nn::LSTMCell;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use crate::gigaam::Result;
use crate::gigaam::error::TensorSnafu;

/// RNN-T predictor: token embedding + multi-layer LSTM.
///
/// The empty-prefix predictor call (Python `predict(None, None, batch_size)`)
/// is realized by passing `prev_token = blank_id` with zero `(h, c)`. PyTorch's
/// `nn.Embedding(padding_idx=blank_id)` keeps the blank row at zero through
/// training, so this is equivalent to "embedding of zero vector". We assert
/// the row is in fact zero at load time.
///
/// The LSTM stack reuses [`LSTMCell`] from `morok_tensor::nn`, which applies
/// PyTorch's `[i, f, g, o]` gate order — matching the reference exactly so
/// checkpoints load without gate-axis remapping.
pub struct RnntPredictor {
    /// `[num_classes, pred_hidden]`. Row `blank_id` must be zeros.
    pub embed: Tensor,
    pub layers: Vec<LSTMCell>,
    pub pred_hidden: usize,
    pub num_classes: usize,
    pub blank_id: usize,
}

impl RnntPredictor {
    pub fn empty(pred_hidden: usize, num_layers: usize, num_classes: usize) -> Self {
        let blank_id = num_classes - 1;
        let h4 = 4 * pred_hidden;
        Self {
            embed: Tensor::zeros(&[num_classes, pred_hidden], DType::Float32).unwrap(),
            layers: (0..num_layers)
                .map(|_| {
                    LSTMCell::new(
                        Tensor::zeros(&[h4, pred_hidden], DType::Float32).unwrap(),
                        Tensor::zeros(&[h4, pred_hidden], DType::Float32).unwrap(),
                        Tensor::zeros(&[h4], DType::Float32).unwrap(),
                        Tensor::zeros(&[h4], DType::Float32).unwrap(),
                    )
                })
                .collect(),
            pred_hidden,
            num_classes,
            blank_id,
        }
    }

    /// Run one predictor step. Returns a single tensor of shape
    /// `[1, 1, pred_hidden + 2 * num_layers * pred_hidden]` containing
    /// `[g | h_out_flat | c_out_flat]` concatenated along the last axis.
    ///
    /// The flat layout is so the result fits one output tensor (the JIT
    /// macro is single-output). Caller splits by known offsets after copyout.
    pub fn forward_concat(&self, prev_token: &Tensor, h_in: &Tensor, c_in: &Tensor) -> Result<Tensor> {
        let p = self.pred_hidden as isize;
        let l = self.layers.len() as isize;

        // Embed lookup: prev_token [1, 1] -> emb [1, 1, P].
        // Squeeze the seq-len axis to feed the LSTM cell shape [B, P].
        let emb = self.embed.embedding(prev_token).context(TensorSnafu)?;
        let mut layer_in = emb.try_squeeze(Some(1)).context(TensorSnafu)?; // [1, P]

        let mut new_hs: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        let mut new_cs: Vec<Tensor> = Vec::with_capacity(self.layers.len());
        for (i, cell) in self.layers.iter().enumerate() {
            let i_i = i as isize;
            // Slice layer i's h, c → [1, 1, P], squeeze leading axis → [1, P].
            let h_i = h_in
                .try_shrink([(i_i, i_i + 1), (0, 1), (0, p)])
                .context(TensorSnafu)?
                .try_squeeze(Some(0))
                .context(TensorSnafu)?;
            let c_i = c_in
                .try_shrink([(i_i, i_i + 1), (0, 1), (0, p)])
                .context(TensorSnafu)?
                .try_squeeze(Some(0))
                .context(TensorSnafu)?;
            let (new_h, new_c) = cell.step(&layer_in, &h_i, &c_i).context(TensorSnafu)?;
            new_hs.push(new_h.clone());
            new_cs.push(new_c.clone());
            layer_in = new_h;
        }

        // g = last layer output [1, P] → [1, 1, P].
        let g = layer_in.try_unsqueeze(1).context(TensorSnafu)?;

        // Stack per-layer h, c → [L, 1, P]. Reshape to [1, 1, L * P] for concat.
        let new_h_stacked = Tensor::stack(&new_hs.iter().collect::<Vec<_>>(), 0).context(TensorSnafu)?;
        let new_c_stacked = Tensor::stack(&new_cs.iter().collect::<Vec<_>>(), 0).context(TensorSnafu)?;
        let new_h_flat = new_h_stacked.try_reshape([1, 1, l * p]).context(TensorSnafu)?;
        let new_c_flat = new_c_stacked.try_reshape([1, 1, l * p]).context(TensorSnafu)?;

        // Concat along the last axis: [1, 1, P + L*P + L*P].
        Tensor::cat(&[&g, &new_h_flat, &new_c_flat], 2).context(TensorSnafu)
    }

    /// Cast predictor weights to fp32 (predictor is small; the encoder may
    /// keep its native fp16/bf16 path) and force the blank-id embedding
    /// row to zero, in place. Together these implement the Python
    /// `predict(None, None, batch_size)` empty-prefix path without a
    /// separate fresh-step JIT.
    ///
    /// Some checkpoints (notably `v3_e2e_rnnt`) do NOT keep
    /// `embed.weight[blank_id]` at zero — fine-tuning updated it. We patch
    /// it here so morok matches the Python decoder regardless of how the
    /// checkpoint was trained.
    pub(crate) fn prepare_for_inference(&mut self) -> Result<()> {
        self.embed = self.embed.cast(DType::Float32).context(TensorSnafu)?;
        self.embed.realize().context(TensorSnafu)?;
        for cell in &mut self.layers {
            for t in [&mut cell.weight_ih, &mut cell.weight_hh, &mut cell.bias_ih, &mut cell.bias_hh] {
                *t = t.cast(DType::Float32).context(TensorSnafu)?;
                t.realize().context(TensorSnafu)?;
            }
        }
        let mut view = self.embed.array_view_mut::<f32>().context(TensorSnafu)?;
        let p = self.pred_hidden;
        let row_start = self.blank_id * p;
        let slice = view.as_slice_mut().expect("contiguous embed");
        slice[row_start..row_start + p].fill(0.0);
        Ok(())
    }
}

impl HasStateDict for RnntPredictor {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "embed"), self.embed.clone());
        for (i, cell) in self.layers.iter().enumerate() {
            let p = prefixed(prefix, &format!("lstm.{i}"));
            sd.insert(prefixed(&p, "w_ih"), cell.weight_ih.clone());
            sd.insert(prefixed(&p, "w_hh"), cell.weight_hh.clone());
            sd.insert(prefixed(&p, "b_ih"), cell.bias_ih.clone());
            sd.insert(prefixed(&p, "b_hh"), cell.bias_hh.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.embed = get_tensor(sd, &prefixed(prefix, "embed"))?;
        for (i, cell) in self.layers.iter_mut().enumerate() {
            let p = prefixed(prefix, &format!("lstm.{i}"));
            cell.weight_ih = get_tensor(sd, &prefixed(&p, "w_ih"))?;
            cell.weight_hh = get_tensor(sd, &prefixed(&p, "w_hh"))?;
            cell.bias_ih = get_tensor(sd, &prefixed(&p, "b_ih"))?;
            cell.bias_hh = get_tensor(sd, &prefixed(&p, "b_hh"))?;
        }
        Ok(())
    }
}
