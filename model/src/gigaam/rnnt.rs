//! GigaAM RNN-T (transducer) head: predictor + joint network + per-utterance
//! step backend implementing [`morok_arch::rnnt::JointStep`].
//!
//! Layout mirrors the reference Python `RNNTDecoder` / `RNNTJoint` / `RNNTHead`
//! in `submodules/GigaAM/gigaam/decoder.py`. The predictor is a multi-layer
//! hand-coded LSTM (matching `model/src/vad/mod.rs`'s Silero pattern), and the
//! joint is a two-Linear sum + ReLU + Linear + log-softmax projection.
//!
//! `RnntPredictor::forward_concat` packs `(g, h_out, c_out)` into a single
//! output tensor (Silero-style multi-output trick) so it fits the
//! single-output [`jit_wrapper!`] macro. The caller splits the buffer back
//! into the three components after copyout.

use std::path::Path;

use morok_dtype::DType;
use morok_tensor::Tensor;
use snafu::ResultExt;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::error::{ConfigSnafu, HubSnafu, StateSnafu, TensorSnafu};
use super::{Encoder, GigaAmConfig, Result, build_encoder_from_sd, build_rope_cache};

// ─── Predictor ────────────────────────────────────────────────────────────

/// One layer of the multi-layer LSTM stack inside [`RnntPredictor`].
///
/// Gate packing follows PyTorch's `nn.LSTM`: `[i, f, g, o]`. The hand-coded
/// LSTM cell at the bottom of this file applies the gates in that order, so a
/// straight checkpoint load (no remap of gate dimensions) gives bit-for-bit
/// equivalence to the reference.
pub struct RnntPredictorLstmLayer {
    /// `[4 * pred_hidden, input_size]`. For layer 0, `input_size = pred_hidden`
    /// (embed output); for higher layers, `input_size = pred_hidden`
    /// (preceding layer's output).
    pub w_ih: Tensor,
    /// `[4 * pred_hidden, pred_hidden]`.
    pub w_hh: Tensor,
    /// `[4 * pred_hidden]`.
    pub b_ih: Tensor,
    /// `[4 * pred_hidden]`.
    pub b_hh: Tensor,
}

impl RnntPredictorLstmLayer {
    pub fn empty(pred_hidden: usize) -> Self {
        let h4 = 4 * pred_hidden;
        Self {
            w_ih: Tensor::full(&[h4, pred_hidden], 0.0, DType::Float32).unwrap(),
            w_hh: Tensor::full(&[h4, pred_hidden], 0.0, DType::Float32).unwrap(),
            b_ih: Tensor::full(&[h4], 0.0, DType::Float32).unwrap(),
            b_hh: Tensor::full(&[h4], 0.0, DType::Float32).unwrap(),
        }
    }

    fn load(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.w_ih = get_tensor(sd, &prefixed(prefix, "w_ih"))?;
        self.w_hh = get_tensor(sd, &prefixed(prefix, "w_hh"))?;
        self.b_ih = get_tensor(sd, &prefixed(prefix, "b_ih"))?;
        self.b_hh = get_tensor(sd, &prefixed(prefix, "b_hh"))?;
        Ok(())
    }
}

/// RNN-T predictor: token embedding + multi-layer LSTM. Stateful per-utterance:
/// the search loop carries `(h, c)` across calls and resets to zeros at the
/// start of a new utterance.
///
/// The empty-prefix predictor call (Python `predict(None, None, batch_size)`)
/// is realized by passing `prev_token = blank_id` with zero `(h, c)`. PyTorch's
/// `nn.Embedding(padding_idx=blank_id)` keeps the blank row at zero through
/// training, so this is equivalent to "embedding of zero vector". We assert
/// the row is in fact zero at load time.
pub struct RnntPredictor {
    /// `[num_classes, pred_hidden]`. Row `blank_id` must be zeros.
    pub embed: Tensor,
    pub layers: Vec<RnntPredictorLstmLayer>,
    pub pred_hidden: usize,
    pub num_classes: usize,
    pub blank_id: usize,
}

impl RnntPredictor {
    pub fn empty(pred_hidden: usize, num_layers: usize, num_classes: usize) -> Self {
        let blank_id = num_classes - 1;
        Self {
            embed: Tensor::full(&[num_classes, pred_hidden], 0.0, DType::Float32).unwrap(),
            layers: (0..num_layers).map(|_| RnntPredictorLstmLayer::empty(pred_hidden)).collect(),
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
        for (i, layer) in self.layers.iter().enumerate() {
            let i_i = i as isize;
            // Slice layer i's h, c → [1, 1, P], squeeze leading axis → [1, P].
            let h_i = h_in
                .try_shrink([(i_i, i_i + 1), (0, 1), (0, p)])
                .context(TensorSnafu)?
                .try_squeeze(Some(0))
                .context(TensorSnafu)?; // [1, P]
            let c_i = c_in
                .try_shrink([(i_i, i_i + 1), (0, 1), (0, p)])
                .context(TensorSnafu)?
                .try_squeeze(Some(0))
                .context(TensorSnafu)?; // [1, P]
            let (new_h, new_c) =
                lstm_cell(&layer_in, &h_i, &c_i, &layer.w_ih, &layer.w_hh, &layer.b_ih, &layer.b_hh, self.pred_hidden)?;
            new_hs.push(new_h.clone());
            new_cs.push(new_c.clone());
            layer_in = new_h; // next layer's input
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
    /// separate fresh-step JIT: the search loop encodes "no prev_token yet"
    /// by passing `blank_id`, and we ensure embedding lookup returns zeros
    /// so the LSTM step gets the same `(zero embed, zero state)` inputs as
    /// the reference.
    ///
    /// Some checkpoints (notably `v3_e2e_rnnt`) do NOT keep
    /// `embed.weight[blank_id]` at zero — fine-tuning updated it. We patch
    /// it here so morok matches the Python decoder regardless of how the
    /// checkpoint was trained.
    ///
    /// TODO: drop the f32 promotion in favor of dtype-aware JIT input
    /// buffers + a dtype-aware blank patch. The promotion is a v1
    /// expediency — predictor weights are small but per-step compute could
    /// be ~2× cheaper at fp16 on Apple AMX.
    fn prepare_for_inference(&mut self) -> Result<()> {
        self.embed = self.embed.cast(DType::Float32).context(TensorSnafu)?;
        self.embed.realize().context(TensorSnafu)?;
        for layer in &mut self.layers {
            layer.w_ih = layer.w_ih.cast(DType::Float32).context(TensorSnafu)?;
            layer.w_hh = layer.w_hh.cast(DType::Float32).context(TensorSnafu)?;
            layer.b_ih = layer.b_ih.cast(DType::Float32).context(TensorSnafu)?;
            layer.b_hh = layer.b_hh.cast(DType::Float32).context(TensorSnafu)?;
            layer.w_ih.realize().context(TensorSnafu)?;
            layer.w_hh.realize().context(TensorSnafu)?;
            layer.b_ih.realize().context(TensorSnafu)?;
            layer.b_hh.realize().context(TensorSnafu)?;
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
        for (i, layer) in self.layers.iter().enumerate() {
            let p = prefixed(prefix, &format!("lstm.{i}"));
            sd.insert(prefixed(&p, "w_ih"), layer.w_ih.clone());
            sd.insert(prefixed(&p, "w_hh"), layer.w_hh.clone());
            sd.insert(prefixed(&p, "b_ih"), layer.b_ih.clone());
            sd.insert(prefixed(&p, "b_hh"), layer.b_hh.clone());
        }
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.embed = get_tensor(sd, &prefixed(prefix, "embed"))?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load(sd, &prefixed(prefix, &format!("lstm.{i}")))?;
        }
        Ok(())
    }
}

// ─── Joint network ────────────────────────────────────────────────────────

/// RNN-T joint: `log_softmax(out_w · ReLU(enc_w · enc_t + enc_b + pred_w · g + pred_b) + out_b)`.
///
/// All Linear weights stored PyTorch-style `[out_features, in_features]` so
/// they plug straight into the `linear()` builder (which transposes
/// internally).
pub struct RnntJoint {
    pub enc_w: Tensor,  // [joint_hidden, enc_hidden]
    pub enc_b: Tensor,  // [joint_hidden]
    pub pred_w: Tensor, // [joint_hidden, pred_hidden]
    pub pred_b: Tensor, // [joint_hidden]
    pub out_w: Tensor,  // [num_classes, joint_hidden]
    pub out_b: Tensor,  // [num_classes]
}

impl RnntJoint {
    pub fn empty(enc_hidden: usize, pred_hidden: usize, joint_hidden: usize, num_classes: usize) -> Self {
        Self {
            enc_w: Tensor::full(&[joint_hidden, enc_hidden], 0.0, DType::Float32).unwrap(),
            enc_b: Tensor::full(&[joint_hidden], 0.0, DType::Float32).unwrap(),
            pred_w: Tensor::full(&[joint_hidden, pred_hidden], 0.0, DType::Float32).unwrap(),
            pred_b: Tensor::full(&[joint_hidden], 0.0, DType::Float32).unwrap(),
            out_w: Tensor::full(&[num_classes, joint_hidden], 0.0, DType::Float32).unwrap(),
            out_b: Tensor::full(&[num_classes], 0.0, DType::Float32).unwrap(),
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
    ///
    /// Python reference produces `[B, T, U, V+1]` but at inference T=U=1 so we
    /// skip the explicit unsqueeze; broadcast across the missing axis is
    /// equivalent.
    pub fn forward(&self, enc_t: &Tensor, g: &Tensor) -> Result<Tensor> {
        // Encoder may emit fp16; joint runs in fp32 (matches predictor's f32
        // path so all weights are uniform). Cast enc_t up here.
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
        // policy as the CTC head (`head.rs`).
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

// ─── Head + model wrapper ─────────────────────────────────────────────────

/// RNN-T head = predictor + joint. Composed similarly to Python `RNNTHead`.
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

/// Top-level GigaAM RNN-T model: shared encoder + transducer head + vocab.
///
/// Vocabulary stays a plain `Vec<String>` matching the CTC loader's shape;
/// for `v3_e2e_rnnt` the entries are SentencePiece pieces (with the `▁`
/// space marker), and the post-processing at the call site replaces `▁`
/// with a literal space.
pub struct GigaAmRnnt {
    pub config: GigaAmConfig,
    pub encoder: Encoder,
    pub head: RnntHead,
    pub vocabulary: Vec<String>,
    pub max_symbols_per_step: usize,
    /// True if `vocabulary` entries are SentencePiece pieces (use `▁ → space`
    /// post-processing on output).
    pub sentencepiece: bool,
}

impl GigaAmRnnt {
    pub fn from_hub(model_id: &str) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main")
    }

    pub fn from_hub_with_revision(model_id: &str, revision: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let config_path = repo.get("config.json").context(HubSnafu)?;
        let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
        // SentencePiece RNN-T variants (e.g. `v3_e2e_rnnt`) ship the tokenizer
        // as `tokenizer.model` (SP protobuf). Char-wise variants ship the
        // vocabulary inline in `config.json` and don't have a tokenizer file.
        let tokenizer_path = repo.get("tokenizer.model").ok();
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors_with_tokenizer(&weights_path, tokenizer_path.as_deref(), config)
    }

    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let tokenizer_path = dir.join("tokenizer.model");
        let tokenizer_path = if tokenizer_path.exists() { Some(tokenizer_path) } else { None };
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors_with_tokenizer(&weights_path, tokenizer_path.as_deref(), config)
    }

    /// Load weights + (optional) SentencePiece tokenizer and assemble the
    /// model. When `tokenizer` is provided, the SP pieces (after SP-side
    /// detokenization to natural form, e.g. `▁hello → " hello"`) are used as
    /// the arch decoder's vocabulary, so the decoder's concatenation of
    /// emitted pieces produces a properly detokenized transcript.
    pub fn from_safetensors_with_tokenizer(
        weights: &Path,
        tokenizer: Option<&Path>,
        config: GigaAmConfig,
    ) -> Result<Self> {
        let sd = state::load_safetensors(weights).context(StateSnafu)?;
        let vocab_override = tokenizer.map(load_sentencepiece_vocab).transpose()?;
        Self::from_state_dict(&sd, config, vocab_override)
    }

    /// Build from a pre-loaded state dict. `vocab_override` takes precedence
    /// over `config.transducer.vocabulary` if `Some`.
    pub fn from_state_dict(sd: &StateDict, config: GigaAmConfig, vocab_override: Option<Vec<String>>) -> Result<Self> {
        let transducer = config.transducer.as_ref().ok_or_else(|| super::error::Error::DecoderConfig {
            message: "GigaAmRnnt requires a transducer config (decoding._target_ ending in RNNTGreedyDecoding); \
                 found CTC config"
                .into(),
        })?;
        let pred_hidden = transducer.pred_hidden;
        let pred_rnn_layers = transducer.pred_rnn_layers;
        let joint_hidden = transducer.joint_hidden;
        let num_classes = transducer.num_classes;
        let max_symbols_per_step = transducer.max_symbols_per_step;
        let sentencepiece = transducer.sentencepiece;
        let vocabulary = vocab_override.unwrap_or_else(|| transducer.vocabulary.clone());
        if vocabulary.len() + 1 != num_classes {
            return Err(super::error::Error::DecoderConfig {
                message: format!(
                    "RNN-T vocabulary length + 1 ({}) != num_classes ({}); \
                     convention is one blank token at the end",
                    vocabulary.len() + 1,
                    num_classes
                ),
            });
        }

        let is_pytorch = sd.keys().any(|k| {
            k.starts_with("encoder.")
                || k.starts_with("model.encoder.")
                || k.starts_with("head.decoder.")
                || k.starts_with("head.joint.")
        });
        let sd_owned = if is_pytorch { super::remap::remap_pytorch(sd.clone(), &config)? } else { sd.clone() };
        let sd = &sd_owned;

        let encoder = build_encoder_from_sd(sd, &config)?;

        let mut head = RnntHead::empty(config.d_model, pred_hidden, pred_rnn_layers, joint_hidden, num_classes);
        head.load_state_dict(sd, "head").context(StateSnafu)?;
        head.predictor.prepare_for_inference()?;
        head.joint.cast_to_f32()?;

        Ok(Self { config, encoder, head, vocabulary, max_symbols_per_step, sentencepiece })
    }

    pub fn with_random_weights(config: GigaAmConfig) -> Self {
        let transducer = config.transducer.as_ref().expect("transducer config required");
        let pred_hidden = transducer.pred_hidden;
        let pred_rnn_layers = transducer.pred_rnn_layers;
        let joint_hidden = transducer.joint_hidden;
        let num_classes = transducer.num_classes;
        let max_symbols_per_step = transducer.max_symbols_per_step;
        let sentencepiece = transducer.sentencepiece;
        let vocabulary = transducer.vocabulary.clone();

        let _ = MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            win_length: config.win_length,
            n_mels: config.n_mels,
            center: config.mel_center,
        };
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            win_length: config.win_length,
            n_mels: config.n_mels,
            center: config.mel_center,
        });
        let (cos_cache, sin_cache) = build_rope_cache(&config);
        let subsampling = super::StridingSubsampling::empty(&config);
        let layers = (0..config.n_layers).map(|_| super::ConformerLayer::empty(&config)).collect();
        let encoder = Encoder {
            mel,
            subsampling,
            layers,
            cos_cache,
            sin_cache,
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_encoder_frames: config.max_encoder_frames,
        };

        let head = RnntHead::empty(config.d_model, pred_hidden, pred_rnn_layers, joint_hidden, num_classes);
        Self { config, encoder, head, vocabulary, max_symbols_per_step, sentencepiece }
    }

    /// dtype the encoder + heads operate in (read from the loaded weights).
    pub fn input_dtype(&self) -> DType {
        self.encoder.input_dtype()
    }
}

// ─── LSTM cell ────────────────────────────────────────────────────────────

/// Hand-coded LSTM cell. Inputs:
/// - `x`: `[B, input_size]`
/// - `h`, `c`: `[B, hidden_size]`
/// - `w_ih`: `[4 * hidden_size, input_size]`
/// - `w_hh`: `[4 * hidden_size, hidden_size]`
/// - `b_ih`, `b_hh`: `[4 * hidden_size]`
///
/// Gate order is PyTorch's `[i, f, g, o]`. Matches the Silero VAD cell at
/// `model/src/vad/mod.rs:156` and the reference `nn.LSTM` packing exactly,
/// so PyTorch checkpoints load without gate-axis remapping.
#[allow(clippy::too_many_arguments)]
fn lstm_cell(
    x: &Tensor,
    h: &Tensor,
    c: &Tensor,
    w_ih: &Tensor,
    w_hh: &Tensor,
    b_ih: &Tensor,
    b_hh: &Tensor,
    hidden: usize,
) -> Result<(Tensor, Tensor)> {
    let gates_x = x.linear().weight(w_ih).bias(b_ih).call().context(TensorSnafu)?;
    let gates_h = h.linear().weight(w_hh).bias(b_hh).call().context(TensorSnafu)?;
    let gates = gates_x.try_add(&gates_h).context(TensorSnafu)?;

    let parts = gates.split(&[hidden, hidden, hidden, hidden], 1).context(TensorSnafu)?;
    let i = parts[0].sigmoid().context(TensorSnafu)?;
    let f = parts[1].sigmoid().context(TensorSnafu)?;
    let g = parts[2].tanh().context(TensorSnafu)?;
    let o = parts[3].sigmoid().context(TensorSnafu)?;

    let new_c =
        f.try_mul(c).context(TensorSnafu)?.try_add(&i.try_mul(&g).context(TensorSnafu)?).context(TensorSnafu)?;
    let new_h = o.try_mul(&new_c.tanh().context(TensorSnafu)?).context(TensorSnafu)?;
    Ok((new_h, new_c))
}

// ─── SentencePiece tokenizer loader ───────────────────────────────────────

/// Minimal subset of the SentencePiece `ModelProto` schema needed to read
/// out the `pieces` array. We don't need the trainer/normalizer specs, the
/// score field, or any other top-level fields — prost silently skips
/// unknown tags during decode, so this partial schema suffices.
///
/// Source of truth for tags: `submodules/GigaAM/.../sentencepiece_model.proto`
/// (or upstream `google/sentencepiece` repo `src/sentencepiece_model.proto`).
#[derive(prost::Message)]
struct SpModelProto {
    #[prost(message, repeated, tag = "1")]
    pieces: Vec<SpPiece>,
}

#[derive(prost::Message)]
struct SpPiece {
    /// The piece string, e.g. `"▁hello"` (`U+2581` = SP space marker) or
    /// `"<unk>"` for control tokens.
    #[prost(string, optional, tag = "1")]
    piece: Option<String>,
    /// `enum Type { NORMAL = 1; UNKNOWN = 2; CONTROL = 3; USER_DEFINED = 4; BYTE = 6; UNUSED = 5 }`.
    #[prost(int32, optional, tag = "3")]
    r#type: Option<i32>,
}

/// Read a SentencePiece `.model` file and return per-id raw pieces. Pieces
/// retain their `▁` (U+2581) prefix on word-initial tokens; the call site
/// concatenates and replaces `▁` with a space for natural detokenization.
///
/// Special tokens (UNKNOWN=2, CONTROL=3, BYTE=6, UNUSED=5) are mapped to
/// the empty string so they elide from the transcript on the (rare) chance
/// the model emits one.
fn load_sentencepiece_vocab(path: &Path) -> Result<Vec<String>> {
    use prost::Message;
    let bytes = std::fs::read(path).context(super::error::ConfigIoSnafu)?;
    let proto = SpModelProto::decode(&*bytes).map_err(|e| super::error::Error::DecoderConfig {
        message: format!("failed to parse SentencePiece model at {}: {e}", path.display()),
    })?;
    let mut pieces = Vec::with_capacity(proto.pieces.len());
    for (i, p) in proto.pieces.into_iter().enumerate() {
        let kind = p.r#type.unwrap_or(1); // default = NORMAL
        // Type 1 = NORMAL, 4 = USER_DEFINED. Everything else (UNKNOWN,
        // CONTROL, BYTE, UNUSED) is non-emittable: store empty so the
        // transcript stays clean if the predictor accidentally lands there.
        let s = if kind == 1 || kind == 4 { p.piece.unwrap_or_default() } else { String::new() };
        let _ = i;
        pieces.push(s);
    }
    Ok(pieces)
}

// ─── Step backend (impl arch::rnnt::JointStep) ────────────────────────────

use std::sync::Arc;

use morok_arch::rnnt::JointStep;

use super::{RnntJointStepJit, RnntPredictorStepJit};

/// Per-utterance RNN-T step backend. Wraps the predictor and joint JITs +
/// committed/tentative LSTM state, implementing
/// [`morok_arch::rnnt::JointStep`].
///
/// For B=1 the search loop owns one of these and drives it through the
/// per-frame inner loop. JIT plans (the heavy ones — predictor + joint) are
/// prepared once at construction and reused; the only per-step overhead is
/// the buffer-pack / execute / read-out cycle.
pub struct RnntStepBackend {
    predictor_jit: RnntPredictorStepJit,
    joint_jit: RnntJointStepJit,

    /// Latest committed LSTM hidden state, flat layout `[L * P]` row-major
    /// (layer-major). Reset to zeros at the start of each utterance.
    committed_h: Vec<f32>,
    committed_c: Vec<f32>,
    /// Post-step LSTM state stash. The search loop calls `commit` to promote
    /// these into `committed_*` on a non-blank emission.
    tentative_h: Vec<f32>,
    tentative_c: Vec<f32>,
    /// Last predictor `g` output (`[P]`). Stashed here so we can drop the
    /// predictor's output borrow before mutably accessing the joint JIT's
    /// input buffer.
    g_tentative: Vec<f32>,

    blank_id: usize,
    pred_hidden: usize,
    pred_rnn_layers: usize,
    enc_hidden: usize,
    total_vocab: usize,

    /// Per-step timing aggregates. Reset by [`reset_stats`]; printed by
    /// [`stats`]. Cheap (one `Instant::now()` per substage) — kept always-on so
    /// the example can profile without recompilation.
    pub stats: StepStats,
}

/// Aggregate timings for [`RnntStepBackend`]. Six sub-stages per `step` call
/// + commit/reset counters; printed via [`Display`](std::fmt::Display).
#[derive(Default, Clone, Debug)]
pub struct StepStats {
    pub n_steps: u64,
    pub n_commits: u64,
    pub n_resets: u64,
    pub t_pred_pack: std::time::Duration,
    pub t_pred_exec: std::time::Duration,
    pub t_pred_read: std::time::Duration,
    pub t_joint_pack: std::time::Duration,
    pub t_joint_exec: std::time::Duration,
    pub t_joint_read: std::time::Duration,
}

impl StepStats {
    pub fn total(&self) -> std::time::Duration {
        self.t_pred_pack
            + self.t_pred_exec
            + self.t_pred_read
            + self.t_joint_pack
            + self.t_joint_exec
            + self.t_joint_read
    }
}

impl RnntStepBackend {
    /// Build the backend from a shared model. Constructs predictor + joint
    /// JIT plans (one of each) and zero state buffers.
    pub fn from_model(model: Arc<GigaAmRnnt>) -> crate::jit::Result<Self> {
        let pred_hidden = model.head.pred_hidden;
        let pred_rnn_layers = model.head.pred_rnn_layers;
        let total_vocab = model.head.num_classes;
        let blank_id = total_vocab - 1;
        let enc_hidden = model.config.d_model;
        let lp = pred_rnn_layers * pred_hidden;

        // Placeholder inputs for `prepare()`. Shape and dtype must match the
        // tensors we'll feed each step. JIT captures their buffer ids and
        // reuses them as in-place input slots.
        let mut prev_token = Tensor::full(&[1, 1], blank_id as i64, DType::Int64)
            .map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        prev_token.realize().map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;

        let mut h_in = Tensor::full(&[pred_rnn_layers, 1, pred_hidden], 0.0f32, DType::Float32)
            .map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        h_in.realize().map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;

        let mut c_in = Tensor::full(&[pred_rnn_layers, 1, pred_hidden], 0.0f32, DType::Float32)
            .map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        c_in.realize().map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;

        let mut enc_t = Tensor::full(&[1, 1, enc_hidden], 0.0f32, DType::Float32)
            .map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        enc_t.realize().map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;

        let mut g = Tensor::full(&[1, 1, pred_hidden], 0.0f32, DType::Float32)
            .map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;
        g.realize().map_err(|e| crate::jit::JitError::Tensor { source: Box::new(e) })?;

        let mut predictor_jit = RnntPredictorStepJit::new(Arc::clone(&model));
        predictor_jit.prepare(&prev_token, &h_in, &c_in)?;

        let mut joint_jit = RnntJointStepJit::new(Arc::clone(&model));
        joint_jit.prepare(&enc_t, &g)?;

        Ok(Self {
            predictor_jit,
            joint_jit,
            committed_h: vec![0.0f32; lp],
            committed_c: vec![0.0f32; lp],
            tentative_h: vec![0.0f32; lp],
            tentative_c: vec![0.0f32; lp],
            g_tentative: vec![0.0f32; pred_hidden],
            blank_id,
            pred_hidden,
            pred_rnn_layers,
            enc_hidden,
            total_vocab,
            stats: StepStats::default(),
        })
    }

    pub fn total_vocab(&self) -> usize {
        self.total_vocab
    }

    pub fn reset_stats(&mut self) {
        self.stats = StepStats::default();
    }
}

impl JointStep for RnntStepBackend {
    type Error = crate::jit::JitError;

    fn step(
        &mut self,
        encoder_frame: &[f32],
        prev_token: Option<usize>,
        logits_out: &mut [f32],
    ) -> std::result::Result<(), Self::Error> {
        debug_assert_eq!(encoder_frame.len(), self.enc_hidden);
        debug_assert_eq!(logits_out.len(), self.total_vocab);

        let t0 = std::time::Instant::now();
        // 1. Pack prev_token (or blank_id) into the predictor's prev_token input.
        {
            let buf = self.predictor_jit.prev_token_mut()?;
            let mut view =
                buf.as_array_mut::<i64>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            let slice = view.as_slice_mut().expect("contiguous prev_token");
            slice[0] = prev_token.unwrap_or(self.blank_id) as i64;
        }
        // 2. Pack committed h, c into predictor inputs.
        {
            let buf = self.predictor_jit.h_in_mut()?;
            let mut view =
                buf.as_array_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            view.as_slice_mut().expect("contiguous h_in").copy_from_slice(&self.committed_h);
        }
        {
            let buf = self.predictor_jit.c_in_mut()?;
            let mut view =
                buf.as_array_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            view.as_slice_mut().expect("contiguous c_in").copy_from_slice(&self.committed_c);
        }
        let t1 = std::time::Instant::now();
        // 3. Execute predictor.
        self.predictor_jit.execute()?;
        let t2 = std::time::Instant::now();

        // 4. Read predictor output and split into g | h_out | c_out. Copy
        //    everything into self-owned buffers so the predictor's output
        //    borrow ends before we touch the joint JIT.
        {
            let out = self.predictor_jit.output()?;
            let arr = out.as_array::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            let flat = arr.as_slice().expect("contiguous predictor output");
            let p = self.pred_hidden;
            let lp = self.pred_rnn_layers * self.pred_hidden;
            debug_assert_eq!(flat.len(), p + 2 * lp);
            self.g_tentative.copy_from_slice(&flat[..p]);
            self.tentative_h.copy_from_slice(&flat[p..p + lp]);
            self.tentative_c.copy_from_slice(&flat[p + lp..p + 2 * lp]);
        }
        let t3 = std::time::Instant::now();

        // 5. Pack enc_t and g into joint inputs.
        {
            let buf = self.joint_jit.enc_t_mut()?;
            let mut view =
                buf.as_array_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            view.as_slice_mut().expect("contiguous enc_t").copy_from_slice(encoder_frame);
        }
        {
            let buf = self.joint_jit.g_mut()?;
            let mut view =
                buf.as_array_mut::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            view.as_slice_mut().expect("contiguous g").copy_from_slice(&self.g_tentative);
        }
        let t4 = std::time::Instant::now();
        // 6. Execute joint.
        self.joint_jit.execute()?;
        let t5 = std::time::Instant::now();

        // 7. Copy log-probs into the caller's slice.
        {
            let out = self.joint_jit.output()?;
            let arr = out.as_array::<f32>().map_err(|e| crate::jit::JitError::Build { source: Box::new(e) })?;
            let flat = arr.as_slice().expect("contiguous joint output");
            // `flat.len() == 1 * 1 * total_vocab` for the [1, 1, V+1] joint output.
            logits_out.copy_from_slice(&flat[..self.total_vocab]);
        }
        let t6 = std::time::Instant::now();

        self.stats.n_steps += 1;
        self.stats.t_pred_pack += t1 - t0;
        self.stats.t_pred_exec += t2 - t1;
        self.stats.t_pred_read += t3 - t2;
        self.stats.t_joint_pack += t4 - t3;
        self.stats.t_joint_exec += t5 - t4;
        self.stats.t_joint_read += t6 - t5;
        Ok(())
    }

    fn commit(&mut self) {
        self.stats.n_commits += 1;
        std::mem::swap(&mut self.committed_h, &mut self.tentative_h);
        std::mem::swap(&mut self.committed_c, &mut self.tentative_c);
    }

    fn reset(&mut self) {
        self.stats.n_resets += 1;
        self.committed_h.fill(0.0);
        self.committed_c.fill(0.0);
    }
}

// ConfigSnafu is reserved for future config-error context capturing in this
// module — silence the unused-import warning until the first use lands.
#[allow(dead_code)]
fn _config_snafu_marker() {
    let _ = ConfigSnafu;
}
