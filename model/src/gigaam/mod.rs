mod encoder;
mod error;
mod head;
pub(crate) mod remap;
mod rnnt;
mod rope;

pub use encoder::*;
pub use error::{Error, Result};
pub use head::*;
pub use rnnt::*;
pub use rope::*;

extern crate self as morok_model;

use std::path::Path;

use morok_arch::ctc::{CtcDecoder, GreedyDecoder};
use morok_dtype::DType;
use morok_ir::SInt;
use morok_macros::jit_wrapper;
use morok_tensor::{BoundVariable, Tensor};
use snafu::ResultExt;

use crate::audio::{MelConfig, MelSpectrogram};
use crate::state::{self, HasStateDict, StateDict};

use error::{ConfigIoSnafu, ConfigSnafu, StateSnafu, TensorSnafu};

pub enum SubsamplingMode {
    Conv1d,
    Conv2d,
}

pub enum ConvNormType {
    LayerNorm,
    BatchNorm,
}

pub struct GigaAmConfig {
    pub max_batch_size: usize,
    pub n_mels: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub conv_kernel: usize,
    pub subsampling_factor: usize,
    pub subsampling_mode: SubsamplingMode,
    pub subs_kernel_size: usize,
    pub conv_norm_type: ConvNormType,
    pub vocab_size: usize,
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub mel_center: bool,
    pub max_mel_frames: usize,
    pub max_encoder_frames: usize,
    /// CTC decoder built from the `decoding` section of the config, or an
    /// empty-vocabulary greedy decoder for synthetic configs that don't
    /// declare one.
    pub decoder: CtcDecoder,
    /// Transducer-specific config, populated when `decoding._target_` ends
    /// in `RNNTGreedyDecoding` (or the head config has predictor/joint
    /// blocks). `None` for CTC checkpoints.
    pub transducer: Option<TransducerConfig>,
}

/// RNN-T-specific config extracted from the JSON `head.decoder` /
/// `head.joint` / `decoding` blocks. See `submodules/GigaAM/gigaam/decoder.py`
/// for the reference shape.
#[derive(Clone, Debug)]
pub struct TransducerConfig {
    pub pred_hidden: usize,
    pub pred_rnn_layers: usize,
    pub joint_hidden: usize,
    /// `vocabulary.len() + 1` — includes the blank token at the end.
    pub num_classes: usize,
    pub max_symbols_per_step: usize,
    pub vocabulary: Vec<String>,
    /// True when the vocabulary entries are SentencePiece pieces (apply
    /// `▁ → space` post-processing on the decoded string).
    pub sentencepiece: bool,
}

impl GigaAmConfig {
    pub fn from_json(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).context(ConfigIoSnafu)?;
        let root: serde_json::Value = serde_json::from_str(&data).context(ConfigSnafu)?;
        let cfg = &root["cfg"]["model"]["cfg"];
        let pre = &cfg["preprocessor"];
        let enc = &cfg["encoder"];
        let head = &cfg["head"];
        let decoding = &cfg["decoding"];

        let d_model = enc["d_model"].as_u64().expect("d_model") as usize;
        let ff_expansion_factor = enc["ff_expansion_factor"].as_u64().expect("ff_expansion_factor") as usize;

        let subsampling_str = enc["subsampling"].as_str().unwrap_or("conv2d");
        let subsampling_mode = match subsampling_str {
            "conv1d" => SubsamplingMode::Conv1d,
            _ => SubsamplingMode::Conv2d,
        };

        let conv_norm_str = enc["conv_norm_type"].as_str().unwrap_or("batch_norm");
        let conv_norm_type = match conv_norm_str {
            "layer_norm" => ConvNormType::LayerNorm,
            _ => ConvNormType::BatchNorm,
        };

        let subsampling_factor = enc["subsampling_factor"].as_u64().expect("subsampling_factor") as usize;
        let subs_kernel_size = enc["subs_kernel_size"].as_u64().unwrap_or(3) as usize;
        let max_encoder_frames = enc["pos_emb_max_len"].as_u64().unwrap_or(5000) as usize;
        // `max_mel_frames` is the pre-subsampling sequence-length bound. Configs that
        // only specify `pos_emb_max_len` (the post-subsampling encoder bound) need it
        // multiplied by `subsampling_factor` so audio approaching the encoder cap
        // isn't rejected at the JIT input stage.
        let max_mel_frames = enc["max_mel_frames"]
            .as_u64()
            .or_else(|| enc["max_seq_len"].as_u64())
            .unwrap_or((max_encoder_frames * subsampling_factor) as u64) as usize;

        // CTC configs put `num_classes` directly on `head`; RNN-T configs nest
        // it under `head.decoder.num_classes` / `head.joint.num_classes`.
        let vocab_size = head["num_classes"]
            .as_u64()
            .or_else(|| head["decoder"]["num_classes"].as_u64())
            .or_else(|| head["joint"]["num_classes"].as_u64())
            .expect("num_classes (head.num_classes or head.{decoder,joint}.num_classes)")
            as usize;
        let decoder = build_decoder(decoding, vocab_size)?;
        let transducer = build_transducer(head, decoding, vocab_size, d_model)?;

        Ok(Self {
            max_batch_size: enc["max_batch_size"].as_u64().unwrap_or(32) as usize,
            n_mels: pre["features"].as_u64().expect("features") as usize,
            d_model,
            n_heads: enc["n_heads"].as_u64().expect("n_heads") as usize,
            n_layers: enc["n_layers"].as_u64().expect("n_layers") as usize,
            d_ff: d_model * ff_expansion_factor,
            conv_kernel: enc["conv_kernel_size"].as_u64().expect("conv_kernel_size") as usize,
            subsampling_factor,
            subsampling_mode,
            subs_kernel_size,
            conv_norm_type,
            vocab_size,
            sample_rate: pre["sample_rate"].as_u64().expect("sample_rate") as usize,
            n_fft: pre["n_fft"].as_u64().expect("n_fft") as usize,
            hop_length: pre["hop_length"].as_u64().expect("hop_length") as usize,
            win_length: pre["win_length"].as_u64().expect("win_length") as usize,
            mel_center: pre["center"].as_bool().unwrap_or(true),
            max_mel_frames,
            max_encoder_frames,
            decoder,
            transducer,
        })
    }
}

/// Build a [`TransducerConfig`] from the `head` and `decoding` blocks of a
/// GigaAM config, or `None` if the head doesn't declare a transducer (CTC
/// checkpoints).
fn build_transducer(
    head: &serde_json::Value,
    decoding: &serde_json::Value,
    vocab_size: usize,
    d_model: usize,
) -> Result<Option<TransducerConfig>> {
    let _ = d_model;
    // Detect: either decoding._target_ names RNNT, or head.{decoder,joint}
    // sub-blocks exist.
    let target = decoding["_target_"].as_str().unwrap_or("");
    let has_decoder_block = head.get("decoder").map(|v| !v.is_null()).unwrap_or(false);
    let has_joint_block = head.get("joint").map(|v| !v.is_null()).unwrap_or(false);
    if !(target.contains("RNNT") || has_decoder_block && has_joint_block) {
        return Ok(None);
    }

    let dec = &head["decoder"];
    let joint = &head["joint"];
    let pred_hidden = dec["pred_hidden"].as_u64().expect("head.decoder.pred_hidden") as usize;
    let pred_rnn_layers = dec["pred_rnn_layers"].as_u64().expect("head.decoder.pred_rnn_layers") as usize;
    let joint_hidden = joint["joint_hidden"].as_u64().expect("head.joint.joint_hidden") as usize;
    let max_symbols_per_step = decoding["max_symbols_per_step"].as_u64().unwrap_or(10) as usize;

    // Vocabulary preference: `decoding.vocabulary` (CTC convention reused for
    // RNN-T configs) or `tokenizer.vocab` if the tokenizer is char-wise.
    // For SentencePiece RNN-T checkpoints (e.g. v3_e2e_rnnt) the JSON config
    // typically stores only the size; the actual pieces ship as
    // `tokenizer.txt` and are loaded via `from_safetensors_with_vocab`. Empty
    // here is fine — `from_state_dict` will splice in the override.
    let vocabulary: Vec<String> = decoding["vocabulary"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    // Heuristic: SentencePiece if (a) decoding declares a non-empty model_path
    // or (b) the vocabulary is empty (will be loaded from tokenizer.txt) and
    // the model id hints at SP. For now: SP iff `decoding.model_path` is
    // non-null, else char-wise.
    let sentencepiece = decoding.get("model_path").and_then(|v| v.as_str()).map(|s| !s.is_empty()).unwrap_or(false);

    Ok(Some(TransducerConfig {
        pred_hidden,
        pred_rnn_layers,
        joint_hidden,
        num_classes: vocab_size,
        max_symbols_per_step,
        vocabulary,
        sentencepiece,
    }))
}

/// Construct a [`CtcDecoder`] from the `decoding` block of a GigaAM config.
///
/// Dispatches on the (PyTorch/Hydra-style) `_target_` string. The trailing
/// fields of the JSON object are deserialized into the leaf decoder type via
/// `serde_json::from_value` — `_target_` itself is silently ignored by serde.
///
/// On a missing/empty block we fall back to an empty-vocabulary
/// [`GreedyDecoder`] so synthetic configs (no `decoding` section) still
/// round-trip through `from_json`.
fn build_decoder(decoding: &serde_json::Value, vocab_size: usize) -> Result<CtcDecoder> {
    if decoding.is_null() {
        return Ok(CtcDecoder::Greedy(GreedyDecoder::new(Vec::new())));
    }
    let target = decoding["_target_"].as_str().unwrap_or("");
    let decoder: CtcDecoder = if target.contains("CTCGreedyDecoding") {
        let g: GreedyDecoder = serde_json::from_value(decoding.clone()).context(ConfigSnafu)?;
        CtcDecoder::Greedy(g)
    } else if target.contains("CTCBeamDecoding") {
        let b: morok_arch::ctc::BeamDecoder = serde_json::from_value(decoding.clone()).context(ConfigSnafu)?;
        CtcDecoder::Beam(Box::new(b))
    } else {
        // Unknown / missing target. If there's a vocabulary array, default to
        // greedy; otherwise empty.
        let vocab: Vec<String> = decoding["vocabulary"]
            .as_array()
            .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
            .unwrap_or_default();
        CtcDecoder::Greedy(GreedyDecoder::new(vocab))
    };
    if !decoder.vocabulary().is_empty() && decoder.total_vocab() != vocab_size {
        return Err(error::Error::DecoderConfig {
            message: format!(
                "decoder vocabulary length + 1 ({}) != head.num_classes ({}); \
                 CTC convention is one blank token appended after the vocabulary",
                decoder.total_vocab(),
                vocab_size
            ),
        });
    }
    Ok(decoder)
}

/// Audio preprocessor + Conformer encoder. Shared by `GigaAm` (CTC) and
/// `GigaAmRnnt` (transducer); they layer different heads on top of the same
/// encoder. Encoder-only path: `forward` for single-batch, `forward_batch`
/// for batched JIT execution.
pub struct Encoder {
    pub mel: MelSpectrogram,
    pub subsampling: StridingSubsampling,
    pub layers: Vec<ConformerLayer>,
    pub cos_cache: Tensor,
    pub sin_cache: Tensor,
    pub d_model: usize,
    pub n_heads: usize,
    pub max_encoder_frames: usize,
}

impl Encoder {
    /// dtype the encoder operates in. Read off the first subsampling
    /// conv weight (the model's compute dtype is determined by the
    /// weights it was loaded with). Falls back to f32 when the weight
    /// isn't itself a float type — should never happen in practice but
    /// avoids producing an integer dtype here.
    pub fn input_dtype(&self) -> DType {
        let dtype = self.subsampling.conv1_weight.uop().dtype();
        if dtype.is_float() { dtype } else { DType::Float32 }
    }

    /// Encoder pass on a single mel batch with no padding mask.
    /// Input: tensor `[B, n_mels, T]`. Output: lazy tensor `[B, d_model, T/4]`.
    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = x.cast(self.input_dtype()).context(TensorSnafu)?;
        let x = self.subsampling.forward(&x)?;

        let shape = x.shape().context(TensorSnafu)?;
        let seq_len = shape[1].clone();

        let d_half = self.d_model / self.n_heads / 2;

        let cos = self
            .cos_cache
            .try_shrink([
                (SInt::Const(0), seq_len.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;
        let sin = self
            .sin_cache
            .try_shrink([
                (SInt::Const(0), seq_len.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, None, None)?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    /// Batched encoder path with dynamic batch and mel-frame length.
    /// Input: `mel` `[B, n_mels, T_mel]`, `lengths` `[B]` valid lengths in mel frames.
    /// Output: `[B, d_model, T_sub]`.
    pub fn forward_batch(
        &self,
        mel: &Tensor,
        lengths: &Tensor,
        batch: &BoundVariable,
        mel_len: &BoundVariable,
    ) -> Result<Tensor> {
        let b = batch.as_sint();
        let t_mel = mel_len.as_sint();

        let lengths = lengths.try_shrink([Some((SInt::Const(0), b.clone()))]).context(TensorSnafu)?;
        let lengths = lengths.cast(DType::Index).context(TensorSnafu)?;

        let two_t = Tensor::const_(2i64, DType::Index);
        let one_t = Tensor::const_(1i64, DType::Index);

        let mut lengths_sub = lengths;
        for _ in 0..2 {
            lengths_sub = lengths_sub.try_add(&one_t).context(TensorSnafu)?.try_div(&two_t).context(TensorSnafu)?;
        }

        let mel = mel
            .try_shrink([Some((SInt::Const(0), b.clone())), None, Some((SInt::Const(0), t_mel))])
            .context(TensorSnafu)?;
        let x = mel.try_transpose(-1, -2).context(TensorSnafu)?;
        let x = x.cast(self.input_dtype()).context(TensorSnafu)?;
        let x = self.subsampling.forward(&x)?;

        let shape = x.shape().context(TensorSnafu)?;
        let t_sub = shape[1].clone();

        let range = Tensor::arange(self.max_encoder_frames as i64, None, None).context(TensorSnafu)?;
        let range = range.cast(DType::Index).context(TensorSnafu)?;
        let range = range.try_shrink([(SInt::Const(0), t_sub.clone())]).context(TensorSnafu)?;
        let range = range.try_reshape([SInt::Const(1), t_sub.clone()]).context(TensorSnafu)?;
        let lens = lengths_sub;
        let lens = lens.try_reshape([b.clone(), SInt::Const(1)]).context(TensorSnafu)?;
        let pad_valid = range.try_lt(&lens).context(TensorSnafu)?;

        let pv1 = pad_valid.try_unsqueeze(1).context(TensorSnafu)?;
        let pv2 = pad_valid.try_unsqueeze(2).context(TensorSnafu)?;
        let att_mask = Some(
            pv1.bitwise_and(&pv2)
                .context(TensorSnafu)?
                .logical_not()
                .context(TensorSnafu)?
                .try_unsqueeze(1)
                .context(TensorSnafu)?,
        );
        let pad_mask = pad_valid.logical_not().context(TensorSnafu)?;

        let d_half = self.d_model / self.n_heads / 2;
        let cos = self
            .cos_cache
            .try_shrink([
                (SInt::Const(0), t_sub.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;
        let sin = self
            .sin_cache
            .try_shrink([
                (SInt::Const(0), t_sub.clone()),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(1)),
                (SInt::Const(0), SInt::Const(d_half)),
            ])
            .context(TensorSnafu)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, &cos, &sin, att_mask.as_ref(), Some(&pad_mask))?;
        }

        x.try_transpose(-1, -2).context(TensorSnafu)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.subsampling.output_length(mel_frames)
    }
}

/// GigaAM model: audio preprocessor + Conformer encoder + CTC head.
pub struct GigaAm {
    pub config: GigaAmConfig,
    pub encoder: Encoder,
    pub head: CTCHead,
}

impl GigaAm {
    /// Load from a HuggingFace Hub repository.
    pub fn from_hub(model_id: &str) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main")
    }

    /// Load from a HuggingFace Hub repository at a specific branch/revision.
    pub fn from_hub_with_revision(model_id: &str, revision: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(error::HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let config_path = repo.get("config.json").context(error::HubSnafu)?;
        let weights_path = repo.get("model.safetensors").context(error::HubSnafu)?;
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors(&weights_path, config)
    }

    /// Load from a safetensors file with a config.json in the same directory.
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let config = GigaAmConfig::from_json(&config_path)?;
        Self::from_safetensors(&weights_path, config)
    }

    /// Load a GigaAM model from a safetensors file.
    pub fn from_safetensors(path: &Path, config: GigaAmConfig) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Self::from_state_dict(&sd, config)
    }

    /// Build from a pre-loaded state dict.
    ///
    /// Auto-detects PyTorch key format (keys starting with `encoder.` or `model.encoder.`) and remaps.
    pub fn from_state_dict(sd: &StateDict, config: GigaAmConfig) -> Result<Self> {
        let is_pytorch = sd.keys().any(|k| k.starts_with("encoder.") || k.starts_with("model.encoder."));
        let sd_owned = if is_pytorch { remap::remap_pytorch(sd.clone(), &config)? } else { sd.clone() };
        let sd = &sd_owned;

        let encoder = build_encoder_from_sd(sd, &config)?;

        let mut head = CTCHead::empty(&config);
        head.load_state_dict(sd, "head").context(StateSnafu)?;

        Ok(Self { config, encoder, head })
    }

    pub fn with_random_weights(config: GigaAmConfig) -> Self {
        let encoder = Encoder::with_random_weights(&config);
        let head = CTCHead::empty(&config);
        Self { config, encoder, head }
    }

    /// Run full inference: waveform -> CTC log-probabilities.
    ///
    /// Input: raw audio samples at 16kHz, mono, float32.
    /// Output: lazy tensor `[1, vocab_size, T/4]` of log-probabilities.
    pub fn forward(&self, waveform: &[f32], mel_tensor: &mut Tensor) -> Result<Tensor> {
        {
            let mut view = mel_tensor.array_view_mut::<f32>().context(TensorSnafu)?;
            self.encoder.mel.forward_into(waveform, &mut view);
        }
        let encoded = self.encode(mel_tensor)?;
        self.head.forward(&encoded)
    }

    /// Encoder-only: mel features -> encoded representation.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.encoder.subsampling_output_length(mel_frames)
    }

    /// Batched encoder path with dynamic batch and mel-frame length.
    pub fn encode_batch(
        &self,
        mel: &Tensor,
        lengths: &Tensor,
        batch: &BoundVariable,
        mel_len: &BoundVariable,
    ) -> Result<Tensor> {
        self.encoder.forward_batch(mel, lengths, batch, mel_len)
    }

    /// dtype the encoder + heads operate in (read from the loaded weights).
    pub fn input_dtype(&self) -> DType {
        self.encoder.input_dtype()
    }
}

impl Encoder {
    pub fn with_random_weights(config: &GigaAmConfig) -> Self {
        let mel = MelSpectrogram::new(&MelConfig {
            sample_rate: config.sample_rate,
            n_fft: config.n_fft,
            hop_length: config.hop_length,
            win_length: config.win_length,
            n_mels: config.n_mels,
            center: config.mel_center,
        });
        let (cos_cache, sin_cache) = build_rope_cache(config);
        let subsampling = StridingSubsampling::empty(config);
        let layers = (0..config.n_layers).map(|_| ConformerLayer::empty(config)).collect();
        Self {
            mel,
            subsampling,
            layers,
            cos_cache,
            sin_cache,
            d_model: config.d_model,
            n_heads: config.n_heads,
            max_encoder_frames: config.max_encoder_frames,
        }
    }
}

/// Construct an `Encoder` from an already-remapped state dict + config.
/// Shared by `GigaAm::from_state_dict` and `GigaAmRnnt::from_state_dict`
/// (in `rnnt.rs`).
pub(crate) fn build_encoder_from_sd(sd: &StateDict, config: &GigaAmConfig) -> Result<Encoder> {
    let mel = MelSpectrogram::new(&MelConfig {
        sample_rate: config.sample_rate,
        n_fft: config.n_fft,
        hop_length: config.hop_length,
        win_length: config.win_length,
        n_mels: config.n_mels,
        center: config.mel_center,
    });
    let (cos_cache, sin_cache) = build_rope_cache(config);

    let mut subsampling = StridingSubsampling::empty(config);
    subsampling.load_state_dict(sd, "subsampling").context(StateSnafu)?;

    let mut layers = Vec::with_capacity(config.n_layers);
    for i in 0..config.n_layers {
        let mut layer = ConformerLayer::empty(config);
        layer.load_state_dict(sd, &format!("layers.{i}")).context(StateSnafu)?;
        layers.push(layer);
    }

    Ok(Encoder {
        mel,
        subsampling,
        layers,
        cos_cache,
        sin_cache,
        d_model: config.d_model,
        n_heads: config.n_heads,
        max_encoder_frames: config.max_encoder_frames,
    })
}

jit_wrapper! {
    GigaAmJit(GigaAm) {
        mel: Tensor,

        build(mel) {
            let encoded = model.encode(mel)?;
            model.head.forward(&encoded)
        }
    }
}

jit_wrapper! {
    GigaAmBatchedJit(GigaAm) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t: (1, model.config.max_mel_frames),
        }

        build(mel, lengths, b, t) {
            let encoded = model.encode_batch(mel, lengths, &b, &t)?;
            model.head.forward(&encoded)
        }
    }
}

// ─── RNN-T JITs ────────────────────────────────────────────────────────────
//
// Encoder JIT for RNN-T is encoder-only (no head); the head's predictor +
// joint run as their own per-step JITs since their input shape depends on
// `prev_token` and the LSTM state, which evolve through the search loop.

// All three JITs take an `Arc<GigaAmRnnt>` so the example can build them from
// a single underlying model — Tensor weights are Arc-backed and shared across
// clones, so the duplication is structural only.
jit_wrapper! {
    GigaAmRnntEncoderJit(std::sync::Arc<GigaAmRnnt>) {
        mel: Tensor,
        lengths: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
            t: (1, model.config.max_mel_frames),
        }

        build(mel, lengths, b, t) {
            let out = model.encoder.forward_batch(mel, lengths, &b, &t)?;
            // Encoder may run in fp16 (depending on weight dtype); promote
            // to fp32 at the JIT boundary so the joint step + the host-side
            // copyout are uniform.
            out.cast(morok_dtype::DType::Float32).context(TensorSnafu)
        }
    }
}

jit_wrapper! {
    RnntPredictorStepJit(std::sync::Arc<GigaAmRnnt>) {
        prev_token: Tensor,
        h_in: Tensor,
        c_in: Tensor,

        build(prev_token, h_in, c_in) {
            model.head.predictor.forward_concat(prev_token, h_in, c_in)
        }
    }
}

jit_wrapper! {
    RnntJointStepJit(std::sync::Arc<GigaAmRnnt>) {
        enc_t: Tensor,
        g: Tensor,

        build(enc_t, g) {
            model.head.joint.forward(enc_t, g)
        }
    }
}
