//! Unified GigaAM model: shared Conformer encoder + variant head (CTC | RN-T).
//!
//! `GigaAm` collapses the previously parallel `GigaAm` (CTC) and
//! `GigaAmRnnt` (RN-T) types into one config-driven struct. The
//! `config.transducer.is_some()` discriminator picks the head at load time:
//! `None` ⇒ `Head::Ctc(CTCHead)`, `Some(_)` ⇒ `Head::Rnnt { head, runtime }`
//! where `RnntRuntime` carries the RN-T-only inference metadata (vocabulary,
//! max symbols per step, SentencePiece flag).
//!
//! The decoder layer in `morok_arch` is not unified by this struct — CTC and
//! RN-T still use their respective `CtcDecoder` / `JointStep` shapes — but the
//! model construction, weight loading, and encoder JIT all flow through one
//! type.

use std::path::Path;

use morok_dtype::DType;
use morok_tensor::{BoundVariable, Tensor};
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict};

use crate::gigaam::ctc::CTCHead;
use crate::gigaam::encoder::{Encoder, build_encoder_from_sd};
use crate::gigaam::error::{Error, HubSnafu, StateSnafu, TensorSnafu};
use crate::gigaam::rnnt::{RnntHead, load_sentencepiece_vocab};
use crate::gigaam::{GigaAmConfig, Result, remap};

/// Unified GigaAM model. The `head` enum carries either a CTC projection or
/// an RN-T predictor+joint pair; pattern-match (or use [`Head::as_ctc`] /
/// [`Head::as_rnnt`]) to drive the head-specific inference path.
pub struct GigaAm {
    pub config: GigaAmConfig,
    pub encoder: Encoder,
    pub head: Head,
}

/// Head variant. `Ctc` holds the small Conv1d projection consumed by
/// `morok_arch::ctc` decoders. `Rnnt` holds the predictor+joint pair plus
/// the runtime metadata (vocab, max-symbols-per-step, SP flag) used by the
/// arch's `JointStep`-driven decoder.
pub enum Head {
    Ctc(CTCHead),
    Rnnt { head: RnntHead, runtime: RnntRuntime },
}

/// RN-T-only runtime metadata. Lives inside [`Head::Rnnt`] so the CTC path
/// stays free of fields it would never use.
pub struct RnntRuntime {
    /// Token strings indexed by predictor class. Length is `num_classes - 1`
    /// (the last class is the blank, not a vocabulary entry).
    pub vocabulary: Vec<String>,
    /// Max non-blank emissions per encoder frame in the greedy search.
    pub max_symbols_per_step: usize,
    /// `true` if `vocabulary` is SentencePiece pieces (post-process `▁` → space
    /// on the output transcript).
    pub sentencepiece: bool,
}

impl Head {
    pub fn is_ctc(&self) -> bool {
        matches!(self, Head::Ctc(_))
    }

    pub fn is_rnnt(&self) -> bool {
        matches!(self, Head::Rnnt { .. })
    }

    pub fn as_ctc(&self) -> Option<&CTCHead> {
        if let Head::Ctc(h) = self { Some(h) } else { None }
    }

    pub fn as_rnnt(&self) -> Option<(&RnntHead, &RnntRuntime)> {
        if let Head::Rnnt { head, runtime } = self { Some((head, runtime)) } else { None }
    }

    /// Infallible accessor. Panics with a clear message if the head is RN-T.
    pub fn ctc(&self) -> &CTCHead {
        self.as_ctc().expect("head variant is RN-T, expected CTC")
    }

    /// Infallible accessor. Panics with a clear message if the head is CTC.
    pub fn rnnt(&self) -> (&RnntHead, &RnntRuntime) {
        self.as_rnnt().expect("head variant is CTC, expected RN-T")
    }
}

impl GigaAm {
    /// Load from a HuggingFace Hub repository (`main` revision).
    pub fn from_hub(model_id: &str) -> Result<Self> {
        Self::from_hub_with_revision(model_id, "main")
    }

    /// Load from a HuggingFace Hub repository at a specific branch/revision.
    /// Auto-detects head type from `config.transducer.is_some()`; fetches
    /// `tokenizer.model` only when the config asks for RN-T.
    pub fn from_hub_with_revision(model_id: &str, revision: &str) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
        let config_path = repo.get("config.json").context(HubSnafu)?;
        let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
        let config = GigaAmConfig::from_json(&config_path)?;
        // SentencePiece-RN-T variants (e.g. `v3_e2e_rnnt`) ship the tokenizer
        // as `tokenizer.model`. CTC variants don't have one; skip the fetch.
        let tokenizer_path = if config.transducer.is_some() { repo.get("tokenizer.model").ok() } else { None };
        Self::from_safetensors(&weights_path, tokenizer_path.as_deref(), config)
    }

    /// Load from a directory containing `config.json` + `model.safetensors`
    /// (and optionally `tokenizer.model` for RN-T configs).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let config_path = dir.join("config.json");
        let weights_path = dir.join("model.safetensors");
        let config = GigaAmConfig::from_json(&config_path)?;
        let tokenizer_path = dir.join("tokenizer.model");
        let tokenizer_path =
            if config.transducer.is_some() && tokenizer_path.exists() { Some(tokenizer_path) } else { None };
        Self::from_safetensors(&weights_path, tokenizer_path.as_deref(), config)
    }

    /// Load weights + (optional) SentencePiece tokenizer and assemble the
    /// model. `tokenizer` is ignored for CTC configs.
    pub fn from_safetensors(weights: &Path, tokenizer: Option<&Path>, config: GigaAmConfig) -> Result<Self> {
        let sd = state::load_safetensors(weights).context(StateSnafu)?;
        let vocab_override = tokenizer.map(load_sentencepiece_vocab).transpose()?;
        Self::from_state_dict(&sd, config, vocab_override)
    }

    /// Build from a pre-loaded state dict. `vocab_override` (RN-T only) wins
    /// over `config.transducer.vocabulary` if `Some`.
    ///
    /// Auto-detects PyTorch key format (`encoder.` / `model.encoder.` /
    /// `head.decoder.` / `head.joint.` prefixes) and remaps to morok layout
    /// before loading.
    pub fn from_state_dict(sd: &StateDict, config: GigaAmConfig, vocab_override: Option<Vec<String>>) -> Result<Self> {
        let is_pytorch = sd.keys().any(|k| {
            k.starts_with("encoder.")
                || k.starts_with("model.encoder.")
                || k.starts_with("head.decoder.")
                || k.starts_with("head.joint.")
        });
        let sd_owned = if is_pytorch { remap::remap_pytorch(sd.clone(), &config)? } else { sd.clone() };
        let sd = &sd_owned;

        let encoder = build_encoder_from_sd(sd, &config)?;

        let head = match &config.transducer {
            None => {
                let mut h = CTCHead::empty(&config);
                h.load_state_dict(sd, "head").context(StateSnafu)?;
                Head::Ctc(h)
            }
            Some(tr) => {
                let vocabulary = vocab_override.unwrap_or_else(|| tr.vocabulary.clone());
                if vocabulary.len() + 1 != tr.num_classes {
                    return Err(Error::DecoderConfig {
                        message: format!(
                            "RN-T vocabulary length + 1 ({}) != num_classes ({}); \
                             convention is one blank token at the end",
                            vocabulary.len() + 1,
                            tr.num_classes
                        ),
                    });
                }
                let mut h = RnntHead::empty(
                    config.d_model,
                    tr.pred_hidden,
                    tr.pred_rnn_layers,
                    tr.joint_hidden,
                    tr.num_classes,
                );
                h.load_state_dict(sd, "head").context(StateSnafu)?;
                h.predictor.prepare_for_inference()?;
                h.joint.cast_to_f32()?;
                Head::Rnnt {
                    head: h,
                    runtime: RnntRuntime {
                        vocabulary,
                        max_symbols_per_step: tr.max_symbols_per_step,
                        sentencepiece: tr.sentencepiece,
                    },
                }
            }
        };

        Ok(Self { config, encoder, head })
    }

    /// Build a model with zero-initialized weights from `config` alone.
    /// Head variant follows `config.transducer.is_some()`.
    pub fn with_random_weights(config: GigaAmConfig) -> Self {
        let encoder = Encoder::with_random_weights(&config);
        let head = match &config.transducer {
            None => Head::Ctc(CTCHead::empty(&config)),
            Some(tr) => Head::Rnnt {
                head: RnntHead::empty(
                    config.d_model,
                    tr.pred_hidden,
                    tr.pred_rnn_layers,
                    tr.joint_hidden,
                    tr.num_classes,
                ),
                runtime: RnntRuntime {
                    vocabulary: tr.vocabulary.clone(),
                    max_symbols_per_step: tr.max_symbols_per_step,
                    sentencepiece: tr.sentencepiece,
                },
            },
        };
        Self { config, encoder, head }
    }

    /// dtype the encoder operates in (read from the loaded weights).
    pub fn input_dtype(&self) -> DType {
        self.encoder.input_dtype()
    }

    /// Encoder-only single-batch forward: mel `[1, n_mels, T]` → encoded
    /// `[1, d_model, T_sub]`.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward(mel)
    }

    /// Batched encoder forward with dynamic batch and mel-frame length.
    /// Input: `mel` `[B, n_mels, T_mel]`, `lengths` `[B]`. Output:
    /// `[B, d_model, T_sub]`.
    pub fn encode_batch(
        &self,
        mel: &Tensor,
        lengths: &Tensor,
        batch: &BoundVariable,
        mel_len: &BoundVariable,
    ) -> Result<Tensor> {
        self.encoder.forward_batch(mel, lengths, batch, mel_len)
    }

    pub fn subsampling_output_length(&self, mel_frames: usize) -> usize {
        self.encoder.subsampling_output_length(mel_frames)
    }

    /// Full CTC pipeline: waveform → mel → encoder → head log-probs. Panics
    /// if the model has an RN-T head; the RN-T path runs through the search
    /// loop, not a single forward call.
    pub fn forward_ctc(&self, waveform: &[f32], mel_tensor: &mut Tensor) -> Result<Tensor> {
        {
            let mut view = mel_tensor.array_view_mut::<f32>().context(TensorSnafu)?;
            self.encoder.mel.forward_into(waveform, &mut view);
        }
        let encoded = self.encode(mel_tensor)?;
        self.head.ctc().forward(&encoded)
    }
}
