//! Top-level GigaAM RNN-T model: shared encoder + transducer head + vocab.

use std::path::Path;

use morok_dtype::DType;
use snafu::ResultExt;

use crate::state::{self, HasStateDict, StateDict};

use super::head::RnntHead;
use super::tokenizer::load_sentencepiece_vocab;
use crate::gigaam::encoder::{Encoder, build_encoder_from_sd};
use crate::gigaam::error::{Error, HubSnafu, StateSnafu};
use crate::gigaam::{GigaAmConfig, Result, remap};

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
        let transducer = config.transducer.as_ref().ok_or_else(|| Error::DecoderConfig {
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
            return Err(Error::DecoderConfig {
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
        let sd_owned = if is_pytorch { remap::remap_pytorch(sd.clone(), &config)? } else { sd.clone() };
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

        let encoder = Encoder::with_random_weights(&config);
        let head = RnntHead::empty(config.d_model, pred_hidden, pred_rnn_layers, joint_hidden, num_classes);
        Self { config, encoder, head, vocabulary, max_symbols_per_step, sentencepiece }
    }

    /// dtype the encoder + heads operate in (read from the loaded weights).
    pub fn input_dtype(&self) -> DType {
        self.encoder.input_dtype()
    }
}
