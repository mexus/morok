//! One-call hub loaders for ModernBERT: fetches `config.json`,
//! `model.safetensors`, and `tokenizer.json`, then assembles the tokenizer +
//! model pair ready for a pipeline.

use snafu::ResultExt;
use svod_arch::pipelines::text::HfTokenizer;
use svod_dtype::DType;

use super::classifier::{ModernBertClassificationModel, ModernBertClassifier};
use super::config::ModernBertConfig;
use super::embedder::ModernBertEmbedder;
use super::error::{
    ClassifierSnafu, EmbedderSnafu, HubSnafu, Result, StateSnafu, TokenClassifierSnafu, TokenizerSnafu,
};
use super::model::ModernBert;
use super::token_classifier::{ModernBertTokenClassificationModel, ModernBertTokenClassifier};

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository (default revision `"main"`) and assemble the
/// `(HfTokenizer, ModernBertEmbedder)` pair ready for an
/// [`EmbeddingsPipeline`](svod_arch::pipelines::text::EmbeddingsPipeline).
///
/// `max_seq` is derived from the checkpoint's `max_position_embeddings`;
/// `max_batch` is caller-chosen (not in `config.json`). `dtype` selects the
/// compute precision (bf16 for GPU, f32 for CPU parity).
///
/// See [`from_hub_with_revision`] for the per-revision form.
pub fn from_hub(model_id: &str, max_batch: usize, dtype: DType) -> Result<(HfTokenizer, ModernBertEmbedder)> {
    from_hub_with_revision(model_id, "main", max_batch, dtype)
}

/// Per-revision form of [`from_hub`]: fetches the same three artifacts from a
/// pinned revision and assembles the pair. Config is seeded from
/// [`ModernBertConfig::default`] with the caller's `dtype` + `max_batch`, then spliced
/// with the structural fields from the downloaded `config.json` (see
/// [`ModernBert::from_hub_with_revision`]).
pub fn from_hub_with_revision(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(HfTokenizer, ModernBertEmbedder)> {
    let mut config = ModernBertConfig { dtype, max_batch_size: max_batch, ..Default::default() };
    let model = ModernBert::from_hub_with_revision(model_id, revision, &mut config)?;
    let max_seq = model.config.max_position_embeddings;

    let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
    let repo =
        api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));
    let tok_path = repo.get("tokenizer.json").context(HubSnafu)?;
    let tokenizer = HfTokenizer::from_path(&tok_path, max_seq).context(TokenizerSnafu)?;

    let embedder = ModernBertEmbedder::new(model, max_batch, max_seq).context(EmbedderSnafu)?;
    Ok((tokenizer, embedder))
}

// ── classifier ─────────────────────────────────────────────────────────────

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository (default revision `"main"`) and assemble the
/// `(HfTokenizer, ModernBertClassifier)` pair ready for a
/// [`ClassifyPipeline`](svod_arch::pipelines::text::ClassifyPipeline).
///
/// `max_seq` is derived from the checkpoint's `max_position_embeddings`;
/// `max_batch` is caller-chosen. `dtype` selects the compute precision.
pub fn from_hub_classifier(
    model_id: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(HfTokenizer, ModernBertClassifier)> {
    from_hub_classifier_with_revision(model_id, "main", max_batch, dtype)
}

/// Per-revision form of [`from_hub_classifier`].
pub fn from_hub_classifier_with_revision(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(HfTokenizer, ModernBertClassifier)> {
    let mut config = ModernBertConfig { dtype, max_batch_size: max_batch, ..Default::default() };

    let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
    let repo =
        api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));

    // Parse config.json and splice ALL fields (backbone + classifier) into the
    // caller-seeded config. dtype / max_batch_size stay caller-chosen.
    let cfg_path = repo.get("config.json").context(HubSnafu)?;
    let parsed = ModernBertConfig::from_json(&cfg_path)?;
    config.apply_checkpoint(&parsed);

    let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
    let sd = crate::state::load_safetensors(&weights_path).context(StateSnafu)?;

    let model = ModernBertClassificationModel::from_state_dict(&sd, &config)?;
    let max_seq = config.max_position_embeddings;

    let tok_path = repo.get("tokenizer.json").context(HubSnafu)?;
    let tokenizer = HfTokenizer::from_path(&tok_path, max_seq).context(TokenizerSnafu)?;

    let classifier = ModernBertClassifier::new(model, max_batch, max_seq).context(ClassifierSnafu)?;
    Ok((tokenizer, classifier))
}

// ── token classification ───────────────────────────────────────────────────

/// Download `config.json` + `model.safetensors` + `tokenizer.json` from a
/// HuggingFace Hub repository (default revision `"main"`) and assemble the
/// `(HfTokenizer, ModernBertTokenClassifier)` pair ready for a
/// [`RecognizePipeline`](svod_arch::pipelines::text::RecognizePipeline).
///
/// `max_seq` is derived from the checkpoint's `max_position_embeddings`;
/// `max_batch` is caller-chosen. `dtype` selects the compute precision.
pub fn from_hub_token_classification(
    model_id: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(HfTokenizer, ModernBertTokenClassifier)> {
    from_hub_token_classification_with_revision(model_id, "main", max_batch, dtype)
}

/// Per-revision form of [`from_hub_token_classification`]. Config is seeded from
/// [`ModernBertConfig::default`] with the caller's `dtype` + `max_batch`, then spliced
/// with the structural fields from the downloaded `config.json` (via
/// [`ModernBertConfig::apply_checkpoint`]). `classifier_pooling` is parsed but
/// unused — the token head never pools.
pub fn from_hub_token_classification_with_revision(
    model_id: &str,
    revision: &str,
    max_batch: usize,
    dtype: DType,
) -> Result<(HfTokenizer, ModernBertTokenClassifier)> {
    let mut config = ModernBertConfig { dtype, max_batch_size: max_batch, ..Default::default() };

    let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
    let repo =
        api.repo(hf_hub::Repo::with_revision(model_id.to_string(), hf_hub::RepoType::Model, revision.to_string()));

    let cfg_path = repo.get("config.json").context(HubSnafu)?;
    let parsed = ModernBertConfig::from_json(&cfg_path)?;
    config.apply_checkpoint(&parsed);

    let weights_path = repo.get("model.safetensors").context(HubSnafu)?;
    let sd = crate::state::load_safetensors(&weights_path).context(StateSnafu)?;

    let model = ModernBertTokenClassificationModel::from_state_dict(&sd, &config)?;
    let max_seq = config.max_position_embeddings;

    let tok_path = repo.get("tokenizer.json").context(HubSnafu)?;
    let tokenizer = HfTokenizer::from_path(&tok_path, max_seq).context(TokenizerSnafu)?;

    let classifier = ModernBertTokenClassifier::new(model, max_batch, max_seq).context(TokenClassifierSnafu)?;
    Ok((tokenizer, classifier))
}
