//! Translate a DiariZen `pytorch_model.bin` state-dict into the morok layout.
//!
//! DiariZen wraps the WavLM backbone under `self.wavlm_model = ...` in the
//! Python `Model.__init__` (`model_wavlm_conformer.py:58`). The published
//! checkpoint therefore has two sets of keys:
//!
//! - `wavlm_model.<...>`  — the WavLM backbone (`feature_extractor.*` /
//!   `encoder.*`); strip the prefix and feed to [`crate::wavlm::WavLm`].
//! - `<rest>`             — `weight_sum.weight`, `proj.{weight,bias}`,
//!   `lnorm.{weight,bias}`, `conformer.conformer_layer.<i>.*`,
//!   `classifier.{weight,bias}`. These stay verbatim.

use crate::state::{self, StateDict};

const WAVLM_PREFIX: &str = "wavlm_model.";

/// `(wavlm_sd, head_sd)` split. Pure key-routing; no BN folding here — apply
/// [`crate::blocks::remap::fold_batchnorm`] to the `head_sd` *separately* in
/// the production loader. Folding replaces `running_var` keys with `invstd`
/// keys (and computes the transform), so a raw PyTorch checkpoint must be
/// folded before `load_state_dict`; round-tripped state dicts already use
/// `invstd` keys and skip folding.
///
/// - `wavlm_sd` has the `wavlm_model.` prefix stripped on each key (so it can
///   be passed directly to `WavLm::load_state_dict(&sd, "")`).
/// - `head_sd` keeps all non-WavLM keys verbatim. Inert PyTorch buffers
///   (`*.num_batches_tracked`, `*.hard_concrete_for_*`) are dropped.
pub fn split_diarizen_state_dict(sd: StateDict) -> Result<(StateDict, StateDict), state::Error> {
    let mut wavlm = StateDict::new();
    let mut head = StateDict::new();
    for (k, v) in sd {
        if is_inert_key(&k) {
            continue;
        }
        if let Some(rest) = k.strip_prefix(WAVLM_PREFIX) {
            wavlm.insert(rest.to_string(), v);
        } else {
            head.insert(k, v);
        }
    }
    Ok((wavlm, head))
}

fn is_inert_key(key: &str) -> bool {
    key.ends_with("num_batches_tracked") || key.contains("hard_concrete_for_")
}
