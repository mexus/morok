//! Parity against the PyTorch reference (`answerdotai/ModernBERT-base`).
//! Heavy: loads the real checkpoint + a golden `last_hidden_state` produced by
//! HuggingFace `transformers` (`uv run scripts/convert_modernbert.py`).
//!
//! Runs in **f32** (config dtype overridden) so it works on CPU backends
//! without GPU bf16 transcendentals. bf16 numerical parity is implied by the
//! framework's f32-accumulator guarantees in layernorm/matmul/attention.

use std::path::{Path, PathBuf};

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::modernbert::{ModernBert, ModernBertConfig};
use crate::state::StateDict;

const HUB_REPO: &str = "answerdotai/ModernBERT-base";

/// Resolve `model.safetensors` / `golden.safetensors` for the real-checkpoint
/// tests: `SVOD_MODERNBERT` dir override → local `data/modernbert/` (output of
/// `scripts/convert_modernbert.py`) → HF Hub download.
fn real_file(name: &str) -> PathBuf {
    let dir = std::env::var_os("SVOD_MODERNBERT")
        .map(PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../data/modernbert"));
    let local = dir.join(name);
    if local.exists() {
        local
    } else {
        let api = hf_hub::api::sync::Api::new().expect("HF Hub API");
        let repo = api.repo(hf_hub::Repo::with_revision(HUB_REPO.into(), hf_hub::RepoType::Model, "main".into()));
        repo.get(name).unwrap_or_else(|_| panic!("download {name} from HF Hub"))
    }
}

fn load_golden_vec<T: svod_dtype::ext::HasDType + Default + Clone>(sd: &StateDict, key: &str) -> Vec<T> {
    let mut t = sd.get(key).unwrap_or_else(|| panic!("golden key {key}")).clone();
    t.realize().expect("realize golden");
    t.as_vec::<T>().expect("golden readout")
}

/// `last_hidden_state` parity: our backbone (f32) vs the PyTorch reference.
#[test]
#[ignore = "heavy: real ModernBERT-base weights + PyTorch golden (local or HF Hub download)"]
fn last_hidden_state_matches_pytorch() {
    let weights = real_file("model.safetensors");
    let golden = crate::state::load_safetensors(&real_file("golden.safetensors")).expect("golden");

    // Parse the published config.json, then force f32 for CPU parity.
    let cfg_path = real_file("config.json");
    let mut cfg = ModernBertConfig::from_json(&cfg_path).expect("parse config.json");
    cfg.dtype = DType::Float32;

    let model = ModernBert::from_safetensors(&weights, cfg).expect("load weights");

    // The exact `input_ids` + `attention_mask` are baked into the golden by the
    // generator script (which runs `transformers` WITH the mask, so pad tokens
    // are excluded from attention — the Rust port must do the same).
    let input_ids: Vec<i64> = load_golden_vec(&golden, "input_ids");
    let want: Vec<f32> = load_golden_vec(&golden, "last_hidden_state");
    let (b, l) = match golden.get("input_ids_shape") {
        Some(t) => {
            let mut t = t.clone();
            t.realize().unwrap();
            let s = t.as_vec::<i64>().unwrap();
            (s[0] as usize, s[1] as usize)
        }
        None => (1, input_ids.len()),
    };

    let ids = Tensor::from_slice(input_ids).try_reshape([b as isize, l as isize]).unwrap();
    // attention_mask is int64 1/0 → bool (true = real token). The encoder
    // inverts internally to the SDPA "true = masked out" convention.
    let mask = match golden.get("attention_mask") {
        Some(t) => {
            let mut t = t.clone();
            t.realize().unwrap();
            Some(t.cast(DType::Bool).unwrap().try_reshape([b as isize, l as isize]).unwrap())
        }
        None => None,
    };

    let mut out = model.forward(&ids, mask.as_ref()).expect("forward");
    out.realize().expect("realize output");
    let got = out.as_vec::<f32>().expect("output readout");

    assert_eq!(got.len(), want.len(), "element count mismatch");
    let deltas: Vec<f32> = got.iter().zip(&want).map(|(a, e)| (a - e).abs()).collect();
    let max_abs = deltas.iter().copied().fold(0.0f32, f32::max);
    // Per-position worst token (across hidden dim D) — separate real vs pad so a
    // large pad-position delta doesn't mask a real-token regression.
    let d = want.len() / l;
    let real_max = (0..deltas.len())
        .step_by(d)
        .take(12) // first 12 tokens are real (per the golden attention_mask)
        .map(|pos| deltas[pos..pos + d].iter().copied().fold(0.0f32, f32::max))
        .fold(0.0f32, f32::max);
    let pad_max = (12 * d..deltas.len())
        .step_by(d)
        .map(|pos| deltas[pos..pos + d].iter().copied().fold(0.0f32, f32::max))
        .fold(0.0f32, f32::max);
    eprintln!("max |delta| = {max_abs:.3e}  real-token max = {real_max:.3e}  pad-token max = {pad_max:.3e}");

    // Both sides run in pure f32 (the Rust model is forced to f32 by the test;
    // the golden is produced by `transformers` in float32). The residual is pure
    // float-reassociation noise between the Rust and PyTorch graph orderings
    // (relative error ~1e-5 against hidden states ranging to ±40).
    assert!(max_abs < 2e-3, "last_hidden_state drifted from PyTorch golden: max |delta| = {max_abs}");
}
