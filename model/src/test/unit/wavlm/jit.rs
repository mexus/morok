use crate::jit::InputSpec;
use crate::wavlm::{WavLm, WavLmConfig, WavLmJit, wavlm_large_s80_md};

/// Construct a JIT wrapper and call `prepare` to validate the build closure
/// constructs a well-formed graph. Gated behind `#[ignore]` because compiling
/// even a tiny 2-layer transformer graph through the CPU backend takes ~60s.
/// Real coverage comes from the parity test (which uses eager `.realize()`)
/// and from full-stack DiariZen runs.
#[test]
#[ignore = "heavy: CPU JIT prepare of a 2-layer WavLM transformer is ~60s"]
fn wavlm_jit_prepares_on_tiny_config() {
    let cfg = tiny_cfg();
    let mut jit = WavLmJit::new(WavLm::empty(cfg.clone()));
    // 4096 samples → at least one feature-extractor output frame even at the
    // full s80-md-v2 stride. Tiny enough that the prepare graph build stays
    // fast (no kernel compile here — that's what `ignored` tests cover).
    jit.prepare(InputSpec::f32(&[cfg.max_batch_size, 4096])).expect("prepare");
}

fn tiny_cfg() -> WavLmConfig {
    let mut cfg = wavlm_large_s80_md();
    cfg.encoder_embed_dim = 32;
    cfg.encoder_head_dim = 8;
    cfg.encoder_num_layers = 2;
    cfg.encoder_use_attention = vec![true; 2];
    cfg.encoder_use_feed_forward = vec![true; 2];
    cfg.encoder_total_num_heads = vec![4; 2];
    cfg.encoder_remaining_heads = vec![vec![0, 1, 2, 3]; 2];
    cfg.encoder_ff_interm_features = vec![64; 2];
    cfg.max_batch_size = 1;
    cfg
}
