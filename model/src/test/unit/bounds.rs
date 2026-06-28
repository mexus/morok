//! Unit tests for [`EncoderBounds`](crate::audio::EncoderBounds) getters.
//!
//! Chunking now lives in the arch `pipelines::audio` layer (tested there); only
//! the model-config → sizing derivations are exercised here.

use crate::audio::EncoderBounds;

/// 16 kHz / hop=160 / subsampling=4 → align_to = 640 samples (40 ms),
/// max_mel_frames=2_080 → max_samples = (2080 - 8) * 160 = 331_520 samples
/// (~20.7 s).
fn bounds_realistic() -> EncoderBounds {
    EncoderBounds {
        sample_rate: 16_000,
        hop_length: 160,
        subsampling_factor: 4,
        max_mel_frames: 2_080,
        recommended_target_secs: None,
    }
}

#[test]
fn encoder_bounds_getters() {
    let b = bounds_realistic();
    assert_eq!(b.align_to_samples(), 640);
    assert_eq!(b.max_samples(), 331_520);
    let secs = b.encoder_capacity_secs();
    assert!((secs - 20.72).abs() < 0.01, "got {secs}");
}

#[test]
fn encoder_bounds_underflow_safe() {
    // max_mel_frames < 2 * subsampling_factor → saturating_sub clamps to 0.
    let b = EncoderBounds {
        sample_rate: 16_000,
        hop_length: 160,
        subsampling_factor: 8,
        max_mel_frames: 4,
        recommended_target_secs: None,
    };
    assert_eq!(b.max_samples(), 0);
    assert_eq!(b.encoder_capacity_secs(), 0.0);
}
