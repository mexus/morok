//! Correctness tests for fixed-slot decode scheduling.
//!
//! The heavy tests (marked `#[ignore]`) load real `whisper-tiny` weights and
//! compare stable-slot refill with the serial greedy path. Run with `cargo test
//! -- --ignored`.

use svod_arch::pipelines::audio::Transcriber;

use crate::whisper::{
    DecodeOptions, ModelDimensions, WhisperAlignedTranscriber, WhisperPlan, WhisperSize, WhisperTask, WhisperTokenizer,
};

/// Loads whisper-tiny + tokenizer from HuggingFace Hub and builds a
/// transcriber with the same options soroka uses.
fn tiny_transcriber() -> WhisperAlignedTranscriber {
    let repo = "openai/whisper-tiny";
    let dims = ModelDimensions::for_size(WhisperSize::Tiny);
    let model = crate::whisper::Whisper::from_hub(repo, "main", dims).unwrap();
    let multilingual = model.is_multilingual();
    let num_languages = model.dims.num_languages();
    let tokenizer = WhisperTokenizer::from_hub(multilingual, num_languages).unwrap();
    // Greedy + no fallback makes the two scheduling strategies directly
    // comparable.
    let options = DecodeOptions {
        task: WhisperTask::Transcribe,
        language: None,
        beam_size: None,
        temperature_inc: 0.0,
        ..DecodeOptions::default()
    };
    let mut plan = WhisperPlan::for_model(&model.dims, WhisperSize::Tiny);
    plan.decoder_slots = 1; // force slot refill across the two test windows
    plan.alignment_batch = 1;
    WhisperAlignedTranscriber::new_with_plan(model, tokenizer, options, WhisperSize::Tiny, 480_000, plan).unwrap()
}

/// Two sine-wave "audio" windows (distinct frequencies) — enough to drive the
/// encoder/decoder without needing a real speech fixture. The point is
/// *deterministic equality* between paths, not transcript quality.
fn fake_windows() -> Vec<Vec<f32>> {
    let sr = 16_000usize;
    (0..2)
        .map(|wi| {
            let freq = 220.0 * 2_f32.powi(wi);
            (0..sr).map(|t| (freq * t as f32 * 2.0 * std::f32::consts::PI / sr as f32).sin() * 0.3).collect()
        })
        .collect()
}

/// Stable slot refill must produce the same text as serial greedy decoding.
#[test]
#[ignore = "heavy: real whisper-tiny weights + JIT compile"]
fn batched_greedy_matches_serial_greedy() {
    let mut tx = tiny_transcriber();
    tx.set_language(Some("en".to_string())); // pin language so detection doesn't diverge

    let windows = fake_windows();
    let refs: Vec<&[f32]> = windows.iter().map(|w| w.as_slice()).collect();

    let (serial, _) = tx.transcribe_windows(&refs, false).expect("serial transcribe");

    let batched = tx.transcribe_windows_batched_greedy(&refs).expect("batched greedy transcribe");

    assert_eq!(serial.len(), batched.len(), "transcript count mismatch");
    for (i, (serial, batched)) in serial.iter().zip(batched.iter()).enumerate() {
        assert_eq!(serial.text, batched.text, "window {i}: scheduling changed decoded text");
    }
}
