//! Correctness tests for the batched decode path.
//!
//! The heavy tests (marked `#[ignore]`) load real `whisper-tiny` weights and
//! verify that the batched path produces the same transcripts as the legacy
//! single-stream path. Run with `cargo test -- --ignored`.

use svod_arch::pipelines::audio::Transcriber;

use crate::whisper::{DecodeOptions, ModelDimensions, WhisperSize, WhisperTokenizer, WhisperTranscriber};

/// Loads whisper-tiny + tokenizer from HuggingFace Hub and builds a
/// transcriber with the same options soroka uses.
fn tiny_transcriber() -> WhisperTranscriber {
    let repo = "openai/whisper-tiny";
    let dims = ModelDimensions::for_size(WhisperSize::Tiny);
    let model = crate::whisper::Whisper::from_hub(repo, "main", dims).unwrap();
    let multilingual = model.is_multilingual();
    let num_languages = model.dims.num_languages();
    let tokenizer = WhisperTokenizer::from_hub(multilingual, num_languages).unwrap();
    // Greedy + no fallback: matches the batched path's decode strategy so the
    // two are directly comparable. The batched path forces these internally
    // (temperature_inc=0, beam_size=None); the legacy path honors whatever the
    // transcriber was built with, so set them here too.
    let options = DecodeOptions {
        task: "transcribe".to_string(),
        language: None,
        without_timestamps: true,
        beam_size: None,
        temperature_inc: 0.0,
        ..DecodeOptions::default()
    };
    WhisperTranscriber::new(model, tokenizer, options, WhisperSize::Tiny, 480_000).unwrap()
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

/// `transcribe_windows_batched` must produce the same text as the legacy
/// `transcribe_windows` for the same windows. Both paths run the same encoder
/// and prefill; the difference is the step loop (batched vs serial). This is
/// the end-to-end correctness guard for the continuous-batching primitive.
#[test]
#[ignore = "heavy: real whisper-tiny weights + JIT compile"]
fn batched_matches_legacy_transcribe() {
    let mut tx = tiny_transcriber();
    tx.set_language(Some("en".to_string())); // pin language so detection doesn't diverge

    let windows = fake_windows();
    let refs: Vec<&[f32]> = windows.iter().map(|w| w.as_slice()).collect();

    // Legacy single-stream path.
    let (legacy, _) = tx.transcribe_windows(&refs, false).expect("legacy transcribe");

    // Batched path.
    let batched = tx.transcribe_windows_batched(&refs).expect("batched transcribe");

    assert_eq!(legacy.len(), batched.len(), "transcript count mismatch");
    for (i, (l, b)) in legacy.iter().zip(batched.iter()).enumerate() {
        assert_eq!(l.text, b.text, "window {i}: text differs between legacy and batched");
    }
}
