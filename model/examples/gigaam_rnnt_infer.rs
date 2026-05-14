//! GigaAM RNN-T (transducer) inference demo.
//!
//! Same pipeline shape as `gigaam_infer.rs` — loads WAV, runs a
//! [`Transcriber`] with an explicit [`SileroVadSplitter`] — but defaults to
//! an RN-T revision (`e2e_rnnt`) so the `Head::Rnnt` path drives the
//! predictor + joint JITs through the per-step `JointStep` backend.
//! SentencePiece `▁ → space` post-processing happens inside the transcriber.
//!
//! Usage:
//!   cargo run -p morok-model --release --example gigaam_rnnt_infer -- audio.wav
//!
//! Env knobs (all optional):
//!   MOROK_AMX=1                 Enable AMX renderer (Apple Silicon).
//!   MOROK_TIMESTAMPS=1          Emit per-word `[start - end] word` lines.
//!   MOROK_RNNT_REPO=<repo>      HF Hub repo (default `vpermilp/GigaAM-v3`).
//!   MOROK_RNNT_REVISION=<rev>   HF Hub revision (default `e2e_rnnt`).
//!   MOROK_MAX_SCORES_MIB=N      SDPA scores buffer budget (default 256).
//!   MOROK_VAD_THRESHOLD=f       Silero VAD threshold (default 0.5).

use std::env;
use std::time::Instant;

use morok_model::gigaam::{GigaAm, TranscribeOpts, Transcriber};
use morok_model::silero_vad::SileroVadSplitter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let t_total = Instant::now();
    let wav_path = env::args().nth(1).ok_or("usage: gigaam_rnnt_infer <audio.wav>")?;
    let repo = env::var("MOROK_RNNT_REPO").unwrap_or_else(|_| "vpermilp/GigaAM-v3".to_string());
    let revision = env::var("MOROK_RNNT_REVISION").unwrap_or_else(|_| "e2e_rnnt".to_string());
    let opts = TranscribeOpts::from_env();

    println!("Loading audio: {wav_path}");
    let (waveform, sample_rate) = load_wav(&wav_path)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, sample_rate);

    println!("\nLoading GigaAM RNN-T from {repo} ({revision})...");
    let model = GigaAm::from_hub_with_revision(&repo, &revision)?;
    if model.head.as_rnnt().is_none() {
        return Err(format!(
            "{repo}@{revision} has a CTC head, not RN-T. Set MOROK_RNNT_REVISION to an RN-T revision."
        )
        .into());
    }
    let splitter = SileroVadSplitter::from_hub()?;
    let mut transcriber = Transcriber::new(model, splitter, opts.clone())?;

    println!("Transcribing...");
    let t_transcribe = Instant::now();
    let result = transcriber.transcribe(&waveform, sample_rate)?;
    let dt_transcribe = t_transcribe.elapsed();

    if opts.word_timestamps {
        for word in result.words() {
            println!("  [{:>6.2} - {:>6.2}] {}", word.start, word.end, word.text);
        }
    } else {
        for chunk in &result.chunks {
            if !chunk.text.is_empty() {
                println!("  [{:>6.1}s] {}", chunk.start_sec, chunk.text);
            }
        }
    }

    println!("\n--- Transcript ---\n{}", result.text);
    println!(
        "\nTotal: {:.2}s; transcribe: {:.2}s; loop RTF: {:.2}x",
        t_total.elapsed().as_secs_f32(),
        dt_transcribe.as_secs_f32(),
        if duration_s > 0.0 { dt_transcribe.as_secs_f32() / duration_s } else { 0.0 },
    );
    Ok(())
}

fn load_wav(path: &str) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            reader.samples::<i16>().map(|s| s.map(|v| v as f32 / 32768.0)).collect::<Result<_, _>>()?
        }
    };
    Ok((samples, spec.sample_rate))
}
