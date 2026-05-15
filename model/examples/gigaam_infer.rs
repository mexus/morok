//! GigaAM CTC inference demo.
//!
//! Loads a WAV, hands it to a [`Transcriber`] over a CTC-head [`GigaAm`] with
//! an explicit [`SileroVadSplitter`], and prints the transcript. The pipeline
//! (mel features, splitter-driven chunking, batched encoder JIT, head decode)
//! lives inside `Transcriber`; this example is just a thin CLI on top.
//!
//! Substitute `FixedLengthSplitter::new()` for the VAD splitter to skip the
//! Silero hub download — useful for tests, short utterances, or pipelines
//! that already segmented the input.
//!
//! Usage:
//!   cargo run -p morok-model --release --example gigaam_infer -- audio.wav
//!
//! Env knobs (all optional):
//!   MOROK_AMX=1                 Enable AMX renderer (Apple Silicon).
//!   MOROK_BEAM_DECODE=1         Promote greedy CTC to beam search.
//!   MOROK_TIMESTAMPS=1          Emit per-word `[start - end] word` lines.
//!   MOROK_GIGAAM_REVISION=name  HF Hub revision (default `ctc`).
//!   MOROK_MAX_SCORES_MIB=N      SDPA scores buffer budget (default 256).
//!   MOROK_VAD_THRESHOLD=f       Silero VAD threshold (default 0.5).
//!
//! See `gigaam_rnnt_infer.rs` for the RN-T variant (same pattern, RN-T-default
//! revision).

use std::env;
use std::time::Instant;

use morok_model::gigaam::{GigaAm, TranscribeOpts, Transcriber};
use morok_model::silero_vad::SileroVadSplitter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_total = Instant::now();
    let wav_path = env::args().nth(1).ok_or("usage: gigaam_infer <audio.wav>")?;
    let revision = env::var("MOROK_GIGAAM_REVISION").unwrap_or_else(|_| "ctc".to_string());
    let opts = TranscribeOpts::from_env();

    println!("Loading audio: {wav_path}");
    let (waveform, sample_rate) = load_wav(&wav_path)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, sample_rate);

    println!("\nLoading GigaAM ({revision})...");
    let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", &revision)?;
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
