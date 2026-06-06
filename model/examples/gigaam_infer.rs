//! GigaAM inference demo (CTC + RN-T).
//!
//! Loads a WAV, hands it to a [`Transcriber`] over a [`GigaAm`] with an
//! explicit [`SileroVadSplitter`], and prints the transcript. The head is
//! dispatched from the loaded weights: a CTC revision drives the fused
//! encoder+head JIT, an RN-T revision the encoder + per-step predictor/joint
//! backend (SentencePiece `▁ → space` post-processing inside the transcriber).
//!
//! Substitute `FixedLengthSplitter::new()` for the VAD splitter to skip the
//! Silero hub download — useful for tests, short utterances, or pipelines
//! that already segmented the input.
//!
//! Usage:
//!   cargo run -p svod-model --release --example gigaam_infer -- audio.wav
//!   cargo run -p svod-model --release --example gigaam_infer -- audio.wav --rnnt --profile
//!
//! Env knobs (all optional):
//!   SVOD_AMX=1                 Enable AMX renderer (Apple Silicon).
//!   SVOD_VAD_THRESHOLD=f       Silero VAD threshold (default 0.5).

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use svod_model::gigaam::{GigaAm, TranscribeOpts, Transcriber};
use svod_model::silero_vad::SileroVadSplitter;

#[derive(Parser, Debug)]
#[command(about = "GigaAM transcription demo (CTC + RN-T)", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono; ints or floats).
    wav: PathBuf,

    /// HF Hub repo with the model weights.
    #[arg(long, default_value = "vpermilp/GigaAM-v3")]
    repo: String,

    /// HF Hub revision; the head (CTC vs RN-T) follows the weights.
    /// Defaults to `ctc`, or `e2e_rnnt` under `--rnnt`.
    #[arg(long)]
    revision: Option<String>,

    /// Shorthand for the default RN-T revision.
    #[arg(long)]
    rnnt: bool,

    /// Emit per-word `[start - end] word` lines.
    #[arg(long)]
    timestamps: bool,

    /// Promote greedy CTC to beam search (no-op for RN-T).
    #[arg(long)]
    beam_decode: bool,

    /// Collect and print the typed per-stage GPU profile.
    #[arg(long)]
    profile: bool,

    /// SDPA scores buffer budget (MiB).
    #[arg(long, default_value_t = 256)]
    max_scores_mib: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let t_total = Instant::now();
    let args = Args::parse();
    let revision = args.revision.clone().unwrap_or_else(|| if args.rnnt { "e2e_rnnt" } else { "ctc" }.to_string());
    let opts = TranscribeOpts::builder()
        .word_timestamps(args.timestamps)
        .beam_decode(args.beam_decode)
        .profile(args.profile)
        .max_scores_mib(args.max_scores_mib)
        .build();

    println!("Loading audio: {}", args.wav.display());
    let (waveform, sample_rate) = load_wav(&args.wav)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, sample_rate);

    println!("\nLoading GigaAM from {} ({revision})...", args.repo);
    let model = GigaAm::from_hub_with_revision(&args.repo, &revision)?;
    if args.rnnt && model.head.as_rnnt().is_none() {
        return Err(format!("{}@{revision} has a CTC head, not RN-T.", args.repo).into());
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
    if let Some(profile) = &result.profile {
        println!("\n--- Profile ---\n{profile}");
    }

    println!("\n--- Transcript ---\n{}", result.text);
    println!(
        "\nTotal: {:.2}s; transcribe: {:.2}s; loop RTF: {:.3}x",
        t_total.elapsed().as_secs_f32(),
        dt_transcribe.as_secs_f32(),
        if duration_s > 0.0 { dt_transcribe.as_secs_f32() / duration_s } else { 0.0 },
    );
    Ok(())
}

fn load_wav(path: &PathBuf) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
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
