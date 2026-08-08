//! Whisper inference demo: transcribe a WAV file using Whisper ASR.
//!
//! Loads a WAV and runs it through an [`Asr`]: a [`FixedLengthSplitter`] feeds
//! a [`WhisperAlignedTranscriber`]. Each 30-second window is mel → encoder JIT →
//! cached beam decoding with temperature fallback, followed by DTW alignment.
//!
//! Usage:
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav --profile
//!   cargo run -p svod-model --release --example whisper_infer -- audio.wav --size base
//!
//! Options:
//!   --size       Model size: tiny, base, small, medium, large-v3, turbo (default: tiny)
//!   --language   Language code: en, fr, de, ..., or auto (default: auto)
//!   --task       transcribe or translate (default: transcribe)
//!   --profile    Print per-stage GPU profile
//!   --timestamps Print word timestamps
//!   --repo       HF Hub repo (default: openai/whisper-{size})

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use svod_arch::pipelines::audio::{Asr, FixedLengthSplitter, RunOptions};
use svod_model::whisper::{
    CHUNK_LENGTH, DecodeOptions, SAMPLE_RATE, Whisper, WhisperAlignedTranscriber, WhisperSize, WhisperTask,
    WhisperTokenizer,
};

#[derive(Parser, Debug)]
#[command(about = "Whisper ASR transcription demo", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono; ints or floats).
    wav: PathBuf,

    /// Model size name.
    #[arg(long, default_value = "tiny")]
    size: String,

    /// HF Hub repo override (default: openai/whisper-{size}).
    #[arg(long)]
    repo: Option<String>,

    /// Spoken language code, or "auto" to detect.
    #[arg(long, default_value = "auto")]
    language: String,

    /// Task: transcribe or translate.
    #[arg(long, default_value = "transcribe")]
    task: WhisperTask,

    /// Print word timestamps.
    #[arg(long)]
    timestamps: bool,

    /// Collect and print per-stage profile.
    #[arg(long)]
    profile: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let t_total = Instant::now();
    let args = Args::parse();

    let size = WhisperSize::from_name(&args.size)
        .ok_or_else(|| format!("unknown size {:?}; try: tiny, base, small, medium, large-v3, turbo", args.size))?;

    println!("Loading audio: {}", args.wav.display());
    let (waveform, sample_rate) = load_wav(&args.wav)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, sample_rate);

    if sample_rate != SAMPLE_RATE as u32 {
        return Err(format!("WAV is {sample_rate} Hz; Whisper expects {} Hz", SAMPLE_RATE).into());
    }

    let repo = args.repo.clone().unwrap_or_else(|| format!("openai/whisper-{}", size.name()));
    println!("\nLoading Whisper from {repo} ...");
    let model = Whisper::from_hub(&repo, "main", svod_model::whisper::ModelDimensions::for_size(size))?;

    let multilingual = model.is_multilingual();
    let num_languages = model.dims.num_languages();
    let tokenizer = WhisperTokenizer::from_hub(multilingual, num_languages)?;

    let options = DecodeOptions {
        task: args.task,
        language: if args.language == "auto" { None } else { Some(args.language.clone()) },
        beam_size: Some(5), // beam search — uses KV-cached fast path
        ..Default::default()
    };

    // Fixed-length 30s windows (no VAD dependency)
    let window_samples = CHUNK_LENGTH * SAMPLE_RATE;
    let splitter = FixedLengthSplitter::new(window_samples, SAMPLE_RATE);

    let transcriber = WhisperAlignedTranscriber::new(model, tokenizer, options, size, window_samples)?;

    let mut asr = Asr::new(splitter, transcriber);

    println!("Transcribing...");
    let t_transcribe = Instant::now();
    let result =
        asr.transcribe(&waveform, RunOptions { words: args.timestamps, profile: args.profile, ..Default::default() })?;
    let dt_transcribe = t_transcribe.elapsed();

    if args.timestamps {
        for chunk in &result.chunks {
            for word in chunk.words.as_deref().unwrap_or_default() {
                println!(
                    "  [{:>6.2}s - {:>6.2}s] {}",
                    chunk.start_sec + word.start,
                    chunk.start_sec + word.end,
                    word.text.trim(),
                );
            }
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
        "\nTotal: {:.2}s; transcribe: {:.2}s; loop RTF: {:.4}x",
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
