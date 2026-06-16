//! VAD benchmark: Silero vs FireRedVAD through the production [`Splitter`]
//! path (probabilities + chunking), with per-run wall time and RTF.
//!
//! Both models load from HF Hub. The first `split` after prepare is reported
//! separately as warmup (it absorbs lazy allocations); the following `--runs`
//! iterations are the steady-state measurement.
//!
//! Usage:
//!   cargo run -p svod-model --release --example vad_bench -- audio.wav
//!   cargo run -p svod-model --release --example vad_bench -- audio.wav --runs 5 --skip-silero
//!
//! Env knobs:
//!   RUST_LOG=svod_model=info   Per-stage breakdowns (Silero: feature/scan;
//!                              FireRedVAD: fbank/DFSMN).
//!   SVOD_VAD_THRESHOLD=f       Speech threshold for both splitters.

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use svod_model::audio::{AudioChunk, EncoderBounds, Splitter};
use svod_model::firered_vad::FireRedVadSplitter;
use svod_model::silero_vad::SileroVadSplitter;

#[derive(Parser, Debug)]
#[command(about = "Silero vs FireRedVAD timing benchmark", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono; ints or floats).
    wav: PathBuf,

    /// Timed iterations after the warmup run.
    #[arg(long, default_value_t = 3)]
    runs: usize,

    #[arg(long)]
    skip_silero: bool,

    #[arg(long)]
    skip_firered: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let (waveform, sample_rate) = load_wav(&args.wav)?;
    let duration_s = waveform.len() as f32 / sample_rate as f32;
    println!("Audio: {} ({} samples, {duration_s:.1}s @ {sample_rate} Hz)", args.wav.display(), waveform.len());
    if sample_rate != 16_000 {
        return Err(format!("expected 16 kHz input, got {sample_rate} Hz").into());
    }
    // GigaAM-like encoder bounds (10 ms hop, 4x subsampling, ~30 s capacity) —
    // the same shape Transcriber would hand the splitter.
    let bounds = EncoderBounds { sample_rate, hop_length: 160, subsampling_factor: 4, max_mel_frames: 3000 };

    if !args.skip_silero {
        println!("\n=== Silero VAD ===");
        let t_init = Instant::now();
        let splitter = SileroVadSplitter::from_hub()?;
        bench("silero", splitter, t_init, &waveform, &bounds, duration_s, args.runs)?;
    }
    if !args.skip_firered {
        println!("\n=== FireRedVAD ===");
        let t_init = Instant::now();
        let splitter = FireRedVadSplitter::from_hub()?;
        bench("firered", splitter, t_init, &waveform, &bounds, duration_s, args.runs)?;
    }
    Ok(())
}

fn bench<S: Splitter>(
    name: &str,
    mut splitter: S,
    t_init: Instant,
    waveform: &[f32],
    bounds: &EncoderBounds,
    duration_s: f32,
    runs: usize,
) -> Result<(), S::Error> {
    println!("init (hub load + JIT prepare): {:.2}s", t_init.elapsed().as_secs_f32());

    let t_warmup = Instant::now();
    let chunks = splitter.split(waveform, bounds)?;
    println!("warmup split: {:.1} ms", t_warmup.elapsed().as_secs_f64() * 1e3);

    let mut times_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let t = Instant::now();
        splitter.split(waveform, bounds)?;
        times_ms.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let min = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
    let avg = times_ms.iter().sum::<f64>() / times_ms.len().max(1) as f64;
    println!(
        "{name}: split min {min:.1} ms / avg {avg:.1} ms over {runs} runs; RTF {:.5}",
        (avg / 1e3) / duration_s as f64,
    );

    println!("chunks ({}):", chunks.len());
    for (i, AudioChunk { start_sample, end_sample, .. }) in chunks.iter().enumerate() {
        let (s, e) = (*start_sample as f32 / 16_000.0, *end_sample as f32 / 16_000.0);
        println!("  {i}: {s:.2}s - {e:.2}s ({:.2}s)", e - s);
    }
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
