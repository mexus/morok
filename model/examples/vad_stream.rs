//! Streaming FireRedVAD demo: feed a WAV in fixed-size blocks (a simulated
//! microphone), print speech start/end events as they fire, then flush for
//! the final segment timestamps and throughput.
//!
//! The model weights load from HF Hub (`Stream-VAD` conversion); the conv
//! caches recycle on-device between chunks, so steady-state cost is one small
//! JIT dispatch per `--chunk-frames` of audio.
//!
//! Usage:
//!   cargo run -p svod-model --release --example vad_stream -- audio.wav
//!   cargo run -p svod-model --release --example vad_stream -- audio.wav --block-ms 20
//!
//! Env knobs:
//!   SVOD_VAD_THRESHOLD=f       Speech threshold (default 0.5).

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;

use svod_model::firered_vad::{FireRedVadStream, FireRedVadStreamer, VadEvent};

#[derive(Parser, Debug)]
#[command(about = "Streaming FireRedVAD demo (simulated live feed)", long_about = None)]
struct Args {
    /// Input WAV (16 kHz mono; ints or floats).
    wav: PathBuf,

    /// Samples per push, as milliseconds of audio (simulated mic block).
    #[arg(long, default_value_t = 100)]
    block_ms: usize,

    /// Frames per JIT dispatch (fixed graph shape; 16 -> 160 ms latency).
    #[arg(long, default_value_t = 16)]
    chunk_frames: usize,

    /// Local converted weights; defaults to the HF Hub download.
    #[arg(long)]
    weights: Option<PathBuf>,
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

    let t_init = Instant::now();
    let model = match &args.weights {
        Some(path) => FireRedVadStream::from_safetensors(path)?,
        None => FireRedVadStream::from_hub()?,
    };
    let mut streamer = FireRedVadStreamer::builder().model(model).chunk_frames(args.chunk_frames).build()?;
    println!("init (hub load + JIT prepare): {:.2}s", t_init.elapsed().as_secs_f32());

    let block = args.block_ms * 16; // samples per push at 16 kHz
    let t_stream = Instant::now();
    for chunk in waveform.chunks(block.max(1)) {
        for event in streamer.push(chunk)? {
            match event {
                VadEvent::SpeechStart { frame } => {
                    println!("  [{:>7.2}s] speech start", (frame - 1) as f32 / 100.0);
                }
                VadEvent::SpeechEnd { start_frame, end_frame } => {
                    let (s, e) = ((start_frame - 1) as f32 / 100.0, (end_frame - 1) as f32 / 100.0);
                    println!("  [{e:>7.2}s] speech end   ({s:.2}s - {e:.2}s, {:.2}s)", e - s);
                }
            }
        }
    }
    let flush = streamer.flush()?;
    for event in &flush.events {
        if let VadEvent::SpeechEnd { start_frame, end_frame } = event {
            let (s, e) = ((start_frame - 1) as f32 / 100.0, (end_frame - 1) as f32 / 100.0);
            println!("  [{e:>7.2}s] speech end   ({s:.2}s - {e:.2}s, {:.2}s) [flush]", e - s);
        }
    }
    let wall = t_stream.elapsed().as_secs_f64();

    println!("\nsegments ({}):", flush.timestamps.len());
    for (i, (s, e)) in flush.timestamps.iter().enumerate() {
        println!("  {i}: {s:.2}s - {e:.2}s ({:.2}s)", e - s);
    }
    println!(
        "\nstreamed {duration_s:.1}s in {:.1} ms ({} pushes of {} ms); RTF {:.5}",
        wall * 1e3,
        waveform.len().div_ceil(block.max(1)),
        args.block_ms,
        wall / duration_s as f64,
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
