//! GigaAM RNN-T (transducer) inference example.
//!
//! Mirrors `gigaam_infer.rs` end-to-end but swaps the CTC head for the RNN-T
//! head: encoder is JIT'd and batched across VAD chunks; per-chunk decoding
//! is a B=1 greedy search driven by `arch::rnnt::RnntDecoder` against the
//! model's predictor + joint JITs.
//!
//! Usage:
//!   cargo run -p morok-model --release --example gigaam_rnnt_infer -- audio.wav
//!
//! Optional env vars:
//!   MOROK_RNNT_REPO=<repo>           HuggingFace repo id (default
//!                                    "vpermilp/GigaAM-v3")
//!   MOROK_RNNT_REVISION=<branch>     Repo revision (default "rnnt")
//!   MOROK_AMX=1                      Enable AMX renderer (Apple Silicon).
//!   MOROK_TIMESTAMPS=1               Emit per-word `[start - end] word`
//!                                    lines (seconds) instead of one
//!                                    `[start_sec] text` line per VAD chunk.
//!                                    Mirrors upstream GigaAM's
//!                                    `transcribe(..., word_timestamps=True)`.

use std::env;
use std::sync::Arc;
use std::time::Instant;

use morok_dtype::DType;
use morok_model::audio::MelSpectrogram;
use morok_model::gigaam::{GigaAmRnnt, GigaAmRnntEncoderJit, RnntStepBackend, SubsamplingMode};
use morok_tensor::{PrepareConfig, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    let t_total = Instant::now();

    let wav_path = env::args().nth(1).ok_or("usage: gigaam_rnnt_infer <audio.wav>")?;
    let amx_enabled = env::var("MOROK_AMX").as_deref() == Ok("1");
    let want_timestamps = env::var("MOROK_TIMESTAMPS").as_deref() == Ok("1");
    let repo = env::var("MOROK_RNNT_REPO").unwrap_or_else(|_| "vpermilp/GigaAM-v3".to_string());
    let revision = env::var("MOROK_RNNT_REVISION").unwrap_or_else(|_| "e2e_rnnt".to_string());

    println!("Loading audio: {wav_path}");
    let t_audio = Instant::now();
    let (waveform, wav_sample_rate) = load_wav(&wav_path)?;
    let dt_audio = t_audio.elapsed();
    let duration_s = waveform.len() as f32 / wav_sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, wav_sample_rate);

    println!("\nLoading GigaAM RNN-T from {repo} ({revision})...");
    let t_model = Instant::now();
    let model = GigaAmRnnt::from_hub_with_revision(&repo, &revision)?;
    let model = Arc::new(model);
    if wav_sample_rate as usize != model.config.sample_rate {
        return Err(format!(
            "WAV is {} Hz, model expects {} Hz (resample first)",
            wav_sample_rate, model.config.sample_rate
        )
        .into());
    }
    let dt_model = t_model.elapsed();

    let sample_rate = model.config.sample_rate;
    let n_mels = model.config.n_mels;
    let max_t_mel = model.config.max_mel_frames;
    let hop_length = model.config.hop_length;
    let subsampling_factor = model.config.subsampling_factor;
    let subs_kernel_size = match model.config.subsampling_mode {
        SubsamplingMode::Conv1d => model.config.subs_kernel_size,
        SubsamplingMode::Conv2d => 3,
    };
    let d_model = model.config.d_model;
    println!(
        "Loaded: {} layers, d_model={}, vocab_size={} (+blank), max_symbols/step={}, sentencepiece={}",
        model.config.n_layers,
        d_model,
        model.vocabulary.len(),
        model.max_symbols_per_step,
        model.sentencepiece,
    );

    // ─── Mel features ────────────────────────────────────────────────────
    let mel = MelSpectrogram::new(&morok_model::audio::MelConfig {
        sample_rate,
        n_fft: model.config.n_fft,
        hop_length,
        win_length: model.config.win_length,
        n_mels,
        center: model.config.mel_center,
    });
    let total_mel_frames = mel.num_frames(waveform.len());
    if total_mel_frames == 0 {
        println!("No frames produced from audio.");
        return Ok(());
    }
    let t_mel = Instant::now();
    let mut full_mel = Tensor::full(&[1, n_mels, total_mel_frames], 0.0f32, DType::Float32)?;
    full_mel.realize().unwrap();
    {
        let mut view = full_mel.array_view_mut::<f32>()?;
        mel.forward_into(&waveform, &mut view);
    }
    let full_mel_data = full_mel.as_vec::<f32>()?;
    let dt_mel = t_mel.elapsed();

    // ─── VAD chunking ────────────────────────────────────────────────────
    println!("\nLoading Silero VAD...");
    let t_vad_prepare = Instant::now();
    let vad_model = morok_model::silero_vad::SileroVad::from_hub()?;
    let mut vad = morok_model::silero_vad::VadInference::new(vad_model)?;
    let dt_vad_prepare = t_vad_prepare.elapsed();

    let t_vad = Instant::now();
    let probs = vad.probs(&waveform)?;
    let mel_headroom = 2 * subsampling_factor;
    let encoder_capacity_secs =
        (max_t_mel.saturating_sub(mel_headroom) as f32 * hop_length as f32) / sample_rate as f32;
    let default_opts = morok_arch::vad::ChunkerOpts::default();
    let chunker_opts = morok_arch::vad::ChunkerOpts {
        sample_rate: sample_rate as u32,
        samples_per_prob: morok_model::silero_vad::NUM_SAMPLES,
        max_duration: default_opts.max_duration.min(encoder_capacity_secs),
        strict_limit_duration: default_opts.strict_limit_duration.min(encoder_capacity_secs),
        align_to: hop_length * subsampling_factor,
        ..default_opts
    };
    let vad_chunks = morok_arch::vad::chunks_from_probs(&probs, &chunker_opts)?;
    let dt_vad = t_vad.elapsed();
    println!(
        "VAD: {} probs → {} chunks (max_chunk={:.1}s, strict={:.1}s, encoder_cap={:.1}s) in {}",
        probs.len(),
        vad_chunks.len(),
        chunker_opts.max_duration,
        chunker_opts.strict_limit_duration,
        encoder_capacity_secs,
        fmt_duration(dt_vad),
    );

    // (mel_start, mel_len, start_sec) per chunk.
    let chunks_meta: Vec<(usize, usize, f32)> = vad_chunks
        .iter()
        .filter_map(|c| {
            let mel_start = c.start_sample / hop_length;
            let mel_end = (c.end_sample / hop_length).min(total_mel_frames);
            if mel_end <= mel_start {
                return None;
            }
            let start_sec = c.start_sample as f32 / sample_rate as f32;
            Some((mel_start, mel_end - mel_start, start_sec))
        })
        .collect();
    if chunks_meta.is_empty() {
        println!("\nNo speech detected; transcript is empty.");
        return Ok(());
    }

    // ─── Encoder JIT ─────────────────────────────────────────────────────
    let num_chunks = chunks_meta.len();

    // Shrink JIT bounds to the actual VAD chunk extent (see `gigaam_infer.rs`
    // for the full reasoning). `max_mel_frames`/`max_batch_size` from the
    // model config are encoder budgets, not per-audio bounds — sizing the JIT
    // to them allocates `[B, n_heads, T_max, T_max]` scores buffers that
    // dwarf the actual chunk shape (`T_actual ≪ T_max`).
    let actual_max_chunk_mel = chunks_meta.iter().map(|(_, len, _)| *len).max().unwrap_or(0);
    let jit_t_mel = (actual_max_chunk_mel + 2 * subsampling_factor)
        .next_multiple_of(subsampling_factor)
        .min(max_t_mel)
        .max(subsampling_factor);

    let target_scores_mib: usize = env::var("MOROK_MAX_SCORES_MIB").ok().and_then(|s| s.parse().ok()).unwrap_or(256);
    let target_scores_bytes = target_scores_mib * 1024 * 1024;
    let t_sub_max = (jit_t_mel / subsampling_factor).max(1);
    let scores_dtype_bytes = model.input_dtype().bytes();
    let bytes_per_batch = model.config.n_heads * t_sub_max * t_sub_max * scores_dtype_bytes;
    let max_batch_by_memory = (target_scores_bytes / bytes_per_batch.max(1)).max(1);
    let max_batch = max_batch_by_memory.min(model.config.max_batch_size).min(num_chunks);
    println!(
        "Chunking into {} VAD chunks (longest {} mel frames); JIT bounds [B={}, T_mel={}] (budget={} MiB/scores)",
        num_chunks, actual_max_chunk_mel, max_batch, jit_t_mel, target_scores_mib
    );

    let t_prepare = Instant::now();
    let mut encoder_jit = GigaAmRnntEncoderJit::new(Arc::clone(&model)).with_b_bound(max_batch).with_t_bound(jit_t_mel);
    let prepare_config = PrepareConfig::from_env();
    println!("AMX renderer      {}", if amx_enabled { "enabled (MOROK_AMX=1)" } else { "disabled" });
    println!("Preparing encoder JIT plan... [{max_batch}, {n_mels}, {jit_t_mel}]");
    encoder_jit.prepare_with_config(
        morok_model::jit::InputSpec::f32(&[max_batch, n_mels, jit_t_mel]),
        morok_model::jit::InputSpec::i32(&[max_batch]),
        &prepare_config,
    )?;
    let dt_prepare_enc = t_prepare.elapsed();

    // ─── Predictor + joint backend ───────────────────────────────────────
    let t_step = Instant::now();
    println!("Preparing predictor + joint step JITs (B=1)...");
    let mut step_backend = RnntStepBackend::from_model(Arc::clone(&model))?;
    let dt_prepare_step = t_step.elapsed();

    let decoder = morok_arch::rnnt::RnntDecoder::new(
        model.vocabulary.clone(),
        morok_arch::rnnt::RnntOpts { max_symbols_per_step: model.max_symbols_per_step },
    );

    // ─── Per-chunk loop ─────────────────────────────────────────────────
    let t_loop = Instant::now();
    let mut dt_pack = std::time::Duration::ZERO;
    let mut dt_enc_exec = std::time::Duration::ZERO;
    let mut dt_enc_read = std::time::Duration::ZERO;
    let mut dt_transpose = std::time::Duration::ZERO;
    let mut dt_decode = std::time::Duration::ZERO;
    let mut chunk_texts: Vec<String> = Vec::with_capacity(num_chunks);
    for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
        let b = (num_chunks - chunk_batch_start).min(max_batch);
        let mut chunk_lengths = vec![0usize; b];

        // Pack mel + lengths.
        let t_pack = Instant::now();
        {
            let mut view = encoder_jit.mel_mut()?.as_array_mut::<f32>()?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice.fill(0.0);
            for (bi, chunk_len) in chunk_lengths.iter_mut().enumerate() {
                let &(mel_start, valid, _start_sec) = &chunks_meta[chunk_batch_start + bi];
                *chunk_len = valid;
                for mel_bin in 0..n_mels {
                    let src = mel_bin * total_mel_frames + mel_start;
                    let dst = ((bi * n_mels) + mel_bin) * jit_t_mel;
                    slice[dst..dst + valid].copy_from_slice(&full_mel_data[src..src + valid]);
                }
            }
        }
        {
            let mut view = encoder_jit.lengths_mut()?.as_array_mut::<i32>()?;
            let slice = view.as_slice_mut().expect("contiguous");
            slice.fill(0);
            for (i, len) in chunk_lengths.iter().enumerate() {
                slice[i] = *len as i32;
            }
        }
        dt_pack += t_pack.elapsed();

        // Execute encoder.
        let t_exec = chunk_lengths.iter().copied().max().unwrap_or(1).max(1);
        let t_exec_sub = subs_output_length(subs_kernel_size, t_exec);
        let t_enc = Instant::now();
        encoder_jit.execute_with_vars(&[("b", b as i64), ("t", t_exec as i64)])?;
        dt_enc_exec += t_enc.elapsed();

        // Read encoder output [B, d_model, T_sub] and decode each chunk.
        let t_read = Instant::now();
        let enc_out = encoder_jit.output()?.as_array::<f32>()?;
        let enc_slice = enc_out.as_slice().expect("contiguous encoder output");
        dt_enc_read += t_read.elapsed();
        let item_stride = d_model * t_exec_sub;
        for (bi, mel_len) in chunk_lengths.iter().enumerate() {
            let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
            // Per-sample slab is [d_model, T_sub_stride] row-major; we need
            // [T_sub_actual, d_model] frame-major for the arch decoder.
            let item = &enc_slice[bi * item_stride..bi * item_stride + item_stride];
            let t_tr = Instant::now();
            let mut frames = vec![0.0f32; actual_sub * d_model];
            for t in 0..actual_sub {
                for d in 0..d_model {
                    frames[t * d_model + d] = item[d * t_exec_sub + t];
                }
            }
            dt_transpose += t_tr.elapsed();
            // SP pieces retain `▁` (U+2581) on word-initial tokens; after
            // concatenation, replace them with spaces for natural Russian
            // text. Char-wise checkpoints (no SP) skip the replace.
            let &(_, _, start_sec) = &chunks_meta[chunk_batch_start + bi];
            let t_dec = Instant::now();
            let (raw, words) = if want_timestamps {
                let (raw, emissions) =
                    decoder.decode_with_timestamps(&frames, actual_sub, actual_sub, d_model, &mut step_backend)?;
                // Per-chunk frame_shift mirrors upstream's
                // `audio_length_samples / SAMPLE_RATE / encoder_seq_len`
                // (gigaam/timestamps_utils.py:8): chunk_duration_secs / actual_sub.
                let chunk_duration = (*mel_len as f32) * hop_length as f32 / sample_rate as f32;
                let frame_shift = chunk_duration / (actual_sub as f32).max(1.0);
                (raw, Some(decoder.frames_to_words(&emissions, frame_shift)))
            } else {
                (decoder.decode(&frames, actual_sub, actual_sub, d_model, &mut step_backend)?, None)
            };
            dt_decode += t_dec.elapsed();
            let text = if model.sentencepiece { raw.replace('\u{2581}', " ").trim().to_string() } else { raw };
            if let Some(words) = words {
                for w in &words {
                    println!("  [{:>6.2} - {:>6.2}] {}", start_sec + w.start, start_sec + w.end, w.text);
                }
            } else if !text.is_empty() {
                println!("  [{:>6.1}s] {}", start_sec, text);
            }
            chunk_texts.push(text);
        }
    }
    let dt_loop = t_loop.elapsed();
    let full_text = chunk_texts.iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(" ");

    println!("\nAudio duration: {:.1}s", duration_s);
    println!("\n--- Timings ---");
    println!("audio load        {:>9}", fmt_duration(dt_audio));
    println!("model load        {:>9}", fmt_duration(dt_model));
    println!("mel extract       {:>9}", fmt_duration(dt_mel));
    println!("vad prepare       {:>9}", fmt_duration(dt_vad_prepare));
    println!("vad exec          {:>9}", fmt_duration(dt_vad));
    println!("encoder prepare   {:>9}", fmt_duration(dt_prepare_enc));
    println!("step prepare      {:>9}", fmt_duration(dt_prepare_step));
    println!("chunk loop        {:>9}", fmt_duration(dt_loop));
    println!("  pack mel+len    {:>9}", fmt_duration(dt_pack));
    println!("  encoder exec    {:>9}", fmt_duration(dt_enc_exec));
    println!("  encoder read    {:>9}", fmt_duration(dt_enc_read));
    println!("  transpose       {:>9}", fmt_duration(dt_transpose));
    println!("  decode          {:>9}", fmt_duration(dt_decode));
    let s = &step_backend.stats;
    let s_total = s.total();
    println!("    n_steps          {}", s.n_steps);
    println!("    n_commits        {}", s.n_commits);
    println!("    n_resets         {}", s.n_resets);
    println!("    pred  pack    {:>9}", fmt_duration(s.t_pred_pack));
    println!("    pred  exec    {:>9}", fmt_duration(s.t_pred_exec));
    println!("    pred  read    {:>9}", fmt_duration(s.t_pred_read));
    println!("    joint pack    {:>9}", fmt_duration(s.t_joint_pack));
    println!("    joint exec    {:>9}", fmt_duration(s.t_joint_exec));
    println!("    joint read    {:>9}", fmt_duration(s.t_joint_read));
    println!("    step total    {:>9}", fmt_duration(s_total));
    if let Some(avg_ns) = (s_total.as_nanos() as u64).checked_div(s.n_steps) {
        println!("    avg/step      {:>9}", fmt_duration(std::time::Duration::from_nanos(avg_ns)));
    }
    println!("total             {:>9}", fmt_duration(t_total.elapsed()));
    if duration_s > 0.0 {
        println!("loop RTF          {:>9.2}x", dt_loop.as_secs_f32() / duration_s);
    }

    println!("\n--- Transcript ---\n{}", full_text);
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

fn subs_output_length(kernel_size: usize, mel_frames: usize) -> usize {
    let pad = (kernel_size - 1) / 2;
    let mut len = mel_frames;
    for _ in 0..2 {
        len = (len + 2 * pad - kernel_size) / 2 + 1;
    }
    len
}

fn fmt_duration(d: std::time::Duration) -> String {
    if d.as_secs() > 0 {
        format!("{:.2}s", d.as_secs_f32())
    } else if d.as_millis() > 0 {
        format!("{}ms", d.as_millis())
    } else {
        format!("{}μs", d.as_micros())
    }
}
