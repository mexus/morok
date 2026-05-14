//! GigaAM CTC inference example: VAD-chunked, batched, dynamic-shape JIT.
//!
//! End-to-end pipeline:
//!   1. Load WAV + extract log-mel features once.
//!   2. Run Silero VAD to find speech-bearing chunks (silence dropped).
//!   3. Build a single batched JIT plan sized to the longest VAD chunk so the
//!      encoder's `[B, n_heads, T, T]` attention scores stay bounded
//!      independent of audio length (see `MOROK_MAX_SCORES_MIB` below).
//!   4. Loop over chunks in batches: pack mel, execute encoder, greedy/beam
//!      decode the CTC head, emit per-chunk text.
//!
//! Usage:
//!   cargo run -p morok-model --release --example gigaam_infer -- audio.wav
//!
//! Env knobs:
//!   MOROK_AMX=1              Enable AMX renderer (Apple Silicon).
//!   MOROK_BEAM_DECODE=1      Promote greedy CTC to beam search.
//!   MOROK_MAX_SCORES_MIB=N   Per-allocation budget for the SDPA scores
//!                            buffer; caps `max_batch` so two simultaneously
//!                            live `[B, H, T_sub², dtype]` scores tensors
//!                            stay under `2 × N MiB`. Default 256.

use std::env;
use std::sync::Arc;
use std::time::{Duration, Instant};

use morok_arch::ctc::CtcDecoder;
use morok_dtype::DType;
use morok_model::audio::MelSpectrogram;
use morok_model::gigaam::{CtcHeadJit, GigaAm, GigaAmEncoderJit, SubsamplingMode};
use morok_tensor::{PrepareConfig, Tensor};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let t_total = Instant::now();

    let wav_path = env::args().nth(1).ok_or("usage: gigaam_infer <audio.wav>")?;
    let amx_enabled = env::var("MOROK_AMX").as_deref() == Ok("1");
    let beam_decode = env::var("MOROK_BEAM_DECODE").as_deref() == Ok("1");

    // ─── Audio ─────────────────────────────────────────────────────────────
    let t_audio = Instant::now();
    println!("Loading audio: {wav_path}");
    let (waveform, wav_sample_rate) = load_wav(&wav_path)?;
    let duration_s = waveform.len() as f32 / wav_sample_rate as f32;
    println!("Samples: {} ({:.1}s @ {} Hz)", waveform.len(), duration_s, wav_sample_rate);
    let dt_audio = t_audio.elapsed();

    // ─── Model + decoder ───────────────────────────────────────────────────
    let t_model = Instant::now();
    println!("\nLoading GigaAM...");
    let model = GigaAm::from_hub_with_revision("vpermilp/GigaAM-v3", "ctc")?;
    if wav_sample_rate as usize != model.config.sample_rate {
        return Err(format!(
            "WAV is {} Hz, model expects {} Hz (resample first)",
            wav_sample_rate, model.config.sample_rate
        )
        .into());
    }
    let mut decoder = if beam_decode {
        // Promote a greedy config to a beam decoder using its vocabulary.
        match &model.config.decoder {
            CtcDecoder::Greedy(g) => CtcDecoder::Beam(Box::new(morok_arch::ctc::BeamDecoder::new(
                g.vocabulary().to_vec(),
                morok_arch::ctc::BeamOpts::default(),
            ))),
            other => other.clone(),
        }
    } else {
        model.config.decoder.clone()
    };
    let total_vocab = decoder.total_vocab();
    println!(
        "Loaded: {} layers, d_model={}, vocab_size={}, decoder={}",
        model.config.n_layers,
        model.config.d_model,
        model.config.vocab_size,
        if beam_decode { "beam" } else { "greedy" },
    );
    let dt_model = t_model.elapsed();

    // ─── Mel features (whole audio, sliced per-chunk later) ────────────────
    let mel = MelSpectrogram::new(&morok_model::audio::MelConfig {
        sample_rate: model.config.sample_rate,
        n_fft: model.config.n_fft,
        hop_length: model.config.hop_length,
        win_length: model.config.win_length,
        n_mels: model.config.n_mels,
        center: model.config.mel_center,
    });
    let n_mels = mel.n_mels();
    let max_t_mel = model.config.max_mel_frames;
    let hop_length = model.config.hop_length;
    let subsampling_factor = model.config.subsampling_factor;
    let subs_kernel_size = match model.config.subsampling_mode {
        SubsamplingMode::Conv1d => model.config.subs_kernel_size,
        SubsamplingMode::Conv2d => 3,
    };
    let total_mel_frames = mel.num_frames(waveform.len());
    if total_mel_frames == 0 {
        println!("No frames produced from audio.");
        return Ok(());
    }

    let t_mel = Instant::now();
    println!("\nExtracting mel features: [1, {n_mels}, {total_mel_frames}]");
    let mut full_mel = Tensor::full(&[1, n_mels, total_mel_frames], 0.0f32, DType::Float32)?;
    full_mel.realize().unwrap();
    {
        let mut view = full_mel.array_view_mut::<f32>()?;
        mel.forward_into(&waveform, &mut view);
    }
    let full_mel_data = full_mel.as_vec::<f32>()?;
    let dt_mel = t_mel.elapsed();

    // ─── VAD chunking ──────────────────────────────────────────────────────
    let t_vad = Instant::now();
    println!("\nLoading Silero VAD...");
    let vad_model = morok_model::silero_vad::SileroVad::from_hub()?;
    let mut vad = morok_model::silero_vad::VadInference::new(vad_model)?;
    let probs = vad.probs(&waveform)?;
    // Each chunk's mel-frame count must fit max_t_mel. The chunker rounds
    // chunk boundaries to align_to-sample multiples; the start/end snap
    // can each shift by up to one alignment step, so a chunk's mel length
    // can grow by 2 × subsampling_factor beyond the nominal max_duration.
    let mel_headroom = 2 * subsampling_factor;
    let encoder_capacity_secs =
        (max_t_mel.saturating_sub(mel_headroom) as f32 * hop_length as f32) / model.config.sample_rate as f32;
    let default_opts = morok_arch::vad::ChunkerOpts::default();
    let chunker_opts = morok_arch::vad::ChunkerOpts {
        sample_rate: model.config.sample_rate as u32,
        samples_per_prob: morok_model::silero_vad::NUM_SAMPLES,
        max_duration: default_opts.max_duration.min(encoder_capacity_secs),
        strict_limit_duration: default_opts.strict_limit_duration.min(encoder_capacity_secs),
        align_to: hop_length * subsampling_factor,
        ..default_opts
    };
    let vad_chunks = morok_arch::vad::chunks_from_probs(&probs, &chunker_opts)?;
    let dt_vad = t_vad.elapsed();
    println!(
        "VAD: {} probs → {} chunks (max_chunk={:.1}s, strict={:.1}s) in {}",
        probs.len(),
        vad_chunks.len(),
        chunker_opts.max_duration,
        chunker_opts.strict_limit_duration,
        fmt_duration(dt_vad),
    );

    // (mel_start, mel_len, start_sec) per chunk.
    let chunks_meta: Vec<(usize, usize, f32)> = vad_chunks
        .iter()
        .filter_map(|c| {
            let mel_start = c.start_sample / hop_length;
            let mel_end = (c.end_sample / hop_length).min(total_mel_frames);
            (mel_end > mel_start)
                .then(|| (mel_start, mel_end - mel_start, c.start_sample as f32 / model.config.sample_rate as f32))
        })
        .collect();
    if chunks_meta.is_empty() {
        println!("\nNo speech detected; transcript is empty.");
        return Ok(());
    }

    // ─── JIT bounds: shrink to actual chunk extent ────────────────────────
    // `max_mel_frames` / `max_batch_size` from the model config are encoder
    // worst-case budgets, not per-audio bounds. Sizing the JIT to them
    // allocates `[B, n_heads, T_max, T_max]` scores buffers that dwarf the
    // actual VAD chunk shape (`T_actual ≪ T_max`). Compute the real bound
    // here so the JIT plan matches what we'll execute.
    let num_chunks = chunks_meta.len();
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
        "Chunking into {} VAD chunks (longest {} mel frames); JIT bounds [B={}, T_mel={}]",
        num_chunks, actual_max_chunk_mel, max_batch, jit_t_mel,
    );

    // ─── JIT prepare ───────────────────────────────────────────────────────
    let t_prepare = Instant::now();
    let model = Arc::new(model);
    let d_model = model.config.d_model;
    let jit_t_sub = subs_output_length(subs_kernel_size, jit_t_mel);
    let mut encoder_jit = GigaAmEncoderJit::new(Arc::clone(&model)).with_b_bound(max_batch).with_t_bound(jit_t_mel);
    let mut head_jit = CtcHeadJit::new(Arc::clone(&model)).with_b_bound(max_batch).with_t_sub_bound(jit_t_sub);
    println!("Preparing encoder JIT plan... [{max_batch}, {n_mels}, {jit_t_mel}]");
    println!("AMX renderer {}", if amx_enabled { "enabled" } else { "disabled" });
    let prepare_config = PrepareConfig::from_env();
    encoder_jit.prepare_with_config(
        morok_model::jit::InputSpec::f32(&[max_batch, n_mels, jit_t_mel]),
        morok_model::jit::InputSpec::i32(&[max_batch]),
        &prepare_config,
    )?;
    println!("Preparing CTC head JIT plan... [{max_batch}, {d_model}, {jit_t_sub}]");
    head_jit
        .prepare_with_config(morok_model::jit::InputSpec::f32(&[max_batch, d_model, jit_t_sub]), &prepare_config)?;
    let dt_prepare = t_prepare.elapsed();
    println!("Plans captured.");

    // ─── Inference loop ────────────────────────────────────────────────────
    let t_loop = Instant::now();
    let mut chunk_texts: Vec<String> = Vec::with_capacity(num_chunks);
    for chunk_batch_start in (0..num_chunks).step_by(max_batch) {
        let b = (num_chunks - chunk_batch_start).min(max_batch);
        let mut chunk_lengths = vec![0usize; b];

        // Pack mel slices + valid lengths into the encoder's input buffers.
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

        // Execute encoder with the actual batch shape (b, t) for this iteration.
        let t_exec = chunk_lengths.iter().copied().max().unwrap_or(1).max(1);
        let t_exec_sub = subs_output_length(subs_kernel_size, t_exec);
        encoder_jit.execute_with_vars(&[("b", b as i64), ("t", t_exec as i64)])?;

        // Forward the encoder output into the head's input buffer. The encoder
        // output buffer is flat 1D (kernel-packed at the dynamic stride
        // `[b, d_model, t_exec_sub]`); the head input keeps the static 3D
        // shape `[max_batch, d_model, jit_t_sub]` from `prepare()`. Reshape
        // the source view to 3D and let ndarray cross-stride the assign into
        // the destination's matching sub-slice.
        {
            let n = b * d_model * t_exec_sub;
            let src_flat = encoder_jit.output()?.as_array::<f32>()?;
            let src_3d = src_flat
                .slice(ndarray::s![0..n])
                .into_shape_with_order((b, d_model, t_exec_sub))
                .expect("encoder output reshape");
            let dst_flat = head_jit.encoded_mut()?.as_array_mut::<f32>()?;
            let mut dst_3d =
                dst_flat.into_shape_with_order((max_batch, d_model, jit_t_sub)).expect("head input reshape");
            dst_3d.slice_mut(ndarray::s![0..b, 0..d_model, 0..t_exec_sub]).assign(&src_3d);
        }
        head_jit.execute_with_vars(&[("b", b as i64), ("t_sub", t_exec_sub as i64)])?;

        // Decode per batch item.
        let logits_array = head_jit.output()?.as_array::<f32>().expect("failed to read output logits");
        let logits_slice = logits_array.as_slice().expect("contiguous output logits");
        let item_stride = t_exec_sub * total_vocab;
        for (bi, mel_len) in chunk_lengths.iter().enumerate() {
            let actual_sub = subs_output_length(subs_kernel_size, *mel_len);
            let item_base = bi * item_stride;
            let item_slice = &logits_slice[item_base..item_base + item_stride];
            let text = decoder.decode(item_slice, t_exec_sub, actual_sub)?;
            let &(_, _, start_sec) = &chunks_meta[chunk_batch_start + bi];
            if !text.is_empty() {
                println!("  [{:>6.1}s] {}", start_sec, text);
            }
            chunk_texts.push(text);
        }
    }
    let dt_loop = t_loop.elapsed();

    // ─── Output ────────────────────────────────────────────────────────────
    let full_text = chunk_texts.iter().filter(|s| !s.is_empty()).cloned().collect::<Vec<_>>().join(" ");
    println!("\n--- Timings ---");
    println!("audio load      {:>9}", fmt_duration(dt_audio));
    println!("model load      {:>9}", fmt_duration(dt_model));
    println!("mel extract     {:>9}", fmt_duration(dt_mel));
    println!("vad             {:>9}", fmt_duration(dt_vad));
    println!("jit prepare     {:>9}", fmt_duration(dt_prepare));
    println!("chunk loop      {:>9}", fmt_duration(dt_loop));
    println!("total           {:>9}", fmt_duration(t_total.elapsed()));
    if duration_s > 0.0 {
        println!("loop RTF        {:>9.2}x", dt_loop.as_secs_f32() / duration_s);
    }
    println!("\n--- Transcript ---\n{full_text}");
    Ok(())
}

fn fmt_duration(d: Duration) -> String {
    if d.as_secs() >= 1 { format!("{:.2}s", d.as_secs_f64()) } else { format!("{}ms", d.as_millis()) }
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
