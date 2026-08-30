//! Audio preprocessing: mel spectrogram, STFT, mel filterbanks.
//!
//! Runs eagerly on CPU using `realfft` — not through Svod's lazy tensor pipeline.

use std::f32::consts::PI;

use std::sync::Arc;

use ndarray::{Array2, ArrayViewMutD};
use realfft::{RealFftPlanner, RealToComplex};

/// Mel filterbank scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[derive(Default)]
pub enum MelScale {
    /// HTK mel scale: `2595·log10(1+f/700)`, peak height 1 (unnormalized).
    /// Matches torchaudio's `melscale_fbanks(slk_norm=None)`.
    #[default]
    Htk,
    /// librosa Slaney scale: linear below 1 kHz, log above; area-normalized
    /// triangles. Matches `librosa.filters.mel(norm='slaney')` and
    /// Whisper's pre-computed `mel_filters.npz`.
    Slaney,
}

/// Configuration for mel spectrogram extraction.
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    pub center: bool,
    pub mel_scale: MelScale,
}

/// CPU-based log-mel spectrogram extractor.
pub struct MelSpectrogram {
    r2c: Arc<dyn RealToComplex<f32>>,
    /// Sparse filterbank rows: `(first_bin, weights)` per mel — each triangular
    /// filter covers only a handful of contiguous FFT bins, so the dense
    /// `[n_mels, n_bins]` matvec wastes ~40x. Built dense, sparsified once;
    /// ascending-bin accumulation keeps the matvec bit-identical to dense.
    mel_fb: Vec<(usize, Vec<f32>)>,
    window: Vec<f32>,
    n_fft: usize,
    hop_length: usize,
    center: bool,
}

impl MelSpectrogram {
    pub fn new(config: &MelConfig) -> Self {
        let n_fft = config.n_fft;

        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(n_fft);

        let window = hann_window(config.n_fft, config.win_length);

        let dense = build_mel_filterbank(config.n_mels, n_fft, config.sample_rate as f32, config.mel_scale);
        let mel_fb = dense
            .rows()
            .into_iter()
            .map(|row| {
                let first = row.iter().position(|&w| w != 0.0).unwrap_or(0);
                let last = row.iter().rposition(|&w| w != 0.0).map_or(first, |l| l + 1);
                (first, row.slice(ndarray::s![first..last]).to_vec())
            })
            .collect();

        Self { r2c, mel_fb, window, n_fft, hop_length: config.hop_length, center: config.center }
    }

    pub fn n_mels(&self) -> usize {
        self.mel_fb.len()
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        let signal_len = if self.center { waveform_len + self.n_fft } else { waveform_len };
        if signal_len >= self.n_fft { (signal_len - self.n_fft) / self.hop_length + 1 } else { 0 }
    }

    pub fn forward_into(&self, waveform: &[f32], out: &mut ArrayViewMutD<'_, f32>) {
        let n_fft = self.n_fft;
        let signal_owned = self.center.then(|| reflect_pad(waveform, n_fft / 2));
        let signal: &[f32] = signal_owned.as_deref().unwrap_or(waveform);

        let n_frames = if signal.len() >= n_fft { (signal.len() - n_fft) / self.hop_length + 1 } else { 0 };
        let n_bins = n_fft / 2 + 1;
        let n_mels = self.mel_fb.len();

        debug_assert!(
            {
                let shape = out.shape();
                shape.len() >= 2
                    && shape[shape.len() - 2] == n_mels
                    && shape[shape.len() - 1] == n_frames
                    && shape[..shape.len() - 2].iter().all(|&d| d == 1)
            },
            "forward_into: expected output trailing dims [.., {n_mels}, {n_frames}] with leading 1s, got {:?}",
            out.shape(),
        );

        let out_slice = out.as_slice_mut().expect("output must be contiguous");

        out_slice[..n_mels * n_frames].fill(0.0);

        let mut indata = self.r2c.make_input_vec();
        let mut outdata = self.r2c.make_output_vec();
        let mut power = vec![0.0f32; n_bins];

        for frame_idx in 0..n_frames {
            let start = frame_idx * self.hop_length;
            for i in 0..n_fft {
                indata[i] = signal[start + i] * self.window[i];
            }
            self.r2c.process(&mut indata, &mut outdata).expect("FFT failed");

            for (i, c) in outdata.iter().enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            for (mel_idx, (first, weights)) in self.mel_fb.iter().enumerate() {
                let mut sum = 0.0f32;
                for (w, &p) in weights.iter().zip(&power[*first..]) {
                    sum += w * p;
                }
                out_slice[mel_idx * n_frames + frame_idx] = sum.clamp(1e-9, 1e9).ln();
            }
        }
    }

    /// Compute the raw mel power spectrogram (no log compression) into a flat
    /// `Vec<f32>` of length `n_mels * n_frames`. Used by Whisper which applies
    /// its own `log10` + clamp + normalize.
    pub fn forward_power(&self, waveform: &[f32]) -> Vec<f32> {
        let n_fft = self.n_fft;
        let signal_owned = self.center.then(|| reflect_pad(waveform, n_fft / 2));
        let signal: &[f32] = signal_owned.as_deref().unwrap_or(waveform);

        let n_frames_raw = if signal.len() >= n_fft { (signal.len() - n_fft) / self.hop_length + 1 } else { 0 };
        // Match torch.stft(...)[..., :-1]: drop the last frame.
        // torch.stft with center=True produces ceil(L/hop) frames but Whisper
        // drops the trailing one for exact N_FRAMES alignment.
        let n_frames = n_frames_raw.saturating_sub(1);
        let n_bins = n_fft / 2 + 1;
        let n_mels = self.mel_fb.len();

        let mut result = vec![0.0f32; n_mels * n_frames];

        let mut indata = self.r2c.make_input_vec();
        let mut outdata = self.r2c.make_output_vec();
        let mut power = vec![0.0f32; n_bins];

        for frame_idx in 0..n_frames_raw {
            let start = frame_idx * self.hop_length;
            for i in 0..n_fft {
                indata[i] = signal[start + i] * self.window[i];
            }
            self.r2c.process(&mut indata, &mut outdata).expect("FFT failed");

            for (i, c) in outdata.iter().enumerate() {
                power[i] = c.re * c.re + c.im * c.im;
            }

            // Skip the last frame (matching torch.stft[..., :-1])
            if frame_idx >= n_frames {
                continue;
            }

            for (mel_idx, (first, weights)) in self.mel_fb.iter().enumerate() {
                let mut sum = 0.0f32;
                for (w, &p) in weights.iter().zip(&power[*first..]) {
                    sum += w * p;
                }
                result[mel_idx * n_frames + frame_idx] = sum;
            }
        }

        result
    }
}

/// Periodic Hann window, matching `torch.hann_window(periodic=True)`, which is
/// torchaudio's default in `MelSpectrogram`. `realfft` handles only the FFT;
/// it does not provide STFT window builders.
pub(crate) fn hann_window(n_fft: usize, win_length: usize) -> Vec<f32> {
    let mut window = vec![0.0f32; n_fft];
    for (i, w) in window.iter_mut().enumerate().take(win_length) {
        *w = 0.5 * (1.0 - (2.0 * PI * i as f32 / win_length as f32).cos());
    }
    window
}

/// Build mel filterbank matrix of shape `[n_mels, n_fft/2+1]`.
fn build_mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32, scale: MelScale) -> Array2<f32> {
    match scale {
        MelScale::Htk => build_htk_filterbank(n_mels, n_fft, sample_rate),
        MelScale::Slaney => build_slaney_filterbank(n_mels, n_fft, sample_rate),
    }
}

/// HTK mel scale: `2595·log10(1+f/700)`, peak height 1 (unnormalized).
fn build_htk_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Array2<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate / 2.0;

    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);

    let mel_points: Vec<f32> =
        (0..n_mels + 2).map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32).collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f32> = hz_points.iter().map(|&f| f * n_fft as f32 / sample_rate).collect();

    let mut fb = Array2::zeros((n_mels, n_bins));
    for i in 0..n_mels {
        let left = bin_points[i];
        let center = bin_points[i + 1];
        let right = bin_points[i + 2];

        for j in 0..n_bins {
            let freq = j as f32;
            if freq >= left && freq <= center && center > left {
                fb[[i, j]] = (freq - left) / (center - left);
            } else if freq > center && freq <= right && right > center {
                fb[[i, j]] = (right - freq) / (right - center);
            }
        }
    }
    fb
}

/// librosa Slaney mel scale: linear below 1 kHz, log above; area-normalized.
/// Matches `librosa.filters.mel(sr, n_fft, n_mels, norm='slaney')`.
fn build_slaney_filterbank(n_mels: usize, n_fft: usize, sample_rate: f32) -> Array2<f32> {
    let n_bins = n_fft / 2 + 1;
    let f_max = sample_rate / 2.0;

    // Slaney mel scale (librosa hz_to_mel with htk=False)
    let f_min = 0.0f32;
    let f_sp = 200.0f32 / 3.0; // linear slope below 1 kHz (~66.67 Hz/mel)
    let min_log_hz = 1000.0f32;
    let min_log_mel = (min_log_hz - f_min) / f_sp; // = 15 mel
    let logstep = 0.068_751_775; // ln(6.4)/27
    let hz_to_mel = |freq: f32| -> f32 {
        if freq >= min_log_hz { min_log_mel + (freq / min_log_hz).ln() / logstep } else { (freq - f_min) / f_sp }
    };
    let mel_to_hz = |mel: f32| -> f32 {
        if mel >= min_log_mel { min_log_hz * ((mel - min_log_mel) * logstep).exp() } else { f_min + mel * f_sp }
    };

    // FFT bin frequencies
    let fft_freqs: Vec<f32> = (0..n_bins).map(|i| i as f32 * sample_rate / n_fft as f32).collect();

    // Mel-spaced center points
    let mel_min = hz_to_mel(0.0);
    let mel_max = hz_to_mel(f_max);
    let mel_points: Vec<f32> =
        (0..n_mels + 2).map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_mels + 1) as f32).collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Build triangular filters using librosa's ramping approach
    let mut fb = Array2::zeros((n_mels, n_bins));
    for i in 0..n_mels {
        let left = hz_points[i];
        let center = hz_points[i + 1];
        let right = hz_points[i + 2];

        // Triangular ramp
        let lower_slope = if center > left {
            fft_freqs.iter().map(|&f| ((f - left) / (center - left)).max(0.0)).collect::<Vec<_>>()
        } else {
            vec![0.0; n_bins]
        };
        let upper_slope = if right > center {
            fft_freqs.iter().map(|&f| ((right - f) / (right - center)).max(0.0)).collect::<Vec<_>>()
        } else {
            vec![0.0; n_bins]
        };

        // enorm (Slaney area normalization): 2 / (right - left) in Hz
        let enorm = 2.0 / (right - left).max(1e-10);

        for j in 0..n_bins {
            fb[[i, j]] = lower_slope[j].min(upper_slope[j]) * enorm;
        }
    }
    fb
}

/// Reflect-pad a signal by `pad` samples on each side, mirroring PyTorch's
/// `Reflect1d`: the boundary element is not duplicated, and `pad` must be
/// strictly less than the signal length (single-bounce reflection only).
pub(crate) fn reflect_pad(signal: &[f32], pad: usize) -> Vec<f32> {
    let len = signal.len();
    assert!(
        pad < len,
        "reflect_pad requires pad ({pad}) < signal length ({len}); multi-bounce reflection is not supported",
    );

    let mut padded = Vec::with_capacity(len + 2 * pad);
    for i in (1..=pad).rev() {
        padded.push(signal[i]);
    }
    padded.extend_from_slice(signal);
    for i in 1..=pad {
        padded.push(signal[len - 1 - i]);
    }
    padded
}
