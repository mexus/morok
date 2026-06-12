//! Kaldi-compatible log-mel fbank for FireRedVAD, matching
//! `kaldi-native-fbank` with the upstream options (16 kHz, 25 ms / 10 ms,
//! `snip_edges`, `dither = 0`, 80 bins): per-frame DC removal, per-frame
//! pre-emphasis 0.97, Povey window, power spectrum over a 512-point FFT,
//! triangular mel bins on `1127·ln(1 + f/700)` between 20 Hz and Nyquist,
//! natural log floored at `f32::EPSILON`.
//!
//! Runs eagerly on CPU via `realfft`, like [`crate::audio::mel`]. The math
//! differs from that extractor everywhere that matters (window function,
//! framing, mel scale, pre-emphasis), hence a separate implementation.

use std::f32::consts::PI;
use std::sync::Arc;

use realfft::{RealFftPlanner, RealToComplex};

use super::{FRAME_LENGTH, FRAME_SHIFT, N_MELS};

const N_FFT: usize = 512; // next power of two >= FRAME_LENGTH
const LOW_FREQ: f32 = 20.0;
const SAMPLE_RATE: f32 = 16_000.0;
/// Kaldi reads 16-bit PCM without normalization; Svod waveforms are `[-1, 1]`.
const INT16_SCALE: f32 = 32_768.0;
const PREEMPH: f32 = 0.97;

pub struct FireRedFbank {
    r2c: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    /// Sparse mel rows `(first_bin, weights)` — same layout as
    /// [`crate::audio::mel::MelSpectrogram`].
    mel_fb: Vec<(usize, Vec<f32>)>,
}

impl Default for FireRedFbank {
    fn default() -> Self {
        Self::new()
    }
}

impl FireRedFbank {
    pub fn new() -> Self {
        let r2c = RealFftPlanner::<f32>::new().plan_fft_forward(N_FFT);
        Self { r2c, window: povey_window(FRAME_LENGTH), mel_fb: kaldi_mel_banks() }
    }

    /// `snip_edges` frame count: only complete 25 ms windows produce a frame.
    pub fn num_frames(&self, n_samples: usize) -> usize {
        if n_samples < FRAME_LENGTH { 0 } else { 1 + (n_samples - FRAME_LENGTH) / FRAME_SHIFT }
    }

    /// Extract `[num_frames * N_MELS]` row-major log-mel features from a
    /// `[-1, 1]`-scale waveform (pre-CMVN — normalization happens inside the
    /// model graph).
    pub fn forward(&self, waveform: &[f32]) -> Vec<f32> {
        let n_frames = self.num_frames(waveform.len());
        let mut feat = vec![0.0f32; n_frames * N_MELS];
        let mut indata = self.r2c.make_input_vec();
        let mut outdata = self.r2c.make_output_vec();
        let mut power = vec![0.0f32; N_FFT / 2 + 1];

        for (frame_idx, out) in feat.chunks_exact_mut(N_MELS).enumerate() {
            let frame = &waveform[frame_idx * FRAME_SHIFT..frame_idx * FRAME_SHIFT + FRAME_LENGTH];
            let buf = &mut indata[..FRAME_LENGTH];
            for (d, &s) in buf.iter_mut().zip(frame) {
                *d = s * INT16_SCALE;
            }
            let mean = buf.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for d in buf.iter_mut() {
                *d -= mean;
            }
            for i in (1..FRAME_LENGTH).rev() {
                buf[i] -= PREEMPH * buf[i - 1];
            }
            buf[0] -= PREEMPH * buf[0];
            for (d, w) in buf.iter_mut().zip(&self.window) {
                *d *= w;
            }
            indata[FRAME_LENGTH..].fill(0.0);

            self.r2c.process(&mut indata, &mut outdata).expect("FFT failed");
            for (p, c) in power.iter_mut().zip(&outdata) {
                *p = c.re * c.re + c.im * c.im;
            }
            for (mel, (first, weights)) in out.iter_mut().zip(&self.mel_fb) {
                let sum: f32 = weights.iter().zip(&power[*first..]).map(|(w, &p)| w * p).sum();
                *mel = sum.max(f32::EPSILON).ln();
            }
        }
        feat
    }
}

/// Povey window: `(0.5 - 0.5·cos(2π i / (L-1)))^0.85` — Kaldi's default.
fn povey_window(len: usize) -> Vec<f32> {
    (0..len).map(|i| (0.5 - 0.5 * (2.0 * PI * i as f32 / (len - 1) as f32).cos()).powf(0.85)).collect()
}

/// Kaldi `MelBanks`: triangles on the `1127·ln(1 + f/700)` mel axis over the
/// first `N_FFT/2` FFT bins (Nyquist bin excluded, as in Kaldi), 20 Hz to
/// Nyquist, returned as sparse `(first_bin, weights)` rows.
fn kaldi_mel_banks() -> Vec<(usize, Vec<f32>)> {
    let mel = |f: f32| 1127.0 * (1.0 + f / 700.0).ln();
    let num_fft_bins = N_FFT / 2;
    let fft_bin_width = SAMPLE_RATE / N_FFT as f32;
    let (mel_low, mel_high) = (mel(LOW_FREQ), mel(SAMPLE_RATE / 2.0));
    let delta = (mel_high - mel_low) / (N_MELS + 1) as f32;

    (0..N_MELS)
        .map(|m| {
            let (left, center, right) =
                (mel_low + m as f32 * delta, mel_low + (m + 1) as f32 * delta, mel_low + (m + 2) as f32 * delta);
            let mut first = None;
            let mut weights = Vec::new();
            for i in 0..num_fft_bins {
                let mel_f = mel(i as f32 * fft_bin_width);
                if mel_f > left && mel_f < right {
                    first.get_or_insert(i);
                    weights.push(if mel_f <= center { (mel_f - left) / delta } else { (right - mel_f) / delta });
                } else if first.is_some() {
                    break;
                }
            }
            (first.expect("non-empty mel bin"), weights)
        })
        .collect()
}
