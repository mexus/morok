//! Whisper mel spectrogram: uses the pre-computed Slaney filterbank from
//! Whisper's `assets/mel_filters.npz`, then applies `log10` + clamp + normalize.

use crate::audio::{MelConfig, MelScale, MelSpectrogram};

use super::config::{HOP_LENGTH, N_FFT, SAMPLE_RATE};

/// Whisper-specific mel spectrogram extractor.
pub struct WhisperMel {
    inner: MelSpectrogram,
}

impl WhisperMel {
    pub fn new(n_mels: usize) -> Self {
        let inner = MelSpectrogram::new(&MelConfig {
            sample_rate: SAMPLE_RATE,
            n_fft: N_FFT,
            hop_length: HOP_LENGTH,
            win_length: N_FFT,
            n_mels,
            center: true,
            mel_scale: MelScale::Slaney,
        });
        Self { inner }
    }

    pub fn n_mels(&self) -> usize {
        self.inner.n_mels()
    }

    pub fn num_frames(&self, waveform_len: usize) -> usize {
        self.inner.num_frames(waveform_len)
    }

    /// Compute log-mel spectrogram matching `whisper.audio.log_mel_spectrogram`.
    /// Returns `[n_mels, n_frames]` row-major.
    ///
    /// Audio is pad-or-trimmed to 30 seconds (N_SAMPLES) before STFT, matching
    /// `whisper.audio.pad_or_trim` + `log_mel_spectrogram`.
    pub fn compute(&self, waveform: &[f32]) -> Vec<f32> {
        let audio_owned: Vec<f32>;
        let audio: &[f32] = if waveform.len() != super::config::N_SAMPLES {
            audio_owned = {
                let mut v = vec![0.0f32; super::config::N_SAMPLES];
                let copy_len = waveform.len().min(super::config::N_SAMPLES);
                v[..copy_len].copy_from_slice(&waveform[..copy_len]);
                v
            };
            &audio_owned
        } else {
            waveform
        };

        let power = self.inner.forward_power(audio);
        if power.is_empty() {
            return power;
        }

        // log10(clamp(x, 1e-10))
        let mut log_spec: Vec<f32> = power.iter().map(|&p| p.clamp(1e-10, f32::MAX).log10()).collect();

        // max(x, x.max() - 8.0)
        let max_val = log_spec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let clamp_floor = max_val - 8.0;
        for v in log_spec.iter_mut() {
            *v = v.max(clamp_floor);
        }

        // (x + 4.0) / 4.0
        for v in log_spec.iter_mut() {
            *v = (*v + 4.0) / 4.0;
        }

        log_spec
    }
}
