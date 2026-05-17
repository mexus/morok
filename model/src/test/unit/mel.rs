use crate::audio::mel::hann_window;
use crate::audio::{MelConfig, MelSpectrogram};

struct MelOutput {
    data: ndarray::Array3<f32>,
}

impl MelOutput {
    fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }

    fn as_slice(&self) -> &[f32] {
        self.data.as_slice().expect("contiguous mel buffer")
    }
}

fn run_mel(config: &MelConfig, waveform: &[f32]) -> MelOutput {
    let mel = MelSpectrogram::new(config);
    let n_mels = mel.n_mels();
    let n_frames = mel.num_frames(waveform.len());
    let mut data = ndarray::Array3::<f32>::zeros((1, n_mels, n_frames));
    let mut view = data.view_mut().into_dyn();
    mel.forward_into(waveform, &mut view);
    MelOutput { data }
}

#[test]
fn test_mel_spectrogram_shape_center_true() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 400, hop_length: 160, win_length: 400, n_mels: 64, center: true };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    assert_eq!(output.shape(), (1, 64, 101));
}

#[test]
fn test_mel_spectrogram_shape_center_false() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 320, hop_length: 160, win_length: 320, n_mels: 64, center: false };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    assert_eq!(output.shape(), (1, 64, 99));
}

#[test]
fn test_mel_spectrogram_values_finite() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 400, hop_length: 160, win_length: 400, n_mels: 64, center: true };

    let waveform: Vec<f32> = vec![0.0; 1600];
    let output = run_mel(&config, &waveform);

    for v in output.as_slice() {
        assert!(v.is_finite(), "mel output contains non-finite value: {v}");
    }
}

#[test]
fn test_mel_spectrogram_sine_wave() {
    let config =
        MelConfig { sample_rate: 16000, n_fft: 400, hop_length: 160, win_length: 400, n_mels: 64, center: true };

    let waveform: Vec<f32> =
        (0..16000).map(|i| (i as f32 * 440.0 * 2.0 * std::f32::consts::PI / 16000.0).sin()).collect();
    let output = run_mel(&config, &waveform);

    let vals = output.as_slice();
    let (_, n_mels, n_frames) = output.shape();

    let mut avg_energy: Vec<f32> = vec![0.0; n_mels];
    for mel_idx in 0..n_mels {
        for frame in 0..n_frames {
            avg_energy[mel_idx] += vals[mel_idx * n_frames + frame];
        }
        avg_energy[mel_idx] /= n_frames as f32;
    }

    let lower_avg: f32 = avg_energy[..20].iter().sum::<f32>() / 20.0;
    let upper_avg: f32 = avg_energy[40..].iter().sum::<f32>() / 24.0;
    assert!(
        lower_avg > upper_avg,
        "Expected lower mel bins to have more energy for 440Hz sine: lower={lower_avg:.2}, upper={upper_avg:.2}"
    );
}

#[test]
fn test_hann_window_matches_torch_periodic_default() {
    let window = hann_window(8, 8);
    let expected = [0.0, 0.14644662, 0.5, 0.8535534, 1.0, 0.8535533, 0.5, 0.1464465];

    for (got, want) in window.iter().zip(expected) {
        assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
    }
}
