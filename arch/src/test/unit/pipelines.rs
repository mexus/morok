use std::convert::Infallible;

use crate::pipelines::audio::{
    Asr, ChunkResult, FixedLengthSplitter, RunProfile, Splitter, Transcriber, Transcript, Vad, VadSplitter,
    crop_words_to_core, words_to_text,
};
use crate::rnnt::Word;
use crate::vad::{AudioChunk, ChunkerOpts};

fn word(text: &str, start: f32, end: f32) -> Word {
    Word { text: text.to_string(), start, end }
}

// ─── Mocks ───────────────────────────────────────────────────────────────────

/// VAD that ignores the audio and returns a fixed probability vector.
struct MockVad {
    probs: Vec<f32>,
    samples_per_prob: usize,
}

impl Vad for MockVad {
    type Error = Infallible;
    fn samples_per_prob(&self) -> usize {
        self.samples_per_prob
    }
    fn probs(&mut self, _waveform: &[f32]) -> Result<Vec<f32>, Infallible> {
        Ok(self.probs.clone())
    }
}

/// Transcriber that returns a preset transcript per window (1 sample = 1 s).
struct PresetTranscriber {
    out: Vec<Transcript>,
    want_words: bool,
}

impl Transcriber for PresetTranscriber {
    type Error = Infallible;
    fn sample_rate(&self) -> u32 {
        1
    }
    fn wants_words(&self) -> bool {
        self.want_words
    }
    fn transcribe_windows(&mut self, windows: &[&[f32]]) -> Result<(Vec<Transcript>, Option<RunProfile>), Infallible> {
        assert_eq!(windows.len(), self.out.len(), "preset length must match window count");
        Ok((self.out.clone(), None))
    }
}

fn fast_chunker_opts() -> ChunkerOpts {
    ChunkerOpts {
        sample_rate: 1,
        samples_per_prob: 1,
        threshold: 0.5,
        min_duration: 1.0,
        max_duration: 10.0,
        strict_limit_duration: 15.0,
        min_speech_probs: 1,
        min_silence_probs: 1,
        merge_gap_probs: 0,
        trough_search_probs: None,
        trough_threshold: None,
        pad_samples: 0,
        preroll_samples: 0,
        align_to: 1,
        max_total_samples: None,
    }
}

// ─── crop / stitch ────────────────────────────────────────────────────────────

#[test]
fn crop_keeps_midpoints_inside_core_and_rebases() {
    // core starts 2s into the window, spans 5s (window time [2, 7)).
    let words = vec![word("a", 0.5, 1.5), word("b", 3.5, 4.5), word("c", 7.5, 8.5)];
    let cropped = crop_words_to_core(words, 2.0, 5.0);
    assert_eq!(cropped, vec![word("b", 1.5, 2.5)]);
}

#[test]
fn words_to_text_joins_and_drops_empties() {
    let words = vec![word("hello", 0.0, 0.1), word("", 0.1, 0.1), word("world", 0.2, 0.3)];
    assert_eq!(words_to_text(&words), "hello world");
    assert_eq!(words_to_text(&[]), "");
}

// ─── Transcriber::transcribe_chunks default (geometry + crop + stitch) ─────────

#[test]
fn transcribe_chunks_slices_decode_windows_crops_and_stitches() {
    let waveform = vec![0.0_f32; 50];
    let chunks = vec![AudioChunk::with_decode(10, 20, 5, 25), AudioChunk::with_decode(30, 40, 28, 42)];
    // Window A: only "a1" (mid 7, in core [5,15)) survives; "pre"/"post" are pad.
    // Window B: "b1" (mid 4, in core [2,12)) survives.
    let mut asr = PresetTranscriber {
        want_words: true,
        out: vec![
            Transcript {
                text: String::new(),
                words: vec![word("pre", 1.0, 3.0), word("a1", 6.0, 8.0), word("post", 16.0, 18.0)],
            },
            Transcript { text: String::new(), words: vec![word("b1", 3.0, 5.0)] },
        ],
    };
    let out = asr.transcribe_chunks(&waveform, &chunks).unwrap();
    assert_eq!(out.text, "a1 b1");
    assert_eq!(
        out.chunks,
        vec![
            ChunkResult {
                start_sec: 10.0,
                end_sec: 20.0,
                text: "a1".to_string(),
                words: Some(vec![word("a1", 1.0, 3.0)])
            },
            ChunkResult {
                start_sec: 30.0,
                end_sec: 40.0,
                text: "b1".to_string(),
                words: Some(vec![word("b1", 1.0, 3.0)])
            },
        ]
    );
    assert!(out.profile.is_none());
}

#[test]
fn transcribe_chunks_omits_words_when_not_wanted() {
    let waveform = vec![0.0_f32; 20];
    let chunks = vec![AudioChunk::new(0, 10)];
    let mut asr = PresetTranscriber {
        want_words: false,
        out: vec![Transcript { text: String::new(), words: vec![word("hi", 1.0, 2.0)] }],
    };
    let out = asr.transcribe_chunks(&waveform, &chunks).unwrap();
    assert_eq!(out.text, "hi");
    assert_eq!(out.chunks[0].words, None);
}

// ─── Splitters ────────────────────────────────────────────────────────────────

#[test]
fn vad_splitter_runs_probs_then_chunks() {
    let mut splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    let chunks = splitter.split(&[0.0_f32; 4]).unwrap();
    assert_eq!(chunks, vec![AudioChunk::new(0, 4)]);
}

#[test]
fn fixed_length_splitter_strides_aligned_with_unaligned_tail() {
    let mut splitter = FixedLengthSplitter::new(10, 4);
    let chunks = splitter.split(&[0.0_f32; 26]).unwrap();
    // window 10 floored to align 4 → 8; final tail keeps its remainder.
    assert_eq!(chunks, vec![AudioChunk::new(0, 8), AudioChunk::new(8, 16), AudioChunk::new(16, 26)]);
}

#[test]
fn fixed_length_splitter_empty_waveform() {
    let mut splitter = FixedLengthSplitter::new(10, 1);
    assert!(splitter.split(&[]).unwrap().is_empty());
}

#[test]
fn vad_splitter_records_profile_and_max_chunk_bound() {
    let mut splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    // strict_chunk_sample_bound((15+2)*1 + 0 + 1) = 18 from fast_chunker_opts.
    assert_eq!(splitter.max_chunk_samples(), 18);
    assert!(splitter.take_profile().is_none(), "no split yet");
    splitter.split(&[0.0_f32; 4]).unwrap();
    let profile = splitter.take_profile().expect("split records a vad stage");
    assert_eq!(profile.stages.len(), 1);
    assert!(profile.stage("vad").is_some());
    assert!(splitter.take_profile().is_none(), "take clears the stash");
}

// ─── Asr composer end-to-end ──────────────────────────────────────────────────

#[test]
fn asr_composes_split_and_transcribe() {
    // 3 fixed windows → 3 preset transcripts → stitched.
    let splitter = FixedLengthSplitter::new(10, 1);
    let transcriber = PresetTranscriber {
        want_words: false,
        out: vec![
            Transcript { text: String::new(), words: vec![word("one", 1.0, 2.0)] },
            Transcript { text: String::new(), words: vec![word("two", 1.0, 2.0)] },
            Transcript { text: String::new(), words: vec![word("three", 1.0, 2.0)] },
        ],
    };
    let mut asr = Asr::new(splitter, transcriber);
    let out = asr.transcribe(&[0.0_f32; 25]).unwrap();
    assert_eq!(out.text, "one two three");
}

// ─── Profile merge (default transcribe_windows over a single-window model) ─────

/// Single-window transcriber emitting a 1 ms `decode` stage per window — the
/// default `transcribe_windows` must merge these, not drop them.
struct ProfilingTranscriber;

impl Transcriber for ProfilingTranscriber {
    type Error = Infallible;
    fn sample_rate(&self) -> u32 {
        1
    }
    fn wants_words(&self) -> bool {
        false
    }
    fn transcribe_window(&mut self, _window: &[f32]) -> Result<(Transcript, Option<RunProfile>), Infallible> {
        let mut profile = RunProfile::default();
        profile.push(svod_runtime::StageProfile::host("decode", std::time::Duration::from_millis(1)));
        Ok((Transcript { text: "x".to_string(), words: Vec::new() }, Some(profile)))
    }
}

#[test]
fn transcribe_windows_default_merges_per_window_profiles() {
    // Two windows, no batch override → the default loops transcribe_window and
    // merges both `decode` stages into one with summed wall (1 ms × 2).
    let waveform = vec![0.0_f32; 20];
    let chunks = vec![AudioChunk::new(0, 10), AudioChunk::new(10, 20)];
    let out = ProfilingTranscriber.transcribe_chunks(&waveform, &chunks).unwrap();
    let profile = out.profile.expect("profile collected");
    assert_eq!(profile.stages.len(), 1, "stages merged by name");
    assert_eq!(profile.stage("decode").unwrap().wall, std::time::Duration::from_millis(2));
}

#[test]
fn asr_prepends_vad_stage_when_transcriber_profiles() {
    // VadSplitter records a `vad` stage; the profiling transcriber a `decode`
    // one. Asr surfaces both with VAD leading.
    let splitter = VadSplitter::new(MockVad { probs: vec![1.0; 4], samples_per_prob: 1 }, fast_chunker_opts());
    let mut asr = Asr::new(splitter, ProfilingTranscriber);
    let out = asr.transcribe(&[0.0_f32; 4]).unwrap();
    let profile = out.profile.expect("profile");
    let stages: Vec<&str> = profile.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(stages, vec!["vad", "decode"], "VAD stage leads the merged profile");
}
