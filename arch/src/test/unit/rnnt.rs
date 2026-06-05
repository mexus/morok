use std::cmp::Ordering;
use std::convert::Infallible;

use proptest::prelude::*;

use crate::rnnt::{JointStep, RnntDecoder, RnntOpts, TokenEmission, Word};

/// NaN-safe argmax — the device backend computes the joint argmax on-GPU and
/// returns the index; the mock mirrors that over its scripted logits. NaN is
/// treated as smaller than any number (first non-NaN wins; all-NaN → 0).
fn argmax_nan_safe(frame: &[f32]) -> usize {
    let cmp = |a: f32, b: f32| {
        a.partial_cmp(&b).unwrap_or_else(|| match (a.is_nan(), b.is_nan()) {
            (true, false) => Ordering::Less,
            (false, true) => Ordering::Greater,
            _ => Ordering::Equal,
        })
    };
    let mut best = 0usize;
    for i in 1..frame.len() {
        if cmp(frame[i], frame[best]).is_gt() {
            best = i;
        }
    }
    best
}

// ─── Mock backend ─────────────────────────────────────────────────────────

/// Returns a scripted sequence of logits frames, one per `step` call. The
/// search loop advances through the script in order.
///
/// `committed_token` records the `prev_token` parameter at each step. `commit`
/// increments `commits`; `reset` increments `resets`. The mock exposes enough
/// observability to assert every contract on `JointStep` from the search loop.
struct MockJointStep {
    /// Logits for each `step` call, in order.
    script: Vec<Vec<f32>>,
    /// Cursor into `script`.
    cursor: usize,
    /// `(encoder_frame_first_value, prev_token)` recorded at each `step`.
    /// Stores only the first encoder element so equality checks stay cheap.
    pub seen: Vec<(f32, Option<usize>)>,
    /// Mirror of the search loop's notion of committed prev_token. After
    /// `commit`, this latches whatever `prev_token` would be next: the
    /// search-loop observable `prev_token` for the *next* step.
    pub commits: usize,
    pub resets: usize,
}

impl MockJointStep {
    fn new(script: Vec<Vec<f32>>) -> Self {
        Self { script, cursor: 0, seen: Vec::new(), commits: 0, resets: 0 }
    }
}

impl JointStep for MockJointStep {
    type Error = Infallible;

    fn step(&mut self, encoder_frame: &[f32], prev_token: Option<usize>) -> Result<usize, Self::Error> {
        let frame_first = *encoder_frame.first().unwrap_or(&0.0);
        self.seen.push((frame_first, prev_token));
        let idx = self.cursor.min(self.script.len() - 1);
        self.cursor += 1;
        Ok(argmax_nan_safe(&self.script[idx]))
    }

    fn commit(&mut self) {
        self.commits += 1;
    }

    fn reset(&mut self) {
        self.resets += 1;
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn abc_vocab() -> Vec<String> {
    vec!["a".into(), "b".into(), "c".into()]
}

fn one_hot_logits(winner: usize, total_vocab: usize) -> Vec<f32> {
    let mut v = vec![-10.0f32; total_vocab];
    v[winner] = 0.0;
    v
}

/// Build encoder frames where frame `t` is filled with value `t as f32`.
/// `enc_hidden = 1` keeps tests focused on the search loop.
fn linspace_encoder(n_frames: usize) -> Vec<f32> {
    (0..n_frames).map(|t| t as f32).collect()
}

fn decoder_with(vocab: Vec<String>, max_symbols: usize) -> RnntDecoder {
    RnntDecoder::new(vocab, RnntOpts { max_symbols_per_step: max_symbols })
}

// ─── Unit tests ───────────────────────────────────────────────────────────

#[test]
fn test_rnnt_blank_only_emits_empty() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Every frame: blank wins on the first inner call → outer loop advances.
    let mut mock = MockJointStep::new(vec![one_hot_logits(blank, total); 5]);

    let enc = linspace_encoder(5);
    let text = decoder.decode(&enc, 5, 5, 1, &mut mock).unwrap();
    assert_eq!(text, "");
    assert_eq!(mock.commits, 0, "blank-only decode must not commit");
    assert_eq!(mock.resets, 1, "exactly one reset per decode");
    assert_eq!(mock.seen.len(), 5, "one step per frame");
    // prev_token stays None throughout (no commits).
    for (_, prev) in &mock.seen {
        assert_eq!(*prev, None);
    }
}

#[test]
fn test_rnnt_one_emit_per_frame() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Per frame: one non-blank, then blank → emit once and advance.
    let script = vec![
        one_hot_logits(0, total), // emit 'a'
        one_hot_logits(blank, total),
        one_hot_logits(1, total), // emit 'b'
        one_hot_logits(blank, total),
        one_hot_logits(2, total), // emit 'c'
        one_hot_logits(blank, total),
    ];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(3);
    let text = decoder.decode(&enc, 3, 3, 1, &mut mock).unwrap();
    assert_eq!(text, "abc");
    assert_eq!(mock.commits, 3);
    assert_eq!(mock.resets, 1);
    // First step is empty-prefix (None); subsequent steps see the previously
    // emitted token.
    assert_eq!(mock.seen[0].1, None);
    assert_eq!(mock.seen[1].1, Some(0)); // prev_token after emitting 'a'
    assert_eq!(mock.seen[2].1, Some(0)); // first inner call of frame 1
    assert_eq!(mock.seen[3].1, Some(1));
    assert_eq!(mock.seen[4].1, Some(1));
    assert_eq!(mock.seen[5].1, Some(2));
}

#[test]
fn test_rnnt_multi_emit_per_frame() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Frame 0: emit 'a', 'b', then blank → moves on. Frame 1: blank only.
    let script = vec![
        one_hot_logits(0, total),
        one_hot_logits(1, total),
        one_hot_logits(blank, total),
        one_hot_logits(blank, total),
    ];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(2);
    let text = decoder.decode(&enc, 2, 2, 1, &mut mock).unwrap();
    assert_eq!(text, "ab");
    assert_eq!(mock.commits, 2);
}

#[test]
fn test_rnnt_max_symbols_caps_inner_loop() {
    let decoder = decoder_with(abc_vocab(), 3);
    let total = decoder.total_vocab();
    // Always emit 'a' — without max_symbols cap this would loop forever.
    // Single frame, so max_symbols_per_step caps to 3 emissions.
    let mut mock = MockJointStep::new(vec![one_hot_logits(0, total); 32]);

    let enc = linspace_encoder(1);
    let text = decoder.decode(&enc, 1, 1, 1, &mut mock).unwrap();
    assert_eq!(text, "aaa", "max_symbols_per_step must cap emissions");
    assert_eq!(mock.commits, 3);
    assert_eq!(mock.seen.len(), 3, "no extra step calls past the cap");
}

#[test]
fn test_rnnt_nan_logits_fall_through_argmax() {
    let decoder = decoder_with(abc_vocab(), 10);
    let total = decoder.total_vocab();
    // Frame 0: NaN-poisoned frame. argmax_nan_safe treats NaN as
    // smaller-than-anything, so the first non-NaN wins. With logits
    // [NaN, NaN, NaN, NaN] argmax is index 0 — emits 'a'. We then schedule a
    // blank to escape the inner loop.
    let blank = decoder.blank_id();
    let script = vec![vec![f32::NAN; total], one_hot_logits(blank, total)];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(1);
    let text = decoder.decode(&enc, 1, 1, 1, &mut mock).unwrap();
    assert_eq!(text, "a");
}

#[test]
fn test_rnnt_empty_vocab() {
    let decoder = decoder_with(Vec::new(), 10);
    let mut mock = MockJointStep::new(vec![vec![1.0; 1]]);
    let text = decoder.decode(&[], 0, 0, 1, &mut mock).unwrap();
    assert_eq!(text, "");
    assert_eq!(mock.resets, 0, "no reset for empty vocab — early-exit before loop");
    assert_eq!(mock.seen.len(), 0);
}

#[test]
fn test_rnnt_zero_valid_frames() {
    let decoder = decoder_with(abc_vocab(), 10);
    let mut mock = MockJointStep::new(vec![one_hot_logits(0, 4)]);
    let enc = linspace_encoder(3);
    let text = decoder.decode(&enc, 3, 0, 1, &mut mock).unwrap();
    assert_eq!(text, "");
    assert_eq!(mock.seen.len(), 0);
}

#[test]
fn test_rnnt_stride_clamps_valid_frames() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // valid_frames > stride: only `stride` frames are consumed.
    let mut mock = MockJointStep::new(vec![one_hot_logits(blank, total); 2]);
    let enc = linspace_encoder(2);
    let _ = decoder.decode(&enc, 2, 5, 1, &mut mock).unwrap();
    assert_eq!(mock.seen.len(), 2);
}

#[test]
fn test_rnnt_with_timestamps() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    // Frame 0: emit 'a', blank. Frame 1: blank. Frame 2: emit 'b', 'c', blank.
    let script = vec![
        one_hot_logits(0, total),
        one_hot_logits(blank, total),
        one_hot_logits(blank, total),
        one_hot_logits(1, total),
        one_hot_logits(2, total),
        one_hot_logits(blank, total),
    ];
    let mut mock = MockJointStep::new(script);

    let enc = linspace_encoder(3);
    let (text, emissions) = decoder.decode_with_timestamps(&enc, 3, 3, 1, &mut mock).unwrap();
    assert_eq!(text, "abc");
    let pairs: Vec<(usize, usize)> = emissions.iter().map(|e| (e.token_id, e.frame)).collect();
    assert_eq!(pairs, vec![(0, 0), (1, 2), (2, 2)]);
}

#[test]
fn test_rnnt_step_receives_correct_encoder_frame() {
    let decoder = decoder_with(abc_vocab(), 10);
    let blank = decoder.blank_id();
    let total = decoder.total_vocab();
    let mut mock = MockJointStep::new(vec![one_hot_logits(blank, total); 4]);

    // Per-frame encoder values: 10, 20, 30, 40.
    let enc = vec![10.0f32, 20.0, 30.0, 40.0];
    let _ = decoder.decode(&enc, 4, 4, 1, &mut mock).unwrap();
    let frame_values: Vec<f32> = mock.seen.iter().map(|(v, _)| *v).collect();
    assert_eq!(frame_values, vec![10.0, 20.0, 30.0, 40.0]);
}

#[test]
fn test_rnnt_backend_error_surfaces_with_frame_index() {
    use snafu::Snafu;

    #[derive(Debug, Snafu)]
    #[snafu(display("boom"))]
    struct Boom;

    struct FailingStep;
    impl JointStep for FailingStep {
        type Error = Boom;
        fn step(&mut self, _: &[f32], _: Option<usize>) -> Result<usize, Self::Error> {
            Err(Boom)
        }
        fn commit(&mut self) {}
        fn reset(&mut self) {}
    }

    let decoder = decoder_with(abc_vocab(), 10);
    let enc = linspace_encoder(3);
    let mut step = FailingStep;
    let err = decoder.decode(&enc, 3, 3, 1, &mut step).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("frame 0"), "error message preserves frame index, got: {msg}");
}

#[cfg(feature = "serde")]
#[test]
fn test_rnnt_opts_serde_default_roundtrip() {
    let opts: RnntOpts = serde_json::from_str("{}").unwrap();
    assert_eq!(opts.max_symbols_per_step, 10);

    let opts: RnntOpts = serde_json::from_str(r#"{"max_symbols_per_step": 5}"#).unwrap();
    assert_eq!(opts.max_symbols_per_step, 5);
}

// ─── frames_to_words ──────────────────────────────────────────────────────

fn em(token_id: usize, frame: usize) -> TokenEmission {
    TokenEmission { token_id, frame }
}

fn assert_words(got: &[Word], expected: &[(&str, f32, f32)]) {
    assert_eq!(got.len(), expected.len(), "word count mismatch: got {got:?}, expected {expected:?}");
    for (g, (text, start, end)) in got.iter().zip(expected.iter()) {
        assert_eq!(&g.text, text);
        assert!((g.start - start).abs() < 1e-5, "start: got {}, expected {}", g.start, start);
        assert!((g.end - end).abs() < 1e-5, "end: got {}, expected {}", g.end, end);
    }
}

#[test]
fn test_frames_to_words_sentencepiece_boundaries() {
    let decoder = decoder_with(vec!["\u{2581}hello".into(), "\u{2581}world".into(), "!".into()], 10);
    let emissions = vec![em(0, 0), em(1, 10), em(2, 12)];
    let words = decoder.frames_to_words(&emissions, 0.04);
    assert_words(&words, &[("hello", 0.0, 0.04), ("world!", 0.40, 0.52)]);
}

#[test]
fn test_frames_to_words_multi_piece_word() {
    let decoder = decoder_with(vec!["\u{2581}при".into(), "ве".into(), "т".into()], 10);
    let emissions = vec![em(0, 5), em(1, 6), em(2, 7)];
    let words = decoder.frames_to_words(&emissions, 0.04);
    assert_words(&words, &[("привет", 0.20, 0.32)]);
}

#[test]
fn test_frames_to_words_empty_emissions() {
    let decoder = decoder_with(vec!["\u{2581}x".into()], 10);
    assert!(decoder.frames_to_words(&[], 0.04).is_empty());
}

#[test]
fn test_frames_to_words_literal_space_separator() {
    let decoder = decoder_with(vec!["hello".into(), " ".into(), "world".into()], 10);
    let emissions = vec![em(0, 0), em(1, 2), em(2, 3)];
    let words = decoder.frames_to_words(&emissions, 0.1);
    assert_words(&words, &[("hello", 0.0, 0.1), ("world", 0.3, 0.4)]);
}

// ─── Property tests ───────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    /// Output token count never exceeds `valid_frames * max_symbols_per_step`.
    /// The cap is the only termination guarantee for adversarial backends.
    #[test]
    fn prop_rnnt_output_length_bounded(
        valid_frames in 0usize..40,
        max_symbols in 1usize..6,
    ) {
        let decoder = decoder_with(abc_vocab(), max_symbols);
        let total = decoder.total_vocab();
        // Pathological backend: always emits 'a' (never blank) — forces the
        // cap to fire on every frame.
        let script_len = valid_frames * max_symbols + 1;
        let mut mock = MockJointStep::new(vec![one_hot_logits(0, total); script_len.max(1)]);
        let enc = linspace_encoder(valid_frames.max(1));
        let text = decoder.decode(&enc, valid_frames, valid_frames, 1, &mut mock).unwrap();
        prop_assert!(text.chars().count() <= valid_frames * max_symbols);
    }

    /// Number of `commit` calls equals output token count. The search loop
    /// must commit exactly once per non-blank emission, never on blank.
    #[test]
    fn prop_rnnt_commit_count_matches_emissions(
        n_frames in 1usize..15,
        seed in any::<u64>(),
    ) {
        let decoder = decoder_with(abc_vocab(), 10);
        let blank = decoder.blank_id();
        let total = decoder.total_vocab();
        // Pseudo-random script of (token | blank) entries, large enough that
        // the search loop always finds a blank within max_symbols.
        let mut state = seed;
        let mut script = Vec::with_capacity(n_frames * 12);
        for _ in 0..(n_frames * 12) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // 60% blank (so inner loop terminates promptly), 40% non-blank.
            let pick = if (state >> 32) % 5 < 3 { blank } else { ((state >> 32) % 3) as usize };
            script.push(one_hot_logits(pick, total));
        }
        let mut mock = MockJointStep::new(script);
        let enc = linspace_encoder(n_frames);
        let text = decoder.decode(&enc, n_frames, n_frames, 1, &mut mock).unwrap();
        prop_assert_eq!(text.chars().count(), mock.commits);
    }

    /// Timestamps are non-decreasing and each frame index is in `[0, valid_frames)`.
    #[test]
    fn prop_rnnt_timestamps_monotonic_and_in_bounds(
        n_frames in 1usize..15,
        seed in any::<u64>(),
    ) {
        let decoder = decoder_with(abc_vocab(), 10);
        let blank = decoder.blank_id();
        let total = decoder.total_vocab();
        let mut state = seed;
        let mut script = Vec::with_capacity(n_frames * 12);
        for _ in 0..(n_frames * 12) {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let pick = if (state >> 32) % 5 < 3 { blank } else { ((state >> 32) % 3) as usize };
            script.push(one_hot_logits(pick, total));
        }
        let mut mock = MockJointStep::new(script);
        let enc = linspace_encoder(n_frames);
        let (_, emissions) = decoder.decode_with_timestamps(&enc, n_frames, n_frames, 1, &mut mock).unwrap();
        for w in emissions.windows(2) {
            prop_assert!(w[0].frame <= w[1].frame);
        }
        for e in &emissions {
            prop_assert!(e.frame < n_frames);
        }
    }
}
