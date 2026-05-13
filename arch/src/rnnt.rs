//! RNN-T (transducer) greedy decoder.
//!
//! The decoder owns the search loop and the vocabulary. The predictor + joint
//! networks live behind the [`JointStep`] trait, which the model crate
//! implements with JIT calls. This crate stays Tensor-free and operates only
//! on plain `&[f32]` slices.
//!
//! Algorithm (B=1) mirrors `gigaam/decoding.py:139–207`:
//! ```text
//! step.reset()
//! prev_token = None
//! for t in 0..valid_frames:
//!   enc_t = encoder_frames[t * enc_hidden ..][..enc_hidden]
//!   for _ in 0..max_symbols_per_step:
//!     step.step(enc_t, prev_token, &mut logits_buf)?
//!     k = argmax_nan_safe(logits_buf)
//!     if k == blank_id { break }
//!     emit token k; prev_token = Some(k); step.commit()
//! ```
//!
//! The "tentative / committed" predictor-state semantics on [`JointStep`] keep
//! the trait stateless from the search loop's perspective: the search loop
//! only knows about `prev_token` and `logits_out` and never sees the LSTM
//! hidden state. The implementation owns and rolls its own state.
//!
//! # Layout convention
//!
//! `encoder_frames` is row-major `[stride_frames, enc_hidden]` for a single
//! batch item. `valid_frames` clamps padding from JIT static shapes — only
//! frames `0..valid_frames` are consumed.

use std::cmp::Ordering;

use snafu::{IntoError, Snafu};

#[cfg(feature = "serde")]
use serde::Deserialize;

// ─── Errors ───────────────────────────────────────────────────────────────

/// Failure modes for [`RnntDecoder`]. The only failure surface is the user's
/// [`JointStep`] backend; anything else is a usage bug.
#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum RnntDecodeError<E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    #[snafu(display("predictor/joint backend failed at frame {frame}"))]
    Backend { source: E, frame: usize },
}

// ─── Options ──────────────────────────────────────────────────────────────

/// Tunables for [`RnntDecoder`]. Default `max_symbols_per_step = 10` matches
/// `gigaam/decoding.py:108`.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
#[cfg_attr(feature = "serde", serde(default))]
pub struct RnntOpts {
    /// Maximum non-blank tokens emitted per encoder frame. Caps a runaway
    /// inner loop if the joint never emits blank.
    pub max_symbols_per_step: usize,
}

impl Default for RnntOpts {
    fn default() -> Self {
        Self { max_symbols_per_step: 10 }
    }
}

// ─── Token emissions / words ──────────────────────────────────────────────

/// One non-blank token emitted by the greedy decoder. `frame` is the
/// encoder-frame index (`t` in the outer search loop) at which the joint
/// network selected this token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenEmission {
    pub token_id: usize,
    pub frame: usize,
}

/// A grouped word and its `[start, end)` time span in seconds. Produced by
/// [`frames_to_words`]. Mirrors upstream GigaAM's `Word` dataclass in
/// `gigaam/types.py`.
#[derive(Clone, Debug, PartialEq)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

struct PendingWord {
    text: String,
    first_frame: usize,
    last_frame: usize,
}

fn flush_pending(pending: &mut Option<PendingWord>, frame_shift: f32, words: &mut Vec<Word>) {
    let Some(p) = pending.take() else { return };
    let trimmed = p.text.trim();
    if !trimmed.is_empty() {
        words.push(Word {
            text: trimmed.to_string(),
            start: p.first_frame as f32 * frame_shift,
            end: (p.last_frame + 1) as f32 * frame_shift,
        });
    }
}

// ─── JointStep trait ──────────────────────────────────────────────────────

/// Bridge between the search loop and the predictor/joint backend.
///
/// The implementation owns the per-utterance LSTM state. The search loop
/// never sees it — it only drives `step` (compute logits), `commit` (promote
/// the last step's predictor state to the new committed state, on non-blank
/// emission) and `reset` (clear committed state at the start of an
/// utterance).
///
/// On a blank emission, the search loop simply does not call `commit`, and
/// the next `step` call's predictor input is the same `prev_token` as before
/// the blank. The implementation must therefore make `step` idempotent w.r.t.
/// committed state: each call reads from committed state and writes only to
/// tentative state.
pub trait JointStep {
    /// Backend-specific error (typically a JIT execution error).
    type Error: std::error::Error + Send + Sync + 'static;

    /// Compute joint logits for `(encoder_frame, prev_token)` against the
    /// LATEST COMMITTED predictor state. `prev_token = None` selects the
    /// empty-prefix initial state. Writes `total_vocab` logits into
    /// `logits_out`. The post-step predictor state is stashed internally as
    /// "tentative" and is only used if [`commit`](Self::commit) is called.
    fn step(
        &mut self,
        encoder_frame: &[f32],
        prev_token: Option<usize>,
        logits_out: &mut [f32],
    ) -> Result<(), Self::Error>;

    /// Promote the last `step`'s tentative state to committed. The arch
    /// decoder calls this exactly once per non-blank emission.
    fn commit(&mut self);

    /// Reset committed state to the empty-prefix initial. Called once per
    /// utterance, before the first frame.
    fn reset(&mut self);
}

// ─── Decoder ──────────────────────────────────────────────────────────────

/// Greedy RNN-T decoder. Owns the vocabulary; the blank token id is implicit
/// at `vocabulary.len()`.
///
/// Decode is `&self` — all per-utterance mutable state lives in the
/// [`JointStep`] backend the caller supplies.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Deserialize))]
pub struct RnntDecoder {
    vocabulary: Vec<String>,
    #[cfg_attr(feature = "serde", serde(default))]
    opts: RnntOpts,
}

impl RnntDecoder {
    pub fn new(vocabulary: Vec<String>, opts: RnntOpts) -> Self {
        Self { vocabulary, opts }
    }

    pub fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    pub fn opts(&self) -> &RnntOpts {
        &self.opts
    }

    pub fn blank_id(&self) -> usize {
        self.vocabulary.len()
    }

    pub fn total_vocab(&self) -> usize {
        self.vocabulary.len() + 1
    }

    /// Group token emissions from [`decode_with_timestamps`](Self::decode_with_timestamps)
    /// into words and assign each a `[start, end)` time span in seconds.
    /// A new word begins on:
    ///
    /// - a piece prefixed with `▁` (U+2581, SentencePiece word marker), or
    /// - a piece equal to a single ASCII space (char-based vocabularies).
    ///
    /// Other pieces concatenate onto the current word. `start = first_frame *
    /// frame_shift`; `end = (last_frame + 1) * frame_shift`. Whitespace-only
    /// candidates are dropped.
    ///
    /// Port of `gigaam/timestamps_utils.py:frames_to_words`.
    pub fn frames_to_words(&self, emissions: &[TokenEmission], frame_shift: f32) -> Vec<Word> {
        const SP_MARK: char = '\u{2581}';

        let mut words: Vec<Word> = Vec::new();
        let mut pending: Option<PendingWord> = None;

        for e in emissions {
            let piece = match self.vocabulary.get(e.token_id) {
                Some(p) => p.as_str(),
                None => continue,
            };
            if let Some(stripped) = piece.strip_prefix(SP_MARK) {
                flush_pending(&mut pending, frame_shift, &mut words);
                pending = Some(PendingWord { text: stripped.to_string(), first_frame: e.frame, last_frame: e.frame });
            } else if piece == " " {
                flush_pending(&mut pending, frame_shift, &mut words);
            } else {
                match &mut pending {
                    Some(p) => {
                        p.text.push_str(piece);
                        p.last_frame = e.frame;
                    }
                    None => {
                        pending =
                            Some(PendingWord { text: piece.to_string(), first_frame: e.frame, last_frame: e.frame });
                    }
                }
            }
        }
        flush_pending(&mut pending, frame_shift, &mut words);
        words
    }

    /// Greedy decode. `encoder_frames` is row-major `[stride_frames, enc_hidden]`.
    /// Empty vocabulary or `valid_frames == 0` yield an empty string.
    pub fn decode<S: JointStep>(
        &self,
        encoder_frames: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        enc_hidden: usize,
        step: &mut S,
    ) -> Result<String, RnntDecodeError<S::Error>> {
        let (text, _emissions) =
            self.decode_inner(encoder_frames, stride_frames, valid_frames, enc_hidden, step, false)?;
        Ok(text)
    }

    /// Greedy decode + per-emission `(token_id, frame)` pairs. `emissions[i]`
    /// records which token was emitted and at which encoder frame, in
    /// decoder output order. Pair with [`frames_to_words`] to recover
    /// word-level timestamps from a SentencePiece vocabulary.
    pub fn decode_with_timestamps<S: JointStep>(
        &self,
        encoder_frames: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        enc_hidden: usize,
        step: &mut S,
    ) -> Result<(String, Vec<TokenEmission>), RnntDecodeError<S::Error>> {
        self.decode_inner(encoder_frames, stride_frames, valid_frames, enc_hidden, step, true)
    }

    fn decode_inner<S: JointStep>(
        &self,
        encoder_frames: &[f32],
        stride_frames: usize,
        valid_frames: usize,
        enc_hidden: usize,
        step: &mut S,
        keep_emissions: bool,
    ) -> Result<(String, Vec<TokenEmission>), RnntDecodeError<S::Error>> {
        if self.vocabulary.is_empty() || valid_frames == 0 {
            return Ok((String::new(), Vec::new()));
        }
        let blank_id = self.blank_id();
        let total_vocab = self.total_vocab();
        let n_frames = stride_frames.min(valid_frames);

        step.reset();

        let mut text = String::new();
        let mut emissions = if keep_emissions { Vec::with_capacity(n_frames) } else { Vec::new() };
        let mut logits = vec![0.0f32; total_vocab];
        let mut prev_token: Option<usize> = None;

        for t in 0..n_frames {
            let base = t * enc_hidden;
            let enc_t = &encoder_frames[base..base + enc_hidden];

            for _ in 0..self.opts.max_symbols_per_step {
                step.step(enc_t, prev_token, &mut logits).map_err(|e| BackendSnafu { frame: t }.into_error(e))?;

                let k = argmax_nan_safe(&logits);
                if k == blank_id {
                    break;
                }
                // Non-blank: emit and commit. The commit advances the
                // backend's committed predictor state to the post-step
                // state we just produced.
                text.push_str(&self.vocabulary[k]);
                if keep_emissions {
                    emissions.push(TokenEmission { token_id: k, frame: t });
                }
                prev_token = Some(k);
                step.commit();
            }
        }

        Ok((text, emissions))
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────

fn argmax_nan_safe(frame: &[f32]) -> usize {
    let mut best = 0usize;
    for i in 1..frame.len() {
        if compare_logits(frame[i], frame[best]).is_gt() {
            best = i;
        }
    }
    best
}

fn compare_logits(a: f32, b: f32) -> Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Less,
        (false, true) => Ordering::Greater,
        (false, false) => Ordering::Equal,
    })
}
