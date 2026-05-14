//! Audio preprocessing for ASR: mel features ([`mel`]) and chunking
//! ([`splitter`]).
//!
//! Both submodules operate on raw `&[f32]` PCM, before any encoder JIT
//! involvement. They're grouped here because they share the same domain
//! (audio-level transforms over a waveform) and because the splitter +
//! mel extractor sit back-to-back in the inference pipeline.

pub(crate) mod mel;
mod splitter;

pub use mel::{MelConfig, MelSpectrogram};
pub use splitter::{AudioChunk, EncoderBounds, FixedLengthSplitter, Splitter};
