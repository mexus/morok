//! OpenAI Whisper: encoder-decoder transformer for speech recognition.
//!
//! Architecture: Conv-BN frontend → sinusoidal position embeddings → standard
//! pre-norm transformer encoder; learned-position decoder with self-attention +
//! cross-attention to encoder output.  Supports DTW word-level alignment via
//! cross-attention weight extraction.
//!
//! # Quick start
//!
//! ```no_run
//! use svod_model::whisper::{Whisper, WhisperSize, ModelDimensions};
//!
//! let dims = ModelDimensions::for_size(WhisperSize::Tiny);
//! let model = Whisper::empty(dims);
//! ```

pub mod attention;
pub mod blocks;
pub mod config;
pub mod decode;
pub mod decoder;
pub mod dtw;
pub mod encoder;
pub mod error;
pub mod jit;
pub mod mel;
pub mod model;
pub mod tokenizer;
pub mod transcribe;

mod loader;

pub use attention::{MultiHeadAttention, causal_mask};
pub use blocks::{Conv1dWeights, LayerNormWeights, LinearWeights, sinusoids};
pub use config::{ModelDimensions, WhisperSize};
pub use decode::{
    DecodeLane, DecodeOptions, DecodeResult, LanguageDetection, beam_decode_cached, decode_with_fallback_cached,
    detect_language, greedy_decode_cached, greedy_decode_with_alignment, run_batched_decode,
};
pub use decoder::{DecoderBlock, TextDecoder};
pub use dtw::{WordTiming, dtw, find_alignment_path, median_filter, path_to_word_timings};
pub use encoder::{AudioEncoder, EncoderBlock};
pub use error::{Error, Result};
pub use jit::{WhisperDecoderJit, WhisperDecoderStepBatchedJit, WhisperDecoderStepJit, WhisperEncoderJit, WhisperPrefillJit};
pub use mel::WhisperMel;
pub use model::Whisper;
pub use tokenizer::WhisperTokenizer;
pub use transcribe::{TranscribeError, WhisperTranscriber};

// Re-export audio constants
pub use config::{
    CHUNK_LENGTH, FRAMES_PER_SECOND, HOP_LENGTH, N_AUDIO_CTX, N_FFT, N_FRAMES, N_SAMPLES, N_SAMPLES_PER_TOKEN,
    N_TEXT_CTX, SAMPLE_RATE, TOKENS_PER_SECOND,
};
