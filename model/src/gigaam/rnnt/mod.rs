//! RNN-T (transducer) head for GigaAM.
//!
//! Layout mirrors the reference Python `RNNTDecoder` / `RNNTJoint` / `RNNTHead`
//! in `submodules/GigaAM/gigaam/decoder.py`. The predictor is a multi-layer
//! LSTM stack of [`morok_tensor::nn::LSTMCell`]s (PyTorch `[i, f, g, o]` gate
//! order, matching the Silero VAD predictor); the joint is a two-Linear sum
//! + ReLU + Linear + log-softmax projection.
//!
//! Submodules:
//! - [`predictor`] — `RnntPredictor` (token embed + multi-layer LSTM).
//! - [`joint`] — `RnntJoint` (sum projection + ReLU + linear).
//! - [`head`] — `RnntHead = predictor + joint`.
//! - [`tokenizer`] — SentencePiece protobuf loader (private).
//! - [`jit`] — `RnntPredictorStepJit`, `RnntJointStepJit` (the encoder JIT
//!   is shared and lives in [`crate::gigaam::jit`]).
//! - [`backend`] — `RnntStepBackend` (impl `morok_arch::rnnt::JointStep`).
//!
//! The model wrapper itself lives in [`crate::gigaam::model`]; the RN-T
//! runtime metadata (vocabulary, max-symbols-per-step, SP flag) is carried
//! inside the `Head::Rnnt` variant.

mod backend;
mod head;
mod jit;
mod joint;
mod predictor;
mod tokenizer;

pub(crate) use tokenizer::load_sentencepiece_vocab;

pub use backend::*;
pub use head::*;
pub use jit::*;
pub use joint::*;
pub use predictor::*;
