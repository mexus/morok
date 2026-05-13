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
//! - [`model`] — `GigaAmRnnt`: encoder + head + vocab + loaders.
//! - [`tokenizer`] — SentencePiece protobuf loader (private).
//! - [`jit`] — `GigaAmRnntEncoderJit`, `RnntPredictorStepJit`, `RnntJointStepJit`.
//! - [`backend`] — `RnntStepBackend` (impl `morok_arch::rnnt::JointStep`).

mod backend;
mod head;
mod jit;
mod joint;
mod model;
mod predictor;
mod tokenizer;

pub use backend::*;
pub use head::*;
pub use jit::*;
pub use joint::*;
pub use model::*;
pub use predictor::*;
