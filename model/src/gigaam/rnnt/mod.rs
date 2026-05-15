//! RNN-T (transducer) head for GigaAM. Layout mirrors the reference Python
//! `RNNTDecoder` / `RNNTJoint` / `RNNTHead` in
//! `submodules/GigaAM/gigaam/decoder.py`: multi-layer LSTM predictor +
//! two-Linear-sum + ReLU + Linear + log-softmax joint. LSTM gate order is
//! PyTorch's `[i, f, g, o]` so checkpoints load without remapping.

pub(crate) mod backend;
pub(crate) mod head;
pub(crate) mod jit;
pub(crate) mod joint;
pub(crate) mod predictor;

pub(crate) use backend::RnntStepBackend;
pub(crate) use head::RnntHead;
