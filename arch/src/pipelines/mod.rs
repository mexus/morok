//! Composable inference pipelines (host-side orchestration, GPU-free).
//!
//! [`audio`] is the long-form speech-to-text pipeline: VAD segmentation →
//! per-window transcription → core-crop → stitch, with the heavy machinery in
//! trait defaults so a model only implements its irreducible part. Future
//! pipeline families (e.g. speaker diarization) get sibling sub-modules here.

pub mod audio;
