//! Typed per-stage GPU profile for one [`Transcriber::transcribe`] call.
//!
//! [`TranscribeProfile`] carries one representative profiled execution per GPU
//! stage as raw [`KernelProfile`]s — queryable through
//! [`StageProfile::top`]/[`gpu_total`](StageProfile::gpu_total) — plus
//! host wall-clock for the host-only stages. `Display` renders the human
//! histograms the CLI logs.
//!
//! [`Transcriber::transcribe`]: super::Transcriber::transcribe

use std::time::Duration;

use svod_runtime::{KernelAggregate, KernelProfile, aggregate_profiles, render_histogram};

/// Pipeline stage owning a profiled JIT execution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// Fused encoder+head plan (CTC models).
    CtcHead,
    /// Standalone encoder plan (RN-T models).
    Encoder,
    /// One fused predictor+joint decode step (RN-T models).
    RnntStep,
}

/// One stage's profile: accumulated host wall across the whole call plus the
/// per-dispatch kernels of ONE representative profiled execution.
#[derive(Debug)]
pub struct StageProfile {
    pub stage: Stage,
    /// Host wall accumulated over every execution of the stage.
    pub wall: Duration,
    /// Per-dispatch kernels of the profiled execution (GPU-stamped on AMD).
    pub kernels: Vec<KernelProfile>,
}

impl StageProfile {
    /// Sum of GPU (or wall-fallback) time across the profiled execution.
    pub fn gpu_total(&self) -> Duration {
        self.kernels.iter().map(KernelProfile::gpu_or_wall).sum()
    }

    /// Top kernels by total time, aggregated by entry point.
    pub fn top(&self, n: usize) -> Vec<KernelAggregate> {
        let mut aggs = aggregate_profiles(&self.kernels);
        aggs.truncate(n);
        aggs
    }
}

/// Per-call profile: host-only stage walls + one [`StageProfile`] per GPU
/// stage that executed.
#[derive(Debug, Default)]
pub struct TranscribeProfile {
    /// VAD split wall (host conv + LSTM scan).
    pub vad: Duration,
    /// Mel extraction + input packing wall.
    pub mel: Duration,
    pub stages: Vec<StageProfile>,
}

impl TranscribeProfile {
    pub fn stage(&self, stage: Stage) -> Option<&StageProfile> {
        self.stages.iter().find(|s| s.stage == stage)
    }
}

impl std::fmt::Display for TranscribeProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "vad {:.1} ms | mel {:.1} ms", self.vad.as_secs_f64() * 1e3, self.mel.as_secs_f64() * 1e3)?;
        for s in &self.stages {
            writeln!(
                f,
                "{:?}: wall {:.1} ms, profiled exec GPU {:.3} ms\n{}",
                s.stage,
                s.wall.as_secs_f64() * 1e3,
                s.gpu_total().as_secs_f64() * 1e3,
                render_histogram(&s.kernels, 20),
            )?;
        }
        Ok(())
    }
}
