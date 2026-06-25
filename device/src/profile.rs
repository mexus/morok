//! Backend-agnostic profiling data shared across the device boundary.
//!
//! These types live in `device` (not `runtime`) because the device traits
//! [`crate::DispatchTimestamps::counters`] and [`crate::Program::resource_usage`]
//! return them. `runtime` re-exports them through its `profiler` module so callers
//! see one profiling vocabulary.

use std::collections::BTreeMap;

/// A hardware performance counter selectable via PMC. The current set is the
/// SQ block (gfx11/RDNA3.5), which answers the ILP/occupancy question: VALU
/// instructions issued vs SQ-busy cycles, plus waves launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PmcCounter {
    /// Cycles the shader sequencer (SQ) was busy.
    SqBusyCycles,
    /// Waves launched.
    SqWaves,
    /// VALU instructions issued.
    SqInstsValu,
}

impl PmcCounter {
    /// Short token used in `SVOD_PMC=…` selection and as a table header.
    pub fn token(self) -> &'static str {
        match self {
            Self::SqBusyCycles => "sqbusy",
            Self::SqWaves => "waves",
            Self::SqInstsValu => "valu",
        }
    }

    /// Parse a `SVOD_PMC` token; unknown tokens return `None`.
    pub fn from_token(s: &str) -> Option<Self> {
        Some(match s.trim().to_ascii_lowercase().as_str() {
            "sqbusy" | "busy" => Self::SqBusyCycles,
            "waves" => Self::SqWaves,
            "valu" => Self::SqInstsValu,
            _ => return None,
        })
    }

    /// All implemented counters (the default selection).
    pub fn all() -> [PmcCounter; 3] {
        [Self::SqBusyCycles, Self::SqWaves, Self::SqInstsValu]
    }
}

/// Hardware counter values harvested for one dispatch. Sparse: only the
/// requested counters are present.
#[derive(Debug, Clone, Default)]
pub struct CounterSet {
    pub values: BTreeMap<PmcCounter, u64>,
}

/// Per-kernel static GPU resource usage, decoded from the compiled program's
/// kernel descriptor. Pure static — no runtime cost.
#[derive(Debug, Clone, Copy)]
pub struct KernelResources {
    /// Vector GPRs allocated per lane.
    pub vgprs: u32,
    /// Scalar GPRs allocated.
    pub sgprs: u32,
    /// LDS (group segment) bytes per workgroup.
    pub lds_bytes: u32,
    /// Scratch (private segment) bytes per lane.
    pub scratch_bytes: u32,
    /// Wave width (32 or 64).
    pub wave_size: u32,
    /// VGPR-limited occupancy as a fraction of the SIMD's max resident waves
    /// (0.0–1.0), for architectures with known register-file geometry. `None`
    /// when the geometry is unknown — the first-order limiter only (LDS and
    /// workgroup limits are not modeled).
    pub occupancy: Option<f32>,
}
