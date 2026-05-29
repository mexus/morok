//! AMD GPU support for the device crate (KFD-direct, no HIP runtime).
//!
//! Linux-only: on other OSes this module exposes only the type names and
//! returns `Err(NoAmdGpu)` / `Err(DeviceUnavailable)` at runtime. See
//! `~/.claude/plans/hi-buddy-we-re-working-lovely-pancake.md` for the phasing.

pub mod sys;
pub mod topology;

#[cfg(target_os = "linux")]
pub mod allocator;
#[cfg(target_os = "linux")]
pub mod connector;
#[cfg(target_os = "linux")]
pub mod device;
#[cfg(target_os = "linux")]
pub mod graph;
#[cfg(target_os = "linux")]
pub mod hw_queue;
#[cfg(target_os = "linux")]
pub mod iface;
#[cfg(target_os = "linux")]
pub mod kernarg;
#[cfg(target_os = "linux")]
pub mod program;
#[cfg(target_os = "linux")]
pub mod queue;
#[cfg(target_os = "linux")]
pub mod signal;

#[cfg(target_os = "linux")]
pub use allocator::AmdAllocator;
#[cfg(target_os = "linux")]
pub use connector::{AmdConnector, ConnectorLease};
#[cfg(target_os = "linux")]
pub use device::{AmdDevice, AmdDeviceCore};
#[cfg(target_os = "linux")]
pub use graph::AmdGraph;
#[cfg(target_os = "linux")]
pub use hw_queue::{AmdArgsState, AmdHwQueue, Fmt, Sym, VarVals};
#[cfg(target_os = "linux")]
pub use iface::{AllocKind, AllocResult, AmdIface, KfdIface, QueueHandle, RingDesc};
#[cfg(target_os = "linux")]
pub use kernarg::KernargArena;
#[cfg(target_os = "linux")]
pub use program::AmdProgram;
#[cfg(target_os = "linux")]
pub use queue::{AmdComputeQueue, AmdCopyQueue};
#[cfg(target_os = "linux")]
pub use signal::{AmdSignal, SignalPool, Timeline};
pub use topology::{AmdNode, enumerate};
