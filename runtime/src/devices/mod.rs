//! Device implementations for different backends.

#[cfg(target_os = "linux")]
pub mod amd;
pub mod cpu;
pub mod cpu_queue;

#[cfg(target_os = "linux")]
pub use amd::create_amd_device;
pub use cpu::{CpuBackend, create_cpu_device, create_cpu_device_with_backend};
pub use cpu_queue::CpuQueue;
