//! Low-level FFI types for the AMD KFD path: bindgen output for the KFD
//! ioctl headers and handcrafted HSA ABI structs.

pub mod hsa;
#[cfg(target_os = "linux")]
pub mod ioctl;
pub mod kfd;
pub mod pm4;
