//! Low-level FFI types for the AMD KFD path: bindgen output for the KFD ioctl
//! headers (`kfd`) and the HSA / AQL ABI (`hsa`), plus hand-written PM4 / SDMA
//! packet builders.

#[cfg(unix)]
pub mod hsa;
#[cfg(unix)]
pub mod ioctl;
pub mod kfd;
pub mod pm4;
pub mod sdma;
