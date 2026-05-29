//! Userspace AM driver internals — gfx11/RDNA3 first, structured for the rest.
//!
//! **Status: unprivileged scaffolding.** The [`mm`] submodule (TLSF
//! sub-allocators + the GMMU page-table / PTE encoding) is pure logic with no
//! MMIO, so the whole address-space math is unit-tested without a GPU. The
//! privileged bring-up (PCI BAR mapping, IP discovery, PSP firmware load, GFX
//! ring setup) — and the `AmIface` implementor that ties them to the
//! [`crate::amd::iface::AmdIface`] seam — land incrementally once a
//! root-capable environment is available (AM unbinds amdgpu + mmaps the BARs,
//! which needs `cap_sys_rawio`/root).
//!
//! Arch parametrization is data-driven: register tables keyed by the `ip_ver`
//! tuples read from IP discovery, with the gfx11/gfx12/gfx9 deltas as small
//! inline `if ip_ver >= (X,Y,Z)` branches inside shared modules (mirrors
//! tinygrad's `runtime/support/am`). gfx11 is wired now; the rest are data adds.

pub mod mm;
