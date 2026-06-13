//! Target-capability gate for hand-built tile kernels.
//!
//! A tile kernel is built for a specific GPU arch (its WMMA descriptor, wave width,
//! and lane distribution are arch-specific) and compiles via `clang -x ir`. This
//! gate validates the kernel inputs' [`DeviceSpec`] against the **arch(es) the
//! kernel declares it supports** and that the AMD LLVM toolchain is present —
//! failing fast with a clear message instead of mis-rendering or failing deep in
//! compile.
//!
//! The gate is generic over the supported-arch set: a kernel passes its own
//! `&[AmdArch]` (flash-attention declares `[Gfx942]` today). Adding a GPU is
//! "declare its arch here (and supply its arch-specific kernel bits)", not "rewrite
//! this"; the generic launch infra (`compile`/`run_kernel`/`graph_launch`) stays
//! arch-agnostic — only the per-kernel launcher invokes this.
//!
//! It validates **from the `DeviceSpec`** (no full-`Device` open): `DeviceSpec::Amd`
//! deliberately omits the arch (it's a hardware property — baking it into the spec
//! invites the "two specs, one physical device" trap; see `svod_dtype::DeviceSpec`),
//! so the arch is resolved from the spec's `device_id` via the KFD topology.

use svod_dtype::{AmdArch, DeviceSpec};

use crate::launch::{Result, UnsupportedTargetSnafu};

/// Verify the kernel inputs' device `spec` is one of the kernel's `supported` AMD
/// arches **and** the AMD LLVM (`clang` amdgcn) toolchain is available. Resolves the
/// arch from `spec`'s `device_id` via the topology (no `Device` open); a non-AMD
/// spec (or an unreadable/unsupported device) fails the supported-arch check. Call
/// from a kernel launcher with `Tensor::device()`.
pub fn check_target(spec: &DeviceSpec, supported: &[AmdArch]) -> Result<()> {
    let arch = match spec {
        DeviceSpec::Amd { device_id } => svod_device::registry::resolve_amd_arch_from_topology(*device_id).ok(),
        _ => None,
    };
    if !arch.is_some_and(|a| supported.contains(&a)) {
        return UnsupportedTargetSnafu {
            reason: format!(
                "kernel supports AMD arch(es) {supported:?}, but device {spec:?} resolved to arch {arch:?}"
            ),
        }
        .fail();
    }
    if !svod_runtime::amd::has_amdgpu_target() {
        return UnsupportedTargetSnafu {
            reason: "AMD LLVM target unavailable — `clang` with the amdgcn backend is required to compile \
                     tile kernels (install ROCm/LLVM clang)"
                .to_string(),
        }
        .fail();
    }
    Ok(())
}
