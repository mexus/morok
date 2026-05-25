//! LLVM target selection (CPU vs AMD GPU).
//!
//! Threaded through the renderer so that op-emission helpers (address spaces,
//! kernel attributes, intrinsic names) can branch on the target without
//! introducing separate renderer types.

use svod_dtype::{AddrSpace, AmdArch};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlvmTarget {
    /// Host CPU. Uses x86/AArch64 ELF triple, generic LLVM IR.
    Cpu,
    /// AMD GPU at the named `gfx{family}` target. Uses
    /// `amdgcn-amd-amdhsa` triple, `amdgpu_kernel` calling convention,
    /// and amdgcn-specific intrinsics for SPECIAL/BARRIER/WMMA.
    Amd(AmdArch),
}

impl LlvmTarget {
    pub fn is_amd(&self) -> bool {
        matches!(self, Self::Amd(_))
    }

    pub fn amd_arch(&self) -> Option<AmdArch> {
        match self {
            Self::Amd(a) => Some(*a),
            _ => None,
        }
    }
}

/// Numeric address space encoded in LLVM IR pointer types for this target.
///
/// CPU: addrspace(0) is the generic flat space; LLVM's IR-level distinction
/// between Global and Local doesn't really apply (we use `alloca` for Local).
/// AMD: AMDGPU mandates explicit address spaces — Global=1, Constant=4,
/// Local=3, Private=5, Generic=0. Kernel-arg pointers are passed unannotated
/// (`ptr`) and the backend implicitly promotes to addrspace(1).
///
/// See <https://llvm.org/docs/AMDGPUUsage.html#address-spaces>.
pub fn addr_space_num(target: LlvmTarget, addrspace: AddrSpace) -> u32 {
    match (target, addrspace) {
        (LlvmTarget::Cpu, AddrSpace::Global) => 0,
        (LlvmTarget::Cpu, AddrSpace::Local) => 3,
        (LlvmTarget::Cpu, AddrSpace::Reg) => 5,
        (LlvmTarget::Amd(_), AddrSpace::Global) => 1,
        (LlvmTarget::Amd(_), AddrSpace::Local) => 3,
        (LlvmTarget::Amd(_), AddrSpace::Reg) => 5,
    }
}
