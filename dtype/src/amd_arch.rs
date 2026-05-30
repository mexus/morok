//! AMD GPU architecture targets (gfx-family).
//!
//! Mirrors the arch set covered by tinygrad's AMDLLVMRenderer
//! (`renderer/llvmir.py:259-289`) and HIPRenderer's `is_cdna` predicate
//! (`renderer/cstyle.py:472`).

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum AmdArch {
    // CDNA — datacenter; MFMA, wave64.
    Gfx942,
    Gfx950,
    // RDNA3 — Radeon 7000; WMMA, wave32.
    Gfx1100,
    Gfx1101,
    Gfx1102,
    Gfx1151,
    // RDNA4 — next-gen Radeon; WMMA with bf16/fp8 packing, wave32.
    Gfx1200,
    Gfx1201,
}

impl AmdArch {
    /// CDNA family (datacenter; MFMA intrinsics, wave64).
    pub const fn is_cdna(self) -> bool {
        matches!(self, Self::Gfx942 | Self::Gfx950)
    }

    /// RDNA3 family (Radeon 7000; WMMA intrinsics, wave32).
    pub const fn is_rdna3(self) -> bool {
        matches!(self, Self::Gfx1100 | Self::Gfx1101 | Self::Gfx1102 | Self::Gfx1151)
    }

    /// RDNA4 family (next-gen Radeon; WMMA with bf16/fp8 packing, wave32).
    pub const fn is_rdna4(self) -> bool {
        matches!(self, Self::Gfx1200 | Self::Gfx1201)
    }

    /// `true` when the arch has hardware matrix-multiply intrinsics
    /// (CDNA's MFMA or RDNA3+'s WMMA). All supported arches have matrix cores
    /// since RDNA2 was dropped to match tinygrad's `assert target[0] in (11, 12)
    /// or target in ((9,4,2),(9,5,0))` (`ops_amd.py:962`).
    pub const fn has_matrix_cores(self) -> bool {
        self.is_cdna() || self.is_rdna3() || self.is_rdna4()
    }

    /// Default wave size for this arch (clang `-mcpu` selects this; the kernel
    /// descriptor's `kernel_code_properties` bit 10 encodes wave32 when set).
    pub const fn wave_size(self) -> u32 {
        if self.is_cdna() { 64 } else { 32 }
    }

    /// clang `-mcpu=...` string.
    pub const fn mcpu(self) -> &'static str {
        match self {
            Self::Gfx942 => "gfx942",
            Self::Gfx950 => "gfx950",
            Self::Gfx1100 => "gfx1100",
            Self::Gfx1101 => "gfx1101",
            Self::Gfx1102 => "gfx1102",
            Self::Gfx1151 => "gfx1151",
            Self::Gfx1200 => "gfx1200",
            Self::Gfx1201 => "gfx1201",
        }
    }

    /// Map KFD's `gfx_target_version` integer to an `AmdArch`.
    ///
    /// Format: `major*10000 + minor*100 + stepping` (mirrors tinygrad
    /// `ops_amd.py:961` decoding of `/sys/.../properties` `gfx_target_version`).
    pub fn from_gfx_target_version(v: u32) -> Option<Self> {
        // KFD's `gfx_target_version` encodes (major, minor, step) as
        //   v = major*10_000 + minor*100 + step
        // (decimal, not hex — the `gfx%d%x%x` string formatting in tinygrad's
        // `ops_amd.py:960` re-hexifies minor/step purely for display).
        Some(match v {
            90_402 => Self::Gfx942,
            90_500 => Self::Gfx950,
            110_000 => Self::Gfx1100,
            110_001 => Self::Gfx1101,
            110_002 => Self::Gfx1102,
            110_501 => Self::Gfx1151,
            120_000 => Self::Gfx1200,
            120_001 => Self::Gfx1201,
            _ => return None,
        })
    }

    /// Parse a `gfx{family}` string (case-insensitive).
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "gfx942" => Some(Self::Gfx942),
            "gfx950" => Some(Self::Gfx950),
            "gfx1100" => Some(Self::Gfx1100),
            "gfx1101" => Some(Self::Gfx1101),
            "gfx1102" => Some(Self::Gfx1102),
            "gfx1151" => Some(Self::Gfx1151),
            "gfx1200" => Some(Self::Gfx1200),
            "gfx1201" => Some(Self::Gfx1201),
            _ => None,
        }
    }
}

impl fmt::Display for AmdArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.mcpu())
    }
}

#[cfg(test)]
#[path = "test/unit/amd_arch.rs"]
mod tests;
