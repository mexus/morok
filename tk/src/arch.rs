//! Arch-derived capability bundle for the tile DSL.
//!
//! `svod-tk` kernels are built for a specific GPU arch: the wave width, the
//! cross-lane reduce tree, and the per-lane WMMA-fragment row stride are all
//! arch properties, as is the WMMA descriptor itself ([`crate::group`] looks it
//! up from the shared `TensorCore` table by [`ArchCaps::arch`]). [`ArchCaps`] is
//! the single place those are derived from an [`AmdArch`], so the builders thread
//! one value instead of hardcoding gfx942 (wave64) literals.
//!
//! Today only gfx942 (CDNA3, wave64) is actually *built*: it is the validated
//! target, and the register-tile fragment-layout tables ([`crate::tiles`] strides
//! and `group::mma`'s per-lane upcast counts) are calibrated for it (see
//! [`crate::WARP_THREADS`]).
//!
//! What generalizes cleanly to RDNA3.5 (gfx1151, wave32): [`ArchCaps::wave_size`]
//! (the control path — warp/lane math, launch block) and the WMMA descriptor
//! (sourced by arch from the shared `TensorCore` table — RDNA routes to the
//! 32-thread WMMA core). What does **NOT**: [`ArchCaps::reduce_tree`] and
//! [`ArchCaps::frag_row_stride`] encode the **CDNA MFMA** fragment geometry, and
//! RDNA WMMA is a *different* layout (`ept=(16,16,8)`, inputs replicated across the
//! two wave-halves, an even/odd-interleaved `<8×float>` accumulator) — so the
//! RDNA3.5 fragment path is a separate implementation, not a wave-size reparam.
//! gfx1151 therefore stays out of `FA_/MATMUL_SUPPORTED_ARCHS` (it falls back) and
//! [`Kernel::new`](crate::Kernel::new)'s `debug_assert` blocks building it until
//! that layout lands and is validated on hardware.

use smallvec::SmallVec;
use svod_dtype::AmdArch;

/// The arch-derived constants the tile builders thread instead of the wave64
/// literals. `Copy`; [`Self::for_arch`] is `const`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArchCaps {
    /// The target GPU arch — drives the WMMA descriptor lookup
    /// (`Renderer::for_amd_arch`) in [`crate::group`].
    pub arch: AmdArch,
    /// Lanes per wave. `threadIdx` splits into warp = `idx / wave_size` and lane
    /// = `idx % wave_size`; the launch block is `warps * wave_size`. 64 on CDNA3,
    /// 32 on RDNA3/4.
    pub wave_size: usize,
}

impl ArchCaps {
    /// Derive the caps from `arch` (wave size from [`AmdArch::wave_size`]).
    pub const fn for_arch(arch: AmdArch) -> Self {
        Self { arch, wave_size: arch.wave_size() as usize }
    }

    /// The validated default target: gfx942 (CDNA3, wave64).
    pub const GFX942: ArchCaps = ArchCaps::for_arch(AmdArch::Gfx942);

    /// Cross-lane (`ds_bpermute`) reduce-tree offsets **for the CDNA MFMA fragment
    /// layout**: a lane folds the partials of the `wave_size / 16` sibling
    /// row-groups, each one WMMA-column span (16 lanes) apart → wave64
    /// `[16, 32, 48]` (reproduces the prior literal). At wave32 this formula yields
    /// `[16]`, which is **not** the RDNA WMMA reduce — RDNA's accumulator lane
    /// layout differs, so RDNA3.5 needs its own tree (deferred; see module docs).
    pub fn reduce_tree(&self) -> SmallVec<[i64; 3]> {
        (1..self.wave_size as i64 / 16).map(|i| i * 16).collect()
    }

    /// Per-lane row stride of a 16×16 **CDNA MFMA** accumulator fragment
    /// (`256 / wave_size`): wave64 → 4. The FA causal/padding mask maps each
    /// `laneid / 16` row-group to KV rows with this stride. (RDNA's `<8×float>`
    /// accumulator is also 8/lane at wave32, but its rows are even/odd interleaved
    /// across the wave-halves, not this contiguous stride — deferred.)
    pub const fn frag_row_stride(&self) -> i64 {
        (16 * 16 / self.wave_size) as i64
    }
}
