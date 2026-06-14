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
//! (the control path — warp/lane math, launch block), the WMMA descriptor (sourced
//! by arch from the shared `TensorCore` table — RDNA routes to the 32-thread WMMA
//! core), and [`ArchCaps::reduce_tree`] (the `wave_size/16 − 1` sibling-fold formula
//! yields the correct `[16]` for the wave32 even/odd accumulator — see below). The
//! RDNA WMMA *fragment* layout is otherwise different (`ept=(16,16,8)`, inputs
//! replicated across the two wave-halves, an even/odd-interleaved `<8×float>`
//! accumulator), so it is carried by dedicated tile shapes (`RT_16X16_W32_*` in
//! [`crate::tiles`]) selected per arch in the kernels, not by reparameterizing the
//! CDNA shapes. Both matmul and FA are now built for gfx1151 (in
//! `MATMUL_/FA_SUPPORTED_ARCHS`); [`ArchCaps::frag_row_stride`] is the one remaining
//! CDNA-only datum (the legacy direct-launch FA mask — the production rolled-db
//! kernel derives its mask from the accumulator's own `lane_rc` instead).

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

    /// Cross-lane (`ds_bpermute`) reduce-tree offsets: a lane folds the partials of
    /// the `wave_size / 16` sibling row-groups, each one WMMA-column span (16 lanes)
    /// apart → wave64 `[16, 32, 48]`, wave32 `[16]`. This is correct for **both** the
    /// CDNA MFMA layout *and* the RDNA even/odd accumulator: at wave32 a softmax row
    /// (16 KV) is split across a lane's 8 in-register elements (the even/odd half)
    /// and its sibling lane `L+16` (the other half), so the single `[16]` fold plus
    /// the in-register reduce covers the whole row (HW-validated reduce structure).
    pub fn reduce_tree(&self) -> SmallVec<[i64; 3]> {
        (1..self.wave_size as i64 / 16).map(|i| i * 16).collect()
    }

    /// Per-lane row stride of a 16×16 **CDNA MFMA** accumulator fragment
    /// (`256 / wave_size`): wave64 → 4. Used only by the legacy direct-launch FA
    /// builders' causal/padding mask (which map each `laneid / 16` row-group to KV
    /// rows with this contiguous stride). The production rolled-db FA derives its
    /// mask from the att accumulator's own `lane_rc` instead — arch-correct for both
    /// the CDNA stride and the RDNA even/odd interleave — so it does not call this.
    pub const fn frag_row_stride(&self) -> i64 {
        (16 * 16 / self.wave_size) as i64
    }
}
