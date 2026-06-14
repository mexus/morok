//! The [`ArchCaps`] capability layer: the wave size / cross-lane reduce tree /
//! WMMA-fragment row stride are *derived from the detected `AmdArch`*, not
//! hand-set per call. gfx942 must reproduce the prior wave64 literals
//! bit-for-bit (the builders now thread these instead of the old constants);
//! gfx1151 (RDNA3.5, wave32) gets the correct control-path caps even though it
//! stays out of `FA_/MATMUL_SUPPORTED_ARCHS` (falls back) until its
//! fragment-layout table is validated on hardware.

use svod_dtype::{AmdArch, DType};
use svod_schedule::optimizer::Renderer;

use crate::ArchCaps;

/// Behavior-preserving guard: the caps derived for gfx942 reproduce exactly the
/// wave64 literals the builders previously hardcoded (`64`, the `[16,32,48]`
/// `ds_bpermute` sibling tree, the `*4` FA fragment row stride). If the
/// derivation drifts, the gfx942 path is no longer bit-identical — fail loudly.
#[test]
fn gfx942_caps_reproduce_wave64_literals() {
    let c = ArchCaps::for_arch(AmdArch::Gfx942);
    assert_eq!(c, ArchCaps::GFX942, "GFX942 const == for_arch(Gfx942)");
    assert_eq!(c.wave_size, 64);
    assert_eq!(c.reduce_tree().as_slice(), &[16, 32, 48], "prior ds_bpermute sibling tree");
    assert_eq!(c.frag_row_stride(), 4, "prior FA `*4` lane→KV-row stride (256/64)");
}

/// RDNA3.5 (gfx1151) control-path caps that genuinely generalize: wave32 (the
/// warp/lane math + launch block) and the RDNA3.5 classification. The WMMA
/// descriptor resolution is covered by [`wmma_descriptor_resolves_per_detected_arch`].
/// NOTE: `reduce_tree`/`frag_row_stride` are the CDNA-MFMA formulas evaluated at 32
/// lanes — **not** the RDNA WMMA fragment layout (replicated inputs, even/odd
/// accumulator), which is a separate deferred implementation — so they are
/// deliberately not asserted as RDNA-correct here.
#[test]
fn gfx1151_caps_are_wave32() {
    let c = ArchCaps::for_arch(AmdArch::Gfx1151);
    assert_eq!(c.wave_size, 32);
    assert!(c.arch.is_rdna3_5() && !c.arch.is_rdna3(), "gfx1151 is RDNA3.5, distinct from RDNA3");
}

/// The WMMA descriptor is sourced from the shared `TensorCore` table *by the
/// detected arch* (`group::wmma_desc` looks up `Renderer::for_amd_arch(caps.arch)`),
/// so it tracks the GPU in use — not a hand-built descriptor. Confirm the
/// 16×16×16 f16 core resolves with the arch's wave thread count on both the
/// validated CDNA3 path (64) and the deferred RDNA3.5 path (32).
#[test]
fn wmma_descriptor_resolves_per_detected_arch() {
    let core_threads = |arch: AmdArch| {
        Renderer::for_amd_arch(arch)
            .tensor_cores
            .into_iter()
            .find(|tc| tc.dtype_in == DType::Float16 && tc.dims == (16, 16, 16))
            .map(|tc| tc.threads)
    };
    assert_eq!(core_threads(AmdArch::Gfx942), Some(64), "gfx942 f16 WMMA = wave64 MFMA core");
    assert_eq!(core_threads(AmdArch::Gfx1151), Some(32), "gfx1151 f16 WMMA = wave32 RDNA core");
}
