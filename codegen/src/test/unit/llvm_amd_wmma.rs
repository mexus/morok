use super::*;

#[test]
fn cdna_mfma_naming() {
    let name =
        resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::BFloat16), Some(ScalarDType::Float32), (16, 16, 16));
    assert_eq!(name.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x16bf16.1k"));
}

#[test]
fn cdna_fp8_requires_k32() {
    // fp8 MFMA selects only at K=32 (`16x16x32`); the K=16 form is a non-existent
    // intrinsic that LLVM silently lowers to an extern call (verified: `llc
    // -mcpu=gfx942` "Cannot select" for `16x16x16.fp8.fp8`).
    let k32 = resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::FP8E4M3), Some(ScalarDType::Float32), (16, 16, 32));
    assert_eq!(k32.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8"));
    let bf8 = resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::FP8E5M2), Some(ScalarDType::Float32), (16, 16, 32));
    assert_eq!(bf8.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x32.bf8.bf8"));
    // K=16 fp8 has no intrinsic on either CDNA generation.
    for arch in [AmdArch::Gfx942, AmdArch::Gfx950] {
        assert!(
            resolve_intrinsic(arch, Some(ScalarDType::FP8E4M3), Some(ScalarDType::Float32), (16, 16, 16)).is_none(),
            "fp8 K=16 must not resolve on {arch}"
        );
    }
}

#[test]
fn cdna_k32_dotted_forms_are_gfx950_only() {
    // The dotted K=32 `.f16`/`.bf16` double-rate forms select on CDNA4 (gfx950)
    // only; on gfx942 they "Cannot select", so the renderer must return `None`
    // there (forcing decomposition) rather than name an unselectable intrinsic.
    let f16 = resolve_intrinsic(AmdArch::Gfx950, Some(ScalarDType::Float16), Some(ScalarDType::Float32), (16, 16, 32));
    assert_eq!(f16.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x32.f16"));
    let bf16 =
        resolve_intrinsic(AmdArch::Gfx950, Some(ScalarDType::BFloat16), Some(ScalarDType::Float32), (16, 16, 32));
    assert_eq!(bf16.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x32.bf16"));
    for in_dt in [ScalarDType::Float16, ScalarDType::BFloat16] {
        assert!(
            resolve_intrinsic(AmdArch::Gfx942, Some(in_dt), Some(ScalarDType::Float32), (16, 16, 32)).is_none(),
            "{in_dt:?} K=32 dotted form must not resolve on gfx942"
        );
    }
    // K=16 f16/bf16 still resolve to the plain forms on both generations.
    assert_eq!(
        resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::Float16), Some(ScalarDType::Float32), (16, 16, 16))
            .as_deref(),
        Some("llvm.amdgcn.mfma.f32.16x16x16f16")
    );
}

#[test]
fn rdna3_wmma_naming() {
    let name =
        resolve_intrinsic(AmdArch::Gfx1100, Some(ScalarDType::Float16), Some(ScalarDType::Float32), (16, 16, 16));
    assert_eq!(name.as_deref(), Some("llvm.amdgcn.wmma.f32.16x16x16.f16"));
}

#[test]
fn rdna4_fp8_wmma_naming() {
    let name =
        resolve_intrinsic(AmdArch::Gfx1201, Some(ScalarDType::FP8E4M3), Some(ScalarDType::Float32), (16, 16, 16));
    assert_eq!(name.as_deref(), Some("llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8"));
}

#[test]
fn unsupported_returns_none() {
    // Bool inputs aren't supported by any WMMA flavor.
    let name = resolve_intrinsic(AmdArch::Gfx1100, Some(ScalarDType::Bool), Some(ScalarDType::Float32), (16, 16, 16));
    assert!(name.is_none());
}
