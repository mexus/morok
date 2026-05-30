
use super::*;

#[test]
fn cdna_mfma_naming() {
    let name =
        resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::BFloat16), Some(ScalarDType::Float32), (16, 16, 16));
    assert_eq!(name.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x16bf16.1k"));
}

#[test]
fn cdna_mfma_k32_and_fp8() {
    let k32 = resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::Float16), Some(ScalarDType::Float32), (16, 16, 32));
    assert_eq!(k32.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x32.f16"));
    let fp8 = resolve_intrinsic(AmdArch::Gfx942, Some(ScalarDType::FP8E4M3), Some(ScalarDType::Float32), (16, 16, 16));
    assert_eq!(fp8.as_deref(), Some("llvm.amdgcn.mfma.f32.16x16x16.fp8.fp8"));
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
