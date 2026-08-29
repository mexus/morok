use super::*;

/// The ROCm device libraries are only searched when the IR still calls an f64
/// `__ocml_*` entry point; everything else is an `@llvm.*` intrinsic the
/// AMDGPU backend selects on its own.
#[test_case::test_case("  %r = call float @llvm.exp2.f32(float %v)" => true; "llvm intrinsics only")]
#[test_case::test_case("  %r = call double @__ocml_exp2_f64(double %v)" => false; "f64 ocml transcendental")]
fn nogpulib_tracks_device_library_use(body: &str) -> bool {
    amd_object_flags(body, AmdArch::Gfx1100).iter().any(|flag| flag == "-nogpulib")
}

#[test]
fn has_amdgpu_target_returns_consistent_bool() {
    // Behavior depends on local clang; we just check it doesn't panic and
    // returns the same value across calls (the OnceLock caches it).
    let a = has_amdgpu_target();
    let b = has_amdgpu_target();
    assert_eq!(a, b);
}

#[test]
fn compile_returns_clean_err_without_target() {
    if has_amdgpu_target() {
        // Real compile path tested by the integration test below.
        return;
    }
    let err = compile_ir_to_amd_object("; empty\n", AmdArch::Gfx1100).expect_err("must fail without amdgpu target");
    let msg = format!("{err}");
    assert!(msg.contains("AMDGPU"), "unexpected error message: {msg}");
}

/// Round-trips a tiny AMD LLVM kernel through the Phase 1 renderer + clang.
/// Skipped when the host clang lacks the AMDGPU target.
#[test]
fn compile_smoke_gfx1100() {
    if !has_amdgpu_target() {
        eprintln!("skipping: host clang has no amdgpu target");
        return;
    }
    let ir = r#"; ModuleID = 'amd_smoke'
source_filename = "amd_smoke"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()
declare float @llvm.exp2.f32(float)

define amdgpu_kernel void @amd_smoke(ptr noalias %buf0) #0 {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  %v = load float, ptr %p
  %e = call float @llvm.exp2.f32(float %v)
  store float %e, ptr %p
  ret void
}

attributes #0 = { alwaysinline nounwind "no-builtins" "amdgpu-flat-work-group-size"="1,256" "no-trapping-math"="true" }
"#;
    let obj = compile_ir_to_amd_object(ir, AmdArch::Gfx1100).expect("amdgcn compile");
    assert!(!obj.is_empty(), "empty AMD code object");
    // AMDGPU code objects are ELF; check the magic header.
    assert_eq!(&obj[..4], b"\x7fELF", "output is not ELF");
    validate_amd_object(&obj, AmdArch::Gfx1100, "amd_smoke").expect("valid gfx1100 object");
    assert!(validate_amd_object(&obj, AmdArch::Gfx1101, "amd_smoke").is_err(), "wrong target arch must fail");
    assert!(validate_amd_object(&obj, AmdArch::Gfx1100, "other_kernel").is_err(), "wrong kernel must fail");
}
