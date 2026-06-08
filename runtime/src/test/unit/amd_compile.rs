use super::*;

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

define amdgpu_kernel void @amd_smoke(ptr noalias %buf0) #0 {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}

attributes #0 = { alwaysinline nounwind "no-builtins" "amdgpu-flat-work-group-size"="1,256" "no-trapping-math"="true" }
"#;
    let obj = compile_ir_to_amd_object(ir, AmdArch::Gfx1100).expect("amdgcn compile");
    assert!(!obj.is_empty(), "empty AMD code object");
    // AMDGPU code objects are ELF; check the magic header.
    assert_eq!(&obj[..4], b"\x7fELF", "output is not ELF");
}
