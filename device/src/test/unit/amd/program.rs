
use super::*;

/// Compile a trivial amdgcn kernel via Phase 2, then parse it back and
/// verify the kernel descriptor round-trips. Skipped when host clang
/// lacks AMDGPU target.
#[test]
fn parse_kernel_descriptor_from_compiled_elf() {
    // We can't pull svod-runtime here (dependency would cycle), so we
    // shell out to clang ourselves with the same flags as
    // `runtime::amd::compile`. Lighter than wiring a dev-dep.
    let ir = r#"; ModuleID = 'p6_smoke'
source_filename = "p6_smoke"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @p6_smoke(ptr noalias %buf0) #0 {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}

attributes #0 = { alwaysinline nounwind "no-builtins" "amdgpu-flat-work-group-size"="1,64" "no-trapping-math"="true" }
"#;
    let out = match std::process::Command::new("clang")
        .args([
            "-x",
            "ir",
            "-c",
            "-O2",
            "--target=amdgcn-amd-amdhsa",
            "-mcpu=gfx1100",
            "-mcumode",
            "-nogpulib",
            "-nogpuinc",
            "-Wno-override-module",
            "-",
            "-o",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping: clang not available");
            return;
        }
    };
    use std::io::Write;
    let mut out = out;
    out.stdin.take().unwrap().write_all(ir.as_bytes()).unwrap();
    let output = out.wait_with_output().unwrap();
    if !output.status.success() {
        eprintln!("skipping: clang amdgcn compile failed (target may be unavailable)");
        return;
    }
    let bytes = output.stdout;
    let parsed = parse_kernel(&bytes, "p6_smoke").expect("parse");
    // Sanity: kernarg_size is at least one ptr (8 bytes), aligned.
    let kernarg_size = parsed.kd.kernarg_size;
    assert!(kernarg_size >= 8, "kernarg_size {} should hold at least one pointer", kernarg_size);
    // Sanity: descriptor offset is inside the image.
    assert!((parsed.kd_offset as usize) < parsed.image.len());
}
