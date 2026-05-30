//! `clang -target amdgcn-amd-amdhsa` driver: lowers AMD LLVM text-IR
//! to an AMDGPU code object (ELF) that the KFD runtime can dispatch.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use svod_dtype::AmdArch;
use tracing::debug;

/// Compile AMD LLVM IR text into a fully-linked AMDGPU code object.
///
/// `clang --target=amdgcn-amd-amdhsa -mcpu={arch}` produces an `ET_DYN` ELF
/// that already has lld's amdgpu-link step applied (clang invokes lld
/// internally for single-TU compilations), so the output is directly loadable
/// by the KFD runtime — no further link step required.
///
/// # Errors
///
/// - [`crate::Error::JitCompilation`] when `clang` is missing, the AMDGPU
///   target is not enabled in the host LLVM, or compilation fails.
pub fn compile_ir_to_amd_object(ir: &str, arch: AmdArch) -> crate::Result<Vec<u8>> {
    if !has_amdgpu_target() {
        return Err(crate::Error::JitCompilation {
            reason: "AMD GPU support requires clang built with the AMDGPU target. \
                     Reinstall clang from your distro or build with \
                     -DLLVM_TARGETS_TO_BUILD='X86;AArch64;AMDGPU'."
                .to_string(),
        });
    }

    if let Ok(dir) = std::env::var("SVOD_DUMP_AMD_IR") {
        // Extract the kernel name from the `ModuleID = '<name>'` directive so
        // each kernel lands in its own file. Without this, every compile
        // overwrites the same path and debugging a specific failing kernel is
        // impossible (the dispatcher pre-compiles many kernels ahead of any
        // dispatch, so the dumped file would always be the LAST one compiled,
        // never the failing one).
        let kernel_name = ir
            .lines()
            .find_map(|l| l.strip_prefix("; ModuleID = '").and_then(|s| s.strip_suffix("'")))
            .unwrap_or("unknown");
        // Sanitize for filesystem (kernel names contain only [A-Za-z0-9_] in
        // practice, but be defensive against future renderer changes).
        let safe: String =
            kernel_name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' }).collect();
        let path = std::path::Path::new(&dir).join(format!("{}_{}.ll", arch.mcpu(), safe));
        let _ = std::fs::create_dir_all(&dir);
        let _ = std::fs::write(&path, ir);
    }

    debug!(arch = arch.mcpu(), ir.length = ir.len(), "compiling amdgcn IR via clang");

    let mcpu_arg = format!("-mcpu={}", arch.mcpu());
    let args: &[&str] = &[
        "-x",
        "ir",
        "-c",
        "-O3",
        "--target=amdgcn-amd-amdhsa",
        &mcpu_arg,
        "-mcumode",
        "-nogpulib",
        "-nogpuinc",
        "-Wno-override-module",
        "-fno-math-errno",
        "-",
        "-o",
        "-",
    ];

    let mut child = Command::new("clang")
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| crate::Error::JitCompilation { reason: format!("failed to spawn clang: {e}") })?;

    child
        .stdin
        .take()
        .expect("stdin piped")
        .write_all(ir.as_bytes())
        .map_err(|e| crate::Error::JitCompilation { reason: format!("failed to write IR to clang stdin: {e}") })?;

    let output = child
        .wait_with_output()
        .map_err(|e| crate::Error::JitCompilation { reason: format!("failed to wait for clang: {e}") })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::Error::JitCompilation {
            reason: format!("clang amdgcn compilation failed (mcpu={}):\n{stderr}", arch.mcpu()),
        });
    }
    if output.stdout.is_empty() {
        return Err(crate::Error::JitCompilation {
            reason: format!("clang produced empty output for amdgcn mcpu={}", arch.mcpu()),
        });
    }

    Ok(output.stdout)
}

/// Does the host `clang` advertise the `amdgpu` target?
///
/// Cached for the lifetime of the process: clang installation doesn't change
/// during a run, and the subprocess is too slow to do per-call.
pub fn has_amdgpu_target() -> bool {
    static CACHE: OnceLock<bool> = OnceLock::new();
    *CACHE.get_or_init(|| {
        let output = match Command::new("clang").arg("--print-targets").output() {
            Ok(o) => o,
            Err(_) => return false,
        };
        if !output.status.success() {
            return false;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout.lines().any(|line| line.split_whitespace().next().map(|t| t == "amdgcn").unwrap_or(false))
    })
}

#[cfg(test)]
#[path = "../test/unit/amd_compile.rs"]
mod tests;
