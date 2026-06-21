//! LLVM JIT compilation via external clang + ELF loader.
//!
//! Compiles LLVM IR text via `clang -x ir -c -O2` stdin→stdout and loads the
//! resulting object via the shared JIT ELF loader. No linked LLVM required.

use tracing::debug;

use crate::Result;
use crate::dispatch::KernelCif;
use crate::error::JitResultExt;

/// LLVM JIT-compiled kernel using external clang + mmap ELF loader.
pub struct LlvmKernel {
    _mmap: memmap2::MmapMut,
    fn_ptr: *const (),
    entry_point: String,
    name: String,
    var_names: Vec<String>,
    cif: KernelCif,
}

// SAFETY: Function pointer points to read-only compiled code in mmap'd memory.
// Multiple threads can call it concurrently.
unsafe impl Send for LlvmKernel {}
unsafe impl Sync for LlvmKernel {}

impl LlvmKernel {
    /// Compile LLVM IR text to executable code via external clang.
    pub fn compile_ir(
        ir: &str,
        entry_point: impl Into<String>,
        name: impl Into<String>,
        var_names: Vec<String>,
        buf_count: usize,
    ) -> Result<Self> {
        let entry_point = entry_point.into();
        let name = name.into();

        debug!(kernel.name = %name, ir.length = ir.len(), "Compiling LLVM IR via external clang");

        if let Ok(dir) = std::env::var("SVOD_DUMP_LLVM_IR") {
            let path = std::path::Path::new(&dir).join(format!("{name}.ll"));
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(&path, ir);
        }

        if let Ok(dir) = std::env::var("SVOD_DUMP_POST_O2_IR") {
            // Run the same `-O2 -funroll-loops -fvectorize -fslp-vectorize`
            // pipeline as the JIT compile but emit textual LLVM IR instead
            // of an object file. Writes `<dir>/<name>.post.ll`.
            let _ = std::fs::create_dir_all(&dir);
            if let Some(post_ir) = compile_ir_to_post_o2_text(ir) {
                let path = std::path::Path::new(&dir).join(format!("{name}.post.ll"));
                let _ = std::fs::write(&path, post_ir);
            }
        }

        let obj = compile_ir_to_object(ir)?;
        let (fn_ptr, mmap) = crate::jit_loader::jit_load(&obj, &entry_point)?;
        let cif = KernelCif::new(buf_count + var_names.len());

        debug!(kernel.name = %name, "LLVM kernel compiled and loaded");

        Ok(Self { _mmap: mmap, fn_ptr, entry_point, name, var_names, cif })
    }

    /// Compile a RenderedKernel from the codegen crate.
    pub fn compile(kernel: &svod_codegen::RenderedKernel) -> Result<Self> {
        Self::compile_ir(&kernel.code, &kernel.name, &kernel.name, kernel.var_names.clone(), kernel.buffer_args.len())
    }

    pub fn var_names(&self) -> &[String] {
        &self.var_names
    }

    pub fn fn_ptr(&self) -> *const () {
        self.fn_ptr
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// Execute the kernel with buffer pointers and variable values.
    ///
    /// # Safety
    ///
    /// Caller must ensure buffer pointers are valid/aligned and `vals` length
    /// matches `var_names`.
    pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
        debug!(
            kernel.entry_point = %self.entry_point,
            kernel.num_buffers = buffers.len(),
            kernel.num_vals = vals.len(),
            "Executing LLVM kernel"
        );

        unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None) };

        Ok(())
    }

    pub(crate) fn cif(&self) -> &KernelCif {
        &self.cif
    }
}

/// Compile LLVM IR text to a relocatable object via `clang -x ir`.
///
/// Uses `--target=<arch>-none-unknown-elf` to produce a relocatable ELF object
/// (same as the C path in jit_loader), so the JIT ELF loader can handle
/// relocations consistently.
fn compile_ir_to_object(ir: &str) -> Result<Vec<u8>> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let target = crate::jit_loader::elf_target_triple();

    let mut args = vec![
        "-x",
        "ir",
        "-c",
        "-O2",
        "-march=native",
        "-fPIC",
        "-fno-math-errno",
        "-fno-stack-protector",
        "-funroll-loops",
        "-fvectorize",
        "-fslp-vectorize",
    ];
    args.push(&target);
    args.extend_from_slice(crate::jit_loader::platform_clang_flags());
    args.extend_from_slice(&["-", "-o", "-"]);

    let mut child = Command::new("clang")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .jit("spawn clang for IR (is clang installed?)")?;

    child.stdin.take().expect("stdin was piped").write_all(ir.as_bytes()).jit("write IR to clang stdin")?;

    let output = child.wait_with_output().jit("wait for clang (IR)")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(crate::Error::JitCompilation { reason: format!("clang IR compilation failed:\n{stderr}") });
    }

    if output.stdout.is_empty() {
        return Err(crate::Error::JitCompilation { reason: "clang produced empty output from IR".to_string() });
    }

    Ok(output.stdout)
}

/// Run the same `-O2` LLVM pass pipeline as the JIT compile but emit
/// textual LLVM IR. Returns `None` on compile failure (silent — this
/// is a diagnostic-only path, never load-bearing).
fn compile_ir_to_post_o2_text(ir: &str) -> Option<String> {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let mut args = vec![
        "-x",
        "ir",
        "-S",
        "-emit-llvm",
        "-O2",
        "-march=native",
        "-fno-math-errno",
        "-funroll-loops",
        "-fvectorize",
        "-fslp-vectorize",
    ];
    args.extend_from_slice(crate::jit_loader::platform_clang_flags());
    args.extend_from_slice(&["-", "-o", "-"]);

    let mut child = Command::new("clang")
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.take()?.write_all(ir.as_bytes()).ok()?;
    let output = child.wait_with_output().ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout).ok()
}

#[cfg(test)]
#[path = "test/unit/llvm.rs"]
mod tests;
