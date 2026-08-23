//! Clang compilation backend for C codegen.
//!
//! By default, compiles C source via `clang -c` stdin→stdout and loads the
//! resulting object via custom ELF parsing + mmap (no temp files, no dlopen).
//!
//! With `dlopen-fallback` feature: compiles via `clang -shared -O2` and loads
//! the resulting shared library via `dlopen` for kernel execution.

// Default: JIT ELF loader (no temp files, no dlopen)
#[cfg(not(feature = "dlopen-fallback"))]
pub use crate::jit_loader::JitKernel as ClangKernel;

// Fallback: dlopen-based loading
#[cfg(feature = "dlopen-fallback")]
mod dlopen_impl {
    use crate::Result;
    use crate::dispatch::KernelCif;
    use crate::error::JitResultExt;

    /// A compiled C kernel loaded as a shared library.
    pub struct ClangKernel {
        _lib: libloading::Library,
        fn_ptr: *const (),
        name: String,
        var_names: Vec<String>,
        cif: KernelCif,
        _tmp_dir: tempfile::TempDir,
    }

    // SAFETY: The function pointer points to read-only compiled code
    // in the loaded shared library. Multiple threads can call it concurrently.
    unsafe impl Send for ClangKernel {}
    unsafe impl Sync for ClangKernel {}

    impl ClangKernel {
        pub fn compile_with_abi(
            src: &str,
            name: &str,
            var_names: Vec<String>,
            abi: &[svod_device::device::AbiParamDescriptor],
        ) -> Result<Self> {
            use std::io::Write;

            let buffer_count = abi.iter().filter(|arg| arg.is_storage()).count();
            svod_device::device::validate_abi_descriptors(abi, buffer_count, &var_names)?;

            let tmp_dir = tempfile::tempdir().jit("create temp directory")?;

            let src_path = tmp_dir.path().join(format!("{name}.c"));
            let so_path = tmp_dir.path().join(format!("{name}.so"));

            let mut src_file = std::fs::File::create(&src_path).jit("create source file")?;
            src_file.write_all(src.as_bytes()).jit("write source file")?;
            drop(src_file);

            // On ARM, `-mcpu=native` enables CPU-specific tuning. `-march=native`
            // only sets the base ISA family on ARM.
            let march = match std::env::consts::ARCH {
                "x86_64" | "loongarch64" => "-march=native",
                "riscv64" => "-march=rv64g",
                _ => "-mcpu=native",
            };
            let mut args = vec!["-shared", "-O2", march, "-fPIC", "-fno-math-errno", "-fno-ident", "-lm"];
            // Reserve x18 only on macOS ARM, where the kernel clobbers it on
            // context switch. Linux ARM treats x18 as a free GPR; Windows ARM
            // is not a target svod currently supports.
            #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
            args.push("-ffixed-x18");
            let so_str = so_path.to_str().ok_or_else(|| crate::Error::JitCompilation {
                reason: format!("temp .so path is not valid UTF-8: {}", so_path.display()),
            })?;
            let src_str = src_path.to_str().ok_or_else(|| crate::Error::JitCompilation {
                reason: format!("temp source path is not valid UTF-8: {}", src_path.display()),
            })?;
            args.extend_from_slice(&["-o", so_str, src_str]);
            let output =
                std::process::Command::new("clang").args(&args).output().jit("run clang (is clang installed?)")?;

            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                return Err(crate::Error::JitCompilation {
                    reason: format!("clang compilation failed:\n{stderr}\nSource:\n{src}"),
                });
            }

            let lib = unsafe { libloading::Library::new(&so_path).jit("load shared library")? };

            let fn_ptr = unsafe {
                let func: libloading::Symbol<unsafe extern "C" fn()> = lib
                    .get(name.as_bytes())
                    .map_err(|e| crate::Error::FunctionNotFound { name: format!("{name}: {e}") })?;
                *func as *const ()
            };

            let cif = KernelCif::from_abi(abi);
            tracing::debug!(kernel.name = %name, "Clang kernel compiled and loaded (dlopen)");

            Ok(Self { _lib: lib, fn_ptr, name: name.to_string(), var_names, cif, _tmp_dir: tmp_dir })
        }

        pub unsafe fn execute_with_vals(&self, buffers: &[*mut u8], vals: &[i64]) -> Result<()> {
            unsafe { self.cif.dispatch(self.fn_ptr, buffers, vals, None)? };
            Ok(())
        }

        pub(crate) fn cif(&self) -> &KernelCif {
            &self.cif
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
    }
}

#[cfg(feature = "dlopen-fallback")]
pub use dlopen_impl::ClangKernel;

#[cfg(test)]
#[path = "test/unit/clang.rs"]
mod tests;
