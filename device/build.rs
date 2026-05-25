//! Build script: bindgen for the vendored Linux KFD ioctl header.
//!
//! Only runs on Linux. Other platforms emit an empty stub so downstream
//! `mod kfd { include!(...); }` compiles unconditionally (and the AMD path
//! returns `Err(NoAmdGpu)` at runtime on non-Linux hosts).

use std::env;
use std::path::PathBuf;

fn main() {
    let out_dir: PathBuf = env::var("OUT_DIR").expect("OUT_DIR set by cargo").into();
    let kfd_rs = out_dir.join("kfd_sys.rs");

    #[cfg(target_os = "linux")]
    {
        println!("cargo:rerun-if-changed=include/kfd_ioctl.h");
        let header = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("include/kfd_ioctl.h");
        let bindings = bindgen::Builder::default()
            .header(header.to_string_lossy().into_owned())
            .allowlist_type("kfd_ioctl_.*_args")
            .allowlist_type("kfd_process_device_apertures")
            .allowlist_type("kfd_event_data")
            .allowlist_type("kfd_hsa_signal_event_data")
            .allowlist_type("kfd_hsa_memory_exception_data")
            .allowlist_type("kfd_hsa_hw_exception_data")
            .allowlist_type("kfd_memory_exception_failure")
            .allowlist_type("__u\\d+")
            .allowlist_type("__s\\d+")
            .allowlist_var("KFD_IOC_.*")
            .allowlist_var("KFD_MMAP_TYPE.*")
            .allowlist_var("KFD_MAX_QUEUE_PERCENTAGE")
            .allowlist_var("AMDKFD_IOC_.*")
            .derive_default(true)
            .layout_tests(false)
            .generate_comments(false)
            .generate()
            .expect("bindgen kfd_ioctl.h");
        bindings.write_to_file(&kfd_rs).expect("write kfd_sys.rs");
    }

    #[cfg(not(target_os = "linux"))]
    {
        // Empty stub — the AMD module's `sys/kfd.rs` includes this via
        // `include!()`, so the file must always exist.
        std::fs::write(&kfd_rs, "// KFD bindings unavailable on non-Linux hosts.\n")
            .expect("write empty kfd_sys.rs stub");
    }
}
