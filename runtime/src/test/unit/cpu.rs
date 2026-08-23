//! CPU device integration tests.
//!
//! Tests the full Device pipeline: render → compile → runtime factory → execute

use crate::devices::cpu::{CpuBackend, create_cpu_device_with_backend};
use svod_device::registry::DeviceRegistry;
use svod_ir::UOp;

#[test]
fn llvm_jit_emits_reusable_object_bytes() {
    use svod_device::device::ProgramSpec;
    use svod_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();
    let spec = ProgramSpec::new(
        "source_only".into(),
        "define void @source_only() { ret void }".into(),
        DeviceSpec::Cpu,
        UOp::sink(vec![]),
    );
    let compiled = device.compiler.compile(&spec).unwrap();
    assert!(compiled.src.is_none());
    assert!(!compiled.bytes.is_empty());
    crate::clang::validate_relocatable_object(&compiled.bytes, "source_only").unwrap();
    assert!(device.compiler.cache_key().starts_with("cpu-llvm-clang:"));
}

#[cfg(unix)]
#[test]
fn clang_object_cache_survives_fresh_process_without_invoking_clang() {
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    const HELPER: &str = "SVOD_TEST_CLANG_CACHE_CHILD";
    if std::env::var_os(HELPER).is_some() {
        use svod_device::device::ProgramSpec;
        use svod_dtype::DeviceSpec;

        let registry = DeviceRegistry::default();
        let device = create_cpu_device_with_backend(&registry, CpuBackend::Clang).unwrap();
        let spec = ProgramSpec::new(
            "fresh_process_kernel".into(),
            "void fresh_process_kernel(float *out) { out[0] = 7.0f; }\n".into(),
            DeviceSpec::Cpu,
            UOp::sink(vec![]),
        );
        let compiled = device.compiler.compile(&spec).unwrap();
        assert!(!compiled.bytes.is_empty());
        return;
    }

    let directory = tempfile::tempdir().unwrap();
    let bin = directory.path().join("bin");
    std::fs::create_dir(&bin).unwrap();
    let count = directory.path().join("clang-invocations");
    let real_clang = Command::new("sh").args(["-c", "command -v clang"]).output().unwrap();
    assert!(real_clang.status.success());
    let real_clang = std::fs::canonicalize(String::from_utf8(real_clang.stdout).unwrap().trim()).unwrap();
    let wrapper = bin.join("clang");
    std::fs::write(
        &wrapper,
        format!("#!/bin/sh\nprintf 'invoked\\n' >> '{}'\nexec '{}' \"$@\"\n", count.display(), real_clang.display()),
    )
    .unwrap();
    let mut permissions = std::fs::metadata(&wrapper).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&wrapper, permissions).unwrap();

    let test_name = std::thread::current().name().unwrap().to_string();
    let executable = std::env::current_exe().unwrap();
    let mut paths = vec![bin];
    paths.extend(std::env::split_paths(&std::env::var_os("PATH").unwrap()));
    let path = std::env::join_paths(paths).unwrap();
    let run_child = || {
        Command::new(&executable)
            .args(["--exact", &test_name, "--nocapture"])
            .env(HELPER, "1")
            .env("PATH", &path)
            .env("SVOD_OBJECT_CACHE_DIR", directory.path().join("cache"))
            .env("SVOD_OBJECT_CACHE_MAX_BYTES", "10485760")
            .status()
            .unwrap()
    };

    assert!(run_child().success());
    let cold_count = std::fs::read_to_string(&count).unwrap().lines().count();
    assert!(cold_count >= 3, "cold process must probe version/target and compile");
    assert!(run_child().success());
    let warm_count = std::fs::read_to_string(&count).unwrap().lines().count();
    assert_eq!(warm_count, cold_count, "warm fresh process invoked clang");
}

#[test]
fn test_cpu_device_creation_llvm() {
    let registry = DeviceRegistry::default();
    let device =
        create_cpu_device_with_backend(&registry, CpuBackend::Llvm).expect("Failed to create CPU device with LLVM");

    // Verify device properties
    assert_eq!(device.base_device_key(), "CPU");
    assert!(device.compiler.cache_key().starts_with("cpu-llvm-clang:"));
}

#[test]
fn test_compile_and_runtime_pipeline_llvm() {
    use svod_device::device::ProgramSpec;
    use svod_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    // Create a minimal valid LLVM IR program
    // This is a no-op kernel that just returns
    let llvm_ir = r#"
define void @test_kernel() {
entry:
  ret void
}
"#;

    let sink = UOp::sink(vec![]);
    let spec = ProgramSpec::new("test_kernel".to_string(), llvm_ir.to_string(), DeviceSpec::Cpu, sink);

    // Test 1: Compile
    let compiled = device.compiler.compile(&spec).expect("Compile should succeed");
    assert!(compiled.src.is_none(), "LLVM compiler should hand reusable object bytes to runtime");
    assert!(!compiled.bytes.is_empty(), "LLVM compiler should emit a relocatable object");
    assert_eq!(compiled.name, "test_kernel");

    // Direct-source compilation remains supported, but executable loading requires
    // a semantic PROGRAM stage identity.
    let err = match (device.runtime)(&compiled) {
        Ok(_) => panic!("identity-less compiler output must not reach the runtime"),
        Err(err) => err,
    };
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { .. }), "{err:?}");

    let staged = svod_codegen::program_pipeline::program_from_sink(UOp::sink(vec![]), DeviceSpec::Cpu).unwrap();
    let (staged, _) = svod_codegen::program_pipeline::do_render(&staged, device.renderer.as_ref()).unwrap();
    let (_, mut compiled) = svod_codegen::program_pipeline::do_compile(&staged, device.compiler.as_ref()).unwrap();
    let program = (device.runtime)(&compiled).expect("validated RuntimeFactory should succeed");
    // Note: program.name() might not match spec.name (it's a TODO in LlvmProgram)
    assert!(!program.name().is_empty(), "Program should have a name");

    // Test 3: Execute (no buffers needed for this kernel)
    let pointers: Vec<*mut u8> = vec![];

    unsafe {
        program.execute(&pointers, &[], None, None, /*wait=*/ true).expect("Execution should succeed");
    }

    compiled.bytes.push(0);
    let err = match (device.runtime)(&compiled) {
        Ok(_) => panic!("tampered compiler output must not reach the runtime"),
        Err(err) => err,
    };
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

#[test]
fn test_compile_invalid_ir() {
    use svod_device::device::ProgramSpec;
    use svod_dtype::DeviceSpec;

    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();

    // Create a ProgramSpec with invalid LLVM IR
    let sink = UOp::sink(vec![]);
    let spec = ProgramSpec::new("test".to_string(), "this is not valid LLVM IR".to_string(), DeviceSpec::Cpu, sink);

    // Object emission validates LLVM IR before it can reach the runtime.
    let result = device.compiler.compile(&spec);
    assert!(result.is_err(), "invalid LLVM IR must fail during object compilation");
}

#[test]
fn cpu_runtime_rejects_missing_stage_identity_before_loading() {
    let registry = DeviceRegistry::default();
    let device = create_cpu_device_with_backend(&registry, CpuBackend::Llvm).unwrap();
    let mut compiled = svod_device::device::CompiledSpec::from_source(
        "bad_abi".into(),
        "define void @bad_abi(ptr %data) { ret void }".into(),
        UOp::sink(vec![]),
        vec![],
    )
    .unwrap();
    compiled.buf_count = 1;

    let err = match (device.runtime)(&compiled) {
        Ok(_) => panic!("runtime must reject count-only ABI before JIT creation"),
        Err(err) => err,
    };
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { .. }), "{err:?}");
}

#[test]
fn cpu_dispatch_binds_interleaved_abi_values() {
    use svod_device::device::{AbiParamDescriptor, AbiParamKind};
    use svod_dtype::{AddrSpace, DType};

    let abi = vec![
        AbiParamDescriptor { slot: 0, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Int32, name: None },
        AbiParamDescriptor { slot: 1, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("low".into()) },
        AbiParamDescriptor { slot: 2, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Int32, name: None },
        AbiParamDescriptor { slot: 3, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("high".into()) },
    ];
    let kernel = crate::jit_loader::JitKernel::compile_with_abi(
        "void interleaved(int *data0, int data1, int *data2, int data3) { *data0 = data1; *data2 = data3; }",
        "interleaved",
        vec!["low".into(), "high".into()],
        &abi,
    )
    .expect("compile interleaved ABI fixture");
    let (mut low, mut high) = (0i32, 0i32);
    let buffers = vec![(&mut low as *mut i32).cast::<u8>(), (&mut high as *mut i32).cast::<u8>()];
    let err = unsafe { kernel.execute_with_vals(&buffers[..1], &[17, -9]) }
        .expect_err("runtime ABI arity mismatch must be typed");
    assert!(matches!(err, crate::Error::Device { source: svod_device::Error::ProgramAbiMismatch { .. } }), "{err:?}");
    unsafe { kernel.execute_with_vals(&buffers, &[17, -9]).expect("execute interleaved ABI fixture") };
    assert_eq!((low, high), (17, -9));
}

#[test]
fn cpu_dispatch_binds_sparse_storage_slots_by_ordinal() {
    use svod_device::device::{AbiParamDescriptor, AbiParamKind};
    use svod_dtype::{AddrSpace, DType};

    let abi = vec![
        AbiParamDescriptor { slot: 0, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Int32, name: None },
        AbiParamDescriptor { slot: 5, kind: AbiParamKind::Storage(AddrSpace::Global), dtype: DType::Int32, name: None },
    ];
    let kernel = crate::jit_loader::JitKernel::compile_with_abi(
        "void sparse(int *data0, int *data5) { *data0 = 17; *data5 = -9; }",
        "sparse",
        vec![],
        &abi,
    )
    .expect("compile sparse storage ABI fixture");
    let (mut first, mut second) = (0i32, 0i32);
    let buffers = [(&mut first as *mut i32).cast::<u8>(), (&mut second as *mut i32).cast::<u8>()];

    unsafe { kernel.execute_with_vals(&buffers, &[]).unwrap() };
    assert_eq!((first, second), (17, -9));
}
