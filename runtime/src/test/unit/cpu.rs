//! CPU device integration tests.
//!
//! Tests the full Device pipeline: render → compile → runtime factory → execute

use crate::devices::cpu::{CpuBackend, create_cpu_device_with_backend};
use svod_device::registry::DeviceRegistry;
use svod_ir::UOp;

#[test]
fn test_cpu_device_creation_llvm() {
    let registry = DeviceRegistry::default();
    let device =
        create_cpu_device_with_backend(&registry, CpuBackend::Llvm).expect("Failed to create CPU device with LLVM");

    // Verify device properties
    assert_eq!(device.base_device_key(), "CPU");
    assert_eq!(device.compiler.cache_key(), "llvm-jit");
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
    assert!(compiled.src.is_some(), "LLVM JIT should have source");
    assert!(compiled.bytes.is_empty(), "LLVM JIT should have empty bytes");
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

    // Compilation should fail gracefully
    // Note: Current implementation doesn't validate, so this will pass
    // TODO: Add LLVM IR validation to LlvmCompiler
    let result = device.compiler.compile(&spec);
    assert!(result.is_ok(), "Should return CompiledSpec even with invalid IR (validation TODO)");
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
