use super::*;

fn storage_abi(count: usize) -> Vec<svod_device::device::AbiParamDescriptor> {
    (0..count)
        .map(|slot| svod_device::device::AbiParamDescriptor {
            slot,
            kind: svod_device::device::AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        })
        .collect()
}

/// A JIT'd LLVM module runs through the declared storage ABI, with or without
/// pointer arguments, and unloads cleanly when the kernel handle is dropped.
#[test_case::test_case("define void @kernel() {\n  ret void\n}\n", 0; "no arguments")]
#[test_case::test_case("define void @kernel(ptr noalias %data0, ptr noalias %data1) {\n  ret void\n}\n", 2; "two buffers")]
fn llvm_kernel_compiles_and_executes(ir: &str, buffers: usize) {
    let kernel = LlvmKernel::compile_ir_with_abi(ir, "kernel", "kernel", vec![], &storage_abi(buffers)).unwrap();
    assert_eq!(kernel.name(), "kernel");

    let mut storage = vec![vec![0u8; 16]; buffers];
    let pointers = storage.iter_mut().map(|buffer| buffer.as_mut_ptr()).collect::<Vec<_>>();
    unsafe { kernel.execute_with_vals(&pointers, &[]).unwrap() };
    drop(kernel);
}
