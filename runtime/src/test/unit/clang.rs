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

#[test]
fn test_clang_kernel_noop() {
    let src = "void test_kernel(void) { }\n";
    let kernel = ClangKernel::compile_with_abi(src, "test_kernel", vec![], &storage_abi(0)).unwrap();
    assert_eq!(kernel.name(), "test_kernel");
    unsafe {
        kernel.execute_with_vals(&[], &[]).unwrap();
    }
}

#[test]
fn test_clang_kernel_add() {
    let src = r#"
void add_kernel(float* restrict a, float* restrict b, float* restrict out) {
    out[0] = a[0] + b[0];
}
"#;
    let kernel = ClangKernel::compile_with_abi(src, "add_kernel", vec![], &storage_abi(3)).unwrap();

    let mut a = [1.0f32];
    let mut b = [2.0f32];
    let mut out = [0.0f32];

    let buffers = vec![a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }

    assert_eq!(out[0], 3.0);
}

#[test]
fn cpu_object_validation_checks_header_and_entry_symbol() {
    let toolchain = ClangToolchain::discover(None).unwrap();
    let flags = c_object_flags();
    let object = compile_c_object(&toolchain, "void cached_kernel(void) {}\n", &flags).unwrap();
    validate_c_object(&object, "cached_kernel").unwrap();
    assert!(validate_c_object(&object, "other_kernel").is_err());

    let mut wrong_machine = object;
    wrong_machine[18..20].copy_from_slice(&0xffffu16.to_le_bytes());
    assert!(validate_c_object(&wrong_machine, "cached_kernel").is_err());
}

#[test]
fn clean_compiler_process_produces_valid_object() {
    let toolchain = ClangToolchain::discover(None).unwrap();
    let mut process = spawn_compile_process(&toolchain, "void isolated_kernel(void) {}\n", &c_object_flags()).unwrap();
    while process.try_wait().unwrap().is_none() {
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let object = process.finish().unwrap();
    validate_c_object(&object, "isolated_kernel").unwrap();
}

/// `-march=native` resolves against the running CPU, so its `-###` probe must
/// never be shared between hosts; an explicit arch must be.
#[test_case::test_case("-march=native" => false; "host-resolved arch")]
#[test_case::test_case("-mcpu=native" => false; "host-resolved cpu")]
#[test_case::test_case("-march=x86-64" => true; "explicit arch")]
fn probe_key_is_shared_between_hosts(flag: &str) -> bool {
    let flags = vec![flag.to_string()];
    let executable = [7u8; 32];
    probe_key(&executable, &flags, Some(&[1u8; 32])) == probe_key(&executable, &flags, Some(&[2u8; 32]))
}

#[test]
fn unfingerprintable_host_disables_probe_sharing_for_native_flags() {
    let executable = [7u8; 32];
    assert!(probe_key(&executable, &["-march=native".into()], None).is_none());
    assert!(probe_key(&executable, &["-march=x86-64".into()], None).is_some());
}
