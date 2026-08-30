use super::*;

use svod_runtime::CpuBackend;
use test_case::test_case;

/// `for_cpu_backend` resolves a CPU device per schedule item; the toolchain
/// probe behind it must run once per backend, i.e. resolves must be memoized.
#[test_case(CpuBackend::Clang; "clang")]
#[test_case(CpuBackend::Llvm; "llvm")]
fn resolve_device_is_memoized_per_backend(backend: CpuBackend) {
    let config = PrepareConfig::for_cpu_backend(backend);
    let registry = svod_device::registry::registry();

    let first = config.resolve_device(&DeviceSpec::Cpu, registry).expect("resolve");
    let second = config.resolve_device(&DeviceSpec::Cpu, registry).expect("resolve");
    assert!(Arc::ptr_eq(&first, &second), "repeated resolves must share one device");

    // A second config over the same backend sees the same process-wide device.
    let other = PrepareConfig::for_cpu_backend(backend).resolve_device(&DeviceSpec::Cpu, registry).expect("resolve");
    assert!(Arc::ptr_eq(&first, &other));
}

#[test]
fn resolve_device_separates_backends() {
    let registry = svod_device::registry::registry();
    let clang = PrepareConfig::for_cpu_backend(CpuBackend::Clang).resolve_device(&DeviceSpec::Cpu, registry).unwrap();
    let llvm = PrepareConfig::for_cpu_backend(CpuBackend::Llvm).resolve_device(&DeviceSpec::Cpu, registry).unwrap();
    assert!(!Arc::ptr_eq(&clang, &llvm));
    assert_ne!(clang.compiler.cache_key(), llvm.compiler.cache_key());
}
