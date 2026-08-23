use super::test_support::{MockAmdCall, MockAmdIface, MockFreeIssue, amd_alloc_or_skip};
use crate::allocator::{Allocator, AmdBufferGuard, BufferSpec, RawBuffer};
use crate::amd::allocator::*;
use crate::amd::iface::{AllocKind, AmdIface};
use crate::amd::va_registry::AllocTag;
use crate::error::Error;
use std::sync::Arc;

fn mock_allocator(iface: &Arc<MockAmdIface>) -> AmdAllocator {
    AmdAllocator { dev: iface.device(), device_id: 0 }
}

/// Construction either succeeds (real hardware + supported arch) or
/// returns a clean error variant; never panics.
#[test]
fn allocator_construction_is_clean() {
    match AmdAllocator::new(0) {
        Ok(_alloc) => {}
        Err(Error::NoAmdGpu { .. }) | Err(Error::AmdAllocFailed { .. }) | Err(Error::AmdIoctl { .. }) => {}
        Err(e) => panic!("unexpected error: {e:?}"),
    }
}

/// Live VRAM alloc → free round-trip. Skipped on hosts that can't open an
/// AmdDevice (no GPU, unsupported arch, missing perms).
#[test]
fn alloc_free_roundtrip_if_hw_supports() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let buf = alloc.alloc(4096, &opts, /*zero=*/ true).expect("alloc 4 KiB");
    assert_eq!(buf.size(), 4096);
    assert!(buf.cpu_accessible());
    alloc.free(buf, 4096, &opts);
}

#[test]
fn mock_raw_allocation_is_aligned_stable_and_freed_once() {
    let iface = Arc::new(MockAmdIface::default());
    let alloc = mock_allocator(&iface);
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let buffer = alloc.alloc(17, &opts, true).expect("mock allocation");
    let (gpu_addr, host_ptr, size) = match &buffer {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host_ptr), size, .. } => (*gpu_addr, *host_ptr, *size),
        other => panic!("unexpected buffer: {other:?}"),
    };
    assert_eq!(gpu_addr as usize % 0x1000, 0);
    assert_eq!(host_ptr.as_ptr() as u64, gpu_addr);
    assert_eq!(size, 0x1000);
    unsafe { host_ptr.as_ptr().write(0x5a) };
    assert_eq!(unsafe { host_ptr.as_ptr().read() }, 0x5a);
    assert_eq!(iface.allocation_count(), 1);
    assert_eq!(iface.live_handle_count(), 1);

    alloc.free(buffer, 17, &opts);
    assert_eq!(iface.free_count(), 1);
    assert_eq!(iface.live_handle_count(), 0);
    assert_eq!(
        iface.transcript().iter().map(std::mem::discriminant).collect::<Vec<_>>(),
        [
            std::mem::discriminant(&MockAmdCall::Alloc { size: 0, cpu_access: false, zero: false }),
            std::mem::discriminant(&MockAmdCall::Free { gpu_va: 0, size: 0, handle: 0 }),
        ]
    );
}

#[test]
fn construction_guard_reclaims_on_error_unwind() {
    fn fail_after_alloc(alloc: &AmdAllocator) -> crate::error::Result<()> {
        let options = BufferSpec::default();
        let _guard = AmdBufferGuard::new(alloc.alloc(64, &options, true)?);
        Err(Error::Runtime { message: "later construction step failed".into() })
    }

    let iface = Arc::new(MockAmdIface::default());
    let alloc = mock_allocator(&iface);
    assert!(fail_after_alloc(&alloc).is_err());
    assert_eq!(iface.allocation_count(), 1);
    assert_eq!(iface.free_count(), 1);
    assert_eq!(iface.live_handle_count(), 0);
}

#[test]
fn construction_guard_quarantines_when_device_is_poisoned() {
    let iface = Arc::new(MockAmdIface::default());
    let alloc = mock_allocator(&iface);
    let guard = AmdBufferGuard::new(alloc.alloc(64, &BufferSpec::default(), true).unwrap());
    alloc.dev.core().poison("synthetic fault");
    drop(guard);
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), 1);
}

#[test]
fn construction_guard_quarantines_during_panic() {
    let iface = Arc::new(MockAmdIface::default());
    let alloc = mock_allocator(&iface);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = AmdBufferGuard::new(alloc.alloc(64, &BufferSpec::default(), true).unwrap());
        panic!("synthetic construction panic");
    }));
    assert!(result.is_err());
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), 1);
}

#[test]
fn mock_detects_double_and_unknown_frees() {
    let iface = MockAmdIface::default();
    let allocation = iface.alloc_raw(1, AllocKind::UncachedGtt, AllocTag::Gtt, true, false).expect("allocation");
    iface.free_raw(allocation.gpu_va, allocation.size, allocation.handle);
    iface.free_raw(allocation.gpu_va, allocation.size, allocation.handle);
    iface.free_raw(0xdead_0000, 0x1000, 999);
    assert_eq!(
        iface.free_issues(),
        [
            MockFreeIssue::DoubleFree { handle: allocation.handle },
            MockFreeIssue::UnknownFree { gpu_va: 0xdead_0000, size: 0x1000, handle: 999 },
        ]
    );
}

#[test]
fn mock_allocation_outcomes_are_scripted_fifo() {
    let iface = Arc::new(MockAmdIface::default());
    iface.script_alloc(Err(Error::AmdIoctl { ioctl: "mock alloc", errno: 12 }));
    let alloc = mock_allocator(&iface);
    let error = alloc.alloc(64, &BufferSpec::default(), false).expect_err("scripted allocation failure");
    assert!(matches!(error, Error::AmdIoctl { ioctl: "mock alloc", errno: 12 }));
    let buffer = alloc.alloc(64, &BufferSpec::default(), false).expect("default scripted success");
    assert_eq!(iface.allocation_count(), 1);
    alloc.free(buffer, 64, &BufferSpec::default());
}
