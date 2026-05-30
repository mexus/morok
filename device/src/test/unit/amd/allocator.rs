use super::*;
use crate::error::Error;

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
    let alloc = match AmdAllocator::new(0) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("skipping: AmdAllocator::new failed (no supported AMD GPU on this host)");
            return;
        }
    };
    let opts = BufferSpec { cpu_access: true, ..Default::default() };
    let buf = alloc.alloc(4096, &opts, /*zero=*/ true).expect("alloc 4 KiB");
    assert_eq!(buf.size(), 4096);
    assert!(buf.cpu_accessible());
    alloc.free(buf, 4096, &opts);
}
