use crate::amd::device::*;
use crate::amd::iface::{AmdIface, QueueTeardown, RingDesc};
use crate::error::Error;
use std::sync::Arc;
use svod_dtype::AmdArch;

use super::test_support::{MockAmdCall, MockAmdIface};

/// On hosts without `/dev/kfd` (or without a supported GPU), `open` must
/// surface a clean `Err` — never panic.
#[test]
fn open_without_gpu_or_unsupported_arch_is_clean_err() {
    let result = AmdDevice::open(0);
    match result {
        Ok(_) => {
            // Host has a supported AMD GPU — exercise the happy path too.
            // (We can't assert much without hardware-specific data.)
        }
        Err(Error::NoAmdGpu { .. }) | Err(Error::AmdAllocFailed { .. }) | Err(Error::AmdIoctl { .. }) => {
            // All acceptable.
        }
        Err(e) => panic!("unexpected error variant: {e:?}"),
    }
}

#[test]
fn aql_scratch_descriptor_gfx9_encoding() {
    // gfx9 SQ_BUF_RSRC scratch descriptor layout:
    //   WORD0 = lo32(va)
    //   WORD1 = hi32(va)[15:0] | SWIZZLE_ENABLE(bit31)
    //   WORD2 = lo32(size_per_xcc)   (NUM_RECORDS)
    //   WORD3 = SQ_BUF_RSRC: DST_SEL=XYZW, NUM_FORMAT=UINT, DATA_FORMAT=32,
    //           ELEMENT_SIZE=1, INDEX_STRIDE=3, ADD_TID_ENABLE=1 = 0x00EA4FAC
    let va: u64 = 0x1234_5678_9abc_d000;
    let d = AqlScratchDesc::gfx9(va, 0x0004_0000, 0xDEAD, 256);
    assert_eq!(d.resource_descriptor[0], 0x9abc_d000);
    assert_eq!(d.resource_descriptor[1], 0x8000_5678); // (0x12345678 & 0xFFFF) | 0x80000000
    assert_eq!(d.resource_descriptor[2], 0x0004_0000);
    assert_eq!(d.resource_descriptor[3], 0x00EA_4FAC);
    assert_eq!(d.backing_va, va);
    assert_eq!(d.tmpring_size, 0xDEAD);
    assert_eq!(d.wave64_lane_byte_size, 256); // wave64: priv_seg * 64 / 64
}

/// Exclusive lanes fill the bounded hardware pool, then become available for
/// atomic reuse only after their non-clone lease drops.
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU"]
fn queue_leases_are_exclusive_and_reused() {
    use std::collections::HashSet;
    let Some(alloc) = super::test_support::amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 256).expect("signal pool"));
    }
    let n = core.hw_queues();
    assert!(n >= 1, "hw_queues must be >= 1");
    let mut leases: Vec<_> = (0..n).map(|_| core.lease_queue(&alloc).expect("lease")).collect();
    let distinct: HashSet<_> = leases.iter().map(|lease| lease.queue_ptr()).collect();
    assert_eq!(distinct.len(), n, "simultaneous leases must own distinct queues");
    let released = leases.pop().unwrap().queue_ptr();
    let reused = core.lease_queue(&alloc).expect("reuse released lane");
    assert_eq!(reused.queue_ptr(), released, "released lane must be reused without creating another queue");
}

#[test]
fn pack_tmpring_wavesize_width_by_arch() {
    // wave_scratch=0x3FFFF: cdna(13b) truncates, rdna3(15b) truncates, rdna4(18b) keeps it.
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx942) >> 12, 0x1FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1100) >> 12, 0x7FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1200) >> 12, 0x3FFFF);
    assert_eq!(pack_tmpring(0xABC, 0, &AmdArch::Gfx1100) & 0xFFF, 0xABC);
}

#[test]
fn mock_queue_setup_and_teardown_are_accounted_and_scripted() {
    let iface = Arc::new(MockAmdIface::default());
    let _device = iface.device();
    let desc = RingDesc {
        ring_gpu: 0x1000,
        gart_gpu: 0x2000,
        wptr_offset: 0,
        rptr_offset: 8,
        eop_gpu: 0,
        eop_size: 0,
        ctx_gpu: 0,
        ctx_save_restore_size: 0,
        ctl_stack_size: 0,
        ring_size: 0x4000,
        gpu_id: 1,
        queue_type: 2,
    };

    iface.script_setup(Err(Error::AmdIoctl { ioctl: "mock setup", errno: 5 }));
    assert!(matches!(iface.setup_ring(&desc), Err(Error::AmdIoctl { ioctl: "mock setup", errno: 5 })));
    let queue = iface.setup_ring(&desc).expect("default setup");
    assert_eq!(queue.doorbell_base.as_ptr() as usize % 0x1000, 0);
    assert_eq!(iface.queue_setup_count(), 1);
    assert_eq!(iface.live_queue_count(), 1);

    iface.script_teardown(Err(Error::AmdIoctl { ioctl: "mock teardown", errno: 16 }));
    assert!(matches!(
        iface.teardown_ring(queue.queue_id, queue.doorbell_base),
        Err(Error::AmdIoctl { ioctl: "mock teardown", errno: 16 })
    ));
    assert_eq!(iface.live_queue_count(), 1);
    iface.script_teardown(Ok(QueueTeardown::Complete));
    assert_eq!(iface.teardown_ring(queue.queue_id, queue.doorbell_base).unwrap(), QueueTeardown::Complete);
    assert_eq!(iface.queue_teardown_count(), 1);
    assert_eq!(iface.live_queue_count(), 0);
    assert_eq!(iface.transcript().iter().filter(|call| matches!(call, MockAmdCall::SetupRing { .. })).count(), 2);
    assert_eq!(iface.transcript().iter().filter(|call| matches!(call, MockAmdCall::TeardownRing { .. })).count(), 2);
}
