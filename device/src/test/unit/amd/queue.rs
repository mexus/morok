use super::*;

#[test]
fn aql_packet_header_layout() {
    let h = kernel_dispatch_header();
    // AQL packet header = TYPE_KERNEL_DISPATCH | barrier | sys-acq | sys-rel
    let expected: u16 = 2 | (1 << 8) | (HSA_FENCE_SCOPE_SYSTEM << 9) | (HSA_FENCE_SCOPE_SYSTEM << 11);
    assert_eq!(h, expected);
}

#[test]
fn aql_packet_is_64_bytes() {
    assert_eq!(size_of::<HsaKernelDispatchPacket>(), AQL_PACKET_BYTES);
}

#[test]
fn build_dispatch_picks_correct_dims() {
    let p1 = build_dispatch_packet([64, 1, 1], [1024, 1, 1], 0, 0, 0, 0, 0);
    assert_eq!(p1.setup & 0b11, 1);
    let p2 = build_dispatch_packet([8, 8, 1], [256, 256, 1], 0, 0, 0, 0, 0);
    assert_eq!(p2.setup & 0b11, 2);
    let p3 = build_dispatch_packet([4, 4, 4], [64, 64, 64], 0, 0, 0, 0, 0);
    assert_eq!(p3.setup & 0b11, 3);
}

#[test]
fn sdma_linear_copy_dwords_layout() {
    let dw = build_sdma_linear_copy(0x1_0000_2000, 0x2_0000_3000, 4096);
    assert_eq!(dw[0], 0x01);
    assert_eq!(dw[1], 4095);
    assert_eq!(dw[3], 0x0000_2000);
    assert_eq!(dw[4], 0x0000_0001);
    assert_eq!(dw[5], 0x0000_3000);
    assert_eq!(dw[6], 0x0000_0002);
}

/// Live compute queue creation (exercises the KFD CREATE_QUEUE path).
/// Skipped without a supported AMD GPU. A real dispatch needs the device
/// timeline wired up by the factory, so we only assert creation here.
#[test]
fn compute_queue_create_if_hw_supports() {
    let alloc = match AmdAllocator::new(0) {
        Ok(a) => a,
        Err(_) => return,
    };
    let _q = AmdComputeQueue::create(&alloc).expect("create compute queue");
}

/// On real AQL hardware (multi-XCC CDNA), `set_aql_scratch` must land the
/// scratch descriptor at the right `amd_queue_t` offsets in the GART page the
/// firmware reads. Exercises the offsets + volatile writes end-to-end against a
/// live queue; on PM4 hardware the queue has no descriptor and the write is a
/// no-op (we skip the assertion there).
#[test]
fn set_aql_scratch_round_trips_through_gart() {
    let alloc = match AmdAllocator::new(0) {
        Ok(a) => a,
        Err(_) => return,
    };
    let q = AmdComputeQueue::create(&alloc).expect("create compute queue");
    if q.is_pm4() {
        return; // PM4 queues program scratch via registers; no GART descriptor.
    }
    // A realistic descriptor, sized exactly as a 256-byte/thread scratch alloc
    // would be on this device.
    let (va, _size, tmpring, _rounded, _handle, desc) =
        crate::amd::device::alloc_scratch(alloc.dev.core().iface(), &alloc.dev.node, &alloc.dev.arch, 256)
            .expect("alloc scratch");
    assert_ne!(desc, crate::amd::device::AqlScratchDesc::default(), "CDNA must synthesize a descriptor");
    q.set_aql_scratch(&desc);
    assert_eq!(q.read_aql_scratch(), desc, "GART descriptor must match what we wrote");
    // Sanity: the descriptor points at the freshly allocated scratch buffer.
    assert_eq!(desc.backing_va, va);
    assert_eq!(desc.tmpring_size, tmpring);
    // Free the scratch we allocated for the test.
    alloc.dev.core().iface().free_raw(va, _size, _handle);
}
