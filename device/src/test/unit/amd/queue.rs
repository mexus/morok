
use super::*;

#[test]
fn aql_packet_header_matches_tinygrad() {
    let h = kernel_dispatch_header();
    // tinygrad AQL_HDR = TYPE_KERNEL_DISPATCH | barrier | sys-acq | sys-rel
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

#[test]
fn barrier_packet_packs_dep_signals() {
    let p = build_barrier_packet(&[0xdead_beef, 0xcafe_babe], 7, 0xfeed_face);
    let words: &[u64] = unsafe { std::slice::from_raw_parts(&p as *const _ as *const u64, 8) };
    assert_eq!(words[1], 0xdead_beef);
    assert_eq!(words[2], 0xcafe_babe);
    assert_eq!(words[6], 7);
    assert_eq!(words[7], 0xfeed_face);
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
