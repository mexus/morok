
use super::*;
use proptest::prelude::*;
use std::collections::BTreeMap;

#[test]
fn l2_cnt_is_bit_length_not_count() {
    // The classic TLSF gotcha: l2_cnt = bit_length(16) = 5, so there are
    // 1<<5 = 32 second-level iteration slots (lv2 itself stays < 16).
    let a = TlsfAllocator::with_defaults(1 << 20, 0);
    assert_eq!(a.l2_cnt, 5);
    assert!(a.lv2(0xFFFF) < 16);
}

#[test]
fn bit_length_matches_python() {
    assert_eq!(bit_length(0), 0);
    assert_eq!(bit_length(1), 1);
    assert_eq!(bit_length(16), 5);
    assert_eq!(bit_length(0xFFFF), 16);
    assert_eq!(bit_length(1 << 43), 44);
}

#[test]
fn sequential_allocs_are_contiguous_and_based() {
    let mut a = TlsfAllocator::with_defaults(1024, 0x1000);
    let x = a.alloc(16, 1).unwrap();
    let y = a.alloc(16, 1).unwrap();
    assert_eq!(x, 0x1000); // base is applied
    assert_eq!(y, 0x1010); // 16 bytes after x
}

#[test]
fn alloc_respects_alignment() {
    let mut a = TlsfAllocator::with_defaults(1 << 20, 0);
    let _pad = a.alloc(16, 1).unwrap(); // push the cursor off zero
    let aligned = a.alloc(64, 4096).unwrap();
    assert_eq!(aligned % 4096, 0);
}

#[test]
fn exhaustion_returns_none() {
    let mut a = TlsfAllocator::with_defaults(64, 0);
    assert!(a.alloc(64, 1).is_some());
    assert!(a.alloc(16, 1).is_none());
}

#[test]
fn free_coalesces_back_to_whole_arena() {
    let mut a = TlsfAllocator::with_defaults(4096, 0);
    let p0 = a.alloc(1024, 1).unwrap();
    let p1 = a.alloc(1024, 1).unwrap();
    let p2 = a.alloc(2048, 1).unwrap();
    // Free out of order; everything must coalesce so the full arena allocs.
    a.free(p1);
    a.free(p0);
    a.free(p2);
    assert!(a.alloc(4096, 1).is_some(), "arena did not coalesce back to one block");
}

#[test]
fn freed_block_is_reused() {
    let mut a = TlsfAllocator::with_defaults(1 << 16, 0);
    let p = a.alloc(256, 1).unwrap();
    a.free(p);
    let q = a.alloc(256, 1).unwrap();
    assert_eq!(p, q, "freed block of the same size should be handed back");
}

proptest! {
    /// A random alloc/free workload never hands out overlapping or
    /// out-of-bounds regions, and never double-allocates a byte.
    #[test]
    fn no_overlaps_and_in_bounds(ops in prop::collection::vec((0u64..2048, 0usize..4, 0u64..8), 1..200)) {
        const SIZE: u64 = 1 << 16;
        const BASE: u64 = 0x4000;
        let mut a = TlsfAllocator::with_defaults(SIZE, BASE);
        // addr -> size of each live allocation.
        let mut live: BTreeMap<u64, u64> = BTreeMap::new();
        // a fifo of live addresses so "free" picks a real one.
        let mut order: Vec<u64> = Vec::new();

        for (raw_size, action, free_idx) in ops {
            if action == 0 && !order.is_empty() {
                // free
                let i = (free_idx as usize) % order.len();
                let addr = order.remove(i);
                let sz = live.remove(&addr).unwrap();
                // bounds sanity before freeing
                prop_assert!(addr >= BASE && addr + sz <= BASE + SIZE);
                a.free(addr);
            } else {
                // alloc
                let align = 1u64 << (free_idx.min(3) as u32); // 1,2,4,8
                let size = raw_size.max(1);
                if let Some(addr) = a.alloc(size, align) {
                    let alloc_sz = size.max(TlsfAllocator::DEFAULT_BLOCK_SIZE);
                    prop_assert!(addr >= BASE, "addr {addr:#x} below base");
                    prop_assert!(addr + alloc_sz <= BASE + SIZE, "alloc past arena end");
                    prop_assert_eq!(addr % align, 0, "alignment violated");
                    // no overlap with any live allocation
                    for (&oaddr, &osz) in &live {
                        let disjoint = addr + alloc_sz <= oaddr || oaddr + osz <= addr;
                        prop_assert!(disjoint, "overlap: [{addr:#x},+{alloc_sz:#x}) vs [{oaddr:#x},+{osz:#x})");
                    }
                    live.insert(addr, alloc_sz);
                    order.push(addr);
                }
            }
        }

        // After freeing everything, the whole arena must be allocatable again.
        for addr in order.drain(..) {
            a.free(addr);
        }
        prop_assert!(a.alloc(SIZE, 1).is_some(), "arena did not fully coalesce");
    }
}
