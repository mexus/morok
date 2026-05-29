//! Two-Level Segregated Fit (TLSF) allocator — port of tinygrad's
//! `TLSFAllocator` (`runtime/support/memory.py`).
//!
//! The AM driver uses three instances of this: the process-global virtual
//! address space, the per-device physical-VRAM pool, and the page-table pool.
//! It is pure bookkeeping over an integer address range (no GPU), so the whole
//! thing is unit- and property-tested here.
//!
//! Two levels of free-list buckets:
//!   * level 1 (`fl`) = the most-significant bit position of the block size;
//!   * level 2 (`sl`) splits each level-1 span into `lv2_cnt` sub-buckets.
//!
//! `alloc` finds the smallest block that fits (its bucket, then upward); `free`
//! coalesces with free neighbours. `blocks` is an address-ordered intrusive
//! doubly-linked list (`start -> {size, next, prev, is_free}`).

use std::collections::HashMap;

/// One contiguous region in the address-ordered block list. `next`/`prev` are
/// the start offsets of the neighbouring regions (`None` at the ends); `next`
/// equals `start + size` and may point one-past-the-end (not a live key).
#[derive(Clone, Copy, Debug)]
struct Block {
    size: u64,
    next: Option<u64>,
    prev: Option<u64>,
    is_free: bool,
}

/// `64 - leading_zeros` — the 1-based position of the most-significant set bit
/// (Python's `int.bit_length`). `bit_length(0) == 0`, `bit_length(16) == 5`.
#[inline]
fn bit_length(x: u64) -> u32 {
    u64::BITS - x.leading_zeros()
}

/// Round `x` up to a multiple of `m` (must be a power of two).
#[inline]
fn round_up_pow2(x: u64, m: u64) -> u64 {
    debug_assert!(m.is_power_of_two());
    (x + m - 1) & !(m - 1)
}

pub struct TlsfAllocator {
    size: u64,
    base: u64,
    block_size: u64,
    /// `bit_length(lv2_cnt)` — the second-level index width. With the default
    /// `lv2_cnt = 16` this is **5**, not 16 and not 4 (matches tinygrad's
    /// `self.l2_cnt = lv2_cnt.bit_length()`). Iteration over level-2 buckets
    /// runs `0..(1 << l2_cnt)`.
    l2_cnt: u32,
    /// `storage[fl][sl]` → list of free-block start offsets in that bucket.
    storage: Vec<Vec<Vec<u64>>>,
    /// Count of free blocks per level-1 bucket (fast "is this `fl` empty").
    lv1_entries: Vec<usize>,
    /// `start_offset -> Block`. All keys are base-relative.
    blocks: HashMap<u64, Block>,
}

impl TlsfAllocator {
    /// Default minimum-block / second-level-count (tinygrad's defaults).
    pub const DEFAULT_BLOCK_SIZE: u64 = 16;
    pub const DEFAULT_LV2_CNT: u64 = 16;

    /// A TLSF arena covering `[base, base + size)`. `block_size` is the minimum
    /// allocation/alignment granularity; `lv2_cnt` the second-level fan-out.
    pub fn new(size: u64, base: u64, block_size: u64, lv2_cnt: u64) -> Self {
        let l2_cnt = bit_length(lv2_cnt);
        let sl_slots = 1usize << l2_cnt;
        let fl_count = bit_length(size) as usize + 1;
        let storage = vec![vec![Vec::new(); sl_slots]; fl_count];
        let lv1_entries = vec![0usize; fl_count];
        let mut a = Self { size, base, block_size, l2_cnt, storage, lv1_entries, blocks: HashMap::new() };
        a.blocks.insert(0, Block { size, next: None, prev: None, is_free: true });
        if size > 0 {
            a.insert_block(0, size, None);
        }
        a
    }

    /// A TLSF arena with the default block size (16) and `lv2_cnt` (16).
    pub fn with_defaults(size: u64, base: u64) -> Self {
        Self::new(size, base, Self::DEFAULT_BLOCK_SIZE, Self::DEFAULT_LV2_CNT)
    }

    #[inline]
    pub fn base(&self) -> u64 {
        self.base
    }
    #[inline]
    pub fn size(&self) -> u64 {
        self.size
    }

    #[inline]
    fn lv1(&self, size: u64) -> usize {
        bit_length(size) as usize
    }

    #[inline]
    fn lv2(&self, size: u64) -> usize {
        let bl = bit_length(size);
        ((size - (1u64 << (bl - 1))) >> bl.saturating_sub(self.l2_cnt)) as usize
    }

    /// Insert a free block. `prev = None` means "inherit the existing block's
    /// `prev`" (tinygrad's `prev is None` sentinel — it never sets `prev` to a
    /// genuine `None` after construction).
    fn insert_block(&mut self, start: u64, size: u64, prev: Option<u64>) {
        let prev = prev.or_else(|| self.blocks.get(&start).and_then(|b| b.prev));
        let (fl, sl) = (self.lv1(size), self.lv2(size));
        self.storage[fl][sl].push(start);
        self.lv1_entries[fl] += 1;
        self.blocks.insert(start, Block { size, next: Some(start + size), prev, is_free: true });
    }

    /// Remove a block from the free lists and mark it allocated. `prev` follows
    /// the same inherit-on-`None` rule as [`insert_block`].
    fn remove_block(&mut self, start: u64, size: u64, prev: Option<u64>) {
        let prev = prev.or_else(|| self.blocks.get(&start).and_then(|b| b.prev));
        let (fl, sl) = (self.lv1(size), self.lv2(size));
        let bucket = &mut self.storage[fl][sl];
        if let Some(pos) = bucket.iter().position(|&s| s == start) {
            bucket.remove(pos);
        }
        self.lv1_entries[fl] -= 1;
        self.blocks.insert(start, Block { size, next: Some(start + size), prev, is_free: false });
    }

    /// Split the free block at `start` (size `size`) into `[start, new_size)` +
    /// `[start + new_size, size - new_size)`, fixing the following block's back
    /// pointer.
    fn split_block(&mut self, start: u64, size: u64, new_size: u64) {
        let nxt = self.blocks[&start].next;
        debug_assert!(self.blocks[&start].is_free, "block must be free");
        self.remove_block(start, size, None);
        self.insert_block(start, new_size, None);
        self.insert_block(start + new_size, size - new_size, Some(start));
        if let Some(n) = nxt
            && let Some(b) = self.blocks.get_mut(&n)
        {
            b.prev = Some(start + new_size);
        }
    }

    /// Absorb every consecutive free right-neighbour into the block at `start`.
    fn merge_right(&mut self, start: u64) {
        debug_assert!(self.blocks[&start].is_free, "block must be free");
        let mut size = self.blocks[&start].size;
        let mut nxt = self.blocks[&start].next;
        while let Some(n) = nxt {
            match self.blocks.get(&n) {
                Some(blk) if blk.is_free => {
                    let blk_size = blk.size;
                    self.remove_block(start, size, None);
                    self.remove_block(n, blk_size, None);
                    size += blk_size;
                    self.insert_block(start, size, None);
                    // Drop the absorbed block; continue from its old `next`.
                    nxt = self.blocks.remove(&n).and_then(|b| b.next);
                }
                _ => break,
            }
        }
        if let Some(n) = nxt
            && let Some(b) = self.blocks.get_mut(&n)
        {
            b.prev = Some(start);
        }
    }

    /// Coalesce the block at `start` with free neighbours on both sides: walk
    /// left while free, then merge everything to the right.
    fn merge_block(&mut self, mut start: u64) {
        while let Some(x) = self.blocks[&start].prev {
            if self.blocks[&x].is_free {
                start = x;
            } else {
                break;
            }
        }
        self.merge_right(start);
    }

    /// Allocate `req_size` bytes aligned to `align` (a power of two ≥ 1).
    /// Returns the absolute address (`base`-relative offset + `base`), or `None`
    /// if no block fits. Mirrors tinygrad's `TLSFAllocator.alloc`.
    pub fn alloc(&mut self, req_size: u64, align: u64) -> Option<u64> {
        let align = align.max(1);
        let req_size = req_size.max(self.block_size);
        // Round the search size up to the next sub-bucket boundary so any block
        // in the chosen bucket is guaranteed to fit the request.
        let mut size = (req_size + align - 1).max(self.block_size);
        size = round_up_pow2(size, 1u64 << bit_length(size).saturating_sub(self.l2_cnt));
        let size_bl = bit_length(size) as usize;

        for l1 in self.lv1(size)..self.storage.len() {
            if self.lv1_entries[l1] == 0 {
                continue;
            }
            let sl_start = if l1 == size_bl { self.lv2(size) } else { 0 };
            for l2 in sl_start..(1usize << self.l2_cnt) {
                let Some(&first) = self.storage[l1][l2].first() else { continue };
                let mut start = first;
                let mut nsize = self.blocks[&start].size;
                debug_assert!(nsize >= size, "bucketed block must be large enough");
                // Alignment split: carve off a misaligned prefix.
                let new_start = round_up_pow2(start, align);
                if new_start != start {
                    self.split_block(start, nsize, new_start - start);
                    start = new_start;
                    nsize = self.blocks[&start].size;
                }
                // Size split: carve off the unused tail.
                if nsize > req_size {
                    self.split_block(start, nsize, req_size);
                }
                self.remove_block(start, req_size, None);
                return Some(start + self.base);
            }
        }
        None
    }

    /// Free an address previously returned by [`alloc`](Self::alloc).
    pub fn free(&mut self, addr: u64) {
        let start = addr - self.base;
        let size = self.blocks[&start].size;
        self.insert_block(start, size, None);
        self.merge_block(start);
    }
}

#[cfg(test)]
mod tests {
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
}
