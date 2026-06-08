//! AM memory manager: TLSF sub-allocators + GMMU page tables.
//!
//! Pure logic — no MMIO. The page-table entries are written through an
//! injectable [`PhysMem`] backing (a plain buffer in tests, BAR-mapped VRAM in
//! the real driver), so VA allocation, PTE/PDE encoding, huge-page selection,
//! and the table walk are all unit-testable on any host.

pub mod manager;
pub mod pagetable;
pub mod tlsf;

/// Physical-memory backing for the page tables (and zero-fill of fresh table
/// pages). The real driver maps this onto the GPU's VRAM BAR; tests back it
/// with an in-process buffer. Addresses are byte physical addresses; page-table
/// entries are 64-bit and 8-byte aligned.
pub trait PhysMem {
    /// Read the u64 at `paddr` (8-aligned). Unwritten memory reads as 0.
    fn read_u64(&self, paddr: u64) -> u64;
    /// Write the u64 at `paddr` (8-aligned).
    fn write_u64(&mut self, paddr: u64, val: u64);
    /// Zero `size` bytes at `paddr` (used when allocating a fresh table page).
    fn zero(&mut self, paddr: u64, size: u64);
}

/// [`PhysMem`] backed by the CPU-mapped VRAM BAR (BAR0). Page-table physical
/// addresses are 0-based VRAM offsets, which map directly into the BAR, so the
/// page tables are written straight through the aperture.
pub struct VramPhys {
    base: *mut u8,
    len: usize,
}

// SAFETY: the backing is a fixed MMIO mapping; concurrent table edits are
// serialized by the owning driver (one MemoryManager behind a mutex).
unsafe impl Send for VramPhys {}
unsafe impl Sync for VramPhys {}

impl VramPhys {
    /// `bar0_ptr`/`bar0_len` come from the mapped VRAM BAR. Safe to use for the
    /// whole BAR; the manager only ever touches addresses below the reserved
    /// page-table / data region.
    pub fn new(bar0_ptr: *mut u8, bar0_len: usize) -> Self {
        Self { base: bar0_ptr, len: bar0_len }
    }
}

impl PhysMem for VramPhys {
    fn read_u64(&self, paddr: u64) -> u64 {
        assert!((paddr as usize) + 8 <= self.len, "VRAM read_u64 out of bounds");
        unsafe { (self.base.add(paddr as usize) as *const u64).read_volatile() }
    }
    fn write_u64(&mut self, paddr: u64, val: u64) {
        assert!((paddr as usize) + 8 <= self.len, "VRAM write_u64 out of bounds");
        unsafe { (self.base.add(paddr as usize) as *mut u64).write_volatile(val) }
    }
    fn zero(&mut self, paddr: u64, size: u64) {
        assert!((paddr + size) as usize <= self.len, "VRAM zero out of bounds");
        unsafe { std::ptr::write_bytes(self.base.add(paddr as usize), 0, size as usize) }
    }
}

/// Where a mapping's physical pages live. gfx11 cares about `Phys` (device
/// VRAM) vs `Sys` (host system memory, sets the PTE SYSTEM bit).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddrSpace {
    Phys,
    Sys,
}

/// A live virtual mapping returned by [`manager::MemoryManager::valloc`] /
/// [`map_range`](manager::MemoryManager::map_range).
#[derive(Clone, Debug)]
pub struct VirtMapping {
    pub va_addr: u64,
    pub size: u64,
    /// The `(paddr, size)` physical segments backing the VA range.
    pub paddrs: Vec<(u64, u64)>,
    pub aspace: AddrSpace,
    pub uncached: bool,
    pub snooped: bool,
}
