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
