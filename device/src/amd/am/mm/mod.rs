//! AM memory manager: TLSF sub-allocators + GMMU page tables.
//!
//! Pure logic — no MMIO. The page-table entries are written through an
//! injectable backing store (a plain buffer in tests, BAR-mapped VRAM in the
//! real driver), so VA allocation, PTE/PDE encoding, huge-page selection, and
//! the table walk are all unit-testable on any host. Port of tinygrad's
//! `runtime/support/memory.py` + the AM page-table bits in `am/amdev.py`.

pub mod pagetable;
pub mod tlsf;
