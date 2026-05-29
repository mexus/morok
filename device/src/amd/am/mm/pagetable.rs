//! GMMU page-table geometry + PTE/PDE bit encoding.
//!
//! Pure bit/arithmetic logic — no MMIO, no backing store — so it is fully
//! unit-tested against tinygrad's exact constants. Port of the AM page-table
//! pieces of tinygrad's `runtime/support/am/{amdev,ip}.py` + `memory.py`.
//!
//! **gfx11 / RDNA3 is implemented and tested.** The arch split is by the
//! `ip_ver` tuple read from IP discovery (tinygrad's `self.ip_ver[GC_HWIP]`),
//! exactly where gfx9 (VG10) and gfx12 differ — those branches are marked and
//! filled in when their hardware can validate them. The page-table *shape*
//! (4-level / 48-bit) is shared across gfx9/11/12, so geometry does not branch.

/// `64 - leading_zeros` (Python `int.bit_length`). `bit_length(0) == 0`.
#[inline]
fn bit_length(x: u64) -> u32 {
    u64::BITS - x.leading_zeros()
}

// ── PTE/PDE flag bits (`am.py` AMDGPU_PTE_* / AMDGPU_PDE_*) ──────────────────
pub const PTE_VALID: u64 = 1 << 0;
pub const PTE_SYSTEM: u64 = 1 << 1;
pub const PTE_SNOOPED: u64 = 1 << 2;
pub const PTE_TMZ: u64 = 1 << 3;
pub const PTE_EXECUTABLE: u64 = 1 << 4;
pub const PTE_READABLE: u64 = 1 << 5;
pub const PTE_WRITEABLE: u64 = 1 << 6;
/// Huge-page / "this PDE slot is actually a PTE" bit (gfx10/11).
pub const PDE_PTE: u64 = 1 << 54;

/// Fragment field is bits 7..=11 (5 bits): `(frag & 0x1f) << 7`.
#[inline]
pub fn pte_frag(frag: u32) -> u64 {
    ((frag as u64) & 0x1f) << 7
}

/// MTYPE field for gfx10/11 (NV10) is bits 48..=50.
const MTYPE_NV10_SHIFT: u32 = 48;
/// gfx11 uncached memory type (`soc_11.MTYPE_UC`).
const MTYPE_UC_GFX11: u64 = 3;

/// Physical-address field: bits 12..=47 (page-aligned, 36 bits).
pub const PADDR_MASK: u64 = 0x0000_FFFF_FFFF_F000;

// ── Page-table level indices (`am.AMDGPU_VM_*`) ─────────────────────────────
pub const VM_PDB2: usize = 0; // root
pub const VM_PDB1: usize = 1;
pub const VM_PDB0: usize = 2;
pub const VM_PTB: usize = 3; // leaf (4 KiB)

/// An AMD IP-block version tuple `(major, minor, revision)` — the source of
/// truth for every arch decision (read from IP discovery at boot).
pub type IpVer = (u8, u8, u8);

#[inline]
fn ip_ge(v: IpVer, major: u8) -> bool {
    v.0 >= major
}

/// 4-level / 48-bit page-table geometry (gfx9/11/12 all share this shape).
/// `pte_covers[lv]` = bytes one entry at level `lv` maps; `pte_cnt[lv]` =
/// entries at that level. Port of `MemoryManager.__init__`'s `pte_covers` /
/// `pte_cnt` setup for `va_shifts=[12,21,30,39]`, `va_bits=48`.
#[derive(Clone, Debug)]
pub struct Geometry {
    pub pte_covers: Vec<u64>,
    pub pte_cnt: Vec<u64>,
    pub level_cnt: usize,
    pub va_bits: u32,
}

impl Geometry {
    /// Build from `va_shifts` (per-level low bit, ascending) + total `va_bits`.
    /// gfx11: `Geometry::new(&[12, 21, 30, 39], 48)`.
    pub fn new(va_shifts: &[u32], va_bits: u32) -> Self {
        // lvl_msb = va_shifts ++ [va_bits + 1]; covers = (1<<shift) reversed;
        // cnt = (1 << (msb[i+1]-msb[i])) reversed.
        let mut lvl_msb: Vec<u32> = va_shifts.to_vec();
        lvl_msb.push(va_bits + 1);
        let mut pte_covers: Vec<u64> = va_shifts.iter().map(|&s| 1u64 << s).collect();
        pte_covers.reverse();
        let mut pte_cnt: Vec<u64> = (0..lvl_msb.len() - 1).map(|i| 1u64 << (lvl_msb[i + 1] - lvl_msb[i])).collect();
        pte_cnt.reverse();
        Self { pte_covers, pte_cnt, level_cnt: va_shifts.len(), va_bits }
    }

    /// gfx11/RDNA3 geometry.
    pub fn gfx11() -> Self {
        Self::new(&[12, 21, 30, 39], 48)
    }

    /// The entry index at level `lv` for a (page-table-base-relative) `vaddr`:
    /// `(vaddr / pte_covers[lv]) % pte_cnt[lv]`.
    #[inline]
    pub fn pte_idx(&self, lv: usize, vaddr: u64) -> usize {
        ((vaddr / self.pte_covers[lv]) % self.pte_cnt[lv]) as usize
    }
}

/// Compose the PTE/PDE flag word (without the physical-address field) for a
/// table entry. Port of `AM_GMC.get_pte_flags`.
///
/// `is_table` = this is a PDE pointing at a child table (vs a leaf mapping).
/// `pte_lv` = the level the entry lives at. gfx11 sets the huge-page `PDE_PTE`
/// bit for any leaf placed above the 4 KiB PTB level (i.e. a 2 MiB leaf at
/// PDB0 or a 1 GiB leaf at PDB1).
#[allow(clippy::too_many_arguments)]
pub fn get_pte_flags(
    ip_ver: IpVer,
    pte_lv: usize,
    is_table: bool,
    frag: u32,
    uncached: bool,
    system: bool,
    snooped: bool,
    valid: bool,
) -> u64 {
    let mut extra = 0u64;
    if system {
        extra |= PTE_SYSTEM;
    }
    if snooped {
        extra |= PTE_SNOOPED;
    }
    if valid {
        extra |= PTE_VALID;
    }
    extra |= pte_frag(frag);
    if !is_table {
        extra |= PTE_WRITEABLE | PTE_READABLE | PTE_EXECUTABLE;
    }

    if ip_ge(ip_ver, 12) {
        // gfx12 (RDNA4): MTYPE at bit 54 (2-bit), PDE_PTE_GFX12 = 1<<63,
        // IS_PTE = 1<<63. Constants captured but not validated on hardware.
        unimplemented!("gfx12 PTE encoding — bring up with hardware validation against tinygrad AM_DEBUG");
    } else if ip_ge(ip_ver, 10) {
        // gfx10/11 (NV10): MTYPE bits 48..=50; huge-page bit = PDE_PTE (54).
        let mtype = if uncached { MTYPE_UC_GFX11 } else { 0 };
        extra |= mtype << MTYPE_NV10_SHIFT;
        if !is_table && pte_lv != VM_PTB {
            extra |= PDE_PTE;
        }
    } else {
        // gfx9 (VG10/CDNA): MTYPE at bit 57 (2-bit); PDB1 tables set BFS(0x9),
        // PDB0 tables set TF, leaves above PDB0 set PDE_PTE. Captured for the
        // arch-parametric structure but not validated on hardware.
        unimplemented!("gfx9 VG10 PTE encoding — bring up with hardware validation against tinygrad AM_DEBUG");
    }
    extra
}

/// Is `pte` (at level `pte_lv`) a leaf page rather than a child-table pointer?
/// Port of `AM_GMC.is_pte_huge_page` (gfx10/11 path = the `PDE_PTE` bit).
pub fn is_pte_huge_page(ip_ver: IpVer, _pte_lv: usize, pte: u64) -> bool {
    if ip_ge(ip_ver, 12) {
        unimplemented!("gfx12 huge-page bit — bring up with hardware validation");
    }
    if ip_ge(ip_ver, 10) {
        return pte & PDE_PTE != 0;
    }
    unimplemented!("gfx9 huge-page bit — bring up with hardware validation");
}

/// The full PTE/PDE word: flags `|` page-aligned physical address. Port of
/// `AMPageTableEntry.set_entry`'s `... | (paddr & 0x0000FFFFFFFFF000)`.
#[allow(clippy::too_many_arguments)]
pub fn encode_entry(
    ip_ver: IpVer,
    pte_lv: usize,
    paddr: u64,
    is_table: bool,
    uncached: bool,
    system: bool,
    snooped: bool,
    frag: u32,
    valid: bool,
) -> u64 {
    debug_assert_eq!(paddr & !PADDR_MASK & !0xFFF, 0, "paddr has bits above the 48-bit phys field");
    get_pte_flags(ip_ver, pte_lv, is_table, frag, uncached, system, snooped, valid) | (paddr & PADDR_MASK)
}

/// Extract the physical address a PTE/PDE points at (bits 12..=47).
#[inline]
pub fn entry_paddr(entry: u64) -> u64 {
    entry & PADDR_MASK
}

/// Is the entry valid (`PTE_VALID`)?
#[inline]
pub fn entry_valid(entry: u64) -> bool {
    entry & PTE_VALID != 0
}

/// TLB fragment exponent for a `[va, va+sz)` mapping: `log2(min(lowest set bit
/// of va, lowest set bit of sz)) - 12` (fragment 0 = 4 KiB). Port of
/// `MemoryManager._frag_size` (`must_cover=True`). Saturates at 0.
pub fn frag_size(va: u64, sz: u64) -> u32 {
    let va_pwr2 = if va > 0 { va & va.wrapping_neg() } else { 1u64 << 63 };
    let sz_pwr2 = sz & sz.wrapping_neg();
    bit_length(va_pwr2.min(sz_pwr2)).saturating_sub(13)
}

#[cfg(test)]
mod tests {
    use super::*;

    const GFX11: IpVer = (11, 0, 0);

    #[test]
    fn gfx11_geometry_matches_tinygrad() {
        let g = Geometry::gfx11();
        // pte_covers (level 0..3 = PDB2,PDB1,PDB0,PTB).
        assert_eq!(g.pte_covers, vec![1 << 39, 1 << 30, 1 << 21, 1 << 12]);
        // root (PDB2) is 1024 entries (48-39+1=10 bits); the rest 512.
        assert_eq!(g.pte_cnt, vec![1024, 512, 512, 512]);
        assert_eq!(g.level_cnt, 4);
        assert_eq!(g.va_bits, 48);
    }

    #[test]
    fn pte_idx_decomposes_va() {
        let g = Geometry::gfx11();
        // A VA picking index 1 at every level.
        let va = (1u64 << 39) | (1u64 << 30) | (1u64 << 21) | (1u64 << 12);
        assert_eq!(g.pte_idx(VM_PDB2, va), 1);
        assert_eq!(g.pte_idx(VM_PDB1, va), 1);
        assert_eq!(g.pte_idx(VM_PDB0, va), 1);
        assert_eq!(g.pte_idx(VM_PTB, va), 1);
        // PTB index wraps modulo 512.
        assert_eq!(g.pte_idx(VM_PTB, 512u64 << 12), 0);
        assert_eq!(g.pte_idx(VM_PTB, 513u64 << 12), 1);
    }

    #[test]
    fn gfx11_leaf_4k_pte_cached() {
        // A normal 4 KiB leaf at PTB, cached: VALID|R|W|X, MTYPE 0, no PDE_PTE.
        let f = get_pte_flags(GFX11, VM_PTB, false, 0, false, false, false, true);
        assert_eq!(f & PTE_VALID, PTE_VALID);
        assert_eq!(f & (PTE_READABLE | PTE_WRITEABLE | PTE_EXECUTABLE), PTE_READABLE | PTE_WRITEABLE | PTE_EXECUTABLE);
        assert_eq!(f & PDE_PTE, 0, "4K leaf at PTB must not set the huge-page bit");
        assert_eq!((f >> MTYPE_NV10_SHIFT) & 0x7, 0, "cached → MTYPE 0");
        assert_eq!(f & PTE_SYSTEM, 0);
    }

    #[test]
    fn gfx11_leaf_uncached_sets_mtype_uc() {
        let f = get_pte_flags(GFX11, VM_PTB, false, 0, true, false, false, true);
        assert_eq!((f >> MTYPE_NV10_SHIFT) & 0x7, MTYPE_UC_GFX11, "uncached → MTYPE_UC=3 in bits 48..=50");
    }

    #[test]
    fn gfx11_huge_leaf_sets_pde_pte() {
        // 2 MiB leaf at PDB0 and 1 GiB leaf at PDB1 both set the huge-page bit.
        let f2m = get_pte_flags(GFX11, VM_PDB0, false, 0, false, false, false, true);
        let f1g = get_pte_flags(GFX11, VM_PDB1, false, 0, false, false, false, true);
        assert_ne!(f2m & PDE_PTE, 0);
        assert_ne!(f1g & PDE_PTE, 0);
        assert!(is_pte_huge_page(GFX11, VM_PDB0, f2m));
        assert!(is_pte_huge_page(GFX11, VM_PDB1, f1g));
    }

    #[test]
    fn gfx11_pde_is_valid_only() {
        // A PDE (child-table pointer): VALID set, but no R/W/X and no PDE_PTE.
        let f = get_pte_flags(GFX11, VM_PDB2, true, 0, false, false, false, true);
        assert_eq!(f & PTE_VALID, PTE_VALID);
        assert_eq!(f & (PTE_READABLE | PTE_WRITEABLE | PTE_EXECUTABLE), 0, "PDE has no R/W/X");
        assert_eq!(f & PDE_PTE, 0, "PDE is not a huge page");
        assert!(!is_pte_huge_page(GFX11, VM_PDB2, f));
    }

    #[test]
    fn encode_entry_round_trips_paddr() {
        let paddr = 0x1234_5000u64; // page-aligned
        let e = encode_entry(GFX11, VM_PTB, paddr, false, false, false, false, 0, true);
        assert_eq!(entry_paddr(e), paddr);
        assert!(entry_valid(e));
        // A PDE pointing at a child table.
        let pde = encode_entry(GFX11, VM_PDB1, 0xABCD_0000, true, false, false, false, 0, true);
        assert_eq!(entry_paddr(pde), 0xABCD_0000);
        assert_eq!(pde & (PTE_READABLE | PTE_WRITEABLE | PTE_EXECUTABLE), 0);
    }

    #[test]
    fn system_and_snooped_bits() {
        let f = get_pte_flags(GFX11, VM_PTB, false, 0, false, true, true, true);
        assert_ne!(f & PTE_SYSTEM, 0);
        assert_ne!(f & PTE_SNOOPED, 0);
    }

    #[test]
    fn frag_field_packs_into_bits_7_11() {
        let f = get_pte_flags(GFX11, VM_PTB, false, 9, false, false, false, true);
        assert_eq!((f >> 7) & 0x1f, 9);
    }

    #[test]
    fn frag_size_matches_alignment() {
        // 4 KiB-aligned, 4 KiB region → fragment 0.
        assert_eq!(frag_size(0x1000, 0x1000), 0);
        // 2 MiB-aligned, 2 MiB region → fragment log2(2MiB)-12 = 21-12 = 9.
        assert_eq!(frag_size(2 << 20, 2 << 20), 9);
        // Alignment limited by the smaller of va/sz lowest set bit.
        assert_eq!(frag_size(0x1000, 2 << 20), 0); // va only 4K-aligned
        assert_eq!(frag_size(1 << 30, 0x1000), 0); // sz only 4K
    }

    #[test]
    #[should_panic(expected = "gfx12")]
    fn gfx12_pte_is_gated() {
        let _ = get_pte_flags((12, 0, 0), VM_PTB, false, 0, false, false, false, true);
    }

    #[test]
    #[should_panic(expected = "gfx9")]
    fn gfx9_pte_is_gated() {
        let _ = get_pte_flags((9, 4, 2), VM_PTB, false, 0, false, false, false, true);
    }
}
