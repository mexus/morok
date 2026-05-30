use super::*;

const GFX11: IpVer = (11, 0, 0);

#[test]
fn gfx11_geometry_layout() {
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
