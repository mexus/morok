use super::*;

#[test]
fn version_round_trip() {
    for arch in [AmdArch::Gfx942, AmdArch::Gfx1100, AmdArch::Gfx1201] {
        let s = arch.mcpu();
        assert_eq!(AmdArch::parse(s), Some(arch));
    }
}

#[test]
fn family_predicates() {
    assert!(AmdArch::Gfx942.is_cdna() && !AmdArch::Gfx942.is_rdna3());
    assert!(AmdArch::Gfx1100.is_rdna3() && !AmdArch::Gfx1100.is_cdna());
    assert!(AmdArch::Gfx1201.is_rdna4() && !AmdArch::Gfx1201.is_rdna3());
    assert!(AmdArch::Gfx1100.has_matrix_cores());
    assert!(AmdArch::Gfx942.has_matrix_cores());
    // RDNA3.5 (Strix Halo) is its own family: not RDNA3, not CDNA, gfx11, wave32,
    // but still a WMMA matrix-core part.
    assert!(AmdArch::Gfx1151.is_rdna3_5() && !AmdArch::Gfx1151.is_rdna3() && !AmdArch::Gfx1151.is_cdna());
    assert_eq!(AmdArch::Gfx1151.gfx_major(), 11);
    assert_eq!(AmdArch::Gfx1151.wave_size(), 32);
    assert!(AmdArch::Gfx1151.has_matrix_cores());
}

#[test]
fn wave_size_by_family() {
    assert_eq!(AmdArch::Gfx942.wave_size(), 64);
    assert_eq!(AmdArch::Gfx1100.wave_size(), 32);
    assert_eq!(AmdArch::Gfx1200.wave_size(), 32);
}

#[test]
fn from_kfd_version() {
    assert_eq!(AmdArch::from_gfx_target_version(110_000), Some(AmdArch::Gfx1100));
    assert_eq!(AmdArch::from_gfx_target_version(90_402), Some(AmdArch::Gfx942));
    assert_eq!(AmdArch::from_gfx_target_version(0), None);
}
