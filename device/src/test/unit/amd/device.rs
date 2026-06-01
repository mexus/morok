use super::*;

/// On hosts without `/dev/kfd` (or without a supported GPU), `open` must
/// surface a clean `Err` — never panic.
#[test]
fn open_without_gpu_or_unsupported_arch_is_clean_err() {
    let result = AmdDevice::open(0);
    match result {
        Ok(_) => {
            // Host has a supported AMD GPU — exercise the happy path too.
            // (We can't assert much without hardware-specific data.)
        }
        Err(Error::NoAmdGpu { .. }) | Err(Error::AmdAllocFailed { .. }) | Err(Error::AmdIoctl { .. }) => {
            // All acceptable.
        }
        Err(e) => panic!("unexpected error variant: {e:?}"),
    }
}

#[test]
fn aql_scratch_descriptor_gfx9_encoding() {
    // gfx9 SQ_BUF_RSRC scratch descriptor layout:
    //   WORD0 = lo32(va)
    //   WORD1 = hi32(va)[15:0] | SWIZZLE_ENABLE(bit31)
    //   WORD2 = lo32(size_per_xcc)   (NUM_RECORDS)
    //   WORD3 = SQ_BUF_RSRC: DST_SEL=XYZW, NUM_FORMAT=UINT, DATA_FORMAT=32,
    //           ELEMENT_SIZE=1, INDEX_STRIDE=3, ADD_TID_ENABLE=1 = 0x00EA4FAC
    let va: u64 = 0x1234_5678_9abc_d000;
    let d = AqlScratchDesc::gfx9(va, 0x0004_0000, 0xDEAD, 256);
    assert_eq!(d.resource_descriptor[0], 0x9abc_d000);
    assert_eq!(d.resource_descriptor[1], 0x8000_5678); // (0x12345678 & 0xFFFF) | 0x80000000
    assert_eq!(d.resource_descriptor[2], 0x0004_0000);
    assert_eq!(d.resource_descriptor[3], 0x00EA_4FAC);
    assert_eq!(d.backing_va, va);
    assert_eq!(d.tmpring_size, 0xDEAD);
    assert_eq!(d.wave64_lane_byte_size, 256); // wave64: priv_seg * 64 / 64
}

#[test]
fn pack_tmpring_wavesize_width_by_arch() {
    // wave_scratch=0x3FFFF: cdna(13b) truncates, rdna3(15b) truncates, rdna4(18b) keeps it.
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx942) >> 12, 0x1FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1100) >> 12, 0x7FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1200) >> 12, 0x3FFFF);
    assert_eq!(pack_tmpring(0xABC, 0, &AmdArch::Gfx1100) & 0xFFF, 0xABC);
}
