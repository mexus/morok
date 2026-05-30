
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
fn pack_tmpring_wavesize_width_by_arch() {
    // wave_scratch=0x3FFFF: cdna(13b) truncates, rdna3(15b) truncates, rdna4(18b) keeps it.
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx942) >> 12, 0x1FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1100) >> 12, 0x7FFF);
    assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1200) >> 12, 0x3FFFF);
    assert_eq!(pack_tmpring(0xABC, 0, &AmdArch::Gfx1100) & 0xFFF, 0xABC);
}
