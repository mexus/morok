//! Ownership probe: take the VF from the GIM via the mailbox, then prove the
//! RLCG indirect path works (direct GC reads return 0xffffffff on a VF).
//!
//! DESTRUCTIVE — requires amdgpu unbound first:
//!     device/tools/am_unbind.sh
//!     sudo ./target/debug/examples/am_own
//!     device/tools/am_rebind.sh     # restore KFD
//!
//! Read-only against device state otherwise: it only echoes scratch registers
//! and reads GRBM_STATUS; it does not program GMC/queues yet.

use svod_device::amd::am::discovery::{self, Discovery};
use svod_device::amd::am::mailbox::Mailbox;
use svod_device::amd::am::pci::PciDevice;
use svod_device::amd::am::regs;
use svod_device::amd::am::rlcg::RlcgChannel;

fn main() {
    let bdf = std::env::args().nth(1).unwrap_or_else(|| PciDevice::discover().expect("AMD GPU"));
    if std::fs::metadata(format!("/sys/bus/pci/devices/{bdf}/driver")).is_ok() {
        eprintln!("WARNING: a driver is still bound to {bdf}; run device/tools/am_unbind.sh first");
    }
    let mut dev = PciDevice::open(&bdf, false).expect("open read-write (needs root + amdgpu unbound)");
    dev.enable_bus_master();

    // Bootstrap: on a VF the framebuffer (hence the discovery table) is gated
    // until the GIM grants access via the mailbox, so the mailbox uses the
    // hardcoded NBIO base and runs FIRST.
    let mbox = Mailbox::new(svod_device::amd::am::mailbox::NBIO_7_9_SEG2_BASE);
    println!("is_vf = {}", mbox.is_vf(&dev.mmio));
    let csum = mbox.request_init_access(&dev.mmio).expect("init access handshake");
    println!("GIM granted init access (checksum {csum:#x})");

    // FB now readable — parse discovery and confirm the hardcoded NBIO base.
    let vram_size = (dev.mmio.read_u32(discovery::MM_RCC_CONFIG_MEMSIZE) as u64) << 20;
    let mut tbl = vec![0u8; discovery::TABLE_SIZE];
    dev.vram.read_bytes((vram_size - discovery::TABLE_TAIL_OFFSET) as usize, &mut tbl);
    let d = Discovery::parse(&tbl, vram_size).expect("discovery");
    assert_eq!(
        d.regs_offset[&discovery::NBIO_HWIP][&0][2],
        svod_device::amd::am::mailbox::NBIO_7_9_SEG2_BASE,
        "discovered NBIO seg2 base disagrees with the bootstrap constant"
    );
    println!("discovery OK: {} XCC, vram {} MiB", d.xccs(), vram_size >> 20);

    // RLCG echo on XCC0: write a sentinel to SCRATCH_REG5 and read it back.
    // SCRATCH_REG5 (raw header offset 0x2045, GC segment 1) is unused by the
    // RLCG channel (0..3) and by the AM boot flags (6/7), so it is safe scratch.
    let gc = regs::select("gc", d.ip_ver[&discovery::GC_HWIP]).unwrap();
    let bases0 = &d.regs_offset[&discovery::GC_HWIP][&0];
    let ch0 = RlcgChannel::new(bases0);
    let sr5 = bases0[1] as usize + 0x2045;
    for sentinel in [0xdead_beefu32, 0x0bad_f00d, 0] {
        ch0.write(&dev.mmio, sr5, sentinel).expect("rlcg write");
        let got = ch0.read(&dev.mmio, sr5).expect("rlcg read");
        println!(
            "SCRATCH_REG5 echo: wrote {sentinel:#010x} read {got:#010x} {}",
            if got == sentinel { "OK" } else { "MISMATCH" }
        );
        assert_eq!(got, sentinel, "RLCG scratch echo mismatch");
    }

    // GRBM_STATUS via RLCG should now read a sane value (not 0xffffffff) on
    // every XCC.
    let grbm = regs::find(gc, "regGRBM_STATUS").unwrap();
    for xcc in 0..d.xccs() as u16 {
        let bases = &d.regs_offset[&discovery::GC_HWIP][&xcc];
        let ch = RlcgChannel::new(bases);
        let v = ch.read(&dev.mmio, grbm.dword_index(bases)).expect("rlcg grbm read");
        let direct = dev.mmio.read_u32(grbm.dword_index(bases));
        println!("XCC{xcc} GRBM_STATUS: rlcg={v:#010x} direct={direct:#010x}");
        assert_ne!(v, 0xffff_ffff, "RLCG read still gated on XCC{xcc}");
    }

    mbox.release_init_access(&dev.mmio).expect("release");
    println!("M1 handshake + RLCG validated; released access");
}
