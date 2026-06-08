//! M0 read-only AM bring-up probe: map the BARs alongside a bound amdgpu,
//! parse IP discovery, and dump the registers we can read directly.
//!
//!     sudo -E cargo run -p svod-device --example am_discovery [BDF]

use svod_device::amd::am::discovery::{self, Discovery};
use svod_device::amd::am::pci::PciDevice;
use svod_device::amd::am::regs;

fn main() {
    let bdf = std::env::args().nth(1).unwrap_or_else(|| PciDevice::discover().expect("AMD GPU"));
    let dev = PciDevice::open(&bdf, true).expect("open read-only");
    println!(
        "{bdf}: BAR0 {} GiB, BAR2 {} KiB, BAR5 {} KiB",
        dev.vram.len() >> 30,
        dev.doorbell.len() >> 10,
        dev.mmio.len() >> 10
    );

    let vram_size = (dev.mmio.read_u32(discovery::MM_RCC_CONFIG_MEMSIZE) as u64) << 20;
    println!("RCC_CONFIG_MEMSIZE = {} MiB, large_bar = {}", vram_size >> 20, dev.vram.len() as u64 >= vram_size);
    assert!(dev.vram.len() as u64 >= vram_size, "M0 requires large BAR for direct table reads");

    let mut tbl = vec![0u8; discovery::TABLE_SIZE];
    dev.vram.read_bytes((vram_size - discovery::TABLE_TAIL_OFFSET) as usize, &mut tbl);
    let d = Discovery::parse(&tbl, vram_size).expect("discovery parse");

    println!("xccs = {}, vmhubs = {}, reserved_vram = {} MiB", d.xccs(), d.vmhubs(), d.reserved_vram_size >> 20);
    for (hwip, ver) in &d.ip_ver {
        let insts = d.regs_offset[hwip].len();
        println!("  hwip {hwip:>2}: v{}.{}.{} x{insts} bases[0]={:#x?}", ver.0, ver.1, ver.2, d.regs_offset[hwip][&0]);
    }
    println!(
        "gc_info: se={} cu/sh={} sh/se={} wave={} waves/simd={} scratch/cu={} lds={}KB",
        d.gc.num_se,
        d.gc.num_cu_per_sh,
        d.gc.num_sh_per_se,
        d.gc.wave_size,
        d.gc.max_waves_per_simd,
        d.gc.max_scratch_slots_per_cu,
        d.gc.lds_size_kb
    );

    // Direct dword reads of known registers across the IPs we vendored.
    let rd = |hwip: u16, prefix: &str, name: &str, inst: u16| {
        let def = regs::find(regs::select(prefix, d.ip_ver[&hwip]).expect(prefix), name).expect(name);
        let idx = def.dword_index(&d.regs_offset[&hwip][&inst]);
        if idx * 4 >= dev.mmio.len() {
            println!("  {name}[{inst}] @{idx:#x} beyond BAR5 — needs indirect (M1)");
            return None;
        }
        let v = dev.mmio.read_u32(idx);
        println!("  {name}[{inst}] @{idx:#x} = {v:#010x}");
        Some(v)
    };

    rd(discovery::MP0_HWIP, "mp", "regMP0_SMN_C2PMSG_81", 0); // SCRATCH_REG7 analog dump
    for inst in 0..d.xccs() as u16 {
        rd(discovery::GC_HWIP, "gc", "regGRBM_STATUS", inst);
    }
    rd(discovery::GC_HWIP, "gc", "regCP_MEC_CNTL", 0);
    rd(discovery::GC_HWIP, "gc", "regGCVM_CONTEXT0_CNTL", 0);
    rd(discovery::GC_HWIP, "gc", "regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 0);
    rd(discovery::MMHUB_HWIP, "mmhub", "regMMMC_VM_FB_LOCATION_BASE", 0);
    rd(discovery::MMHUB_HWIP, "mmhub", "regMMMC_VM_FB_LOCATION_TOP", 0);
    rd(discovery::OSSSYS_HWIP, "osssys", "regIH_RB_CNTL", 0);
    rd(discovery::SDMA0_HWIP, "sdma", "regSDMA_GFX_RB_CNTL", 0);
    // VF mailbox regs are absent from the bare-metal AM tables (vendored from
    // kernel nbio_7_9_0 headers in M1); a direct NBIO read still validates the
    // segment bases.
    rd(discovery::NBIO_HWIP, "nbio", "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", 0);
    println!("M0 read-only probe complete");
}
