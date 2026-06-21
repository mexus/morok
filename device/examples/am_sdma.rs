//! SDMA copy: the real GMMU end-to-end test. Allocate VA-mapped VRAM
//! buffers (ring/src/dst/rptr), fill src with a pattern, have the SDMA engine
//! copy src→dst through vmid-0 VAs, and verify dst matches. If the page tables
//! or context are wrong, the engine faults instead of copying.
//!
//! DESTRUCTIVE — amdgpu unbound; recovery needs a VM reboot.
//!     sudo ./target/debug/examples/am_sdma

use svod_device::amd::am::dev::AmDev;
use svod_device::amd::am::ip::sdma::SdmaRing;
use svod_device::amd::am::pci::PciDevice;
use svod_device::amd::sys::sdma;

const BUF: u64 = 0x1000;

fn main() {
    let bdf = std::env::args().nth(1).unwrap_or_else(|| PciDevice::discover().expect("AMD GPU"));
    let mut dev = AmDev::open(&bdf).expect("AmDev::open");
    println!("AmDev up: {} XCC", dev.disc.xccs());
    {
        let r = dev.regs();
        println!(
            "FB_BASE={:#x} FB_TOP={:#x} XGMI_CNTL={:#x} XGMI_SIZE={:#x} -> mc_base={:#x}",
            r.read("mmhub", "regMMMC_VM_FB_LOCATION_BASE", 0).unwrap(),
            r.read("mmhub", "regMMMC_VM_FB_LOCATION_TOP", 0).unwrap(),
            r.read("mmhub", "regMMMC_VM_XGMI_LFB_CNTL", 0).unwrap(),
            r.read("mmhub", "regMMMC_VM_XGMI_LFB_SIZE", 0).unwrap(),
            dev.gmc.mc_base,
        );
        // vram_base_offset = MC_VM_FB_OFFSET<<24 (+xgmi): the real PTE base.
        println!(
            "GC_FB_OFFSET={:#x?} MM_FB_OFFSET={:#x} GC_FB_LOCATION_BASE={:#x?}",
            r.gc_read("regGCMC_VM_FB_OFFSET", 0),
            r.read("mmhub", "regMMMC_VM_FB_OFFSET", 0).unwrap(),
            r.gc_read("regGCMC_VM_FB_LOCATION_BASE", 0),
        );
        // Did our CONTEXT0 page-table base actually land on the MM hub, or is it
        // PF-gated (→ MM walks the PF's tables, not ours)? Compare to root_pt|1.
        let want = (dev.mm.root_pt() | 1) as u32;
        println!(
            "MM CONTEXT0 PT_BASE_LO={:#x} (want {want:#x}) START_LO={:#x} END_LO={:#x} | GC PT_BASE_LO={:#x?}",
            r.read("mmhub", "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 0).unwrap(),
            r.read("mmhub", "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR_LO32", 0).unwrap(),
            r.read("mmhub", "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR_LO32", 0).unwrap(),
            r.gc_read("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR_LO32", 0),
        );
    }

    let ring = dev.valloc(BUF, true).expect("ring");
    let src = dev.valloc(BUF, false).expect("src");
    let dst = dev.valloc(BUF, false).expect("dst");
    let rptr = dev.valloc(0x100, true).expect("rptr wb");
    let (ring_pa, src_pa, dst_pa, rptr_pa) = (ring.paddrs[0].0, src.paddrs[0].0, dst.paddrs[0].0, rptr.paddrs[0].0);
    println!(
        "ring va {:#x} -> pa {:#x}; resolve(ring_va)={:#x?} (want {ring_pa:#x})",
        ring.va_addr,
        ring_pa,
        dev.mm.resolve(ring.va_addr)
    );

    // Pattern into src; clear dst; both via BAR0.
    let pattern: Vec<u8> = (0..BUF as usize).map(|i| (i * 7 + 3) as u8).collect();
    dev.vram_write(src_pa, &pattern);
    dev.vram_write(dst_pa, &vec![0u8; BUF as usize]);
    // Sanity: does a CPU BAR0 write even round-trip at this paddr?
    let mut echo = vec![0u8; 16];
    dev.vram_read(src_pa, &mut echo);
    println!("src BAR0 write→read roundtrip: {:?} (want {:?})", &echo, &pattern[..16]);

    // Build the command stream: copy src→dst, then fence a sentinel to rptr+64.
    let done_va = rptr.va_addr + 64;
    let done_pa = rptr_pa + 64;
    dev.vram_write(done_pa, &0u32.to_le_bytes());
    let mut cmds: Vec<u32> = Vec::new();
    cmds.extend_from_slice(&sdma::copy_linear(src.va_addr, dst.va_addr, BUF as usize));
    cmds.extend_from_slice(&sdma::fence(done_va, 0xc0ffee));
    let mut ring_bytes = Vec::with_capacity(cmds.len() * 4);
    for c in &cmds {
        ring_bytes.extend_from_slice(&c.to_le_bytes());
    }
    dev.vram_write(ring_pa, &ring_bytes);

    {
        // Probe F32_CNTL writability: try to clear HALT and read back.
        let r = dev.regs();
        let before = r.read_raw("sdma", 0, 0, 0x2);
        r.write_raw("sdma", 0, 0, 0x2, before & !1);
        let after = r.read_raw("sdma", 0, 0, 0x2);
        println!(
            "F32_CNTL before={before:#x} after-clear-HALT={after:#x} (write {})",
            if after != before { "took" } else { "IGNORED (VF-protected?)" }
        );
    }
    // Make the CPU's ring/src writes (BAR0 write-combining) visible to the GPU.
    dev.gmc.flush_hdp(&dev.regs());
    let mut sdma_ring = SdmaRing::setup(&dev.regs(), 0, ring.va_addr, BUF, rptr.va_addr).expect("sdma setup");
    let r = dev.regs();
    println!(
        "after setup: RB_CNTL={:#x} RB_BASE={:#x} RB_RPTR={:#x} RB_WPTR={:#x} IB_CNTL={:#x} DOORBELL={:#x}",
        r.read("sdma", "regSDMA_GFX_RB_CNTL", 0).unwrap(),
        r.read("sdma", "regSDMA_GFX_RB_BASE", 0).unwrap(),
        r.read("sdma", "regSDMA_GFX_RB_RPTR", 0).unwrap(),
        r.read("sdma", "regSDMA_GFX_RB_WPTR", 0).unwrap(),
        r.read("sdma", "regSDMA_GFX_IB_CNTL", 0).unwrap(),
        r.read("sdma", "regSDMA_GFX_DOORBELL", 0).unwrap(),
    );
    sdma_ring.submit(&dev.regs(), &dev.pci.doorbell, cmds.len() as u64).expect("submit");
    std::thread::sleep(std::time::Duration::from_millis(50));
    println!(
        "after kick: RB_RPTR={:#x} RB_WPTR={:#x}",
        r.read("sdma", "regSDMA_GFX_RB_RPTR", 0).unwrap(),
        r.read("sdma", "regSDMA_GFX_RB_WPTR", 0).unwrap(),
    );
    let drained = sdma_ring.wait_idle(&dev.regs());
    // Flush the HDP read cache so the engine's VRAM writes are visible via BAR0.
    dev.gmc.flush_hdp(&dev.regs());

    let mut done = [0u8; 4];
    dev.vram_read(done_pa, &mut done);
    let mut rwb = [0u8; 4];
    dev.vram_read(rptr_pa, &mut rwb);
    println!(
        "rptr drained: {drained:?}; rptr writeback (MM-VM mem write) = {:#x}; fence slot = {:#x}",
        u32::from_le_bytes(rwb),
        u32::from_le_bytes(done)
    );

    let mut got = vec![0u8; BUF as usize];
    dev.vram_read(dst_pa, &mut got);
    if got == pattern {
        println!("SDMA copy OK — {BUF} bytes matched through vmid-0 VAs (GMMU validated)");
    } else {
        let first = (0..BUF as usize).find(|&i| got[i] != pattern[i]);
        println!("MISMATCH at byte {first:?}: got {:?} want {:?}", &got[..16], &pattern[..16]);
    }
    println!(
        "GC fault = {:?}  MM fault = {:?}",
        dev.gmc.fault_status(&dev.regs()),
        dev.gmc.mm_fault_status(&dev.regs())
    );
    dev.release();
}
