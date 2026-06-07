//! Minimal GC-hub translation probe: activate one MEC compute queue on XCC0 and
//! run a PM4 WRITE_DATA that writes 0xDEADBEEF to a vmid-0 GPU VA. If the value
//! lands at the buffer's physical address, the GC-hub walks our page tables.
//!
//! DESTRUCTIVE — amdgpu unbound; recovery needs a VM reboot.
//!     sudo ./target/debug/examples/am_compute

use svod_device::amd::am::dev::AmDev;
use svod_device::amd::am::ip::gfx;
use svod_device::amd::am::pci::PciDevice;

const PAGE: u64 = 0x1000;

fn main() {
    let bdf = std::env::args().nth(1).unwrap_or_else(|| PciDevice::discover().expect("AMD GPU"));
    let mut dev = AmDev::open(&bdf).expect("AmDev::open");
    println!("AmDev up: {} XCC, mc_base={:#x}", dev.disc.xccs(), dev.gmc.mc_base);

    // VA-mapped buffers (vmid-0). The MQD is addressed by MC/physical.
    let ring = dev.valloc(PAGE, true).expect("ring");
    let rptr = dev.valloc(PAGE, true).expect("rptr");
    let wptr = dev.valloc(PAGE, true).expect("wptr");
    let eop = dev.valloc(PAGE, true).expect("eop");
    let mqd = dev.valloc(2 * PAGE, true).expect("mqd");
    let dst = dev.valloc(PAGE, false).expect("dst");
    // CP writes queue state back to the MQD through vmid-0 (it faulted on the raw
    // MC addr), so CP_MQD_BASE_ADDR must be the mapped VA, not the physical addr.
    let mqd_mc = mqd.va_addr;

    // Clear dst + writeback slots via BAR0.
    dev.vram_write(dst.paddrs[0].0, &[0u8; 8]);
    dev.vram_write(rptr.paddrs[0].0, &[0u8; 8]);
    dev.vram_write(wptr.paddrs[0].0, &[0u8; 8]);
    println!(
        "ring va {:#x} -> pa {:#x}; dst va {:#x} -> pa {:#x}; mqd pa {:#x} mc {:#x}",
        ring.va_addr, ring.paddrs[0].0, dst.va_addr, dst.paddrs[0].0, mqd.paddrs[0].0, mqd_mc
    );

    // Build + write the MQD struct, then activate the queue on XCC0.
    let mqd_dwords = {
        let r = dev.regs();
        // DO NOT enable: writing BIFC_DOORBELL_ACCESS_EN_PF (PF-owned, over the
        // same BIF as the GIM mailbox) WEDGES the VF — the mailbox handshake then
        // times out and recovery needs a VM reboot. The doorbell aperture is
        // PF-owned on this VF. gfx::enable_doorbell_aperture(&r);
        gfx::enable_mec(&r, 0);
        gfx::build_mqd(
            &r,
            mqd_mc,
            ring.va_addr,
            PAGE,
            rptr.va_addr,
            wptr.va_addr,
            eop.va_addr,
            PAGE,
            gfx::DOORBELL_MEC_RING0,
        )
        .expect("build_mqd")
    };
    let mut mqd_bytes = Vec::with_capacity(mqd_dwords.len() * 4);
    for d in &mqd_dwords {
        mqd_bytes.extend_from_slice(&d.to_le_bytes());
    }
    dev.vram_write(mqd.paddrs[0].0, &mqd_bytes);
    dev.gmc.flush_hdp(&dev.regs());
    gfx::activate(&dev.regs(), 0, &mqd_dwords).expect("activate");

    // Diagnostics: is the MEC running and the queue active?
    {
        let r = dev.regs();
        println!(
            "MEC_CNTL={:#x?} CP_STAT={:#x?} DBELL_RANGE=[{:#x?},{:#x?}]",
            r.gc_read("regCP_MEC_CNTL", 0),
            r.gc_read("regCP_STAT", 0),
            r.gc_read("regCP_MEC_DOORBELL_RANGE_LOWER", 0),
            r.gc_read("regCP_MEC_DOORBELL_RANGE_UPPER", 0),
        );
        let _ = r.gc_write_fields("regGRBM_GFX_CNTL", 0, &[("meid", 1), ("pipeid", 0), ("queueid", 0), ("vmid", 0)]);
        println!(
            "HQD ACTIVE={:#x?} RPTR={:#x?} WPTR_LO={:#x?} DBELL_CTL={:#x?} PQ_BASE_LO={:#x?}",
            r.gc_read("regCP_HQD_ACTIVE", 0),
            r.gc_read("regCP_HQD_PQ_RPTR", 0),
            r.gc_read("regCP_HQD_PQ_WPTR_LO", 0),
            r.gc_read("regCP_HQD_PQ_DOORBELL_CONTROL", 0),
            r.gc_read("regCP_HQD_PQ_BASE", 0),
        );
        let _ = r.gc_write_fields("regGRBM_GFX_CNTL", 0, &[("meid", 0), ("pipeid", 0), ("queueid", 0), ("vmid", 0)]);
    }

    // Clear any latched GC fault so the addr we read reflects THIS submission.
    let _ = dev.regs().gc_write_fields(
        "regGCVM_INVALIDATE_ENG17_REQ",
        0,
        &[
            ("clear_protection_fault_status_addr", 1),
            ("per_vmid_invalidate_req", 1),
            ("invalidate_l2_ptes", 1),
            ("invalidate_l2_pde0", 1),
            ("invalidate_l1_ptes", 1),
        ],
    );

    // Submit a WRITE_DATA packet: 0xDEADBEEF -> dst_va.
    let pkt = gfx::write_data(dst.va_addr, 0xDEAD_BEEF);
    let mut ring_bytes = Vec::with_capacity(pkt.len() * 4);
    for d in &pkt {
        ring_bytes.extend_from_slice(&d.to_le_bytes());
    }
    dev.vram_write(ring.paddrs[0].0, &ring_bytes);
    let wptr_dwords = pkt.len() as u64;
    dev.vram_write(wptr.paddrs[0].0, &wptr_dwords.to_le_bytes());
    dev.gmc.flush_hdp(&dev.regs());
    gfx::ring_doorbell(&dev.pci.doorbell, gfx::DOORBELL_MEC_RING0, wptr_dwords);
    // Fallback kick: poke the wptr register directly (in case the VF doorbell
    // aperture doesn't route to the CP). GRBM-select the queue first.
    {
        let r = dev.regs();
        let _ = r.gc_write_fields("regGRBM_GFX_CNTL", 0, &[("meid", 1), ("pipeid", 0), ("queueid", 0), ("vmid", 0)]);
        let _ = r.gc_write("regCP_HQD_PQ_WPTR_LO", 0, wptr_dwords as u32);
        let _ = r.gc_write_fields("regGRBM_GFX_CNTL", 0, &[("meid", 0), ("pipeid", 0), ("queueid", 0), ("vmid", 0)]);
    }

    // Poll dst for the value.
    let mut got = [0u8; 4];
    for _ in 0..2_000_000 {
        dev.gmc.flush_hdp(&dev.regs());
        dev.vram_read(dst.paddrs[0].0, &mut got);
        if u32::from_le_bytes(got) == 0xDEAD_BEEF {
            break;
        }
        std::hint::spin_loop();
    }
    let mut rwb = [0u8; 4];
    dev.vram_read(rptr.paddrs[0].0, &mut rwb);
    println!("rptr writeback = {:#x}; dst = {:#x}", u32::from_le_bytes(rwb), u32::from_le_bytes(got));
    {
        // Post-kick: did the doorbell deliver the wptr to the CP, and did it run?
        let r = dev.regs();
        let _ = r.gc_write_fields("regGRBM_GFX_CNTL", 0, &[("meid", 1), ("pipeid", 0), ("queueid", 0), ("vmid", 0)]);
        println!(
            "post-kick: HQD WPTR_LO={:#x?} RPTR={:#x?} ACTIVE={:#x?} | CP_STAT={:#x?}",
            r.gc_read("regCP_HQD_PQ_WPTR_LO", 0),
            r.gc_read("regCP_HQD_PQ_RPTR", 0),
            r.gc_read("regCP_HQD_ACTIVE", 0),
            r.gc_read("regCP_STAT", 0),
        );
        let _ = r.gc_write_fields("regGRBM_GFX_CNTL", 0, &[("meid", 0), ("pipeid", 0), ("queueid", 0), ("vmid", 0)]);
    }
    if u32::from_le_bytes(got) == 0xDEAD_BEEF {
        println!("GC COMPUTE OK — WRITE_DATA landed through vmid-0 VA (GC-hub GMMU validated)");
    } else {
        println!("MISMATCH — dst never became 0xDEADBEEF");
    }
    {
        let r = dev.regs();
        let st = r.gc_read("regGCVM_L2_PROTECTION_FAULT_STATUS", 0).unwrap_or(0);
        let alo = r.gc_read("regGCVM_L2_PROTECTION_FAULT_ADDR_LO32", 0).unwrap_or(0) as u64;
        let ahi = r.gc_read("regGCVM_L2_PROTECTION_FAULT_ADDR_HI32", 0).unwrap_or(0) as u64;
        let fault_va = ((ahi << 32) | alo) << 12; // addr field is the faulting page number
        println!(
            "GC fault status={st:#x} (walker_error={}, perm={}, mapping_err={}, cid={}, rw={}, vmid={}) fault_va={:#x}",
            (st >> 1) & 0x7,
            (st >> 4) & 0xf,
            (st >> 8) & 1,
            (st >> 9) & 0x1ff,
            (st >> 18) & 1,
            (st >> 20) & 0xf,
            fault_va,
        );
    }
    dev.release();
}
