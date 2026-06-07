//! GMC (graphics memory controller) bring-up for the GC hub on an SR-IOV VF.
//!
//! On a VF the GIM/PF owns the L2 cache config, system aperture and protection-
//! fault defaults (`gfxhub_v1_2` skips them under `amdgpu_sriov_vf`); the guest
//! programs only its page-table context (start/end/base + CNTL), the per-engine
//! invalidation address ranges, and issues TLB invalidations. All these are GC
//! registers, so every access goes through RLCG. Port of `AM_GMC` minus the
//! PF-owned pieces.

use crate::error::Error;

use super::super::discovery::Discovery;
use super::super::regaccess::Regs;

type Result<T> = std::result::Result<T, Error>;

/// Runtime-flush invalidation engine (matches AM / amdgpu choice of 17).
const ENG: u32 = 17;
const INVALIDATE_POLL: u32 = 1_000_000;

pub struct Gmc {
    /// XGMI/MC offset added to a raw VRAM paddr to form a GPU MC address.
    pub paddr_base: u64,
    pub fb_base: u64,
    pub fb_end: u64,
    /// Inclusive VA window covered by context0.
    pub vm_base: u64,
    pub vm_end: u64,
    /// XGMI/`vram_base_offset` added to raw VRAM paddrs for PTEs / PT base
    /// (tinygrad `paddr2xgmi`; 0 on a single-node VF — see `probe`).
    pub mc_base: u64,
    xccs: u16,
    vmhubs: u16,
}

/// Context-CNTL fields, shared by the GC and MM hubs (identical layout).
/// gfx9 (`trans_futher`): page_table_depth = 2 (not 3) and block_size = 9 — the
/// gfx9 walker is configured differently from gfx10+ for the same 4-level shape.
const CNTL_FIELDS: &[(&str, u32)] = &[
    ("enable_context", 1),
    ("page_table_depth", 2),
    ("page_table_block_size", 9),
    ("range_protection_fault_enable_interrupt", 1),
    ("range_protection_fault_enable_default", 1),
    ("dummy_page_protection_fault_enable_interrupt", 1),
    ("dummy_page_protection_fault_enable_default", 1),
    ("pde0_protection_fault_enable_interrupt", 1),
    ("pde0_protection_fault_enable_default", 1),
    ("valid_protection_fault_enable_interrupt", 1),
    ("valid_protection_fault_enable_default", 1),
    ("read_protection_fault_enable_interrupt", 1),
    ("read_protection_fault_enable_default", 1),
    ("write_protection_fault_enable_interrupt", 1),
    ("write_protection_fault_enable_default", 1),
    ("execute_protection_fault_enable_interrupt", 1),
    ("execute_protection_fault_enable_default", 1),
];

/// `MC_VM_MX_L1_TLB_CNTL` fields (both hubs): enable the L1 TLB and the
/// advanced-driver-model walker so the UTCL uses the per-context page tables.
/// `mtype = 3` = `MTYPE_UC` (soc_9). Matches tinygrad `init_hub`.
const MX_L1_TLB_FIELDS: &[(&str, u32)] = &[
    ("enable_l1_tlb", 1),
    ("system_access_mode", 3),
    ("system_aperture_unmapped_access", 0),
    ("enable_advanced_driver_model", 1),
    ("mtype", 3),
];

// L2 cache config (tinygrad `init_hub`, gfx9 `trans_futher` values). On a VF
// the kernel normally lets the PF own these; we program them best-effort and
// read back (SVOD_AM_DEBUG) to learn which the GIM actually lets the guest set.
const L2_CNTL_FIELDS: &[(&str, u32)] = &[
    ("enable_l2_cache", 1),
    ("enable_l2_fragment_processing", 1), // gfx9 (< 10)
    ("l2_pde0_cache_tag_generation_mode", 0),
    ("pde_fault_classification", 0),
    ("context1_identity_access_mode", 1),
    ("identity_mode_fragment_size", 0),
    ("enable_default_page_out_to_system_memory", 1),
];
const L2_CNTL2_FIELDS: &[(&str, u32)] = &[("invalidate_all_l1_tlbs", 1), ("invalidate_l2_cache", 1)];
const L2_CNTL3_FIELDS: &[(&str, u32)] = &[
    ("l2_cache_4k_associativity", 1),
    ("l2_cache_bigk_associativity", 1),
    ("bank_select", 12),
    ("l2_cache_bigk_fragment_size", 9),
];
const L2_CNTL4_FIELDS: &[(&str, u32)] = &[("l2_cache_4k_partition_count", 1)];

#[inline]
fn dbg_on() -> bool {
    std::env::var("SVOD_AM_DEBUG").is_ok()
}

const INVALIDATE_REQ_FIELDS: &[(&str, u32)] = &[
    ("flush_type", 0),
    ("invalidate_l2_ptes", 1),
    ("invalidate_l2_pde0", 1),
    ("invalidate_l2_pde1", 1),
    ("invalidate_l2_pde2", 1),
    ("invalidate_l1_ptes", 1),
];

impl Gmc {
    /// Read FB/XGMI apertures (direct MMHUB reads) and record the VA window.
    pub fn probe(regs: &Regs, disc: &Discovery, va_base: u64, va_size: u64) -> Result<Self> {
        let fb_base = (regs.read("mmhub", "regMMMC_VM_FB_LOCATION_BASE", 0)? as u64 & 0xff_ffff) << 24;
        let fb_end = (regs.read("mmhub", "regMMMC_VM_FB_LOCATION_TOP", 0)? as u64 & 0xff_ffff) << 24;
        // XGMI fabric offset of this node. On the VF these registers are GIM-
        // gated (read all-ones); a single-node VF has no XGMI offset, so treat a
        // gated read as 0. `pf_lfb_region` is the full 4-bit field.
        let xgmi_cntl = regs.read("mmhub", "regMMMC_VM_XGMI_LFB_CNTL", 0)?;
        let xgmi_size = regs.read("mmhub", "regMMMC_VM_XGMI_LFB_SIZE", 0)?;
        let paddr_base = if xgmi_cntl == u32::MAX || xgmi_size == u32::MAX {
            0
        } else {
            (xgmi_cntl as u64 & 0xf) * ((xgmi_size as u64 & 0xffff) << 24)
        };
        let vm_end = (va_base + va_size - 1).min(0x7fff_ffff_ffff);
        // Page-table entries and PAGE_TABLE_BASE_ADDR hold `raw_paddr +
        // vram_base_offset` (amdgpu `amdgpu_gmc_vram_mc2pa`); the FB *location*
        // CANCELS out and is never part of the stored address. vram_base_offset
        // = (MC_VM_FB_OFFSET << 24) + XGMI node offset. Read FB_OFFSET via the GC
        // RLCG channel (the MM-hub direct read is GIM-gated to all-ones); on this
        // VF it is 0, so PTEs hold raw paddrs. Override with
        // SVOD_AM_MCBASE={raw,fb,fbxgmi} only while characterizing the HW.
        let fb_offset = (regs.gc_read("regGCMC_VM_FB_OFFSET", 0)? as u64 & 0xff_ffff) << 24;
        let mc_base = match std::env::var("SVOD_AM_MCBASE").as_deref() {
            Ok("raw") => 0,
            Ok("fb") => fb_base,
            Ok("fbxgmi") => fb_base + paddr_base,
            _ => fb_offset + paddr_base,
        };
        Ok(Self {
            mc_base,
            paddr_base,
            fb_base,
            fb_end,
            vm_base: va_base,
            vm_end,
            xccs: disc.xccs() as u16,
            vmhubs: disc.vmhubs() as u16,
        })
    }

    /// A raw VRAM paddr → page-table address (XGMI/`vram_base_offset` base, for
    /// PTEs and the PT base register). tinygrad `paddr2xgmi`.
    pub fn paddr2xgmi(&self, paddr: u64) -> u64 {
        self.mc_base + paddr
    }

    /// Program context0 (vmid 0) + invalidation ranges on both hubs (GC per XCC
    /// via RLCG; MM per AID via direct MMIO — SDMA translates through the MM
    /// hub), then flush. `root_pt` is the raw VRAM paddr of the root page table.
    ///
    /// Also enables the per-hub L1 TLB (`MX_L1_TLB_CNTL`): the kernel
    /// (`gfxhub_v1_2`/`mmhub_v1_8`) programs this unconditionally on the VF, and
    /// without `enable_advanced_driver_model` the UTCL never walks these tables.
    pub fn enable(&self, regs: &Regs, root_pt: u64) -> Result<()> {
        let base = self.paddr2xgmi(root_pt) | 1;

        for xcc in 0..self.xccs {
            regs.gc_update("regGCMC_VM_MX_L1_TLB_CNTL", xcc, MX_L1_TLB_FIELDS)?;
            self.init_gc_caches(regs, xcc);
            regs.gc_write_pair("regGCVM_CONTEXT0_PAGE_TABLE_START_ADDR", xcc, self.vm_base >> 12)?;
            regs.gc_write_pair("regGCVM_CONTEXT0_PAGE_TABLE_END_ADDR", xcc, self.vm_end >> 12)?;
            regs.gc_write_pair("regGCVM_CONTEXT0_PAGE_TABLE_BASE_ADDR", xcc, base)?;
            regs.gc_write_fields("regGCVM_CONTEXT0_CNTL", xcc, CNTL_FIELDS)?;
            for eng in 0..18u32 {
                regs.gc_write(&format!("regGCVM_INVALIDATE_ENG{eng}_ADDR_RANGE_LO32"), xcc, 0xffff_ffff)?;
                regs.gc_write(&format!("regGCVM_INVALIDATE_ENG{eng}_ADDR_RANGE_HI32"), xcc, 0x1f)?;
            }
        }

        for hub in 0..self.vmhubs {
            regs.update("mmhub", "regMMMC_VM_MX_L1_TLB_CNTL", hub, MX_L1_TLB_FIELDS)?;
            self.init_mm_caches(regs, hub);
            mm_write_pair(regs, "regMMVM_CONTEXT0_PAGE_TABLE_START_ADDR", hub, self.vm_base >> 12)?;
            mm_write_pair(regs, "regMMVM_CONTEXT0_PAGE_TABLE_END_ADDR", hub, self.vm_end >> 12)?;
            mm_write_pair(regs, "regMMVM_CONTEXT0_PAGE_TABLE_BASE_ADDR", hub, base)?;
            regs.write_fields("mmhub", "regMMVM_CONTEXT0_CNTL", hub, CNTL_FIELDS)?;
            for eng in 0..18u32 {
                regs.write("mmhub", &format!("regMMVM_INVALIDATE_ENG{eng}_ADDR_RANGE_LO32"), hub, 0xffff_ffff)?;
                regs.write("mmhub", &format!("regMMVM_INVALIDATE_ENG{eng}_ADDR_RANGE_HI32"), hub, 0x1f)?;
            }
        }

        self.flush_tlb(regs, 0)
    }

    /// Best-effort L2 + system/identity-aperture init for the GC hub (RLCG).
    /// On the VF some of these may be PF-owned and rejected by RLCG; we log and
    /// continue, then read back to see what stuck.
    fn init_gc_caches(&self, regs: &Regs, xcc: u16) {
        let try_gc = |name: &str, fields: &[(&str, u32)], full: bool| {
            let r = if full { regs.gc_write_fields(name, xcc, fields) } else { regs.gc_update(name, xcc, fields) };
            if let Err(e) = r
                && dbg_on()
            {
                eprintln!("[am-gmc] GC {name} xcc{xcc} rejected: {e}");
            }
        };
        try_gc("regGCVM_L2_CNTL", L2_CNTL_FIELDS, false);
        try_gc("regGCVM_L2_CNTL2", L2_CNTL2_FIELDS, false);
        try_gc("regGCVM_L2_CNTL3", L2_CNTL3_FIELDS, true);
        try_gc("regGCVM_L2_CNTL4", L2_CNTL4_FIELDS, true);
        let _ = regs.gc_write("regGCMC_VM_SYSTEM_APERTURE_LOW_ADDR", xcc, (self.fb_base >> 18) as u32);
        let _ = regs.gc_write("regGCMC_VM_SYSTEM_APERTURE_HIGH_ADDR", xcc, (self.fb_end >> 18) as u32);
        let _ = regs.gc_write_pair("regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", xcc, 0xf_ffff_ffff);
        let _ = regs.gc_write_pair("regGCVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", xcc, 0);
        let _ = regs.gc_write_pair("regGCVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", xcc, 0);
        let _ =
            regs.gc_update("regGCVM_L2_PROTECTION_FAULT_CNTL2", xcc, &[("active_page_migration_pte_read_retry", 1)]);
        if dbg_on() && xcc == 0 {
            eprintln!(
                "[am-gmc] GC xcc0 readback: L2_CNTL={:#x?} MX_L1_TLB={:#x?} CONTEXT0_CNTL={:#x?}",
                regs.gc_read("regGCVM_L2_CNTL", xcc),
                regs.gc_read("regGCMC_VM_MX_L1_TLB_CNTL", xcc),
                regs.gc_read("regGCVM_CONTEXT0_CNTL", xcc),
            );
        }
    }

    /// Best-effort L2 + system/identity-aperture init for the MM hub (direct).
    fn init_mm_caches(&self, regs: &Regs, hub: u16) {
        let _ = regs.update("mmhub", "regMMVM_L2_CNTL", hub, L2_CNTL_FIELDS);
        let _ = regs.update("mmhub", "regMMVM_L2_CNTL2", hub, L2_CNTL2_FIELDS);
        let _ = regs.write_fields("mmhub", "regMMVM_L2_CNTL3", hub, L2_CNTL3_FIELDS);
        let _ = regs.write_fields("mmhub", "regMMVM_L2_CNTL4", hub, L2_CNTL4_FIELDS);
        let _ = regs.write("mmhub", "regMMMC_VM_SYSTEM_APERTURE_LOW_ADDR", hub, (self.fb_base >> 18) as u32);
        let _ = regs.write("mmhub", "regMMMC_VM_SYSTEM_APERTURE_HIGH_ADDR", hub, (self.fb_end >> 18) as u32);
        let _ = mm_write_pair(regs, "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_LOW_ADDR", hub, 0xf_ffff_ffff);
        let _ = mm_write_pair(regs, "regMMVM_L2_CONTEXT1_IDENTITY_APERTURE_HIGH_ADDR", hub, 0);
        let _ = mm_write_pair(regs, "regMMVM_L2_CONTEXT_IDENTITY_PHYSICAL_OFFSET", hub, 0);
        let _ = regs.update(
            "mmhub",
            "regMMVM_L2_PROTECTION_FAULT_CNTL2",
            hub,
            &[("active_page_migration_pte_read_retry", 1)],
        );
        if dbg_on() {
            eprintln!(
                "[am-gmc] MM hub{hub} readback: L2_CNTL={:#x} MX_L1_TLB={:#x} CONTEXT0_CNTL={:#x}",
                regs.read("mmhub", "regMMVM_L2_CNTL", hub).unwrap_or(0xdead_beef),
                regs.read("mmhub", "regMMMC_VM_MX_L1_TLB_CNTL", hub).unwrap_or(0xdead_beef),
                regs.read("mmhub", "regMMVM_CONTEXT0_CNTL", hub).unwrap_or(0xdead_beef),
            );
        }
    }

    /// Flush the HDP read cache so GPU writes to VRAM become visible to the CPU
    /// over the BAR (and vice-versa). The remap register holds a byte address in
    /// MMIO; writing 0 to that dword triggers the flush. Port of `flush_hdp`.
    pub fn flush_hdp(&self, regs: &Regs) {
        if let Ok(v) = regs.read("nbio", "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL", 0) {
            regs.mmio_write_abs((v / 4) as usize, 0);
        }
    }

    /// Invalidate all caches for `vmid` on engine 17 of both hubs; poll the ACK.
    pub fn flush_tlb(&self, regs: &Regs, vmid: u32) -> Result<()> {
        self.flush_hdp(regs);
        let req = |extra: u32| -> Vec<(&str, u32)> {
            let mut f = INVALIDATE_REQ_FIELDS.to_vec();
            f.push(("per_vmid_invalidate_req", extra));
            f
        };
        // GC hub, per XCC (RLCG).
        for xcc in 0..self.xccs {
            regs.gc_write_fields(&format!("regGCVM_INVALIDATE_ENG{ENG}_REQ"), xcc, &req(1 << vmid))?;
            poll_ack(|| regs.gc_read(&format!("regGCVM_INVALIDATE_ENG{ENG}_ACK"), xcc).map(|a| a & (1 << vmid)))
                .map_err(|_| Error::Runtime { message: format!("GC TLB invalidate ACK timeout on XCC{xcc}") })?;
        }
        // MM hub, per AID (direct); honors the per-hub semaphore.
        for hub in 0..self.vmhubs {
            poll_ack(|| regs.read("mmhub", &format!("regMMVM_INVALIDATE_ENG{ENG}_SEM"), hub).map(|s| s & 1))
                .map_err(|_| Error::Runtime { message: format!("MM sem timeout on hub{hub}") })?;
            regs.write_fields("mmhub", &format!("regMMVM_INVALIDATE_ENG{ENG}_REQ"), hub, &req(1 << vmid))?;
            poll_ack(|| regs.read("mmhub", &format!("regMMVM_INVALIDATE_ENG{ENG}_ACK"), hub).map(|a| a & (1 << vmid)))
                .map_err(|_| Error::Runtime { message: format!("MM TLB invalidate ACK timeout on hub{hub}") })?;
            regs.write("mmhub", &format!("regMMVM_INVALIDATE_ENG{ENG}_SEM"), hub, 0)?;
        }
        Ok(())
    }

    /// Decode a protection fault, if any latched (XCC0).
    pub fn fault_status(&self, regs: &Regs) -> Option<u32> {
        match regs.gc_read("regGCVM_L2_PROTECTION_FAULT_STATUS", 0) {
            Ok(s) if s != 0 => Some(s),
            _ => None,
        }
    }

    /// MM-hub protection fault, if any (hub 0).
    pub fn mm_fault_status(&self, regs: &Regs) -> Option<u32> {
        match regs.read("mmhub", "regMMVM_L2_PROTECTION_FAULT_STATUS", 0) {
            Ok(s) if s != 0 => Some(s),
            _ => None,
        }
    }
}

/// Write a 64-bit value across `<base>_LO32`/`_HI32` on an MM-hub instance.
fn mm_write_pair(regs: &Regs, base: &str, hub: u16, val: u64) -> Result<()> {
    regs.write("mmhub", &format!("{base}_LO32"), hub, val as u32)?;
    regs.write("mmhub", &format!("{base}_HI32"), hub, (val >> 32) as u32)
}

/// Spin until `read()` returns nonzero, or give up after `INVALIDATE_POLL`.
fn poll_ack(mut read: impl FnMut() -> Result<u32>) -> Result<()> {
    for _ in 0..INVALIDATE_POLL {
        if read()? != 0 {
            return Ok(());
        }
        std::hint::spin_loop();
    }
    Err(Error::Runtime { message: "poll timeout".into() })
}
