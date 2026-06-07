//! Minimal MEC compute-queue bring-up to probe GC-hub GMMU translation.
//!
//! Activates ONE PM4 compute queue on one XCC by direct CP_HQD register
//! programming (amdgpu `kiq_init_register` style — no KIQ/MAP_QUEUES), then a
//! `WRITE_DATA` packet writes a constant to a vmid-0 GPU VA. If the value lands
//! at the buffer's physical address (read back over the BAR), the GC-hub walks
//! our page tables. Ports the relevant slice of tinygrad `AM_GFX.setup_ring`.

use crate::error::Error;

use super::super::pci::Bar;
use super::super::regaccess::Regs;

type Result<T> = std::result::Result<T, Error>;

/// MEC compute-ring doorbell index (`AMDGPU_NAVI10_DOORBELL_MEC_RING0`). tinygrad
/// uses this exact value on gfx9.4.3 — the layout1 `MEC_RING_START` (8) is wrong
/// (sDMA0 = 0x100 matches both enums, so it didn't validate the MEC slot).
pub const DOORBELL_MEC_RING0: u64 = 3;

/// `v9_mqd` is 512 dwords; the register block CP_MQD_BASE_ADDR..CP_HQD_PQ_WPTR_HI
/// mirrors dwords 0x80..=0xb7 (56 contiguous dwords).
const MQD_DWORDS: usize = 512;
const BLIT_START: usize = 0x80; // cp_mqd_base_addr_lo
const BLIT_LEN: usize = 56; // through cp_hqd_pq_wptr_hi (0xb7)

/// `log2(bytes/4)` (ring/eop size field convention = dwords' bit_length − 2).
fn size_field(bytes: u64) -> u32 {
    (64 - (bytes / 4).leading_zeros()).saturating_sub(2)
}

/// Build the `v9_mqd` for a PM4 (non-AQL) compute queue, vmid 0. All `*_va`
/// are GPU VAs except `mqd_mc`, which is the MQD's MC/physical address.
#[allow(clippy::too_many_arguments)]
pub fn build_mqd(
    regs: &Regs,
    mqd_mc: u64,
    ring_va: u64,
    ring_size: u64,
    rptr_va: u64,
    wptr_va: u64,
    eop_va: u64,
    eop_size: u64,
    doorbell_idx: u64,
) -> Result<[u32; MQD_DWORDS]> {
    let lo = |v: u64| v as u32;
    let hi = |v: u64| (v >> 32) as u32;
    let mut m = [0u32; MQD_DWORDS];
    m[0x00] = 0xC031_0800; // header
    m[0x17] = 0xffff_ffff; // compute_static_thread_mgmt_se0
    m[0x18] = 0xffff_ffff; // se1
    m[0x1a] = 0xffff_ffff; // se2
    m[0x1b] = 0xffff_ffff; // se3
    m[0x80] = lo(mqd_mc); // cp_mqd_base_addr (MC/physical)
    m[0x81] = hi(mqd_mc);
    m[0x82] = 0; // cp_hqd_active — written =1 separately, after the blit
    m[0x83] = 0; // cp_hqd_vmid = 0
    m[0x84] = regs.gc_encode("regCP_HQD_PERSISTENT_STATE", &[("preload_size", 0x55), ("preload_req", 1)])?;
    m[0x85] = 2; // cp_hqd_pipe_priority
    m[0x86] = 0xf; // cp_hqd_queue_priority
    m[0x87] = 0x111; // cp_hqd_quantum
    m[0x88] = lo(ring_va >> 8); // cp_hqd_pq_base (VA>>8, translated)
    m[0x89] = hi(ring_va >> 8);
    m[0x8b] = lo(rptr_va); // rptr_report_addr (VA)
    m[0x8c] = hi(rptr_va);
    m[0x8d] = lo(wptr_va); // wptr_poll_addr (VA)
    m[0x8e] = hi(wptr_va);
    m[0x8f] = regs.gc_encode(
        "regCP_HQD_PQ_DOORBELL_CONTROL",
        &[("doorbell_offset", (doorbell_idx * 2) as u32), ("doorbell_en", 1)],
    )?;
    m[0x91] = regs.gc_encode(
        "regCP_HQD_PQ_CONTROL",
        &[("rptr_block_size", 5), ("unord_dispatch", 0), ("queue_size", size_field(ring_size))],
    )?;
    m[0x95] = regs.gc_encode("regCP_HQD_IB_CONTROL", &[("min_ib_avail_size", 3)])?;
    m[0xa0] = 0x2000_4000; // cp_hqd_hq_status0
    m[0xa2] = regs.gc_encode("regCP_MQD_CONTROL", &[("priv_state", 1)])?;
    m[0xa5] = lo(eop_va >> 8); // cp_hqd_eop_base_addr (VA>>8)
    m[0xa6] = hi(eop_va >> 8);
    m[0xa7] = regs.gc_encode("regCP_HQD_EOP_CONTROL", &[("eop_size", size_field(eop_size))])?;
    Ok(m)
}

/// Open the CP/MEC doorbell aperture (tinygrad `AM_SOC.init_hw`, NBIO 7.9 path).
///
/// WARNING: on this SR-IOV VF these are PF-owned. Writing
/// `BIFC_DOORBELL_ACCESS_EN_PF` (over the same BIF as the GIM mailbox) WEDGES the
/// VF — the next `request_init_access` times out and recovery needs a VM reboot.
/// Kept for reference / bare-metal; DO NOT call on the VF.
pub fn enable_doorbell_aperture(regs: &Regs) {
    let _ = regs.write("nbio", "regXCC_DOORBELL_FENCE", 0, 0);
    let _ = regs.write("nbio", "regBIFC_GFX_INT_MONITOR_MASK", 0, 0x7ff);
    let _ = regs.write("nbio", "regBIFC_DOORBELL_ACCESS_EN_PF", 0, 0xf_ffff);
    let _ = regs.write("nbio", "regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN", 0, 1);
}

/// Best-effort GFX/MEC bring-up for `xcc`, mirroring tinygrad `AM_GFX.init_hw`
/// (the plain-register slice — firmware/PSP steps are GIM-owned on the VF). Some
/// writes may be PF-gated and rejected by RLCG; that's fine, we continue.
pub fn enable_mec(regs: &Regs, xcc: u16) {
    // _config_mec (gfx9): invalidate the MEC icache + reset me1/me2 pipe0, halted.
    let _ = regs.gc_write_fields(
        "regCP_MEC_CNTL",
        xcc,
        &[
            ("mec_invalidate_icache", 1),
            ("mec_me1_pipe0_reset", 1),
            ("mec_me2_pipe0_reset", 1),
            ("mec_me1_halt", 1),
            ("mec_me2_halt", 1),
        ],
    );
    // Core GFX register init the CP needs before any queue.
    let _ = regs.gc_write("regGB_ADDR_CONFIG", xcc, 0x2a11_4042); // golden for mi300
    let _ = regs.gc_write("regRLC_CNTL", xcc, 0x1); // rlc_enable_f32
    let _ = regs.gc_write_fields("regRLC_SRM_CNTL", xcc, &[("srm_enable", 1), ("auto_incr_addr", 1)]);
    let _ = regs.gc_write_fields("regGRBM_CNTL", xcc, &[("read_timeout", 0xff)]);
    let _ = regs.gc_write_fields("regSH_MEM_BASES", xcc, &[("shared_base", 1), ("private_base", 2)]);
    let _ = regs.gc_write("regCP_MEC_DOORBELL_RANGE_LOWER", xcc, 0x100 * xcc as u32);
    let _ = regs.gc_write("regCP_MEC_DOORBELL_RANGE_UPPER", xcc, 0x100 * xcc as u32 + 0xf8);
    // _enable_mec: clear halt, then let the microengine settle.
    let _ = regs.gc_write("regCP_MEC_CNTL", xcc, 0);
    std::thread::sleep(std::time::Duration::from_millis(50));
}

/// GRBM-select (me1,pipe0,queue0,vmid0) → blit the MQD register block via RLCG
/// → CP_HQD_ACTIVE=1 → deselect. Activates the queue on `xcc`.
pub fn activate(regs: &Regs, xcc: u16, mqd: &[u32; MQD_DWORDS]) -> Result<()> {
    regs.gc_write_fields("regGRBM_GFX_CNTL", xcc, &[("meid", 1), ("pipeid", 0), ("queueid", 0), ("vmid", 0)])?;
    let base = regs.gc_index("regCP_MQD_BASE_ADDR", xcc)?;
    for k in 0..BLIT_LEN {
        regs.gc_write_index(xcc, base + k, mqd[BLIT_START + k])?;
    }
    regs.gc_write("regCP_HQD_ACTIVE", xcc, 1)?;
    // Deselect: back to broadcast (me0,pipe0,queue0,vmid0).
    regs.gc_write_fields("regGRBM_GFX_CNTL", xcc, &[("meid", 0), ("pipeid", 0), ("queueid", 0), ("vmid", 0)])?;
    Ok(())
}

/// A `WRITE_DATA` PM4 packet: write `value` to destination GPU VA `dst_va`
/// (DST_SEL=memory, ENGINE_SEL=ME, WR_CONFIRM). 5 dwords.
pub fn write_data(dst_va: u64, value: u32) -> [u32; 5] {
    const PACKET3_WRITE_DATA: u32 = 0x37;
    let header = 0xC000_0000 | (PACKET3_WRITE_DATA << 8) | (3u32 << 16); // count = 4 dwords - 1
    let control = (1 << 20) | (5 << 8); // WR_CONFIRM | DST_SEL(memory) | ENGINE_SEL(ME=0)
    [header, control, dst_va as u32, (dst_va >> 32) as u32, value]
}

/// Ring the MEC doorbell (BAR2, 64-bit) with the new wptr (in dwords).
pub fn ring_doorbell(doorbell: &Bar, doorbell_idx: u64, wptr_dwords: u64) {
    std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    doorbell.write_u64(doorbell_idx as usize, wptr_dwords);
}
