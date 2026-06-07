//! RLCG indirect register access for GC/GCVM on an SR-IOV VF.
//!
//! On a VF the host protects direct GC MMIO (every direct read returns
//! `0xffffffff`), so the guest pokes GC registers through the RLC: stage the
//! value + (address | command) into per-XCC scratch registers, kick `RLC_SPARE_INT`,
//! and poll the address field clear. Port of `amdgpu_virt_rlcg_reg_rw`.
//!
//! The four scratch registers and the kick are NOT in tinygrad's bare-metal
//! tables, so their dword offsets are vendored here from `gc_9_4_3_offset.h`
//! (raw header offsets, matching the `regs_gen.rs` convention — see
//! `regSCRATCH_REG6` = 0x2046).

use crate::error::Error;

use super::pci::Bar;

// GC register offsets (raw header offset, segment = SOC15 BASE_IDX).
const SCRATCH_REG0: (u32, u8) = (0x2040, 1);
const SCRATCH_REG1: (u32, u8) = (0x2041, 1);
const SCRATCH_REG2: (u32, u8) = (0x2042, 1);
const SCRATCH_REG3: (u32, u8) = (0x2043, 1);
const GRBM_GFX_CNTL: (u32, u8) = (0x22, 0);
const GRBM_GFX_INDEX: (u32, u8) = (0x2200, 1);
const RLC_SPARE_INT: (u32, u8) = (0x4ccc, 1);

// Command flags packed into scratch_reg1[30:28].
const GC_WRITE_LEGACY: u32 = 0x8 << 28;
const GC_READ: u32 = 0x1 << 28;

// scratch_reg1 status/error fields.
const ADDRESS_MASK: u32 = 0xfffff; // [19:0]: busy while nonzero
const REG_NOT_IN_RANGE: u32 = 0x100_0000;
const WRONG_OPERATION_TYPE: u32 = 0x200_0000;
const VFGATE_DISABLED: u32 = 0x400_0000;
const POLL_ITERS: u32 = 50_000;

type Result<T> = std::result::Result<T, Error>;

/// One XCC's RLCG channel: the absolute dword indices of its scratch quad +
/// kick, computed from that XCC's GC segment bases.
#[derive(Clone, Copy, Debug)]
pub struct RlcgChannel {
    sr0: usize,
    sr1: usize,
    sr2: usize,
    sr3: usize,
    grbm_cntl: usize,
    grbm_idx: usize,
    spare_int: usize,
}

impl RlcgChannel {
    /// `gc_bases` = the GC IP segment bases for this XCC instance (from
    /// discovery `regs_offset[GC][xcc]`).
    pub fn new(gc_bases: &[u64]) -> Self {
        let idx = |(off, seg): (u32, u8)| gc_bases[seg as usize] as usize + off as usize;
        Self {
            sr0: idx(SCRATCH_REG0),
            sr1: idx(SCRATCH_REG1),
            sr2: idx(SCRATCH_REG2),
            sr3: idx(SCRATCH_REG3),
            grbm_cntl: idx(GRBM_GFX_CNTL),
            grbm_idx: idx(GRBM_GFX_INDEX),
            spare_int: idx(RLC_SPARE_INT),
        }
    }

    fn run(&self, mmio: &Bar, dword_addr: usize, val: u32, flag: u32) -> Result<u32> {
        // GRBM select registers shortcut straight to scratch_reg2/3 — but only on
        // a write: amdgpu mirrors to the direct register solely for the legacy
        // write, and these are write-only select registers, so a read must NOT
        // clobber the live broadcast-select with 0.
        if flag == GC_WRITE_LEGACY && dword_addr == self.grbm_cntl {
            mmio.write_u32(self.sr2, val);
            mmio.write_u32(self.grbm_cntl, val);
            return Ok(0);
        }
        if flag == GC_WRITE_LEGACY && dword_addr == self.grbm_idx {
            mmio.write_u32(self.sr3, val);
            mmio.write_u32(self.grbm_idx, val);
            return Ok(0);
        }
        mmio.write_u32(self.sr0, val);
        mmio.write_u32(self.sr1, (dword_addr as u32 & ADDRESS_MASK) | flag);
        mmio.write_u32(self.spare_int, 1);

        let mut last = 0u32;
        for _ in 0..POLL_ITERS {
            last = mmio.read_u32(self.sr1);
            if last & ADDRESS_MASK == 0 {
                return Ok(mmio.read_u32(self.sr0));
            }
            std::hint::spin_loop();
        }
        let why = if last & REG_NOT_IN_RANGE != 0 {
            "reg not in RLCG range"
        } else if last & WRONG_OPERATION_TYPE != 0 {
            "wrong operation type"
        } else if last & VFGATE_DISABLED != 0 {
            "VF gate disabled"
        } else {
            "timeout"
        };
        Err(Error::Runtime { message: format!("rlcg {why} (addr {dword_addr:#x}, sr1={last:#x})") })
    }

    /// Read a GC register by absolute dword index.
    pub fn read(&self, mmio: &Bar, dword_addr: usize) -> Result<u32> {
        self.run(mmio, dword_addr, 0, GC_READ)
    }

    /// Write a GC register by absolute dword index (legacy mode also mirrors
    /// the value to direct MMIO, matching amdgpu).
    pub fn write(&self, mmio: &Bar, dword_addr: usize, val: u32) -> Result<()> {
        self.run(mmio, dword_addr, val, GC_WRITE_LEGACY).map(|_| ())
    }
}
