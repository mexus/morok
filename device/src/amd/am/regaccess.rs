//! Runtime register access bound to a live device: resolves a register by
//! `(prefix, name)` + instance from discovery, then routes the access — GC/GCVM
//! through the per-XCC RLCG indirect channel (VF-gated), everything else
//! (MMHUB, OSSSYS, SDMA, NBIO, HDP) through direct MMIO.

use super::discovery::{self, Discovery};
use super::pci::Bar;
use super::regs::{self, RegDef};
use super::rlcg::RlcgChannel;
use crate::error::Error;

type Result<T> = std::result::Result<T, Error>;

pub struct Regs<'a> {
    mmio: &'a Bar,
    disc: &'a Discovery,
    /// One RLCG channel per GC instance (XCC), indexed by logical XCC.
    rlcg: Vec<RlcgChannel>,
}

impl<'a> Regs<'a> {
    pub fn new(mmio: &'a Bar, disc: &'a Discovery) -> Self {
        let rlcg =
            (0..disc.xccs() as u16).map(|xcc| RlcgChannel::new(&disc.regs_offset[&discovery::GC_HWIP][&xcc])).collect();
        Self { mmio, disc, rlcg }
    }

    fn def(&self, prefix: &str, name: &str) -> Result<&'static RegDef> {
        let ver = self.disc.ip_ver.get(&hwip_for(prefix)).copied().unwrap_or((0, 0, 0));
        let table = regs::select(prefix, ver)
            .ok_or_else(|| Error::Runtime { message: format!("no {prefix} module for {ver:?}") })?;
        regs::find(table, name).ok_or_else(|| Error::Runtime { message: format!("unknown register {name}") })
    }

    fn bases(&self, hwip: u16, inst: u16) -> &[u64] {
        &self.disc.regs_offset[&hwip][&inst]
    }

    // ── GC / GCVM (RLCG-indirect, per XCC) ─────────────────────────────────
    pub fn gc_read(&self, name: &str, xcc: u16) -> Result<u32> {
        let def = self.def("gc", name)?;
        let idx = def.dword_index(self.bases(discovery::GC_HWIP, xcc));
        self.rlcg[xcc as usize].read(self.mmio, idx)
    }

    pub fn gc_write(&self, name: &str, xcc: u16, val: u32) -> Result<()> {
        let def = self.def("gc", name)?;
        let idx = def.dword_index(self.bases(discovery::GC_HWIP, xcc));
        self.rlcg[xcc as usize].write(self.mmio, idx, val)
    }

    /// Compose `fields` into the register word and write it (GC/RLCG).
    pub fn gc_write_fields(&self, name: &str, xcc: u16, fields: &[(&str, u32)]) -> Result<()> {
        let def = self.def("gc", name)?;
        let idx = def.dword_index(self.bases(discovery::GC_HWIP, xcc));
        self.rlcg[xcc as usize].write(self.mmio, idx, def.encode(fields) as u32)
    }

    /// Read-modify-write `fields` into a GC register (RLCG), preserving the
    /// bits the PF/GIM set (a gated read of all-ones is treated as a fresh 0).
    pub fn gc_update(&self, name: &str, xcc: u16, fields: &[(&str, u32)]) -> Result<()> {
        let def = self.def("gc", name)?;
        let idx = def.dword_index(self.bases(discovery::GC_HWIP, xcc));
        let cur = self.rlcg[xcc as usize].read(self.mmio, idx)?;
        let base = if cur == u32::MAX { 0 } else { cur as u64 };
        self.rlcg[xcc as usize].write(self.mmio, idx, def.encode_onto(base, fields) as u32)
    }

    /// Encode `fields` into a GC register's word (no write) — for staging MQD
    /// dwords with the correct field bit positions.
    pub fn gc_encode(&self, name: &str, fields: &[(&str, u32)]) -> Result<u32> {
        Ok(self.def("gc", name)?.encode(fields) as u32)
    }

    /// Absolute MMIO dword index of a GC register on `xcc` (for raw blits).
    pub fn gc_index(&self, name: &str, xcc: u16) -> Result<usize> {
        Ok(self.def("gc", name)?.dword_index(self.bases(discovery::GC_HWIP, xcc)))
    }

    /// RLCG write to an absolute GC dword index (MQD register-block blit).
    pub fn gc_write_index(&self, xcc: u16, dword_idx: usize, val: u32) -> Result<()> {
        self.rlcg[xcc as usize].write(self.mmio, dword_idx, val)
    }

    /// Write a 64-bit value across `<name>_LO32`/`_HI32` (GC/RLCG).
    pub fn gc_write_pair(&self, base: &str, xcc: u16, val: u64) -> Result<()> {
        self.gc_write(&format!("{base}_LO32"), xcc, val as u32)?;
        self.gc_write(&format!("{base}_HI32"), xcc, (val >> 32) as u32)
    }

    // ── direct MMIO (MMHUB / OSSSYS / SDMA / NBIO / HDP) ────────────────────
    pub fn read(&self, prefix: &str, name: &str, inst: u16) -> Result<u32> {
        let def = self.def(prefix, name)?;
        Ok(self.mmio.read_u32(def.dword_index(self.bases(hwip_for(prefix), inst))))
    }

    pub fn write(&self, prefix: &str, name: &str, inst: u16, val: u32) -> Result<()> {
        let def = self.def(prefix, name)?;
        self.mmio.write_u32(def.dword_index(self.bases(hwip_for(prefix), inst)), val);
        Ok(())
    }

    pub fn write_fields(&self, prefix: &str, name: &str, inst: u16, fields: &[(&str, u32)]) -> Result<()> {
        let def = self.def(prefix, name)?;
        self.mmio.write_u32(def.dword_index(self.bases(hwip_for(prefix), inst)), def.encode(fields) as u32);
        Ok(())
    }

    /// Read-modify-write `fields` into a direct-MMIO register, preserving the
    /// bits the PF/GIM set (a gated read of all-ones is treated as a fresh 0).
    pub fn update(&self, prefix: &str, name: &str, inst: u16, fields: &[(&str, u32)]) -> Result<()> {
        let def = self.def(prefix, name)?;
        let idx = def.dword_index(self.bases(hwip_for(prefix), inst));
        let cur = self.mmio.read_u32(idx);
        let base = if cur == u32::MAX { 0 } else { cur as u64 };
        self.mmio.write_u32(idx, def.encode_onto(base, fields) as u32);
        Ok(())
    }

    pub fn xccs(&self) -> u16 {
        self.disc.xccs() as u16
    }

    // ── raw access for registers absent from the vendored tables ────────────
    /// Direct MMIO read at `bases[seg] + off` (dword) for `prefix`/`inst`.
    pub fn read_raw(&self, prefix: &str, inst: u16, seg: u8, off: u32) -> u32 {
        let idx = self.bases(hwip_for(prefix), inst)[seg as usize] as usize + off as usize;
        self.mmio.read_u32(idx)
    }

    /// Direct MMIO write at `bases[seg] + off` (dword) for `prefix`/`inst`.
    pub fn write_raw(&self, prefix: &str, inst: u16, seg: u8, off: u32, val: u32) {
        let idx = self.bases(hwip_for(prefix), inst)[seg as usize] as usize + off as usize;
        self.mmio.write_u32(idx, val);
    }

    /// Direct MMIO write at an absolute dword index (no base resolution).
    pub fn mmio_write_abs(&self, dword_idx: usize, val: u32) {
        if dword_idx * 4 + 4 <= self.mmio.len() {
            self.mmio.write_u32(dword_idx, val);
        }
    }
}

/// Map a register prefix to its discovery hwip index.
fn hwip_for(prefix: &str) -> u16 {
    match prefix {
        "gc" => discovery::GC_HWIP,
        "mmhub" => discovery::MMHUB_HWIP,
        "osssys" => discovery::OSSSYS_HWIP,
        "sdma" => discovery::SDMA0_HWIP,
        "nbio" => discovery::NBIO_HWIP,
        "hdp" => discovery::HDP_HWIP,
        "mp" => discovery::MP0_HWIP,
        p => panic!("unknown register prefix {p}"),
    }
}
