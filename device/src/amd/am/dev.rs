//! `AmDev`: the userspace-driver device object. Owns the PCI mapping, the
//! discovered IP layout, the GMMU memory manager (over the VRAM BAR), and the
//! IP blocks. VF bring-up = mailbox handshake → discovery → GMC contexts.

use crate::error::Error;

use super::discovery::{self, Discovery};
use super::ip::gmc::Gmc;
use super::mailbox::{Mailbox, NBIO_7_9_SEG2_BASE};
use super::mm::manager::{MemoryManager, gfx11_palloc_ranges};
use super::mm::pagetable::Geometry;
use super::mm::{VirtMapping, VramPhys};
use super::pci::PciDevice;
use super::regaccess::Regs;

type Result<T> = std::result::Result<T, Error>;

/// Global VA window (matches tinygrad's `AMMemoryManager.va_allocator`).
const VA_BASE: u64 = 0x2000_0000_0000;
const VA_SIZE: u64 = 1 << 44;

pub struct AmDev {
    pub pci: PciDevice,
    pub disc: Discovery,
    pub mailbox: Mailbox,
    pub mm: MemoryManager<VramPhys>,
    pub gmc: Gmc,
}

impl AmDev {
    /// Take the VF from the GIM and bring up the memory controller.
    /// Requires amdgpu unbound + root.
    pub fn open(bdf: &str) -> Result<Self> {
        let mut pci = PciDevice::open(bdf, false)?;
        pci.enable_bus_master()?;

        // Bootstrap: FB is gated until the GIM grants access via the mailbox,
        // so the handshake (hardcoded NBIO base) runs before discovery.
        let mailbox = Mailbox::new(NBIO_7_9_SEG2_BASE);
        if !mailbox.is_vf(&pci.mmio) {
            return Err(Error::Runtime {
                message: "not a VF: AM bring-up here only supports the SR-IOV guest flavor".into(),
            });
        }
        mailbox.request_init_access(&pci.mmio)?;

        // Now read + parse the discovery table out of VRAM.
        let vram_size = (pci.mmio.read_u32(discovery::MM_RCC_CONFIG_MEMSIZE) as u64) << 20;
        let mut tbl = vec![0u8; discovery::TABLE_SIZE];
        pci.vram.read_bytes((vram_size - discovery::TABLE_TAIL_OFFSET) as usize, &mut tbl);
        let disc = Discovery::parse(&tbl, vram_size)?;
        let gc_ver = disc.ip_ver[&discovery::GC_HWIP];
        if (gc_ver.0, gc_ver.1) != (9, 4) {
            return Err(Error::Runtime { message: format!("unsupported GC {gc_ver:?} (AM VF path targets gfx9.4.x)") });
        }

        // GMMU over the VRAM BAR. Pool excludes the reserved top region.
        let phys = VramPhys::new(pci.vram.as_ptr(), pci.vram.len());
        let pool = vram_size - disc.reserved_vram_size;
        let mut mm = MemoryManager::new(
            phys,
            gc_ver,
            Geometry::gfx11(), // gfx9 shares the 4-level / 48-bit shape
            pool,
            VA_BASE,
            VA_SIZE,
            false, // large BAR: no need to reserve a page-table slice
            gfx11_palloc_ranges(),
        );

        // Bring up both hubs: program context0 + invalidation ranges (GC per
        // XCC via RLCG, MM per AID direct), then flush. Page-table entries hold
        // MC addresses, so teach the manager the XGMI base before any mapping.
        let gmc = {
            let regs = Regs::new(&pci.mmio, &disc);
            let gmc = Gmc::probe(&regs, &disc, VA_BASE, VA_SIZE)?;
            mm.set_mc_base(gmc.mc_base);
            gmc.enable(&regs, mm.root_pt())?;
            gmc
        };

        Ok(Self { pci, disc, mailbox, mm, gmc })
    }

    /// A register accessor bound to this device.
    pub fn regs(&self) -> Regs<'_> {
        Regs::new(&self.pci.mmio, &self.disc)
    }

    /// Allocate a contiguous, VA-mapped VRAM buffer and flush the TLB so the
    /// GPU sees the new mapping. Returns the [`VirtMapping`] (its `va_addr` is
    /// the GPU VA; `paddrs[0].0` is the BAR0 offset for CPU access).
    pub fn valloc(&mut self, size: u64, uncached: bool) -> Result<VirtMapping> {
        let vm = self
            .mm
            .valloc(size, 0x1000, uncached, true)
            .ok_or_else(|| Error::Runtime { message: format!("VRAM valloc {size} failed") })?;
        let regs = Regs::new(&self.pci.mmio, &self.disc);
        self.gmc.flush_tlb(&regs, 0)?;
        Ok(vm)
    }

    /// CPU write into a VRAM buffer through BAR0 (by its physical address).
    pub fn vram_write(&self, paddr: u64, src: &[u8]) {
        self.pci.vram.write_bytes(paddr as usize, src);
    }

    /// CPU read from a VRAM buffer through BAR0 (by its physical address).
    pub fn vram_read(&self, paddr: u64, out: &mut [u8]) {
        self.pci.vram.read_bytes(paddr as usize, out);
    }

    /// Release exclusive access back to the GIM (best-effort on teardown).
    pub fn release(&self) {
        let _ = self.mailbox.release_init_access(&self.pci.mmio);
    }
}
