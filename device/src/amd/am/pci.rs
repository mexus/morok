//! PCI device access for the AM driver: BAR mmap via sysfs + config space.
//!
//! Read-only opens map the BARs `PROT_READ` and never touch config space, so
//! the device can be inspected alongside a bound amdgpu (the M0 milestone).
//! Ownership mode (`readonly=false`) requires amdgpu unbound and enables
//! bus mastering for GPU-initiated DMA.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::fs::OpenOptionsExt;
use std::path::PathBuf;

use snafu::ensure;

use crate::error::{Error, RuntimeSnafu};

type Result<T> = std::result::Result<T, Error>;

const PCI_COMMAND: u64 = 0x04;
const PCI_COMMAND_MASTER: u16 = 1 << 2;

/// One mapped PCI BAR.
pub struct Bar {
    ptr: *mut u8,
    len: usize,
}

// The mapping is plain memory from the process's point of view; volatile
// accessors below keep MMIO semantics.
unsafe impl Send for Bar {}
unsafe impl Sync for Bar {}

impl Bar {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    /// Volatile dword read at dword index `idx`. Bounds are checked in release
    /// too: the index derives from device-supplied IP-discovery bases.
    #[inline]
    pub fn read_u32(&self, idx: usize) -> u32 {
        assert!(idx * 4 + 4 <= self.len, "BAR read_u32 out of bounds");
        unsafe { (self.ptr as *const u32).add(idx).read_volatile() }
    }

    /// Volatile dword write at dword index `idx`.
    #[inline]
    pub fn write_u32(&self, idx: usize, val: u32) {
        assert!(idx * 4 + 4 <= self.len, "BAR write_u32 out of bounds");
        unsafe { (self.ptr as *mut u32).add(idx).write_volatile(val) }
    }

    /// Volatile qword write at qword index `idx` (doorbell64).
    #[inline]
    pub fn write_u64(&self, idx: usize, val: u64) {
        assert!(idx * 8 + 8 <= self.len, "BAR write_u64 out of bounds");
        unsafe { (self.ptr as *mut u64).add(idx).write_volatile(val) }
    }

    /// Volatile byte read/write (mailbox control bytes).
    #[inline]
    pub fn read_u8(&self, off: usize) -> u8 {
        assert!(off < self.len, "BAR read_u8 out of bounds");
        unsafe { self.ptr.add(off).read_volatile() }
    }

    #[inline]
    pub fn write_u8(&self, off: usize, val: u8) {
        assert!(off < self.len, "BAR write_u8 out of bounds");
        unsafe { self.ptr.add(off).write_volatile(val) }
    }

    /// Copy `out.len()` bytes from BAR offset `off` (e.g. the discovery table
    /// out of the VRAM BAR).
    pub fn read_bytes(&self, off: usize, out: &mut [u8]) {
        assert!(off + out.len() <= self.len, "read past BAR end");
        unsafe { std::ptr::copy_nonoverlapping(self.ptr.add(off), out.as_mut_ptr(), out.len()) };
    }

    /// Copy `src` into BAR offset `off` (CPU → VRAM through the aperture).
    pub fn write_bytes(&self, off: usize, src: &[u8]) {
        assert!(off + src.len() <= self.len, "write past BAR end");
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), self.ptr.add(off), src.len()) };
    }
}

impl Drop for Bar {
    fn drop(&mut self) {
        unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
    }
}

/// An AMD GPU on the PCI bus, addressed by BDF (`0000:ff:00.0`).
pub struct PciDevice {
    pub bdf: String,
    /// BAR0: VRAM aperture (CPU-visible framebuffer).
    pub vram: Bar,
    /// BAR2: 64-bit doorbells.
    pub doorbell: Bar,
    /// BAR5: register MMIO.
    pub mmio: Bar,
    cfg: File,
    readonly: bool,
}

impl PciDevice {
    /// Open by BDF. `readonly` maps the BARs `PROT_READ` and opens config
    /// space read-only — safe alongside a bound kernel driver.
    pub fn open(bdf: &str, readonly: bool) -> Result<Self> {
        let dir = PathBuf::from(format!("/sys/bus/pci/devices/{bdf}"));
        ensure!(dir.exists(), RuntimeSnafu { message: format!("no PCI device {bdf}") });
        let vendor = std::fs::read_to_string(dir.join("vendor")).unwrap_or_default();
        ensure!(
            vendor.trim() == "0x1002",
            RuntimeSnafu { message: format!("{bdf}: vendor {} is not AMD", vendor.trim()) }
        );

        let map_bar = |n: u32| -> Result<Bar> {
            let path = dir.join(format!("resource{n}"));
            let file = OpenOptions::new()
                .read(true)
                .write(!readonly)
                .custom_flags(libc::O_SYNC)
                .open(&path)
                .map_err(|e| Error::Runtime { message: format!("open {path:?}: {e}") })?;
            let len = file.metadata().expect("BAR metadata").len() as usize;
            let prot = if readonly { libc::PROT_READ } else { libc::PROT_READ | libc::PROT_WRITE };
            let ptr = unsafe { libc::mmap(std::ptr::null_mut(), len, prot, libc::MAP_SHARED, file.as_raw_fd(), 0) };
            ensure!(
                ptr != libc::MAP_FAILED,
                RuntimeSnafu { message: format!("mmap BAR{n} ({len} bytes): {}", std::io::Error::last_os_error()) }
            );
            // Keep the BAR mapping out of forked children. `MADV_DONTFORK` is
            // Linux-only in `libc`; elsewhere this path is dead at runtime (no
            // sysfs PCI), so skipping it only affects an unreachable code path.
            #[cfg(target_os = "linux")]
            unsafe {
                libc::madvise(ptr, len, libc::MADV_DONTFORK);
            }
            Ok(Bar { ptr: ptr as *mut u8, len })
        };

        let (vram, doorbell, mmio) = (map_bar(0)?, map_bar(2)?, map_bar(5)?);
        let cfg = OpenOptions::new()
            .read(true)
            .write(!readonly)
            .custom_flags(libc::O_SYNC)
            .open(dir.join("config"))
            .map_err(|e| Error::Runtime { message: format!("open {bdf} config: {e}") })?;
        Ok(Self { bdf: bdf.into(), vram, doorbell, mmio, cfg, readonly })
    }

    /// First AMD display/accelerator function on the bus.
    pub fn discover() -> Result<String> {
        for entry in std::fs::read_dir("/sys/bus/pci/devices")
            .map_err(|e| Error::Runtime { message: format!("scan PCI bus: {e}") })?
            .flatten()
        {
            let dir = entry.path();
            let is_amd = std::fs::read_to_string(dir.join("vendor")).is_ok_and(|v| v.trim() == "0x1002");
            let class = std::fs::read_to_string(dir.join("class")).unwrap_or_default();
            // Display controller (0x03) or processing accelerator (0x12).
            if is_amd && (class.starts_with("0x03") || class.starts_with("0x12")) {
                return Ok(entry.file_name().to_string_lossy().into_owned());
            }
        }
        Err(Error::NoAmdGpu { reason: "no AMD display/accelerator on the PCI bus".into() })
    }

    pub fn config_read16(&mut self, off: u64) -> u16 {
        let mut b = [0u8; 2];
        self.cfg.seek(SeekFrom::Start(off)).and_then(|_| self.cfg.read_exact(&mut b)).expect("config read");
        u16::from_le_bytes(b)
    }

    pub fn config_write16(&mut self, off: u64, val: u16) {
        assert!(!self.readonly, "config write in read-only mode");
        self.cfg.seek(SeekFrom::Start(off)).and_then(|_| self.cfg.write_all(&val.to_le_bytes())).expect("config write");
        let _ = self.config_read16(off); // flush posted write
    }

    /// Enable bus mastering (GPU-initiated DMA). Ownership mode only.
    pub fn enable_bus_master(&mut self) {
        let cmd = self.config_read16(PCI_COMMAND);
        self.config_write16(PCI_COMMAND, cmd | PCI_COMMAND_MASTER);
    }
}
