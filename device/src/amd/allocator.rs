//! `AmdAllocator`: KFD-direct VRAM/GTT allocator.
//!
//! Mirrors tinygrad's `ops_amd.py:748-776`. Each `alloc` reserves a host VA
//! via PROT_NONE mmap, hands the VA to `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU`,
//! optionally maps it host-visible via `mmap(drm_fd, ...)`, and binds it
//! into the GPU page table with `AMDKFD_IOC_MAP_MEMORY_TO_GPU`.
//!
//! Always-compiled-on-Linux: construction returns `Err(NoAmdGpu)` cleanly on
//! hosts without `/dev/kfd`, so the existence of `AmdAllocator` doesn't gate
//! anything; the runtime AMD path simply isn't reachable.

#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;
use std::ptr::NonNull;
use std::sync::Arc;

use libc::{
    MAP_ANONYMOUS, MAP_FIXED, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE, mmap, munmap,
};
use svod_dtype::DeviceSpec;
use tracing::debug;

use crate::allocator::{Allocator, BufferSpec, RawBuffer};
use crate::amd::device::AmdDevice;
use crate::amd::sys::{ioctl, kfd};
use crate::error::{Error, Result, UnsupportedSnafu};

/// VRAM-/GTT-backed buffer allocator routed through KFD ioctls.
pub struct AmdAllocator {
    pub dev: Arc<AmdDevice>,
    pub device_id: usize,
}

impl AmdAllocator {
    /// Open the `device_id`-th KFD GPU node and bind a VM.
    ///
    /// Returns `Err(NoAmdGpu)` cleanly when the host has no AMD GPU, no
    /// `/dev/kfd`, or the index is out of range. Never panics.
    pub fn new(device_id: usize) -> Result<Self> {
        let dev = AmdDevice::open(device_id)?;
        Ok(Self { dev, device_id })
    }

    /// Allocate GTT-pinned system memory with `COHERENT | UNCACHED | PUBLIC`
    /// flags — host-visible, uncached, suitable for queue rings, GART pages,
    /// and signal slots. Mirrors tinygrad's
    /// `alloc(uncached=True, cpu_access=True)` at `ops_amd.py:751-757`:
    /// the `uncached` branch sets **GTT**, not VRAM (uncached and VRAM are
    /// mutually exclusive in tinygrad's flag composition).
    pub fn alloc_uncached(&self, size: usize) -> Result<RawBuffer> {
        do_alloc(
            &self.dev,
            size,
            kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
                | kfd::KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
            /*cpu_accessible=*/ true,
            /*zero_init=*/ true,
        )
    }
}

impl std::fmt::Debug for AmdAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdAllocator")
            .field("device_id", &self.device_id)
            .field("arch", &self.dev.arch.mcpu())
            .field("gpu_id", &self.dev.node.gpu_id)
            .finish()
    }
}

impl Allocator for AmdAllocator {
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        let mut flags = kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE;
        if options.cpu_access {
            flags |= kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC;
        }
        do_alloc(&self.dev, size, flags, options.cpu_access, zero)
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        match dest {
            RawBuffer::AmdDevice { host_ptr: Some(ptr), .. } => {
                // Drain first: a recycled VA may still be referenced by an
                // in-flight kernel (dispatch is async + the LRU recycles
                // without syncing). Direct host writes aren't ordered on the
                // GPU timeline, so synchronize. Matches tinygrad's
                // no-copy-queue `_copyin` (hcq.py:578).
                self.dev.synchronize()?;
                // SAFETY: BAR-backed VRAM mapping valid for the buffer's lifetime;
                // scheduler exclusivity. `dest_off + src.len()` is bounded by the caller.
                let dst = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().add(dest_off), src.len()) };
                dst.copy_from_slice(src);
                Ok(())
            }
            RawBuffer::AmdDevice { host_ptr: None, .. } => {
                todo!("Phase 3: copyin into device-only AMD VRAM via SDMA")
            }
            other => unreachable!("AmdAllocator::_copyin on non-AMD buffer: {other:?}"),
        }
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::AmdDevice { host_ptr: Some(ptr), .. } => {
                // Dispatch is async (`AmdProgram::execute` does not block), so
                // drain the device timeline before reading GPU-written results.
                // Mirrors tinygrad `HCQAllocator._copyout` (hcq.py:613).
                self.dev.synchronize()?;
                let src_slice = unsafe { std::slice::from_raw_parts(ptr.as_ptr().add(src_off), dest.len()) };
                dest.copy_from_slice(src_slice);
                Ok(())
            }
            RawBuffer::AmdDevice { host_ptr: None, .. } => {
                todo!("Phase 3: copyout from device-only AMD VRAM via SDMA")
            }
            other => unreachable!("AmdAllocator::_copyout on non-AMD buffer: {other:?}"),
        }
    }

    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        match (dest, src) {
            (
                RawBuffer::AmdDevice { host_ptr: Some(dst_ptr), .. },
                RawBuffer::AmdDevice { host_ptr: Some(src_ptr), .. },
            ) => {
                let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr.as_ptr().add(dest_off), sz) };
                let src_slice = unsafe { std::slice::from_raw_parts(src_ptr.as_ptr().add(src_off), sz) };
                dst.copy_from_slice(src_slice);
                Ok(())
            }
            (RawBuffer::AmdDevice { .. }, RawBuffer::AmdDevice { .. }) => {
                todo!("Phase 3: AMD↔AMD transfer involving device-only VRAM via SDMA")
            }
            _ => UnsupportedSnafu { op: "transfer" }.fail(),
        }
    }

    fn _free(&self, buffer: RawBuffer, _options: &BufferSpec) {
        let (gpu_addr, host_ptr, size, handle, device) = match buffer {
            RawBuffer::AmdDevice { gpu_addr, host_ptr, size, handle, device } => {
                (gpu_addr, host_ptr, size, handle, device)
            }
            // Wrong-allocator-for-buffer-type would be a programming bug.
            // Falling through means we leak the buffer; CPU/CUDA arms would
            // just drop their backing storage.
            other => {
                debug!(?other, "AmdAllocator::free called with non-AMD buffer; dropping");
                return;
            }
        };
        // 0. Drain the device's submitted work before tearing down the
        //    mapping. Mirrors tinygrad `HCQAllocatorBase._free` at
        //    `hcq.py:566` (`for dev in buf.mapped_devs: dev.synchronize()`).
        //    Without this, the GPU can still hold pending references to
        //    `gpu_addr` when we call `unmap_memory_from_gpu` below — the
        //    KFD tears down the page-table entries and the kernel then
        //    faults at the now-orphaned VA. Logged-and-ignore on failure:
        //    free is called from `Drop`, so we can't propagate.
        if let Err(e) = device.synchronize() {
            tracing::warn!(?e, gpu_addr, "AmdAllocator::free: device synchronize failed; freeing anyway");
        }
        // 1. Unmap from GPU.
        let mut gpu_id = device.node.gpu_id;
        let mut unmap_args = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
            handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        // SAFETY: fd is alive; handle is from a successful alloc.
        let _ = unsafe { ioctl::kfd_unmap_memory_from_gpu(device.kfd_fd.as_raw_fd(), &mut unmap_args as *mut _) };
        // 2. Drop host mapping (PROT_READ|PROT_WRITE for host-visible, or the
        //    PROT_NONE reservation for device-only). Both cases munmap the
        //    same VA region.
        let _ = host_ptr;
        // SAFETY: gpu_addr is the VA returned by our own mmap.
        unsafe { munmap(gpu_addr as *mut _, size) };
        // 3. Free the KFD allocation.
        let mut free_args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
        // SAFETY: same as above.
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(device.kfd_fd.as_raw_fd(), &mut free_args as *mut _) };
    }

    fn name(&self) -> &str {
        "AMD"
    }

    fn device_spec(&self) -> DeviceSpec {
        DeviceSpec::Amd { device_id: self.device_id }
    }
}

/// Shared body for VRAM and GTT allocations. Differences (flag bits) are
/// already encoded in `flags`; everything else (VA reservation, KFD alloc,
/// host mmap, map_memory_to_gpu) is identical.
fn do_alloc(dev: &Arc<AmdDevice>, size: usize, flags: u32, cpu_accessible: bool, zero_init: bool) -> Result<RawBuffer> {
    // KFD VA reservation + map are page-granular; a 0-byte mmap is EINVAL.
    let size = size.max(1).next_multiple_of(0x1000);
    let va = reserve_va(size)?;
    let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
        va_addr: va as u64,
        size: size as u64,
        gpu_id: dev.node.gpu_id,
        flags,
        ..Default::default()
    };
    if let Err(e) = unsafe { ioctl::kfd_alloc_memory_of_gpu(dev.kfd_fd.as_raw_fd(), &mut args as *mut _) } {
        unsafe { munmap(va as *mut _, size) };
        return Err(map_alloc_err(e, cpu_accessible));
    }
    let mem_handle = args.handle;
    let mmap_offset = args.mmap_offset;

    let host_ptr = if cpu_accessible {
        let p = unsafe {
            mmap(
                va as *mut _,
                size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED | MAP_FIXED,
                dev.drm_fd.as_raw_fd(),
                mmap_offset as i64,
            )
        };
        if p == libc::MAP_FAILED || !std::ptr::eq(p, va) {
            free_kfd(dev, mem_handle);
            unsafe { munmap(va as *mut _, size) };
            return Err(Error::AmdAllocFailed {
                reason: "host-visible mmap failed (enable resizable BAR for VRAM, or check GTT availability)".into(),
            });
        }
        Some(unsafe { NonNull::new_unchecked(p as *mut u8) })
    } else {
        None
    };

    let mut gpu_id = dev.node.gpu_id;
    let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
        handle: mem_handle,
        device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
        n_devices: 1,
        n_success: 0,
    };
    if let Err(e) = unsafe { ioctl::kfd_map_memory_to_gpu(dev.kfd_fd.as_raw_fd(), &mut map_args as *mut _) } {
        free_kfd(dev, mem_handle);
        unsafe { munmap(va as *mut _, size) };
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU", errno: e as i32 });
    }
    if map_args.n_success != 1 {
        free_kfd(dev, mem_handle);
        unsafe { munmap(va as *mut _, size) };
        return Err(Error::AmdAllocFailed {
            reason: format!("KFD map_memory_to_gpu reported {} success(es)", map_args.n_success),
        });
    }

    if zero_init && let Some(p) = host_ptr {
        unsafe { std::ptr::write_bytes(p.as_ptr(), 0, size) };
    }

    debug!(size, gpu_addr = va as u64, "AmdAllocator alloc done");

    Ok(RawBuffer::AmdDevice { gpu_addr: va as u64, host_ptr, size, handle: mem_handle, device: Arc::clone(dev) })
}

/// Reserve `size` bytes of host VA so KFD can bind VRAM into it.
fn reserve_va(size: usize) -> Result<*mut libc::c_void> {
    // SAFETY: standard libc::mmap signature; no aliasing concerns at this point.
    let p = unsafe { mmap(std::ptr::null_mut(), size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
    if p == libc::MAP_FAILED {
        let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
        return Err(Error::AmdAllocFailed { reason: format!("VA reservation mmap failed (errno {errno})") });
    }
    Ok(p)
}

fn free_kfd(dev: &AmdDevice, handle: u64) {
    let mut args = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
    // SAFETY: dev.kfd_fd is alive; handle is from a successful alloc.
    let _ = unsafe { ioctl::kfd_free_memory_of_gpu(dev.kfd_fd.as_raw_fd(), &mut args as *mut _) };
}

fn map_alloc_err(e: nix::errno::Errno, cpu_accessible: bool) -> Error {
    match e {
        nix::errno::Errno::ENOMEM => Error::AmdAllocFailed { reason: "ENOMEM (VRAM exhausted)".into() },
        nix::errno::Errno::EINVAL if cpu_accessible => {
            Error::AmdAllocFailed { reason: "EINVAL on host-visible VRAM alloc — enable resizable BAR".into() }
        }
        other => Error::AmdIoctl { ioctl: "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", errno: other as i32 },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction either succeeds (real hardware + supported arch) or
    /// returns a clean error variant; never panics.
    #[test]
    fn allocator_construction_is_clean() {
        match AmdAllocator::new(0) {
            Ok(_alloc) => {}
            Err(Error::NoAmdGpu { .. }) | Err(Error::AmdAllocFailed { .. }) | Err(Error::AmdIoctl { .. }) => {}
            Err(e) => panic!("unexpected error: {e:?}"),
        }
    }

    /// Live VRAM alloc → free round-trip. Skipped on hosts that can't open an
    /// AmdDevice (no GPU, unsupported arch, missing perms).
    #[test]
    fn alloc_free_roundtrip_if_hw_supports() {
        let alloc = match AmdAllocator::new(0) {
            Ok(a) => a,
            Err(_) => {
                eprintln!("skipping: AmdAllocator::new failed (no supported AMD GPU on this host)");
                return;
            }
        };
        let opts = BufferSpec { cpu_access: true, ..Default::default() };
        let buf = alloc.alloc(4096, &opts, /*zero=*/ true).expect("alloc 4 KiB");
        assert_eq!(buf.size(), 4096);
        assert!(buf.cpu_accessible());
        alloc.free(buf, 4096, &opts);
    }
}
