//! `KernargArena`: bump allocator for AMDGPU kernel-argument buffers.
//!
//! Sized at 16 MiB GTT-coherent. **One per `PoolQueue`**. A dispatch holds the
//! queue's dispatch lock across bump + kernarg write + ring submission, so the
//! bump cursor order matches the ring order — a wrapped slot is provably free
//! once the whole pool drains. Each `Program::execute` claims `kernarg_size`
//! bytes (16-byte aligned per ABI). The arena wraps when it fills; on wrap we
//! drain every live `PoolQueue` via the arena's owning `AmdDeviceCore` —
//! without that drain a wrap can clobber kernargs the GPU is still consuming
//! (the host can sprint ahead of the GPU on a `wait=false` burst).

#![cfg(target_os = "linux")]

use std::ptr::NonNull;
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDeviceCore;
use crate::error::{Error, Result};

const ARENA_BYTES: usize = 16 * 1024 * 1024;

pub struct KernargArena {
    pub base_gpu: u64,
    pub base_host: NonNull<u8>,
    pub size: usize,
    cursor: Mutex<usize>,
    /// Back-reference to the device core, used to drain every live pool queue
    /// on wrap (`AmdDeviceCore::synchronize_all`). `Weak` because the program
    /// that owns this arena also indirectly owns the core via `Arc<AmdDevice>`
    /// — a strong handle here would form a cycle.
    core: Weak<AmdDeviceCore>,
    _buffer: RawBuffer,
}

// SAFETY: cursor is Mutex-protected; pointer + buffer are stable for the
// arena's lifetime.
unsafe impl Send for KernargArena {}
unsafe impl Sync for KernargArena {}

impl Drop for KernargArena {
    /// Free the 16 MiB GTT-coherent backing. `RawBuffer` lacks a `Drop` (the
    /// allocator path consumes it by destructure), so the arena — owned
    /// directly by `PoolQueue` — would otherwise leak its allocation every
    /// time a queue drops. Safe against unmap-while-busy because
    /// `PoolQueue::Drop` drains the queue before the `arena` field (and hence
    /// this `Drop`) runs.
    fn drop(&mut self) {
        self._buffer.free_amd_device_in_place();
    }
}

impl KernargArena {
    pub fn new(allocator: &AmdAllocator, core: &Arc<AmdDeviceCore>) -> Result<Box<Self>> {
        // Allocate as COHERENT|UNCACHED GTT (the ring/GART/signal mapping), NOT
        // via `alloc()` — `AmdAllocator::_alloc` ignores `BufferSpec::uncached`
        // and would place the arena in write-combining VRAM. Kernargs are written
        // by the host every dispatch and read by the CP; on a WC mapping the
        // doorbell path (a no-op `Release` fence on x86, not an `sfence`) does not
        // drain the WC buffer, so the CP can fetch a STALE kernarg — a wrong/freed
        // buffer pointer — and wedge with no fault event. UncachedGtt is coherent,
        // so the write is visible without a WC flush.
        let buffer = allocator.alloc_uncached(ARENA_BYTES)?;
        let (base_gpu, base_host) = match &buffer {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::AmdAllocFailed { reason: "kernarg arena requires host-visible buffer".into() }),
        };
        Ok(Box::new(Self {
            base_gpu,
            base_host,
            size: ARENA_BYTES,
            cursor: Mutex::new(0),
            core: Arc::downgrade(core),
            _buffer: buffer,
        }))
    }

    /// Reserve `size` bytes (aligned to `align`) and return the byte offset
    /// into the arena. Wraps to the beginning if `size` doesn't fit — and on
    /// wrap drains every live connector first so we don't overwrite kernargs
    /// the GPU is still reading.
    pub fn bump(&self, size: usize, align: usize) -> Result<usize> {
        if size > self.size {
            return Err(Error::AmdAllocFailed {
                reason: format!("kernarg request {size} exceeds arena {}", self.size),
            });
        }
        let mut cur = self.cursor.lock();
        let aligned = (*cur).next_multiple_of(align);
        if aligned + size > self.size {
            // Wrap. Drop the cursor lock before the potentially multi-second
            // drain so other threads aren't blocked; then re-take and reset.
            drop(cur);
            if let Some(core) = self.core.upgrade()
                && let Err(e) = core.synchronize_all()
            {
                // A poisoned device is the only way this fails on the happy
                // path; the caller will hit the same error on the very next
                // dispatch anyway. Warn and proceed: the host will clobber
                // some slot, but the GPU is also dead, so it's moot.
                tracing::warn!(?e, "kernarg arena wrap: synchronize_all failed");
            }
            let mut cur = self.cursor.lock();
            *cur = size;
            return Ok(0);
        }
        *cur = aligned + size;
        Ok(aligned)
    }

    pub fn gpu_at(&self, offset: usize) -> u64 {
        self.base_gpu + offset as u64
    }

    /// # Safety
    /// Caller must ensure `offset + size <= self.size` and that no concurrent
    /// writer holds the same slot. The dispatch lock held across bump + write +
    /// dispatch (in `execute_on`) guarantees the only producer of a given slot
    /// is the caller of `bump`, and that the GPU reads it before it is reused.
    pub unsafe fn host_at(&self, offset: usize) -> *mut u8 {
        unsafe { self.base_host.as_ptr().add(offset) }
    }
}

impl std::fmt::Debug for KernargArena {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernargArena")
            .field("base_gpu", &format_args!("{:#x}", self.base_gpu))
            .field("size", &self.size)
            .field("cursor", &*self.cursor.lock())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
#[path = "../test/unit/amd/kernarg.rs"]
mod tests;
