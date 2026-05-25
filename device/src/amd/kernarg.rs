//! `KernargArena`: shared bump allocator for AMDGPU kernel-argument buffers.
//!
//! One arena per device, sized at 16 MiB GTT-coherent. Each `Program::execute`
//! claims `kernarg_size` bytes (16-byte aligned per ABI). The arena wraps when
//! it fills — safe because the device's AQL queue completes packets FIFO, so
//! by the time the cursor laps an earlier slot that slot's dispatch has
//! finished consuming its kernargs.

#![cfg(target_os = "linux")]

use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::allocator::{Allocator, BufferOptions, RawBuffer};
use crate::amd::AmdAllocator;
use crate::error::{Error, Result};

const ARENA_BYTES: usize = 16 * 1024 * 1024;

pub struct KernargArena {
    pub base_gpu: u64,
    pub base_host: NonNull<u8>,
    pub size: usize,
    cursor: Mutex<usize>,
    _buffer: RawBuffer,
}

// SAFETY: cursor is Mutex-protected; pointer + buffer are stable for the
// arena's lifetime.
unsafe impl Send for KernargArena {}
unsafe impl Sync for KernargArena {}

impl KernargArena {
    pub fn new(allocator: &AmdAllocator) -> Result<Arc<Self>> {
        let opts = BufferOptions { zero_init: true, cpu_accessible: true, uncached: true, nolru: true };
        let buffer = allocator.alloc(ARENA_BYTES, &opts)?;
        let (base_gpu, base_host) = match &buffer {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::AmdAllocFailed { reason: "kernarg arena requires host-visible buffer".into() }),
        };
        Ok(Arc::new(Self { base_gpu, base_host, size: ARENA_BYTES, cursor: Mutex::new(0), _buffer: buffer }))
    }

    /// Reserve `size` bytes (aligned to `align`) and return the byte offset
    /// into the arena. Wraps to the beginning if `size` doesn't fit in the
    /// remaining space.
    pub fn bump(&self, size: usize, align: usize) -> Result<usize> {
        if size > self.size {
            return Err(Error::AmdAllocFailed {
                reason: format!("kernarg request {size} exceeds arena {}", self.size),
            });
        }
        let mut cur = self.cursor.lock();
        let aligned = (*cur).next_multiple_of(align);
        let (start, next) = if aligned + size > self.size { (0, size) } else { (aligned, aligned + size) };
        *cur = next;
        Ok(start)
    }

    pub fn gpu_at(&self, offset: usize) -> u64 {
        self.base_gpu + offset as u64
    }

    /// # Safety
    /// Caller must ensure `offset + size <= self.size` and that no concurrent
    /// writer holds the same slot. With FIFO AQL execution and bump
    /// semantics, the only producer of a given slot is the caller of `bump`.
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
mod tests {
    use super::*;

    #[test]
    fn kernarg_bump_wraps_when_full() {
        let alloc = match AmdAllocator::new(0) {
            Ok(a) => a,
            Err(_) => {
                eprintln!("skipping: no supported AMD GPU");
                return;
            }
        };
        let arena = KernargArena::new(&alloc).expect("arena");
        let half = arena.size / 2;
        let a = arena.bump(half, 16).expect("first");
        assert_eq!(a, 0);
        let b = arena.bump(half / 2, 16).expect("second");
        assert!(b > a && b < arena.size);
        // Now request something that would overflow → wrap.
        let c = arena.bump(arena.size - 16, 16).expect("third (wrap)");
        assert_eq!(c, 0, "expected wrap to start of arena");
    }
}
