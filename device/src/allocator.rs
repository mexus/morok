use std::alloc::Layout;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::ptr::NonNull;
use std::sync::Mutex;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaSlice, UnifiedSlice};
#[cfg(feature = "cuda")]
use snafu::ResultExt;
#[cfg(feature = "cuda")]
use std::sync::Arc;

use crate::error::*;

/// 64-byte aligned buffer for SIMD operations (covers SSE/AVX/AVX-512).
///
/// The C codegen emits vector types with alignment attributes (e.g. `aligned(32)` for
/// `double4`). Clang then generates aligned load/store instructions (`vmovaps`) that
/// segfault on unaligned pointers. This buffer guarantees all allocations are
/// 64-byte aligned to satisfy any current SIMD width.
pub struct AlignedBuffer {
    ptr: NonNull<u8>,
    len: usize,
}

const BUFFER_ALIGN: usize = 64;

impl AlignedBuffer {
    pub fn new_zeroed(size: usize) -> Self {
        if size == 0 {
            return Self { ptr: NonNull::dangling(), len: 0 };
        }
        let layout = Layout::from_size_align(size, BUFFER_ALIGN).expect("invalid buffer layout");
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        Self { ptr, len: size }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Deref for AlignedBuffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        if self.len == 0 { &[] } else { unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) } }
    }
}

impl DerefMut for AlignedBuffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        if self.len == 0 { &mut [] } else { unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) } }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.len > 0 {
            let layout = Layout::from_size_align(self.len, BUFFER_ALIGN).unwrap();
            unsafe { std::alloc::dealloc(self.ptr.as_ptr(), layout) };
        }
    }
}

/// Opaque handle to device memory.
///
/// # Safety
///
/// `RawBuffer` uses `UnsafeCell` for interior mutability without locking overhead.
/// Thread safety is guaranteed at a higher level by the scheduler:
///
/// 1. **Allocation**: `OnceLock` in `BufferData` ensures single initialization
/// 2. **Buffer Access**: The scheduler guarantees exclusive access to each buffer
///    during kernel execution - no two kernels access the same buffer concurrently
/// 3. **Kernel Execution**: Raw pointers passed to JIT code; Rust doesn't access
///    buffer data during execution
///
/// This design follows Tinygrad's approach where buffer synchronization is the
/// scheduler's responsibility, not the buffer's.
pub enum RawBuffer {
    Cpu {
        data: UnsafeCell<AlignedBuffer>,
        cpu_accessible: bool,
    },
    /// Memory-mapped file region (read-only). Used by DISK device.
    Mmap {
        data: memmap2::Mmap,
        size: usize,
    },
    #[cfg(feature = "cuda")]
    CudaDevice {
        data: UnsafeCell<CudaSlice<u8>>,
        device: Arc<CudaContext>,
    },
    #[cfg(feature = "cuda")]
    CudaUnified {
        data: UnsafeCell<UnifiedSlice<u8>>,
        device: Arc<CudaContext>,
    },
    /// AMD GPU VRAM/GTT buffer allocated via KFD ioctls.
    ///
    /// `gpu_addr` is the GPU virtual address that kernels see in their
    /// kernarg slot. `host_ptr` is `Some(_)` only when `cpu_accessible`; the
    /// pointer is a host-side mmap of the same buffer, suitable for memcpy.
    /// `handle` is KFD's opaque allocation handle, used for the matching
    /// free/unmap ioctls. `device` keeps the underlying KFD/DRM fds alive
    /// for the lifetime of the buffer.
    #[cfg(target_os = "linux")]
    AmdDevice {
        gpu_addr: u64,
        host_ptr: Option<std::ptr::NonNull<u8>>,
        size: usize,
        handle: u64,
        device: std::sync::Arc<crate::amd::AmdDevice>,
    },
}

// SAFETY: RawBuffer access is synchronized by the scheduler at a higher level.
// See RawBuffer documentation for detailed safety invariants.
unsafe impl Send for RawBuffer {}
unsafe impl Sync for RawBuffer {}

// UnsafeCell doesn't implement Debug, so we implement it manually
impl std::fmt::Debug for RawBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RawBuffer::Cpu { cpu_accessible, .. } => {
                f.debug_struct("Cpu").field("cpu_accessible", cpu_accessible).finish_non_exhaustive()
            }
            RawBuffer::Mmap { size, .. } => f.debug_struct("Mmap").field("size", size).finish_non_exhaustive(),
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { device, .. } => {
                f.debug_struct("CudaDevice").field("device", device).finish_non_exhaustive()
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { device, .. } => {
                f.debug_struct("CudaUnified").field("device", device).finish_non_exhaustive()
            }
            #[cfg(target_os = "linux")]
            RawBuffer::AmdDevice { gpu_addr, size, host_ptr, .. } => f
                .debug_struct("AmdDevice")
                .field("gpu_addr", gpu_addr)
                .field("size", size)
                .field("cpu_accessible", &host_ptr.is_some())
                .finish_non_exhaustive(),
        }
    }
}

impl RawBuffer {
    /// Get the size of the buffer in bytes.
    pub fn size(&self) -> usize {
        // SAFETY: Reading .len() doesn't alias with content access and is immutable after allocation
        match self {
            RawBuffer::Cpu { data, .. } => unsafe { (&*data.get()).len() },
            RawBuffer::Mmap { size, .. } => *size,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, .. } => unsafe { (&*data.get()).len() },
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, .. } => unsafe { (&*data.get()).len() },
            #[cfg(target_os = "linux")]
            RawBuffer::AmdDevice { size, .. } => *size,
        }
    }

    /// Get whether this buffer is CPU-accessible.
    pub fn cpu_accessible(&self) -> bool {
        match self {
            RawBuffer::Cpu { cpu_accessible, .. } => *cpu_accessible,
            RawBuffer::Mmap { .. } => true,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { .. } => false,
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { .. } => true,
            #[cfg(target_os = "linux")]
            RawBuffer::AmdDevice { host_ptr, .. } => host_ptr.is_some(),
        }
    }
}

/// Options for buffer allocation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
pub struct BufferOptions {
    /// Whether to zero-initialize the buffer.
    pub zero_init: bool,
    /// Whether this buffer is CPU-accessible.
    ///
    /// CPU allocator: always true (host memory is always accessible).
    /// CUDA allocator: false = device-only (cuMemAlloc), true = unified (cuMemAllocManaged).
    pub cpu_accessible: bool,
}

impl Default for BufferOptions {
    fn default() -> Self {
        Self { zero_init: false, cpu_accessible: true }
    }
}

pub trait Allocator: Send + Sync + std::fmt::Debug {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer>;
    fn free(&self, _buffer: RawBuffer, _options: &BufferOptions) {}
    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
    fn name(&self) -> &str;

    /// Get the device specification for this allocator.
    fn device_spec(&self) -> svod_dtype::DeviceSpec;
}

/// CPU allocator using system memory.
#[derive(Debug, Clone)]
pub struct CpuAllocator;

impl Allocator for CpuAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        let data = AlignedBuffer::new_zeroed(size);
        Ok(RawBuffer::Cpu { data: UnsafeCell::new(data), cpu_accessible: options.cpu_accessible })
    }

    fn name(&self) -> &str {
        "CPU"
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        svod_dtype::DeviceSpec::Cpu
    }
}

/// DISK allocator using memory-mapped files (Tinygrad: ops_disk.py).
/// Read-only — cannot execute kernels. Data is transferred via COPY.
#[derive(Debug, Clone)]
pub struct DiskAllocator {
    path: std::path::PathBuf,
}

impl DiskAllocator {
    pub fn new(path: std::path::PathBuf) -> Self {
        Self { path }
    }
}

impl Allocator for DiskAllocator {
    fn alloc(&self, size: usize, _options: &BufferOptions) -> Result<RawBuffer> {
        let file = std::fs::File::open(&self.path).map_err(|e| crate::Error::CopyFailed {
            reason: format!("DISK: failed to open {}: {e}", self.path.display()),
        })?;
        let file_size = file
            .metadata()
            .map_err(|e| crate::Error::CopyFailed {
                reason: format!("DISK: failed to read metadata for {}: {e}", self.path.display()),
            })?
            .len() as usize;
        if size > file_size {
            return Err(crate::Error::CopyFailed {
                reason: format!("DISK: requested {size} bytes but {} is only {file_size} bytes", self.path.display()),
            });
        }
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|e| crate::Error::CopyFailed {
            reason: format!("DISK: mmap failed for {}: {e}", self.path.display()),
        })?;
        Ok(RawBuffer::Mmap { data: mmap, size })
    }

    fn name(&self) -> &str {
        "DISK"
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        svod_dtype::DeviceSpec::Disk { path: self.path.clone() }
    }
}

/// CUDA allocator using GPU memory.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
pub struct CudaAllocator {
    device: Arc<CudaContext>,
    device_id: usize,
}

#[cfg(feature = "cuda")]
impl CudaAllocator {
    pub fn new(device_id: usize) -> Result<Self> {
        let device = CudaContext::new(device_id).context(CudaSnafu)?;
        Ok(Self { device, device_id })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

#[cfg(feature = "cuda")]
impl Allocator for CudaAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        if options.cpu_accessible {
            // Allocate unified memory (CPU-accessible)
            let mut data = unsafe { self.device.alloc_unified::<u8>(size, true) }.context(CudaSnafu)?;

            if options.zero_init {
                self.device.default_stream().memset_zeros(&mut data).context(CudaSnafu)?;
            }

            Ok(RawBuffer::CudaUnified { data: UnsafeCell::new(data), device: Arc::clone(&self.device) })
        } else {
            // Allocate device-only memory (faster GPU access)
            let stream = self.device.default_stream();
            let data =
                if options.zero_init { stream.alloc_zeros::<u8>(size) } else { unsafe { stream.alloc::<u8>(size) } }
                    .context(CudaSnafu)?;

            Ok(RawBuffer::CudaDevice { data: UnsafeCell::new(data), device: Arc::clone(&self.device) })
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.device.default_stream().synchronize().context(CudaSnafu)
    }

    fn name(&self) -> &str {
        "CUDA"
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        svod_dtype::DeviceSpec::Cuda { device_id: self.device_id }
    }
}

/// Cache key for buffer reuse in LRU allocator.
///
/// Includes size and cpu_accessible (hardware property that affects allocation).
/// zero_init is NOT included - it's a software operation handled after cache retrieval.
///
/// Design rationale (following Tinygrad):
/// - cpu_accessible is included because it represents different memory types:
///   - false: Device-only memory (cuMemAlloc) - faster GPU access
///   - true: Unified memory (cuMemAllocManaged) - CPU-accessible, not yet implemented
/// - These are immutable hardware properties that cannot be changed post-allocation
/// - Buffers allocated with different cpu_accessible values cannot be safely reused
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
struct CacheKey {
    size: usize,
    cpu_accessible: bool,
}

/// LRU allocator that caches freed buffers for reuse.
#[derive(Debug)]
pub(crate) struct LruAllocator {
    inner: Box<dyn Allocator>,
    cache: Mutex<HashMap<CacheKey, Vec<RawBuffer>>>,
    max_buffers_per_size: usize,
    name: String,
}

impl LruAllocator {
    pub fn new(inner: Box<dyn Allocator>) -> Self {
        Self::with_capacity(inner, 32)
    }

    pub fn with_capacity(inner: Box<dyn Allocator>, max_buffers_per_size: usize) -> Self {
        let name = inner.name().to_string();
        Self { inner, cache: Mutex::new(HashMap::new()), max_buffers_per_size, name }
    }

    /// Get the number of cached buffers for a specific size and cpu_accessible flag.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    pub(crate) fn cache_count(&self, size: usize, cpu_accessible: bool) -> usize {
        let key = CacheKey { size, cpu_accessible };
        let cache = self.cache.lock().unwrap();
        cache.get(&key).map(|v| v.len()).unwrap_or(0)
    }

    /// Get the total number of cached buffers across all keys.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    #[allow(dead_code)]
    pub(crate) fn total_cached(&self) -> usize {
        let cache = self.cache.lock().unwrap();
        cache.values().map(|v| v.len()).sum()
    }
}

impl Allocator for LruAllocator {
    fn alloc(&self, size: usize, options: &BufferOptions) -> Result<RawBuffer> {
        let key = CacheKey { size, cpu_accessible: options.cpu_accessible };

        // Try cache first
        let buffer = {
            let mut cache = self.cache.lock().unwrap();
            if let Some(buffers) = cache.get_mut(&key)
                && let Some(buffer) = buffers.pop()
            {
                if buffers.is_empty() {
                    cache.remove(&key);
                }
                Some(buffer)
            } else {
                None
            }
        }; // Drop lock before expensive allocation

        // If found in cache, optionally zero and return
        if let Some(buffer) = buffer {
            if options.zero_init {
                // Zero the cached buffer if requested
                // SAFETY: Buffer just retrieved from cache, not yet returned - no other references exist
                match &buffer {
                    RawBuffer::Cpu { data, .. } => {
                        unsafe { (*data.get()).fill(0) };
                    }
                    RawBuffer::Mmap { .. } => panic!("DISK device is read-only: cannot zero-init mmap buffer"),
                    #[cfg(target_os = "linux")]
                    RawBuffer::AmdDevice { host_ptr: Some(ptr), size, .. } => {
                        // SAFETY: Buffer just retrieved from cache; no other references.
                        unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, *size) };
                    }
                    #[cfg(target_os = "linux")]
                    RawBuffer::AmdDevice { host_ptr: None, .. } => {
                        // Device-only AMD VRAM: SDMA memset lands in Phase 5.
                        // For now, drop the cached buffer and force a fresh
                        // KFD alloc — never silently hand back un-zeroed data.
                        drop(buffer);
                        return self.inner.alloc(size, options);
                    }
                    #[cfg(feature = "cuda")]
                    RawBuffer::CudaDevice { data, device } => {
                        let cuda_data = unsafe { &mut *data.get() };
                        device.default_stream().memset_zeros(cuda_data).context(CudaSnafu)?;
                    }
                    #[cfg(feature = "cuda")]
                    RawBuffer::CudaUnified { data, device } => {
                        let unified_data = unsafe { &mut *data.get() };
                        device.default_stream().memset_zeros(unified_data).context(CudaSnafu)?;
                    }
                }
            }
            return Ok(buffer);
        }

        // Cache miss - allocate from inner
        match self.inner.alloc(size, options) {
            Ok(buffer) => Ok(buffer),
            Err(e) => {
                // On allocation failure, clear cache and retry
                self.cache.lock().unwrap().clear();
                self.inner.alloc(size, options).map_err(|_| e)
            }
        }
    }

    fn free(&self, buffer: RawBuffer, options: &BufferOptions) {
        let key = CacheKey { size: buffer.size(), cpu_accessible: options.cpu_accessible };

        // Push onto the per-key cache if space remains. When the cache is
        // full, route the overflow through `inner.free` so the underlying
        // allocator can actually release the handle. Without this, AMD/CUDA
        // backends leak the GPU mapping (RawBuffer has no Drop). Mirrors
        // tinygrad's `HCQAllocatorBase._free` eviction path
        // (`hcq.py:564-568`), which is also where tinygrad synchronizes the
        // device timeline before tearing down the GPU pagetable entries.
        let overflow = {
            let mut cache = self.cache.lock().unwrap();
            let buffers = cache.entry(key).or_default();
            if buffers.len() < self.max_buffers_per_size {
                buffers.push(buffer);
                None
            } else {
                Some(buffer)
            }
        };
        if let Some(buf) = overflow {
            self.inner.free(buf, options);
        }
    }

    fn synchronize(&self) -> Result<()> {
        self.inner.synchronize()
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn device_spec(&self) -> svod_dtype::DeviceSpec {
        self.inner.device_spec()
    }
}
