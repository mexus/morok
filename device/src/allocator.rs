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

/// Buffer allocation spec. Mirrors tinygrad `BufferSpec` (device.py:77-84): it
/// is the *whole* LRU cache key `(size, spec)`, hence `Hash + Eq + Copy`.
///
/// `zero_init` is intentionally NOT a field — tinygrad's allocator never zeroes
/// (`_alloc` returns raw memory); Svod threads it as a separate `alloc`
/// argument so it does not split the cache. A zeroed and a non-zeroed buffer of
/// the same spec are interchangeable, because a cache hit re-zeroes on demand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "proptest", derive(proptest_derive::Arbitrary))]
pub struct BufferSpec {
    /// GTT-coherent uncached memory (signal/ring/kernarg). Distinct cache type
    /// from VRAM — can't be reused as cached. tinygrad `BufferSpec.uncached`.
    pub uncached: bool,
    /// CPU-accessible mapping.
    ///
    /// CPU allocator: always honored (host memory is always accessible).
    /// CUDA allocator: false = device-only (cuMemAlloc), true = unified (cuMemAllocManaged).
    /// AMD allocator: adds a host BAR mmap (`host_ptr: Some`). tinygrad `BufferSpec.cpu_access`.
    pub cpu_access: bool,
    /// Host (GTT/userptr) memory rather than device VRAM. tinygrad `BufferSpec.host`.
    pub host: bool,
    /// Never cache this buffer in the LRU pool: free goes straight to teardown.
    /// For lifetime-bound buffers (code object, scratch, queue/signal infra).
    /// tinygrad `BufferSpec.nolru`.
    pub nolru: bool,
    /// Wraps a pre-existing pointer; bypasses LRU + ownership accounting.
    /// tinygrad `BufferSpec.external_ptr`.
    pub external_ptr: Option<usize>,
}

impl Default for BufferSpec {
    fn default() -> Self {
        Self { uncached: false, cpu_access: true, host: false, nolru: false, external_ptr: None }
    }
}

/// Device memory allocator. Mirrors tinygrad's `Allocator` (device.py:224-248):
/// the public `alloc`/`free` are thin wrappers over the runtime-implemented
/// `_alloc`/`_free`; copy/transfer/offset/map are overridable hooks that
/// default to "unsupported" (tinygrad raises `NotImplementedError`).
///
/// Object-safe (used as `Arc<dyn Allocator>`): the opaque is the single
/// [`RawBuffer`] enum, so copy hooks take `&RawBuffer` + an explicit byte
/// offset (the view offset lives on [`crate::Buffer`], not on `RawBuffer`).
pub trait Allocator: Send + Sync + std::fmt::Debug {
    /// Allocate `size` bytes. `zero` requests zero-initialized memory (a Svod
    /// extension applied on top of `_alloc`, not part of the cache key).
    fn alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        self._alloc(size, options, zero)
    }

    /// Free a buffer. `size` is the originally-requested allocation size (the
    /// LRU cache key, mirroring tinygrad `free(opaque, size, options)`); the
    /// base allocator ignores it and just releases the handle. The `RawBuffer`
    /// is consumed (and dropped) here.
    fn free(&self, buffer: RawBuffer, size: usize, options: &BufferSpec) {
        let _ = size;
        self._free(buffer, options);
    }

    /// Backend allocation. tinygrad `Allocator._alloc`.
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer>;

    /// Backend free. Default drops the `RawBuffer` (CPU/host memory frees via
    /// `Drop`); device backends override to release driver handles.
    /// tinygrad `Allocator._free`.
    fn _free(&self, _buffer: RawBuffer, _options: &BufferSpec) {}

    /// Copy host bytes into `dest[dest_off..dest_off+src.len()]`.
    /// tinygrad `Allocator._copyin`.
    fn _copyin(&self, _dest: &RawBuffer, _dest_off: usize, _src: &[u8]) -> Result<()> {
        UnsupportedSnafu { op: "copyin" }.fail()
    }

    /// Copy `src[src_off..src_off+dest.len()]` out into host bytes.
    /// tinygrad `Allocator._copyout`.
    fn _copyout(&self, _dest: &mut [u8], _src: &RawBuffer, _src_off: usize) -> Result<()> {
        UnsupportedSnafu { op: "copyout" }.fail()
    }

    /// Same-device copy of `sz` bytes. tinygrad `Allocator._transfer`.
    fn _transfer(
        &self,
        _dest: &RawBuffer,
        _dest_off: usize,
        _src: &RawBuffer,
        _src_off: usize,
        _sz: usize,
    ) -> Result<()> {
        UnsupportedSnafu { op: "transfer" }.fail()
    }

    /// Mint a sub-buffer view (for cross-device base views). tinygrad `Allocator._offset`.
    fn _offset(&self, _buf: &RawBuffer, _size: usize, _offset: usize) -> Result<RawBuffer> {
        UnsupportedSnafu { op: "offset" }.fail()
    }

    /// Map a foreign buffer into this device's address space. tinygrad `Allocator._map`.
    fn _map(&self, _buf: &RawBuffer) -> Result<RawBuffer> {
        UnsupportedSnafu { op: "map" }.fail()
    }

    /// Unmap a previously mapped buffer. tinygrad `Allocator._unmap`.
    fn _unmap(&self, _mb: &RawBuffer) {}

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
    fn _alloc(&self, size: usize, options: &BufferSpec, _zero: bool) -> Result<RawBuffer> {
        // `AlignedBuffer::new_zeroed` always zeroes, so `_zero` is implicitly
        // satisfied on CPU regardless of the flag.
        let data = AlignedBuffer::new_zeroed(size);
        Ok(RawBuffer::Cpu { data: UnsafeCell::new(data), cpu_accessible: options.cpu_access })
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        match dest {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: scheduler guarantees exclusive access during buffer ops.
                let buf = unsafe { &mut *data.get() };
                buf[dest_off..dest_off + src.len()].copy_from_slice(src);
                Ok(())
            }
            other => unreachable!("CpuAllocator::_copyin on non-CPU buffer: {other:?}"),
        }
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::Cpu { data, .. } => {
                // SAFETY: scheduler guarantees no concurrent writes during buffer ops.
                let buf = unsafe { &*data.get() };
                dest.copy_from_slice(&buf[src_off..src_off + dest.len()]);
                Ok(())
            }
            other => unreachable!("CpuAllocator::_copyout on non-CPU buffer: {other:?}"),
        }
    }

    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        match (dest, src) {
            (RawBuffer::Cpu { data: dst, .. }, RawBuffer::Cpu { data: src, .. }) => {
                // SAFETY: distinct allocations (no aliasing); scheduler exclusivity.
                let dst_buf = unsafe { &mut *dst.get() };
                let src_buf = unsafe { &*src.get() };
                dst_buf[dest_off..dest_off + sz].copy_from_slice(&src_buf[src_off..src_off + sz]);
                Ok(())
            }
            _ => UnsupportedSnafu { op: "transfer" }.fail(),
        }
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
    fn _alloc(&self, size: usize, _options: &BufferSpec, _zero: bool) -> Result<RawBuffer> {
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

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::Mmap { data, .. } => {
                dest.copy_from_slice(&data[src_off..src_off + dest.len()]);
                Ok(())
            }
            other => unreachable!("DiskAllocator::_copyout on non-Mmap buffer: {other:?}"),
        }
    }

    fn _copyin(&self, _dest: &RawBuffer, _dest_off: usize, _src: &[u8]) -> Result<()> {
        // DISK is read-only (ops_disk.py never writes through the mmap).
        Err(crate::Error::CopyFailed { reason: "DISK device is read-only: copyin not supported".into() })
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
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        if options.cpu_access {
            // Allocate unified memory (CPU-accessible)
            let mut data = unsafe { self.device.alloc_unified::<u8>(size, true) }.context(CudaSnafu)?;

            if zero {
                self.device.default_stream().memset_zeros(&mut data).context(CudaSnafu)?;
            }

            Ok(RawBuffer::CudaUnified { data: UnsafeCell::new(data), device: Arc::clone(&self.device) })
        } else {
            // Allocate device-only memory (faster GPU access)
            let stream = self.device.default_stream();
            let data = if zero { stream.alloc_zeros::<u8>(size) } else { unsafe { stream.alloc::<u8>(size) } }
                .context(CudaSnafu)?;

            Ok(RawBuffer::CudaDevice { data: UnsafeCell::new(data), device: Arc::clone(&self.device) })
        }
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        match dest {
            RawBuffer::CudaDevice { data, device } => {
                let cuda_data = unsafe { &mut *data.get() };
                let mut view = cuda_data.slice_mut(dest_off..dest_off + src.len());
                device.default_stream().memcpy_htod(src, &mut view).context(CudaSnafu)
            }
            RawBuffer::CudaUnified { data, .. } => {
                let unified_data = unsafe { &mut *data.get() };
                let slice = unified_data.as_mut_slice().context(CudaSnafu)?;
                slice[dest_off..dest_off + src.len()].copy_from_slice(src);
                Ok(())
            }
            other => unreachable!("CudaAllocator::_copyin on non-CUDA buffer: {other:?}"),
        }
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        match src {
            RawBuffer::CudaDevice { data, device } => {
                device.synchronize().context(CudaSnafu)?;
                let cuda_data = unsafe { &*data.get() };
                let view = cuda_data.slice(src_off..src_off + dest.len());
                device.default_stream().memcpy_dtoh(&view, dest).context(CudaSnafu)
            }
            RawBuffer::CudaUnified { data, .. } => {
                let unified_data = unsafe { &*data.get() };
                let slice = unified_data.as_slice().context(CudaSnafu)?;
                dest.copy_from_slice(&slice[src_off..src_off + dest.len()]);
                Ok(())
            }
            other => unreachable!("CudaAllocator::_copyout on non-CUDA buffer: {other:?}"),
        }
    }

    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        let stream = self.device.default_stream();
        match (dest, src) {
            (RawBuffer::CudaDevice { data: dst_data, .. }, RawBuffer::CudaDevice { data: src_data, .. }) => {
                let dst_cuda = unsafe { &mut *dst_data.get() };
                let src_cuda = unsafe { &*src_data.get() };
                let mut dst_view = dst_cuda.slice_mut(dest_off..dest_off + sz);
                let src_view = src_cuda.slice(src_off..src_off + sz);
                stream.memcpy_dtod(&src_view, &mut dst_view).context(CudaSnafu)
            }
            (RawBuffer::CudaUnified { data: dst_data, .. }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let dst_unified = unsafe { &mut *dst_data.get() };
                let src_unified = unsafe { &*src_data.get() };
                let dst_slice = dst_unified.as_mut_slice().context(CudaSnafu)?;
                let src_slice = src_unified.as_slice().context(CudaSnafu)?;
                dst_slice[dest_off..dest_off + sz].copy_from_slice(&src_slice[src_off..src_off + sz]);
                Ok(())
            }
            (RawBuffer::CudaUnified { data: dst_data, .. }, RawBuffer::CudaDevice { data: src_data, .. }) => {
                let src_cuda = unsafe { &*src_data.get() };
                let src_view = src_cuda.slice(src_off..src_off + sz);
                let dst_unified = unsafe { &mut *dst_data.get() };
                let mut dst_target = dst_unified.slice_mut(dest_off..dest_off + sz);
                stream.memcpy_dtod(&src_view, &mut dst_target).context(CudaSnafu)
            }
            (RawBuffer::CudaDevice { data: dst_data, .. }, RawBuffer::CudaUnified { data: src_data, .. }) => {
                let dst_cuda = unsafe { &mut *dst_data.get() };
                let mut dst_view = dst_cuda.slice_mut(dest_off..dest_off + sz);
                let src_unified = unsafe { &*src_data.get() };
                let src_source = src_unified.slice(src_off..src_off + sz);
                stream.memcpy_htod(&src_source, &mut dst_view).context(CudaSnafu)
            }
            _ => UnsupportedSnafu { op: "transfer" }.fail(),
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

/// LRU allocator that caches freed buffers for reuse. Mirrors tinygrad
/// `LRUAllocator` (device.py:250-270):
///
/// - the cache is keyed on the whole `(size, BufferSpec)` (device.py:259);
/// - `free` recycles into the pool *without synchronizing* — the
///   timeline-drain-before-teardown lives in the backend `_free` (e.g.
///   `AmdAllocator::_free`), reached only on real release (overflow, `nolru`,
///   `external_ptr`, or `free_cache`);
/// - on allocation failure `free_cache` releases every pooled buffer through
///   the backend `_free` and the alloc is retried (device.py:260-263).
///
/// The cache key uses the *requested* `size` for both `alloc` and `free` (the
/// `size` arg to `free`), so a backend that rounds up its actual allocation
/// (e.g. AMD page-rounding) still reuses buffers — unlike keying on the
/// buffer's rounded size, which would never match the request.
#[derive(Debug)]
pub(crate) struct LruAllocator {
    inner: Box<dyn Allocator>,
    cache: Mutex<HashMap<(usize, BufferSpec), Vec<RawBuffer>>>,
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

    /// Release every pooled buffer through the backend `_free` (device.py:264-267).
    /// Routing through `inner.free` is essential: `RawBuffer` has no `Drop`, so
    /// merely clearing the map would leak GPU mappings.
    fn free_cache(&self) {
        let drained: Vec<((usize, BufferSpec), Vec<RawBuffer>)> = {
            let mut cache = self.cache.lock().unwrap();
            cache.drain().collect()
        };
        for ((size, options), buffers) in drained {
            for buf in buffers {
                self.inner.free(buf, size, &options);
            }
        }
    }

    /// Get the number of cached buffers for a specific size and cpu_access flag.
    /// Only available in tests for cache introspection.
    #[cfg(test)]
    pub(crate) fn cache_count(&self, size: usize, cpu_access: bool) -> usize {
        let key = (size, BufferSpec { cpu_access, ..Default::default() });
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

    /// Re-zero a buffer popped from the cache. Returns `Err` to signal "drop
    /// this buffer and allocate fresh instead" (device-only AMD VRAM, where a
    /// host memset is impossible until SDMA lands).
    fn zero_cached(&self, buffer: &RawBuffer) -> Result<bool> {
        // SAFETY: buffer just popped from cache — no other references exist.
        match buffer {
            RawBuffer::Cpu { data, .. } => {
                unsafe { (*data.get()).fill(0) };
                Ok(true)
            }
            RawBuffer::Mmap { .. } => panic!("DISK device is read-only: cannot zero-init mmap buffer"),
            #[cfg(target_os = "linux")]
            RawBuffer::AmdDevice { host_ptr: Some(ptr), size, device, .. } => {
                // Drain first: this VA was just recycled from the pool and its
                // previous owner's async kernel may still be writing it. A host
                // memset isn't ordered on the GPU timeline, so synchronize.
                device.synchronize()?;
                unsafe { std::ptr::write_bytes(ptr.as_ptr(), 0, *size) };
                Ok(true)
            }
            #[cfg(target_os = "linux")]
            RawBuffer::AmdDevice { host_ptr: None, .. } => Ok(false),
            #[cfg(feature = "cuda")]
            RawBuffer::CudaDevice { data, device } => {
                let cuda_data = unsafe { &mut *data.get() };
                device.default_stream().memset_zeros(cuda_data).context(CudaSnafu)?;
                Ok(true)
            }
            #[cfg(feature = "cuda")]
            RawBuffer::CudaUnified { data, device } => {
                let unified_data = unsafe { &mut *data.get() };
                device.default_stream().memset_zeros(unified_data).context(CudaSnafu)?;
                Ok(true)
            }
        }
    }
}

impl Allocator for LruAllocator {
    fn alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        // nolru / external_ptr never pool: deterministic free (device.py:269).
        if options.nolru || options.external_ptr.is_some() {
            return self.inner.alloc(size, options, zero);
        }
        let key = (size, *options);

        // Pop from the per-key pool if present (device.py:259).
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
        }; // Drop lock before any (re)allocation.

        if let Some(buffer) = buffer {
            if zero && !self.zero_cached(&buffer)? {
                // Device-only buffer we can't memset on the host: drop it and
                // allocate fresh so we never hand back un-zeroed data.
                drop(buffer);
                return self.inner.alloc(size, options, zero);
            }
            return Ok(buffer);
        }

        // Cache miss → backend alloc; on failure drain the pool and retry once
        // (device.py:260-263).
        match self.inner.alloc(size, options, zero) {
            Ok(buffer) => Ok(buffer),
            Err(e) => {
                self.free_cache();
                self.inner.alloc(size, options, zero).map_err(|_| e)
            }
        }
    }

    fn free(&self, buffer: RawBuffer, size: usize, options: &BufferSpec) {
        // nolru / external_ptr bypass the pool — real free now (device.py:269-270).
        if options.nolru || options.external_ptr.is_some() {
            self.inner.free(buffer, size, options);
            return;
        }

        // Recycle into the pool. NOTE: no synchronize here — tinygrad's LRU
        // recycle is intentionally undrained (device.py:269); the timeline
        // drain happens in the backend `_free` on real teardown
        // (`AmdAllocator::_free` / hcq.py:566). On overflow route through
        // `inner.free` so the handle is actually released (RawBuffer has no Drop).
        let overflow = {
            let mut cache = self.cache.lock().unwrap();
            let buffers = cache.entry((size, *options)).or_default();
            if buffers.len() < self.max_buffers_per_size {
                buffers.push(buffer);
                None
            } else {
                Some(buffer)
            }
        };
        if let Some(buf) = overflow {
            self.inner.free(buf, size, options);
        }
    }

    // The decorator forwards the backend hooks to the wrapped allocator.
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> Result<RawBuffer> {
        self.inner._alloc(size, options, zero)
    }
    fn _free(&self, buffer: RawBuffer, options: &BufferSpec) {
        self.inner._free(buffer, options);
    }
    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> Result<()> {
        self.inner._copyin(dest, dest_off, src)
    }
    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> Result<()> {
        self.inner._copyout(dest, src, src_off)
    }
    fn _transfer(&self, dest: &RawBuffer, dest_off: usize, src: &RawBuffer, src_off: usize, sz: usize) -> Result<()> {
        self.inner._transfer(dest, dest_off, src, src_off, sz)
    }
    fn _offset(&self, buf: &RawBuffer, size: usize, offset: usize) -> Result<RawBuffer> {
        self.inner._offset(buf, size, offset)
    }
    fn _map(&self, buf: &RawBuffer) -> Result<RawBuffer> {
        self.inner._map(buf)
    }
    fn _unmap(&self, mb: &RawBuffer) {
        self.inner._unmap(mb);
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
