//! AMD KFD-direct command queues.
//!
//! - [`AmdComputeQueue`]: 16 MiB AQL ring, doorbell-driven kernel dispatch.
//! - [`AmdCopyQueue`]: SDMA queue for device↔device / device↔host copies.
//!
//! Both share the same KFD `AMDKFD_IOC_CREATE_QUEUE` mechanism but use
//! different `queue_type` codes. AQL packets are 64 bytes (`HsaKernelDispatchPacket`
//! + `HsaBarrierAndPacket`); SDMA submissions are raw dword sequences.
//!
//! Phase 5 scope: data structures, packet construction, and `submit()` writes
//! to ring + doorbell. The full `HardwareQueue` trait implementation
//! (`exec`/`copy`/`memory_barrier`) is wired in Phase 6 once `AmdProgram`
//! provides the kernel descriptor.

#![cfg(target_os = "linux")]

use std::cell::UnsafeCell;
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;
use tracing::debug;

use crate::allocator::{Allocator, BufferSpec};

use crate::amd::AmdAllocator;
use crate::amd::connector::AmdConnector;
use crate::amd::device::AmdDeviceCore;
use crate::amd::sys::hsa::{
    HSA_FENCE_SCOPE_SYSTEM, HSA_PACKET_HEADER_BARRIER, HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
    HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, HSA_PACKET_TYPE_VENDOR_SPECIFIC,
    HsaKernelDispatchPacket, HsaSignal, kernel_dispatch_header,
};
use crate::amd::sys::kfd;
use crate::amd::sys::pm4;
use crate::error::{Error, Result};

/// AQL packets are exactly 64 bytes.
pub const AQL_PACKET_BYTES: usize = 64;
/// 16 MiB ring — the compute-ring default size.
pub const COMPUTE_RING_BYTES: usize = 16 * 1024 * 1024;
/// SDMA ring is smaller; 1 MiB is plenty for short copy bursts.
pub const COPY_RING_BYTES: usize = 1024 * 1024;

/// Conservative upper bound on the dwords a single PM4 dispatch writes to the
/// ring (wait, HDP flush, acquire_mem, the SET_SH_REG stream, DISPATCH_DIRECT,
/// RELEASE_MEM — a typical dispatch is ~150). Bounds in-flight dispatches so
/// the host can never lap the ring.
const MAX_DISPATCH_DWORDS: usize = 1024;
/// Max un-retired dispatches allowed before back-pressure blocks the host.
/// Chosen so the combined ring footprint stays at half the ring even in the
/// worst case (`* MAX_DISPATCH_DWORDS`), leaving generous margin while still
/// letting the host run thousands of dispatches ahead of the GPU.
const RING_MAX_INFLIGHT: u64 = (COMPUTE_RING_BYTES / 4 / MAX_DISPATCH_DWORDS / 2) as u64;

/// AQL vendor-specific packet that wraps a PM4 indirect-buffer reference.
///
/// 16 dwords / 64 bytes:
/// ```text
/// dw0  = AQL_HDR | (VENDOR_SPECIFIC << TYPE) | (1 << 16)
/// dw1  = PACKET3(INDIRECT_BUFFER, count=2)
/// dw2  = pm4_addr lo
/// dw3  = pm4_addr hi
/// dw4  = pm4_count | INDIRECT_BUFFER_VALID
/// dw5  = 10                       (vendor magic)
/// dw6..15 = 10 reserved zero dwords
/// ```
pub fn build_aql_vendor_ib_packet(pm4_addr: u64, pm4_count: u32) -> [u32; 16] {
    let aql_hdr_low: u16 = (1 << HSA_PACKET_HEADER_BARRIER)
        | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE)
        | (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE)
        | HSA_PACKET_TYPE_VENDOR_SPECIFIC;
    let dw0: u32 = (aql_hdr_low as u32) | (1 << 16);
    [
        dw0,
        pm4::packet3(pm4::PACKET3_INDIRECT_BUFFER, 2),
        pm4_addr as u32,
        (pm4_addr >> 32) as u32,
        pm4_count | pm4::INDIRECT_BUFFER_VALID,
        10,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
}

/// Pack a kernel-dispatch packet describing a single launch.
///
/// `kernel_object` = GPU VA of the kernel descriptor (from the loaded code
/// object — Phase 6 will fill this in).
#[allow(clippy::too_many_arguments)]
pub fn build_dispatch_packet(
    workgroup_size: [u16; 3],
    grid_size: [u32; 3],
    private_segment_size: u32,
    group_segment_size: u32,
    kernel_object: u64,
    kernarg_address: u64,
    completion_signal: u64,
) -> HsaKernelDispatchPacket {
    let dims: u16 = if grid_size[2] > 1 {
        3
    } else if grid_size[1] > 1 {
        2
    } else {
        1
    };
    HsaKernelDispatchPacket {
        header: kernel_dispatch_header(),
        // bits 0-1 = dimensions
        setup: dims,
        workgroup_size_x: workgroup_size[0],
        workgroup_size_y: workgroup_size[1],
        workgroup_size_z: workgroup_size[2],
        reserved0: 0,
        grid_size_x: grid_size[0],
        grid_size_y: grid_size[1],
        grid_size_z: grid_size[2],
        private_segment_size,
        group_segment_size,
        kernel_object,
        kernarg_address,
        reserved2: 0,
        completion_signal: HsaSignal { handle: completion_signal },
    }
}

/// SDMA linear copy packet. All values are u32 dwords stored little-endian
/// in the ring; the GPU consumes them as a packed command stream.
pub fn build_sdma_linear_copy(src: u64, dst: u64, size: usize) -> [u32; 7] {
    // DW0: opcode 0x01 (copy) | sub-opcode 0x00 (linear). The sub-opcode
    // sits in bits 8..16, so the constant happens to be just 0x01 for the
    // linear-copy combo. The count bits go in DW1; with SDMA v5+
    // the limit per packet is 0x4000_0000 bytes (caller chunks if needed).
    let dw0: u32 = 0x01;
    let dw1: u32 = (size as u32) - 1;
    let dw2: u32 = 0;
    let dw3: u32 = (src & 0xFFFF_FFFF) as u32;
    let dw4: u32 = (src >> 32) as u32;
    let dw5: u32 = (dst & 0xFFFF_FFFF) as u32;
    let dw6: u32 = (dst >> 32) as u32;
    [dw0, dw1, dw2, dw3, dw4, dw5, dw6]
}

/// Compute queue. Wraps either a `KFD_IOC_QUEUE_TYPE_COMPUTE` (PM4) ring on
/// single-XCC GPUs (gfx11/12 default) or a `KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`
/// ring on multi-XCC CDNA. The two paths share the same KFD setup, doorbell
/// mapping, and submit primitive — the only differences are the packet
/// format we write into the ring and whether the GART contains an
/// `amd_queue_t` AQL descriptor.
/// # Safety — single-owner interior mutability
///
/// `inner` is mutated through `&self` without a lock. The owning
/// `ConnectorLease` guarantees exactly one thread issues sequential,
/// non-reentrant dispatch/submit calls against this queue for its lifetime:
/// the connector lives in exactly one owner slot (plan / graph / per-call
/// lease) or the idle pool (where nobody dispatches), and a single owner
/// dispatches its ops serially. The shared drainer (`synchronize_all`) reads
/// only the timeline atomics + signal slot (via `Timeline`), NEVER this cell.
/// Mirrors `RawBuffer`'s `UnsafeCell` + scheduler-exclusivity pattern
/// (allocator.rs). This is the lock-free dispatch path the per-owner model is
/// built for: distinct connectors' queues are interleaved by the GPU's MES,
/// not by a CPU lock.
pub struct AmdComputeQueue {
    inner: UnsafeCell<QueueInner>,
    /// Immutable device identity (kfd_fd, drm_fd, node, arch, poison latch).
    core: Arc<AmdDeviceCore>,
    /// `true` when this queue submits raw PM4 dwords directly; `false` when
    /// it submits AQL packets (with PM4 wrapped in AQL vendor IB packets).
    /// Decided at queue creation from `num_xcc`, fixed for the queue's lifetime.
    is_pm4: bool,
}

// SAFETY: `QueueInner` is `Send`; the single-owner invariant above means no
// two threads access `inner` concurrently, so it is sound to share `&self`
// across threads (e.g. a plan moved between dispatches). `UnsafeCell` makes
// the type `!Sync` by default, hence the manual impl.
unsafe impl Sync for AmdComputeQueue {}

/// Copy queue (SDMA).
pub struct AmdCopyQueue {
    inner: Mutex<QueueInner>,
    core: Arc<AmdDeviceCore>,
}

struct QueueInner {
    /// 16 MiB ring buffer; host-visible so we can write packets directly.
    ring_host: NonNull<u8>,
    ring_size: usize,
    /// Per-queue doorbell (`*mut u64` MMIO).
    doorbell: NonNull<u64>,
    /// mmap base of the doorbell page, kept so the queue can `munmap` it on
    /// teardown (each queue maps its own page).
    doorbell_base: NonNull<u8>,
    /// Host pointer to the GART-resident `write_dispatch_id` slot — KFD
    /// reads this in addition to the doorbell. It must be updated before
    /// every doorbell ring. Skipping it makes the GPU's
    /// command processor see the doorbell change but stall on a stale wptr.
    write_ptr_host: NonNull<u64>,
    /// 16 MiB host-visible uncached-coherent buffer for PM4 indirect
    /// buffers (AQL path only). The AQL vendor-specific packet
    /// references PM4 dwords stored in this buffer via PACKET3_INDIRECT_BUFFER.
    /// Bump-allocated; wraps on overflow.
    pm4_ibs_host: NonNull<u8>,
    pm4_ibs_gpu: u64,
    pm4_ibs_size: usize,
    pm4_ibs_cursor: usize,
    /// Index of the next packet (in AQL_PACKET_BYTES-sized slots). For SDMA
    /// queues this is the next byte offset; type checks ensure callers don't
    /// confuse them.
    write_idx: u64,
    /// Owned KFD queue id (held for the future destroy ioctl; reading it
    /// inside the queue isn't useful since the ioctl takes it directly).
    #[allow(dead_code)]
    queue_id: u32,
    /// Owned bookkeeping buffers we need to keep alive. The EOP and ctx-save
    /// buffers stay alive for the lifetime of the queue — KFD reads them
    /// asynchronously as part of the compute dispatch hardware state.
    _ring_buf: crate::allocator::RawBuffer,
    _gart_buf: crate::allocator::RawBuffer,
    _eop_buf: Option<crate::allocator::RawBuffer>,
    _ctx_buf: Option<crate::allocator::RawBuffer>,
    _pm4_ibs_buf: Option<crate::allocator::RawBuffer>,
}

// SAFETY: ring/doorbell access goes through Mutex; underlying buffers are
// allocator-owned and stable.
unsafe impl Send for QueueInner {}
unsafe impl Sync for QueueInner {}

impl Drop for QueueInner {
    /// Free the queue's KFD-allocated VRAM/GTT backings. `RawBuffer` itself
    /// has no `Drop` (the existing `AmdAllocator::_free` consumes RawBuffer
    /// by destructure), so a queue dropped directly — as happens for
    /// per-connector queues — would otherwise leak ~50 MiB of ring + GART +
    /// EOP + ctx-save + pm4_ibs every time. We call the in-place free path
    /// (`RawBuffer::free_amd_device_in_place`) for each. `AmdComputeQueue::
    /// Drop` has already invoked `kfd_destroy_queue` AND `AmdConnector::Drop`
    /// has synchronised the timeline, so the GPU is idle on these buffers.
    ///
    /// Skipped during panic unwind: `AmdConnector::Drop` and
    /// `AmdComputeQueue::Drop` both skip their synchronize/destroy on panic, so
    /// the GPU's CP may still be reading the ring/GART. Unmapping them here
    /// would fault the VM mid-unwind and could crash before the panic's
    /// diagnostics flush. Accept the buffer leak — the process is unwinding and
    /// the OS reclaims at exit.
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        self._ring_buf.free_amd_device_in_place();
        self._gart_buf.free_amd_device_in_place();
        if let Some(eop) = self._eop_buf.as_ref() {
            eop.free_amd_device_in_place();
        }
        if let Some(ctx) = self._ctx_buf.as_ref() {
            ctx.free_amd_device_in_place();
        }
        if let Some(pm4) = self._pm4_ibs_buf.as_ref() {
            pm4.free_amd_device_in_place();
        }
    }
}

impl QueueInner {
    /// Append raw PM4 dwords to the ring, wrapping at dword granularity.
    /// `write_idx` is counted in dwords for PM4 queues.
    /// Caller holds the queue lock — this is part of one atomic dispatch.
    ///
    /// Ring overflow is prevented up-stream by `wait_dispatch_headroom` (which
    /// bounds in-flight dispatches via the timeline signal), so a single push
    /// must never exceed the per-dispatch budget.
    fn push_pm4(&mut self, dwords: &[u32]) {
        let ring_dwords = self.ring_size / 4;
        debug_assert!(
            dwords.len() <= MAX_DISPATCH_DWORDS,
            "single dispatch ({} dwords) exceeds MAX_DISPATCH_DWORDS ({MAX_DISPATCH_DWORDS}); \
             raise the bound or lower RING_MAX_INFLIGHT",
            dwords.len(),
        );
        let mut idx = (self.write_idx as usize) % ring_dwords;
        for &dw in dwords {
            // SAFETY: ring_host points to ring_size bytes; idx < ring_dwords.
            unsafe { std::ptr::write_volatile((self.ring_host.as_ptr() as *mut u32).add(idx), dw) };
            idx = (idx + 1) % ring_dwords;
        }
        self.write_idx += dwords.len() as u64;
    }

    /// Write one 64-byte AQL packet at the current slot. `write_idx` counts
    /// 64-byte slots for AQL queues.
    fn push_aql(&mut self, bytes: &[u8]) {
        debug_assert_eq!(bytes.len(), AQL_PACKET_BYTES);
        let off = (self.write_idx as usize * AQL_PACKET_BYTES) % self.ring_size;
        // SAFETY: ring_host is mmapped + size-validated; off bounded by ring_size.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ring_host.as_ptr().add(off), AQL_PACKET_BYTES) };
        self.write_idx += 1;
    }

    /// Bump-allocate `dwords` into the pm4_ibs arena (AQL path only) and return
    /// the GPU VA of the copied region.
    fn pm4_ib(&mut self, dwords: &[u32]) -> u64 {
        let bytes = std::mem::size_of_val(dwords);
        let aligned = self.pm4_ibs_cursor.next_multiple_of(16);
        let start = if aligned + bytes > self.pm4_ibs_size { 0 } else { aligned };
        let gpu_addr = self.pm4_ibs_gpu + start as u64;
        // SAFETY: pm4_ibs_host is mmapped GTT; start + bytes ≤ size by construction.
        unsafe {
            std::ptr::copy_nonoverlapping(dwords.as_ptr() as *const u8, self.pm4_ibs_host.as_ptr().add(start), bytes)
        };
        self.pm4_ibs_cursor = start + bytes;
        gpu_addr
    }

    /// Publish the current `write_idx` to GART + ring the doorbell. AQL uses
    /// the **last completed** slot (`write_idx - 1`); PM4 uses the **next**
    /// dword (`write_idx`).
    fn ring_doorbell(&self, is_pm4: bool) {
        // GART wptr first: without it KFD sees the doorbell
        // change but reads a stale wptr.
        unsafe { std::ptr::write_volatile(self.write_ptr_host.as_ptr(), self.write_idx) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        let doorbell_value = if is_pm4 { self.write_idx } else { self.write_idx - 1 };
        // SAFETY: doorbell is mmapped MMIO; aligned 64-bit store.
        unsafe { std::ptr::write_volatile(self.doorbell.as_ptr(), doorbell_value) };
    }
}

impl AmdComputeQueue {
    /// Exclusive access to `inner` for the single owner. See the struct's
    /// safety doc — the `ConnectorLease` guarantees one sequential dispatcher.
    #[allow(clippy::mut_from_ref)]
    #[inline]
    unsafe fn inner_mut(&self) -> &mut QueueInner {
        // SAFETY: single-owner invariant; no concurrent accessor of `inner`.
        unsafe { &mut *self.inner.get() }
    }

    /// Create a compute queue. The queue kind is selected by `is_aql =
    /// xccs > 1`. Single-XCC GPUs (the gfx11/12 default) use the
    /// PM4 path (`KFD_IOC_QUEUE_TYPE_COMPUTE`), submitting raw PM4 dwords
    /// directly into the ring. Multi-XCC CDNA falls back to AQL, where each
    /// dispatch is a 64-byte AQL packet and PM4 helpers are wrapped via
    /// the vendor IB packet.
    /// Predict whether `create` would build a PM4 queue for this device,
    /// WITHOUT allocating anything. Used by `AmdGraph::capture` to skip the
    /// (multi-MiB) per-graph connector build on AQL hardware where the graph
    /// path is unsupported anyway. Same logic as `create`'s `is_pm4` decision.
    pub fn will_use_pm4(core: &AmdDeviceCore) -> bool {
        let force_aql = std::env::var("SVOD_AMD_AQL").ok().map(|s| s != "0").unwrap_or(false);
        !force_aql && core.node.num_xcc.max(1) == 1
    }

    pub fn create(allocator: &AmdAllocator) -> Result<Box<Self>> {
        let core = allocator.dev.core();
        // `SVOD_AMD_AQL=1` forces AQL even on single-XCC, useful for
        // bisecting PM4 vs AQL bring-up issues.
        let is_pm4 = Self::will_use_pm4(core);
        let queue_type = if is_pm4 { kfd::KFD_IOC_QUEUE_TYPE_COMPUTE } else { kfd::KFD_IOC_QUEUE_TYPE_COMPUTE_AQL };
        let inner = create_queue(allocator, queue_type, COMPUTE_RING_BYTES, !is_pm4)?;
        debug!(gpu_id = core.node.gpu_id, num_xcc = core.node.num_xcc, is_pm4 = is_pm4, "AmdComputeQueue created");
        Ok(Box::new(Self { inner: UnsafeCell::new(inner), core: Arc::clone(core), is_pm4 }))
    }

    /// `true` when this queue submits raw PM4 dwords (single-XCC); `false`
    /// for the AQL path. Read by callers in `program.rs` to pick the right
    /// dispatch builder.
    pub fn is_pm4(&self) -> bool {
        self.is_pm4
    }

    /// Block until at most `RING_MAX_INFLIGHT` dispatches are un-retired, so a
    /// host running `wait=false` faster than the GPU can't lap the ring and
    /// overwrite unconsumed packets. Bounds the combined ring footprint to
    /// `RING_MAX_INFLIGHT * MAX_DISPATCH_DWORDS` (half the ring).
    ///
    /// Gates on the connector's timeline SIGNAL — the proven completion
    /// primitive `synchronize` already uses — not the PM4 read pointer (whose
    /// COMPUTE-queue semantics are unreliable, which would deadlock a spin).
    /// The dispatches we wait on were submitted (doorbell rung) in prior calls,
    /// so the GPU will signal them; the wait always makes progress.
    fn wait_dispatch_headroom(&self, conn: &AmdConnector) -> Result<()> {
        let last_reserved = conn.timeline_value().saturating_sub(1);
        if last_reserved > RING_MAX_INFLIGHT {
            let target = last_reserved - RING_MAX_INFLIGHT;
            conn.timeline_signal()
                .wait_signal_value(target, 30_000)
                .inspect_err(|e| self.core.poison(&e.to_string()))?;
        }
        Ok(())
    }

    /// Atomically build + submit one PM4 (single-XCC) kernel dispatch.
    ///
    /// The queue's `inner` lock serializes packet assembly + ring blit +
    /// doorbell. With one queue per `AmdConnector` (single-owner per plan or
    /// graph) the lock is uncontended in practice, but it stays as a
    /// defensive primitive — any future async use within one connector
    /// (multi-threaded JIT replay, etc.) would still need it.
    ///
    /// Sequence:
    /// `wait(timeline, prev) → memory_barrier → exec → signal(timeline, next)`.
    /// Returns the timeline value this dispatch signals.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_pm4(
        &self,
        conn: &AmdConnector,
        rsrc1: u32,
        rsrc2: u32,
        rsrc3: u32,
        prog_addr: u64,
        enable_private_segment_sgpr: bool,
        user_data: &[u32],
        local: [u32; 3],
        grid: [u32; 3],
        wave32: bool,
        target_major: u32,
    ) -> Result<u64> {
        debug_assert!(self.is_pm4, "dispatch_pm4 called on AQL queue");
        debug_assert!(
            Arc::ptr_eq(&self.core, conn.core()),
            "dispatch_pm4: connector core ≠ queue core (queue gpu_id={}, conn gpu_id={}); \
             cross-device dispatch silently corrupts scratch/timeline VAs",
            self.core.node.gpu_id,
            conn.core().node.gpu_id,
        );
        // Single-queue mode: serialize the whole dispatch (headroom waits +
        // timeline reservation + ring write + doorbell) against other owners
        // sharing this connector. A no-op in multi-queue mode (exclusive
        // ownership → `exec_guard` returns `None`). Held for the method body so
        // the timeline state and ring stay consistent across the back-pressure
        // and wrap waits.
        let _dispatch_guard = self.core.exec_guard();
        // Keep the timeline < 2^32 (drain+reset at the watermark) before
        // reserving this dispatch's value.
        conn.ensure_timeline_headroom()?;
        // Ring back-pressure: block if too many dispatches are in flight, so an
        // async (`wait=false`) burst can't lap the ring. Outside the lock.
        self.wait_dispatch_headroom(conn)?;
        let timeline_addr = conn.timeline_signal().value_addr();
        // The connector is single-owner — its scratch and timeline are not
        // concurrently mutated, so we read them outside any lock.
        let scratch_addr = conn.scratch_gpu_va();
        let tmpring_size = conn.tmpring_size();
        // Assemble the full USER_DATA prefix here, under the lock, so the scratch
        // SGPR descriptor (words 0-3) is derived from the SAME `scratch_addr` as
        // the `COMPUTE_DISPATCH_SCRATCH_BASE` register below. Building it in
        // `AmdProgram::execute` (outside the lock) let a concurrent scratch
        // realloc slip in between the two reads, so the descriptor and the
        // register could point at different buffers. The scratch base address
        // is read exactly once and reused for the descriptor and the register.
        let mut full_user_data: Vec<u32> = Vec::with_capacity(user_data.len() + 4);
        if enable_private_segment_sgpr {
            full_user_data.push(scratch_addr as u32);
            full_user_data.push((scratch_addr >> 32) as u32 | (1u32 << 31));
            full_user_data.push(0xFFFF_FFFF);
            full_user_data.push(0x20c1_4000);
        }
        full_user_data.extend_from_slice(user_data);
        // SAFETY: single-owner invariant (see struct doc) — exclusive, no lock.
        let g = unsafe { self.inner_mut() };
        let prev = conn.timeline_value().saturating_sub(1);
        let next = conn.next_timeline();

        let mut q: Vec<u32> = Vec::with_capacity(96);
        // wait(timeline, prev): no-op on the first dispatch (prev == 0).
        q.extend_from_slice(&pm4::wait_reg_mem(timeline_addr, prev as u32, 0xFFFF_FFFF));
        // memory_barrier: HDP flush handshake + ACQUIRE_MEM cache invalidate.
        q.extend_from_slice(&pm4::hdp_flush());
        q.extend_from_slice(&pm4::acquire_mem());
        // exec: SET_SH_REG stream + DISPATCH_DIRECT.
        build_exec_pm4(
            &mut q,
            rsrc1,
            rsrc2,
            rsrc3,
            prog_addr,
            &full_user_data,
            scratch_addr,
            tmpring_size,
            local,
            grid,
            wave32,
            target_major,
        );
        // signal(timeline, next): RELEASE_MEM after a system-scope cache flush.
        q.extend_from_slice(&pm4::release_mem(timeline_addr, next as u32, /*cache_flush=*/ true));

        g.push_pm4(&q);
        g.ring_doorbell(/*is_pm4=*/ true);
        Ok(next)
    }

    /// Push a pre-built PM4 dword stream into the ring with ONE doorbell — the
    /// primitive behind `AmdHwQueue::submit` (the graph's atomic submit).
    /// Blits `cmds` into the ring, advances the write index, rings the doorbell.
    ///
    /// `dwords` is normally the 4-dword `PACKET3_INDIRECT_BUFFER` reference to
    /// the graph's bound `hw_page`; the CP then runs the whole captured chain
    /// inline.
    ///
    /// With per-connector queues, each graph owns its connector and submits
    /// through ITS OWN ring. The graph's own `comp_queue` `Mutex<AmdHwQueue>`
    /// serialises capture vs replay within one graph; this primitive only
    /// takes the queue's inner lock.
    pub fn submit_dwords(&self, dwords: &[u32]) -> Result<()> {
        debug_assert!(self.is_pm4, "submit_dwords on AQL queue");
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Single-queue serialization (cf. dispatch_pm4); no-op in multi-queue.
        // The graph factory falls back to per-call dispatch in single-queue
        // mode, so this guard is defensive — it keeps `submit_dwords` correct
        // if a shared connector ever drives a captured chain.
        let _dispatch_guard = self.core.exec_guard();
        // No `Release` fence here — `ring_doorbell` already issues its own
        // publication barrier.
        // SAFETY: exclusive access — single-owner in multi-queue mode, or held
        // under `exec_guard` in single-queue mode (see struct doc).
        let g = unsafe { self.inner_mut() };
        g.push_pm4(dwords);
        g.ring_doorbell(/*is_pm4=*/ true);
        Ok(())
    }

    /// Atomically build + submit one AQL (multi-XCC CDNA) kernel dispatch. Same
    /// ordering guarantee as [`dispatch_pm4`]; PM4 helpers are wrapped in AQL
    /// vendor-IB packets and the kernel launch is a real
    /// `HsaKernelDispatchPacket`.
    pub fn dispatch_aql(&self, conn: &AmdConnector, packet: &HsaKernelDispatchPacket) -> Result<u64> {
        debug_assert!(!self.is_pm4, "dispatch_aql called on PM4 queue");
        debug_assert!(
            Arc::ptr_eq(&self.core, conn.core()),
            "dispatch_aql: connector core ≠ queue core (queue gpu_id={}, conn gpu_id={})",
            self.core.node.gpu_id,
            conn.core().node.gpu_id,
        );
        debug_assert_eq!(size_of::<HsaKernelDispatchPacket>(), AQL_PACKET_BYTES);
        // Single-queue serialization (cf. dispatch_pm4); no-op in multi-queue.
        let _dispatch_guard = self.core.exec_guard();
        // Keep the timeline < 2^32 (cf. dispatch_pm4) before reserving.
        conn.ensure_timeline_headroom()?;
        // Ring back-pressure (cf. dispatch_pm4).
        self.wait_dispatch_headroom(conn)?;
        let timeline_addr = conn.timeline_signal().value_addr();
        // SAFETY: single-owner invariant (see struct doc) — exclusive, no lock.
        let g = unsafe { self.inner_mut() };
        let prev = conn.timeline_value().saturating_sub(1);
        let next = conn.next_timeline();

        // wait → barrier(hdp, acquire) → exec → signal, each PM4 op wrapped in a
        // vendor-IB AQL packet (exec is a native dispatch packet).
        let wait = pm4::wait_reg_mem(timeline_addr, prev as u32, 0xFFFF_FFFF);
        let ib = g.pm4_ib(&wait);
        let p = build_aql_vendor_ib_packet(ib, wait.len() as u32);
        g.push_aql(dwords_as_bytes(&p));

        let hdp = pm4::hdp_flush();
        let ib = g.pm4_ib(&hdp);
        let p = build_aql_vendor_ib_packet(ib, hdp.len() as u32);
        g.push_aql(dwords_as_bytes(&p));

        let acq = pm4::acquire_mem();
        let ib = g.pm4_ib(&acq);
        let p = build_aql_vendor_ib_packet(ib, acq.len() as u32);
        g.push_aql(dwords_as_bytes(&p));

        // exec: native 64-byte kernel-dispatch packet.
        let packet_bytes = unsafe { std::slice::from_raw_parts(packet as *const _ as *const u8, AQL_PACKET_BYTES) };
        g.push_aql(packet_bytes);

        let sig = pm4::release_mem(timeline_addr, next as u32, /*cache_flush=*/ true);
        let ib = g.pm4_ib(&sig);
        let p = build_aql_vendor_ib_packet(ib, sig.len() as u32);
        g.push_aql(dwords_as_bytes(&p));

        g.ring_doorbell(/*is_pm4=*/ false);
        Ok(next)
    }
}

impl Drop for AmdComputeQueue {
    /// Destroy the in-kernel KFD compute queue object. Without this, every
    /// `kfd_create_queue` ioctl leaves a queue id permanently registered with
    /// the kernel until process exit — and the per-process compute-queue
    /// limit (typically 32) is over LIFETIME creations, not concurrent ones,
    /// so a long-running process that creates+drops plans (BEAM-style) would
    /// eventually hit the cap with zero live connectors. The userspace ring
    /// / GART / EOP / ctx-save / pm4_ibs buffers free via the underlying
    /// `RawBuffer::Drop` chain when `self.inner` drops next.
    ///
    /// `AmdConnector::Drop` has already synchronised the timeline before
    /// reaching this point on the happy path. During panic unwind the
    /// connector skips `synchronize` to keep teardown bounded — destroying
    /// the KFD queue with in-flight CP work risks a kernel-side fault that
    /// crashes the process before useful diagnostics flush, so we also
    /// skip and accept the queue-id leak (process exit reclaims it).
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        // `&mut self` → exclusive; `get_mut` needs no unsafe.
        let inner = self.inner.get_mut();
        let (queue_id, doorbell_base) = (inner.queue_id, inner.doorbell_base);
        self.core.iface().teardown_ring(queue_id, doorbell_base);
    }
}

impl Drop for AmdCopyQueue {
    fn drop(&mut self) {
        let (queue_id, doorbell_base) = {
            let g = self.inner.lock();
            (g.queue_id, g.doorbell_base)
        };
        self.core.iface().teardown_ring(queue_id, doorbell_base);
    }
}

/// View a `[u32; 16]` AQL packet as its 64 little-endian bytes.
fn dwords_as_bytes(p: &[u32; 16]) -> &[u8] {
    // SAFETY: 16 u32 == 64 bytes, contiguous, any bit pattern valid.
    unsafe { std::slice::from_raw_parts(p.as_ptr() as *const u8, AQL_PACKET_BYTES) }
}

/// Build the PM4 SET_SH_REG + DISPATCH_DIRECT stream for a single-XCC dispatch,
/// appending into `q` (minus SQTT/PMC/dispatch_ptr).
/// The shader entry point is pre-shifted right by 8 (COMPUTE_PGM_LO/HI hold the
/// upper bits of a 256-byte-aligned address). `wave32` comes from
/// `kd.kernel_code_properties & 0x400`; `cs_w32_en` is gfx11/12-only.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_exec_pm4(
    q: &mut Vec<u32>,
    rsrc1: u32,
    rsrc2: u32,
    rsrc3: u32,
    prog_addr: u64,
    user_data: &[u32],
    scratch_addr: u64,
    tmpring_size: u32,
    local: [u32; 3],
    grid: [u32; 3],
    wave32: bool,
    target_major: u32,
) {
    // 1. Pre-dispatch cache-invalidate, skipping GLI/GL2 (already flushed by the
    //    preceding memory_barrier).
    q.extend_from_slice(&pm4::acquire_mem_with(pm4::GCR_FLAGS_NO_GLI_GL2));

    // 2. Shader address: COMPUTE_PGM_LO/HI hold (prog_addr >> 8).
    let prog_shr = prog_addr >> 8;
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_PGM_LO, &[prog_shr as u32, (prog_shr >> 32) as u32]));

    // 3. RSRC1/2 together; RSRC3 separately (gfx9 uses a different SH offset).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_PGM_RSRC1, &[rsrc1, rsrc2]));
    let rsrc3_reg = if target_major == 9 { pm4::COMPUTE_PGM_RSRC3_GFX9 } else { pm4::COMPUTE_PGM_RSRC3 };
    q.extend(pm4::set_sh_reg(rsrc3_reg, &[rsrc3]));

    // 4. Scratch / tmpring (valid base required for wave init on RDNA3+ even
    //    when SCRATCH_EN=0).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_TMPRING_SIZE, &[tmpring_size]));
    let scratch_shr = scratch_addr >> 8;
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_DISPATCH_SCRATCH_BASE_LO, &[scratch_shr as u32, (scratch_shr >> 32) as u32]));

    // 5. Restart points always zero (no preempt-resume).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_RESTART_X, &[0, 0, 0]));

    // 6. COMPUTE_USER_DATA_0..N — user SGPR pre-load (scratch desc + kernarg ptr),
    //    assembled by the caller.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_USER_DATA_0, user_data));

    // 7. RESOURCE_LIMITS: 0 = no per-SH wave caps.
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_RESOURCE_LIMITS, &[0]));

    // 8. START_X..NUM_THREAD_Z + 2 reserved (local size in NUM_THREAD_*).
    q.extend(pm4::set_sh_reg(pm4::COMPUTE_START_X, &[0, 0, 0, local[0], local[1], local[2], 0, 0]));

    // 9. Launch.
    let mut di = pm4::DISPATCH_INITIATOR_FORCE_START_AT_000 | pm4::DISPATCH_INITIATOR_COMPUTE_SHADER_EN;
    if target_major != 9 && wave32 {
        di |= pm4::DISPATCH_INITIATOR_CS_W32_EN;
    }
    q.extend_from_slice(&pm4::dispatch_direct(grid, di));

    // 10. CS_PARTIAL_FLUSH so the next dispatch sees clean state.
    q.extend_from_slice(&pm4::event_write(pm4::CS_PARTIAL_FLUSH, pm4::EVENT_INDEX_PARTIAL_FLUSH));
}

impl AmdCopyQueue {
    pub fn create(allocator: &AmdAllocator) -> Result<Arc<Self>> {
        let inner = create_queue(allocator, kfd::KFD_IOC_QUEUE_TYPE_SDMA, COPY_RING_BYTES, /*aql=*/ false)?;
        Ok(Arc::new(Self { inner: Mutex::new(inner), core: Arc::clone(allocator.dev.core()) }))
    }

    /// Submit a linear copy command (caller chunks for >`MAX_COPY_BYTES`).
    pub fn enqueue_linear_copy(&self, src: u64, dst: u64, size: usize) -> Result<()> {
        let dwords = build_sdma_linear_copy(src, dst, size);
        let mut g = self.inner.lock();
        let byte_off = (g.write_idx as usize) % g.ring_size;
        // SAFETY: ring is host-visible, byte_off bounded by ring_size.
        unsafe {
            std::ptr::copy_nonoverlapping(
                dwords.as_ptr() as *const u8,
                g.ring_host.as_ptr().add(byte_off),
                std::mem::size_of_val(&dwords),
            );
        }
        g.write_idx += std::mem::size_of_val(&dwords) as u64;
        Ok(())
    }

    pub fn submit(&self) -> Result<()> {
        let g = self.inner.lock();
        // GART wptr first, then the doorbell — same ordering as the compute
        // queue's `ring_doorbell`. Skipping the wptr makes the SDMA engine see
        // the doorbell change but stall on a stale write pointer.
        unsafe { std::ptr::write_volatile(g.write_ptr_host.as_ptr(), g.write_idx) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        unsafe { std::ptr::write_volatile(g.doorbell.as_ptr(), g.write_idx) };
        Ok(())
    }
}

impl std::fmt::Debug for AmdComputeQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdComputeQueue").field("gpu_id", &self.core.node.gpu_id).finish_non_exhaustive()
    }
}

impl std::fmt::Debug for AmdCopyQueue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdCopyQueue").field("gpu_id", &self.core.node.gpu_id).finish_non_exhaustive()
    }
}

fn create_queue(allocator: &AmdAllocator, queue_type: u32, ring_size: usize, aql: bool) -> Result<QueueInner> {
    let dev = allocator.dev.core();
    // Ring + GART are both VRAM with COHERENT | UNCACHED | PUBLIC flags
    // (uncached + cpu-accessible). Using plain VRAM (no UNCACHED) makes
    // KFD reject the create_queue ioctl with EINVAL.
    let ring_buf = allocator.alloc_uncached(ring_size)?;
    let (ring_gpu, ring_host) = match &ring_buf {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => return Err(Error::AmdAllocFailed { reason: "queue ring requires host-visible buffer".into() }),
    };
    // GART page holds the AQL queue descriptor (`amd_queue_t`, 256 bytes).
    // rptr/wptr live at fixed offsets inside it; KFD reads the descriptor
    // when wiring up the queue. The GART page is a 0x100-byte uncached,
    // cpu-accessible allocation.
    let gart_buf = allocator.alloc_uncached(0x100)?;
    let (gart_gpu, gart_host) = match &gart_buf {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => return Err(Error::AmdAllocFailed { reason: "GART page requires host-visible buffer".into() }),
    };

    if aql {
        // Initialize the GART descriptor.
        // max_cu_id is total CUs across all XCCs - 1 (cu_cnt*xccs-1).
        let cu_cnt = dev.node.simd_count.max(1) / dev.node.simd_per_cu.max(1);
        let waves_per_cu = dev.node.max_waves_per_simd * dev.node.simd_per_cu;
        let desc = crate::amd::sys::hsa::AmdQueueT {
            queue_properties: crate::amd::sys::hsa::AMD_QUEUE_PROPERTIES_IS_PTR64
                | crate::amd::sys::hsa::AMD_QUEUE_PROPERTIES_ENABLE_PROFILING,
            read_dispatch_id_field_base_byte_offset: crate::amd::sys::hsa::OFFSET_READ_DISPATCH_ID as u32,
            max_cu_id: cu_cnt.saturating_sub(1),
            max_wave_id: waves_per_cu.saturating_sub(1),
            ..Default::default()
        };
        // SAFETY: gart_host points to a 4 KiB region we just allocated.
        unsafe {
            std::ptr::copy_nonoverlapping(
                &desc as *const _ as *const u8,
                gart_host.as_ptr(),
                std::mem::size_of::<crate::amd::sys::hsa::AmdQueueT>(),
            );
        }
    }

    // Both AQL and plain COMPUTE queues use the same rptr/wptr offsets — the
    // `amd_queue_t::{read,write}_dispatch_id` byte offsets are passed
    // unconditionally. KFD validates these
    // against the queue type's expected layout; using (0, 8) for plain
    // COMPUTE produces EINVAL.
    let wptr_offset: u64 = crate::amd::sys::hsa::OFFSET_WRITE_DISPATCH_ID as u64;
    let rptr_offset: u64 = crate::amd::sys::hsa::OFFSET_READ_DISPATCH_ID as u64;
    // Compute queues need EOP + ctx-save buffers. Sizing:
    //   ctx_save_restore_size (ioctl arg) = wg_data_size + ctl_stack_size
    //   cwsr_buffer_size (alloc size)     = round_up((ctx_save_restore_size
    //                                          + debug_memory_size) * xccs,
    //                                          PAGESIZE)
    // The buffer is larger than what we
    // tell KFD by `debug_memory_size * xccs` — that overflow region is where
    // KFD writes debug-trap state. Undersizing causes corruption when CWSR
    // fires; oversizing is harmless.
    //
    // EOP and ctx-save are *plain VRAM* (no PUBLIC/COHERENT/UNCACHED flags):
    // they're written by the GPU during preemption and never read from the
    // CPU, so the default allocation flags suffice.
    let (wg_data_size, ctl_stack_size, debug_memory_size) = compute_ctx_sizes(dev);
    let xccs = dev.node.num_xcc.max(1) as usize;
    let ctx_save_restore_size = wg_data_size + ctl_stack_size;
    let cwsr_buffer_size = (ctx_save_restore_size + debug_memory_size) * xccs;
    let cwsr_buffer_size = cwsr_buffer_size.next_multiple_of(0x1000);
    let plain = BufferSpec { cpu_access: false, nolru: true, ..Default::default() };
    let eop_buf = allocator.alloc(0x1000, &plain, /*zero=*/ false)?;
    let ctx_buf = allocator.alloc(cwsr_buffer_size, &plain, /*zero=*/ false)?;
    let eop_gpu = match &eop_buf {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, .. } => *gpu_addr,
        _ => 0,
    };
    let ctx_gpu = match &ctx_buf {
        crate::allocator::RawBuffer::AmdDevice { gpu_addr, .. } => *gpu_addr,
        _ => 0,
    };
    let (eop_buf, ctx_buf) = (Some(eop_buf), Some(ctx_buf));
    let _ = aql; // queue_type already encodes AQL vs plain COMPUTE

    // CREATE_QUEUE + doorbell mmap through the backend seam. The ring/GART/EOP/
    // ctx buffers above are allocated by us (above the seam); the iface only
    // activates the HQD (register the queue + map its doorbell).
    let desc = crate::amd::iface::RingDesc {
        ring_gpu,
        gart_gpu,
        wptr_offset,
        rptr_offset,
        eop_gpu,
        eop_size: 0x1000,
        ctx_gpu,
        ctx_save_restore_size: ctx_save_restore_size as u32,
        ctl_stack_size: ctl_stack_size as u32,
        ring_size,
        gpu_id: dev.node.gpu_id,
        queue_type,
    };
    let qh = dev.iface().setup_ring(&desc)?;
    let queue_id = qh.queue_id;
    let doorbell = qh.doorbell;
    let doorbell_base = qh.doorbell_base;

    // SAFETY: gart_host points to the GART page we just mmapped; the
    // write/read_dispatch_id fields live at fixed offsets inside the
    // AmdQueueT descriptor we wrote into the page.
    let write_ptr_host = unsafe { NonNull::new_unchecked(gart_host.as_ptr().add(wptr_offset as usize) as *mut u64) };

    // PM4 indirect-buffer arena. AQL compute queues need it (PM4 helpers
    // get wrapped in vendor-IB packets); PM4 single-XCC compute queues and SDMA
    // queues write straight into their ring with no IB indirection. So the
    // arena is allocated only on the AQL path.
    const PM4_IBS_BYTES: usize = 16 * 1024 * 1024;
    let pm4_needed = aql && queue_type == kfd::KFD_IOC_QUEUE_TYPE_COMPUTE_AQL;
    let (pm4_ibs_host, pm4_ibs_gpu, pm4_ibs_size, pm4_ibs_buf) = if pm4_needed {
        let buf = allocator.alloc_uncached(PM4_IBS_BYTES)?;
        let (gpu, host) = match &buf {
            crate::allocator::RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => return Err(Error::AmdAllocFailed { reason: "pm4_ibs requires host-visible buffer".into() }),
        };
        (host, gpu, PM4_IBS_BYTES, Some(buf))
    } else {
        // Use the ring_host as a dummy non-null pointer; size 0 prevents use.
        (ring_host, 0, 0, None)
    };

    Ok(QueueInner {
        ring_host,
        ring_size,
        doorbell,
        doorbell_base,
        write_ptr_host,
        pm4_ibs_host,
        pm4_ibs_gpu,
        pm4_ibs_size,
        pm4_ibs_cursor: 0,
        write_idx: 0,
        queue_id,
        _ring_buf: ring_buf,
        _gart_buf: gart_buf,
        _eop_buf: eop_buf,
        _ctx_buf: ctx_buf,
        _pm4_ibs_buf: pm4_ibs_buf,
    })
}

/// Compute (wg_data_size, ctl_stack_size, debug_memory_size) for the ctx-save /
/// restore region.
fn compute_ctx_sizes(dev: &AmdDeviceCore) -> (usize, usize, usize) {
    const PAGE: usize = 0x1000;
    let sgrp_per_cu: usize = 0x4000;
    let hwreg_per_cu: usize = 0x1000;
    let is_cdna4 = dev.arch == svod_dtype::AmdArch::Gfx950;
    let lds_per_cu: usize = if is_cdna4 { (dev.node.lds_size_in_kb as usize) << 10 } else { 0x10000 };

    // VGPR-per-CU branches on a small whitelist of gfx-target
    // tuples: CDNA (gfx9.x) uses 0x80000, the listed
    // RDNA3/RDNA4 tuples use 0x60000, Gfx1102 alone uses 0x40000.
    let vgpr_per_cu: usize = match dev.arch {
        svod_dtype::AmdArch::Gfx942 | svod_dtype::AmdArch::Gfx950 => 0x80000,
        svod_dtype::AmdArch::Gfx1100
        | svod_dtype::AmdArch::Gfx1101
        | svod_dtype::AmdArch::Gfx1151
        | svod_dtype::AmdArch::Gfx1200
        | svod_dtype::AmdArch::Gfx1201 => 0x60000,
        svod_dtype::AmdArch::Gfx1102 => 0x40000,
    };

    let xccs = dev.node.num_xcc.max(1) as usize;
    let cu_cnt = ((dev.node.simd_count.max(1) / dev.node.simd_per_cu.max(1)) as usize / xccs).max(1);
    let waves_per_cu = (dev.node.max_waves_per_simd as usize) * (dev.node.simd_per_cu as usize);
    let wave_cnt = if dev.arch.is_cdna() {
        // gfx9 caps waves at min(cu_cnt*40, se_cnt*xccs*512); we don't have a
        // sysfs se_cnt yet so we use the conservative cu_cnt*40.
        (cu_cnt * 40).min(cu_cnt * waves_per_cu)
    } else {
        cu_cnt * waves_per_cu
    };

    let wg_data_size = (vgpr_per_cu + sgrp_per_cu + lds_per_cu + hwreg_per_cu) * cu_cnt;
    let wg_data_size = wg_data_size.next_multiple_of(PAGE);

    let waves_factor = if dev.arch.is_cdna() { 8 } else { 12 };
    let ctl_stack_size = (waves_factor * wave_cnt + 8 + 40).next_multiple_of(PAGE);
    // `debug_memory_size = round_up(wave_cnt * 32, 64)`.
    let debug_memory_size = (wave_cnt * 32).next_multiple_of(64);

    (wg_data_size, ctl_stack_size, debug_memory_size)
}

#[cfg(test)]
#[path = "../test/unit/amd/queue.rs"]
mod tests;
