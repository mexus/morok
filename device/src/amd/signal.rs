//! `AmdSignal`: GTT-coherent timeline counter polled by the CPU.
//!
//! A `SignalPool` carves a single host-visible GTT page into 64-byte slots;
//! each [`AmdSignal`] owns a slot and exposes the `value_addr` GPU virtual
//! address so kernels / AQL barrier packets can write completion values.
//! CPU polling reads the same memory through the slot's `host_ptr`.

#![cfg(unix)]

use std::any::Any;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::allocator::{AmdBufferGuard, RawBuffer};
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDeviceCore;
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

/// Spin / yield budget before escalating to a KFD WAIT_EVENTS sleep: cheap
/// polling for short waits, kernel blocking for long ones so a stalled wait
/// doesn't pin a CPU.
#[cfg(not(test))]
const WAIT_EVENTS_ESCALATE_MS: u64 = 200;
#[cfg(test)]
const WAIT_EVENTS_ESCALATE_MS: u64 = 1;

/// 64-byte slot laid out as an `amd_signal_t` (kind@0, value@8). The AQL packet
/// processor reads a dispatch packet's `completion_signal.handle` as the struct
/// base and atomically decrements the `value` field at offset 8; PM4 RELEASE_MEM
/// and SDMA fence writes target that same `value`. So every signal — compute
/// completion or copy timeline — shares this one layout.
const SLOT_BYTES: usize = 64;
/// 4 KiB page / 64 B per slot = 64 signals per page — the page-alignment quantum
/// the pool rounds its slot count up to.
const SLOTS_PER_PAGE: usize = 64;
/// Byte offset of the `value` counter inside an `amd_signal_t` slot.
const SIGNAL_VALUE_OFFSET: usize = 8;
/// Byte offsets of dispatch timestamps (`amd_signal_t.start_ts` / `.end_ts`).
/// AMD HCQ submission finalizers write these fields with explicit PM4 timestamp
/// commands on both PM4 and AQL queues.
const SIGNAL_START_TS_OFFSET: usize = 32;
const SIGNAL_END_TS_OFFSET: usize = 40;
/// The GPU clock counter feeding the timestamps ticks at the architected
/// 100 MHz on AMD GPUs (10 ns per tick — tinygrad's `timestamp_divider=100`).
const NS_PER_TICK: u64 = 10;

/// A pool-allocated AMD signal.
///
/// The atomic counter the GPU writes to lives at `value_addr` (GPU VA) and
/// is also reachable via `host_ptr` for CPU polling. `Drop` returns the
/// slot to its pool. The pool keeps the underlying VRAM allocation alive.
pub struct AmdSignal {
    slot: u32,
    /// `amd_signal_t` struct base (GPU VA), used to derive timestamp fields.
    base_gpu: u64,
    /// GPU VA of the `value` counter (`base_gpu + SIGNAL_VALUE_OFFSET`) — what
    /// PM4/SDMA packets write and what the host polls.
    value_addr: u64,
    host_ptr: NonNull<AtomicU64>,
    pool: Weak<SignalPool>,
    /// Owning device core — used to escalate long waits to
    /// `AMDKFD_IOC_WAIT_EVENTS` on the device's `queue_event`. `Weak` so
    /// signals don't extend device lifetime.
    device: Weak<AmdDeviceCore>,
}

// SAFETY: AtomicU64 covers all reads/writes; the host pointer comes from a
// shared mmap and is stable for the pool's lifetime.
unsafe impl Send for AmdSignal {}
unsafe impl Sync for AmdSignal {}

impl AmdSignal {
    /// GPU VA of the `value` counter — what PM4/SDMA wait/signal packets write
    /// and the host polls (`amd_signal_t.value`, at +8 from the struct base).
    pub fn value_addr(&self) -> u64 {
        self.value_addr
    }

    /// GPU VA of the dispatch `start_ts` field (`base_gpu + 32`). On the
    /// single-XCC PM4 path the CP does not auto-stamp dispatches (the AQL path's
    /// `ENABLE_PROFILING` does), so a profiling dispatch targets this address
    /// with a `release_mem_timestamp` GPU-clock probe before the kernel launches.
    #[inline]
    pub fn start_ts_addr(&self) -> u64 {
        self.base_gpu + SIGNAL_START_TS_OFFSET as u64
    }

    /// GPU VA of the dispatch `end_ts` field (`base_gpu + 40`). See
    /// [`start_ts_addr`](Self::start_ts_addr).
    #[inline]
    pub fn end_ts_addr(&self) -> u64 {
        self.base_gpu + SIGNAL_END_TS_OFFSET as u64
    }

    /// Slot index inside the pool. Useful for debugging.
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// Current value (host read of the coherent slot).
    #[inline]
    fn load(&self) -> u64 {
        // SAFETY: NonNull valid for the pool's lifetime; AtomicU64 is race-free.
        unsafe { self.host_ptr.as_ref().load(Ordering::Acquire) }
    }

    /// Reset a slot before assigning it to a timeline or timestamp probe.
    /// Monotonic PM4/SDMA timelines start at zero and receive literal stores;
    /// stale profiling stamps are always cleared on reuse.
    #[inline]
    pub(crate) fn reset(&self, value: u64) {
        // SAFETY: the full 64-byte slot is mapped; ts fields at +32/+40.
        unsafe {
            let base = (self.host_ptr.as_ptr() as *mut u8).sub(SIGNAL_VALUE_OFFSET);
            std::ptr::write_volatile(base.add(SIGNAL_START_TS_OFFSET) as *mut u64, 0);
            std::ptr::write_volatile(base.add(SIGNAL_END_TS_OFFSET) as *mut u64, 0);
            self.host_ptr.as_ref().store(value, Ordering::Release);
        }
    }

    /// Tiered busy-wait until `ready(value)` holds, or `timeout_ms` of *no
    /// progress* elapses, or KFD reports a GPU fault. Shared by the
    /// monotonic timeline waits.
    ///
    /// Early-exit on fault is load-bearing for BEAM search: a bad kernel config
    /// may fault the GPU, and paying the full timeout per rejected candidate is
    /// unaffordable.
    fn poll_until(&self, ready: impl Fn(u64) -> bool, timeout_ms: u64, what: &'static str) -> Result<()> {
        let mut start = std::time::Instant::now();
        let mut prev = u64::MAX;
        loop {
            if let Some(error) = self.device.upgrade().and_then(|device| device.poison_error()) {
                return Err(error);
            }
            let v = self.load();
            if ready(v) {
                return Ok(());
            }
            if v != prev {
                prev = v; // progress: timeout is for *no* progress, so reset.
                start = std::time::Instant::now();
            }
            if timeout_ms > 0 && start.elapsed().as_millis() as u64 >= timeout_ms {
                // A hung kernel almost always raised a fault; surface it
                // alongside the deadline.
                let fault = self.device.upgrade().and_then(|d| d.poll_faults_nonblocking());
                // The wait predicate is opaque here, so `target` is reported as
                // 0 and `what` names the operation.
                return Err(fault.unwrap_or(Error::TimelineTimeout {
                    what,
                    target: 0,
                    current: v,
                    waited_ms: timeout_ms,
                }));
            }
            if let Some(fault) = self.spin_or_escalate(start)? {
                return Err(fault);
            }
        }
    }

    /// Spin-wait until the value is ≥ `target` (increment convention — SDMA
    /// fence / monotonic timeline writes a literal increasing value).
    pub(crate) fn wait_signal_value(&self, target: u64, timeout_ms: u64) -> Result<()> {
        self.poll_until(|v| v >= target, timeout_ms, "wait_signal_value")
    }

    /// CP-written dispatch timestamps in nanoseconds, valid only after the
    /// signal retired. `None` until then, or when
    /// the slot was never targeted by timestamp commands (both stamps
    /// zeroed by [`reset`](Self::reset)).
    pub fn timestamps_ns(&self) -> Option<(u64, u64)> {
        // SAFETY: the full 64-byte slot is mapped; value lives at +8, so the
        // slot base is host_ptr − SIGNAL_VALUE_OFFSET.
        let (start, end) = unsafe {
            let base = (self.host_ptr.as_ptr() as *const u8).sub(SIGNAL_VALUE_OFFSET);
            (
                std::ptr::read_volatile(base.add(SIGNAL_START_TS_OFFSET) as *const u64),
                std::ptr::read_volatile(base.add(SIGNAL_END_TS_OFFSET) as *const u64),
            )
        };
        (start != 0 && end >= start).then(|| (start * NS_PER_TICK, end * NS_PER_TICK))
    }

    /// Tiered polling backoff: tight spin → `yield_now` → KFD `WAIT_EVENTS`
    /// once we've burned `WAIT_EVENTS_ESCALATE_MS` of wall time. The kernel
    /// wakes us when the device's `queue_event` fires, eliminating CPU burn
    /// for stalled or long-running dispatches.
    ///
    /// Returns the typed `WAIT_EVENTS` failure directly. Otherwise, `Some`
    /// carries a reported GPU fault and `None` means a normal wake-up, timeout,
    /// dropped device, or the pure spin/yield path.
    #[inline]
    fn spin_or_escalate(&self, start: std::time::Instant) -> Result<Option<Error>> {
        let elapsed_ms = start.elapsed().as_millis() as u64;
        if elapsed_ms >= WAIT_EVENTS_ESCALATE_MS
            && let Some(dev) = self.device.upgrade()
        {
            // Sleep in the kernel for at most another tier worth of time;
            // on return we re-check the host value and either complete
            // or escalate again. wait_events watches the three KFD
            // events (queue, mem fault, hw fault); a fault is returned
            // here so we bail with the actual error instead of grinding
            // through the rest of the timeout.
            match dev.wait_events(WAIT_EVENTS_ESCALATE_MS as u32) {
                Ok(Some(fault)) => return Ok(Some(fault)),
                Ok(None) => return Ok(None),
                Err(error) => return Err(error),
            }
        }
        std::hint::spin_loop();
        if start.elapsed().as_micros() >= 100 {
            std::thread::yield_now();
        }
        Ok(None)
    }
}

impl std::fmt::Debug for AmdSignal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AmdSignal")
            .field("slot", &self.slot)
            .field("value_addr", &format_args!("{:#x}", self.value_addr))
            .field("value", &self.value())
            .finish()
    }
}

impl Drop for AmdSignal {
    fn drop(&mut self) {
        if std::thread::panicking() || self.device.upgrade().is_some_and(|device| device.is_poisoned()) {
            return;
        }
        if let Some(pool) = self.pool.upgrade() {
            pool.release_slot(self.slot);
        }
    }
}

/// Watermark for the 2^32 timeline wraparound. PM4 WAIT_REG_MEM/RELEASE_MEM
/// compare the low 32 bits of the signal slot, so the counter must stay below
/// 2^32; we drain + reset at 2^31 to keep headroom.
pub const TIMELINE_WRAP_WATERMARK: u64 = 1 << 31;

/// A connector's timeline: an owned monotonic counter plus the shared signal
/// the GPU writes on dispatch completion. This is the ONE primitive that
/// crosses owners — a `PoolQueue` dispatches against it (advancing `value`), and
/// any thread can *drain* it (read `value`, poll the signal slot) without taking
/// lane publication authority. The registry on `AmdDeviceCore` holds
/// `Weak<PoolQueue>`, and `drain_all` fences in-flight work purely through these
/// atomics plus retained linked-plan timelines, keeping concurrent dispatch unblocked.
#[derive(Debug)]
pub struct Timeline {
    signal: Arc<AmdSignal>,
    /// Highest reserved value + ... i.e. the next value `next()` hands out.
    /// Starts at 1; the value a dispatch SIGNALS is `next()`'s return.
    value: AtomicU64,
}

impl Timeline {
    pub fn new(signal: Arc<AmdSignal>) -> Arc<Self> {
        signal.reset(0);
        Arc::new(Self { signal, value: AtomicU64::new(1) })
    }

    /// The shared completion timeline (for emitting wait/signal packets).
    #[inline]
    pub fn signal(&self) -> &Arc<AmdSignal> {
        &self.signal
    }

    /// GPU VA of the signal counter — what PM4/AQL wait/signal packets target.
    #[inline]
    pub fn value_addr(&self) -> u64 {
        self.signal.value_addr()
    }

    /// Reserve the next timeline value (`fetch_add(1)`); the caller emits a
    /// signal packet writing this value on completion.
    #[inline]
    pub fn next(&self) -> u64 {
        self.value.fetch_add(1, Ordering::AcqRel)
    }

    /// Highest value reserved so far (the value the next `signal` packet writes
    /// is `current()`; the last reserved is `current() - 1`).
    #[inline]
    pub fn current(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Block until the GPU has written the current `value - 1` snapshot. This
    /// never resets the generation because callers that do not hold the queue's
    /// publication lock can race a later reservation.
    pub fn drain(&self, timeout_ms: u64) -> Result<()> {
        let target = self.value.load(Ordering::Acquire).saturating_sub(1);
        if target == 0 {
            return Ok(());
        }
        self.signal.wait_signal_value(target, timeout_ms)?;
        Ok(())
    }

    /// Reset a drained generation. The caller must hold the same lock that
    /// serializes `next()` with queue publication and must have just drained.
    pub fn reset_after_drain(&self) {
        if self.value.load(Ordering::Acquire) > TIMELINE_WRAP_WATERMARK {
            debug_assert!(self.signal.value() >= self.value.load(Ordering::Acquire).saturating_sub(1));
            self.signal.reset(0);
            self.value.store(1, Ordering::Release);
        }
    }
}

impl crate::sync::DispatchTimestamps for AmdSignal {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        AmdSignal::timestamps_ns(self)
    }
}

impl TimelineSignal for AmdSignal {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn value(&self) -> u64 {
        // SAFETY: NonNull is valid for the pool's lifetime; AtomicU64 reads
        // are race-free.
        unsafe { self.host_ptr.as_ref().load(Ordering::Acquire) }
    }

    fn set(&self, value: u64) {
        // SAFETY: same as `value`.
        unsafe { self.host_ptr.as_ref().store(value, Ordering::Release) };
    }

    fn wait(&self, target: u64, timeout_ms: u64) -> Result<()> {
        // Tiered strategy (spin → yield → KFD WAIT_EVENTS sleep) + fault
        // surfacing live in the shared `poll_until` helper.
        self.poll_until(|v| v >= target, timeout_ms, "wait")
    }
}

/// Pool over a shared host-visible GTT region (one or more pages); hands out
/// [`AmdSignal`]s. Sized at construction: a flat per-owner model needs only a
/// handful, but DAG dispatch reserves one slot per kernel of the largest
/// captured graph (low hundreds for GigaAM), so the pool spans several pages.
pub struct SignalPool {
    /// Owning buffer — held to keep the GTT mapping alive while signals exist.
    _buffer: RawBuffer,
    base_gpu: u64,
    base_host: NonNull<u8>,
    /// Total slots carved from the region (rounded up to whole pages).
    slots: usize,
    free_slots: Mutex<Vec<u32>>,
    /// Captured at pool creation; signals downgrade-clone this into a `Weak`
    /// so `wait` can call `AmdDeviceCore::wait_events` for KFD escalation.
    device: Arc<AmdDeviceCore>,
}

// SAFETY: AtomicU64 covers concurrent reads/writes through `base_host`;
// `free_slots` is mutex-protected; `_buffer` is owned and not aliased.
unsafe impl Send for SignalPool {}
unsafe impl Sync for SignalPool {}

impl Drop for SignalPool {
    fn drop(&mut self) {
        if std::thread::panicking() {
            return;
        }
        if let Err(error) = self.device.synchronize_all() {
            tracing::warn!(?error, "SignalPool drop: backing allocation quarantined");
            return;
        }
        self._buffer.free_amd_device_in_place();
    }
}

impl SignalPool {
    /// Allocate the backing GTT page from `allocator` and partition it.
    ///
    /// Critical: the signal page must be **GTT-coherent + uncached** so that
    /// the GPU's decrement of the completion_signal field is immediately
    /// visible to the host (otherwise it sits in GPU L2 and we spin
    /// forever).
    pub fn new(allocator: &AmdAllocator, slots: usize) -> Result<Arc<Self>> {
        // Round up to a whole page so the GTT allocation is page-aligned and
        // every byte is usable as a slot.
        let slots = slots.max(1).next_multiple_of(SLOTS_PER_PAGE);
        let buffer = AmdBufferGuard::new(
            allocator.alloc_uncached_tagged(SLOT_BYTES * slots, crate::amd::va_registry::AllocTag::SignalPool)?,
        );
        let (base_gpu, base_host) = match buffer.buffer() {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => {
                return Err(Error::NotHostVisible { what: "SignalPool" });
            }
        };
        let free_slots = Mutex::new((0..slots as u32).rev().collect()); // pop low slots first
        let device = Arc::clone(allocator.dev.core());
        Ok(Arc::new(Self { _buffer: buffer.into_inner(), base_gpu, base_host, slots, free_slots, device }))
    }

    /// Carve off a new signal from the pool. Returns `Err` when exhausted.
    pub fn acquire(self: &Arc<Self>) -> Result<AmdSignal> {
        let slot = self.free_slots.lock().pop().ok_or_else(|| Error::AmdAllocFailed {
            reason: format!("SignalPool exhausted ({} slots in use)", self.slots),
        })?;
        let offset = slot as usize * SLOT_BYTES;
        let base_gpu = self.base_gpu + offset as u64;
        let base_host = self.base_host.as_ptr();
        // Lay out the amd_signal_t: zero the 64-byte slot, then set kind=USER so
        // the AQL packet processor treats it as a value signal, value stays 0.
        // SAFETY: offset + SLOT_BYTES <= page size by construction.
        let slot_host = unsafe { base_host.add(offset) };
        unsafe {
            std::ptr::write_bytes(slot_host, 0, SLOT_BYTES);
            std::ptr::write_volatile(
                slot_host as *mut i64,
                crate::amd::sys::hsa::amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64,
            );
        }
        let value_addr = base_gpu + SIGNAL_VALUE_OFFSET as u64;
        // SAFETY: the 8-byte value field sits at +SIGNAL_VALUE_OFFSET in the slot.
        let host_ptr = unsafe { NonNull::new_unchecked(slot_host.add(SIGNAL_VALUE_OFFSET) as *mut AtomicU64) };
        unsafe { host_ptr.as_ref().store(0, Ordering::Release) };
        Ok(AmdSignal {
            slot,
            base_gpu,
            value_addr,
            host_ptr,
            pool: Arc::downgrade(self),
            device: Arc::downgrade(&self.device),
        })
    }

    fn release_slot(&self, slot: u32) {
        self.free_slots.lock().push(slot);
    }

    /// Currently-free slot count. Used by graph capture to decide whether a DAG
    /// reservation (one slot per kernel, held for the graph's life) would leave
    /// enough headroom for per-op AQL back-pressure + PM4 counters; if not,
    /// capture falls back to blanket-BARRIER instead of starving dispatch.
    pub fn free(&self) -> usize {
        self.free_slots.lock().len()
    }
}

impl std::fmt::Debug for SignalPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let free = self.free_slots.lock().len();
        f.debug_struct("SignalPool")
            .field("base_gpu", &format_args!("{:#x}", self.base_gpu))
            .field("slots_total", &self.slots)
            .field("slots_free", &free)
            .finish()
    }
}
