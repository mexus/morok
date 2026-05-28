//! `AmdSignal`: GTT-coherent timeline counter polled by the CPU.
//!
//! A `SignalPool` carves a single host-visible GTT page into 64-byte slots;
//! each [`AmdSignal`] owns a slot and exposes the `value_addr` GPU virtual
//! address so kernels / AQL barrier packets can write completion values.
//! CPU polling reads the same memory through the slot's `host_ptr`.

#![cfg(target_os = "linux")]

use std::any::Any;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDeviceCore;
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

/// Spin / yield budget before escalating to a KFD WAIT_EVENTS sleep. Matches
/// tinygrad's threshold in `hcq.py:294`: cheap polling for short waits, kernel
/// blocking for long ones so a stalled wait doesn't pin a CPU.
const WAIT_EVENTS_ESCALATE_MS: u64 = 200;

/// 16-byte slot inside the [`SignalPool`]'s shared GTT page. Matches
/// tinygrad's `hcq.py:455` (`range(0, alc.size, 16)`); slot holds the u64
/// value at offset 0 plus 8 bytes of padding.
const SLOT_BYTES: usize = 16;
/// 4 KiB page / 16 B per slot = 256 signals per pool. Matches tinygrad's
/// `sigalloc_size = 0x1000` default.
const SLOTS_PER_POOL: usize = 256;

/// A pool-allocated AMD signal.
///
/// The atomic counter the GPU writes to lives at `value_addr` (GPU VA) and
/// is also reachable via `host_ptr` for CPU polling. `Drop` returns the
/// slot to its pool. The pool keeps the underlying VRAM allocation alive.
pub struct AmdSignal {
    slot: u32,
    value_addr: u64,
    host_ptr: NonNull<AtomicU64>,
    pool: Weak<SignalPool>,
    /// Owning device core — used to escalate long waits to
    /// `AMDKFD_IOC_WAIT_EVENTS` on the device's `queue_event` (tinygrad
    /// `ops_amd.py:811`). `Weak` so signals don't extend device lifetime.
    device: Weak<AmdDeviceCore>,
}

// SAFETY: AtomicU64 covers all reads/writes; the host pointer comes from a
// shared mmap and is stable for the pool's lifetime.
unsafe impl Send for AmdSignal {}
unsafe impl Sync for AmdSignal {}

impl AmdSignal {
    /// Raw GPU virtual address of the signal counter (for AQL/PM4 packets).
    pub fn value_addr(&self) -> u64 {
        self.value_addr
    }

    /// Slot index inside the pool. Useful for debugging.
    pub fn slot(&self) -> u32 {
        self.slot
    }

    /// Spin-wait until the host-visible signal value is ≥ `target`. The PM4
    /// RELEASE_MEM packet emitted via AQL vendor IB writes the literal value
    /// (not decrement), so we use increment-convention here. Returns
    /// `Err(Runtime)` immediately if KFD reports a memory or hw fault during
    /// the wait, or after `timeout_ms` if the signal never reaches `target`
    /// — preferred over a silent infinite spin when something upstream
    /// prevents the GPU from running the kernel.
    ///
    /// Early-exit on fault is load-bearing for BEAM search: bad kernel
    /// configurations may fault the GPU, and waiting the full 30 s timeout
    /// for each rejected candidate is unaffordable.
    pub fn wait_signal_value(&self, target: u64, timeout_ms: u64) -> Result<()> {
        let mut start = std::time::Instant::now();
        let mut prev = u64::MAX;
        loop {
            let v = unsafe { self.host_ptr.as_ref().load(Ordering::Acquire) };
            if v >= target {
                return Ok(());
            }
            if v != prev {
                prev = v; // progress: timeout is for *no* progress, so reset.
                start = std::time::Instant::now();
            }
            if timeout_ms > 0 && start.elapsed().as_millis() as u64 >= timeout_ms {
                // Drain any pending fault info before reporting the timeout —
                // a hung kernel almost always raised a fault that we want to
                // surface alongside the deadline message.
                let fault = self.device.upgrade().and_then(|d| d.poll_faults_nonblocking());
                return Err(match fault {
                    Some(e) => e,
                    None => Error::Runtime {
                        message: format!(
                            "AmdSignal::wait_signal_value({target}) timed out after {timeout_ms} ms (current value={v})"
                        ),
                    },
                });
            }
            if let Some(fault) = self.spin_or_escalate(start) {
                return Err(fault);
            }
        }
    }

    /// Spin-wait until the host-visible signal value reaches `target` going
    /// *down* (HSA completion_signal convention: GPU decrements by 1 per
    /// completed dispatch). Returns `Err(Runtime)` immediately on a GPU
    /// fault, or after `timeout_ms` if the signal never reaches `target`.
    pub fn wait_decrement_to(&self, target: u64, timeout_ms: u64) -> Result<()> {
        let mut start = std::time::Instant::now();
        let mut prev = u64::MAX;
        loop {
            // Read as i64 to compare across the HSA decrement convention: if
            // the GPU goes below `target` (shouldn't happen for a balanced
            // pre-set + 1 dispatch) we still wake up.
            let v = unsafe { self.host_ptr.as_ref().load(Ordering::Acquire) };
            if (v as i64) <= (target as i64) {
                return Ok(());
            }
            if v != prev {
                prev = v;
                start = std::time::Instant::now();
            }
            if timeout_ms > 0 && start.elapsed().as_millis() as u64 >= timeout_ms {
                let fault = self.device.upgrade().and_then(|d| d.poll_faults_nonblocking());
                return Err(match fault {
                    Some(e) => e,
                    None => Error::Runtime {
                        message: format!(
                            "AmdSignal::wait_decrement_to({target}) timed out after {timeout_ms} ms (current value={v})"
                        ),
                    },
                });
            }
            if let Some(fault) = self.spin_or_escalate(start) {
                return Err(fault);
            }
        }
    }

    /// Tiered polling backoff: tight spin → `yield_now` → KFD `WAIT_EVENTS`
    /// once we've burned `WAIT_EVENTS_ESCALATE_MS` of wall time. The kernel
    /// wakes us when the device's `queue_event` fires, eliminating CPU burn
    /// for stalled or long-running dispatches. Mirrors tinygrad `hcq.py:294`
    /// → `KFDIface.sleep` (`ops_amd.py:811`).
    ///
    /// Returns `Some(Error)` when KFD reports a GPU fault during the kernel
    /// wait so callers can break out of the spin immediately. `None` for
    /// normal wake-ups (queue_event fired, WAIT_EVENTS timed out internally,
    /// device dropped, or the pure spin/yield path).
    #[inline]
    fn spin_or_escalate(&self, start: std::time::Instant) -> Option<Error> {
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
                Ok(Some(fault)) => return Some(fault),
                Ok(None) | Err(_) => return None,
            }
        }
        std::hint::spin_loop();
        if start.elapsed().as_micros() >= 100 {
            std::thread::yield_now();
        }
        None
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
        if let Some(pool) = self.pool.upgrade() {
            pool.release_slot(self.slot);
        }
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
        let mut start = std::time::Instant::now();
        let mut prev = u64::MAX;
        // Tinygrad's tiered strategy (`hcq.py:294`): spin → yield → KFD
        // WAIT_EVENTS sleep. The shared helper handles the escalation tiers
        // and surfaces GPU faults so BEAM search can bail on a bad config
        // instead of blocking for the full timeout.
        loop {
            let v = self.value();
            if v >= target {
                return Ok(());
            }
            if v != prev {
                prev = v;
                start = std::time::Instant::now();
            }
            if timeout_ms > 0 && start.elapsed().as_millis() as u64 >= timeout_ms {
                let fault = self.device.upgrade().and_then(|d| d.poll_faults_nonblocking());
                return Err(match fault {
                    Some(e) => e,
                    None => Error::Runtime { message: format!("AmdSignal::wait timed out (>= {target})") },
                });
            }
            if let Some(fault) = self.spin_or_escalate(start) {
                return Err(fault);
            }
        }
    }
}

/// Pool over one shared host-visible GTT page; hands out [`AmdSignal`]s.
pub struct SignalPool {
    /// Owning buffer — held to keep the GTT mapping alive while signals exist.
    _buffer: RawBuffer,
    base_gpu: u64,
    base_host: NonNull<u8>,
    free_slots: Mutex<Vec<u32>>,
    /// Captured at pool creation; signals downgrade-clone this into a `Weak`
    /// so `wait` can call `AmdDeviceCore::wait_events` for KFD escalation.
    device: Arc<AmdDeviceCore>,
}

// SAFETY: AtomicU64 covers concurrent reads/writes through `base_host`;
// `free_slots` is mutex-protected; `_buffer` is owned and not aliased.
unsafe impl Send for SignalPool {}
unsafe impl Sync for SignalPool {}

impl SignalPool {
    /// Allocate the backing GTT page from `allocator` and partition it.
    ///
    /// Critical: the signal page must be **GTT-coherent + uncached** so that
    /// the GPU's decrement of the completion_signal field is immediately
    /// visible to the host (otherwise it sits in GPU L2 and we spin
    /// forever). Matches tinygrad's
    /// `iface.alloc(..., uncached=True, cpu_access=True)` for the signal
    /// pool at `hcq.py:HCQCompiled.new_signal`.
    pub fn new(allocator: &AmdAllocator) -> Result<Arc<Self>> {
        let buffer = allocator.alloc_uncached(SLOT_BYTES * SLOTS_PER_POOL)?;
        let (base_gpu, base_host) = match &buffer {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
            _ => {
                return Err(Error::AmdAllocFailed { reason: "SignalPool requires host-visible AMD buffer".into() });
            }
        };
        let free_slots = Mutex::new((0..SLOTS_PER_POOL as u32).rev().collect()); // pop low slots first
        let device = Arc::clone(allocator.dev.core());
        Ok(Arc::new(Self { _buffer: buffer, base_gpu, base_host, free_slots, device }))
    }

    /// Carve off a new signal from the pool. Returns `Err` when exhausted.
    pub fn acquire(self: &Arc<Self>) -> Result<AmdSignal> {
        let slot = self.free_slots.lock().pop().ok_or_else(|| Error::AmdAllocFailed {
            reason: format!("SignalPool exhausted ({SLOTS_PER_POOL} slots in use)"),
        })?;
        let offset = slot as usize * SLOT_BYTES;
        let value_addr = self.base_gpu + offset as u64;
        // SAFETY: offset < page size; AtomicU64 fits in a 16-byte slot.
        let host_ptr = unsafe { NonNull::new_unchecked(self.base_host.as_ptr().add(offset) as *mut AtomicU64) };
        // Zero the slot so the new signal starts at 0.
        // SAFETY: same as above.
        unsafe { host_ptr.as_ref().store(0, Ordering::Release) };
        Ok(AmdSignal { slot, value_addr, host_ptr, pool: Arc::downgrade(self), device: Arc::downgrade(&self.device) })
    }

    fn release_slot(&self, slot: u32) {
        self.free_slots.lock().push(slot);
    }
}

impl std::fmt::Debug for SignalPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let free = self.free_slots.lock().len();
        f.debug_struct("SignalPool")
            .field("base_gpu", &format_args!("{:#x}", self.base_gpu))
            .field("slots_total", &SLOTS_PER_POOL)
            .field("slots_free", &free)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Live pool round-trip on real hardware (skipped when no supported AMD
    /// GPU is present).
    #[test]
    fn signal_pool_acquire_release_roundtrip() {
        let alloc = match AmdAllocator::new(0) {
            Ok(a) => a,
            Err(_) => {
                eprintln!("skipping: no supported AMD GPU");
                return;
            }
        };
        let pool = SignalPool::new(&alloc).expect("create pool");
        let s1 = pool.acquire().expect("acquire 1");
        let s2 = pool.acquire().expect("acquire 2");
        assert_ne!(s1.value_addr(), s2.value_addr());
        assert_eq!(s1.value(), 0);
        s1.set(7);
        assert_eq!(s1.value(), 7);
        drop(s1);
        // After drop, slot is back in the pool; acquiring should give it back
        // (slot count restored).
        let s3 = pool.acquire().expect("acquire 3");
        let _ = s3;
        let _ = s2;
    }

    #[test]
    fn signal_pool_exhaustion_is_clean_err() {
        let alloc = match AmdAllocator::new(0) {
            Ok(a) => a,
            Err(_) => {
                eprintln!("skipping: no supported AMD GPU");
                return;
            }
        };
        let pool = SignalPool::new(&alloc).expect("create pool");
        let mut sigs = Vec::new();
        for _ in 0..SLOTS_PER_POOL {
            sigs.push(pool.acquire().expect("ack"));
        }
        let err = pool.acquire().expect_err("pool must be exhausted");
        assert!(matches!(err, Error::AmdAllocFailed { .. }));
    }
}
