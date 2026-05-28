//! `AmdConnector`: per-owner dispatch context.
//!
//! Step 2 of the per-owner connector refactor (plan `snug-honking-robin`).
//! Pulls the mutable per-queue state — scratch buffer, timeline signal +
//! counter, dispatch lock — off `AmdDevice` into a struct that future steps
//! will instantiate per-`ExecutionPlan` / per-graph. Each connector owns its
//! own dispatch slice; sharing happens only through the immutable
//! `Arc<AmdDeviceCore>`.
//!
//! Step 2 ships exactly **one** connector per `AmdDevice` (the "default"
//! connector), so visible behavior is unchanged. Steps 3-7 thread the
//! connector through `Program::execute_on`, `AmdGraph`, and `ExecutionPlan`,
//! at which point the `dispatch_lock` (still here transitionally) becomes
//! removable.
//!
//! `queue` / `kernargs_arena` / `signal_pool` migrate into this struct in
//! Step 3, where the factory + program-load wiring is reshaped so they
//! don't need to be passed in from outside.
//!
//! Tinygrad analogue: there is no direct equivalent — tinygrad bundles all
//! of this into `AMDDevice` (`runtime/ops_amd.py`) and relies on the GIL to
//! serialize across concurrent dispatchers. Splitting it out is a deliberate
//! Rust-side ownership-eliminates-locks divergence.

#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::amd::device::{AmdDeviceCore, ScratchState, alloc_scratch};
use crate::amd::signal::AmdSignal;
use crate::amd::sys::{ioctl, kfd};
use crate::error::Result;
use crate::sync::TimelineSignal;

/// Per-owner dispatch state. One instance per `ExecutionPlan` / `AmdGraph` in
/// the final architecture; Step 2 ships exactly one per `AmdDevice` (the
/// `default` connector) to keep the diff small.
///
/// Owns: scratch backing, timeline signal + counter, transitional
/// `dispatch_lock`. Step 3 adds `queue`, `kernargs_arena`, and
/// `signal_pool` ownership.
#[derive(Debug)]
pub struct AmdConnector {
    /// Shared immutable identity. Cloned across all connectors backed by the
    /// same physical AMD:N (and across `AmdDevice` for back-compat).
    core: Arc<AmdDeviceCore>,
    /// Per-connector scratch backing. Mirrors what used to live on
    /// `AmdDevice::scratch_state`; see that field's docstring for the
    /// `_ensure_has_local_memory` story (`ops_amd.py:1065-1081`). Once each
    /// connector has its own scratch, the dispatch_lock around
    /// realloc-vs-dispatch becomes unnecessary (Step 6).
    scratch_state: Mutex<ScratchState>,
    /// Per-connector timeline signal. Every kernel/copy submitted via this
    /// connector waits on the previous timeline value and signals the next
    /// one. Lazy-init from the factory (signal pool depends on an
    /// `AmdAllocator`, which depends on `Arc<AmdDevice>`, which contains
    /// this connector — chicken/egg avoided via `OnceLock`).
    /// Mirrors `HCQCompiled.timeline_signal` (`hcq.py:415`).
    timeline_signal: OnceLock<Arc<AmdSignal>>,
    /// Highest timeline value submitted through this connector. Reserved by
    /// `next_timeline()` via `fetch_add(1)`. Mirrors `timeline_value`
    /// (`hcq.py:405`).
    timeline_value: AtomicU64,
    /// Serializes [timeline acquire + live-scratch read + ring blit] against
    /// teardown drains. **Transitional**: while connectors are shared across
    /// dispatchers in Steps 2-6 this lock prevents scratch-realloc-vs-dispatch
    /// VA races. Targeted for deletion in Step 7 once each connector has
    /// exactly one owner.
    dispatch_lock: Mutex<()>,
}

impl AmdConnector {
    /// Build a connector against a freshly opened device core. Allocates the
    /// initial 128 B/thread scratch buffer (matches tinygrad's
    /// `_ensure_has_local_memory(128)` at `ops_amd.py:1010`).
    pub fn new(core: Arc<AmdDeviceCore>) -> Result<Arc<Self>> {
        let (scratch_va, scratch_size, tmpring_size, size_per_thread, scratch_handle) =
            alloc_scratch(&core.kfd_fd, &core.node, &core.arch, 128)?;
        Ok(Arc::new(Self {
            core,
            scratch_state: Mutex::new(ScratchState {
                gpu_va: scratch_va,
                size_per_thread,
                tmpring_size,
                handle: scratch_handle,
                size: scratch_size,
            }),
            timeline_signal: OnceLock::new(),
            timeline_value: AtomicU64::new(1),
            dispatch_lock: Mutex::new(()),
        }))
    }

    /// The immutable core this connector dispatches against.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// Acquire the dispatch/teardown serialization lock. See the field
    /// docstring for the invariant and the deletion path (Step 7).
    pub fn lock_dispatch(&self) -> parking_lot::MutexGuard<'_, ()> {
        self.dispatch_lock.lock()
    }

    /// Install the connector's timeline signal. Called exactly once from the
    /// device factory after the `SignalPool` is constructed. Subsequent
    /// calls are a no-op.
    pub fn init_timeline(&self, signal: Arc<AmdSignal>) {
        let _ = self.timeline_signal.set(signal);
    }

    /// Connector timeline signal (panics if [`init_timeline`] hasn't been
    /// called — that's a factory-wiring bug, not a runtime condition).
    pub fn timeline_signal(&self) -> &Arc<AmdSignal> {
        self.timeline_signal.get().expect("timeline_signal not initialized; call init_timeline from factory")
    }

    /// Reserve the next timeline value (`fetch_add(1)`). The caller emits a
    /// queue signal packet that writes this value to the connector's signal
    /// slot. Mirrors `HCQCompiled.next_timeline` (`hcq.py:447`).
    pub fn next_timeline(&self) -> u64 {
        self.timeline_value.fetch_add(1, Ordering::AcqRel)
    }

    /// Highest submitted timeline value (i.e. the value the next `signal`
    /// packet would write). `synchronize` waits until the GPU has written
    /// `value - 1`.
    pub fn timeline_value(&self) -> u64 {
        self.timeline_value.load(Ordering::Acquire)
    }

    /// Current scratch buffer GPU VA. Read under the scratch mutex; tiny
    /// lock window on the dispatch hot path.
    pub fn scratch_gpu_va(&self) -> u64 {
        self.scratch_state.lock().gpu_va
    }

    /// Packed `COMPUTE_TMPRING_SIZE` for the current scratch buffer.
    pub fn tmpring_size(&self) -> u32 {
        self.scratch_state.lock().tmpring_size
    }

    /// Drain all submitted GPU work on this connector. Blocks until the
    /// connector's timeline signal observes `timeline_value() - 1`.
    pub fn synchronize(&self) -> Result<()> {
        if let Some(err) = self.core.poison_error() {
            return Err(err);
        }
        // Skip cleanly when the timeline hasn't been wired up yet — the
        // factory installs it after `open()`, but allocator paths run during
        // device construction (e.g. ring/GART buffers) need to be a no-op.
        let Some(signal) = self.timeline_signal.get() else {
            return Ok(());
        };
        let target = self.timeline_value.load(Ordering::Acquire).saturating_sub(1);
        if target == 0 {
            return Ok(());
        }
        signal.wait_signal_value(target, 30_000).inspect_err(|e| self.core.poison(&e.to_string()))?;

        // Timeline wraparound (`hcq.py:442,480`). PM4 WAIT_REG_MEM / RELEASE_MEM
        // compare the *low 32 bits* of the signal, so the counter must stay
        // below 2^32. We've just drained to `target` (GPU idle), so it's safe
        // to reset the signal slot to 0 and restart the counter at 1.
        if self.timeline_value.load(Ordering::Acquire) > (1u64 << 31) {
            signal.set(0);
            self.timeline_value.store(1, Ordering::Release);
        }
        Ok(())
    }

    /// Ensure the connector's scratch backing has at least
    /// `private_segment_size` bytes per thread. Mirrors tinygrad's
    /// `_ensure_has_local_memory` (`ops_amd.py:1065-1081`). The old scratch
    /// buffer is freed (drain → unmap → munmap → free).
    pub fn ensure_has_local_memory(&self, private_segment_size: u32) -> Result<()> {
        let current = self.scratch_state.lock().size_per_thread;
        if private_segment_size <= current {
            return Ok(());
        }
        let (va, size, tmpring, rounded, handle) =
            alloc_scratch(&self.core.kfd_fd, &self.core.node, &self.core.arch, private_segment_size)?;
        // Hold the dispatch lock across the swap + free so the old scratch VA
        // is unmapped only when no in-flight dispatch can still program it.
        let _disp = self.lock_dispatch();
        let stale = {
            let mut state = self.scratch_state.lock();
            if rounded > state.size_per_thread {
                let old = (state.gpu_va, state.size, state.handle);
                *state = ScratchState { gpu_va: va, size_per_thread: rounded, tmpring_size: tmpring, handle, size };
                old
            } else {
                (va, size, handle)
            }
        };
        self.free_scratch(stale.0, stale.1, stale.2);
        Ok(())
    }

    /// Drain → unmap → munmap → free a scratch backing buffer. Old scratch is
    /// no longer referenced once the timeline drains (sole user is dispatch
    /// on this connector).
    fn free_scratch(&self, va: u64, size: usize, handle: u64) {
        if let Err(e) = self.synchronize() {
            tracing::warn!(?e, va, "scratch realloc: synchronize failed; freeing anyway");
        }
        let mut gpu_id = self.core.node.gpu_id;
        let mut unmap = kfd::kfd_ioctl_unmap_memory_from_gpu_args {
            handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        // SAFETY: fd alive; handle from a successful alloc_scratch.
        let _ = unsafe { ioctl::kfd_unmap_memory_from_gpu(self.core.kfd_fd.as_raw_fd(), &mut unmap as *mut _) };
        // SAFETY: va is the VA reserved by alloc_scratch's mmap.
        unsafe { libc::munmap(va as *mut _, size) };
        let mut free = kfd::kfd_ioctl_free_memory_of_gpu_args { handle };
        // SAFETY: same handle.
        let _ = unsafe { ioctl::kfd_free_memory_of_gpu(self.core.kfd_fd.as_raw_fd(), &mut free as *mut _) };
    }
}
