//! `AmdConnector`: per-owner dispatch context.
//!
//! Holds the dispatch state that must NOT be shared across independent
//! callers: a KFD compute queue (ring + doorbell), a kernarg bump arena,
//! the scratch backing, and the timeline signal + counter. Every
//! `ExecutionPlan` and every `AmdGraph` owns its own `AmdConnector` —
//! `AmdDevice` keeps one default connector for the few callers that bypass
//! the plan path (`AmdAllocator::_copyin`/`_copyout`/`_free` device-wide
//! synchronize; the `Program::execute` trait fallback used by
//! `benchmark_kernel`).
//!
//! Single-owner by construction means there's no dispatch lock: scratch
//! realloc, timeline reservation, kernarg bump, and ring submission all run
//! serially by ownership rather than under an explicit `Mutex<()>`. The
//! queue's internal `Mutex<QueueInner>` stays as a defensive primitive but
//! has exactly one writer in steady state.
//!
//! Tinygrad analogue: `AMDDevice` (`runtime/ops_amd.py`) bundles all of
//! this into one per-physical-GPU object and relies on the Python GIL to
//! serialize concurrent dispatchers. The Svod split is a deliberate
//! ownership-eliminates-locks divergence — sibling plans on the same
//! physical GPU contend only at the per-core synchronize barrier (rare),
//! not on a coarse device-wide dispatch mutex.

#![cfg(target_os = "linux")]

use std::os::fd::AsRawFd;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

use parking_lot::Mutex;

use crate::amd::AmdAllocator;
use crate::amd::device::{AmdDeviceCore, ScratchState, alloc_scratch};
use crate::amd::kernarg::KernargArena;
use crate::amd::queue::AmdComputeQueue;
use crate::amd::signal::AmdSignal;
use crate::amd::sys::{ioctl, kfd};
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

/// Per-owner dispatch state. One per `ExecutionPlan`, one per `AmdGraph`,
/// plus one default per `AmdDevice` for trait-fallback callers (the rest of
/// the runtime always goes through the plan/graph path).
///
/// Owns: KFD compute queue, kernarg arena, scratch backing, timeline signal,
/// timeline counter. Single ownership means no dispatch lock — every
/// mutation is serialized by the owner's exclusive access.
#[derive(Debug)]
pub struct AmdConnector {
    /// Shared immutable identity. Cloned across all connectors backed by the
    /// same physical AMD:N (and across `AmdDevice` for back-compat).
    core: Arc<AmdDeviceCore>,
    /// Per-connector KFD compute queue — own ring + doorbell + GART. Built
    /// fresh by `new_with_resources`; with one connector per plan/graph this
    /// means each owner pushes packets into an isolated ring, so the queue's
    /// internal `Mutex<QueueInner>` is exercised only by intra-owner traffic
    /// (effectively uncontended).
    queue: Arc<AmdComputeQueue>,
    /// Per-connector kernel-argument bump arena (16 MiB GTT). Wraps drain
    /// every live connector via the core's registry (see
    /// `KernargArena::bump`).
    arena: Arc<KernargArena>,
    /// Per-connector scratch backing. Mirrors what used to live on
    /// `AmdDevice::scratch_state`; see that field's docstring for the
    /// `_ensure_has_local_memory` story (`ops_amd.py:1065-1081`).
    scratch_state: Mutex<ScratchState>,
    /// Per-connector timeline signal. Every kernel/copy submitted via this
    /// connector waits on the previous timeline value and signals the next
    /// one. Acquired from the core's shared `SignalPool` at construction.
    /// Mirrors `HCQCompiled.timeline_signal` (`hcq.py:415`).
    timeline_signal: OnceLock<Arc<AmdSignal>>,
    /// Highest timeline value submitted through this connector. Reserved by
    /// `next_timeline()` via `fetch_add(1)`. Mirrors `timeline_value`
    /// (`hcq.py:405`).
    timeline_value: AtomicU64,
}

impl AmdConnector {
    /// Build a connector with its own KFD compute queue + kernarg arena +
    /// initial scratch. The timeline signal is acquired from the core's
    /// shared `SignalPool` (which the factory must have installed before any
    /// connector is built — `AmdDeviceCore::install_signal_pool`).
    ///
    /// Registers `Weak::self` in the core's connector list so device-wide
    /// `synchronize_all` (called by `AmdAllocator::_copyin`/`_copyout`/`_free`)
    /// drains every connector before any host-visible buffer free.
    pub fn new_with_resources(core: Arc<AmdDeviceCore>, allocator: &AmdAllocator) -> Result<Arc<Self>> {
        // Order matters: every step that allocates must come BEFORE
        // `alloc_scratch`. Earlier-built resources (`AmdComputeQueue`,
        // `KernargArena`, signal slot, timeline Arc) all have RAII cleanup
        // (queue: `Drop for AmdComputeQueue` + `Drop for QueueInner`; arena:
        // `Drop for KernargArena`; signal: `AmdSignal::Drop` returns slot to
        // pool). The scratch backing is the lone raw KFD allocation — keeping
        // it last means a failure before line 95 unwinds via `?` and the
        // RAII cleanups run; failure of `alloc_scratch` itself returns
        // without anything to leak.
        let queue = AmdComputeQueue::create(allocator)?;
        let arena = KernargArena::new(allocator, &core)?;
        let pool = core.signal_pool().cloned().ok_or_else(|| Error::Runtime {
            message: "AmdConnector::new_with_resources: signal pool not installed on core — \
                      install via AmdDeviceCore::install_signal_pool before building any connector"
                .into(),
        })?;
        let timeline_sig = Arc::new(pool.acquire()?);
        let (scratch_va, scratch_size, tmpring_size, size_per_thread, scratch_handle) =
            alloc_scratch(&core.kfd_fd, &core.node, &core.arch, 128)?;
        let conn = Arc::new(Self {
            core,
            queue,
            arena,
            scratch_state: Mutex::new(ScratchState {
                gpu_va: scratch_va,
                size_per_thread,
                tmpring_size,
                handle: scratch_handle,
                size: scratch_size,
            }),
            timeline_signal: OnceLock::new(),
            timeline_value: AtomicU64::new(1),
        });
        let _ = conn.timeline_signal.set(timeline_sig);
        // Register with the core so `synchronize_all` finds us. Opportunistic
        // GC: drop entries whose connector has already been dropped — this is
        // also what bounds the live KFD queue count (Drop returns the id via
        // `AmdComputeQueue::Drop`), so the previous >32 soft-warn is moot.
        {
            let mut list = conn.core.connectors.lock();
            list.retain(|w| w.strong_count() > 0);
            list.push(Arc::downgrade(&conn));
        }
        Ok(conn)
    }

    /// Borrow this connector's KFD compute queue.
    #[inline]
    pub fn queue(&self) -> &Arc<AmdComputeQueue> {
        &self.queue
    }

    /// Borrow this connector's kernarg arena.
    #[inline]
    pub fn arena(&self) -> &Arc<KernargArena> {
        &self.arena
    }

    /// The immutable core this connector dispatches against.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// Connector timeline signal (always installed by `new_with_resources`).
    pub fn timeline_signal(&self) -> &Arc<AmdSignal> {
        self.timeline_signal.get().expect("timeline_signal must be set by new_with_resources")
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
        // `new_with_resources` installs the timeline signal before returning,
        // so by the time any caller can reach `synchronize` it is set.
        let signal = self.timeline_signal.get().expect("timeline signal installed by new_with_resources");
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
        // Step 7: per-connector ownership means there's no in-flight dispatch
        // on a sibling thread to race against. `free_scratch` still drains
        // *this* connector's timeline before the unmap (just below) — that's
        // the per-connector invariant that replaced the device-wide lock.
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

impl Drop for AmdConnector {
    /// Drain in-flight GPU work before the connector dies, so a downstream
    /// `AmdAllocator::_copyout` (or any host read of a buffer this connector
    /// wrote) doesn't race the still-running kernel. Without this, a graph
    /// dropped at the end of `ExecutionPlan::execute` would lose its
    /// `timeline_signal` while its async dispatch is still pending — the
    /// device-wide `synchronize_all` would then skip the dead connector and
    /// host reads would observe partial / zero-initialized buffer state.
    ///
    /// Skipped during panic unwind: `synchronize` can block up to ~30 s per
    /// connector (signal timeout) and an unwinding test with N live
    /// connectors would pay N × 30 s before process teardown. The in-flight
    /// work is then implicitly abandoned — the caller saw a panic anyway, so
    /// observing partial GPU state is the lesser evil.
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!(
                "AmdConnector drop during panic unwind: skipping synchronize; \
                 in-flight GPU work + scratch backing abandoned"
            );
            return;
        }
        if let Err(e) = self.synchronize() {
            tracing::warn!(?e, "AmdConnector drop: synchronize failed (in-flight work lost)");
        }
        // Free the scratch backing. `ScratchState` is `Copy` with no `Drop`,
        // so without this every dropped connector would leak its ~50-200 MiB
        // KFD scratch alloc + host VA reservation. `free_scratch` does its
        // own synchronize internally (no-op now that we drained above) then
        // unmaps/munmaps/frees via KFD.
        let state = *self.scratch_state.lock();
        self.free_scratch(state.gpu_va, state.size, state.handle);
    }
}
