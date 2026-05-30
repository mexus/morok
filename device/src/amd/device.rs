//! `AmdDevice`: KFD-direct device handle.
//!
//! Opens `/dev/kfd` and `/dev/dri/renderD*`, parses topology, calls
//! `AMDKFD_IOC_ACQUIRE_VM`. Owns an `Arc<AmdDeviceCore>` (the immutable
//! per-physical-AMD:N identity — KFD/DRM fds, topology, event-page state,
//! poison latch, shared signal pool) plus a lazily-installed default
//! `AmdConnector` used by trait-fallback callers (`benchmark_kernel`) and
//! by the device-wide synchronize chain (`AmdAllocator::_copyin`/`_copyout`
//! /`_free`). Per-plan and per-graph callers build their own
//! `AmdConnector` against the same shared core — they don't touch the
//! default connector.

#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use nix::fcntl::{OFlag, open};
use nix::sys::stat::Mode;
use once_cell::sync::{Lazy, OnceCell};
use parking_lot::Mutex;
use svod_dtype::AmdArch;
use tracing::debug;

use crate::amd::sys::{ioctl, kfd};
use crate::amd::topology::{AmdNode, enumerate};
use crate::error::{Error, Result};

/// Per-process cache of opened `AmdDevice`s, keyed by `device_id`. KFD only
/// accepts one `ACQUIRE_VM` per (process, GPU); the cache ensures that
/// concurrent `AmdAllocator::new(0)` calls — e.g. registry-cached
/// LRU-wrapped allocator + factory-side queue/arena setup — share the same
/// `Arc<AmdDevice>` instead of double-opening.
///
/// The cached `Arc<AmdDevice>` carries the shared `Arc<AmdDeviceCore>` —
/// per-plan and per-graph callers reach the core via `AmdDevice::core()` and
/// build their own `AmdConnector` against it (no extra KFD opens).
static DEVICE_CACHE: Lazy<Mutex<HashMap<usize, Arc<AmdDevice>>>> = Lazy::new(Default::default);

/// Process-wide `/dev/kfd` handle. Tinygrad opens KFD once per process
/// (`KFDIface.kfd`) and all devices share it — events created on a per-device
/// basis are addressed by `event_id` against this shared fd.
static GLOBAL_KFD: Lazy<Mutex<Option<Arc<OwnedFd>>>> = Lazy::new(Default::default);

/// Process-wide event-page state. Tinygrad allocates the 0x8000 GTT event
/// page exactly once per process (`KFDIface.event_page`); the first device
/// allocates+binds it via `CREATE_EVENT(event_page_offset=handle)`,
/// subsequent devices just `MAP_MEMORY_TO_GPU` it to their `gpu_id`. Matches
/// `ops_amd.py:731-733`.
static EVENT_PAGE: Lazy<Mutex<Option<EventPageState>>> = Lazy::new(Default::default);

#[derive(Debug, Clone, Copy)]
pub(crate) struct EventPageState {
    pub(crate) handle: u64,
    pub(crate) va: u64,
    pub(crate) size: usize,
}

/// Scratch backing memory + `COMPUTE_TMPRING_SIZE` packing. Held under a
/// mutex on `AmdConnector` so [`AmdConnector::ensure_has_local_memory`] can
/// grow the scratch buffer when a freshly-loaded program demands more bytes
/// per thread than what's currently allocated. `pub(crate)` because Step 2
/// moved the owning field into the sibling `connector` module.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScratchState {
    /// GPU VA of the current scratch buffer.
    pub gpu_va: u64,
    /// Bytes per thread (rounded up to 4-byte slot stride for wave64).
    /// Equivalent to tinygrad's `max_private_segment_size` (ops_amd.py:1066).
    pub size_per_thread: u32,
    /// Packed `COMPUTE_TMPRING_SIZE` register value.
    pub tmpring_size: u32,
    /// KFD handle + total byte size of the backing alloc — needed to free the
    /// old buffer when scratch grows (mirror tinygrad `_realloc`).
    pub handle: u64,
    pub size: usize,
}

/// Immutable per-physical-AMD:N identity: KFD/DRM fds, topology, event-page
/// state, fault latch, shared signal pool. One instance per physical GPU
/// (KFD rejects double `ACQUIRE_VM`); shared as `Arc<AmdDeviceCore>` by
/// every `AmdConnector` built against the device. Connector registry and
/// `synchronize_all` live here so the host can drain every live connector
/// before any destructive operation (`AmdAllocator::_free`, etc.).
///
/// The backend seam: all KFD ioctls (memory alloc/free, queue ring setup/
/// teardown, event waits) route through `iface`. KFD-specific state (the kfd/
/// drm fds, ABI version, event ids, event page) lives on the [`KfdIface`]
/// implementor, not the core. `node` + `arch` stay here as the device identity.
#[derive(Debug)]
pub struct AmdDeviceCore {
    pub node: AmdNode,
    pub arch: AmdArch,
    /// Backend implementation (KFD today). All ioctls route through this.
    iface: Arc<dyn crate::amd::iface::AmdIface>,
    /// Whether an SDMA copy queue is available on this physical device. Set
    /// by the factory after it tries to create one (`ops_amd.py:1000`).
    /// Today every AMD buffer is host-visible + memcpy'd, so this stays
    /// `false` and the SDMA queue is dead code — kept on the core for the
    /// future SDMA revival.
    has_sdma_queue: AtomicBool,
    /// Poison latch. Once a GPU fault/timeout is observed, the device is dead:
    /// every `synchronize`/`execute` against any connector on this device
    /// fails fast. Mirrors tinygrad's `error_state` (`hcq.py:421`). Per-physical
    /// device because a memory fault corrupts the whole VM, not just one queue.
    poisoned: AtomicBool,
    error_msg: OnceLock<String>,
    /// Registry of every connector's `Timeline` built against this core. Weak
    /// so dropped connectors don't keep timelines alive. Used by
    /// [`AmdDeviceCore::synchronize_all`] to drain ALL in-flight GPU work
    /// before destructive host-visible operations
    /// (`AmdAllocator::_copyin`/`_copyout`/`_free`). Holding `Weak<Timeline>`
    /// (not `Weak<AmdConnector>`) means the drainer reads only the timeline
    /// atomic + signal slot and NEVER touches a connector's queue — that
    /// decoupling is what lets dispatch run lock-free.
    pub(crate) timelines: parking_lot::Mutex<Vec<Weak<crate::amd::signal::Timeline>>>,
    /// Process-global signal pool, allocated once per physical device. Lazily
    /// installed by the device factory and shared across every `AmdConnector`
    /// (timeline signal acquired here at connector construction) — pool access
    /// is rare (slot alloc on connector build), and one pool covers many
    /// connectors at 4 KiB total VRAM.
    signal_pool: OnceLock<Arc<crate::amd::signal::SignalPool>>,
    /// How owners obtain a connector and whether dispatch is serialized. Built
    /// once at open from `SVOD_AMD_SINGLE_QUEUE`; see [`Dispatcher`]. Each mode
    /// owns only its own state — no cross-mode dead fields, and the mode is a
    /// type rather than a boolean re-checked at every call site.
    dispatcher: Dispatcher,
}

/// Per-device dispatch strategy: how an owner (plan / graph / the
/// `Program::execute` fallback) gets an [`AmdConnector`](crate::amd::connector::AmdConnector)
/// and whether dispatch is serialized.
#[derive(Debug)]
enum Dispatcher {
    /// Lock-free per-owner (`SVOD_AMD_SINGLE_QUEUE=0`): each owner leases an
    /// exclusively-owned connector from the idle pool (bounded by
    /// [`CONNECTOR_POOL_CAP`]); the GPU's MES interleaves the N queues, so
    /// dispatch needs no CPU lock. Returning a lease puts the connector back in
    /// the pool (or drops it over cap → `AmdComputeQueue::Drop` destroys the KFD
    /// queue). The lease being exclusive + un-aliasable is what stops two
    /// dispatchers from sharing one KFD queue.
    MultiQueue { pool: parking_lot::Mutex<Vec<Arc<crate::amd::connector::AmdConnector>>> },
    /// KFD-safe single-queue (default): every owner shares ONE connector per
    /// physical device — built on first lease, kept for the device's lifetime —
    /// and dispatch + scratch-realloc are serialized behind `lock`. The kernel
    /// then only ever sees one compute queue per GPU (tinygrad's model),
    /// sidestepping the MES/runlist overload the multi-queue path triggers under
    /// load. `lock` guards two *non-nested* critical sections — dispatch
    /// (`dispatch_pm4`/`dispatch_aql`/`submit_dwords`) and the realloc branch of
    /// `AmdConnector::ensure_has_local_memory` — so there is no reentrant
    /// deadlock. Returning a lease is a no-op: the shared connector lives in
    /// `shared`, and a dropped lease just decrements its `Arc` refcount.
    SingleQueue { shared: OnceCell<Arc<crate::amd::connector::AmdConnector>>, lock: parking_lot::Mutex<()> },
}

/// Max idle connectors retained per physical AMD:N. Each pooled connector
/// holds a live KFD compute queue, and the per-process hardware budget is
/// small (~24 user compute queues on CDNA; HIP's `GPU_MAX_HW_QUEUES` defaults
/// to 4). Retaining more idle queues than the GPU can run concurrently just
/// invites runlist oversubscription, so we keep the pool small and reuse
/// aggressively. Over-cap returns drop normally (queue destroyed via
/// `AmdComputeQueue::Drop`).
pub const CONNECTOR_POOL_CAP: usize = 4;

/// Open handle to one AMD GPU node.
///
/// A thin owner of the immutable `AmdDeviceCore`. There is no per-device
/// "default" connector: every dispatcher holds its own connector — plans and
/// graphs build/lease one for their lifetime, and the `Program::execute`
/// trait fallback leases one per call from `core.lease_connector`. The
/// device-wide synchronize chain (`AmdAllocator::_copyin`/`_copyout`/`_free`)
/// routes through `dev.synchronize() → core.synchronize_all()`, which drains
/// EVERY connector registered on the core (pooled, leased, plan, and graph).
///
/// Immutable Core fields stay reachable via [`Deref`] — `self.dev.node`,
/// `self.dev.kfd_fd`, `self.dev.poison_error()`, etc.
#[derive(Debug)]
pub struct AmdDevice {
    /// Immutable identity (cloneable across connectors).
    core: Arc<AmdDeviceCore>,
}

impl std::ops::Deref for AmdDevice {
    type Target = AmdDeviceCore;
    #[inline]
    fn deref(&self) -> &AmdDeviceCore {
        &self.core
    }
}

impl AmdDevice {
    /// Open the `device_id`-th GPU node.
    ///
    /// Returns:
    /// - `Err(NoAmdGpu)` when there is no `/dev/kfd`, no GPU nodes in
    ///   topology, or `device_id` is out of range. Never panics.
    /// - `Err(AmdAllocFailed)` when the host has hardware we don't support
    ///   (currently only RDNA3+CDNA per the Phase 0 `AmdArch` set).
    /// - `Err(AmdIoctl)` for KFD failures (permission denied, no event page).
    pub fn open(device_id: usize) -> Result<Arc<Self>> {
        // Fast path: device already opened by another caller (registry +
        // factory share via DEVICE_CACHE).
        {
            let cache = DEVICE_CACHE.lock();
            if let Some(dev) = cache.get(&device_id) {
                return Ok(Arc::clone(dev));
            }
        }
        let dev = Self::open_uncached(device_id)?;
        DEVICE_CACHE.lock().insert(device_id, Arc::clone(&dev));
        Ok(dev)
    }

    fn open_uncached(device_id: usize) -> Result<Arc<Self>> {
        let nodes = enumerate();
        if nodes.is_empty() {
            return Err(Error::NoAmdGpu { reason: "no KFD topology nodes (no /dev/kfd?)".into() });
        }
        let node = nodes
            .get(device_id)
            .ok_or_else(|| Error::NoAmdGpu {
                reason: format!("device_id {device_id} out of range; {} GPU node(s) present", nodes.len()),
            })?
            .clone();
        let arch = AmdArch::from_gfx_target_version(node.gfx_target_version).ok_or_else(|| Error::AmdAllocFailed {
            reason: format!(
                "unsupported gfx target {} (decoded major.minor.step = {}.{}.{}); supported families: \
                 CDNA gfx942/950, RDNA3 gfx1100/1101/1102/1151, RDNA4 gfx1200/1201 \
                 (matches tinygrad's `assert target[0] in (11, 12) or target in ((9,4,2),(9,5,0))` \
                 in ops_amd.py:962)",
                node.gfx_target_version,
                node.gfx_target_version / 10_000,
                (node.gfx_target_version / 100) % 100,
                node.gfx_target_version % 100,
            ),
        })?;

        // Backend selection. Today only the KFD-direct backend exists; the
        // `SVOD_AMD_BACKEND` knob is the seam where the userspace AM driver
        // will plug in. All KFD bring-up + ioctls live on `KfdIface`.
        let backend = std::env::var("SVOD_AMD_BACKEND").unwrap_or_else(|_| "kfd".into());
        if backend != "kfd" {
            return Err(Error::NoAmdGpu {
                reason: format!("unknown SVOD_AMD_BACKEND={backend} (only 'kfd' supported)"),
            });
        }
        let iface: Arc<dyn crate::amd::iface::AmdIface> = Arc::new(crate::amd::iface::KfdIface::open(&node)?);

        debug!(node = node.node_id, gpu_id = node.gpu_id, arch = arch.mcpu(), backend = %backend, "AmdDevice opened");

        // KFD-safe single-queue is the default; SVOD_AMD_SINGLE_QUEUE=0 opts into
        // lock-free multi-queue. See `Dispatcher`.
        let dispatcher = if std::env::var("SVOD_AMD_SINGLE_QUEUE").ok().map(|s| s != "0").unwrap_or(true) {
            Dispatcher::SingleQueue { shared: OnceCell::new(), lock: parking_lot::Mutex::new(()) }
        } else {
            Dispatcher::MultiQueue { pool: parking_lot::Mutex::new(Vec::new()) }
        };

        let core = Arc::new(AmdDeviceCore {
            node,
            arch,
            iface,
            has_sdma_queue: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            error_msg: OnceLock::new(),
            timelines: parking_lot::Mutex::new(Vec::new()),
            signal_pool: OnceLock::new(),
            dispatcher,
        });
        Ok(Arc::new(Self { core }))
    }

    /// Borrow the shared immutable core — used to build/lease per-owner
    /// `AmdConnector`s against the same physical device without re-acquiring
    /// KFD.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// Drain all submitted GPU work on every connector backed by this device.
    /// Must drain ALL connectors (not just the default) — once Steps 4-5 of
    /// the connector refactor make `ExecutionPlan`/`AmdGraph` own their own
    /// connectors, kernels signal on the OWNER's timeline. Skipping the
    /// per-owner drain would let `AmdAllocator::_copyout`/`_copyin`/`_free`
    /// observe an unfinished kernel's buffer.
    pub fn synchronize(&self) -> Result<()> {
        self.core.synchronize_all()
    }
}

impl AmdDeviceCore {
    /// Drain every connector backed by this core — the per-VM fence before any
    /// destructive host-visible op (`AmdAllocator::_copyin`/`_copyout`/`_free`).
    /// Iterates the `timelines` registry and drains each via `Timeline::drain`,
    /// which reads only the timeline atomic + signal slot — it NEVER touches a
    /// connector's queue, so a concurrent owner can keep dispatching lock-free
    /// while this runs. A freed/read buffer has no live handle, so the owner
    /// can't add new work referencing it; draining each timeline to its current
    /// value fences all in-flight readers. Fast on idle timelines (target 0).
    pub fn synchronize_all(&self) -> Result<()> {
        if let Some(err) = self.poison_error() {
            return Err(err);
        }
        // Snapshot strong refs to release the registry lock before the
        // potentially multi-second waits, keeping each timeline alive across
        // its drain so a concurrent connector drop can't pull the rug out.
        let live: Vec<Arc<crate::amd::signal::Timeline>> =
            self.timelines.lock().iter().filter_map(|w| w.upgrade()).collect();
        // Drain every timeline; collect the first error but keep going so a
        // single stuck connector doesn't strand buffer-frees on the others.
        let mut first_err: Option<Error> = None;
        for t in live {
            if let Err(e) = t.drain(30_000) {
                tracing::warn!(?e, "synchronize_all: timeline drain failed; continuing");
                self.poison(&e.to_string());
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        // Opportunistic GC of dropped timeline entries. The registry is touched
        // here on every host read/free, so dead Weaks don't accumulate.
        self.timelines.lock().retain(|w| w.strong_count() > 0);
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Borrow the backend seam — all KFD ioctls (alloc/free/ring/wait) route
    /// through it. The allocator, queue, and connector helpers call this.
    #[inline]
    pub(crate) fn iface(&self) -> &Arc<dyn crate::amd::iface::AmdIface> {
        &self.iface
    }

    /// Borrow the process-global signal pool (lazy-installed by the device
    /// factory). `None` before the factory has run; once initialized, every
    /// connector built against this core shares it.
    pub fn signal_pool(&self) -> Option<&Arc<crate::amd::signal::SignalPool>> {
        self.signal_pool.get()
    }

    /// Whether dispatch is serialized onto one shared connector (single-queue
    /// mode). The graph factory checks this to fall back to per-call dispatch —
    /// the captured-replay path keeps its own connector/ring, which the
    /// single-queue dispatch lock doesn't cover.
    #[inline]
    pub fn is_single_queue(&self) -> bool {
        matches!(self.dispatcher, Dispatcher::SingleQueue { .. })
    }

    /// Acquire the dispatch lock in single-queue mode; `None` in multi-queue.
    /// Held by the dispatch methods and the scratch-realloc path so the shared
    /// connector's ring/timeline/scratch are mutated by one thread at a time. In
    /// multi-queue mode each connector is exclusively owned, so this is `None`
    /// and dispatch stays lock-free.
    #[inline]
    pub(crate) fn exec_guard(&self) -> Option<parking_lot::MutexGuard<'_, ()>> {
        match &self.dispatcher {
            Dispatcher::SingleQueue { lock, .. } => Some(lock.lock()),
            Dispatcher::MultiQueue { .. } => None,
        }
    }

    /// Lease an `AmdConnector` for this core, per the [`Dispatcher`] mode:
    /// the shared connector (single-queue) or a pooled exclusive one
    /// (multi-queue). Either way the returned
    /// [`ConnectorLease`](crate::amd::connector::ConnectorLease) `Deref`s to
    /// `&AmdConnector`, so callers are mode-agnostic; [`return_connector`](Self::return_connector)
    /// (run on lease drop) does the mode-appropriate thing.
    pub fn lease_connector(
        self: &Arc<Self>,
        allocator: &crate::amd::AmdAllocator,
    ) -> Result<crate::amd::connector::ConnectorLease> {
        let conn = match &self.dispatcher {
            // Build the one shared connector on first lease, reuse thereafter.
            Dispatcher::SingleQueue { shared, .. } => shared
                .get_or_try_init(|| {
                    crate::amd::connector::AmdConnector::new_with_resources(Arc::clone(self), allocator)
                })
                .map(Arc::clone)?,
            Dispatcher::MultiQueue { pool } => match pool.lock().pop() {
                Some(c) => c,
                None => crate::amd::connector::AmdConnector::new_with_resources(Arc::clone(self), allocator)?,
            },
        };
        Ok(crate::amd::connector::ConnectorLease::new(conn, Arc::clone(self)))
    }

    /// Return a connector when its lease drops. Multi-queue: clone it back into
    /// the idle pool, unless at [`CONNECTOR_POOL_CAP`] (then the lease's `Arc`
    /// drops → `AmdComputeQueue::Drop` destroys the KFD queue). Single-queue: a
    /// no-op — the shared connector lives in the `Dispatcher`; the lease's `Arc`
    /// clone just decrements the refcount when it drops.
    pub(crate) fn return_connector(&self, conn: &Arc<crate::amd::connector::AmdConnector>) {
        if let Dispatcher::MultiQueue { pool } = &self.dispatcher {
            let mut pool = pool.lock();
            if pool.len() < CONNECTOR_POOL_CAP {
                pool.push(Arc::clone(conn));
            }
        }
    }

    /// Install the signal pool. Called once per physical device by the
    /// runtime factory; subsequent calls are a no-op.
    pub fn install_signal_pool(&self, pool: Arc<crate::amd::signal::SignalPool>) {
        let _ = self.signal_pool.set(pool);
    }
}

impl AmdDeviceCore {
    /// Record whether an SDMA copy queue was successfully created. Called once
    /// from the device factory. When `false`, `AmdAllocator::_alloc` forces
    /// `cpu_access` so every buffer is host-visible and copies use `memmove`.
    pub fn set_has_sdma_queue(&self, present: bool) {
        self.has_sdma_queue.store(present, Ordering::Release);
    }

    /// Whether an SDMA copy queue is available (`ops_amd.py` `has_sdma_queue`).
    #[inline]
    pub fn has_sdma_queue(&self) -> bool {
        self.has_sdma_queue.load(Ordering::Acquire)
    }

    /// Block in the kernel for up to `timeout_ms` waiting on **any** of the
    /// device's three events (queue completion, memory fault, hw exception).
    /// Mirrors tinygrad `KFDIface.sleep` at `ops_amd.py:811`: signal polling
    /// escalates to this after a fixed spin/yield budget so a stalled wait
    /// doesn't burn CPU.
    ///
    /// Returns `Ok(Some(fault))` when a fault event fired (caller should bail
    /// with that error rather than continue polling the signal value).
    /// Returns `Ok(None)` for normal wake-ups (queue_event fired, timeout,
    /// or no event yet).
    pub fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>> {
        let r = self.iface.wait_events(timeout_ms)?;
        // Poison with the bare fault message (not `Error::Display`, which would
        // prepend "runtime error: "). The backend already built the rich string.
        if let Some(Error::Runtime { message }) = &r {
            self.poison(message);
        }
        Ok(r)
    }

    /// `true` once a fault/timeout has poisoned the device. Hot-path gate.
    #[inline]
    pub fn is_poisoned(&self) -> bool {
        self.poisoned.load(Ordering::Relaxed)
    }

    /// Latch a fault: device becomes unusable, message recorded once.
    pub fn poison(&self, msg: &str) {
        let _ = self.error_msg.set(msg.to_string());
        self.poisoned.store(true, Ordering::Relaxed);
    }

    /// Recorded fault if poisoned, else `None`.
    pub fn poison_error(&self) -> Option<Error> {
        self.is_poisoned().then(|| Error::Runtime {
            message: self.error_msg.get().cloned().unwrap_or_else(|| "AMD device poisoned".into()),
        })
    }

    /// Non-blocking check: did the memory-fault or hw-exception event fire
    /// since the last consumption? Issues a `WAIT_EVENTS` with `timeout=0`.
    /// Returns `Some(Error::*)` if a fault is pending, `None` otherwise.
    /// Used (a) from `AmdSignal::wait` on a 30 s timeout to attach the real
    /// error to a stalled dispatch and (b) from the WAIT_EVENTS escalation
    /// path to break out of polling early on a fault.
    pub fn poll_faults_nonblocking(&self) -> Option<Error> {
        // Non-blocking poll = `wait_events` with timeout 0. Preserves the
        // pre-refactor contract: ioctl error / no fault → `None`; fault →
        // poison with the bare message + return `Some`.
        match self.iface.wait_events(0) {
            Ok(Some(Error::Runtime { message })) => {
                self.poison(&message);
                Some(Error::Runtime { message })
            }
            _ => None,
        }
    }
}

/// Ensure the process-wide `/dev/kfd` handle is open and return a shared
/// `Arc<OwnedFd>`. Mirrors tinygrad's `KFDIface.kfd` class attribute
/// (`ops_amd.py:725`): all devices in a process share one KFD fd so events
/// are visible across all of them.
pub(crate) fn ensure_global_kfd() -> Result<Arc<OwnedFd>> {
    let mut g = GLOBAL_KFD.lock();
    if let Some(fd) = g.as_ref() {
        return Ok(Arc::clone(fd));
    }
    let fd = Arc::new(open_owned("/dev/kfd")?);
    *g = Some(Arc::clone(&fd));
    Ok(fd)
}

/// Ensure the process-wide event page is allocated, bound, and mapped to
/// `node.gpu_id`. Mirrors tinygrad `ops_amd.py:731-733`:
/// - first device: allocate 0x8000 GTT|COHERENT|UNCACHED|PUBLIC, bind via
///   `CREATE_EVENT(event_page_offset=handle)`, map into the first GPU.
/// - subsequent devices: only `MAP_MEMORY_TO_GPU` the existing page into
///   their `gpu_id` (no re-alloc, no re-bind).
pub(crate) fn ensure_event_page(kfd_fd: &OwnedFd, drm_fd: &OwnedFd, node: &AmdNode) -> Result<EventPageState> {
    let mut g = EVENT_PAGE.lock();
    if let Some(ep) = g.as_ref() {
        // Reuse: map the existing page into this device's GPU page table.
        let mut gpu_id = node.gpu_id;
        let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
            handle: ep.handle,
            device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
            n_devices: 1,
            n_success: 0,
        };
        if let Err(e) = unsafe { ioctl::kfd_map_memory_to_gpu(kfd_fd.as_raw_fd(), &mut map_args as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU(event page reuse)", errno: e as i32 });
        }
        return Ok(*ep);
    }

    let (va, size, handle) = alloc_event_page(kfd_fd, drm_fd, node)?;
    // Bind the page to this KFD process — only on first init.
    let mut bind = kfd::kfd_ioctl_create_event_args { event_page_offset: handle, ..Default::default() };
    if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut bind as *mut _) } {
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(bind page)", errno: e as i32 });
    }
    let ep = EventPageState { handle, va, size };
    *g = Some(ep);
    Ok(ep)
}

/// Allocate the 0x8000-byte event page (GTT-pinned, uncached, host-visible).
/// Returns `(va, size, kfd_handle)`. The handle goes into the
/// `event_page_offset` field of the bind `AMDKFD_IOC_CREATE_EVENT` call
/// (`ops_amd.py:733`).
fn alloc_event_page(kfd_fd: &OwnedFd, drm_fd: &OwnedFd, node: &AmdNode) -> Result<(u64, usize, u64)> {
    use libc::{
        MAP_ANONYMOUS, MAP_FIXED, MAP_NORESERVE, MAP_PRIVATE, MAP_SHARED, PROT_NONE, PROT_READ, PROT_WRITE, mmap,
        munmap,
    };
    let size: usize = 0x8000;
    // SAFETY: standard libc::mmap; PROT_NONE reservation.
    let va = unsafe { mmap(std::ptr::null_mut(), size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
    if va == libc::MAP_FAILED {
        return Err(Error::AmdAllocFailed { reason: "event-page VA reservation failed".into() });
    }
    let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
        va_addr: va as u64,
        size: size as u64,
        gpu_id: node.gpu_id,
        flags: kfd::KFD_IOC_ALLOC_MEM_FLAGS_GTT
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_EXECUTABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_PUBLIC
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_COHERENT
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_UNCACHED,
        ..Default::default()
    };
    // SAFETY: kfd_fd is alive; args type-correct.
    if let Err(e) = unsafe { ioctl::kfd_alloc_memory_of_gpu(kfd_fd.as_raw_fd(), &mut args as *mut _) } {
        unsafe { munmap(va, size) };
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(event page)", errno: e as i32 });
    }
    let handle = args.handle;
    let mmap_offset = args.mmap_offset;

    // Map host-visible via drm_fd at the reserved VA.
    let host = unsafe {
        mmap(va, size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, drm_fd.as_raw_fd(), mmap_offset as i64)
    };
    if host == libc::MAP_FAILED || !std::ptr::eq(host, va) {
        unsafe { munmap(va, size) };
        return Err(Error::AmdAllocFailed { reason: "event-page host mmap failed".into() });
    }

    // Map into the first GPU's page table.
    let mut gpu_id = node.gpu_id;
    let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
        handle,
        device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
        n_devices: 1,
        n_success: 0,
    };
    if let Err(e) = unsafe { ioctl::kfd_map_memory_to_gpu(kfd_fd.as_raw_fd(), &mut map_args as *mut _) } {
        unsafe { munmap(va, size) };
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU(event page)", errno: e as i32 });
    }

    Ok((va as u64, size, handle))
}

/// Allocate a scratch buffer sized for `private_segment_size` bytes per
/// thread and compute the packed `COMPUTE_TMPRING_SIZE` value. Returns
/// `(scratch_gpu_va, scratch_size, tmpring_size, rounded_size_per_thread)`.
/// Mirrors tinygrad's `_ensure_has_local_memory` at `ops_amd.py:1065-1081`.
///
/// Sizing (gfx11/12):
/// - `lanes_per_wave = 64` (scratch lane stride is wave64-aligned per AMD)
/// - `mem_alignment_size = 256`
/// - `size_per_thread = round_up(private_segment_size, 4)` (= 256/64)
/// - `cu_cnt = simd_count / simd_per_cu / xccs`
/// - `size_per_xcc = size_per_thread * lanes_per_wave * max_slots_scratch_cu * cu_cnt`
/// - `total = size_per_xcc * xccs` (page-aligned for KFD)
///
/// `COMPUTE_TMPRING_SIZE` packs `WAVES` (bits 0-11) and `WAVESIZE`
/// (bits 12-26 on gfx11):
/// - `wave_scratch = ceildiv(lanes_per_wave * size_per_thread, 256)`
/// - `num_waves = (size_per_xcc / (wave_scratch * 256)) / se_cnt`
/// - `max_scratch_waves = cu_cnt * max_slots_scratch_cu * xccs`
/// - `WAVES = min(num_waves, max_scratch_waves)`, `WAVESIZE = wave_scratch`
pub(crate) fn alloc_scratch(
    iface: &Arc<dyn crate::amd::iface::AmdIface>,
    node: &AmdNode,
    arch: &AmdArch,
    private_segment_size: u32,
) -> Result<(u64, usize, u32, u32, u64)> {
    const LANES_PER_WAVE: u32 = 64;
    const PAGE: usize = 0x1000;
    // gfx9 (CDNA) scratch is 1024-byte aligned; gfx11/12 (RDNA) use 256.
    let mem_alignment_size: u32 = if arch.is_cdna() { 1024 } else { 256 };

    let xccs = node.num_xcc.max(1);
    let simd_per_cu = node.simd_per_cu.max(1);
    let cu_cnt = ((node.simd_count.max(1) / simd_per_cu) / xccs).max(1);
    let max_slots = node.max_slots_scratch_cu.max(1);
    let se_cnt = (node.array_count.max(1) / node.simd_arrays_per_engine.max(1) / xccs).max(1);

    // Round up to the per-lane alignment stride.
    let size_per_thread = private_segment_size.max(1).next_multiple_of(mem_alignment_size / LANES_PER_WAVE);
    let size_per_xcc =
        (size_per_thread as usize) * (LANES_PER_WAVE as usize) * (max_slots as usize) * (cu_cnt as usize);
    let total = (size_per_xcc * xccs as usize).next_multiple_of(PAGE);

    // KFD alloc as plain VRAM (GPU-only; no host access needed — the GPU writes
    // register spills here and reads them back). Plain VRAM = no EXECUTABLE, no
    // PUBLIC (`cpu_access=false` keeps PUBLIC off); see `AllocKind::DeviceVram`.
    let r = iface.alloc_raw(
        total,
        crate::amd::iface::AllocKind::DeviceVram { executable: false },
        crate::amd::va_registry::AllocTag::Scratch,
        /*cpu_access=*/ false,
        /*zero=*/ false,
    )?;
    let va = r.gpu_va;
    let total = r.size;
    let handle = r.handle;

    // gfx9 divides scratch evenly across SEs (1); gfx11/12 divide by se_cnt.
    let wave_scratch = (LANES_PER_WAVE * size_per_thread).div_ceil(mem_alignment_size);
    let max_scratch_waves = cu_cnt * max_slots * xccs;
    let se_div = if arch.is_cdna() { 1 } else { se_cnt };
    let num_waves = ((size_per_xcc as u32) / (wave_scratch * mem_alignment_size)) / se_div;
    let waves = num_waves.min(max_scratch_waves);
    let tmpring_size = pack_tmpring(waves, wave_scratch, arch);

    Ok((va, total, tmpring_size, size_per_thread, handle))
}

/// Pack `COMPUTE_TMPRING_SIZE`: WAVES in bits 0..12, WAVESIZE at bit 12 with an
/// arch-specific field width — gfx9 13b, gfx11 15b, gfx12 18b.
fn pack_tmpring(waves: u32, wave_scratch: u32, arch: &AmdArch) -> u32 {
    let wavesize_mask: u32 = if arch.is_cdna() {
        0x1FFF
    } else if arch.is_rdna4() {
        0x3FFFF
    } else {
        0x7FFF
    };
    (waves & 0xFFF) | ((wave_scratch & wavesize_mask) << 12)
}

pub(crate) fn open_owned(path: &str) -> Result<OwnedFd> {
    match open(path, OFlag::O_RDWR | OFlag::O_CLOEXEC, Mode::empty()) {
        Ok(fd) => {
            // `nix::fcntl::open` in our pinned version returns a bare `RawFd`;
            // adopt it as an `OwnedFd` so Drop closes it for us.
            let raw = fd_to_raw(fd);
            // SAFETY: nix just opened this fd and transferred ownership to us;
            // no other code can be observing it.
            Ok(unsafe { OwnedFd::from_raw_fd(raw) })
        }
        Err(nix::errno::Errno::ENOENT) | Err(nix::errno::Errno::EACCES) => {
            Err(Error::NoAmdGpu { reason: format!("cannot open {path}") })
        }
        Err(e) => Err(Error::AmdIoctl { ioctl: "open", errno: e as i32 }),
    }
}

/// Extract the raw fd from whatever `nix::fcntl::open` returns. In older nix
/// versions this is `RawFd`; in 0.30+ it's `OwnedFd`. We use a small trait
/// dispatch so the call site stays version-agnostic.
fn fd_to_raw<T: ToRawFdShim>(fd: T) -> RawFd {
    fd.to_raw()
}

trait ToRawFdShim {
    fn to_raw(self) -> RawFd;
}

impl ToRawFdShim for RawFd {
    fn to_raw(self) -> RawFd {
        self
    }
}

impl ToRawFdShim for OwnedFd {
    fn to_raw(self) -> RawFd {
        std::os::fd::IntoRawFd::into_raw_fd(self)
    }
}

#[cfg(test)]
#[path = "../test/unit/amd/device.rs"]
mod tests;
