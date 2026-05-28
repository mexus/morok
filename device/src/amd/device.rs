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
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use svod_dtype::AmdArch;
use tracing::debug;

use crate::amd::signal::AmdSignal;
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
struct EventPageState {
    handle: u64,
    va: u64,
    size: usize,
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
/// `event_page_*` is the GTT-pinned per-process event page (bound via
/// `CREATE_EVENT(event_page_offset=handle)`). `queue_event_*` is the SIGNAL
/// event used by `AMDKFD_IOC_CREATE_QUEUE` for completion notification.
/// `mem_fault_event_id` / `hw_fault_event_id` are MEMORY / HW_EXCEPTION
/// events used by the fault-collection path. `queue_event_mailbox_ptr` is
/// the GPU VA inside the event page where SDMA fence packets write the
/// queue event_id (per tinygrad `ops_amd.py:738`).
#[derive(Debug)]
pub struct AmdDeviceCore {
    pub node: AmdNode,
    pub arch: AmdArch,
    /// Shared `/dev/kfd` handle (one per process; see [`GLOBAL_KFD`]).
    pub kfd_fd: Arc<OwnedFd>,
    pub drm_fd: OwnedFd,
    pub kfd_version: (u32, u32),
    /// VA + size of the GTT-pinned event page (held to keep it mapped).
    pub event_page_va: u64,
    pub event_page_size: usize,
    /// KFD ids for the SIGNAL events used by queue completion / fault paths.
    pub queue_event_id: u32,
    pub queue_event_slot_index: u32,
    pub queue_event_mailbox_ptr: u64,
    pub mem_fault_event_id: u32,
    pub hw_fault_event_id: u32,
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
    /// Registry of every `AmdConnector` built against this core. Weak so
    /// dropped connectors don't keep timelines alive. Used by
    /// [`AmdDeviceCore::synchronize_all`] to drain ALL in-flight GPU work
    /// before destructive host-visible operations
    /// (`AmdAllocator::_copyin`/`_copyout`/`_free`). Each connector has its
    /// own timeline signal — without iterating them, a copy-back from a buffer
    /// written by a per-plan/per-graph connector races the dispatch.
    pub(crate) connectors: parking_lot::Mutex<Vec<Weak<crate::amd::connector::AmdConnector>>>,
    /// Process-global signal pool, allocated once per physical device. Lazily
    /// installed by the device factory and shared across every `AmdConnector`
    /// (timeline signal acquired here at connector construction) — pool access
    /// is rare (slot alloc on connector build), and one pool covers many
    /// connectors at 4 KiB total VRAM.
    signal_pool: OnceLock<Arc<crate::amd::signal::SignalPool>>,
}

/// Open handle to one AMD GPU node.
///
/// Holds the immutable `AmdDeviceCore` plus a default `AmdConnector` used
/// by trait-fallback callers (`Program::execute` → `benchmark_kernel` etc.)
/// and by the device-wide synchronize chain (`AmdAllocator::_copyin`/
/// `_copyout`/`_free` route through `dev.synchronize() →
/// core.synchronize_all()` which drains EVERY connector — default + per-
/// plan + per-graph). Plan and graph callers build their own connector
/// via `AmdConnector::new_with_resources` and bypass the default; the
/// default connector is never on their hot path.
///
/// Immutable Core fields stay reachable via [`Deref`] — `self.dev.node`,
/// `self.dev.kfd_fd`, `self.dev.poison_error()`, etc.
#[derive(Debug)]
pub struct AmdDevice {
    /// Immutable identity (cloneable across connectors).
    core: Arc<AmdDeviceCore>,
    /// Default per-device connector — owns its own KFD ring + kernarg arena +
    /// scratch + timeline. Lazy because `AmdConnector::new_with_resources`
    /// needs an allocator, and the allocator needs an `AmdDevice` (DEVICE_CACHE
    /// path); the factory installs this after `AmdDevice::open` returns.
    /// Routed through by:
    /// - `Program::execute` trait fallback (`AmdProgram::execute`) — when a
    ///   caller dispatches a program without going through an
    ///   `ExecutionPlan`/`AmdGraph` (e.g. `benchmark_kernel`).
    /// - `AmdAllocator::_copyin`/`_copyout`/`_free` device-wide synchronize.
    connector: OnceLock<Arc<crate::amd::connector::AmdConnector>>,
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

        let kfd_fd = ensure_global_kfd()?;
        let drm_path = format!("/dev/dri/renderD{}", node.drm_render_minor);
        let drm_fd = open_owned(drm_path.as_str())?;

        // GET_VERSION captures the KFD ABI version so we can gate RUNTIME_ENABLE
        // (which only exists on kfd >= 1.14). Mirrors tinygrad `ops_amd.py:726`.
        let mut ver_args = kfd::kfd_ioctl_get_version_args { major_version: 0, minor_version: 0 };
        if let Err(e) = unsafe { ioctl::kfd_get_version(kfd_fd.as_raw_fd(), &mut ver_args as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_GET_VERSION", errno: e as i32 });
        }
        let kfd_version = (ver_args.major_version, ver_args.minor_version);

        // ACQUIRE_VM tells KFD to register this DRM fd as the process's VM
        // for this GPU. Required before any alloc/map ioctls.
        let mut args = kfd::kfd_ioctl_acquire_vm_args { drm_fd: drm_fd.as_raw_fd() as u32, gpu_id: node.gpu_id };
        // SAFETY: kfd_fd is a valid open fd; `args` is a well-typed ioctl
        // argument matching the AMDKFD_IOC_ACQUIRE_VM signature.
        let rc = unsafe { ioctl::kfd_acquire_vm(kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_ACQUIRE_VM", errno: e as i32 });
        }

        // RUNTIME_ENABLE — only on KFD >= 1.14. Tinygrad gates it the same way
        // (`ops_amd.py:728`); older kernels reject the ioctl with ENOTTY.
        if kfd_version >= (1, 14) {
            let mut rt = kfd::kfd_ioctl_runtime_enable_args { mode_mask: 0, ..Default::default() };
            if let Err(e) = unsafe { ioctl::kfd_runtime_enable(kfd_fd.as_raw_fd(), &mut rt as *mut _) } {
                return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_RUNTIME_ENABLE", errno: e as i32 });
            }
        }

        // Event-page setup. Mirrors `ops_amd.py:731-733`: allocated and bound
        // exactly once per process; subsequent devices reuse it by calling
        // `MAP_MEMORY_TO_GPU` for their `gpu_id`. Without the bound event page,
        // AMDKFD_IOC_CREATE_QUEUE returns EINVAL.
        let EventPageState { handle: _, va: event_page_va, size: event_page_size } =
            ensure_event_page(&kfd_fd, &drm_fd, &node)?;
        let mut qe = kfd::kfd_ioctl_create_event_args {
            event_type: kfd::KFD_IOC_EVENT_SIGNAL,
            auto_reset: 1,
            ..Default::default()
        };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut qe as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(queue signal)", errno: e as i32 });
        }
        let mut mem_event =
            kfd::kfd_ioctl_create_event_args { event_type: kfd::KFD_IOC_EVENT_MEMORY, ..Default::default() };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut mem_event as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(mem fault)", errno: e as i32 });
        }
        let mut hw_event =
            kfd::kfd_ioctl_create_event_args { event_type: kfd::KFD_IOC_EVENT_HW_EXCEPTION, ..Default::default() };
        if let Err(e) = unsafe { ioctl::kfd_create_event(kfd_fd.as_raw_fd(), &mut hw_event as *mut _) } {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_CREATE_EVENT(hw fault)", errno: e as i32 });
        }

        // The mailbox sits at event_page + slot_index * 8 (tinygrad
        // `ops_amd.py:738`). SDMA fence packets write the queue event_id
        // here to wake up `WAIT_EVENTS` from `sleep()`.
        let queue_event_mailbox_ptr = event_page_va + (qe.event_slot_index as u64) * 8;

        debug!(
            node = node.node_id,
            gpu_id = node.gpu_id,
            arch = arch.mcpu(),
            kfd_version = ?kfd_version,
            queue_event_id = qe.event_id,
            mem_fault_event_id = mem_event.event_id,
            hw_fault_event_id = hw_event.event_id,
            "AmdDevice opened"
        );

        let core = Arc::new(AmdDeviceCore {
            node,
            arch,
            kfd_fd,
            drm_fd,
            kfd_version,
            event_page_va,
            event_page_size,
            queue_event_id: qe.event_id,
            queue_event_slot_index: qe.event_slot_index,
            queue_event_mailbox_ptr,
            mem_fault_event_id: mem_event.event_id,
            hw_fault_event_id: hw_event.event_id,
            has_sdma_queue: AtomicBool::new(false),
            poisoned: AtomicBool::new(false),
            error_msg: OnceLock::new(),
            connectors: parking_lot::Mutex::new(Vec::new()),
            signal_pool: OnceLock::new(),
        });
        // Default connector is installed by the factory AFTER this returns —
        // building it here would recursively call `AmdAllocator::new`, which
        // calls `AmdDevice::open`. The lazy `OnceLock` breaks the cycle.
        Ok(Arc::new(Self { core, connector: OnceLock::new() }))
    }

    /// Install the default connector. Called once per device by the factory
    /// after the allocator + signal pool are constructed; subsequent calls
    /// are a no-op.
    pub fn install_default_connector(&self, conn: Arc<crate::amd::connector::AmdConnector>) {
        let _ = self.connector.set(conn);
    }

    /// Borrow the shared immutable core. Used by Step 3+ to build per-owner
    /// `AmdConnector`s against the same physical device without re-acquiring
    /// KFD.
    #[inline]
    pub fn core(&self) -> &Arc<AmdDeviceCore> {
        &self.core
    }

    /// Borrow the device's default connector. Panics if the factory hasn't
    /// installed it yet — that's a wiring bug, not a runtime condition.
    /// Used by `Program::execute` trait fallback (callers who don't go
    /// through `ExecutionPlan::execute_on`) and by the device-wide
    /// synchronize chain in `AmdAllocator`.
    #[inline]
    pub fn connector(&self) -> &Arc<crate::amd::connector::AmdConnector> {
        self.connector.get().expect("AmdDevice default connector not installed; factory wiring bug")
    }

    // === Delegations to the default connector ===
    // Back-compat surface for `AmdAllocator::_copyin`/`_copyout`/`_free` and
    // direct `Program::execute` callers. After the factory installs the
    // default connector these are pure forwarding methods.

    /// Current scratch buffer GPU VA on the default connector.
    pub fn scratch_gpu_va(&self) -> u64 {
        self.connector().scratch_gpu_va()
    }

    /// Packed `COMPUTE_TMPRING_SIZE` on the default connector.
    pub fn tmpring_size(&self) -> u32 {
        self.connector().tmpring_size()
    }

    /// Grow the default connector's scratch backing (delegate).
    pub fn ensure_has_local_memory(&self, private_segment_size: u32) -> Result<()> {
        self.connector().ensure_has_local_memory(private_segment_size)
    }

    /// Default connector timeline signal (delegate; panics if not initialized).
    pub fn timeline_signal(&self) -> &Arc<AmdSignal> {
        self.connector().timeline_signal()
    }

    /// Reserve the next timeline value on the default connector.
    pub fn next_timeline(&self) -> u64 {
        self.connector().next_timeline()
    }

    /// Highest submitted timeline value on the default connector.
    pub fn timeline_value(&self) -> u64 {
        self.connector().timeline_value()
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
    /// Drain every connector currently backed by this core. Iterates the
    /// `connectors` registry (Weak refs), upgrades each, and synchronises in
    /// turn. Cheap when no per-owner connectors exist (the default connector
    /// is registered the same way). Fast on idle connectors — each is a
    /// no-op when its timeline is at value 0.
    pub fn synchronize_all(&self) -> Result<()> {
        // Snapshot strong refs to release the registry lock before doing
        // potentially multi-second waits. The snapshot also keeps every
        // connector alive until we've drained it, so a concurrent drop
        // can't pull the rug out mid-iteration.
        let live: Vec<Arc<crate::amd::connector::AmdConnector>> =
            self.connectors.lock().iter().filter_map(|w| w.upgrade()).collect();
        // Drain every connector; collect the first error but keep going so
        // a single stuck connector doesn't strand buffer-frees on the others.
        let mut first_err: Option<Error> = None;
        for c in live {
            if let Err(e) = c.synchronize() {
                tracing::warn!(?e, "synchronize_all: connector drain failed; continuing");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        // Opportunistic GC of `Drop`ped connector entries. The registry is
        // touched here on every host read/free, so dead Weaks don't
        // accumulate indefinitely between connector-creation events.
        self.connectors.lock().retain(|w| w.strong_count() > 0);
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Borrow the process-global signal pool (lazy-installed by the device
    /// factory). `None` before the factory has run; once initialized, every
    /// connector built against this core shares it.
    pub fn signal_pool(&self) -> Option<&Arc<crate::amd::signal::SignalPool>> {
        self.signal_pool.get()
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
        let mut events: [kfd::kfd_event_data; 3] = [Default::default(); 3];
        events[0].event_id = self.queue_event_id;
        events[1].event_id = self.mem_fault_event_id;
        events[2].event_id = self.hw_fault_event_id;
        let mut args = kfd::kfd_ioctl_wait_events_args {
            events_ptr: events.as_mut_ptr() as u64,
            num_events: events.len() as u32,
            wait_for_all: 0,
            timeout: timeout_ms,
            wait_result: 0,
        };
        // SAFETY: kfd_fd is alive; args + events live for the duration of the call.
        let rc = unsafe { ioctl::kfd_wait_events(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if let Err(e) = rc {
            return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: e as i32 });
        }
        Ok(self.poll_faults_nonblocking())
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
        let mut events: [kfd::kfd_event_data; 2] = [Default::default(); 2];
        events[0].event_id = self.mem_fault_event_id;
        events[1].event_id = self.hw_fault_event_id;
        let mut args = kfd::kfd_ioctl_wait_events_args {
            events_ptr: events.as_mut_ptr() as u64,
            num_events: events.len() as u32,
            wait_for_all: 0,
            timeout: 0,
            wait_result: 0,
        };
        // SAFETY: kfd_fd is alive; args + events live for the duration of the call.
        let rc = unsafe { ioctl::kfd_wait_events(self.kfd_fd.as_raw_fd(), &mut args as *mut _) };
        if rc.is_err() {
            // EAGAIN / ETIMEDOUT = no fault pending; treat as "no fault".
            return None;
        }
        // Inspect each event's union payload. `gpu_id != 0` signals the
        // fault was actually written by KFD (the union is zero-initialized
        // when nothing fired).
        // SAFETY: bindgen union access — we read whichever payload type
        // matches the event we registered.
        let mem = unsafe { events[0].__bindgen_anon_1.memory_exception_data };
        if mem.gpu_id != 0 {
            let msg = format!(
                "AMD GPU memory fault on gpu_id={} va={:#x} (NotPresent={} ReadOnly={} NoExecute={} ErrorType={})",
                mem.gpu_id,
                { mem.va },
                mem.failure.NotPresent,
                mem.failure.ReadOnly,
                mem.failure.NoExecute,
                { mem.ErrorType },
            );
            self.poison(&msg);
            return Some(Error::Runtime { message: msg });
        }
        let hw = unsafe { events[1].__bindgen_anon_1.hw_exception_data };
        if hw.gpu_id != 0 {
            let msg = format!(
                "AMD GPU hardware exception on gpu_id={} reset_type={} reset_cause={} memory_lost={}",
                hw.gpu_id, hw.reset_type, hw.reset_cause, hw.memory_lost,
            );
            self.poison(&msg);
            return Some(Error::Runtime { message: msg });
        }
        None
    }
}

/// Ensure the process-wide `/dev/kfd` handle is open and return a shared
/// `Arc<OwnedFd>`. Mirrors tinygrad's `KFDIface.kfd` class attribute
/// (`ops_amd.py:725`): all devices in a process share one KFD fd so events
/// are visible across all of them.
fn ensure_global_kfd() -> Result<Arc<OwnedFd>> {
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
fn ensure_event_page(kfd_fd: &OwnedFd, drm_fd: &OwnedFd, node: &AmdNode) -> Result<EventPageState> {
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
    kfd_fd: &OwnedFd,
    node: &AmdNode,
    arch: &AmdArch,
    private_segment_size: u32,
) -> Result<(u64, usize, u32, u32, u64)> {
    use libc::{MAP_ANONYMOUS, MAP_NORESERVE, MAP_PRIVATE, PROT_NONE, mmap, munmap};

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

    // Reserve VA + KFD alloc as plain VRAM (GPU-only; no host access needed —
    // the GPU writes register spills here and reads them back).
    // SAFETY: standard libc::mmap; PROT_NONE reservation.
    let va =
        unsafe { mmap(std::ptr::null_mut(), total, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0) };
    if va == libc::MAP_FAILED {
        return Err(Error::AmdAllocFailed { reason: "scratch VA reservation failed".into() });
    }
    let mut args = kfd::kfd_ioctl_alloc_memory_of_gpu_args {
        va_addr: va as u64,
        size: total as u64,
        gpu_id: node.gpu_id,
        flags: kfd::KFD_IOC_ALLOC_MEM_FLAGS_VRAM
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_WRITABLE
            | kfd::KFD_IOC_ALLOC_MEM_FLAGS_NO_SUBSTITUTE,
        ..Default::default()
    };
    // SAFETY: kfd_fd is alive; args type-correct.
    if let Err(e) = unsafe { ioctl::kfd_alloc_memory_of_gpu(kfd_fd.as_raw_fd(), &mut args as *mut _) } {
        unsafe { munmap(va, total) };
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_ALLOC_MEMORY_OF_GPU(scratch)", errno: e as i32 });
    }
    let handle = args.handle;

    let mut gpu_id = node.gpu_id;
    let mut map_args = kfd::kfd_ioctl_map_memory_to_gpu_args {
        handle,
        device_ids_array_ptr: &mut gpu_id as *mut _ as u64,
        n_devices: 1,
        n_success: 0,
    };
    if let Err(e) = unsafe { ioctl::kfd_map_memory_to_gpu(kfd_fd.as_raw_fd(), &mut map_args as *mut _) } {
        unsafe { munmap(va, total) };
        return Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_MAP_MEMORY_TO_GPU(scratch)", errno: e as i32 });
    }

    // gfx9 divides scratch evenly across SEs (1); gfx11/12 divide by se_cnt.
    let wave_scratch = (LANES_PER_WAVE * size_per_thread).div_ceil(mem_alignment_size);
    let max_scratch_waves = cu_cnt * max_slots * xccs;
    let se_div = if arch.is_cdna() { 1 } else { se_cnt };
    let num_waves = ((size_per_xcc as u32) / (wave_scratch * mem_alignment_size)) / se_div;
    let waves = num_waves.min(max_scratch_waves);
    let tmpring_size = pack_tmpring(waves, wave_scratch, arch);

    Ok((va as u64, total, tmpring_size, size_per_thread, handle))
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

fn open_owned(path: &str) -> Result<OwnedFd> {
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
mod tests {
    use super::*;

    /// On hosts without `/dev/kfd` (or without a supported GPU), `open` must
    /// surface a clean `Err` — never panic.
    #[test]
    fn open_without_gpu_or_unsupported_arch_is_clean_err() {
        let result = AmdDevice::open(0);
        match result {
            Ok(_) => {
                // Host has a supported AMD GPU — exercise the happy path too.
                // (We can't assert much without hardware-specific data.)
            }
            Err(Error::NoAmdGpu { .. }) | Err(Error::AmdAllocFailed { .. }) | Err(Error::AmdIoctl { .. }) => {
                // All acceptable.
            }
            Err(e) => panic!("unexpected error variant: {e:?}"),
        }
    }

    #[test]
    fn pack_tmpring_wavesize_width_by_arch() {
        // wave_scratch=0x3FFFF: cdna(13b) truncates, rdna3(15b) truncates, rdna4(18b) keeps it.
        assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx942) >> 12, 0x1FFF);
        assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1100) >> 12, 0x7FFF);
        assert_eq!(pack_tmpring(1, 0x3FFFF, &AmdArch::Gfx1200) >> 12, 0x3FFFF);
        assert_eq!(pack_tmpring(0xABC, 0, &AmdArch::Gfx1100) & 0xFFF, 0xABC);
    }
}
