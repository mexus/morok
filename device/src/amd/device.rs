//! `AmdDevice`: KFD-direct device handle.
//!
//! Phase 3 deliverable — opens `/dev/kfd` and `/dev/dri/renderD*`, parses
//! topology, calls `AMDKFD_IOC_ACQUIRE_VM`. Queues/signals/kernarg arena are
//! filled in by Phase 5/6.

#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

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
/// mutex on `AmdDevice` so [`AmdDevice::ensure_has_local_memory`] can grow
/// the scratch buffer when a freshly-loaded program demands more bytes per
/// thread than what's currently allocated.
#[derive(Debug, Clone, Copy)]
struct ScratchState {
    /// GPU VA of the current scratch buffer.
    gpu_va: u64,
    /// Bytes per thread (rounded up to 4-byte slot stride for wave64).
    /// Equivalent to tinygrad's `max_private_segment_size` (ops_amd.py:1066).
    size_per_thread: u32,
    /// Packed `COMPUTE_TMPRING_SIZE` register value.
    tmpring_size: u32,
}

/// Open handle to one AMD GPU node.
///
/// Holds the KFD and DRM render fds plus parsed topology. Constructed via
/// [`AmdDevice::open`]; callers receive an `Arc<AmdDevice>` so the same
/// device handle can be shared by the allocator, queues, signals, and
/// per-Program state without repeated KFD opens.
///
/// `event_page_*` is the GTT-pinned per-process event page (bound via
/// `CREATE_EVENT(event_page_offset=handle)`). `queue_event_*` is the SIGNAL
/// event used by `AMDKFD_IOC_CREATE_QUEUE` for completion notification.
/// `mem_fault_event_id` / `hw_fault_event_id` are MEMORY / HW_EXCEPTION
/// events used by the fault-collection path. `queue_event_mailbox_ptr` is
/// the GPU VA inside the event page where SDMA fence packets write the
/// queue event_id (per tinygrad `ops_amd.py:738`).
#[derive(Debug)]
pub struct AmdDevice {
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
    /// Per-device scratch buffer + sizing. Initialized at open with
    /// `private_segment_size = 128` (matches tinygrad's
    /// `_ensure_has_local_memory(128)` at `ops_amd.py:1010`). Grown via
    /// [`AmdDevice::ensure_has_local_memory`] when a program load reports a
    /// larger `private_segment_fixed_size` — mirrors tinygrad
    /// `ops_amd.py:589-590, 1065-1071`. Mutex because growth happens during
    /// program load (not on the dispatch hot path); reads on the hot path
    /// take the lock briefly to copy the (va, tmpring_size) pair.
    ///
    /// Even kernels with `SCRATCH_EN=0` need valid
    /// `COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI` for wave init on RDNA3+,
    /// otherwise wave launch faults at random addresses. The old allocation
    /// after a grow is intentionally leaked — AmdDevice lives in
    /// `DEVICE_CACHE` for the whole process lifetime, so KFD frees it on
    /// process exit.
    scratch_state: Mutex<ScratchState>,
    /// Device-global timeline signal. Every kernel/copy submit waits on the
    /// previous timeline value and signals the next one; `synchronize()`
    /// drains the queue by waiting until the GPU writes `timeline_value - 1`.
    /// Mirrors tinygrad `HCQCompiled.timeline_signal` (`hcq.py:415`).
    ///
    /// Lazy-initialized via [`AmdDevice::init_timeline`] from the factory so
    /// the `SignalPool` (which depends on an `AmdAllocator`, which depends on
    /// `Arc<AmdDevice>`) can be wired up after `open()`.
    timeline_signal: OnceLock<Arc<AmdSignal>>,
    /// Highest timeline value submitted so far (next submit will use this and
    /// then increment). Mirrors `HCQCompiled.timeline_value` (`hcq.py:405`).
    timeline_value: AtomicU64,
    /// Poison latch. Once a GPU fault/timeout is observed, the device is dead:
    /// every `synchronize`/`execute` fails fast. Mirrors tinygrad's
    /// `error_state` (`hcq.py:421`). `AtomicBool` so the per-dispatch hot path
    /// gate is a single Relaxed load; the message is written once on latch.
    poisoned: AtomicBool,
    error_msg: OnceLock<String>,
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

        // Default scratch buffer — sized for 128 B/thread per tinygrad's
        // `_ensure_has_local_memory(128)` at `ops_amd.py:1010`. Kernels with
        // private_segment_fixed_size == 0 still need this set up: on RDNA3+
        // the wave-init reads COMPUTE_DISPATCH_SCRATCH_BASE_LO/HI regardless
        // of SCRATCH_EN, and an invalid base produces an MMU fault at the
        // first dispatch. Programs that need more bytes/thread call
        // `ensure_has_local_memory` at load time to grow the backing.
        let (scratch_va, scratch_size, tmpring_size, size_per_thread) = alloc_scratch(&kfd_fd, &node, &arch, 128)?;

        debug!(
            node = node.node_id,
            gpu_id = node.gpu_id,
            arch = arch.mcpu(),
            kfd_version = ?kfd_version,
            queue_event_id = qe.event_id,
            mem_fault_event_id = mem_event.event_id,
            hw_fault_event_id = hw_event.event_id,
            scratch_va = scratch_va,
            scratch_size = scratch_size,
            tmpring_size = tmpring_size,
            "AmdDevice opened"
        );

        Ok(Arc::new(Self {
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
            scratch_state: Mutex::new(ScratchState { gpu_va: scratch_va, size_per_thread, tmpring_size }),
            timeline_signal: OnceLock::new(),
            timeline_value: AtomicU64::new(1),
            poisoned: AtomicBool::new(false),
            error_msg: OnceLock::new(),
        }))
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

    /// Install the device-global timeline signal. Called exactly once from
    /// the device factory after the `SignalPool` is created. Mirrors
    /// tinygrad's `HCQCompiled.__init__` line `hcq.py:415`. Subsequent calls
    /// are a no-op (the existing signal stays).
    pub fn init_timeline(&self, signal: Arc<AmdSignal>) {
        let _ = self.timeline_signal.set(signal);
    }

    /// Current scratch buffer GPU VA (set by [`ensure_has_local_memory`]).
    /// Read under the scratch mutex; the call is on the dispatch hot path so
    /// the lock window is intentionally tiny.
    pub fn scratch_gpu_va(&self) -> u64 {
        self.scratch_state.lock().gpu_va
    }

    /// Packed `COMPUTE_TMPRING_SIZE` for the current scratch buffer.
    pub fn tmpring_size(&self) -> u32 {
        self.scratch_state.lock().tmpring_size
    }

    /// Ensure the device's scratch backing has at least `private_segment_size`
    /// bytes per thread. Mirrors tinygrad's `_ensure_has_local_memory` at
    /// `ops_amd.py:1065-1081` — when a program load reports a larger
    /// `private_segment_fixed_size` than the current backing supports, allocate
    /// a fresh scratch buffer with the larger size and update the device's
    /// (gpu_va, tmpring_size) atomically.
    ///
    /// The old scratch buffer is intentionally leaked: KFD reclaims it on
    /// process exit, and synchronizing the timeline to safely free it would
    /// stall every concurrent dispatch. Tinygrad uses `_realloc` (which DOES
    /// free) but our buffers are larger / fewer growth events expected; the
    /// leak is bounded by the largest kernel's scratch size and lives until
    /// process exit.
    pub fn ensure_has_local_memory(&self, private_segment_size: u32) -> Result<()> {
        let current = self.scratch_state.lock().size_per_thread;
        if private_segment_size <= current {
            return Ok(());
        }
        let (va, _size, tmpring, rounded) = alloc_scratch(&self.kfd_fd, &self.node, &self.arch, private_segment_size)?;
        let mut state = self.scratch_state.lock();
        // Re-check under lock — another caller might have grown to a larger
        // size between our check and this allocation. In that case keep the
        // larger backing.
        if rounded > state.size_per_thread {
            state.gpu_va = va;
            state.size_per_thread = rounded;
            state.tmpring_size = tmpring;
        }
        Ok(())
    }

    /// Device-global timeline signal (panics if [`init_timeline`] hasn't been
    /// called — that's a programming error in the device factory wiring, not a
    /// runtime condition).
    pub fn timeline_signal(&self) -> &Arc<AmdSignal> {
        self.timeline_signal.get().expect("timeline_signal not initialized; call init_timeline from factory")
    }

    /// Reserve the next timeline value (atomic `fetch_add(1)`). The caller
    /// emits a queue signal packet that writes this value to the device
    /// timeline signal slot. Mirrors `HCQCompiled.next_timeline` (`hcq.py:447`).
    pub fn next_timeline(&self) -> u64 {
        self.timeline_value.fetch_add(1, Ordering::AcqRel)
    }

    /// Highest submitted timeline value (i.e. the value the next `signal`
    /// packet would write). `synchronize` waits until the GPU has written
    /// `value - 1`.
    pub fn timeline_value(&self) -> u64 {
        self.timeline_value.load(Ordering::Acquire)
    }

    /// Drain all submitted GPU work on this device. Blocks until the device
    /// timeline signal observes `timeline_value() - 1`.
    ///
    /// Tinygrad calls this from `_free` (`hcq.py:566`) so a buffer's KFD
    /// unmap can't race with an in-flight kernel still holding its VA. Also
    /// safe to call before externally-visible side effects (e.g. host reads
    /// of GPU-written output buffers) for which `AmdProgram::execute` no
    /// longer blocks per-dispatch.
    pub fn synchronize(&self) -> Result<()> {
        // Once latched, the device is dead — re-raise the recorded fault.
        if let Some(err) = self.poison_error() {
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
        signal.wait_signal_value(target, 30_000).inspect_err(|e| self.poison(&e.to_string()))
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
fn alloc_scratch(
    kfd_fd: &OwnedFd,
    node: &AmdNode,
    arch: &AmdArch,
    private_segment_size: u32,
) -> Result<(u64, usize, u32, u32)> {
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

    Ok((va as u64, total, tmpring_size, size_per_thread))
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
