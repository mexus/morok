//! Shared gating helpers for the AMD hardware tests.
//!
//! Sibling modules under `crate::test::unit::amd` reach these via
//! `super::test_support`. Centralising the probe keeps the per-test boilerplate
//! to a single `let … else { return }` and gives every hardware test identical
//! skip semantics on hosts without a supported GPU.

use crate::amd::AmdAllocator;

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{HashMap, HashSet, VecDeque};
use std::ptr::NonNull;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::amd::device::AmdDevice;
use crate::amd::iface::{AllocKind, AllocResult, AmdIface, PublicationStage, QueueHandle, QueueTeardown, RingDesc};
use crate::amd::va_registry::AllocTag;
use crate::error::{Error, Result};

const MOCK_ALIGNMENT: usize = 0x1000;
const MOCK_DOORBELL_BYTES: usize = 0x2000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MockAmdCall {
    Alloc { size: usize, cpu_access: bool, zero: bool },
    Free { gpu_va: u64, size: usize, handle: u64 },
    SetupRing { ring_size: usize, queue_type: u32 },
    TeardownRing { queue_id: u32 },
    WaitEvents { timeout_ms: u32 },
    PublicationCheckpoint { stage: PublicationStage },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum MockFreeIssue {
    DoubleFree { handle: u64 },
    UnknownFree { gpu_va: u64, size: usize, handle: u64 },
}

struct MockAllocation {
    ptr: NonNull<u8>,
    layout: Layout,
}

// SAFETY: the allocation has no thread affinity and access to its lifetime is
// serialized through MockAmdIface's state mutex.
unsafe impl Send for MockAllocation {}

impl MockAllocation {
    fn new(size: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, MOCK_ALIGNMENT)
            .map_err(|_| Error::AmdAllocFailed { reason: format!("invalid mock allocation layout: {size}") })?;
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| Error::AmdAllocFailed { reason: format!("mock host allocation failed: {size}") })?;
        Ok(Self { ptr, layout })
    }
}

impl Drop for MockAllocation {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr.as_ptr(), self.layout) };
    }
}

struct MockLiveAllocation {
    _memory: MockAllocation,
    gpu_va: u64,
    size: usize,
}

struct MockQueue {
    doorbell: MockAllocation,
}

enum MockPublicationOutcome {
    Return(Result<()>),
    Panic,
}

#[derive(Default)]
struct MockAmdState {
    next_handle: u64,
    next_queue_id: u32,
    allocations: usize,
    frees: usize,
    queue_setups: usize,
    queue_teardowns: usize,
    live: HashMap<u64, MockLiveAllocation>,
    freed: HashSet<u64>,
    queues: HashMap<u32, MockQueue>,
    alloc_script: VecDeque<Result<()>>,
    setup_script: VecDeque<Result<()>>,
    teardown_script: VecDeque<Result<QueueTeardown>>,
    wait_script: VecDeque<Result<Option<Error>>>,
    publication_script: VecDeque<MockPublicationOutcome>,
    alloc_tags: Vec<AllocTag>,
    transcript: Vec<MockAmdCall>,
    free_issues: Vec<MockFreeIssue>,
}

/// Host-only AMD backend with page-aligned, stable allocations and explicit
/// lifecycle accounting. Scripted outcomes are consumed FIFO; an empty script
/// uses the successful default for that operation.
#[derive(Default)]
pub(crate) struct MockAmdIface {
    state: Mutex<MockAmdState>,
}

impl std::fmt::Debug for MockAmdIface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock();
        f.debug_struct("MockAmdIface")
            .field("allocations", &state.allocations)
            .field("frees", &state.frees)
            .field("live_handles", &state.live.len())
            .field("live_queues", &state.queues.len())
            .finish()
    }
}

impl MockAmdIface {
    pub(crate) fn device(self: &Arc<Self>) -> Arc<AmdDevice> {
        AmdDevice::synthetic(Arc::clone(self) as Arc<dyn AmdIface>)
    }

    pub(crate) fn script_alloc(&self, outcome: Result<()>) {
        self.state.lock().alloc_script.push_back(outcome);
    }

    pub(crate) fn script_setup(&self, outcome: Result<()>) {
        self.state.lock().setup_script.push_back(outcome);
    }

    pub(crate) fn script_teardown(&self, outcome: Result<QueueTeardown>) {
        self.state.lock().teardown_script.push_back(outcome);
    }

    pub(crate) fn script_wait(&self, outcome: Result<Option<Error>>) {
        self.state.lock().wait_script.push_back(outcome);
    }

    pub(crate) fn script_publication(&self, outcome: Result<()>) {
        self.state.lock().publication_script.push_back(MockPublicationOutcome::Return(outcome));
    }

    pub(crate) fn script_publication_panic(&self) {
        self.state.lock().publication_script.push_back(MockPublicationOutcome::Panic);
    }

    pub(crate) fn allocation_count(&self) -> usize {
        self.state.lock().allocations
    }

    pub(crate) fn free_count(&self) -> usize {
        self.state.lock().frees
    }

    /// Successful allocations recorded for one [`AllocTag`], live or freed.
    pub(crate) fn alloc_count_for_tag(&self, tag: AllocTag) -> usize {
        self.state.lock().alloc_tags.iter().filter(|recorded| **recorded == tag).count()
    }

    pub(crate) fn live_handle_count(&self) -> usize {
        self.state.lock().live.len()
    }

    pub(crate) fn queue_setup_count(&self) -> usize {
        self.state.lock().queue_setups
    }

    pub(crate) fn queue_teardown_count(&self) -> usize {
        self.state.lock().queue_teardowns
    }

    pub(crate) fn live_queue_count(&self) -> usize {
        self.state.lock().queues.len()
    }

    pub(crate) fn transcript(&self) -> Vec<MockAmdCall> {
        self.state.lock().transcript.clone()
    }

    pub(crate) fn free_issues(&self) -> Vec<MockFreeIssue> {
        self.state.lock().free_issues.clone()
    }
}

impl AmdIface for MockAmdIface {
    fn alloc_raw(
        &self,
        size: usize,
        _kind: AllocKind,
        tag: AllocTag,
        cpu_access: bool,
        zero: bool,
    ) -> Result<AllocResult> {
        let size = size.max(1).next_multiple_of(MOCK_ALIGNMENT);
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::Alloc { size, cpu_access, zero });
        if let Some(outcome) = state.alloc_script.pop_front() {
            outcome?;
        }
        if zero && !cpu_access {
            return Err(Error::AmdAllocFailed { reason: "mock cannot zero a device-only allocation".into() });
        }
        let memory = MockAllocation::new(size)?;
        let gpu_va = memory.ptr.as_ptr() as u64;
        let host_ptr = cpu_access.then_some(memory.ptr);
        state.next_handle = state.next_handle.max(1);
        let handle = state.next_handle;
        state.next_handle += 1;
        let previous = state.live.insert(handle, MockLiveAllocation { _memory: memory, gpu_va, size });
        debug_assert!(previous.is_none());
        state.allocations += 1;
        state.alloc_tags.push(tag);
        Ok(AllocResult { gpu_va, host_ptr, handle, size })
    }

    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64) {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::Free { gpu_va, size, handle });
        let Some(allocation) = state.live.remove(&handle) else {
            let issue = if state.freed.contains(&handle) {
                MockFreeIssue::DoubleFree { handle }
            } else {
                MockFreeIssue::UnknownFree { gpu_va, size, handle }
            };
            state.free_issues.push(issue);
            return;
        };
        if allocation.gpu_va != gpu_va || allocation.size != size {
            state.live.insert(handle, allocation);
            state.free_issues.push(MockFreeIssue::UnknownFree { gpu_va, size, handle });
            return;
        }
        // Removing and dropping the backing allocation invalidates the address
        // only after the matching free has been recorded.
        drop(allocation);
        state.freed.insert(handle);
        state.frees += 1;
    }

    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle> {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::SetupRing { ring_size: desc.ring_size, queue_type: desc.queue_type });
        if let Some(outcome) = state.setup_script.pop_front() {
            outcome?;
        }
        let doorbell = MockAllocation::new(MOCK_DOORBELL_BYTES)?;
        let doorbell_base = doorbell.ptr;
        let doorbell_ptr = doorbell.ptr.cast::<u64>();
        state.next_queue_id = state.next_queue_id.max(1);
        let queue_id = state.next_queue_id;
        state.next_queue_id += 1;
        state.queues.insert(queue_id, MockQueue { doorbell });
        state.queue_setups += 1;
        Ok(QueueHandle { queue_id, doorbell_base, doorbell: doorbell_ptr })
    }

    fn teardown_ring(&self, queue_id: u32, doorbell_base: NonNull<u8>) -> Result<QueueTeardown> {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::TeardownRing { queue_id });
        if let Some(outcome) = state.teardown_script.pop_front() {
            let outcome = outcome?;
            let queue = state.queues.remove(&queue_id).ok_or_else(|| Error::AmdQueueStillActive {
                queue_id,
                cause: "mock teardown of unknown queue".into(),
            })?;
            if queue.doorbell.ptr != doorbell_base {
                state.queues.insert(queue_id, queue);
                return Err(Error::AmdQueueStillActive { queue_id, cause: "mock doorbell mismatch".into() });
            }
            state.queue_teardowns += 1;
            return Ok(outcome);
        }
        let queue = state
            .queues
            .remove(&queue_id)
            .ok_or_else(|| Error::AmdQueueStillActive { queue_id, cause: "mock teardown of unknown queue".into() })?;
        if queue.doorbell.ptr != doorbell_base {
            state.queues.insert(queue_id, queue);
            return Err(Error::AmdQueueStillActive { queue_id, cause: "mock doorbell mismatch".into() });
        }
        state.queue_teardowns += 1;
        Ok(QueueTeardown::Complete)
    }

    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>> {
        let scripted = {
            let mut state = self.state.lock();
            state.transcript.push(MockAmdCall::WaitEvents { timeout_ms });
            state.wait_script.pop_front()
        };
        if let Some(outcome) = scripted {
            return outcome;
        }
        if timeout_ms > 0 {
            std::thread::sleep(std::time::Duration::from_millis(timeout_ms.into()));
        }
        Ok(None)
    }

    fn publication_checkpoint(&self, stage: PublicationStage) -> Result<()> {
        let mut state = self.state.lock();
        state.transcript.push(MockAmdCall::PublicationCheckpoint { stage });
        match state.publication_script.pop_front() {
            Some(MockPublicationOutcome::Return(outcome)) => outcome,
            Some(MockPublicationOutcome::Panic) => panic!("scripted publication panic at {stage:?}"),
            None => Ok(()),
        }
    }
}

/// Open the device-0 AMD allocator, or `None` (with a skip note) on any host
/// that lacks a supported AMD GPU — no `/dev/kfd`, unsupported arch, or missing
/// permissions. Hardware tests early-return on `None`:
///
/// ```ignore
/// let Some(alloc) = amd_alloc_or_skip() else { return };
/// ```
pub(crate) fn amd_alloc_or_skip() -> Option<AmdAllocator> {
    match AmdAllocator::new(0) {
        Ok(alloc) => Some(alloc),
        Err(_) => {
            eprintln!("skipping: no supported AMD GPU on this host");
            None
        }
    }
}

/// `true` if `alloc` drives a multi-XCC (CDNA SPX) device. The native-completion
/// and AQL-scratch probes are meaningless on a single-XCC part, so they gate on
/// this and skip (with a note) otherwise.
pub(crate) fn require_multi_xcc(alloc: &AmdAllocator) -> bool {
    if alloc.dev.node.num_xcc.max(1) > 1 {
        return true;
    }
    eprintln!("PROBE skipped: single-XCC device (multi-XCC AQL only)");
    false
}

/// `true` if `alloc` drives a single-XCC (RDNA / gfx11/12) PM4 device. The PM4
/// graph-capture probes only exercise the PM4 indirect-buffer path, so they gate
/// on this and skip (with a note) on multi-XCC AQL parts.
pub(crate) fn require_single_xcc(alloc: &AmdAllocator) -> bool {
    if alloc.dev.node.num_xcc.max(1) == 1 {
        return true;
    }
    eprintln!("PROBE skipped: multi-XCC device (single-XCC PM4 only)");
    false
}
