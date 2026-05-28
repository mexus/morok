//! `AmdGraph`: PM4 graph capture/replay — a 1:1 port of tinygrad's `HCQGraph`
//! (`runtime/graph/hcq.py:11`) for the single-device, single-XCC PM4 compute
//! path.
//!
//! The whole kernel chain is assembled once into one [`AmdHwQueue`] command
//! stream, bound into a host-visible page, and replayed with ONE doorbell — so
//! repeated inference pays graph-launch cost once instead of the per-kernel
//! `wait → barrier → exec → signal → doorbell` round-trip N times.
//!
//! Structure (faithful to `HCQGraph`):
//! - The graph is ONE device-timeline step. A preamble does
//!   `memory_barrier → wait(virt_timeline, timeline-1) → wait(kick, kickoff) →
//!   signal(self, kickoff)`; per kernel just `exec()` (NO inter-kernel
//!   signal/wait — same-queue ordering is the `acquire_mem` + `CS_PARTIAL_FLUSH`
//!   already inside each `exec`); a final `signal(virt_timeline, timeline)`
//!   advances the real device timeline by exactly +1.
//! - The `virt_timeline` signal's address + value are SYMBOLS resolved at replay
//!   to the device timeline signal's `value_addr()` and `timeline_value()-1`, so
//!   the graph composes with per-call dispatch and `AmdDevice::synchronize`.
//! - A per-graph kickoff signal lets the host stage all patches then release the
//!   whole IB atomically by setting the kick signal to `kickoff_value`.
//!
//! Scope: single device, single PM4 compute queue. AQL (multi-XCC), non-AMD
//! programs, or mixed device/queue chains → `Ok(None)` (caller falls back to
//! per-call dispatch). Mirrors `HCQGraph.supports_uop` rejecting non-PROGRAM /
//! MOCK queues (`graph/hcq.py:318-336`).

#![cfg(target_os = "linux")]

use std::sync::Arc;

use parking_lot::Mutex;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::connector::AmdConnector;
use crate::amd::hw_queue::{AmdHwQueue, Sym, VarVals};
use crate::amd::program::AmdProgram;
use crate::amd::queue::AmdComputeQueue;
use crate::amd::signal::{AmdSignal, SignalPool};
use crate::device::{Graph, GraphKernel};
use crate::error::{Error, Result};
use crate::sync::TimelineSignal;

/// A captured, replayable AMD kernel chain.
pub struct AmdGraph {
    /// Per-graph connector — owns the scratch, timeline signal/value, and the
    /// (transitional) dispatch lock used for replay. Built fresh in
    /// [`AmdGraph::capture`] from the captured kernels' shared device core.
    /// Step 5 of the connector refactor: replaces `dev: Arc<AmdDevice>` so
    /// the graph runs on isolated dispatch state — no per-call sibling can
    /// race its timeline reservation or scratch realloc.
    connector: Arc<AmdConnector>,
    /// The single PM4 command stream for the whole chain (preamble + N execs +
    /// final signal), bound into a host-visible page. `submit` mutates its
    /// patch state, so it sits behind a `Mutex` — replay takes `&self`
    /// (`Graph::replay`) but serializes against itself. Mirrors tinygrad's
    /// per-device `comp_queues[dev]` (`graph/hcq.py:51`).
    comp_queue: Mutex<AmdHwQueue>,
    /// Dedicated kernarg page — one fixed slot per kernel, written at capture.
    /// Owning it (vs. the shared rolling `KernargArena` lapped by concurrent
    /// per-call dispatch → stale VAs → `NotPresent`) is what makes replay safe.
    /// Mirrors `HCQGraph`'s per-graph `kernargs_bufs` (`graph/hcq.py:33`).
    _kernargs_buf: RawBuffer,
    /// Per-graph kickoff signal (← `kick_signals`, `graph/hcq.py:65`). The
    /// preamble waits this for `kickoff_value`; replay sets it after staging to
    /// release the whole IB.
    kick_sig: AmdSignal,
    /// Per-graph "queue" signal the preamble sets to `kickoff_value`
    /// (← `signals`, `graph/hcq.py:66`). Reset to 0 each replay
    /// (`queue_signals_to_reset`, `graph/hcq.py:221`).
    self_sig: AmdSignal,
    /// Held to keep the signal pool's GTT page mapped while `kick_sig`/`self_sig`
    /// borrow into it.
    _signal_pool: Arc<SignalPool>,
    /// Per-replay kickoff counter (← `kickoff_value`, `graph/hcq.py:68`).
    kickoff_value: Mutex<u64>,
    /// `last_timeline` device value waited on at the start of each replay
    /// (← `last_timeline`, `graph/hcq.py:220`).
    last_timeline: Mutex<u64>,
}

// SAFETY: every interior-mutable field is behind a `Mutex`; `AmdHwQueue` is
// already `Send + Sync` (its host pointers are stable graph-owned mappings).
unsafe impl Send for AmdGraph {}
unsafe impl Sync for AmdGraph {}

impl AmdGraph {
    /// Capture `kernels` into one PM4 command stream. Returns `Ok(None)` when
    /// the chain isn't graphable on the PM4 path (non-AMD program, AQL queue, or
    /// mixed devices/queues) so the caller falls back to per-call dispatch.
    pub fn capture(allocator: &AmdAllocator, kernels: &[GraphKernel]) -> Result<Option<Box<dyn Graph>>> {
        if kernels.is_empty() {
            return Ok(None);
        }

        // Recover the concrete AmdProgram for every kernel; assert they share one
        // device + one PM4 queue. A chain spanning two queues/devices would need
        // tinygrad's multi-device `_resolve_deps` cross-queue sync — out of scope.
        let mut progs: Vec<&AmdProgram> = Vec::with_capacity(kernels.len());
        for k in kernels {
            let Some(p) = k.program.as_any().downcast_ref::<AmdProgram>() else {
                return Ok(None);
            };
            progs.push(p);
        }
        let dev = Arc::clone(progs[0].device());
        let queue: Arc<AmdComputeQueue> = Arc::clone(progs[0].queue());
        if !queue.is_pm4() {
            return Ok(None);
        }
        for p in &progs[1..] {
            if !Arc::ptr_eq(p.device(), &dev) || !Arc::ptr_eq(p.queue(), &queue) {
                return Ok(None);
            }
        }
        if let Some(err) = dev.core().poison_error() {
            return Err(err);
        }

        // ── Build a fresh per-graph connector (Step 5 of the connector refactor).
        // Owns its own scratch + timeline so replay no longer contends with
        // per-call dispatchers on the device's default connector. Grown to fit
        // the captured kernels' private-segment requirement; timeline signal
        // acquired from the shared pool (same pool the per-call signals come
        // from — pool access is rare, intra-pool Mutex is fine).
        let connector = AmdConnector::new(Arc::clone(dev.core()))?;
        let mut max_priv_seg = 128u32;
        for p in &progs {
            max_priv_seg = max_priv_seg.max(p.private_segment_size());
        }
        connector.ensure_has_local_memory(max_priv_seg)?;
        let timeline_sig = Arc::new(progs[0].signal_pool().acquire()?);
        connector.init_timeline(timeline_sig);

        // ── Lay out one 16-byte-aligned kernarg slot per kernel inside a single
        // dedicated page (← `kernargs_bufs` + per-kernel `BumpAllocator.alloc`,
        // `graph/hcq.py:33-41`). Validate the buffer/val counts up front.
        let mut slot_offsets: Vec<usize> = Vec::with_capacity(kernels.len());
        let mut total = 0usize;
        for (k, p) in kernels.iter().zip(&progs) {
            let (buf_count, var_count) = p.arg_counts();
            if k.buffers.len() != buf_count {
                return Err(Error::Runtime {
                    message: format!(
                        "AmdGraph capture: kernel '{}' expects {buf_count} buffers, got {}",
                        k.program.name(),
                        k.buffers.len()
                    ),
                });
            }
            if k.vals.len() != var_count {
                return Err(Error::Runtime {
                    message: format!(
                        "AmdGraph capture: kernel '{}' expects {var_count} vals, got {}",
                        k.program.name(),
                        k.vals.len()
                    ),
                });
            }
            slot_offsets.push(total);
            total += p.kernarg_record_size().next_multiple_of(16);
        }

        // Dedicated host-visible uncached kernargs page (same flags as the
        // shared arena).
        let kernargs_buf = allocator.alloc_uncached(total.max(16))?;
        let (kernargs_gpu, kernargs_host) = match &kernargs_buf {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, h.as_ptr()),
            _ => {
                return Err(Error::AmdAllocFailed { reason: "graph kernargs require host-visible buffer".into() });
            }
        };

        // ── Per-graph signals from the shared pool (← `kick_signals` +
        // `signals`, `graph/hcq.py:65-66`). Both start at 0.
        let signal_pool = Arc::clone(progs[0].signal_pool());
        let kick_sig = signal_pool.acquire()?;
        let self_sig = signal_pool.acquire()?;

        // ── Build the one command stream. (← `comp_queues[dev]`, plus the
        // preamble/exec/final loop at `graph/hcq.py:158-217`.)
        let mut comp_queue = AmdHwQueue::new(Arc::clone(&connector), Arc::clone(&queue));

        // Preamble (← graph/hcq.py:158-160).
        comp_queue.preamble(kick_sig.value_addr(), self_sig.value_addr());

        // Per-kernel: fill kernargs, then exec (← graph/hcq.py:162-210, AMD
        // subset — no inter-kernel signal/wait).
        for ((k, p), &off) in kernels.iter().zip(&progs).zip(&slot_offsets) {
            // SAFETY: off + record <= total <= allocation; the graph is the sole
            // writer of this slot for its lifetime.
            let slot_host = unsafe { kernargs_host.add(off) };
            let slot_gpu = kernargs_gpu + off as u64;

            // Plain realize: buffer VAs are plan-stable, so every buffer is a
            // concrete VA (`Ok`). JIT input replacement (tinygrad's
            // `input_replace_to_var`, `graph/hcq.py:19-26`) would mark a position
            // `Err(Sym::InputVa(j,pos))`; the current `GraphKernel` carries no
            // input map, so none are symbolic here. The `Sym::InputVa` machinery
            // is in place for when JIT graphs land.
            let bufs: Vec<std::result::Result<u64, Sym>> = k.buffers.iter().map(|&b| Ok(b as u64)).collect();
            // SAFETY: slot_host owns >= kernarg_record_size() bytes (laid out
            // above); fill_kernargs re-validates the layout against the record.
            let args = unsafe { p.fill_kernargs(slot_host, slot_gpu, &bufs, &k.vals)? };

            let g = k.global_size.unwrap_or([1, 1, 1]);
            let l = k.local_size.unwrap_or([1, 1, 1]);
            comp_queue.exec(p, &args, [g[0] as u32, g[1] as u32, g[2] as u32], [l[0] as u32, l[1] as u32, l[2] as u32]);
        }

        // Final signal advancing the real device timeline by +1 (← graph/hcq.py:217).
        comp_queue.final_signal();

        // Bind into a host-visible page + build the indirect-buffer reference
        // (← `bind(dev)`, graph/hcq.py:217).
        comp_queue.bind(allocator)?;

        if std::env::var_os("SVOD_DEBUG_DISPATCH").is_some() {
            eprintln!(
                "[graph capture] kernels={} kernargs_gpu={kernargs_gpu:#x} kick_sig={:#x} self_sig={:#x} scratch={:#x}",
                kernels.len(),
                kick_sig.value_addr(),
                self_sig.value_addr(),
                connector.scratch_gpu_va(),
            );
        }
        // `dev` is now unused — keep the variable name for the construction trace
        // above (it still drives validation) but drop the local: capture stores
        // the connector, not the device.
        drop(dev);

        Ok(Some(Box::new(AmdGraph {
            connector,
            comp_queue: Mutex::new(comp_queue),
            _kernargs_buf: kernargs_buf,
            kick_sig,
            self_sig,
            _signal_pool: signal_pool,
            // last_timeline starts at 0 (← `{dev: (timeline_signal, 0)}`,
            // graph/hcq.py:220).
            kickoff_value: Mutex::new(0),
            last_timeline: Mutex::new(0),
        })))
    }
}

impl Graph for AmdGraph {
    /// Replay the captured chain (← `HCQGraph.__call__`, `graph/hcq.py:263`).
    ///
    /// `vals` is unused: the gated chains are all static (no runtime vars), so
    /// launch `vals` are baked into the kernarg slots at capture. Buffer VAs are
    /// plan-stable too — only the timeline/kickoff symbols change per replay.
    fn replay(&self, vals: &[i64]) -> Result<()> {
        let _ = vals;
        if let Some(err) = self.connector.core().poison_error() {
            return Err(err);
        }

        // 1. Bump kickoff + wait the previous replay's timeline target
        //    (← graph/hcq.py:271-272).
        let kickoff_value = {
            let mut k = self.kickoff_value.lock();
            *k += 1;
            *k
        };
        let last = *self.last_timeline.lock();
        if last > 0 {
            self.connector.timeline_signal().wait(last, 30_000)?;
        }

        // 2. Reserve this replay's timeline step and submit the IB. Under Step 5
        //    of the connector refactor the graph owns its connector
        //    exclusively, so the timeline + ring are not contended by any
        //    sibling dispatcher — the prior `lock_dispatch()` critical section
        //    around [reserve → patch → push] is no longer needed and would
        //    be deleted in Step 7. We keep the bracketed block to scope the
        //    `comp_queue` lock acquisition.
        let signalled = {
            // VirtTimelineVal = timeline_value-1 (what the preamble waits for);
            // the final signal writes +1, advancing the connector timeline by
            // exactly one step. `next_timeline` reserves that same value.
            let prev = self.connector.timeline_value().saturating_sub(1);
            let signalled = self.connector.next_timeline();

            // Resolve the graph's symbols (← `hcq_var_vals`, graph/hcq.py:275-285).
            let mut var_vals: VarVals = VarVals::new();
            var_vals.insert(Sym::Kickoff, kickoff_value);
            var_vals.insert(Sym::VirtTimelineVal, prev);
            var_vals.insert(Sym::VirtTimelineSigAddr, self.connector.timeline_signal().value_addr());

            // submit → apply_var_vals (patch hw_page + kernargs) → _submit (one
            //  doorbell). (← graph/hcq.py:290.)
            self.comp_queue.lock().submit(&var_vals)?;
            signalled
        };
        *self.last_timeline.lock() = signalled;

        // 3. Reset the per-queue signal, then release the IB by setting the kick
        //    signal to kickoff_value (← graph/hcq.py:295-296). Ordering matters:
        //    the GPU's preamble is parked on the kick wait until this store, and
        //    the IB is already in the ring (pushed above).
        self.self_sig.set(0);
        self.kick_sig.set(kickoff_value);

        // Async return — no synchronize here (← `HCQGraph.__call__` only waits
        // when `wait=True`, graph/hcq.py:298-300). Back-pressure is the *next*
        // replay's `last_timeline` wait above; host reads drain to this replay's
        // final signal (`signalled`) via `AmdAllocator::_copyout` / the
        // `Buffer::as_*` guards / an explicit `synchronize`, identical to the
        // per-call dispatch path (`program.rs::execute`).
        Ok(())
    }
}
