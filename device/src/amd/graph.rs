//! `AmdGraph`: capture a static kernel chain into a replayable batch of native
//! AQL kernel-dispatch packets, replayed on the device's **shared** compute
//! queue with one doorbell + one completion signal.
//!
//! Capture builds one `hsa_kernel_dispatch_packet_t` per kernel, kernargs baked
//! into a dedicated page (the chain is static — no runtime vars, and the plan
//! owns its buffers so their VAs are stable across replays). Each packet sets
//! the header BARRIER bit, so the chain is serialised in queue order. Replay
//! acquires a fresh per-op completion signal, blits the N packets plus a
//! terminating [`build_barrier_and`](crate::amd::queue::build_barrier_and)
//! (carrying that signal) into the shared ring with one doorbell (serialised by
//! the queue's inner Mutex), and registers the signal as in-flight — exactly
//! like a per-call dispatch, just N+1 packets in one shot. The barrier_and
//! fires the signal once the last kernel retires; host reads drain it via
//! `AmdDevice::synchronize` → `synchronize_all`.
//!
//! Runs on a shared `PoolQueue` from the device pool. Scope: multi-XCC CDNA
//! (AQL). Single-XCC PM4 (RDNA) chains and non-AMD / mixed-device chains return
//! `Ok(None)` → per-call dispatch.

#![cfg(unix)]

use std::sync::Arc;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::connector::OwnerCtx;
use crate::amd::program::AmdProgram;
use crate::amd::queue::{
    AQL_PACKET_BYTES, build_barrier_and, build_barrier_and_deps, build_dispatch_packet, build_dispatch_packet_barrier,
};
use crate::amd::signal::AmdSignal;
use crate::device::{Graph, GraphKernel};
use crate::error::{Error, Result};

/// A captured, replayable AMD kernel chain (multi-XCC AQL).
pub struct AmdGraph {
    /// This graph's owner context over a shared `PoolQueue` (queue + scratch +
    /// signal pool). Held for the graph's lifetime; the queue is shared with
    /// per-call dispatch and co-tenant owners — they compose via the queue's
    /// FIFO order (replay blits its whole batch under the inner Mutex).
    owner: OwnerCtx,
    /// The captured AQL packet stream, replayed verbatim each call. In DAG mode
    /// this interleaves `barrier_and` dep-gates (barrier-bit=0, gating only on
    /// producer signals) with BARRIER-stripped kernel-dispatch packets that each
    /// carry their own completion signal (`kernel_sigs[e]`). In blanket-BARRIER
    /// fallback mode it is one BARRIER-serialised dispatch packet per kernel
    /// (completion_signal=0). Static: kernargs are baked into `kernargs_buf`.
    packets: Vec<[u32; 16]>,
    /// DAG mode only: one graph-lifetime completion signal per kernel (emission
    /// order). Each dispatch packet fires `kernel_sigs[e]`; dependent kernels'
    /// `barrier_and` packets gate on the producers' handles. Re-armed to 1 before
    /// every replay. Empty ⇒ blanket-BARRIER fallback (no DAG overlap). Held for
    /// the graph's lifetime (reserved off the signal pool, never FIFO in-flight).
    kernel_sigs: Vec<Arc<AmdSignal>>,
    /// Dedicated kernargs page (one slot per kernel, baked at capture). Owned so
    /// concurrent per-call dispatch on the shared rolling arena can't lap it.
    /// `RawBuffer` has no `Drop`; freed in `Drop for AmdGraph` after the drain.
    kernargs_buf: RawBuffer,
}

// SAFETY: `packets` is immutable after capture; the owner + kernargs page are
// graph-owned stable mappings. Replay only reads `packets`/the page and pushes
// through the queue's own synchronisation.
unsafe impl Send for AmdGraph {}
unsafe impl Sync for AmdGraph {}

impl Drop for AmdGraph {
    /// Drain in-flight replays before freeing the kernargs page the GPU reads.
    /// Owner-local drain is sufficient: the graph's own terminating barrier_and
    /// signal is the owner's newest. Skipped on panic unwind (same rationale as
    /// `PoolQueue::Drop`).
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!("AmdGraph drop during panic unwind: skipping synchronize; in-flight replay abandoned");
            return;
        }
        if let Err(e) = self.owner.synchronize() {
            tracing::warn!(?e, "AmdGraph drop: synchronize failed (in-flight replay lost)");
        }
        // `RawBuffer` has no `Drop`; free the kernargs page now that the GPU is
        // idle on it (drained above).
        self.kernargs_buf.free_amd_device_in_place();
    }
}

impl AmdGraph {
    /// Capture `kernels` into one replayable native-AQL batch. Returns `Ok(None)`
    /// when the chain isn't graphable here (non-AMD program, single-XCC PM4, or
    /// mixed devices) so the caller falls back to per-call dispatch.
    pub fn capture(allocator: &AmdAllocator, kernels: &[GraphKernel]) -> Result<Option<Box<dyn Graph>>> {
        if kernels.is_empty() {
            return Ok(None);
        }
        // Recover the concrete AmdProgram for every kernel; all must share one
        // physical device.
        let mut progs: Vec<&AmdProgram> = Vec::with_capacity(kernels.len());
        for k in kernels {
            let Some(p) = k.program.as_any().downcast_ref::<AmdProgram>() else {
                return Ok(None);
            };
            progs.push(p);
        }
        let dev = Arc::clone(progs[0].device());
        for p in &progs[1..] {
            if !Arc::ptr_eq(p.device(), &dev) {
                return Ok(None);
            }
        }
        if let Some(err) = dev.core().poison_error() {
            return Err(err);
        }
        // AQL (multi-XCC CDNA) only. Single-XCC PM4 (RDNA) falls back to per-call
        // dispatch (its native-completion migration is not yet done).
        if crate::amd::queue::AmdComputeQueue::will_use_pm4(dev.core()) {
            return Ok(None);
        }

        // Assign a shared `PoolQueue` to this graph (its own owner context).
        let owner = dev.core().assign_owner(allocator)?;
        let mut max_priv_seg = 128u32;
        for p in &progs {
            max_priv_seg = max_priv_seg.max(p.private_segment_size());
        }
        owner.pool().ensure_has_local_memory(max_priv_seg)?;

        // One 16-byte-aligned kernarg slot per kernel in a dedicated page.
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
        let kernargs_buf = allocator.alloc_uncached(total.max(16))?;
        let (kernargs_gpu, kernargs_host) = match &kernargs_buf {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, h.as_ptr()),
            _ => {
                return Err(Error::AmdAllocFailed { reason: "graph kernargs require host-visible buffer".into() });
            }
        };

        // Bake kernargs once per kernel (shared by both lowering modes); the
        // packet header / completion signal differ per mode but the kernarg slot
        // is identical. Record each kernel's pre-computed launch geometry so the
        // packet builders below don't re-derive it.
        struct Baked {
            workgroup: [u16; 3],
            grid: [u32; 3],
            priv_seg: u32,
            group_seg: u32,
            prog_addr: u64,
            slot_gpu: u64,
        }
        let mut baked: Vec<Baked> = Vec::with_capacity(kernels.len());
        for ((k, p), &off) in kernels.iter().zip(&progs).zip(&slot_offsets) {
            // SAFETY: off + record <= total <= allocation; sole writer.
            let slot_host = unsafe { kernargs_host.add(off) };
            let slot_gpu = kernargs_gpu + off as u64;
            let bufs: Vec<u64> = k.buffers.iter().map(|&b| b as u64).collect();
            // SAFETY: slot_host owns >= kernarg_record_size() bytes (laid out above).
            unsafe { p.write_kernargs(slot_host, &bufs, &k.vals)? };

            let g = k.global_size.unwrap_or([1, 1, 1]);
            let l = k.local_size.unwrap_or([1, 1, 1]);
            baked.push(Baked {
                workgroup: [l[0] as u16, l[1] as u16, l[2] as u16],
                grid: [(g[0] * l[0]) as u32, (g[1] * l[1]) as u32, (g[2] * l[2]) as u32],
                priv_seg: p.private_segment_size(),
                group_seg: p.group_segment_size(),
                prog_addr: p.aql_prog_addr(),
                slot_gpu,
            });
        }

        // Try to reserve one graph-lifetime completion signal per kernel for
        // DAG-driven dispatch (independent kernels overlap; true deps enforced by
        // barrier_and packets). If the signal pool can't fund every slot, drop the
        // partial reservation and fall back to blanket-BARRIER capture — same
        // result, no overlap. Reservation is all-or-nothing.
        // Reserve one graph-lifetime signal per kernel for DAG dep-gating, but
        // only if the pool keeps enough headroom afterwards for per-op AQL
        // back-pressure + PM4 counters — otherwise a large graph would drain the
        // pool and make later per-call `acquire_signal` hard-fail. If headroom is
        // short, skip DAG entirely (no partial reservation) and use the
        // blanket-BARRIER fallback.
        const SIGNAL_RESERVE_HEADROOM: usize = 128;
        let mut kernel_sigs: Vec<Arc<AmdSignal>> = Vec::with_capacity(kernels.len());
        let mut dag_ok =
            !kernels.is_empty() && owner.pool().signal_free() >= kernels.len().saturating_add(SIGNAL_RESERVE_HEADROOM);
        if dag_ok {
            for _ in 0..kernels.len() {
                match owner.pool().reserve_signal() {
                    Ok(s) => kernel_sigs.push(s),
                    Err(_) => {
                        dag_ok = false;
                        break;
                    }
                }
            }
            if !dag_ok {
                kernel_sigs.clear(); // drop the partial reservation back to the pool
            }
        }

        let pack = |dw: &mut [u32; 16], packet: &crate::amd::sys::hsa::hsa_kernel_dispatch_packet_t| {
            // SAFETY: hsa_kernel_dispatch_packet_t is repr(C), exactly 64 bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    packet as *const _ as *const u8,
                    dw.as_mut_ptr() as *mut u8,
                    AQL_PACKET_BYTES,
                );
            }
        };

        let mut packets: Vec<[u32; 16]> = Vec::new();
        if dag_ok {
            // DAG emission: each kernel is preceded by barrier_and dep-gates that
            // wait on its not-yet-satisfied producers' signals, then a
            // BARRIER-stripped dispatch carrying the kernel's own completion
            // signal. `satisfied` is monotonic across the batch: once a producer
            // has been gated on, a later consumer of the same producer needs no
            // fresh barrier_and (the producer is already retired in queue order).
            let mut satisfied: std::collections::HashSet<usize> = std::collections::HashSet::new();
            for (e, (k, b)) in kernels.iter().zip(&baked).enumerate() {
                let needed: Vec<usize> = k.deps.iter().copied().filter(|d| !satisfied.contains(d)).collect();
                for chunk in needed.chunks(5) {
                    let dep_handles: Vec<u64> = chunk.iter().map(|&d| kernel_sigs[d].signal_handle()).collect();
                    packets.push(build_barrier_and_deps(&dep_handles, /*completion=*/ 0));
                }
                satisfied.extend(needed);
                let packet = build_dispatch_packet_barrier(
                    b.workgroup,
                    b.grid,
                    b.priv_seg,
                    b.group_seg,
                    b.prog_addr,
                    b.slot_gpu,
                    /*completion_signal=*/ kernel_sigs[e].signal_handle(),
                    /*barrier=*/ false,
                );
                let mut dwords = [0u32; 16];
                pack(&mut dwords, &packet);
                packets.push(dwords);
            }
        } else {
            tracing::info!(
                kernels = kernels.len(),
                "AmdGraph: DAG overlap disabled (signal budget); falling back to blanket-BARRIER capture"
            );
            // Blanket-BARRIER fallback: one BARRIER-serialised dispatch per kernel
            // (completion_signal=0); the batch's terminating barrier_and carries
            // the replay signal. Identical to the pre-DAG behaviour.
            for b in &baked {
                let packet = build_dispatch_packet(
                    b.workgroup,
                    b.grid,
                    b.priv_seg,
                    b.group_seg,
                    b.prog_addr,
                    b.slot_gpu,
                    /*completion_signal=*/ 0,
                );
                let mut dwords = [0u32; 16];
                pack(&mut dwords, &packet);
                packets.push(dwords);
            }
        }

        if std::env::var_os("SVOD_DEBUG_DISPATCH").is_some() {
            eprintln!(
                "[graph capture] kernels={} dag={dag_ok} kernargs_gpu={kernargs_gpu:#x} scratch={:#x}",
                kernels.len(),
                owner.pool().scratch_gpu_va(),
            );
        }

        Ok(Some(Box::new(AmdGraph { owner, packets, kernel_sigs, kernargs_buf })))
    }
}

impl Graph for AmdGraph {
    /// Replay the captured chain: acquire a fresh completion signal, blit the N
    /// dispatch packets + a terminating barrier_and (carrying that signal) into
    /// the shared ring with one doorbell, register the signal in-flight. Async —
    /// host reads drain via `synchronize_all`, identical to per-call `wait=false`.
    ///
    /// `vals` is unused: the captured chain is static (no runtime vars), so launch
    /// vals are baked into the kernarg slots at capture.
    fn replay(&self, vals: &[i64]) -> Result<()> {
        let _ = vals;
        let pool = self.owner.pool();
        if let Some(err) = pool.core().poison_error() {
            return Err(err);
        }
        // Serialize replays of THIS graph on the (shared) queue: hold the queue's
        // dispatch lock — which also fences against a co-tenant scratch grow that
        // would unmap scratch mid-replay — and drain the PREVIOUS batch before
        // touching the shared graph-lifetime per-kernel signals. The per-kernel
        // `kernel_sigs` are re-armed each replay (below); re-arming them while a
        // prior batch is still in flight corrupts its dep-gates (a producer's
        // signal reset to 1 under a consumer still polling for 0) and lets a
        // second batch lap the ring. Waiting the owner's last batch restores the
        // one-replay-in-flight invariant the per-call path gets for free; it is a
        // no-op when a host readback already drained it (the common case).
        let _disp = pool.dispatch_guard();
        self.owner.synchronize()?;
        // DAG mode: re-arm every per-kernel completion signal to 1 (the previous
        // batch, drained above, left them at 0). The Release store in `arm`
        // precedes the `submit_aql` doorbell's SeqCst fence, so the GPU never
        // observes a stale 0. The owner's batch completion still rides the
        // terminating barrier_and below — never the kernel_sigs.
        for s in &self.kernel_sigs {
            s.arm(1);
        }
        // Fresh per-op completion signal (armed to 1; handles pool back-pressure
        // by draining the oldest in-flight replay/dispatch when exhausted).
        let sig = pool.acquire_signal()?;
        // Captured packets + one terminating barrier_and (barrier-bit=1) that
        // fires `sig` once the WHOLE batch retires — the owner's single
        // batch-completion handle, independent of the per-kernel signals.
        let mut batch = self.packets.clone();
        batch.push(build_barrier_and(sig.signal_handle()));
        pool.queue().submit_aql(&batch)?;
        pool.register_inflight(Arc::clone(&sig));
        self.owner.set_newest(Arc::clone(&sig));
        Ok(())
    }
}
