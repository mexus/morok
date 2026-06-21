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
use crate::amd::device::AmdDevice;
use crate::amd::program::AmdProgram;
use crate::amd::queue::{
    AQL_PACKET_BYTES, GraphBarrier, GraphKernelPm4, append_graph_kernel_pm4, build_barrier_and, build_barrier_and_deps,
    build_dispatch_packet, build_dispatch_packet_barrier,
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
        // Single-XCC PM4 (RDNA, e.g. gfx1151) can capture the chain into one
        // resident PM4 indirect buffer, replayed with a single doorbell — see
        // `AmdGraphPm4::capture`. Multi-XCC CDNA uses the native-AQL path below.
        //
        // OPT-IN via the per-device `pm4_graph` flag, default OFF (per-call
        // dispatch): on gfx1151 the CP executes one big inlined IB measurably
        // SLOWER than the per-call ring stream it pipelines across dispatches —
        // the GigaAM RN-T encoder is ~36% slower captured (21.7s vs 15.9s host
        // wall, full audio.wav, 277 chunks) despite producing a BIT-IDENTICAL
        // transcript. So the chain is captured correctly but per-call stays the
        // default fast path; the flag exposes the (correct) capture for hardware
        // that benefits or future barrier-granularity work. The fallback below is
        // unchanged. (Flag lives on `AmdDeviceCore`, not an env var, so a value of
        // `0` can't accidentally enable it and tests toggle it race-free.)
        if crate::amd::queue::AmdComputeQueue::will_use_pm4(dev.core()) {
            if !dev.core().pm4_graph() {
                return Ok(None);
            }
            return AmdGraphPm4::capture(allocator, kernels, progs, dev);
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
            _ => return Err(Error::NotHostVisible { what: "graph kernargs" }),
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

/// A captured, replayable AMD kernel chain (single-XCC PM4 / RDNA).
///
/// Capture bakes every kernel's full PM4 exec stream (the `dispatch_pm4`
/// SET_SH_REG + DISPATCH_DIRECT sequence, each preceded by an `hdp_flush` +
/// full `acquire_mem` hazard barrier) into ONE resident indirect buffer, with a
/// dedicated kernarg page holding every kernel's baked args. Replay submits a
/// single `PACKET3_INDIRECT_BUFFER` referencing that IB — wrapped in the queue's
/// monotonic-counter `wait`/`release_mem` discipline — with one doorbell; the CP
/// runs the whole chain inline. Mirrors [`AmdGraph`] (the AQL analogue) but uses
/// raw PM4 since single-XCC queues are PM4, not AQL.
pub struct AmdGraphPm4 {
    /// Shared `PoolQueue` owner held for the graph's lifetime (queue + scratch +
    /// PM4 counter). Replay rides this owner's counter so its `synchronize`
    /// drains the chain, identical to per-call PM4 dispatch.
    owner: OwnerCtx,
    /// Per-kernel baked geometry/identity, in replay (FIFO) order. Used to
    /// re-assemble the IB when the queue's shared scratch VA changes after
    /// capture (a co-tenant grow); a no-op in the common pre-sized case.
    kernels: Vec<GraphKernelPm4>,
    /// Per-kernel hazard-barrier strength (parallel to `kernels`), computed once
    /// from each kernel's position + `deps` at capture (see [`GraphBarrier`]).
    barriers: Vec<GraphBarrier>,
    /// Resident indirect buffer holding the concatenated per-kernel PM4 streams.
    /// `RawBuffer` has no `Drop`; freed in `Drop for AmdGraphPm4` after the drain.
    ib_buf: RawBuffer,
    ib_gpu: u64,
    ib_host: *mut u8,
    /// Capacity of `ib_buf` in dwords — a rebuild must not exceed it (scratch VA
    /// changes don't change the IB length, so this never trips, but it bounds the
    /// host write).
    ib_cap_dwords: u32,
    /// Dedicated kernarg page (one slot per kernel, baked at capture). Owned so a
    /// concurrent per-call dispatch on the shared rolling arena can't lap it.
    kernargs_buf: RawBuffer,
    /// Mutable replay state: the live IB dword count + the scratch VA/tmpring the
    /// IB is currently baked against. Guarded so replay's staleness check + IB
    /// rewrite are race-free; the `dispatch_lock` already serialises THIS graph's
    /// replays, so the inner lock is uncontended.
    state: parking_lot::Mutex<Pm4IbState>,
}

/// Mutable IB-bake state (see [`AmdGraphPm4::state`]).
struct Pm4IbState {
    /// Dword count of the currently-baked IB content (the `ib_size` field of the
    /// INDIRECT_BUFFER packet).
    ib_dwords: u32,
    /// Scratch VA/tmpring the IB was last baked against. Replay rebuilds the IB
    /// iff the queue's live scratch VA differs (co-tenant grow), so the baked
    /// descriptor never points at unmapped VRAM.
    baked_scratch_va: u64,
    baked_tmpring: u32,
}

// SAFETY: `kernels`/`ib_*` are graph-owned stable mappings; the IB host pointer
// is written only through `state`'s lock (replay rebuild), and the kernargs page
// is immutable after capture. Raw pointers address allocator-owned buffers held
// for the graph's lifetime.
unsafe impl Send for AmdGraphPm4 {}
unsafe impl Sync for AmdGraphPm4 {}

impl Drop for AmdGraphPm4 {
    /// Drain in-flight replays before freeing the IB + kernargs the GPU reads.
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!("AmdGraphPm4 drop during panic unwind: skipping synchronize; in-flight replay abandoned");
            return;
        }
        if let Err(e) = self.owner.synchronize() {
            tracing::warn!(?e, "AmdGraphPm4 drop: synchronize failed (in-flight replay lost)");
        }
        self.ib_buf.free_amd_device_in_place();
        self.kernargs_buf.free_amd_device_in_place();
    }
}

impl AmdGraphPm4 {
    /// Capture `kernels` into one resident PM4 indirect buffer. `progs` is the
    /// already-validated concrete program for each kernel (same device, checked
    /// by `AmdGraph::capture`).
    fn capture(
        allocator: &AmdAllocator,
        kernels: &[GraphKernel],
        progs: Vec<&AmdProgram>,
        dev: Arc<AmdDevice>,
    ) -> Result<Option<Box<dyn Graph>>> {
        // Own a shared `PoolQueue` and pre-size its scratch for the biggest
        // kernel, so no replay triggers a mid-chain scratch grow.
        let owner = dev.core().assign_owner(allocator)?;
        let mut max_priv_seg = 128u32;
        for p in &progs {
            max_priv_seg = max_priv_seg.max(p.private_segment_size());
        }
        owner.pool().ensure_has_local_memory(max_priv_seg)?;

        // One 16-byte-aligned kernarg slot per kernel in a dedicated page,
        // validated against each program's arity (mirrors the AQL path).
        let mut slot_offsets: Vec<usize> = Vec::with_capacity(kernels.len());
        let mut total = 0usize;
        for (k, p) in kernels.iter().zip(&progs) {
            let (buf_count, var_count) = p.arg_counts();
            if k.buffers.len() != buf_count {
                return Err(Error::Runtime {
                    message: format!(
                        "AmdGraphPm4 capture: kernel '{}' expects {buf_count} buffers, got {}",
                        k.program.name(),
                        k.buffers.len()
                    ),
                });
            }
            if k.vals.len() != var_count {
                return Err(Error::Runtime {
                    message: format!(
                        "AmdGraphPm4 capture: kernel '{}' expects {var_count} vals, got {}",
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
            _ => return Err(Error::NotHostVisible { what: "graph kernargs" }),
        };

        // Bake each kernel's kernarg slot once and record its geometry/identity.
        let mut baked: Vec<GraphKernelPm4> = Vec::with_capacity(kernels.len());
        for ((k, p), &off) in kernels.iter().zip(&progs).zip(&slot_offsets) {
            // SAFETY: off + record <= total <= allocation; sole writer.
            let slot_host = unsafe { kernargs_host.add(off) };
            let slot_gpu = kernargs_gpu + off as u64;
            let bufs: Vec<u64> = k.buffers.iter().map(|&b| b as u64).collect();
            // SAFETY: slot_host owns >= kernarg_record_size() bytes (laid out above).
            unsafe { p.write_kernargs(slot_host, &bufs, &k.vals)? };

            // PM4 launch geometry matches `execute_on`'s PM4 arm: `grid` is the
            // workgroup count (`global_size`), `local` the workgroup size — NOT
            // the AQL `grid = global*local` convention.
            let g = k.global_size.unwrap_or([1, 1, 1]);
            let l = k.local_size.unwrap_or([1, 1, 1]);
            let (rsrc1, rsrc2, rsrc3) = p.rsrc();
            let (wave32, target_major) = p.wave32_target();
            baked.push(GraphKernelPm4 {
                rsrc1,
                rsrc2,
                rsrc3,
                prog_addr: p.pm4_prog_addr(),
                enable_private_segment_sgpr: p.enable_private_segment_sgpr(),
                kernarg_user_data: [slot_gpu as u32, (slot_gpu >> 32) as u32],
                local: [l[0] as u32, l[1] as u32, l[2] as u32],
                grid: [g[0] as u32, g[1] as u32, g[2] as u32],
                wave32,
                target_major,
            });
        }

        // Per-kernel hazard-barrier strength from `deps`: a kernel with an
        // in-graph producer (RAW/WAW/WAR) does a FULL L2 invalidate to observe it;
        // a kernel with no in-graph producer reads only resident/host-stable
        // inputs (covered by the one IB-head HDP flush) so a NARROW per-CU acquire
        // suffices. Correctness rests on `deps` being the COMPLETE in-graph hazard
        // set (the GVA-keyed RAW/WAW/WAR walk in `ExecutionPlan::build_graph`).
        //
        // WRITE-THROUGH ASSUMPTION (load-bearing): inside the IB a producer ends
        // with only `CS_PARTIAL_FLUSH` — there is NO per-kernel EOP cache flush
        // (only the wrapping `replay_indirect_buffer` release_mem flushes). So the
        // Full consumer's `acquire_mem(GL2_INV|GL2_WB)` is the ONLY thing making
        // the producer's stores visible, which is correct ONLY because RDNA L0/L1
        // are write-through to GL2 (producer stores have reached L2 by the time
        // `CS_PARTIAL_FLUSH` drains). This holds on the gfx1151 target (matches the
        // bit-identical transcript); a write-BACK part would need a producer-side
        // GL2_WB on real RAW/WAW edges before the Full consumer.
        let barriers: Vec<GraphBarrier> =
            kernels.iter().map(|k| if k.deps.is_empty() { GraphBarrier::Narrow } else { GraphBarrier::Full }).collect();

        // Assemble the IB once against the current (pre-sized) scratch.
        let scratch_va = owner.pool().scratch_gpu_va();
        let tmpring = owner.pool().tmpring_size();
        let ib = Self::assemble_ib(&baked, &barriers, scratch_va, tmpring);
        let ib_dwords = ib.len() as u32;
        // Resident IB buffer (uncached GTT, CP-readable like the ring). Round the
        // capacity to a page so a co-tenant scratch-grow rebuild has identical
        // length headroom.
        let ib_bytes = (ib.len() * 4).max(16).next_multiple_of(0x1000);
        let ib_buf = allocator.alloc_uncached(ib_bytes)?;
        let (ib_gpu, ib_host) = match &ib_buf {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, h.as_ptr()),
            _ => {
                kernargs_buf.free_amd_device_in_place();
                return Err(Error::NotHostVisible { what: "graph IB" });
            }
        };
        // SAFETY: ib_host owns ib_bytes >= ib.len()*4; sole writer at capture.
        unsafe { std::ptr::copy_nonoverlapping(ib.as_ptr() as *const u8, ib_host, ib.len() * 4) };

        if std::env::var_os("SVOD_DEBUG_DISPATCH").is_some() {
            eprintln!(
                "[graph capture pm4] kernels={} ib_dwords={ib_dwords} ib_gpu={ib_gpu:#x} kernargs_gpu={kernargs_gpu:#x} scratch={scratch_va:#x}",
                kernels.len(),
            );
        }

        Ok(Some(Box::new(AmdGraphPm4 {
            owner,
            kernels: baked,
            barriers,
            ib_buf,
            ib_gpu,
            ib_host,
            ib_cap_dwords: (ib_bytes / 4) as u32,
            kernargs_buf,
            state: parking_lot::Mutex::new(Pm4IbState {
                ib_dwords,
                baked_scratch_va: scratch_va,
                baked_tmpring: tmpring,
            }),
        })))
    }

    /// Assemble the full IB dword stream: one IB-head HDP-flush handshake (makes
    /// all host writes to GTT — packed mel/lengths etc. — visible to the chain),
    /// then each kernel's `[barrier]? + exec` (see `append_graph_kernel_pm4`).
    fn assemble_ib(kernels: &[GraphKernelPm4], barriers: &[GraphBarrier], scratch_va: u64, tmpring: u32) -> Vec<u32> {
        let mut ib: Vec<u32> = Vec::new();
        // One global HDP flush up front: a host-data-path flush is GPU-wide (not
        // per-buffer), so a single handshake makes every host-written input read
        // anywhere in the chain visible — replacing the per-call path's per-kernel
        // HDP flush, the dominant inline-IB overhead.
        ib.extend_from_slice(&crate::amd::sys::pm4::hdp_flush());
        for (k, &b) in kernels.iter().zip(barriers) {
            append_graph_kernel_pm4(&mut ib, k, b, scratch_va, tmpring);
        }
        ib
    }

    /// Re-bake the IB against the live scratch VA when a co-tenant grew it after
    /// capture (rare; pre-sizing avoids it). Called holding both the queue's
    /// dispatch lock and the `state` guard, so the rewrite can't race a
    /// concurrent grow or another replay. The IB length is invariant under a
    /// scratch VA change, so this fits the original capacity.
    fn rebuild_ib(&self, st: &mut Pm4IbState, scratch_va: u64, tmpring: u32) {
        let ib = Self::assemble_ib(&self.kernels, &self.barriers, scratch_va, tmpring);
        debug_assert!(ib.len() as u32 <= self.ib_cap_dwords, "rebuilt PM4 graph IB overflows its buffer");
        let n = (ib.len() as u32).min(self.ib_cap_dwords) as usize;
        // SAFETY: ib_host owns ib_cap_dwords*4 bytes; n <= ib_cap_dwords; sole
        // writer under the held dispatch lock + `state` guard.
        unsafe { std::ptr::copy_nonoverlapping(ib.as_ptr() as *const u8, self.ib_host, n * 4) };
        st.ib_dwords = n as u32;
        st.baked_scratch_va = scratch_va;
        st.baked_tmpring = tmpring;
    }
}

impl Graph for AmdGraphPm4 {
    /// Replay the captured chain: one `PACKET3_INDIRECT_BUFFER` (wrapped in the
    /// counter `wait`/`release_mem` discipline) + one doorbell. Async — host
    /// reads drain via the owner's counter (`synchronize_all`), identical to
    /// per-call PM4 `wait=false`.
    ///
    /// `vals` is unused: the captured chain is static (no runtime vars); launch
    /// vals are baked into the kernarg slots at capture.
    fn replay(&self, vals: &[i64]) -> Result<()> {
        let _ = vals;
        let pool = self.owner.pool();
        if let Some(err) = pool.core().poison_error() {
            return Err(err);
        }
        // Hold the dispatch lock across the whole op — same fence `dispatch_pm4`
        // and `ensure_has_local_memory` use. This pins the queue's scratch VA for
        // the duration (no co-tenant grow can slip in between the staleness check
        // and the submit) and orders the counter reservation against co-tenants.
        let _disp = pool.dispatch_guard();
        let mut st = self.state.lock();
        // If a co-tenant grew scratch since capture/last-replay, the baked
        // descriptor VA is stale — re-bake the IB before submitting.
        let live_scratch = pool.scratch_gpu_va();
        let live_tmpring = pool.tmpring_size();
        if live_scratch != st.baked_scratch_va || live_tmpring != st.baked_tmpring {
            self.rebuild_ib(&mut st, live_scratch, live_tmpring);
        }
        let v = pool.queue().replay_indirect_buffer(pool, self.ib_gpu, st.ib_dwords)?;
        drop(st);
        self.owner.set_pm4_high(v);
        Ok(())
    }
}
