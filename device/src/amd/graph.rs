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
//! (carrying that signal) into the shared ring under `exec_guard`, and registers
//! the signal as in-flight — exactly like a per-call dispatch, just N+1 packets
//! in one shot. The barrier_and fires the signal once the last kernel retires;
//! host reads drain it via `AmdDevice::synchronize` → `synchronize_all`.
//!
//! This runs on the default single-queue path — no separate ring, no virtual
//! timeline, no kick/self gating, no vendor-IB/PRED_EXEC. Scope: multi-XCC CDNA
//! (AQL). Single-XCC PM4 (RDNA) chains and non-AMD / mixed-device chains return
//! `Ok(None)` → per-call dispatch.

#![cfg(target_os = "linux")]

use std::sync::Arc;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::connector::ConnectorLease;
use crate::amd::program::AmdProgram;
use crate::amd::queue::{AQL_PACKET_BYTES, build_barrier_and, build_dispatch_packet};
use crate::device::{Graph, GraphKernel};
use crate::error::{Error, Result};

/// A captured, replayable AMD kernel chain (multi-XCC AQL).
pub struct AmdGraph {
    /// The device's shared compute connector (queue + scratch + signal pool).
    /// Held for the graph's lifetime; in single-queue mode this is the device
    /// singleton, shared with per-call dispatch — they compose via the queue's
    /// FIFO order and `exec_guard`.
    connector: ConnectorLease,
    /// One native AQL kernel-dispatch packet per captured kernel. Static:
    /// kernargs are baked into `_kernargs_buf` and replayed verbatim.
    packets: Vec<[u32; 16]>,
    /// Dedicated kernargs page (one slot per kernel, baked at capture). Owned so
    /// concurrent per-call dispatch on the shared rolling arena can't lap it.
    /// `RawBuffer` has no `Drop`; freed in `Drop for AmdGraph` after the drain.
    kernargs_buf: RawBuffer,
}

// SAFETY: `packets` is immutable after capture; the connector + kernargs page
// are graph-owned stable mappings. Replay only reads `packets`/the page and
// pushes through the queue's own synchronisation.
unsafe impl Send for AmdGraph {}
unsafe impl Sync for AmdGraph {}

impl Drop for AmdGraph {
    /// Drain in-flight replays before freeing the kernargs page the GPU reads.
    /// Skipped on panic unwind (same rationale as `AmdConnector::Drop`).
    fn drop(&mut self) {
        if std::thread::panicking() {
            tracing::warn!("AmdGraph drop during panic unwind: skipping synchronize; in-flight replay abandoned");
            return;
        }
        if let Err(e) = self.connector.synchronize() {
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

        // Use the device's shared connector (the singleton in single-queue mode).
        let connector = dev.core().lease_connector(allocator)?;
        let mut max_priv_seg = 128u32;
        for p in &progs {
            max_priv_seg = max_priv_seg.max(p.private_segment_size());
        }
        connector.ensure_has_local_memory(max_priv_seg)?;

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

        // Bake one native dispatch packet per kernel (completion_signal = 0; the
        // batch's single terminating barrier_and carries the replay signal).
        let mut packets: Vec<[u32; 16]> = Vec::with_capacity(kernels.len());
        for ((k, p), &off) in kernels.iter().zip(&progs).zip(&slot_offsets) {
            // SAFETY: off + record <= total <= allocation; sole writer.
            let slot_host = unsafe { kernargs_host.add(off) };
            let slot_gpu = kernargs_gpu + off as u64;
            let bufs: Vec<u64> = k.buffers.iter().map(|&b| b as u64).collect();
            // SAFETY: slot_host owns >= kernarg_record_size() bytes (laid out above).
            unsafe { p.write_kernargs(slot_host, &bufs, &k.vals)? };

            let g = k.global_size.unwrap_or([1, 1, 1]);
            let l = k.local_size.unwrap_or([1, 1, 1]);
            let grid = [(g[0] * l[0]) as u32, (g[1] * l[1]) as u32, (g[2] * l[2]) as u32];
            let packet = build_dispatch_packet(
                [l[0] as u16, l[1] as u16, l[2] as u16],
                grid,
                p.private_segment_size(),
                p.group_segment_size(),
                p.aql_prog_addr(),
                slot_gpu,
                /*completion_signal=*/ 0,
            );
            let mut dwords = [0u32; 16];
            // SAFETY: hsa_kernel_dispatch_packet_t is repr(C), exactly 64 bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    &packet as *const _ as *const u8,
                    dwords.as_mut_ptr() as *mut u8,
                    AQL_PACKET_BYTES,
                );
            }
            packets.push(dwords);
        }

        if std::env::var_os("SVOD_DEBUG_DISPATCH").is_some() {
            eprintln!(
                "[graph capture] kernels={} kernargs_gpu={kernargs_gpu:#x} scratch={:#x}",
                kernels.len(),
                connector.scratch_gpu_va(),
            );
        }

        Ok(Some(Box::new(AmdGraph { connector, packets, kernargs_buf })))
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
        if let Some(err) = self.connector.core().poison_error() {
            return Err(err);
        }
        // Fresh per-op completion signal (armed to 1; handles pool back-pressure
        // by draining the oldest in-flight replay/dispatch when exhausted).
        let sig = self.connector.acquire_signal()?;
        // N dispatch packets (BARRIER-serialised) + one barrier_and terminator
        // that fires `sig` after the last kernel retires.
        let mut batch = self.packets.clone();
        batch.push(build_barrier_and(sig.signal_handle()));
        self.connector.queue().submit_aql(&batch)?;
        self.connector.register_inflight(Arc::clone(&sig));
        Ok(())
    }
}
