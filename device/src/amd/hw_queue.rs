//! `AmdHwQueue`: a symbolic PM4 command builder for graph capture/replay.
//!
//! 1:1 port of tinygrad's `HWQueue` (`runtime/support/hcq.py:75`) +
//! `AMDComputeQueue` (`runtime/ops_amd.py:51`) for the single-device, single-XCC
//! PM4 compute path. The builder accumulates a dword stream (`q`) with symbolic
//! patch points so the whole stream can be bound into a host-visible page once
//! and replayed with one doorbell, re-resolving the symbolic dwords (timeline
//! values/addresses, JIT input-buffer VAs) each call without rebuilding.
//!
//! Why a port (vs. the previous bespoke IB patching): tinygrad runs the whole
//! graph as ONE device-timeline step gated by a kickoff signal, with same-queue
//! kernels ordered purely by the `acquire_mem` + `CS_PARTIAL_FLUSH` already
//! inside each `exec` (no inter-kernel signal/wait). The previous design threaded
//! a per-kernel timeline chain and bumped the device timeline N times per replay,
//! which drifted on multi-kernel chains and failed single-kernel under load. This
//! mirrors the proven structure exactly. See `graph.rs` for the assembly order.
//!
//! Symbolic model (← tinygrad `UOp.variable` + `sym_infer`): Svod PM4 isn't
//! UOp-symbolic, so a [`Sym`] is an enum resolved through a `HashMap<Sym,u64>`
//! ([`VarVals`]) at submit. Each use site additionally carries a `shift`
//! (hi/lo of a 64-bit address) and an `add` (tinygrad's `var + 1` for the final
//! signal), so one symbol covers `lo`, `hi`, and `+1` uses without extra enum
//! variants. `q(&[QVal])` records concrete dwords directly and symbolic dwords
//! in `q_sints` (← `q_sints`); `bind_sints_to_mem` records symbolic kernarg
//! fields in `mv_sints` (← `mv_sints`). `apply_var_vals` patches both, skipping
//! unchanged values via `prev_resolved` (← `_apply_var_vals`, `hcq.py:217`).

#![cfg(target_os = "linux")]

use std::collections::HashMap;
use std::sync::Arc;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::device::AmdDevice;
use crate::amd::program::AmdProgram;
use crate::amd::queue::{AmdComputeQueue, build_exec_pm4};
use crate::amd::sys::pm4;
use crate::error::{Error, Result};

/// A symbol that resolves to a `u64` at submit time. Replaces tinygrad's
/// `UOp.variable` (`graph/hcq.py:25,69,154-155`) — Svod PM4 dwords are concrete,
/// so we model the small fixed set of graph-replay symbols as an enum keyed into
/// [`VarVals`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Sym {
    /// `kickoff_var` (`graph/hcq.py:69`): the per-replay kickoff counter. The
    /// preamble waits the kick signal for this value; the host sets the kick
    /// signal to it after staging to release the whole IB.
    Kickoff,
    /// `timeline_var_<dev>` (`graph/hcq.py:155`): the virtual device-timeline
    /// value the graph's preamble waits for (resolved to `timeline_value-1`).
    /// The final signal uses this symbol with `add = 1` (← `var + 1`).
    VirtTimelineVal,
    /// `timeline_sig_<dev>` (`graph/hcq.py:154`): the GPU VA of the device
    /// timeline signal (the graph's wait/signal target address is itself a
    /// symbol so the graph drives the real device timeline at replay).
    VirtTimelineSigAddr,
    /// `inp_<iidx>_<dev>` (`graph/hcq.py:25`): a JIT input-buffer VA patched per
    /// replay. `(kernel_index, buffer_position)` identifies the kernarg slot.
    InputVa(usize, usize),
    /// A caller-supplied launch variable by name (`var_vals` in `__call__`).
    Var(String),
}

/// Resolved values for every [`Sym`] referenced by a queue, supplied at submit.
/// Mirrors tinygrad's `hcq_var_vals` dict (`graph/hcq.py:275-285`).
pub type VarVals = HashMap<Sym, u64>;

/// Number-format of a symbolic kernarg field (← tinygrad `fmt` arg, `hcq.py:211`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    /// `'Q'` — 64-bit (buffer VAs).
    U64,
    /// `'I'` — 32-bit (scalar vals).
    U32,
}

/// One value enqueued via [`AmdHwQueue::q`] — concrete dword or a symbol slot.
/// Mirrors the `int` vs `UOp` branch in tinygrad's `q()` (`hcq.py:102-106`).
enum QVal {
    /// A literal dword written verbatim.
    Dword(u32),
    /// A symbolic dword: `((resolved >> shift) as u32 + add)`. `shift` selects
    /// lo/hi of a 64-bit address; `add` is tinygrad's `var + 1`.
    Sym { sym: Sym, shift: u32, add: u32 },
}

/// A symbolic dword inside `q`: where it lives + how to derive it (← `q_sints`,
/// extended with `shift`/`add` so one [`Sym`] covers lo/hi/`+1` uses).
struct QSint {
    /// Offset into `q` (dword index) of the patched word.
    off: usize,
    /// Index into `syms`.
    sym_idx: usize,
    shift: u32,
    add: u32,
}

/// A symbolic kernarg field: where + how (← `mv_sints`). Unlike tinygrad we
/// carry `fmt` (not a bit-`mask`) because Svod's kernarg fields are whole
/// little-endian integers, never masked sub-fields (← `mv_sints`, `hcq.py:84`).
struct MvSint {
    /// Host pointer to the start of the field's containing region.
    host: *mut u8,
    /// Element index from that base, in `fmt`-sized units.
    elem: usize,
    sym_idx: usize,
    fmt: Fmt,
}

/// What `bind` produced: the host-visible page holding `q` and the 4-dword
/// indirect-buffer reference that `submit` pushes into the ring.
struct Binded {
    /// Host mapping of the page (where `apply_var_vals` patches `q` dwords).
    host: *mut u32,
    /// `[PACKET3(INDIRECT_BUFFER,2), va_lo, va_hi, len|VALID]` (← `indirect_cmd`).
    indirect_cmd: [u32; 4],
    /// Owns the page allocation — freed on drop (the allocator's `RawBuffer`
    /// Drop runs sync→unmap→munmap).
    _page: RawBuffer,
}

/// One captured kernel-arg state: where its kernarg slot lives + which fields
/// are symbolic. 1:1 with tinygrad's `CLikeArgsState` (`hcq.py:322`), narrowed
/// to the buffer-VAs-then-vals layout Svod's renderer emits.
pub struct AmdArgsState {
    /// GPU VA of the kernarg slot (goes into USER_DATA, concrete at capture).
    buf_gpu: u64,
    /// `bind_data`: each entry is `(syms, host_ptr, fmt)` — the symbolic field
    /// values `bind_args_state` records into the queue's `mv_sints`
    /// (← `bind_data`, `hcq.py:318`). Concrete buffer VAs / vals are written
    /// straight into the page at construction; only JIT-replaced inputs are
    /// symbolic.
    bind_data: Vec<(Vec<Sym>, *mut u8, Fmt)>,
}

impl AmdArgsState {
    /// Build the kernarg slot for one kernel. Port of `CLikeArgsState.__init__`
    /// (`hcq.py:322-330`): write buffer VAs (`fmt='Q'`) then scalar vals
    /// (`fmt='I'`) into the slot at `host`/`gpu`. `bufs[pos]` is either a
    /// concrete VA (`Ok`) or a [`Sym`] for a JIT input (`Err`) — symbolic ones
    /// are recorded in `bind_data` so the queue re-patches them each replay.
    /// `vals` are always concrete here (capture-time launch vars).
    ///
    /// # Safety
    /// `host` must point at a writable kernarg slot of at least
    /// `bufs.len()*8 + vals.len()*4` bytes that the caller owns.
    pub unsafe fn new(host: *mut u8, gpu: u64, bufs: &[std::result::Result<u64, Sym>], vals: &[i64]) -> Self {
        let mut bind_data = Vec::new();
        let mut cursor = 0usize;
        // Buffer VAs: 8 bytes each (`fmt='Q'`, `hcq.py:328`).
        for b in bufs {
            // SAFETY: cursor + 8 <= buf_count*8 <= slot size by caller contract.
            let field = unsafe { host.add(cursor) };
            match b {
                Ok(va) => unsafe {
                    std::ptr::copy_nonoverlapping(va.to_le_bytes().as_ptr(), field, 8);
                },
                Err(sym) => {
                    // Symbolic input VA — patched per replay. Write a poison
                    // placeholder so a missing resolution faults loudly rather
                    // than reading a stale/zero VA.
                    unsafe { std::ptr::copy_nonoverlapping(0xdead_c0de_dead_c0deu64.to_le_bytes().as_ptr(), field, 8) };
                    bind_data.push((vec![sym.clone()], field, Fmt::U64));
                }
            }
            cursor += 8;
        }
        // Scalar vals: 4 bytes each (`fmt='I'`, `hcq.py:330`); i64→i32 matching
        // the kernel's `i32` var dtype (same truncation as `AmdProgram::execute`).
        for v in vals {
            // SAFETY: cursor + 4 <= slot size.
            let field = unsafe { host.add(cursor) };
            unsafe { std::ptr::copy_nonoverlapping((*v as i32).to_le_bytes().as_ptr(), field, 4) };
            cursor += 4;
        }
        Self { buf_gpu: gpu, bind_data }
    }
}

/// A symbolic PM4 compute command builder. One per graph (single queue).
pub struct AmdHwQueue {
    dev: Arc<AmdDevice>,
    queue: Arc<AmdComputeQueue>,
    /// The dword stream (← `_q`). Concrete until `bind`, after which it lives in
    /// the host-visible page and `apply_var_vals` patches it in place.
    q: Vec<u32>,
    /// Distinct symbols in first-reference order (← `syms`).
    syms: Vec<Sym>,
    /// Per-symbol last-resolved value, parallel to `syms`; `apply_var_vals`
    /// skips patches whose symbol didn't change (← `_prev_resolved_syms`).
    prev_resolved: Vec<Option<u64>>,
    /// Symbolic dwords in `q` (← `q_sints`).
    q_sints: Vec<QSint>,
    /// Symbolic kernarg fields (← `mv_sints`).
    mv_sints: Vec<MvSint>,
    /// `None` until `bind`; then the host page + indirect-buffer reference.
    binded: Option<Binded>,
}

// SAFETY: the host pointers in `mv_sints`/`binded` are stable host-visible
// mappings the graph owns for its lifetime; the only writers are `bind`
// (capture) and `submit` (replay), which the graph serializes under the device
// dispatch lock (see `queue::submit_dwords`).
unsafe impl Send for AmdHwQueue {}
unsafe impl Sync for AmdHwQueue {}

impl AmdHwQueue {
    /// New empty queue (← `HWQueue.__init__`, `hcq.py:80`).
    pub fn new(dev: Arc<AmdDevice>, queue: Arc<AmdComputeQueue>) -> Self {
        Self {
            dev,
            queue,
            q: Vec::new(),
            syms: Vec::new(),
            prev_resolved: Vec::new(),
            q_sints: Vec::new(),
            mv_sints: Vec::new(),
            binded: None,
        }
    }

    /// Intern a symbol, returning its index (← `_new_sym`, `hcq.py:88`).
    fn new_sym(&mut self, sym: &Sym) -> usize {
        if let Some(i) = self.syms.iter().position(|s| s == sym) {
            return i;
        }
        self.syms.push(sym.clone());
        self.prev_resolved.push(None);
        self.syms.len() - 1
    }

    /// Enqueue values — concrete dwords verbatim, symbols recorded for later
    /// resolution (← `q`, `hcq.py:94`).
    fn q(&mut self, values: Vec<QVal>) {
        for v in values {
            match v {
                QVal::Dword(d) => self.q.push(d),
                QVal::Sym { sym, shift, add } => {
                    let sym_idx = self.new_sym(&sym);
                    self.q_sints.push(QSint { off: self.q.len(), sym_idx, shift, add });
                    self.q.push(0xbadc_0ded); // placeholder (← `hcq.py:105`)
                }
            }
        }
    }

    /// Push a (possibly symbolic) 64-bit address as `[lo, hi]` dwords.
    fn addr64(&self, addr: &SymU64) -> Vec<QVal> {
        match addr {
            SymU64::Concrete(a) => vec![QVal::Dword(*a as u32), QVal::Dword((*a >> 32) as u32)],
            SymU64::Sym(s) => {
                vec![QVal::Sym { sym: s.clone(), shift: 0, add: 0 }, QVal::Sym { sym: s.clone(), shift: 32, add: 0 }]
            }
        }
    }

    // *** commands (← AMDComputeQueue) ***

    /// `wait_reg_mem`/`wait` (← `ops_amd.py:85,370`): poll memory at `addr`
    /// until `(*addr & mask) >= value`. Both may be symbolic (the graph waits
    /// the virtual device-timeline signal whose address is `VirtTimelineSigAddr`).
    /// Layout matches `pm4::wait_reg_mem`:
    /// `[hdr, info, addr_lo, addr_hi, value, mask, poll]`.
    fn wait(&mut self, addr: SymU64, value: SymU32, mask: u32) {
        let info = pm4::wait_reg_mem_mem_space(1)
            | pm4::wait_reg_mem_function(pm4::WAIT_REG_MEM_FUNC_GEQ)
            | pm4::wait_reg_mem_engine(0);
        let mut pkt = vec![QVal::Dword(pm4::packet3(pm4::PACKET3_WAIT_REG_MEM, 5)), QVal::Dword(info)];
        pkt.extend(self.addr64(&addr));
        pkt.push(value.into_qval());
        pkt.push(QVal::Dword(mask));
        pkt.push(QVal::Dword(4));
        self.q(pkt);
    }

    /// `signal` (← `ops_amd.py:385`) = `release_mem(addr, value, cache_flush=true)`.
    /// Writes the low 32 bits of `value` to `addr` after a full system-scope
    /// cache flush. Both may be symbolic (the graph signals the virtual device
    /// timeline). Layout matches `pm4::release_mem`:
    /// `[hdr, event, memsel, addr_lo, addr_hi, value, value_hi, ctxid]`.
    fn signal(&mut self, addr: SymU64, value: SymU32) {
        let event_dw = pm4::release_mem_event_type(pm4::CACHE_FLUSH_AND_INV_TS_EVENT)
            | pm4::release_mem_event_index(pm4::EVENT_INDEX_END_OF_PIPE)
            | pm4::RELEASE_MEM_CACHE_FLUSH_ALL;
        let memsel_dw = pm4::release_mem_data_sel(pm4::DATA_SEL_SEND_32_BIT_LOW)
            | pm4::release_mem_int_sel(pm4::INT_SEL_INTERRUPT_AFTER_WRITE)
            | pm4::release_mem_dst_sel(pm4::DST_SEL_MEMORY);
        let mut pkt =
            vec![QVal::Dword(pm4::packet3(pm4::PACKET3_RELEASE_MEM, 6)), QVal::Dword(event_dw), QVal::Dword(memsel_dw)];
        pkt.extend(self.addr64(&addr));
        pkt.push(value.into_qval());
        pkt.push(QVal::Dword(0)); // value_hi
        pkt.push(QVal::Dword(0)); // ctxid
        self.q(pkt);
    }

    /// `memory_barrier` (← `ops_amd.py:133`): HDP-flush register handshake then
    /// a full `acquire_mem`. Concrete (no symbols).
    fn memory_barrier(&mut self) {
        self.q.extend_from_slice(&pm4::hdp_flush());
        self.q.extend_from_slice(&pm4::acquire_mem());
    }

    /// `exec` (← `ops_amd.py:320`) — the critical command. Records the kernarg
    /// slot's symbolic fields into `mv_sints` (`bind_args_state`, `hcq.py:321`),
    /// then emits the exact dword sequence `build_exec_pm4` produces for a
    /// per-call dispatch: `acquire_mem(gli=0,gl2=0)` → PGM/RSRC/TMPRING/SCRATCH/
    /// RESTART/USER_DATA/RESOURCE_LIMITS/START_X regs → `DISPATCH_DIRECT` →
    /// `EVENT_WRITE(CS_PARTIAL_FLUSH)`. USER_DATA holds the concrete kernarg page
    /// VA + optional scratch prefix; same-queue ordering comes from the
    /// `acquire_mem` + `CS_PARTIAL_FLUSH`, so there is NO inter-kernel
    /// signal/wait.
    pub fn exec(&mut self, prg: &AmdProgram, args: &AmdArgsState, global_size: [u32; 3], local_size: [u32; 3]) {
        // bind_args_state (← hcq.py:205): record symbolic kernarg fields.
        for (syms, mem, fmt) in &args.bind_data {
            for (i, sym) in syms.iter().enumerate() {
                let sym_idx = self.new_sym(sym);
                self.mv_sints.push(MvSint { host: *mem, elem: i, sym_idx, fmt: *fmt });
            }
        }

        // Snapshot scratch under the dispatch lock — same invariant as
        // `dispatch_pm4`: a concurrent scratch realloc holds this lock while it
        // unmaps the old VA, so the captured base stays live for the graph's
        // life (program load already grew scratch to fit every captured kernel).
        let (scratch_addr, tmpring_size) = {
            let _disp = self.dev.lock_dispatch();
            (self.dev.scratch_gpu_va(), self.dev.tmpring_size())
        };

        // USER_DATA SGPR prefix: optional 4-dword scratch descriptor, then the
        // 2-dword kernarg pointer — identical to `AmdProgram::execute`
        // (`program.rs:646-655` / `ops_amd.py:325-342`).
        let mut user_data: smallvec::SmallVec<[u32; 8]> = smallvec::SmallVec::new();
        if prg.enable_private_segment_sgpr() {
            user_data.push(scratch_addr as u32);
            user_data.push((scratch_addr >> 32) as u32 | (1u32 << 31));
            user_data.push(0xFFFF_FFFF);
            user_data.push(0x20c1_4000);
        }
        user_data.push(args.buf_gpu as u32);
        user_data.push((args.buf_gpu >> 32) as u32);

        let (rsrc1, rsrc2, rsrc3) = prg.rsrc();
        let (wave32, target_major) = prg.wave32_target();
        // build_exec_pm4 appends the concrete reg-set + dispatch + CS_PARTIAL_FLUSH
        // — byte-identical to the per-call path (no symbols in the body).
        build_exec_pm4(
            &mut self.q,
            rsrc1,
            rsrc2,
            rsrc3,
            prg.pm4_prog_addr(),
            &user_data,
            scratch_addr,
            tmpring_size,
            local_size,
            global_size,
            wave32,
            target_major,
        );
    }

    /// Preamble: `memory_barrier().wait(virt_timeline).wait(kick).signal(self)`
    /// (← `graph/hcq.py:158-160`). `self_sig_addr` is the per-graph signal the
    /// preamble sets to `kickoff`; the kick wait gates the whole IB until the
    /// host stages the replay and releases it by setting the kick signal.
    pub fn preamble(&mut self, kick_sig_addr: u64, self_sig_addr: u64) {
        self.memory_barrier();
        self.wait(SymU64::Sym(Sym::VirtTimelineSigAddr), SymU32::Sym(Sym::VirtTimelineVal), 0xFFFF_FFFF);
        self.wait(SymU64::Concrete(kick_sig_addr), SymU32::Sym(Sym::Kickoff), 0xFFFF_FFFF);
        self.signal(SymU64::Concrete(self_sig_addr), SymU32::Sym(Sym::Kickoff));
    }

    /// Final: `signal(virt_timeline_sig, virt_timeline_val + 1)`
    /// (← `graph/hcq.py:217`). Advances the real device timeline by exactly +1
    /// per replay (the preamble waited `timeline-1`), so the graph composes with
    /// per-call dispatch and `AmdDevice::synchronize`.
    pub fn final_signal(&mut self) {
        self.signal(SymU64::Sym(Sym::VirtTimelineSigAddr), SymU32::SymAdd(Sym::VirtTimelineVal, 1));
    }

    /// `bind` (← `ops_amd.py:396`): allocate a host-visible uncached page, copy
    /// `q` into it, build the `INDIRECT_BUFFER` reference, and repoint future
    /// patches at the page (so `apply_var_vals` rewrites GPU-resident dwords).
    pub fn bind(&mut self, allocator: &AmdAllocator) -> Result<()> {
        let page = allocator.alloc_uncached((self.q.len() * 4).max(16))?;
        let (gpu, host) = match &page {
            RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, h.as_ptr() as *mut u32),
            _ => return Err(Error::AmdAllocFailed { reason: "graph hw_page requires host-visible buffer".into() }),
        };
        // SAFETY: host maps >= q.len() u32s (page-rounded); source is exactly
        // q.len() dwords.
        unsafe { std::ptr::copy_nonoverlapping(self.q.as_ptr(), host, self.q.len()) };
        let indirect_cmd = [
            pm4::packet3(pm4::PACKET3_INDIRECT_BUFFER, 2),
            gpu as u32,
            (gpu >> 32) as u32,
            (self.q.len() as u32) | pm4::INDIRECT_BUFFER_VALID,
        ];
        self.binded = Some(Binded { host, indirect_cmd, _page: page });
        Ok(())
    }

    /// `_apply_var_vals` (← `hcq.py:217`): resolve every symbol, patch changed
    /// `q` dwords and `mv_sints` kernarg fields in place. Skips symbols whose
    /// resolved value is unchanged since the last submit (`prev_resolved`).
    fn apply_var_vals(&mut self, var_vals: &VarVals) -> Result<()> {
        let resolved: Vec<u64> = self
            .syms
            .iter()
            .map(|s| {
                var_vals
                    .get(s)
                    .copied()
                    .ok_or_else(|| Error::Runtime { message: format!("AmdHwQueue: unresolved graph symbol {s:?}") })
            })
            .collect::<Result<_>>()?;

        let host = self.binded.as_ref().map(|b| b.host);
        for qs in &self.q_sints {
            if self.prev_resolved[qs.sym_idx] == Some(resolved[qs.sym_idx]) {
                continue;
            }
            let word = ((resolved[qs.sym_idx] >> qs.shift) as u32).wrapping_add(qs.add);
            match host {
                // SAFETY: off < q.len() <= page dwords; page is live + writable.
                Some(h) => unsafe { std::ptr::write_volatile(h.add(qs.off), word) },
                None => self.q[qs.off] = word, // unbound (test/diagnostic only)
            }
        }

        for mv in &self.mv_sints {
            if self.prev_resolved[mv.sym_idx] == Some(resolved[mv.sym_idx]) {
                continue;
            }
            let val = resolved[mv.sym_idx];
            // SAFETY: host points into a kernarg slot the graph owns; elem index
            // is within the slot by construction.
            unsafe {
                match mv.fmt {
                    Fmt::U64 => std::ptr::write_unaligned((mv.host as *mut u64).add(mv.elem), val),
                    Fmt::U32 => std::ptr::write_unaligned((mv.host as *mut u32).add(mv.elem), val as u32),
                }
            }
        }

        for (slot, r) in self.prev_resolved.iter_mut().zip(&resolved) {
            *slot = Some(*r);
        }
        Ok(())
    }

    /// `submit` (← `hcq.py:230`): apply `var_vals` then `_submit`. Patches the
    /// bound page's symbolic dwords + kernarg fields, fences, then pushes the
    /// indirect-buffer reference with one doorbell.
    pub fn submit(&mut self, var_vals: &VarVals) -> Result<()> {
        self.apply_var_vals(var_vals)?;
        std::sync::atomic::fence(std::sync::atomic::Ordering::Release);
        let cmd = self.binded.as_ref().expect("AmdHwQueue::submit before bind").indirect_cmd;
        // `_submit` (← `ops_amd.py:407`): one doorbell via the queue primitive.
        self.queue.submit_dwords(&cmd)
    }
}

/// A 64-bit field that may be concrete or symbolic (addresses).
enum SymU64 {
    Concrete(u64),
    Sym(Sym),
}

/// A 32-bit symbolic field (timeline/kickoff values), optionally with a `+add`
/// (← tinygrad's `virt_timeline_val + 1`). Every graph wait/signal value is a
/// symbol, so there is no concrete variant.
enum SymU32 {
    Sym(Sym),
    SymAdd(Sym, u32),
}

impl SymU32 {
    fn into_qval(self) -> QVal {
        match self {
            SymU32::Sym(s) => QVal::Sym { sym: s, shift: 0, add: 0 },
            SymU32::SymAdd(s, add) => QVal::Sym { sym: s, shift: 0, add },
        }
    }
}
