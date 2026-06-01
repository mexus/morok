//! `AmdHwQueue`: a symbolic PM4 command builder for graph capture/replay.
//!
//! A hardware-queue command builder for the single-device, single-XCC
//! PM4 compute path. The builder accumulates a dword stream (`q`) with symbolic
//! patch points so the whole stream can be bound into a host-visible page once
//! and replayed with one doorbell, re-resolving the symbolic dwords (timeline
//! values/addresses, JIT input-buffer VAs) each call without rebuilding.
//!
//! Why this structure (vs. the previous bespoke IB patching): the whole
//! graph runs as ONE device-timeline step gated by a kickoff signal, with same-queue
//! kernels ordered purely by the `acquire_mem` + `CS_PARTIAL_FLUSH` already
//! inside each `exec` (no inter-kernel signal/wait). The previous design threaded
//! a per-kernel timeline chain and bumped the device timeline N times per replay,
//! which drifted on multi-kernel chains and failed single-kernel under load. See
//! `graph.rs` for the assembly order.
//!
//! Symbolic model: Svod PM4 dwords are concrete, so a [`Sym`] is an enum
//! resolved through a `HashMap<Sym,u64>`
//! ([`VarVals`]) at submit. Each use site additionally carries a `shift`
//! (hi/lo of a 64-bit address) and an `add` (`var + 1` for the final
//! signal), so one symbol covers `lo`, `hi`, and `+1` uses without extra enum
//! variants. `q(&[QVal])` records concrete dwords directly and symbolic dwords
//! in `q_sints`; `bind_sints_to_mem` records symbolic kernarg
//! fields in `mv_sints`. `apply_var_vals` patches both, skipping
//! unchanged values via `prev_resolved`.

#![cfg(target_os = "linux")]

use std::collections::HashMap;

use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::connector::AmdConnector;
use crate::amd::program::AmdProgram;
use crate::amd::queue::build_exec_pm4;
use crate::amd::sys::pm4;
use crate::error::{Error, Result};

/// A symbol that resolves to a `u64` at submit time. Svod PM4 dwords are
/// concrete, so we model the small fixed set of graph-replay symbols as an enum
/// keyed into [`VarVals`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Sym {
    /// The per-replay kickoff counter. The
    /// preamble waits the kick signal for this value; the host sets the kick
    /// signal to it after staging to release the whole IB.
    Kickoff,
    /// The virtual device-timeline
    /// value the graph's preamble waits for (resolved to `timeline_value-1`).
    /// The final signal uses this symbol with `add = 1` (i.e. `var + 1`).
    VirtTimelineVal,
    /// The GPU VA of the device
    /// timeline signal (the graph's wait/signal target address is itself a
    /// symbol so the graph drives the real device timeline at replay).
    VirtTimelineSigAddr,
    /// A JIT input-buffer VA patched per
    /// replay. `(kernel_index, buffer_position)` identifies the kernarg slot.
    InputVa(usize, usize),
    /// A caller-supplied launch variable by name (`var_vals` in `__call__`).
    Var(String),
}

/// Resolved values for every [`Sym`] referenced by a queue, supplied at submit.
pub type VarVals = HashMap<Sym, u64>;

/// Number-format of a symbolic kernarg field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fmt {
    /// `'Q'` — 64-bit (buffer VAs).
    U64,
    /// `'I'` — 32-bit (scalar vals).
    U32,
}

/// One value enqueued via [`AmdHwQueue::q`] — concrete dword or a symbol slot.
enum QVal {
    /// A literal dword written verbatim.
    Dword(u32),
    /// A symbolic dword: `((resolved >> shift) as u32 + add)`. `shift` selects
    /// lo/hi of a 64-bit address; `add` applies the `var + 1` offset.
    Sym { sym: Sym, shift: u32, add: u32 },
}

/// A symbolic dword inside `q`: where it lives + how to derive it
/// (extended with `shift`/`add` so one [`Sym`] covers lo/hi/`+1` uses).
struct QSint {
    /// Offset into `q` (dword index) of the patched word.
    off: usize,
    /// Index into `syms`.
    sym_idx: usize,
    shift: u32,
    add: u32,
}

/// A symbolic kernarg field: where + how. We
/// carry `fmt` (not a bit-`mask`) because Svod's kernarg fields are whole
/// little-endian integers, never masked sub-fields.
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
/// are symbolic, narrowed
/// to the buffer-VAs-then-vals layout Svod's renderer emits.
pub struct AmdArgsState {
    /// GPU VA of the kernarg slot (goes into USER_DATA, concrete at capture).
    buf_gpu: u64,
    /// `bind_data`: each entry is `(syms, host_ptr, fmt)` — the symbolic field
    /// values `bind_args_state` records into the queue's `mv_sints`.
    /// Concrete buffer VAs / vals are written
    /// straight into the page at construction; only JIT-replaced inputs are
    /// symbolic.
    bind_data: Vec<(Vec<Sym>, *mut u8, Fmt)>,
}

impl AmdArgsState {
    /// Build the kernarg slot for one kernel: write buffer VAs (`fmt='Q'`) then
    /// scalar vals
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
        // Buffer VAs: 8 bytes each (`fmt='Q'`).
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
        // Scalar vals: 4 bytes each (`fmt='I'`); i64→i32 matching
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
    /// The dword stream. Concrete until `bind`, after which it lives in
    /// the host-visible page and `apply_var_vals` patches it in place.
    q: Vec<u32>,
    /// Distinct symbols in first-reference order.
    syms: Vec<Sym>,
    /// Per-symbol last-resolved value, parallel to `syms`; `apply_var_vals`
    /// skips patches whose symbol didn't change.
    prev_resolved: Vec<Option<u64>>,
    /// Symbolic dwords in `q`.
    q_sints: Vec<QSint>,
    /// Symbolic kernarg fields.
    mv_sints: Vec<MvSint>,
    /// `None` until `bind`; then the host page + indirect-buffer reference.
    binded: Option<Binded>,
    /// gfx major version of the capturing device (the graph is single-device,
    /// so it's fixed). The `signal`/`memory_barrier` cache encodings branch on
    /// `target_major == 9`, same as `exec`'s per-kernel `target_major`.
    target_major: u32,
}

// SAFETY: the host pointers in `mv_sints`/`binded` are stable host-visible
// mappings the graph owns for its lifetime; the only writers are `bind`
// (capture, single-threaded) and `submit` (replay, serialised by
// `AmdGraph::comp_queue` `Mutex<AmdHwQueue>`).
unsafe impl Send for AmdHwQueue {}
unsafe impl Sync for AmdHwQueue {}

impl AmdHwQueue {
    /// New empty queue. The connector is
    /// NOT held here — it's supplied by the owning `AmdGraph` (which holds the
    /// `ConnectorLease`) to `exec`/`submit`, so the lease stays the sole owner
    /// of the connector and can't be aliased.
    pub fn new(target_major: u32) -> Self {
        Self {
            q: Vec::new(),
            syms: Vec::new(),
            prev_resolved: Vec::new(),
            q_sints: Vec::new(),
            mv_sints: Vec::new(),
            binded: None,
            target_major,
        }
    }

    /// Intern a symbol, returning its index.
    fn new_sym(&mut self, sym: &Sym) -> usize {
        if let Some(i) = self.syms.iter().position(|s| s == sym) {
            return i;
        }
        self.syms.push(sym.clone());
        self.prev_resolved.push(None);
        self.syms.len() - 1
    }

    /// Enqueue values — concrete dwords verbatim, symbols recorded for later
    /// resolution.
    fn q(&mut self, values: Vec<QVal>) {
        for v in values {
            match v {
                QVal::Dword(d) => self.q.push(d),
                QVal::Sym { sym, shift, add } => {
                    let sym_idx = self.new_sym(&sym);
                    self.q_sints.push(QSint { off: self.q.len(), sym_idx, shift, add });
                    self.q.push(0xbadc_0ded); // placeholder
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

    // *** commands ***

    /// `wait_reg_mem`/`wait`: poll memory at `addr`
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

    /// `signal` = `release_mem(addr, value, cache_flush=true)`.
    /// Writes the low 32 bits of `value` to `addr` after a full system-scope
    /// cache flush. Both may be symbolic (the graph signals the virtual device
    /// timeline). Layout matches `pm4::release_mem`:
    /// `[hdr, event, memsel, addr_lo, addr_hi, value, value_hi, ctxid]`.
    fn signal(&mut self, addr: SymU64, value: SymU32) {
        // gfx9 (CDNA) and gfx10+ (RDNA) encode the cache flush differently, and
        // DST_SEL only exists on gfx10+ — same split as `pm4::release_mem`.
        let gfx9 = self.target_major == 9;
        let cache = if gfx9 { pm4::EOP_CACHE_FLUSH_GFX9 } else { pm4::RELEASE_MEM_CACHE_FLUSH_ALL };
        let event_dw = pm4::release_mem_event_type(pm4::CACHE_FLUSH_AND_INV_TS_EVENT)
            | pm4::release_mem_event_index(pm4::EVENT_INDEX_END_OF_PIPE)
            | cache;
        let mut memsel_dw = pm4::release_mem_data_sel(pm4::DATA_SEL_SEND_32_BIT_LOW)
            | pm4::release_mem_int_sel(pm4::INT_SEL_INTERRUPT_AFTER_WRITE);
        if !gfx9 {
            memsel_dw |= pm4::release_mem_dst_sel(pm4::DST_SEL_MEMORY);
        }
        let mut pkt =
            vec![QVal::Dword(pm4::packet3(pm4::PACKET3_RELEASE_MEM, 6)), QVal::Dword(event_dw), QVal::Dword(memsel_dw)];
        pkt.extend(self.addr64(&addr));
        pkt.push(value.into_qval());
        pkt.push(QVal::Dword(0)); // value_hi
        pkt.push(QVal::Dword(0)); // ctxid
        self.q(pkt);
    }

    /// `memory_barrier`: HDP-flush register handshake then
    /// a full `acquire_mem`. Concrete (no symbols).
    fn memory_barrier(&mut self) {
        self.q.extend_from_slice(&pm4::hdp_flush());
        if self.target_major == 9 {
            self.q.extend_from_slice(&pm4::acquire_mem_gfx9());
        } else {
            self.q.extend_from_slice(&pm4::acquire_mem());
        }
    }

    /// `exec` — the critical command. Records the kernarg
    /// slot's symbolic fields into `mv_sints`,
    /// then emits the exact dword sequence `build_exec_pm4` produces for a
    /// per-call dispatch: `acquire_mem(gli=0,gl2=0)` → PGM/RSRC/TMPRING/SCRATCH/
    /// RESTART/USER_DATA/RESOURCE_LIMITS/START_X regs → `DISPATCH_DIRECT` →
    /// `EVENT_WRITE(CS_PARTIAL_FLUSH)`. USER_DATA holds the concrete kernarg page
    /// VA + optional scratch prefix; same-queue ordering comes from the
    /// `acquire_mem` + `CS_PARTIAL_FLUSH`, so there is NO inter-kernel
    /// signal/wait.
    pub fn exec(
        &mut self,
        conn: &AmdConnector,
        prg: &AmdProgram,
        args: &AmdArgsState,
        global_size: [u32; 3],
        local_size: [u32; 3],
    ) {
        // bind_args_state: record symbolic kernarg fields.
        for (syms, mem, fmt) in &args.bind_data {
            for (i, sym) in syms.iter().enumerate() {
                let sym_idx = self.new_sym(sym);
                self.mv_sints.push(MvSint { host: *mem, elem: i, sym_idx, fmt: *fmt });
            }
        }

        // Read the graph connector's own scratch. The connector is held by the
        // owning graph's `ConnectorLease` (exclusive), so there's no concurrent
        // realloc to guard against.
        let scratch_addr = conn.scratch_gpu_va();
        let tmpring_size = conn.tmpring_size();

        // USER_DATA SGPR prefix: optional 4-dword scratch descriptor, then the
        // 2-dword kernarg pointer — identical to `AmdProgram::execute`
        // (`program.rs:646-655`).
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

    /// Preamble: `memory_barrier().wait(virt_timeline).wait(kick).signal(self)`.
    /// `self_sig_addr` is the per-graph signal the
    /// preamble sets to `kickoff`; the kick wait gates the whole IB until the
    /// host stages the replay and releases it by setting the kick signal.
    pub fn preamble(&mut self, kick_sig_addr: u64, self_sig_addr: u64) {
        self.memory_barrier();
        self.wait(SymU64::Sym(Sym::VirtTimelineSigAddr), SymU32::Sym(Sym::VirtTimelineVal), 0xFFFF_FFFF);
        self.wait(SymU64::Concrete(kick_sig_addr), SymU32::Sym(Sym::Kickoff), 0xFFFF_FFFF);
        self.signal(SymU64::Concrete(self_sig_addr), SymU32::Sym(Sym::Kickoff));
    }

    /// Final: `signal(virt_timeline_sig, virt_timeline_val + 1)`.
    /// Advances the real device timeline by exactly +1
    /// per replay (the preamble waited `timeline-1`), so the graph composes with
    /// per-call dispatch and `AmdDevice::synchronize`.
    pub fn final_signal(&mut self) {
        self.signal(SymU64::Sym(Sym::VirtTimelineSigAddr), SymU32::SymAdd(Sym::VirtTimelineVal, 1));
    }

    /// `bind`: allocate a host-visible uncached page, copy
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

    /// `apply_var_vals`: resolve every symbol, patch changed
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

    /// `submit`: apply `var_vals` then push to the ring. Patches the
    /// bound page's symbolic dwords + kernarg fields, then pushes the
    /// indirect-buffer reference with one doorbell.
    ///
    /// Per-graph connector ownership means there's no concurrent reader of
    /// the patched IB page to publish to — the doorbell store inside
    /// `submit_dwords::ring_doorbell` provides the host→GPU publication
    /// barrier.
    pub fn submit(&mut self, conn: &AmdConnector, var_vals: &VarVals) -> Result<()> {
        self.apply_var_vals(var_vals)?;
        let cmd = self.binded.as_ref().expect("AmdHwQueue::submit before bind").indirect_cmd;
        // Submit: one doorbell via the queue primitive.
        conn.queue().submit_dwords(&cmd)
    }
}

/// A 64-bit field that may be concrete or symbolic (addresses).
enum SymU64 {
    Concrete(u64),
    Sym(Sym),
}

/// A 32-bit symbolic field (timeline/kickoff values), optionally with a `+add`
/// (the `virt_timeline_val + 1` offset). Every graph wait/signal value is a
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
