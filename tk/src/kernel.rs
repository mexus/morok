//! The [`Kernel`] builder — the eager context that mints ranges, allocations,
//! and tiles, and assembles the final hand-lowered SINK.
//!
//! All builder methods take `&self` over interior-mutable counters/stacks, so
//! many tiles and [`crate::group::Group`]s can borrow one `&Kernel` at once —
//! the borrow-checker-friendly mapping of tinygrad's `Kernel` context manager.

use std::cell::{Cell, RefCell};
use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, KernelInfo, Op, UOp};

use crate::WARP_THREADS;
use crate::index::cidx;

pub struct Kernel {
    pub name: String,
    /// `blockIdx.{x,y,z}` as `Special` ops (only rendered if referenced).
    pub block_idx: [Arc<UOp>; 3],
    /// `threadIdx.x` as a `Special` op.
    pub thread_idx: Arc<UOp>,

    /// Flat (1-D pointer) `Param` placeholders, one per bound buffer in
    /// declaration order. The kernel body references these; the concrete buffers
    /// bind positionally at launch (`ProgramSpec.globals` slot order).
    globals: Vec<Arc<UOp>>,
    global_slot: Cell<usize>,
    shared_slot: Cell<usize>,
    range_id: Cell<usize>,

    /// Tracked ranges, closed together by [`Kernel::finish`] / [`Kernel::endrange`].
    range_stack: RefCell<Vec<Arc<UOp>>>,
    /// Terminal `(store, buffer)` pairs, consumed by `finish`/`endrange`.
    store_stack: RefCell<Vec<(Arc<UOp>, Arc<UOp>)>>,
}

impl Kernel {
    /// Build a kernel context bound to concrete realized buffers. `buffers` are
    /// the `BUFFER` UOps of the realized tensors in declaration order (output(s)
    /// first, then inputs). Each is converted to a flat 1-D `Param` placeholder
    /// (slot = declaration index); [`Kernel::next_global`] hands those out as GL
    /// tiles bind them, and [`crate::launch`] binds the concrete buffers
    /// positionally at dispatch — the svod analog of tinygrad `sink.call(bufs)`.
    pub fn new(name: impl Into<String>, grid: [i64; 3], block: i64, buffers: Vec<Arc<UOp>>) -> Self {
        let globals = buffers.iter().enumerate().map(|(slot, buf)| flat_param(slot, buf)).collect();
        let block_idx = [
            UOp::special(cidx(grid[0]), "gidx0".to_string()),
            UOp::special(cidx(grid[1]), "gidx1".to_string()),
            UOp::special(cidx(grid[2]), "gidx2".to_string()),
        ];
        let thread_idx = UOp::special(cidx(block), "lidx0".to_string());
        Kernel {
            name: name.into(),
            block_idx,
            thread_idx,
            globals,
            global_slot: Cell::new(0),
            shared_slot: Cell::new(0),
            range_id: Cell::new(0),
            range_stack: RefCell::new(Vec::new()),
            store_stack: RefCell::new(Vec::new()),
        }
    }

    // ── lane / warp helpers ────────────────────────────────────────────────

    pub fn warpid(&self) -> Arc<UOp> {
        self.thread_idx.try_div(&cidx(WARP_THREADS as i64)).expect("warpid: index div")
    }
    pub fn laneid(&self) -> Arc<UOp> {
        self.thread_idx.try_mod(&cidx(WARP_THREADS as i64)).expect("laneid: index mod")
    }

    // ── ranges ─────────────────────────────────────────────────────────────

    fn fresh_range(&self, end: i64, axis_type: AxisType) -> Arc<UOp> {
        let rid = self.range_id.get();
        self.range_id.set(rid + 1);
        UOp::range_axis(cidx(end), AxisId::Renumbered(rid), axis_type)
    }

    /// A tracked `Loop` range closed by `finish`.
    pub fn range(&self, end: i64) -> Arc<UOp> {
        let r = self.fresh_range(end, AxisType::Loop);
        self.range_stack.borrow_mut().push(r.clone());
        r
    }

    /// A tracked `Loop` range with a *dynamic* (runtime-valued) end — e.g. a
    /// `Special`-derived bound for causal block-skip (`q_seq + 1`) — closed by
    /// `finish`/`endrange` like [`Kernel::range`]. `end` must be `Index`-typed
    /// (or const-coercible; `UOp::range_axis` handles the coercion). The renderer
    /// lowers it to a real runtime-trip loop.
    pub fn range_uop(&self, end: Arc<UOp>) -> Arc<UOp> {
        let rid = self.range_id.get();
        self.range_id.set(rid + 1);
        let r = UOp::range_axis(end, AxisId::Renumbered(rid), AxisType::Loop);
        self.range_stack.borrow_mut().push(r.clone());
        r
    }

    /// A range with an explicit axis type; tracked only when `track`.
    pub fn range_typed(&self, end: i64, axis_type: AxisType, track: bool) -> Arc<UOp> {
        let r = self.fresh_range(end, axis_type);
        if track {
            self.range_stack.borrow_mut().push(r.clone());
        }
        r
    }

    /// An untracked range, closed manually via `store(..).end([r])`.
    pub fn raw_range(&self, end: i64, axis_type: AxisType) -> Arc<UOp> {
        self.fresh_range(end, axis_type)
    }

    /// The currently tracked (outer) ranges — tinygrad `ker.range_stack`. A
    /// reduction's per-iteration re-init must depend on these so it re-runs once
    /// per outer-loop iteration instead of hoisting above the enclosing loops.
    pub fn tracked_ranges(&self) -> SmallVec<[Arc<UOp>; 4]> {
        self.range_stack.borrow().iter().cloned().collect()
    }

    // ── allocations ────────────────────────────────────────────────────────

    /// Allocate shared (LDS) memory. The slot is a per-kernel monotonic id (the
    /// renderer names LDS `@local{id}`, so it MUST be unique within a kernel).
    pub fn alloc_local(&self, flat_size: usize, elem: DType) -> Arc<UOp> {
        let slot = self.shared_slot.get();
        self.shared_slot.set(slot + 1);
        UOp::define_local(slot, elem.ptr(Some(flat_size), AddrSpace::Local))
    }

    /// Allocate register (per-lane) memory (auto-unique id).
    pub fn alloc_reg(&self, flat_size: usize, elem: DType) -> Arc<UOp> {
        UOp::define_reg_typed(flat_size, elem)
    }

    /// Hand out the next global buffer placeholder (a flat 1-D `Param`) as a GL
    /// tile binds it. Already flat — no `flat_ptr` unwrap is needed.
    pub fn next_global(&self) -> Arc<UOp> {
        let slot = self.global_slot.get();
        self.global_slot.set(slot + 1);
        self.globals[slot].clone()
    }

    // ── store bookkeeping / finalization ───────────────────────────────────

    /// Record a terminal `(store, buffer)` pair for `finish`/`endrange`.
    pub fn push_store(&self, store: Arc<UOp>, buf: Arc<UOp>) {
        self.store_stack.borrow_mut().push((store, buf));
    }

    /// Close every tracked range and group the last `stores` terminal stores
    /// into the final kernel SINK (carrying `opts_to_apply = Some(vec![])` so
    /// the optimizer leaves this hand-lowered body untouched).
    pub fn finish(&self, stores: usize) -> Arc<UOp> {
        let rngs: SmallVec<[Arc<UOp>; 4]> = self.range_stack.borrow_mut().drain(..).collect();

        let mut store_uops = Vec::with_capacity(stores);
        for _ in 0..stores {
            let (store, _buf) = self.store_stack.borrow_mut().pop().expect("finish: store stack underflow");
            store_uops.push(store);
        }
        store_uops.reverse(); // restore declaration order

        // Each terminal store is already an `END(STORE)` / `END(GROUP(STORE..))`
        // closing its own loops (the Group ops self-end so their `After`-rewraps
        // carry a completed-loop edge). svod's GROUP may only hold *bare* STOREs,
        // so we don't re-wrap these in a GROUP; instead close any remaining
        // tracked (outer) ranges around each — a no-op `END` when `rngs` is empty
        // (the matmul, whose tile loop `endrange` already consumed) — and SINK
        // them directly (the native `SINK(END(STORE, ..))` kernel shape).
        let sources: Vec<Arc<UOp>> = store_uops.into_iter().map(|s| s.end(rngs.clone())).collect();
        UOp::sink_with_info(sources, KernelInfo { opts_to_apply: Some(vec![]) })
    }

    /// Close `ranges` inner (accumulation) loops around the last store and
    /// return the store's buffer rewrapped with the close as a dependency.
    pub fn endrange(&self, ranges: usize) -> Arc<UOp> {
        let (store, buf) = self.store_stack.borrow_mut().pop().expect("endrange: store stack underflow");
        let mut rngs: Vec<Arc<UOp>> = Vec::with_capacity(ranges);
        for _ in 0..ranges {
            rngs.push(self.range_stack.borrow_mut().pop().expect("endrange: range stack underflow"));
        }
        let ended = store.end(SmallVec::from_vec(rngs));
        buf.after(smallvec![ended])
    }

    /// Like [`Self::endrange`] but returns the loop-closing `END` node directly
    /// (rather than one rewrapped buffer), so several accumulators sharing one K
    /// loop can each be rewrapped `.after([end])` to read the final value
    /// outside the loop. Only the last store is ended (a `RANGE` may have a
    /// single `END`, else a double loop footer): the caller must chain the
    /// other accumulators' stores into it (via a shared input) so they are
    /// scoped inside the loop and survive dead-code elimination.
    pub fn endrange_to(&self, ranges: usize) -> Arc<UOp> {
        let (store, _buf) = self.store_stack.borrow_mut().pop().expect("endrange_to: store stack underflow");
        let mut rngs: Vec<Arc<UOp>> = Vec::with_capacity(ranges);
        for _ in 0..ranges {
            rngs.push(self.range_stack.borrow_mut().pop().expect("endrange_to: range stack underflow"));
        }
        store.end(SmallVec::from_vec(rngs))
    }
}

/// Mint a flat 1-D `Param` placeholder for a concrete `BUFFER` UOp at `slot`.
///
/// The buffer's element dtype + flat size become a `Ptr` param (global address
/// space), exactly the shape `placeholder_like` builds for a rank-≤1 source.
/// Keeping it flat (no RESHAPE wrapper) means GL tiles index it directly — the
/// renderer's `globals` derivation counts this `Param` and `launch` binds the
/// concrete buffer to its slot. A reshaped/lazy source is unwrapped to its base
/// `BUFFER` first, and a non-buffer source (already a `Param`/`Ptr`) is reused.
fn flat_param(slot: usize, src: &Arc<UOp>) -> Arc<UOp> {
    let base = src.base();
    match base.op() {
        Op::Buffer { size, .. } => {
            let elem = base.dtype();
            UOp::param(slot, *size, elem.ptr(Some(*size), AddrSpace::Global), None)
        }
        // Already a buffer-like pointer (e.g. a pre-built Param): reuse as-is.
        _ => base,
    }
}
