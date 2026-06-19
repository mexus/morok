//! The [`Group`] — a cooperating set of warps that owns the movement, load,
//! store, and WMMA ops, ported from tinygrad `extra/thunder/tiny/tk/group.py`.
//!
//! Each op opens its own *untracked* loops ([`Kernel::raw_range`]), builds the
//! terminal store closing those loops (`store.end(ranges)`, with a workgroup
//! `barrier` for the coalesced GLOBAL→LOCAL fill), records it via
//! [`Kernel::push_store`], and returns the destination tile *rewrapped* with an
//! `After([END(STORE)])` dependency — so a later read of the tile is ordered
//! after the write (tinygrad's `dst.after(dst_store)`).

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::{AmdArch, DType};
use svod_ir::{AxisType, ConstValue, RendererDevice, UOp, WmmaMetadata, WmmaUpcastAxes};
use svod_schedule::optimizer::{Renderer, TensorCore};

use crate::index::{Idx, cidx, flat_index, flat_offset, index_off, load_at, load_off, load_vec};
use crate::kernel::Kernel;
use crate::tile::{GL, RT, RV, RegTile, ST};
use crate::tiles::TileLayout;

/// The index inputs to a [`Group::load`]/[`Group::store`], named by ROLE so the two
/// ops no longer disagree on positional slots. `block` is the wave sub-tile / global
/// block offset (the old `idxs`); `frag` is the REG-side fragment offset (the old
/// `dst_idxs` on load / `src_idxs` on store). `axis` is the global-tile row-stride
/// split (ignored by the LOCAL↔REG hops). Borrows the slices for the call.
#[derive(Clone, Copy, Default)]
pub struct MoveIdx<'a> {
    pub block: &'a [Idx],
    pub frag: &'a [Idx],
    pub axis: usize,
}

impl<'a> MoveIdx<'a> {
    /// A wave/global `block` offset at `axis` (the common fill/gather/store case).
    pub fn block(block: &'a [Idx], axis: usize) -> Self {
        Self { block, frag: &[], axis }
    }
    /// A REG-side `frag` offset only.
    pub fn frag(frag: &'a [Idx]) -> Self {
        Self { frag, block: &[], axis: 0 }
    }
    /// Both a `block` and a `frag` offset at `axis`.
    pub fn at(block: &'a [Idx], frag: &'a [Idx], axis: usize) -> Self {
        Self { block, frag, axis }
    }
}

/// A source tile that can be loaded INTO `Dst` by a [`Group`]. The legal
/// address-space pairs each have an impl (ST←GL, RT←ST, RT←GL); an illegal pair is a
/// *compile* error (no impl), not a runtime panic. `Output` is the rewrapped dst.
pub trait LoadInto<'k, Dst> {
    type Output;
    fn load_into(self, g: &Group<'k>, dst: Dst, ix: MoveIdx<'_>) -> Self::Output;
}

/// A source REG tile that can be stored INTO `Dst` (`ST` or `GL`); illegal pairs are
/// a compile error.
pub trait StoreInto<'k, Dst> {
    type Output;
    fn store_into(self, g: &Group<'k>, dst: Dst, ix: MoveIdx<'_>) -> Self::Output;
}

impl<'k> LoadInto<'k, ST<'k>> for GL<'k> {
    type Output = ST<'k>;
    fn load_into(self, g: &Group<'k>, dst: ST<'k>, ix: MoveIdx<'_>) -> ST<'k> {
        g.load_global_to_local(dst, &self, ix.block, ix.axis, true)
    }
}
impl<'k> LoadInto<'k, RT<'k>> for ST<'k> {
    type Output = RT<'k>;
    fn load_into(self, g: &Group<'k>, dst: RT<'k>, ix: MoveIdx<'_>) -> RT<'k> {
        g.load_local_to_reg(dst, &self, ix.frag, ix.block)
    }
}
impl<'k> LoadInto<'k, RT<'k>> for GL<'k> {
    type Output = RT<'k>;
    fn load_into(self, g: &Group<'k>, dst: RT<'k>, ix: MoveIdx<'_>) -> RT<'k> {
        g.load_global_to_reg(dst, &self, ix.frag, ix.block, ix.axis)
    }
}
impl<'k> StoreInto<'k, ST<'k>> for RT<'k> {
    type Output = ST<'k>;
    fn store_into(self, g: &Group<'k>, dst: ST<'k>, ix: MoveIdx<'_>) -> ST<'k> {
        g.store_reg_to_local(dst, &self, ix.block, ix.frag)
    }
}
impl<'k> StoreInto<'k, GL<'k>> for RT<'k> {
    type Output = GL<'k>;
    fn store_into(self, g: &Group<'k>, dst: GL<'k>, ix: MoveIdx<'_>) -> GL<'k> {
        g.store_reg_to_global(dst, &self, ix.block, ix.frag, ix.axis)
    }
}

// ── Index (i64-typed) arithmetic helpers ───────────────────────────────────

fn idiv(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_div(&cidx(k)).expect("idiv")
}
fn imod(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_mod(&cidx(k)).expect("imod")
}
fn imul(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    if k == 1 { a.clone() } else { a.try_mul(&cidx(k)).expect("imul") }
}
fn iadd(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    a.try_add(b).expect("iadd")
}
fn ixor(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_xor_op(&cidx(k)).expect("ixor")
}
fn iand(a: &Arc<UOp>, k: i64) -> Arc<UOp> {
    a.try_and_op(&cidx(k)).expect("iand")
}

/// Compare-exchange direction for [`Group::compare_exchange`] (sorting networks).
#[derive(Clone, Copy, Debug)]
pub enum SwapDir {
    /// The lower-index lane of each pair keeps the min (the larger goes high).
    Ascending,
    /// The lower-index lane keeps the max.
    Descending,
    /// Bitonic merge: ascending where `(laneid & bit) == 0`, else descending.
    ByLaneBit(i64),
}
fn idx_mul(idx: &Idx, k: i64) -> Idx {
    match idx {
        Idx::Const(c) => Idx::Const(c * k),
        Idx::Uop(u) => Idx::Uop(imul(u, k)),
    }
}

/// The wave sub-tile fragment index for a shared-tile axis (SI-1):
/// `block * frags + local`, where `block` (a wave's row/col in the wave grid,
/// already including `warp_row`/`warp_col`) selects which `frags`-tall slice of
/// the shared tile this wave reads/writes. `None` ⇒ no offset (single-warp).
fn wave_offset(block: Option<&Idx>, frags: i64, local: &Arc<UOp>) -> Idx {
    match block {
        None => Idx::from(local),
        Some(b) => Idx::Uop(iadd(&imul(&b.to_uop(), frags), local)),
    }
}

/// The per-lane (row, col) within a base fragment. `transpose` selects the
/// "fragment is laid out column-major in registers" branch (group.py: either
/// `rt.layout != st.layout` for the LDS hops, or `rt.layout == COL` for the
/// global hops); `inner` is the upcast element index.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lane_rc(
    transpose: bool,
    interleave: bool,
    interleave_t: bool,
    laneid: &Arc<UOp>,
    rows: i64,
    cols: i64,
    stride: i64,
    inner: &Arc<UOp>,
) -> (Arc<UOp>, Arc<UOp>) {
    if interleave_t {
        // The transpose of the RDNA accumulator interleave: `row = lane%16,
        // col = 2·j + lane/16`. Stores an even/odd-interleaved accumulator to memory
        // along the transposed (N-major) axis — the FA output tile `O[q,d]` from the
        // `[d,q]` PV accumulator. Checked before `interleave` (and ignores
        // `transpose`) so it never perturbs the matmul accumulator store.
        return (imod(laneid, cols), iadd(&imul(inner, 2), &idiv(laneid, cols)));
    }
    if interleave {
        // RDNA wave32 WMMA f32 accumulator: even/odd row interleave across the two
        // wave-halves — `m = 2·j + lane/16, n = lane%16` (j = `inner`; the ×2 is the
        // wave32 subgroup count `wave_size/16`). The lane-half is the +1 unit and
        // the register the ×2 — the opposite weighting from the stride branches, so
        // it can't be expressed as a stride (tinygrad `ops_python` RDNA3 `c_map`).
        return (iadd(&imul(inner, 2), &idiv(laneid, cols)), imod(laneid, cols));
    }
    if transpose {
        (iadd(&imul(&idiv(laneid, cols), stride), inner), imod(laneid, cols))
    } else {
        (imod(laneid, rows), iadd(&imul(&idiv(laneid, rows), stride), inner))
    }
}

/// Bridge a scheduler [`TensorCore`] (the per-arch×dtype matrix-op table, the
/// single source of truth — `schedule::optimizer::renderer`) into the IR
/// [`WmmaMetadata`] that a hand-built [`Op::Wmma`](svod_ir::Op::Wmma) consumes.
///
/// `dims`/`dtype_in`/`dtype_out`/`threads`/`tile_grid` copy straight across. The
/// `upcast_axes` are `log2(elements_per_thread)` size-2 entries per operand
/// (mirrors the optimizer's `tc.rs` construction, where every upcast/reduce split
/// is by 2); the axis-id *values* are cosmetic on tk's expander-free direct path —
/// codegen/devectorize read only the sizes — so we descend from 4, which
/// reproduces the prior hand layout `[(4,2),(3,2)]` for the gfx942 16×16×16 case.
/// `reduce_axes` is empty: tk's `mma` carries the K reduce as its own `inner`
/// range, not inside the WMMA metadata.
fn wmma_from_tc(tc: &TensorCore, device: RendererDevice) -> WmmaMetadata {
    let axes = |ept: usize| -> Vec<(usize, usize)> { (0..(ept as f64).log2() as usize).map(|i| (4 - i, 2)).collect() };
    WmmaMetadata {
        name: format!("WMMA_{}_{}_{}_{:?}_{:?}", tc.dims.0, tc.dims.1, tc.dims.2, tc.dtype_in, tc.dtype_out),
        dims: tc.dims,
        dtype_in: tc.dtype_in.clone(),
        dtype_out: tc.dtype_out.clone(),
        device,
        threads: tc.threads,
        upcast_axes: WmmaUpcastAxes {
            a: axes(tc.elements_per_thread.0),
            b: axes(tc.elements_per_thread.1),
            c: axes(tc.elements_per_thread.2),
        },
        reduce_axes: vec![],
        tile_grid: tc.tile_grid,
    }
}

/// The K=16 WMMA descriptor for input dtype `dtype_in` on `arch`, looked up from
/// the shared per-arch tensor-core table (`Renderer::for_amd_arch`) rather than
/// re-encoded here — so bf16/f16 on CDNA's MFMA cores and the RDNA wave32 cores
/// come from one source. Both accumulate in f32. `arch` is threaded from
/// [`crate::ArchCaps::arch`] (gfx942 in practice; the table already carries the
/// RDNA cores for when a wave32 arch is enabled).
fn wmma_desc(arch: AmdArch, dtype_in: &DType) -> WmmaMetadata {
    let ren = Renderer::for_amd_arch(arch);
    let tc =
        ren.tensor_cores.iter().find(|tc| &tc.dtype_in == dtype_in && tc.dims == (16, 16, 16)).unwrap_or_else(|| {
            // Precondition violation by the kernel author, not end-user input: the
            // matrix-core operand dtype must be bf16/f16 (the only 16×16×16 WMMA
            // inputs). The USE-face kernels pre-cast; an AUTHOR calling `mma_*`
            // with an unsupported RT dtype lands here.
            unimplemented!(
                "mma: operand dtype {dtype_in:?} has no 16×16×16 WMMA on {arch:?} — operands must be bf16 or f16"
            )
        });
    wmma_from_tc(tc, ren.device)
}

/// Per-lane element count for a WMMA operand = product of its upcast-axis sizes
/// (`wmma_from_tc` builds these as `log2(elements_per_thread)` size-2 entries, so
/// the product is the elements-per-thread). gfx942 16×16×16 → A/B/C = 4/4/4; RDNA
/// → 16/16/8 (replicated 16-wide inputs, 8-wide accumulator). Empty axes ⇒ 1.
fn upcast_count(axes: &[(usize, usize)]) -> i64 {
    axes.iter().map(|(_, sz)| *sz as i64).product()
}

/// Scalar geometry of the coalesced GLOBAL↔LDS fill for one ST tile (the part
/// independent of the global source / tile position). Shared by the direct fill
/// and the register-staged prefetch so both address LDS identically.
struct LdsGeom {
    ept: i64,
    st_cols: i64,
    memcpy_per_row: i64,
    base_rows: i64,
    base_cols: i64,
    total_calls: i64,
    num_valid: i64,
    clamp: bool,
}

/// A cooperating set of `warps` waves laid out in a `rows_waves × cols_waves`
/// grid (tinygrad `Group` / HK `group<NUM_WARPS>`). Each wave owns a sub-tile of
/// the shared tiles; the GLOBAL→LDS fill is collaborative over all
/// `group_threads`.
pub struct Group<'k> {
    pub warps: usize,
    pub rows_waves: usize,
    pub cols_waves: usize,
    group_threads: usize,
    ker: &'k Kernel,
}

impl Kernel {
    /// A single-warp group (tinygrad `ker.warp`).
    pub fn warp(&self) -> Group<'_> {
        self.group_2d(1, 1)
    }
    /// An `n`-warp group laid out `1×n` (tinygrad `ker.group`).
    pub fn group(&self, n: usize) -> Group<'_> {
        self.group_2d(1, n)
    }
    /// An `R×C`-wave group: one workgroup runs `rows_waves * cols_waves` waves
    /// (`group_threads = warps * 64`), each owning a sub-tile of the shared
    /// tiles (HK 2×4 wave grid, `GEMM:67-68`).
    pub fn group_2d(&self, rows_waves: usize, cols_waves: usize) -> Group<'_> {
        let warps = rows_waves * cols_waves;
        Group { warps, rows_waves, cols_waves, group_threads: warps * self.caps.wave_size, ker: self }
    }
}

impl<'k> Group<'k> {
    /// The group lane id (`threadIdx % group_threads`).
    fn laneid(&self) -> Arc<UOp> {
        imod(&self.ker.thread_idx, self.group_threads as i64)
    }

    /// Total threads in the workgroup (`warps * 64`) — the launch block size.
    pub fn group_threads(&self) -> usize {
        self.group_threads
    }

    /// The wave's flat index within the group (`(threadIdx % group_threads)/64`).
    pub fn warpid_in_group(&self) -> Arc<UOp> {
        idiv(&imod(&self.ker.thread_idx, self.group_threads as i64), self.ker.caps.wave_size as i64)
    }
    /// The wave's row in the `rows_waves × cols_waves` wave grid (`GEMM:67`).
    pub fn warp_row(&self) -> Arc<UOp> {
        idiv(&self.warpid_in_group(), self.cols_waves as i64)
    }
    /// The wave's column in the wave grid (`GEMM:68`).
    pub fn warp_col(&self) -> Arc<UOp> {
        imod(&self.warpid_in_group(), self.cols_waves as i64)
    }

    // ── single-warp register ops ────────────────────────────────────────────

    /// Anchor a constant-address unrolled **read** to the enclosing rolled
    /// (tracked) loops, so loop-invariant code motion cannot hoist a read of a
    /// loop-*carried* register out of the loop. The looped primitives dodge this
    /// incidentally (their loop-variable index makes the read non-hoistable); the
    /// unrolled bodies use constant indices, so a read of a carried accumulator
    /// (`max_vec`, `o_reg`, …) would otherwise be lifted to the entry block and
    /// see the *initial* value every iteration. A no-op when looped or when there
    /// is no enclosing tracked loop. (Over-anchors genuinely loop-invariant
    /// read-only tiles — harmless: a redundant ordering edge.)
    pub(crate) fn anchor(&self, buf: &Arc<UOp>) -> Arc<UOp> {
        let tracked = self.ker.tracked_ranges();
        if self.ker.unrolled() && !tracked.is_empty() { buf.after(tracked) } else { buf.clone() }
    }

    /// Build a per-element register op body — one bare `STORE` per logical
    /// element. Looped (the default): open a `Loop` `RANGE` per dim and close one
    /// store around them. Fully **unrolled** (the kernel's [`Kernel::unrolled`]
    /// flag): emit a bare store per element position, grouped into one node (no
    /// `RANGE`), so the body renders flat for the FA scheduling comb. `store_at`
    /// builds one element's `STORE` from its index tuple.
    fn elementwise<F>(&self, shape: &[usize], store_at: F) -> Arc<UOp>
    where
        F: Fn(&[Idx]) -> Arc<UOp>,
    {
        if self.ker.unrolled() {
            let stores: Vec<Arc<UOp>> = cartesian(shape).iter().map(|idxs| store_at(idxs)).collect();
            if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) }
        } else {
            let rngs: Vec<Arc<UOp>> = shape.iter().map(|&d| self.ker.raw_range(d as i64, AxisType::Loop)).collect();
            let idxs: Vec<Idx> = rngs.iter().map(Idx::from).collect();
            store_at(&idxs).end(SmallVec::from_vec(rngs))
        }
    }

    /// Fill a register tile with `value` (tinygrad `clear`).
    fn clear(&self, reg: RT<'k>, value: f64) -> RT<'k> {
        // Per-lane register fill: identical on every wave (each clears its own RT).
        let (buf, shape, is_float, elem) =
            (reg.uop().clone(), reg.shape().to_vec(), reg.elem().is_float(), reg.elem().clone());
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let cv = if is_float { ConstValue::Float(value) } else { ConstValue::Int(value as i64) };
            flat_index(&buf, &shape, idxs).store(UOp::const_(elem.clone(), cv))
        });
        self.finalize_reg(reg, ended)
    }

    /// Zero a register tile.
    pub fn zero(&self, reg: RT<'k>) -> RT<'k> {
        self.clear(reg, 0.0)
    }
    /// Fill a register tile with `1` (tinygrad `ones`).
    pub fn ones(&self, reg: RT<'k>) -> RT<'k> {
        self.clear(reg, 1.0)
    }
    /// Fill a register tile with `-∞` (tinygrad `neg_inf`).
    pub fn neg_inf(&self, reg: RT<'k>) -> RT<'k> {
        self.clear(reg, f64::NEG_INFINITY)
    }
    /// Fill a register *vector* with `value` (the [`RV`] analog of [`clear`]).
    pub fn clear_rv(&self, rv: RV<'k>, value: f64) -> RV<'k> {
        assert_eq!(self.warps, 1, "clear_rv is a single-warp op");
        let (buf, shape, is_float, elem) =
            (rv.uop().clone(), rv.shape().to_vec(), rv.elem().is_float(), rv.elem().clone());
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let cv = if is_float { ConstValue::Float(value) } else { ConstValue::Int(value as i64) };
            flat_index(&buf, &shape, idxs).store(UOp::const_(elem.clone(), cv))
        });
        self.finalize_tile(rv, ended)
    }
    /// Zero a register vector.
    pub fn zero_rv(&self, rv: RV<'k>) -> RV<'k> {
        self.clear_rv(rv, 0.0)
    }
    /// Fill a register vector with `1`.
    pub fn ones_rv(&self, rv: RV<'k>) -> RV<'k> {
        self.clear_rv(rv, 1.0)
    }
    /// Fill a register vector with `-∞`.
    pub fn neg_inf_rv(&self, rv: RV<'k>) -> RV<'k> {
        self.clear_rv(rv, f64::NEG_INFINITY)
    }

    /// Copy `src` into `dst` element-wise (tinygrad `copy`), casting on a dtype
    /// mismatch. Generic over [`RT`]/[`RV`] (softmax copies a register vector).
    pub fn copy<T: RegTile<'k>>(&self, dst: T, src: &T) -> T {
        // Per-lane register op: wave-safe (each wave copies its own RT).
        assert_eq!(dst.shape(), src.shape(), "copy: shape mismatch");
        let (sbuf, sshape, selem) = (self.anchor(src.uop()), src.shape().to_vec(), src.elem().clone());
        let (dbuf, dshape, delem) = (dst.uop().clone(), dst.shape().to_vec(), dst.elem().clone());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let mut load = load_at(&sbuf, &sshape, idxs);
            if selem != delem {
                load = load.cast(delem.clone());
            }
            flat_index(&dbuf, &dshape, idxs).store(load)
        });
        self.finalize_tile(dst, ended)
    }

    /// Transpose `src` into `dst` element-wise (tinygrad `transpose`): write
    /// `src[h, w, inner]` to `dst[w, h, inner]`, casting on a dtype mismatch.
    /// Used by FA to swap a register fragment's height/width before a WMMA.
    pub fn transpose(&self, dst: RT<'k>, src: &RT<'k>) -> RT<'k> {
        // Per-lane register op: wave-safe (each wave transposes its own RT).
        let (sbuf, sshape, selem) = (self.anchor(src.uop()), src.shape().to_vec(), src.elem().clone());
        let (dbuf, dshape, delem) = (dst.uop().clone(), dst.shape().to_vec(), dst.elem().clone());
        // Iterate the source `[height, width, inner]`; write `dst[width, height, inner]`.
        let ended = self.elementwise(&sshape.clone(), move |idxs| {
            let mut load = load_at(&sbuf, &sshape, idxs);
            if selem != delem {
                load = load.cast(delem.clone());
            }
            flat_index(&dbuf, &dshape, &[idxs[1].clone(), idxs[0].clone(), idxs[2].clone()]).store(load)
        });
        self.finalize_reg(dst, ended)
    }

    // ── elementwise map (tinygrad `Group.map`) ───────────────────────────────

    /// Apply `op` to every element of a register tile (tinygrad `Group.map`):
    /// open a loop per logical dim, load the element, store `op(value, idx)`
    /// back, and rewrap the tile after the store. `idx` lets `op` index *other*
    /// tiles at the same position (the RV-broadcast and FA causal-mask path).
    pub fn map<T, F>(&self, a: T, op: F) -> T
    where
        T: RegTile<'k>,
        F: Fn(&Arc<UOp>, &[Idx]) -> Arc<UOp>,
    {
        // Per-lane register op: wave-safe (each wave maps its own RT).
        let (buf, shape) = (a.uop().clone(), a.shape().to_vec());
        let rbuf = self.anchor(&buf); // anchored read; store to the raw buffer
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let val = load_at(&rbuf, &shape, idxs);
            flat_index(&buf, &shape, idxs).store(op(&val, idxs))
        });
        self.finalize_tile(a, ended)
    }

    // ── cross-lane reductions (tinygrad `row_reduce` / `col_reduce`) ──────────

    /// Reduce each row of `src` into `vec` (tinygrad `row_reduce`): per
    /// row-tile `height`, fold `op` over the `(width, inner)` lane-local
    /// elements into a 1-element REG accumulator, publish it to an LDS scratch
    /// slot at this lane, `barrier`, then fold the three sibling 16-lane slots
    /// (`(laneid + (1+i)*16) % group_threads`) to complete the warp-wide reduce,
    /// and fold the result into `vec[height]`.
    pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init_value: f64) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let n = src.shape().len();
        self.reduce(vec, src, op, init_value, src.shape()[n - 3] as i64, src.shape()[n - 2] as i64, true)
    }

    /// Reduce each column of `src` into `vec` (tinygrad `col_reduce`): the
    /// transpose of [`row_reduce`] — outer loop over column-tiles, accumulate
    /// over the `(height, inner)` elements.
    pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init_value: f64) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let n = src.shape().len();
        self.reduce(vec, src, op, init_value, src.shape()[n - 2] as i64, src.shape()[n - 3] as i64, false)
    }

    /// Read this lane's `value` from lane `src_lane` within the wave (gfx9
    /// wave64) via `llvm.amdgcn.ds.bpermute` — an in-register cross-lane gather
    /// with no LDS and no barrier. The intrinsic is i32-typed (lane `L` receives
    /// `data` from lane `byte_addr(L) >> 2`), so f32 is bitcast through i32 and
    /// the byte address is `src_lane * 4`. Emitted via the typed `Op::Custom`
    /// path (the `declare` is auto-hoisted+deduped to the module prefix).
    fn shuffle_lane(&self, value: &Arc<UOp>, src_lane: &Arc<UOp>) -> Arc<UOp> {
        let is_f32 = value.dtype() == DType::Float32;
        let data_i = if is_f32 { value.bitcast(DType::Int32) } else { value.clone() };
        let addr = imul(src_lane, 4).cast(DType::Int32);
        let sh = UOp::custom(
            smallvec![addr, data_i],
            "declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)\n\
             call i32 @llvm.amdgcn.ds.bpermute(i32 {0}, i32 {1})"
                .to_string(),
            DType::Int32,
        );
        if is_f32 { sh.bitcast(DType::Float32) } else { sh }
    }

    /// Per-element cross-lane gather (the public face of [`Self::shuffle_lane`]): for
    /// each logical element, `dst` receives `src`'s value at the SAME position but
    /// from lane `src_lane(laneid)`. Single-warp; one `ds_bpermute` per element (no
    /// LDS, no barrier). The shared foundation for `shuffle_xor`/`compare_exchange`
    /// (and, later, scan / arg-reduce). f32 (bitcast) and i32 transports are
    /// supported today; f16/bf16/i64 are a follow-up.
    pub fn shuffle<F>(&self, dst: RT<'k>, src: &RT<'k>, src_lane: F) -> RT<'k>
    where
        F: Fn(&Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "shuffle is a single-warp op");
        assert_eq!(dst.shape(), src.shape(), "shuffle: shape mismatch");
        let sl = src_lane(&self.laneid());
        let (sbuf, sshape) = (self.anchor(src.uop()), src.shape().to_vec());
        let (dbuf, dshape) = (dst.uop().clone(), dst.shape().to_vec());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let v = load_at(&sbuf, &sshape, idxs);
            flat_index(&dbuf, &dshape, idxs).store(self.shuffle_lane(&v, &sl))
        });
        self.finalize_reg(dst, ended)
    }

    /// Butterfly exchange: `dst[pos] = src[pos]` from lane `laneid ^ mask`. Arch-blind
    /// — for any `mask < wave_size` the XOR partner stays in `[0, wave_size)`, so no
    /// modulus is needed (cheaper than [`Self::shuffle_down`]). The sort/reduce primitive.
    pub fn shuffle_xor(&self, dst: RT<'k>, src: &RT<'k>, mask: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "shuffle_xor mask {mask} must be in 1..{w}");
        self.shuffle(dst, src, |laneid| ixor(laneid, mask))
    }

    /// Shift down: `dst[L] = src[(L + delta) mod wave_size]`.
    pub fn shuffle_down(&self, dst: RT<'k>, src: &RT<'k>, delta: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(delta > 0 && delta < w, "shuffle_down delta {delta} must be in 1..{w}");
        self.shuffle(dst, src, move |laneid| imod(&iadd(laneid, &cidx(delta)), w))
    }

    /// Shift up: `dst[L] = src[(L - delta) mod wave_size]` (the scan primitive).
    pub fn shuffle_up(&self, dst: RT<'k>, src: &RT<'k>, delta: i64) -> RT<'k> {
        let w = self.ker.caps.wave_size as i64;
        assert!(delta > 0 && delta < w, "shuffle_up delta {delta} must be in 1..{w}");
        self.shuffle(dst, src, move |laneid| imod(&iadd(laneid, &cidx(w - delta)), w))
    }

    /// One bitonic compare-exchange stage across the butterfly partner `laneid ^
    /// mask`: each lane keeps the min or max of its element and the partner's, per
    /// `dir` — the building block of sorting networks. Per element: one `ds_bpermute`
    /// gather + an ALU min/max select (no LDS, no barrier).
    pub fn compare_exchange(&self, dst: RT<'k>, src: &RT<'k>, mask: i64, dir: SwapDir) -> RT<'k> {
        assert_eq!(self.warps, 1, "compare_exchange is a single-warp op");
        assert_eq!(dst.shape(), src.shape(), "compare_exchange: shape mismatch");
        let w = self.ker.caps.wave_size as i64;
        assert!(mask > 0 && mask < w, "compare_exchange mask {mask} must be in 1..{w}");
        let laneid = self.laneid();
        let partner = ixor(&laneid, mask);
        // `keep_min`: this lane keeps the smaller of the pair (else the larger). The
        // lower-index lane of a pair is `(laneid & mask) == 0`.
        let is_low = iand(&laneid, mask).try_cmpeq(&cidx(0)).expect("ce is_low");
        let keep_min = match dir {
            SwapDir::Ascending => is_low,
            SwapDir::Descending => iand(&laneid, mask).try_cmpne(&cidx(0)).expect("ce desc"),
            // Bitonic merge: ascending where `(laneid & bit) == 0`. Keep min iff the
            // low-lane flag equals the ascending flag.
            SwapDir::ByLaneBit(bit) => {
                let asc = iand(&laneid, bit).try_cmpeq(&cidx(0)).expect("ce dir bit");
                is_low.try_cmpeq(&asc).expect("ce keep_min")
            }
        };
        let (sbuf, sshape) = (self.anchor(src.uop()), src.shape().to_vec());
        let (dbuf, dshape) = (dst.uop().clone(), dst.shape().to_vec());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let v = load_at(&sbuf, &sshape, idxs);
            let p = self.shuffle_lane(&v, &partner);
            let lt = v.try_cmplt(&p).expect("ce lt");
            let mn = UOp::try_where(lt, v.clone(), p.clone()).expect("ce min");
            let mx = v.try_max(&p).expect("ce max");
            let out = UOp::try_where(keep_min.clone(), mn, mx).expect("ce select");
            flat_index(&dbuf, &dshape, idxs).store(out)
        });
        self.finalize_reg(dst, ended)
    }

    /// Shared reduction body. `outer_end` is the tile dim mapped to `vec`
    /// (row-tiles for `row_reduce`, col-tiles for `col_reduce`); `acc_end` is the
    /// in-lane reduce dim; `row` selects the `src[outer, acc, inner]` vs
    /// `src[acc, outer, inner]` element order.
    #[allow(clippy::too_many_arguments)]
    fn reduce<F>(
        &self,
        vec: RV<'k>,
        src: &RT<'k>,
        op: F,
        init_value: f64,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        assert_eq!(self.warps, 1, "reduce is a single-warp op");
        if self.ker.unrolled() {
            return self.reduce_u(vec, src, op, init_value, outer_end, acc_end, row);
        }
        let elem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let red_reg = self.ker.alloc_reg(1, elem.clone());
        let laneid = self.laneid();

        let read0 = |buf: &Arc<UOp>| load_at(buf, &[1], &[Idx::Const(0)]);
        let init_val = UOp::const_(elem.clone(), ConstValue::Float(init_value));

        let outer = self.ker.raw_range(outer_end, AxisType::Loop);

        // Re-init the REG accumulator each outer iteration: the init store must
        // depend on `outer` (and the enclosing tracked loops), or it hoists above
        // them and the accumulator carries stale state across iterations.
        let mut init_deps: SmallVec<[Arc<UOp>; 4]> = smallvec![outer.clone()];
        init_deps.extend(self.ker.tracked_ranges());
        let init_buf = red_reg.after(init_deps);
        let i = self.ker.raw_range(1, AxisType::Loop);
        let mut latest = flat_index(&init_buf, &[1], &[Idx::from(&i)]).store(init_val).end(smallvec![i]);

        // In-lane fold over (acc, inner). The accumulator read must observe both
        // the prior store (`latest`) and the live reduce ranges, else it hoists.
        let acc = self.ker.raw_range(acc_end, AxisType::Reduce);
        let inner = self.ker.raw_range(ept, AxisType::Reduce);
        let acc_read = read0(&red_reg.after(smallvec![latest.clone(), acc.clone(), inner.clone()]));
        let src_idx = if row {
            [Idx::from(&outer), Idx::from(&acc), Idx::from(&inner)]
        } else {
            [Idx::from(&acc), Idx::from(&outer), Idx::from(&inner)]
        };
        let src_v = load_at(src.uop(), src.shape(), &src_idx);
        latest = flat_index(&red_reg, &[1], &[Idx::Const(0)]).store(op(&acc_read, &src_v)).end(smallvec![acc, inner]);

        // Cross-lane fold via `ds_bpermute`: read this lane's in-lane `partial`
        // once, then gather the three sibling 16-lane slots' *original* partials
        // (lanes L+16, L+32, L+48 mod warp) straight from registers — no LDS and
        // no barrier. The wave executes the gather in lockstep, so every lane's
        // `partial` is live before any lane reads it (the LDS barrier's old job).
        // Lane L thus folds the partials of {L, L+16, L+32, L+48} — bit-for-bit
        // the prior LDS sibling tree.
        let partial = read0(&red_reg.after(smallvec![latest]));
        let mut acc = partial.clone();
        for d in self.ker.caps.reduce_tree() {
            let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads as i64);
            acc = op(&acc, &self.shuffle_lane(&partial, &src_lane));
        }

        // Fold the lane result into vec[outer]: the vec read carries the incoming
        // vec state plus `outer` so it accumulates across outer iterations.
        let vec_acc =
            load_at(&vec.uop().after(smallvec![outer.clone()]), vec.shape(), &[Idx::from(&outer), Idx::Const(0)]);
        let vec_store = flat_index(vec.uop(), vec.shape(), &[Idx::from(&outer), Idx::Const(0)])
            .store(op(&vec_acc, &acc))
            .end(smallvec![outer]);
        self.finalize_tile(vec, vec_store)
    }

    /// Fully **unrolled** [`Self::reduce`]: the `outer`/`acc`/`inner` `RANGE`s
    /// become Rust `for`s, so the in-lane fold and the cross-lane `ds_bpermute`
    /// gather render loop-free (the softmax max/sum reduce must sit in the flat
    /// region with the MFMAs for the attention comb). Bit-identical fold order to
    /// the looped form.
    #[allow(clippy::too_many_arguments)]
    fn reduce_u<F>(
        &self,
        vec: RV<'k>,
        src: &RT<'k>,
        op: F,
        init_value: f64,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let elem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let laneid = self.laneid();
        let read0 = |buf: &Arc<UOp>| load_at(buf, &[1], &[Idx::Const(0)]);
        // Anchor the `src` read so a constant-address read of a carried tile is
        // not hoisted out of the enclosing rolled loop (see `Group::anchor`).
        let src_buf = self.anchor(src.uop());

        // Chain the per-`outer` vec stores so the LAST scopes them all under the
        // enclosing (rolled KV) loop's `END`.
        let mut vec_prev: Option<Arc<UOp>> = None;
        for o in 0..outer_end {
            // Fresh 1-element accumulator per `outer` (no cross-`outer` reuse, so
            // the unrolled folds stay independent).
            let red_reg = self.ker.alloc_reg(1, elem.clone());

            // Re-init: anchor the init store inside the enclosing tracked (KV)
            // loop, or — having only a constant input — it hoists above the rolled
            // loop and the accumulator carries stale state across KV iterations
            // (the looped form's `init_deps` invariant).
            let init_buf = red_reg.after(self.ker.tracked_ranges());
            let init_val = UOp::const_(elem.clone(), ConstValue::Float(init_value));
            let mut latest = flat_index(&init_buf, &[1], &[Idx::Const(0)]).store(init_val);

            // In-lane fold over (acc, inner): each step observes the prior store.
            for a in 0..acc_end {
                for i in 0..ept {
                    let acc_read = read0(&red_reg.after(smallvec![latest.clone()]));
                    let src_idx = if row {
                        [Idx::Const(o), Idx::Const(a), Idx::Const(i)]
                    } else {
                        [Idx::Const(a), Idx::Const(o), Idx::Const(i)]
                    };
                    let src_v = load_at(&src_buf, src.shape(), &src_idx);
                    latest = flat_index(&red_reg, &[1], &[Idx::Const(0)]).store(op(&acc_read, &src_v));
                }
            }

            // Cross-lane fold via `ds_bpermute` (the same sibling 16-lane tree as
            // the looped form): read this lane's partial once, gather L+{16,32,48}.
            let partial = read0(&red_reg.after(smallvec![latest]));
            let mut acc = partial.clone();
            for d in self.ker.caps.reduce_tree() {
                let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads as i64);
                acc = op(&acc, &self.shuffle_lane(&partial, &src_lane));
            }

            // Fold into vec[o], carrying the incoming (running) vec state; chain
            // across `outer` for loop scoping.
            let vbuf = match &vec_prev {
                Some(p) => vec.uop().after(smallvec![p.clone()]),
                None => self.anchor(vec.uop()),
            };
            let vec_acc = load_at(&vbuf, vec.shape(), &[Idx::Const(o), Idx::Const(0)]);
            let vstore = flat_index(vec.uop(), vec.shape(), &[Idx::Const(o), Idx::Const(0)]).store(op(&vec_acc, &acc));
            vec_prev = Some(vstore);
        }
        let terminal = vec_prev.expect("reduce_u: at least one outer tile");
        self.finalize_tile(vec, terminal)
    }

    /// `C += A·B` over a tile (tinygrad `mma_AB`): for every output fragment
    /// `(height, width)` accumulate `WMMA(A[height,inner], B[inner,width])`
    /// across the reduce axis `inner`. One [`Op::Wmma`](svod_ir::Op::Wmma) per
    /// K-iteration → one `mfma.f32.16x16x16bf16.1k`.
    ///
    /// # Panics
    /// The operand tiles `a`/`b` must be **bf16 or f16** — the only 16×16×16
    /// matrix-core input dtypes. An operand of any other dtype panics (a kernel-
    /// authoring error). Accumulation is always f32; this precondition holds for
    /// all four `mma_{ab,abt,atb,atbt}` variants.
    pub fn mma_ab(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, false)
    }

    /// `C += A·Bᵀ` (tinygrad `mma_ABt`): B fragment is read transposed
    /// (`b[width, inner]`); reduce axis stays `a.shape[-2]`.
    pub fn mma_abt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, false, true)
    }

    /// `C += Aᵀ·B` (tinygrad `mma_AtB`): A fragment is read transposed
    /// (`a[inner, height]`) and the reduce axis is `a.shape[-3]`.
    pub fn mma_atb(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, false)
    }

    /// `C += Aᵀ·Bᵀ` (tinygrad `mma_AtBt`): both fragments read transposed.
    pub fn mma_atbt(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>) -> RT<'k> {
        self.mma(c, a, b, true, true)
    }

    /// The shared WMMA body. The four `mma_{AB,ABt,AtB,AtBt}` variants differ
    /// only in the operand index permutation and the reduce-axis selection:
    /// - `a_t` (Aᵀ): A is read `a[inner, height]` and the reduce axis is
    ///   `a.shape[-3]`; otherwise `a[height, inner]`, reduce axis `a.shape[-2]`.
    /// - `b_t` (Bᵀ): B is read `b[width, inner]`; otherwise `b[inner, width]`.
    fn mma(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>, a_t: bool, b_t: bool) -> RT<'k> {
        // Flat (cross-tile-pipeline) FA opts into the fully-unrolled body so the
        // QKᵀ / A·V MFMAs render loop-free for the attention scheduling comb.
        if self.ker.unrolled() {
            return self.mma_u(c, a, b, a_t, b_t);
        }
        // Wave-agnostic: each wave runs the WMMA on its own per-lane RT operands
        // (the wave sub-tile selection happens in the LDS→REG load, not here). The
        // per-lane operand widths come from the descriptor (gfx942 4/4/4; RDNA
        // 16/16/8), not a hardcoded 4.
        assert_eq!(a.base.base.cols, 16, "mma: only the 16-col WMMA base is supported");
        let meta = wmma_desc(self.ker.caps.arch, a.elem());
        let (a_w, b_w, c_w) =
            (upcast_count(&meta.upcast_axes.a), upcast_count(&meta.upcast_axes.b), upcast_count(&meta.upcast_axes.c));

        let h_end = c.shape()[c.shape().len() - 3] as i64;
        let w_end = c.shape()[c.shape().len() - 2] as i64;
        let k_end = if a_t { a.shape()[a.shape().len() - 3] } else { a.shape()[a.shape().len() - 2] } as i64;
        let height = self.ker.raw_range(h_end, AxisType::Loop);
        let width = self.ker.raw_range(w_end, AxisType::Loop);
        let inner = self.ker.raw_range(k_end, AxisType::Reduce);

        let a_in = UOp::vectorize(
            (0..a_w)
                .map(|i| {
                    let idx = if a_t {
                        [Idx::from(&inner), Idx::from(&height), Idx::Const(i)]
                    } else {
                        [Idx::from(&height), Idx::from(&inner), Idx::Const(i)]
                    };
                    load_at(a.uop(), a.shape(), &idx)
                })
                .collect(),
        );
        let b_in = UOp::vectorize(
            (0..b_w)
                .map(|i| {
                    let idx = if b_t {
                        [Idx::from(&width), Idx::from(&inner), Idx::Const(i)]
                    } else {
                        [Idx::from(&inner), Idx::from(&width), Idx::Const(i)]
                    };
                    load_at(b.uop(), b.shape(), &idx)
                })
                .collect(),
        );
        // The accumulator read must depend on the reduce range `inner`, or it is
        // loop-invariant w.r.t. the K loop and gets hoisted *out* of it — every
        // K-iteration would then re-read the pre-loop C and the WMMA's
        // accumulation chain breaks. Mirrors svod's `reduce_to_acc`
        // (`acc.after([..reduce_range]).index(..)`): the `After([inner])` keeps
        // the read inside the K loop so it observes the prior iteration's store.
        let c_acc = c.uop().after(smallvec![inner.clone()]);
        let d_in = UOp::vectorize(
            (0..c_w)
                .map(|i| load_at(&c_acc, c.shape(), &[Idx::from(&height), Idx::from(&width), Idx::Const(i)]))
                .collect(),
        );

        let out = UOp::wmma(a_in, b_in, d_in, meta);
        let c_i: Vec<Arc<UOp>> = (0..c_w)
            .map(|i| {
                flat_index(c.uop(), c.shape(), &[Idx::from(&height), Idx::from(&width), Idx::Const(i)])
                    .store(out.gep(vec![i as usize]))
            })
            .collect();
        let c_store = UOp::group(c_i).end(smallvec![height, width, inner]);
        self.finalize_reg(c, c_store)
    }

    /// Fully **unrolled** [`Self::mma`]: emit one [`Op::Wmma`](svod_ir::Op::Wmma)
    /// per `(height, width, k)` fragment via Rust `for` loops — **no inner
    /// `RANGE`** — so the MFMAs render as a *flat* schedulable LLVM region the
    /// attention scheduling comb can weave the online softmax through. tk's
    /// direct-launch path skips the optimizer's `pre_expand`, so the looped
    /// [`Self::mma`] stays rolled (three `loop_body_*` around the mfma); explicit
    /// unroll is the only way to flatten it (route b — the cheap axis-flip is dead
    /// on the direct path).
    ///
    /// Each fragment's K-accumulation chains (`c[h,w]`'s k-step read observes the
    /// k−1 store); fragments chain into one terminal store so the enclosing rolled
    /// KV loop's `END` scopes them all (cf. the matmul accumulator chain,
    /// `kernels/matmul.rs:201`). Bit-identical accumulation order to [`Self::mma`].
    fn mma_u(&self, c: RT<'k>, a: &RT<'k>, b: &RT<'k>, a_t: bool, b_t: bool) -> RT<'k> {
        assert_eq!(a.base.base.cols, 16, "mma_u: only the 16-col WMMA base is supported");
        let meta = wmma_desc(self.ker.caps.arch, a.elem());
        let (a_w, b_w, c_w) =
            (upcast_count(&meta.upcast_axes.a), upcast_count(&meta.upcast_axes.b), upcast_count(&meta.upcast_axes.c));

        let h_end = c.shape()[c.shape().len() - 3] as i64;
        let w_end = c.shape()[c.shape().len() - 2] as i64;
        let k_end = if a_t { a.shape()[a.shape().len() - 3] } else { a.shape()[a.shape().len() - 2] } as i64;

        // Fragment-scoping chain: each fragment's first (k=0) accumulator read
        // orders after the previous fragment's terminal store, so the LAST
        // fragment's store transitively scopes them all under one loop `END`.
        let mut prev_frag: Option<Arc<UOp>> = None;
        for h in 0..h_end {
            for w in 0..w_end {
                // Per-fragment K accumulation: the k-step read observes the k−1
                // store to this same fragment (the unrolled analog of the looped
                // `c.after([inner])` loop-carry).
                let mut frag_prev: Option<Arc<UOp>> = None;
                for k in 0..k_end {
                    let a_in = UOp::vectorize(
                        (0..a_w)
                            .map(|i| {
                                let idx = if a_t {
                                    [Idx::Const(k), Idx::Const(h), Idx::Const(i)]
                                } else {
                                    [Idx::Const(h), Idx::Const(k), Idx::Const(i)]
                                };
                                load_at(a.uop(), a.shape(), &idx)
                            })
                            .collect(),
                    );
                    let b_in = UOp::vectorize(
                        (0..b_w)
                            .map(|i| {
                                let idx = if b_t {
                                    [Idx::Const(w), Idx::Const(k), Idx::Const(i)]
                                } else {
                                    [Idx::Const(k), Idx::Const(w), Idx::Const(i)]
                                };
                                load_at(b.uop(), b.shape(), &idx)
                            })
                            .collect(),
                    );
                    // Accumulator source: the prior k-step's store for this
                    // fragment; on k==0 the incoming `c` carrying the
                    // fragment-scoping dep on the previous fragment's store.
                    let mut deps: SmallVec<[Arc<UOp>; 4]> = SmallVec::new();
                    match &frag_prev {
                        Some(fp) => deps.push(fp.clone()),
                        None => {
                            if let Some(pf) = &prev_frag {
                                deps.push(pf.clone());
                            }
                        }
                    }
                    // Anchor the incoming accumulator read (no chain dep yet) to the
                    // enclosing rolled loop so a carried accumulator (`o_reg`) is not
                    // hoisted out (see `Group::anchor`); subsequent k/fragment reads
                    // chain through their stores, which are already loop-scoped.
                    let c_src = if deps.is_empty() { self.anchor(c.uop()) } else { c.uop().after(deps) };
                    let d_in = UOp::vectorize(
                        (0..c_w)
                            .map(|i| load_at(&c_src, c.shape(), &[Idx::Const(h), Idx::Const(w), Idx::Const(i)]))
                            .collect(),
                    );
                    let out = UOp::wmma(a_in, b_in, d_in, meta.clone());
                    let c_i: Vec<Arc<UOp>> = (0..c_w)
                        .map(|i| {
                            flat_index(c.uop(), c.shape(), &[Idx::Const(h), Idx::Const(w), Idx::Const(i)])
                                .store(out.gep(vec![i as usize]))
                        })
                        .collect();
                    frag_prev = Some(UOp::group(c_i));
                }
                prev_frag = frag_prev;
            }
        }
        let terminal = prev_frag.expect("mma_u: at least one (height, width) fragment");
        self.finalize_reg(c, terminal)
    }

    // ── load (tinygrad `Group.load`) ────────────────────────────────────────

    /// Move data into `dst` (tinygrad `Group.load`), with the legal (dst, src)
    /// address-space pair resolved at **compile time** via [`LoadInto`]: ST←GL
    /// (coalesced fill + barrier), RT←ST / RT←GL (fragment gather). An illegal pair
    /// (e.g. RT←RT) has no impl, so it is a compile error — not a runtime panic:
    ///
    /// ```compile_fail
    /// # use svod_tk::{ArchCaps, Kernel, MoveIdx};
    /// # use svod_tk::tiles::{RT_16X16, TileLayout};
    /// # use svod_dtype::DType;
    /// let ker = Kernel::new("x", [1, 1, 1], 64, vec![], ArchCaps::GFX942);
    /// let g = ker.warp();
    /// let a = ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16);
    /// let b = ker.rt((16, 16), DType::Float32, TileLayout::Row, RT_16X16);
    /// let _ = g.load(a, b, MoveIdx::default()); // RT ← RT: no LoadInto impl ⇒ won't compile
    /// ```
    pub fn load<Dst, Src>(&self, dst: Dst, src: Src, ix: MoveIdx<'_>) -> Src::Output
    where
        Src: LoadInto<'k, Dst>,
    {
        src.load_into(self, dst, ix)
    }

    /// Coalesced GLOBAL→LOCAL fill **without** the trailing workgroup barrier —
    /// the software-pipeline primitive (stage ii). The caller is responsible
    /// for inserting one barrier per buffer before the LDS→REG gather (so the
    /// fill is visible) and before the next overwrite (the WAR edge); decoupling
    /// the fill from its sync lets the next block's GLOBAL loads issue *ahead* of
    /// the current block's compute, overlapping memory latency with the MFMA.
    pub fn fill_local_nobar(&self, dst: ST<'k>, src: GL<'k>, idxs: &[Idx], axis: usize) -> ST<'k> {
        self.load_global_to_local(dst, &src, idxs, axis, false)
    }

    /// Stage one tile of `src` (GLOBAL) into a fresh per-lane register buffer —
    /// the GLOBAL→VGPR half of the register prefetch. Uses the *same*
    /// coalesced per-lane addressing as [`Self::load_global_to_local`], but lands
    /// the loaded (unswizzled) values in a flat `[total_calls, ept]` DEFINE_REG
    /// instead of LDS, so the load can be issued ahead of the consuming MFMAs.
    /// Commit it with [`Self::commit_reg_to_local`] (same `st`/`idxs`/`axis`).
    pub fn stage_global_to_reg(&self, st: &ST<'k>, src: &GL<'k>, idxs: &[Idx], axis: usize) -> Arc<UOp> {
        let geom = self.lds_fill_geom(st);
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, st.rows as i64);
                }
                if i == 3 {
                    e = idx_mul(&e, st.cols as i64);
                }
                e
            })
            .collect();
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let stage = self.ker.alloc_reg((geom.total_calls * geom.ept) as usize, st.elem().clone());
        let outer = self.ker.raw_range(geom.total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(geom.ept, AxisType::Upcast);
        let (height, width, row, col) = self.fill_lane_rc(&geom, &outer, &inner);

        let off = iadd(
            &src_i_base,
            &iadd(
                &iadd(&imul(&height, geom.base_rows * row_stride), &imul(&width, geom.base_cols)),
                &iadd(&imul(&row, row_stride), &col),
            ),
        );
        let mut load = load_off(src.uop(), off);
        if src.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        let stage_shape = [geom.total_calls as usize, geom.ept as usize];
        let stored = flat_index(&stage, &stage_shape, &[Idx::from(&outer), Idx::from(&inner)])
            .store(load)
            .end(smallvec![outer, inner]);
        self.ker.push_store(stored.clone(), stage.clone());
        stage.after(smallvec![stored])
    }

    /// Commit a staged register buffer (from [`Self::stage_global_to_reg`]) into
    /// the swizzled LDS tile — the VGPR→LDS `ds_write` half of the prefetch.
    /// Recomputes the identical per-lane addressing. Ends in a workgroup barrier
    /// when `barrier` (the single-buffer commit); the double-buffered pipeline
    /// passes `false` and shares one barrier per iteration.
    pub fn commit_reg_to_local(&self, st: ST<'k>, stage: &Arc<UOp>, barrier: bool) -> ST<'k> {
        // The LDS destination geometry is fully determined by the tile shape (the
        // global tile position only mattered when *staging* into the registers).
        let geom = self.lds_fill_geom(&st);
        let outer = self.ker.raw_range(geom.total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(geom.ept, AxisType::Upcast);
        let (height, width, row, col) = self.fill_lane_rc(&geom, &outer, &inner);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());

        let stage_shape = [geom.total_calls as usize, geom.ept as usize];
        let load = load_at(stage, &stage_shape, &[Idx::from(&outer), Idx::from(&inner)]);
        let stored = st_index(&st, &[Idx::Uop(height), Idx::Uop(width), Idx::Uop(srow), Idx::Uop(scol)])
            .store(load)
            .end(smallvec![outer, inner]);
        let stored = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, stored)
    }

    /// Move data out of `src` (tinygrad `Group.store`): REG→LOCAL (fragment
    /// scatter) and REG→GLOBAL (coalesced write-back).
    pub fn store<Dst, Src>(&self, dst: Dst, src: Src, ix: MoveIdx<'_>) -> Src::Output
    where
        Src: StoreInto<'k, Dst>,
    {
        src.store_into(self, dst, ix)
    }

    /// Cross-wave WAR fence over two just-loaded register reads `a`/`b`: builds ONE
    /// workgroup `Barrier` (passthrough `a`, deps = `b` + `extra` — the cross-iteration
    /// prefetch commits the double-buffer pipeline folds in) and returns BOTH reads
    /// re-threaded `.after([sync])`. The barrier is internal (never returned), so a
    /// read cannot be left un-fenced (you get the fenced tiles back) and nothing can
    /// depend on the barrier as a value (the AMD renderer emits the `s.barrier` fence
    /// but registers no SSA value for it). Emits the identical graph as the hand-built
    /// `a.uop().barrier([b] + extra)` + per-read `.after([sync])`.
    pub fn war_fence2<T: RegTile<'k>>(&self, a: T, b: T, extra: &[Arc<UOp>]) -> (T, T) {
        let mut deps: SmallVec<[Arc<UOp>; 4]> = smallvec![b.uop().clone()];
        deps.extend(extra.iter().cloned());
        let sync = a.uop().barrier(deps);
        (a.after(smallvec![sync.clone()]), b.after(smallvec![sync]))
    }

    /// The [`LdsGeom`] for filling `st` collaboratively across all group
    /// threads (`elements_per_thread`, pass count, last-pass clamp).
    fn lds_fill_geom(&self, st: &ST<'k>) -> LdsGeom {
        let ept = st.base.base.elements_per_thread() as i64;
        let st_cols = st.cols as i64;
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let slots = self.group_threads as i64 * ept;
        let total_calls = (total_elems + slots - 1) / slots;
        LdsGeom {
            ept,
            st_cols,
            memcpy_per_row: st_cols / ept,
            base_rows,
            base_cols,
            total_calls,
            num_valid: total_elems / ept,
            clamp: total_calls * slots != total_elems,
        }
    }

    /// The `(height, width, row, col)` LDS fragment coordinate this lane fills at
    /// collaborative pass `(outer, inner)` — the shared per-lane addressing of
    /// the direct fill and the register-staged prefetch (over-subscribed last
    /// pass clamps to the final valid fragment, idempotent).
    fn fill_lane_rc(
        &self,
        geom: &LdsGeom,
        outer: &Arc<UOp>,
        inner: &Arc<UOp>,
    ) -> (Arc<UOp>, Arc<UOp>, Arc<UOp>, Arc<UOp>) {
        let mut load_idx = iadd(&imul(outer, self.group_threads as i64), &self.laneid());
        if geom.clamp {
            let cond = load_idx.try_cmplt(&cidx(geom.num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(geom.num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, geom.memcpy_per_row);
        let col0 = iadd(&imod(&imul(&load_idx, geom.ept), geom.st_cols), inner);
        (
            idiv(&row0, geom.base_rows),
            idiv(&col0, geom.base_cols),
            imod(&row0, geom.base_rows),
            imod(&col0, geom.base_cols),
        )
    }

    /// Coalesced GLOBAL→LOCAL fill: every group thread streams
    /// `elements_per_thread` contiguous global elements into the swizzled LDS
    /// tile. When `barrier`, it is closed with a workgroup barrier so the
    /// subsequent gather sees it (the default); the software-pipeline path passes
    /// `false` and inserts the barrier itself (see [`Self::fill_local_nobar`]).
    fn load_global_to_local(&self, st: ST<'k>, src: &GL<'k>, idxs: &[Idx], axis: usize, barrier: bool) -> ST<'k> {
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, st.rows as i64);
                }
                if i == 3 {
                    e = idx_mul(&e, st.cols as i64);
                }
                e
            })
            .collect();
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let ept = st.base.base.elements_per_thread() as i64;
        let st_cols = st.cols as i64;
        let memcpy_per_row = st_cols / ept;
        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let height_dim = st.shape()[n - 4] as i64;
        let width_dim = st.shape()[n - 3] as i64;
        let total_elems = height_dim * width_dim * num_elements;
        let slots = self.group_threads as i64 * ept;
        // Round the pass count *up*: a tile smaller than one full group-pass (the
        // multi-wave FA 16×64 K/V block streamed by 512 threads) would otherwise
        // floor to zero passes and load nothing.
        let total_calls = (total_elems + slots - 1) / slots;
        // Over-subscribed last pass (more lane-loads than fragment-loads): clamp
        // the load index to the last valid fragment so the excess lanes redo it
        // (idempotent — same source, same swizzled slot) instead of writing past
        // the tile. A no-op when the tile divides the group evenly (matmul,
        // single-warp FA): `clamp` is false and the index passes through.
        let num_valid = total_elems / ept;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Upcast);
        let laneid = self.laneid();

        let mut load_idx = iadd(&imul(&outer, self.group_threads as i64), &laneid);
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = iadd(&imod(&imul(&load_idx, ept), st_cols), &inner);
        let height = idiv(&row0, base_rows);
        let width = idiv(&col0, base_cols);
        let row = imod(&row0, base_rows);
        let col = imod(&col0, base_cols);
        let (srow, scol) = st.base.swizzle.swizzle_rc(row.clone(), col.clone(), st.base.base.cols, st.elem().base());

        let off = iadd(
            &src_i_base,
            &iadd(
                &iadd(&imul(&height, base_rows * row_stride), &imul(&width, base_cols)),
                &iadd(&imul(&row, row_stride), &col),
            ),
        );
        let mut load = load_off(src.uop(), off);
        if src.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        let dst_idx = st_index(&st, &[Idx::Uop(height), Idx::Uop(width), Idx::Uop(srow), Idx::Uop(scol)]);
        let stored = dst_idx.store(load).end(smallvec![outer, inner]);
        let ended = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, ended)
    }

    /// Vectorized GLOBAL→LOCAL fill: the [`Self::load_global_to_local`]
    /// counterpart that issues **128-bit** (`vec8` bf16) coalesced global loads
    /// (one `global_load_dwordx4`/lane) and commits each into the XOR-swizzled
    /// LDS as `vec8/sw` contiguous `vec_sw` stores. The swizzle's XOR delta is
    /// always a multiple of 8 bytes (`st.cuh:96` `<<3`), so a `sw = 8/itemsize`
    /// element group is never re-ordered (the `vec4` halves stay contiguous);
    /// a single `vec8` LDS store would split on the odd deltas, so we keep the
    /// wide *global* load but narrow the swizzled *LDS* store. Ends in a
    /// workgroup barrier (the matmul fill). bf16-only.
    pub fn fill_local_vec(&self, dst: ST<'k>, src: GL<'k>, idxs: &[Idx], axis: usize) -> ST<'k> {
        self.load_global_to_local_vec(dst, &src, idxs, axis, true)
    }

    fn load_global_to_local_vec(&self, st: ST<'k>, src: &GL<'k>, idxs: &[Idx], axis: usize, barrier: bool) -> ST<'k> {
        let itemsize = st.elem().base().bytes() as i64;
        assert_eq!(itemsize, 2, "vec fill: bf16-only (128-bit = vec8)");
        assert_eq!(src.elem(), st.elem(), "vec fill: cast unsupported (use the scalar fill)");
        let vw: i64 = 16 / itemsize; // 8 bf16 — the 128-bit global load width
        let sw: i64 = 8 / itemsize; // 4 bf16 — the swizzle-order-safe LDS store width

        let base_rows = st.base.base.rows as i64;
        let base_cols = st.base.base.cols as i64;
        let st_cols = st.cols as i64;
        // Alignment invariants: the swizzle period and the
        // tile/fragment widths must admit `vw`-aligned 16-byte groups.
        if let Some(period) = st.base.swizzle.period_bytes(st.base.base.cols, itemsize) {
            assert_eq!(period % 16, 0, "vec fill: swizzle period {period}B not 16B-aligned");
        }
        assert_eq!(base_cols % vw, 0, "vec fill: base cols {base_cols} not a multiple of vec width {vw}");
        assert_eq!(st_cols % vw, 0, "vec fill: st cols {st_cols} not a multiple of vec width {vw}");

        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        assert_eq!(row_stride % vw, 0, "vec fill: row stride {row_stride} not {vw}-aligned (need N % 8 == 0)");

        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, st.rows as i64);
                }
                if i == 3 {
                    e = idx_mul(&e, st.cols as i64);
                }
                e
            })
            .collect();
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let num_elements = st.base.base.num_elements() as i64;
        let n = st.shape().len();
        let total_elems = st.shape()[n - 4] as i64 * st.shape()[n - 3] as i64 * num_elements;
        let memcpy_per_row = st_cols / vw;
        let slots = self.group_threads as i64 * vw;
        let total_calls = (total_elems + slots - 1) / slots;
        let num_valid = total_elems / vw;
        let clamp = total_calls * slots != total_elems;

        let outer = self.ker.raw_range(total_calls, AxisType::Loop);
        let mut load_idx = iadd(&imul(&outer, self.group_threads as i64), &self.laneid());
        if clamp {
            let cond = load_idx.try_cmplt(&cidx(num_valid)).expect("load_idx < num_valid");
            load_idx = UOp::try_where(cond, load_idx.clone(), cidx(num_valid - 1)).expect("clamp load_idx");
        }
        // The thread's `vw`-wide run: row `row0`, columns `[col0, col0+vw)` (a
        // `vw`-aligned slice within one base fragment, since `vw | base_cols`).
        let row0 = idiv(&load_idx, memcpy_per_row);
        let col0 = imod(&imul(&load_idx, vw), st_cols);
        let height = idiv(&row0, base_rows);
        let row = imod(&row0, base_rows);
        let width = idiv(&col0, base_cols);

        // One 128-bit coalesced global load of the contiguous `vw`-run.
        let off = iadd(&src_i_base, &iadd(&imul(&row0, row_stride), &col0));
        let loaded = load_vec(src.uop(), off, vw as usize);

        // Commit as `vw/sw` swizzle-safe `vec_sw` LDS stores (delta is constant
        // across the fragment row, so each `sw`-group maps contiguously).
        let stores: Vec<Arc<UOp>> = (0..vw / sw)
            .map(|j| {
                let col = imod(&iadd(&col0, &cidx(j * sw)), base_cols);
                let (srow, scol) = st.base.swizzle.swizzle_rc(row.clone(), col, st.base.base.cols, st.elem().base());
                let val = loaded.gep(((j * sw) as usize..(j * sw + sw) as usize).collect());
                let didx = [Idx::Uop(height.clone()), Idx::Uop(width.clone()), Idx::Uop(srow), Idx::Uop(scol)];
                st_index(&st, &didx).store(val)
            })
            .collect();
        let grouped = if stores.len() == 1 { stores.into_iter().next().unwrap() } else { UOp::group(stores) };
        let stored = grouped.end(smallvec![outer]);
        let ended = if barrier { stored.barrier(SmallVec::new()) } else { stored };
        self.finalize_st(st, ended)
    }

    /// LOCAL→REG fragment gather: each lane reads its WMMA fragment lanes from
    /// the (swizzled) LDS tile.
    fn load_local_to_reg(&self, rt: RT<'k>, st: &ST<'k>, dst_idxs: &[Idx], idxs: &[Idx]) -> RT<'k> {
        let laneid = self.ker.laneid();
        let ept = rt.base.base.elements_per_thread() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        // SI-1 off-by-one guard: the wave's RT sub-tile must fit inside the ST.
        let sn = st.shape().len();
        let (st_h, st_w) = (st.shape()[sn - 4] as i64, st.shape()[sn - 3] as i64);
        assert!(rt_h <= st_h && rt_w <= st_w, "load LOCAL→REG: RT {rt_h}×{rt_w} exceeds ST {st_h}×{st_w}");
        let height = self.ker.raw_range(rt_h, AxisType::Loop);
        let width = self.ker.raw_range(rt_w, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let (row, col) = lane_rc(
            rt.layout != st.layout,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());

        // Wave sub-tile fragment offset (SI-1): the caller passes the wave's
        // `(row_block, col_block)` via `idxs` (already including warp_row/col);
        // empty ⇒ no offset (single-warp).
        let h_idx = wave_offset(idxs.first(), rt_h, &height);
        let w_idx = wave_offset(idxs.get(1), rt_w, &width);
        let src_idx = [h_idx, w_idx, Idx::Uop(srow), Idx::Uop(scol)];
        let mut load = st_load(st, &src_idx);
        if st.elem() != rt.elem() {
            load = load.cast(rt.elem().clone());
        }
        let mut didx: Vec<Idx> = dst_idxs.to_vec();
        didx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let ended = flat_index(rt.uop(), rt.shape(), &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_reg(rt, ended)
    }

    /// GLOBAL→REG fragment gather: each lane reads its register fragment
    /// straight from global memory (the FA Q-tile load). The mirror of
    /// [`Self::store_reg_to_global`].
    fn load_global_to_reg(&self, rt: RT<'k>, src: &GL<'k>, dst_idxs: &[Idx], idxs: &[Idx], axis: usize) -> RT<'k> {
        let row_stride: i64 = src.shape()[axis + 1..].iter().product::<usize>() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let s3 = rt.shape()[n - 3] as i64;
        let s2 = rt.shape()[n - 2] as i64;

        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, s3 * base_rows);
                }
                if i == 3 {
                    e = idx_mul(&e, s2 * base_cols);
                }
                e
            })
            .collect();
        let src_i_base = flat_offset(src.shape(), &idxs_t);

        let laneid = self.ker.laneid();
        let height = self.ker.raw_range(s3, AxisType::Loop);
        let width = self.ker.raw_range(s2, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let base_row = imul(&height, base_rows);
        let base_col = imul(&width, base_cols);
        let (row, col) = lane_rc(
            rt.layout == TileLayout::Col,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let srow = iadd(&base_row, &row);
        let scol = iadd(&base_col, &col);
        let off = iadd(&src_i_base, &iadd(&imul(&srow, row_stride), &scol));

        let mut load = load_off(src.uop(), off);
        if src.elem() != rt.elem() {
            load = load.cast(rt.elem().clone());
        }
        let mut didx: Vec<Idx> = dst_idxs.to_vec();
        didx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let ended = flat_index(rt.uop(), rt.shape(), &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_reg(rt, ended)
    }

    /// REG→LOCAL fragment scatter: each lane writes its register fragment into
    /// the (swizzled) LDS tile (the layout-transpose hop before write-back).
    fn store_reg_to_local(&self, st: ST<'k>, rt: &RT<'k>, idxs: &[Idx], src_idxs: &[Idx]) -> ST<'k> {
        let laneid = self.ker.laneid();
        let ept = rt.base.base.elements_per_thread() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let n = rt.shape().len();
        let (rt_h, rt_w) = (rt.shape()[n - 3] as i64, rt.shape()[n - 2] as i64);
        let height = self.ker.raw_range(rt_h, AxisType::Loop);
        let width = self.ker.raw_range(rt_w, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let (row, col) = lane_rc(
            rt.layout != st.layout,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let (srow, scol) = st.base.swizzle.swizzle_rc(row, col, st.base.base.cols, st.elem().base());

        let mut sidx: Vec<Idx> = src_idxs.to_vec();
        sidx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let mut load = load_at(rt.uop(), rt.shape(), &sidx);
        if rt.elem() != st.elem() {
            load = load.cast(st.elem().clone());
        }
        // Wave sub-tile fragment offset (SI-1), symmetric with `load_local_to_reg`.
        let h_idx = wave_offset(idxs.first(), rt_h, &height);
        let w_idx = wave_offset(idxs.get(1), rt_w, &width);
        let didx = [h_idx, w_idx, Idx::Uop(srow), Idx::Uop(scol)];
        let ended = st_index(&st, &didx).store(load).end(smallvec![height, width, inner]);
        self.finalize_st(st, ended)
    }

    /// REG→GLOBAL write-back: each lane writes its register fragment to the
    /// correct global position.
    fn store_reg_to_global(&self, dst: GL<'k>, rt: &RT<'k>, idxs: &[Idx], src_idxs: &[Idx], axis: usize) -> GL<'k> {
        let row_stride: i64 = dst.shape()[axis + 1..].iter().product::<usize>() as i64;
        let base_rows = rt.base.base.rows as i64;
        let base_cols = rt.base.base.cols as i64;
        let stride = rt.base.stride as i64;
        let ept = rt.base.base.elements_per_thread() as i64;
        let n = rt.shape().len();
        let s3 = rt.shape()[n - 3] as i64;
        let s2 = rt.shape()[n - 2] as i64;

        let idxs_t: Vec<Idx> = idxs
            .iter()
            .enumerate()
            .map(|(i, idx)| {
                let mut e = idx.clone();
                if i == axis {
                    e = idx_mul(&e, s3 * base_rows);
                }
                if i == 3 {
                    e = idx_mul(&e, s2 * base_cols);
                }
                e
            })
            .collect();
        let dst_i_base = flat_offset(dst.shape(), &idxs_t);

        let laneid = self.ker.laneid();
        let height = self.ker.raw_range(s3, AxisType::Loop);
        let width = self.ker.raw_range(s2, AxisType::Loop);
        let inner = self.ker.raw_range(ept, AxisType::Loop);

        let base_row = imul(&height, base_rows);
        let base_col = imul(&width, base_cols);
        let (row, col) = lane_rc(
            rt.layout == TileLayout::Col,
            rt.base.interleave,
            rt.base.interleave_t,
            &laneid,
            base_rows,
            base_cols,
            stride,
            &inner,
        );
        let srow = iadd(&base_row, &row);
        let scol = iadd(&base_col, &col);
        let off = iadd(&dst_i_base, &iadd(&imul(&srow, row_stride), &scol));

        let mut sidx: Vec<Idx> = src_idxs.to_vec();
        sidx.extend([Idx::from(&height), Idx::from(&width), Idx::from(&inner)]);
        let mut load = load_at(rt.uop(), rt.shape(), &sidx);
        if rt.elem() != dst.elem() {
            load = load.cast(dst.elem().clone());
        }
        let ended = index_off(dst.uop(), off).store(load).end(smallvec![height, width, inner]);
        self.finalize_gl(dst, ended)
    }

    // ── store bookkeeping helpers ───────────────────────────────────────────

    fn finalize_reg(&self, t: RT<'k>, ended: Arc<UOp>) -> RT<'k> {
        self.finalize_tile(t, ended)
    }
    /// Record `ended` as a terminal store and rewrap the register tile so later
    /// reads order after it (tinygrad `dst.after(dst_store)`).
    fn finalize_tile<T: RegTile<'k>>(&self, t: T, ended: Arc<UOp>) -> T {
        self.ker.push_store(ended.clone(), t.uop().clone());
        let after = t.uop().after(smallvec![ended]);
        t.rewrap(after)
    }
    fn finalize_st(&self, t: ST<'k>, ended: Arc<UOp>) -> ST<'k> {
        self.ker.push_store(ended.clone(), t.uop().clone());
        let after = t.uop().after(smallvec![ended]);
        t.rewrap(after)
    }
    fn finalize_gl(&self, t: GL<'k>, ended: Arc<UOp>) -> GL<'k> {
        self.ker.push_store(ended.clone(), t.uop().clone());
        let after = t.uop().after(smallvec![ended]);
        t.rewrap(after)
    }
}

impl<'k> ST<'k> {
    /// A zero-copy view of the `(row_blk, col_blk)`-th sub-rectangle of warp-tile
    /// element size `dims` — folds the per-warp band into the tile's additive base
    /// offset (composing with any existing double-buffer parity offset), so a
    /// subsequent [`Group::load`]/[`Group::store`] needs **no** wave-block index
    /// (pass [`MoveIdx::default`]). `dims` is the consuming register tile's element
    /// shape (the warp-tile size). Addresses the SAME element as the equivalent
    /// `wave_offset` block — the band is whole-fragment-granular (so the LDS swizzle,
    /// applied within a fragment, is unaffected); only the offset op-tree differs
    /// from the folded form (`imul(a·k + local, stride)` → `imul(local, stride) +
    /// imul(a·k, stride)`), so it is correct-by-construction but changes the kernel's
    /// content hash.
    pub fn subtile(&self, dims: (usize, usize), blk: (Idx, Idx)) -> ST<'k> {
        let frag_h = (dims.0 / self.base.base.rows) as i64;
        let frag_w = (dims.1 / self.base.base.cols) as i64;
        let band = flat_offset(
            self.shape(),
            &[idx_mul(&blk.0, frag_h), idx_mul(&blk.1, frag_w), Idx::Const(0), Idx::Const(0)],
        );
        let off = match self.base_offset() {
            Some(bo) => band.try_add(bo).expect("subtile band + base offset"),
            None => band,
        };
        self.with_base_offset(off)
    }
}

/// ST flat INDEX honoring the optional double-buffer parity [`ST::base_offset`].
/// Identical to [`crate::index::flat_index`] for an ordinary (`base_offset:None`)
/// tile; adds the parity offset for a [`Kernel::st_db`] half-view.
fn st_index(st: &ST, idxs: &[Idx]) -> Arc<UOp> {
    let mut off = flat_offset(st.shape(), idxs);
    if let Some(bo) = st.base_offset() {
        off = off.try_add(bo).expect("st_index: parity base offset add");
    }
    index_off(st.uop(), off)
}
/// ST flat LOAD honoring [`ST::base_offset`] — the [`crate::index::load_at`] analog.
fn st_load(st: &ST, idxs: &[Idx]) -> Arc<UOp> {
    let idx = st_index(st, idxs);
    UOp::load().buffer(st.uop().clone()).index(idx).call()
}

/// The row-major cartesian product of `0..d` for each `d` in `shape` — the
/// constant index tuples an unrolled register op iterates (the analog of the
/// nested `Loop` `RANGE`s it replaces).
fn cartesian(shape: &[usize]) -> Vec<Vec<Idx>> {
    let mut acc = vec![Vec::new()];
    for &d in shape {
        acc = acc
            .into_iter()
            .flat_map(|prefix| {
                (0..d as i64).map(move |i| {
                    let mut next = prefix.clone();
                    next.push(Idx::Const(i));
                    next
                })
            })
            .collect();
    }
    acc
}
