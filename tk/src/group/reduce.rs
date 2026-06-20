//! Cross-lane reductions: the value-only `row_reduce`/`col_reduce` (shared
//! `reduce`/`reduce_u` bodies) and the index-carrying argmin/argmax
//! `row_arg_reduce`/`col_arg_reduce` (shared `arg_reduce` body). Each folds the
//! lane-local elements, then the sibling 16-lane slots via `ds_bpermute`.

use std::sync::Arc;

use smallvec::{SmallVec, smallvec};
use svod_dtype::DType;
use svod_ir::{AxisType, ConstValue, UOp};

use super::{ArgDir, Group, arg_fold, iadd, imod, imul, lane_rc};
use crate::index::{Idx, cidx, flat_index, load_at};
use crate::tile::{RT, RV};
use crate::tiles::TileLayout;

impl<'k> Group<'k> {
    /// Reduce each row of `src` into `vec` (tinygrad `row_reduce`): per
    /// row-tile `height`, fold `op` over the `(width, inner)` lane-local
    /// elements into a 1-element REG accumulator, publish it to an LDS scratch
    /// slot at this lane, `barrier`, then fold the three sibling 16-lane slots
    /// (`(laneid + (1+i)*16) % group_threads`) to complete the warp-wide reduce,
    /// and fold the result into `vec[height]`.
    ///
    /// # Panics
    /// Panics if the tile rank is less than 3 (it reads the trailing
    /// `[.., height, width, inner]` dims).
    pub fn row_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init_value: f64) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let n = src.shape().len();
        self.reduce(vec, src, op, init_value, src.shape()[n - 3] as i64, src.shape()[n - 2] as i64, true)
    }

    /// Reduce each column of `src` into `vec` (tinygrad `col_reduce`): the
    /// transpose of [`Self::row_reduce`] — outer loop over column-tiles, accumulate
    /// over the `(height, inner)` elements.
    ///
    /// # Panics
    /// Panics if the tile rank is less than 3, or if the group has more than one
    /// warp.
    pub fn col_reduce<F>(&self, vec: RV<'k>, src: &RT<'k>, op: F, init_value: f64) -> RV<'k>
    where
        F: Fn(&Arc<UOp>, &Arc<UOp>) -> Arc<UOp>,
    {
        let n = src.shape().len();
        self.reduce(vec, src, op, init_value, src.shape()[n - 2] as i64, src.shape()[n - 3] as i64, false)
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
            let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads() as i64);
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
                let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads() as i64);
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

    /// The global index, along the **folded** axis, contributed by element
    /// `(laneid, inner)` of fragment-tile `acc`: `acc*extent + lane_rc(..)`.
    /// Reuses the source fragment's [`lane_rc`] mapping — the same one the value
    /// load uses — and picks the lane_rc coordinate that *varies with `inner`*,
    /// since that (with the cross-lane tree) is exactly the axis the reduce folds.
    /// It is the **column** for the normal (gfx942 stride-4) and `interleave_t`
    /// layouts, and the **row** for the `transpose` (`Col`-layout) and the wave32
    /// even/odd `interleave` accumulator — where the 16-wide reduced axis is split
    /// across a lane's `inner` elements and its `L+16` sibling. So `row_arg_reduce`
    /// on a wave32 accumulator reduces the interleave's `inner`-carrying axis,
    /// exactly as `row_reduce` does (the caller arranges the tile to match).
    fn axis_index_of(&self, src: &RT<'k>, acc: &Arc<UOp>, inner: &Arc<UOp>) -> Arc<UOp> {
        let base_rows = src.base.base.rows as i64;
        let base_cols = src.base.base.cols as i64;
        let (r, c) = lane_rc(
            src.layout == TileLayout::Col,
            src.base.interleave,
            src.base.interleave_t,
            &self.laneid(),
            base_rows,
            base_cols,
            src.base.stride as i64,
            inner,
        );
        // Which lane_rc coordinate carries `inner` (the folded axis)?
        let inner_is_col = if src.base.interleave_t {
            true
        } else if src.base.interleave {
            false
        } else {
            src.layout != TileLayout::Col
        };
        let (folded, extent) = if inner_is_col { (c, base_cols) } else { (r, base_rows) };
        iadd(&imul(acc, extent), &folded).cast(DType::Int32)
    }

    /// Record one grouped two-output terminal store and rewrap BOTH result tiles
    /// after it — the [`Group::finalize_tile`](super::Group) analog for
    /// arg-reduce's paired value/index outputs. One `END(GROUP(STORE, STORE))`
    /// closes the shared loop exactly once; a per-store `.end()` would
    /// double-`END` the range (cf. the grouped accumulator store in `mma`).
    fn finalize_pair(&self, val: RV<'k>, idx: RV<'k>, ended: Arc<UOp>) -> (RV<'k>, RV<'k>) {
        self.ker.push_store(ended.clone(), val.uop().clone());
        let val = val.rewrap(val.uop().after(smallvec![ended.clone()]));
        let idx = idx.rewrap(idx.uop().after(smallvec![ended]));
        (val, idx)
    }

    /// Argmin/argmax each row of `src` into `(val, idx)` — the index-carrying
    /// [`Self::row_reduce`]. Folds the reduced-axis `(width, inner)` lane-local
    /// elements and the sibling 16-lane `ds_bpermute` tree, keeping the
    /// extremum's value AND its global column index (ties → smaller index,
    /// matching `Tensor::topk`/`argmin`). The value `RV` is seeded by `dir`
    /// (`+∞`/`−∞`); the index `RV` must be `Int32`. Inside a rolled loop each trip
    /// is a **fresh** reduce (the output pair re-seeds per the enclosing tracked
    /// range), not a running extremum folded across trips.
    ///
    /// The reduced data must be **NaN-free**: the value compare lowers to an
    /// unordered `fcmp ult`, so a NaN can win the fold and propagate as the kept
    /// value (unlike `Tensor::argmin`, whose `==`-mask yields an out-of-range
    /// index) — finite KNN distances satisfy this. A non-16-multiple reduced
    /// width must be `±∞`-padded by the caller so padded lanes never win.
    ///
    /// # Panics
    /// Panics if the group has more than one warp, the kernel is unrolled (the
    /// flat form is a follow-up), the value `RV` dtype is not the (float) source
    /// dtype, or the index `RV` is not `Int32`.
    pub fn row_arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>) {
        let n = src.shape().len();
        self.arg_reduce(val, idx, src, dir, src.shape()[n - 3] as i64, src.shape()[n - 2] as i64, true)
    }

    /// Argmin/argmax each column of `src` into `(val, idx)` — the transpose of
    /// [`Self::row_arg_reduce`] (folds `(height, inner)`, returns the row index).
    /// Same dtype/padding preconditions.
    pub fn col_arg_reduce(&self, val: RV<'k>, idx: RV<'k>, src: &RT<'k>, dir: ArgDir) -> (RV<'k>, RV<'k>) {
        let n = src.shape().len();
        self.arg_reduce(val, idx, src, dir, src.shape()[n - 2] as i64, src.shape()[n - 3] as i64, false)
    }

    /// Shared arg-reduce body (the index-carrying [`Self::reduce`]): threads a
    /// second `Int32` index accumulator alongside the value through the in-lane
    /// fold and the cross-lane tree. The partner's index rides its OWN
    /// `ds_bpermute` with its value, so it is never re-derived from the lane id.
    /// `outer_end` is the tile dim mapped to `(val, idx)`; `acc_end` is the
    /// reduced dim; `row` selects `src[outer, acc, inner]` vs `src[acc, outer, inner]`.
    #[allow(clippy::too_many_arguments)]
    fn arg_reduce(
        &self,
        val: RV<'k>,
        idx: RV<'k>,
        src: &RT<'k>,
        dir: ArgDir,
        outer_end: i64,
        acc_end: i64,
        row: bool,
    ) -> (RV<'k>, RV<'k>) {
        assert_eq!(self.warps, 1, "arg_reduce is a single-warp op");
        assert!(!self.ker.unrolled(), "arg_reduce: unrolled (flat) form not yet implemented");
        assert!(src.elem().is_float(), "arg_reduce: value dtype must be float");
        assert_eq!(val.elem(), src.elem(), "arg_reduce: value RV dtype must match src");
        assert_eq!(idx.elem(), &DType::Int32, "arg_reduce: index RV must be Int32");

        let velem = src.elem().clone();
        let ept = src.shape()[src.shape().len() - 1] as i64;
        let val_reg = self.ker.alloc_reg(1, velem.clone());
        let idx_reg = self.ker.alloc_reg(1, DType::Int32);
        let laneid = self.laneid();
        let read0 = |buf: &Arc<UOp>| load_at(buf, &[1], &[Idx::Const(0)]);

        let outer = self.ker.raw_range(outer_end, AxisType::Loop);

        // Re-init both accumulators each outer iteration: the init stores must
        // depend on `outer` + enclosing tracked loops, or they hoist above the
        // loop and carry stale state (cf. `reduce`). One grouped END closes the
        // tiny init loop once.
        let mut init_deps: SmallVec<[Arc<UOp>; 4]> = smallvec![outer.clone()];
        init_deps.extend(self.ker.tracked_ranges());
        let i_range = self.ker.raw_range(1, AxisType::Loop);
        let v_init = flat_index(&val_reg.after(init_deps.clone()), &[1], &[Idx::from(&i_range)])
            .store(UOp::const_(velem.clone(), ConstValue::Float(dir.init())));
        let i_init = flat_index(&idx_reg.after(init_deps), &[1], &[Idx::from(&i_range)])
            .store(UOp::const_(DType::Int32, ConstValue::Int(-1)));
        let init_grp = UOp::group(vec![v_init, i_init]).end(smallvec![i_range]);

        // In-lane fold over (acc, inner): fold this element's value + its global
        // axis index into the running pair, storing both under one grouped END.
        let acc = self.ker.raw_range(acc_end, AxisType::Reduce);
        let inner = self.ker.raw_range(ept, AxisType::Reduce);
        let va = read0(&val_reg.after(smallvec![init_grp.clone(), acc.clone(), inner.clone()]));
        let ia = read0(&idx_reg.after(smallvec![init_grp.clone(), acc.clone(), inner.clone()]));
        let src_idx = if row {
            [Idx::from(&outer), Idx::from(&acc), Idx::from(&inner)]
        } else {
            [Idx::from(&acc), Idx::from(&outer), Idx::from(&inner)]
        };
        let vb = load_at(src.uop(), src.shape(), &src_idx);
        let ib = self.axis_index_of(src, &acc, &inner);
        let (vf, idf) = arg_fold(dir, &va, &ia, &vb, &ib);
        let v_fold = flat_index(&val_reg, &[1], &[Idx::Const(0)]).store(vf);
        let i_fold = flat_index(&idx_reg, &[1], &[Idx::Const(0)]).store(idf);
        let fold_grp = UOp::group(vec![v_fold, i_fold]).end(smallvec![acc, inner]);

        // Cross-lane fold via `ds_bpermute`: value and index each ride their own
        // shuffle, so the partner's winning index is transported, not re-derived.
        let v_partial = read0(&val_reg.after(smallvec![fold_grp.clone()]));
        let i_partial = read0(&idx_reg.after(smallvec![fold_grp]));
        let (mut vacc, mut iacc) = (v_partial.clone(), i_partial.clone());
        for d in self.ker.caps.reduce_tree() {
            let src_lane = imod(&iadd(&laneid, &cidx(d)), self.group_threads() as i64);
            let pv = self.shuffle_lane(&v_partial, &src_lane);
            let pi = self.shuffle_lane(&i_partial, &src_lane);
            let (v, i) = arg_fold(dir, &vacc, &iacc, &pv, &pi);
            vacc = v;
            iacc = i;
        }

        // Re-seed the OUTPUT pair to `dir.init()`/`-1` once per outer trip AND per
        // enclosing tracked loop, so a reduce *inside* a rolled loop (the KNN corpus
        // stream) starts fresh each trip instead of folding onto the previous trip's
        // result — the running-extremum hoist that an `outer`-only edge leaves open
        // (the output RVs' seed `clear_rv` carries no tracked-loop dependency, so it
        // is hoisted to `run_count = 1`; this re-seed restores the per-trip start).
        // A reduce with no enclosing tracked loop re-seeds once (`init_deps = [outer]`),
        // identical to the prior single-fold behavior. The fold then reads THIS seed,
        // not the carried buffer, so it is a fresh per-trip reduce, not a running one.
        let mut out_init: SmallVec<[Arc<UOp>; 4]> = smallvec![outer.clone()];
        out_init.extend(self.ker.tracked_ranges());
        let vo_seed = flat_index(&val.uop().after(out_init.clone()), val.shape(), &[Idx::from(&outer), Idx::Const(0)])
            .store(UOp::const_(velem.clone(), ConstValue::Float(dir.init())));
        let io_seed = flat_index(&idx.uop().after(out_init), idx.shape(), &[Idx::from(&outer), Idx::Const(0)])
            .store(UOp::const_(DType::Int32, ConstValue::Int(-1)));
        let oseed_grp = UOp::group(vec![vo_seed, io_seed]);

        // Fold the lane result into the freshly re-seeded (val[outer], idx[outer]).
        let v_in = load_at(
            &val.uop().after(smallvec![oseed_grp.clone(), outer.clone()]),
            val.shape(),
            &[Idx::from(&outer), Idx::Const(0)],
        );
        let i_in = load_at(
            &idx.uop().after(smallvec![oseed_grp, outer.clone()]),
            idx.shape(),
            &[Idx::from(&outer), Idx::Const(0)],
        );
        let (vout, iout) = arg_fold(dir, &v_in, &i_in, &vacc, &iacc);
        let v_store = flat_index(val.uop(), val.shape(), &[Idx::from(&outer), Idx::Const(0)]).store(vout);
        let i_store = flat_index(idx.uop(), idx.shape(), &[Idx::from(&outer), Idx::Const(0)]).store(iout);
        let out_grp = UOp::group(vec![v_store, i_store]).end(smallvec![outer]);
        self.finalize_pair(val, idx, out_grp)
    }
}
