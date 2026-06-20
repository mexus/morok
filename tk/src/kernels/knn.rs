//! Stages 1–2 of a fused brute-force KNN.
//!
//! **Stage 1** ([`build_knn_score`]) is the **x²-free score tile**. For query
//! rows `x[query, d]` and corpus rows `c[corpus, d]` the score is
//! `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩`. The query self-term `‖x[n]‖²` is
//! dropped (it is constant per query row `n`, so it never changes the argmin over
//! the corpus `m` that the running top-K in Stage 2 takes). The dominant distance
//! term `‖c[m]‖²` (`c_sq`) is precomputed in **f32** outside the kernel and passed
//! in (an augmentation that smuggled it through a bf16 WMMA operand would lose its
//! precision), replicated along the query axis so every `(m, n)` reads `c_sq[m]`.
//!
//! **Stage 2** ([`build_knn_topk`]) streams the corpus in `BLK`-tall tiles and
//! keeps, per query, the running unsorted top-K nearest corpus rows via a
//! flashlib-style **argmin-insert**: no score recompute, no in-kernel sort. The
//! final K-ordering is offloaded to the generic graph in Stage 3.
//!
//! Orientation (mirrors [`crate::kernels::fa::fa_qk`]'s `QKᵀ`): the **corpus `m`
//! is the reduced / row axis** and the query `n` the column, so Stage 2's running
//! top-K over the corpus folds the score tile's row — the inner-carrying axis on
//! both the gfx942 normal accumulator (matrix-col reduce) and the gfx1151 wave32
//! even/odd interleave accumulator (matrix-row reduce); the caller arranges the
//! tile, exactly as FA does. The cross MMA is `mma_atb(score, cᵀ, xᵀ)` (both
//! operands the corpus/query tiles transposed to `[d, *]` Col fragments), giving
//! `score[m, n] = Σ_d c[m,d]·x[n,d] = ⟨c[m], x[n]⟩` in the f32 accumulator.
//!
//! The `c_sq` global is loaded into a tile declared with the SAME accumulator
//! fragment as the cross MMA output, so it aligns lane-for-lane (both index the
//! accumulator frag's `lane_rc`); the combine `score = c_sq − 2·cross` is then a
//! pair of per-lane f32 elementwise ops. Arch-portable (gfx942 wave64 / gfx1151
//! wave32) via the role-based fragment shortcuts — no hardcoded fragment.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};

use crate::ArgDir;
use crate::Group;
use crate::arch::FragRole;
use crate::group::{MoveIdx, lane_rc};
use crate::index::{Idx, cidx, load_at};
use crate::kernel::Kernel;
use crate::scaffold::GlSpec;
use crate::tile::{GL, RT, RV, RegTile};
use crate::tiles::{TileLayout, VecLayout};

/// The WMMA tile edge (K=16); the cross MMA operates on 16×16 fragments, so the
/// corpus / query / D dims must each be a multiple of it. Also the corpus stream
/// tile height (`BM`) and the top-K slot padding (`K_pad`).
const BLK: usize = 16;

/// The GPU arch(es) this kernel is built for: gfx942 (CDNA3 MFMA, wave64) and
/// gfx1151 (RDNA3.5 WMMA, wave32). Both resolve the accumulator/operand fragments
/// by role through [`crate::ArchCaps`]; the launcher gates against this list.
/// Validated on gfx942 (CDNA3) and gfx1151 (RDNA3.5).
pub const KNN_SUPPORTED_ARCHS: &[svod_dtype::AmdArch] = &[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151];

const POS_INF: f64 = f64::INFINITY;
const NEG_INF: f64 = f64::NEG_INFINITY;

fn fconst(dt: &DType, v: f64) -> Arc<UOp> {
    UOp::const_(dt.clone(), ConstValue::Float(v))
}
fn iconst32(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(v))
}

/// Anchor a register tile's next op on the corpus-loop range carried by `m_blk`:
/// `t.after([m_blk])` when `m_blk` is the rolled loop index (`Idx::Uop`), a no-op for
/// a `Const` tile index (single-tile Stage 1). The rolled-loop re-init footgun
/// ([`crate::loop_scope`]): a constant fill with no loop dependency hoists to
/// `run_count = 1`.
fn reinit_on<'k>(t: RT<'k>, m_blk: &Idx) -> RT<'k> {
    match m_blk {
        Idx::Uop(u) => t.after(u),
        Idx::Const(_) => t,
    }
}

/// The cross-term + `c_sq` combine yielding one `[BLK, query]` Col f32 score tile
/// `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩` for corpus rows `[m_blk·BLK, +BLK)` —
/// the Stage-1 machinery, factored so Stage 2 calls it per `BLK`-tall corpus tile
/// (and [`build_knn_score`] still uses it for the whole corpus in one tile).
/// `m_rows` is the corpus-tile height (the full `corpus` for Stage 1, [`BLK`] per
/// stream tile for Stage 2). `x_reg_t` is the loop-invariant query operand `[d,
/// query]` (the caller loads it once); `m_blk` is the corpus tile index (its
/// row-block offset, in units of `m_rows`). `masked` gates the GLOBAL→LDS/REG hops
/// against the true corpus extent so a ragged final tile reads `0.0` instead of
/// touching out-of-bounds memory (the caller then masks those rows to `+∞` for the
/// argmin).
#[allow(clippy::too_many_arguments)]
fn score_tile<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    m_rows: usize,
    query: usize,
    d: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL<'k>,
    c_sq_gl: &GL<'k>,
    m_blk: &Idx,
    masked: bool,
) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // GLOBAL(corpus tile m_blk) → LDS (swizzled) → REG, transposed to the `[d, m_rows]`
    // Col operand for the contraction over D (mirrors `fa_qk`'s Kᵀ).
    let c_smem = ker.shared_sw((m_rows, d), bf16.clone(), row);
    let c_reg = ker.operand((m_rows, d), bf16.clone(), row);
    let c_reg_t = ker.operand((d, m_rows), bf16.clone(), col);

    let c_idx = [Idx::Const(0), Idx::Const(0), m_blk.clone(), Idx::Const(0)];
    let c_smem = warp.load(c_smem, c_gl.clone(), MoveIdx::block(&c_idx, 2));
    let c_reg = warp.load(c_reg, c_smem, MoveIdx::default());
    let c_reg_t = warp.transpose(c_reg_t, &c_reg);

    // score[m, n] = Σ_d c[m,d]·x[n,d] = ⟨c[m], x[n]⟩ (corpus = row, query = col).
    // The MMA accumulator must be RE-ZEROED each corpus iteration: when `m_blk` is the
    // rolled corpus-loop index, anchor the zero-fill on it (`cross.after([loop_range])`,
    // exactly `fa_qk`'s `warp.zero(lp.reinit(att))`) or the constant fill hoists out of
    // the loop (`run_count = 1`) and the MMA accumulates the cross term across ALL tiles.
    // A `Const` `m_blk` (the single-tile Stage-1 path) adds no dependency.
    let cross = warp.zero(reinit_on(ker.acc((m_rows, query), col), m_blk));
    let cross = warp.mma_atb(cross, &c_reg_t, x_reg_t);

    // Load c_sq_rep[m_blk] into the SAME accumulator fragment + layout as the cross
    // MMA output, so the two align lane-for-lane (both index the accumulator frag's
    // `lane_rc`; orientation-robust per the reductions/masked tests).
    let cs_idx = [Idx::Const(0), Idx::Const(0), m_blk.clone(), Idx::Const(0)];
    let cs_mi = MoveIdx::block(&cs_idx, 2);
    let cs_mi = if masked { cs_mi.masked() } else { cs_mi };
    let c_sq = warp.load(ker.acc((m_rows, query), col), c_sq_gl.clone(), cs_mi);

    // score = c_sq − 2·cross, all f32.
    let cross = warp.mul_scalar(cross, -2.0);
    warp.add(cross, &c_sq)
}

/// Build the x²-free KNN score-tile kernel into the bound ABI.
///
/// ABI (outputs then inputs, fixed by [`Kernel::bind_abi`]):
/// - `score` (`[1, 1, corpus, query]`, f32) — the output `‖c[m]‖² − 2·⟨x[n],c[m]⟩`.
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, corpus, query]`, f32) — `‖c[m]‖²` precomputed outside the
///   kernel and replicated along the query axis (each `(m, n)` holds `c_sq[m]`).
///
/// Single-warp; `corpus`, `query`, `d` must each be a multiple of [`BLK`] (16).
///
/// # Panics
/// Panics unless `corpus`, `query`, and `d` are each a multiple of 16.
pub fn build_knn_score(ker: &Kernel, corpus: usize, query: usize, d: usize) {
    Kernel::assert_divisible(corpus, BLK, "KNN corpus");
    Kernel::assert_divisible(query, BLK, "KNN query");
    Kernel::assert_divisible(d, BLK, "KNN D");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let warp = ker.warp();

    // ABI: output (score, f32) then inputs (x, c — bf16; c_sq_rep — f32).
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, corpus, query], f32.clone())],
        &[
            GlSpec::new(&[1, 1, query, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, query], f32.clone()),
        ],
    );
    let (score_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL) =
        (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    // Query tile loaded once and transposed to its `[d, query]` Col fragment.
    let x_reg_t = load_query_t(ker, &warp, query, d, &x_gl);

    // The whole corpus in one `(corpus, query)` tile (the Stage-1 single-store shape).
    let score = score_tile(ker, &warp, corpus, query, d, &x_reg_t, &c_gl, &c_sq_gl, &Idx::Const(0), false);
    let zero4 = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::Const(0)];
    let _ = warp.store(score_gl, score, MoveIdx::block(&zero4, 2));
}

/// Load the query tile `[query, d]` and transpose it to its `[d, query]` Col
/// operand fragment for the cross contraction over D — loop-invariant, so both
/// builders load it once.
fn load_query_t<'k>(ker: &'k Kernel, warp: &Group<'k>, query: usize, d: usize, x_gl: &GL<'k>) -> RT<'k> {
    let bf16 = DType::BFloat16;
    let (row, col) = (TileLayout::Row, TileLayout::Col);
    let x_smem = ker.shared_sw((query, d), bf16.clone(), row);
    let x_reg = ker.operand((query, d), bf16.clone(), row);
    let x_reg_t = ker.operand((d, query), bf16.clone(), col);
    let zero4 = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::Const(0)];
    let x_smem = warp.load(x_smem, x_gl.clone(), MoveIdx::block(&zero4, 2));
    let x_reg = warp.load(x_reg, x_smem, MoveIdx::default());
    warp.transpose(x_reg_t, &x_reg)
}

/// Per-query running top-K state: two Col-layout `[K_pad=BLK, query]` register
/// tiles — `val` (f32, K-slot = row, seeded `+∞`) and `idx` (Int32, seeded `−1`).
/// `row_arg_reduce` folds the K-slot (row) axis on both archs, so a `Max` reduce
/// yields the per-query running-worst slot to evict.
struct TopK<'k> {
    val: RT<'k>,
    idx: RT<'k>,
}

/// Build the x²-free KNN running-top-K kernel into the bound ABI.
///
/// ABI (outputs then inputs):
/// - `idx` (`[1, 1, query, k]`, Int32) — the **unsorted** K nearest corpus indices
///   per query (final K-ordering offloaded to the generic graph in Stage 3).
/// - `val` (`[1, 1, query, k]`, f32) — their x²-free scores.
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, corpus, query]`, f32) — `‖c[m]‖²` replicated along query.
///
/// Single-warp, correctness-first, arch-portable via role fragments. The corpus is
/// streamed in [`BLK`]-tall tiles through a [`crate::loop_scope::Loop`]; per tile
/// the running top-K is updated by up to `k` argmin-insert steps. Built **rolled**
/// (`arg_reduce` panics under unroll).
///
/// # Panics
/// Panics unless `query`/`d` are multiples of [`BLK`], `corpus > 0`, `1 ≤ k ≤ BLK`,
/// and `query ≤ BLK` (the v1 single-query-fragment constraint: a wider query would
/// fold distinct queries together in the per-query `row_arg_reduce`).
pub fn build_knn_topk(ker: &Kernel, corpus: usize, query: usize, d: usize, k: usize) {
    Kernel::assert_divisible(query, BLK, "KNN topk query");
    Kernel::assert_divisible(d, BLK, "KNN topk D");
    assert!(corpus > 0, "KNN topk corpus must be > 0");
    assert!((1..=BLK).contains(&k), "KNN topk k must be in 1..=16");
    assert!(query <= BLK, "KNN topk query must be <= 16 (single query fragment) for v1");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let i32 = DType::Int32;
    let col = TileLayout::Col;
    let warp = ker.warp();
    let acc_frag = ker.caps.frag(FragRole::Accumulator);

    // ABI: outputs (idx i32, val f32) then inputs (x, c — bf16; c_sq_rep — f32).
    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[1, 1, query, k], i32.clone()), GlSpec::new(&[1, 1, query, k], f32.clone())],
        &[
            GlSpec::new(&[1, 1, query, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, d], bf16.clone()),
            GlSpec::new(&[1, 1, corpus, query], f32.clone()),
        ],
    );
    let (idx_gl, val_gl, x_gl, c_gl, c_sq_gl): (GL, GL, GL, GL, GL) =
        (outs[0].clone(), outs[1].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());

    let x_reg_t = load_query_t(ker, &warp, query, d, &x_gl);

    // Running top-K state (Col `[K_pad=BLK, query]`). The fragment is 16-wide but only
    // the first `k` K-slots are live; seed slots `[0, k)` to `+∞` (empty, fillable) and
    // the padding `[k, 16)` to `−∞` so the `row_arg_reduce(Max)` worst-slot search NEVER
    // evicts into a padding slot (a `−∞` always loses the Max to a real `+∞`/finite
    // slot). Without this the worst is forever the `+∞` of an unused slot and inserts
    // leak past the stored first-`k` columns. `idx` seeds to `−1` everywhere.
    let val0 = seed_topk_val(ker, &warp, query, k);
    let idx0 = warp.map(ker.rt((BLK, query), i32.clone(), col, acc_frag), |_, _| iconst32(-1));
    let topk = TopK { val: val0, idx: idx0 };

    // Stream the corpus in BLK-tall tiles via the FA running-state Loop carry.
    let tiles = corpus.div_ceil(BLK);
    let masked = !corpus.is_multiple_of(BLK);
    let lp = ker.loop_static(tiles as i64);
    let m_tile = lp.index().clone();
    let topk = TopK { val: lp.reinit(topk.val), idx: lp.reinit(topk.idx) };

    let topk = topk_insert(ker, &warp, corpus, query, d, k, &x_reg_t, &c_gl, &c_sq_gl, &m_tile, masked, topk);

    // Close the loop once: `topk_insert` chained the idx-evict store (the last
    // terminal) to depend on the val-evict store and the score updates, so the single
    // loop-closing END scopes the whole insert body (the matmul multi-accumulator
    // idiom). Both carried tiles then read their post-loop value via `.after([end])`.
    let ended = lp.close();
    let idx_after = topk.idx.after(&ended);
    let val_after = topk.val.after(&ended);

    store_topk(ker, &warp, query, k, &idx_gl, &val_gl, &idx_after, &val_after);
}

/// One corpus tile's argmin-insert: compute the score sub-tile, mask ragged rows
/// to `+∞`, then run up to `k` steps of (find the per-query tile-min over corpus,
/// compare to the running worst slot, conditionally evict, remove the consumed
/// element). Returns the updated running top-K.
#[allow(clippy::too_many_arguments)]
fn topk_insert<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    corpus: usize,
    query: usize,
    d: usize,
    k: usize,
    x_reg_t: &RT<'k>,
    c_gl: &GL<'k>,
    c_sq_gl: &GL<'k>,
    m_tile: &Arc<UOp>,
    masked: bool,
    mut topk: TopK<'k>,
) -> TopK<'k> {
    // score[m, n] for this corpus tile (corpus = row, query = col).
    let mut score = score_tile(ker, warp, BLK, query, d, x_reg_t, c_gl, c_sq_gl, &Idx::Uop(m_tile.clone()), masked);
    if masked {
        score = mask_ragged_rows(ker, warp, score, m_tile, corpus);
    }

    // Each step's stores are CHAINED so a single loop-closing END (in the caller)
    // scopes the whole insert body inside the rolled corpus loop (the matmul
    // multi-accumulator idiom): idx-evict ← val-evict, score-remove ← idx-evict, and
    // the next step's reduces read the chained `score`/`topk.val`. The idx-evict of
    // the LAST step is the loop's terminal store, so its `remove_used` (dead — no next
    // argmin) is skipped, leaving idx-evict last on the store stack for `lp.close()`.
    for step in 0..k {
        // a. per-query tile-min value + in-tile corpus index; global_m = m_tile·BLK + in-tile.
        let (row_min, row_arg) =
            warp.row_arg_reduce(seed_val(ker, warp, POS_INF), seed_idx(ker, warp), &score, ArgDir::Min);
        // `global_m = m_tile·BLK + row_arg` in a FRESH RV — `warp.map` rewrites its
        // tile in place, so mapping `row_arg` directly would clobber the in-tile index
        // that `remove_used` still needs to mask the consumed element (the `k > 1`
        // per-step argmin-skip). Read `row_arg` (anchored) into the new buffer instead.
        let mbase = m_tile.mul(&cidx(BLK as i64)).cast(DType::Int32);
        let (ra_buf, ra_shape) = (warp.anchor(row_arg.uop()), row_arg.shape().to_vec());
        let global_m = warp.map(seed_idx(ker, warp), move |_, idx| {
            load_at(&ra_buf, &ra_shape, idx).try_add(&mbase).expect("global_m")
        });

        // b. per-query running-worst value + its K-slot index.
        let (worst, evict) =
            warp.row_arg_reduce(seed_val(ker, warp, NEG_INF), seed_idx(ker, warp), &topk.val, ArgDir::Max);

        // d. Evict (conditional rewrite by K-slot): write row_min/global_m into the
        //    `evict[query]` K-slot where `row_min[query] < worst[query]` (do_insert).
        //    Chain idx-evict after val-evict so both carried writes share one END.
        topk.val = evict_slot(ker, warp, topk.val, &evict, &row_min, &worst, &row_min);
        topk.idx = evict_slot(ker, warp, topk.idx.after(&topk.val), &evict, &row_min, &worst, &global_m);

        // e. Remove the consumed corpus element so the next step's argmin skips it
        //    (chained after idx-evict). Skipped on the last step (no next argmin), so
        //    idx-evict stays the loop's terminal store.
        if step + 1 < k {
            score = remove_used(ker, warp, score.after(&topk.idx), &row_arg, &row_min, &worst);
        }
    }
    topk
}

/// A length-`BLK` f32 RV seeded to `init` (the `row_arg_reduce` value accumulator;
/// it actually overwrites the seed with `dir.init()`, but a same-dtype seed keeps
/// the alloc explicit). Length `BLK` matches the single query/K fragment edge.
fn seed_val<'k>(ker: &'k Kernel, warp: &Group<'k>, init: f64) -> RV<'k> {
    let frag = ker.caps.frag(FragRole::Accumulator);
    warp.clear_rv(ker.rv(BLK, DType::Float32, VecLayout::Ortho, frag), init)
}
/// A length-`BLK` Int32 index RV seeded to `−1` (the `row_arg_reduce` index acc).
fn seed_idx<'k>(ker: &'k Kernel, warp: &Group<'k>) -> RV<'k> {
    let frag = ker.caps.frag(FragRole::Accumulator);
    warp.clear_rv(ker.rv(BLK, DType::Int32, VecLayout::Ortho, frag), -1.0)
}

/// Seed the running-top-K value tile (Col `[K_pad=BLK, query]`): K-slots `[0, k)`
/// to `+∞` (empty, fillable), the padding slots `[k, 16)` to `−∞` so the worst-slot
/// `row_arg_reduce(Max)` never evicts into a padding slot. The per-element K-slot is
/// its matrix-row coordinate off the frag's OWN `lane_rc` (arch-correct for the
/// gfx942 stride and the gfx1151 even/odd interleave).
fn seed_topk_val<'k>(ker: &'k Kernel, warp: &Group<'k>, query: usize, k: usize) -> RT<'k> {
    let tile = ker.acc((BLK, query), TileLayout::Col);
    let laneid = ker.laneid();
    let (interleave, interleave_t, stride) = (tile.base.interleave, tile.base.interleave_t, tile.base.stride as i64);
    let transpose = tile.layout == TileLayout::Col;
    warp.map(tile, move |x, idx| {
        let (k_if, _q_if) = lane_rc(transpose, interleave, interleave_t, &laneid, 16, 16, stride, &idx[2].to_uop());
        let k_pos = idx[0].to_uop().mul(&cidx(BLK as i64)).add(&k_if);
        UOp::try_where(k_pos.lt(&cidx(k as i64)), fconst(&x.dtype(), POS_INF), fconst(&x.dtype(), NEG_INF))
            .expect("topk val seed where")
    })
}

/// Read a per-query RV scalar inside a `map` over a Col `[*, query]` tile: the
/// query is the tile's column / width axis, so it selects RV slot `idx[1]` exactly
/// as [`crate::math`]'s `combine_rv` does for a Col tile. The RV buffer is anchored
/// so a constant-address read of a carried (loop) RV is not hoisted out of the
/// corpus loop. Returns `(anchored_buf, shape)` to capture into the `map` closure.
fn rv_query_src<'k>(warp: &Group<'k>, rv: &RV<'k>) -> (Arc<UOp>, Vec<usize>) {
    (warp.anchor(rv.uop()), rv.shape().to_vec())
}

/// Mask ragged corpus rows (`global_m ≥ corpus`) of a Col `[BLK, query]` score
/// tile to `+∞`, so the per-query argmin never selects the padding the masked
/// score load zeroed. The per-element corpus row is read off the score frag's OWN
/// `lane_rc` map (arch-correct for the gfx942 stride and the gfx1151 even/odd
/// interleave) — exactly `fa_qk`'s `kv_pos ≥ valid_len` form, with the bound the
/// (build-time constant) corpus extent.
fn mask_ragged_rows<'k>(ker: &'k Kernel, warp: &Group<'k>, score: RT<'k>, m_tile: &Arc<UOp>, corpus: usize) -> RT<'k> {
    let bound = cidx(corpus as i64);
    let laneid = ker.laneid();
    let (interleave, interleave_t, stride) = (score.base.interleave, score.base.interleave_t, score.base.stride as i64);
    let transpose = score.layout == TileLayout::Col;
    let m_tile = m_tile.clone();
    warp.map(score, move |x, idx| {
        let (m_if, _q_if) = lane_rc(transpose, interleave, interleave_t, &laneid, 16, 16, stride, &idx[2].to_uop());
        let global_m = m_tile.mul(&cidx(BLK as i64)).add(&idx[0].to_uop().mul(&cidx(BLK as i64))).add(&m_if);
        UOp::try_where(global_m.ge(&bound), fconst(&x.dtype(), POS_INF), x.clone()).expect("ragged-row mask where")
    })
}

/// Evict step (used for BOTH the f32 value tile and its Int32 index partner): in
/// the Col `[BLK, query]` tile, overwrite the element at the per-query worst K-slot
/// `evict[query]` with `repl[query]` when `do_insert = row_min[query] <
/// worst[query]`, leaving every other slot. Selecting both outputs (value tile and
/// index tile) by the SAME predicate keeps the kept value paired with its index.
/// The K-slot of an element is its matrix-row coordinate off the frag's `lane_rc`;
/// the per-query RVs are read by the col-tile index (`idx[1]`), the multi-RV
/// generalization of `combine_rv`.
#[allow(clippy::too_many_arguments)]
fn evict_slot<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    tile: RT<'k>,
    evict: &RV<'k>,
    row_min: &RV<'k>,
    worst: &RV<'k>,
    repl: &RV<'k>,
) -> RT<'k> {
    let laneid = ker.laneid();
    let (interleave, interleave_t, stride) = (tile.base.interleave, tile.base.interleave_t, tile.base.stride as i64);
    let transpose = tile.layout == TileLayout::Col;
    let (e_buf, e_shape) = rv_query_src(warp, evict);
    let (rmin_buf, rmin_shape) = rv_query_src(warp, row_min);
    let (worst_buf, worst_shape) = rv_query_src(warp, worst);
    let (repl_buf, repl_shape) = rv_query_src(warp, repl);
    warp.map(tile, move |x, idx| {
        let (k_if, _q_if) = lane_rc(transpose, interleave, interleave_t, &laneid, 16, 16, stride, &idx[2].to_uop());
        let k_pos = idx[0].to_uop().mul(&cidx(BLK as i64)).add(&k_if).cast(DType::Int32);
        let q = idx[1].clone();
        let e = load_at(&e_buf, &e_shape, &[q.clone(), Idx::Const(0)]);
        let rmin = load_at(&rmin_buf, &rmin_shape, &[q.clone(), Idx::Const(0)]);
        let wst = load_at(&worst_buf, &worst_shape, &[q.clone(), Idx::Const(0)]);
        let mut rpl = load_at(&repl_buf, &repl_shape, &[q, Idx::Const(0)]);
        if rpl.dtype() != x.dtype() {
            rpl = rpl.cast(x.dtype());
        }
        let do_insert = rmin.lt(&wst);
        let hit = k_pos.eq(&e).and_(&do_insert);
        UOp::try_where(hit, rpl, x.clone()).expect("evict where")
    })
}

/// Remove the consumed corpus element from a Col `[BLK, query]` score tile: set
/// `score[m == row_arg[query], query] = +∞` where `row_min[query] < worst[query]`
/// (i.e. the element actually inserted this step), so the next step's argmin skips
/// it. `row_arg` is the in-tile corpus row (0..BLK) returned by `row_arg_reduce`,
/// compared directly against the element's in-tile matrix-row coordinate.
fn remove_used<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    score: RT<'k>,
    row_arg: &RV<'k>,
    row_min: &RV<'k>,
    worst: &RV<'k>,
) -> RT<'k> {
    let laneid = ker.laneid();
    let (interleave, interleave_t, stride) = (score.base.interleave, score.base.interleave_t, score.base.stride as i64);
    let transpose = score.layout == TileLayout::Col;
    let (ra_buf, ra_shape) = rv_query_src(warp, row_arg);
    let (rmin_buf, rmin_shape) = rv_query_src(warp, row_min);
    let (worst_buf, worst_shape) = rv_query_src(warp, worst);
    warp.map(score, move |x, idx| {
        let (m_if, _q_if) = lane_rc(transpose, interleave, interleave_t, &laneid, 16, 16, stride, &idx[2].to_uop());
        let m_local = idx[0].to_uop().mul(&cidx(BLK as i64)).add(&m_if).cast(DType::Int32);
        let q = idx[1].clone();
        let ra = load_at(&ra_buf, &ra_shape, &[q.clone(), Idx::Const(0)]);
        let rmin = load_at(&rmin_buf, &rmin_shape, &[q.clone(), Idx::Const(0)]);
        let wst = load_at(&worst_buf, &worst_shape, &[q, Idx::Const(0)]);
        let do_insert = rmin.lt(&wst);
        let hit = m_local.eq(&ra).and_(&do_insert);
        UOp::try_where(hit, fconst(&x.dtype(), POS_INF), x.clone()).expect("remove-used where")
    })
}

/// Store the unsorted running top-K to the `[1, 1, query, k]` outputs. The running
/// tiles are Col `[K_slot=BLK, query]`; the output wants `[query, K_slot]`, the
/// transpose. So each tile is `transpose`d into a Row `[query, BLK]`
/// AccumulatorT-fragment tile (the FA output-store relayout) and stored with the
/// boundary mask, which drops the columns `≥ k` (the K-slots past the requested
/// `k`). `k == BLK` needs no mask; a partial `k` gates the trailing columns.
#[allow(clippy::too_many_arguments)]
fn store_topk<'k>(
    ker: &'k Kernel,
    warp: &Group<'k>,
    query: usize,
    k: usize,
    idx_gl: &GL<'k>,
    val_gl: &GL<'k>,
    idx_after: &RT<'k>,
    val_after: &RT<'k>,
) {
    let row = TileLayout::Row;
    let i32 = DType::Int32;
    let acc_t = ker.caps.frag(FragRole::AccumulatorT);
    let zero4 = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::Const(0)];
    let mi = if k.is_multiple_of(BLK) { MoveIdx::block(&zero4, 2) } else { MoveIdx::block(&zero4, 2).masked() };

    // Transpose Col [BLK, query] → Row [query, BLK] (AccumulatorT), then masked-store
    // its first `k` columns to the [query, k] output. Both transposes (which push
    // intermediate REG stores) come BEFORE both global stores, so the kernel's final
    // two terminal stores — popped by `finish(2)` as the SINK sources — are exactly
    // the two output writes.
    let val_t = warp.transpose(ker.acc_t((query, BLK), row), val_after);
    let idx_t = warp.transpose(ker.rt((query, BLK), i32.clone(), row, acc_t), idx_after);
    let _ = warp.store(val_gl.clone(), val_t, mi);
    let _ = warp.store(idx_gl.clone(), idx_t, mi);
}
