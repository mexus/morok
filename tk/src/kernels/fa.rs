//! Flash-attention forward — a port of tinygrad `test_tk.py::test_fa`'s forward
//! kernel (the online-softmax attention, *not* `fa.py`'s jit/backward).
//!
//! One workgroup (single wave64 warp) owns one `(head, q_block, batch)` triple:
//! it loads its Q tile into registers, then streams the K/V blocks, computing
//! `QKᵀ` with [`mma_atb`](crate::Group::mma_atb), applying the causal mask, the
//! running-max online softmax (the LDS cross-lane [`row_reduce`]s), and the `A·V`
//! accumulation, before normalizing and writing the transposed output tile back.

use std::sync::Arc;

use smallvec::smallvec;
use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::Group;
use crate::index::{Idx, load_at};
use crate::kernel::Kernel;
use crate::tile::{RT, RV, RegTile, ST};
use crate::tiles::{RT_16X16, ST_16X16, TileLayout, VecLayout};

/// The WMMA tile edge (gfx942 K=16). The QKᵀ / A·V WMMAs always operate on
/// 16×16 fragments; Q/KV per-warp *tiles* are grids of `BLK`-edged fragments
/// ([`Q_BLK`]/[`KV_BLK`]).
const BLK: usize = 16;

/// Multi-wave warps per workgroup (the FA-5 occupancy lift): 8 wave64 warps =
/// `8 * 64 = 512` threads per block. Each warp owns a distinct Q-tile; all 8
/// share one K/V LDS slot, filled collaboratively across the 512 threads.
const NUM_WARPS: usize = 8;

/// Default per-warp Q-tile height for the production double-buffered path
/// ([`flash_attention_forward_mw_db`]). FA-4 measured `{16,16}` (the WMMA edge)
/// as the non-regressing default on gfx942: bigger register tiles raise VGPR
/// pressure and drop occupancy (the bottleneck). `{32,32}`/`{32,64}` stay
/// opt-in via the explicit-tile [`build_fa_mw_db`] args.
const Q_BLK: usize = 16;
/// Default per-warp KV-tile (super-block) height. See [`Q_BLK`].
const KV_BLK: usize = 16;

fn iconst(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

/// The GPU arch(es) the flash-attention kernels are built for. They use the gfx942
/// WMMA descriptor (`mfma.f32.16x16x16bf16.1k`) + 64-lane `ds.bpermute`, so only
/// gfx942 today. Adding another GPU (e.g. gfx1151) means adding its arch here once
/// its arch-specific kernel bits exist — the launchers gate against this list, the
/// generic launch infra stays arch-agnostic.
pub const FA_SUPPORTED_ARCHS: &[svod_dtype::AmdArch] = &[svod_dtype::AmdArch::Gfx942];

/// Validate the kernel inputs' device against [`FA_SUPPORTED_ARCHS`] + LLVM toolchain.
fn fa_check_target(t: &Tensor) -> crate::LaunchResult<()> {
    crate::target::check_target(&t.device(), FA_SUPPORTED_ARCHS)
}

/// Tuning knobs for the multi-wave flash-attention builders ([`build_fa_mw_db`],
/// [`build_fa_mw_rdb`]) — the structured replacement for their former positional
/// `bool`/tile args (mirrors [`crate::kernels::matmul::MatmulCfg`]). [`Default`] is
/// the production baseline: `{16,16}` per-warp tile, rolled (looped) compute.
///
/// `unroll` is read only by [`build_fa_mw_rdb`]; `pipelined` only by
/// [`build_fa_mw_db`] — the shape (`b,n,h,h_kv,d`) stays a positional arg since it's
/// derived from the input tensors, not a tuning choice. `causal` is read by
/// [`build_fa_mw_rdb`] (the legacy builders are causal-only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FaConfig {
    /// Per-warp Q-tile height (a multiple of the WMMA edge `16`).
    pub q_blk: usize,
    /// Per-warp KV super-block height (a multiple of `16`).
    pub kv_blk: usize,
    /// [`build_fa_mw_rdb`] only: emit the fully-unrolled (flat) QKᵀ/softmax/A·V body.
    pub unroll: bool,
    /// [`build_fa_mw_db`] only: barrier-reduced pipelined fills (stage 2) vs naive (stage 1).
    pub pipelined: bool,
    /// [`build_fa_mw_rdb`] only: causal masking + KV block-skip. `false` is the full
    /// (bidirectional) attention sweep over every KV super-block.
    pub causal: bool,
}

impl Default for FaConfig {
    fn default() -> Self {
        Self { q_blk: Q_BLK, kv_blk: KV_BLK, unroll: false, pipelined: false, causal: true }
    }
}

/// Build the flash-attention forward SINK for `[B, N, H, D]` Q/O and
/// `[B, N, H_KV, D]` K/V (`H` a multiple of `H_KV` for GQA). `D` and the
/// sequence length `N` must be multiples of [`BLK`].
///
/// Delegates to [`build_fa_kv`] with `causal_skip = true` (the production
/// causal block-skip path).
pub fn build_fa(ker: &Kernel, b: usize, n: usize, h: usize, h_kv: usize, d: usize) {
    build_fa_kv(ker, b, n, h, h_kv, d, true);
}

/// Flash-attention forward, parameterized on the KV-loop bound for benchmarking.
///
/// `causal_skip = true` is the production path: query block `q_seq` trips the KV
/// loop only over blocks `0..=q_seq` (dynamic bound `q_seq + 1`), halving the
/// average KV trip count. `causal_skip = false` trips the *full* `n / BLK` blocks
/// and relies solely on the per-element causal mask for correctness — the A/B
/// baseline that isolates the block-skip win. Both produce identical results;
/// `true` reproduces [`build_fa`]'s structure byte-for-byte.
pub(crate) fn build_fa_kv(ker: &Kernel, b: usize, n: usize, h: usize, h_kv: usize, d: usize, causal_skip: bool) {
    assert_eq!(d % BLK, 0, "D must be a multiple of {BLK}");
    assert_eq!(n % BLK, 0, "N must be a multiple of {BLK}");
    assert_eq!(h % h_kv, 0, "H must be a multiple of H_KV");
    let group_size = (h / h_kv) as i64;
    let warp = ker.warp();

    // out, then q, k, v (declaration order = ABI slot order).
    let o = ker.gl(&[b, n, h, d], DType::BFloat16);
    let q = ker.gl(&[b, n, h, d], DType::BFloat16);
    let k = ker.gl(&[b, n, h_kv, d], DType::BFloat16);
    let v = ker.gl(&[b, n, h_kv, d], DType::BFloat16);

    let head = ker.block_idx[0].clone();
    let head_kv = head.idiv(&iconst(group_size));
    let batch = ker.block_idx[2].clone();
    let q_seq = ker.block_idx[1].clone();

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    let k_smem = ker.st((BLK, d), bf16.clone(), row, ST_16X16);
    let v_smem = ker.st((BLK, d), bf16.clone(), row, ST_16X16);

    let q_reg_fl = ker.rt((BLK, d), f32.clone(), row, RT_16X16);
    let q_reg = ker.rt((BLK, d), bf16.clone(), row, RT_16X16);
    let q_reg_t = ker.rt((d, BLK), bf16.clone(), col, RT_16X16);
    let k_reg = ker.rt((BLK, d), bf16.clone(), row, RT_16X16);
    let k_reg_t = ker.rt((d, BLK), bf16.clone(), col, RT_16X16);
    let v_reg = ker.rt((BLK, d), bf16.clone(), col, RT_16X16);
    let o_reg = ker.rt((d, BLK), f32.clone(), col, RT_16X16);
    let o_reg_t = ker.rt((BLK, d), f32.clone(), row, RT_16X16);
    let att = ker.rt((BLK, BLK), f32.clone(), col, RT_16X16);
    let att_mma = ker.rt((BLK, BLK), bf16, col, RT_16X16);
    let max_vec = ker.rv(BLK, f32.clone(), VecLayout::Ortho, RT_16X16);
    let norm_vec = ker.rv(BLK, f32.clone(), VecLayout::Ortho, RT_16X16);
    let max_vec_last = ker.rv(BLK, f32.clone(), VecLayout::Ortho, RT_16X16);
    let scale_vec = ker.rv(BLK, f32, VecLayout::Ortho, RT_16X16);

    let mut max_vec = warp.neg_inf_rv(max_vec);
    let mut norm_vec = warp.zero_rv(norm_vec);
    let mut o_reg = warp.zero(o_reg);
    let mut scale_vec = warp.ones_rv(scale_vec);
    let mut max_vec_last = max_vec_last;

    // Load + scale the Q tile, then transpose it for the QKᵀ contraction.
    let q_idx = [Idx::from(&batch), Idx::from(&q_seq), Idx::from(&head), Idx::Const(0)];
    let q_reg_fl = warp.load(q_reg_fl.into(), q.into(), &[], &q_idx, 1).rt();
    let q_reg_fl = q_reg_fl * ((1.0 / (d as f64).sqrt()) * std::f64::consts::LOG2_E);
    let q_reg = warp.copy(q_reg, &q_reg_fl);
    let q_reg_t = warp.transpose(q_reg_t, &q_reg);

    // Causal block-skip: query block `q_seq` attends only to KV blocks
    // `0..=q_seq` (all keys in block j ≤ q_seq are valid). The dynamic bound
    // `q_seq + 1` halves the average KV trip count vs the full `n/BLK`. The
    // per-element mask below stays correct for free: it is a no-op on the
    // sub-diagonal blocks and the triangular mask on the diagonal (last trip).
    // With `causal_skip = false` the loop trips the full `n/BLK` blocks and the
    // mask alone enforces causality (the A/B baseline).
    let kv_idx = if causal_skip {
        let kv_bound = q_seq.add(&iconst(1));
        ker.range_uop(kv_bound)
    } else {
        ker.range((n / BLK) as i64)
    };
    {
        let kidx = [Idx::from(&batch), Idx::from(&kv_idx), Idx::from(&head_kv), Idx::Const(0)];
        let k_smem = warp.load(k_smem.into(), k.into(), &[], &kidx, 1).st();
        let v_smem = warp.load(v_smem.into(), v.into(), &[], &kidx, 1).st();
        let k_reg = warp.load(k_reg.into(), k_smem.into(), &[], &[], 0).rt();
        let v_reg = warp.load(v_reg.into(), v_smem.into(), &[], &[], 0).rt();

        // QKᵀ into a freshly-zeroed att tile.
        let att = warp.zero(att.after(smallvec![kv_idx.clone()]));
        let k_reg_t = warp.transpose(k_reg_t, &k_reg);
        let att = warp.mma_atb(att, &k_reg_t, &q_reg_t);

        // Causal mask: drop keys ahead of this query (set the score to -∞).
        let laneid = ker.laneid();
        let q_base = laneid.mod_(&iconst(16)).add(&q_seq.mul(&iconst(BLK as i64)));
        let kv_base = laneid.idiv(&iconst(16)).mul(&iconst(4)).add(&kv_idx.mul(&iconst(BLK as i64)));
        let att = warp.map(att, move |x, idx| {
            let kv_pos = kv_base.add(&idx[0].to_uop().mul(&iconst(16))).add(&idx[2].to_uop());
            let q_pos = q_base.add(&idx[1].to_uop().mul(&iconst(16)));
            let cond = kv_pos.gt(&q_pos);
            let neg_inf = UOp::const_(x.dtype(), ConstValue::Float(f64::NEG_INFINITY));
            UOp::try_where(cond, neg_inf, x.clone()).expect("causal where")
        });

        // Online softmax: update the running max, rescale the running stats by
        // exp2(prev_max - new_max), exponentiate, and accumulate the norm.
        max_vec_last = warp.copy(max_vec_last.after(smallvec![kv_idx.clone()]), &max_vec);
        max_vec = warp.row_reduce(
            max_vec.after(smallvec![max_vec_last.uop().clone()]),
            &att,
            |a, b| a.max(b),
            f64::NEG_INFINITY,
        );

        let (mvl_buf, mvl_shape) = (max_vec_last.uop().clone(), max_vec_last.shape().to_vec());
        let (mv_buf, mv_shape) = (max_vec.uop().clone(), max_vec.shape().to_vec());
        scale_vec = warp.map(scale_vec.after(smallvec![mvl_buf.clone(), mv_buf.clone()]), move |_, idx| {
            let a = load_at(&mvl_buf, &mvl_shape, idx);
            let b = load_at(&mv_buf, &mv_shape, idx);
            a.sub(&b)
        });
        scale_vec = scale_vec.exp2();

        o_reg = o_reg * &scale_vec;
        norm_vec = norm_vec * &scale_vec;

        let att = (att - &max_vec).exp2();

        norm_vec = warp.row_reduce(norm_vec.after(smallvec![scale_vec.uop().clone()]), &att, |a, b| a.add(b), 0.0);

        // A·V accumulation.
        let att_mma = warp.copy(att_mma.after(smallvec![kv_idx.clone(), norm_vec.uop().clone()]), &att);
        o_reg = warp.mma_atb(o_reg, &v_reg, &att_mma);
    }
    o_reg = o_reg.rewrap(ker.endrange(1));
    norm_vec = norm_vec.after(smallvec![o_reg.uop().clone()]);

    let o_reg = o_reg / &norm_vec;
    let o_reg_t = warp.transpose(o_reg_t, &o_reg);
    let o_idx = [Idx::from(&batch), Idx::from(&q_seq), Idx::from(&head), Idx::Const(0)];
    let _ = warp.store(o.into(), o_reg_t.into(), &o_idx, &[], 1);
}

/// Multi-wave flash-attention forward (FA-5 stage i): [`NUM_WARPS`] warps per
/// workgroup (512 threads) for the occupancy lift.
///
/// Each warp owns a **distinct** Q-tile `q_blk = block_q_base*NUM_WARPS + warpid`
/// (grid dim1 = `n/BLK/NUM_WARPS`, block 512). The 8 warps share one `(head,
/// batch)`, so K/V is loaded **once** into shared LDS collaboratively over all
/// 512 threads ([`Group::load`] GLOBAL→LOCAL with the 8-wave group), then **each
/// warp** reads K/V from LDS and runs its own QKᵀ → softmax → A·V on its own
/// Q-tile with per-warp [`Group::warp`] ops (the `ds_bpermute` reduce stays
/// inside each warp's 64 lanes). The KV loop bound is the **group max**
/// `(block_q_base+1)*NUM_WARPS` (all 8 warps trip the same count; each warp's
/// per-element diagonal mask drops its own out-of-range keys); since `n/BLK` is a
/// multiple of `NUM_WARPS` this is `≤ n/BLK` and needs no clamp.
///
/// A workgroup barrier after the per-warp LDS→REG reads guards the shared K/V LDS
/// from the next iteration's collaborative fill (the cross-wave WAR hazard — cf.
/// the matmul M1 sync, `test/unit/matmul.rs`).
pub(crate) fn build_fa_mw(ker: &Kernel, b: usize, n: usize, h: usize, h_kv: usize, d: usize) {
    assert_eq!(d % BLK, 0, "D must be a multiple of {BLK}");
    assert_eq!(n % BLK, 0, "N must be a multiple of {BLK}");
    assert_eq!(h % h_kv, 0, "H must be a multiple of H_KV");
    assert_eq!((n / BLK) % NUM_WARPS, 0, "multi-wave FA needs n/BLK a multiple of {NUM_WARPS} q-tiles");
    let group_size = (h / h_kv) as i64;
    let g = ker.group(NUM_WARPS); // 512 threads — collaborative K/V GLOBAL→LDS fill
    let warp = ker.warp(); // per-warp register/WMMA/softmax ops

    // out, then q, k, v (declaration order = ABI slot order).
    let o = ker.gl(&[b, n, h, d], DType::BFloat16);
    let q = ker.gl(&[b, n, h, d], DType::BFloat16);
    let k = ker.gl(&[b, n, h_kv, d], DType::BFloat16);
    let v = ker.gl(&[b, n, h_kv, d], DType::BFloat16);

    let head = ker.block_idx[0].clone();
    let head_kv = head.idiv(&iconst(group_size));
    let batch = ker.block_idx[2].clone();
    // This warp's Q-tile: `block_q_base * NUM_WARPS + warpid`.
    let block_q_base = ker.block_idx[1].clone();
    let warpid = g.warpid_in_group();
    let q_blk = block_q_base.mul(&iconst(NUM_WARPS as i64)).add(&warpid);

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // One shared K/V LDS slot for all 8 warps (filled once per KV block).
    let k_smem = ker.st((BLK, d), bf16.clone(), row, ST_16X16);
    let v_smem = ker.st((BLK, d), bf16.clone(), row, ST_16X16);

    // Per-warp register tiles (every warp runs the full pipeline on its own tile).
    let q_reg_fl = ker.rt((BLK, d), f32.clone(), row, RT_16X16);
    let q_reg = ker.rt((BLK, d), bf16.clone(), row, RT_16X16);
    let q_reg_t = ker.rt((d, BLK), bf16.clone(), col, RT_16X16);
    let k_reg = ker.rt((BLK, d), bf16.clone(), row, RT_16X16);
    let k_reg_t = ker.rt((d, BLK), bf16.clone(), col, RT_16X16);
    let v_reg = ker.rt((BLK, d), bf16.clone(), col, RT_16X16);
    let o_reg = ker.rt((d, BLK), f32.clone(), col, RT_16X16);
    let o_reg_t = ker.rt((BLK, d), f32.clone(), row, RT_16X16);
    let att = ker.rt((BLK, BLK), f32.clone(), col, RT_16X16);
    let att_mma = ker.rt((BLK, BLK), bf16, col, RT_16X16);
    let max_vec = ker.rv(BLK, f32.clone(), VecLayout::Ortho, RT_16X16);
    let norm_vec = ker.rv(BLK, f32.clone(), VecLayout::Ortho, RT_16X16);
    let max_vec_last = ker.rv(BLK, f32.clone(), VecLayout::Ortho, RT_16X16);
    let scale_vec = ker.rv(BLK, f32, VecLayout::Ortho, RT_16X16);

    let mut max_vec = warp.neg_inf_rv(max_vec);
    let mut norm_vec = warp.zero_rv(norm_vec);
    let mut o_reg = warp.zero(o_reg);
    let mut scale_vec = warp.ones_rv(scale_vec);
    let mut max_vec_last = max_vec_last;

    // Load + scale this warp's Q tile, then transpose it for the QKᵀ contraction.
    let q_idx = [Idx::from(&batch), Idx::from(&q_blk), Idx::from(&head), Idx::Const(0)];
    let q_reg_fl = warp.load(q_reg_fl.into(), q.into(), &[], &q_idx, 1).rt();
    let q_reg_fl = q_reg_fl * ((1.0 / (d as f64).sqrt()) * std::f64::consts::LOG2_E);
    let q_reg = warp.copy(q_reg, &q_reg_fl);
    let q_reg_t = warp.transpose(q_reg_t, &q_reg);

    // Group-level causal block-skip: the whole block trips `0..=max q_blk`, i.e.
    // `(block_q_base+1)*NUM_WARPS` blocks (≤ n/BLK; the per-warp mask drops each
    // warp's out-of-range keys, so warps with a smaller q_blk just compute
    // fully-masked tiles that contribute 0 to the running-max online softmax).
    let kv_bound = block_q_base.add(&iconst(1)).mul(&iconst(NUM_WARPS as i64));
    let kv_idx = ker.range_uop(kv_bound);
    {
        let kidx = [Idx::from(&batch), Idx::from(&kv_idx), Idx::from(&head_kv), Idx::Const(0)];
        // Collaborative GLOBAL→LDS fill across all 512 threads (each ends in a
        // workgroup barrier so every warp sees the full K/V block).
        let k_smem = g.load(k_smem.into(), k.into(), &[], &kidx, 1).st();
        let v_smem = g.load(v_smem.into(), v.into(), &[], &kidx, 1).st();
        // Per-warp LDS→REG gather: every warp reads the *same* shared K/V block.
        let k_reg = warp.load(k_reg.into(), k_smem.into(), &[], &[], 0).rt();
        let v_reg = warp.load(v_reg.into(), v_smem.into(), &[], &[], 0).rt();

        // Cross-wave WAR sync: all 8 warps must finish reading the shared K/V LDS
        // before the next iteration's collaborative fill overwrites it. Emitted in
        // program order inside the loop body, so the back-edge serializes it (cf.
        // matmul M1, `test/unit/matmul.rs`).
        let sync = k_reg.uop().barrier(smallvec![v_reg.uop().clone()]);
        let k_reg = k_reg.after(smallvec![sync.clone()]);
        let v_reg = v_reg.after(smallvec![sync]);

        // QKᵀ into a freshly-zeroed att tile.
        let att = warp.zero(att.after(smallvec![kv_idx.clone()]));
        let k_reg_t = warp.transpose(k_reg_t, &k_reg);
        let att = warp.mma_atb(att, &k_reg_t, &q_reg_t);

        // Causal mask: drop keys ahead of this warp's own query rows (→ -∞).
        let laneid = ker.laneid();
        let q_base = laneid.mod_(&iconst(16)).add(&q_blk.mul(&iconst(BLK as i64)));
        let kv_base = laneid.idiv(&iconst(16)).mul(&iconst(4)).add(&kv_idx.mul(&iconst(BLK as i64)));
        let att = warp.map(att, move |x, idx| {
            let kv_pos = kv_base.add(&idx[0].to_uop().mul(&iconst(16))).add(&idx[2].to_uop());
            let q_pos = q_base.add(&idx[1].to_uop().mul(&iconst(16)));
            let cond = kv_pos.gt(&q_pos);
            let neg_inf = UOp::const_(x.dtype(), ConstValue::Float(f64::NEG_INFINITY));
            UOp::try_where(cond, neg_inf, x.clone()).expect("causal where")
        });

        // Online softmax: update the running max, rescale the running stats by
        // exp2(prev_max - new_max), exponentiate, and accumulate the norm. The
        // running max carries across iterations, so a fully-masked tile (all -∞)
        // leaves the finite max untouched and contributes exp2(-∞)=0 — no NaN.
        max_vec_last = warp.copy(max_vec_last.after(smallvec![kv_idx.clone()]), &max_vec);
        max_vec = warp.row_reduce(
            max_vec.after(smallvec![max_vec_last.uop().clone()]),
            &att,
            |a, b| a.max(b),
            f64::NEG_INFINITY,
        );

        let (mvl_buf, mvl_shape) = (max_vec_last.uop().clone(), max_vec_last.shape().to_vec());
        let (mv_buf, mv_shape) = (max_vec.uop().clone(), max_vec.shape().to_vec());
        scale_vec = warp.map(scale_vec.after(smallvec![mvl_buf.clone(), mv_buf.clone()]), move |_, idx| {
            let a = load_at(&mvl_buf, &mvl_shape, idx);
            let b = load_at(&mv_buf, &mv_shape, idx);
            a.sub(&b)
        });
        scale_vec = scale_vec.exp2();

        o_reg = o_reg * &scale_vec;
        norm_vec = norm_vec * &scale_vec;

        let att = (att - &max_vec).exp2();

        norm_vec = warp.row_reduce(norm_vec.after(smallvec![scale_vec.uop().clone()]), &att, |a, b| a.add(b), 0.0);

        // A·V accumulation.
        let att_mma = warp.copy(att_mma.after(smallvec![kv_idx.clone(), norm_vec.uop().clone()]), &att);
        o_reg = warp.mma_atb(o_reg, &v_reg, &att_mma);
    }
    o_reg = o_reg.rewrap(ker.endrange(1));
    norm_vec = norm_vec.after(smallvec![o_reg.uop().clone()]);

    let o_reg = o_reg / &norm_vec;
    let o_reg_t = warp.transpose(o_reg_t, &o_reg);
    let o_idx = [Idx::from(&batch), Idx::from(&q_blk), Idx::from(&head), Idx::Const(0)];
    let _ = warp.store(o.into(), o_reg_t.into(), &o_idx, &[], 1);
}

// =============================================================================
// FA-5 stage (ii): double-buffered K/V (unroll-by-2, two static LDS buffers).
// =============================================================================

/// Online-softmax state carried *across* KV iterations. Read-modify-written in a
/// single linear chain, so the two unrolled slices safely share one set (each
/// slice's ops take the prior slice's rewrapped handle).
struct FaAcc<'k> {
    max_vec: RV<'k>,
    norm_vec: RV<'k>,
    o_reg: RT<'k>,
}

/// Per-slice scratch register tiles. The double-buffered builder allocates a
/// **distinct** set per unrolled half: the two slices live in one loop body with
/// no back-edge between them, so a shared scratch register would race (a slice-1
/// write could be scheduled before a slice-0 read — the priority toposort only
/// orders on data deps). Distinct tiles make the halves independent by
/// construction; only [`FaAcc`] is threaded.
struct FaScratch<'k> {
    k_reg: RT<'k>,
    k_reg_t: RT<'k>,
    v_reg: RT<'k>,
    att: RT<'k>,
    att_mma: RT<'k>,
    scale_vec: RV<'k>,
    max_vec_last: RV<'k>,
}

/// Fill one slice's K/V LDS buffers. `pipelined = false` (stage 1) uses the
/// barrier-coupled [`Group::load`] — a workgroup barrier after each of the K and
/// V fills — and keeps the per-slice WAR barrier in [`fa_kv_slice`]. `pipelined =
/// true` (stage 2) batches the K+V fills with [`Group::fill_local_nobar`] under a
/// SINGLE combined visibility barrier (consumed by this slice's gathers); the
/// alternating slices' barriers also separate each buffer's last read from its
/// next overwrite, so the WAR barrier drops too — **2** workgroup syncs per
/// double-iteration vs the naive **6**.
fn fill_kv_pair<'k>(
    g: &Group<'k>,
    k: &crate::tile::GL<'k>,
    v: &crate::tile::GL<'k>,
    k_smem: ST<'k>,
    v_smem: ST<'k>,
    kidx: &[Idx],
    pipelined: bool,
) -> (ST<'k>, ST<'k>) {
    if pipelined {
        let ksf = g.fill_local_nobar(k_smem, k.clone(), kidx, 1);
        let vsf = g.fill_local_nobar(v_smem, v.clone(), kidx, 1);
        let bar = ksf.uop().barrier(smallvec![vsf.uop().clone()]);
        let ksf = ksf.rewrap(ksf.uop().after(smallvec![bar.clone()]));
        let vsf = vsf.rewrap(vsf.uop().after(smallvec![bar]));
        (ksf, vsf)
    } else {
        let ksf = g.load(k_smem.into(), k.clone().into(), &[], kidx, 1).st();
        let vsf = g.load(v_smem.into(), v.clone().into(), &[], kidx, 1).st();
        (ksf, vsf)
    }
}

/// One KV-slice of the multi-wave online-softmax pipeline: gather this warp's
/// K/V fragments from the already-filled shared `(k_smem, v_smem)` LDS buffers,
/// compute `QKᵀ` → causal mask → online-softmax rescale → `A·V`, and thread the
/// updated [`FaAcc`] out. `slice_idx` is the KV block this half consumes (the
/// causal-mask base + the GLOBAL fill index); `reinit_dep` is the *loop* RANGE
/// uop, so the per-iteration re-inits (`att` zero, `max_vec_last`/`att_mma`
/// copies) re-run each trip instead of hoisting above the loop. `war_barrier`
/// gates the LDS→REG read behind a cross-wave WAR barrier (stage-1 naive); the
/// pipelined path drops it (the alternating fill barriers cover the WAR).
#[allow(clippy::too_many_arguments)]
fn fa_kv_slice<'k>(
    warp: &Group<'k>,
    ker: &Kernel,
    acc: FaAcc<'k>,
    sc: FaScratch<'k>,
    k_smem: ST<'k>,
    v_smem: ST<'k>,
    q_reg_t: &RT<'k>,
    q_blk: &Arc<UOp>,
    slice_idx: &Arc<UOp>,
    q_blk_rows: usize,
    kv_blk_rows: usize,
    reinit_dep: &Arc<UOp>,
    war_barrier: bool,
    extra_war: &[Arc<UOp>],
    causal: bool,
    valid_len: Option<Arc<UOp>>,
) -> FaAcc<'k> {
    let FaScratch { k_reg, k_reg_t, v_reg, att, att_mma, scale_vec, max_vec_last } = sc;
    let (att, v_reg) = fa_qk(
        warp,
        ker,
        k_reg,
        k_reg_t,
        v_reg,
        att,
        k_smem,
        v_smem,
        q_reg_t,
        q_blk,
        slice_idx,
        q_blk_rows,
        kv_blk_rows,
        reinit_dep,
        war_barrier,
        extra_war,
        causal,
        valid_len,
    );
    fa_softmax_pv(warp, acc, att_mma, scale_vec, max_vec_last, att, &v_reg, reinit_dep)
}

/// Stage 1 of a KV slice — `QKᵀ`: gather this warp's K/V fragments from the
/// already-filled shared `(k_smem, v_smem)` LDS, compute `QKᵀ` into a
/// freshly-zeroed `att`, and apply the causal mask. Returns the masked raw scores
/// `att` and the gathered `v_reg` (carried to [`fa_softmax_pv`]). Splitting QK off
/// the softmax/PV lets the cross-tile pipeline emit `qk(cur)` out of phase with
/// `softmax_pv(prev)`. `war_barrier`/`extra_war` gate the LDS→REG read behind a
/// cross-wave WAR barrier (with the double-buffer prefetch commits folded in).
#[allow(clippy::too_many_arguments)]
fn fa_qk<'k>(
    warp: &Group<'k>,
    ker: &Kernel,
    k_reg: RT<'k>,
    k_reg_t: RT<'k>,
    v_reg: RT<'k>,
    att: RT<'k>,
    k_smem: ST<'k>,
    v_smem: ST<'k>,
    q_reg_t: &RT<'k>,
    q_blk: &Arc<UOp>,
    slice_idx: &Arc<UOp>,
    q_blk_rows: usize,
    kv_blk_rows: usize,
    reinit_dep: &Arc<UOp>,
    war_barrier: bool,
    extra_war: &[Arc<UOp>],
    causal: bool,
    valid_len: Option<Arc<UOp>>,
) -> (RT<'k>, RT<'k>) {
    // Per-warp LDS→REG gather: every warp reads the shared K/V block.
    let k_reg = warp.load(k_reg.into(), k_smem.into(), &[], &[], 0).rt();
    let v_reg = warp.load(v_reg.into(), v_smem.into(), &[], &[], 0).rt();
    // Cross-wave WAR sync: all 8 warps must finish reading this buffer before the
    // next fill overwrites it. `extra_war` folds in the rolled double-buffer's
    // prefetch commits, so this single in-loop barrier (consumed by the gathers)
    // also gates the cross-iteration RAW/WAR.
    let (k_reg, v_reg) = if war_barrier {
        let mut deps: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![v_reg.uop().clone()];
        deps.extend(extra_war.iter().cloned());
        let sync = k_reg.uop().barrier(deps);
        (k_reg.after(smallvec![sync.clone()]), v_reg.after(smallvec![sync]))
    } else {
        (k_reg, v_reg)
    };

    // QKᵀ into a freshly-zeroed att tile.
    let att = warp.zero(att.after(smallvec![reinit_dep.clone()]));
    let k_reg_t = warp.transpose(k_reg_t, &k_reg);
    let att = warp.mma_atb(att, &k_reg_t, q_reg_t);

    // Score masking: drop keys ahead of this warp's own query rows (causal → -∞)
    // and/or keys at/after the per-batch valid length (padding mask → -∞). The
    // per-fragment offsets stay at the WMMA edge (`*16`/`/16`/`*4`); only the
    // tile *base* scales with the per-warp Q/KV tile heights (`q_blk_rows`,
    // `kv_blk_rows`), so the mask generalizes from 16×16 to T×T (and asymmetric)
    // with no change to the within-fragment derivation. With neither knob set,
    // the map is skipped entirely (full bidirectional attention).
    let att = if causal || valid_len.is_some() {
        let laneid = ker.laneid();
        let q_base = laneid.mod_(&iconst(16)).add(&q_blk.mul(&iconst(q_blk_rows as i64)));
        let kv_base = laneid.idiv(&iconst(16)).mul(&iconst(4)).add(&slice_idx.mul(&iconst(kv_blk_rows as i64)));
        warp.map(att, move |x, idx| {
            let kv_pos = kv_base.add(&idx[0].to_uop().mul(&iconst(16))).add(&idx[2].to_uop());
            let q_pos = q_base.add(&idx[1].to_uop().mul(&iconst(16)));
            let neg_inf = UOp::const_(x.dtype(), ConstValue::Float(f64::NEG_INFINITY));
            let mut out = x.clone();
            if causal {
                out = UOp::try_where(kv_pos.gt(&q_pos), neg_inf.clone(), out).expect("causal where");
            }
            if let Some(vl) = &valid_len {
                // Mask keys at/after the valid length: `kv_pos >= vl` (the index-typed
                // valid count comes pre-loaded from the per-batch `lens` buffer).
                out = UOp::try_where(kv_pos.ge(vl), neg_inf, out).expect("padding where");
            }
            out
        })
    } else {
        att
    };
    (att, v_reg)
}

/// Stage 2 of a KV slice — online softmax + `A·V`: given the masked raw scores
/// `att` (from [`fa_qk`]) and the gathered `v_reg`, update the running max,
/// rescale the running stats by `exp2(prev_max - new_max)`, exponentiate, fold the
/// norm, and accumulate `A·V` into `o_reg`. Threads the updated [`FaAcc`] out.
///
/// `att` is col-layout `(KV=height, Q=width)`; softmax reduces over KV and
/// broadcasts per Q, so the reduce folds the *height* (KV) via [`Group::col_reduce`]
/// → a per-*width* (Q) vector. At `{16,16}` this is bit-identical to `row_reduce`;
/// for multi-fragment tiles it is the only orientation that folds the right axis.
#[allow(clippy::too_many_arguments)]
fn fa_softmax_pv<'k>(
    warp: &Group<'k>,
    acc: FaAcc<'k>,
    att_mma: RT<'k>,
    scale_vec: RV<'k>,
    max_vec_last: RV<'k>,
    att: RT<'k>,
    v_reg: &RT<'k>,
    reinit_dep: &Arc<UOp>,
) -> FaAcc<'k> {
    let FaAcc { mut max_vec, mut norm_vec, mut o_reg } = acc;

    let max_vec_last = warp.copy(max_vec_last.after(smallvec![reinit_dep.clone()]), &max_vec);
    max_vec =
        warp.col_reduce(max_vec.after(smallvec![max_vec_last.uop().clone()]), &att, |a, b| a.max(b), f64::NEG_INFINITY);

    let (mvl_buf, mvl_shape) = (max_vec_last.uop().clone(), max_vec_last.shape().to_vec());
    let (mv_buf, mv_shape) = (max_vec.uop().clone(), max_vec.shape().to_vec());
    let mut scale_vec = warp.map(scale_vec.after(smallvec![mvl_buf.clone(), mv_buf.clone()]), move |_, idx| {
        let a = load_at(&mvl_buf, &mvl_shape, idx);
        let b = load_at(&mv_buf, &mv_shape, idx);
        a.sub(&b)
    });
    scale_vec = scale_vec.exp2();

    o_reg = o_reg * &scale_vec;
    norm_vec = norm_vec * &scale_vec;

    let att = (att - &max_vec).exp2();

    norm_vec = warp.col_reduce(norm_vec.after(smallvec![scale_vec.uop().clone()]), &att, |a, b| a.add(b), 0.0);

    // A·V accumulation.
    let att_mma = warp.copy(att_mma.after(smallvec![reinit_dep.clone(), norm_vec.uop().clone()]), &att);
    o_reg = warp.mma_atb(o_reg, v_reg, &att_mma);

    FaAcc { max_vec, norm_vec, o_reg }
}

/// Double-buffered multi-wave flash-attention forward (FA-5 stage ii + FA-4).
///
/// Same grid/semantics as [`build_fa_mw`], but the KV loop is **unrolled by 2**
/// over two static LDS buffers (`k_smem{0,1}` / `v_smem{0,1}`): trip `t` consumes
/// KV slices `2t` (buf0) and `2t+1` (buf1).
///
/// **FA-4 per-warp tiles.** `q_blk_rows`/`kv_blk_rows` (both multiples of [`BLK`])
/// set each warp's Q-tile and KV super-block heights, so the online-softmax
/// max/sum reduce amortizes over a larger fragment grid (`q_blk_rows*kv_blk_rows`
/// scores). `att` is `(KV=height, Q=width)` col-layout, so the softmax reduce folds
/// the KV (height) axis via [`Group::col_reduce`] producing a per-Q vector — at the
/// baseline single-fragment `{16,16}` this is bit-identical to `row_reduce`. The
/// group-max causal bound is `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK` super-blocks
/// (exact + even for `{16,16}`/`{32,32}`/`{32,64}`), so the unroll-by-2 half-count
/// `(block_q_base+1)*NUM_WARPS*Q_BLK/(2*KV_BLK)` skips none; the per-element causal
/// mask (absolute positions per fragment) covers sub-diagonal/diagonal for free,
/// including the asymmetric `Q_BLK<KV_BLK` case. **Bigger tiles raise VGPR pressure
/// (66→183→242 VGPRs for 16/32/64) and LDS (8→16→32 KB), dropping occupancy:** a
/// net win only when the machine is already block-saturated (B=8,H=16: `{32,32}`
/// ~1.2× at N=2048), a ~10–20% regression at low occupancy (B=1 inference). The
/// default entry point keeps `{16,16}`; larger tiles are opt-in.
///
/// Two static buffers sidestep the UOp model's inability to pick an LDS region by
/// runtime parity (`i % 2`): each slice is statically bound to its own buffer.
/// `pipelined = false` keeps a per-slice cross-wave WAR barrier (stage-1 naive,
/// the correctness reference for the unroll). `pipelined = true` reduces the
/// barriers — for the baseline `{16,16}` tile it also drops the WAR barrier (the
/// alternating fill barriers already separate a single-pass buffer's last-read
/// from its next overwrite); bigger tiles retain the WAR barrier (their longer
/// compute / multi-pass KV fill shifts the hazard window so the drop races — see
/// the inline note).
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_fa_mw_db(ker: &Kernel, b: usize, n: usize, h: usize, h_kv: usize, d: usize, cfg: FaConfig) {
    let FaConfig { q_blk: q_blk_rows, kv_blk: kv_blk_rows, pipelined, .. } = cfg;
    assert_eq!(d % BLK, 0, "D must be a multiple of {BLK}");
    assert_eq!(q_blk_rows % BLK, 0, "Q_BLK must be a multiple of the WMMA edge {BLK}");
    assert_eq!(kv_blk_rows % BLK, 0, "KV_BLK must be a multiple of the WMMA edge {BLK}");
    assert_eq!(h % h_kv, 0, "H must be a multiple of H_KV");
    // Grid: each block owns `NUM_WARPS` Q-tiles of `q_blk_rows` rows each.
    assert_eq!(
        n % (q_blk_rows * NUM_WARPS),
        0,
        "multi-wave FA needs N a multiple of Q_BLK*{NUM_WARPS} ({} here)",
        q_blk_rows * NUM_WARPS
    );
    // Unroll-by-2 over `kv_blk_rows`-row KV super-blocks. The group-max causal
    // bound is `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK` super-blocks (exact, and
    // always even for these tiles), so `2*KV_BLK | NUM_WARPS*Q_BLK`.
    assert_eq!(
        (NUM_WARPS * q_blk_rows) % (2 * kv_blk_rows),
        0,
        "FA-4 unroll-by-2 needs 2*KV_BLK to divide NUM_WARPS*Q_BLK ({}*{} / {})",
        NUM_WARPS,
        q_blk_rows,
        2 * kv_blk_rows
    );
    let group_size = (h / h_kv) as i64;
    let g = ker.group(NUM_WARPS); // 512 threads — collaborative K/V GLOBAL→LDS fill
    let warp = ker.warp(); // per-warp register/WMMA/softmax ops

    // out, then q, k, v (declaration order = ABI slot order).
    let o = ker.gl(&[b, n, h, d], DType::BFloat16);
    let q = ker.gl(&[b, n, h, d], DType::BFloat16);
    let k = ker.gl(&[b, n, h_kv, d], DType::BFloat16);
    let v = ker.gl(&[b, n, h_kv, d], DType::BFloat16);

    let head = ker.block_idx[0].clone();
    let head_kv = head.idiv(&iconst(group_size));
    let batch = ker.block_idx[2].clone();
    let block_q_base = ker.block_idx[1].clone();
    let warpid = g.warpid_in_group();
    let q_blk = block_q_base.mul(&iconst(NUM_WARPS as i64)).add(&warpid);

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // TWO shared K/V LDS buffers — buf0 holds even slices, buf1 holds odd ones.
    // Each buffer is one `kv_blk_rows × d` KV super-block.
    let k_smem0 = ker.st((kv_blk_rows, d), bf16.clone(), row, ST_16X16);
    let k_smem1 = ker.st((kv_blk_rows, d), bf16.clone(), row, ST_16X16);
    let v_smem0 = ker.st((kv_blk_rows, d), bf16.clone(), row, ST_16X16);
    let v_smem1 = ker.st((kv_blk_rows, d), bf16.clone(), row, ST_16X16);

    // Q tile + its transpose (shared, read-only across both slices).
    let q_reg_fl = ker.rt((q_blk_rows, d), f32.clone(), row, RT_16X16);
    let q_reg = ker.rt((q_blk_rows, d), bf16.clone(), row, RT_16X16);
    let q_reg_t = ker.rt((d, q_blk_rows), bf16.clone(), col, RT_16X16);
    let o_reg_t = ker.rt((q_blk_rows, d), f32.clone(), row, RT_16X16);

    // Distinct per-slice scratch (no inter-slice register WAR — see `FaScratch`).
    // `att`/`o` tiles are `(KV=height, Q=width)` / `(d, Q)`; the RVs index per Q.
    let mk_scratch = || FaScratch {
        k_reg: ker.rt((kv_blk_rows, d), bf16.clone(), row, RT_16X16),
        k_reg_t: ker.rt((d, kv_blk_rows), bf16.clone(), col, RT_16X16),
        v_reg: ker.rt((kv_blk_rows, d), bf16.clone(), col, RT_16X16),
        att: ker.rt((kv_blk_rows, q_blk_rows), f32.clone(), col, RT_16X16),
        att_mma: ker.rt((kv_blk_rows, q_blk_rows), bf16.clone(), col, RT_16X16),
        scale_vec: ker.rv(q_blk_rows, f32.clone(), VecLayout::Ortho, RT_16X16),
        max_vec_last: ker.rv(q_blk_rows, f32.clone(), VecLayout::Ortho, RT_16X16),
    };
    let sc0 = mk_scratch();
    let sc1 = mk_scratch();

    // Carried online-softmax accumulators.
    let o_reg = ker.rt((d, q_blk_rows), f32.clone(), col, RT_16X16);
    let max_vec = ker.rv(q_blk_rows, f32.clone(), VecLayout::Ortho, RT_16X16);
    let norm_vec = ker.rv(q_blk_rows, f32, VecLayout::Ortho, RT_16X16);
    let acc = FaAcc { max_vec: warp.neg_inf_rv(max_vec), norm_vec: warp.zero_rv(norm_vec), o_reg: warp.zero(o_reg) };

    // Load + scale this warp's Q tile, then transpose for the QKᵀ contraction.
    let q_idx = [Idx::from(&batch), Idx::from(&q_blk), Idx::from(&head), Idx::Const(0)];
    let q_reg_fl = warp.load(q_reg_fl.into(), q.into(), &[], &q_idx, 1).rt();
    let q_reg_fl = q_reg_fl * ((1.0 / (d as f64).sqrt()) * std::f64::consts::LOG2_E);
    let q_reg = warp.copy(q_reg, &q_reg_fl);
    let q_reg_t = warp.transpose(q_reg_t, &q_reg);

    // Unrolled-by-2 KV loop: half the super-block trips, two slices per body. The
    // group-max bound `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK` is exact + even,
    // so the half-count `(block_q_base+1)*NUM_WARPS*Q_BLK/(2*KV_BLK)` skips none.
    let half_mult = (NUM_WARPS * q_blk_rows / (2 * kv_blk_rows)) as i64;
    let kv_half = block_q_base.add(&iconst(1)).mul(&iconst(half_mult));
    let t = ker.range_uop(kv_half);
    let slice0 = t.mul(&iconst(2));
    let slice1 = slice0.add(&iconst(1));

    // Cross-wave WAR barrier. Dropping it (relying on the alternating fill
    // barriers to cover a buffer's last-read/next-fill gap) is only validated for
    // the baseline single-fragment tile `{16,16}`. Bigger register tiles lengthen
    // the per-slice compute and the collaborative fill (KV_BLK=64 even spills to a
    // 2-pass fill), which shifts the WAR hazard window so the dropped barrier
    // races the next-trip overwrite against an in-flight read — GPU-confirmed
    // corruption at depth (flaky on `{32,32}` @ N=1024, deterministic on `{32,64}`
    // @ N=2048). So only `{16,16}` keeps the barrier-reduced drop; every larger
    // tile retains the explicit WAR barrier (still 1 fewer barrier/slice than the
    // naive path's separate K and V fills).
    let baseline_tile = q_blk_rows == BLK && kv_blk_rows == BLK;
    let war = !pipelined || !baseline_tile;

    // Slice 0 → buf0.
    let kidx0 = [Idx::from(&batch), Idx::from(&slice0), Idx::from(&head_kv), Idx::Const(0)];
    let (k_smem0, v_smem0) = fill_kv_pair(&g, &k, &v, k_smem0, v_smem0, &kidx0, pipelined);
    let acc = fa_kv_slice(
        &warp,
        ker,
        acc,
        sc0,
        k_smem0,
        v_smem0,
        &q_reg_t,
        &q_blk,
        &slice0,
        q_blk_rows,
        kv_blk_rows,
        &t,
        war,
        &[],
        true,
        None,
    );

    // Slice 1 → buf1.
    let kidx1 = [Idx::from(&batch), Idx::from(&slice1), Idx::from(&head_kv), Idx::Const(0)];
    let (k_smem1, v_smem1) = fill_kv_pair(&g, &k, &v, k_smem1, v_smem1, &kidx1, pipelined);
    let acc = fa_kv_slice(
        &warp,
        ker,
        acc,
        sc1,
        k_smem1,
        v_smem1,
        &q_reg_t,
        &q_blk,
        &slice1,
        q_blk_rows,
        kv_blk_rows,
        &t,
        war,
        &[],
        true,
        None,
    );

    let FaAcc { norm_vec, o_reg, .. } = acc;
    let o_reg = o_reg.rewrap(ker.endrange(1));
    let norm_vec = norm_vec.after(smallvec![o_reg.uop().clone()]);

    let o_reg = o_reg / &norm_vec;
    let o_reg_t = warp.transpose(o_reg_t, &o_reg);
    let o_idx = [Idx::from(&batch), Idx::from(&q_blk), Idx::from(&head), Idx::Const(0)];
    let _ = warp.store(o.into(), o_reg_t.into(), &o_idx, &[], 1);
}

// =============================================================================
// Software-pipelined double-buffered KV loop.
// =============================================================================

/// Software-pipelined double-buffered multi-wave flash-attention. Same
/// grid/semantics as [`build_fa_mw_db`], but the KV loop is a rolled `Range` over a
/// **2×-size LDS** K/V double buffer indexed by `kv_idx % 2`: each iteration
/// register-stages the next KV block's GLOBAL→VGPR load, gathers the current buffer
/// half into the WMMA fragments, runs the online-softmax body, then `ds_write`-
/// commits the staged registers into the other half, under one workgroup barrier
/// per iteration.
///
/// Unlike the unroll-by-2 [`build_fa_mw_db`] (two static buffers, two slices and
/// two [`FaScratch`] sets per body), this keeps one scratch set and one loop body.
/// LDS is the same (one `st_db` = two halves); FA's K/V are small so 2× fits the
/// 64 KB budget. The online-softmax [`FaAcc`] carries across the back-edge via the
/// memory-accumulator (`kv_idx` re-init) pattern, as in [`build_fa_mw`]. The
/// `kv_idx % 2` parity makes the gather/commit counter-dependent so they stay
/// loop-scoped; the per-iteration WAR barrier (consumed by the gathers, with the
/// prefetch commits folded into its deps) provides the cross-iteration RAW/WAR
/// ordering, closed with plain [`Kernel::endrange`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_fa_mw_rdb(
    ker: &Kernel,
    b: usize,
    n: usize,
    h: usize,
    h_kv: usize,
    d: usize,
    cfg: FaConfig,
    in_dtype: DType,
    masked: bool,
) {
    let FaConfig { q_blk: q_blk_rows, kv_blk: kv_blk_rows, unroll, causal, .. } = cfg;
    // Flat compute (unrolled QKᵀ/softmax/A·V) is the prerequisite for the Stage-2
    // attention scheduling comb; the rolled (`unroll = false`) form is the iglp
    // baseline. Same numerics either way (the unroll only changes the loop
    // mechanism, not the fold order).
    ker.set_unroll(unroll);
    assert_eq!(d % BLK, 0, "D must be a multiple of {BLK}");
    assert_eq!(q_blk_rows % BLK, 0, "Q_BLK must be a multiple of the WMMA edge {BLK}");
    assert_eq!(kv_blk_rows % BLK, 0, "KV_BLK must be a multiple of the WMMA edge {BLK}");
    assert_eq!(h % h_kv, 0, "H must be a multiple of H_KV");
    assert_eq!(
        n % (q_blk_rows * NUM_WARPS),
        0,
        "multi-wave FA needs N a multiple of Q_BLK*{NUM_WARPS} ({} here)",
        q_blk_rows * NUM_WARPS
    );
    // Rolled (no unroll halving): the group-max causal bound is
    // `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK` super-blocks (exact for these tiles).
    assert_eq!(
        (NUM_WARPS * q_blk_rows) % kv_blk_rows,
        0,
        "FA rolled db needs KV_BLK to divide NUM_WARPS*Q_BLK ({NUM_WARPS}*{q_blk_rows} / {kv_blk_rows})"
    );
    let group_size = (h / h_kv) as i64;
    let g = ker.group(NUM_WARPS); // 512 threads — collaborative K/V GLOBAL→LDS fill
    let warp = ker.warp();

    let o = ker.gl(&[b, n, h, d], in_dtype.clone());
    let q = ker.gl(&[b, n, h, d], in_dtype.clone());
    let k = ker.gl(&[b, n, h_kv, d], in_dtype.clone());
    let v = ker.gl(&[b, n, h_kv, d], in_dtype.clone());
    // Per-batch valid key-length buffer (padding mask), declared AFTER o,q,k,v so the
    // ABI slot order stays stable; only bound when `masked`. The scalar `lens[batch]`
    // is read and cast to `Index` so the score-mask compare (`kv_pos >= vl`) matches
    // the causal path's Index-typed position arithmetic exactly.
    let valid_len = masked.then(|| {
        let lens = ker.gl(&[b], DType::Int32);
        load_at(lens.uop(), lens.shape(), &[Idx::from(&ker.block_idx[2])]).cast(DType::Index)
    });

    let head = ker.block_idx[0].clone();
    let head_kv = head.idiv(&iconst(group_size));
    let batch = ker.block_idx[2].clone();
    let block_q_base = ker.block_idx[1].clone();
    let warpid = g.warpid_in_group();
    let q_blk = block_q_base.mul(&iconst(NUM_WARPS as i64)).add(&warpid);

    let in_dt = in_dtype.clone();
    let f32 = DType::Float32;
    let (row, col) = (TileLayout::Row, TileLayout::Col);

    // 2×-size shared K/V LDS double buffers (one `kv_blk_rows × d` block per half).
    let k_smem = ker.st_db((kv_blk_rows, d), in_dt.clone(), row, ST_16X16);
    let v_smem = ker.st_db((kv_blk_rows, d), in_dt.clone(), row, ST_16X16);
    let half_k = k_smem.half_elems() as i64;
    let half_v = v_smem.half_elems() as i64;

    // Q tile + transpose (shared, read-only across the loop).
    let q_reg_fl = ker.rt((q_blk_rows, d), f32.clone(), row, RT_16X16);
    let q_reg = ker.rt((q_blk_rows, d), in_dt.clone(), row, RT_16X16);
    let q_reg_t = ker.rt((d, q_blk_rows), in_dt.clone(), col, RT_16X16);
    let o_reg_t = ker.rt((q_blk_rows, d), f32.clone(), row, RT_16X16);

    // One scratch set (vs the unroll's two): the rolled body has a back-edge, so
    // the carried FaAcc + a single scratch suffice.
    let sc = FaScratch {
        k_reg: ker.rt((kv_blk_rows, d), in_dt.clone(), row, RT_16X16),
        k_reg_t: ker.rt((d, kv_blk_rows), in_dt.clone(), col, RT_16X16),
        v_reg: ker.rt((kv_blk_rows, d), in_dt.clone(), col, RT_16X16),
        att: ker.rt((kv_blk_rows, q_blk_rows), f32.clone(), col, RT_16X16),
        att_mma: ker.rt((kv_blk_rows, q_blk_rows), in_dt.clone(), col, RT_16X16),
        scale_vec: ker.rv(q_blk_rows, f32.clone(), VecLayout::Ortho, RT_16X16),
        max_vec_last: ker.rv(q_blk_rows, f32.clone(), VecLayout::Ortho, RT_16X16),
    };

    // Carried online-softmax accumulators.
    let o_reg = ker.rt((d, q_blk_rows), f32.clone(), col, RT_16X16);
    let max_vec = ker.rv(q_blk_rows, f32.clone(), VecLayout::Ortho, RT_16X16);
    let norm_vec = ker.rv(q_blk_rows, f32, VecLayout::Ortho, RT_16X16);
    let acc = FaAcc { max_vec: warp.neg_inf_rv(max_vec), norm_vec: warp.zero_rv(norm_vec), o_reg: warp.zero(o_reg) };

    // Load + scale this warp's Q tile, then transpose for the QKᵀ contraction.
    let q_idx = [Idx::from(&batch), Idx::from(&q_blk), Idx::from(&head), Idx::Const(0)];
    let q_reg_fl = warp.load(q_reg_fl.into(), q.into(), &[], &q_idx, 1).rt();
    let q_reg_fl = q_reg_fl * ((1.0 / (d as f64).sqrt()) * std::f64::consts::LOG2_E);
    let q_reg = warp.copy(q_reg, &q_reg_fl);
    let q_reg_t = warp.transpose(q_reg_t, &q_reg);

    // Total KV super-blocks (the full bidirectional sweep). With `causal`, the
    // per-q-block bound is the causal block-skip `(block_q_base+1)*NUM_WARPS*Q_BLK/KV_BLK`
    // super-blocks; without it every q-block attends to all `total_kv_blocks`.
    let total_kv_blocks = (n / kv_blk_rows) as i64;
    let kv_bound = if causal {
        let blocks_mult = (NUM_WARPS * q_blk_rows / kv_blk_rows) as i64;
        block_q_base.add(&iconst(1)).mul(&iconst(blocks_mult))
    } else {
        iconst(total_kv_blocks)
    };

    // Prologue: stage KV block 0 → VGPR, commit → buf[0], barrier.
    let p_kidx = [Idx::from(&batch), Idx::Const(0), Idx::from(&head_kv), Idx::Const(0)];
    let s0_k = g.stage_global_to_reg(&k_smem, &k, &p_kidx, 1);
    let s0_v = g.stage_global_to_reg(&v_smem, &v, &p_kidx, 1);
    let k_smem = g.commit_reg_to_local(k_smem, &s0_k, true);
    let v_smem = g.commit_reg_to_local(v_smem, &s0_v, true);

    // Rolled KV loop. `kv_bound` (the dynamic per-q-block causal trip count) is the
    // Range end. The prefetch-block index is `(kv+1) % total_kv_blocks` (a Mod): the
    // final trip's prefetch (`kv+1 == total`) wraps to block 0, which is never
    // gathered, keeping the GLOBAL read in bounds. A `min`/`where` clamp is avoided
    // — a `WHERE` in the prefetch-address path is mis-ordered past its address-MUL
    // consumer in this kernel's linearization, leaving the renderer without its SSA
    // value; Mod (like the parity) lowers and orders cleanly.
    let kv_idx = ker.range_uop(kv_bound);
    let kvp1 = kv_idx.add(&iconst(1));
    let pf = kvp1.try_mod(&iconst(total_kv_blocks)).expect("(kv+1) % total blocks");
    let par_cur = kv_idx.try_mod(&iconst(2)).expect("kv % 2");
    let par_nxt = kvp1.try_mod(&iconst(2)).expect("(kv+1) % 2");

    let k_cur = k_smem.with_base_offset(par_cur.mul(&iconst(half_k)));
    let v_cur = v_smem.with_base_offset(par_cur.mul(&iconst(half_v)));
    let k_nxt = k_smem.with_base_offset(par_nxt.mul(&iconst(half_k)));
    let v_nxt = v_smem.with_base_offset(par_nxt.mul(&iconst(half_v)));

    // Mark the KV-loop as an attention compute pipeline (MFMA + online softmax),
    // threaded through the in-loop K/V buffers so the marker precedes the first
    // prefetch load and stays loop-scoped (dep = `kv_idx`). The prologue keeps the
    // un-rewrapped `k`/`v`. The post-linearization scheduling pass brackets the MFMAs
    // and (Stage 2) weaves the softmax under them (supersedes the prior `iglp_opt(0)`).
    let pf_kidx = [Idx::from(&batch), Idx::from(&pf), Idx::from(&head_kv), Idx::Const(0)];
    let mark = crate::sched::pipeline(crate::sched::SchedKind::Attention, kv_idx.clone());
    let k_l = k.rewrap(k.uop().after(smallvec![mark.clone()]));
    let v_l = v.rewrap(v.uop().after(smallvec![mark]));
    let s_k = g.stage_global_to_reg(&k_smem, &k_l, &pf_kidx, 1);
    let s_v = g.stage_global_to_reg(&v_smem, &v_l, &pf_kidx, 1);

    // Commit the staged registers into the *other* half (no per-commit barrier — the
    // single in-loop WAR barrier below covers both RAW and WAR). Emitted before the
    // slice so the slice's `o_reg` A·V store stays the last terminal store on the stack.
    let commit_k = g.commit_reg_to_local(k_nxt, &s_k, false);
    let commit_v = g.commit_reg_to_local(v_nxt, &s_v, false);

    // Gather buf[cur] (counter-dependent ⇒ loop-scoped; reads the block committed
    // last iteration, or the prologue for block 0) and run QKᵀ → causal mask →
    // online softmax → A·V. The WAR barrier (consumed by the gathers, an in-loop
    // anchor) folds in the prefetch commits via `extra_war`, so one barrier gates
    // the cross-iteration RAW/WAR. The barrier-wrapped END (`endrange_barrier_to`)
    // is NOT used here: it reorders the causal-mask WHERE past its consumer, leaving
    // the renderer without its SSA value — plain `endrange` keeps the render order.
    let extra_war = [commit_k.uop().clone(), commit_v.uop().clone()];
    let FaAcc { norm_vec, o_reg, .. } = fa_kv_slice(
        &warp,
        ker,
        acc,
        sc,
        k_cur,
        v_cur,
        &q_reg_t,
        &q_blk,
        &kv_idx,
        q_blk_rows,
        kv_blk_rows,
        &kv_idx,
        true,
        &extra_war,
        causal,
        valid_len,
    );

    let o_reg = o_reg.rewrap(ker.endrange(1));
    let norm_vec = norm_vec.after(smallvec![o_reg.uop().clone()]);

    let o_reg = o_reg / &norm_vec;
    let o_reg_t = warp.transpose(o_reg_t, &o_reg);
    let o_idx = [Idx::from(&batch), Idx::from(&q_blk), Idx::from(&head), Idx::Const(0)];
    let _ = warp.store(o.into(), o_reg_t.into(), &o_idx, &[], 1);
}

/// Run flash-attention forward into `o` against realized `q`/`k`/`v`.
///
/// Shapes: `q`,`o` = `[B, N, H, D]`, `k`,`v` = `[B, N, H_KV, D]`, with `H` a
/// multiple of `H_KV` (grouped-query attention) and `N`/`D` multiples of 16.
/// Causal forward attention; writes `o` in place.
pub fn flash_attention_forward(o: &mut Tensor, q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<()> {
    fa_check_target(q)?;
    let qs = q.shape().expect("q shape");
    let ks = k.shape().expect("k shape");
    let dim = |s: &svod_ir::shape::Shape, i: usize| s[i].as_const().expect("concrete dim");
    let (b, n, h, d) = (dim(&qs, 0), dim(&qs, 1), dim(&qs, 2), dim(&qs, 3));
    let h_kv = dim(&ks, 2);
    let grid = [h as i64, (n / BLK) as i64, b as i64];

    crate::run_kernel("fa", grid, 64, &mut [o], &[q, k, v], |ker| {
        build_fa(ker, b, n, h, h_kv, d);
        ker.finish(1)
    })
}

/// Run **multi-wave** ([`NUM_WARPS`]-warp) flash-attention forward into `o`.
///
/// Same shapes/semantics as [`flash_attention_forward`], but requires `n / BLK`
/// to be a multiple of [`NUM_WARPS`] (the 512-thread block owns `NUM_WARPS`
/// Q-tiles). Grid dim1 is `n / BLK / NUM_WARPS`, block `NUM_WARPS * 64`.
pub fn flash_attention_forward_mw(o: &mut Tensor, q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<()> {
    fa_check_target(q)?;
    let qs = q.shape().expect("q shape");
    let ks = k.shape().expect("k shape");
    let dim = |s: &svod_ir::shape::Shape, i: usize| s[i].as_const().expect("concrete dim");
    let (b, n, h, d) = (dim(&qs, 0), dim(&qs, 1), dim(&qs, 2), dim(&qs, 3));
    let h_kv = dim(&ks, 2);
    let grid = [h as i64, (n / BLK / NUM_WARPS) as i64, b as i64];

    crate::run_kernel("fa_mw", grid, (NUM_WARPS * 64) as i64, &mut [o], &[q, k, v], |ker| {
        build_fa_mw(ker, b, n, h, h_kv, d);
        ker.finish(1)
    })
}

/// Run **double-buffered** multi-wave flash-attention forward into `o` (FA-5
/// stage ii). Same grid/semantics as [`flash_attention_forward_mw`]; `pipelined`
/// selects the unroll-only stage-1 path (`false`) vs the barrier-reduced
/// pipelined path (`true`). See [`build_fa_mw_db`].
pub fn flash_attention_forward_mw_db(
    o: &mut Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    pipelined: bool,
) -> crate::LaunchResult<()> {
    fa_check_target(q)?;
    let qs = q.shape().expect("q shape");
    let ks = k.shape().expect("k shape");
    let dim = |s: &svod_ir::shape::Shape, i: usize| s[i].as_const().expect("concrete dim");
    let (b, n, h, d) = (dim(&qs, 0), dim(&qs, 1), dim(&qs, 2), dim(&qs, 3));
    let h_kv = dim(&ks, 2);
    let grid = [h as i64, (n / Q_BLK / NUM_WARPS) as i64, b as i64];

    crate::run_kernel("fa_mw_db", grid, (NUM_WARPS * 64) as i64, &mut [o], &[q, k, v], |ker| {
        build_fa_mw_db(
            ker,
            b,
            n,
            h,
            h_kv,
            d,
            FaConfig { q_blk: Q_BLK, kv_blk: KV_BLK, pipelined, ..Default::default() },
        );
        ker.finish(1)
    })
}

/// Per-warp tile for [`build_fa_mw_rdb`]: the bigger `{32,32}` (which amortizes the
/// softmax over more MFMA) once its grid `b·h·n/(32·NUM_WARPS)` covers the ~304-CU
/// machine and `N` divides `32·NUM_WARPS`; otherwise the baseline `{16,16}` (the
/// bigger tile halves the grid, so it loses at low occupancy). The 304 crossover is
/// a first cut from the gfx942 bench.
fn adaptive_fa_tile(b: usize, n: usize, h: usize) -> (usize, usize) {
    const NUM_CU: usize = 304;
    const BIG: usize = 32;
    if n.is_multiple_of(BIG * NUM_WARPS) && b * h * (n / (BIG * NUM_WARPS)) >= NUM_CU {
        (BIG, BIG)
    } else {
        (Q_BLK, KV_BLK)
    }
}

/// Run the rolled double-buffered multi-wave flash-attention forward into `o`
/// ([`build_fa_mw_rdb`]). One rolled KV loop over a parity-indexed 2× LDS double
/// buffer (one [`FaScratch`]); the per-warp tile is [`adaptive_fa_tile`].
pub fn flash_attention_forward_mw_rdb(o: &mut Tensor, q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<()> {
    fa_check_target(q)?;
    let qs = q.shape().expect("q shape");
    let ks = k.shape().expect("k shape");
    let dim = |s: &svod_ir::shape::Shape, i: usize| s[i].as_const().expect("concrete dim");
    let (b, n, h, d) = (dim(&qs, 0), dim(&qs, 1), dim(&qs, 2), dim(&qs, 3));
    let h_kv = dim(&ks, 2);
    let (q_blk, kv_blk) = adaptive_fa_tile(b, n, h);
    let grid = [h as i64, (n / q_blk / NUM_WARPS) as i64, b as i64];

    let in_dtype = q.uop().dtype();
    crate::run_kernel("fa_mw_rdb", grid, (NUM_WARPS * 64) as i64, &mut [o], &[q, k, v], |ker| {
        build_fa_mw_rdb(
            ker,
            b,
            n,
            h,
            h_kv,
            d,
            FaConfig { q_blk, kv_blk, ..Default::default() },
            in_dtype.clone(),
            false,
        );
        ker.finish(1)
    })
}

/// Options for the unified [`flash_attention_with`] entry point.
///
/// `causal` selects the triangular (causal block-skip) sweep vs the full
/// bidirectional sweep. `key_lens` is an optional realized `[B]`-shaped `i32`
/// tensor of valid **key** counts per batch — a *key-only* padding mask: keys at
/// `kv_pos >= key_lens[batch]` are masked out of every query row. Queries beyond
/// the valid length are still computed (the kernel does not mask query rows); the
/// caller is expected to discard those padded output rows. The scheduler fallback
/// mirrors this exactly with a `[B,1,1,N]` key mask, so the hand kernel and the
/// fallback agree on every row (valid and padded alike).
#[derive(Clone, Copy)]
pub struct FaOpts<'a> {
    /// Causal (triangular) attention when `true`; full bidirectional when `false`.
    pub causal: bool,
    /// Optional `[B]` `i32` per-batch valid-key-count padding mask (key-only).
    pub key_lens: Option<&'a Tensor>,
}

impl Default for FaOpts<'_> {
    fn default() -> Self {
        Self { causal: true, key_lens: None }
    }
}

/// **Unified** graph-native flash-attention forward with a scheduler fallback.
///
/// Q is `[B,N,H,D]`, K/V are `[B,N,H_KV,D]`. When the device + shapes are eligible
/// for the hand-built gfx942 kernel — target in [`FA_SUPPORTED_ARCHS`], dtype ∈
/// {bf16, f16}, `D % 16 == 0`, `N % (q_blk·NUM_WARPS) == 0`, and `H % H_KV == 0`
/// (GQA) — this builds the rolled double-buffered kernel ([`build_fa_mw_rdb`]) via
/// [`crate::graph_launch`], honoring `opts.causal` and the optional `opts.key_lens`
/// **key-only** padding mask (a 5th `[B]` `i32` global bound after `o,q,k,v`).
///
/// Otherwise (wrong arch / unsupported dtype / non-tiling shape — including
/// gfx1151 and CPU) it FALLS BACK to [`Tensor::scaled_dot_product_attention`] so
/// the call is correct on every backend: permute `[B,N,H,D] → [B,H,N,D]`, cast to
/// f32, run SDPA with `is_causal = opts.causal` plus (when `key_lens` is set) a
/// `[B,1,1,N]` boolean key mask (`true = masked` where `arange(N) >= key_lens`,
/// matching the kernel's `kv_pos >= lens[batch]`), then permute back and cast to
/// the original dtype. Both paths use key-only masking, so they AGREE on all rows.
pub fn flash_attention_with(q: &Tensor, k: &Tensor, v: &Tensor, opts: FaOpts) -> crate::LaunchResult<Tensor> {
    let qs = q.shape().expect("q shape");
    let ks = k.shape().expect("k shape");
    let dim = |s: &svod_ir::shape::Shape, i: usize| s[i].as_const().expect("concrete dim");
    let (b, n, h, d) = (dim(&qs, 0), dim(&qs, 1), dim(&qs, 2), dim(&qs, 3));
    let h_kv = dim(&ks, 2);
    let (q_blk, _kv_blk) = adaptive_fa_tile(b, n, h);
    let dtype = q.uop().dtype();

    let arch_ok = crate::target::check_target(&q.device(), FA_SUPPORTED_ARCHS).is_ok();
    let dtype_ok = dtype == DType::BFloat16 || dtype == DType::Float16;
    let shape_ok = d % BLK == 0 && n % (q_blk * NUM_WARPS) == 0 && h % h_kv == 0;

    if arch_ok && dtype_ok && shape_ok {
        let (q_blk, kv_blk) = adaptive_fa_tile(b, n, h);
        let grid = [h as i64, (n / q_blk / NUM_WARPS) as i64, b as i64];
        let out = Tensor::empty(&[b, n, h, d], dtype.clone());
        let masked = opts.key_lens.is_some();
        let causal = opts.causal;
        let build_dtype = dtype.clone();
        // ABI/global order is o, q, k, v, (lens) — `out` is global[0], `ins` map to
        // global[1..] in order, so `key_lens` (the 5th global) goes last.
        let mut ins: Vec<&Tensor> = vec![q, k, v];
        if let Some(lens) = opts.key_lens {
            ins.push(lens);
        }
        return crate::graph_launch("flash_attention", grid, (NUM_WARPS * 64) as i64, out, &ins, move |ker| {
            build_fa_mw_rdb(
                ker,
                b,
                n,
                h,
                h_kv,
                d,
                FaConfig { q_blk, kv_blk, causal, ..Default::default() },
                build_dtype.clone(),
                masked,
            );
            ker.finish(1)
        });
    }

    // Fallback: scheduler SDPA in the inputs' NATIVE dtype (f16/bf16), mirroring the
    // kernel's key-only masking — matching what the model's native attention runs, so
    // the QKᵀ/PV use the fast WMMA path (a prior f32 cast forced slow `v_mfma_f32` +
    // extra copies). Each fallible step `?`s through `.context(FallbackSnafu)` (which
    // boxes the large `svod_tensor` error) so no closure ever carries the un-boxed error.
    let fb = crate::launch::FallbackSnafu;
    let perm = |t: &Tensor| -> crate::LaunchResult<Tensor> { t.try_permute(&[0, 2, 1, 3]).context(fb) };
    let (qp, kp, vp) = (perm(q)?, perm(k)?, perm(v)?);
    // [B,1,1,N] bool key mask: true (masked) where arange(N) >= key_lens[batch],
    // matching the kernel's `kv_pos >= lens[batch]`.
    let mask = match opts.key_lens {
        Some(lens) => {
            let range = Tensor::arange(n as i64, None, None).context(fb)?;
            let range = range.try_reshape([1usize, 1, 1, n]).context(fb)?;
            let lens = lens.cast(DType::Int32).context(fb)?;
            let lens = lens.try_reshape([b, 1, 1, 1]).context(fb)?;
            Some(range.try_ge(&lens).context(fb)?)
        }
        None => None,
    };
    let out_bhnd = qp
        .scaled_dot_product_attention()
        .key(&kp)
        .value(&vp)
        .is_causal(opts.causal)
        .maybe_attn_mask(mask.as_ref())
        .call()
        .context(fb)?;
    out_bhnd.try_permute(&[0, 2, 1, 3]).context(fb)?.cast(dtype).context(fb)
}

/// **Graph-native** causal flash-attention forward: returns a lazy output
/// [`Tensor`] (a `custom_kernel` / `Op::Call` graph node) instead of writing in
/// place, so it composes into a model's UOp graph and benchmarks through the
/// normal `prepare()` → `execute_profiled` path like any other op. Same kernel as
/// [`flash_attention_forward_mw_rdb`] (rolled double-buffer, [`adaptive_fa_tile`]),
/// just launched via [`crate::graph_launch`] rather than direct dispatch.
///
/// Thin wrapper over [`flash_attention_with`] with [`FaOpts::default`] (causal,
/// unmasked) — unchanged behavior. On a non-gfx942 / unsupported-shape device this
/// transparently falls back to the scheduler SDPA path.
pub fn flash_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> crate::LaunchResult<Tensor> {
    flash_attention_with(q, k, v, FaOpts::default())
}
