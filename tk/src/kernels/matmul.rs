//! The bf16→f32 tile matmul builders: the parametrized multi-wave kernel (M1 +
//! M7 size-adaptive) and the M2 K-loop register-prefetch pipeline. A port of
//! tinygrad `test_tk.py::test_simple_matmul` lifted to a reusable kernel builder.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;
use svod_tensor::Tensor;

use crate::index::{Idx, cidx};
use crate::tiles::{RT_16X16, RT_16X16_W32_ACC, RT_16X16_W32_IN, ST_16X16_SWIZZLED, ST_16X16_SWIZZLED_W32, TileLayout};
use crate::{Kernel, RT, RegTile};

/// K-reduction step (the LDS strip depth, shared by every config). HK `GEMM:6`.
pub const K_STEP: usize = 64;

/// K-reduction step for [`build_matmul_db`]. Halved vs [`K_STEP`] so the 2×-size
/// LDS double buffer fits the 64 KB budget at the 256×256/8-wave geometry
/// (A `256×32`×2 + B `32×256`×2 = 64 KB).
pub const K_STEP_DB: usize = 32;

/// Block / wave geometry of a multi-wave matmul (HK `GEMM:5-8,67-68`): a
/// `wave_rows × wave_cols`-wave workgroup computes a `block × block` C tile,
/// each wave owning `n_accum` col-major `reg × reg` f32 accumulators
/// (`reg = block / wave_cols`) reduced over K in [`K_STEP`]-wide steps. The
/// `wave_cols = wave_rows * n_accum` invariant keeps `reg` square: the M side is
/// split into `wave_rows * n_accum` row-blocks, the N side into `wave_cols`.
#[derive(Clone, Copy)]
pub struct MatmulCfg {
    pub block: usize,
    pub wave_rows: usize,
    pub wave_cols: usize,
    pub n_accum: usize,
    /// M4: drive `(pid_m, pid_n)` from a flattened 1-D grid via the chiplet/L2
    /// [`l2_swizzle`](crate::grid::l2_swizzle) instead of the plain 2-D
    /// `block_idx`. Grid becomes `[grid² , 1, 1]`.
    pub l2_swizzle: bool,
    /// M3: fill the GLOBAL→LDS strips with 128-bit (`vec8` bf16) coalesced loads
    /// instead of the scalar/`vec4`-folded path.
    pub vec_load: bool,
}

impl MatmulCfg {
    /// The per-accumulator square edge (`block / wave_cols`).
    pub const fn reg(&self) -> usize {
        self.block / self.wave_cols
    }
    /// `reg`-blocks per C-tile side (= `wave_cols` = `wave_rows * n_accum`); the
    /// grid→C-block coordinate multiplier.
    pub const fn blocks_per_side(&self) -> usize {
        self.block / self.reg()
    }
    /// Launch block size (threads) = `wave_rows * wave_cols * wave_size`.
    pub const fn threads(&self, wave_size: usize) -> i64 {
        (self.wave_rows * self.wave_cols * wave_size) as i64
    }
    /// Grid edge (`n / block`).
    pub const fn grid(&self, n: usize) -> i64 {
        (n / self.block) as i64
    }
    /// Launch grid dims: a flattened 1-D `[grid², 1, 1]` when M4 ([`l2_swizzle`])
    /// is on (the chiplet swizzle re-derives `(pid_m, pid_n)`), else the plain
    /// 2-D `[grid, grid, 1]`.
    pub const fn grid_dims(&self, n: usize) -> [i64; 3] {
        let g = self.grid(n);
        if self.l2_swizzle { [g * g, 1, 1] } else { [g, g, 1] }
    }
}

/// M1+M3+M4: 8-wave (2×4) 256×256 block, two 64×64 accumulators/wave, 512
/// threads, the chiplet/L2 grid swizzle, and 128-bit vectorized LDS fills.
pub const M1_CFG: MatmulCfg =
    MatmulCfg { block: 256, wave_rows: 2, wave_cols: 4, n_accum: 2, l2_swizzle: true, vec_load: true };
/// M7 small-N: single-warp 64×64 block, one 64×64 accumulator, 64 threads — the
/// grid is `(n/64)²` workgroups, ~16× M1's at a given N, so a small N keeps the
/// 304-CU machine fed instead of collapsing to a handful of 256×256 blocks.
/// Keeps the plain 2-D grid + scalar fill (the swizzle/vec wins are large-N).
pub const SMALL_CFG: MatmulCfg =
    MatmulCfg { block: 64, wave_rows: 1, wave_cols: 1, n_accum: 1, l2_swizzle: false, vec_load: false };

/// M7 size-adaptive config selection: small N (where the 256×256/8-wave grid
/// starves the machine) uses [`SMALL_CFG`]; everything else keeps [`M1_CFG`].
/// The 768 threshold is the empirical crossover from the GPU-time bench.
pub fn cfg_for_n(n: usize) -> MatmulCfg {
    if n <= 768 && n.is_multiple_of(SMALL_CFG.block) { SMALL_CFG } else { M1_CFG }
}

/// The M-row C-block coordinate of accumulator `a` (`warp_row + a*wave_rows`,
/// in `reg`-block units) — HK `GEMM:92-94` wave sub-tile row selection.
fn acc_row(warp_row: &Arc<UOp>, a: usize, cfg: &MatmulCfg) -> Arc<UOp> {
    if a == 0 { warp_row.clone() } else { warp_row.add(&cidx((a * cfg.wave_rows) as i64)) }
}

/// The `(pid_m, pid_n)` C-block coordinate (in `block` units) for this workgroup
/// — M4's chiplet/L2 [`l2_swizzle`](crate::grid::l2_swizzle) off a flattened 1-D
/// grid (`block_idx[0]`) when enabled, else the plain 2-D `block_idx`.
fn block_coords(ker: &Kernel, n: usize, cfg: &MatmulCfg) -> (Arc<UOp>, Arc<UOp>) {
    if cfg.l2_swizzle {
        let g = cfg.grid(n);
        crate::grid::l2_swizzle(ker.block_idx[0].clone(), g * g, g, g)
    } else {
        (ker.block_idx[1].clone(), ker.block_idx[0].clone())
    }
}

/// Build the `simple_matmul` SINK for an `n×n` bf16→f32 matmul with [`M1_CFG`]
/// (the default large-N path: 256×256 block, M4 chiplet grid, M3 vec fills).
pub fn build_matmul(ker: &Kernel, n: usize) {
    build_matmul_cfg(ker, n, M1_CFG);
}

/// M7: build the matmul with the size-adaptive config ([`cfg_for_n`]). The
/// caller must launch with the matching grid/threads ([`MatmulCfg::grid`] /
/// [`MatmulCfg::threads`]) and `finish(cfg.n_accum)`.
pub fn build_matmul_adaptive(ker: &Kernel, n: usize) {
    build_matmul_cfg(ker, n, cfg_for_n(n));
}

/// The GPU arch(es) the tile matmul is built for: gfx942 (CDNA MFMA, wave64) and
/// gfx1151 (RDNA3.5 WMMA, wave32 — the `_W32_*` fragment shapes). The launcher
/// gates against this; see [`crate::target::check_target`]. (gfx1151 is pending
/// hardware validation — a wrong fragment map shows as a permuted output.)
pub const MATMUL_SUPPORTED_ARCHS: &[svod_dtype::AmdArch] = &[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151];

/// **Graph-native** `n×n` bf16→f32 tile matmul: returns a lazy output [`Tensor`]
/// (a `custom_kernel` / `Op::Call` node) — the matmul peer of
/// [`crate::flash_attention`]. Composes into a model graph and benchmarks through
/// the normal `prepare()` → `execute_profiled` path. `a`/`b` are square `[n, n]`
/// bf16; uses the size-adaptive config ([`cfg_for_n`]).
pub fn matmul(a: &svod_tensor::Tensor, b: &svod_tensor::Tensor) -> crate::LaunchResult<Tensor> {
    // Single resolve: gate to a supported arch (+ toolchain) and reuse the arch to
    // build caps so the launch block (`waves * wave_size`) and the kernel's wave
    // math track the real wave width.
    let caps = crate::ArchCaps::for_arch(crate::target::resolve_supported_arch(&a.device(), MATMUL_SUPPORTED_ARCHS)?);
    let dim = |t: &Tensor, i: usize| t.shape().expect("shape")[i].as_const().expect("concrete dim");
    let (am, an) = (dim(a, 0), dim(a, 1));
    let (bm, bn) = (dim(b, 0), dim(b, 1));
    assert_eq!(
        (am, an, bm, bn),
        (am, am, am, am),
        "tk matmul requires square, equal-size a/b ([n,n]); got a={am}x{an} b={bm}x{bn}"
    );
    let n = am;

    let cfg = cfg_for_n(n);
    let out = Tensor::empty(&[n, n], DType::Float32);
    crate::graph_launch("matmul", cfg.grid_dims(n), cfg.threads(caps.wave_size), out, &[a, b], caps, move |ker| {
        build_matmul_cfg(ker, n, cfg);
        ker.finish(cfg.n_accum)
    })
}

/// The parametrized multi-wave matmul (M1 + M7). One `cfg.block × cfg.block` C
/// tile per workgroup, `cfg.n_accum` col-major `reg × reg` accumulators/wave
/// reduced over a tracked K-loop; each wave streams its A-strip rows and shared
/// B-strip cols out of XOR-swizzled LDS. A single `END` closes the K-loop around
/// the last accumulator's store; the rest stay scoped inside it by chaining
/// their A-inputs through the prior accumulator's MFMA (a `RANGE` admits one
/// `END`). The epilogue stores each accumulator to global C at its `reg`-block.
pub fn build_matmul_cfg(ker: &Kernel, n: usize, cfg: MatmulCfg) {
    assert_eq!(n % cfg.block, 0, "matmul N={n} must be a multiple of the {} block", cfg.block);
    assert_eq!(cfg.wave_cols, cfg.wave_rows * cfg.n_accum, "config invariant wave_cols == wave_rows*n_accum");
    let reg = cfg.reg();
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols);

    // WMMA fragment layout is arch-specific: gfx942 (CDNA MFMA, wave64) vs gfx11
    // (RDNA WMMA, wave32). The A/B input *orientation* is the SAME on both — lane =
    // M (A) / N (B), element = K (tinygrad `a_elem(x,k,row)=x[k][row]`) — only the
    // packing differs: CDNA spreads K across lane-groups (`ept=4, stride=4`); RDNA
    // holds all 16 K per lane, replicated across wave-halves (`ept=16, stride=0`).
    // The accumulator differs more (even/odd row interleave — see `RT_16X16_W32_ACC`).
    let (st_sw, rt_acc, rt_in) = if ker.caps.arch.is_cdna() {
        (ST_16X16_SWIZZLED, RT_16X16, RT_16X16)
    } else {
        (ST_16X16_SWIZZLED_W32, RT_16X16_W32_ACC, RT_16X16_W32_IN)
    };

    // GL params bind in declaration order: out (c, f32), then ins (a, b — bf16).
    let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
    let a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);

    // A strip [block×K_STEP] = [M,K]; B strip [K_STEP×block] = [K,N]; both
    // XOR-swizzled, single-buffered.
    let a_smem = ker.st((cfg.block, K_STEP), DType::BFloat16, TileLayout::Row, st_sw);
    let b_smem = ker.st((K_STEP, cfg.block), DType::BFloat16, TileLayout::Row, st_sw);

    let (row, col) = block_coords(ker, n, &cfg); // (pid_m, pid_n) in block units
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();

    // `n_accum` col-major reg×reg f32 accumulators per wave.
    let accs: Vec<RT> =
        (0..cfg.n_accum).map(|_| g.zero(ker.rt((reg, reg), DType::Float32, TileLayout::Col, rt_acc))).collect();

    let tile = ker.range((n / K_STEP) as i64);

    // Collaborative GLOBAL→LDS fill over all threads (each ends in a barrier);
    // M3 uses 128-bit vectorized loads for the large-N strips.
    let a_idx = [Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::from(&tile)];
    let b_idx = [Idx::Const(0), Idx::Const(0), Idx::from(&tile), Idx::from(&col)];
    let (a_smem, b_smem) = if cfg.vec_load {
        (g.fill_local_vec(a_smem, a_gl, &a_idx, 2), g.fill_local_vec(b_smem, b_gl, &b_idx, 2))
    } else {
        (
            g.load(a_smem.into(), a_gl.into(), &[], &a_idx, 2).st(),
            g.load(b_smem.into(), b_gl.into(), &[], &b_idx, 2).st(),
        )
    };

    // Shared B sub-tile (N col-block {warp_col}, same for every accumulator) and
    // per-accumulator A sub-tiles (M row-block {warp_row + a*wave_rows}).
    let bb = g
        .load(
            ker.rt((K_STEP, reg), DType::BFloat16, TileLayout::Col, rt_in).into(),
            b_smem.into(),
            &[],
            &[Idx::Const(0), Idx::from(&warp_col)],
            0,
        )
        .rt();
    let a_subs: Vec<RT> = (0..cfg.n_accum)
        .map(|a| {
            g.load(
                ker.rt((reg, K_STEP), DType::BFloat16, TileLayout::Row, rt_in).into(),
                a_smem.clone().into(),
                &[],
                &[Idx::Uop(acc_row(&warp_row, a, &cfg))],
                0,
            )
            .rt()
        })
        .collect();

    // Cross-wave WAR barrier: every wave must finish reading LDS before the next
    // K iteration's collaborative fill overwrites it.
    let mut bar_deps: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![bb.uop().clone()];
    bar_deps.extend(a_subs.iter().skip(1).map(|t| t.uop().clone()));
    let sync = a_subs[0].uop().barrier(bar_deps);
    let bb = bb.after(smallvec![sync.clone()]);
    let a_subs: Vec<RT> = a_subs.into_iter().map(|t| t.after(smallvec![sync.clone()])).collect();

    // MFMA-accumulate each accumulator over the K sub-steps; chain accumulator
    // `a`'s A-input through accumulator `a-1`'s MFMA so a single `END` scopes
    // them all inside the K-loop.
    let mut prev_out: Option<Arc<UOp>> = None;
    for (a, a_sub) in a_subs.iter().enumerate() {
        let a_sub = match &prev_out {
            Some(p) => a_sub.after(smallvec![p.clone()]),
            None => a_sub.clone(),
        };
        prev_out = Some(g.mma_ab(accs[a].clone(), &a_sub, &bb).uop().clone());
    }
    let ended = ker.endrange_to(1);
    // Each accumulator reads its fully-reduced register value *outside* the loop.
    let final_accs: Vec<RT> = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();

    // Epilogue: store each col-major accumulator to global C at its reg-block
    // coords {row*bps + warp_row + a*wave_rows, col*bps + warp_col} (GEMM:222-223).
    let bps = cfg.blocks_per_side() as i64;
    let nidx = col.mul(&cidx(bps)).add(&warp_col);
    let mut c_t = c_gl;
    for (a, c) in final_accs.into_iter().enumerate() {
        let m = row.mul(&cidx(bps)).add(&acc_row(&warp_row, a, &cfg));
        c_t =
            g.store(c_t.into(), c.into(), &[Idx::Const(0), Idx::Const(0), Idx::Uop(m), Idx::from(&nidx)], &[], 2).gl();
    }
}

// =============================================================================
// M2 — K-loop register-prefetch pipeline (single-LDS, manually unrolled).
// =============================================================================

/// Which M2 pipeline stage to emit (staged bring-up, each gated on gfx942
/// correctness before the next). `Unroll` is the numerically-transparent
/// baseline; `Prefetch` stages the next tile's GLOBAL→VGPR load under the
/// current tile's MFMAs (committed to LDS at the cluster tail); `Hints` adds the
/// `s_setprio`/`s_waitcnt`/`sched_barrier` scheduling fences.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PipeStage {
    Unroll,
    Prefetch,
    Hints,
}

impl PipeStage {
    fn prefetch(self) -> bool {
        matches!(self, PipeStage::Prefetch | PipeStage::Hints)
    }
    fn hints(self) -> bool {
        matches!(self, PipeStage::Hints)
    }
    /// `SVOD_TK_PIPELINE` = 1→Unroll, 2→Prefetch, 3(or other)→Hints.
    pub fn from_env() -> Self {
        match std::env::var("SVOD_TK_PIPELINE").ok().as_deref() {
            Some("1") => PipeStage::Unroll,
            Some("2") => PipeStage::Prefetch,
            _ => PipeStage::Hints,
        }
    }
}

/// M2: the register-prefetch K-loop pipeline. Same [`M1_CFG`] geometry as
/// [`build_matmul`], but the K-loop is **manually unrolled** in Rust (HK
/// `#pragma unroll`, `GEMM:84`) over a **single** XOR-swizzled LDS buffer, and
/// the next tile's GLOBAL→VGPR load is staged under the current tile's MFMAs
/// (`PipeStage::Prefetch`), then `ds_write`-committed into the same LDS at the
/// cluster tail. Accumulators chain across tiles by register reassignment (each
/// `mma_ab` is self-contained — no `RANGE`/`END`), so the result is numerically
/// identical to M1. Launch with [`M1_CFG`] grid/threads and `finish(n_accum)`.
///
/// **M2 bring-up verdict (opt-in only — does not beat M1).** The `Unroll` stage
/// is numerically transparent at every N but the fully-unrolled IR regresses on
/// GPU time at N ≥ 1024 (looser register allocation than M1's tight K-loop) and
/// its compile time grows steeply (~30 s at N=2048 / 32 tiles). The register
/// staging (`Prefetch`/`Hints`) is correct only up to ~8 unrolled tiles (≤ 512):
/// beyond that the single-LDS commit hazards corrupt/wedge. A *looped* single-
/// LDS prefetch — the small-IR fix — is not expressible here: carrying the
/// committed LDS across an `Op::Range` back-edge needs a loop-carried phi the
/// UOp model lacks (the plan's KEY DESIGN NOTE). So M2 ships as validated
/// infrastructure ([`crate::asm`] + the staging primitives), gated behind
/// `SVOD_TK_PIPELINE`, with M1 as the perf baseline. The size-adaptive **M7**
/// ([`build_matmul_adaptive`]) is the matmul lever that *did* land.
pub fn build_matmul_pipelined(ker: &Kernel, n: usize, stage: PipeStage) {
    let cfg = M1_CFG;
    assert_eq!(n % cfg.block, 0, "matmul N={n} must be a multiple of the {} block", cfg.block);
    let reg = cfg.reg();
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols);

    let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
    let a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let (row, col) = block_coords(ker, n, &cfg);
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();

    let mut accs: Vec<RT> =
        (0..cfg.n_accum).map(|_| g.zero(ker.rt((reg, reg), DType::Float32, TileLayout::Col, RT_16X16))).collect();

    // A single XOR-swizzled LDS pair, threaded across the unrolled tiles (the
    // WAR back-edge is the explicit barrier, since there is no loop RANGE).
    let mut a_smem = ker.st((cfg.block, K_STEP), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);
    let mut b_smem = ker.st((K_STEP, cfg.block), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);

    let num_tiles = n / K_STEP;
    // Prefetch staging buffers carried one tile ahead (filled at tile-1's tail,
    // committed at tile's head). `None` until the first tile primes them.
    let mut staged: Option<(Arc<UOp>, Arc<UOp>)> = None;

    for tile in 0..num_tiles {
        // ── Commit / fill the current tile's LDS ──────────────────────────────
        let (af, bf) = if stage.prefetch() {
            match staged.take() {
                // Tiles ≥1: ds_write-commit the registers staged during tile-1.
                Some((sa, sb)) => {
                    let af = g.commit_reg_to_local(a_smem.clone(), &sa, true);
                    let bf = g.commit_reg_to_local(b_smem.clone(), &sb, true);
                    (af, bf)
                }
                // Tile 0 prologue: no prefetch yet — fill straight from global.
                None => (
                    g.load(a_smem.clone().into(), a_gl.clone().into(), &[], &tile_a_idx(tile, &row), 2).st(),
                    g.load(b_smem.clone().into(), b_gl.clone().into(), &[], &tile_b_idx(tile, &col), 2).st(),
                ),
            }
        } else {
            (
                g.load(a_smem.clone().into(), a_gl.clone().into(), &[], &tile_a_idx(tile, &row), 2).st(),
                g.load(b_smem.clone().into(), b_gl.clone().into(), &[], &tile_b_idx(tile, &col), 2).st(),
            )
        };

        // ── Stage the next tile's GLOBAL→VGPR load (overlaps this tile's MFMAs) ─
        if stage.prefetch() && tile + 1 < num_tiles {
            let sa = g.stage_global_to_reg(&a_smem, &a_gl, &tile_a_idx(tile + 1, &row), 2);
            let sb = g.stage_global_to_reg(&b_smem, &b_gl, &tile_b_idx(tile + 1, &col), 2);
            staged = Some((sa, sb));
        }

        // ── LDS→REG gather ────────────────────────────────────────────────────
        let bb = g
            .load(
                ker.rt((K_STEP, reg), DType::BFloat16, TileLayout::Col, RT_16X16).into(),
                bf.clone().into(),
                &[],
                &[Idx::Const(0), Idx::from(&warp_col)],
                0,
            )
            .rt();
        let a_subs: Vec<RT> = (0..cfg.n_accum)
            .map(|a| {
                g.load(
                    ker.rt((reg, K_STEP), DType::BFloat16, TileLayout::Row, RT_16X16).into(),
                    af.clone().into(),
                    &[],
                    &[Idx::Uop(acc_row(&warp_row, a, &cfg))],
                    0,
                )
                .rt()
            })
            .collect();

        // Cross-wave WAR barrier (its src is a value node, as M1 — the asm
        // scheduling nodes must NOT depend on a Barrier: the AMD renderer emits
        // the fence/`s.barrier` but registers no value for it, so a `ctx.get`
        // on a barrier-dep is undefined).
        let mut reads: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![bb.uop().clone()];
        reads.extend(a_subs.iter().map(|t| t.uop().clone()));
        let sync = a_subs[0].uop().barrier(reads);
        let bb = bb.after(smallvec![sync.clone()]);
        let a_subs: Vec<RT> = a_subs.into_iter().map(|t| t.after(smallvec![sync.clone()])).collect();

        // ── MFMA burst: `s_waitcnt lgkmcnt(0)` to drain the deferred LDS reads,
        // then `s_setprio(1)` around the burst — chained off a post-sync value
        // node (not the barrier) so the asm nodes order between the barrier and
        // the MFMAs without taking the Barrier as a dep.
        let prio_in: Vec<RT> = if stage.hints() {
            let wait = crate::asm::s_waitcnt_lgkmcnt(0, a_subs[0].uop().clone());
            let hi = crate::asm::s_setprio(1, wait);
            a_subs.iter().map(|t| t.after(smallvec![hi.clone()])).collect()
        } else {
            a_subs
        };
        let mut last_out: Option<Arc<UOp>> = None;
        for (a, a_sub) in prio_in.iter().enumerate() {
            accs[a] = g.mma_ab(accs[a].clone(), a_sub, &bb);
            last_out = Some(accs[a].uop().clone());
        }
        // Lower priority + pin the cluster with a scheduling barrier (so the
        // staged loads / commits cannot float across the MFMA region). Both hang
        // off the last MFMA output (a value node).
        let cluster_tail = if stage.hints() {
            let lo = crate::asm::s_setprio(0, last_out.clone().unwrap());
            crate::asm::sched_barrier(0, lo)
        } else {
            sync.clone()
        };

        // Thread the LDS handles to the next tile, ordered after this tile's
        // reads (WAR) and the cluster tail (so the next commit lands after).
        a_smem = af.rewrap(af.uop().after(smallvec![cluster_tail.clone()]));
        b_smem = bf.rewrap(bf.uop().after(smallvec![cluster_tail]));
    }

    // Epilogue: store each accumulator to global C at its reg-block.
    let bps = cfg.blocks_per_side() as i64;
    let nidx = col.mul(&cidx(bps)).add(&warp_col);
    let mut c_t = c_gl;
    for (a, c) in accs.into_iter().enumerate() {
        let m = row.mul(&cidx(bps)).add(&acc_row(&warp_row, a, &cfg));
        c_t =
            g.store(c_t.into(), c.into(), &[Idx::Const(0), Idx::Const(0), Idx::Uop(m), Idx::from(&nidx)], &[], 2).gl();
    }
}

/// A-strip tile index `[0, 0, row-block, K-tile]` for the unrolled K-loop.
fn tile_a_idx(tile: usize, row: &Arc<UOp>) -> [Idx; 4] {
    [Idx::Const(0), Idx::Const(0), Idx::from(row), Idx::Const(tile as i64)]
}
/// B-strip tile index `[0, 0, K-tile, col-block]` for the unrolled K-loop.
fn tile_b_idx(tile: usize, col: &Arc<UOp>) -> [Idx; 4] {
    [Idx::Const(0), Idx::Const(0), Idx::Const(tile as i64), Idx::from(col)]
}

// =============================================================================
// Software-pipelined double-buffered K-loop.
// =============================================================================

/// Software-pipelined double-buffered matmul ([`M1_CFG`] geometry). The K-loop is
/// a rolled `Range` over a **2×-size XOR-swizzled LDS** double buffer indexed by
/// `tile % 2` ([`K_STEP_DB`] keeps the doubled buffer within the 64 KB LDS
/// budget). Each iteration register-stages the next K-tile's GLOBAL→VGPR load,
/// gathers the current buffer half into the WMMA fragments, MFMA-accumulates, then
/// `ds_write`-commits the staged registers into the other half, under one
/// workgroup barrier per iteration.
///
/// The `tile % 2` parity both selects the buffer half and makes the gather/commit
/// addresses counter-dependent, which keeps them inside the K-loop (a single-LDS
/// variant would let LICM hoist the gather out and read tile 0 every iteration).
/// The per-iteration barrier has no in-iteration value consumer; it is kept live
/// by wrapping the loop-closing MFMA store ([`Kernel::endrange_barrier_to`]) so it
/// becomes the closing `END`'s computation — emitted after the MFMAs, which read
/// the raw gathers and so do not wait on it. Cross-iteration RAW (commit `i`
/// visible to gather `i+1`) and WAR (gather `i` done before commit `i+1`
/// overwrites the half) ordering comes from that barrier plus rolled-loop program
/// order. `iglp_opt(0)` at the K-loop top hands the AMDGPU machine scheduler the
/// GEMM (MFMA/memory interleave) pipeline.
///
/// Launch with [`M1_CFG`] grid/threads and `finish(n_accum)`.
pub fn build_matmul_db(ker: &Kernel, n: usize) {
    let cfg = M1_CFG;
    assert_eq!(n % cfg.block, 0, "matmul N={n} must be a multiple of the {} block", cfg.block);
    assert_eq!(n % K_STEP_DB, 0, "matmul N={n} must be a multiple of K_STEP_DB={K_STEP_DB}");
    let reg = cfg.reg();
    let g = ker.group_2d(cfg.wave_rows, cfg.wave_cols);

    let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
    let a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
    let (row, col) = block_coords(ker, n, &cfg);
    let warp_row = g.warp_row();
    let warp_col = g.warp_col();

    let accs: Vec<RT> =
        (0..cfg.n_accum).map(|_| g.zero(ker.rt((reg, reg), DType::Float32, TileLayout::Col, RT_16X16))).collect();

    // 2×-size XOR-swizzled LDS strips; the loop selects a half by `parity*half`.
    let a_smem = ker.st_db((cfg.block, K_STEP_DB), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);
    let b_smem = ker.st_db((K_STEP_DB, cfg.block), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);
    let half_a = a_smem.half_elems() as i64;
    let half_b = b_smem.half_elems() as i64;
    let num_tiles = n / K_STEP_DB;

    // ── Prologue: stage global tile 0 → VGPR, commit → buf[0], barrier ────────
    let p_a_idx = [Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::Const(0)];
    let p_b_idx = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::from(&col)];
    let s0_a = g.stage_global_to_reg(&a_smem, &a_gl, &p_a_idx, 2);
    let s0_b = g.stage_global_to_reg(&b_smem, &b_gl, &p_b_idx, 2);
    let a_smem = g.commit_reg_to_local(a_smem, &s0_a, true); // buf[0], with barrier
    let b_smem = g.commit_reg_to_local(b_smem, &s0_b, true);

    // ── Rolled, software-pipelined K-loop over the double buffer ──────────────
    let tile = ker.range(num_tiles as i64);
    let tp1 = tile.add(&cidx(1));
    // Clamp the prefetch tile to the last valid one (the final iteration's
    // prefetch is never consumed; clamping keeps the GLOBAL read in bounds).
    let pf = UOp::try_where(
        tp1.try_cmplt(&cidx(num_tiles as i64)).expect("tile+1 < num_tiles"),
        tp1.clone(),
        cidx(num_tiles as i64 - 1),
    )
    .expect("clamp prefetch tile");
    let par_cur = tile.try_mod(&cidx(2)).expect("tile % 2");
    let par_nxt = tp1.try_mod(&cidx(2)).expect("(tile+1) % 2");

    let a_cur = a_smem.with_base_offset(par_cur.mul(&cidx(half_a)));
    let b_cur = b_smem.with_base_offset(par_cur.mul(&cidx(half_b)));
    let a_nxt = a_smem.with_base_offset(par_nxt.mul(&cidx(half_a)));
    let b_nxt = b_smem.with_base_offset(par_nxt.mul(&cidx(half_b)));

    // Mark the K-loop as a GEMM compute pipeline, threaded through the GLOBAL buffers
    // the in-loop stage reads so the marker precedes the first prefetch load and stays
    // loop-scoped (dep = the counter `tile`). The prologue keeps the un-rewrapped
    // `a_gl`/`b_gl`. The post-linearization scheduling pass brackets each MFMA with
    // `s_setprio` and a `sched.barrier` fence (supersedes the prior `iglp_opt(0)`).
    let mark = crate::sched::pipeline(crate::sched::SchedKind::Gemm, tile.clone());
    let a_gl_l = a_gl.rewrap(a_gl.uop().after(smallvec![mark.clone()]));
    let b_gl_l = b_gl.rewrap(b_gl.uop().after(smallvec![mark]));

    // Stage the next tile's GLOBAL→VGPR load (overlaps this tile's MFMAs).
    let pf_a_idx = [Idx::Const(0), Idx::Const(0), Idx::from(&row), Idx::from(&pf)];
    let pf_b_idx = [Idx::Const(0), Idx::Const(0), Idx::from(&pf), Idx::from(&col)];
    let s_a = g.stage_global_to_reg(&a_smem, &a_gl_l, &pf_a_idx, 2);
    let s_b = g.stage_global_to_reg(&b_smem, &b_gl_l, &pf_b_idx, 2);

    // Gather the current buffer half → WMMA fragments. The counter-dependent
    // address keeps these loop-scoped; they read the tile committed last
    // iteration (or the prologue for tile 0), made visible by the prior barrier.
    let bb = g
        .load(
            ker.rt((K_STEP_DB, reg), DType::BFloat16, TileLayout::Col, RT_16X16).into(),
            b_cur.into(),
            &[],
            &[Idx::Const(0), Idx::from(&warp_col)],
            0,
        )
        .rt();
    let a_subs: Vec<RT> = (0..cfg.n_accum)
        .map(|a| {
            g.load(
                ker.rt((reg, K_STEP_DB), DType::BFloat16, TileLayout::Row, RT_16X16).into(),
                a_cur.clone().into(),
                &[],
                &[Idx::Uop(acc_row(&warp_row, a, &cfg))],
                0,
            )
            .rt()
        })
        .collect();

    // Commit the staged registers into the *other* half (no per-commit barrier —
    // the single loop-tail barrier below covers both the RAW and the WAR edge).
    let commit_a = g.commit_reg_to_local(a_nxt, &s_a, false);
    let commit_b = g.commit_reg_to_local(b_nxt, &s_b, false);

    // MFMA-accumulate on the gathers — the MFMA does not depend on the tail
    // barrier, so it stays independent of the prefetch's commit/global-load and the
    // two can overlap. Chain accumulator `a`'s A-input through `a-1`'s MFMA so a
    // single `END` scopes them all in the K-loop (as [`build_matmul_cfg`]).
    let mut prev_out: Option<Arc<UOp>> = None;
    for (a, a_sub) in a_subs.iter().enumerate() {
        let a_sub = match &prev_out {
            Some(p) => a_sub.after(smallvec![p.clone()]),
            None => a_sub.clone(),
        };
        prev_out = Some(g.mma_ab(accs[a].clone(), &a_sub, &bb).uop().clone());
    }

    // Close the K-loop by wrapping the last accumulator's MFMA store in a workgroup
    // barrier (the cross-iteration RAW/WAR fence): the barrier becomes the
    // loop-closing `END`'s computation — kept live + tile-scoped with no value
    // consumer, and (its passthrough being the MFMA store) emitted after the MFMAs.
    // Its deps are the prefetch commits.
    let bar_deps: smallvec::SmallVec<[Arc<UOp>; 4]> = smallvec![commit_a.uop().clone(), commit_b.uop().clone()];
    let ended = ker.endrange_barrier_to(1, bar_deps);
    let final_accs: Vec<RT> = accs.iter().map(|c| c.after(smallvec![ended.clone()])).collect();

    // Epilogue: store each accumulator to global C at its reg-block.
    let bps = cfg.blocks_per_side() as i64;
    let nidx = col.mul(&cidx(bps)).add(&warp_col);
    let mut c_t = c_gl;
    for (a, c) in final_accs.into_iter().enumerate() {
        let m = row.mul(&cidx(bps)).add(&acc_row(&warp_row, a, &cfg));
        c_t =
            g.store(c_t.into(), c.into(), &[Idx::Const(0), Idx::Const(0), Idx::Uop(m), Idx::from(&nidx)], &[], 2).gl();
    }
}
