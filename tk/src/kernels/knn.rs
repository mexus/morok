//! Stage 1 of a fused brute-force KNN: the **x²-free score tile**.
//!
//! For query rows `x[query, d]` and corpus rows `c[corpus, d]` the score is
//! `score[m, n] = ‖c[m]‖² − 2·⟨x[n], c[m]⟩`. The query self-term `‖x[n]‖²` is
//! dropped (it is constant per query row `n`, so it never changes the argmin over
//! the corpus `m` that the running top-K in Stage 2 takes). The dominant distance
//! term `‖c[m]‖²` (`c_sq`) is precomputed in **f32** outside the kernel and passed
//! in (an augmentation that smuggled it through a bf16 WMMA operand would lose its
//! precision), replicated along the query axis so every `(m, n)` reads `c_sq[m]`.
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

use svod_dtype::DType;

use crate::index::Idx;
use crate::kernel::Kernel;
use crate::scaffold::GlSpec;
use crate::tiles::TileLayout;
use crate::{GL, MoveIdx};

/// The WMMA tile edge (K=16); the cross MMA operates on 16×16 fragments, so the
/// corpus / query / D dims must each be a multiple of it.
const BLK: usize = 16;

/// The GPU arch(es) this kernel is built for: gfx942 (CDNA3 MFMA, wave64) and
/// gfx1151 (RDNA3.5 WMMA, wave32). Both resolve the accumulator/operand fragments
/// by role through [`crate::ArchCaps`]; the launcher gates against this list.
/// Validated on gfx942 (CDNA3) and gfx1151 (RDNA3.5).
pub const KNN_SUPPORTED_ARCHS: &[svod_dtype::AmdArch] = &[svod_dtype::AmdArch::Gfx942, svod_dtype::AmdArch::Gfx1151];

/// Build the x²-free KNN score-tile kernel into the bound ABI.
///
/// ABI (outputs then inputs, fixed by [`Kernel::bind_abi`]):
/// - `score` (`[1, 1, corpus, query]`, f32) — the output `‖c[m]‖² − 2·⟨x[n],c[m]⟩`.
/// - `x` (`[1, 1, query, d]`, bf16) — the query rows.
/// - `c` (`[1, 1, corpus, d]`, bf16) — the corpus rows.
/// - `c_sq_rep` (`[1, 1, corpus, query]`, f32) — `‖c[m]‖²` precomputed outside the
///   kernel and replicated along the query axis (each `(m, n)` holds `c_sq[m]`).
///
/// Single-tile, single-warp (`ker.warp()`); `corpus`, `query`, `d` must each be a
/// multiple of [`BLK`] (16).
///
/// # Panics
/// Panics unless `corpus`, `query`, and `d` are each a multiple of 16.
pub fn build_knn_score(ker: &Kernel, corpus: usize, query: usize, d: usize) {
    Kernel::assert_divisible(corpus, BLK, "KNN corpus");
    Kernel::assert_divisible(query, BLK, "KNN query");
    Kernel::assert_divisible(d, BLK, "KNN D");

    let bf16 = DType::BFloat16;
    let f32 = DType::Float32;
    let (row, col) = (TileLayout::Row, TileLayout::Col);
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

    // Tiles are declared by ROLE via the scaffold shortcuts (`ker.acc`/`operand`/
    // `shared_sw`), which resolve the arch fragment through `caps.frag` — so the
    // kernel names no physical fragment constant.

    // Cross term ⟨c[m], x[n]⟩ into a Col f32 accumulator `score[corpus, query]`:
    // mirror `fa_qk`'s QKᵀ load → LDS → reg → transpose → mma_atb. The corpus
    // plays K's role (the reduced row axis), the query plays Q's (the column).
    let x_smem = ker.shared_sw((query, d), bf16.clone(), row);
    let c_smem = ker.shared_sw((corpus, d), bf16.clone(), row);
    let x_reg = ker.operand((query, d), bf16.clone(), row);
    let c_reg = ker.operand((corpus, d), bf16.clone(), row);
    let x_reg_t = ker.operand((d, query), bf16.clone(), col);
    let c_reg_t = ker.operand((d, corpus), bf16.clone(), col);

    let zero4 = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::Const(0)];

    // GLOBAL → LDS (swizzled) → REG, then transpose each operand to its `[d, *]`
    // Col fragment for the contraction over D.
    let x_smem = warp.load(x_smem, x_gl, MoveIdx::block(&zero4, 2));
    let c_smem = warp.load(c_smem, c_gl, MoveIdx::block(&zero4, 2));
    let x_reg = warp.load(x_reg, x_smem, MoveIdx::default());
    let c_reg = warp.load(c_reg, c_smem, MoveIdx::default());
    let x_reg_t = warp.transpose(x_reg_t, &x_reg);
    let c_reg_t = warp.transpose(c_reg_t, &c_reg);

    // score[m, n] = Σ_d c[m,d]·x[n,d] = ⟨c[m], x[n]⟩ (corpus = row, query = col).
    let cross = warp.zero(ker.acc((corpus, query), col));
    let cross = warp.mma_atb(cross, &c_reg_t, &x_reg_t);

    // Load c_sq_rep into a tile with the SAME accumulator fragment + layout as the
    // cross MMA output, so the two align lane-for-lane (both index the accumulator
    // frag's `lane_rc`; orientation-robust per the reductions/masked tests).
    let c_sq = warp.load(ker.acc((corpus, query), col), c_sq_gl, MoveIdx::block(&zero4, 2));

    // score = c_sq − 2·cross, all f32. `mul_scalar(cross, −2)` then `add(.., c_sq)`.
    let cross = warp.mul_scalar(cross, -2.0);
    let score = warp.add(cross, &c_sq);

    let _ = warp.store(score_gl, score, MoveIdx::block(&zero4, 2));
}
