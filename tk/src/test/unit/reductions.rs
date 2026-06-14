//! Cross-lane reductions (`row_reduce`) and an end-to-end softmax — a port of
//! tinygrad `test_tk.py::test_softmax`.
//!
//! The graph-shape checks run GPU-free; the softmax comparison is `#[ignore]`
//! and validates on gfx942 (lane-distributed, so the CPU backend can't run it).

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::Op;

use crate::index::Idx;
use crate::tile::RegTile;
use crate::tiles::{RT_16X16, ST_16X16, TileLayout, VecLayout};
use crate::{Kernel, MoveIdx};

const INV_LN2: f64 = std::f64::consts::LOG2_E; // 1 / ln(2) == log2(e)

/// Build the softmax-over-axis-3 SINK for a `block × n` row-softmax with a
/// `block × block` tile, mirroring tinygrad `test_softmax`.
fn build_softmax(ker: &Kernel, n: usize, block: usize) {
    let warp = ker.warp();

    // out (b, f32), then in (a, f32).
    let b = ker.gl(&[1, 1, block, n], DType::Float32);
    let a = ker.gl(&[1, 1, block, n], DType::Float32);

    let max_vec = ker.rv(block, DType::Float32, VecLayout::Ortho, RT_16X16);
    let norm_vec = ker.rv(block, DType::Float32, VecLayout::Ortho, RT_16X16);
    let max_vec_last = ker.rv(block, DType::Float32, VecLayout::Ortho, RT_16X16);

    let mut max_vec = warp.neg_inf_rv(max_vec);
    let mut norm_vec = warp.zero_rv(norm_vec);
    let mut max_vec_last = max_vec_last;

    // Pass 1: running max + normalization accumulator over the column tiles.
    let tile_col = ker.range((n / block) as i64);
    {
        let a_smem = ker.st((block, block), DType::Float32, TileLayout::Row, ST_16X16);
        let a_reg = ker.rt((block, block), DType::Float32, TileLayout::Row, RT_16X16);
        let idxs = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::from(&tile_col)];
        let a_smem = warp.load(a_smem, a.clone(), MoveIdx::block(&idxs, 2));
        let a_reg = warp.load(a_reg, a_smem, MoveIdx::default());
        let a_reg = warp.mul_scalar(a_reg, INV_LN2);

        max_vec_last = warp.copy(max_vec_last.after(smallvec![tile_col.clone()]), &max_vec);
        let mv_in = max_vec.after(smallvec![max_vec_last.uop().clone()]);
        max_vec = warp.row_reduce(mv_in, &a_reg, |x, y| x.try_max(y).expect("row max"), f64::NEG_INFINITY);

        let a_reg = warp.sub_rv(a_reg, &max_vec);
        let a_reg = warp.exp2(a_reg);
        max_vec_last = warp.exp2(warp.sub(max_vec_last, &max_vec));
        norm_vec = warp.mul(norm_vec, &max_vec_last);
        norm_vec = warp.row_reduce(norm_vec, &a_reg, |x, y| x.try_add(y).expect("row add"), 0.0);
    }
    norm_vec = norm_vec.rewrap(ker.endrange(1));
    max_vec = max_vec.after(smallvec![norm_vec.uop().clone()]);

    // Pass 2: recompute the (scaled) exponentials and normalize.
    let tile_col = ker.range((n / block) as i64);
    {
        let a_smem = ker.st((block, block), DType::Float32, TileLayout::Row, ST_16X16);
        let a_reg = ker.rt((block, block), DType::Float32, TileLayout::Row, RT_16X16);
        let idxs = [Idx::Const(0), Idx::Const(0), Idx::Const(0), Idx::from(&tile_col)];
        let a_smem = warp.load(a_smem, a.clone(), MoveIdx::block(&idxs, 2));
        let a_reg = warp.load(a_reg, a_smem, MoveIdx::default());
        let a_reg = warp.mul_scalar(a_reg, INV_LN2);
        let a_reg = warp.sub_rv(a_reg, &max_vec);
        let a_reg = warp.exp2(a_reg);
        let a_reg = warp.div_rv(a_reg, &norm_vec);
        let _ = warp.store(b, a_reg, MoveIdx::block(&idxs, 2));
    }
}

/// A bare `row_reduce` folds the three sibling 16-lane slots with an in-register
/// `ds_bpermute` wave shuffle (`Op::Custom`) — no LDS scratch (`DefineLocal`),
/// no workgroup `Barrier`, and no WMMA.
#[test]
fn test_row_reduce_graph_shape() {
    let ker = Kernel::new("row_reduce_probe", [1, 1, 1], 64, vec![], crate::ArchCaps::GFX942);
    let warp = ker.warp();

    let src = ker.rt((32, 32), DType::Float32, TileLayout::Row, RT_16X16);
    let src = warp.zero(src);
    let vec = ker.rv(32, DType::Float32, VecLayout::Ortho, RT_16X16);
    let vec = warp.zero_rv(vec);
    let out = warp.row_reduce(vec, &src, |x, y| x.try_add(y).expect("add"), 0.0);

    let topo = out.uop().toposort();
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Custom { .. })),
        "row_reduce gathers sibling lanes with a ds_bpermute Op::Custom shuffle"
    );
    assert!(
        !topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))),
        "the wave-shuffle reduce allocates no LDS scratch"
    );
    assert!(
        !topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })),
        "the wave-shuffle reduce needs no workgroup barrier"
    );
    assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "a reduction has no WMMA");
}

// =============================================================================
// Hardware-gated end-to-end softmax on gfx942.
// =============================================================================

/// Whether the active device is CDNA (wave64). These softmax tests hardcode a
/// 64-thread launch block + single-warp geometry tuned for wave64, so they skip on
/// a wave32 (RDNA) device — the reduce itself is arch-blind; only the test's launch
/// geometry is wave64-specific (a 64-thread block is 2 waves on wave32, racing the
/// shared output). A wave32 softmax would need an arch-derived block.
fn is_cdna_device() -> bool {
    let dev = svod_tensor::Tensor::rand(&[16, 16]).expect("probe tensor").device();
    crate::target::resolve_arch(&dev).is_some_and(|a| a.is_cdna())
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib reductions::test_softmax_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_softmax_amd() {
    use svod_tensor::Tensor;

    if !is_cdna_device() {
        eprintln!("skip test_softmax_amd: wave64 launch geometry (CDNA-only)");
        return;
    }
    let (n, block) = (64usize, 32usize);

    let a = Tensor::rand(&[1, 1, block, n]).expect("rand a");
    let mut a = a.cast(DType::Float32).expect("cast a");
    a.realize().expect("realize a");
    let mut out = Tensor::empty(&[1, 1, block, n], DType::Float32);

    crate::run_kernel("softmax", [1, 1, 1], 64, &mut [&mut out], &[&a], |ker| {
        build_softmax(ker, n, block);
        ker.finish(1)
    })
    .expect("softmax launch");

    let got = out.as_vec::<f32>().expect("read out");

    let mut reference = a.softmax(3isize).expect("ref softmax");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    println!("softmax N={n} block={block}: max abs error = {max_abs:e}");
    assert!(max_abs < 1e-4, "max abs error {max_abs} exceeds f32 softmax tolerance 1e-4");
}

/// P1 isolation (`SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib reductions::test_softmax_unroll_amd -- --ignored --nocapture`):
/// the **fully-unrolled** softmax (reduce_u + unrolled map/copy, no mma/db) must
/// match the reference — isolates the unrolled reduce/elementwise from the FA
/// double-buffer + mma context. Swept across block sizes so `outer_end` varies
/// (16 → 1 fragment, 32 → 2).
#[test]
#[ignore]
fn test_softmax_unroll_amd() {
    use svod_tensor::Tensor;

    if !is_cdna_device() {
        eprintln!("skip test_softmax_unroll_amd: wave64 launch geometry (CDNA-only)");
        return;
    }
    for (n, block) in [(64usize, 16usize), (64, 32)] {
        let a = Tensor::rand(&[1, 1, block, n]).expect("rand a");
        let mut a = a.cast(DType::Float32).expect("cast a");
        a.realize().expect("realize a");
        let mut out = Tensor::empty(&[1, 1, block, n], DType::Float32);

        crate::run_kernel("softmax_u", [1, 1, 1], 64, &mut [&mut out], &[&a], |ker| {
            ker.set_unroll(true);
            build_softmax(ker, n, block);
            ker.finish(1)
        })
        .expect("softmax_u launch");

        let got = out.as_vec::<f32>().expect("read out");
        let mut reference = a.softmax(3isize).expect("ref softmax");
        reference.realize().expect("realize reference");
        let expected = reference.as_vec::<f32>().expect("read reference");

        let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
        println!("softmax_u N={n} block={block}: max abs error = {max_abs:e}");
        assert!(max_abs < 1e-4, "softmax_u N={n} block={block}: max abs error {max_abs} exceeds 1e-4");
    }
}
