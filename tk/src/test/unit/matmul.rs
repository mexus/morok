//! Tests for the bf16→f32 tile matmul ([`crate::kernels::matmul`]): a port of
//! tinygrad `test_tk.py::test_simple_matmul` plus a GPU-free graph-shape check of
//! the `mma_AB` WMMA construction and the hardware-gated end-to-end checks.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{Op, UOp};

use crate::Kernel;
use crate::kernels::matmul::*;
use crate::tiles::{RT_16X16, TileLayout};

/// Dummy `(c, a, b)` BUFFER UOps for GPU-free graph-shape kernel builds.
fn dummy_buffers(n: usize) -> Vec<Arc<UOp>> {
    let sz = n * n;
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, sz, DType::BFloat16),
    ]
}

/// Graph-shape check (no GPU): the `Hints` stage injects the
/// `s_setprio`/`s_waitcnt`/`sched.barrier` `Op::Custom` scheduling nodes into
/// the SINK (one cluster per unrolled tile), and the `Unroll` stage injects
/// none — so a refactor cannot silently drop them or leave them in the baseline.
#[test]
fn test_pipeline_hints_emit_asm() {
    let n = 256usize; // n / K_STEP = 4 unrolled tiles.
    let tiles = n / K_STEP;

    let custom_codes = |stage| {
        let ker = Kernel::new("pipe", M1_CFG.grid_dims(n), M1_CFG.threads(), dummy_buffers(n));
        build_matmul_pipelined(&ker, n, stage);
        ker.finish(M1_CFG.n_accum)
            .toposort()
            .into_iter()
            .filter_map(|u| match u.op() {
                Op::Custom { code, .. } => Some(code.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
    };

    let unroll = custom_codes(PipeStage::Unroll);
    assert!(unroll.is_empty(), "Unroll stage emits no asm scheduling nodes, got {}", unroll.len());

    let hints = custom_codes(PipeStage::Hints);
    let count = |needle: &str| hints.iter().filter(|c| c.contains(needle)).count();
    // The pre-MFMA fences (`s_waitcnt`, `s_setprio(1)`) are consumed by every
    // tile's MFMAs, so all `tiles` survive. The cluster-tail fences
    // (`s_setprio(0)`, `sched.barrier`) are consumed only by the *next* tile's
    // LDS commit, so the final tile's pair is dead-code-eliminated (`tiles - 1`).
    assert_eq!(count("s_waitcnt lgkmcnt(0)"), tiles, "one waitcnt per tile");
    assert_eq!(count("s_setprio 1"), tiles, "one setprio(1) per tile");
    assert_eq!(count("s_setprio 0"), tiles - 1, "one setprio(0) per tile but the last (cluster-tail, unconsumed)");
    assert_eq!(count("llvm.amdgcn.sched.barrier"), tiles - 1, "one sched.barrier per tile but the last");
}

/// Pure graph-shape check (no GPU): `mma_AB` emits exactly one `WMMA` per
/// K-iteration with `bf16.vec(4)` × `bf16.vec(4)` → `f32.vec(4)` operands and a
/// 16×16×16 / 4-4-4 descriptor.
#[test]
fn test_mma_ab_wmma_graph_shape() {
    let ker = Kernel::new("mma_probe", [1, 1, 1], 64, vec![]);
    let warp = ker.warp();

    let a = ker.rt((64, 64), DType::BFloat16, TileLayout::Row, RT_16X16);
    let b = ker.rt((64, 64), DType::BFloat16, TileLayout::Col, RT_16X16);
    let c = ker.rt((64, 64), DType::Float32, TileLayout::Col, RT_16X16);

    let c0 = warp.zero(c);
    let out = warp.mma_ab(c0, &a, &b);

    let wmmas: Vec<_> = out.uop().toposort().into_iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).collect();
    assert_eq!(wmmas.len(), 1, "exactly one symbolic WMMA per K-iteration");

    let bf16x4 = DType::BFloat16.vec(4);
    let f32x4 = DType::Float32.vec(4);
    let Op::Wmma { a: wa, b: wb, c: wc, metadata } = wmmas[0].op() else { unreachable!() };
    assert_eq!(wa.dtype(), bf16x4, "A operand is bf16.vec(4)");
    assert_eq!(wb.dtype(), bf16x4, "B operand is bf16.vec(4)");
    assert_eq!(wc.dtype(), f32x4, "C (accumulator) operand is f32.vec(4)");
    assert_eq!(wmmas[0].dtype(), f32x4, "WMMA result is f32.vec(4)");

    assert_eq!(metadata.dims, (16, 16, 16));
    assert_eq!(metadata.dtype_in, DType::BFloat16);
    assert_eq!(metadata.dtype_out, DType::Float32);
    let prod = |axes: &[(usize, usize)]| axes.iter().map(|(_, s)| s).product::<usize>();
    assert_eq!(prod(&metadata.upcast_axes.a), 4, "A upcast product");
    assert_eq!(prod(&metadata.upcast_axes.b), 4, "B upcast product");
    assert_eq!(prod(&metadata.upcast_axes.c), 4, "C upcast product");
}

/// The fully-unrolled MMA ([`Kernel::set_unroll`]) emits one symbolic `WMMA` per
/// `(height, width, k)` fragment — a 32×32 = 2×2 output over a 32-wide K (2
/// reduce steps) is 8 flat nodes — vs the looped form's single symbolic node, and
/// renders to gfx942 with 8 distinct `mfma` instructions (no enclosing
/// `loop_body`), which the looped form cannot (it renders one mfma inside loops).
/// This is the P1 flatness de-risk: explicit Rust-`for` unroll *does* flatten the
/// MFMAs on tk's optimizer-skipping direct-launch path (route b).
#[test]
fn test_mma_unroll_flattens_mfma() {
    let build = |unroll: bool| {
        let n = 32usize;
        let ker = Kernel::new("mma_unroll_probe", [1, 1, 1], 64, dummy_buffers(n));
        ker.set_unroll(unroll);
        let c_gl = ker.gl(&[1, 1, n, n], DType::Float32);
        let _a_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
        let _b_gl = ker.gl(&[1, 1, n, n], DType::BFloat16);
        let warp = ker.warp();
        let a = warp.zero(ker.rt((n, n), DType::BFloat16, TileLayout::Row, RT_16X16));
        // `mma_ab` reads `a[h,k] b[k,w]`; a 32×32 col `b` is a 2×2 K-tiled operand.
        let b = warp.zero(ker.rt((n, n), DType::BFloat16, TileLayout::Col, RT_16X16));
        let c = warp.zero(ker.rt((n, n), DType::Float32, TileLayout::Col, RT_16X16));
        let c = warp.mma_ab(c, &a, &b);
        let z = || crate::index::Idx::Const(0);
        let _ = warp.store(c_gl.into(), c.into(), &[z(), z(), z(), z()], &[], 2);
        ker.finish(1)
    };

    let wmma_count = |sink: &Arc<UOp>| sink.toposort().iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).count();
    assert_eq!(wmma_count(&build(false)), 1, "looped mma → one symbolic WMMA node");
    assert_eq!(wmma_count(&build(true)), 8, "unrolled mma → 8 flat WMMA nodes (2×2 output × 2 K-steps)");

    let render = |sink: Arc<UOp>| {
        let lowered = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
        let program = svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu);
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
        let linear_uop =
            linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR present");
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("mma_unroll_probe")).expect("render").code
    };
    // Count mfma *call sites* — exclude the single (deduped) `declare` line.
    let mfma =
        |code: &str| code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    let (looped_mfma, unrolled_mfma) = (mfma(&render(build(false))), mfma(&render(build(true))));
    // The flatness proof: unrolling renders all 8 MFMAs as distinct flat
    // instructions (a rolled K/fragment loop cannot — it renders strictly fewer).
    assert_eq!(unrolled_mfma, 8, "unrolled mma renders 8 flat mfma — no rolled K/fragment loop");
    assert!(looped_mfma < 8, "looped mma keeps the K/fragment loops rolled ({looped_mfma} < 8 static mfma)");
}

/// Graph-shape check that a full matmul kernel builds a well-formed SINK with
/// the expected number of `WMMA` ops (one per output fragment × K reduce, all
/// symbolic ⇒ one node) and one global output store.
#[test]
fn test_matmul_kernel_builds() {
    let n = 512usize;
    let ker = Kernel::new("simple_matmul", M1_CFG.grid_dims(n), M1_CFG.threads(), dummy_buffers(n));
    build_matmul(&ker, n);
    let sink = ker.finish(2);

    assert!(matches!(sink.op(), Op::Sink { .. }), "kernel finishes in a SINK");
    let topo = sink.toposort();
    let wmmas = topo.iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).count();
    assert_eq!(wmmas, 2, "two symbolic WMMA nodes (two per-wave accumulators)");
    // Two terminal global C stores (one per accumulator).
    let Op::Sink { sources, .. } = sink.op() else { unreachable!() };
    assert_eq!(sources.len(), 2, "two terminal C stores");
    // LDS fills + the cross-wave WAR sync emit workgroup barriers.
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "workgroup barrier present");
}

/// The GLOBAL→LDS strip fills issue 128-bit `bf16.vec(8)` coalesced loads
/// (one per strip's collaborative pass), and only those — the LDS→REG gather
/// stays scalar-into-WMMA (`bf16.vec(4)` operands).
#[test]
fn test_matmul_vec_fill_graph_shape() {
    let n = 512usize;
    let ker = Kernel::new("vec_fill", M1_CFG.grid_dims(n), M1_CFG.threads(), dummy_buffers(n));
    build_matmul(&ker, n);
    let sink = ker.finish(M1_CFG.n_accum);

    let bf16x8 = DType::BFloat16.vec(8);
    let wide = sink.toposort().into_iter().filter(|u| matches!(u.op(), Op::Load { .. }) && u.dtype() == bf16x8).count();
    // One symbolic vec8 global LOAD per strip (A, B) — the outer pass is a Range,
    // so each strip's coalesced fill is a single node.
    assert_eq!(wide, 2, "two bf16.vec(8) GLOBAL→LDS loads (A and B strips), got {wide}");
}

/// Graph-shape (no GPU): the double-buffered matmul builds a well-formed
/// **acyclic** SINK with the 2× LDS double buffer, a `tile % 2` parity select
/// (the buffer-half index that also keeps the gather/commit loop-scoped), two
/// WMMA accumulators, a workgroup barrier, and two terminal C stores. Catches a
/// dropped/cyclic anchoring edge before any GPU run.
#[test]
fn test_matmul_db_builds() {
    use svod_ir::BinaryOp;

    let n = 512usize;
    let ker = Kernel::new("matmul_db", M1_CFG.grid_dims(n), M1_CFG.threads(), dummy_buffers(n));
    build_matmul_db(&ker, n);
    let sink = ker.finish(M1_CFG.n_accum);

    assert!(matches!(sink.op(), Op::Sink { .. }), "kernel finishes in a SINK");
    let topo = sink.toposort(); // would diverge on a cyclic graph

    let wmmas = topo.iter().filter(|u| matches!(u.op(), Op::Wmma { .. })).count();
    assert_eq!(wmmas, 2, "two symbolic WMMA nodes (two per-wave accumulators)");

    let Op::Sink { sources, .. } = sink.op() else { unreachable!() };
    assert_eq!(sources.len(), 2, "two terminal C stores");

    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "workgroup barrier present");

    // The `tile % 2` parity select lowers to a Mod-by-2 binary.
    let has_parity = topo.iter().any(|u| {
        matches!(u.op(), Op::Binary(BinaryOp::Mod, _, d)
            if matches!(d.op(), Op::Const(c) if matches!(c.0, svod_ir::ConstValue::Int(2))))
    });
    assert!(has_parity, "the tile % 2 double-buffer parity select is present");

    // Two double-buffered LDS allocations (A and B strips); accumulators/stages
    // are DefineReg, not DefineLocal.
    let lds = topo.iter().filter(|u| matches!(u.op(), Op::DefineLocal { .. })).count();
    assert_eq!(lds, 2, "two LDS double buffers (A, B), got {lds}");
}

/// Scheduling pass (host render, no GPU): marking the K-loop a GEMM pipeline makes
/// the post-linearization pass splice one backend-delegated interleave
/// (`@llvm.amdgcn.iglp.opt(0)`) at the loop top — the dataflow model's lever, since
/// hand-placed `s_setprio`/`sched.barrier` measured *slower* for GEMM (they pin the
/// load/MFMA overlap the double buffer exists to create). Renders `build_matmul_db`
/// to gfx942 LLVM IR and asserts the marker lowered to exactly that.
#[test]
fn test_matmul_db_sched_pass_amd_text() {
    let n = 512usize;
    let ker = Kernel::new("matmul_db", M1_CFG.grid_dims(n), M1_CFG.threads(), dummy_buffers(n));
    build_matmul_db(&ker, n);
    let sink = ker.finish(M1_CFG.n_accum);
    let lowered = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
    let program = svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu);
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop =
        linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR present");
    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
    let code = svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("matmul_db")).expect("render").code;

    let count = |needle: &str| code.matches(needle).count();
    // The GEMM marker lowers to a single backend-delegated interleave at the loop
    // top — hand-placed s_setprio/sched.barrier measured slower for GEMM (they pin
    // the load/MFMA overlap), so the dataflow path delegates to iglp.
    assert_eq!(count("call void @llvm.amdgcn.iglp.opt(i32 0)"), 1, "marker lowered to one iglp delegation");
    assert!(!code.contains("s_setprio"), "GEMM delegates to iglp — no manual priority brackets");
}

/// A `group_2d(2,4)` is 8 waves / 512 threads, with `warp_row`/`warp_col`
/// derived as `div`/`mod` of the wave id by `cols_waves`.
#[test]
fn test_group_2d_wave_index_shape() {
    use svod_ir::{BinaryOp, Op};

    let ker = Kernel::new("wave_probe", [1, 1, 1], 512, vec![]);
    let g = ker.group_2d(2, 4);
    assert_eq!(g.warps, 8, "2×4 wave grid = 8 waves");
    assert_eq!(g.rows_waves, 2);
    assert_eq!(g.cols_waves, 4);
    assert_eq!(g.group_threads(), 512, "8 waves × 64 = 512 threads/block");

    // warp_row = warpid / cols_waves (=4); warp_col = warpid % 4.
    let by_four = |u: &Arc<UOp>, op| {
        u.toposort().into_iter().any(|n| {
            matches!(n.op(), Op::Binary(o, _, d) if *o == op
                && matches!(d.op(), Op::Const(c) if matches!(c.0, svod_ir::ConstValue::Int(4))))
        })
    };
    assert!(by_four(&g.warp_row(), BinaryOp::Idiv), "warp_row divides the wave id by cols_waves=4");
    assert!(by_four(&g.warp_col(), BinaryOp::Mod), "warp_col mods the wave id by cols_waves=4");

    // Single-warp group keeps the 1×1 grid.
    let w = ker.warp();
    assert_eq!((w.warps, w.rows_waves, w.cols_waves, w.group_threads()), (1, 1, 1, 64));
}

/// `st_db` allocates a 2×-size LDS buffer, and a parity `with_base_offset` view
/// threads a runtime offset into the LDS flat address (so a double-buffer
/// gather/fill is counter-dependent and stays loop-scoped), while an ordinary
/// `st` tile's addresses carry no such offset.
#[test]
fn test_st_db_base_offset_infra() {
    use crate::tiles::ST_16X16_SWIZZLED;

    let ker = Kernel::new("db_infra", [1, 1, 1], 512, vec![]);
    // Single-half flat element count for a 256×32 bf16 tile (base 16×16):
    // (256/16)*(32/16)*16*16 = 16*2*256 = 8192.
    let db = ker.st_db((256, 32), DType::BFloat16, TileLayout::Row, ST_16X16_SWIZZLED);
    assert_eq!(db.half_elems(), 8192, "half_elems = height*width*base.rows*base.cols");
    assert!(db.base_offset().is_none(), "fresh st_db addresses half 0 (no base_offset)");

    // A parity view adds `parity * half_elems` to the flat address.
    let tile = ker.range(4); // a Loop range counter
    let parity = tile.try_mod(&crate::index::cidx(2)).expect("tile % 2");
    let off = parity.try_mul(&crate::index::cidx(db.half_elems() as i64)).expect("parity*half");
    let view = db.with_base_offset(off.clone());
    assert!(view.base_offset().is_some(), "with_base_offset sets the parity select");

    // Sanity: the underlying buffer is shared (same DefineLocal), only the view differs.
    assert!(std::sync::Arc::ptr_eq(db.uop(), view.uop()), "with_base_offset shares the backing buffer");
}

// =============================================================================
// Hardware-gated end-to-end matmul on gfx942.
// =============================================================================

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_simple_matmul_amd -- --ignored --nocapture`.
///
/// Runs the 8-wave 256×256 tile matmul on the real GPU across several N and
/// checks each against a reference `a.matmul(b)` over the *same* bf16-rounded
/// operands (bf16 tolerance ~5e-2).
#[test]
#[ignore]
fn test_simple_matmul_amd() {
    for n in [256usize, 512, 1024, 2048] {
        run_matmul_check(n);
    }
}

fn run_matmul_check(n: usize) {
    let (a, b) = matmul_inputs(n);
    let got = launch_matmul("simple_matmul", n, M1_CFG, |ker| build_matmul(ker, n), &a, &b);
    let expected = matmul_reference(&a, &b);
    let max_abs = max_abs_err(&got, &expected);
    println!("matmul N={n}: max abs error = {max_abs:e}");
    assert!(max_abs < 5e-2, "N={n}: max abs error {max_abs} exceeds bf16 tolerance 5e-2");
}

/// The chiplet/L2 grid swizzle in **isolation** (1-D grid + [`l2_swizzle`],
/// scalar fill). It permutes which workgroup computes which 256-block, so the
/// full C must be bit-identical-up-to-bf16-tolerance to `a.matmul(b)`.
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_l2swizzle_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_l2swizzle_amd() {
    let cfg = MatmulCfg { vec_load: false, ..M1_CFG };
    for n in [2048usize, 4096] {
        let (a, b) = matmul_inputs(n);
        let got = launch_matmul("matmul_l2sw", n, cfg, |ker| build_matmul_cfg(ker, n, cfg), &a, &b);
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("l2swizzle N={n}: max abs error = {max_abs:e}");
        assert!(max_abs < 5e-2, "l2swizzle N={n}: max abs error {max_abs} exceeds 5e-2");
    }
}

/// Realized bf16 `(a, b)` inputs so kernel + reference see identical rounding.
fn matmul_inputs(n: usize) -> (svod_tensor::Tensor, svod_tensor::Tensor) {
    use svod_tensor::Tensor;
    let mut a = Tensor::rand(&[n, n]).expect("rand a").cast(DType::BFloat16).expect("cast a→bf16");
    let mut b = Tensor::rand(&[n, n]).expect("rand b").cast(DType::BFloat16).expect("cast b→bf16");
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    (a, b)
}

/// f32 ground-truth `a·b` over the bf16-rounded operands.
fn matmul_reference(a: &svod_tensor::Tensor, b: &svod_tensor::Tensor) -> Vec<f32> {
    let mut reference =
        a.cast(DType::Float32).expect("a→f32").matmul(&b.cast(DType::Float32).expect("b→f32")).expect("ref matmul");
    reference.realize().expect("realize reference");
    reference.as_vec::<f32>().expect("read reference")
}

fn max_abs_err(got: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    got.iter().zip(expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max)
}

/// Build + dispatch a matmul `cfg` over `(a, b)` once, returning the f32 C.
fn launch_matmul<F>(
    name: &str,
    n: usize,
    cfg: MatmulCfg,
    build: F,
    a: &svod_tensor::Tensor,
    b: &svod_tensor::Tensor,
) -> Vec<f32>
where
    F: FnOnce(&Kernel),
{
    use svod_tensor::Tensor;
    let mut c = Tensor::empty(&[n, n], DType::Float32);
    crate::run_kernel(name, cfg.grid_dims(n), cfg.threads(), &mut [&mut c], &[a, b], |ker| {
        build(ker);
        ker.finish(cfg.n_accum)
    })
    .expect("matmul launch");
    c.as_vec::<f32>().expect("read c")
}

/// The size-adaptive matmul is correct at every N, picking [`SMALL_CFG`] for
/// small N (where the 256×256 block under-occupies the machine) and [`M1_CFG`]
/// otherwise.
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_adaptive_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_adaptive_amd() {
    for n in [256usize, 512, 768, 1024, 2048] {
        let (a, b) = matmul_inputs(n);
        let cfg = cfg_for_n(n);
        let got = launch_matmul("matmul_adaptive", n, cfg, |ker| build_matmul_adaptive(ker, n), &a, &b);
        let expected = matmul_reference(&a, &b);
        let max_abs = max_abs_err(&got, &expected);
        println!("adaptive N={n} (block={}): max abs error = {max_abs:e}", cfg.block);
        assert!(max_abs < 5e-2, "adaptive N={n}: max abs error {max_abs} exceeds 5e-2");
    }
}

/// gfx942: the double-buffered matmul ([`build_matmul_db`]) is correct vs the
/// reference and bit-identical to the single-buffered baseline — the rolled
/// pipeline only reorders memory/compute, the per-fragment WMMA accumulation
/// order is unchanged, so it must not change the result.
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_db_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_db_amd() {
    for n in [256usize, 512, 1024, 2048] {
        let (a, b) = matmul_inputs(n);
        let expected = matmul_reference(&a, &b);
        let m1 = launch_matmul("simple_matmul", n, M1_CFG, |ker| build_matmul(ker, n), &a, &b);
        let got = launch_matmul("matmul_db", n, M1_CFG, |ker| build_matmul_db(ker, n), &a, &b);
        let max_abs = max_abs_err(&got, &expected);
        let vs_m1 = max_abs_err(&got, &m1);
        println!("matmul_db N={n}: max abs error = {max_abs:e}, max |Δ vs M1| = {vs_m1:e}");
        assert!(max_abs < 5e-2, "matmul_db N={n}: max abs error {max_abs} exceeds bf16 tolerance 5e-2");
        assert!(vs_m1 < 1e-3, "matmul_db N={n}: differs from M1 by {vs_m1} (must be ~identical)");
    }
}

/// The K-loop pipeline is correct against the reference AND numerically
/// identical (bit-for-bit) to the single-buffered baseline — the pipeline only
/// reorders memory/compute, it must not change the result.
///
/// Scope (see the module note on [`build_matmul_pipelined`]): the `Unroll` stage
/// at N ≤ 1024 (full-unroll IR compile time grows steeply past that), and the
/// register-staged `Prefetch` / `Hints` stages at N ≤ 512 (their single-LDS
/// commit hazards beyond ~8 unrolled tiles, so they stay opt-in and small-N only).
///
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib matmul::test_matmul_pipelined_amd -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_pipelined_amd() {
    // (stage, max N the stage is exercised at).
    let cases = [(PipeStage::Unroll, 1024usize), (PipeStage::Prefetch, 512), (PipeStage::Hints, 512)];
    for n in [256usize, 512, 1024] {
        let (a, b) = matmul_inputs(n);
        let expected = matmul_reference(&a, &b);
        let m1 = launch_matmul("simple_matmul", n, M1_CFG, |ker| build_matmul(ker, n), &a, &b);
        for (stage, max_n) in cases {
            if n > max_n {
                continue;
            }
            let got = launch_matmul("matmul_pipe", n, M1_CFG, |ker| build_matmul_pipelined(ker, n, stage), &a, &b);
            let max_abs = max_abs_err(&got, &expected);
            let vs_m1 = max_abs_err(&got, &m1);
            println!("pipeline[{stage:?}] N={n}: max abs error = {max_abs:e}, max |Δ vs M1| = {vs_m1:e}");
            assert!(max_abs < 5e-2, "pipeline[{stage:?}] N={n}: max abs error {max_abs} exceeds 5e-2");
            assert!(vs_m1 < 1e-3, "pipeline[{stage:?}] N={n}: differs from M1 by {vs_m1} (must be identical)");
        }
    }
}
