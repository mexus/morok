//! Flash-attention forward — a GPU-free graph-shape check plus a gfx942
//! comparison against `Tensor::scaled_dot_product_attention`.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::{ConstValue, Op, TernaryOp, UOp, UnaryOp};

use crate::Kernel;
use crate::kernels::fa::{build_fa, build_fa_mw, build_fa_mw_db, build_fa_mw_rdb};

/// `(o, q, k, v)` dummy BUFFER UOps for a GPU-free FA build.
fn dummy_fa_buffers(b: usize, n: usize, h: usize, h_kv: usize, d: usize) -> Vec<Arc<UOp>> {
    let q_sz = b * n * h * d;
    let kv_sz = b * n * h_kv * d;
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, q_sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, q_sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, kv_sz, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, kv_sz, DType::BFloat16),
    ]
}

/// The FA forward SINK is well-formed and carries the expected structure: WMMA
/// (QKᵀ and A·V), the K/V smem fill (`DefineLocal` + `Barrier`), an in-register
/// `ds_bpermute` softmax reduce (`Op::Custom`, no reduce LDS/barrier), the causal
/// `WHERE` mask, and the `exp2` online softmax.
#[test]
fn test_fa_kernel_builds() {
    let (b, n, h, h_kv, d) = (1usize, 32, 2, 2, 64);
    let ker = Kernel::new("fa", [h as i64, (n / 16) as i64, b as i64], 64, dummy_fa_buffers(b, n, h, h_kv, d));
    build_fa(&ker, b, n, h, h_kv, d);
    let sink = ker.finish(1);

    assert!(matches!(sink.op(), Op::Sink { .. }), "FA finishes in a SINK");
    let topo = sink.toposort();
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "FA emits WMMA (QKᵀ / A·V)");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "FA uses LDS (K/V smem)");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "FA inserts K/V-fill workgroup barriers");
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Custom { .. })),
        "FA's softmax row_reduce uses a ds_bpermute Op::Custom shuffle"
    );
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Ternary(TernaryOp::Where, _, _, _))),
        "FA applies the causal WHERE mask"
    );
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Exp2, _))), "FA uses exp2 for the online softmax");
    // Causal block-skip: the KV loop bound is dynamic (`q_seq + 1`), so its
    // RANGE end is not a constant — a runtime-trip loop, not `0..n/BLK`.
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Range { end, .. } if !matches!(end.op(), Op::Const(_)))),
        "FA's causal KV loop must have a dynamic (q_seq+1) bound, not a constant trip count"
    );
}

/// The multi-wave FA forward SINK carries the multi-wave structure: a 512-thread
/// block (`Special` `lidx0` range = `NUM_WARPS * 64`), shared K/V LDS, WMMA,
/// the in-register `ds_bpermute` reduce, the causal mask + `exp2`, and a dynamic
/// (group-max) KV loop bound.
#[test]
fn test_fa_mw_kernel_builds() {
    // N=128 → 8 q-blocks → 1 multi-wave block (grid dim1 = 1).
    let (b, n, h, h_kv, d) = (1usize, 128, 2, 2, 64);
    let block = 8 * 64;
    let ker = Kernel::new("fa_mw", [h as i64, 1, b as i64], block, dummy_fa_buffers(b, n, h, h_kv, d));
    build_fa_mw(&ker, b, n, h, h_kv, d);
    let sink = ker.finish(1);

    assert!(matches!(sink.op(), Op::Sink { .. }), "multi-wave FA finishes in a SINK");
    let topo = sink.toposort();
    // The block thread index (`lidx0`) ranges over all 512 threads.
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Special { .. })
            && u.toposort().iter().any(|c| matches!(c.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(512))))),
        "multi-wave block spans 512 threads"
    );
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "QKᵀ / A·V WMMA");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "shared K/V LDS");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "K/V fill + cross-wave WAR barriers");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Custom { .. })), "ds_bpermute softmax reduce");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(TernaryOp::Where, _, _, _))), "causal WHERE mask");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Exp2, _))), "exp2 online softmax");
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Range { end, .. } if !matches!(end.op(), Op::Const(_)))),
        "group-max KV loop has a dynamic bound"
    );
}

/// The double-buffered multi-wave FA SINK (both stages) carries the unroll-by-2
/// structure: 4 LDS tiles (2×K, 2×V), WMMA, the causal mask + `exp2`, and a
/// dynamic half-count KV loop. Stage 1 keeps the cross-wave WAR barrier; the
/// pipelined stage drops it, so it emits strictly fewer `Barrier`s.
#[test]
fn test_fa_mw_db_kernel_builds() {
    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let block = 8 * 64;
    // (q_blk, kv_blk, n): n must be a multiple of q_blk*NUM_WARPS(=8).
    let count_barriers = |pipelined: bool, q_blk: usize, kv_blk: usize, n: usize| {
        let gd1 = (n / q_blk / 8) as i64;
        let ker = Kernel::new("fa_mw_db", [h as i64, gd1, b as i64], block, dummy_fa_buffers(b, n, h, h_kv, d));
        build_fa_mw_db(&ker, b, n, h, h_kv, d, pipelined, q_blk, kv_blk);
        let sink = ker.finish(1);
        let topo = sink.toposort();
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "QKᵀ / A·V WMMA");
        // 4 LDS allocations: k_smem{0,1}, v_smem{0,1}.
        let lds = topo.iter().filter(|u| matches!(u.op(), Op::DefineLocal(_))).count();
        assert_eq!(lds, 4, "double-buffer allocates 2×K + 2×V LDS tiles, got {lds}");
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(TernaryOp::Where, _, _, _))), "causal WHERE mask");
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Exp2, _))), "exp2 online softmax");
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Range { end, .. } if !matches!(end.op(), Op::Const(_)))),
            "half-count KV loop has a dynamic bound"
        );
        // Count loop RANGEs whose constant trip = 2 — a 2-fragment tile axis
        // (`KV_BLK/16` or `Q_BLK/16` == 2) the WMMA/transpose/reduce loops fold.
        // The optimizer is disabled, so a bigger tile shows up as larger *loop
        // bounds*, not more static WMMA nodes; a `{16,16}` tile (all axes = 1
        // fragment, d = 4) has none.
        let frag2 = topo
            .iter()
            .filter(|u| matches!(u.op(), Op::Range { end, .. } if matches!(end.op(), Op::Const(cv) if matches!(cv.0, ConstValue::Int(2)))))
            .count();
        (topo.iter().filter(|u| matches!(u.op(), Op::Barrier { .. })).count(), frag2)
    };
    // {16,16} baseline: pipelined drops the redundant WAR barriers, and the
    // single-fragment tile loops have no trip-2 RANGE.
    let (naive, b16) = count_barriers(false, 16, 16, 128);
    let (pipelined, _) = count_barriers(true, 16, 16, 128);
    assert!(naive > 0, "stage-1 emits fill + WAR barriers");
    assert!(pipelined < naive, "pipelined stage drops the redundant WAR barriers ({pipelined} < {naive})");
    assert_eq!(b16, 0, "{{16,16}} single-fragment tiles have no trip-2 loop, got {b16}");
    // Bigger per-warp tiles still build a well-formed SINK, and the larger
    // (2-fragment) tile axes introduce trip-2 fragment loops.
    let (_, f32) = count_barriers(true, 32, 32, 512);
    let (_, f3264) = count_barriers(true, 32, 64, 512);
    assert!(f32 > 0, "{{32,32}} 2-fragment tiles add trip-2 fragment loops, got {f32}");
    assert!(f3264 > 0, "{{32,64}} 2-fragment Q-tile adds trip-2 fragment loops, got {f3264}");
}

/// Graph-shape (no GPU): the rolled double-buffered FA builds an **acyclic** SINK
/// with TWO LDS double buffers (one `st_db` each for K and V — vs the unroll-by-2's
/// four static tiles), a `kv_idx % 2` parity select, WMMA, the causal mask +
/// `exp2`, a dynamic-bound rolled KV loop, and a workgroup barrier. Catches a
/// dropped/cyclic anchoring edge before any GPU run.
#[test]
fn test_fa_mw_rdb_kernel_builds() {
    use svod_ir::BinaryOp;

    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let n = 128usize; // multiple of Q_BLK(16)*NUM_WARPS(8) = 128.
    let block = 8 * 64;
    let ker =
        Kernel::new("fa_mw_rdb", [h as i64, (n / 16 / 8) as i64, b as i64], block, dummy_fa_buffers(b, n, h, h_kv, d));
    build_fa_mw_rdb(&ker, b, n, h, h_kv, d, 16, 16, false);
    let sink = ker.finish(1);

    assert!(matches!(sink.op(), Op::Sink { .. }), "rolled-db FA finishes in a SINK");
    let topo = sink.toposort(); // would diverge on a cyclic graph

    assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "QKᵀ / A·V WMMA");
    // TWO LDS double buffers (k_smem, v_smem), each one `st_db` DefineLocal of 2×
    // size — half the allocation *count* of the unroll-by-2's four static tiles.
    let lds = topo.iter().filter(|u| matches!(u.op(), Op::DefineLocal(_))).count();
    assert_eq!(lds, 2, "rolled db allocates one st_db each for K and V, got {lds}");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Ternary(TernaryOp::Where, _, _, _))), "causal WHERE mask");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Unary(UnaryOp::Exp2, _))), "exp2 online softmax");
    assert!(topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "workgroup barrier present");
    assert!(
        topo.iter().any(|u| matches!(u.op(), Op::Range { end, .. } if !matches!(end.op(), Op::Const(_)))),
        "rolled KV loop has a dynamic bound"
    );
    // The `kv_idx % 2` parity select (buffer half + loop-scoping) → a Mod-by-2.
    let has_parity = topo.iter().any(|u| {
        matches!(u.op(), Op::Binary(BinaryOp::Mod, _, dv)
            if matches!(dv.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(2))))
    });
    assert!(has_parity, "the kv_idx % 2 double-buffer parity select is present");
}

/// Render-regression (CPU, no GPU): the rolled-db FA must render to AMD LLVM IR
/// in **bounded** memory/time. The `Mod`-based prefetch clamp in
/// [`build_fa_mw_rdb`] avoids a `WHERE` in the prefetch-block index that the
/// renderer mis-orders past its address-MUL consumer (which would make it hit
/// `RenderContext::get` on an unrendered node and `{:?}` the whole shared graph).
/// This test renders the full kernel and asserts it returns.
#[test]
fn test_fa_mw_rdb_renders_bounded() {
    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let n = 128usize;
    let render = |unroll: bool| {
        let ker = Kernel::new(
            "fa_mw_rdb",
            [h as i64, (n / 16 / 8) as i64, b as i64],
            8 * 64,
            dummy_fa_buffers(b, n, h, h_kv, d),
        );
        build_fa_mw_rdb(&ker, b, n, h, h_kv, d, 16, 16, unroll);
        let sink = ker.finish(1);
        let lowered = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
        let program = svod_codegen::program_pipeline::program_from_sink(lowered, svod_dtype::DeviceSpec::Cpu);
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
        let linear_uop = linearized
            .toposort()
            .into_iter()
            .find(|u| matches!(u.op(), svod_ir::Op::Linear { .. }))
            .expect("LINEAR present");
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
        // Returns (no OOM/hang) ⇒ the Mod-clamped prefetch index renders.
        svod_codegen::traits::Renderer::render(&renderer, &linear_uop, Some("fa_rdb")).expect("render").code
    };

    // The attention marker lowers (Stage 1) to a single backend-delegated interleave
    // at the loop top (Stage 2 will swap this for the softmax/MFMA comb).
    let rolled = render(false);
    assert_eq!(
        rolled.matches("call void @llvm.amdgcn.iglp.opt(i32 0)").count(),
        1,
        "marker lowered to one iglp delegation"
    );

    // Flatness (P1): with the unroll flag the QKᵀ (4 K-steps) and A·V (4 output
    // fragments) MFMAs render as 8 distinct flat `mfma` call sites — the rolled
    // form keeps them looped (strictly fewer). The exp2 online softmax likewise
    // leaves the rolled loop bodies. This is the comb's prerequisite.
    let mfma =
        |code: &str| code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    let flat = render(true);
    assert_eq!(mfma(&flat), 8, "unrolled FA slice renders 8 flat mfma (4 QKᵀ + 4 A·V)");
    assert!(mfma(&rolled) < 8, "rolled FA slice keeps the QKᵀ/A·V loops ({} < 8 mfma)", mfma(&rolled));
}

// =============================================================================
// Hardware-gated end-to-end flash-attention on gfx942.
// =============================================================================

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_amd -- --ignored --nocapture`.
///
/// Compares the FA forward to `q.scaled_dot_product_attention(k, v, is_causal)`
/// over identical bf16 operands (svod has no `enable_gqa`, so `H == H_KV`).
/// Sweeps several sequence lengths so the causal block-skip is exercised across
/// distinct trip counts — causal SDPA *is* the full-mask oracle, so each pass
/// checks the truncated `0..=q_seq` loop equals the full loop + triangular mask.
#[test]
#[ignore]
fn test_fa_amd() {
    for n in [32usize, 64, 128] {
        run_fa_amd_case(1, n, 2, 64, FaPath::Single);
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_amd -- --ignored --nocapture`.
///
/// The multi-wave (8-warp) FA correctness gate vs causal SDPA, at sequence
/// lengths where `n/16` is a multiple of `NUM_WARPS` (so the 512-thread block
/// tiles evenly): N=128 (1 block) through 2048. Also sweeps an occupancy-bound
/// batched shape (B=2, H=4) so the grid spans many blocks.
#[test]
#[ignore]
fn test_fa_mw_amd() {
    for n in [128usize, 512, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath::Mw);
    }
    run_fa_amd_case(2, 256, 4, 64, FaPath::Mw);
}

/// Which FA kernel a hardware case runs.
#[derive(Clone, Copy)]
enum FaPath {
    /// Single-warp [`crate::kernels::fa::flash_attention_forward`].
    Single,
    /// Multi-wave [`crate::kernels::fa::flash_attention_forward_mw`].
    Mw,
    /// Double-buffered multi-wave (`pipelined` toggles stage 1 vs 2), with the
    /// per-warp tile heights `(q_blk, kv_blk)`.
    MwDb { pipelined: bool, q_blk: usize, kv_blk: usize },
    /// Rolled double-buffered multi-wave ([`crate::kernels::fa::build_fa_mw_rdb`]),
    /// with the per-warp tile heights `(q_blk, kv_blk)`; `unroll` toggles the
    /// fully-unrolled (flat) compute body.
    Rdb { q_blk: usize, kv_blk: usize, unroll: bool },
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_db_amd -- --ignored --nocapture`.
///
/// The double-buffered (unroll-by-2) FA correctness gate vs causal SDPA, at the
/// same shapes as [`test_fa_mw_amd`], for BOTH stages (naive unroll + pipelined
/// barrier-reduced). The unrolled path must stay bit-compatible with multi-wave.
#[test]
#[ignore]
fn test_fa_mw_db_amd() {
    for &pipelined in &[false, true] {
        // Baseline {16,16}: N a multiple of 16*8=128.
        for n in [128usize, 512, 1024, 2048] {
            run_fa_amd_case(1, n, 2, 64, FaPath::MwDb { pipelined, q_blk: 16, kv_blk: 16 });
        }
        run_fa_amd_case(2, 256, 4, 64, FaPath::MwDb { pipelined, q_blk: 16, kv_blk: 16 });
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_db_tiled_amd -- --ignored --nocapture`.
///
/// Bigger-per-warp-tile correctness gate: the symmetric `{32,32}` (2 row
/// fragments) and asymmetric `{32,64}` (HK-like, opt-in) configs vs causal SDPA,
/// for both the naive and pipelined double-buffer stages. N must be a multiple
/// of `Q_BLK * NUM_WARPS = 32*8 = 256`.
#[test]
#[ignore]
fn test_fa_mw_db_tiled_amd() {
    for &pipelined in &[false, true] {
        for &(q_blk, kv_blk) in &[(32usize, 32usize), (32, 64)] {
            for n in [512usize, 1024, 2048] {
                run_fa_amd_case(1, n, 2, 64, FaPath::MwDb { pipelined, q_blk, kv_blk });
            }
            run_fa_amd_case(2, 256, 4, 64, FaPath::MwDb { pipelined, q_blk, kv_blk });
        }
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_rdb_amd -- --ignored --nocapture`.
///
/// Rolled double-buffer FA correctness gate vs causal SDPA, at `{16,16}` and the
/// bigger `{32,32}` per-warp tile. N must be a multiple of `q_blk * NUM_WARPS`.
#[test]
#[ignore]
fn test_fa_mw_rdb_amd() {
    for n in [128usize, 512, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath::Rdb { q_blk: 16, kv_blk: 16, unroll: false });
    }
    run_fa_amd_case(2, 256, 4, 64, FaPath::Rdb { q_blk: 16, kv_blk: 16, unroll: false });
    for n in [512usize, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath::Rdb { q_blk: 32, kv_blk: 32, unroll: false });
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_mw_rdb_unroll_amd -- --ignored --nocapture`.
///
/// P1 in-situ bit-exactness: the **fully-unrolled** rolled-db FA (same structure,
/// flat QKᵀ/softmax/A·V compute) must match causal SDPA — validating `mma_u` /
/// `reduce_u` / the unrolled `map`/`copy`/`transpose` against the looped forms
/// before the cross-tile-pipeline restructure builds on them.
#[test]
#[ignore]
fn test_fa_mw_rdb_unroll_amd() {
    for n in [128usize, 512, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath::Rdb { q_blk: 16, kv_blk: 16, unroll: true });
    }
    run_fa_amd_case(2, 256, 4, 64, FaPath::Rdb { q_blk: 16, kv_blk: 16, unroll: true });
    for n in [512usize, 1024, 2048] {
        run_fa_amd_case(1, n, 2, 64, FaPath::Rdb { q_blk: 32, kv_blk: 32, unroll: true });
    }
}

fn run_fa_amd_case(b: usize, n: usize, h: usize, d: usize, path: FaPath) {
    use svod_tensor::Tensor;

    let mk = || {
        let t = Tensor::randn(&[b, n, h, d]).expect("randn");
        let mut t = t.cast(DType::BFloat16).expect("cast bf16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());
    let mut o = Tensor::empty(&[b, n, h, d], DType::BFloat16);

    let h_kv = h;
    match path {
        FaPath::Single => crate::kernels::fa::flash_attention_forward(&mut o, &q, &k, &v).expect("fa launch"),
        FaPath::Mw => crate::kernels::fa::flash_attention_forward_mw(&mut o, &q, &k, &v).expect("fa_mw launch"),
        FaPath::MwDb { pipelined, q_blk, kv_blk } => {
            // Explicit tile config: 8-warp block, grid dim1 = n / q_blk / NUM_WARPS.
            let grid = [h as i64, (n / q_blk / 8) as i64, b as i64];
            crate::run_kernel("fa_mw_db", grid, 8 * 64, &mut [&mut o], &[&q, &k, &v], |ker| {
                crate::kernels::fa::build_fa_mw_db(ker, b, n, h, h_kv, d, pipelined, q_blk, kv_blk);
                ker.finish(1)
            })
            .expect("fa_mw_db tiled launch")
        }
        FaPath::Rdb { q_blk, kv_blk, unroll } => {
            let grid = [h as i64, (n / q_blk / 8) as i64, b as i64];
            crate::run_kernel("fa_mw_rdb", grid, 8 * 64, &mut [&mut o], &[&q, &k, &v], |ker| {
                crate::kernels::fa::build_fa_mw_rdb(ker, b, n, h, h_kv, d, q_blk, kv_blk, unroll);
                ker.finish(1)
            })
            .expect("fa_mw_rdb tiled launch")
        }
    }
    let mut of = o.cast(DType::Float32).expect("o→f32");
    of.realize().expect("realize o→f32");
    let got: Vec<f32> = of.as_vec::<f32>().expect("read o");

    // Reference: permute [B,N,H,D] → [B,H,N,D], SDPA (causal), permute back.
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
    let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let (atol, rtol) = (2e-2f32, 2e-2f32);
    let mut max_abs = 0.0f32;
    let mut worst = 0.0f32;
    for (g, e) in got.iter().zip(&expected) {
        let abs = (g - e).abs();
        max_abs = max_abs.max(abs);
        worst = worst.max(abs - rtol * e.abs());
    }
    let label = match path {
        FaPath::Single => "sw".to_string(),
        FaPath::Mw => "mw".to_string(),
        FaPath::MwDb { pipelined, q_blk, kv_blk } => {
            format!("mw_db[{},{}x{}]", if pipelined { "pipe" } else { "naive" }, q_blk, kv_blk)
        }
        FaPath::Rdb { q_blk, kv_blk, unroll } => {
            format!("mw_rdb[{}{q_blk}x{kv_blk}]", if unroll { "u," } else { "" })
        }
    };
    println!("fa[{label}] B={b} N={n} H={h} D={d}: max abs error = {max_abs:e}");
    assert!(worst <= atol, "FA exceeds atol+rtol*|e| (max abs {max_abs:e}, tol {atol}+{rtol}*|e|)");
}
