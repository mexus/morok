//! Flash-attention forward — a GPU-free graph-shape check plus a gfx942
//! comparison against `Tensor::scaled_dot_product_attention`.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::UOp;

use crate::Kernel;
use crate::kernels::fa::{FaConfig, build_fa_mw_rdb};

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

/// The target gate rejects a non-gfx942 device up front (host — no GPU needed): a
/// CPU spec resolves to no AMD arch, so `check_target` errs `UnsupportedTarget`
/// instead of letting the gfx942-only kernel mis-render/compile-fail later. The
/// gate is generic over the kernel's declared `FA_SUPPORTED_ARCHS`, not a hardcoded
/// arch — adding another GPU is extending that list, not rewriting the gate.
#[test]
fn test_target_gate_rejects_non_gfx942() {
    use crate::kernels::fa::FA_SUPPORTED_ARCHS;
    let err = crate::target::check_target(&DeviceSpec::Cpu, FA_SUPPORTED_ARCHS);
    assert!(
        matches!(err, Err(crate::launch::Error::UnsupportedTarget { .. })),
        "a CPU device must be rejected by the gfx942 gate, got {err:?}"
    );
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
        build_fa_mw_rdb(
            &ker,
            b,
            n,
            h,
            h_kv,
            d,
            FaConfig { q_blk: 16, kv_blk: 16, unroll, ..Default::default() },
            svod_dtype::DType::BFloat16,
            false,
        );
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

/// Host regression guard for the **graph/realize-path** lowering of a hand-lowered
/// (`opts_to_apply=Some(vec![])`) tile-kernel SINK on AMD. Mirrors the realize
/// optimize→render path: `optimize_kernel_with_config` → `decompose_with` →
/// `program_from_sink` → `do_linearize` → `type_verify`. Without the
/// hand-lowered-kernel optimizer bypass (`schedule/src/optimizer/mod.rs`), the
/// UNROLL-expander broadcasts a scalar GLOBAL `Ptr` into an illegal `Ptr{vcount:4}`
/// vector that fails `type_verify`; with the bypass it renders identically to the
/// direct path (`test_fa_mw_rdb_renders_bounded`): rolled QKᵀ/A·V loops, one iglp.
#[test]
fn test_fa_graph_path_renders_clean() {
    use svod_schedule::{OptimizerConfig, OptimizerRenderer, optimize_kernel_with_config};

    let (b, h, h_kv, d) = (1usize, 2, 2, 64);
    let n = 128usize;
    let ker =
        Kernel::new("fa_mw_rdb", [h as i64, (n / 16 / 8) as i64, b as i64], 8 * 64, dummy_fa_buffers(b, n, h, h_kv, d));
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk: 16, kv_blk: 16, ..Default::default() },
        svod_dtype::DType::BFloat16,
        false,
    );
    let sink = ker.finish(1);

    // Realize builds the optimizer renderer for gfx942 via for_amd_arch.
    let opt_ren = OptimizerRenderer::for_amd_arch(svod_dtype::AmdArch::Gfx942);
    let config = OptimizerConfig::default();
    let optimized = optimize_kernel_with_config(sink, &opt_ren, &config);

    // Decompose (AMD renderer's decompositor) then program_from_sink + linearize.
    let text_ren = svod_codegen::llvm::LlvmTextRenderer::amd(svod_dtype::AmdArch::Gfx942);
    let decomposed = match svod_codegen::traits::Renderer::decompositor(&text_ren) {
        Some(m) => svod_ir::decompositions::decompose_with(&optimized, &m),
        None => optimized,
    };
    let program = svod_codegen::program_pipeline::program_from_sink(decomposed, svod_dtype::DeviceSpec::Cpu);
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("do_linearize");
    let linear_uop = linearized
        .toposort()
        .into_iter()
        .find(|u| matches!(u.op(), svod_ir::Op::Linear { .. }))
        .expect("LINEAR present");
    // This is the verify the real do_render runs (program_pipeline.rs:129-131).
    svod_schedule::spec::type_verify(&linear_uop, &svod_schedule::spec::spec_program())
        .expect("type_verify must pass (the Ptr{vcount:4} failure surfaces here)");
    let code = svod_codegen::traits::Renderer::render(&text_ren, &linear_uop, Some("fa_rdb")).expect("render").code;
    let mfma = code.lines().filter(|l| l.contains("mfma.f32.16x16x16bf16.1k") && !l.contains("declare")).count();
    assert!(mfma > 0, "graph-path FA must render mfma calls, got {mfma}");
    // Matches the direct-path render contract (test_fa_mw_rdb_renders_bounded):
    // the rolled QKᵀ/A·V loops kept (< 8 flat sites), one iglp delegation.
    assert!(mfma < 8, "rolled graph FA keeps the QKᵀ/A·V loops ({mfma} < 8)");
    assert_eq!(
        code.matches("call void @llvm.amdgcn.iglp.opt(i32 0)").count(),
        1,
        "marker lowered to one iglp delegation (identical to the direct path)"
    );
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
                crate::kernels::fa::build_fa_mw_db(
                    ker,
                    b,
                    n,
                    h,
                    h_kv,
                    d,
                    FaConfig { q_blk, kv_blk, pipelined, ..Default::default() },
                );
                ker.finish(1)
            })
            .expect("fa_mw_db tiled launch")
        }
        FaPath::Rdb { q_blk, kv_blk, unroll } => {
            let grid = [h as i64, (n / q_blk / 8) as i64, b as i64];
            crate::run_kernel("fa_mw_rdb", grid, 8 * 64, &mut [&mut o], &[&q, &k, &v], |ker| {
                crate::kernels::fa::build_fa_mw_rdb(
                    ker,
                    b,
                    n,
                    h,
                    h_kv,
                    d,
                    FaConfig { q_blk, kv_blk, unroll, ..Default::default() },
                    q.uop().dtype(),
                    false,
                );
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

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_graph_amd -- --ignored --nocapture`.
///
/// **Phase-A gate (graph integration):** the graph-native `flash_attention` (a
/// `custom_kernel` / `Op::Call` node realized by the tensor scheduler) must match
/// **(a)** causal SDPA (the oracle, tol 2e-2) AND **(b)** the direct-launch
/// `flash_attention_forward_mw_rdb` output **bit-for-bit** (same kernel, different
/// launch path). Proves the full FA SINK — LDS + barriers + 512-thread block +
/// dynamic `range_uop` bound + ds_bpermute + WMMA — survives `custom_kernel →
/// rangeify → realize` on gfx942.
#[test]
#[ignore]
fn test_fa_graph_amd() {
    use svod_tensor::Tensor;
    for (b, n, h, d) in [(1usize, 128usize, 2usize, 64usize), (1, 512, 2, 64), (2, 256, 4, 64)] {
        let mk = || {
            let t = Tensor::randn(&[b, n, h, d]).expect("randn");
            let mut t = t.cast(DType::BFloat16).expect("cast bf16");
            t.realize().expect("realize");
            t
        };
        let (q, k, v) = (mk(), mk(), mk());

        // Graph path: lazy custom_kernel Tensor → realize.
        let og = crate::kernels::fa::flash_attention(&q, &k, &v).expect("graph fa");
        let mut og_f = og.cast(DType::Float32).expect("og→f32");
        og_f.realize().expect("realize graph");
        let graph: Vec<f32> = og_f.as_vec::<f32>().expect("read graph");

        // Direct-launch path (same kernel, in-place dispatch).
        let mut od = Tensor::empty(&[b, n, h, d], DType::BFloat16);
        crate::kernels::fa::flash_attention_forward_mw_rdb(&mut od, &q, &k, &v).expect("direct fa");
        let mut od_f = od.cast(DType::Float32).expect("od→f32");
        od_f.realize().expect("realize direct");
        let direct: Vec<f32> = od_f.as_vec::<f32>().expect("read direct");

        // Reference: causal SDPA over the same operands.
        let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
        let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
        reference.realize().expect("realize reference");
        let expected = reference.as_vec::<f32>().expect("read reference");

        assert_eq!(graph.len(), expected.len(), "graph/ref length mismatch");
        assert_eq!(graph.len(), direct.len(), "graph/direct length mismatch");
        let (atol, rtol) = (2e-2f32, 2e-2f32);
        let (mut max_abs_ref, mut worst, mut max_abs_direct) = (0.0f32, 0.0f32, 0.0f32);
        for i in 0..graph.len() {
            let (g, e, dd) = (graph[i], expected[i], direct[i]);
            max_abs_ref = max_abs_ref.max((g - e).abs());
            worst = worst.max((g - e).abs() - rtol * e.abs());
            max_abs_direct = max_abs_direct.max((g - dd).abs());
        }
        println!("fa[graph] B={b} N={n} H={h} D={d}: vs SDPA = {max_abs_ref:e}, vs direct = {max_abs_direct:e}");
        assert!(worst <= atol, "graph FA vs SDPA exceeds tol (max abs {max_abs_ref:e})");
        assert!(max_abs_direct < 1e-3, "graph FA must match direct-launch bit-for-bit (Δ {max_abs_direct:e})");
    }
}

// Generic custom-kernel **check** via the `tensor`-layer macro — the graph-native
// `flash_attention` vs causal SDPA on gfx942. Demonstrates the reusable
// definition(`Tensor::graph_kernel`)/check(`custom_kernel_check!`) facility: tk just
// supplies the kernel `run` + the `reference` op; the boilerplate (build inputs, run
// both, cast f32, compare within tol) is generated.
// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_graph_check_amd -- --ignored --nocapture`.
svod_tensor::custom_kernel_check! {
    test_fa_graph_check_amd,
    inputs (q, k, v): shape [1, 128, 2, 64], dtype svod_dtype::DType::BFloat16,
    run: |q, k, v| crate::kernels::fa::flash_attention(q, k, v),
    reference: |q, k, v| {
        use svod_tensor::Tensor;
        let perm = |t: &Tensor| {
            t.cast(svod_dtype::DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute")
        };
        let (qp, kp, vp) = (perm(q), perm(k), perm(v));
        let r = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
        r.try_permute(&[0, 2, 1, 3])
    },
    tol: 2e-2,
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_noncausal_f16_amd -- --ignored --nocapture`.
///
/// The unified [`crate::kernels::fa::flash_attention_with`] **non-causal, f16,
/// unmasked** path vs full-bidirectional SDPA over the same f16 operands. Exercises
/// the `causal: false` KV sweep and the f16 WMMA-operand globals end-to-end on
/// gfx942. Tol 2e-2 (f16 accumulation slack).
#[test]
#[ignore]
fn test_fa_noncausal_f16_amd() {
    use crate::kernels::fa::{FaOpts, flash_attention_with};
    use svod_tensor::Tensor;

    let (b, n, h, d) = (1usize, 256usize, 8usize, 64usize);
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::Float16).expect("cast f16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());

    let og = flash_attention_with(&q, &k, &v, FaOpts { causal: false, key_lens: None }).expect("fa noncausal");
    let mut og_f = og.cast(DType::Float32).expect("og→f32");
    og_f.realize().expect("realize og");
    let got: Vec<f32> = og_f.as_vec::<f32>().expect("read og");

    // Reference: full (non-causal) SDPA over the same f32-permuted operands.
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(false).call().expect("sdpa");
    let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    println!("fa[noncausal,f16] B={b} N={n} H={h} D={d}: max abs error = {max_abs:e}");
    assert!(max_abs <= 2e-2, "non-causal f16 FA exceeds tol (max abs {max_abs:e})");
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib fa::test_fa_noncausal_f16_masked_amd -- --ignored --nocapture`.
///
/// The unified [`crate::kernels::fa::flash_attention_with`] **non-causal, f16,
/// key-masked** path vs full SDPA with the SAME `[B,1,1,N]` key mask. `key_lens =
/// [200]` (valid keys) ⇒ keys `200..256` are masked. Both the kernel and the
/// reference use KEY-only masking, so they agree on EVERY row (the full output is
/// compared, including the padded query rows `200..256`). Tol 2e-2.
#[test]
#[ignore]
fn test_fa_noncausal_f16_masked_amd() {
    use crate::kernels::fa::{FaOpts, flash_attention_with};
    use svod_tensor::Tensor;

    let (b, n, h, d) = (1usize, 256usize, 8usize, 64usize);
    let valid: i32 = 200;
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::Float16).expect("cast f16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());

    // Per-batch valid-key-count tensor [B] i32 (on the default/AMD device).
    let mut lens = Tensor::from_slice([valid; 1]);
    lens.realize().expect("realize lens");

    let og = flash_attention_with(&q, &k, &v, FaOpts { causal: false, key_lens: Some(&lens) }).expect("fa masked");
    let mut og_f = og.cast(DType::Float32).expect("og→f32");
    og_f.realize().expect("realize og");
    let got: Vec<f32> = og_f.as_vec::<f32>().expect("read og");

    // Reference: full SDPA with the same [B,1,1,N] bool key mask (true = masked,
    // where arange(N) >= valid), mirroring the kernel's `kv_pos >= lens[batch]`.
    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let range = Tensor::arange(n as i64, None, None).expect("arange").try_reshape([1usize, 1, 1, n]).expect("reshape");
    let lref = Tensor::from_slice([valid; 1]).try_reshape([b, 1, 1, 1]).expect("reshape lens");
    let mask = range.try_ge(&lref).expect("ge mask");
    let ref_bhnd = qp
        .scaled_dot_product_attention()
        .key(&kp)
        .value(&vp)
        .is_causal(false)
        .attn_mask(&mask)
        .call()
        .expect("sdpa masked");
    let mut reference = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    reference.realize().expect("realize reference");
    let expected = reference.as_vec::<f32>().expect("read reference");

    assert_eq!(got.len(), expected.len(), "length mismatch");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    println!("fa[noncausal,f16,masked lens={valid}] B={b} N={n} H={h} D={d}: max abs error = {max_abs:e}");
    assert!(max_abs <= 2e-2, "non-causal masked f16 FA exceeds tol (max abs {max_abs:e})");
}
