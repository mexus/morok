//! Tests for the x²-free KNN score-tile kernel ([`crate::kernels::knn`]): a
//! GPU-free graph-shape check (both archs) of the cross WMMA + `c_sq` load + f32
//! combine, and a hardware-gated end-to-end score check vs a generic-Tensor
//! reference (arch-portable: gfx942 wave64 / gfx1151 wave32).

use std::sync::Arc;

use svod_dtype::{AmdArch, DType, DeviceSpec};
use svod_ir::{Op, UOp};

use crate::kernels::knn::{KNN_SUPPORTED_ARCHS, build_knn_score};
use crate::{ArchCaps, Kernel};

/// Placeholder buffers for a GPU-free build: `score` (f32), `x`/`c` (bf16),
/// `c_sq_rep` (f32), in ABI order.
fn knn_bufs(corpus: usize, query: usize, d: usize) -> Vec<Arc<UOp>> {
    vec![
        UOp::new_buffer(DeviceSpec::Cpu, corpus * query, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, query * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, corpus * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, corpus * query, DType::Float32),
    ]
}

/// Build the KNN score SINK for `(corpus, query, d)` on `caps` (GPU-free).
fn knn_sink(corpus: usize, query: usize, d: usize, caps: ArchCaps) -> Arc<UOp> {
    let ker = Kernel::new("knn_score", [1, 1, 1], caps.wave_size as i64, knn_bufs(corpus, query, d), caps);
    build_knn_score(&ker, corpus, query, d);
    ker.finish(1)
}

/// The score kernel's graph carries: a cross-term `WMMA`, a `Load` of the f32
/// `c_sq` input (the 4th / last `Param`) into the accumulator fragment, and the
/// f32 combine (the `−2·cross` `Mul` by a const + the `+ c_sq` `Add`). No stray
/// LDS / barrier surprises beyond the two operand strips the cross MMA needs.
/// Holds on both wave64 (gfx942) and wave32 (gfx1151).
#[test]
fn test_knn_score_graph_shape() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_arch(AmdArch::Gfx1151)] {
        let arch = caps.arch;
        let topo = knn_sink(32, 32, 32, caps).toposort();

        // The cross term emits a WMMA (looped: one symbolic node per K-iteration).
        assert!(topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "{arch:?}: cross term emits a WMMA");

        // The c_sq input is the last (4th) Param; its load reads PARAM slot 3.
        let loads_param3 = topo.iter().any(|u| {
            let Op::Load { .. } = u.op() else { return false };
            u.toposort().iter().any(|s| matches!(s.op(), Op::Param { slot: 3, .. }))
        });
        assert!(loads_param3, "{arch:?}: a Load reads the c_sq input (Param slot 3)");

        // The f32 combine: a `*(-2)` multiply against a const and an add. The score
        // path is entirely f32 (no cast back to a narrow dtype before the store).
        let has_neg2 = topo.iter().any(|u| {
            let Op::Binary(svod_ir::BinaryOp::Mul, _, rhs) = u.op() else { return false };
            matches!(rhs.op(), Op::Const(c) if matches!(c.0, svod_ir::ConstValue::Float(f) if (f + 2.0).abs() < 1e-9))
        });
        assert!(has_neg2, "{arch:?}: the −2·cross scale is a Mul by the const −2.0");
        assert!(
            topo.iter().any(|u| matches!(u.op(), Op::Binary(svod_ir::BinaryOp::Add, ..))),
            "{arch:?}: the + c_sq add"
        );
    }
}

// =============================================================================
// Hardware-gated end-to-end score on gfx942 / gfx1151.
// =============================================================================

/// Whether the env-selected device is a supported AMD GPU (with the AMD-LLVM
/// toolchain) — else the `#[ignore]`d test self-skips instead of erroring on CPU.
fn device_supported() -> bool {
    let spec = svod_tensor::Tensor::empty(&[1], DType::Float32).device();
    crate::target::check_target(&spec, KNN_SUPPORTED_ARCHS).is_ok()
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib knn::test_knn_score_amd -- --ignored --nocapture`.
///
/// Random `x[query, d]`, `c[corpus, d]` (bf16); `c_sq[m] = Σ_d c[m,d]²` computed
/// host-side in f32 and replicated along the query axis. The kernel score is
/// compared to the generic-Tensor reference `c_sq[:,None] − 2·(c.f32 @ x.f32ᵀ)`,
/// oriented to the kernel's `score[m, n]` (corpus = row, query = col) tile. The
/// tolerance scales with √D like the matmul proptest (the cross term reduces over
/// D in bf16; `c_sq` is exact f32).
#[test]
#[ignore]
fn test_knn_score_amd() {
    use svod_tensor::Tensor;
    use svod_tensor::testing::allclose_f32;

    if !device_supported() {
        eprintln!("skip test_knn_score_amd: no supported AMD GPU / toolchain");
        return;
    }
    let dev = Tensor::rand(&[16, 16]).expect("probe").device();
    let arch = crate::target::resolve_arch(&dev).expect("resolve arch");
    let w = arch.wave_size() as i64;

    for (corpus, query, d) in [(16usize, 16usize, 16usize), (32, 32, 32), (32, 16, 48)] {
        // Realized bf16 inputs so the kernel and the reference see identical rounding.
        let mut x = Tensor::randn(&[query, d]).expect("randn x").cast(DType::BFloat16).expect("x→bf16");
        let mut c = Tensor::randn(&[corpus, d]).expect("randn c").cast(DType::BFloat16).expect("c→bf16");
        x.realize().expect("realize x");
        c.realize().expect("realize c");

        // c_sq[m] = Σ_d c[m,d]² in f32, replicated to [corpus, query] (each (m,n) → c_sq[m]).
        let cf = c.cast(DType::Float32).expect("c→f32");
        let mut c_sq_rep = cf
            .try_mul(&cf)
            .expect("c²")
            .sum_with()
            .axes(1isize)
            .keepdim(true)
            .call()
            .expect("Σ_d c²")
            .try_expand([corpus, query])
            .expect("replicate along query");
        c_sq_rep.realize().expect("realize c_sq_rep");

        let mut score = Tensor::empty(&[1, 1, corpus, query], DType::Float32);
        crate::run_kernel("knn_score", [1, 1, 1], w, &mut [&mut score], &[&x, &c, &c_sq_rep], |ker| {
            build_knn_score(ker, corpus, query, d);
            ker.finish(1)
        })
        .expect("knn_score launch");
        let got = score.as_vec::<f32>().expect("read score");

        // Reference: c_sq[m] − 2·⟨c[m], x[n]⟩, oriented score[m, n] (corpus row, query col).
        let xf = x.cast(DType::Float32).expect("x→f32");
        let cross = cf.matmul(&xf.try_permute(&[1, 0]).expect("xᵀ")).expect("c @ xᵀ"); // [corpus, query]
        let two = Tensor::from_slice([2.0f32]);
        let mut refb = c_sq_rep.try_sub(&cross.try_mul(&two).expect("2·cross")).expect("c_sq − 2·cross");
        refb.realize().expect("realize ref");
        let exp = refb.as_vec::<f32>().expect("read ref");

        let (atol, rtol) = (0.02 * (d as f32).sqrt(), 2e-2);
        let r = allclose_f32(&got, &exp, atol, rtol);
        let max_abs = got.iter().zip(&exp).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
        println!("knn_score corpus={corpus} query={query} d={d}: max abs error = {max_abs:e} on {arch:?}");
        assert!(r.ok, "knn_score corpus={corpus} query={query} d={d} on {arch:?}: {}", r.message);
    }
}
