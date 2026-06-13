//! Criterion GPU-device-time benches for the svod-tk kernels.
//!
//! These feed criterion **GPU device time** via `iter_custom` (HW-stamped on-device
//! kernel time), so its outlier rejection, confidence intervals, and HTML plots
//! operate on the real signal — not host wall-clock, which at these tiny kernels is
//! dominated by launch overhead. They complement (do not replace) the in-tree
//! interleaved A/B harness (`src/test/unit/bench.rs`), which stays for
//! variance-sensitive *paired* comparisons under the MI300X VF's ±20% clock swing.
//!
//! Run: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench kernels`
//! Self-skips (records no samples) when no supported AMD GPU is present.

use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use svod_dtype::DType;
use svod_tensor::Tensor;

/// A realized random bf16 tensor on the env-selected device.
fn randn_bf16(shape: &[usize]) -> Tensor {
    let mut t = Tensor::randn(shape).expect("randn").cast(DType::BFloat16).expect("→bf16");
    t.realize().expect("realize");
    t
}

/// Whether the kernel's *actual* requirement is met on the env-selected device —
/// the SAME gate the kernel enforces (`check_target`: declared arch + AMD-LLVM
/// toolchain), not a weaker "any AMD GPU" probe. `cargo bench` has no `#[ignore]`,
/// so the bench self-skips cleanly here instead of panicking later in the gate.
fn requirements_met(archs: &[svod_dtype::AmdArch]) -> bool {
    let spec = Tensor::empty(&[1], DType::Float32).device(); // the env/default device
    svod_tk::target::check_target(&spec, archs).is_ok()
}

/// Sum GPU device time (ns) over `iters` replays of a prepared graph plan — the
/// graph (Tensor) path, via `execute_profiled`'s per-kernel HW stamps.
fn plan_gpu_ns(plan: &svod_runtime::ExecutionPlan, iters: u64) -> u64 {
    let mut total = 0u64;
    for _ in 0..iters {
        let profiles = plan.execute_profiled().expect("execute_profiled");
        total += profiles
            .iter()
            .filter_map(|p| match (p.gpu_start_ns, p.gpu_end_ns) {
                (Some(s), Some(e)) => Some(e - s),
                _ => None,
            })
            .sum::<u64>();
    }
    total
}

/// Graph-native flash-attention (the `custom_kernel` node) vs causal SDPA, both
/// timed through `prepare()` → `execute_profiled` (benchmark-as-normal-code).
fn bench_fa(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::fa::FA_SUPPORTED_ARCHS) {
        eprintln!(
            "svod-tk flash_attention bench: skipped (target does not meet the kernel's gfx942 + AMD-LLVM requirement)"
        );
        return;
    }
    let (b, h, d) = (1usize, 16usize, 64usize);
    let mut group = c.benchmark_group("flash_attention");
    for &n in &[512usize, 1024, 2048] {
        // Causal useful FLOPs ≈ 2·B·H·N²·D.
        group.throughput(Throughput::Elements((2.0 * (b * h * d) as f64 * (n as f64).powi(2)) as u64));
        let (q, k, v) = (randn_bf16(&[b, n, h, d]), randn_bf16(&[b, n, h, d]), randn_bf16(&[b, n, h, d]));

        let mut fa = svod_tk::flash_attention(&q, &k, &v).expect("flash_attention");
        let fa_plan = fa.prepare().expect("prepare fa");
        group.bench_with_input(BenchmarkId::new("tk", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&fa_plan, iters))));
        });

        let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("perm");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let refb = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
        let mut ref_t = refb.try_permute(&[0, 2, 1, 3]).expect("perm back");
        let ref_plan = ref_t.prepare().expect("prepare ref");
        group.bench_with_input(BenchmarkId::new("sdpa", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&ref_plan, iters))));
        });
    }
    group.finish();
}

/// The tile matmul: the **graph** path (`svod_tk::matmul` → `prepare()` →
/// `execute_profiled`, mirroring the FA bench — the real model/UOp path) and, for
/// comparison, the isolated **direct**-launch path (`compile_kernel` →
/// `dispatch_gpu_ns`). Both read the same HW kernel-dispatch stamps.
fn bench_matmul(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("svod-tk matmul bench: skipped (target does not meet the kernel's gfx942 + AMD-LLVM requirement)");
        return;
    }
    use svod_tk::kernels::matmul::{M1_CFG, build_matmul};
    let mut group = c.benchmark_group("matmul");
    for &n in &[1024usize, 2048] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64));
        let a = randn_bf16(&[n, n]);
        let b = randn_bf16(&[n, n]);

        // Graph path (benchmark-as-normal-code), consistent with the FA bench.
        let mut mm = svod_tk::matmul(&a, &b).expect("graph matmul");
        let mm_plan = mm.prepare().expect("prepare matmul");
        group.bench_with_input(BenchmarkId::new("tk", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&mm_plan, iters))));
        });

        // Direct launch (isolated-kernel timing), for comparison.
        let mut out = Tensor::empty(&[n, n], DType::Float32);
        let launch = svod_tk::compile_kernel(
            "matmul",
            M1_CFG.grid_dims(n),
            M1_CFG.threads(),
            &mut [&mut out],
            &[&a, &b],
            |ker| {
                build_matmul(ker, n);
                ker.finish(M1_CFG.n_accum)
            },
        )
        .expect("compile matmul");
        group.bench_with_input(BenchmarkId::new("direct", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| {
                let mut total = 0u64;
                for _ in 0..iters {
                    total += launch.dispatch_gpu_ns().expect("gpu ns").unwrap_or(0);
                }
                Duration::from_nanos(black_box(total))
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fa, bench_matmul);
criterion_main!(benches);
