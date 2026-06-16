//! Criterion GPU-device-time benches for svod-tk's kernels.
//!
//! These bench svod-tk's two kernels through their public `Tensor` interface,
//! timed the way the model runs them (`prepare()` → `execute_profiled`):
//! `svod_tk::flash_attention_with` — the production attention kernel (non-causal,
//! optional key padding) vs the SDPA fallback — and `svod_tk::matmul` — the hand
//! kernel (a DSL perf-canary; the model itself uses the generic optimizer) vs a
//! generic-GEMM reference. GPU device time comes from `execute_profiled`'s
//! per-kernel HW stamps (the criterion `iter_custom` source), so outlier rejection
//! / CIs operate on real on-device time, not host wall-clock.
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

/// Whether the env-selected device is a supported AMD GPU with the AMD-LLVM
/// toolchain (`check_target`), so HW dispatch stamps are available. `cargo bench`
/// has no `#[ignore]`, so the bench self-skips cleanly here instead of recording
/// garbage (or panicking) on CPU.
fn requirements_met(archs: &[svod_dtype::AmdArch]) -> bool {
    let spec = Tensor::empty(&[1], DType::Float32).device(); // the env/default device
    svod_tk::target::check_target(&spec, archs).is_ok()
}

/// Sum GPU device time (ns) over `iters` replays of a prepared graph plan, via
/// `execute_profiled`'s per-kernel HW stamps (an op may lower to several kernels).
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

/// The svod-tk hand matmul (`svod_tk::matmul`, a graph-native `custom_kernel`
/// node) vs svod's generic `Tensor::matmul` reference — both timed through
/// `prepare()` → `execute_profiled`. The hand kernel carries no production load
/// (the model uses the generic optimizer via `Tensor::linear`); this is its DSL
/// perf-canary. Square `M = N = K`, bf16 in, f32 accumulate.
fn bench_matmul(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::matmul::MATMUL_SUPPORTED_ARCHS) {
        eprintln!("svod-tk matmul bench: skipped (no supported AMD GPU / toolchain)");
        return;
    }
    let mut group = c.benchmark_group("matmul");
    for &n in &[1024usize, 2048] {
        group.throughput(Throughput::Elements((2.0 * (n as f64).powi(3)) as u64)); // 2·M·N·K
        let a = randn_bf16(&[n, n]);
        let b = randn_bf16(&[n, n]);

        let mut y = svod_tk::matmul(&a, &b).expect("tk matmul").expect("matmul kernel applies");
        let plan = y.prepare().expect("prepare matmul");
        group.bench_with_input(BenchmarkId::new("tk", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&plan, iters))));
        });

        // Reference: svod's generic bf16→f32 GEMM (the matmul a user would write).
        let mut reft = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("ref matmul");
        let ref_plan = reft.prepare().expect("prepare ref");
        group.bench_with_input(BenchmarkId::new("generic", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&ref_plan, iters))));
        });
    }
    group.finish();
}

/// Attention as the model runs it: `svod_tk::flash_attention_with` — the exact
/// GigaAM encoder call (non-causal, `[B, T, H, d_k]` layout, optional key padding;
/// here unpadded) — vs the **non-causal** SDPA fallback the model would otherwise
/// take. Both timed through `prepare()` → `execute_profiled`.
fn bench_fa(c: &mut Criterion) {
    if !requirements_met(svod_tk::kernels::fa::FA_SUPPORTED_ARCHS) {
        eprintln!("svod-tk flash_attention bench: skipped (target does not meet the kernel's AMD-LLVM requirement)");
        return;
    }
    let (b, h, d) = (1usize, 16usize, 64usize);
    let mut group = c.benchmark_group("flash_attention");
    for &n in &[512usize, 1024, 2048] {
        // Non-causal attention FLOPs: QKᵀ + P·V, each 2·B·H·N²·d.
        group.throughput(Throughput::Elements((4.0 * (b * h * d) as f64 * (n as f64).powi(2)) as u64));
        let (q, k, v) = (randn_bf16(&[b, n, h, d]), randn_bf16(&[b, n, h, d]), randn_bf16(&[b, n, h, d]));

        // The model's exact call: non-causal, no key padding.
        let mut fa = svod_tk::flash_attention_with(&q, &k, &v, svod_tk::FaOpts { causal: false, key_lens: None })
            .expect("flash_attention_with")
            .expect("FA kernel applies for bench shape");
        let fa_plan = fa.prepare().expect("prepare fa");
        group.bench_with_input(BenchmarkId::new("tk", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&fa_plan, iters))));
        });

        // The model's fallback: non-causal SDPA. SDPA wants `[B, H, T, d]`.
        let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("perm");
        let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
        let refb = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(false).call().expect("sdpa");
        let mut ref_t = refb.try_permute(&[0, 2, 1, 3]).expect("perm back");
        let ref_plan = ref_t.prepare().expect("prepare ref");
        group.bench_with_input(BenchmarkId::new("sdpa", n), &n, |bencher, _| {
            bencher.iter_custom(|iters| Duration::from_nanos(black_box(plan_gpu_ns(&ref_plan, iters))));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_fa, bench_matmul);
criterion_main!(benches);
