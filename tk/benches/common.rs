//! Shared harness for svod-tk's per-kernel criterion benches (`fa`, `matmul`, `knn` —
//! one bench binary each). They bench svod-tk's kernels through their public `Tensor`
//! interface, timed the way the model runs them (`prepare()` → `execute_profiled`): GPU
//! device time comes from `execute_profiled`'s per-kernel HW stamps (the criterion
//! `iter_custom` source), so outlier rejection / CIs operate on real on-device time, not
//! host wall-clock.
//!
//! Run one kernel: `SVOD_DEVICE=AMD:0 cargo bench -p svod-tk --bench knn`
//! GPU benches self-skip (record no samples) when no supported AMD GPU is present.

use svod_dtype::DType;
use svod_tensor::Tensor;

/// A realized random bf16 tensor on the env-selected device.
pub fn randn_bf16(shape: &[usize]) -> Tensor {
    let mut t = Tensor::randn(shape).expect("randn").cast(DType::BFloat16).expect("→bf16");
    t.realize().expect("realize");
    t
}

/// Whether the env-selected device is a supported AMD GPU with the AMD-LLVM toolchain
/// (`check_target`), so HW dispatch stamps are available. `cargo bench` has no
/// `#[ignore]`, so a bench self-skips cleanly here instead of recording garbage (or
/// panicking) on CPU.
pub fn requirements_met(archs: &'static [svod_dtype::AmdArch]) -> bool {
    let spec = Tensor::empty(&[1], DType::Float32).device(); // the env/default device
    svod_tk::target::check_target(&spec, archs).is_ok()
}

/// Sum GPU device time (ns) over `iters` replays of a prepared graph plan, via
/// `execute_profiled`'s per-kernel HW stamps (an op may lower to several kernels).
pub fn plan_gpu_ns(plan: &svod_runtime::ExecutionPlan, iters: u64) -> u64 {
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
