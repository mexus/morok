//! Per-kernel execution profiling.
//!
//! Provides structured timing data for kernel execution via
//! [`ExecutionPlan::execute_profiled()`](crate::ExecutionPlan::execute_profiled).

use std::sync::Arc;
use std::time::Duration;

use svod_dtype::DeviceSpec;

use crate::kernel_cache::CachedKernel;

/// Per-kernel timing from a profiled execution.
///
/// Holds an `Arc<CachedKernel>` for zero-copy access to kernel metadata
/// (entry point, generated code, global/local size, variable names).
///
/// # Example
///
/// ```ignore
/// let plan = tensor.prepare()?;
/// let profiles = plan.execute_profiled()?;
///
/// for (i, p) in profiles.iter().enumerate() {
///     println!("{:4} {:>8.3}ms  {}  ({} bufs, {:?})",
///         i, p.elapsed.as_secs_f64() * 1000.0,
///         p.kernel.entry_point, p.num_buffers, p.device);
/// }
/// ```
pub struct KernelProfile {
    /// Compiled kernel (entry_point, code, global_size, local_size, var_names).
    /// Debug shows the entry point only — the code/program are not printable.
    pub kernel: Arc<CachedKernel>,
    /// Device this kernel executed on.
    pub device: DeviceSpec,
    /// Number of buffer arguments.
    pub num_buffers: usize,
    /// Wall-clock execution time. On async backends (AMD AQL) this is mere
    /// submission overhead; the GPU truth is in the fields below.
    pub elapsed: Duration,
    /// HW dispatch start/end on the GPU clock (ns), when the backend stamps
    /// dispatches ([`svod_device::DispatchTimestamps`]).
    pub gpu_start_ns: Option<u64>,
    pub gpu_end_ns: Option<u64>,
}

impl std::fmt::Debug for KernelProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelProfile")
            .field("kernel", &self.kernel.entry_point)
            .field("elapsed", &self.elapsed)
            .field("gpu_start_ns", &self.gpu_start_ns)
            .field("gpu_end_ns", &self.gpu_end_ns)
            .finish_non_exhaustive()
    }
}

impl KernelProfile {
    /// GPU execution time when stamped, else the wall-clock fallback.
    pub fn gpu_or_wall(&self) -> Duration {
        match (self.gpu_start_ns, self.gpu_end_ns) {
            (Some(s), Some(e)) => Duration::from_nanos(e - s),
            _ => self.elapsed,
        }
    }
}

/// Per-kernel-name aggregate over a profiled execution, sorted by total time
/// descending. Render with [`render_histogram`].
pub struct KernelAggregate {
    pub name: String,
    pub count: usize,
    pub total: Duration,
    pub mean: Duration,
}

/// Group profiles by entry point, sum GPU (or wall-fallback) durations.
pub fn aggregate_profiles(profiles: &[KernelProfile]) -> Vec<KernelAggregate> {
    let mut map: std::collections::HashMap<&str, (usize, Duration)> = std::collections::HashMap::new();
    for p in profiles {
        let e = map.entry(&p.kernel.entry_point).or_insert((0, Duration::ZERO));
        e.0 += 1;
        e.1 += p.gpu_or_wall();
    }
    let mut out: Vec<KernelAggregate> = map
        .into_iter()
        .map(|(name, (count, total))| KernelAggregate {
            name: name.to_string(),
            count,
            total,
            mean: total / count as u32,
        })
        .collect();
    out.sort_by_key(|p| std::cmp::Reverse(p.total));
    out
}

/// Multi-line histogram of the top-`n` kernels by total time.
pub fn render_histogram(profiles: &[KernelProfile], n: usize) -> String {
    let total: Duration = profiles.iter().map(KernelProfile::gpu_or_wall).sum();
    let stamped = profiles.iter().filter(|p| p.gpu_start_ns.is_some()).count();
    let mut s = format!(
        "{} dispatches ({} GPU-stamped), total {:.3} ms\n{:>10}  {:>5}  {:>9}  {:>5}  name\n",
        profiles.len(),
        stamped,
        total.as_secs_f64() * 1e3,
        "total ms",
        "count",
        "mean µs",
        "%",
    );
    for a in aggregate_profiles(profiles).into_iter().take(n) {
        let pct = 100.0 * a.total.as_secs_f64() / total.as_secs_f64().max(f64::EPSILON);
        s.push_str(&format!(
            "{:>10.3}  {:>5}  {:>9.1}  {:>5.1}  {}\n",
            a.total.as_secs_f64() * 1e3,
            a.count,
            a.mean.as_secs_f64() * 1e6,
            pct,
            a.name,
        ));
    }
    s
}
