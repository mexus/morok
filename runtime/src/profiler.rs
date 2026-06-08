//! Per-kernel execution profiling.
//!
//! Provides structured timing data for kernel execution via
//! [`ExecutionPlan::execute_profiled()`](crate::ExecutionPlan::execute_profiled).

use std::collections::BTreeMap;
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
///         i, p.wall.as_secs_f64() * 1000.0,
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
    /// Host wall-clock around the dispatch submit. On async backends (GPU) this
    /// is mostly launch/submission overhead, NOT on-device execution time — for
    /// that use `gpu_start_ns`/`gpu_end_ns` (or [`Self::gpu_or_wall`]).
    pub wall: Duration,
    /// HW dispatch start/end on the GPU clock (ns), when the backend stamps
    /// dispatches ([`svod_device::DispatchTimestamps`]).
    pub gpu_start_ns: Option<u64>,
    pub gpu_end_ns: Option<u64>,
}

impl std::fmt::Debug for KernelProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KernelProfile")
            .field("kernel", &self.kernel.entry_point)
            .field("wall", &self.wall)
            .field("gpu_start_ns", &self.gpu_start_ns)
            .field("gpu_end_ns", &self.gpu_end_ns)
            .finish_non_exhaustive()
    }
}

impl KernelProfile {
    /// GPU execution time when stamped, else the host wall-clock fallback.
    pub fn gpu_or_wall(&self) -> Duration {
        match (self.gpu_start_ns, self.gpu_end_ns) {
            (Some(s), Some(e)) => Duration::from_nanos(e - s),
            _ => self.wall,
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

/// One stage of a profiled run: a named span owning the per-dispatch kernels of
/// ONE representative profiled execution (GPU-stamped when the backend supports
/// it), plus the host wall accumulated over the stage and an extensible metadata
/// bag. Host-only stages (no GPU work) carry empty `kernels`.
///
/// Model-agnostic by design: the stage identity is a free-form `name` (data, not
/// a typed enum), so any model populates the same shape and a generic UI /
/// histogram renders it uniformly. Stages are a flat, ordered list — any
/// grouping/hierarchy is a render-time concern, not stored here.
#[derive(Debug, Default)]
pub struct StageProfile {
    /// Stage identity, e.g. `"vad"`, `"mel"`, `"encoder"`, `"ctc_head"`.
    pub name: String,
    /// Host wall accumulated over the stage. On async GPUs this is mostly submit
    /// overhead; the on-device truth is in `kernels`.
    pub wall: Duration,
    /// Per-dispatch kernels of the profiled execution. Empty for host-only stages.
    pub kernels: Vec<KernelProfile>,
    /// Extensible per-stage metadata (rtf, chunk index, …). Keeps the format
    /// stable across models without schema churn; consumed as-is by the UI.
    pub meta: BTreeMap<String, String>,
}

impl StageProfile {
    /// A host-only stage (no GPU kernels).
    pub fn host(name: impl Into<String>, wall: Duration) -> Self {
        Self { name: name.into(), wall, kernels: Vec::new(), meta: BTreeMap::new() }
    }

    /// A GPU stage carrying one profiled execution's per-dispatch kernels.
    pub fn gpu(name: impl Into<String>, wall: Duration, kernels: Vec<KernelProfile>) -> Self {
        Self { name: name.into(), wall, kernels, meta: BTreeMap::new() }
    }

    /// Sum of GPU (or wall-fallback) time across the profiled execution.
    pub fn gpu_total(&self) -> Duration {
        self.kernels.iter().map(KernelProfile::gpu_or_wall).sum()
    }

    /// Top kernels by total time, aggregated by entry point.
    pub fn top(&self, n: usize) -> Vec<KernelAggregate> {
        let mut aggs = aggregate_profiles(&self.kernels);
        aggs.truncate(n);
        aggs
    }
}

/// A model-agnostic profile of one inference run: an ordered, flat list of named
/// stages. Any model emits this same shape, so a generic UI / the `Display`
/// histogram renders an arbitrary model's profile uniformly.
#[derive(Debug, Default)]
pub struct RunProfile {
    pub stages: Vec<StageProfile>,
}

impl RunProfile {
    /// Append a stage.
    pub fn push(&mut self, stage: StageProfile) {
        self.stages.push(stage);
    }

    /// First stage with the given name, if any.
    pub fn stage(&self, name: &str) -> Option<&StageProfile> {
        self.stages.iter().find(|s| s.name == name)
    }
}

impl std::fmt::Display for RunProfile {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for s in &self.stages {
            if s.kernels.is_empty() {
                writeln!(f, "{}: wall {:.1} ms", s.name, s.wall.as_secs_f64() * 1e3)?;
            } else {
                writeln!(
                    f,
                    "{}: wall {:.1} ms, profiled exec GPU {:.3} ms\n{}",
                    s.name,
                    s.wall.as_secs_f64() * 1e3,
                    s.gpu_total().as_secs_f64() * 1e3,
                    render_histogram(&s.kernels, 20),
                )?;
            }
        }
        Ok(())
    }
}
