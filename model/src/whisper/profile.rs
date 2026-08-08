//! Internal Whisper copy profiling.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use snafu::ResultExt;
use svod_device::Buffer;
use svod_runtime::{KernelProfile, StageProfile};

use super::error::{DeviceSnafu, Result};

#[derive(Debug, Default)]
pub(crate) struct GraphProfile {
    pub(crate) wall: Duration,
    pub(crate) executions: usize,
    pub(crate) kernels: Vec<KernelProfile>,
}

impl GraphProfile {
    pub(crate) fn record(&mut self, wall: Duration, kernels: Vec<KernelProfile>) {
        self.wall = self.wall.saturating_add(wall);
        self.executions = self.executions.saturating_add(1);
        self.kernels.extend(kernels);
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.wall = self.wall.saturating_add(other.wall);
        self.executions = self.executions.saturating_add(other.executions);
        self.kernels.extend(other.kernels);
    }

    pub(crate) fn stage(self, name: &str) -> StageProfile {
        let kernel_dispatches = self.kernels.len();
        let average_wall_ms =
            if self.executions == 0 { 0.0 } else { self.wall.as_secs_f64() * 1e3 / self.executions as f64 };
        let mut stage = StageProfile::gpu(name, self.wall, self.kernels);
        stage.meta.insert("executions".into(), self.executions.to_string());
        stage.meta.insert("kernel_dispatches".into(), kernel_dispatches.to_string());
        stage.meta.insert("accumulated_wall_ms".into(), format!("{:.3}", self.wall.as_secs_f64() * 1e3));
        stage.meta.insert("average_execution_wall_ms".into(), format!("{average_wall_ms:.3}"));
        stage.meta.insert(
            "timing_semantics".into(),
            "accumulated host wall per execution from profiled submission through explicit output synchronization"
                .into(),
        );
        stage
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct CopyStats {
    pub(crate) ops: usize,
    pub(crate) bytes: usize,
    pub(crate) wall: Duration,
}

impl CopyStats {
    pub(crate) fn add(&mut self, ops: usize, bytes: usize, wall: Duration) {
        self.ops = self.ops.saturating_add(ops);
        self.bytes = self.bytes.saturating_add(bytes);
        self.wall = self.wall.saturating_add(wall);
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.add(other.ops, other.bytes, other.wall);
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct CopyCategory {
    total: CopyStats,
    breakdown: BTreeMap<&'static str, CopyStats>,
}

impl CopyCategory {
    fn record(&mut self, name: &'static str, ops: usize, bytes: usize, wall: Duration) {
        self.total.add(ops, bytes, wall);
        self.breakdown.entry(name).or_default().add(ops, bytes, wall);
    }

    fn merge(&mut self, other: Self) {
        self.total.merge(other.total);
        for (name, stats) in other.breakdown {
            self.breakdown.entry(name).or_default().merge(stats);
        }
    }

    fn stage(&self, name: &str) -> Option<StageProfile> {
        (self.total.bytes != 0).then(|| {
            let mut stage = StageProfile::host(name, self.total.wall);
            stage.meta.insert("ops".into(), self.total.ops.to_string());
            stage.meta.insert("bytes".into(), self.total.bytes.to_string());
            let gbps = if self.total.wall.is_zero() {
                0.0
            } else {
                self.total.bytes as f64 / self.total.wall.as_secs_f64() / 1e9
            };
            stage.meta.insert("effective_gbps".into(), format!("{gbps:.3}"));
            let semantics = match name {
                "copy_d2d" => {
                    "device synchronized immediately before and after each transfer group; host wall, not hardware DMA timestamps"
                }
                "copy_h2d" => {
                    "prior device work fenced before host-visible writes; synchronized host wall, not hardware DMA timestamps"
                }
                _ => {
                    "producer work fenced before host-visible reads; synchronized host copy wall, not hardware DMA timestamps"
                }
            };
            stage.meta.insert("timing_semantics".into(), semantics.into());
            for (breakdown, stats) in &self.breakdown {
                stage.meta.insert(format!("{breakdown}_ops"), stats.ops.to_string());
                stage.meta.insert(format!("{breakdown}_bytes"), stats.bytes.to_string());
                stage.meta.insert(format!("{breakdown}_wall_ms"), format!("{:.3}", stats.wall.as_secs_f64() * 1e3));
                let gbps = if stats.wall.is_zero() {
                    0.0
                } else {
                    stats.bytes as f64 / stats.wall.as_secs_f64() / 1e9
                };
                stage.meta.insert(format!("{breakdown}_effective_gbps"), format!("{gbps:.3}"));
            }
            stage
        })
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct CopyProfile {
    h2d: CopyCategory,
    d2d: CopyCategory,
    d2h: CopyCategory,
}

impl CopyProfile {
    pub(crate) fn h2d(&mut self, name: &'static str, ops: usize, bytes: usize, wall: Duration) {
        self.h2d.record(name, ops, bytes, wall);
    }

    pub(crate) fn d2d(&mut self, name: &'static str, ops: usize, bytes: usize, wall: Duration) {
        self.d2d.record(name, ops, bytes, wall);
    }

    pub(crate) fn d2h(&mut self, name: &'static str, ops: usize, bytes: usize, wall: Duration) {
        self.d2h.record(name, ops, bytes, wall);
    }

    pub(crate) fn merge(&mut self, other: Self) {
        self.h2d.merge(other.h2d);
        self.d2d.merge(other.d2d);
        self.d2h.merge(other.d2h);
    }

    pub(crate) fn stages(&self) -> impl Iterator<Item = StageProfile> + '_ {
        [self.h2d.stage("copy_h2d"), self.d2d.stage("copy_d2d"), self.d2h.stage("copy_d2h")].into_iter().flatten()
    }
}

/// Fence before starting so prior graph work is excluded, then fence after the
/// final transfer. This measures synchronized group wall, not SDMA timestamps.
pub(crate) fn timed_d2d<T>(enabled: bool, fence: &Buffer, work: impl FnOnce() -> Result<T>) -> Result<(T, Duration)> {
    if !enabled {
        return work().map(|value| (value, Duration::ZERO));
    }
    fence.synchronize().context(DeviceSnafu)?;
    let started = Instant::now();
    let value = work()?;
    fence.synchronize().context(DeviceSnafu)?;
    Ok((value, started.elapsed()))
}

/// Drain prior device work before timing a host-visible read or write. Without
/// this fence, the mapping's implicit synchronization is incorrectly charged
/// to the copy instead of the graph that produced the buffer.
pub(crate) fn begin_host_copy(enabled: bool, buffer: &Buffer) -> Result<Option<Instant>> {
    if !enabled {
        return Ok(None);
    }
    buffer.synchronize().context(DeviceSnafu)?;
    Ok(Some(Instant::now()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copy_stats_merge_saturates() {
        let mut stats = CopyStats { ops: usize::MAX, bytes: usize::MAX - 1, wall: Duration::MAX };
        stats.merge(CopyStats { ops: 1, bytes: 4, wall: Duration::from_nanos(1) });
        assert_eq!(stats, CopyStats { ops: usize::MAX, bytes: usize::MAX, wall: Duration::MAX });
    }

    #[test]
    fn copy_stage_formats_totals_and_breakdown() {
        let mut profile = CopyProfile::default();
        profile.d2d("cache_append", 2, 1024, Duration::from_micros(2));
        let stage = profile.stages().next().unwrap();
        assert_eq!(stage.name, "copy_d2d");
        assert_eq!(stage.meta["ops"], "2");
        assert_eq!(stage.meta["bytes"], "1024");
        assert_eq!(stage.meta["effective_gbps"], "0.512");
        assert_eq!(stage.meta["cache_append_bytes"], "1024");
        assert!(stage.meta["timing_semantics"].contains("not hardware DMA timestamps"));
    }

    #[test]
    fn graph_profile_accumulates_execution_wall_and_metadata() {
        let mut profile = GraphProfile::default();
        profile.record(Duration::from_millis(2), Vec::new());
        let mut other = GraphProfile::default();
        other.record(Duration::from_millis(3), Vec::new());
        profile.merge(other);

        let stage = profile.stage("graph");
        assert_eq!(stage.wall, Duration::from_millis(5));
        assert_eq!(stage.meta["executions"], "2");
        assert_eq!(stage.meta["kernel_dispatches"], "0");
        assert_eq!(stage.meta["accumulated_wall_ms"], "5.000");
        assert_eq!(stage.meta["average_execution_wall_ms"], "2.500");
        assert!(stage.meta["timing_semantics"].contains("output synchronization"));
    }
}
