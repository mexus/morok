//! Whisper copy/graph profiling internals: saturating accumulation and the
//! stage metadata the profile renders.

use crate::whisper::profile::{CopyProfile, CopyStats, GraphProfile};
use std::time::Duration;

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
