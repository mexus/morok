//! Unit tests for the profiler data model, table rendering (GPU-free), and
//! [`RunProfile::merge`] accumulation semantics.

use std::time::Duration;

use svod_device::PmcCounter;
use svod_device::hcq::{
    DeviceQueue, QueueKind, QueueMergeLimits, SemanticLinkedPlan, TopologyOperation, TopologyOperationKind,
    TopologyResource, schedule_device_lanes,
};
use svod_dtype::DeviceSpec;

use crate::profiler::{
    OperationTiming, PmcSelection, ProfileOptions, RunProfile, StageProfile, analyze_execution_lanes, parse_pmc,
};

#[test]
fn pmc_counter_token_roundtrip() {
    for c in [PmcCounter::SqBusyCycles, PmcCounter::SqWaves, PmcCounter::SqInstsValu] {
        assert_eq!(PmcCounter::from_token(c.token()), Some(c), "token roundtrip for {c:?}");
    }
    assert_eq!(PmcCounter::from_token("nope"), None);
    assert_eq!(PmcCounter::from_token("BUSY"), Some(PmcCounter::SqBusyCycles), "case-insensitive alias");
}

#[test]
fn pmc_selection_resolution() {
    assert!(!PmcSelection::None.is_enabled());
    assert!(PmcSelection::None.counters().is_empty());
    assert!(PmcSelection::Default.is_enabled());
    assert_eq!(PmcSelection::Default.counters().len(), 3);
    let custom = PmcSelection::Custom(vec![PmcCounter::SqInstsValu]);
    assert_eq!(custom.counters(), vec![PmcCounter::SqInstsValu]);
}

#[test]
fn parse_pmc_values() {
    assert_eq!(parse_pmc(""), PmcSelection::None);
    assert_eq!(parse_pmc("0"), PmcSelection::None);
    assert_eq!(parse_pmc("1"), PmcSelection::Default);
    assert_eq!(parse_pmc("valu,waves"), PmcSelection::Custom(vec![PmcCounter::SqInstsValu, PmcCounter::SqWaves]));
    // All-unknown tokens fall back to the default set rather than an empty selection.
    assert_eq!(parse_pmc("bogus"), PmcSelection::Default);
}

#[test]
fn profile_options_default() {
    let o = ProfileOptions::default();
    assert_eq!(o.iters, 1);
    assert!(o.static_analysis);
    assert_eq!(o.counters, PmcSelection::None);
}

#[test]
fn render_table_empty_and_host_only() {
    // Empty report renders nothing.
    assert_eq!(RunProfile::default().render_table(), "");

    // A host-only stage (no kernels) renders a single wall line, no metric table.
    let mut rp = RunProfile::default();
    rp.push(StageProfile::host("mel", Duration::from_millis(3)));
    let out = rp.render_table();
    assert!(out.contains("mel"), "host stage name present: {out:?}");
    assert!(out.contains("host"), "host stage tagged host: {out:?}");
    assert!(!out.contains("GFLOP/s"), "no metric columns for host-only: {out:?}");
}

#[test]
fn merge_accumulates_same_named_stages_and_appends_new() {
    let mut a = RunProfile::default();
    a.push(StageProfile::host("mel", Duration::from_millis(2)));
    let mut enc = StageProfile::host("encoder", Duration::from_millis(10));
    enc.meta.insert("rtf".into(), "0.02".into());
    a.push(enc);

    let mut b = RunProfile::default();
    let mut enc2 = StageProfile::host("encoder", Duration::from_millis(5)); // same name → sum wall + meta
    enc2.meta.insert("chunks".into(), "4".into());
    b.push(enc2);
    b.push(StageProfile::host("decode", Duration::from_millis(3))); // new name → appended

    a.merge(b);

    let names: Vec<&str> = a.stages.iter().map(|s| s.name.as_str()).collect();
    assert_eq!(names, ["mel", "encoder", "decode"], "matched stays in place, new appends");
    assert_eq!(a.stage("mel").unwrap().wall, Duration::from_millis(2), "untouched");

    let enc = a.stage("encoder").unwrap();
    assert_eq!(enc.wall, Duration::from_millis(15), "10 + 5 summed");
    assert_eq!(enc.meta.get("rtf").map(String::as_str), Some("0.02"), "kept");
    assert_eq!(enc.meta.get("chunks").map(String::as_str), Some("4"), "folded in");
}

fn resource(id: u64) -> TopologyResource {
    TopologyResource { id, owner: DeviceSpec::Cpu, start: 0, end: 64 }
}

fn topology_op(operation: usize, queue: QueueKind, reads: &[u64], writes: &[u64]) -> TopologyOperation {
    TopologyOperation {
        operation,
        lane: DeviceQueue { device: DeviceSpec::Cpu, queue },
        reads: reads.iter().copied().map(resource).collect(),
        writes: writes.iter().copied().map(resource).collect(),
        kind: TopologyOperationKind::Execute,
    }
}

fn semantic_plan(operations: &[TopologyOperation]) -> SemanticLinkedPlan {
    let lanes = schedule_device_lanes(operations, QueueMergeLimits::NO_MERGE, |executor, owner| executor == owner);
    let mut signal = 0x1000;
    SemanticLinkedPlan::from_lane_submissions(lanes, |_| {
        signal += 16;
        [signal - 8, signal]
    })
    .unwrap()
}

fn timing(operation: usize, millis: u64) -> OperationTiming {
    OperationTiming { operation, copy_leg: None, duration: Duration::from_millis(millis) }
}

#[test]
fn host_fork_join_lane_metrics_measure_overlap_and_join_wait() {
    // Independent compute/copy forks at t=0. The compute join waits three ms
    // after its first command for the longer copy lane.
    let plan = semantic_plan(&[
        topology_op(0, QueueKind::Compute(0), &[], &[1]),
        topology_op(1, QueueKind::Copy(0), &[], &[2]),
        topology_op(2, QueueKind::Compute(0), &[1, 2], &[3]),
    ]);
    let metrics = analyze_execution_lanes(&plan, &[timing(0, 5), timing(1, 8), timing(2, 2)]);

    assert_eq!(metrics.makespan, Duration::from_millis(10));
    assert_eq!(metrics.busy, Duration::from_millis(15));
    assert_eq!(metrics.wait, Duration::from_millis(3));
    assert_eq!(metrics.overlap, Duration::from_millis(5));
    let compute = metrics.lanes.iter().find(|lane| lane.lane.queue == QueueKind::Compute(0)).unwrap();
    assert_eq!(compute.makespan, Duration::from_millis(10));
    assert_eq!(compute.busy, Duration::from_millis(7));
    assert_eq!(compute.wait, Duration::from_millis(3));
    assert_eq!(compute.overlap, Duration::from_millis(5));
}

#[test]
fn alternating_copy_compute_metrics_preserve_serial_hazards() {
    let plan = semantic_plan(&[
        topology_op(0, QueueKind::Compute(0), &[], &[1]),
        topology_op(1, QueueKind::Copy(0), &[1], &[2]),
        topology_op(2, QueueKind::Compute(0), &[2], &[3]),
    ]);
    let metrics = analyze_execution_lanes(&plan, &[timing(0, 2), timing(1, 3), timing(2, 4)]);

    assert_eq!(metrics.makespan, Duration::from_millis(9));
    assert_eq!(metrics.busy, Duration::from_millis(9));
    assert_eq!(metrics.wait, Duration::from_millis(5));
    assert_eq!(metrics.overlap, Duration::ZERO);
    assert!(metrics.lanes.iter().all(|lane| lane.overlap == Duration::ZERO));
}
