//! `native_topology_decline` gating: which semantic linked plans an AMD backend
//! refuses to replay natively.

use crate::amd::linked_plan::native_topology_decline;
use crate::device::NativeReplayDecline;
use crate::hcq::{CopyLeg, DeviceQueue, LaneSubmission, QueueKind, SemanticLinkedPlan, TopologyCommand};
use svod_dtype::DeviceSpec;
fn plan(lanes: Vec<LaneSubmission>) -> SemanticLinkedPlan {
    SemanticLinkedPlan::from_lane_submissions(lanes, |_| [0x1000, 0x1008]).unwrap()
}

#[test]
fn native_topology_rejects_staged_copy() {
    let semantic = plan(vec![LaneSubmission {
        lane: DeviceQueue { device: DeviceSpec::Amd { device_id: 0 }, queue: QueueKind::Copy(0) },
        waits: vec![],
        commands: vec![TopologyCommand { operation: 4, copy_leg: Some(CopyLeg::ToHost) }],
        signal_value: 1,
    }]);
    assert_eq!(native_topology_decline(&semantic, true), Some(NativeReplayDecline::StagedCopy { operation: 4 }));
}

#[test]
fn native_topology_rejects_copy_without_hardware_queue() {
    let semantic = plan(vec![LaneSubmission {
        lane: DeviceQueue { device: DeviceSpec::Amd { device_id: 0 }, queue: QueueKind::Copy(0) },
        waits: vec![],
        commands: vec![TopologyCommand { operation: 4, copy_leg: None }],
        signal_value: 1,
    }]);
    assert_eq!(native_topology_decline(&semantic, false), Some(NativeReplayDecline::BackendUnsupported));
}

#[test]
fn native_topology_rejects_mixed_devices() {
    let semantic = plan(vec![
        LaneSubmission {
            lane: DeviceQueue { device: DeviceSpec::Amd { device_id: 0 }, queue: QueueKind::Compute(0) },
            waits: vec![],
            commands: vec![TopologyCommand { operation: 0, copy_leg: None }],
            signal_value: 1,
        },
        LaneSubmission {
            lane: DeviceQueue { device: DeviceSpec::Amd { device_id: 1 }, queue: QueueKind::Compute(0) },
            waits: vec![],
            commands: vec![TopologyCommand { operation: 1, copy_leg: None }],
            signal_value: 1,
        },
    ]);
    assert!(matches!(
        native_topology_decline(&semantic, true),
        Some(NativeReplayDecline::MixedComputeDevices {
            expected: DeviceSpec::Amd { device_id: 0 },
            actual: DeviceSpec::Amd { device_id: 1 },
        })
    ));
}
