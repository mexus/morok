//! `native_topology_decline` gating: which semantic linked plans an AMD backend
//! refuses to replay natively.

use crate::amd::linked_plan::native_topology_decline;
use crate::device::NativeReplayDecline;
use crate::hcq::{CopyLeg, DeviceQueue, LaneSubmission, QueueKind, SemanticLinkedPlan, TopologyCommand};
use svod_dtype::DeviceSpec;

fn lane(device_id: usize, queue: QueueKind, operation: usize, copy_leg: Option<CopyLeg>) -> LaneSubmission {
    LaneSubmission {
        lane: DeviceQueue { device: DeviceSpec::Amd { device_id }, queue },
        waits: vec![],
        commands: vec![TopologyCommand { operation, copy_leg }],
        signal_value: 1,
    }
}

fn decline(lanes: Vec<LaneSubmission>, has_copy_queue: bool) -> Option<NativeReplayDecline> {
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, |_| [0x1000, 0x1008]).unwrap();
    native_topology_decline(&plan, has_copy_queue)
}

#[test]
fn native_topology_rejects_staged_copies_and_copies_without_a_hardware_queue() {
    let copy = |leg| vec![lane(0, QueueKind::Copy(0), 4, leg)];
    assert_eq!(decline(copy(Some(CopyLeg::ToHost)), true), Some(NativeReplayDecline::StagedCopy { operation: 4 }));
    assert_eq!(decline(copy(None), false), Some(NativeReplayDecline::BackendUnsupported));
    assert_eq!(decline(copy(None), true), None, "a direct copy on a device with an SDMA queue replays natively");
}

#[test]
fn native_topology_rejects_mixed_devices() {
    let lanes = vec![lane(0, QueueKind::Compute(0), 0, None), lane(1, QueueKind::Compute(0), 1, None)];
    assert!(matches!(
        decline(lanes, true),
        Some(NativeReplayDecline::MixedComputeDevices {
            expected: DeviceSpec::Amd { device_id: 0 },
            actual: DeviceSpec::Amd { device_id: 1 },
        })
    ));
}
