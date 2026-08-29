use crate::hcq::{
    ClikeKernargLayout, Command, CommandBufferCache, ComputeDispatch, CpuQueueExecutor, LinkPatchValues,
    LoweredCommandBuffer, NullHcq, PatchEncoding, PatchSite, PatchSource, PatchTable, PlaceholderKind,
    PlaceholderPacking, PlaceholderRequest, QueueKind, RuntimePatchValues, SemanticLinkedSubmission, Submission,
    SubmissionExecutionError, SystemField, SystemPatchValues,
};
use svod_dtype::DeviceSpec;

#[test]
fn clike_kernargs_pack_dense_abi_order() {
    use crate::device::{AbiParamDescriptor, AbiParamKind};
    use svod_dtype::{AddrSpace, DType};
    let layout = ClikeKernargLayout::from_abi(&[
        AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor {
            slot: 1,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor { slot: 2, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("low".into()) },
        AbiParamDescriptor { slot: 3, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("high".into()) },
    ]);
    let mut dst = [0xcc; 32];
    let written = layout.pack(&mut dst, &[0x1122_3344_5566_7788, 0x99aa_bbcc_ddee_ff00], &[-2, 0x1234_5678]).unwrap();
    assert_eq!(written, 24);
    assert_eq!(&dst[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&dst[8..16], &0x99aa_bbcc_ddee_ff00u64.to_le_bytes());
    assert_eq!(&dst[16..20], &(-2i32).to_le_bytes());
    assert_eq!(&dst[20..24], &0x1234_5678i32.to_le_bytes());
    assert_eq!(&dst[24..], &[0xcc; 8]);
}

#[test]
fn null_hcq_enforces_timeline_dependencies_and_order() {
    let signal = 0x1000;
    let mut null = NullHcq::default();
    let mut blocked = Submission::new(QueueKind::Compute(0));
    blocked.push(Command::Wait { signal_address: signal, value: 2 });
    assert!(null.submit(&blocked).is_err());

    null.set_signal(signal, 1);
    let dispatch = ComputeDispatch {
        workgroup_size: [64, 1, 1],
        grid_size: [1024, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0x2000,
        kernarg_address: 0x3000,
        completion_signal: 0x4000,
        barrier: true,
        amd_pm4: None,
    };
    let mut submit = Submission::new(QueueKind::Compute(0));
    submit
        .push(Command::Wait { signal_address: signal, value: 1 })
        .push(Command::MemoryBarrier)
        .push(Command::Compute(dispatch.clone()))
        .push(Command::Store { dst: signal, value: 2 });
    null.submit(&submit).unwrap();

    assert_eq!(null.trace().len(), 4);
    assert_eq!(null.trace()[0].1, Command::Wait { signal_address: signal, value: 1 });
    assert_eq!(null.trace()[2].1, Command::Compute(dispatch));
    null.submit(&blocked).unwrap();
}

#[test]
fn null_hcq_timestamps_use_deterministic_queue_clock() {
    let mut null = NullHcq::with_clock(1_000, 25);
    let mut compute = Submission::new(QueueKind::Compute(0));
    compute
        .push(Command::Timestamp { dst: 0x40 })
        .push(Command::Execute { operation: 0 })
        .push(Command::Timestamp { dst: 0x48 })
        .push(Command::Store { dst: 0x20, value: 1 });
    let mut copy = Submission::new(QueueKind::Copy(0));
    copy.push(Command::Wait { signal_address: 0x20, value: 1 })
        .push(Command::Timestamp { dst: 0x50 })
        .push(Command::Copy { dst: 0x2000, src: 0x1000, bytes: 16 })
        .push(Command::Timestamp { dst: 0x58 });

    null.submit(&compute).unwrap();
    null.submit(&copy).unwrap();
    assert_eq!(null.signal_value(0x40), Some(1_000));
    assert_eq!(null.signal_value(0x48), Some(1_025));
    assert_eq!(null.signal_value(0x50), Some(1_050));
    assert_eq!(null.signal_value(0x58), Some(1_075));
    assert_eq!(null.trace().len(), compute.commands.len() + copy.commands.len());
}

#[test]
fn cpu_hcq_mixed_compute_copy_waits_and_finalizers_are_ordered() {
    let source = [3u8, 1, 4, 1];
    let mut intermediate = [0u8; 4];
    let mut destination = [0u8; 4];
    let mut executor = CpuQueueExecutor::with_clock(100, 10);
    let mut compute = Submission::new(QueueKind::Compute(0));
    compute
        .push(Command::MemoryBarrier)
        .push(Command::Execute { operation: 7 })
        .push(Command::Copy { dst: intermediate.as_mut_ptr() as u64, src: source.as_ptr() as u64, bytes: source.len() })
        .push(Command::Timestamp { dst: 0x30 })
        .push(Command::Store { dst: 0x20, value: 1 });
    let mut copy = Submission::new(QueueKind::Copy(0));
    copy.push(Command::Wait { signal_address: 0x20, value: 1 })
        .push(Command::Copy {
            dst: destination.as_mut_ptr() as u64,
            src: intermediate.as_ptr() as u64,
            bytes: intermediate.len(),
        })
        .push(Command::Timestamp { dst: 0x38 })
        .push(Command::Store { dst: 0x28, value: 2 });

    let mut operations = Vec::new();
    unsafe {
        executor.submit(&compute, |operation| {
            operations.push(operation);
            Ok::<_, ()>(())
        })
    }
    .unwrap();
    unsafe { executor.submit(&copy, |_| Ok::<_, ()>(())) }.unwrap();

    assert_eq!(operations, [7]);
    assert_eq!(destination, source);
    assert_eq!(executor.signal_value(0x30), Some(100));
    assert_eq!(executor.signal_value(0x38), Some(110));
    assert_eq!(executor.signal_value(0x28), Some(2));
}

#[test]
fn cpu_and_null_compute_errors_do_not_publish_finalizers() {
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Execute { operation: 9 }).push(Command::Store { dst: 0x20, value: 1 });

    let mut cpu = CpuQueueExecutor::default();
    let error = unsafe { cpu.submit(&submission, |_| Err("CPU failure")) }.unwrap_err();
    assert!(matches!(error, SubmissionExecutionError::Execute("CPU failure")));
    assert_eq!(cpu.signal_value(0x20), None);

    let mut null = NullHcq::default();
    let error = null.submit_with(&submission, |_| Err("null failure")).unwrap_err();
    assert!(matches!(error, SubmissionExecutionError::Execute("null failure")));
    assert_eq!(null.signal_value(0x20), None);
}

#[test]
fn profile_disabled_submission_has_no_commands_or_metadata_overhead() {
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Execute { operation: 7 });
    assert!(!submission.profile_requested());
    assert_eq!(submission.commands, [Command::Execute { operation: 7 }]);

    submission.request_profile();
    assert!(submission.profile_requested());
    assert_eq!(submission.commands, [Command::Execute { operation: 7 }]);
}

#[test]
fn inserting_profile_commands_preserves_patch_ownership() {
    use crate::hcq::{CommandField, PatchSource};

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Wait { signal_address: 0, value: 1 });
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.insert(0, Command::Timestamp { dst: 0x80 });
    assert_eq!(submission.patches()[0].command, 1);
    assert_eq!(submission.commands[1], Command::Wait { signal_address: 0, value: 1 });
}

#[test]
fn semantic_link_retains_structure_and_repatches_runtime_and_system_fields() {
    use crate::hcq::CommandField;

    let mut submission = Submission::new(QueueKind::Copy(0));
    submission.push(Command::Wait { signal_address: 0, value: 0 });
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.push(Command::Copy { dst: 0, src: 0, bytes: 4 });
    submission.bind(1, CommandField::CopyDst, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(1, CommandField::CopySrc, PatchSource::RuntimeBuffer(1)).unwrap();

    let linked = SemanticLinkedSubmission::new(submission);
    let static_ptr = linked.static_submission().commands.as_ptr();
    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::TimelineSignal(0), 0x1000);
    system.0.insert(SystemField::TimelineValue(0), 7);
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x2000, 0x3000], vars: vec![] }, &system).unwrap();
    assert_eq!(replay.submission().commands[0], Command::Wait { signal_address: 0x1000, value: 7 });
    assert_eq!(replay.submission().commands[1], Command::Copy { dst: 0x2000, src: 0x3000, bytes: 4 });

    system.0.insert(SystemField::TimelineValue(0), 8);
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x4000, 0x5000], vars: vec![] }, &system).unwrap();
    assert_eq!(replay.submission().commands[0], Command::Wait { signal_address: 0x1000, value: 8 });
    assert_eq!(replay.submission().commands[1], Command::Copy { dst: 0x4000, src: 0x5000, bytes: 4 });
    assert_eq!(linked.static_submission().commands.as_ptr(), static_ptr);
    assert_eq!(linked.static_submission().commands[0], Command::Wait { signal_address: 0, value: 0 });
}

#[test]
fn canonical_program_kernargs_interleave_storage_and_scalars_by_slot() {
    let mut info = svod_ir::ProgramInfo::default();
    info.globals = vec![0, 2];
    let slotted = |name: &str, slot: usize| {
        let var = svod_ir::UOp::variable(name.into(), 0, 16, svod_dtype::DType::Int32);
        let svod_ir::Op::Param { shape, arg } = var.op() else { panic!("variable PARAM") };
        let mut arg = arg.clone();
        arg.slot = slot;
        svod_ir::UOp::new(svod_ir::Op::Param { shape: shape.clone(), arg }, svod_dtype::DType::Int32)
    };
    info.vars = vec![slotted("low", 1), slotted("high", 3)];
    let mut dst = [0xcc; 32];
    let abi = vec![
        crate::device::AbiParamDescriptor {
            slot: 0,
            kind: crate::device::AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        },
        crate::device::AbiParamDescriptor::from_param(&info.vars[0]).unwrap(),
        crate::device::AbiParamDescriptor {
            slot: 2,
            kind: crate::device::AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        },
        crate::device::AbiParamDescriptor::from_param(&info.vars[1]).unwrap(),
    ];
    let written = ClikeKernargLayout::pack_program(&info, &abi, &mut dst, &[0x1000, 0x3000], &[7, -3]).unwrap();
    assert_eq!(written, 28);
    assert_eq!(&dst[0..8], &0x1000u64.to_le_bytes());
    assert_eq!(&dst[8..12], &7i32.to_le_bytes());
    assert_eq!(&dst[12..16], &[0; 4]);
    assert_eq!(&dst[16..24], &0x3000u64.to_le_bytes());
    assert_eq!(&dst[24..28], &(-3i32).to_le_bytes());
}

#[test]
fn sparse_program_slots_use_compact_buffer_ordinals() {
    let mut info = svod_ir::ProgramInfo::default();
    info.globals = vec![0, 5];
    let abi = info
        .globals
        .iter()
        .map(|&slot| crate::device::AbiParamDescriptor {
            slot,
            kind: crate::device::AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        })
        .collect::<Vec<_>>();
    let mut dst = [0u8; 16];

    ClikeKernargLayout::pack_program(&info, &abi, &mut dst, &[0x1111, 0x5555], &[]).unwrap();
    assert_eq!(&dst[..8], &0x1111u64.to_le_bytes());
    assert_eq!(&dst[8..], &0x5555u64.to_le_bytes());
    let err = ClikeKernargLayout::pack_program(&info, &abi, &mut dst, &[0x1111], &[])
        .expect_err("compact buffer arity must be exact");
    assert!(matches!(err, crate::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

#[test]
fn neutral_patch_tables_cache_link_bytes_and_scatter_replays() {
    let lowered = LoweredCommandBuffer {
        bytes: vec![0xaa; 24],
        patches: PatchTable::from_sites(vec![
            PatchSite { byte_offset: 0, encoding: PatchEncoding::U64, source: PatchSource::LinkAddress(0), addend: 0 },
            PatchSite {
                byte_offset: 8,
                encoding: PatchEncoding::U64,
                source: PatchSource::RuntimeBuffer(0),
                addend: 16,
            },
            PatchSite { byte_offset: 16, encoding: PatchEncoding::U32, source: PatchSource::RuntimeVar(0), addend: 0 },
            PatchSite {
                byte_offset: 20,
                encoding: PatchEncoding::U32,
                source: PatchSource::System(SystemField::TimelineValue(0)),
                addend: 0,
            },
        ]),
    };
    assert_eq!(lowered.patches.link.len(), 1);
    assert_eq!(lowered.patches.runtime.len(), 2);
    assert_eq!(lowered.patches.system.len(), 1);

    let mut cache = CommandBufferCache::default();
    let linked = cache.link(&lowered, &LinkPatchValues(vec![0x1122_3344_5566_7788])).unwrap();
    let linked_again = cache.link(&lowered, &LinkPatchValues(vec![0x1122_3344_5566_7788])).unwrap();
    assert!(std::sync::Arc::ptr_eq(&linked, &linked_again));
    let immutable = linked.static_bytes().to_vec();

    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::TimelineValue(0), 3);
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x2000], vars: vec![7] }, &system).unwrap();
    assert_eq!(&replay.bytes()[8..16], &0x2010u64.to_le_bytes());

    system.0.insert(SystemField::TimelineValue(0), 4);
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x9000], vars: vec![-2] }, &system).unwrap();
    assert_eq!(&replay.bytes()[8..16], &0x9010u64.to_le_bytes());
    assert_eq!(&replay.bytes()[16..20], &(-2i32).to_le_bytes());
    assert_eq!(linked.static_bytes(), immutable);
}

#[test]
fn hcq_placeholder_packing_aliases_scratch_and_aligns_kernargs() {
    let packing = PlaceholderPacking::pack(&[
        PlaceholderRequest { kind: PlaceholderKind::Scratch, bytes: 64 },
        PlaceholderRequest { kind: PlaceholderKind::Kernargs, bytes: 20 },
        PlaceholderRequest { kind: PlaceholderKind::Scratch, bytes: 256 },
        PlaceholderRequest { kind: PlaceholderKind::Kernargs, bytes: 12 },
    ]);
    assert_eq!(packing.offsets, [0, 0, 0, 32]);
    assert_eq!(packing.scratch_bytes, 256);
    assert_eq!(packing.kernarg_bytes, 44);
}

/// One packing rule for every kernarg site. Tinygrad bump-allocates its kernarg
/// blocks at alignment 8 (`runtime/support/hcq.py:352`); 16 covers the largest
/// AMDHSA member alignment without the 128-byte inflation morok used to apply.
#[test_case::test_case(&[8, 12, 4], 16 => (vec![0, 16, 32], 36); "records are aligned, the total is not padded")]
#[test_case::test_case(&[16, 16], 16 => (vec![0, 16], 32); "already-aligned records pack tight")]
#[test_case::test_case(&[], 16 => (vec![], 0); "no records")]
fn kernarg_offsets_pack_records_at_one_alignment(sizes: &[usize], align: usize) -> (Vec<usize>, usize) {
    crate::hcq::kernarg_offsets(sizes.iter().copied(), align)
}

#[test]
fn timeline_rollover_switches_signal_and_requests_one_reset() {
    let mut timeline = crate::hcq::EpochTimeline::with_next([0x1000, 0x2000], crate::hcq::TIMELINE_ROLLOVER + 1);
    let point = timeline.reserve();
    assert_eq!(point, crate::hcq::TimelinePoint { signal_address: 0x2000, value: 1 });
    assert_eq!(timeline.take_reset(), Some(0x2000));
    assert_eq!(timeline.take_reset(), None, "rollover reset ownership is consumed once");

    let mut timelines = crate::hcq::SubmissionTimelines::new([0x10, 0x18], [0x20, 0x28], [0x30, 0x38]);
    assert!(timelines.take_resets().is_empty());
}

#[test]
fn submission_finalizer_helpers_own_wait_and_signal_commands() {
    let producer = crate::hcq::TimelinePoint { signal_address: 0x20, value: 7 };
    let completion = crate::hcq::TimelinePoint { signal_address: 0x10, value: 3 };
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.wait_for(producer).push(Command::MemoryBarrier).signal(completion);
    assert_eq!(
        submission.commands,
        [
            Command::Wait { signal_address: 0x20, value: 7 },
            Command::MemoryBarrier,
            Command::Store { dst: 0x10, value: 3 },
        ]
    );
}

fn gpu(id: usize) -> DeviceSpec {
    DeviceSpec::Amd { device_id: id }
}

fn resource(id: u64, owner: DeviceSpec) -> crate::hcq::TopologyResource {
    resource_range(id, owner, 0, 16)
}

fn resource_range(id: u64, owner: DeviceSpec, start: usize, end: usize) -> crate::hcq::TopologyResource {
    crate::hcq::TopologyResource { id, owner, start, end }
}

fn lane_signals(lane: &crate::hcq::DeviceQueue) -> [u64; 2] {
    let device = match &lane.device {
        DeviceSpec::Amd { device_id } => *device_id as u64 + 1,
        DeviceSpec::Cpu => 0,
        _ => 0x100,
    };
    let queue = match lane.queue {
        QueueKind::Compute(number) => number as u64 * 4,
        QueueKind::Copy(number) => number as u64 * 4 + 2,
    };
    let first = 0x1000 + device * 0x100 + queue * 0x10;
    [first, first + 8]
}

fn copy_op(
    operation: usize,
    src: crate::hcq::TopologyResource,
    dst: crate::hcq::TopologyResource,
) -> crate::hcq::TopologyOperation {
    crate::hcq::TopologyOperation {
        operation,
        lane: crate::hcq::DeviceQueue { device: dst.owner.clone(), queue: QueueKind::Copy(0) },
        reads: vec![src.clone()],
        writes: vec![dst.clone()],
        kind: crate::hcq::TopologyOperationKind::Copy { src, dst, bytes: 16 },
    }
}

#[test]
fn direct_copy_uses_declared_executor_access() {
    use crate::hcq::{CopyLeg, QueueMergeLimits, SemanticLinkedPlan, schedule_device_lanes};
    let mut op = copy_op(0, resource(1, gpu(0)), resource(2, gpu(1)));
    op.lane.device = gpu(2);
    let scheduled = schedule_device_lanes(&[op], QueueMergeLimits::UNLIMITED, |executor, owner| {
        executor == &gpu(2) && matches!(owner, DeviceSpec::Amd { .. })
    });
    assert_eq!(scheduled.len(), 1);
    assert_eq!(scheduled[0].lane.device, gpu(2));
    assert_eq!(scheduled[0].commands[0].copy_leg, Some(CopyLeg::Direct));

    let plan = SemanticLinkedPlan::from_lane_submissions(scheduled, lane_signals).unwrap();
    let mut null = NullHcq::default();
    let mut executed = Vec::new();
    plan.execute_null(&mut null, |lane, command| {
        executed.push((lane.clone(), command.clone()));
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(executed.len(), 1);
    assert_eq!(executed[0].1.copy_leg, Some(CopyLeg::Direct));
}

#[test]
fn two_device_inaccessible_copy_inserts_ordered_host_staging() {
    use crate::hcq::{CopyLeg, QueueMergeLimits, SemanticLinkedPlan, schedule_device_lanes};
    let op = copy_op(4, resource(10, gpu(0)), resource(11, gpu(1)));
    let scheduled = schedule_device_lanes(&[op], QueueMergeLimits::UNLIMITED, |executor, owner| executor == owner);
    assert_eq!(scheduled.len(), 2);
    assert_eq!(scheduled[0].lane.device, gpu(0));
    assert_eq!(scheduled[0].commands[0].copy_leg, Some(CopyLeg::ToHost));
    assert_eq!(scheduled[1].lane.device, gpu(1));
    assert_eq!(scheduled[1].commands[0].copy_leg, Some(CopyLeg::FromHost));
    assert_eq!(scheduled[1].waits[0].lane, scheduled[0].lane);
    assert_eq!(scheduled[1].waits[0].value, scheduled[0].signal_value);

    let plan = SemanticLinkedPlan::from_lane_submissions(scheduled, lane_signals).unwrap();
    let mut null = NullHcq::default();
    let mut legs = Vec::new();
    plan.execute_null(&mut null, |_, command| {
        legs.push(command.copy_leg.unwrap());
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(legs, [CopyLeg::ToHost, CopyLeg::FromHost]);
}

#[test]
fn compute_copy_cross_device_dependencies_are_lane_local_not_global() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, TopologyOperation, TopologyOperationKind, schedule_device_lanes};
    let produced = resource(20, gpu(0));
    let output = resource(21, gpu(1));
    let compute = TopologyOperation {
        operation: 0,
        lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
        reads: vec![],
        writes: vec![produced.clone()],
        kind: TopologyOperationKind::Execute,
    };
    let copy = copy_op(1, produced, output);
    let scheduled = schedule_device_lanes(&[compute, copy], QueueMergeLimits::UNLIMITED, |a, b| a == b);
    // compute, source->host, target<-host. Each transfer waits only its producer.
    assert_eq!(scheduled.len(), 3);
    assert_eq!(scheduled[1].waits, [crate::hcq::LaneWait { lane: scheduled[0].lane.clone(), value: 1 }]);
    assert_eq!(scheduled[2].waits, [crate::hcq::LaneWait { lane: scheduled[1].lane.clone(), value: 1 }]);
}

#[test]
fn queue_merge_limits_split_exactly_after_boundary() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, TopologyOperation, TopologyOperationKind, schedule_device_lanes};
    let ops = (0..5)
        .map(|operation| TopologyOperation {
            operation,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        })
        .collect::<Vec<_>>();
    let scheduled =
        schedule_device_lanes(&ops, QueueMergeLimits { max_submissions: 2, max_commands: 2 }, |a, b| a == b);
    assert_eq!(scheduled.iter().map(|s| s.commands.len()).collect::<Vec<_>>(), [2, 2, 1]);

    let unmerged = schedule_device_lanes(&ops, QueueMergeLimits::NO_MERGE, |a, b| a == b);
    assert_eq!(unmerged.iter().map(|s| s.commands.len()).collect::<Vec<_>>(), [1, 1, 1, 1, 1]);
}

#[test]
fn equal_queue_numbers_on_different_devices_keep_distinct_timelines() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, SemanticLinkedPlan, TopologyOperation, TopologyOperationKind};
    let operations = [
        TopologyOperation {
            operation: 0,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        },
        TopologyOperation {
            operation: 1,
            lane: DeviceQueue { device: gpu(1), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        },
    ];
    let lanes = crate::hcq::schedule_device_lanes(&operations, QueueMergeLimits::UNLIMITED, |a, b| a == b);
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    assert_ne!(plan.bindings()[0].point.signal_address, plan.bindings()[1].point.signal_address);

    let mut null = NullHcq::default();
    let mut devices = Vec::new();
    plan.execute_null(&mut null, |lane, _| {
        devices.push(lane.device.clone());
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(devices, [gpu(0), gpu(1)]);
}

#[test]
fn compute_copy_dependencies_execute_in_both_directions() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, SemanticLinkedPlan, TopologyOperation, TopologyOperationKind};
    let first = resource(30, gpu(0));
    let second = resource(31, gpu(0));
    let operations = [
        TopologyOperation {
            operation: 0,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![first.clone()],
            kind: TopologyOperationKind::Execute,
        },
        copy_op(1, first, second.clone()),
        TopologyOperation {
            operation: 2,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![second],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        },
    ];
    let lanes = crate::hcq::schedule_device_lanes(&operations, QueueMergeLimits::NO_MERGE, |a, b| a == b);
    assert_eq!(lanes[1].waits[0].lane.queue, QueueKind::Compute(0));
    assert_eq!(lanes[2].waits[0].lane.queue, QueueKind::Copy(0));

    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let mut null = NullHcq::default();
    let mut order = Vec::new();
    plan.execute_null(&mut null, |_, command| {
        order.push(command.operation);
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(order, [0, 1, 2]);
}

#[test]
fn raw_war_and_waw_hazards_wait_for_the_producer_lane() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, TopologyOperation, TopologyOperationKind};
    let compute = DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) };
    let copy = DeviceQueue { device: gpu(0), queue: QueueKind::Copy(0) };
    let cases = [
        (vec![], vec![resource(40, gpu(0))], vec![resource(40, gpu(0))], vec![]),
        (vec![resource(41, gpu(0))], vec![], vec![], vec![resource(41, gpu(0))]),
        (vec![], vec![resource(42, gpu(0))], vec![], vec![resource(42, gpu(0))]),
    ];
    for (producer_reads, producer_writes, consumer_reads, consumer_writes) in cases {
        let operations = [
            TopologyOperation {
                operation: 0,
                lane: compute.clone(),
                reads: producer_reads,
                writes: producer_writes,
                kind: TopologyOperationKind::Execute,
            },
            TopologyOperation {
                operation: 1,
                lane: copy.clone(),
                reads: consumer_reads,
                writes: consumer_writes,
                kind: TopologyOperationKind::Execute,
            },
        ];
        let lanes = crate::hcq::schedule_device_lanes(&operations, QueueMergeLimits::NO_MERGE, |a, b| a == b);
        assert_eq!(lanes[1].waits, [crate::hcq::LaneWait { lane: compute.clone(), value: 1 }]);
        let plan = crate::hcq::SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
        let mut null = NullHcq::default();
        let mut order = Vec::new();
        plan.execute_null(&mut null, |_, command| {
            order.push(command.operation);
            Ok::<_, ()>(())
        })
        .unwrap();
        assert_eq!(order, [0, 1]);
    }
}

#[test]
fn topology_hazards_require_overlapping_byte_ranges() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, TopologyOperation, TopologyOperationKind};
    let producer = |write| TopologyOperation {
        operation: 0,
        lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
        reads: vec![],
        writes: vec![write],
        kind: TopologyOperationKind::Execute,
    };
    let consumer = |operation, read| TopologyOperation {
        operation,
        lane: DeviceQueue { device: gpu(0), queue: QueueKind::Copy(0) },
        reads: vec![read],
        writes: vec![],
        kind: TopologyOperationKind::Execute,
    };

    let disjoint = crate::hcq::schedule_device_lanes(
        &[producer(resource_range(50, gpu(0), 0, 8)), consumer(1, resource_range(50, gpu(0), 8, 16))],
        QueueMergeLimits::NO_MERGE,
        |a, b| a == b,
    );
    assert!(disjoint[1].waits.is_empty());
    let overlap = crate::hcq::schedule_device_lanes(
        &[producer(resource_range(50, gpu(0), 0, 9)), consumer(1, resource_range(50, gpu(0), 8, 16))],
        QueueMergeLimits::NO_MERGE,
        |a, b| a == b,
    );
    assert_eq!(overlap[1].waits[0].value, 1);
    let plan = crate::hcq::SemanticLinkedPlan::from_lane_submissions(overlap, lane_signals).unwrap();
    let mut null = NullHcq::default();
    let mut order = Vec::new();
    plan.execute_null(&mut null, |_, command| {
        order.push(command.operation);
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(order, [0, 1]);
}

#[test]
fn merged_waits_target_and_execute_published_boundaries() {
    use crate::hcq::{DeviceQueue, QueueMergeLimits, SemanticLinkedPlan, TopologyOperation, TopologyOperationKind};
    let first = resource(60, gpu(0));
    let operations = [
        TopologyOperation {
            operation: 0,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![first.clone()],
            kind: TopologyOperationKind::Execute,
        },
        TopologyOperation {
            operation: 1,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        },
        TopologyOperation {
            operation: 2,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Copy(0) },
            reads: vec![first],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        },
    ];
    let lanes = crate::hcq::schedule_device_lanes(&operations, QueueMergeLimits::UNLIMITED, |a, b| a == b);
    assert_eq!(lanes.len(), 2);
    assert_eq!(lanes[0].signal_value, 2);
    assert_eq!(lanes[1].waits[0].value, 2);

    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let mut null = NullHcq::default();
    let mut null_order = Vec::new();
    plan.execute_null(&mut null, |_, command| {
        null_order.push(command.operation);
        Ok::<_, ()>(())
    })
    .unwrap();
    assert_eq!(null_order, [0, 1, 2]);

    let mut cpu = CpuQueueExecutor::default();
    let mut order = Vec::new();
    unsafe {
        plan.execute_cpu(&mut cpu, |_, command| {
            order.push(command.operation);
            Ok::<_, ()>(())
        })
    }
    .unwrap();
    assert_eq!(order, [0, 1, 2]);
}

#[test]
fn semantic_linked_submission_retains_concrete_lane_identity() {
    let lane = crate::hcq::DeviceQueue { device: gpu(1), queue: QueueKind::Compute(0) };
    let linked = SemanticLinkedSubmission::new_for_lane(lane.clone(), Submission::new(QueueKind::Compute(0))).unwrap();
    assert_eq!(linked.lane(), &lane);
    assert_eq!(linked.replay_buffer().lane(), &lane);
}

#[test]
fn native_adapter_preserves_single_device_submission_shape() {
    use crate::hcq::{
        Command, DeviceQueue, QueueMergeLimits, SemanticLinkedPlan, TopologyOperation, TopologyOperationKind,
    };

    let value = resource(70, gpu(0));
    let operations = [
        TopologyOperation {
            operation: 0,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Compute(0) },
            reads: vec![],
            writes: vec![value.clone()],
            kind: TopologyOperationKind::Execute,
        },
        TopologyOperation {
            operation: 1,
            lane: DeviceQueue { device: gpu(0), queue: QueueKind::Copy(0) },
            reads: vec![value],
            writes: vec![],
            kind: TopologyOperationKind::Execute,
        },
    ];
    let lanes = crate::hcq::schedule_device_lanes(&operations, QueueMergeLimits::NO_MERGE, |a, b| a == b);
    let plan = SemanticLinkedPlan::from_lane_submissions(lanes, lane_signals).unwrap();
    let native = plan.native_submissions().unwrap();

    assert_eq!(native.len(), 3);
    assert_eq!(native[0].lane().queue, QueueKind::Compute(0));
    assert!(matches!(
        native[0].static_submission().commands.as_slice(),
        [Command::MemoryBarrier, Command::Wait { .. }, Command::Execute { operation: 0 }, Command::Store { .. }]
    ));
    assert!(matches!(
        native[1].static_submission().commands.as_slice(),
        [
            Command::MemoryBarrier,
            Command::Wait { .. },
            Command::Wait { .. },
            Command::Execute { operation: 1 },
            Command::Store { .. }
        ]
    ));
    assert!(matches!(native[2].static_submission().commands.as_slice(), [Command::Wait { .. }, Command::Store { .. }]));
}

#[test]
fn linked_buffer_cache_is_scoped_by_context_and_device() {
    let lowered = LoweredCommandBuffer { bytes: vec![0; 8], patches: PatchTable::default() };
    let values = LinkPatchValues::default();
    let mut cache = CommandBufferCache::default();
    let a = cache.link_for_context(1, &gpu(0), &lowered, &values).unwrap();
    let replay = cache.link_for_context(1, &gpu(0), &lowered, &values).unwrap();
    let other_context = cache.link_for_context(2, &gpu(0), &lowered, &values).unwrap();
    let other_device = cache.link_for_context(1, &gpu(1), &lowered, &values).unwrap();
    assert!(std::sync::Arc::ptr_eq(&a, &replay));
    assert!(!std::sync::Arc::ptr_eq(&a, &other_context));
    assert!(!std::sync::Arc::ptr_eq(&a, &other_device));
}

#[test]
fn concurrent_device_replays_patch_private_buffers() {
    use crate::hcq::CommandField;
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission.push(Command::Copy { dst: 0, src: 0, bytes: 8 });
    submission.bind(0, CommandField::CopyDst, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(0, CommandField::CopySrc, PatchSource::RuntimeBuffer(1)).unwrap();
    let linked = std::sync::Arc::new(SemanticLinkedSubmission::new(submission));
    std::thread::scope(|scope| {
        for lane in 0..2u64 {
            let linked = std::sync::Arc::clone(&linked);
            scope.spawn(move || {
                let mut replay = linked.replay_buffer();
                linked
                    .patch(
                        &mut replay,
                        &RuntimePatchValues { buffers: vec![0x1000 + lane, 0x2000 + lane], vars: vec![] },
                        &SystemPatchValues::default(),
                    )
                    .unwrap();
                assert_eq!(
                    replay.submission().commands[0],
                    Command::Copy { dst: 0x1000 + lane, src: 0x2000 + lane, bytes: 8 }
                );
            });
        }
    });
    assert_eq!(linked.static_submission().commands[0], Command::Copy { dst: 0, src: 0, bytes: 8 });
}
