use super::test_support::{MockAmdCall, MockAmdIface, amd_alloc_or_skip};
use crate::amd::AmdAllocator;
use crate::amd::connector::PoolQueue;
use crate::amd::iface::PublicationStage;
use crate::amd::queue::*;
use crate::amd::sys::hsa::{
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM, hsa_kernel_dispatch_packet_t, kernel_dispatch_header,
};
use crate::error::Error;
use std::sync::Arc;

fn mock_allocator(xccs: u32) -> (Arc<MockAmdIface>, AmdAllocator) {
    let iface = Arc::new(MockAmdIface::default());
    let dev = crate::amd::device::AmdDevice::synthetic_with_xcc(
        Arc::clone(&iface) as Arc<dyn crate::amd::iface::AmdIface>,
        xccs,
    );
    (iface, AmdAllocator { dev, device_id: 0 })
}

fn install_signal_pool(allocator: &AmdAllocator) {
    allocator.dev.core().install_signal_pool(crate::amd::signal::SignalPool::new(allocator, 64).expect("signal pool"));
}

fn scripted_error(stage: &'static str) -> Error {
    Error::Runtime { message: format!("scripted {stage} failure") }
}

#[test]
fn aql_packet_header_layout() {
    let h = kernel_dispatch_header();
    // AQL packet header = TYPE_KERNEL_DISPATCH | barrier | sys-acq | sys-rel
    let sys = hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16;
    let expected: u16 = 2 | (1 << 8) | (sys << 9) | (sys << 11);
    assert_eq!(h, expected);
}

#[test]
fn aql_packet_is_64_bytes() {
    assert_eq!(size_of::<hsa_kernel_dispatch_packet_t>(), AQL_PACKET_BYTES);
}

#[test]
fn hcq_command_stream_limits_are_checked_in_release_builds() {
    assert!(validate_pm4_dword_count(1).is_ok());
    assert!(validate_pm4_dword_count(1024).is_ok());
    assert!(matches!(
        validate_pm4_dword_count(1025),
        Err(crate::error::Error::CommandStreamTooLarge { kind: "PM4 ring submission", actual: 1025, limit: 1024 })
    ));

    let aql_slots = COMPUTE_RING_BYTES / AQL_PACKET_BYTES;
    assert!(validate_aql_packet_count(aql_slots - 1).is_ok());
    assert!(matches!(
        validate_aql_packet_count(aql_slots),
        Err(crate::error::Error::CommandStreamTooLarge { kind: "AQL ring submission", .. })
    ));

    assert!(build_aql_vendor_ib_packet(0x1000, crate::amd::sys::pm4::INDIRECT_BUFFER_SIZE_MASK).is_ok());
    assert!(matches!(
        build_aql_vendor_ib_packet(0x1000, crate::amd::sys::pm4::INDIRECT_BUFFER_SIZE_MASK + 1),
        Err(crate::error::Error::CommandStreamTooLarge { kind: "PM4 indirect buffer", .. })
    ));
    let ib = build_pm4_indirect_buffer(0x1122_3344_5566_7788, 1025).unwrap();
    assert_eq!(ib[0], crate::amd::sys::pm4::packet3(crate::amd::sys::pm4::PACKET3_INDIRECT_BUFFER, 2));
    assert_eq!(ib[1], 0x5566_7788);
    assert_eq!(ib[2], 0x1122_3344);
    assert_eq!(ib[3], 1025 | crate::amd::sys::pm4::INDIRECT_BUFFER_VALID);
}

#[test]
fn linked_transaction_limits_include_aggregate_and_sdma_wrap_padding() {
    assert_eq!(validate_linked_compute_lengths(true, 64, &[16, 20]).unwrap(), 9);
    assert!(matches!(
        validate_linked_compute_lengths(true, 64, &[32, 32]),
        Err(crate::error::Error::CommandStreamTooLarge { kind: "PM4 linked transaction", actual: 16, limit: 15 })
    ));
    assert_eq!(validate_linked_compute_lengths(false, 256, &[64, 128]).unwrap(), 3);
    assert!(validate_linked_compute_lengths(false, 256, &[65]).is_err());

    // Starting 8 bytes before ring end, a 16-byte packet needs 8 bytes of NOP
    // padding before it, then the next packet follows without another wrap.
    assert_eq!(linked_sdma_published_bytes(56, 64, &[16, 8]).unwrap(), 32);
    assert!(matches!(
        linked_sdma_published_bytes(56, 64, &[32, 32]),
        Err(crate::error::Error::CommandStreamTooLarge { kind: "SDMA linked transaction", .. })
    ));
}

#[test]
fn pm4_read_pointer_reconstructs_the_producer_epoch() {
    const CAPACITY: usize = 4_194_304;

    assert_eq!(absolute_pm4_read_idx(20_000, 7_485, CAPACITY), 7_485);
    assert_eq!(absolute_pm4_read_idx(CAPACITY as u64 + 24_588, 7_485, CAPACITY), CAPACITY as u64 + 7_485);
    assert_eq!(absolute_pm4_read_idx(CAPACITY as u64 + 7_485, CAPACITY as u64 - 10, CAPACITY), CAPACITY as u64 - 10);
}

#[test]
fn mock_compute_queue_construction_unwinds_every_allocation_stage() {
    for (xccs, allocation_stages) in [(1, 4), (2, 5)] {
        for fail_at in 0..allocation_stages {
            let (iface, allocator) = mock_allocator(xccs);
            for _ in 0..fail_at {
                iface.script_alloc(Ok(()));
            }
            iface.script_alloc(Err(scripted_error("compute queue allocation")));

            assert!(AmdComputeQueue::create(&allocator).is_err(), "xccs={xccs}, fail_at={fail_at}");
            assert_eq!(iface.allocation_count(), fail_at, "xccs={xccs}, fail_at={fail_at}");
            assert_eq!(iface.free_count(), fail_at, "xccs={xccs}, fail_at={fail_at}");
            assert_eq!(iface.live_handle_count(), 0, "xccs={xccs}, fail_at={fail_at}");
            assert!(iface.free_issues().is_empty());
        }
    }
}

#[test]
fn mock_compute_queue_setup_success_failure_and_active_rollback_are_owned() {
    let (iface, allocator) = mock_allocator(1);
    iface.script_setup(Err(scripted_error("setup")));
    assert!(AmdComputeQueue::create(&allocator).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (4, 4, 0));

    let queue = AmdComputeQueue::create(&allocator).expect("queue");
    assert_eq!(iface.live_queue_count(), 1);
    drop(queue);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (8, 8, 0));
    assert_eq!((iface.queue_setup_count(), iface.queue_teardown_count(), iface.live_queue_count()), (1, 1, 0));
    assert!(iface.free_issues().is_empty());

    let queue = AmdComputeQueue::create(&allocator).expect("queue with leaked doorbell");
    iface.script_teardown(Ok(crate::amd::iface::QueueTeardown::DoorbellLeaked { errno: 12 }));
    drop(queue);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (12, 12, 0));
    assert_eq!(iface.live_queue_count(), 0);

    let (iface, allocator) = mock_allocator(1);
    iface.script_setup(Err(Error::AmdQueueStillActive { queue_id: 77, cause: "scripted rollback failure".into() }));
    assert!(matches!(AmdComputeQueue::create(&allocator), Err(Error::AmdQueueStillActive { .. })));
    assert!(allocator.dev.is_poisoned());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (4, 0, 4));
}

#[test]
fn mock_copy_queue_construction_unwinds_ring_signal_and_staging_stages() {
    for fail_at in 0..3 {
        let (iface, allocator) = mock_allocator(1);
        install_signal_pool(&allocator);
        let baseline_allocs = iface.allocation_count();
        for _ in 0..fail_at {
            iface.script_alloc(Ok(()));
        }
        iface.script_alloc(Err(scripted_error("copy queue allocation")));

        assert!(AmdCopyQueue::create(&allocator).is_err(), "fail_at={fail_at}");
        assert_eq!(iface.allocation_count() - baseline_allocs, fail_at, "fail_at={fail_at}");
        assert_eq!(iface.free_count(), fail_at, "fail_at={fail_at}");
        assert_eq!(iface.live_handle_count(), baseline_allocs, "fail_at={fail_at}");
        assert_eq!(iface.queue_setup_count(), usize::from(fail_at == 2));
        assert_eq!(iface.queue_teardown_count(), usize::from(fail_at == 2));
        assert!(iface.free_issues().is_empty());
    }

    let (iface, allocator) = mock_allocator(1);
    let pool = crate::amd::signal::SignalPool::new(&allocator, 64).expect("signal pool");
    let held = (0..64).map(|_| pool.acquire().unwrap()).collect::<Vec<_>>();
    allocator.dev.core().install_signal_pool(Arc::clone(&pool));
    assert!(AmdCopyQueue::create(&allocator).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count()), (3, 2));
    assert_eq!((iface.queue_setup_count(), iface.queue_teardown_count()), (1, 1));
    drop(held);

    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    iface.script_alloc(Ok(()));
    iface.script_alloc(Ok(()));
    iface.script_alloc(Err(scripted_error("staging")));
    iface.script_teardown(Err(scripted_error("partial queue destroy")));
    assert!(AmdCopyQueue::create(&allocator).is_err());
    assert!(allocator.dev.is_poisoned());
    assert_eq!(iface.allocation_count(), baseline + 2);
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), baseline + 2);
    assert_eq!(iface.live_queue_count(), 1);
}

#[test]
fn mock_copy_queue_drop_balances_or_quarantines_after_destroy_failure() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");
    drop(queue);
    assert_eq!(iface.allocation_count() - baseline, 3);
    assert_eq!(iface.free_count(), 3);
    assert_eq!((iface.queue_teardown_count(), iface.live_queue_count()), (1, 0));

    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");
    iface.script_teardown(Err(scripted_error("destroy")));
    drop(queue);
    assert!(allocator.dev.is_poisoned());
    assert_eq!(iface.allocation_count() - baseline, 3);
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), baseline + 3);
    assert_eq!(iface.live_queue_count(), 1);
}

#[test]
fn mock_pool_queue_construction_unwinds_queue_arena_and_scratch_stages() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    for fail_at in 0..6 {
        let allocations_before = iface.allocation_count();
        let frees_before = iface.free_count();
        for _ in 0..fail_at {
            iface.script_alloc(Ok(()));
        }
        iface.script_alloc(Err(scripted_error("pool allocation")));
        assert!(PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).is_err());
        assert_eq!(iface.allocation_count() - allocations_before, fail_at, "fail_at={fail_at}");
        assert_eq!(iface.free_count() - frees_before, fail_at, "fail_at={fail_at}");
        assert_eq!(iface.live_handle_count(), baseline, "fail_at={fail_at}");
        assert!(iface.free_issues().is_empty());
    }

    let allocations_before = iface.allocation_count();
    let frees_before = iface.free_count();
    let queue = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    drop(queue);
    assert_eq!(iface.allocation_count() - allocations_before, 6);
    assert_eq!(iface.free_count() - frees_before, 6);
    assert_eq!(iface.live_handle_count(), baseline);
}

#[test]
fn mock_pm4_and_aql_publication_failures_restore_or_poison_by_doorbell_stage() {
    for xccs in [1, 2] {
        let (iface, allocator) = mock_allocator(xccs);
        install_signal_pool(&allocator);
        let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
        let mut submission = crate::hcq::Submission::new(crate::hcq::QueueKind::Compute(0));
        submission.push(crate::hcq::Command::MemoryBarrier).push(crate::hcq::Command::Compute(
            crate::hcq::ComputeDispatch {
                workgroup_size: [1, 1, 1],
                grid_size: [1, 1, 1],
                private_segment_size: 0,
                group_segment_size: 0,
                kernel_object: 0x1000,
                kernarg_address: 0x2000,
                completion_signal: 0,
                barrier: true,
                amd_pm4: Some(crate::hcq::AmdPm4Dispatch {
                    rsrc: [0, 0, 0],
                    program_address: 0x1000,
                    enable_private_segment_sgpr: false,
                    workgroup_count: [1, 1, 1],
                    wave32: true,
                    target_major: 11,
                }),
            },
        ));

        iface.script_publication(Err(scripted_error("after reservation")));
        assert!(pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]).is_err());
        assert_eq!(pool.pm4_value(), 1);
        assert_eq!(pool.queue().ring_write_idx(), 0, "xccs={xccs}");
        assert!(!allocator.dev.is_poisoned());

        iface.script_publication(Ok(()));
        iface.script_publication(Err(scripted_error("before doorbell")));
        assert!(pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]).is_err());
        assert_eq!(pool.pm4_value(), 1);
        assert_eq!(pool.queue().ring_write_idx(), 0, "xccs={xccs}");
        assert!(!allocator.dev.is_poisoned());

        iface.script_publication(Ok(()));
        iface.script_publication(Ok(()));
        iface.script_publication(Err(scripted_error("after doorbell")));
        let post_doorbell_error = match pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]) {
            Ok(_) => panic!("scripted post-doorbell failure unexpectedly succeeded"),
            Err(error) => error,
        };
        assert_eq!(pool.pm4_value(), 2, "post-doorbell reservation xccs={xccs}, error={post_doorbell_error:?}");
        assert!(allocator.dev.is_poisoned(), "xccs={xccs}, error={post_doorbell_error:?}");
        let stages = iface
            .transcript()
            .into_iter()
            .filter_map(|call| match call {
                MockAmdCall::PublicationCheckpoint { stage } => Some(stage),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            stages,
            [
                PublicationStage::AfterReservation,
                PublicationStage::AfterReservation,
                PublicationStage::BeforeDoorbell,
                PublicationStage::AfterReservation,
                PublicationStage::BeforeDoorbell,
                PublicationStage::AfterDoorbell,
            ]
        );
        drop(pool);
        assert_eq!(iface.free_count(), 0, "poisoned xccs={xccs} resources must be quarantined");
    }
}

#[test]
fn mock_copy_publication_restores_before_doorbell_and_poisons_after() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");

    iface.script_publication(Err(scripted_error("copy after reservation")));
    assert!(queue.copy_fenced(0x1000, 0x2000, 4).is_err());
    assert_eq!(queue.ring_write_idx(), 0, "the abandoned copy packets must be rolled back");
    assert!(!allocator.dev.is_poisoned());

    iface.script_publication(Ok(()));
    iface.script_publication(Err(scripted_error("copy before doorbell")));
    assert!(queue.copy_fenced(0x1000, 0x2000, 4).is_err());
    assert_eq!(queue.ring_write_idx(), 0);
    assert!(!allocator.dev.is_poisoned());

    iface.script_publication(Ok(()));
    iface.script_publication(Ok(()));
    iface.script_publication(Err(scripted_error("copy after doorbell")));
    assert!(queue.copy_fenced(0x1000, 0x2000, 4).is_err());
    assert!(allocator.dev.is_poisoned());
    drop(queue);
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), baseline + 3);
    assert_eq!(iface.live_queue_count(), 1);
}

#[test]
fn mock_copy_publication_panic_rolls_the_ring_back_to_the_pre_copy_index() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let queue = AmdCopyQueue::create(&allocator).expect("copy queue");

    iface.script_publication(Ok(()));
    iface.script_publication_panic();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = queue.copy_fenced(0x1000, 0x2000, 4);
    }));
    assert!(result.is_err());
    // The rollback owns the ring index from before the first copy packet, so an
    // unwound publication leaves nothing enqueued.
    assert_eq!(queue.ring_write_idx(), 0);
    assert!(!allocator.dev.is_poisoned());
}

#[test]
fn mock_publication_panic_restores_before_doorbell_and_poisons_after() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
    let mut submission = crate::hcq::Submission::new(crate::hcq::QueueKind::Compute(0));
    submission.push(crate::hcq::Command::MemoryBarrier);

    iface.script_publication(Ok(()));
    iface.script_publication_panic();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]);
    }));
    assert!(result.is_err());
    assert_eq!(pool.pm4_value(), 1);
    assert!(!allocator.dev.is_poisoned());

    iface.script_publication(Ok(()));
    iface.script_publication(Ok(()));
    iface.script_publication_panic();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = pool.queue().submit_hcq_dispatch(&pool, &submission, &[], &[]);
    }));
    assert!(result.is_err());
    assert_eq!(pool.pm4_value(), 2);
    assert!(allocator.dev.is_poisoned());
    drop(pool);
    assert_eq!(iface.free_count(), 0);
}

#[test]
fn mock_pool_failed_drain_and_panic_abandonment_quarantine_every_backing() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    pool.next_pm4();
    iface.script_wait(Err(scripted_error("drain")));
    drop(pool);
    assert!(allocator.dev.is_poisoned());
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), baseline + 6);

    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).unwrap();
        panic!("scripted pool abandonment");
    }));
    assert!(result.is_err());
    assert!(allocator.dev.is_poisoned());
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), baseline + 6);
}

#[test]
fn mock_scratch_growth_preserves_old_state_on_drain_or_allocation_failure() {
    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    let allocations = iface.allocation_count();
    pool.next_pm4();
    iface.script_wait(Err(scripted_error("scratch drain")));
    assert!(pool.ensure_has_local_memory(4096).is_err());
    assert_eq!(iface.allocation_count(), allocations, "failed drain must not allocate replacement scratch");
    assert_eq!(iface.free_count(), 0, "failed drain must not free old scratch");
    assert!(allocator.dev.is_poisoned());
    drop(pool);
    assert_eq!(iface.free_count(), 0);

    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    iface.script_alloc(Err(scripted_error("replacement scratch allocation")));
    assert!(pool.ensure_has_local_memory(4096).is_err());
    assert!(!allocator.dev.is_poisoned());
    assert_eq!(iface.allocation_count(), baseline + 6);
    assert_eq!(iface.free_count(), 0, "old scratch must survive replacement allocation failure");
    drop(pool);
    assert_eq!(iface.free_count(), 6);

    let (iface, allocator) = mock_allocator(1);
    install_signal_pool(&allocator);
    let baseline = iface.allocation_count();
    let pool = PoolQueue::new_with_resources(Arc::clone(allocator.dev.core()), &allocator).expect("pool queue");
    pool.ensure_has_local_memory(4096).expect("scratch growth");
    assert_eq!(iface.allocation_count(), baseline + 7);
    assert_eq!(iface.free_count(), 1, "successful publication frees exactly the drained old scratch");
    drop(pool);
    assert_eq!(iface.free_count(), 7);
    assert!(iface.free_issues().is_empty());
}

#[test]
fn build_dispatch_picks_correct_dims() {
    // `setup` (dims) is the high 16 bits of the header/setup union's full_header.
    let dims = |p: &hsa_kernel_dispatch_packet_t| (unsafe { p.__bindgen_anon_1.full_header } >> 16) & 0b11;
    let p1 = build_dispatch_packet([64, 1, 1], [1024, 1, 1], 0, 0, 0, 0, 0);
    assert_eq!(dims(&p1), 1);
    let p2 = build_dispatch_packet([8, 8, 1], [256, 256, 1], 0, 0, 0, 0, 0);
    assert_eq!(dims(&p2), 2);
    let p3 = build_dispatch_packet([4, 4, 4], [64, 64, 64], 0, 0, 0, 0, 0);
    assert_eq!(dims(&p3), 3);
}

#[test]
fn hcq_compute_lowers_to_exact_aql_fields() {
    let command = crate::hcq::ComputeDispatch {
        workgroup_size: [8, 4, 2],
        grid_size: [128, 64, 2],
        private_segment_size: 96,
        group_segment_size: 512,
        kernel_object: 0x1122_3344_5566_7788,
        kernarg_address: 0x8877_6655_4433_2210,
        completion_signal: 0x1234_5678_9abc_def0,
        barrier: false,
        amd_pm4: None,
    };
    let packet = lower_hcq_compute(&command).unwrap();
    assert_eq!([packet.workgroup_size_x, packet.workgroup_size_y, packet.workgroup_size_z], [8, 4, 2]);
    assert_eq!([packet.grid_size_x, packet.grid_size_y, packet.grid_size_z], command.grid_size);
    assert_eq!(packet.private_segment_size, 96);
    assert_eq!(packet.group_segment_size, 512);
    assert_eq!(packet.kernel_object, command.kernel_object);
    assert_eq!(packet.kernarg_address as u64, command.kernarg_address);
    assert_eq!(packet.completion_signal.handle, command.completion_signal);
    let header = unsafe { packet.__bindgen_anon_1.full_header } as u16;
    assert_eq!(header & (1 << crate::amd::sys::hsa::hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER), 0);
}

#[test]
fn hcq_compute_rejects_oversize_workgroup_dimension() {
    let command = crate::hcq::ComputeDispatch {
        workgroup_size: [u16::MAX as u32 + 1, 1, 1],
        grid_size: [1, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0,
        kernarg_address: 0,
        completion_signal: 0,
        barrier: true,
        amd_pm4: None,
    };
    assert!(lower_hcq_compute(&command).is_err());
}

#[test]
fn hcq_pm4_and_aql_lower_the_same_neutral_compute_intent() {
    use crate::hcq::{AmdPm4Dispatch, Command, ComputeDispatch, NullHcq, QueueKind, Submission};

    let dispatch = ComputeDispatch {
        workgroup_size: [8, 4, 2],
        grid_size: [128, 32, 8],
        private_segment_size: 96,
        group_segment_size: 512,
        kernel_object: 0x1122_3344_5566_7788,
        kernarg_address: 0x8877_6655_4433_2210,
        completion_signal: 0x1234_5678_9abc_def0,
        barrier: true,
        amd_pm4: Some(AmdPm4Dispatch {
            rsrc: [1, 2, 3],
            program_address: 0x12_3456_7800,
            enable_private_segment_sgpr: false,
            workgroup_count: [16, 8, 4],
            wave32: true,
            target_major: 11,
        }),
    };
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::MemoryBarrier).push(Command::Compute(dispatch.clone()));

    let mut mock = NullHcq::default();
    mock.submit(&submission).unwrap();
    assert_eq!(
        mock.trace(),
        &[(QueueKind::Compute(0), Command::MemoryBarrier), (QueueKind::Compute(0), Command::Compute(dispatch.clone())),]
    );

    let aql = lower_hcq_aql(&submission).unwrap();
    assert_eq!(aql.len(), 1);
    let packet = lower_hcq_compute(&dispatch).unwrap();
    let mut expected_aql = [0u32; 16];
    // SAFETY: the AQL dispatch packet is exactly 16 POD dwords.
    unsafe { std::ptr::copy_nonoverlapping(&packet as *const _ as *const u32, expected_aql.as_mut_ptr(), 16) };
    assert_eq!(aql[0], expected_aql);

    let pm4 = lower_hcq_pm4(&submission, pm4_state()).unwrap();
    assert!(!pm4.is_empty());
    assert_eq!(pm4[0], 0xc005_3c00); // neutral barrier lowers before compute setup
    assert!(pm4.contains(&0xc003_1500)); // DISPATCH_DIRECT packet
}

fn pm4_state() -> Pm4LoweringState {
    Pm4LoweringState {
        scratch_address: 0x1111_2222_3333_4400,
        tmpring_size: 0x55,
        target_major: 11,
        completion_xcc_mask: None,
    }
}

fn aql_control_state(multi_xcc: bool) -> Pm4LoweringState {
    Pm4LoweringState {
        scratch_address: 0x1111_2222_3333_4400,
        tmpring_size: 0x55,
        target_major: 9,
        completion_xcc_mask: multi_xcc.then_some(1),
    }
}

#[test]
fn hcq_pm4_wait_barrier_store_timestamp_goldens() {
    use crate::hcq::{Command, QueueKind, Submission};
    let one = |command| {
        let mut submission = Submission::new(QueueKind::Compute(0));
        submission.push(command);
        lower_hcq_pm4(&submission, pm4_state()).unwrap()
    };

    assert_eq!(
        one(Command::Wait { signal_address: 0x1122_3344_5566_7788, value: 7 }),
        [0xc005_3c00, 0x15, 0x5566_7788, 0x1122_3344, 7, 0xffff_ffff, 4]
    );
    assert_eq!(
        one(Command::MemoryBarrier),
        [
            0xc005_3c00,
            0x45,
            0xe26,
            0xe27,
            0xffff_ffff,
            0xffff_ffff,
            4,
            0xc006_5800,
            0,
            0xffff_ffff,
            0xffff_ffff,
            0,
            0,
            0,
            0xc3f1,
        ]
    );
    assert_eq!(
        one(Command::Store { dst: 0x1122_3344_5566_7788, value: 0xaabb_ccdd_eeff_0011 }),
        [0xc006_4900, 0x70f514, 0x4000_0000, 0x5566_7788, 0x1122_3344, 0xeeff_0011, 0xaabb_ccdd, 0]
    );
    assert_eq!(
        one(Command::Timestamp { dst: 0x1122_3344_5566_7788 }),
        [
            0xc006_4900,
            0x514,
            0x0200_0000,
            0,
            0,
            0,
            0,
            0,
            0xc006_4900,
            0x514,
            0x6000_0000,
            0x5566_7788,
            0x1122_3344,
            0,
            0,
            0,
            0xc006_5800,
            0,
            0xffff_ffff,
            0xffff_ffff,
            0,
            0,
            0,
            0xc3f1,
        ]
    );
}

#[test]
fn hcq_pm4_compute_golden() {
    use crate::hcq::{AmdPm4Dispatch, Command, ComputeDispatch, QueueKind, Submission};
    let dispatch = ComputeDispatch {
        workgroup_size: [8, 4, 2],
        grid_size: [128, 32, 8],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0,
        kernarg_address: 0x0000_00ab_cdef_0010,
        completion_signal: 0,
        barrier: true,
        amd_pm4: Some(AmdPm4Dispatch {
            rsrc: [1, 2, 3],
            program_address: 0x12_3456_7800,
            enable_private_segment_sgpr: false,
            workgroup_count: [16, 8, 4],
            wave32: true,
            target_major: 11,
        }),
    };
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Compute(dispatch));
    assert_eq!(
        lower_hcq_pm4(&submission, pm4_state()).unwrap(),
        [
            0xc006_5800,
            0,
            0xffff_ffff,
            0xffff_ffff,
            0,
            0,
            0,
            0x3f0,
            0xc002_7600,
            0x20c,
            0x1234_5678,
            0,
            0xc002_7600,
            0x212,
            1,
            2,
            0xc001_7600,
            0x228,
            3,
            0xc001_7600,
            0x218,
            0x55,
            0xc002_7600,
            0x210,
            0x2233_3344,
            0x0011_1122,
            0xc003_7600,
            0x21b,
            0,
            0,
            0,
            0xc002_7600,
            0x240,
            0xcdef_0010,
            0xab,
            0xc001_7600,
            0x215,
            0,
            0xc008_7600,
            0x204,
            0,
            0,
            0,
            8,
            4,
            2,
            0,
            0,
            0xc003_1500,
            16,
            8,
            4,
            0x8005,
            0xc000_4600,
            0x407,
        ]
    );
}

#[test]
fn hcq_pm4_mixed_submission_keeps_dependency_and_release_order() {
    use crate::hcq::{Command, QueueKind, Submission};
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission
        .push(Command::Wait { signal_address: 0x1000, value: 3 })
        .push(Command::MemoryBarrier)
        .push(Command::Timestamp { dst: 0x2000 })
        .push(Command::Store { dst: 0x3000, value: 4 });
    let q = lower_hcq_pm4(&submission, pm4_state()).unwrap();
    assert_eq!(q.len(), 7 + 15 + 24 + 8);
    assert_eq!(&q[..7], &[0xc005_3c00, 0x15, 0x1000, 0, 3, u32::MAX, 4]);
    assert_eq!(q[7], 0xc005_3c00); // release/acquire barrier follows dependency wait
    assert_eq!(q[22], 0xc006_4900); // timestamp's ordering release
    assert_eq!(&q[q.len() - 8..], &[0xc006_4900, 0x70f514, 0x4000_0000, 0x3000, 0, 4, 0, 0]);
}

#[test]
fn hcq_sdma_command_and_mixed_goldens() {
    use crate::hcq::{Command, QueueKind, Submission};
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission
        .push(Command::MemoryBarrier)
        .push(Command::Wait { signal_address: 0x1_0000_1000, value: 9 })
        .push(Command::Copy { dst: 0x2_0000_2000, src: 0x3_0000_3000, bytes: 16 })
        .push(Command::Timestamp { dst: 0x4_0000_4000 })
        .push(Command::Store { dst: 0x5_0000_5000, value: 0x5566_7788 });
    assert_eq!(
        lower_hcq_sdma(&submission, 11).unwrap(),
        [
            0xd000_0008,
            0x1000,
            1,
            9,
            u32::MAX,
            0x0fff_0004,
            1,
            15,
            0,
            0x3000,
            3,
            0x2000,
            2,
            0x20d,
            0x4000,
            4,
            0x0003_0005,
            0x5000,
            5,
            0x5566_7788,
        ]
    );
}

#[test]
fn hcq_amd_rejects_unsupported_packet_forms_and_wide_waits() {
    use crate::hcq::{Command, QueueKind, Submission};
    let mut compute = Submission::new(QueueKind::Compute(0));
    compute.push(Command::Copy { dst: 1, src: 2, bytes: 4 });
    assert!(lower_hcq_pm4(&compute, pm4_state()).unwrap_err().to_string().contains("does not support"));
    let mut copy = Submission::new(QueueKind::Copy(0));
    copy.push(Command::Execute { operation: 0 });
    assert!(lower_hcq_sdma(&copy, 11).unwrap_err().to_string().contains("does not support"));
    let mut wide = Submission::new(QueueKind::Compute(0));
    wide.push(Command::Wait { signal_address: 0x1000, value: u32::MAX as u64 + 1 });
    assert!(lower_hcq_pm4(&wide, pm4_state()).unwrap_err().to_string().contains("32-bit"));
}

fn replay_dwords(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|word| u32::from_le_bytes(word.try_into().unwrap())).collect()
}

#[test]
fn hcq_pm4_dynamic_normal_replay_patches_vars_addresses_without_relowering() {
    use crate::hcq::{
        AmdPm4Dispatch, Command, CommandBufferCache, CommandField, ComputeDispatch, LinkPatchValues, PatchEncoding,
        PatchSource, QueueKind, RuntimePatchValues, Submission, SystemField, SystemPatchValues,
    };

    let dispatch = ComputeDispatch {
        workgroup_size: [8, 1, 1],
        grid_size: [16, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0,
        kernarg_address: 0,
        completion_signal: 0,
        barrier: true,
        amd_pm4: Some(AmdPm4Dispatch {
            rsrc: [1, 2, 3],
            program_address: 0,
            enable_private_segment_sgpr: true,
            workgroup_count: [16, 1, 1],
            wave32: true,
            target_major: 11,
        }),
    };
    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Wait { signal_address: 0, value: 0 }).push(Command::Compute(dispatch));
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.bind(1, CommandField::ComputeProgramAddress, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(1, CommandField::ComputeKernargAddress, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(1, CommandField::ComputeGrid(0), PatchSource::RuntimeVar(0)).unwrap();
    submission.bind(1, CommandField::ComputeScratchAddress, PatchSource::System(SystemField::ScratchAddress)).unwrap();
    submission.bind(1, CommandField::ComputeScratchTmpring, PatchSource::System(SystemField::ScratchTmpring)).unwrap();

    let lowered = lower_hcq_pm4_command_buffer(&submission, pm4_state()).unwrap();
    assert_eq!(lowered.patches.link.len(), 2);
    assert_eq!(lowered.patches.runtime.len(), 3);
    assert_eq!(lowered.patches.system.len(), 8);
    let mut cache = CommandBufferCache::default();
    let linked = cache.link(&lowered, &LinkPatchValues(vec![0x12_3456_7800])).unwrap();
    let static_bytes = linked.static_bytes().to_vec();
    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::TimelineSignal(0), 0x1000);
    system.0.insert(SystemField::TimelineValue(0), 3);
    system.0.insert(SystemField::ScratchAddress, 0x1234_5678_9000);
    system.0.insert(SystemField::ScratchTmpring, 0x55);
    linked
        .patch(&mut replay, &RuntimePatchValues { buffers: vec![0xaaaa_bbbb_cccc_0000], vars: vec![32] }, &system)
        .unwrap();
    let first = replay.bytes().to_vec();

    system.0.insert(SystemField::TimelineSignal(0), 0x9000);
    system.0.insert(SystemField::TimelineValue(0), 4);
    system.0.insert(SystemField::ScratchAddress, 0x2234_5678_a000);
    system.0.insert(SystemField::ScratchTmpring, 0x66);
    linked
        .patch(&mut replay, &RuntimePatchValues { buffers: vec![0x1111_2222_3333_0000], vars: vec![64] }, &system)
        .unwrap();
    assert_ne!(first, replay.bytes());
    assert_eq!(linked.static_bytes(), static_bytes);
    let dwords = replay_dwords(replay.bytes());
    assert_eq!(&dwords[2..5], &[0x9000, 0, 4]);
    let descriptor_high =
        lowered.patches.system.iter().find(|site| site.encoding == PatchEncoding::High32Or(1 << 31)).unwrap();
    assert_eq!(dwords[descriptor_high.byte_offset / 4], 0x8000_2234);
    for site in &lowered.patches.link {
        assert_eq!(
            &replay.bytes()[site.byte_offset..site.byte_offset + 4],
            &static_bytes[site.byte_offset..site.byte_offset + 4]
        );
    }
}

#[test]
fn hcq_sdma_dynamic_normal_replay_patches_chunk_addresses_without_relowering() {
    use crate::hcq::{
        Command, CommandBufferCache, CommandField, LinkPatchValues, PatchSource, QueueKind, RuntimePatchValues,
        Submission, SystemField, SystemPatchValues,
    };

    let bytes = crate::amd::sys::sdma::SDMA_MAX_COPY_BYTES + 8;
    let mut submission = Submission::new(QueueKind::Copy(0));
    submission
        .push(Command::Copy { dst: 0, src: 0, bytes })
        .push(Command::Timestamp { dst: 0 })
        .push(Command::Store { dst: 0, value: 0 });
    submission.bind(0, CommandField::CopySrc, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(0, CommandField::CopyDst, PatchSource::RuntimeBuffer(1)).unwrap();
    submission.bind(1, CommandField::TimestampDst, PatchSource::System(SystemField::Timestamp(0))).unwrap();
    submission.bind(2, CommandField::StoreDst, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(2, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();

    let lowered = lower_hcq_sdma_command_buffer(&submission, 11).unwrap();
    assert_eq!(lowered.patches.runtime.len(), 8); // src/dst lo+hi for both chunks
    let linked = CommandBufferCache::default().link(&lowered, &LinkPatchValues(vec![0x7000])).unwrap();
    let static_bytes = linked.static_bytes().to_vec();
    let mut replay = linked.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::Timestamp(0), 0x8000);
    system.0.insert(SystemField::TimelineValue(0), 5);
    linked
        .patch(&mut replay, &RuntimePatchValues { buffers: vec![0x1_0000_0000, 0x2_0000_0000], vars: vec![] }, &system)
        .unwrap();
    let first = replay_dwords(replay.bytes());
    assert_eq!([first[3], first[4]], [0, 1]);
    assert_eq!([first[10], first[11]], [crate::amd::sys::sdma::SDMA_MAX_COPY_BYTES as u32, 1]);

    system.0.insert(SystemField::Timestamp(0), 0xa000);
    system.0.insert(SystemField::TimelineValue(0), 6);
    linked
        .patch(&mut replay, &RuntimePatchValues { buffers: vec![0x3_0000_1000, 0x4_0000_2000], vars: vec![] }, &system)
        .unwrap();
    let second = replay_dwords(replay.bytes());
    assert_eq!([second[3], second[4]], [0x1000, 3]);
    assert_eq!([second[10], second[11]], [0x0040_1000, 3]);
    assert_eq!(linked.static_bytes(), static_bytes);
}

#[test]
fn hcq_aql_dynamic_normal_replay_patches_vars_and_addresses_without_kernel_completion() {
    use crate::amd::queue::lower_hcq_aql_command_buffer;
    use crate::hcq::{
        Command, CommandBufferCache, CommandField, ComputeDispatch, LinkPatchValues, PatchSource, QueueKind,
        RuntimePatchValues, Submission, SystemPatchValues,
    };

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::MemoryBarrier).push(Command::Compute(ComputeDispatch {
        workgroup_size: [8, 1, 1],
        grid_size: [16, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0,
        kernarg_address: 0,
        completion_signal: 0,
        barrier: true,
        amd_pm4: None,
    }));
    submission.bind(1, CommandField::ComputeKernelObject, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(1, CommandField::ComputeKernargAddress, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(1, CommandField::ComputeGrid(0), PatchSource::RuntimeVar(0)).unwrap();
    let lowered = lower_hcq_aql_command_buffer(&submission).unwrap();
    let linked = CommandBufferCache::default().link(&lowered, &LinkPatchValues(vec![0x1234_5600])).unwrap();
    let immutable = linked.static_bytes().to_vec();
    let mut replay = linked.replay_buffer();
    let system = SystemPatchValues::default();
    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x1_0000_1000], vars: vec![32] }, &system).unwrap();
    let first = replay.bytes().to_vec();
    assert_eq!(&first[12..16], &32u32.to_le_bytes());
    assert_eq!(&first[40..48], &0x1_0000_1000u64.to_le_bytes());
    assert_eq!(&first[56..64], &0u64.to_le_bytes());

    linked.patch(&mut replay, &RuntimePatchValues { buffers: vec![0x2_0000_2000], vars: vec![64] }, &system).unwrap();
    assert_ne!(replay.bytes(), first);
    assert_eq!(&replay.bytes()[12..16], &64u32.to_le_bytes());
    assert_eq!(&replay.bytes()[40..48], &0x2_0000_2000u64.to_le_bytes());
    assert_eq!(&replay.bytes()[56..64], &0u64.to_le_bytes());
    assert_eq!(linked.static_bytes(), immutable);
}

#[test]
fn hcq_aql_submission_program_keeps_wait_store_and_dispatch_on_device() {
    use crate::hcq::{
        Command, CommandBufferCache, CommandField, ComputeDispatch, LinkPatchValues, PatchSource, QueueKind,
        RuntimePatchValues, Submission, SystemField, SystemPatchValues,
    };

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission
        .push(Command::Wait { signal_address: 0, value: 0 })
        .push(Command::MemoryBarrier)
        .push(Command::Compute(ComputeDispatch {
            workgroup_size: [8, 1, 1],
            grid_size: [16, 1, 1],
            private_segment_size: 0,
            group_segment_size: 0,
            kernel_object: 0,
            kernarg_address: 0,
            completion_signal: 0,
            barrier: true,
            amd_pm4: None,
        }))
        .push(Command::Store { dst: 0, value: 0 });
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.bind(2, CommandField::ComputeKernelObject, PatchSource::LinkAddress(0)).unwrap();
    submission.bind(2, CommandField::ComputeKernargAddress, PatchSource::RuntimeBuffer(0)).unwrap();
    submission.bind(3, CommandField::StoreDst, PatchSource::System(SystemField::TimelineSignal(1))).unwrap();
    submission.bind(3, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(1))).unwrap();

    let lowered =
        lower_hcq_aql_submission_program(&submission, aql_control_state(true), PatchSource::LinkAddress(1)).unwrap();
    assert_eq!(lowered.aql.bytes.len(), 3 * AQL_PACKET_BYTES, "IB, dispatch, IB");
    assert!(!lowered.control.bytes.is_empty());

    let links = LinkPatchValues(vec![0x1234_5600, 0x8000_0000]);
    let aql = CommandBufferCache::default().link(&lowered.aql, &links).unwrap();
    let control = CommandBufferCache::default().link(&lowered.control, &links).unwrap();
    let mut aql_replay = aql.replay_buffer();
    let mut control_replay = control.replay_buffer();
    let mut system = SystemPatchValues::default();
    system.0.insert(SystemField::TimelineSignal(0), 0x1000);
    system.0.insert(SystemField::TimelineValue(0), 7);
    system.0.insert(SystemField::TimelineSignal(1), 0x2000);
    system.0.insert(SystemField::TimelineValue(1), 8);
    let runtime = RuntimePatchValues { buffers: vec![0x3000], vars: vec![] };
    aql.patch(&mut aql_replay, &runtime, &system).unwrap();
    control.patch(&mut control_replay, &runtime, &system).unwrap();

    assert_eq!(&aql_replay.bytes()[8..16], &0x8000_0000u64.to_le_bytes());
    assert_eq!(&aql_replay.bytes()[AQL_PACKET_BYTES + 40..AQL_PACKET_BYTES + 48], &0x3000u64.to_le_bytes());
    let trailing_ib =
        u64::from_le_bytes(aql_replay.bytes()[2 * AQL_PACKET_BYTES + 8..2 * AQL_PACKET_BYTES + 16].try_into().unwrap());
    assert!(trailing_ib > 0x8000_0000, "trailing IB points at the linked store run");
    let control_words = replay_dwords(control_replay.bytes());
    assert_eq!(&control_words[2..5], &[0x1000, 0, 7]);
    assert_eq!(&control_words[control_words.len() - 5..control_words.len() - 2], &[0x2000, 0, 8]);
    assert_eq!(
        &control_words[control_words.len() - 10..control_words.len() - 8],
        &crate::amd::sys::pm4::pred_exec(1, 8)
    );
}

#[test]
fn hcq_aql_submission_program_supports_control_only_finalizer() {
    use crate::hcq::{Command, CommandField, PatchSource, QueueKind, Submission, SystemField};

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Wait { signal_address: 0, value: 0 }).push(Command::Store { dst: 0, value: 0 });
    submission.bind(0, CommandField::WaitAddress, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::WaitValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    submission.bind(1, CommandField::StoreDst, PatchSource::System(SystemField::TimelineSignal(1))).unwrap();
    submission.bind(1, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(1))).unwrap();

    let lowered =
        lower_hcq_aql_submission_program(&submission, aql_control_state(true), PatchSource::LinkAddress(0)).unwrap();
    assert_eq!(lowered.aql.bytes.len(), AQL_PACKET_BYTES, "control-only finalizer needs one vendor IB packet");
    assert!(!lowered.control.bytes.is_empty());
    let words = replay_dwords(&lowered.control.bytes);
    assert_eq!(&words[words.len() - 10..words.len() - 8], &crate::amd::sys::pm4::pred_exec(1, 8));
}

#[test]
fn hcq_aql_direct_timeline_packets_leave_kernel_completion_unset() {
    use crate::hcq::{Command, ComputeDispatch, PatchSource, QueueKind, Submission};

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::MemoryBarrier).push(Command::Compute(ComputeDispatch {
        workgroup_size: [8, 1, 1],
        grid_size: [64, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0x1234_5600,
        kernarg_address: 0x9000,
        completion_signal: 0,
        barrier: true,
        amd_pm4: None,
    }));
    let finalized = finalize_hcq_aql_timeline_submission(&submission, 0x2000, 7, 8, None).unwrap();
    let lowered =
        lower_hcq_aql_submission_program(&finalized, aql_control_state(true), PatchSource::LinkAddress(0)).unwrap();

    assert_eq!(lowered.aql.bytes.len(), 3 * AQL_PACKET_BYTES, "prefix IB, kernel, terminal IB");
    for packet in lowered.aql.bytes.chunks_exact(AQL_PACKET_BYTES) {
        assert_eq!(&packet[56..64], &0u64.to_le_bytes(), "no AQL packet owns native completion");
    }
    let kernel = &lowered.aql.bytes[AQL_PACKET_BYTES..2 * AQL_PACKET_BYTES];
    assert_eq!(&kernel[32..40], &0x1234_5600u64.to_le_bytes());
    let control = replay_dwords(&lowered.control.bytes);
    assert_eq!(&control[2..5], &[0x2000, 0, 7]);
    assert_eq!(&control[control.len() - 10..control.len() - 8], &crate::amd::sys::pm4::pred_exec(1, 8));
    assert_eq!(&control[control.len() - 5..control.len() - 2], &[0x2000, 0, 8]);

    let mut invalid = submission;
    let Command::Compute(dispatch) = &mut invalid.commands[1] else { unreachable!() };
    dispatch.completion_signal = 0x4000;
    let err = lower_hcq_aql_command_buffer(&invalid).unwrap_err();
    assert!(err.to_string().contains("completion must remain unset"));
}

#[test]
fn hcq_aql_profile_uses_predicated_pm4_timestamps_not_kernel_completion() {
    use crate::hcq::{Command, ComputeDispatch, PatchSource, QueueKind, Submission};

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::MemoryBarrier).push(Command::Compute(ComputeDispatch {
        workgroup_size: [8, 1, 1],
        grid_size: [64, 1, 1],
        private_segment_size: 0,
        group_segment_size: 0,
        kernel_object: 0x1234_5600,
        kernarg_address: 0x9000,
        completion_signal: 0,
        barrier: true,
        amd_pm4: None,
    }));
    let finalized = finalize_hcq_aql_timeline_submission(&submission, 0x2000, 7, 8, Some((0x3000, 0x3008))).unwrap();
    let lowered =
        lower_hcq_aql_submission_program(&finalized, aql_control_state(true), PatchSource::LinkAddress(0)).unwrap();

    assert_eq!(&lowered.aql.bytes[AQL_PACKET_BYTES + 56..2 * AQL_PACKET_BYTES], &0u64.to_le_bytes());
    let control = replay_dwords(&lowered.control.bytes);
    let pred = crate::amd::sys::pm4::packet3(crate::amd::sys::pm4::PACKET3_PRED_EXEC, 0);
    assert_eq!(control.iter().filter(|&&word| word == pred).count(), 3, "start, end, and terminal store");
    assert!(control.windows(2).any(|words| words == crate::amd::sys::pm4::pred_exec(1, 23)));
    assert_eq!(control.iter().filter(|&&word| word == 0x3000).count(), 1);
    assert_eq!(control.iter().filter(|&&word| word == 0x3008).count(), 1);
}

#[test]
fn hcq_aql_single_xcc_control_omits_pred_exec() {
    use crate::hcq::{Command, CommandField, PatchSource, QueueKind, Submission, SystemField};

    let mut submission = Submission::new(QueueKind::Compute(0));
    submission.push(Command::Store { dst: 0, value: 0 });
    submission.bind(0, CommandField::StoreDst, PatchSource::System(SystemField::TimelineSignal(0))).unwrap();
    submission.bind(0, CommandField::StoreValue, PatchSource::System(SystemField::TimelineValue(0))).unwrap();
    let lowered =
        lower_hcq_aql_submission_program(&submission, aql_control_state(false), PatchSource::LinkAddress(0)).unwrap();
    let words = replay_dwords(&lowered.control.bytes);
    let pred = crate::amd::sys::pm4::packet3(crate::amd::sys::pm4::PACKET3_PRED_EXEC, 0);
    assert!(!words.contains(&pred));
}

#[test]
fn sdma_linear_copy_dwords_layout() {
    let dw = crate::amd::sys::sdma::copy_linear(0x1_0000_2000, 0x2_0000_3000, 4096);
    assert_eq!(dw[0], 0x01);
    assert_eq!(dw[1], 4095);
    assert_eq!(dw[3], 0x0000_2000);
    assert_eq!(dw[4], 0x0000_0001);
    assert_eq!(dw[5], 0x0000_3000);
    assert_eq!(dw[6], 0x0000_0002);
}

#[test]
fn sdma_fence_mtype_matches_tinygrad_by_arch() {
    assert_eq!(crate::amd::sys::sdma::fence(0x1_0000_2000, 7, 9), [5, 0x2000, 1, 7]);
    assert_eq!(crate::amd::sys::sdma::fence(0x1_0000_2000, 7, 11), [0x0003_0005, 0x2000, 1, 7]);
}

/// Live SDMA staging roundtrip: a device-local (host_ptr: None) buffer is
/// filled via `_copyin` and read back via `_copyout`, exercising the real SDMA
/// copy + fence + signal-wait path. Skipped without an AMD GPU. A wrong fence
/// fails via the 30 s copy timeout rather than hanging.
#[test]
fn sdma_device_local_roundtrip() {
    use crate::allocator::{Allocator, BufferSpec, RawBuffer};
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    // Bring up signal pool + copy queue (the device factory normally does this);
    // both installers are idempotent, so this is safe if a factory already ran.
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    if core.copy_queue().is_none() {
        core.install_copy_queue(AmdCopyQueue::create(&alloc).expect("copy queue"));
        core.set_has_sdma_queue(true);
    }

    let spec = BufferSpec { cpu_access: false, ..Default::default() };
    // Span > staging size (4 MiB) to exercise multi-chunk staging.
    let n = 5 * 1024 * 1024usize;
    let buf = alloc._alloc(n, &spec, false).expect("device-local alloc");
    assert!(matches!(buf, RawBuffer::AmdDevice { host_ptr: None, .. }), "buffer must be device-only");

    let src: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    alloc._copyin(&buf, 0, &src).expect("copyin");
    let mut out = vec![0u8; n];
    alloc._copyout(&mut out, &buf, 0).expect("copyout");
    assert_eq!(src, out, "SDMA host↔device roundtrip must preserve bytes");

    // Device→device transfer into a second device-local buffer.
    let buf2 = alloc._alloc(n, &spec, false).expect("device-local alloc 2");
    alloc._transfer(&buf2, 0, &buf, 0, n).expect("transfer");
    let mut out2 = vec![0u8; n];
    alloc._copyout(&mut out2, &buf2, 0).expect("copyout 2");
    assert_eq!(src, out2, "SDMA device→device transfer must preserve bytes");

    alloc._free(buf, &spec);
    alloc._free(buf2, &spec);
}

/// Live compute queue creation (exercises the KFD CREATE_QUEUE path).
/// Skipped without a supported AMD GPU. A real dispatch needs the device
/// timeline wired up by the factory, so we only assert creation here.
#[test]
fn compute_queue_create_if_hw_supports() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let _q = AmdComputeQueue::create(&alloc).expect("create compute queue");
}

/// On real AQL hardware (multi-XCC CDNA), `set_aql_scratch` must land the
/// scratch descriptor at the right `amd_queue_t` offsets in the GART page the
/// firmware reads. Exercises the offsets + volatile writes end-to-end against a
/// live queue; on PM4 hardware the queue has no descriptor and the write is a
/// no-op (we skip the assertion there).
#[test]
fn set_aql_scratch_round_trips_through_gart() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let q = AmdComputeQueue::create(&alloc).expect("create compute queue");
    if q.is_pm4() {
        return; // PM4 queues program scratch via registers; no GART descriptor.
    }
    // A realistic descriptor, sized exactly as a 256-byte/thread scratch alloc
    // would be on this device.
    let (va, _size, tmpring, _rounded, _handle, desc) =
        crate::amd::device::alloc_scratch(alloc.dev.core().iface(), &alloc.dev.node, &alloc.dev.arch, 256)
            .expect("alloc scratch");
    assert_ne!(desc, crate::amd::device::AqlScratchDesc::default(), "CDNA must synthesize a descriptor");
    q.set_aql_scratch(&desc);
    assert_eq!(q.read_aql_scratch(), desc, "GART descriptor must match what we wrote");
    // Sanity: the descriptor points at the freshly allocated scratch buffer.
    assert_eq!(desc.backing_va, va);
    assert_eq!(desc.tmpring_size, tmpring);
    // Free the scratch we allocated for the test.
    alloc.dev.core().iface().free_raw(va, _size, _handle);
}
