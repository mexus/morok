use super::super::types::{OptArgExt, OptOps};
use super::*;

#[test]
fn test_beam_config_default() {
    let config = BeamConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.max_upcast, 256);
    assert_eq!(config.max_local, 1024);
}

#[test]
fn test_beam_actions_not_empty() {
    assert!(!BEAM_ACTIONS.is_empty());
    // Should have a reasonable number of actions
    // UPCAST: 8 axes * 6 amounts = 48
    // UNROLL: 5 axes * 3 amounts = 15
    // LOCAL: 6 axes * 7 amounts = 42
    // GROUPTOP: 3 axes * 8 amounts = 24
    // GROUP: 3 axes * 4 amounts = 12
    // TC: 1 + 9 = 10
    // SWAP: 10 pairs
    // NOLOCALS: 1
    // Total: ~162 actions
    assert!(BEAM_ACTIONS.len() > 100, "Expected >100 actions, got {}", BEAM_ACTIONS.len());
    assert!(BEAM_ACTIONS.len() < 500, "Expected <500 actions, got {}", BEAM_ACTIONS.len());
}

#[test]
fn test_beam_actions_contains_expected_types() {
    let has_upcast = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::UPCAST);
    let has_local = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::LOCAL);
    let has_unroll = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::UNROLL);
    let has_tc = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::TC);
    let has_swap = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::SWAP);

    assert!(has_upcast);
    assert!(has_local);
    assert!(has_unroll);
    assert!(has_tc);
    assert!(has_swap);
    // NOLOCALS is env-gated (`SVOD_NOLOCALS`), tested separately.
}

#[test]
fn test_beam_action_grid_uses_tinygrad_amount_major_order() {
    let upcasts = BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::UPCAST).take(9).collect::<Vec<_>>();
    assert_eq!(
        upcasts.iter().take(8).map(|action| action.axis).collect::<Vec<_>>(),
        (0..8).map(Some).collect::<Vec<_>>()
    );
    assert!(upcasts.iter().take(8).all(|action| action.arg.int() == Ok(0)));
    assert_eq!(upcasts[8].axis, Some(0));
    assert_eq!(upcasts[8].arg.int(), Ok(2));
}

#[test]
fn test_beam_tensor_core_actions_keep_strict_default_and_padded_axes() {
    let tensor_core_actions = BEAM_ACTIONS.iter().filter(|action| action.op == OptOps::TC).collect::<Vec<_>>();
    assert_eq!(tensor_core_actions.len(), 10);
    assert!(tensor_core_actions.iter().any(|action| action.arg.tc().unwrap().1 == 0));
    assert_eq!(tensor_core_actions.iter().filter(|action| action.arg.tc().unwrap().1 == 2).count(), 9);
}

#[test]
fn test_beam_cache_key_includes_post_optimization_behavior() {
    let scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1i32)]), crate::optimizer::Renderer::cpu());
    let config = BeamConfig::default();
    assert_ne!(
        CacheKey::from_scheduler(&scheduler, &config, "compiler", 0).to_bytes(),
        CacheKey::from_scheduler(&scheduler, &config, "compiler", 1).to_bytes()
    );
}

#[test]
fn test_beam_cache_key_includes_exact_compiler_identity() {
    let scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1i32)]), crate::optimizer::Renderer::cpu());
    let config = BeamConfig::default();
    assert_ne!(
        CacheKey::from_scheduler(&scheduler, &config, "cpu-clang:17", 0).to_bytes(),
        CacheKey::from_scheduler(&scheduler, &config, "cpu-clang:18", 0).to_bytes()
    );
}

#[test]
fn test_beam_cache_key_includes_behavior_controls() {
    let scheduler = Scheduler::new(UOp::sink(vec![UOp::native_const(1i32)]), crate::optimizer::Renderer::cpu());
    let base = BeamConfig::default();
    let base_key = CacheKey::from_scheduler(&scheduler, &base, "compiler", 0).to_bytes();
    let variants = [
        BeamConfig { min_progress_ns: base.min_progress_ns + 1, ..base.clone() },
        BeamConfig { enable_nolocals: !base.enable_nolocals, ..base.clone() },
        BeamConfig { compile_timeout_secs: base.compile_timeout_secs + 1, ..base.clone() },
        BeamConfig { num_runs: base.num_runs + 1, ..base.clone() },
    ];
    for variant in variants {
        assert_ne!(base_key, CacheKey::from_scheduler(&scheduler, &variant, "compiler", 0).to_bytes());
    }
    let parallel = BeamConfig { compile_workers: base.compile_workers + 1, ..base.clone() };
    assert_eq!(base_key, CacheKey::from_scheduler(&scheduler, &parallel, "compiler", 0).to_bytes());
    let recycling = BeamConfig { max_tasks_per_child: base.max_tasks_per_child + 1, ..base.clone() };
    assert_eq!(base_key, CacheKey::from_scheduler(&scheduler, &recycling, "compiler", 0).to_bytes());
}

#[test]
fn test_remote_beam_parent_tracks_only_opt_sequences() {
    let scheduler = weak_axis_scheduler(0x4b31);
    let config =
        BeamConfig { beam_width: 2, min_progress_ns: 1_000_000_000, disable_cache: true, ..Default::default() };
    let base_opt_count = scheduler.applied_opts.len();
    let worker_scheduler = scheduler.clone();
    let result = beam_search_remote_staged(
        scheduler,
        &config,
        |candidates, emit| {
            assert!(candidates.iter().all(|opts| opts.len() == base_opt_count + 1));
            for (index, opts) in candidates.iter().enumerate() {
                if apply_remote_candidate(worker_scheduler.clone(), base_opt_count, opts, &config).is_some() {
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: index,
                            binary_key: index.to_le_bytes().to_vec(),
                            compute_ops: 1,
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            }
            Ok(())
        },
        |index, _| Some(Duration::from_nanos(10_000 - *index as u64)),
    )
    .unwrap();
    assert_eq!(result.iterations, 1);
    assert_eq!(result.scheduler.applied_opts.len(), base_opt_count + 1);
    assert!(result.compiled > 0);
}

#[test]
fn test_beam_cache_key_distinguishes_exact_amd_targets() {
    use svod_dtype::AmdArch;

    let ast = UOp::sink(vec![UOp::native_const(1i32)]);
    let config = BeamConfig::default();
    let gfx1100 = Scheduler::new(ast.clone(), crate::optimizer::Renderer::for_amd_arch(AmdArch::Gfx1100));
    let gfx1151 = Scheduler::new(ast, crate::optimizer::Renderer::for_amd_arch(AmdArch::Gfx1151));
    assert_ne!(
        CacheKey::from_scheduler(&gfx1100, &config, "amd", 0).to_bytes(),
        CacheKey::from_scheduler(&gfx1151, &config, "amd", 0).to_bytes()
    );
}

fn weak_axis_scheduler(constant: i32) -> Scheduler {
    use svod_ir::{AxisId, AxisType};

    let range = UOp::range_axis(UOp::index_const(64), AxisId::Renumbered(0), AxisType::Weak);
    Scheduler::new(UOp::sink(vec![UOp::native_const(constant), range]), crate::optimizer::Renderer::cpu())
}

#[test]
fn test_staged_beam_streams_unordered_compiles_dedups_and_serializes_timing() {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct FakeArtifact {
        index: usize,
    }

    let scheduler = weak_axis_scheduler(0x51a9);
    let config = BeamConfig {
        beam_width: 2,
        min_progress_ns: 1_000_000_000,
        compile_workers: 3,
        disable_cache: true,
        ..BeamConfig::default()
    };
    let opts_by_index = Arc::new(Mutex::new(HashMap::new()));
    let compile_calls = Arc::new(AtomicUsize::new(0));
    let benchmark_calls = Arc::new(AtomicUsize::new(0));
    let benchmark_active = Arc::new(AtomicUsize::new(0));
    let benchmark_max = Arc::new(AtomicUsize::new(0));

    let result = beam_search_staged(
        scheduler,
        &config,
        {
            let opts_by_index = Arc::clone(&opts_by_index);
            let calls = Arc::clone(&compile_calls);
            move |candidates: &[Scheduler], emit: &mut dyn FnMut(usize, CompiledCandidate<FakeArtifact>)| {
                for index in (0..candidates.len()).rev() {
                    calls.fetch_add(1, Ordering::SeqCst);
                    opts_by_index.lock().unwrap().insert(index, candidates[index].applied_opts.clone());
                    let binary_key = if matches!(index, 3 | 4) { vec![0xdd] } else { index.to_le_bytes().to_vec() };
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: FakeArtifact { index },
                            binary_key,
                            compute_ops: if index == 2 { 1001 } else { 1 },
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            }
        },
        {
            let calls = Arc::clone(&benchmark_calls);
            let active = Arc::clone(&benchmark_active);
            let maximum = Arc::clone(&benchmark_max);
            move |artifact: &FakeArtifact, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                maximum.fetch_max(now, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(1));
                active.fetch_sub(1, Ordering::SeqCst);
                Some(Duration::from_nanos(10_000 - artifact.index as u64))
            }
        },
    )
    .unwrap();

    let generated = compile_calls.load(Ordering::SeqCst);
    assert!(generated > 6, "test scheduler must expose enough candidates");
    assert_eq!(result.generated, generated);
    assert_eq!(result.unique_ir, 0);
    assert_eq!(result.compiled, compile_calls.load(Ordering::SeqCst));
    assert_eq!(
        result.unique_binary,
        result.compiled - 2,
        "one excessive-compute and one duplicate binary must be removed"
    );
    assert_eq!(benchmark_calls.load(Ordering::SeqCst), result.unique_binary);
    assert_eq!(result.benchmarked, result.unique_binary);
    assert_eq!(benchmark_max.load(Ordering::SeqCst), 1, "backend timing must be serialized");

    let winning_index = opts_by_index
        .lock()
        .unwrap()
        .keys()
        .copied()
        .filter(|index| *index != 1 && *index != 2 && *index != 4)
        .max()
        .unwrap();
    assert_eq!(result.scheduler.applied_opts, opts_by_index.lock().unwrap()[&winning_index]);
}

#[test]
fn test_staged_beam_cache_cold_and_warm_choose_same_winner() {
    use std::hash::{Hash, Hasher};

    let scheduler = weak_axis_scheduler(0x6b17);
    let config = BeamConfig {
        min_progress_ns: 1_000_000_000,
        compile_workers: 2,
        disable_cache: false,
        ..BeamConfig::default()
    };
    let compiler_identity = "fake-compiler:beam-cold-warm-v1";
    let key = CacheKey::from_scheduler(&scheduler, &config, compiler_identity, 0x1234);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return; // Another Svod process may hold sled's exclusive database lock.
    }

    let run = |scheduler: Scheduler| {
        beam_search_cached_staged(
            scheduler,
            &config,
            compiler_identity,
            0x1234,
            |candidates, emit| {
                for (index, candidate) in candidates.iter().enumerate() {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    candidate.applied_opts.hash(&mut hasher);
                    let identity = hasher.finish();
                    emit(
                        index,
                        CompiledCandidate {
                            artifact: identity,
                            binary_key: identity.to_le_bytes().to_vec(),
                            compute_ops: 1,
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
            },
            |identity, _| Some(Duration::from_nanos(1 + identity % 10_000)),
        )
        .unwrap()
    };

    let cold = run(scheduler.clone());
    let warm = run(scheduler);
    cache_invalidate(&key);

    assert!(cold.iterations > 0);
    assert_eq!(warm.iterations, 0, "second search should replay the persistent BEAM entry");
    assert_eq!(cold.scheduler.applied_opts, warm.scheduler.applied_opts);
    assert_eq!(cold.timing, warm.timing);
}

#[test]
fn test_remote_beam_cache_reuses_winner_across_parallel_and_recycling_changes() {
    use std::hash::{Hash, Hasher};

    let scheduler = weak_axis_scheduler(0x7193);
    let cold_config = BeamConfig {
        min_progress_ns: 1_000_000_000,
        compile_workers: 1,
        max_tasks_per_child: 1,
        disable_cache: false,
        ..Default::default()
    };
    let warm_config = BeamConfig { compile_workers: 8, max_tasks_per_child: 99, ..cold_config.clone() };
    let identity = "fake-compiler:remote-cache-v1";
    let key = CacheKey::from_scheduler(&scheduler, &cold_config, identity, 0x22);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return;
    }

    let run = |config: &BeamConfig| {
        let worker_scheduler = scheduler.clone();
        let base_opt_count = scheduler.applied_opts.len();
        beam_search_cached_remote(
            scheduler.clone(),
            config,
            identity,
            0x22,
            |candidates, emit| {
                for (index, opts) in candidates.iter().enumerate() {
                    if apply_remote_candidate(worker_scheduler.clone(), base_opt_count, opts, config).is_none() {
                        continue;
                    }
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    opts.hash(&mut hasher);
                    let artifact = hasher.finish();
                    emit(
                        index,
                        CompiledCandidate {
                            artifact,
                            binary_key: artifact.to_le_bytes().to_vec(),
                            compute_ops: 1,
                            preparation: Duration::ZERO,
                            compilation: Duration::ZERO,
                        },
                    );
                }
                Ok(())
            },
            |artifact, _| Some(Duration::from_nanos(1 + artifact % 10_000)),
        )
        .unwrap()
    };
    let cold = run(&cold_config);
    let warm = run(&warm_config);
    cache_invalidate(&key);
    assert!(cold.iterations > 0);
    assert_eq!(warm.iterations, 0);
    assert_eq!(cold.scheduler.applied_opts, warm.scheduler.applied_opts);
    assert_eq!(cold.timing, warm.timing);
}

#[test]
fn test_remote_beam_does_not_cache_unbenchmarked_search() {
    let scheduler = weak_axis_scheduler(0x7a21);
    let config = BeamConfig { min_progress_ns: 1_000_000_000, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:remote-no-empty-cache-v1";
    let key = CacheKey::from_scheduler(&scheduler, &config, identity, 0x31);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return;
    }

    let failed = beam_search_cached_remote(
        scheduler.clone(),
        &config,
        identity,
        0x31,
        |_candidates, _emit: &mut dyn FnMut(usize, CompiledCandidate<usize>)| Ok(()),
        |_artifact, _| Some(Duration::from_nanos(1)),
    )
    .unwrap();
    assert_eq!(failed.benchmarked, 0);
    assert_eq!(failed.timing, Duration::MAX);
    assert!(cache_get(&key).is_none());

    let cold = beam_search_cached_remote(
        scheduler,
        &config,
        identity,
        0x31,
        |candidates, emit| {
            for index in 0..candidates.len() {
                emit(
                    index,
                    CompiledCandidate {
                        artifact: index,
                        binary_key: index.to_le_bytes().to_vec(),
                        compute_ops: 1,
                        preparation: Duration::ZERO,
                        compilation: Duration::ZERO,
                    },
                );
            }
            Ok(())
        },
        |artifact, _| Some(Duration::from_nanos(10_000 - *artifact as u64)),
    )
    .unwrap();
    assert!(cold.iterations > 0, "the failed search must not create a cache hit");
    assert!(cache_get(&key).is_some());
    cache_invalidate(&key);
}

#[test]
fn test_remote_beam_worker_error_invalidates_cache() {
    let scheduler = weak_axis_scheduler(0x7a22);
    let config = BeamConfig { min_progress_ns: 1_000_000_000, disable_cache: false, ..Default::default() };
    let identity = "fake-compiler:remote-worker-error-v1";
    let key = CacheKey::from_scheduler(&scheduler, &config, identity, 0x32);
    cache_invalidate(&key);
    if CACHE_DB.is_none() {
        return;
    }

    cache_put(&key, &[Opt::upcast(0, 2)]);
    let result = beam_search_cached_remote(
        scheduler,
        &config,
        identity,
        0x32,
        |_candidates, _emit: &mut dyn FnMut(usize, CompiledCandidate<usize>)| {
            Err(OptError::BeamWorker { message: "disconnected".into() })
        },
        |_artifact, _| Some(Duration::from_nanos(1)),
    );
    assert!(matches!(result, Err(OptError::BeamWorker { .. })));
    assert!(cache_get(&key).is_none());
}

#[test]
fn test_beam_search_with_mock_scoring() {
    use super::super::renderer::Renderer;
    use svod_ir::UOp;

    // Create a simple scheduler
    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);
    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    let config = BeamConfig { beam_width: 2, ..Default::default() };

    // Mock scoring: return constant timing + a hash that varies by scheduler
    // pointer so dedup doesn't collapse every candidate to one entry.
    let mock_score = |s: &Scheduler, _early_stop: Option<Duration>| {
        Some(CandidateMetrics {
            timing: Duration::from_micros(100),
            ir_hash: s as *const Scheduler as u64,
            compute_ops: 1,
        })
    };

    let result = beam_search(scheduler, &config, mock_score);
    assert!(result.is_ok());

    let result = result.unwrap();
    assert!(result.iterations > 0 || result.candidates_evaluated == 0);
}

#[test]
fn test_validate_limits() {
    use super::super::renderer::Renderer;
    use svod_ir::UOp;

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);
    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    let config = BeamConfig::default();

    // Simple scheduler should pass limits
    assert!(validate_limits(&scheduler, &config));

    // With very restrictive limits
    let strict_config = BeamConfig { max_upcast: 1, max_local: 1, max_uops: 1, ..Default::default() };

    // May or may not pass depending on UOp count
    let _result = validate_limits(&scheduler, &strict_config);
}

#[test]
fn test_replay_opts_empty() {
    use super::super::renderer::Renderer;
    use svod_ir::UOp;

    let val = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![val]);
    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    // Empty replay should succeed
    let result = replay_opts(scheduler, &[]);
    assert!(result.is_ok());
}

#[test]
fn test_serialize_deserialize_opts_empty() {
    let opts: Vec<Opt> = vec![];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    assert!(deserialized.unwrap().is_empty());
}

#[test]
fn test_serialize_deserialize_opts_upcast() {
    let opts = vec![Opt::upcast(0, 4), Opt::upcast(1, 8)];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].op, OptOps::UPCAST);
    assert_eq!(result[0].axis, Some(0));
    assert_eq!(result[1].op, OptOps::UPCAST);
    assert_eq!(result[1].axis, Some(1));
}

#[test]
fn test_serialize_deserialize_opts_tc() {
    use super::super::types::OptArg;

    let opts = vec![Opt::tc(None, -1, 2, 1)];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].op, OptOps::TC);
    assert_eq!(result[0].axis, None);
    if let OptArg::TensorCore { tc_select, opt_level, use_tc } = &result[0].arg {
        assert_eq!(*tc_select, -1);
        assert_eq!(*opt_level, 2);
        assert_eq!(*use_tc, 1);
    } else {
        panic!("Expected TensorCore arg");
    }
}

#[test]
fn test_serialize_deserialize_opts_swap() {
    use super::super::types::OptArg;

    let opts = vec![Opt::swap(0, 2)];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].op, OptOps::SWAP);
    assert_eq!(result[0].axis, Some(0));
    if let OptArg::Swap { other_axis } = &result[0].arg {
        assert_eq!(*other_axis, 2);
    } else {
        panic!("Expected Swap arg");
    }
}

#[test]
fn test_serialize_deserialize_opts_mixed() {
    let opts = vec![Opt::upcast(0, 4), Opt::local(1, 16), Opt::unroll(0, 8), Opt::nolocals()];
    let serialized = serialize_opts(&opts);
    let deserialized = deserialize_opts(&serialized);

    assert!(deserialized.is_some());
    let result = deserialized.unwrap();
    assert_eq!(result.len(), 4);
    assert_eq!(result[0].op, OptOps::UPCAST);
    assert_eq!(result[1].op, OptOps::LOCAL);
    assert_eq!(result[2].op, OptOps::UNROLL);
    assert_eq!(result[3].op, OptOps::NOLOCALS);
}

#[test]
fn test_beam_actions_contains_thread() {
    let has_thread = BEAM_ACTIONS.iter().any(|a| a.op == OptOps::THREAD);
    assert!(has_thread, "BEAM_ACTIONS should contain THREAD actions");

    // Count thread actions
    let thread_count = BEAM_ACTIONS.iter().filter(|a| a.op == OptOps::THREAD).count();
    assert!(thread_count >= 6, "Expected at least 6 THREAD actions (3 axes × 2+ amounts), got {}", thread_count);
}

#[test]
fn test_thread_action_applied_to_loop_axis() {
    use super::super::renderer::Renderer;
    use svod_ir::{AxisId, AxisType, UOp};

    // Create a kernel with Weak axis (CPU threading target)
    let end_64 = UOp::index_const(64);
    let r_loop = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Weak);
    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_loop]);

    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    // Verify renderer supports threading
    assert!(scheduler.renderer().has_threads, "CPU renderer should have has_threads=true");

    // Try to apply THREAD opt with a divisor that fits available parallelism.
    let max_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let thread_count = [32usize, 16, 8, 4, 2].into_iter().find(|&t| t <= max_threads && 64 % t == 0).unwrap_or(1);
    if thread_count == 1 {
        return;
    }
    let mut test_scheduler = scheduler.clone();
    let result = apply_opt(&mut test_scheduler, &Opt::thread(0, thread_count), true);
    assert!(result.is_ok(), "THREAD(0, {}) should succeed on Weak axis: {:?}", thread_count, result);

    // Verify Thread axis was created
    let thread_axes = test_scheduler.axes_of(&[AxisType::Thread]);
    assert!(!thread_axes.is_empty(), "Should have Thread axis after THREAD opt");
}

#[test]
fn test_generate_actions_includes_thread_for_cpu() {
    use super::super::renderer::Renderer;
    use svod_ir::{AxisId, AxisType, UOp};

    // Create a kernel with Weak axis
    let end_64 = UOp::index_const(64);
    let r_loop = UOp::range_axis(end_64, AxisId::Renumbered(0), AxisType::Weak);
    let compute = UOp::native_const(1.0f32);
    let sink = UOp::sink(vec![compute, r_loop]);

    let renderer = Renderer::cpu();
    let scheduler = Scheduler::new(sink, renderer);

    let config = BeamConfig::default();
    let candidates = generate_actions(&scheduler, &config);

    // Check if any candidate has a Thread axis
    let has_threaded = candidates.iter().any(|s| !s.axes_of(&[AxisType::Thread]).is_empty());
    assert!(has_threaded, "generate_actions should produce candidates with Thread axes for CPU");
}
