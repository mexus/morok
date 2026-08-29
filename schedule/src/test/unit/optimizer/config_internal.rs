use super::*;

#[test]
fn test_opt_strategy_default_is_heuristic() {
    assert_eq!(OptStrategy::default(), OptStrategy::Heuristic);
}

#[test]
fn test_opt_strategy_is_none() {
    assert!(OptStrategy::None.is_none());
    assert!(!OptStrategy::Heuristic.is_none());
    assert!(!OptStrategy::Beam { width: 4 }.is_none());
}

#[test]
fn test_opt_strategy_is_beam() {
    assert!(!OptStrategy::None.is_beam());
    assert!(!OptStrategy::Heuristic.is_beam());
    assert!(OptStrategy::Beam { width: 4 }.is_beam());
}

#[test]
fn test_beam_config_default() {
    let config = BeamConfig::default();
    assert_eq!(config.beam_width, 4);
    assert_eq!(config.max_upcast, 256);
    assert_eq!(config.max_local, 1024);
    assert_eq!(config.min_progress_ns, 10);
    assert!(!config.enable_nolocals);
    assert_eq!(config.compile_workers, 0);
    assert_eq!(config.max_tasks_per_child, 16);
    assert_eq!(config.compile_timeout_secs, 10);
}

#[test]
fn test_beam_min_progress_matches_tinygrad_microseconds_env() {
    assert_eq!(parse_beam_min_progress(None), 10);
    assert_eq!(parse_beam_min_progress(Some("0.01")), 10);
    assert_eq!(parse_beam_min_progress(Some("1")), 1_000);
    assert_eq!(parse_beam_min_progress(Some("invalid")), 10);
}

#[test]
fn test_beam_config_builder() {
    let config = BeamConfig::builder()
        .beam_width(8)
        .max_upcast(512)
        .min_progress_ns(25)
        .enable_nolocals(true)
        .compile_workers(3)
        .max_tasks_per_child(5)
        .compile_timeout_secs(7)
        .build();

    assert_eq!(config.beam_width, 8);
    assert_eq!(config.max_upcast, 512);
    assert_eq!(config.max_local, 1024); // default
    assert_eq!(config.min_progress_ns, 25);
    assert!(config.enable_nolocals);
    assert_eq!(config.compile_workers, 3);
    assert_eq!(config.max_tasks_per_child, 5);
    assert_eq!(config.compile_timeout_secs, 7);
}

#[test]
fn test_heuristics_config_default() {
    let config = HeuristicsConfig::default();
    assert_eq!(config.tc_enabled, TcUsage::Enabled);
    assert_eq!(config.tc_opt, TcOpt::Strict);
    assert!(config.matvec_enabled);
    assert_eq!(config.threads_per_row, 8);
    assert_eq!(config.rows_per_thread, 4);
    assert_eq!(config.grouped_threshold, 256);
}

#[test]
fn test_heuristics_config_builder() {
    let config = HeuristicsConfig::builder()
        .tc_enabled(TcUsage::Disabled)
        .matvec_enabled(false)
        .threads_per_row(16)
        .rows_per_thread(2)
        .grouped_threshold(128)
        .build();

    assert_eq!(config.tc_enabled, TcUsage::Disabled);
    assert!(!config.matvec_enabled);
    assert_eq!(config.threads_per_row, 16);
    assert_eq!(config.rows_per_thread, 2);
    assert_eq!(config.grouped_threshold, 128);
}

#[test]
fn test_optimizer_config_default() {
    let config = OptimizerConfig::default();
    assert_eq!(config.strategy, OptStrategy::Heuristic);
    assert_eq!(config.beam.beam_width, 4);
    // tinygrad `helpers.py:245`: DISABLE_FAST_IDIV defaults to 1.
    assert!(config.disable_fast_idiv);
}

/// `disable_fast_idiv` gates the magic-multiply rewrite in the late pattern set.
#[test_case::test_case(true, true; "disabled keeps cdiv")]
#[test_case::test_case(false, false; "enabled rewrites cdiv")]
fn test_disable_fast_idiv_gates_late_rewrites(disable_fast_idiv: bool, expect_cdiv: bool) {
    use svod_ir::{BinaryOp, DType, Op, UOp};

    let x = UOp::var("x", DType::Int32, 0, 255);
    let cdiv = UOp::new(Op::Binary(BinaryOp::CDiv, x, UOp::native_const(3i32)), DType::Int32);
    let renderer = crate::optimizer::Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let patterns = crate::optimizer::get_late_rewrite_patterns(&renderer, disable_fast_idiv);
    let rewritten = crate::rewrite::graph_rewrite(&patterns, cdiv, &mut ());

    let has_cdiv = rewritten.toposort().iter().any(|u| matches!(u.op(), Op::Binary(BinaryOp::CDiv, ..)));
    assert_eq!(has_cdiv, expect_cdiv, "{}", rewritten.tree());
}

#[test]
fn test_optimizer_config_builder() {
    let config = OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 8 })
        .beam(BeamConfig::builder().max_upcast(512).build())
        .build();

    assert_eq!(config.strategy, OptStrategy::Beam { width: 8 });
    assert_eq!(config.beam.beam_width, 8);
    assert_eq!(config.beam.max_upcast, 512);
}

#[test]
fn test_tc_usage_as_usize() {
    assert_eq!(TcUsage::Disabled.as_usize(), 0);
    assert_eq!(TcUsage::Enabled.as_usize(), 1);
    assert_eq!(TcUsage::ShapeOnly.as_usize(), 2);
}

#[test]
fn test_tc_opt_as_usize() {
    assert_eq!(TcOpt::Strict.as_usize(), 0);
    assert_eq!(TcOpt::Relaxed.as_usize(), 1);
    assert_eq!(TcOpt::Padded.as_usize(), 2);
}

#[test]
fn test_tc_select_as_i32() {
    assert_eq!(TcSelect::Auto.as_i32(), -1);
    assert_eq!(TcSelect::Index(5).as_i32(), 5);
}
