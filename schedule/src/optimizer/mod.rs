//! Kernel optimization layer for svod-schedule.
//!
//! Provides a `Scheduler` that applies optimization primitives (OptOps) to transform
//! kernel execution for better performance on specific backends.
//!
//! # Architecture
//!
//! The optimization process follows this flow:
//!
//! 1. **Initialization**: Create `Scheduler` from UOp AST + `Renderer` (backend capabilities)
//! 2. **Initial Transform**: Convert eligible LOOP axes to GLOBAL (parallelization)
//! 3. **Optimization**: Apply `Opt` operations via `apply_opt()`
//!    - UPCAST: Vectorization (SIMD)
//!    - LOCAL: GPU workgroup dimensions (shared memory)
//!    - UNROLL: Loop unrolling for reductions
//!    - GROUP: Two-stage reductions with synchronization
//!    - TC: Tensor core acceleration
//!    - PADTO, SWAP, THREAD, NOLOCALS: Layout and configuration
//! 4. **Finalization**: Extract optimized AST with `get_optimized_ast()`
//!
//! # Optimization Strategies
//!
//! - **Hand-coded heuristics** (`heuristics` module): Fast, reasonable performance
//! - **Beam search** (`beam` module, optional): Slow, ML-quality performance
//!
//! # Example
//!
//! ```ignore
//! use svod_schedule::optimizer::{Scheduler, Renderer, Opt, OptOps};
//!
//! // Create scheduler with CUDA backend
//! let renderer = Renderer::cuda();
//! let mut scheduler = Scheduler::new(kernel_ast, renderer);
//!
//! // Apply optimizations
//! scheduler.convert_loop_to_global();
//! scheduler.apply_opt(Opt::upcast(0, 4), true)?; // Stack axis 0 by 4
//! scheduler.apply_opt(Opt::local(1, 16), true)?; // Local memory for axis 1
//!
//! // Get optimized kernel
//! let optimized_ast = scheduler.get_optimized_ast(None);
//! ```

pub mod beam;
pub mod config;
pub mod error;
pub mod heuristics;
mod implicit_barriers;
pub mod kernel_info;
pub mod opts;
pub mod renderer;
pub mod scheduler;
pub mod tc;
pub mod types;

// Re-exports
pub use beam::{
    BeamResult, CandidateMetrics, apply_remote_candidate, beam_search, beam_search_cached, beam_search_cached_remote,
    beam_search_cached_with_behavior, clear_cache, compute_ops_estimate, hash_post_codegen_ir, replay_opts,
};
pub use config::{BeamConfig, HeuristicsConfig, OptStrategy, OptimizerConfig, TcOpt as TcOptLevel, TcSelect, TcUsage};
pub use error::OptError;
pub use heuristics::hand_coded_optimizations;
pub use kernel_info::KernelInfo;
pub use opts::apply_opt;
pub use renderer::{Renderer, TcOpt, TensorCore};
pub use scheduler::Scheduler;
#[cfg(test)]
pub use scheduler::clear_kernel_name_counts;
pub use types::{AxisType, Opt, OptArg, OptArgExt, OptOps};

use crate::devectorize::{
    Fp8DecompCtx, pm_add_loads, pm_expand_broadcast, pm_float_decomp, pm_long_decomp, pm_reduce_local,
};
use crate::gpudims::pm_add_gpudims;
use crate::rangeify::patterns::{
    pm_comparison_negations, pm_demorgan, pm_div_to_shr, pm_fdiv_to_mul, pm_fma_decomposition, pm_half_bf16_cast,
    pm_load_collapse, pm_mod_to_and, pm_mul_to_shl, pm_neg_from_mul, pm_shl_add_to_mulacc, pm_threefry_decomp,
};
use crate::rangeify::transforms::{SimplifyRangesContext, pm_flatten_range, pm_simplify_ranges, pm_split_ranges};
use crate::rewrite::graph_rewrite;
use crate::symbolic::patterns::{pm_fold_cast_const, sym, symbolic, symbolic_simple};
use implicit_barriers::pm_implicit_barriers;
use std::collections::HashSet;
use std::sync::{Arc, LazyLock};
use svod_ir::{AxisId, Op, UOp};

#[derive(Default)]
struct LocalBufferContext {
    next_fallback_slot: usize,
    used_slots: HashSet<usize>,
}

impl LocalBufferContext {
    fn axis_slot(axis: &AxisId) -> usize {
        if axis.path().len() == 1 {
            return axis.value();
        }

        // Preserve Tinygrad's numeric slot for scalar axes. Nested Rust axes
        // occupy a separate deterministic namespace and retain their full path.
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for value in std::iter::once(axis.is_renumbered() as usize).chain(axis.path().iter().copied()) {
            for byte in value.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x100_0000_01b3);
            }
        }
        (1usize << (usize::BITS - 1)) | (hash as usize & (usize::MAX >> 1))
    }

    fn allocate(&mut self, axis: Option<&AxisId>) -> usize {
        let mut slot = axis.map(Self::axis_slot).unwrap_or(self.next_fallback_slot);
        while !self.used_slots.insert(slot) {
            slot = slot.wrapping_add(1);
        }
        if axis.is_none() {
            self.next_fallback_slot = slot.wrapping_add(1);
        }
        slot
    }
}

fn add_local_buffer(stage: &Arc<UOp>, ctx: &mut LocalBufferContext) -> Option<Arc<UOp>> {
    let Op::Stage { compute, ranges, opts } = stage.op() else { return None };
    let shape = stage.shape().ok().flatten()?.clone();
    let max_shape =
        shape.iter().map(|dim| dim.vmax().map(svod_ir::SInt::Const)).collect::<Option<svod_ir::shape::Shape>>()?;
    let slot = ctx.allocate(opts.local_axis.as_ref());
    let buffer = UOp::placeholder(&max_shape, stage.dtype(), slot, opts.addrspace, None).ok()?;
    let index = UOp::index().buffer(buffer.clone()).indices(ranges.to_vec()).call().ok()?;
    let store = index.store(compute.clone());
    let end = store.end(ranges.clone());
    Some(buffer.after(smallvec::smallvec![end]))
}

fn pm_add_local_buffers() -> crate::TypedPatternMatcher<LocalBufferContext> {
    let add_local_buffers = crate::patterns! {
        @context LocalBufferContext;
        stage @ Stage { compute: _ } => |stage, ctx| add_local_buffer(stage, ctx),
    };
    add_local_buffers + crate::rangeify::patterns::movement_op_patterns().with_context::<LocalBufferContext>()
}

#[cfg(test)]
mod stage_local_tests {
    use super::*;
    use svod_dtype::{AddrSpace, DType};

    #[test]
    fn add_local_buffer_matches_stage_mapping_and_numbering() {
        let r0 = UOp::range_const(2, 0);
        let r1 = UOp::range_const(3, 1);
        let compute = UOp::native_const(7.0f32);
        let stage = UOp::stage_local(compute.clone(), vec![r0.clone(), r1.clone()]);
        assert_eq!(
            stage.shape().unwrap().unwrap().iter().map(|x| x.as_const()).collect::<Vec<_>>(),
            [Some(2), Some(3)]
        );

        let mut ctx = LocalBufferContext::default();
        let lowered = add_local_buffer(&stage, &mut ctx).unwrap();
        assert_eq!(lowered.dtype(), DType::Float32);

        let Op::After { passthrough, deps } = lowered.op() else { panic!("expected AFTER") };
        let storage = passthrough.base();
        assert!(matches!(storage.op(), Op::Buffer { arg, .. }
            if arg.slot == 0 && arg.dtype == DType::Float32 && arg.addrspace == Some(AddrSpace::Local)));
        let [end] = deps.as_slice() else { panic!("expected one dependency") };
        let Op::End { computation, ranges } = end.op() else { panic!("expected END") };
        assert!(ranges.iter().zip([&r0, &r1]).all(|(actual, expected)| Arc::ptr_eq(actual, expected)));
        let Op::Store { index, value, gate: None } = computation.op() else { panic!("expected STORE") };
        assert!(Arc::ptr_eq(value, &compute));
        let Op::Index { buffer, indices } = index.op() else { panic!("expected INDEX") };
        assert!(Arc::ptr_eq(buffer, passthrough));
        assert!(indices.iter().zip([&r0, &r1]).all(|(actual, expected)| Arc::ptr_eq(actual, expected)));

        let second = UOp::stage_local(UOp::native_const(8.0f32), vec![]);
        let second = add_local_buffer(&second, &mut ctx).unwrap();
        assert!(matches!(second.buf_uop().op(), Op::Buffer { arg, .. } if arg.slot == 1));
    }

    #[test]
    fn grouped_local_axis_drives_slot_without_colliding_with_nested_axes() {
        let scalar_axis = AxisId::Renumbered(7);
        let nested_axis = scalar_axis.child(0);
        let scalar =
            UOp::stage(UOp::native_const(1.0f32), vec![], svod_ir::BufferizeOpts::local_for_axis(scalar_axis.clone()));
        let nested =
            UOp::stage(UOp::native_const(2.0f32), vec![], svod_ir::BufferizeOpts::local_for_axis(nested_axis.clone()));
        assert!(!Arc::ptr_eq(&scalar, &nested));

        let mut ctx = LocalBufferContext::default();
        let scalar = add_local_buffer(&scalar, &mut ctx).unwrap();
        let nested = add_local_buffer(&nested, &mut ctx).unwrap();
        let scalar_slot = match scalar.buf_uop().op() {
            Op::Buffer { arg, .. } => arg.slot,
            _ => unreachable!(),
        };
        let nested_slot = match nested.buf_uop().op() {
            Op::Buffer { arg, .. } => arg.slot,
            _ => unreachable!(),
        };
        assert_eq!(scalar_slot, 7);
        assert_ne!(nested_slot, scalar_slot);
        assert_eq!(nested_slot, LocalBufferContext::axis_slot(&nested_axis));
    }

    #[test]
    fn grouped_local_slots_repeat_across_kernel_rewrites() {
        fn lower() -> Vec<usize> {
            let stages = [AxisId::Renumbered(3), AxisId::Renumbered(3).child(1)]
                .into_iter()
                .enumerate()
                .map(|(value, axis)| {
                    UOp::stage(UOp::native_const(value as f32), vec![], svod_ir::BufferizeOpts::local_for_axis(axis))
                })
                .collect::<Vec<_>>();
            let mut ctx = LocalBufferContext::default();
            stages
                .iter()
                .map(|stage| match add_local_buffer(stage, &mut ctx).unwrap().buf_uop().op() {
                    Op::Buffer { arg, .. } => arg.slot,
                    _ => unreachable!(),
                })
                .collect()
        }
        assert_eq!(lower(), lower());
    }
}

// Tinygrad 8c8b43de codegen/__init__.py:314. Keep source order because the
// unified rewrite lets later patterns consume nodes rewritten by earlier ones.
pub(crate) static POST_OPT_SYM: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
    sym().clone()
        + crate::symbolic::patterns::pm_move_where_on_load()
        + pm_flatten_range()
        + crate::rangeify::patterns::pm_reduce_unparented()
});

fn extra_symbolic_patterns() -> &'static crate::TypedPatternMatcher {
    static PM: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| sym().clone() + crate::late::indexing_simplify().clone());
    &PM
}

fn lower_index_patterns() -> &'static crate::TypedPatternMatcher {
    static PM: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
        symbolic_simple().clone()
            + pm_fold_cast_const().clone()
            + crate::symbolic::pm_lower_index_dtype()
            + crate::late::indexing_simplify().clone()
    });
    &PM
}

#[cfg(test)]
mod lower_index_stage_tests {
    use super::*;
    use svod_dtype::DType;
    use svod_ir::{BinaryOp, ConstValue, Op};

    fn weak(value: i64) -> Arc<UOp> {
        UOp::const_(DType::WeakInt, ConstValue::Int(value))
    }

    fn weak_float(value: f64) -> Arc<UOp> {
        UOp::const_(DType::WeakFloat, ConstValue::Float(value))
    }

    fn production_value(value: Arc<UOp>) -> Arc<UOp> {
        let root = graph_rewrite(extra_symbolic_patterns(), UOp::sink(vec![value]), &mut ());
        let root = graph_rewrite(lower_index_patterns(), root, &mut ());
        let lowered = graph_rewrite(symbolic(), root, &mut ());
        let Op::Sink { sources, .. } = lowered.op() else { panic!("expected SINK") };
        sources[0].clone()
    }

    fn lower_value(value: Arc<UOp>) -> Arc<UOp> {
        let lowered = graph_rewrite(lower_index_patterns(), UOp::sink(vec![value]), &mut ());
        let Op::Sink { sources, .. } = lowered.op() else { panic!("expected SINK") };
        sources[0].clone()
    }

    #[test]
    fn lower_index_composition_pushes_long_cast_through_invalid() {
        let buffer = UOp::param(0, 16, DType::Float32, None);
        let x = UOp::variable("x".into(), 0, 15, DType::WeakInt);
        let valid = x.lt(&weak(8));
        let index = UOp::index().buffer(buffer).indices(vec![x.valid(valid).cast(DType::Int64)]).call().unwrap();

        let lowered = graph_rewrite(lower_index_patterns(), index, &mut ());
        let Op::Index { indices, .. } = lowered.op() else { panic!("expected INDEX") };
        let Op::Ternary(_, _, value, invalid) = indices[0].op() else { panic!("expected gated index") };
        assert_eq!(value.dtype(), DType::Int32, "{}", lowered.tree());
        assert!(UOp::is_invalid_marker(invalid));
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
    }

    #[test]
    fn post_optimization_propagates_stale_index_before_decomposition() {
        let stale = UOp::param(0, 1, DType::Index, None);
        let renderer = Renderer::cpu().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
        let err = apply_post_optimization_with_renderer(UOp::sink(vec![stale]), &renderer)
            .expect_err("legacy Index must fail at the post-index-lowering invariant");
        assert!(err.to_string().contains("post-index-lowering"), "unexpected error: {err}");
        assert!(err.to_string().contains("legacy Index dtype"), "unexpected error: {err}");
    }

    #[test]
    fn extra_symbolic_distributes_weak_index_before_lowering() {
        let buffer = UOp::param(0, 64, DType::Float32, None);
        let x = UOp::variable("x".into(), 0, 7, DType::WeakInt);
        let index_expr = x.add(&weak(2)).mul(&weak(4));
        let index = UOp::index().buffer(buffer).indices(vec![index_expr]).call().unwrap();

        let distributed = graph_rewrite(extra_symbolic_patterns(), index, &mut ());
        let Op::Index { indices, .. } = distributed.op() else { panic!("expected INDEX") };
        assert!(matches!(indices[0].op(), Op::Binary(BinaryOp::Add, ..)), "{}", distributed.tree());

        let lowered = graph_rewrite(lower_index_patterns(), distributed, &mut ());
        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
    }

    #[test]
    fn lower_index_commits_weak_vconsts_before_comparison_folding() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let neighbor = 1.0 + 2f64.powi(-23);
        let lhs = UOp::vconst(
            vec![ConstValue::Float(midpoint), ConstValue::Float(neighbor), ConstValue::Invalid],
            DType::WeakFloat,
        );
        let rhs =
            UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(1.0), ConstValue::Invalid], DType::WeakFloat);
        let comparison = lhs.try_cmpeq(&rhs).unwrap();

        let lowered = production_value(comparison);

        assert_eq!(lowered.dtype(), DType::Bool.vec(3).unwrap(), "{}", lowered.tree());
        assert!(matches!(lowered.op(), Op::VConst { values }
            if values == &vec![ConstValue::Bool(true), ConstValue::Bool(false), ConstValue::Invalid]));
    }

    #[test]
    fn lower_index_commits_weak_vconsts_before_arithmetic_folding() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let neighbor = 1.0 + 2f64.powi(-23);
        let lhs = UOp::vconst(
            vec![ConstValue::Float(midpoint), ConstValue::Float(neighbor), ConstValue::Invalid],
            DType::WeakFloat,
        );
        let rhs = UOp::vconst(
            vec![ConstValue::Float(midpoint), ConstValue::Float(midpoint), ConstValue::Invalid],
            DType::WeakFloat,
        );
        let add = lhs.try_add(&rhs).unwrap();

        let lowered = production_value(add);

        assert_eq!(lowered.dtype(), DType::Float32.vec(3).unwrap());
        assert!(matches!(lowered.op(), Op::VConst { values }
            if values == &vec![ConstValue::Float(2.0), ConstValue::Float(2.0), ConstValue::Invalid]));
    }

    #[test]
    fn lower_index_commits_scalar_weak_float_before_folding() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let comparison = weak_float(midpoint).try_cmpeq(&weak_float(1.0)).unwrap();
        let add = weak_float(midpoint).try_add(&weak_float(midpoint)).unwrap();

        let lowered_comparison = production_value(comparison);
        let lowered_add = production_value(add);

        assert_eq!(lowered_comparison.dtype(), DType::Bool);
        assert!(matches!(lowered_comparison.op(), Op::Const(value) if value.0 == ConstValue::Bool(true)));
        assert_eq!(lowered_add.dtype(), DType::Float32);
        assert!(matches!(lowered_add.op(), Op::Const(value) if value.0 == ConstValue::Float(2.0)));
    }

    #[test]
    fn lower_index_commits_constant_stack_lanes_before_their_consumer() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let lhs = UOp::stack(vec![weak_float(midpoint), UOp::invalid_marker()].into());
        let rhs = UOp::stack(vec![weak_float(1.0), UOp::invalid_marker()].into());
        let comparison = lhs.try_cmpeq(&rhs).unwrap();

        let lowered = lower_value(comparison);

        assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
        let Op::Binary(BinaryOp::Eq, lhs, rhs) = lowered.op() else {
            panic!("expected comparison, got {}", lowered.tree())
        };
        for stack in [lhs, rhs] {
            assert_eq!(stack.dtype(), DType::Float32);
            let Op::Stack { sources } = stack.op() else { panic!("expected STACK") };
            assert!(matches!(sources[0].op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)));
            assert!(UOp::is_invalid_marker(&sources[1]));
        }
    }

    #[test]
    fn production_commits_weak_coefficients_before_term_combining() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let x = UOp::variable("x".into(), -10, 10, DType::Float32);
        let expression =
            x.try_mul(&weak_float(midpoint)).unwrap().try_add(&x.try_mul(&weak_float(-1.0)).unwrap()).unwrap();

        let lowered = production_value(expression);

        assert!(
            matches!(lowered.op(), Op::Binary(BinaryOp::Mul, value, zero)
                if Arc::ptr_eq(value, &x)
                    && matches!(zero.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0))),
            "{}",
            lowered.tree()
        );
    }

    #[test]
    fn production_commits_weak_comparison_before_where_bounds() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let condition = weak_float(1.0).try_cmplt(&weak_float(midpoint)).unwrap();
        let expression = UOp::try_where(condition, UOp::native_const(7i32), UOp::native_const(9i32)).unwrap();

        let lowered = production_value(expression);

        assert!(matches!(lowered.op(), Op::Const(value) if value.0 == ConstValue::Int(9)), "{}", lowered.tree());
    }

    #[test]
    fn production_commits_weak_base_before_power_decomposition() {
        let midpoint = 1.0 + 2f64.powi(-24);
        let exponent = weak_float(1.0).try_add(&weak_float(1.0)).unwrap();
        let expression = weak_float(midpoint).try_pow(&exponent).unwrap();

        let lowered = production_value(expression);

        assert!(matches!(lowered.op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)), "{}", lowered.tree());
    }

    #[test]
    fn production_scalar_midpoint_neighbors_match_f32_commitment() {
        let midpoint = 1.0 + 2f64.powi(-24);
        for (value, expected) in [
            (f64::from_bits(midpoint.to_bits() - 1), true),
            (midpoint, true),
            (f64::from_bits(midpoint.to_bits() + 1), false),
        ] {
            let comparison = weak_float(value).try_cmpeq(&weak_float(1.0)).unwrap();
            let lowered = production_value(comparison);
            assert!(
                matches!(lowered.op(), Op::Const(result) if result.0 == ConstValue::Bool(expected)),
                "value={value:?} expected={expected}: {}",
                lowered.tree()
            );
        }
    }
}

/// Apply optimizations to a kernel AST.
///
/// This is the main entry point for optimization in the tensor pipeline.
/// Uses environment variables for configuration (see `OptimizerConfig::from_env`).
///
/// # Pipeline
///
/// 1. **Symbolic simplification** - Constant folding, identities, DCE
/// 2. **Loop→Global conversion** - Enable GPU parallelization
/// 3. **Hand-coded heuristics** - Vectorization, unrolling, tiling
///
/// # Arguments
///
/// * `ast` - The kernel AST (CALL body AST)
/// * `renderer` - Backend capabilities descriptor
///
/// # Returns
///
/// Optimized AST with transformations applied.
///
/// # Environment Variables
///
/// * `SVOD_NOOPT=1` - Disable all optimizations (for debugging)
/// * `BEAM=N` - Use beam search with width N (future)
pub fn optimize_kernel(ast: Arc<svod_ir::UOp>, renderer: &Renderer) -> Result<Arc<svod_ir::UOp>, OptError> {
    optimize_kernel_with_config(ast, renderer, &OptimizerConfig::from_env())
}

/// Apply post-optimization passes to kernel AST.
///
/// These passes run AFTER heuristic/beam optimization and BEFORE codegen:
/// - pm_add_loads: Extract LOAD ops from INDEX
/// - pre_expand: Expand Range(Unroll/Upcast) into shaped lane operations
/// - pm_add_gpudims (GPU only): Convert GLOBAL/LOCAL RANGE to SPECIAL thread indices
/// - devectorize: `symbolic_simple + devectorizer2 + indexing_simplify`
/// - target-selected dtype decomposition
///
/// NOTE: We do NOT apply FMA decomposition (a*b+c → MulAcc) — let LLVM's
/// optimizer fuse MUL+ADD into FMA when beneficial.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
///
/// Called by both heuristic and beam search paths for consistent behavior.
/// The concrete renderer is required because final decomposition is capability-dependent.
#[tracing::instrument(skip_all)]
pub fn apply_post_optimization(ast: Arc<svod_ir::UOp>, renderer: &Renderer) -> Result<Arc<svod_ir::UOp>, OptError> {
    apply_post_optimization_with_renderer(ast, renderer)
}

/// Apply post-optimization passes with renderer context.
///
/// Same as `apply_post_optimization`, retaining the legacy explicit name used by beam callers.
/// When the renderer has GPU capabilities, `pm_add_gpudims` is applied
/// to convert GLOBAL/LOCAL RANGE operations to SPECIAL thread indices.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
/// * `renderer` - Bound optimizer and code-renderer capabilities
#[tracing::instrument(skip_all)]
pub fn apply_post_optimization_with_renderer(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
) -> Result<Arc<svod_ir::UOp>, OptError> {
    apply_post_optimization_with_config(ast, renderer, &OptimizerConfig::from_env())
}

pub fn apply_post_optimization_with_config(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    config: &OptimizerConfig,
) -> Result<Arc<svod_ir::UOp>, OptError> {
    apply_post_optimization_configured_with_capture(
        ast,
        renderer,
        config.transcendental,
        config.disable_fast_idiv,
        None,
    )
}

fn apply_post_optimization_configured_with_capture(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    transcendental: i32,
    disable_fast_idiv: bool,
    final_rewrite_capture: Option<&mut Option<Arc<svod_ir::UOp>>>,
) -> Result<Arc<svod_ir::UOp>, OptError> {
    // Save metadata before graph_rewrite destroys it (e.g., KernelInfo with kernel name)
    let saved_metadata = ast.metadata_raw();

    tracing::trace!(ast.initial = ast.tree(), node_count = ast.node_count(), "kernel initial");

    // Env-gated per-stage diagnostic. Set SVOD_PER_STAGE_UOPS=1 to print
    // node_count to stderr after each post-opt stage. Used to pinpoint
    // which pass blows up on a bloated input. Set SVOD_DUMP_STAGE=<prefix>
    // to also dump the full UOp tree at any stage whose label starts with
    // that prefix (e.g. SVOD_DUMP_STAGE=13 dumps stage 13-pm_add_loads).
    // SVOD_DUMP_CANONICAL_STAGE uses the same prefix matching but emits the
    // allocation-independent canonical JSON used by parity tooling.
    let dump_per_stage = std::env::var("SVOD_PER_STAGE_UOPS").is_ok();
    let dump_stage_prefix = std::env::var("SVOD_DUMP_STAGE").ok();
    let print_stage = |label: &str, node: &Arc<svod_ir::UOp>| {
        if dump_per_stage {
            eprintln!("[per-stage] {} : node_count={}", label, node.node_count());
        }
        if let Some(ref prefix) = dump_stage_prefix
            && label.starts_with(prefix.as_str())
        {
            eprintln!("[dump-stage] {} :", label);
            eprintln!("{}", node.tree());
            eprintln!("[dump-stage] {} : end", label);
        }
        svod_ir::dump_canonical_stage(label, node);
    };
    print_stage("00-initial", &ast);

    // =========================================================================
    // Stage 8: Post-opt symbolic + WHERE movement
    // This MUST run BEFORE expander to optimize conditionals before expansion.
    // pm_move_where_on_load is scoped here, not applied globally.
    // =========================================================================
    let t_stage = std::time::Instant::now();
    let with_symbolic = graph_rewrite(&*POST_OPT_SYM, ast, &mut ());
    tracing::debug!(
        ast.optimized = with_symbolic.tree(),
        node_count = with_symbolic.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 8: after post-opt symbolic"
    );
    print_stage("08-post_opt_sym", &with_symbolic);

    // =========================================================================
    // Stage 9: Expander
    // =========================================================================
    // UNROLL expansion: expand UNROLL ops to vectorized operations.
    // CRITICAL: Must run BEFORE pm_reduce so REDUCE sees its actual vectorized
    // dtype, allowing reduce_to_acc to create accumulators with the correct
    // vector dtype.
    let t_stage = std::time::Instant::now();
    let expanded = crate::expand::pre_expand(&with_symbolic);
    tracing::debug!(
        ast.optimized = expanded.tree(),
        node_count = expanded.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 9: after pre_expand"
    );
    print_stage("09-pre_expand", &expanded);
    svod_ir::dump_canonical_stage("expanded", &expanded);

    // =========================================================================
    // Stage 10: Remove reductions
    // =========================================================================
    let t_stage = std::time::Instant::now();
    static PM_REDUCE_COMBINED: LazyLock<crate::TypedPatternMatcher<crate::devectorize::ReduceContext>> =
        LazyLock::new(|| {
            crate::devectorize::movement_cleanup_patterns().with_context::<crate::devectorize::ReduceContext>()
                + pm_reduce_local()
        });
    let mut reduce_ctx = crate::devectorize::ReduceContext::default();
    let reduced = graph_rewrite(&*PM_REDUCE_COMBINED, expanded, &mut reduce_ctx);
    tracing::debug!(
        ast.optimized = reduced.tree(),
        node_count = reduced.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_reduce"
    );
    print_stage("10-pm_reduce", &reduced);

    // Tinygrad adds local buffers after reduction lowering creates grouped
    // STAGE nodes, and before GPU dimensions are assigned.
    let t_stage = std::time::Instant::now();
    let with_local_buffers = { graph_rewrite(&pm_add_local_buffers(), reduced, &mut LocalBufferContext::default()) };
    tracing::debug!(
        ast.optimized = with_local_buffers.tree(),
        node_count = with_local_buffers.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after add local buffers"
    );
    print_stage("11-local_buffers", &with_local_buffers);

    let t_stage = std::time::Instant::now();
    let with_device_ranges = graph_rewrite(&crate::gpudims::pm_lower_device_ranges(), with_local_buffers, &mut ());
    let with_gpudims = if renderer.has_local || renderer.has_threads {
        graph_rewrite(&pm_add_gpudims(), with_device_ranges, &mut renderer.clone())
    } else {
        with_device_ranges
    };
    tracing::debug!(
        ast.optimized = with_gpudims.tree(),
        node_count = with_gpudims.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_add_gpudims"
    );
    print_stage("12-pm_add_gpudims", &with_gpudims);

    let t_stage = std::time::Instant::now();
    // Tinygrad target order: symbolic_simple + pm_expand_broadcast + pm_add_loads.
    static PM_ADD_LOADS: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| symbolic_simple() + pm_expand_broadcast().clone() + pm_add_loads().clone());
    let with_loads = graph_rewrite(&*PM_ADD_LOADS, with_gpudims, &mut ());
    tracing::debug!(
        ast.optimized = with_loads.tree(),
        node_count = with_loads.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_add_loads"
    );
    print_stage("13-pm_add_loads", &with_loads);

    // ALU devectorization happens inside devectorize() Phase 1, alongside
    // expand_index and full symbolic (including gep_pushing) — single combined
    // pass handles ALL devectorization including bool ALU (via
    // no_vectorized_alu). An earlier isolated pass that combined
    // no_vectorized_alu + gep_pushing without load/store folding caused graph
    // explosion on wide STACK nodes (for example, STACK with 135 sources).
    let t_stage = std::time::Instant::now();
    let renderer_ctx = renderer.clone();
    let devectorized = crate::devectorize::devectorize(&with_loads, &renderer_ctx);
    tracing::debug!(
        ast.optimized = devectorized.tree(),
        node_count = devectorized.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after devectorize"
    );
    print_stage("14-devectorize", &devectorized);

    // Some memory coalescing opportunities only become visible after full symbolic simplification.
    let t_stage = std::time::Instant::now();
    let early_symbolic = graph_rewrite(sym(), devectorized, &mut ());
    tracing::debug!(
        ast.optimized = early_symbolic.tree(),
        node_count = early_symbolic.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after early symbolic"
    );
    print_stage("15-early_symbolic", &early_symbolic);

    let t_stage = std::time::Instant::now();
    let coalesced = crate::late::memory_coalescing(early_symbolic, &renderer_ctx);
    tracing::debug!(
        ast.optimized = coalesced.tree(),
        node_count = coalesced.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after memory coalescing"
    );
    print_stage("16-memory_coalescing", &coalesced);

    // Tinygrad 8c8b43de: symbolic_simple+ew_devectorizer+pm_simplify_add_image,
    // ctx=({}, ren), bottom_up=True.
    let t_stage = std::time::Instant::now();
    static PM_ADD_IMAGES: LazyLock<crate::TypedPatternMatcher<crate::late::AddImageContext>> = LazyLock::new(|| {
        symbolic_simple().clone().with_context::<crate::late::AddImageContext>()
            + crate::devectorize::no_vectorized_alu().clone().with_context()
            + crate::late::pm_simplify_add_image()
    });
    let mut image_ctx = (std::collections::HashMap::new(), renderer_ctx.clone());
    let coalesced = crate::rewrite::graph_rewrite_bottom_up(&*PM_ADD_IMAGES, coalesced, &mut image_ctx);
    tracing::debug!(
        ast.optimized = coalesced.tree(),
        node_count = coalesced.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after bottom-up elementwise/image pass"
    );
    print_stage("17-bottom_up_ew_image", &coalesced);
    svod_ir::dump_canonical_stage("coalesced", &coalesced);

    // Keep indices weak while the distributive/index-validity rules can still fire.
    let t_stage = std::time::Instant::now();
    let extra_symbolic = graph_rewrite(extra_symbolic_patterns(), coalesced, &mut ());
    tracing::debug!(
        ast.optimized = extra_symbolic.tree(),
        node_count = extra_symbolic.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after extra symbolic"
    );
    print_stage("16-extra_symbolic", &extra_symbolic);

    // Tinygrad 8c8b43de codegen/__init__.py:347-349. Source order is required:
    // symbolic/cast folding exposes index lowering and indexing_simplify consumes it.
    let t_stage = std::time::Instant::now();
    let with_lowered_idx = graph_rewrite(lower_index_patterns(), extra_symbolic, &mut ());
    tracing::debug!(
        ast.optimized = with_lowered_idx.tree(),
        node_count = with_lowered_idx.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after pm_lower_index_dtype"
    );
    print_stage("17-pm_lower_index_dtype", &with_lowered_idx);

    // Final full symbolic before decomposition.
    let t_stage = std::time::Instant::now();
    static POST_INDEX_SYM: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| symbolic().clone());
    let with_lowered_idx = graph_rewrite(&*POST_INDEX_SYM, with_lowered_idx, &mut ());
    tracing::debug!(
        ast.optimized = with_lowered_idx.tree(),
        node_count = with_lowered_idx.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after post-index symbolic"
    );
    print_stage("18-final_symbolic", &with_lowered_idx);
    if crate::spec::spec_enabled() {
        crate::spec::verify_no_legacy_index_dtype(&with_lowered_idx).map_err(|source| OptError::Spec { source })?;
    }

    let t_stage = std::time::Instant::now();
    let cast_float = graph_rewrite(pm_cast_float_alu(), with_lowered_idx, &mut ());
    tracing::debug!(
        ast.optimized = cast_float.tree(),
        node_count = cast_float.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after cast float ALU operands"
    );
    print_stage("19-cast_float_alu", &cast_float);

    let t_stage = std::time::Instant::now();
    let supported_ops =
        renderer_ctx.supported_ops().expect("post-optimization requires concrete renderer capabilities");
    let pm_early_decomp = early_decomposition_patterns(supported_ops);
    let early_decomposed = graph_rewrite(&pm_early_decomp, cast_float, &mut ());
    tracing::debug!(
        ast.optimized = early_decomposed.tree(),
        node_count = early_decomposed.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after early decompositions"
    );
    print_stage("19b-early_decompositions", &early_decomposed);

    // Dtype decompositions run before late op decompositions and gate movement.
    let t_stage = std::time::Instant::now();
    let mut dtype_ctx = DTypeDecompCtx::new(renderer_ctx.clone());
    let pm_dtype = pm_dtype_decomps() + crate::symbolic::pm_commit_weak().with_context::<DTypeDecompCtx>();
    let dtype_decomposed = graph_rewrite(&pm_dtype, early_decomposed, &mut dtype_ctx);
    tracing::debug!(
        ast.optimized = dtype_decomposed.tree(),
        node_count = dtype_decomposed.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "after dtype decompositions"
    );
    print_stage("19c-dtype_decompositions", &dtype_decomposed);

    // Stage 18: Late decompositions. Gate movement must run after this pass.
    let t_stage = std::time::Instant::now();
    let force_transcendental = transcendental >= 2;
    let mut pm_decomp = pm_early_decomp
        + get_late_rewrite_patterns(&renderer_ctx, disable_fast_idiv)
        + svod_ir::decompositions::get_transcendental_patterns(supported_ops, force_transcendental);
    if let Some(matcher) = renderer_ctx.decomposition_matcher() {
        pm_decomp = pm_decomp + matcher.clone();
    }
    let late_decomposed = graph_rewrite(&pm_decomp, dtype_decomposed, &mut ());
    tracing::debug!(
        ast.optimized = late_decomposed.tree(),
        node_count = late_decomposed.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 18: after late decompositions"
    );
    print_stage("19d-late_decompositions", &late_decomposed);

    // Tinygrad codegen/__init__.py:369: lower WHERE-Invalid memory validity
    // after late decomposition and before the final pm_remove_invalid.
    let t_stage = std::time::Instant::now();
    let gates_moved = graph_rewrite(&crate::late::pm_move_gates_from_index(), late_decomposed, &mut ());
    let gates_moved =
        graph_rewrite(crate::devectorize::pm_scalarize_register_stack_index_preserve_deps(), gates_moved, &mut ());
    let gates_moved = crate::devectorize::merge_register_read_ends(gates_moved);
    assert!(
        !gates_moved.toposort().iter().any(crate::devectorize::is_register_stack_index),
        "direct register-stack lane selection must preserve all ordering dependencies",
    );
    tracing::debug!(
        ast.optimized = gates_moved.tree(),
        node_count = gates_moved.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 19: after move gates from index"
    );
    print_stage("19e-move_gates_from_index", &gates_moved);
    svod_ir::dump_canonical_stage("gated", &gates_moved);

    // Tinygrad's final rewrite repeats decomposition rules, then performs
    // renderer rewrites/split ENDs and removes any remaining Invalid markers.
    let t_stage = std::time::Instant::now();
    let mut pm_final = crate::symbolic::pm_commit_weak() + crate::symbolic::pm_cast_weak() + pm_decomp;
    if let Some(matcher) = renderer_ctx.extra_matcher() {
        pm_final = pm_final + matcher.clone();
    }
    pm_final = pm_final + crate::linearize::pm_split_ends();
    assert_target_renderer_boundary(&gates_moved);
    let rendered = graph_rewrite(&pm_final, gates_moved, &mut ());
    let final_rewrite = graph_rewrite(crate::symbolic::patterns::pm_remove_invalid(), rendered, &mut ());
    if let Some(capture) = final_rewrite_capture {
        *capture = Some(match saved_metadata.clone() {
            Some(meta) => final_rewrite.clone().with_metadata_raw(meta),
            None => final_rewrite.clone(),
        });
    }
    // Renderer rewrites can create local-memory dependencies, so barriers are
    // inferred after the captured late-final-rewrite boundary.
    let rendered = graph_rewrite(pm_implicit_barriers(), final_rewrite, &mut ());
    tracing::debug!(
        ast.optimized = rendered.tree(),
        node_count = rendered.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "Stage 20: after final rewrite"
    );
    print_stage("20-final_rewrite", &rendered);
    debug_assert!(!rendered.toposort().iter().any(UOp::is_invalid_marker), "final rewrite left an Invalid marker");

    // Re-attach metadata (e.g., KernelInfo) that was lost during graph rewrites
    let optimized = match saved_metadata {
        Some(meta) => rendered.with_metadata_raw(meta),
        None => rendered,
    };
    svod_ir::dump_canonical_stage("final_rewrite", &optimized);
    Ok(optimized)
}

fn assert_target_renderer_boundary(root: &Arc<UOp>) {
    for node in root.toposort() {
        match node.op() {
            Op::Index { buffer, indices } if indices.len() > 1 => {
                let static_tensor_index = buffer.shape().ok().flatten().is_some_and(|shape| {
                    shape.len() == indices.len() && shape.iter().all(|dim| dim.as_const().is_some())
                });
                assert!(!static_tensor_index, "static multi-index INDEX survived rangeify/indexing: {}", node.tree());
            }
            Op::Reshape { src, .. } => {
                let source_shape = src.shape().ok().flatten();
                let target_shape = node.shape().ok().flatten();
                let residual = source_shape.zip(target_shape).is_some_and(|(source, target)| {
                    target.len() > source.len()
                        && target[target.len() - source.len()..] == source[..]
                        && target[..target.len() - source.len()].iter().all(|dim| dim.as_const() == Some(1))
                });
                assert!(!residual, "singleton-prefix broadcast survived devectorize: {}", node.tree());
            }
            Op::Expand { src, .. } if matches!(src.op(), Op::Stack { sources } if sources.len() == 1) => {
                let source_shape = src.shape().ok().flatten();
                let target_shape = node.shape().ok().flatten();
                let residual = source_shape
                    .zip(target_shape)
                    .is_some_and(|(source, target)| source.len() == target.len() && source[1..] == target[1..]);
                assert!(!residual, "singleton STACK broadcast survived devectorize: {}", node.tree());
            }
            Op::Binary(..) | Op::Ternary(..)
                if node.op().sources().iter().any(|src| matches!(src.op(), Op::Stack { .. }))
                    && (node.dtype().vcount() > 1
                        || node.op().sources().iter().any(|src| src.dtype().vcount() > 1)) =>
            {
                panic!("mixed STACK/vector ALU survived devectorize: {}", node.tree());
            }
            _ => {}
        }
    }
}

/// Discover target-sensitive dtype emulation in Tinygrad's deterministic order.
pub fn get_dtype_decomps(
    root: &Arc<UOp>,
    renderer: &Renderer,
) -> Vec<(svod_dtype::ScalarDType, svod_dtype::ScalarDType)> {
    use svod_dtype::ScalarDType;
    let candidates = [
        ScalarDType::FP8E4M3,
        ScalarDType::FP8E4M3FNUZ,
        ScalarDType::FP8E5M2,
        ScalarDType::FP8E5M2FNUZ,
        ScalarDType::Float16,
        ScalarDType::BFloat16,
        ScalarDType::Int64,
        ScalarDType::UInt64,
    ];
    let graph_dtypes: std::collections::BTreeSet<_> = root
        .toposort()
        .into_iter()
        .map(|u| u.dtype().base())
        .filter(|dt| candidates.contains(dt))
        .map(|dt| if dt == ScalarDType::UInt64 { ScalarDType::Int64 } else { dt })
        .collect();
    graph_dtypes
        .into_iter()
        .filter(|dt| !renderer.supports_dtype(*dt))
        .map(|from| {
            let to = if from == ScalarDType::Int64 {
                ScalarDType::Int32
            } else if from.is_fp8() && renderer.supports_dtype(ScalarDType::Float16) {
                ScalarDType::Float16
            } else {
                ScalarDType::Float32
            };
            (from, to)
        })
        .collect()
}

#[derive(Clone)]
struct DTypeDecompCtx {
    selected: std::collections::BTreeSet<svod_dtype::ScalarDType>,
    renderer: Renderer,
}

impl DTypeDecompCtx {
    fn new(renderer: Renderer) -> Self {
        Self { selected: std::collections::BTreeSet::new(), renderer }
    }

    fn should_emulate(&self, dtype: svod_dtype::ScalarDType) -> bool {
        !self.renderer.supports_dtype(dtype)
    }

    fn mapping(&self, from: svod_dtype::ScalarDType) -> svod_dtype::ScalarDType {
        use svod_dtype::ScalarDType;
        if from == ScalarDType::Int64 {
            ScalarDType::Int32
        } else if from.is_fp8() && !self.should_emulate(ScalarDType::Float16) {
            ScalarDType::Float16
        } else {
            ScalarDType::Float32
        }
    }
}

/// Tinygrad `pm_dtype_decomps`: discover while walking, then decompose all
/// selected dtypes from the SINK rule in deterministic dtype order.
fn pm_dtype_decomps() -> crate::TypedPatternMatcher<DTypeDecompCtx> {
    crate::patterns! {
        @context DTypeDecompCtx;

        x if matches!(x.dtype().base(),
            svod_dtype::ScalarDType::FP8E4M3
            | svod_dtype::ScalarDType::FP8E4M3FNUZ
            | svod_dtype::ScalarDType::FP8E5M2
            | svod_dtype::ScalarDType::FP8E5M2FNUZ
            | svod_dtype::ScalarDType::Float16
            | svod_dtype::ScalarDType::BFloat16
            | svod_dtype::ScalarDType::Int64
            | svod_dtype::ScalarDType::UInt64) => |x, ctx| {
                let dtype = if x.dtype().base() == svod_dtype::ScalarDType::UInt64 {
                    svod_dtype::ScalarDType::Int64
                } else {
                    x.dtype().base()
                };
                ctx.selected.insert(dtype);
                None
            },

        sink @ Sink { sources: _ } if ctx.selected.iter().any(|dtype| ctx.should_emulate(*dtype)) => |sink, ctx| {
            use svod_dtype::ScalarDType;
            let selected = std::mem::take(&mut ctx.selected);
            let mut result = sink.clone();
            for from in selected.into_iter().filter(|dtype| ctx.should_emulate(*dtype)) {
                let to = ctx.mapping(from);
                tracing::debug!(?from, ?to, target = ctx.renderer.device.canonical(), "emulating dtype");
                if from == ScalarDType::Int64 {
                    result = svod_ir::rewrite::graph_rewrite_bottom_up(&pm_long_decomp(), result, &mut ());
                } else {
                    result = svod_ir::rewrite::graph_rewrite_bottom_up(
                        &pm_float_decomp(),
                        result,
                        &mut Fp8DecompCtx { from, to },
                    );
                }
            }
            (!Arc::ptr_eq(&result, sink)).then_some(result)
        },
    }
}

#[cfg(test)]
pub(crate) fn apply_dtype_decomps(root: Arc<UOp>, renderer: Renderer) -> Arc<UOp> {
    let mut ctx = DTypeDecompCtx::new(renderer);
    graph_rewrite(
        &(pm_dtype_decomps() + crate::symbolic::pm_commit_weak().with_context::<DTypeDecompCtx>()),
        root,
        &mut ctx,
    )
}

#[cfg(test)]
pub(crate) fn finish_final_rewrite(root: Arc<UOp>) -> Arc<UOp> {
    let root = graph_rewrite(crate::symbolic::patterns::pm_remove_invalid(), root, &mut ());
    graph_rewrite(pm_implicit_barriers(), root, &mut ())
}

#[cfg(test)]
pub(crate) fn final_rewrite_patterns() -> &'static crate::TypedPatternMatcher {
    static PM: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
        crate::symbolic::pm_commit_weak()
            + crate::symbolic::pm_cast_weak()
            + crate::symbolic::patterns::pm_remove_invalid().clone()
    });
    &PM
}

/// Float transcendental operands must have the operation's output dtype before
/// decomposition expands them into dtype-homogeneous polynomial arithmetic.
fn pm_cast_float_alu() -> &'static crate::TypedPatternMatcher {
    static PM: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(|| {
        crate::patterns! {
            for op in unary [Sin, Log2, Exp2, Sqrt, Reciprocal] {
                u @ op(x) if x.dtype() != u.dtype()
                    => Some(UOp::new(Op::Unary(op, x.cast(u.dtype())), u.dtype())),
            }
        }
    });
    &PM
}

/// Tinygrad's simplifying decomposition matcher.
fn early_decomposition_patterns(supported: &svod_ir::RendererOps) -> crate::TypedPatternMatcher {
    let mut pm = symbolic_simple()
        + pm_fold_cast_const()
        + pm_mod_to_and().clone()
        + svod_ir::decompositions::divmod_decomposition_patterns();
    if !supported.supports_binary(svod_ir::BinaryOp::Threefry) {
        pm = pm + pm_threefry_decomp().clone();
    }
    if !supported.supports_binary(svod_ir::BinaryOp::Max) && supported.supports_binary(svod_ir::BinaryOp::Lt) {
        pm = pm + crate::rangeify::patterns::pm_max_decomposition().clone();
    }
    if !supported.supports_unary(svod_ir::UnaryOp::Erf) {
        pm = pm + crate::rangeify::patterns::pm_erf_decomposition().clone();
    }
    pm
}

/// Late rewrite patterns for algebraic decompositions.
///
/// Returns patterns for:
/// - MULACC (FMA): `a*b+c → MulAcc(a,b,c)` for float types
/// - MOD → AND: `x % 2^n → x & (2^n-1)` for power-of-two modulus
/// - MUL → SHL: `x * 2^n → x << n` for power-of-two multiplier
/// - NEG from MUL: `x * -1 → NEG(x)`
/// - Fast integer division (magic number multiplication)
fn get_late_rewrite_patterns(renderer: &Renderer, disable_fast_idiv: bool) -> crate::TypedPatternMatcher {
    use svod_ir::{BinaryOp as B, TernaryOp as T, UnaryOp as U};
    let supported = renderer.supported_ops().expect("late rewrites require concrete renderer capabilities");
    let mut pm = pm_mod_to_and().clone() + pm_half_bf16_cast().clone();
    if supported.supports_binary(B::Or) {
        pm = pm + pm_demorgan().clone();
    }
    if supported.supports_binary(B::Shl) {
        pm = pm + pm_mul_to_shl().clone();
    }
    if supported.supports_binary(B::Shr) {
        pm = pm + pm_div_to_shr().clone();
        if !disable_fast_idiv {
            pm = pm + crate::symbolic::fast_division_patterns(renderer.supported_dtypes()) + pm_mod_to_idiv().clone();
        }
    }
    if supported.supports_unary(U::Neg) {
        pm = pm + pm_neg_from_mul().clone();
    }
    if supported.supports_binary(B::Lt) || supported.supports_binary(B::Eq) {
        pm = pm + pm_comparison_negations().clone();
    }
    if supported.supports_ternary(T::MulAcc) {
        pm = pm + pm_fma_decomposition().clone();
        if supported.supports_binary(B::Shl) {
            pm = pm + pm_shl_add_to_mulacc().clone();
        }
    }
    if supported.supports_binary(B::Fdiv) {
        pm = pm + pm_fdiv_to_mul().clone();
    }
    pm
}

/// CMOD → CDIV decomposition.
///
/// `x % d → x - d*(x//d)` for non-power-of-2 constant divisors.
/// Runs AFTER fast_division_patterns so the resulting CDIV gets decomposed
/// to magic-number multiplication. Without this, standalone CMOD nodes
/// for non-power-of-2 divisors survive to codegen unlowered.
fn pm_mod_to_idiv() -> &'static crate::TypedPatternMatcher {
    crate::cached_patterns! {
        CMod(x, d @const(d_val))
            if x.dtype().is_int()
            && (x.dtype().is_unsigned() || x.vmin().try_int().is_some_and(|v| v >= 0))
            && matches!(d_val.try_int(), Some(v) if v > 1 && !((v as u64).is_power_of_two()))
            => {
                // x % d → x - d * (x // d)
                let div = x.cdiv(d);
                let mul = d.try_mul(&div).ok()?;
                x.try_sub(&mul).ok()
            },
    }
}

/// Apply per-kernel pre-optimization passes.
///
/// These stages run BEFORE heuristic/beam optimization, per-kernel.
///
/// Stages:
/// 1. Movement ops (`pm_mops`, bottom-up)
/// 2. Load collapse (`pm_load_collapse`)
/// 3. Split ranges + flatten (`pm_split_ranges + pm_flatten_range`)
/// 4. Symbolic + flatten (`sym + pm_flatten_range`)
/// 5. Simplify ranges (`pm_simplify_ranges`)
///
/// Called by both heuristic and beam search paths.
#[tracing::instrument(skip_all)]
pub fn apply_pre_optimization(ast: Arc<svod_ir::UOp>) -> Result<Arc<svod_ir::UOp>, OptError> {
    tracing::trace!(ast.initial = ast.tree(), node_count = ast.node_count(), "kernel initial");

    // Tinygrad full_rewrite_to_sink verifies the original kernel DAG at this
    // boundary before per-kernel preprocessing.
    if crate::spec::spec_enabled() {
        crate::spec::type_verify(&ast, &crate::spec::spec_tensor()).map_err(|source| OptError::Spec { source })?;
    }

    use crate::rangeify::transforms::SplitRangesContext;

    let mut sink = ast;

    let t_stage = std::time::Instant::now();
    use crate::rangeify::patterns::movement_op_patterns;
    use crate::rewrite::graph_rewrite_bottom_up;
    static PM_EARLY_MOPS: LazyLock<crate::TypedPatternMatcher> = LazyLock::new(movement_op_patterns);
    sink = graph_rewrite_bottom_up(&*PM_EARLY_MOPS, sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: movement ops complete"
    );

    let t_stage = std::time::Instant::now();
    sink = graph_rewrite(pm_load_collapse(), sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: load collapse complete"
    );

    let t_stage = std::time::Instant::now();
    let mut split_ctx = SplitRangesContext::new();
    let pm_split_flatten = pm_split_ranges() + pm_flatten_range().with_context::<SplitRangesContext>();
    sink = graph_rewrite(&pm_split_flatten, sink, &mut split_ctx);
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: split ranges complete"
    );

    let t_stage = std::time::Instant::now();
    static PM_SYM_FLATTEN: LazyLock<crate::TypedPatternMatcher> =
        LazyLock::new(|| sym() + pm_fold_cast_const() + pm_flatten_range());
    sink = graph_rewrite(&*PM_SYM_FLATTEN, sink, &mut ());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: symbolic + flatten complete"
    );

    let t_stage = std::time::Instant::now();
    static PM_SIMPLIFY_FLATTEN: LazyLock<crate::TypedPatternMatcher<SimplifyRangesContext>> =
        LazyLock::new(|| pm_flatten_range().with_context::<SimplifyRangesContext>() + pm_simplify_ranges());
    sink = graph_rewrite(&*PM_SIMPLIFY_FLATTEN, sink, &mut SimplifyRangesContext::default());
    tracing::debug!(
        ast.pre = sink.tree(),
        node_count = sink.node_count(),
        elapsed_ms = t_stage.elapsed().as_millis() as u64,
        "pre-opt: simplify ranges complete"
    );

    Ok(sink)
}

/// Apply optimizations with explicit configuration.
///
/// Use this when you need explicit control over the optimization settings.
///
/// Note: For beam search strategy, this falls back to heuristics because
/// beam search requires a `compile_and_time` function from the runtime.
/// Use `optimize_kernel_beam()` for actual beam search optimization.
pub fn optimize_kernel_with_config(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    config: &OptimizerConfig,
) -> Result<Arc<svod_ir::UOp>, OptError> {
    optimize_kernel_with_config_impl(ast, renderer, config, None)
}

/// Run the production optimizer and also return its graph immediately after
/// the pinned late `final rewrite`, before implicit barriers, CFG insertion,
/// and PARAM numbering.
pub fn optimize_kernel_with_config_and_final_rewrite(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    config: &OptimizerConfig,
) -> Result<(Arc<svod_ir::UOp>, Arc<svod_ir::UOp>), OptError> {
    let mut final_rewrite = None;
    let optimized = optimize_kernel_with_config_impl(ast, renderer, config, Some(&mut final_rewrite))?;
    Ok((final_rewrite.expect("post-optimization must capture final rewrite"), optimized))
}

fn optimize_kernel_with_config_impl(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    config: &OptimizerConfig,
    final_rewrite_capture: Option<&mut Option<Arc<svod_ir::UOp>>>,
) -> Result<Arc<svod_ir::UOp>, OptError> {
    if renderer.supported_ops().is_none() {
        return Err(OptError::MissingRendererCapabilities);
    }
    // Author-supplied `opts_to_apply` (tinygrad parity) is read from the kernel
    // SINK marker BEFORE pre-optimization. When set, it overrides the strategy:
    // apply exactly those opts (an empty list applies none), never heuristics.
    let explicit_opts = kernel_opts_to_apply(&ast);

    // Pre-optimization: per-kernel stages. Kept ON for the explicit-opts path
    // for parity with `OptStrategy::None` (which also keeps it).
    let pre_optimized = apply_pre_optimization(ast)?;

    let optimized = if let Some(opts) = explicit_opts {
        apply_explicit_opts(pre_optimized, renderer, &opts)?
    } else {
        match config.strategy {
            OptStrategy::None => pre_optimized, // No heuristic optimization, but post-optimization still needed
            OptStrategy::Heuristic => optimize_heuristic(pre_optimized, renderer, &config.heuristics),
            OptStrategy::Beam { .. } => {
                // Beam search requires a compile_and_time function.
                // Use optimize_kernel_beam() for actual beam search.
                // Fall back to heuristics for the simple API.
                optimize_heuristic(pre_optimized, renderer, &config.heuristics)
            }
        }
    };

    // Svod applies PADTO in the same optimizer transaction as the other
    // schedule opts, so optimized and postrange are one concrete boundary.
    svod_ir::dump_canonical_stage("optimized", &optimized);
    svod_ir::dump_canonical_stage("postrange", &optimized);

    // apply_post_optimization contains correctness transforms (pm_add_loads wraps INDEX
    // with LOAD for arithmetic ops) and must run even when optimizations are disabled.
    // Pass the renderer to enable GPU dimension injection for GPU backends.

    apply_post_optimization_configured_with_capture(
        optimized,
        renderer,
        config.transcendental,
        config.disable_fast_idiv,
        final_rewrite_capture,
    )
}

#[cfg(test)]
pub(crate) fn apply_late_rewrites(
    root: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    disable_fast_idiv: bool,
) -> Arc<svod_ir::UOp> {
    graph_rewrite(&get_late_rewrite_patterns(renderer, disable_fast_idiv), root, &mut ())
}

/// Read an author-supplied `opts_to_apply` list off a kernel SINK marker.
///
/// Mirrors tinygrad's `apply_opts` reading `ast.arg.opts_to_apply`. Returns
/// `None` (optimizer chooses) unless the AST is a `Sink` whose `KernelInfo`
/// carries an explicit list.
fn kernel_opts_to_apply(ast: &Arc<svod_ir::UOp>) -> Option<Vec<Opt>> {
    match ast.op() {
        svod_ir::Op::Sink { info: Some(ki), .. } => ki.opts_to_apply.clone(),
        _ => None,
    }
}

/// Apply exactly the author-supplied opts (tinygrad's `apply_opts` inner loop).
///
/// `convert_loop_to_global` runs first (as tinygrad does), then each opt is
/// applied in order. An empty list applies zero opts — the pass-through case
/// for an already-lowered hand-built kernel. An opt that fails to apply is an
/// error: the author asked for *exactly* these opts, so a failure is propagated
/// rather than silently dropped (which would yield a kernel missing a requested
/// transform).
fn apply_explicit_opts(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    opts: &[Opt],
) -> Result<Arc<svod_ir::UOp>, OptError> {
    let mut scheduler = Scheduler::new(ast, renderer.clone());
    scheduler.convert_loop_to_global()?;
    for opt in opts {
        apply_opt(&mut scheduler, opt, true)?;
    }
    Ok(scheduler.get_optimized_ast(None))
}

/// Apply optimizations with explicit strategy selection (legacy API).
///
/// Prefer `optimize_kernel_with_config` for new code.
pub fn optimize_kernel_with_strategy(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    strategy: OptStrategy,
) -> Result<Arc<svod_ir::UOp>, OptError> {
    let config = OptimizerConfig { strategy, ..Default::default() };
    optimize_kernel_with_config(ast, renderer, &config)
}

/// Apply beam search optimization with custom timing function.
///
/// This is the primary entry point for beam search auto-tuning. It requires
/// a `compile_and_time` function that compiles a scheduler state and returns
/// its execution timing.
///
/// # Arguments
///
/// * `ast` - The kernel AST to optimize
/// * `renderer` - Backend capabilities descriptor
/// * `config` - Beam search configuration
/// * `compile_and_time` - Function to compile and time a scheduler
///
/// # Returns
///
/// Result containing `BeamResult` with optimized scheduler and metrics.
///
/// # Example
///
/// ```ignore
/// use svod_schedule::optimizer::{optimize_kernel_beam, BeamConfig, Renderer};
/// use svod_runtime::{BenchmarkConfig, benchmark_kernel};
///
/// let config = BeamConfig::from_env();
/// let renderer = Renderer::cpu();
///
/// let compile_and_time = |scheduler: &Scheduler| -> Option<Duration> {
///     let ast = scheduler.get_optimized_ast(None);
///     let kernel = compile_kernel(&ast)?;
///     let result = benchmark_kernel(&kernel, &buffers, &vars, &bench_config).ok()?;
///     Some(result.min)
/// };
///
/// let result = optimize_kernel_beam(ast, &renderer, &config, compile_and_time)?;
/// let optimized_ast = result.scheduler.get_optimized_ast(None);
/// ```
pub fn optimize_kernel_beam<F>(
    ast: Arc<svod_ir::UOp>,
    renderer: &Renderer,
    config: &BeamConfig,
    compile_and_time: F,
) -> Result<BeamResult, error::OptError>
where
    F: Fn(&Scheduler, Option<std::time::Duration>) -> Option<beam::CandidateMetrics> + Sync,
{
    // Step 0: Per-kernel pre-optimization.
    let pre_optimized = apply_pre_optimization(ast)?;

    // Step 1: Create scheduler (AST already simplified by apply_pre_optimization Stage 3)
    let mut scheduler = Scheduler::new(pre_optimized, renderer.clone());

    // Step 2: Convert loops to global (for GPU parallelization)
    let _ = scheduler.convert_loop_to_global();

    // Step 4: Run beam search (with caching)
    beam::beam_search_cached(scheduler, config, compile_and_time)
}

/// Create a scheduler ready for optimization without applying any opts.
///
/// This is useful when you want to manually control the optimization process
/// or use beam search with custom logic.
///
/// # Arguments
///
/// * `ast` - The kernel AST
/// * `renderer` - Backend capabilities descriptor
///
/// # Returns
///
/// A `Scheduler` with loops converted to globals (if applicable).
pub fn prepare_scheduler(ast: Arc<svod_ir::UOp>, renderer: &Renderer) -> Result<Scheduler, OptError> {
    let pre_optimized = apply_pre_optimization(ast)?;
    let mut scheduler = Scheduler::new(pre_optimized, renderer.clone());
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL
    // Rangeify produces LOOP-typed axes by default; threading is left to the
    // optimizer.
    Ok(scheduler)
}

/// Apply heuristic-based optimizations.
fn optimize_heuristic(ast: Arc<svod_ir::UOp>, renderer: &Renderer, config: &HeuristicsConfig) -> Arc<svod_ir::UOp> {
    // Step 1: Create scheduler (AST already simplified by apply_pre_optimization Stage 3)
    let mut scheduler = Scheduler::new(ast, renderer.clone());

    // Step 3: Convert axes for parallelization/vectorization
    let _ = scheduler.convert_loop_to_global(); // GPU: LOOP→GLOBAL

    // Step 4: Apply hand-coded heuristics with config
    heuristics::hand_coded_optimizations(&mut scheduler, config);

    // Step 5: Extract optimized AST
    scheduler.get_optimized_ast(None)
}
