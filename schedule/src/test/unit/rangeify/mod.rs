pub mod advanced_edge_cases;
pub mod buffer_folding;
pub mod buffer_limits;
pub mod bufferize_to_store;
pub mod codegen_integration;
pub mod codegen_patterns;
pub mod context;
pub mod cycle_detection;
pub mod deduplication;
pub mod device_semantics;
pub mod edge_cases;
pub mod flatten_range;
pub mod fusion;
pub mod helpers;
pub mod indexing;
pub mod kernel_context;
pub mod kernel_count;
pub mod late_decompositions;
pub mod load_collapse;
pub mod movement_patterns;
pub mod patterns;
pub mod pipeline;
pub mod pipeline_integration;
pub mod range_load_guards;
pub mod range_merging;
pub mod realize_map;
pub mod reduce_simplify;
pub mod remove_bufferize;
pub mod resolve_call;
pub mod simplify_ranges;
pub mod split_kernel;
pub mod split_patterns;
pub mod split_ranges;
pub mod split_reduceop;
pub mod transform;

use svod_ir::UOp;

use crate::rangeify::RangeifyContext;
use crate::rangeify::patterns as rangeify_patterns;

#[test]
fn test_rangeify_context_new() {
    let ctx = RangeifyContext::new();
    assert_eq!(ctx.range_counter, 0);
    assert_eq!(ctx.range_map.len(), 0);
}

#[test]
fn test_rangeify_context_next_range_id() {
    let mut ctx = RangeifyContext::new();

    assert_eq!(ctx.next_range_id(), 0);
    assert_eq!(ctx.next_range_id(), 1);
    assert_eq!(ctx.next_range_id(), 2);
    assert_eq!(ctx.range_counter, 3);
}

#[test]
fn test_rangeify_context_record_transform() {
    let mut ctx = RangeifyContext::new();

    let original = UOp::native_const(1.0f32);
    let rangeified = UOp::native_const(2.0f32);

    ctx.record_transform(original.clone(), rangeified.clone());

    let retrieved = ctx.get_rangeified(&original);
    assert!(retrieved.is_some());
    assert!(std::sync::Arc::ptr_eq(retrieved.unwrap(), &rangeified));
}

#[test]
fn test_rangeify_context_get_missing() {
    let ctx = RangeifyContext::new();

    let uop = UOp::native_const(1.0f32);
    assert!(ctx.get_rangeified(&uop).is_none());
}

#[test]
fn test_pattern_matchers_stub() {
    let m = rangeify_patterns::buffer_folding();
    let x = UOp::native_const(1.0f32);

    use crate::pattern::RewriteResult;
    assert!(matches!(m.rewrite(&x, &mut ()), RewriteResult::NoMatch));
}

#[test]
fn test_early_rewrites_detach_removal() {
    use crate::pattern::RewriteResult;

    let matcher = rangeify_patterns::early_rewrites();

    // Test: DETACH(x) -> x
    let x = UOp::native_const(1.0f32);
    let detach = x.detach();

    let result = matcher.rewrite(&detach, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::sync::Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_early_rewrites_contiguous_backward_removal() {
    use crate::pattern::RewriteResult;

    let matcher = rangeify_patterns::early_rewrites();

    // Test: CONTIGUOUS_BACKWARD(x) -> x
    let x = UOp::native_const(1.0f32);
    let contiguous = x.contiguous_backward();

    let result = matcher.rewrite(&contiguous, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::sync::Arc::ptr_eq(&rewritten, &x));
    }
}
