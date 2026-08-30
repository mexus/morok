use super::*;
use svod_dtype::DType;
use svod_ir::types::ConstValue;
use test_case::test_case;

fn loop_end(value: Arc<UOp>, ranges: &[Arc<UOp>]) -> Arc<UOp> {
    value.end(ranges.iter().cloned().collect())
}

fn float(value: f32) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(value as f64))
}

/// Ranges closed by the same END are not chained: predecessor edges only come from loops
/// that actually follow one another.
#[test_case(1; "one range")]
#[test_case(2; "two sibling ranges")]
fn ranges_closed_by_one_end_have_no_predecessor_edges(count: usize) {
    let end = UOp::index_const(10);
    let ranges: Vec<_> = (0..count).map(|axis| UOp::range(end.clone(), axis)).collect();
    let ctx = CFGContext::new(&UOp::sink(vec![loop_end(float(1.0), &ranges)]));

    assert!(!ctx.has_edges());
}

/// The inner END depends on the outer range, so the loops nest rather than being siblings
/// and the outer range keeps no predecessor.
#[test]
fn a_nested_range_is_not_a_sibling_of_its_parent() {
    let end = UOp::index_const(10);
    let outer_range = UOp::range(end.clone(), 1);
    let inner_value = float(1.0).add(&outer_range.cast(DType::Float32));
    let outer_end = loop_end(loop_end(inner_value, &[UOp::range(end, 0)]), std::slice::from_ref(&outer_range));

    let ctx = CFGContext::new(&UOp::sink(vec![outer_end]));

    assert!(ctx.get_predecessor(&outer_range).is_none());
}
