use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{AxisType, Op, ReduceOp, UOp};

use crate::rangeify::{SimplifyRangesContext, pm_simplify_ranges};
use crate::rewrite::graph_rewrite;

fn buffer() -> std::sync::Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32)
}

fn narrowed_end(root: &std::sync::Arc<UOp>, axis: usize) -> i64 {
    root.ranges()
        .iter()
        .find_map(|range| match range.op() {
            Op::Range { end, axis_id: svod_ir::AxisId::Renumbered(id), .. } if *id == axis => end.vmax().try_int(),
            _ => None,
        })
        .expect("range must remain in rewritten graph")
}

fn simplify(sink: std::sync::Arc<UOp>) -> std::sync::Arc<UOp> {
    graph_rewrite(&pm_simplify_ranges(), sink, &mut SimplifyRangesContext::default())
}

#[test]
fn bounded_load_narrows_range() {
    let range = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(0), AxisType::Loop);
    let gate = range.try_cmplt(&UOp::index_const(7)).unwrap();
    let index = UOp::index().buffer(buffer()).indices(vec![range.valid(gate)]).call().unwrap();
    let result = simplify(UOp::sink(vec![UOp::load().index(index).call()]));
    assert_eq!(narrowed_end(&result, 0), 7);
}

#[test]
fn bounded_store_narrows_range() {
    let range = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(1), AxisType::Loop);
    let gate = range.try_cmplt(&UOp::index_const(5)).unwrap();
    let index = UOp::index().buffer(buffer()).indices(vec![range.valid(gate)]).call().unwrap();
    let result = simplify(UOp::sink(vec![index.store(UOp::native_const(1.0f32))]));
    assert_eq!(narrowed_end(&result, 1), 5);
}

#[test]
fn conflicting_gates_choose_largest_bound() {
    let range = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(2), AxisType::Loop);
    let accesses = [4, 9].map(|bound| {
        let gate = range.try_cmplt(&UOp::index_const(bound)).unwrap();
        let index = UOp::index().buffer(buffer()).indices(vec![range.valid(gate)]).call().unwrap();
        UOp::load().index(index).call()
    });
    let result = simplify(UOp::sink(accesses.to_vec()));
    assert_eq!(narrowed_end(&result, 2), 9);
}

#[test]
fn reduce_range_is_protected() {
    let range = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(3), AxisType::Reduce);
    let gate = range.try_cmplt(&UOp::index_const(6)).unwrap();
    let index = UOp::index().buffer(buffer()).indices(vec![range.valid(gate)]).call().unwrap();
    let load = UOp::load().index(index).call();
    let reduce = load.reduce(smallvec![range], ReduceOp::Add);
    let result = simplify(UOp::sink(vec![reduce]));
    assert_eq!(narrowed_end(&result, 3), 16);
}

#[test]
fn ungated_and_noncanonical_gates_are_noops() {
    let ungated = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(4), AxisType::Loop);
    let ungated_index = UOp::index().buffer(buffer()).indices(vec![ungated]).call().unwrap();

    let indirect = UOp::range_axis(UOp::index_const(16), svod_ir::AxisId::Renumbered(5), AxisType::Loop);
    let shifted = indirect.add(&UOp::index_const(1));
    let gate = shifted.try_cmplt(&UOp::index_const(8)).unwrap();
    let indirect_index = UOp::index().buffer(buffer()).indices(vec![indirect.valid(gate)]).call().unwrap();

    let loads = [ungated_index, indirect_index].map(|index| UOp::load().index(index).call());
    let result = simplify(UOp::sink(loads.to_vec()));
    assert_eq!(narrowed_end(&result, 4), 16);
    assert_eq!(narrowed_end(&result, 5), 16);
}
