use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{BinaryOp, Op, ReduceOp, SInt, UOp};

use crate::multi::{lower_allreduce_pm, multi_pm, validate_no_unresolved_allreduce, validate_supported_subset};
use crate::optimizer::apply_pre_optimization;
use crate::rangeify::rangeify_with_map;
use crate::rewrite::graph_rewrite;

fn buffer(size: usize) -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, size, DType::Float32)
}

#[test]
fn mselect_mstack_selects_the_requested_shard() {
    let shard0 = buffer(8);
    let shard1 = buffer(8);
    let selected = UOp::mstack(smallvec![shard0, shard1.clone()]).mselect(1);
    let result = graph_rewrite(&multi_pm(), selected, &mut ());
    assert!(Arc::ptr_eq(&result, &shard1));
}

#[test]
fn pre_optimization_does_not_repeat_multi_rewrite() {
    let shard0 = buffer(6);
    let shard1 = buffer(6);
    let stacked = UOp::mstack(smallvec![shard0, shard1]);
    let reshaped = stacked.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = apply_pre_optimization(reshaped.mselect(1)).unwrap();

    assert!(matches!(result.op(), Op::MSelect { .. }), "the per-kernel optimizer must not rerun multi_pm");
}

#[test]
fn same_axis_alu_runs_per_shard() {
    let local0 = buffer(8);
    let local1 = buffer(8);
    let lhs = UOp::multi(local0.clone(), 0);
    let rhs = UOp::multi(local1.clone(), 0);
    let add = UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Float32);
    let result = graph_rewrite(&multi_pm(), add, &mut ());

    let Op::Multi { src, axis: 0 } = result.op() else { panic!("expected MULTI, got {:?}", result.op()) };
    assert!(matches!(src.op(), Op::Binary(BinaryOp::Add, a, b) if Arc::ptr_eq(a, &local0) && Arc::ptr_eq(b, &local1)));
}

#[test]
fn scalar_absent_layout_is_a_supported_per_shard_operand() {
    let local = buffer(8);
    let scalar = UOp::native_const(2.0f32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, UOp::multi(local.clone(), 0), scalar.clone()), DType::Float32);
    let result = graph_rewrite(&multi_pm(), add, &mut ());

    assert!(matches!(result.op(), Op::Multi { src, axis: 0 }
        if matches!(src.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
            if Arc::ptr_eq(lhs, &local) && Arc::ptr_eq(rhs, &scalar))));
    validate_supported_subset(&result).unwrap();
}

#[test]
fn permute_remaps_the_shard_axis() {
    let local = buffer(6).try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let multi = UOp::multi(local.clone(), 0);
    let permute = UOp::new(Op::Permute { src: multi, axes: vec![1, 0] }, DType::Float32);
    let result = graph_rewrite(&multi_pm(), permute, &mut ());

    assert!(matches!(result.op(), Op::Multi { src, axis: 1 }
        if matches!(src.op(), Op::Permute { src: inner, axes } if Arc::ptr_eq(inner, &local) && axes == &[1, 0])));
}

#[test]
fn non_reduced_multi_axis_survives_reduce() {
    let local = buffer(8);
    let multi = UOp::multi(local.clone(), 1);
    let reduce = multi.reduce_with_num_axes(smallvec![], ReduceOp::Add, 1);
    let result = graph_rewrite(&multi_pm(), reduce, &mut ());

    assert!(matches!(result.op(), Op::Multi { src, axis: 0 }
        if matches!(src.op(), Op::Reduce { src: inner, num_axes: 1, .. } if Arc::ptr_eq(inner, &local))));
}

#[test]
fn non_sharded_reduce_axis_runs_per_shard_before_rangeify() {
    let local = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let reduced = UOp::multi(local.clone(), 1).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let rewritten = graph_rewrite(&multi_pm(), reduced.clone(), &mut ());

    assert!(matches!(rewritten.op(), Op::Multi { src, axis: 0 }
        if matches!(src.op(), Op::ReduceAxis { src: inner, axes, .. }
            if Arc::ptr_eq(inner, &local) && axes == &[0])));
    validate_supported_subset(&rewritten).unwrap();

    let rangeified = rangeify_with_map(UOp::sink(vec![reduced])).unwrap();
    assert!(rangeified.sink.toposort().iter().any(|node| matches!(node.op(), Op::Multi { axis: 0, .. })));
    assert!(
        rangeified
            .sink
            .toposort()
            .iter()
            .all(|node| { !matches!(node.op(), Op::ReduceAxis { src, .. } if matches!(src.op(), Op::Multi { .. })) })
    );
}

#[test]
fn unsupported_multi_forms_are_not_guessed() {
    let local0 = buffer(8);
    let local1 = buffer(8);
    let mismatched = UOp::new(
        Op::Binary(BinaryOp::Add, UOp::multi(local0.clone(), 0), UOp::multi(local1.clone(), 1)),
        DType::Float32,
    );
    let mismatched_result = graph_rewrite(&multi_pm(), mismatched.clone(), &mut ());
    assert!(Arc::ptr_eq(&mismatched_result, &mismatched), "mixed shard axes require resharding metadata");

    let reshape =
        UOp::new(Op::Reshape { src: UOp::multi(local0.clone(), 0), new_shape: UOp::index_const(8) }, DType::Float32);
    let reshape_result = graph_rewrite(&multi_pm(), reshape.clone(), &mut ());
    assert!(Arc::ptr_eq(&reshape_result, &reshape), "reshape needs shard count to prove an intact boundary");

    let stack = UOp::mstack(smallvec![local0, local1]);
    let out_of_range = stack.mselect(2);
    let select_result = graph_rewrite(&multi_pm(), out_of_range.clone(), &mut ());
    assert!(Arc::ptr_eq(&select_result, &out_of_range), "selection must not fall back to another shard");
}

#[test]
fn single_device_graph_is_unchanged() {
    let lhs = buffer(8);
    let rhs = buffer(8);
    let add = UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Float32);
    let result = graph_rewrite(&multi_pm(), add.clone(), &mut ());
    assert!(Arc::ptr_eq(&result, &add));
}

#[test]
fn rangeify_runs_multi_before_tagging() {
    let local0 = buffer(8);
    let local1 = buffer(8);
    let add = UOp::new(
        Op::Binary(BinaryOp::Add, UOp::multi(local0.clone(), 0), UOp::multi(local1.clone(), 0)),
        DType::Float32,
    );
    let result = rangeify_with_map(UOp::sink(vec![add])).unwrap();

    assert!(result.uop_list.iter().all(|node| {
        !matches!(node.op(), Op::Binary(..))
            || node.op().sources().iter().all(|source| !matches!(source.op(), Op::Multi { .. }))
    }));
    assert!(result.sink.toposort().iter().any(|node| matches!(node.op(), Op::Multi { .. })));
}

#[test]
fn rangeify_resolves_mselect_before_movement_lowering() {
    let shard0 = buffer(6);
    let shard1 = buffer(6);
    let stacked = UOp::mstack(smallvec![shard0, shard1.clone()]);
    let reshaped = stacked.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = rangeify_with_map(UOp::sink(vec![reshaped.mselect(1)])).unwrap();

    assert!(result.uop_list.iter().any(|node| Arc::ptr_eq(node, &shard1)));
    assert!(result.sink.toposort().iter().all(|node| !matches!(node.op(), Op::MSelect { .. })));
}

#[test]
fn rangeify_single_device_path_does_not_rewrite_before_tagging() {
    let lhs = buffer(8);
    let rhs = buffer(8);
    let add = UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::Float32);
    let result = rangeify_with_map(UOp::sink(vec![add.clone()])).unwrap();

    assert!(result.uop_list.iter().any(|node| Arc::ptr_eq(node, &add)));
    assert!(result.sink.toposort().iter().all(|node| !matches!(node.op(), Op::Multi { .. })));
}

#[test]
fn rangeify_rejects_unsupported_multi_forms_with_typed_errors() {
    let local0 = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let local1 = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let mismatched =
        UOp::new(Op::Binary(BinaryOp::Add, UOp::multi(local0.clone(), 0), UOp::multi(local1, 1)), DType::Float32);
    assert!(matches!(rangeify_with_map(UOp::sink(vec![mismatched])), Err(svod_ir::Error::MultiAxisMismatch { .. })));

    let nested = UOp::multi(UOp::multi(local0.clone(), 0), 0);
    assert!(matches!(rangeify_with_map(UOp::sink(vec![nested])), Err(svod_ir::Error::MultiNested { .. })));

    let reshape =
        UOp::new(Op::Reshape { src: UOp::multi(local0.clone(), 0), new_shape: UOp::index_const(8) }, DType::Float32);
    assert!(matches!(
        rangeify_with_map(UOp::sink(vec![reshape])),
        Err(svod_ir::Error::MultiMovementUnsupported { operation: "RESHAPE", .. })
    ));

    let flip = UOp::new(Op::Flip { src: UOp::multi(local0.clone(), 0), axes: vec![true, false] }, DType::Float32);
    assert!(matches!(
        rangeify_with_map(UOp::sink(vec![flip])),
        Err(svod_ir::Error::MultiMovementUnsupported { operation: "FLIP", axis: 0, .. })
    ));

    let unsharded = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let missing_layout = UOp::new(Op::Binary(BinaryOp::Add, UOp::multi(local0.clone(), 0), unsharded), DType::Float32);
    assert!(matches!(
        rangeify_with_map(UOp::sink(vec![missing_layout])),
        Err(svod_ir::Error::MultiLayoutMissing { axis: 0, .. })
    ));

    let reduced = UOp::multi(local0, 0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    assert!(matches!(
        rangeify_with_map(UOp::sink(vec![reduced])),
        Err(svod_ir::Error::MultiReductionAcrossShardAxis { axis: 0 })
    ));
}

#[test]
fn independent_outputs_may_have_different_single_axis_layouts() {
    let local0 = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let local1 = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    validate_supported_subset(&UOp::sink(vec![UOp::multi(local0, 0), UOp::multi(local1, 1)])).unwrap();
}

#[test]
fn shard_axis_reduce_emits_local_sum_then_allreduce() {
    let shard0 = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let shard1 = buffer(8).try_reshape(&smallvec![SInt::Const(2), SInt::Const(4)]).unwrap();
    let shards = UOp::mstack(smallvec![shard0.clone(), shard1.clone()]);
    let reduced = UOp::multi(shards, 0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let rewritten = graph_rewrite(&multi_pm(), reduced, &mut ());

    let Op::AllReduce { src, reduce_op: ReduceOp::Add, .. } = rewritten.op() else {
        panic!("expected ALLREDUCE, got {:?}", rewritten.op());
    };
    let Op::MStack { buffers } = src.op() else { panic!("expected local reduction MSTACK") };
    assert_eq!(buffers.len(), 2);
    assert!(matches!(buffers[0].op(), Op::ReduceAxis { src, reduce_op: ReduceOp::Add, axes }
        if Arc::ptr_eq(src, &shard0) && axes == &[0]));
    assert!(matches!(buffers[1].op(), Op::ReduceAxis { src, reduce_op: ReduceOp::Add, axes }
        if Arc::ptr_eq(src, &shard1) && axes == &[0]));
    validate_supported_subset(&rewritten).unwrap();
}

#[test]
fn shard_axis_reduce_supports_max_and_rejects_other_collectives() {
    let shards = UOp::mstack(smallvec![buffer(4), buffer(4)]);
    let max = UOp::multi(shards.clone(), 0).try_reduce_axis(ReduceOp::Max, vec![0]).unwrap();
    let max = graph_rewrite(&multi_pm(), max, &mut ());
    assert!(matches!(max.op(), Op::AllReduce { reduce_op: ReduceOp::Max, .. }));

    let mul = UOp::multi(shards, 0).try_reduce_axis(ReduceOp::Mul, vec![0]).unwrap();
    assert!(matches!(
        rangeify_with_map(UOp::sink(vec![mul])),
        Err(svod_ir::Error::MultiReductionAcrossShardAxis { axis: 0 })
    ));
}

#[test]
fn reduced_precision_cast_is_restored_around_collective() {
    let low0 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    let low1 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float16);
    let shards = UOp::mstack(smallvec![low0.cast(DType::Float32), low1.cast(DType::Float32)]);
    let reduced = UOp::multi(shards, 0).try_reduce_axis(ReduceOp::Add, vec![0]).unwrap();
    let rewritten = graph_rewrite(&multi_pm(), reduced, &mut ());

    let Op::Cast { src: collective, dtype: DType::Scalar(svod_dtype::ScalarDType::Float32) } = rewritten.op() else {
        panic!("expected widened result cast, got {:?}", rewritten.op());
    };
    let Op::AllReduce { src, .. } = collective.op() else { panic!("expected ALLREDUCE") };
    let Op::MStack { buffers } = src.op() else { panic!("expected MSTACK") };
    assert!(buffers.iter().all(|local| local.dtype() == DType::Float16));
}

#[test]
fn allreduce_lowers_to_opaque_host_call_before_program_codegen() {
    let shard0 = buffer(4);
    let allreduce = UOp::allreduce(UOp::mstack(smallvec![shard0.clone(), buffer(4)]), DeviceSpec::Cpu, ReduceOp::Add);
    let lowered = graph_rewrite(&lower_allreduce_pm(), allreduce.clone(), &mut ());
    validate_no_unresolved_allreduce(&lowered).unwrap();

    let Op::After { deps, .. } = lowered.op() else { panic!("expected AFTER output") };
    let Op::Call { body, args, .. } = deps[0].op() else { panic!("expected host collective CALL") };
    assert!(matches!(
        body.op(),
        Op::CustomFunction { kind: svod_ir::CustomFunctionKind::AllReduce { reduce_op: ReduceOp::Add }, .. }
    ));
    assert_eq!(args.len(), 3, "output plus two explicit shard buffers");
    assert!(Arc::ptr_eq(&args[0], &shard0), "collective output must be a concrete in-place shard buffer");
    assert!(matches!(body.op(), Op::CustomFunction { attrs, .. } if attrs.len() == args.len()));
    assert!(lowered.toposort_call_aware(true).iter().all(|node| !matches!(node.op(), Op::AllReduce { .. })));

    let rangeified = rangeify_with_map(UOp::sink(vec![allreduce])).unwrap();
    assert!(rangeified.sink.toposort_call_aware(true).iter().all(|node| !matches!(node.op(), Op::AllReduce { .. })));
    assert!(rangeified.sink.toposort().iter().any(|node| matches!(
        node.op(),
        Op::CustomFunction { kind: svod_ir::CustomFunctionKind::AllReduce { .. }, .. }
    )));
}
