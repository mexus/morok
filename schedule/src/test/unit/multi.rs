use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::{BinaryOp, Op, ReduceOp, SInt, UOp};

use crate::multi::multi_pm;
use crate::optimizer::apply_pre_optimization;
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
fn mselect_moves_before_movement_then_selects() {
    let shard0 = buffer(6);
    let shard1 = buffer(6);
    let stacked = UOp::mstack(smallvec![shard0, shard1.clone()]);
    let reshaped = stacked.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = apply_pre_optimization(reshaped.mselect(1)).unwrap();

    let Op::Reshape { src, .. } = result.op() else { panic!("expected RESHAPE, got {:?}", result.op()) };
    assert!(Arc::ptr_eq(src, &shard1), "MSELECT must preserve the requested shard through movement");
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
