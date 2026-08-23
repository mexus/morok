//! Tests for Tinygrad-aligned cycle detection in kernel splitting.

use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::UOp;

use crate::rangeify::transforms::find_bufs;

#[test]
fn test_find_bufs_accepts_direct_load_and_store_targets() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let output = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let load_index = UOp::index().buffer(input).indices(vec![index.clone()]).call().unwrap();
    let loaded = UOp::load().index(load_index).call();
    let store_index = UOp::index().buffer(output).indices(vec![index]).call().unwrap();

    find_bufs(&store_index.store(loaded));
}

#[test]
fn test_find_bufs_accepts_read_write_through_same_direct_source() {
    // Tinygrad detects conflicting INDEX source ops, not LOAD versus STORE access.
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let load_index = UOp::index().buffer(buffer.clone()).indices(vec![index.clone()]).call().unwrap();
    let store_index = UOp::index().buffer(buffer).indices(vec![index]).call().unwrap();

    find_bufs(&store_index.store(UOp::load().index(load_index).call()));
}

#[test]
fn test_find_bufs_accepts_post_gater_memory_targets() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let output = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let gate = UOp::native_const(true);
    let index = UOp::index_const(0);
    let load_index = UOp::index().buffer(input).indices(vec![index.clone()]).call().unwrap();
    let loaded = UOp::load().index(load_index).alt(UOp::native_const(0.0f32)).gate(gate.clone()).call();
    let store_index = UOp::index().buffer(output).indices(vec![index]).call().unwrap();

    find_bufs(&store_index.store_gated(loaded, gate));
}

#[test]
fn test_find_bufs_accepts_pre_gater_valid_index() {
    let input = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let output = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let valid_index = UOp::index_const(0).valid(UOp::native_const(true));
    let load_index = UOp::index().buffer(input).indices(vec![valid_index.clone()]).call().unwrap();
    let loaded = UOp::load().index(load_index).call();
    let store_index = UOp::index().buffer(output).indices(vec![valid_index]).call().unwrap();

    find_bufs(&store_index.store(loaded));
}

#[test]
#[should_panic(expected = "cycle detected while indexing")]
fn test_find_bufs_rejects_distinct_index_sources_for_same_buffer() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index = UOp::index_const(0);
    let direct = UOp::index().buffer(buffer.clone()).indices(vec![index.clone()]).call().unwrap();
    let selected = UOp::index().buffer(buffer.mselect(0)).indices(vec![index]).call().unwrap();

    find_bufs(&selected.store(UOp::load().index(direct).call()));
}
