//! `find_bufs` panics when one buffer is reached through two different INDEX
//! source ops. Tinygrad keys on the INDEX source, not on LOAD-versus-STORE, so
//! every shape below is legal even when it reads and writes the same buffer.

use std::sync::Arc;

use svod_device::DeviceSpec;
use svod_dtype::DType;
use svod_ir::UOp;
use test_case::test_case;

use crate::rangeify::transforms::find_bufs;

fn buffer() -> Arc<UOp> {
    UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32)
}

fn index(buffer: Arc<UOp>, at: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buffer).indices(vec![at]).call().expect("index")
}

fn distinct_buffers() -> Arc<UOp> {
    let zero = UOp::index_const(0);
    index(buffer(), zero.clone()).store(UOp::load().index(index(buffer(), zero)).call())
}

fn one_buffer_read_and_written() -> Arc<UOp> {
    let (buf, zero) = (buffer(), UOp::index_const(0));
    let loaded = UOp::load().index(index(Arc::clone(&buf), zero.clone())).call();
    index(buf, zero).store(loaded)
}

/// Gate on the LOAD/STORE (post-gater), not on the address.
fn gated_load_and_store() -> Arc<UOp> {
    let (gate, zero) = (UOp::native_const(true), UOp::index_const(0));
    let loaded =
        UOp::load().index(index(buffer(), zero.clone())).alt(UOp::native_const(0.0f32)).gate(Arc::clone(&gate)).call();
    index(buffer(), zero).store_gated(loaded, gate)
}

/// Gate folded into the address itself (pre-gater `VALID`).
fn valid_gated_index() -> Arc<UOp> {
    let valid = UOp::index_const(0).valid(UOp::native_const(true));
    index(buffer(), Arc::clone(&valid)).store(UOp::load().index(index(buffer(), valid)).call())
}

#[test_case(super::distinct_buffers ; "distinct load and store buffers")]
#[test_case(super::one_buffer_read_and_written ; "one buffer through one index source")]
#[test_case(super::gated_load_and_store ; "post-gater load and store")]
#[test_case(super::valid_gated_index ; "pre-gater valid index")]
fn accepted(build: fn() -> Arc<UOp>) {
    find_bufs(&build());
}

#[test]
#[should_panic(expected = "cycle detected while indexing")]
fn distinct_index_sources_for_one_buffer_are_a_cycle() {
    let (buf, zero) = (buffer(), UOp::index_const(0));
    let direct = index(Arc::clone(&buf), zero.clone());
    let selected = index(buf.mselect(0), zero);

    find_bufs(&selected.store(UOp::load().index(direct).call()));
}
