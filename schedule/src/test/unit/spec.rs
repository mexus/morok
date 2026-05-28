//! Tests for the `spec_program` whitelist (port of tinygrad `type_verify`).
//!
//! A lowered, pre-render kernel must match the whitelist exactly: valid ops
//! pass, and the malformed cases tinygrad rejects (`spec.py`) become a
//! recoverable `SpecError` — a non-integer memory index, a surviving
//! `PtrCat`/`Cat`, a movement op that should have been lowered to index
//! arithmetic, an unfolded `Invalid`, or any op outside the program whitelist.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use crate::spec::{spec_program, spec_tensor, type_verify};

/// A GLOBAL-address-space pointer PARAM (a kernel buffer argument).
fn global_param() -> Arc<UOp> {
    UOp::param(0, 16, DType::Float32.ptr(Some(16), AddrSpace::Global), None)
}

fn verify_err(root: &Arc<UOp>) -> String {
    type_verify(root, &spec_program()).expect_err("expected spec_program rejection").to_string()
}

#[test]
fn spec_program_accepts_valid_kernel() {
    // STORE(value -> INDEX(buf, [int])) wrapped in a SINK: every node satisfies a
    // program rule (PARAM=GLOBAL ptr, integer index, matching-dtype CONST, STORE).
    let buf = global_param();
    let idx = UOp::index().buffer(buf).indices(vec![UOp::index_const(0)]).ptr(true).call().unwrap();
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = UOp::new(Op::Store { index: idx, value, ranges: smallvec![] }, DType::Void);
    let sink = UOp::sink(vec![store]);

    assert!(type_verify(&sink, &spec_program()).is_ok(), "a well-formed kernel must pass the whitelist");
}

#[test]
fn spec_program_accepts_integer_alu() {
    // ALU over integer constants: shared-base arithmetic is valid.
    let sum = UOp::index_const(2).add(&UOp::index_const(3));
    let sink = UOp::sink(vec![sum]);
    assert!(type_verify(&sink, &spec_program()).is_ok());
}

#[test]
fn spec_program_rejects_float_index() {
    // A `<float>` reaching an INDEX operand (faulty horizontal-reduction lowering).
    // Built directly: `UOp::index` would reject the non-int index at construction.
    let fidx = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let bad = UOp::new(Op::Index { buffer: global_param(), indices: smallvec![fidx], gate: None }, DType::Float32);
    let sink = UOp::sink(vec![bad]);
    assert!(verify_err(&sink).contains("INDEX"), "float index must be rejected");
}

#[test]
fn spec_program_rejects_surviving_ptrcat() {
    // PtrCat must be distributed into scalar accesses by the devectorizer.
    let ptrcat = UOp::ptrcat().sources(vec![global_param()]).call();
    let sink = UOp::sink(vec![ptrcat]);
    assert!(verify_err(&sink).contains("PtrCat"), "surviving PtrCat must be rejected");
}

#[test]
fn spec_program_rejects_movement_op() {
    // Movement ops are lowered to index arithmetic before linearize; a survivor
    // is a lowering bug (tinygrad `spec.py:205`).
    let reshape = UOp::new(Op::Reshape { src: UOp::index_const(0), new_shape: UOp::index_const(1) }, DType::Index);
    let sink = UOp::sink(vec![reshape]);
    assert!(verify_err(&sink).contains("movement"), "movement op must be rejected in a program");
}

#[test]
fn spec_program_rejects_invalid_op() {
    let inv = UOp::new(Op::Invalid, DType::Void);
    let sink = UOp::sink(vec![inv]);
    assert!(verify_err(&sink).contains("Invalid"), "Invalid must be folded out before a program");
}

#[test]
fn spec_program_rejects_op_outside_whitelist() {
    // MULTI is a tensor-graph op with no program rule — the whitelist fails on it
    // (tinygrad: a uop matching no pattern raises, `spec.py:38`).
    let multi = UOp::new(Op::Multi { src: UOp::index_const(0), axis: 0 }, DType::Index);
    let sink = UOp::sink(vec![multi]);
    assert!(verify_err(&sink).contains("no matching rule"), "an op outside the whitelist must be rejected");
}

#[test]
fn spec_tensor_assertion_mode_passes_unmatched() {
    // spec_tensor runs in assertion mode: an op with no applicable rule passes
    // (only explicit `Err` rules fire). Contrast with the spec_program whitelist.
    let multi = UOp::new(Op::Multi { src: UOp::index_const(0), axis: 0 }, DType::Index);
    let sink = UOp::sink(vec![multi]);
    assert!(type_verify(&sink, &spec_tensor()).is_ok(), "assertion mode passes unmatched ops");
}
