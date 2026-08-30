//! Broadcast ending ranges — `run_rangeify`'s `broadcast_ending_ranges`
//! (tinygrad/schedule/indexing.py:221-225, re-added at :284).

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{BinaryOp, Op, SInt, UOp, UnaryOp};

use crate::rangeify::run_rangeify;

#[test]
fn a_broadcast_source_ends_its_producer_ranges() {
    // `big[4,4] + sqrt(exp(small[4]))`. The ADD iterates (r0, r1) and broadcasts
    // its right operand over r0, so r0 ends there: SQRT carries it, and EXP —
    // the producer that inherits it — is realized. Tinygrad on this exact graph
    // (run_rangeify(debug=True)) prints:
    //
    //        1 Ops.ADD   (4, 4)  0 [r0][r1]
    //        1 Ops.SQRT  (4,)    1 [r1]
    //   ***  1 Ops.EXP2  (4,)    0 [r2]
    //
    // — one ending range on SQRT, and EXP2 marked realized.
    //
    // The frontend materialises size-1 broadcasts as an explicit `Op::Expand`,
    // so only a hand-built graph reaches this path today.
    let big = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 16, DType::Float32)
        .try_reshape(&smallvec![SInt::Const(4), SInt::Const(4)])
        .expect("reshape");
    let small = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 4, DType::Float32);
    let inner = UOp::new(Op::Unary(UnaryOp::Exp, Arc::clone(&small)), DType::Float32);
    let outer = UOp::new(Op::Unary(UnaryOp::Sqrt, Arc::clone(&inner)), DType::Float32);
    let sum = UOp::new(Op::Binary(BinaryOp::Add, big, Arc::clone(&outer)), DType::Float32);

    let (_, ctx) = run_rangeify(UOp::sink(vec![sum.contiguous()])).expect("run_rangeify");

    assert!(ctx.should_realize(&inner), "the producer inheriting the ended range is realized");
    assert!(!ctx.should_realize(&outer), "the broadcast source itself keeps the consumer's ranges");
}
