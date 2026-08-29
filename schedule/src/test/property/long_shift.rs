//! Property tests for the 64-bit shift word split in [`pm_long_decomp`].
//!
//! `pm_long_decomp` has no tinygrad counterpart — tinygrad targets only backends
//! with native 64-bit integers, so the shift rules in `decompositions.py` are the
//! `MUL`/`IDIV` -> shift strengthening, not a word split. Svod emits the split for
//! backends without native i64 (WebGPU), so each word must reproduce the native
//! `<<` / `>>` result bit for bit, including shifts of 32 or more.

use std::sync::Arc;

use proptest::prelude::*;
use svod_dtype::{DType, ScalarDType};
use svod_ir::rewrite::graph_rewrite_bottom_up;
use svod_ir::types::{BinaryOp, ConstValue};
use svod_ir::uop::eval::{eval_binary_op_typed, eval_ternary_op_typed, eval_unary_op_typed};
use svod_ir::{Op, UOp};

use crate::devectorize::pm_long_decomp;
use crate::test::unit::devectorize::helpers::create_buffer_typed;

/// Fold a fully constant word expression, mirroring what the backend would compute.
fn eval_word(expr: &Arc<UOp>) -> Option<ConstValue> {
    let dtype = expr.dtype().base();
    match expr.op() {
        Op::Const(constant) => Some(constant.0),
        Op::Unary(op, src) => eval_unary_op_typed(*op, eval_word(src)?, dtype),
        Op::Binary(op, lhs, rhs) => eval_binary_op_typed(*op, eval_word(lhs)?, eval_word(rhs)?, dtype),
        Op::Ternary(op, a, b, c) => eval_ternary_op_typed(*op, eval_word(a)?, eval_word(b)?, eval_word(c)?, dtype),
        Op::Cast { src, .. } | Op::BitCast { src, .. } => Some(reinterpret(eval_word(src)?, dtype)),
        _ => None,
    }
}

fn reinterpret(value: ConstValue, dtype: ScalarDType) -> ConstValue {
    let bits = match value {
        ConstValue::Int(v) => v as u32,
        ConstValue::UInt(v) => v as u32,
        ConstValue::Bool(v) => v as u32,
        other => return other,
    };
    if dtype.is_signed() { ConstValue::Int(bits as i32 as i64) } else { ConstValue::UInt(bits as u64) }
}

fn word_bits(value: ConstValue) -> u32 {
    match value {
        ConstValue::Int(v) => v as u32,
        ConstValue::UInt(v) => v as u32,
        other => panic!("word is not an integer: {other:?}"),
    }
}

/// Run `pm_long_decomp` over `STORE(index, value <op> shift)` and return `[low, high]`.
pub fn split_shift(op: BinaryOp, value: u64, shift: u64, from: ScalarDType) -> [u32; 2] {
    let long = DType::Scalar(from);
    let konst = |v: u64| {
        UOp::const_(
            long.clone(),
            if from == ScalarDType::Int64 { ConstValue::Int(v as i64) } else { ConstValue::UInt(v) },
        )
    };
    let index = UOp::index()
        .buffer(create_buffer_typed(4, from))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let root = index.store(UOp::new(Op::Binary(op, konst(value), konst(shift)), long));
    let decomposed = graph_rewrite_bottom_up(&pm_long_decomp(), root, &mut ());

    let mut words = [None; 2];
    for node in decomposed.toposort() {
        let Op::Store { value, .. } = node.op() else { continue };
        let word = node.tag().as_ref().expect("split store is word-tagged")[1];
        words[word] = Some(word_bits(eval_word(value).expect("word expression must fold")));
    }
    [words[0].expect("low word"), words[1].expect("high word")]
}

/// Native reference for `value <op> shift` at 64 bits.
pub fn native_shift(op: BinaryOp, value: u64, shift: u64, from: ScalarDType) -> u64 {
    match op {
        BinaryOp::Shl => value << shift,
        BinaryOp::Shr if from == ScalarDType::Int64 => ((value as i64) >> shift) as u64,
        BinaryOp::Shr => value >> shift,
        other => panic!("not a shift: {other:?}"),
    }
}

pub fn assert_shift_words(op: BinaryOp, value: u64, shift: u64, from: ScalarDType) {
    let expected = native_shift(op, value, shift, from);
    assert_eq!(
        split_shift(op, value, shift, from),
        [expected as u32, (expected >> 32) as u32],
        "{from:?} {value:#018x} {op:?} {shift}"
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// The word split of `x << s` / `x >> s` must equal the native 64-bit shift for every `s < 64`.
    #[test]
    fn long_shift_words_match_native(
        value in any::<u64>(),
        shift in 0u64..64,
        signed in any::<bool>(),
        right in any::<bool>(),
    ) {
        let from = if signed { ScalarDType::Int64 } else { ScalarDType::UInt64 };
        let op = if right { BinaryOp::Shr } else { BinaryOp::Shl };
        let expected = native_shift(op, value, shift, from);
        prop_assert_eq!(split_shift(op, value, shift, from), [expected as u32, (expected >> 32) as u32]);
    }
}
