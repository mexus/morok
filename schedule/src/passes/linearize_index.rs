use std::sync::Arc;

use svod_ir::{BinaryOp, Op, UOp};

/// Count integer division and modulo operations when deciding whether adjacent
/// ranges can be merged.
pub fn count_divmod(uop: &Arc<UOp>) -> usize {
    uop.toposort()
        .iter()
        .filter(|node| matches!(node.op(), Op::Binary(BinaryOp::FloorDiv | BinaryOp::FloorMod, _, _)))
        .count()
}
