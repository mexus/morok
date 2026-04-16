//! Common utilities shared between codegen backends.

use std::sync::Arc;

use morok_ir::{Op, UOp};

/// Check whether a buffer (PARAM/DefineGlobal) is used as a STORE target in the graph.
pub fn is_output_buffer(def_global: &Arc<UOp>, nodes: &[Arc<UOp>]) -> bool {
    let buffer_id = def_global.id;

    for node in nodes {
        if let Some(buffer) = node.store_buffer() {
            if buffer.id == buffer_id {
                return true;
            }
            if let Op::Index { buffer: idx_buf, .. } = buffer.op()
                && idx_buf.id == buffer_id
            {
                return true;
            }
        }
    }
    false
}

/// Collect buffer and variable parameters from a UOp graph.
///
/// Collects:
/// - Buffers: PARAM, DEFINE_LOCAL operations
/// - Variables: DEFINE_VAR operations (passed as i64 kernel params)
///
/// Returns (buffers, variables) sorted for deterministic function signatures.
pub fn collect_buffers_and_vars(root: &Arc<UOp>) -> (Vec<Arc<UOp>>, Vec<Arc<UOp>>) {
    let nodes = root.toposort();

    // Collect buffers
    let mut buffers = Vec::new();
    for node in &nodes {
        match node.op() {
            Op::Buffer { .. } | Op::Param { device: None, .. } | Op::DefineLocal(_) => {
                buffers.push(node.clone());
            }
            _ => {}
        }
    }

    // Sort buffers by internal ID (matches split_kernel.rs ordering)
    buffers.sort_by_key(|b| match b.op() {
        Op::Param { slot, device: None, .. } => *slot as u64,
        Op::DefineLocal(id) => (*id as u64) + (1u64 << 32),
        Op::Buffer { .. } => b.id + (1u64 << 48),
        _ => b.id,
    });

    // Collect DefineVar nodes - these become i64 kernel parameters
    let mut variables = Vec::new();
    for node in &nodes {
        if matches!(node.op(), Op::DefineVar { .. }) {
            variables.push(node.clone());
        }
    }

    // Sort variables by name for deterministic function signatures
    variables.sort_by_key(|v| if let Op::DefineVar { name, .. } = v.op() { name.clone() } else { String::new() });

    (buffers, variables)
}
