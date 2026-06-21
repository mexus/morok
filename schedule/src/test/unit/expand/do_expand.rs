//! Tests for do_expand (UNROLL propagation through operations).
//!
//! Ported from Tinygrad's TestExpander class (test_uop_graph.py:663-811).
//!
//! do_expand replicates operations that have UNROLL inputs:
//! - Broadcasts scalar operands
//! - Swizzles UNROLL operands with different axes
//! - Wraps results in UNROLL
//!
//! Value assertions match Tinygrad's exact test expectations.

use super::helpers::*;
use svod_dtype::DType;
use svod_ir::UOp;
use svod_ir::types::ConstValue;

// =============================================================================
// Broadcast Expansion Tests
// =============================================================================

/// Test: UNROLL + scalar broadcast
///
/// Tinygrad: test_expand_add_broadcast
/// ```python
/// e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// sink = expander_rewrite(e1+3)
/// self.assertTupleEqual(sink.src[0].arg, (3,4,5,6))
/// ```
#[test]
fn test_expand_add_broadcast() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // Add scalar constant 3
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(3));
    let add = unroll.try_add(&scalar).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // Assert exact values like Tinygrad: (3, 4, 5, 6)
    assert_result_values(&result, &[3, 4, 5, 6]);

    // Also verify UNROLL structure
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4)], "Should preserve axis");
}

// =============================================================================
// Same-Axis Expansion Tests
// =============================================================================

/// Test: Two UNROLLs with same axis
///
/// Tinygrad: test_expand_same_axis
/// ```python
/// e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((1,4),))
/// e2 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x*4 for x in range(4))),), ((1,4),))
/// sink = expander_rewrite(e1+e2)
/// self.assertTupleEqual(sink.src[0].arg, (0, 5, 10, 15))
/// ```
#[test]
fn test_expand_same_axis() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let e1 = create_unroll_iota(1, 4);

    // Create UNROLL(VCONST([0,4,8,12]), [(1,4)])
    let e2 = create_unroll_scaled(1, 4, 4);

    // Add them
    let add = e1.try_add(&e2).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // Assert exact values: 0+0=0, 1+4=5, 2+8=10, 3+12=15
    assert_result_values(&result, &[0, 5, 10, 15]);

    // Verify UNROLL structure
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4)], "Should preserve axis");
}

// =============================================================================
// Different-Axis Expansion Tests
// =============================================================================

/// Test: Two UNROLLs with different axes
///
/// Tinygrad: test_expand_different_axis
/// ```python
/// e1 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(x*4 for x in range(4))),), ((1,4),))
/// e2 = UOp(Ops.UNROLL, dtypes.int, (UOp.const(dtypes.int.vec(4), tuple(range(4))),), ((2,4),))
/// sink = expander_rewrite(e1+e2)
/// self.assertTupleEqual(sink.arg, ((1, 4), (2, 4)))
/// self.assertTupleEqual(sink.src[0].arg, tuple(range(16)))
/// ```
#[test]
fn test_expand_different_axis() {
    // Create UNROLL(VCONST([0,4,8,12]), [(1,4)])
    let e1 = create_unroll_scaled(1, 4, 4);

    // Create UNROLL(VCONST([0,1,2,3]), [(2,4)])
    let e2 = create_unroll_iota(2, 4);

    // Add them
    let add = e1.try_add(&e2).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // When combining different axes, the result expands to 4*4=16 values.
    // Values follow the pattern: axis1_val + axis2_val
    // Row-major iteration: axis 1 is outer (slower), axis 2 is inner (faster)
    // (0,0)=0, (0,1)=1, (0,2)=2, (0,3)=3, (1,0)=4, (1,1)=5, ...
    // = 0+0, 0+1, 0+2, 0+3, 4+0, 4+1, 4+2, 4+3, 8+0, 8+1, 8+2, 8+3, 12+0, 12+1, 12+2, 12+3
    let expected: Vec<i64> = (0..16).collect();
    assert_result_values(&result, &expected);

    // Verify axes
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4), (2, 4)], "Should have both axes");
}

/// Test: Two UNROLLs with different axes (operands flipped)
///
/// Tinygrad: test_expand_different_axis_flip
/// Same values but operands reversed.
#[test]
fn test_expand_different_axis_flip() {
    // Create UNROLL(VCONST([0,1,2,3]), [(2,4)])
    let e2 = create_unroll_iota(2, 4);

    // Create UNROLL(VCONST([0,4,8,12]), [(1,4)])
    let e1 = create_unroll_scaled(1, 4, 4);

    // Add them (flipped order)
    let add = e2.try_add(&e1).unwrap();

    // Apply expander
    let result = phase2_only(&add);

    // Same result as test_expand_different_axis (addition is commutative)
    let expected: Vec<i64> = (0..16).collect();
    assert_result_values(&result, &expected);

    // Verify axes
    let (_, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4), (2, 4)], "Should have both axes");
}

// =============================================================================
// Three-Axis Expansion Tests
// =============================================================================

/// Test: Three UNROLLs with different axes
///
/// This extends Tinygrad's pattern to verify 3D expansion.
#[test]
fn test_expand_three_axes() {
    // Create UNROLL with axis 1 (stride 4): [0, 4, 8, 12]
    let e1 = create_unroll_scaled(1, 4, 4);

    // Create UNROLL with axis 2 (stride 1): [0, 1, 2, 3]
    let e2 = create_unroll_iota(2, 4);

    // Create UNROLL with axis 3 (stride 16): [0, 16, 32, 48]
    let e3 = create_unroll_scaled(3, 4, 16);

    // Build: e1 + e2 + e3
    let sum = e1.try_add(&e2).unwrap().try_add(&e3).unwrap();

    // Apply expander
    let result = phase2_only(&sum);

    // Result should have 4*4*4=64 elements
    // Verify axes
    let (src, axes) = unwrap_unroll(&result);
    assert_eq!(axes, vec![(1, 4), (2, 4), (3, 4)], "Should have three axes");
    assert_eq!(src.dtype().vcount(), 64, "Inner should be vec64");
}

// =============================================================================
// Multiplication Expansion Tests
// =============================================================================

/// Test: UNROLL * scalar
#[test]
fn test_expand_mul_broadcast() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let unroll = create_unroll_iota(1, 4);

    // Multiply by scalar 2
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(2));
    let mul = unroll.try_mul(&scalar).unwrap();

    // Apply expander
    let result = phase2_only(&mul);

    // Expected: [0*2, 1*2, 2*2, 3*2] = [0, 2, 4, 6]
    assert_result_values(&result, &[0, 2, 4, 6]);
}

/// Test: Two UNROLLs multiplied (same axis)
#[test]
fn test_expand_mul_same_axis() {
    // Create UNROLL(VCONST([1,2,3,4]), [(1,4)])
    let e1 = create_unroll_values(1, vec![1, 2, 3, 4]);

    // Create UNROLL(VCONST([1,2,3,4]), [(1,4)])
    let e2 = create_unroll_values(1, vec![1, 2, 3, 4]);

    // Multiply them
    let mul = e1.try_mul(&e2).unwrap();

    // Apply expander
    let result = phase2_only(&mul);

    // Expected: [1*1, 2*2, 3*3, 4*4] = [1, 4, 9, 16]
    assert_result_values(&result, &[1, 4, 9, 16]);
}

// =============================================================================
// Subtraction Expansion Tests
// =============================================================================

/// Test: UNROLL - scalar
#[test]
fn test_expand_sub_broadcast() {
    // Create UNROLL(VCONST([10,20,30,40]), [(1,4)])
    let unroll = create_unroll_values(1, vec![10, 20, 30, 40]);

    // Subtract scalar 5
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(5));
    let sub = unroll.try_sub(&scalar).unwrap();

    // Apply expander
    let result = phase2_only(&sub);

    // Expected: [10-5, 20-5, 30-5, 40-5] = [5, 15, 25, 35]
    assert_result_values(&result, &[5, 15, 25, 35]);
}

// =============================================================================
// Mixed Operations Expansion Tests
// =============================================================================

/// Test: (UNROLL + scalar) * UNROLL
///
/// Verifies that compound expressions expand correctly.
#[test]
fn test_expand_compound_expression() {
    // Create UNROLL(VCONST([0,1,2,3]), [(1,4)])
    let e1 = create_unroll_iota(1, 4);

    // Create UNROLL(VCONST([2,2,2,2]), [(1,4)])
    let e2 = create_unroll_values(1, vec![2, 2, 2, 2]);

    // Build: (e1 + 1) * e2 = ([0,1,2,3] + 1) * [2,2,2,2]
    let scalar = UOp::const_(DType::Int64, ConstValue::Int(1));
    let sum = e1.try_add(&scalar).unwrap();
    let result = phase2_only(&sum.try_mul(&e2).unwrap());

    // Expected: [1*2, 2*2, 3*2, 4*2] = [2, 4, 6, 8]
    assert_result_values(&result, &[2, 4, 6, 8]);
}

// =============================================================================
// Broadcast on Const short-circuits to VConst.
// =============================================================================

/// `UOp::broadcast(const, N)` emits a single `VConst` (one uop), not a
/// `Vectorize` of N cloned scalar `Const`s (N+1 uops).
#[test]
fn test_broadcast_const_emits_single_vconst() {
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(2.5));
    let broadcast = scalar.broadcast(8);

    assert!(
        matches!(broadcast.op(), svod_ir::Op::VConst { .. }),
        "broadcast(Const, 8) should produce Op::VConst, got {:?}",
        broadcast.op()
    );
    assert_eq!(broadcast.dtype().vcount(), 8);
}

/// `broadcast(N=1)` short-circuits to the same Arc (no wrapper).
#[test]
fn test_broadcast_count_one_is_passthrough() {
    let scalar = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let same = scalar.broadcast(1);

    assert!(std::sync::Arc::ptr_eq(&scalar, &same), "broadcast(_, 1) should clone the Arc, not wrap");
}

// =============================================================================
// Ptr Source Guard (do_expand Case 4)
// =============================================================================
//
// A scalar `Ptr`-typed source (e.g., a global buffer PARAM used by a tk
// per-element INDEX) MUST NOT be broadcast by `do_expand` Case 4. Broadcasting
// it produces `VECTORIZE([PARAM; N])` with dtype `Ptr{vcount:N}` — an illegal
// "vector of N pointers" that trips `rule_vectorize` (each lane's dtype
// `Ptr{vcount:1}` != `scalar_dtype()`). The fix passes the scalar `Ptr`
// through unchanged; the downstream devectorizer handles scalar-buffer +
// vector-index via `expand_index_to_vectorize` + `fold_expanded_index` +
// `distribute_ptrcat_load`.

/// `do_expand` Case 4 on a Ptr-typed buffer source: the PARAM is passed through
/// unchanged (no `VECTORIZE` of pointer-typed elements is created), so the
/// resulting INDEX keeps its scalar `Ptr` buffer. This unblocks the schedule
/// optimizer for hand-lowered (`opts_to_apply=Some(vec![])`) tk kernels.
#[test]
fn test_do_expand_ptr_not_broadcasted() {
    use svod_dtype::{AddrSpace, ScalarDType};
    use svod_ir::Op;

    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global).expect("ptr dtype");
    let buf = UOp::param(0, 64, ptr_dtype, None);

    // INDEX(buf, UNROLL(VCONST([0,1,2,3]), [(1,4)])) with ptr=true — the tk
    // per-element addressing pattern that triggers do_expand Case 4 on `buf`.
    let unroll_offset = create_unroll_iota(1, 4);
    let index = UOp::index().buffer(buf.clone()).indices(vec![unroll_offset]).ptr(true).call().expect("index");

    let result = phase2_only(&index);

    // (a) No VECTORIZE node whose elements are pointer-typed may exist anywhere
    // in the toposort — that would be the illegal "vector of pointers".
    let bad_vec = count_ops(&result, |u| match u.op() {
        Op::Vectorize { elements } => {
            !elements.is_empty() && elements.iter().all(|e| matches!(e.dtype(), DType::Ptr { .. }))
        }
        _ => false,
    });
    assert_eq!(bad_vec, 0, "do_expand Case 4 must not broadcast a Ptr-typed source into a VECTORIZE-of-pointers");

    // (b) The scalar `Ptr` PARAM survives pointer-equal — Case 4 pushed it
    // through unchanged rather than wrapping it in a vector constructor.
    let has_orig = count_ops(&result, |u| std::sync::Arc::ptr_eq(u, &buf)) >= 1;
    assert!(has_orig, "the scalar Ptr PARAM buffer must survive (not be broadcasted away)");
}

/// `do_expand` on a Ptr-typed INDEX (store-target with `ptr=true`) must preserve
/// the scalar `Ptr` dtype — NOT strip it to the element type. Before the fix,
/// `DType::Scalar(base_dtype.base()).vec(...)` converted `Ptr{vcount:1, base:bf16}`
/// to `Vector{bf16, N}`, losing the Ptr-ness that `pm_add_loads` / STORE codegen
/// depend on. This caused `pm_add_loads` to mistake the INDEX for a value-load
/// (wrapping it in `LOAD(INDEX)` with the wrong operand type).
#[test]
fn test_do_expand_ptr_index_preserves_ptr_dtype() {
    use svod_dtype::{AddrSpace, ScalarDType};
    use svod_ir::Op;

    let ptr_dtype = DType::Scalar(ScalarDType::Float32).ptr(Some(64), AddrSpace::Global).expect("ptr dtype");
    let buf = UOp::param(0, 64, ptr_dtype.clone(), None);

    // INDEX(buf, UNROLL(VCONST([0,1,2,3]), [(1,4)])) with ptr=true — a store-target
    // INDEX whose Upcast axis triggers do_expand on both the Add (index) and the
    // INDEX itself.
    let unroll_offset = create_unroll_iota(1, 4);
    let index = UOp::index().buffer(buf).indices(vec![unroll_offset]).ptr(true).call().expect("index");

    let result = phase2_only(&index);

    // After expansion, every surviving INDEX node must still have a Ptr dtype.
    // If do_expand stripped the Ptr type, the INDEX would have element dtype
    // (Vector{Float32, 4}) which breaks pm_add_loads and STORE codegen.
    let stripped =
        count_ops(&result, |u| matches!(u.op(), Op::Index { .. }) && !matches!(u.dtype(), DType::Ptr { .. }));
    assert_eq!(stripped, 0, "do_expand must not strip the Ptr dtype from a Ptr-typed INDEX");
}
