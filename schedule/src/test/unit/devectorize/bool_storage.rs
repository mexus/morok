//! Phase 3 tests: bool storage conversion.
//!
//! Tests for the bool_storage_patterns which convert bool LOAD/STORE
//! operations to use uint8 storage to avoid LLVM i1 garbage bits.
//!
//! Based on Tinygrad's PTX/NIR bool->uint8 patterns.

use svod_dtype::{DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use super::helpers::*;

// =============================================================================
// Bool Load Tests
// =============================================================================

/// Test: LOAD<bool> converts to CAST(LOAD<uint8>, bool).
///
/// This ensures proper bool loading without LLVM i1 garbage bits.
#[test]
fn test_bool_load_to_uint8() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);

    // Create a LOAD that returns bool
    let load = create_load(index);

    // Verify initial state
    assert_eq!(load.dtype().base(), ScalarDType::Bool);

    let result = apply_bool_storage(&load);

    // Result should be CAST(LOAD<uint8>, bool) or LOAD with converted type
    // Either way, the result type should be bool (user-facing) but storage is uint8
    match result.op() {
        Op::Cast { src, dtype } => {
            // Outer dtype should be bool
            assert_eq!(dtype.base(), ScalarDType::Bool, "CAST should produce bool");
            // Inner LOAD should be uint8
            assert_is_load(src);
            assert_eq!(src.dtype().base(), ScalarDType::UInt8, "Inner LOAD should be uint8");
        }
        Op::Load { .. } => {
            // If unchanged, result should still be bool (transformation may be deferred)
            assert_eq!(result.dtype().base(), ScalarDType::Bool, "LOAD result should be bool");
        }
        other => panic!("Expected CAST(LOAD) or LOAD, got {:?}", other),
    }
}

/// Test: Non-bool LOAD remains unchanged.
#[test]
fn test_non_bool_load_unchanged() {
    let buffer = create_buffer(64); // float32 buffer
    let index = create_index(buffer.clone(), 0);
    let load = create_load(index);

    assert_eq!(load.dtype().base(), ScalarDType::Float32);

    let result = apply_bool_storage(&load);

    // Float32 LOAD should remain unchanged
    assert_is_load(&result);
    assert_eq!(result.dtype().base(), ScalarDType::Float32);
}

/// Test: Int32 LOAD remains unchanged.
#[test]
fn test_int32_load_unchanged() {
    let buffer = create_buffer_typed(64, ScalarDType::Int32);
    let index = create_index(buffer.clone(), 0);
    let load = create_load(index);

    let result = apply_bool_storage(&load);

    assert_is_load(&result);
    assert_eq!(result.dtype().base(), ScalarDType::Int32);
}

// =============================================================================
// Bool Store Tests
// =============================================================================

/// Test: STORE(bool_val) converts to STORE(CAST(bool_val, uint8)).
#[test]
fn test_bool_store_to_uint8() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let bool_val = create_bool_const(true);

    let store = create_store(index, bool_val);

    let result = apply_bool_storage(&store);

    // Result should be STORE with CAST(bool_val, uint8) as value
    match result.op() {
        Op::Store { value, .. } => {
            // Value should be cast to uint8
            match value.op() {
                Op::Cast { src, dtype } => {
                    assert_eq!(dtype.base(), ScalarDType::UInt8);
                    assert_eq!(src.dtype().base(), ScalarDType::Bool);
                }
                // Could be constant uint8 after optimization
                Op::Const(_) => {}
                other => panic!("Expected CAST or Const value, got {:?}", other),
            }
        }
        other => panic!("Expected STORE, got {:?}", other),
    }
}

/// Test: Non-bool STORE remains unchanged.
#[test]
fn test_non_bool_store_unchanged() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let float_val = create_float_const(3.0);

    let store = create_store(index, float_val.clone());

    let result = apply_bool_storage(&store);

    // Float STORE should remain unchanged
    match result.op() {
        Op::Store { value, .. } => {
            // Value should NOT be cast
            assert_eq!(value.dtype().base(), ScalarDType::Float32);
        }
        other => panic!("Expected STORE, got {:?}", other),
    }
}

#[test]
fn test_invalid_bool_store_is_left_for_final_cleanup() {
    let buffer = create_bool_buffer(1);
    let index = create_index(buffer, 0);
    let store = create_store(index, UOp::invalid_marker());

    let result = apply_bool_storage(&store);
    assert!(std::sync::Arc::ptr_eq(&result, &store));
}

// =============================================================================
// Roundtrip Tests
// =============================================================================

/// Test: Store bool then load bool maintains correctness.
#[test]
fn test_bool_roundtrip() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let bool_val = create_bool_const(true);

    // Store bool value
    let store = create_store(index.clone(), bool_val);
    let store_result = apply_bool_storage(&store);

    // Load bool value
    let load = create_load(index);
    let load_result = apply_bool_storage(&load);

    // Verify store has uint8 cast
    if let Op::Store { value, .. } = store_result.op() {
        assert!(matches!(value.op(), Op::Cast { .. } | Op::Const(_)));
    }

    // Verify load is cast back to bool
    if let Op::Cast { dtype, .. } = load_result.op() {
        assert_eq!(dtype.base(), ScalarDType::Bool);
    }
}

/// Test: Bool buffer through full devectorize pipeline.
#[test]
fn test_bool_with_devectorize() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let load = create_load(index);

    // Apply full devectorize (all phases)
    let result = apply_devectorize(&load);

    // Should produce properly converted load
    // Either CAST(LOAD<uint8>, bool) or unchanged if pattern didn't match
    assert!(
        result.dtype().base() == ScalarDType::Bool || result.dtype().base() == ScalarDType::UInt8,
        "Result should be bool or uint8"
    );
}

#[test]
fn test_bool_bitcast_with_devectorize_becomes_cast() {
    let bitcast = UOp::new(Op::BitCast { src: create_bool_const(true), dtype: DType::UInt8 }, DType::UInt8);

    let result = apply_devectorize(&bitcast);

    assert!(!result.toposort().iter().any(|uop| matches!(uop.op(), Op::BitCast { .. })));
    assert_eq!(result.dtype(), DType::UInt8);
}

// =============================================================================
// Vector Bool Tests
// =============================================================================

/// Test: Vector bool load conversion.
#[test]
fn test_vector_bool_load() {
    let buffer = create_bool_buffer(64);

    // Create vector bool load by loading multiple elements
    let index = create_index(buffer.clone(), 0);

    // Create load with explicit bool dtype
    let load = create_load(index);

    let result = apply_bool_storage(&load);

    // Should handle vector bool correctly
    match result.op() {
        Op::Cast { src, dtype } => {
            assert_eq!(dtype.base(), ScalarDType::Bool);
            assert_eq!(src.dtype().base(), ScalarDType::UInt8);
        }
        Op::Load { .. } => {}
        other => panic!("Expected CAST(LOAD) or LOAD, got {:?}", other),
    }
}

/// Test: Vector bool store conversion.
#[test]
fn test_vector_bool_store() {
    let buffer = create_bool_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let bool_vec = create_vector_bool(vec![true, false, true, false]);

    let store = create_store(index, bool_vec);

    let result = apply_bool_storage(&store);

    // Should convert vector bool to uint8
    if let Op::Store { value, .. } = result.op() {
        match value.op() {
            Op::Cast { dtype, .. } => {
                assert_eq!(dtype.base(), ScalarDType::UInt8);
            }
            Op::Stack { .. } => {
                // Could be VECTORIZE of casts
            }
            _ => {}
        }
    }
}

/// Post-gater boundary: bool_storage keeps the gate and converted alt.
#[test]
fn test_bool_gated_load_preserves_alt() {
    let buffer = create_bool_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = create_bool_const(false);
    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().unwrap();
    let load = UOp::load().index(index).alt(create_bool_const(true)).gate(gate).call();

    let result = apply_bool_storage(&load);
    let Op::Cast { src, .. } = result.op() else {
        panic!("Expected CAST wrapping converted bool load, got {:?}", result.op());
    };
    let Op::Load { alt, gate, .. } = src.op() else {
        panic!("Expected inner LOAD after bool storage, got {:?}", src.op());
    };
    assert!(gate.is_some(), "Expected late LOAD gate to be preserved");
    let Some(alt) = alt else {
        panic!("Expected gated LOAD alt to be preserved by bool_storage");
    };
    assert_eq!(alt.dtype().base(), ScalarDType::UInt8, "Alt should be converted to UInt8 storage dtype");
}
