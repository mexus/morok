use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, UOp};

use super::helpers::extract_kernel;
use crate::rangeify::kernel::{split_store, try_get_kernel_graph};

/// Helper to call split_store with the new signature
fn call_split_store(x: &Arc<UOp>) -> Option<Arc<UOp>> {
    let mut uop_list = Vec::new();
    split_store(&mut uop_list, x)
}

fn closed_range_count(uop: &Arc<UOp>) -> usize {
    match uop.op() {
        Op::End { ranges, .. } => ranges.len(),
        Op::Sink { sources, .. } => sources.iter().map(closed_range_count).sum(),
        _ => 0,
    }
}

fn effect_body(uop: &Arc<UOp>) -> &Arc<UOp> {
    match uop.op() {
        Op::End { computation, .. } => computation,
        _ => uop,
    }
}

fn expect_end_call(uop: &Arc<UOp>, expected_ranges: usize) -> Arc<UOp> {
    let Op::Call { body, .. } = uop.op() else {
        panic!("Expected CALL, got {:?}", uop.op());
    };
    assert_eq!(closed_range_count(body), expected_ranges);
    uop.clone()
}

#[test]
fn test_split_store_basic() {
    // Create a simple STORE operation with a proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value);

    // Try to split
    let result = call_split_store(&store);

    // Should return a CALL
    assert!(result.is_some());
    let kernel = result.unwrap();
    assert!(matches!(kernel.op(), Op::Call { .. }));

    // BUFFER should be tracked in call args
    if let Op::Call { args: sources, .. } = kernel.op() {
        // With proper BUFFER ops, sources should contain the buffer
        assert!(!sources.is_empty(), "Kernel sources should contain the buffer");
    }
}

#[test]
fn test_split_store_dense_global_params_with_internal_buffer() {
    // Original storage identities are intentionally sparse. Only the two
    // globals belong to the CALL tuple; wrapped local/register allocations and
    // a scalar variable do not consume global slots.
    let output = UOp::buffer(41, 4, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    let local = UOp::buffer(700, 4, DType::Float32, AddrSpace::Local, None);
    let local_peer = UOp::buffer(701, 4, DType::Float32, AddrSpace::Local, None);
    let reg = UOp::buffer(800, 1, DType::Float32, AddrSpace::Reg, None);
    let input = UOp::buffer(990, 4, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    let scalar = UOp::variable("N".to_string(), 1, 4, DType::Float32).bind(UOp::native_const(2.0f32));
    let zero = UOp::index_const(0);

    let output_idx = UOp::index().buffer(output.clone()).indices(vec![zero.clone()]).call().unwrap();
    let local_stack = UOp::new(
        Op::MStack { buffers: smallvec![local.clone(), local_peer] },
        DType::Float32.ptr(Some(8), AddrSpace::Local).unwrap(),
    )
    .after(smallvec![UOp::noop()]);
    let local_idx = UOp::index().buffer(local_stack).indices(vec![zero.clone()]).call().unwrap();
    let reg_idx =
        UOp::index().buffer(reg.clone().after(smallvec![UOp::noop()])).indices(vec![zero.clone()]).call().unwrap();
    let input_idx = UOp::index().buffer(input.clone()).indices(vec![zero]).call().unwrap();
    let value = UOp::load()
        .index(local_idx)
        .call()
        .try_add(&UOp::load().index(reg_idx).call())
        .unwrap()
        .try_add(&UOp::load().index(input_idx).call())
        .unwrap()
        .try_add(&scalar)
        .unwrap();
    let call = call_split_store(&output_idx.store(value)).expect("STORE should split");

    let Op::Call { body, args, .. } = call.op() else { panic!("expected CALL") };
    assert_eq!(args.len(), 3, "CALL sources are two globals followed by the scalar binding");
    assert!(args.iter().any(|arg| Arc::ptr_eq(arg, &output)));
    assert!(args.iter().any(|arg| Arc::ptr_eq(arg, &input)));
    assert!(matches!(args.last().unwrap().op(), Op::Bind { .. }));
    let Op::Bind { var: call_var, value: call_value } = args.last().unwrap().op() else { unreachable!() };
    let Op::Bind { var: body_var, value: body_value } = scalar.op() else { unreachable!() };
    assert!(Arc::ptr_eq(call_value, body_value));
    assert!(!Arc::ptr_eq(call_var, body_var), "CALL binding must not alias the body-local PARAM");
    let (Op::Param { arg: call_arg, .. }, Op::Param { arg: body_arg, .. }) = (call_var.op(), body_var.op()) else {
        panic!("scalar binding must retain PARAM semantics")
    };
    assert_eq!(call_arg, body_arg, "boundary identity must not change scalar metadata");
    assert_eq!(call_var.dtype(), body_var.dtype());

    let mut global_slots: Vec<usize> = body
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Param { arg, .. } if arg.addrspace == Some(AddrSpace::Global) => Some(arg.slot),
            _ => None,
        })
        .collect();
    global_slots.sort_unstable();
    global_slots.dedup();
    assert_eq!(global_slots, vec![0, 1], "PARAM slots must be dense CALL positions");
    let program_info = svod_ir::ProgramInfo::from_sink(body, DeviceSpec::Cpu);
    assert_eq!(program_info.globals, vec![0, 1], "PROGRAM globals are direct CALL tuple positions");
    assert_eq!(program_info.vars.len(), 1, "scalar variables are PROGRAM values, not globals");
    assert!(body.toposort().iter().any(
        |u| matches!(u.op(), Op::Buffer { arg, .. } if arg.addrspace == Some(AddrSpace::Local) && arg.slot == 700)
    ));
    assert!(
        body.toposort().iter().any(
            |u| matches!(u.op(), Op::Buffer { arg, .. } if arg.addrspace == Some(AddrSpace::Reg) && arg.slot == 800)
        )
    );
}

#[test]
fn test_split_store_non_store_returns_none() {
    // Create a non-STORE operation
    let const_op = UOp::native_const(1.0f32);

    // Try to split
    let result = call_split_store(&const_op);

    // Should return None
    assert!(result.is_none());
}

#[test]
fn test_split_store_end_operation() {
    // Create an END operation wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value);
    let range = UOp::range_const(10, 0);
    let end = store.clone().end(smallvec![range.clone()]);

    // Try to split
    let result = call_split_store(&end);

    // Should process END wrapping STORE
    assert!(result.is_some());
    let end_call = result.unwrap();
    let kernel = expect_end_call(&end_call, 1);

    // Verify CALL structure: body should be SINK wrapping transformed END
    if let Op::Call { args: sources, body: ast, .. } = kernel.op() {
        // With proper BUFFER, sources should contain the buffer mapping
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");

        if let Op::Sink { sources: sink_sources, .. } = ast.op() {
            // SINK should wrap the transformed computation
            assert_eq!(sink_sources.len(), 1);
            // Note: The END may be transformed, so we just check it's an END with ranges
            if let Op::End { ranges, .. } = sink_sources[0].op() {
                assert_eq!(ranges.len(), 1);
            } else {
                panic!("Expected END operation in SINK");
            }
        } else {
            panic!("Expected SINK operation");
        }
    } else {
        panic!("Expected CALL operation");
    }
}

#[test]
fn test_split_store_end_non_store_returns_none() {
    // Create an END operation wrapping non-STORE (control flow marker)
    let noop = UOp::noop();
    let range = UOp::range_const(10, 0);
    let end = noop.end(smallvec![range]);

    // Try to split
    let result = call_split_store(&end);

    // Should return None (skip control flow markers)
    assert!(result.is_none());
}

#[test]
fn test_split_store_creates_sink() {
    // Create a STORE operation with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value.clone());

    let result = call_split_store(&store).unwrap();

    // Extract the body from the CALL
    if let Op::Call { args: sources, body: ast, .. } = result.op() {
        // The AST should be a SINK operation
        if let Op::Sink { sources: sink_sources, .. } = ast.op() {
            // SINK should wrap the transformed STORE
            assert_eq!(sink_sources.len(), 1);

            // Verify the STORE structure has codegen PARAM (buffer converted)
            if let Op::Store { index: store_index, value: store_val, .. } = sink_sources[0].op() {
                // Index should contain the buffer reference
                let Op::Index { buffer: store_buf, .. } = store_index.op() else {
                    panic!("Expected INDEX operation in STORE, got {:?}", store_index.op());
                };
                // Buffer should be converted to codegen PARAM
                assert!(
                    matches!(store_buf.op(), Op::Param { arg, .. } if arg.device == Some(svod_dtype::DeviceSpec::Cpu)),
                    "Expected codegen PARAM, got {:?}",
                    store_buf.op()
                );
                // Value should be preserved
                assert!(std::sync::Arc::ptr_eq(store_val, &value));
            } else {
                panic!("Expected STORE in SINK sources");
            }
        } else {
            panic!("Expected SINK operation");
        }

        // With proper BUFFER ops, sources should contain the buffer mapping
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");
    } else {
        panic!("Expected CALL operation");
    }
}

#[test]
fn test_split_store_preserves_computation() {
    // Create STOREs with different value dtypes
    let test_cases = [
        (DType::Float32, ConstValue::Float(1.0)),
        (DType::Int32, ConstValue::Int(1)),
        (DType::Bool, ConstValue::Bool(true)),
    ];

    for (_dtype_idx, (dtype, _const_val)) in test_cases.iter().enumerate() {
        let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, dtype.clone());
        let const_idx = UOp::index_const(0);
        let value = match _dtype_idx {
            0 => UOp::native_const(1.0f32),
            1 => UOp::native_const(1i32),
            2 => UOp::native_const(true),
            _ => panic!("Unsupported dtype index"),
        };
        let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
        let store = store_idx.store(value.clone());

        let result = call_split_store(&store);

        assert!(result.is_some());
        let kernel = result.unwrap();

        // Verify CALL structure
        if let Op::Call { body: ast, .. } = kernel.op()
            && let Op::Sink { sources, .. } = ast.op()
        {
            // Verify the stored value dtype is preserved
            if let Op::Store { value: stored_val, .. } = sources[0].op() {
                assert_eq!(stored_val.dtype(), *dtype);
                assert!(std::sync::Arc::ptr_eq(stored_val, &value));
            }
        }
    }
}

#[test]
fn test_split_store_multiple_calls_independent() {
    // Create two different STORE operations with proper BUFFER ops
    let buffer1 = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let idx_offset1 = UOp::index_const(0);
    let value1 = UOp::native_const(1.0f32);
    let index1 = UOp::index().buffer(buffer1).indices(vec![idx_offset1]).call().unwrap();
    let store1 = index1.store(value1);

    let buffer2 = UOp::new_buffer(DeviceSpec::Cpu, 200, DType::Float32);
    let idx_offset2 = UOp::index_const(0);
    let value2 = UOp::native_const(2.0f32);
    let index2 = UOp::index().buffer(buffer2).indices(vec![idx_offset2]).call().unwrap();
    let store2 = index2.store(value2);

    // Split both
    let kernel1 = call_split_store(&store1).unwrap();
    let kernel2 = call_split_store(&store2).unwrap();

    // Both should be valid calls
    assert!(matches!(kernel1.op(), Op::Call { .. }));
    assert!(matches!(kernel2.op(), Op::Call { .. }));

    // They should be different kernels (different UOps)
    assert!(!std::sync::Arc::ptr_eq(&kernel1, &kernel2));
}

#[test]
fn test_split_store_end_with_multiple_ranges() {
    // Create END with multiple ranges wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value);
    let range1 = UOp::range_const(4, 0);
    let range2 = UOp::range_const(8, 1);
    let end = store.clone().end(smallvec![range1.clone(), range2.clone()]);

    let result = call_split_store(&end);

    assert!(result.is_some());

    // Verify CALL body preserves the closed END ranges.
    let kernel = expect_end_call(&result.unwrap(), 2);
    if let Op::Call { args: sources, body: ast, .. } = kernel.op() {
        // With proper BUFFER, sources should contain buffer mappings
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");

        if let Op::Sink { sources: sink_sources, .. } = ast.op() {
            // SINK should wrap the END (possibly transformed)
            assert_eq!(sink_sources.len(), 1);

            // Verify END structure with multiple ranges is preserved
            if let Op::End { ranges, .. } = sink_sources[0].op() {
                assert_eq!(ranges.len(), 2);
            } else {
                panic!("Expected END operation in SINK");
            }
        } else {
            panic!("Expected SINK operation");
        }
    } else {
        panic!("Expected CALL operation");
    }
}

#[test]
fn test_split_store_end_with_outer_range() {
    // Create END with OUTER range wrapping a STORE with proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value);
    let range_outer = UOp::range_axis(UOp::index_const(10), AxisId::Renumbered(0), AxisType::Loop);
    let end = store.end(smallvec![range_outer]);

    let result = call_split_store(&end);

    // END closes ranges, so this should still split.
    let result = result.expect("OUTER END should create kernel");
    expect_end_call(&result, 1);
}

#[test]
fn test_split_store_end_with_mixed_ranges() {
    // Create END with mix of LOOP and OUTER ranges wrapping a STORE with proper BUFFER.
    // END closes ranges, so range order does not gate splitting.
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value);
    let range_loop = UOp::range_const(4, 0);
    let range_outer = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), AxisType::Loop);

    // [LOOP, OUTER]: kernel is created.
    let end = store.end(smallvec![range_loop.clone(), range_outer.clone()]);
    let result = call_split_store(&end);
    let result = result.expect("LOOP+OUTER END should create kernel");
    expect_end_call(&result, 2);

    // [OUTER, LOOP]: also creates kernel.
    let end = store.end(smallvec![range_outer, range_loop]);
    let result = call_split_store(&end);
    let result = result.expect("OUTER+LOOP END should create kernel");
    expect_end_call(&result, 2);
}

// ============================================================================
// COPY Support Tests
// ============================================================================

#[test]
fn test_split_store_with_copy() {
    // Create a COPY operation with proper BUFFER
    let src_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let copy = src_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    // Create STORE using the COPY result with proper BUFFER
    let output_buffer = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(copy.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is COPY, not SINK
    if let Op::Call { body: ast, .. } = kernel.op() {
        assert!(matches!(ast.op(), Op::Copy { .. }), "Expected COPY operation as kernel AST, got: {:?}", ast.op());
    } else {
        panic!("Expected CALL operation");
    }
}

#[test]
fn test_split_store_normal_computation_uses_sink() {
    // Create normal arithmetic computation (no COPY)
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let value = a.try_add(&b).unwrap();

    // Create STORE with normal computation using proper BUFFER
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(value.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is SINK (normal case)
    if let Op::Call { args: sources, body: ast, .. } = kernel.op() {
        // With proper BUFFER, sources should contain buffer mappings
        assert!(!sources.is_empty(), "Kernel sources should contain buffer mappings");

        if let Op::Sink { sources: sink_sources, .. } = ast.op() {
            // SINK should wrap the transformed STORE
            assert_eq!(sink_sources.len(), 1);
        } else {
            panic!("Expected SINK operation for normal computation, got: {:?}", ast.op());
        }
    } else {
        panic!("Expected CALL operation");
    }
}

#[test]
fn test_split_store_nested_copy_in_store() {
    // Create nested structure: END(STORE(COPY)) with proper BUFFER
    let src_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let copy = src_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });

    let output_buffer = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(copy.clone());

    let range = UOp::range_const(10, 0);
    let end = store.end(smallvec![range]);

    let result = call_split_store(&end);

    assert!(result.is_some());
    let kernel = expect_end_call(&result.unwrap(), 1);

    // Verify COPY kernel AST remains directly recoverable for mixed-op runtime lowering.
    if let Op::Call { body: ast, .. } = kernel.op() {
        assert!(matches!(effect_body(ast).op(), Op::Copy { .. }), "Expected COPY kernel AST, got: {:?}", ast.op());
    } else {
        panic!("Expected CALL operation");
    }
}

#[test]
fn test_split_store_simple_store_splits_directly() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 64, DType::Float32);
    let idx = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let value = UOp::native_const(1.0f32);

    // A STORE with no open ranges (scalar, loop-invariant) splits directly
    // into a single-kernel CALL — no END wrapper needed.
    let store = idx.store(value.clone());
    assert!(store.in_scope_ranges().is_empty(), "simple store should have no open ranges");
    assert!(call_split_store(&store).is_some(), "simple STORE should split directly");
}

#[test]
fn test_split_store_open_loop_range_returns_none() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 64, DType::Float32);
    let range = UOp::range_const(4, 0);
    let idx = UOp::index().buffer(buffer).indices(vec![range.clone()]).call().unwrap();
    let store = idx.store(UOp::native_const(1.0f32));

    assert!(!store.in_scope_ranges().is_empty(), "test fixture should contain an open range");
    assert!(call_split_store(&store).is_none(), "STORE with open LOOP range must not split");
}

#[test]
fn test_split_store_open_device_range_splits() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 2, DType::Float32);
    let device = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(0), AxisType::Device);
    let idx = UOp::index().buffer(buffer).indices(vec![device]).call().unwrap();
    let store = idx.store(UOp::native_const(1.0f32));

    assert!(!store.in_scope_ranges().is_empty());
    assert!(call_split_store(&store).is_some(), "DEVICE is a launch lane, not a computational loop");
}

#[test]
fn test_split_store_open_device_and_loop_ranges_does_not_split() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let device = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(0), AxisType::Device);
    let loop_range = UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(1), AxisType::Loop);
    let index = device.mul(&UOp::index_const(2)).add(&loop_range);
    let idx = UOp::index().buffer(buffer).indices(vec![index]).call().unwrap();
    let store = idx.store(UOp::native_const(1.0f32));

    assert!(call_split_store(&store).is_none(), "open LOOP must remain an interior STORE");
}

#[test]
fn test_split_store_open_outer_range_returns_none() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 64, DType::Float32);
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);
    let idx = UOp::index().buffer(buffer).indices(vec![range.clone()]).call().unwrap();
    let store = idx.store(UOp::native_const(1.0f32));

    assert!(!store.in_scope_ranges().is_empty(), "test fixture should contain an open range");
    assert!(call_split_store(&store).is_none(), "STORE with open OUTER range must not split");
}

#[test]
fn test_try_get_kernel_graph_allows_cross_device_copy() {
    let src = UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32);
    let copy = src.copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    let out = UOp::new_buffer(DeviceSpec::Cuda { device_id: 0 }, 16, DType::Float32);
    let idx = UOp::index().buffer(out).indices(vec![UOp::index_const(0)]).call().unwrap();
    let root = UOp::sink(vec![idx.store(copy)]);

    let graph = match try_get_kernel_graph(root) {
        Ok((graph, _)) => graph,
        Err(err) => panic!("cross-device COPY should be allowed: {err}"),
    };
    let kernel = extract_kernel(&graph).expect("copy call");
    let Op::Call { body, .. } = kernel.op() else {
        panic!("expected CALL");
    };
    assert!(matches!(body.op(), Op::Copy { .. }), "expected direct COPY body, got {:?}", body.op());
}

#[test]
fn test_try_get_kernel_graph_ignores_bind_args_for_device_validation() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32);
    let var = UOp::define_var("i".to_string(), 0, 15);
    let cuda_param = UOp::param(1, 1, DType::Index, Some(DeviceSpec::Cuda { device_id: 0 }));
    let bind_with_device_storage = var.bind(cuda_param);
    let idx = UOp::index().buffer(buffer).indices(vec![bind_with_device_storage]).call().unwrap();
    let root = UOp::sink(vec![idx.store(UOp::native_const(1.0f32))]);

    match try_get_kernel_graph(root) {
        Ok((graph, _)) => assert!(extract_kernel(&graph).is_some(), "expected a split kernel"),
        Err(err) => panic!("BIND args should not participate in kernel device validation: {err}"),
    }
}

#[test]
fn test_split_store_copy_precedence_documented() {
    // This test documents the COPY detection behavior.
    //
    // **Behavior:** The stored value of the STORE is checked directly for
    // COPY ops. If found, the COPY becomes the kernel AST directly.

    // Create nested COPY: COPY(COPY(buffer)) with proper BUFFER
    let base_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let copy1 = base_buffer.copy_to_device(DeviceSpec::Cuda { device_id: 0 });
    let copy2 = copy1.clone().copy_to_device(DeviceSpec::Cpu);

    let output_buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let const_idx = UOp::index_const(0);
    let store_idx = UOp::index().buffer(output_buffer).indices(vec![const_idx]).call().unwrap();
    let store = store_idx.store(copy2.clone());

    let result = call_split_store(&store);

    assert!(result.is_some());
    let kernel = result.unwrap();

    // Verify kernel AST is one of the COPY operations (toposort order determines which)
    if let Op::Call { body: ast, .. } = kernel.op() {
        assert!(matches!(ast.op(), Op::Copy { .. }), "Expected COPY operation as kernel AST, got: {:?}", ast.op());
    } else {
        panic!("Expected CALL operation");
    }
}
