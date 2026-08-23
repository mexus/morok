use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, CallInfo, Error, Op, UOp};

use crate::rangeify::{rangeify, transforms::resolve_calls};

/// Helper: peel a TUPLE wrapper. Per tinygrad, FUNCTION bodies are TUPLE-wrapped
/// value producers; the inlined result of `resolve_function` is the substituted TUPLE.
fn peel_tuple(uop: &Arc<UOp>) -> &Arc<UOp> {
    match uop.op() {
        Op::Tuple { src } if src.len() == 1 => &src[0],
        _ => uop,
    }
}

#[test]
fn test_resolve_call_inlines_function() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0.clone(), a1.clone()], CallInfo::default());

    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    // Inlined body is wrapped in a single-element TUPLE per tinygrad invariant.
    let inlined = peel_tuple(&resolved);
    match inlined.op() {
        Op::Binary(BinaryOp::Add, lhs, rhs) => {
            assert!(Arc::ptr_eq(lhs, &a0));
            assert!(Arc::ptr_eq(rhs, &a1));
        }
        op => panic!("expected inlined add body, got {op:?}"),
    }
    assert!(!resolved.toposort().iter().any(|u| matches!(u.op(), Op::Function { .. })));
}

#[test]
fn test_resolve_call_preserves_opaque_call() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let call = body.call(smallvec![a0.clone(), a1.clone()], CallInfo::default());

    let resolved = resolve_calls(call).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Call { body: call_body, args, .. } => {
            assert!(matches!(call_body.op(), Op::Binary(BinaryOp::Add, _, _)));
            assert_eq!(args.len(), 2);
            assert!(Arc::ptr_eq(&args[0], &a0));
            assert!(Arc::ptr_eq(&args[1], &a1));
        }
        op => panic!("expected opaque CALL, got {op:?}"),
    }
}

/// PROGRAM is in tinygrad's `_OPAQUE_CALL_BODIES` (`ops.py:933`) — opaque bodies
/// belong in CALL, not FUNCTION. CALL is preserved by resolve_calls.
#[test]
fn test_resolve_call_preserves_program_call() {
    let sink = UOp::sink(vec![]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let program = UOp::program(sink, info, None, None, None);
    let call = program.call(smallvec![], CallInfo::default());

    let resolved = resolve_calls(call).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Call { body, .. } => assert!(matches!(body.op(), Op::Program { .. })),
        op => panic!("expected CALL(PROGRAM), got {op:?}"),
    }
}

/// SINK is in tinygrad's `_OPAQUE_CALL_BODIES` — wrap with CALL.
#[test]
fn test_resolve_call_preserves_sink_call() {
    let body = UOp::sink_with_info(vec![UOp::native_const(1.0f32)], svod_ir::KernelInfo::default());
    let call = body.call(smallvec![], CallInfo::default());

    let resolved = resolve_calls(call).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Call { body, .. } => assert!(matches!(body.op(), Op::Sink { .. })),
        op => panic!("expected CALL(SINK), got {op:?}"),
    }
}

#[test]
fn test_resolve_call_preserves_precompile_function() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let body = p0.try_sqrt().unwrap();
    let arg = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let info = CallInfo { precompile: true, ..CallInfo::default() };
    let function = body.function(smallvec![arg.clone()], info);

    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Function { body, args, info } => {
            // FUNCTION body is now TUPLE-wrapped per tinygrad invariant.
            let Op::Tuple { src } = body.op() else {
                panic!("expected FUNCTION body to be TUPLE, got {:?}", body.op())
            };
            assert_eq!(src.len(), 1);
            assert!(matches!(src[0].op(), Op::Unary(_, _)));
            assert_eq!(args.len(), 1);
            assert!(Arc::ptr_eq(&args[0], &arg));
            assert!(info.precompile);
        }
        op => panic!("expected precompile FUNCTION to be preserved, got {op:?}"),
    }
}

#[test]
fn test_resolve_call_preserves_precompile_gettuple_and_actual() {
    let formal = UOp::param(0, 8, DType::Float32, None);
    let actual = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let info = CallInfo { precompile: true, ..CallInfo::default() };
    let function = formal.function(smallvec![actual.clone()], info);
    let gettuple = function.try_gettuple(0).unwrap();

    let resolved = resolve_calls(gettuple).expect("precompiled FUNCTION must remain opaque");
    let Op::GetTuple { src, index: 0 } = resolved.op() else { panic!("expected GETTUPLE root") };
    let Op::Function { args, info, .. } = src.op() else { panic!("expected preserved FUNCTION source") };
    assert!(info.precompile);
    assert_eq!(args.len(), 1);
    assert!(Arc::ptr_eq(&args[0], &actual));
}

#[test]
fn test_resolve_call_keeps_nested_function_under_opaque_outer_function() {
    let nested_formal = UOp::param(0, 8, DType::Float32, None);
    let nested_actual = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let nested = nested_formal.function(smallvec![nested_actual], CallInfo::default());
    let outer_info = CallInfo { precompile: true, ..CallInfo::default() };
    let outer = nested.function(smallvec![], outer_info);

    let resolved = resolve_calls(outer).expect("opaque outer FUNCTION must preserve its body");
    let Op::Function { body, info, .. } = resolved.op() else { panic!("expected outer FUNCTION") };
    assert!(info.precompile);
    assert!(body.toposort().iter().any(|node| matches!(node.op(), Op::Function { info, .. } if !info.precompile)));
}

/// Tinygrad parity: BIND is value-producing (not in `_OPAQUE_CALL_BODIES`), so a
/// FUNCTION wrapping it gets inlined like any value body. With no PARAMs the
/// substitution is a no-op and the resolved value is the TUPLE-wrapped body.
#[test]
fn test_resolve_call_inlines_bind_body_function() {
    let var = UOp::define_var("N".to_string(), 0, 32);
    let bind = var.bind(UOp::index_const(8));
    let function = bind.function(smallvec![], CallInfo::default());

    let resolved = resolve_calls(function).expect("resolve_calls should succeed");
    let inlined = peel_tuple(&resolved);
    assert!(matches!(inlined.op(), Op::Bind { .. }));
    assert!(!resolved.toposort().iter().any(|u| matches!(u.op(), Op::Function { .. })));
}

/// CALL is opaque: a FUNCTION wrapping a CALL is unusual — express the same
/// preservation intent by wrapping with CALL outermost. resolve_calls preserves
/// the outer CALL and leaves the inner CALL alone.
#[test]
fn test_resolve_call_preserves_nested_call_body() {
    let inner_call = UOp::native_const(1.0f32).call(smallvec![], CallInfo::default());
    let outer_call = inner_call.call(smallvec![], CallInfo::default());

    let resolved = resolve_calls(outer_call).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Call { body, .. } => assert!(matches!(body.op(), Op::Call { .. })),
        op => panic!("expected CALL(CALL), got {op:?}"),
    }
}

/// SINK with non-kernel metadata still requires CALL (opaque body).
#[test]
fn test_resolve_call_preserves_sink_call_with_unrelated_metadata() {
    #[derive(Debug)]
    struct UnrelatedMarker;

    let p0 = UOp::param(0, 8, DType::Float32, None);
    let body = UOp::sink(vec![p0.clone()]).with_metadata(UnrelatedMarker);
    let arg = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let call = body.call(smallvec![arg.clone()], CallInfo::default());

    let resolved = resolve_calls(call).expect("resolve_calls should succeed");
    match resolved.op() {
        Op::Call { body, args, .. } => {
            assert!(matches!(body.op(), Op::Sink { .. }));
            assert_eq!(args.len(), 1);
            assert!(Arc::ptr_eq(&args[0], &arg));
        }
        op => panic!("expected CALL(SINK), got {op:?}"),
    }
}

#[test]
fn test_resolve_call_allows_non_contiguous_param_slots_with_unused_args() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p2 = UOp::param(2, 8, DType::Float32, None);
    let body = p0.try_add(&p2).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a2 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0.clone(), a1, a2.clone()], CallInfo::default());

    let resolved = resolve_calls(function).expect("unused argument slots should be allowed");
    assert!(resolved.toposort().iter().all(|u| !matches!(u.op(), Op::Param { .. })));
    assert!(resolved.toposort().iter().any(|u| Arc::ptr_eq(u, &a0)));
    assert!(resolved.toposort().iter().any(|u| Arc::ptr_eq(u, &a2)));
}

#[test]
fn test_resolve_call_error_arg_count_mismatch() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0], CallInfo::default());

    let err = resolve_calls(function).expect_err("resolve_calls should fail");
    assert!(matches!(err, Error::CallFormalSlotMissing { slot: 1, arg_count: 1 }));
}

#[test]
fn test_resolve_call_error_shape_mismatch() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let body = p0.try_sqrt().unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::Float32);
    let function = body.function(smallvec![a0], CallInfo::default());

    let err = resolve_calls(function).expect_err("resolve_calls should fail");
    assert!(matches!(err, Error::CallArgShapeMismatch { arg_index: 0, .. }));
}

#[test]
fn test_resolve_call_error_dtype_mismatch() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let body = p0.try_sqrt().unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Int32);
    let function = body.function(smallvec![a0], CallInfo::default());

    let err = resolve_calls(function).expect_err("resolve_calls should fail");
    assert!(matches!(err, Error::CallArgDTypeMismatch { arg_index: 0, .. }));
}

#[test]
fn test_rangeify_pipeline_runs_resolve_call() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let body = p0.try_add(&p1).unwrap();

    let a0 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let a1 = UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32);
    let function = body.function(smallvec![a0, a1], CallInfo::default());

    let (out, _ctx) = rangeify(function).expect("rangeify should succeed");
    assert!(!out.toposort().iter().any(|u| matches!(u.op(), Op::Function { .. })));
}

#[test]
fn test_rangeify_consumes_expression_valued_function_result_shape() {
    let p1 = UOp::scalar_param(1, Some("p1".into()), DType::WeakInt, 1, 8);
    let extent = p1.try_add(&p1).unwrap();
    let formal = UOp::param_with_shape(0, &smallvec![svod_ir::SInt::Symbolic(extent)], DType::Float32, None);
    let actual_dim = UOp::define_var("actual".into(), 1, 8);
    let actual_extent = actual_dim.try_add(&actual_dim).unwrap();
    let actual =
        UOp::param_with_shape(7, &smallvec![svod_ir::SInt::Symbolic(actual_extent.clone())], DType::Float32, None);
    let output = formal.function(smallvec![actual, actual_dim], CallInfo::default()).try_gettuple(0).unwrap();

    assert_eq!(output.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Symbolic(actual_extent)]);
    let (resolved, _) = rangeify(output).expect("rangeify should consume substituted call shape");
    assert!(resolved.toposort().iter().all(|node| !matches!(node.op(), Op::Function { .. })));
}

#[test]
fn test_rangeify_preserves_kernel_call_body_boundaries() {
    let detached = UOp::native_const(1.0f32).detach();
    let body = UOp::sink_with_info(vec![detached], svod_ir::KernelInfo::default());
    let function = body.call(smallvec![], CallInfo::default());

    let (out, _ctx) = rangeify(function).expect("rangeify should succeed");
    let call_node = out
        .toposort()
        .into_iter()
        .find(|u| matches!(u.op(), Op::Call { .. }))
        .expect("kernel call should be preserved");

    let Op::Call { body, .. } = call_node.op() else { unreachable!("filtered to call node") };
    assert!(
        body.toposort().iter().any(|u| matches!(u.op(), Op::Detach { .. })),
        "call-preserving rangeify should not rewrite inside preserved kernel call bodies"
    );
}

#[test]
fn test_resolve_call_does_not_inline_function_inside_opaque_call_body() {
    let p0 = UOp::param(0, 8, DType::Float32, None);
    let p1 = UOp::param(1, 8, DType::Float32, None);
    let nested_body = p0.try_add(&p1).unwrap();
    let nested_function = nested_body.function(
        smallvec![
            UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32),
            UOp::new_buffer(DeviceSpec::Cpu, 8, DType::Float32)
        ],
        CallInfo::default(),
    );
    let opaque_call = nested_function.call(smallvec![], CallInfo::default());

    let resolved = resolve_calls(opaque_call).expect("resolve_calls should preserve call body");
    let Op::Call { body, .. } = resolved.op() else { panic!("expected CALL root") };
    assert!(
        body.toposort().iter().any(|u| matches!(u.op(), Op::Function { .. })),
        "FUNCTION inside CALL body must remain unresolved"
    );
}
