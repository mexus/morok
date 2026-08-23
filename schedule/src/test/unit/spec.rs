//! Rule-level tests for Tinygrad's `spec_program` at pinned commit 8c8b43de.
//!
//! Program-only rules are tested directly, including ordering-sensitive rules.
//! `spec_tensor` is used where useful to prove that a rejection comes from the
//! program rule rather than an inherited `spec_shared` rule.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::types::ConstValue;
use svod_ir::{BinaryOp, ConstValueHash, Op, ParamArg, ReduceOp, UOp};

use crate::optimizer::apply_pre_optimization;
use crate::spec::{
    SpecError, spec_hcq, spec_program, spec_tensor, type_verify, verify_kernel_graph, verify_no_legacy_index_dtype,
};

fn global_param(slot: usize) -> Arc<UOp> {
    UOp::new(
        Op::Param {
            shape: UOp::stack(smallvec![]),
            arg: ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None),
        },
        DType::Float32,
    )
}

fn int_const(dtype: DType, value: i64) -> Arc<UOp> {
    UOp::const_(dtype, ConstValue::Int(value))
}

fn verify_program_err(root: &Arc<UOp>) -> String {
    type_verify(root, &spec_program()).expect_err("expected spec_program rejection").to_string()
}

fn kernel_param(slot: usize, dtype: DType) -> Arc<UOp> {
    UOp::new(
        Op::Param {
            shape: UOp::stack(smallvec![]),
            arg: ParamArg::buffer(slot, dtype.clone(), AddrSpace::Global, None),
        },
        dtype,
    )
}

fn kernel_call(formals: Vec<Arc<UOp>>, args: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::sink(formals).call(args.into(), svod_ir::CallInfo::default())
}

#[test]
fn spec_kernel_graph_accepts_valid_multi_output_call_graph() {
    let formal = kernel_param(0, DType::Float32);
    let input = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let call = kernel_call(vec![formal], vec![input]);
    let out0 = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32).after(smallvec![call.clone()]);
    let out1 = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32).after(smallvec![call]);

    verify_kernel_graph(&UOp::sink(vec![out0, out1])).expect("valid multi-output kernel graph");
}

#[test]
fn spec_kernel_graph_rejects_call_body_and_argument_order() {
    let bad_body = UOp::native_const(0i32).call(smallvec![], svod_ir::CallInfo::default());
    let err = verify_kernel_graph(&UOp::sink(vec![bad_body])).expect_err("CONST is not an opaque kernel body");
    assert!(err.to_string().contains("supported opaque body"), "unexpected error: {err}");

    let float_formal = kernel_param(0, DType::Float32);
    let int_formal = kernel_param(1, DType::Int32);
    let float_actual = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let int_actual = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Int32);
    let reversed = kernel_call(vec![float_formal, int_formal], vec![int_actual, float_actual]);
    let err = verify_kernel_graph(&UOp::sink(vec![reversed])).expect_err("CALL args are positional");
    assert!(err.to_string().contains("positional arguments"), "unexpected error: {err}");
}

#[test]
fn spec_kernel_graph_checks_mselect_index_and_mstack_layout() {
    let cpu = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let cuda = UOp::new_buffer(svod_dtype::DeviceSpec::Cuda { device_id: 0 }, 4, DType::Float32);
    let stack = UOp::mstack(smallvec![cpu.clone(), cuda]);
    verify_kernel_graph(&UOp::sink(vec![stack.mselect(1)])).expect("concrete-device MSTACK layout");

    let err = verify_kernel_graph(&UOp::sink(vec![stack.mselect(2)])).expect_err("MSELECT index must be in range");
    assert!(err.to_string().contains("in-range MSTACK"), "unexpected error: {err}");

    let device_free = kernel_param(0, DType::Float32);
    let malformed = UOp::mstack(smallvec![cpu, device_free]);
    let err = verify_kernel_graph(&UOp::sink(vec![malformed])).expect_err("mixed device metadata must fail");
    assert!(err.to_string().contains("MSTACK"), "unexpected error: {err}");
}

#[test]
fn spec_kernel_graph_checks_after_dependency_shape_and_context() {
    let output = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let malformed = output.after(smallvec![UOp::native_const(1i32)]);
    let err = verify_kernel_graph(&UOp::sink(vec![malformed.clone()])).expect_err("AFTER dependency must be callable");
    assert!(err.to_string().contains("CALL/AFTER dependencies"), "unexpected error: {err}");
    match err {
        SpecError::Verification { boundary, uop_id, source_path, .. } => {
            assert_eq!(boundary, "kernel graph");
            assert_eq!(uop_id, malformed.id);
            assert_eq!(source_path, vec![0]);
        }
    }
}

#[test]
fn spec_kernel_graph_accepts_copy_only_as_supported_opaque_call_body() {
    let formal = kernel_param(0, DType::Float32);
    let copy = formal.copy_to_device(svod_dtype::DeviceSpec::Cuda { device_id: 0 });
    let input = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let call = copy.call(smallvec![input], svod_ir::CallInfo::default());
    verify_kernel_graph(&UOp::sink(vec![call])).expect("cross-device COPY call");

    let direct = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32)
        .copy_to_device(svod_dtype::DeviceSpec::Cuda { device_id: 0 });
    let err = verify_kernel_graph(&UOp::sink(vec![direct])).expect_err("bare COPY must not survive in the outer graph");
    assert!(err.to_string().contains("no matching rule"), "unexpected error: {err}");
}

#[test]
fn spec_hcq_accepts_exact_getaddr_and_rejects_non_storage_source() {
    let param = global_param(0);
    let address = UOp::new(Op::GetAddr { src: param, device: svod_dtype::DeviceSpec::Cpu }, DType::UInt64);
    assert!(type_verify(&UOp::sink(vec![address]), &spec_hcq()).is_ok());

    let invalid =
        UOp::new(Op::GetAddr { src: UOp::native_const(1u32), device: svod_dtype::DeviceSpec::Cpu }, DType::UInt64);
    assert!(type_verify(&UOp::sink(vec![invalid]), &spec_hcq()).is_err());
}

#[test]
fn spec_program_accepts_only_structured_local_and_reg_buffers() {
    for addrspace in [AddrSpace::Local, AddrSpace::Reg] {
        let buffer = UOp::new(
            Op::Buffer { shape: int_const(DType::Int32, 4), arg: ParamArg::buffer(3, DType::Float32, addrspace, None) },
            DType::Float32,
        );
        assert!(type_verify(&UOp::sink(vec![buffer]), &spec_program()).is_ok());
    }

    let global = UOp::new(
        Op::Buffer {
            shape: int_const(DType::Int32, 4),
            arg: ParamArg::buffer(3, DType::Float32, AddrSpace::Global, Some(svod_dtype::DeviceSpec::Cpu)),
        },
        DType::Float32,
    );
    assert!(type_verify(&UOp::sink(vec![global]), &spec_program()).is_err());
}

#[test]
fn spec_tensor_accepts_structured_global_buffer() {
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    assert!(type_verify(&UOp::sink(vec![buffer]), &spec_tensor()).is_ok());
}

#[test]
fn preoptimization_rejects_malformed_dtype_before_rewrites() {
    let malformed = UOp::new(
        Op::Binary(BinaryOp::Add, UOp::native_const(1i32), UOp::const_(DType::Float32, ConstValue::Float(1.0))),
        DType::Int32,
    );
    let err =
        apply_pre_optimization(UOp::sink(vec![malformed])).expect_err("mixed ALU dtype must fail at target boundary");
    assert!(err.to_string().contains("binary operand/result dtype mismatch"), "unexpected error: {err}");
}

#[test]
fn preoptimization_rejects_malformed_tensor_shape_and_source() {
    let bad_shape = UOp::new(
        Op::Buffer {
            shape: int_const(DType::Int32, 4),
            arg: ParamArg::buffer(0, DType::Float32, AddrSpace::Global, Some(svod_dtype::DeviceSpec::Cpu)),
        },
        DType::Float32,
    );
    assert!(apply_pre_optimization(UOp::sink(vec![bad_shape])).is_err(), "GLOBAL BUFFER shape must be weakint");

    let bad_source = UOp::native_const(1i32).mselect(0);
    let err = apply_pre_optimization(UOp::sink(vec![bad_source])).expect_err("MSELECT requires a target multi source");
    assert!(err.to_string().contains("MSELECT requires"), "unexpected error: {err}");
}

#[test]
fn preoptimization_rejects_unsupported_movement_and_bad_multi_axis() {
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 4, DType::Float32);
    let legacy_reduce =
        UOp::new(Op::ReduceAxis { src: buffer.clone(), reduce_op: ReduceOp::Add, axes: vec![0] }, DType::Float32);
    let err = apply_pre_optimization(UOp::sink(vec![legacy_reduce])).expect_err("REDUCE_AXIS is not target tensor IR");
    assert!(err.to_string().contains("no matching rule"), "unexpected error: {err}");

    let bad_axis = UOp::multi(buffer, 1);
    let err = apply_pre_optimization(UOp::sink(vec![bad_axis])).expect_err("MULTI axis must exist in its source shape");
    assert!(err.to_string().contains("MULTI must"), "unexpected error: {err}");
}

#[test]
fn preoptimization_accepts_valid_custom_kernel_forms() {
    let buffer = global_param(0);
    let index = UOp::index().buffer(buffer).indices(vec![int_const(DType::Int32, 0)]).call().unwrap();
    let loaded = UOp::load().index(index.clone()).call();
    let custom = UOp::custom(smallvec![loaded], "({0} + 1.0f)".to_string(), DType::Float32);
    let kernel = UOp::sink(vec![index.store(custom)]);

    assert!(apply_pre_optimization(kernel).is_ok(), "valid hand-authored custom kernel must cross the tensor boundary");
}

#[test]
fn spec_shared_accepts_weakint_index_but_not_weakfloat_index() {
    let buffer = global_param(0);
    let weakint_index = UOp::new(
        Op::Index { buffer: buffer.clone(), indices: smallvec![UOp::const_(DType::WeakInt, ConstValue::Int(0))] },
        DType::Float32,
    );
    let weakfloat_index = UOp::new(
        Op::Index { buffer, indices: smallvec![UOp::const_(DType::WeakFloat, ConstValue::Float(0.0))] },
        DType::Float32,
    );

    assert!(type_verify(&UOp::sink(vec![weakint_index]), &spec_tensor()).is_ok());
    assert!(type_verify(&UOp::sink(vec![weakfloat_index]), &spec_tensor()).is_err());
}

fn raw_index(index: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Index { buffer: global_param(0), indices: smallvec![index] }, DType::Float32)
}

#[test]
fn spec_shared_accepts_exact_invalid_index_values() {
    let scalar = UOp::invalid_marker();
    let vconst = UOp::vconst(vec![ConstValue::Invalid; 4], DType::Bool);
    let vectorize = UOp::stack(smallvec![
        UOp::invalid_marker(),
        UOp::invalid_marker(),
        UOp::invalid_marker(),
        UOp::invalid_marker(),
    ]);

    for index in [scalar, vconst, vectorize] {
        assert!(type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor()).is_ok());
    }
}

#[test]
fn spec_shared_rejects_bool_and_mixed_invalid_index_values() {
    let ordinary_bool = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let ordinary_bool_vector = UOp::vconst(vec![ConstValue::Bool(false), ConstValue::Bool(true)], DType::Bool);
    let mixed =
        UOp::new(Op::VConst { values: vec![ConstValue::Bool(false), ConstValue::Int(0)] }, DType::Bool.vec(2).unwrap());

    for index in [ordinary_bool, ordinary_bool_vector] {
        let err = type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor())
            .expect_err("ordinary Bool index must be rejected")
            .to_string();
        assert!(err.contains("non-integer value reached a memory INDEX operand"), "unexpected rejection: {err}");
    }
    assert!(
        type_verify(&UOp::sink(vec![raw_index(mixed)]), &spec_tensor()).is_err(),
        "mixed Bool/int vector must be rejected"
    );
}

#[test]
fn spec_shared_preserves_integer_index_rules() {
    let scalar = int_const(DType::Int32, 1);
    let vector = UOp::vconst(vec![ConstValue::Int(0), ConstValue::Int(1)], DType::Int32);

    for index in [scalar, vector] {
        assert!(type_verify(&UOp::sink(vec![raw_index(index)]), &spec_tensor()).is_ok());
    }
}

fn assert_weak_rejected_only_at_program_level(weak: Arc<UOp>, concrete: Arc<UOp>) {
    let weak_sink = UOp::sink(vec![weak]);
    assert!(type_verify(&weak_sink, &spec_tensor()).is_ok(), "weak dtype is legal in spec_shared/spec_tensor");
    assert!(type_verify(&weak_sink, &spec_program()).is_err(), "weak dtype must be rejected by spec_program");
    assert!(type_verify(&UOp::sink(vec![concrete]), &spec_program()).is_ok(), "concrete counterpart must pass");
}

#[test]
fn spec_program_rejects_weakint_only_at_program_level() {
    assert_weak_rejected_only_at_program_level(
        UOp::const_(DType::WeakInt, ConstValue::Int(1)),
        int_const(DType::Int32, 1),
    );
}

#[test]
fn spec_program_rejects_weakfloat_only_at_program_level() {
    assert_weak_rejected_only_at_program_level(
        UOp::const_(DType::WeakFloat, ConstValue::Float(1.0)),
        UOp::const_(DType::Float32, ConstValue::Float(1.0)),
    );
}

#[test]
fn spec_program_explicitly_rejects_legacy_index_dtype() {
    let sink = UOp::sink(vec![UOp::const_(DType::Index, ConstValue::Int(1))]);
    let err = verify_program_err(&sink);
    assert!(err.contains("legacy Index dtype must be lowered"), "unexpected error: {err}");
}

#[test]
fn post_index_lowering_invariant_reports_typed_context() {
    let stale = UOp::new(Op::Noop, DType::Index);
    let err = verify_no_legacy_index_dtype(&UOp::sink(vec![stale.clone()]))
        .expect_err("post-index-lowering boundary must reject Index");
    match err {
        SpecError::Verification { boundary, uop_id, source_path, reason, .. } => {
            assert_eq!(boundary, "post-index-lowering");
            assert_eq!(uop_id, stale.id);
            assert_eq!(source_path, vec![0]);
            assert_eq!(reason, "legacy Index dtype must be lowered before a program");
        }
    }
}

#[test]
fn spec_program_rejects_noncanonical_typed_constants() {
    let midpoint = 1.0 + 2f64.powi(-24);
    let scalar = UOp::new(Op::Const(ConstValueHash(ConstValue::Float(midpoint))), DType::Float32);
    let vector = UOp::new(
        Op::VConst { values: vec![ConstValue::Float(1.0), ConstValue::Float(midpoint)] },
        DType::Float32.vec(2).unwrap(),
    );

    for constant in [scalar, vector] {
        let err = verify_program_err(&UOp::sink(vec![constant]));
        assert!(err.contains("typed constant value is not canonical for its dtype"), "unexpected error: {err}");
    }
}

#[test]
fn spec_program_accepts_canonical_typed_constants_and_nans() {
    let scalar_nan = UOp::new(Op::Const(ConstValueHash(ConstValue::Float(f64::NAN))), DType::Float32);
    let vector_nan = UOp::new(
        Op::VConst { values: vec![ConstValue::Float(f64::NAN), ConstValue::Float(1.0)] },
        DType::Float32.vec(2).unwrap(),
    );

    for constant in [UOp::const_(DType::Float32, ConstValue::Float(1.0)), scalar_nan, vector_nan] {
        assert!(type_verify(&UOp::sink(vec![constant]), &spec_program()).is_ok());
    }

    let payload_nan =
        UOp::new(Op::Const(ConstValueHash(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0001)))), DType::Float32);
    assert!(verify_program_err(&UOp::sink(vec![payload_nan])).contains("typed constant value is not canonical"));
}

#[test]
fn spec_program_allows_special_shrink_before_movement_rejection() {
    // Pinned spec.py:207-208 deliberately places this rule before the general
    // movement-op rejection. The final source must be CONST.
    let shrink = UOp::new(
        Op::Shrink { src: global_param(0), offsets: int_const(DType::Int32, 0), sizes: int_const(DType::Int32, 1) },
        DType::Float32,
    );

    assert!(
        type_verify(&UOp::sink(vec![shrink]), &spec_program()).is_ok(),
        "special SHRINK must win before movement rejection"
    );
}

#[test]
fn spec_program_rejects_other_movement_ops() {
    let reshape =
        UOp::new(Op::Reshape { src: int_const(DType::Int32, 0), new_shape: int_const(DType::Int32, 1) }, DType::Int32);
    let sink = UOp::sink(vec![reshape]);

    assert!(type_verify(&sink, &spec_tensor()).is_ok(), "movement is tensor-graph legal");
    assert!(verify_program_err(&sink).contains("movement"), "movement must be rejected in a program");
}

#[test]
fn spec_program_rejects_invalid_that_shared_spec_accepts() {
    let sink = UOp::sink(vec![UOp::invalid_marker()]);

    assert!(type_verify(&sink, &spec_tensor()).is_ok(), "Invalid is legal in spec_shared/spec_tensor");
    assert!(verify_program_err(&sink).contains("Invalid"), "Invalid must be folded before a program");
}

#[test]
fn spec_program_accepts_devectorized_shaped_stack() {
    let lanes = (0..4).map(|value| UOp::const_(DType::BFloat16, ConstValue::Float(value as f64))).collect();
    let vector = UOp::stack(lanes);

    assert!(type_verify(&UOp::sink(vec![vector]), &spec_program()).is_ok());
}

#[test]
fn spec_program_accepts_shaped_index() {
    let lanes = (0..4).map(|value| UOp::const_(DType::Float32, ConstValue::Float(value as f64))).collect();
    let vector = UOp::stack(lanes);
    assert!(type_verify(&UOp::sink(vec![vector.index_axes(vec![2])]), &spec_program()).is_ok());
}

fn valid_if(condition: Arc<UOp>) -> Arc<UOp> {
    let index = UOp::index().buffer(global_param(0)).indices(vec![int_const(DType::Int32, 0)]).call().unwrap();
    UOp::new(Op::If { condition, body: smallvec![index] }, DType::Void)
}

#[test]
fn spec_program_accepts_well_formed_if_and_endif() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    assert!(type_verify(&UOp::sink(vec![UOp::endif(valid_if(condition))]), &spec_program()).is_ok());
}

#[test]
fn spec_program_rejects_if_with_non_index_dedup_source() {
    let condition = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let bad_if = UOp::new(Op::If { condition, body: smallvec![int_const(DType::Int32, 0)] }, DType::Void);
    assert!(
        type_verify(&UOp::sink(vec![bad_if]), &spec_program()).is_err(),
        "IF dedup source must be CAST/INDEX/SHRINK"
    );
}

#[test]
fn spec_program_rejects_endif_without_if() {
    let bad_endif = UOp::new(Op::EndIf { if_op: int_const(DType::Int32, 0) }, DType::Void);
    assert!(type_verify(&UOp::sink(vec![bad_endif]), &spec_program()).is_err(), "ENDIF must close an IF");
}

#[test]
fn spec_program_requires_special_to_be_lowered_int32() {
    let valid = UOp::new(Op::Special { end: int_const(DType::Int32, 8), name: "lidx0".to_string() }, DType::Int32);
    assert!(type_verify(&UOp::sink(vec![valid]), &spec_program()).is_ok());

    let wrong_width =
        UOp::new(Op::Special { end: int_const(DType::Int64, 8), name: "lidx0".to_string() }, DType::Int64);
    assert!(
        type_verify(&UOp::sink(vec![wrong_width]), &spec_program()).is_err(),
        "SPECIAL source and result must be int32 after index lowering"
    );
}

#[test]
fn spec_program_rejects_op_outside_whitelist() {
    let multi = UOp::new(Op::Multi { src: int_const(DType::Int32, 0), axis: 0 }, DType::Int32);
    let sink = UOp::sink(vec![multi]);
    assert!(verify_program_err(&sink).contains("no matching rule"));
}
