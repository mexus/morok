use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{BinaryOp, ConstValue, ConstValueHash, Op, ParamArg, TernaryOp, UOp};

use crate::rewrite::graph_rewrite;
use crate::symbolic::index_lowering::{WeakMemo, pm_commit_weak, pm_lower_index_dtype, pm_lower_weak};

#[test]
fn weak_const_selects_i32_or_i64() {
    for (value, expected) in [(42, DType::Int32), (i64::MAX / 2, DType::Int64)] {
        let weak = UOp::const_(DType::WeakInt, ConstValue::Int(value));
        let lowered = graph_rewrite(&pm_lower_weak(), weak, &mut ());
        let Op::Cast { src, dtype } = lowered.op() else { panic!("expected weak cast") };
        assert_eq!(*dtype, DType::WeakInt);
        assert_eq!(src.dtype(), expected);
    }
}

#[test]
fn weak_float_const_selects_default_float() {
    let weak = UOp::const_(DType::WeakFloat, ConstValue::Float(1.5));
    let lowered = graph_rewrite(&pm_lower_weak(), weak, &mut ());
    let Op::Cast { src, dtype } = lowered.op() else { panic!("expected weak cast") };
    assert_eq!(*dtype, DType::WeakFloat);
    assert_eq!(src.dtype(), DType::Float32);
}

#[test]
fn weak_vconst_selects_mechanically_shaped_default() {
    for (dtype, values, expected) in [
        (DType::WeakInt, vec![ConstValue::Int(1); 4], DType::Int32),
        (DType::WeakFloat, vec![ConstValue::Float(1.5); 8], DType::Float32),
    ] {
        let weak = UOp::vconst(values, dtype.clone());
        let lowered = graph_rewrite(&pm_lower_weak(), weak, &mut ());
        let Op::Cast { src, dtype: cast_dtype } = lowered.op() else { panic!("expected weak vector cast") };
        assert_eq!(*cast_dtype, dtype.vec(cast_dtype.vcount()).unwrap());
        assert_eq!(src.dtype(), expected.vec(cast_dtype.vcount()).unwrap());
    }
}

#[test]
fn weak_vconst_f32_midpoint_commits_before_comparison() {
    let midpoint = 1.0 + 2f64.powi(-24);
    let weak = UOp::vconst(vec![ConstValue::Float(midpoint); 2], DType::WeakFloat);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), UOp::sink(vec![weak]), &mut WeakMemo::default());
    let Op::Sink { sources, .. } = lowered.op() else { panic!("expected sink") };
    let committed = &sources[0];
    assert_eq!(committed.dtype(), DType::Float32.vec(2).unwrap());
    let Op::VConst { values } = committed.op() else { panic!("expected committed VCONST") };
    assert!(
        values.iter().all(|value| { matches!(value, ConstValue::Float(value) if value.to_bits() == 1.0f64.to_bits()) })
    );

    let ones = UOp::vconst(vec![ConstValue::Float(1.0); 2], DType::Float32);
    let comparison = committed.try_cmpeq(&ones).unwrap();
    let folded = graph_rewrite(crate::symbolic::patterns::symbolic(), comparison, &mut ());
    assert!(matches!(folded.op(), Op::VConst { values } if values == &vec![ConstValue::Bool(true); 2]));
}

#[test]
fn weak_vconst_commit_preserves_invalid_lane_and_width() {
    let midpoint = 1.0 + 2f64.powi(-24);
    let weak =
        UOp::vconst(vec![ConstValue::Float(midpoint), ConstValue::Invalid, ConstValue::Float(2.0)], DType::WeakFloat);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), UOp::sink(vec![weak]), &mut WeakMemo::default());
    let Op::Sink { sources, .. } = lowered.op() else { panic!("expected sink") };
    assert_eq!(sources[0].dtype(), DType::Float32.vec(3).unwrap());
    assert!(matches!(sources[0].op(), Op::VConst { values }
        if values == &vec![ConstValue::Float(1.0), ConstValue::Invalid, ConstValue::Float(2.0)]));
}

#[test]
fn stacked_weak_cast_resolves_at_outer_default() {
    let value = UOp::native_const(7i32);
    let stacked = value.cast(DType::WeakInt).cast(DType::WeakFloat);
    let lowered = graph_rewrite(&pm_lower_weak(), stacked, &mut ());
    let Op::Cast { src, dtype } = lowered.op() else { panic!("expected outer weak cast") };
    assert_eq!(*dtype, DType::WeakFloat);
    assert_eq!(src.dtype(), DType::Float32);
}

#[test]
fn concrete_int_plus_weak_int_resolves_as_target() {
    let index = UOp::variable("idx".into(), 0, 31, DType::Int64);
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let add = UOp::new(Op::Binary(BinaryOp::Add, index, weak), DType::Int64);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), add, &mut WeakMemo::default());
    let Op::Binary(BinaryOp::Add, _, rhs) = lowered.op() else { panic!("expected add") };
    assert_eq!(lowered.dtype(), DType::Int64);
    assert_eq!(rhs.dtype(), DType::Int64);
    assert!(matches!(rhs.op(), Op::Const(_)), "bare weak constants commit directly");
}

#[test]
fn comparison_lowers_whole_node_to_unify_operand_width() {
    let lhs = UOp::const_(DType::WeakInt, ConstValue::Int(i32::MAX as i64 + 1));
    let rhs = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let comparison = UOp::new(Op::Binary(BinaryOp::Lt, lhs, rhs), DType::Bool);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), comparison, &mut WeakMemo::default());
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = lowered.op() else { panic!("expected comparison") };
    assert_eq!(lowered.dtype(), DType::Bool);
    assert_eq!(lhs.dtype(), DType::Int64);
    assert_eq!(rhs.dtype(), DType::Int64);
}

#[test]
fn committing_shift_lhs_rederives_result_dtype() {
    let lhs = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let rhs = UOp::native_const(2i64);
    let shift = UOp::new(Op::Binary(BinaryOp::Shl, lhs, rhs), DType::WeakInt);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), shift, &mut WeakMemo::default());
    let Op::Binary(BinaryOp::Shl, lhs, _) = lowered.op() else { panic!("expected shift") };
    assert_eq!(lhs.dtype(), DType::Int64);
    assert_eq!(lowered.dtype(), DType::Int64);
}

#[test]
fn weak_shift_counts_commit_to_each_integer_lhs_width() {
    let integer_dtypes = [
        DType::Int8,
        DType::UInt8,
        DType::Int16,
        DType::UInt16,
        DType::Int32,
        DType::UInt32,
        DType::Int64,
        DType::UInt64,
    ];

    for dtype in integer_dtypes {
        let value = if dtype.is_unsigned() { ConstValue::UInt(8) } else { ConstValue::Int(8) };
        let lhs = UOp::const_(dtype.clone(), value);
        for op in [BinaryOp::Shl, BinaryOp::Shr] {
            let shift = UOp::new(Op::Binary(op, lhs.clone(), UOp::index_const(1)), dtype.clone());
            let lowered = graph_rewrite(&pm_commit_weak(), shift, &mut ());
            let Op::Binary(actual, actual_lhs, actual_rhs) = lowered.op() else { panic!("expected shift") };
            assert_eq!(*actual, op);
            assert_eq!(lowered.dtype(), dtype);
            assert_eq!(actual_lhs.dtype(), dtype);
            assert_eq!(actual_rhs.dtype(), dtype);
            assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
        }
    }
}

#[test]
fn weak_bitwise_and_shift_indices_commit_before_program() {
    let ptr = DType::Float32.ptr(Some(16), AddrSpace::Global).unwrap();
    let buffer = UOp::param(0, 16, DType::Scalar(ptr.base()), None);
    let value = UOp::index_const(12);
    let operand = UOp::index_const(3);
    let expressions =
        [value.try_shr_op(&operand).unwrap(), value.try_and_op(&operand).unwrap(), value.try_xor_op(&operand).unwrap()];

    for expression in expressions {
        assert_eq!(expression.dtype(), DType::WeakInt, "graph construction must preserve mathematical integers");
        let index = UOp::index().buffer(buffer.clone()).indices(vec![expression]).call().unwrap();
        let lowered = graph_rewrite(&pm_lower_index_dtype(), index, &mut WeakMemo::default());
        assert!(
            lowered.toposort().iter().all(|u| !u.dtype().is_weak()),
            "weak dtype reached program boundary:\n{}",
            lowered.tree()
        );
    }
}

#[test]
fn only_alu_weak_param_is_lowered() {
    let shape = svod_ir::shape::shape_to_uop(&[1usize.into()].into_iter().collect());
    let make_param = |addrspace| {
        UOp::new(
            Op::Param {
                shape: shape.clone(),
                arg: ParamArg {
                    slot: 0,
                    dtype: DType::WeakInt,
                    vmin_vmax: Some((ConstValueHash(ConstValue::Int(0)), ConstValueHash(ConstValue::Int(7)))),
                    multiple_of: None,
                    name: None,
                    addrspace,
                    axis: None,
                    device: None,
                    volatile: false,
                },
            },
            DType::WeakInt,
        )
    };
    let alu = make_param(None);
    let lowered = graph_rewrite(&pm_lower_weak(), alu.clone(), &mut ());
    assert!(!Arc::ptr_eq(&lowered, &alu));
    assert!(
        matches!(lowered.op(), Op::Cast { src, dtype } if *dtype == DType::WeakInt && !src.dtype().is_weak()),
        "{}",
        lowered.tree()
    );

    let buffer = make_param(Some(AddrSpace::Global));
    let lowered_buffer = graph_rewrite(&pm_lower_weak(), buffer, &mut ());
    assert!(matches!(lowered_buffer.op(), Op::Param { arg, .. } if arg.dtype == DType::WeakInt));
}

#[test]
fn shape_weak_constants_commit_at_concrete_consumer() {
    let shape: svod_ir::shape::Shape = [2usize.into(), 3usize.into()].into_iter().collect();
    let weak_shape = svod_ir::shape::shape_to_uop(&shape);
    assert_eq!(weak_shape.dtype(), DType::WeakInt);
    let concrete = UOp::variable("n".into(), 0, 10, DType::Int32);
    let add = UOp::new(Op::Binary(BinaryOp::Add, concrete, weak_shape), DType::Int32);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), add, &mut WeakMemo::default());
    assert!(!lowered.op().sources().iter().any(|s| s.dtype().is_weak()));
}

#[test]
fn lowering_weak_index_preserves_scalar_load_shape_for_index_extraction() {
    let buffer = UOp::param(0, 64, DType::BFloat16, None);
    let offsets = UOp::stack((0..8).map(|offset| UOp::const_(DType::WeakInt, ConstValue::Int(offset))).collect());
    let index = UOp::index().buffer(buffer).indices(vec![offsets]).call().unwrap();
    let load = UOp::load().index(index).call();
    let lane = UOp::index().buffer(load).indices(vec![UOp::index_const(3)]).call().unwrap();

    let matcher = crate::symbolic::patterns::symbolic_simple().with_context::<WeakMemo>() + pm_lower_index_dtype();
    let lowered = graph_rewrite(&matcher, lane, &mut WeakMemo::default());

    let shaped_load = lowered
        .toposort()
        .into_iter()
        .find(|node| matches!(node.op(), Op::Load { .. }))
        .expect("shaped LOAD must remain under extraction");
    assert_eq!(shaped_load.dtype(), DType::BFloat16);
    assert_eq!(shaped_load.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(8)]);
    assert!(matches!(lowered.op(), Op::Index { buffer, .. } if std::sync::Arc::ptr_eq(buffer, &shaped_load)));
    assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()));
}

#[test]
fn where_invalid_is_preserved() {
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let weak = UOp::const_(DType::WeakInt, ConstValue::Int(7));
    let valid = weak.valid(gate.clone());
    let lowered = graph_rewrite(&pm_lower_weak(), valid, &mut ());
    let Op::Cast { src, dtype } = lowered.op() else { panic!("expected weak cast") };
    assert_eq!(*dtype, DType::WeakInt);
    let Op::Ternary(TernaryOp::Where, condition, value, invalid) = src.op() else { panic!("expected WHERE") };
    assert!(Arc::ptr_eq(condition, &gate));
    assert_eq!(value.dtype(), DType::Int32);
    assert!(UOp::is_invalid_marker(invalid));
}

#[test]
fn final_invalid_removal_leaves_addresses_gated() {
    let gate = UOp::var("gate", DType::Bool, 0, 1);
    let address = UOp::var("i", DType::Index, 0, 16).valid(gate);
    let lanes = [address.clone(), UOp::invalid_marker()].into_iter().collect();
    let stacked = UOp::new(Op::Stack { sources: lanes }, DType::Index);

    // Rewriting an address Invalid to 0 would turn a skipped access into an
    // unconditional read of element 0; only pm_lower_index_dtype may lower these.
    for gated in [address, stacked] {
        let result = graph_rewrite(crate::symbolic::patterns::pm_remove_invalid(), gated.clone(), &mut ());
        assert!(Arc::ptr_eq(&result, &gated));
    }
}

#[test]
fn stack_invalid_lowers_weak_lane_without_replacing_marker() {
    let invalid = UOp::invalid_marker();
    let vector = UOp::stack([UOp::const_(DType::WeakInt, ConstValue::Int(7)), invalid.clone()].into_iter().collect());
    let lowered = graph_rewrite(&pm_lower_weak(), vector, &mut ());

    assert_eq!(lowered.dtype(), DType::Int32);
    assert_eq!(lowered.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(2)]);
    let Op::Stack { sources: elements } = lowered.op() else { panic!("expected STACK") };
    assert_eq!(elements[0].dtype(), DType::Int32);
    assert!(Arc::ptr_eq(&elements[1], &invalid), "weak lowering must preserve Invalid for the gater");
}

#[test]
fn weak_lane_index_commits_hardware_vector_source() {
    let lanes = UOp::vconst((0..4).map(ConstValue::Int).collect(), DType::WeakInt);
    let lane = UOp::index().buffer(lanes).indices(vec![UOp::index_const(2)]).call().unwrap();
    let lowered = graph_rewrite(&pm_lower_index_dtype(), UOp::sink(vec![lane]), &mut WeakMemo::default());
    assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
}

#[test]
fn mixed_shaped_and_hardware_vector_weak_binary_commits() {
    let shaped = UOp::stack((0..8).map(|value| UOp::native_const(value as i64)).collect());
    let hardware = UOp::vconst(vec![ConstValue::Int(1); 8], DType::WeakInt);
    let add = UOp::new(Op::Binary(BinaryOp::Add, shaped, hardware), DType::WeakInt.vec(8).unwrap());
    let lowered = graph_rewrite(&pm_lower_index_dtype(), UOp::sink(vec![add]), &mut WeakMemo::default());
    assert!(lowered.toposort().iter().all(|node| !node.dtype().is_weak()), "{}", lowered.tree());
}

#[test]
fn concrete_cast_is_a_width_floor() {
    let lhs = UOp::const_(DType::WeakInt, ConstValue::Int(i32::MAX as i64 + 1));
    let rhs = UOp::const_(DType::WeakInt, ConstValue::Int(1));
    let weak_add = UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), DType::WeakInt);
    let lowered = graph_rewrite(&pm_lower_index_dtype(), weak_add.cast(DType::Int32), &mut WeakMemo::default());
    let Op::Cast { src, dtype } = lowered.op() else { panic!("expected concrete cast") };
    assert_eq!(*dtype, DType::Int32);
    assert!(src.op().sources().iter().all(|s| s.dtype() == DType::Int64));
}

#[test]
fn store_commits_weak_value_to_destination() {
    let ptr = DType::Float32.ptr(Some(16), AddrSpace::Global).unwrap();
    let buffer = UOp::param(0, 16, DType::Scalar(ptr.base()), None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::native_const(0i32)]).call().unwrap();
    let store = index.store(UOp::const_(DType::WeakFloat, ConstValue::Float(1.0)));
    let lowered = graph_rewrite(&pm_commit_weak(), store, &mut ());
    let Op::Store { index: lowered_index, value, .. } = lowered.op() else { panic!("expected store") };
    assert_eq!(value.dtype(), DType::Float32);
    assert_eq!(index.dtype(), DType::Float32, "INDEX exposes the adopted buffer dtype");
    assert!(Arc::ptr_eq(lowered_index, &index), "STORE lowering must not adapt or replace its destination address");
}

#[test]
fn gated_long_index_narrows_only_for_small_buffers() {
    for (size, narrowed) in [(16usize, true), (i32::MAX as usize + 2, false)] {
        let ptr = DType::Float32.ptr(Some(size), AddrSpace::Global).unwrap();
        let buffer = UOp::param(0, size, DType::Scalar(ptr.base()), None);
        let idx = UOp::variable("idx".into(), 0, i64::MAX / 2, DType::Int64);
        let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
        let index = UOp::index().buffer(buffer).indices(vec![idx.valid(gate)]).call().unwrap();
        let lowered = graph_rewrite(&pm_lower_index_dtype(), index, &mut WeakMemo::default());
        let Op::Index { indices, .. } = lowered.op() else { panic!("expected index") };
        let Op::Ternary(TernaryOp::Where, _, idx, invalid) = indices[0].op() else { panic!("expected gated index") };
        assert_eq!(idx.dtype() == DType::Int32, narrowed);
        assert!(UOp::is_invalid_marker(invalid));
    }
}

/// `lower_weak_srcs` (tinygrad/uop/weak.py:29-40) keeps a `ctx` dict keyed by source:
/// `if (r:=ctx.get(s)) is None: r = graph_rewrite(s, pm_lower_weak)`. The memo lives for
/// one `to_program` (`ctx={}`, codegen/__init__.py:349), so a source shared by several
/// consumers is rewritten once, not once per edge.
#[test]
fn shared_weak_sources_are_lowered_once_per_pass() {
    let weak_index = |offset: i64| {
        UOp::new(
            Op::Binary(BinaryOp::Add, UOp::range_const(64, 0), UOp::const_(DType::WeakInt, ConstValue::Int(offset))),
            DType::WeakInt,
        )
    };
    let read = |slot: usize, index: Arc<UOp>| {
        let buffer = UOp::param(slot, 64, DType::Float32, None);
        UOp::index().buffer(buffer).indices(vec![index]).call().unwrap()
    };

    // Three consumers over two distinct weak indices.
    let shared = weak_index(3);
    let sink = UOp::sink(vec![read(0, shared.clone()), read(1, shared), read(2, weak_index(5))]);

    let mut memo = WeakMemo::default();
    graph_rewrite(&pm_lower_index_dtype(), sink, &mut memo);

    // Six weak edges reach a non-weak consumer here: three INDEX indices and the shared
    // WeakInt extent of the three PARAM shapes. They collapse to three rewrites.
    assert_eq!(memo.len(), 3, "one entry per distinct weak source, not per consumer edge");
}
