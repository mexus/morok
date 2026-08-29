use super::*;
use smallvec::smallvec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::types::{AxisId, AxisType, ConstValue, InsArg, RendererDevice, WmmaMetadata};

use crate::linearize::line_rewrite_cleanups;

#[test]
fn tinygrad_partial_range_comparison_rejects_path_prefixes() {
    let short = ArgKey::Range(vec![0], axis_type_value(AxisType::Weak));
    let long = ArgKey::Range(vec![0, 1], axis_type_value(AxisType::Weak));
    assert_eq!(partial_arg_cmp(&short, &long), None);
    assert_eq!(partial_arg_cmp(&long, &short), None);
}

#[test]
fn tinygrad_partial_param_comparison_stops_at_first_difference() {
    let projected = arg_key(&Op::Param {
        shape: UOp::index_const(1),
        arg: ParamArg::variable("projected".to_string(), DType::WeakInt, 0, 8),
    });
    assert!(matches!(projected, ArgKey::Param(ParamKey { addrspace: Some(4), .. })));

    let mut left = param_key(&ParamArg::variable("a".to_string(), DType::WeakInt, 0, 8));
    let mut right = param_key(&ParamArg::variable("b".to_string(), DType::WeakInt, 0, 8));
    left.addrspace = None;
    right.addrspace = Some(1);
    assert_eq!(partial_param_cmp(&left, &right), Some(Ordering::Less));
    assert_eq!(partial_param_cmp(&right, &left), Some(Ordering::Greater));
}

#[test]
fn tinygrad_float_keys_coalesce_signed_zero_and_nan_payloads() {
    assert_eq!(const_key(ConstValue::Float(-0.0)), const_key(ConstValue::Float(0.0)));
    let left = const_key(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0001)));
    let right = const_key(ConstValue::Float(f64::from_bits(0x7ff8_0000_0000_0002)));
    assert_eq!(left, right);
    assert_eq!(partial_const_cmp(&left, &right), None);
}

#[test]
fn tinygrad_vconst_linearizer_key_is_stack_of_constants() {
    let vconst = UOp::vconst(vec![ConstValue::Int(1), ConstValue::Int(2)], DType::WeakInt);
    let keys = compute_tuplize(&vconst.toposort());
    let key = &keys[&vconst.id];
    assert_eq!(key.op, 16);
    assert_eq!(key.arg, ArgKey::None);
    assert_eq!(key.dtype, dtype_key(&DType::WeakInt));
    assert_eq!(key.src.len(), 2);
    assert!(key.src.iter().all(|source| source.op == 61 && source.dtype == dtype_key(&DType::WeakInt)));

    let stack = UOp::stack(smallvec![UOp::range_const(8, 0), UOp::special(UOp::index_const(8), "gidx0".to_string())]);
    assert_eq!(tinygrad_tuplize_cmp(&stack, &vconst), Some(Ordering::Less));
}

#[test]
fn tinygrad_wmma_key_omits_svod_only_metadata() {
    let value = UOp::native_const(0.0f32);
    let mut left = WmmaMetadata {
        name: "z_svod_name".to_string(),
        dims: (16, 16, 16),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: RendererDevice::CudaSm80,
        threads: 32,
        upcast_axes: None,
        reduce_axes: vec![AxisId::Renumbered(3)],
        tile_grid: (2, 2),
    };
    let mut right = left.clone();
    right.name = "a_svod_name".to_string();
    right.dtype_out = DType::Int32;
    right.reduce_axes.clear();
    right.tile_grid = (1, 1);

    let make_op = |metadata| Op::Wmma { a: value.clone(), b: value.clone(), c: value.clone(), metadata };
    assert_eq!(arg_key(&make_op(left.clone())), arg_key(&make_op(right.clone())));

    right.device = RendererDevice::CudaSm89;
    right.upcast_axes = Some(svod_ir::WmmaUpcastAxes { a: vec![(AxisId::Unrenumbered(3), 2)], b: vec![], c: vec![] });
    left.upcast_axes =
        Some(svod_ir::WmmaUpcastAxes { a: vec![(AxisId::RenumberedPath(smallvec![3, 1]), 2)], b: vec![], c: vec![] });
    assert_eq!(arg_key(&make_op(left.clone())), arg_key(&make_op(right.clone())));

    let left_uop = UOp::new(make_op(left.clone()), DType::Float32);
    let right_uop = UOp::new(make_op(right.clone()), DType::Float32);
    assert_eq!(tinygrad_tuplize_cmp(&left_uop, &right_uop), Some(Ordering::Equal));

    left.dims = (8, 16, 16);
    assert_ne!(
        arg_key(&make_op(left)),
        arg_key(&make_op(WmmaMetadata {
            name: "ignored".to_string(),
            dims: (16, 16, 16),
            dtype_in: DType::Float16,
            dtype_out: DType::Float32,
            device: RendererDevice::CudaSm80,
            threads: 32,
            upcast_axes: None,
            reduce_axes: vec![],
            tile_grid: (1, 1),
        }))
    );
}

#[test]
fn test_linearize_single_const() {
    let c = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let sink = UOp::sink(vec![c.clone()]);

    let result = linearize(sink.clone());

    assert_eq!(result.len(), 2); // const + sink
    // Const should come before sink
    assert!(matches!(result[0].op(), Op::Const(_)));
    assert!(matches!(result[1].op(), Op::Sink { .. }));
}

#[test]
fn test_linearize_simple_computation() {
    let a = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let b = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let sum = a.try_add(&b).unwrap();
    let sink = UOp::sink(vec![sum]);

    let result = linearize(sink);

    // Should have: const, const, add, sink
    assert_eq!(result.len(), 4);
    // Constants should come first (priority -10)
    assert!(matches!(result[0].op(), Op::Const(_)));
    assert!(matches!(result[1].op(), Op::Const(_)));
    // Then binary op
    assert!(matches!(result[2].op(), Op::Binary(_, _, _)));
    // Then sink
    assert!(matches!(result[3].op(), Op::Sink { .. }));
}

#[test]
fn test_linearize_with_range() {
    // Create: for i in range(10): end(value)
    let end_val = UOp::index_const(10);
    let range = UOp::range(end_val, 0);
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let end = value.end(smallvec![range.clone()]);
    let sink = UOp::sink(vec![end]);

    let result = linearize(sink);

    // Verify RANGE comes before END (RANGE priority 5, END priority -5)
    // But RANGE should come after its sources
    let range_pos = result.iter().position(|u| matches!(u.op(), Op::Range { .. }));
    let end_pos = result.iter().position(|u| matches!(u.op(), Op::End { .. }));

    assert!(range_pos.is_some());
    assert!(end_pos.is_some());
    // END depends on RANGE, so RANGE must come before END
    assert!(range_pos.unwrap() < end_pos.unwrap());
}

#[test]
fn test_linearize_preserves_dependencies() {
    // Create a diamond dependency: a + b, where both depend on c
    let c = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let c2 = UOp::const_(DType::Float32, ConstValue::Float(2.0));
    let c3 = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let a = c.try_add(&c2).unwrap();
    let b = c.try_add(&c3).unwrap();
    let sum = a.try_add(&b).unwrap();
    let sink = UOp::sink(vec![sum.clone()]);

    let result = linearize(sink);

    // c should appear before both a and b
    let c_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &c));
    let a_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &a));
    let b_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &b));
    let sum_pos = result.iter().position(|u| std::sync::Arc::ptr_eq(u, &sum));

    assert!(c_pos.is_some());
    assert!(a_pos.is_some());
    assert!(b_pos.is_some());
    assert!(sum_pos.is_some());

    // Dependencies: c < a, c < b, a < sum, b < sum
    assert!(c_pos.unwrap() < a_pos.unwrap());
    assert!(c_pos.unwrap() < b_pos.unwrap());
    assert!(a_pos.unwrap() < sum_pos.unwrap());
    assert!(b_pos.unwrap() < sum_pos.unwrap());
}

#[test]
fn test_priority_ordering() {
    let param = UOp::param(0, 1, DType::Float32, None);
    let range = UOp::range_const(10, 0);

    assert!(priority(&param).0 < 0); // PARAM = -20
    assert!(priority(&range).0 == 5); // RANGE = 5
    assert_eq!(priority(&param).1, Some(0)); // PARAM extra = slot
}

#[test]
fn test_pinned_param_and_buffer_priorities() {
    let device_param = UOp::param(3, 1, DType::Float32, Some(DeviceSpec::Cpu));
    let local = UOp::buffer(0, 1, DType::Float32, AddrSpace::Local, None);
    let global = UOp::buffer(1, 1, DType::Float32, AddrSpace::Global, Some(DeviceSpec::Cpu));
    let reg = UOp::buffer(2, 1, DType::Float32, AddrSpace::Reg, None);

    assert_eq!(priority(&device_param), (-20, Some(3)));
    assert_eq!(priority(&local), (-17, None));
    assert_eq!(priority(&global), (-18, None));
    assert_eq!(priority(&reg), (-18, None));

    // The pin has no CONST arm (upstream 52b989c6c "don't place consts early")
    // and no DEFINE_VAR arm (4a4b6956d): a symbolic variable is a PARAM.
    assert_eq!(priority(&UOp::const_(DType::Int32, ConstValue::Int(7))), (0, None));
    assert_eq!(priority(&UOp::variable("n".to_string(), 0, 8, DType::Int32)), (-20, Some(-1)));
}

#[test]
fn test_tuplize_is_recursive_past_128_elements() {
    let mut low = UOp::const_(DType::Int32, ConstValue::Int(1));
    let mut high = UOp::const_(DType::Int32, ConstValue::Int(2));
    for _ in 0..140 {
        low = UOp::new(Op::Precast { src: low }, DType::Int32);
        high = UOp::new(Op::Precast { src: high }, DType::Int32);
    }
    let sink = UOp::sink(vec![high.clone(), low.clone()]);
    let keys = compute_tuplize(&sink.toposort());
    assert!(keys[&low.id] < keys[&high.id]);

    let order = linearize(sink);
    assert!(order.iter().position(|u| Arc::ptr_eq(u, &low)) < order.iter().position(|u| Arc::ptr_eq(u, &high)));
}

#[test]
fn test_equal_dependency_side_effects_use_full_arg_order() {
    let dependency = UOp::native_const(0i32);
    let later = UOp::new(Op::CustomI { deps: smallvec![dependency.clone()], code: "z".into() }, DType::Void);
    let earlier = UOp::new(Op::CustomI { deps: smallvec![dependency], code: "a".into() }, DType::Void);
    let order = linearize(UOp::sink(vec![later.clone(), earlier.clone()]));

    assert!(order.iter().position(|u| Arc::ptr_eq(u, &earlier)) < order.iter().position(|u| Arc::ptr_eq(u, &later)));
}

#[test]
fn test_nested_axis_and_ins_arguments_participate_in_tuplize() {
    let end = UOp::index_const(4);
    let outer = UOp::range_axis(end.clone(), AxisId::RenumberedPath(smallvec![0, 1]), AxisType::Loop);
    let inner = UOp::range_axis(end, AxisId::RenumberedPath(smallvec![0, 2]), AxisType::Loop);
    assert!(arg_key(outer.op()) < arg_key(inner.op()));

    let source = UOp::native_const(1i32);
    let ins_a = UOp::new(
        Op::Ins {
            sources: smallvec![source.clone()],
            arg: InsArg::with_attributes("v_add", vec![("axis".into(), "1".into())]),
        },
        DType::Int32,
    );
    let ins_b = UOp::new(
        Op::Ins {
            sources: smallvec![source],
            arg: InsArg::with_attributes("v_add", vec![("axis".into(), "2".into())]),
        },
        DType::Int32,
    );
    assert!(arg_key(ins_a.op()) < arg_key(ins_b.op()));
    assert_eq!(op_value(ins_a.op()), 64);
}

#[test]
fn test_linearize_cleanup_expands_gated_store() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let gate = UOp::native_const(true);
    let store = index.store_gated(UOp::native_const(1.0f32), gate.clone());

    let result = line_rewrite_cleanups(vec![store]);
    assert_eq!(result.len(), 3);
    let Op::If { condition, body } = result[0].op() else { panic!("expected IF") };
    assert!(Arc::ptr_eq(&condition, &gate));
    assert_eq!(body.len(), 1);
    assert!(matches!(result[1].op(), Op::Store { gate: None, .. }));
    let Op::EndIf { if_op } = result[2].op() else { panic!("expected ENDIF") };
    assert!(Arc::ptr_eq(&if_op, &result[0]));
}
