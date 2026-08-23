use super::*;

/// Build a Vec<Arc<UOp>> of concrete `index_const` dims for tests that only
/// exercise the numeric grouping/splitting logic.
fn d(vals: &[usize]) -> Vec<Arc<UOp>> {
    vals.iter().map(|&v| UOp::index_const(v as i64)).collect()
}

/// Extract dim_max from a slice — round-trips numeric-only test inputs back
/// through the sint abstraction.
fn dmax(vs: &[Arc<UOp>]) -> Vec<usize> {
    vs.iter().map(dim_max).collect()
}

#[test]
fn test_thread_extent_maps_to_exact_core_id_cardinality() {
    let thread = UOp::range_axis(UOp::index_const(2), svod_ir::AxisId::Renumbered(0), AxisType::Thread);
    let sink = UOp::sink(vec![thread]);

    let lowered = add_gpudims(&Renderer::cpu(), &sink).expect("thread range should lower to core_id");
    let core_id = lowered
        .toposort()
        .into_iter()
        .find(|u| matches!(u.op(), Op::Param { arg, .. } if arg.name.as_deref() == Some("core_id")))
        .expect("lowered graph should contain core_id");

    assert_eq!(core_id.vmin(), &ConstValue::Int(0));
    assert_eq!(core_id.vmax(), &ConstValue::Int(1));

    let info = svod_ir::ProgramInfo::from_sink(&lowered, svod_dtype::DeviceSpec::Cpu);
    assert_eq!(info.global_size[0].vmin(), &ConstValue::Int(2));
    assert_eq!(info.global_size[0].vmax(), &ConstValue::Int(2));
}

#[test]
fn test_existing_special_skips_all_gpudims_lowering() {
    let global = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::Global);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let sink = UOp::sink(vec![global, special]);

    assert!(add_gpudims(&Renderer::amd_cdna3(), &sink).is_none());
}

#[test]
fn test_global_product_cap_accounts_for_local_extent() {
    let global = UOp::range_axis(UOp::index_const(1 << 30), svod_ir::AxisId::Renumbered(0), AxisType::Global);
    let local = UOp::range_axis(UOp::index_const(8), svod_ir::AxisId::Renumbered(1), AxisType::Local);
    let sink = UOp::sink(vec![global, local]);

    let lowered = add_gpudims(&Renderer::amd_cdna3(), &sink).expect("GPU ranges should lower");
    let global_special_ends: Vec<usize> = lowered
        .toposort()
        .into_iter()
        .filter_map(|uop| match uop.op() {
            Op::Special { end, name } if name.starts_with("gidx") => Some(dim_max(end)),
            _ => None,
        })
        .collect();

    assert_eq!(global_special_ends, vec![4, 1 << 28]);
    assert!(global_special_ends.iter().all(|&end| end <= u32::MAX as usize / 8));
}

#[test]
fn test_device_range_lowers_and_end_drops_all_params() {
    let device =
        UOp::range_axis_dtype(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::Device, DType::Index);
    let other = UOp::variable("other".to_string(), 0, 7, DType::Index);
    let computation = device.add(&UOp::const_(DType::Index, ConstValue::Int(1)));
    let ended = computation.end(smallvec::smallvec![device, other]);
    let lowered = crate::rewrite::graph_rewrite(&pm_add_gpudims(), ended, &mut Renderer::amd_cdna3());

    let Op::End { ranges, .. } = lowered.op() else { panic!("target keeps an empty END") };
    assert!(ranges.is_empty(), "Tinygrad removes every PARAM when _device_num is present");
    let device_num =
        lowered.toposort().into_iter().find(|uop| is_device_num(uop)).expect("DEVICE range should become _device_num");
    assert_eq!(device_num.dtype(), DType::Index);
    assert_eq!(device_num.vmax(), &ConstValue::Int(3));
}

#[test]
fn test_non_device_end_keeps_param_counterexample() {
    let other = UOp::variable("other".to_string(), 0, 7, DType::Index);
    let ended = UOp::index_const(1).end(smallvec::smallvec![other.clone()]);
    let lowered = crate::rewrite::graph_rewrite(&pm_add_gpudims(), ended.clone(), &mut Renderer::amd_cdna3());

    assert!(Arc::ptr_eq(&lowered, &ended));
}

#[test]
fn test_missing_group_reduce_masks_structured_global_param_store() {
    let group = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::GroupReduce);
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().expect("index");
    let store = index.store(group.cast(DType::Float32));
    let sink = UOp::sink(vec![store]);

    let lowered = add_gpudims(&Renderer::amd_cdna3(), &sink).expect("group range should lower");
    let masked_index = lowered
        .toposort()
        .into_iter()
        .find(|u| matches!(u.op(), Op::Index { indices, .. } if matches!(indices[0].op(), Op::Ternary(..))))
        .expect("global store index should carry missing GroupReduce validity");
    let Op::Index { indices, .. } = masked_index.op() else { unreachable!() };
    assert!(indices[0].toposort().iter().any(|u| matches!(u.op(), Op::Special { name, .. } if name == "lidx0")));
}

#[test]
fn test_group_dims_already_fits() {
    // Dims already fit, no grouping needed.
    let result = group_dims(&d(&[4, 4]), &[16, 16, 16]);
    assert_eq!(dmax(&result.unwrap()), vec![4, 4]);
}

#[test]
fn test_group_dims_needs_grouping() {
    // 4 dims need to be grouped to fit into 3 max_sizes:
    // [4, 4, 4, 4] → [16, 4, 4].
    let result = group_dims(&d(&[4, 4, 4, 4]), &[256, 256, 256]);
    let result = result.unwrap();
    assert!(result.len() <= 3);
    assert_eq!(dmax(&result), vec![16, 4, 4]);
}

#[test]
fn test_group_dims_no_change() {
    // Dims already fit.
    let result = group_dims(&d(&[8, 8, 8]), &[256, 256, 256]);
    assert_eq!(dmax(&result.unwrap()), vec![8, 8, 8]);
}

#[test]
fn test_group_dims_impossible() {
    // Can't fit 1000 into max 10.
    let result = group_dims(&d(&[1000]), &[10]);
    assert!(result.is_none());
}

#[test]
fn test_non_cubic_local_dims_fit_product_cap() {
    // Regression: a 1024-product local shape with per-axis caps = product
    // ([1024;3]) fits [32,2,2] unchanged. The old cube-root cap (10 each)
    // made axis 0 (32) unfittable and panicked at split_dims.
    let result = group_dims(&d(&[32, 2, 2]), &[1024, 1024, 1024]);
    assert_eq!(dmax(&result.unwrap()), vec![32, 2, 2]);
}

#[test]
fn test_split_dims_simple() {
    // 100 exceeds 64, should split.
    let result = split_dims(&d(&[100]), &[64, 64, 64]).unwrap();
    assert!(result.iter().all(|x| dim_max(x) <= 64));
}

#[test]
fn test_split_dims_symbolic_too_big_returns_none() {
    // Symbolic dim with vmax > limit and no concrete factor — must report
    // failure rather than emit a malformed split.
    let v = UOp::variable("n".to_string(), 0, 200, DType::WeakInt);
    let result = split_dims(&[v], &[64, 64, 64]);
    assert!(result.is_none());
}

#[test]
fn test_find_smallest_divisor() {
    assert_eq!(find_smallest_divisor(1), 1);
    assert_eq!(find_smallest_divisor(2), 2);
    assert_eq!(find_smallest_divisor(3), 1); // prime
    assert_eq!(find_smallest_divisor(4), 2);
    assert_eq!(find_smallest_divisor(9), 3);
    assert_eq!(find_smallest_divisor(100), 2);
}

#[test]
fn test_group_dims_symbolic_fits_under_vmax() {
    // Symbolic dim with vmax=100 plus 3 concrete dims gets grouped down to 3
    // since 100*4 = 400 ≤ 65535 (typical y-axis cap).
    let v = UOp::variable("n".to_string(), 0, 100, DType::WeakInt);
    let dims = vec![v.clone(), UOp::index_const(4), UOp::index_const(8), UOp::index_const(8)];
    let result = group_dims(&dims, &[2147483647, 65535, 65535]).unwrap();
    assert_eq!(result.len(), 3);
    // First slot should hold the merged symbolic*4; vmax 400.
    assert_eq!(dim_max(&result[0]), 400);
    assert_eq!(dim_max(&result[1]), 8);
    assert_eq!(dim_max(&result[2]), 8);
}
