use super::*;
use test_case::test_case;

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
    let sink = UOp::sink(vec![thread.clone()]);

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

    // One core_id cannot stand in for two THREAD axes: decline, don't panic.
    let second = UOp::range_axis(UOp::index_const(3), svod_ir::AxisId::Renumbered(1), AxisType::Thread);
    assert!(add_gpudims(&Renderer::cpu(), &UOp::sink(vec![thread, second])).is_none());
}

#[test]
fn test_existing_special_skips_all_gpudims_lowering() {
    let global = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::Global);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let sink = UOp::sink(vec![global, special]);

    assert!(add_gpudims(&Renderer::amd_cdna3(), &sink).is_none());
}

/// Extents of the `gidx*` SPECIALs `add_gpudims` emits for the given axes,
/// sorted so the assertion does not depend on toposort order.
fn global_special_extents(renderer: &Renderer, global_extents: &[i64], local_extents: &[i64]) -> Vec<usize> {
    let mut ranges = Vec::new();
    for (extents, axis_type) in [(global_extents, AxisType::Global), (local_extents, AxisType::Local)] {
        for &extent in extents {
            let axis = svod_ir::AxisId::Renumbered(ranges.len());
            ranges.push(UOp::range_axis(UOp::index_const(extent), axis, axis_type));
        }
    }
    let lowered = add_gpudims(renderer, &UOp::sink(ranges)).expect("GPU ranges should lower");
    let mut ends: Vec<usize> = lowered
        .toposort()
        .into_iter()
        .filter_map(|uop| match uop.op() {
            Op::Special { end, name } if name.starts_with("gidx") => Some(dim_max(end)),
            _ => None,
        })
        .collect();
    ends.sort_unstable();
    ends
}

#[test_case(&[8], &[4, 1 << 28]; "one local axis divides the work item cap")]
#[test_case(&[0], &[1 << 30]; "zero extent local axis does not divide by zero")]
#[test_case(&[64, 64, 64, 4], &[32, 1 << 25]; "contracted locals still cap the grid")]
fn test_global_product_cap_accounts_for_local_extent(local_extents: &[i64], expected: &[usize]) {
    assert_eq!(global_special_extents(&Renderer::amd_cdna3(), &[1 << 30], local_extents), expected);
}

#[test]
fn test_device_range_lowers_and_end_drops_all_params() {
    let device =
        UOp::range_axis_dtype(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::Device, DType::Index);
    let other = UOp::variable("other".to_string(), 0, 7, DType::Index);
    let computation = device.add(&UOp::const_(DType::Index, ConstValue::Int(1)));
    let ended = computation.end(smallvec::smallvec![device, other]);
    let lowered = crate::rewrite::graph_rewrite(&pm_lower_device_ranges(), ended, &mut ());

    let Op::End { ranges, .. } = lowered.op() else { panic!("target keeps an empty END") };
    assert!(ranges.is_empty(), "Tinygrad removes every PARAM when _device_num is present");
    let device_num =
        lowered.toposort().into_iter().find(|uop| is_device_num(uop)).expect("DEVICE range should become _device_num");
    assert_eq!(device_num.dtype(), DType::Index);
    assert_eq!(device_num.vmax(), &ConstValue::Int(3));
}

#[test]
fn test_device_range_lowers_without_gpu_dimension_capability() {
    let range =
        UOp::range_axis_dtype(UOp::native_const(2i32), svod_ir::AxisId::Renumbered(0), AxisType::Device, DType::Int32);
    let lowered = crate::rewrite::graph_rewrite(&pm_lower_device_ranges(), range, &mut ());
    assert!(matches!(lowered.op(), Op::Param { arg, .. } if arg.name.as_deref() == Some("_device_num")));
    assert_eq!(lowered.vmin().try_int(), Some(0));
    assert_eq!(lowered.vmax().try_int(), Some(1));
}

#[test]
fn test_non_device_end_keeps_param_counterexample() {
    let other = UOp::variable("other".to_string(), 0, 7, DType::Index);
    let ended = UOp::index_const(1).end(smallvec::smallvec![other.clone()]);
    let lowered = crate::rewrite::graph_rewrite(&pm_add_gpudims(), ended.clone(), &mut Renderer::amd_cdna3());

    assert!(Arc::ptr_eq(&lowered, &ended));
}

#[test_case(UOp::param(0, 16, DType::Float32, None); "bare global param")]
#[test_case(UOp::param(0, 16, DType::Float32, None).after(smallvec::smallvec![UOp::noop()]); "param behind AFTER")]
#[test_case(UOp::stack(smallvec::smallvec![
    UOp::param(0, 16, DType::Float32, None),
    UOp::param(1, 16, DType::Float32, None),
]); "stack of global params")]
fn test_missing_group_reduce_masks_structured_global_param_store(buffer: Arc<UOp>) {
    let group = UOp::range_axis(UOp::index_const(4), svod_ir::AxisId::Renumbered(0), AxisType::GroupReduce);
    // A symbolic offset keeps the INDEX from folding through the STACK row.
    let offset = UOp::variable("off".to_string(), 0, 15, DType::Index);
    let index = UOp::index().buffer(buffer).indices(vec![offset]).call().expect("index");
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
