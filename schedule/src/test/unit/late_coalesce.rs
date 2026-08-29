use std::sync::Arc;

use svod_dtype::{AddrSpace, DType};
use svod_ir::{BinaryOp, ConstValue, Op, ParamArg, TernaryOp, UOp};

use crate::devectorize::devectorize;
use crate::graph_rewrite;
use crate::late::{AddImageContext, indexing_simplify, memory_coalescing, pm_simplify_add_image};
use crate::optimizer::Renderer;
use crate::symbolic::patterns::sym;

fn weak(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

fn image_param() -> Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![1usize.into(), 4usize.into(), 4usize.into()]);
    let arg = ParamArg::buffer(0, DType::Float32, AddrSpace::Global, None);
    UOp::new(Op::Param { shape, arg }, DType::Float32)
}

fn gated_index(index: &Arc<UOp>) -> (&Arc<UOp>, &Arc<UOp>) {
    let Op::Index { indices, .. } = index.op() else { panic!("expected INDEX, got {}", index.tree()) };
    let Op::Ternary(TernaryOp::Where, valid, idx, invalid) = indices[0].op() else {
        panic!("expected gated index, got {}", index.tree())
    };
    assert!(UOp::is_invalid_marker(invalid));
    (valid, idx)
}

#[test]
fn generic_index_is_simplified_under_its_validity() {
    let x = UOp::define_var("x".into(), 0, 7);
    let valid = x.lt(&weak(4));
    let start = x.mod_(&weak(4));
    let buffer = UOp::param(0, 8, DType::Int32, None);
    let index = UOp::index().buffer(buffer).indices(vec![start.valid(valid.clone())]).call().unwrap();

    let matcher = sym().clone() + indexing_simplify().clone();
    let result = graph_rewrite(&matcher, index, &mut ());
    let (result_valid, result_idx) = gated_index(&result);
    assert!(Arc::ptr_eq(result_valid, &valid));
    assert!(Arc::ptr_eq(result_idx, &x), "expected x % 4 to simplify to x, got {}", result_idx.tree());
}

#[test]
fn image_clause_is_dropped_only_when_wrong_side_is_out_of_bounds() {
    let x = UOp::define_var("x".into(), 0, 7);
    let valid = x.lt(&weak(4));
    let zero = weak(0);
    let image = image_param();
    let index = UOp::index()
        .buffer(image)
        .indices(vec![zero.valid(valid.clone()), x.valid(valid)])
        .dtype(DType::Float32)
        .call()
        .unwrap();

    let result = graph_rewrite(indexing_simplify(), index, &mut ());
    let Op::Index { indices, .. } = result.op() else { panic!("expected INDEX") };
    assert_eq!(indices.len(), 2);
    assert!(indices.iter().all(|idx| !matches!(idx.op(), Op::Ternary(TernaryOp::Where, ..))));
    assert!(matches!(indices[0].op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
    assert!(Arc::ptr_eq(&indices[1], &x));
}

#[test]
fn two_coordinate_non_image_index_is_not_an_image_rewrite() {
    let x = UOp::define_var("x".into(), 0, 7);
    let valid = x.lt(&weak(4));
    let buffer = UOp::param(0, 64, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![weak(0).valid(valid.clone()), x.valid(valid)]).call().unwrap();

    let result = graph_rewrite(indexing_simplify(), index.clone(), &mut ());
    assert!(Arc::ptr_eq(&result, &index));
}

#[test]
fn unparseable_validity_does_not_trigger_index_rewrite() {
    let x = UOp::define_var("x".into(), 0, 7);
    let valid = x.eq(&weak(3));
    let buffer = UOp::param(0, 8, DType::Int32, None);
    let index = UOp::index().buffer(buffer).indices(vec![x.valid(valid)]).call().unwrap();

    let result = graph_rewrite(indexing_simplify(), index.clone(), &mut ());
    assert!(Arc::ptr_eq(&result, &index));
}

#[test]
fn lower_bound_with_constant_on_left_is_parsed() {
    let x = UOp::define_var("x".into(), 0, 7);
    let valid = weak(3).lt(&x);
    let start = x.floor_div(&weak(4));
    let buffer = UOp::param(0, 8, DType::Int32, None);
    let index = UOp::index().buffer(buffer).indices(vec![start.valid(valid.clone())]).call().unwrap();

    let result = graph_rewrite(indexing_simplify(), index, &mut ());
    let (_, result_idx) = gated_index(&result);
    assert!(matches!(result_idx.op(), Op::Const(value) if value.0 == ConstValue::Int(1)));
    assert!(!matches!(result_idx.op(), Op::Binary(BinaryOp::FloorDiv, ..)));
}

fn load_at(buffer: &Arc<UOp>, index: Arc<UOp>) -> Arc<UOp> {
    UOp::load().index(UOp::index().buffer(buffer.clone()).indices(vec![index]).call().unwrap()).call()
}

fn loads(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Load { .. })).collect()
}

fn stores(root: &Arc<UOp>) -> Vec<Arc<UOp>> {
    root.toposort().into_iter().filter(|uop| matches!(uop.op(), Op::Store { .. })).collect()
}

fn shaped_index(buffer: &Arc<UOp>, offsets: &[i64]) -> Arc<UOp> {
    let indices = UOp::stack(offsets.iter().copied().map(UOp::index_const).collect());
    UOp::new(Op::Index { buffer: buffer.clone(), indices: smallvec::smallvec![indices] }, buffer.dtype())
}

fn target_coalesce(sink: Arc<UOp>, renderer: &Renderer) -> Arc<UOp> {
    let devectorized = devectorize(&sink, renderer);
    let simplified = graph_rewrite(sym(), devectorized, &mut ());
    memory_coalescing(simplified, renderer)
}

#[test]
fn shaped_width_four_load_is_devectorized_then_coalesced_as_scalar_memory() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let old = UOp::new(Op::Load { index: shaped_index(&buffer, &[0, 1, 2, 3]), alt: None, gate: None }, DType::Float32);

    let result = target_coalesce(UOp::sink(vec![old]), &Renderer::cpu());
    let folded = loads(&result);
    assert_eq!(folded.len(), 1, "expected one width-four memory operation, got {}", result.tree());
    assert_eq!(folded[0].dtype(), DType::Float32);
    let Op::Load { index, .. } = folded[0].op() else { unreachable!() };
    assert_eq!(index.dtype(), DType::Float32);
    assert_eq!(folded[0].shape().unwrap().unwrap()[0].as_const(), Some(4));
}

#[test]
fn shaped_width_eight_load_uses_two_scalar_dtype_width_four_accesses() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let old = UOp::new(
        Op::Load { index: shaped_index(&buffer, &[0, 1, 2, 3, 4, 5, 6, 7]), alt: None, gate: None },
        DType::Float32,
    );

    let result = target_coalesce(UOp::sink(vec![old]), &Renderer::cpu());
    let folded = loads(&result);
    assert_eq!(folded.len(), 2, "float width eight maps to two target width-four groups");
    assert!(folded.iter().all(|load| load.dtype() == DType::Float32));
}

#[test]
fn shaped_width_sixteen_load_folds_to_one_access_on_apple_amx() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let offsets: Vec<i64> = (0..16).collect();
    let old = UOp::new(Op::Load { index: shaped_index(&buffer, &offsets), alt: None, gate: None }, DType::Float32);

    let result = target_coalesce(UOp::sink(vec![old]), &Renderer::apple_amx());
    let folded = loads(&result);
    assert_eq!(folded.len(), 1, "AMX folds a whole 16-lane register, got {}", result.tree());
    assert_eq!(folded[0].shape().unwrap().unwrap()[0].as_const(), Some(16));
}

#[test]
fn shaped_store_is_devectorized_then_coalesced_with_scalar_memory_dtype() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let value = UOp::stack((0..4).map(|value| UOp::const_(DType::Float32, ConstValue::Float(value as f64))).collect());
    let old = UOp::new(Op::Store { index: shaped_index(&buffer, &[0, 1, 2, 3]), value, gate: None }, DType::Void);

    let result = target_coalesce(UOp::sink(vec![old]), &Renderer::cpu());
    let folded = stores(&result);
    assert_eq!(folded.len(), 1, "expected one width-four shaped store, got {}", result.tree());
    let Op::Store { index, value, .. } = folded[0].op() else { unreachable!() };
    assert_eq!(index.dtype(), DType::Float32);
    assert_eq!(value.dtype(), DType::Float32);
    assert!(matches!(value.op(), Op::Stack { sources } if sources.len() == 4));
}

#[test]
fn shaped_noncontiguous_offsets_form_distinct_target_runs() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let old = UOp::new(
        Op::Load { index: shaped_index(&buffer, &[0, 1, 2, 3, 8, 9, 10, 11]), alt: None, gate: None },
        DType::Float32,
    );

    let result = target_coalesce(UOp::sink(vec![old]), &Renderer::cpu());
    assert_eq!(loads(&result).len(), 2, "the gap must not be bridged into one shaped access");
}

#[test]
fn contiguous_loads_fold_to_one_shaped_scalar_load() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let accesses = (0..4).map(|offset| load_at(&buffer, UOp::index_const(offset))).collect();

    let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
    let folded = loads(&result);
    assert_eq!(folded.len(), 1, "expected one coalesced LOAD, got {}", result.tree());
    assert_eq!(folded[0].dtype(), DType::Float32);
    assert_eq!(folded[0].shape().unwrap().unwrap().iter().map(|x| x.as_const()).collect::<Vec<_>>(), vec![Some(4)]);
    let Op::Load { index, .. } = folded[0].op() else { unreachable!() };
    let Op::Shrink { offsets, sizes, .. } = index.op() else { panic!("expected SHRINK") };
    assert_eq!(offsets.dtype(), DType::WeakInt);
    assert_eq!(sizes.dtype(), DType::WeakInt);
    assert!(matches!(sizes.op(), Op::Const(value) if value.0 == ConstValue::Int(4)));
}

#[test]
fn wrapped_reg_loads_remain_scalar() {
    let buffer = UOp::buffer(0, 4, DType::Float32, AddrSpace::Reg, None).after(smallvec::smallvec![UOp::noop()]);
    let accesses = (0..4).map(|offset| load_at(&buffer, UOp::index_const(offset))).collect();

    let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
    assert_eq!(loads(&result).len(), 4, "REG accesses must not coalesce: {}", result.tree());
    assert!(!result.toposort().iter().any(|u| matches!(u.op(), Op::Shrink { .. })), "{}", result.tree());
}

#[test]
fn grouped_shrink_is_preserved_for_vector_memory_rendering() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let accesses = (0..4).map(|offset| load_at(&buffer, UOp::index_const(offset))).collect();
    let coalesced = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
    assert!(coalesced.toposort().iter().any(|u| matches!(u.op(), Op::Shrink { .. })), "{}", coalesced.tree());
    assert_eq!(loads(&coalesced).len(), 1);
}

#[test]
fn grouped_weak_index_offsets_match_tinygrad_for_widths_four_five_and_eight() {
    let buffer = UOp::param(0, 32, DType::Float32, None);
    let base = UOp::define_var("group_width_base".into(), 0, 3).mul(&weak(8));

    for width in [4usize, 5, 8] {
        let accesses = (0..width).map(|offset| load_at(&buffer, base.add(&base.const_like(offset as i64)))).collect();
        let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
        let folded = loads(&result);
        let expected_groups = width.div_ceil(4);
        assert_eq!(folded.len(), expected_groups, "width {width}: {}", result.tree());

        let mut grouped_widths = Vec::new();
        let mut scalar_offsets = Vec::new();
        for load in folded {
            let Op::Load { index, .. } = load.op() else { unreachable!() };
            match index.op() {
                Op::Shrink { offsets, sizes, .. } => {
                    assert_eq!(offsets.dtype(), DType::WeakInt);
                    assert_eq!(sizes.dtype(), DType::WeakInt);
                    assert_eq!(offsets.shape().unwrap().unwrap().as_slice(), &[]);
                    assert_eq!(sizes.shape().unwrap().unwrap().as_slice(), &[]);
                    let Op::Const(value) = sizes.op() else { panic!("width must be CONST") };
                    grouped_widths.push(value.0.clone());
                }
                Op::Index { indices, .. } => {
                    assert_eq!(indices[0].dtype(), DType::WeakInt);
                    let Op::Binary(BinaryOp::Add, _, offset) = indices[0].op() else {
                        panic!("scalar offset must preserve its base")
                    };
                    let Op::Const(value) = offset.op() else { panic!("offset must be CONST") };
                    scalar_offsets.push(value.0.clone());
                }
                _ => panic!("expected SHRINK or INDEX, got {}", index.tree()),
            }
        }

        let expected_widths =
            if width == 8 { vec![ConstValue::Int(4), ConstValue::Int(4)] } else { vec![ConstValue::Int(4)] };
        assert_eq!(grouped_widths, expected_widths, "width {width}");
        let expected_scalars = if width == 5 { vec![ConstValue::Int(4)] } else { vec![] };
        assert_eq!(scalar_offsets, expected_scalars, "width {width}");
    }
}

#[test]
fn mismatched_validity_masks_do_not_coalesce() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let x = UOp::define_var("x".into(), 0, 7);
    let valid0 = x.lt(&weak(4));
    let valid1 = x.lt(&weak(5));
    let sink = UOp::sink(vec![
        load_at(&buffer, UOp::index_const(0).valid(valid0)),
        load_at(&buffer, UOp::index_const(1).valid(valid1)),
    ]);

    let result = memory_coalescing(sink, &Renderer::cpu());
    assert_eq!(loads(&result).len(), 2, "different validity identities must form different groups");
}

#[test]
fn shaped_load_with_shared_validity_preserves_one_group_gate() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let x = UOp::define_var("x".into(), 0, 7);
    let valid = x.lt(&weak(4));
    let indices = UOp::stack((0..4).map(|offset| UOp::index_const(offset).valid(valid.clone())).collect());
    let index = UOp::new(Op::Index { buffer, indices: smallvec::smallvec![indices] }, DType::Float32);
    let load = UOp::new(Op::Load { index, alt: None, gate: None }, DType::Float32);

    let result = target_coalesce(UOp::sink(vec![load]), &Renderer::cpu());
    let folded = loads(&result);
    assert_eq!(folded.len(), 1, "shared validity should produce one shaped access: {}", result.tree());
    let Op::Load { index, .. } = folded[0].op() else { unreachable!() };
    let Op::Shrink { offsets, sizes, .. } = index.op() else { panic!("expected SHRINK: {}", index.tree()) };
    assert!(Arc::ptr_eq(&offsets.get_valid(), &valid));
    assert!(matches!(offsets.get_idx().op(), Op::Const(value) if value.0 == ConstValue::Int(0)));
    assert!(matches!(sizes.op(), Op::Const(value) if value.0 == ConstValue::Int(4)));
}

#[test]
fn matching_offsets_with_different_bases_do_not_coalesce_together() {
    let buffer = UOp::param(0, 64, DType::Float32, None);
    let x = UOp::define_var("x".into(), 0, 7).mul(&weak(2));
    let y = UOp::define_var("y".into(), 0, 7).mul(&weak(2));
    let mut accesses = Vec::new();
    for base in [x, y] {
        for offset in 0..2 {
            accesses.push(load_at(&buffer, base.add(&weak(offset))));
        }
    }

    let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
    assert_eq!(loads(&result).len(), 2, "different base identities must remain distinct groups");
}

#[test]
fn noncontiguous_offsets_do_not_bridge_runs_or_realign() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let accesses = [0, 1, 3, 4].into_iter().map(|offset| load_at(&buffer, UOp::index_const(offset))).collect();

    let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
    assert_eq!(loads(&result).len(), 3, "[0,1] folds, but unaligned [3,4] remains scalar");
}

#[test]
fn renderer_without_float4_keeps_scalar_accesses() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let accesses = (0..4).map(|offset| load_at(&buffer, UOp::index_const(offset))).collect();
    let mut renderer = Renderer::cpu();
    renderer.supports_float4 = false;

    let result = memory_coalescing(UOp::sink(accesses), &renderer);
    assert_eq!(loads(&result).len(), 4);
}

#[test]
fn contiguous_stores_fold_to_one_shaped_scalar_store() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let accesses = (0..4)
        .map(|offset| {
            UOp::index()
                .buffer(buffer.clone())
                .indices(vec![UOp::index_const(offset)])
                .call()
                .unwrap()
                .store(UOp::const_(DType::Float32, ConstValue::Float(offset as f64)))
        })
        .collect();

    let result = memory_coalescing(UOp::sink(accesses), &Renderer::cpu());
    let folded = stores(&result);
    assert_eq!(folded.len(), 1, "expected one coalesced STORE, got {}", result.tree());
    let Op::Store { index, value, .. } = folded[0].op() else { unreachable!() };
    assert_eq!(index.dtype(), DType::Float32);
    assert!(matches!(value.op(), Op::Stack { sources } if sources.len() == 4));
}

#[test]
fn m5_wmma_stores_remain_distinct_before_and_after_coalescing() {
    let output = UOp::param(0, 80, DType::Float32, None);
    let lidx = UOp::special(weak(32), "lidx0".to_string());
    let valid = lidx.lt(&weak(16));
    let indices = [lidx.clone(), lidx.add(&weak(32)), lidx.add(&weak(64)).valid(valid.clone())];
    let before = UOp::sink(
        indices
            .into_iter()
            .enumerate()
            .map(|(value, index)| {
                UOp::index()
                    .buffer(output.clone())
                    .indices(vec![index])
                    .call()
                    .unwrap()
                    .store(UOp::const_(DType::Float32, ConstValue::Float(value as f64)))
            })
            .collect(),
    );

    assert_eq!(stores(&before).len(), 3, "M=5 pre-coalescing mapping must contain three stores");
    let after = memory_coalescing(before, &Renderer::cpu());
    let after_stores = stores(&after);
    assert_eq!(after_stores.len(), 3, "M=5 post-coalescing mapping must keep all three stores distinct");
    assert_eq!(
        after_stores
            .iter()
            .filter(|store| matches!(store.op(), Op::Store { index, .. }
                if matches!(index.op(), Op::Index { indices, .. }
                    if Arc::ptr_eq(&indices[0].get_valid(), &valid))))
            .count(),
        1,
        "only the C[64..80) store uses the M=5 validity identity",
    );
}

#[test]
fn multiple_stores_to_one_group_offset_are_left_un_coalesced() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let first = index.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
    let second = index.store(UOp::const_(DType::Float32, ConstValue::Float(2.0)));
    let result = memory_coalescing(UOp::sink(vec![first, second]), &Renderer::cpu());

    assert_eq!(stores(&result).len(), 2, "both stores survive; coalescing declines the group");
}

#[test]
fn image_accesses_use_fixed_width_four_without_float4_capability() {
    let image = image_param();
    let accesses = (0..4).map(|offset| load_at(&image, UOp::index_const(offset))).collect();
    let mut renderer = Renderer::cpu();
    renderer.supports_float4 = false;

    let result = memory_coalescing(UOp::sink(accesses), &renderer);
    assert_eq!(loads(&result).len(), 1);
}

#[test]
fn image_float_half_float_roundtrip_is_removed() {
    let value = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let roundtrip = value.cast(DType::Float16).cast(DType::Float32);
    let mut ctx: AddImageContext = (std::collections::HashMap::new(), Renderer::cpu());

    let result = graph_rewrite(&pm_simplify_add_image(), roundtrip, &mut ctx);

    assert!(Arc::ptr_eq(&result, &value));
}

#[test]
fn gated_load_is_skipped_rather_than_aborting_the_pass() {
    let buffer = UOp::param(0, 16, DType::Float32, None);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    let old = UOp::new(
        Op::Load {
            index,
            alt: Some(UOp::const_(DType::Float32, ConstValue::Float(0.0))),
            gate: Some(UOp::const_(DType::Bool, ConstValue::Bool(true))),
        },
        DType::Float32,
    );

    let folded = loads(&memory_coalescing(UOp::sink(vec![old.clone()]), &Renderer::cpu()));
    assert_eq!(folded.len(), 1, "the gated load must survive un-coalesced");
    assert!(Arc::ptr_eq(&folded[0], &old));
}
