use std::sync::Arc;

use svod_dtype::{DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use super::helpers::{create_bool_const, create_buffer_typed};

/// Post-gater boundary: FP8 decomposition preserves the load gate/alt pair.
#[test]
fn test_fp8_decomp_preserves_alt_on_gated_load() {
    let buffer = create_buffer_typed(64, ScalarDType::FP8E5M2);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let gate = create_bool_const(false);
    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().unwrap();
    let load = UOp::load()
        .index(index)
        .alt(UOp::const_(DType::Scalar(ScalarDType::FP8E5M2), ConstValue::Float(0.0)))
        .gate(gate)
        .call();

    assert_eq!(count_gated_loads_without_alt(&load), 0);

    let mut ctx = crate::devectorize::Fp8DecompCtx { from: ScalarDType::FP8E5M2, to: ScalarDType::Float16 };
    let decomposed = svod_ir::rewrite::graph_rewrite_bottom_up(&crate::devectorize::pm_float_decomp(), load, &mut ctx);

    assert!(count_gated_loads(&decomposed) > 0, "expected at least one gated load after FP8 decomposition");
    assert_eq!(count_gated_loads_without_alt(&decomposed), 0, "FP8 decomposition must preserve alt on gated loads");
}

#[test]
fn vector_fp8_load_decomposes_to_scalar_loads_and_stack() {
    let buffer = create_buffer_typed(4, ScalarDType::FP8E4M3);
    let indices = UOp::stack((0..4).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect());
    let index =
        UOp::index().buffer(buffer).indices(vec![indices]).call().unwrap().with_dtype(DType::FP8E4M3.vec(4).unwrap());
    let load = UOp::load().index(index).dtype(DType::FP8E4M3.vec(4).unwrap()).call();
    let mut ctx = crate::devectorize::Fp8DecompCtx { from: ScalarDType::FP8E4M3, to: ScalarDType::Float16 };
    let decomposed = svod_ir::rewrite::graph_rewrite_bottom_up(&crate::devectorize::pm_float_decomp(), load, &mut ctx);

    assert!(matches!(decomposed.op(), Op::Stack { .. }), "{}", decomposed.tree());
    assert_eq!(decomposed.dtype(), DType::Float16);
    assert_eq!(decomposed.toposort().iter().filter(|u| matches!(u.op(), Op::Load { .. })).count(), 4);
}

fn count_gated_loads(root: &Arc<UOp>) -> usize {
    root.toposort().into_iter().filter(|node| matches!(node.op(), Op::Load { gate: Some(_), .. })).count()
}

fn count_gated_loads_without_alt(root: &Arc<UOp>) -> usize {
    root.toposort().into_iter().filter(|node| matches!(node.op(), Op::Load { alt: None, gate: Some(_), .. })).count()
}

#[test]
fn dtype_decomposition_mapping_is_target_sensitive() {
    use crate::optimizer::{Renderer, get_dtype_decomps};
    use svod_dtype::AmdArch;

    let values = [
        (ScalarDType::FP8E4M3, ConstValue::Float(1.0)),
        (ScalarDType::FP8E4M3FNUZ, ConstValue::Float(1.0)),
        (ScalarDType::FP8E5M2, ConstValue::Float(1.0)),
        (ScalarDType::FP8E5M2FNUZ, ConstValue::Float(1.0)),
        (ScalarDType::Float16, ConstValue::Float(1.0)),
        (ScalarDType::BFloat16, ConstValue::Float(1.0)),
        (ScalarDType::Int64, ConstValue::Int(1)),
        (ScalarDType::UInt64, ConstValue::UInt(1)),
    ];
    let sink = UOp::sink(values.into_iter().map(|(dt, value)| UOp::const_(DType::Scalar(dt), value)).collect());

    assert_eq!(
        get_dtype_decomps(&sink, &Renderer::cpu()),
        vec![
            (ScalarDType::FP8E4M3, ScalarDType::Float16),
            (ScalarDType::FP8E5M2, ScalarDType::Float16),
            (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16),
            (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16),
        ]
    );
    assert_eq!(
        get_dtype_decomps(&sink, &Renderer::for_amd_arch(AmdArch::Gfx942)),
        vec![(ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16), (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16),]
    );
    assert_eq!(
        get_dtype_decomps(&sink, &Renderer::for_amd_arch(AmdArch::Gfx950)),
        vec![(ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16), (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16),]
    );
    assert_eq!(
        get_dtype_decomps(&sink, &Renderer::for_amd_arch(AmdArch::Gfx1151)),
        vec![
            (ScalarDType::FP8E4M3, ScalarDType::Float16),
            (ScalarDType::FP8E5M2, ScalarDType::Float16),
            (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float16),
            (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float16),
        ]
    );
    assert_eq!(
        get_dtype_decomps(&sink, &Renderer::webgpu()),
        vec![
            (ScalarDType::Int64, ScalarDType::Int32),
            (ScalarDType::FP8E4M3, ScalarDType::Float32),
            (ScalarDType::FP8E5M2, ScalarDType::Float32),
            (ScalarDType::Float16, ScalarDType::Float32),
            (ScalarDType::BFloat16, ScalarDType::Float32),
            (ScalarDType::FP8E4M3FNUZ, ScalarDType::Float32),
            (ScalarDType::FP8E5M2FNUZ, ScalarDType::Float32),
        ]
    );
}

#[test]
fn fnuz_store_and_load_are_both_decomposed() {
    let buffer = create_buffer_typed(4, ScalarDType::FP8E4M3FNUZ);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().unwrap();
    let load = UOp::load().index(index.clone()).call();
    let root = UOp::sink(vec![
        index.store(UOp::const_(DType::Scalar(ScalarDType::FP8E4M3FNUZ), ConstValue::Float(1.0))),
        load,
    ]);
    let mut ctx = crate::devectorize::Fp8DecompCtx { from: ScalarDType::FP8E4M3FNUZ, to: ScalarDType::Float16 };
    let decomposed = svod_ir::rewrite::graph_rewrite_bottom_up(&crate::devectorize::pm_float_decomp(), root, &mut ctx);

    assert!(!decomposed.toposort().iter().any(|u| u.dtype().base() == ScalarDType::FP8E4M3FNUZ));
    assert!(
        decomposed
            .toposort()
            .iter()
            .any(|u| matches!(u.op(), Op::Store { value, .. } if value.dtype() == DType::UInt8))
    );
    assert!(decomposed.toposort().iter().any(|u| matches!(u.op(), Op::Load { .. }) && u.dtype() == DType::UInt8));
}

#[test]
fn long_store_splits_but_native_long_is_untouched() {
    let buffer = create_buffer_typed(4, ScalarDType::Int64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));
    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().unwrap();
    let root = index.store(UOp::const_(DType::Int64, ConstValue::Int(0x1234_5678_7654_3210)));
    let decomposed =
        svod_ir::rewrite::graph_rewrite_bottom_up(&crate::devectorize::pm_long_decomp(), root.clone(), &mut ());

    assert_eq!(root.toposort().iter().filter(|u| matches!(u.op(), Op::Store { .. })).count(), 1);
    assert_eq!(
        decomposed.toposort().iter().filter(|u| matches!(u.op(), Op::Store { .. })).count(),
        2,
        "{}",
        decomposed.tree()
    );
    assert!(!decomposed.toposort().iter().any(|u| u.dtype().base() == ScalarDType::Int64));
}

#[test]
fn combined_dtype_pass_commits_mixed_fp8_bf16_weak_stores_before_decomposition() {
    let fp8 = create_buffer_typed(4, ScalarDType::FP8E4M3);
    let bf16 = create_buffer_typed(4, ScalarDType::BFloat16);
    let offset = UOp::const_(DType::Index, ConstValue::Int(0));
    let fp8_index = UOp::index().buffer(fp8).indices(vec![offset.clone()]).call().unwrap();
    let bf16_index = UOp::index().buffer(bf16).indices(vec![offset]).call().unwrap();
    let root = UOp::sink(vec![
        fp8_index.store(UOp::const_(DType::WeakFloat, ConstValue::Float(1.5))),
        bf16_index.store(UOp::const_(DType::WeakFloat, ConstValue::Float(-2.0))),
    ]);

    let decomposed = crate::optimizer::apply_dtype_decomps(root, crate::optimizer::Renderer::webgpu());
    assert!(
        decomposed
            .toposort()
            .iter()
            .all(|u| { !matches!(u.dtype().base(), ScalarDType::FP8E4M3 | ScalarDType::BFloat16) }),
        "{}",
        decomposed.tree()
    );
    let store_dtypes: Vec<_> = decomposed
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Store { value, .. } => Some(value.dtype()),
            _ => None,
        })
        .collect();
    assert!(store_dtypes.contains(&DType::UInt8), "{}", decomposed.tree());
    assert!(store_dtypes.contains(&DType::UInt16), "{}", decomposed.tree());
}

#[test]
fn combined_dtype_pass_commits_long_weak_store_before_word_split() {
    let buffer = create_buffer_typed(4, ScalarDType::Int64);
    let offset = UOp::const_(DType::Index, ConstValue::Int(0));
    let index = UOp::index().buffer(buffer).indices(vec![offset]).call().unwrap();
    let value = UOp::new(
        Op::Binary(
            svod_ir::BinaryOp::Add,
            UOp::const_(DType::WeakInt, ConstValue::Int(0x1_0000_0000)),
            UOp::const_(DType::WeakInt, ConstValue::Int(0x7654_3210)),
        ),
        DType::WeakInt,
    );

    let decomposed = crate::optimizer::apply_dtype_decomps(
        UOp::sink(vec![index.store(value)]),
        crate::optimizer::Renderer::webgpu(),
    );
    let stores: Vec<_> = decomposed
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Store { value, .. } => Some(value.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(stores.len(), 2, "{}", decomposed.tree());
    assert!(stores.iter().all(|value| value.dtype() == DType::Int32), "{}", decomposed.tree());
    assert!(stores.iter().all(|value| !value.dtype().is_weak()), "{}", decomposed.tree());
}
