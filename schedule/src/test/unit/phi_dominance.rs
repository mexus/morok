//! Minimal reproducible example for the phi-dominance bug.
//!
//! The kmeans generic baseline (`matmul → min over K`) produces invalid LLVM IR
//! at K≥1024 on gfx1151: a value derived from an inner-loop counter is used
//! after the loop exits. This module builds the minimal graph that triggers the
//! same issue, so we can iterate on the fix without running the full bench.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, Op, ReduceOp, UOp};

use crate::linearize::linearize_with_cfg;
use crate::optimizer::config::OptStrategy;
use crate::optimizer::tc;
use crate::optimizer::{
    OptimizerConfig, Renderer, Scheduler, apply_post_optimization_with_renderer, optimize_kernel_with_config,
};

// ── graph builders ─────────────────────────────────────────────────────────

/// `result[n] = MIN_k ( Σ_d A[n,d] · B[d,k] )`
///
/// Three axes: N (Global), K (Global→Reduce-Min), D (Reduce-Add contraction).
/// Mirrors the kmeans generic baseline: `x @ cᵀ → min(1)`.
///
/// Inputs are BFloat16 so that RDNA4 WMMA (bf16→f32) is selectable.
fn build_matmul_min(n: i64, k: i64, d: i64) -> Arc<UOp> {
    let n_r = UOp::range_axis(UOp::index_const(n), AxisId::Renumbered(0), AxisType::Global);
    let k_r = UOp::range_axis(UOp::index_const(k), AxisId::Renumbered(1), AxisType::Global);
    let d_r = UOp::range_axis(UOp::index_const(d), AxisId::Renumbered(2), AxisType::Reduce);

    let nf = n_r.clone().cast(DType::BFloat16);
    let kf = k_r.clone().cast(DType::BFloat16);
    let df = d_r.clone().cast(DType::BFloat16);

    let a = nf.try_add(&df).unwrap();
    let b = df.try_add(&kf).unwrap();
    let prod = a.try_mul(&b).unwrap();
    let matmul = prod.reduce(smallvec![d_r], ReduceOp::Add);
    let result = matmul.reduce(smallvec![k_r], ReduceOp::Min);
    UOp::sink(vec![result, n_r])
}

/// Plain matmul: `C[n,k] = Σ_d A[n,d] · B[d,k]` — no second reduction.
fn build_matmul_only(n: i64, k: i64, d: i64) -> Arc<UOp> {
    let n_r = UOp::range_axis(UOp::index_const(n), AxisId::Renumbered(0), AxisType::Global);
    let k_r = UOp::range_axis(UOp::index_const(k), AxisId::Renumbered(1), AxisType::Global);
    let d_r = UOp::range_axis(UOp::index_const(d), AxisId::Renumbered(2), AxisType::Reduce);

    let nf = n_r.clone().cast(DType::BFloat16);
    let kf = k_r.clone().cast(DType::BFloat16);
    let df = d_r.clone().cast(DType::BFloat16);

    let a = nf.try_add(&df).unwrap();
    let b = df.try_add(&kf).unwrap();
    let prod = a.try_mul(&b).unwrap();
    let matmul = prod.reduce(smallvec![d_r], ReduceOp::Add);
    UOp::sink(vec![matmul, n_r, k_r])
}

// ── dominance checker ──────────────────────────────────────────────────────

/// Validate that no instruction in the linearized list references a value from
/// a closed (ended) loop scope without going through AFTER.
///
/// Algorithm: scan the list left-to-right, maintaining:
/// - `open_ranges`: RANGEs whose END has not yet been seen.
/// - `range_deps`: transitive closure of RANGE dependencies for each UOp.
///
/// AFTER nodes strip all range deps (they represent the loop-exit value).
fn check_phi_dominance(linear: &[Arc<UOp>]) -> Result<(), String> {
    let mut range_deps: HashMap<u64, HashSet<u64>> = HashMap::new();
    let mut open_ranges: HashSet<u64> = HashSet::new();

    for (idx, uop) in linear.iter().enumerate() {
        match uop.op() {
            Op::Range { .. } => {
                open_ranges.insert(uop.id);
                let mut deps = HashSet::from([uop.id]);
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                range_deps.insert(uop.id, deps);
            }

            Op::End { ranges, .. } => {
                // Check END's own sources before closing.
                let mut deps = HashSet::new();
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                for rid in &deps {
                    if !open_ranges.contains(rid) {
                        return Err(format!("END at [{idx}] depends on closed range {rid}"));
                    }
                }
                range_deps.insert(uop.id, deps);
                for r in ranges {
                    open_ranges.remove(&r.id);
                }
            }

            Op::After { .. } => {
                range_deps.insert(uop.id, HashSet::new());
            }

            _ => {
                let mut deps = HashSet::new();
                for src in uop.op().sources() {
                    deps.extend(range_deps.get(&src.id).cloned().unwrap_or_default());
                }
                for rid in &deps {
                    if !open_ranges.contains(rid) {
                        return Err(format!(
                            "phi-dominance violation at [{idx}]: {:?} depends on closed range {rid}",
                            uop.op()
                        ));
                    }
                }
                range_deps.insert(uop.id, deps);
            }
        }
    }
    Ok(())
}

// ── tests ──────────────────────────────────────────────────────────────────

fn assert_no_phi_violation(sink: Arc<UOp>, renderer: &Renderer, label: &str) {
    let config = OptimizerConfig { strategy: OptStrategy::Heuristic, ..Default::default() };
    let optimized = optimize_kernel_with_config(sink, renderer, &config)
        .unwrap_or_else(|e| panic!("{label}: optimizer failed: {e:?}"));

    // Diagnostics
    let topo = optimized.toposort();
    let has_wmma = topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. }));
    let n_ranges = topo.iter().filter(|u| matches!(u.op(), Op::Range { .. })).count();
    let n_ends = topo.iter().filter(|u| matches!(u.op(), Op::End { .. })).count();
    eprintln!("[{label}] ops={}, wmma={has_wmma}, ranges={n_ranges}, ends={n_ends}", topo.len());

    let linear = linearize_with_cfg(optimized);
    eprintln!("[{label}] linear={}", linear.len());
    check_phi_dominance(&linear).unwrap_or_else(|e| panic!("{label}: {e}"));
}

// ── tests: manual TC pipeline ──────────────────────────────────────────────
//
// The heuristic optimizer in `optimize_kernel_with_config` doesn't always try
// TC for our hand-built graph.  These tests apply TC manually (like tc.rs
// tests), then run the full post-optimization + linearize pipeline.

fn assert_no_phi_with_tc(sink: Arc<UOp>, renderer: &Renderer, label: &str) {
    let mut scheduler = Scheduler::new(sink, renderer.clone());

    // Apply TC: tc_select=-1 (auto), axis0=0, axis1=1.
    let _axes = tc::apply(&mut scheduler, -1, 0, 1).unwrap_or_else(|e| panic!("{label}: TC apply failed: {e:?}"));

    let ast = scheduler.get_optimized_ast(None);
    let has_wmma = ast.toposort().iter().any(|u| matches!(u.op(), Op::Wmma { .. }));
    assert!(has_wmma, "{label}: TC apply did not produce WMMA");

    let post = apply_post_optimization_with_renderer(ast, Some(renderer));
    let linear = linearize_with_cfg(post);
    eprintln!("[{label}] linear={}", linear.len());
    check_phi_dominance(&linear).unwrap_or_else(|e| panic!("{label}: {e}"));
}

#[test]
fn matmul_only_rdna4() {
    let sink = build_matmul_only(64, 1024, 64);
    assert_no_phi_violation(sink, &Renderer::amd_rdna4(), "rdna4 matmul-only");
}

#[test]
fn matmul_only_cdna3() {
    let sink = build_matmul_only(64, 1024, 64);
    assert_no_phi_violation(sink, &Renderer::amd_cdna3(), "cdna3 matmul-only");
}

#[test]
fn matmul_only_rdna4_tc() {
    let sink = build_matmul_only(64, 1024, 64);
    assert_no_phi_with_tc(sink, &Renderer::amd_rdna4(), "rdna4 matmul-only TC");
}

#[test]
fn matmul_min_rdna4_tc() {
    let sink = build_matmul_min(64, 1024, 64);
    assert_no_phi_with_tc(sink, &Renderer::amd_rdna4(), "rdna4 matmul+min TC");
}

#[test]
fn matmul_min_rdna4_small_k() {
    let sink = build_matmul_min(64, 256, 64);
    assert_no_phi_violation(sink, &Renderer::amd_rdna4(), "rdna4 K=256");
}

#[test]
fn matmul_min_rdna4_large_k() {
    let sink = build_matmul_min(64, 1024, 64);
    assert_no_phi_violation(sink, &Renderer::amd_rdna4(), "rdna4 K=1024");
}

#[test]
fn matmul_min_cdna3_large_k() {
    let sink = build_matmul_min(64, 1024, 64);
    assert_no_phi_violation(sink, &Renderer::amd_cdna3(), "cdna3 K=1024");
}
