//! GPU dimension injection for kernel execution.
//!
//! This module implements `pm_add_gpudims`, which transforms RANGE operations
//! with GLOBAL/LOCAL axis types into SPECIAL UOps representing GPU thread indices.
//!
//! Based on Tinygrad's `gpudims.py`.
//!
//! # Pipeline Position
//!
//! Runs between `pm_reduce` (Stage 11) and `pm_add_loads` (Stage 13):
//! - After reduction is lowered to accumulator patterns
//! - Before loads are explicitly extracted from INDEX ops
//!
//! # Transformation
//!
//! ```text
//! RANGE(end, axis_id, GLOBAL) → gidxN (SPECIAL with global thread index)
//! RANGE(end, axis_id, LOCAL)  → lidxN (SPECIAL with local thread index)
//! ```
//!
//! Dimension limiting is applied to fit within hardware constraints:
//! - Grouping: Merge adjacent dimensions that fit within limits
//! - Splitting: Factor dimensions that exceed limits

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::types::{AxisType, ConstValue};
use svod_ir::{Op, UOp, UOpKey};

use crate::optimizer::Renderer;
use crate::pattern::TypedPatternMatcher;

/// Pattern matcher for GPU dimension injection.
///
/// Matches SINK operations and transforms GLOBAL/LOCAL ranges to SPECIAL ops.
/// Must run after pm_reduce and before pm_add_loads.
///
/// # Context
///
/// Requires `&Renderer` context to access device limits (global_max, local_max).
pub fn pm_add_gpudims() -> TypedPatternMatcher<Renderer> {
    crate::patterns! {
        @context Renderer;
        // Match SINK with at least one source
        sink @ Sink { sources: _sources } => |sink| add_gpudims(ctx, sink),
    }
}

/// Main transformation: inject GPU dimensions into SINK.
///
/// Follows Tinygrad's `add_gpudims` function (gpudims.py:59-103):
/// 1. Collect all RANGE operations from topology
/// 2. Check for existing SPECIAL ops (skip if found)
/// 3. Categorize ranges by axis type (GLOBAL/THREAD vs LOCAL/WARP/GROUP_REDUCE)
/// 4. Create SPECIAL indices with dimension limiting
/// 5. Substitute RANGE ops with computed indices
#[allow(clippy::mutable_key_type)]
fn add_gpudims(ctx: &Renderer, sink: &Arc<UOp>) -> Option<Arc<UOp>> {
    let Op::Sink { .. } = sink.op() else {
        return None;
    };

    // Collect topology (all UOps reachable from sink)
    let topo = sink.toposort();

    // Check for existing SPECIAL ops - if found, gpudims already applied
    if topo.iter().any(|u| matches!(u.op(), Op::Special { .. })) {
        return None;
    }

    // Collect all RANGE operations, keyed by (axis_id, axis_type)
    // We exclude axis_type from the key matching for categorization, but track it
    let mut all_ranges: HashMap<(usize, AxisType), Arc<UOp>> = HashMap::new();
    for u in &topo {
        if let Op::Range { axis_id, axis_type, .. } = u.op() {
            all_ranges.insert((axis_id.value(), *axis_type), u.clone());
        }
    }

    if all_ranges.is_empty() {
        return None;
    }

    // Categorize ranges by axis type
    // Global dims: GLOBAL, THREAD
    // Local dims: LOCAL, WARP, GROUP_REDUCE
    let mut global_dims: Vec<(usize, AxisType)> = Vec::new();
    let mut local_dims: Vec<(usize, AxisType)> = Vec::new();

    for (axis_id, axis_type) in all_ranges.keys() {
        match axis_type {
            AxisType::Global | AxisType::Thread if !global_dims.iter().any(|(id, _)| *id == *axis_id) => {
                global_dims.push((*axis_id, *axis_type));
            }
            AxisType::Local | AxisType::Warp | AxisType::GroupReduce
                if !local_dims.iter().any(|(id, _)| *id == *axis_id) =>
            {
                local_dims.push((*axis_id, *axis_type));
            }
            _ => {}
        }
    }

    // Sort by axis_id for consistent ordering
    global_dims.sort_by_key(|(id, _)| *id);
    local_dims.sort_by_key(|(id, _)| *id);

    // No GPU dimensions to inject
    if global_dims.is_empty() && local_dims.is_empty() {
        return None;
    }

    // Extract shapes from RANGE operations (the end values)
    let get_ranges_for_dims = |dims: &[(usize, AxisType)]| -> Vec<Arc<UOp>> {
        dims.iter().filter_map(|(axis_id, axis_type)| all_ranges.get(&(*axis_id, *axis_type))).cloned().collect()
    };

    let global_ranges = get_ranges_for_dims(&global_dims);
    let local_ranges = get_ranges_for_dims(&local_dims);

    // Extract dimension sizes from ranges
    let extract_shape = |ranges: &[Arc<UOp>]| -> Vec<Arc<UOp>> {
        ranges
            .iter()
            .filter_map(|r| match r.op() {
                Op::Range { end, .. } => Some(end.clone()),
                _ => None,
            })
            .collect()
    };

    let global_shape = extract_shape(&global_ranges);
    let local_shape = extract_shape(&local_ranges);

    let (all_idxs, local_idxs_for_masks): (Vec<Arc<UOp>>, Vec<Arc<UOp>>) = if ctx.has_threads {
        // CPU threading expects exactly one global dim and no local dims; the
        // optimizer ensures this. If a beam candidate produces something else,
        // we have to bail (the pattern-matcher framework returns Option, not
        // Result), but make the violation loud — in debug builds this fires
        // a panic so tests catch malformed candidates, and in release a
        // tracing::warn leaves a breadcrumb instead of a silent no-op that
        // would leave extra RANGEs un-substituted for the renderer.
        if global_dims.len() != 1 || !local_dims.is_empty() {
            tracing::warn!(
                global_dims = global_dims.len(),
                local_dims = local_dims.len(),
                "pm_add_gpudims: has_threads contract violated (expected 1 global / 0 local); skipping rewrite"
            );
            debug_assert!(
                global_dims.len() == 1 && local_dims.is_empty(),
                "pm_add_gpudims: has_threads requires exactly one global dim and no locals; got {} global, {} local",
                global_dims.len(),
                local_dims.len(),
            );
            return None;
        }
        let end = global_shape.first()?;
        let end = match end.op() {
            Op::Const(c) => const_to_i64(&c.0)?,
            other => {
                tracing::warn!(
                    op = ?other,
                    "pm_add_gpudims: has_threads requires Const global end; skipping rewrite"
                );
                debug_assert!(false, "pm_add_gpudims: has_threads requires Const global end, got {other:?}");
                return None;
            }
        };
        if end <= 0 {
            tracing::warn!(end, "pm_add_gpudims: has_threads requires positive global end; skipping rewrite");
            debug_assert!(end > 0, "pm_add_gpudims: has_threads requires positive global end, got {end}");
            return None;
        }
        (vec![UOp::define_var("core_id".to_string(), 0, end - 1)], Vec::new())
    } else {
        // Generate GPU indices
        let global_max = ctx.global_max.as_deref();
        let local_max_product = ctx.local_max;

        // For locals, we use product limit rather than per-dimension
        // Convert to per-dimension limits if needed
        let local_max: Option<Vec<usize>> = local_max_product.map(|max| {
            // Simple heuristic: distribute limit evenly if multiple dimensions
            let n = local_shape.len().max(1);
            let per_dim = (max as f64).powf(1.0 / n as f64).floor() as usize;
            vec![per_dim.max(1); n]
        });
        let local_max_slice = local_max.as_deref();

        // Create global indices (gidx0, gidx1, ...)
        let global_idxs = get_grouped_dims("gidx", &global_shape, global_max, true);
        // Create local indices (lidx0, lidx1, ...)
        let local_idxs = get_grouped_dims("lidx", &local_shape, local_max_slice, false);
        let local_idxs_for_masks = local_idxs.clone();

        // Combine indices in order: global, then local
        (global_idxs.into_iter().chain(local_idxs).collect(), local_idxs_for_masks)
    };

    // Build substitution map: RANGE -> corresponding index
    let mut subs: HashMap<UOpKey, Arc<UOp>> = HashMap::new();
    let all_dims: Vec<(usize, AxisType)> = global_dims.iter().chain(local_dims.iter()).cloned().collect();

    for (i, (axis_id, axis_type)) in all_dims.iter().enumerate() {
        if *axis_type == AxisType::Reduce {
            // Don't replace reduce axes (they stay as loops)
            continue;
        }
        if let Some(range_uop) = all_ranges.get(&(*axis_id, *axis_type))
            && i < all_idxs.len()
        {
            subs.insert(UOpKey(range_uop.clone()), all_idxs[i].clone());
        }
    }

    // Handle STORE masking for global stores with missing local indices
    // When a STORE to global memory doesn't use all local indices,
    // we need to mask the store to only execute when unused local indices are 0
    let store_subs = compute_store_masks(&topo, &all_ranges, &local_dims, &local_idxs_for_masks);
    for (id, masked_idx) in store_subs {
        subs.insert(id, masked_idx);
    }

    // Apply substitutions to rebuild the sink
    if subs.is_empty() {
        return None;
    }

    Some(sink.substitute(&subs))
}

/// Compute store masks for global stores with missing local indices.
///
/// Based on Tinygrad's gpudims.py:86-96.
/// When a STORE to global memory doesn't use all local indices,
/// we add a mask so the store only executes when missing locals are 0.
#[allow(clippy::mutable_key_type)]
fn compute_store_masks(
    topo: &[Arc<UOp>],
    all_ranges: &HashMap<(usize, AxisType), Arc<UOp>>,
    local_dims: &[(usize, AxisType)],
    local_idxs: &[Arc<UOp>],
) -> HashMap<UOpKey, Arc<UOp>> {
    let mut masks: HashMap<UOpKey, Arc<UOp>> = HashMap::new();

    for uop in topo {
        let Op::Store { index, .. } = uop.op() else {
            continue;
        };

        // Check if store targets global memory
        // In Svod, we check if the INDEX's buffer has Global addrspace
        let is_global_store = match index.op() {
            Op::Index { buffer, .. } => match buffer.dtype() {
                DType::Ptr { addrspace, .. } => addrspace == svod_dtype::AddrSpace::Global,
                _ => true, // Assume global if not a pointer type
            },
            _ => continue,
        };

        if !is_global_store {
            continue;
        }

        // Find local ranges NOT used in the index computation.
        // Use in_scope_ranges() to get only active (not ended) ranges,
        // rather than toposort().filter(Range) which returns ALL ranges in the graph.
        let index_ranges: HashSet<u64> = index.in_scope_ranges().iter().map(|key| key.0.id).collect();

        let mut missing_locals: Vec<Arc<UOp>> = Vec::new();
        for (i, (axis_id, axis_type)) in local_dims.iter().enumerate() {
            if let Some(range_uop) = all_ranges.get(&(*axis_id, *axis_type))
                && !index_ranges.contains(&range_uop.id)
                && i < local_idxs.len()
            {
                missing_locals.push(local_idxs[i].clone());
            }
        }

        if missing_locals.is_empty() {
            continue;
        }

        // Create mask: (missing_local_1 == 0) & (missing_local_2 == 0) & ...
        // Using eq() and and_() panicking wrappers for cleaner code
        let zero = UOp::index_const(0);
        let mut mask: Option<Arc<UOp>> = None;
        for local_idx in missing_locals {
            let eq_zero = local_idx.eq(&zero);
            mask = Some(match mask {
                None => eq_zero,
                Some(m) => m.and_(&eq_zero),
            });
        }

        // Add gate to INDEX if mask exists
        if let (Some(mask), Op::Index { buffer, indices, gate }) = (mask, index.op()) {
            let new_gate = match gate {
                Some(existing) => existing.and_(&mask),
                None => mask,
            };
            // Use INDEX builder pattern
            let new_index = UOp::index()
                .buffer(buffer.clone())
                .indices(indices.clone())
                .gate(new_gate)
                .call()
                .expect("gpudims: INDEX gate construction failed");
            masks.insert(UOpKey(index.clone()), new_index);
        }
    }

    masks
}

/// Extract i64 value from ConstValue.
fn const_to_i64(cv: &ConstValue) -> Option<i64> {
    match cv {
        ConstValue::Int(v) => Some(*v),
        ConstValue::UInt(v) => Some(*v as i64),
        ConstValue::Bool(v) => Some(*v as i64),
        ConstValue::Float(v) => Some(*v as i64),
    }
}

/// Tinygrad's `_dim_max(d: sint) -> int` (gpudims.py:7): concrete int passes
/// through, symbolic UOp returns its `vmax` upper bound. Used uniformly across
/// grouping/splitting so concrete and symbolic dims go through one code path.
fn dim_max(d: &Arc<UOp>) -> usize {
    const_to_i64(d.vmax()).map(|v| v.max(0) as usize).unwrap_or(usize::MAX)
}

/// True when `a` and `b` are structurally identical (hash-cons identity).
fn dims_eq(a: &[Arc<UOp>], b: &[Arc<UOp>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| Arc::ptr_eq(x, y))
}

/// True when `u` is the concrete CONST integer 1 (for matching tinygrad's
/// `acc != 1` leading-1 special case in `get_contraction`).
fn is_one(u: &Arc<UOp>) -> bool {
    matches!(u.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(1)))
}

/// Create GPU thread indices with dimension limiting.
///
/// Mirrors Tinygrad's `get_grouped_dims` (gpudims.py:28-56). Operates on
/// `sint` dims (concrete `Int` const or symbolic `UOp`) end-to-end via
/// [`dim_max`]; grouping/splitting always returns a fresh `Vec<Arc<UOp>>` so
/// downstream `decompose`/`combine`/`flatten_unflatten` can index into it
/// regardless of whether the input was numeric or symbolic.
///
/// # Arguments
///
/// * `prefix` - Index name prefix ("gidx" or "lidx")
/// * `dims` - Dimension sizes as UOps
/// * `max_sizes` - Hardware limits per dimension (None = unlimited)
/// * `reverse` - Reverse dimension ordering (true for global indices)
///
/// # Returns
///
/// Vector of SPECIAL UOps (plus mod/idiv decomposition where contraction was
/// applied) representing thread indices, one per original `dims` entry.
fn get_grouped_dims(prefix: &str, dims: &[Arc<UOp>], max_sizes: Option<&[usize]>, reverse: bool) -> Vec<Arc<UOp>> {
    // Tinygrad-equivalent (`codegen/gpudims.py:29`): when `reverse=True`,
    // recursively call with reversed dims, then reverse the result. Reversing
    // only the OUTPUT array leaves the SPECIAL UOps named in iteration order
    // while the indices land at swapped positions — manifests as a 21× OOB on
    // matmul+reduce kernels where g_x and g_y are picked for different range
    // axes.
    if reverse {
        let reversed: Vec<Arc<UOp>> = dims.iter().cloned().rev().collect();
        let result = get_grouped_dims(prefix, &reversed, max_sizes, false);
        return result.into_iter().rev().collect();
    }
    if dims.is_empty() {
        return vec![];
    }

    let limited: Vec<Arc<UOp>> = match max_sizes {
        None => dims.to_vec(),
        Some(max) => {
            // First try grouping: (a, b, c, d) → (a*b, c, d). Match tinygrad's
            // fail-fast behaviour (gpudims.py:33-37): if neither grouping nor
            // splitting can fit the dims into the backend's axis cap, panic
            // immediately. Returning unchanged dims and warning is what we
            // used to do, but it produced SPECIAL UOps with `gidx3`/`lidx3+`
            // that the AMD renderer rejects at codegen/src/llvm/amd/ops.rs;
            // the error surfaced at codegen time rather than at scheduling
            // time, which buries the actual problem (a bad scheduler/BEAM
            // candidate). Failing here makes the offending candidate visible.
            let grouped = group_dims(dims, max);
            let after_group = grouped.unwrap_or_else(|| dims.to_vec());
            if after_group.len() > max.len() {
                panic!(
                    "get_grouped_dims: cannot limit dims to {} axes (dims={:?}, max_sizes={:?}); \
                     scheduler emitted more SPECIAL axes than the backend supports",
                    max.len(),
                    dims.iter().map(dim_max).collect::<Vec<_>>(),
                    max,
                );
            }
            if dims_eq(&after_group, dims) {
                // No grouping happened (or every group attempt was a no-op):
                // try splitting up dims (a,) → (b, c).
                split_dims(dims, max).unwrap_or_else(|| {
                    panic!(
                        "get_grouped_dims: split_dims failed (likely non-factorable symbolic dim); \
                         dims={:?}, max_sizes={:?}",
                        dims.iter().map(dim_max).collect::<Vec<_>>(),
                        max,
                    )
                })
            } else {
                after_group
            }
        }
    };

    let raw_idxs: Vec<Arc<UOp>> = limited
        .iter()
        .enumerate()
        .map(|(i, s)| UOp::special(s.clone(), format!("{prefix}{i}")))
        .collect();

    if limited.len() < dims.len() {
        // Contraction: more original dims than limited dims — decompose via
        // divmod (mirrors gpudims.py:39-47).
        decompose_contracted_dims(&raw_idxs, dims, &limited)
    } else if limited.len() > dims.len() {
        // Expansion: more limited dims than original — combine via add/mul
        // (gpudims.py:48-50).
        combine_expanded_dims(&raw_idxs, &limited, dims)
    } else if !dims_eq(&limited, dims) {
        // Same count but different values: flatten then unflatten with new
        // strides (gpudims.py:51-55).
        flatten_unflatten_dims(&raw_idxs, &limited, dims)
    } else {
        raw_idxs
    }
}

/// Group adjacent dimensions to fit within hardware limits.
///
/// Mirrors Tinygrad's `_group_dims` (gpudims.py:9-16).
fn group_dims(dims: &[Arc<UOp>], max_sizes: &[usize]) -> Option<Vec<Arc<UOp>>> {
    let mut result: Vec<Arc<UOp>> = dims.to_vec();
    while result.len() > max_sizes.len() || result.iter().zip(max_sizes).any(|(d, m)| dim_max(d) > *m) {
        let mut grouped = false;
        for (i, &m) in max_sizes.iter().enumerate() {
            if i + 1 < result.len() && dim_max(&result[i]).saturating_mul(dim_max(&result[i + 1])) <= m {
                let merged = result[i].mul(&result[i + 1]);
                result = result[..i]
                    .iter()
                    .cloned()
                    .chain(std::iter::once(merged))
                    .chain(result[i + 2..].iter().cloned())
                    .collect();
                grouped = true;
                break;
            }
        }
        if !grouped {
            return None;
        }
    }
    Some(result)
}

/// Split dimensions that exceed hardware limits.
///
/// Mirrors Tinygrad's `_split_dims` (gpudims.py:18-26). Splitting requires a
/// concrete factor; if any dim that exceeds its limit is symbolic (no
/// `Op::Const` peer to read), the operation is unrepresentable and `None` is
/// returned (tinygrad raises in the same situation).
fn split_dims(dims: &[Arc<UOp>], max_sizes: &[usize]) -> Option<Vec<Arc<UOp>>> {
    if dims.iter().zip(max_sizes).all(|(d, m)| dim_max(d) <= *m) {
        return Some(dims.to_vec());
    }
    let mut working: Vec<Arc<UOp>> = dims.to_vec();
    while working.len() < 3 {
        working.push(UOp::index_const(1));
    }
    for i in 0..3 {
        let m = max_sizes.get(i).copied().unwrap_or(usize::MAX);
        while dim_max(&working[i]) > m {
            let Op::Const(c) = working[i].op() else {
                return None;
            };
            let val = const_to_i64(&c.0)? as usize;
            let div = find_smallest_divisor(val);
            if div == 1 {
                return None;
            }
            let div_uop = UOp::index_const(div as i64);
            let next = (i + 1) % 3;
            working[i] = working[i].idiv(&div_uop);
            working[next] = working[next].mul(&div_uop);
        }
    }
    let result = if is_one(&working[2]) {
        if is_one(&working[1]) {
            vec![working[0].clone()]
        } else {
            vec![working[0].clone(), working[1].clone()]
        }
    } else {
        working
    };
    Some(result)
}

/// Find the smallest divisor of n (excluding 1).
fn find_smallest_divisor(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let sqrt_n = (n as f64).sqrt().ceil() as usize;
    for d in 2..=sqrt_n {
        if n.is_multiple_of(d) {
            return d;
        }
    }
    1 // n is prime
}

/// Decompose contracted dimensions back to original count.
///
/// Mirrors Tinygrad's gpudims.py:39-47. For each SPECIAL index whose limited
/// dim grouped several original dims together, peel them back via repeated
/// `current % dims[c]; current //= dims[c]`.
fn decompose_contracted_dims(
    raw_idxs: &[Arc<UOp>],
    original_dims: &[Arc<UOp>],
    limited_dims: &[Arc<UOp>],
) -> Vec<Arc<UOp>> {
    let contraction = match get_contraction(original_dims, limited_dims) {
        Some(c) => c,
        None => return raw_idxs.to_vec(),
    };
    let mut result: Vec<Arc<UOp>> = Vec::new();
    for (idx, group) in raw_idxs.iter().zip(&contraction) {
        if group.is_empty() {
            // Leading-1 contraction group (`acc != 1` branch in get_contraction
            // produced an empty span); the SPECIAL has no original dim to
            // emit, so it contributes nothing — skip.
            continue;
        }
        let mut current = idx.clone();
        for &c in &group[..group.len() - 1] {
            let d = &original_dims[c];
            result.push(current.mod_(d));
            current = current.idiv(d);
        }
        result.push(current);
    }
    result
}

/// Get contraction mapping: which original dims map to each limited dim.
///
/// Mirrors Tinygrad's `get_contraction` (helpers.py:121-125) with `T = sint`.
/// Accumulated products are built via `UOp::mul`; hash-consing makes equal
/// UOp expressions share an `Arc`, so [`Arc::ptr_eq`] is the correct equality
/// for matching positions — same shape contract as the python implementation
/// where sint `__eq__` collapses to identity for symbolic and value for int.
///
/// # Example
///
/// ```text
/// original = [2, 5, 2], limited = [10, 2]
/// acc_old = [2, 10, 20]
/// acc_new = [10, 20]
/// split = [2, 3]
/// result = [[0, 1], [2]]
/// ```
fn get_contraction(original_dims: &[Arc<UOp>], limited_dims: &[Arc<UOp>]) -> Option<Vec<Vec<usize>>> {
    if original_dims.is_empty() && limited_dims.is_empty() {
        return Some(vec![]);
    }
    if limited_dims.is_empty() {
        return None;
    }
    // Skip the synthetic leading `1` from `itertools.accumulate`'s initial: hash
    // consing only collapses identical sub-trees, and `Mul(1, x)` is not folded
    // at construction time. Seeding with the first dim instead makes the
    // accumulated products from grouping (`result[i].mul(&result[i+1])`) share
    // an Arc with the matching slice of `acc_old`.
    let scan_mul = |dims: &[Arc<UOp>]| -> Vec<Arc<UOp>> {
        let mut out: Vec<Arc<UOp>> = Vec::with_capacity(dims.len());
        for (i, d) in dims.iter().enumerate() {
            let acc = if i == 0 { d.clone() } else { out[i - 1].mul(d) };
            out.push(acc);
        }
        out
    };
    let acc_old = scan_mul(original_dims);
    let acc_new = scan_mul(limited_dims);
    let mut split = Vec::with_capacity(acc_new.len());
    for acc in &acc_new {
        if is_one(acc) {
            split.push(0);
        } else {
            match acc_old.iter().position(|o| Arc::ptr_eq(o, acc)) {
                Some(idx) => split.push(idx + 1),
                None => return None,
            }
        }
    }
    let mut result = Vec::with_capacity(split.len());
    let mut prev = 0;
    for (i, &s) in split.iter().enumerate() {
        if i == split.len() - 1 {
            result.push((prev..original_dims.len()).collect());
        } else {
            result.push((prev..s).collect());
            prev = s;
        }
    }
    Some(result)
}

/// Combine expanded dimensions to match original count (gpudims.py:48-50).
fn combine_expanded_dims(
    raw_idxs: &[Arc<UOp>],
    limited_dims: &[Arc<UOp>],
    original_dims: &[Arc<UOp>],
) -> Vec<Arc<UOp>> {
    match (limited_dims.len(), original_dims.len()) {
        (2, 1) => vec![raw_idxs[0].mul(&limited_dims[1]).add(&raw_idxs[1])],
        (3, 1) => {
            let inner = raw_idxs[0].mul(&limited_dims[1]).add(&raw_idxs[1]);
            vec![inner.mul(&limited_dims[2]).add(&raw_idxs[2])]
        }
        _ => flatten_unflatten_dims(raw_idxs, limited_dims, original_dims),
    }
}

/// Flatten and unflatten when dims have same count but different values
/// (gpudims.py:51-55).
fn flatten_unflatten_dims(
    raw_idxs: &[Arc<UOp>],
    limited_dims: &[Arc<UOp>],
    original_dims: &[Arc<UOp>],
) -> Vec<Arc<UOp>> {
    let flat = match limited_dims.len() {
        2 => raw_idxs[0].mul(&limited_dims[1]).add(&raw_idxs[1]),
        3 => {
            let l12 = limited_dims[1].mul(&limited_dims[2]);
            let t0 = raw_idxs[0].mul(&l12);
            let t1 = raw_idxs[1].mul(&limited_dims[2]);
            t0.add(&t1).add(&raw_idxs[2])
        }
        _ => return raw_idxs.to_vec(),
    };
    match original_dims.len() {
        2 => vec![flat.idiv(&original_dims[1]), flat.mod_(&original_dims[1])],
        3 => {
            let d12 = original_dims[2].mul(&original_dims[1]);
            vec![
                flat.idiv(&d12),
                flat.idiv(&original_dims[2]).mod_(&original_dims[1]),
                flat.mod_(&original_dims[2]),
            ]
        }
        _ => raw_idxs.to_vec(),
    }
}

#[cfg(test)]
#[path = "test/unit/gpudims_internal.rs"]
mod tests;
