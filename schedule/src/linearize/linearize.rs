//! Priority-aware topological sort for linearization.
//!
//! Converts a UOp DAG into a linear instruction sequence suitable for
//! GPU/NPU backends that require sequential instruction streams.

use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use svod_ir::UOp;
use svod_ir::op::Op;
use svod_ir::types::ConstValue;
use svod_ir::uop::core::UOpKey;

/// Priority values for different operation types.
///
/// Lower values = higher priority (scheduled earlier).
/// Based on Tinygrad's linearizer priority assignments.
mod priority {
    pub const PARAM: i32 = -20;
    pub const DEFINE_VAR: i32 = -19;
    pub const DEFINE_LOCAL: i32 = -18;
    pub const DEFINE_REG: i32 = -17;
    pub const END: i32 = -5;
    pub const LOAD: i32 = -1;
    pub const DEFAULT: i32 = 0;
    pub const STORE: i32 = 1;
    pub const RANGE: i32 = 5;
}

/// Ordering key for heap-based scheduling.
///
/// Tuple ordering: (run_count, priority, arg_value, ideal_position, id)
/// - run_count: Higher counts scheduled later (executed in inner loops)
/// - priority: Lower values scheduled earlier
/// - arg_value: For PARAM, slot index for consistent ordering
/// - ideal_position: Position in priority-sorted order
/// - id: UOp ID for tie-breaking (ensures stable ordering)
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct OrderKey {
    run_count: u64,
    priority: i32,
    arg_value: Option<i64>,
    ideal_pos: usize,
    id: u64,
}

/// Convert a UOp DAG into a linear instruction sequence.
///
/// Uses priority-aware topological sorting to produce an optimal
/// instruction order for GPU/NPU execution.
///
/// # Algorithm
///
/// 1. Toposort all nodes from sink
/// 2. Build consumer graph and compute priorities (in REVERSE order!)
/// 3. Create ideal ordering based on priorities
/// 4. Use heap-based linearization respecting data dependencies
/// 5. Reverse result (we build backwards from sink)
///
/// # Priority Assignment
///
/// | Op | Priority | Purpose |
/// |----|----------|---------|
/// | Param | -20 | Function arguments first |
/// | DefineVar | -19 | Symbolic variables early |
/// | DefineLocal | -18 | Local memory early |
/// | DefineReg | -17 | Register definitions early |
/// | End | -5 | Close loops promptly |
/// | Const | 0 | Inlined at use site |
/// | Load | -1 | Loads before compute |
/// | (default) | 0 | Neutral |
/// | Store | 1 | Stores after compute |
/// | Range | 5 | Loop starts late |
///
/// # Example
///
/// ```ignore
/// use svod_schedule::linearize::linearize;
///
/// let kernel_ast = /* ... */;
/// let instructions = linearize(kernel_ast);
///
/// // instructions is now a Vec<Arc<UOp>> in execution order
/// for (i, instr) in instructions.iter().enumerate() {
///     println!("{}: {:?}", i, instr.op());
/// }
/// ```
pub fn linearize(sink: Arc<UOp>) -> Vec<Arc<UOp>> {
    // Step 1: Toposort from sink
    let nodes = sink.toposort();

    if nodes.is_empty() {
        return vec![sink];
    }

    // Step 2: Build consumer graph + priorities
    // CRITICAL: Must iterate in REVERSE order for correct consumer counting
    #[allow(clippy::mutable_key_type)]
    let mut consumers: HashMap<UOpKey, Vec<Arc<UOp>>> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut out_degree: HashMap<UOpKey, usize> = HashMap::new();
    #[allow(clippy::mutable_key_type)]
    let mut priorities: HashMap<UOpKey, OrderKey> = HashMap::new();
    // Map from UOp ID to Arc<UOp> for lookup
    let mut id_to_uop: HashMap<u64, Arc<UOp>> = HashMap::new();

    for u in nodes.iter().rev() {
        id_to_uop.insert(u.id, u.clone());

        // Build consumer graph
        for src in u.op().sources() {
            consumers.entry(UOpKey(src.clone())).or_default().push(u.clone());
        }

        // Compute run count from ranges
        let run_count = compute_run_count(u);

        // Assign priority based on operation type
        let (base_priority, arg_value) = get_priority(u);

        priorities.insert(
            UOpKey(u.clone()),
            OrderKey { run_count, priority: base_priority, arg_value, ideal_pos: 0, id: u.id },
        );
    }

    // Initialize out_degree (number of consumers)
    for node in &nodes {
        let key = UOpKey(node.clone());
        let degree = consumers.get(&key).map_or(0, |c| c.len());
        out_degree.insert(key, degree);
    }

    // Step 3: Create ideal ordering sorted by priority
    let mut sorted: Vec<_> = nodes.to_vec();
    sorted.sort_by_key(|u| {
        priorities.get(&UOpKey(u.clone())).cloned().unwrap_or(OrderKey {
            run_count: 0,
            priority: priority::DEFAULT,
            arg_value: None,
            ideal_pos: 0,
            id: u.id,
        })
    });

    // Assign ideal positions
    // Use reversed position so that nodes earlier in sorted order have larger ideal_pos.
    // Since BinaryHeap is a max-heap, larger values are popped first,
    // ensuring earlier nodes are processed first (consistent with sorted order).
    #[allow(clippy::mutable_key_type)]
    let nkey: HashMap<UOpKey, usize> =
        sorted.iter().enumerate().map(|(i, u)| (UOpKey(u.clone()), sorted.len() - 1 - i)).collect();

    // Update priorities with ideal positions
    for (key, pos) in &nkey {
        if let Some(order_key) = priorities.get_mut(key) {
            order_key.ideal_pos = *pos;
        }
    }

    // Step 4: Heap-based linearization
    // Use MAX-heap: larger OrderKey (worse priority) popped first.
    // After reversal, better priority nodes appear earlier in output.
    // This matches Tinygrad's use of -nkey in a min-heap.
    let mut heap: BinaryHeap<OrderKey> = BinaryHeap::new();

    let sink_key = priorities.get(&UOpKey(sink.clone())).cloned().unwrap_or(OrderKey {
        run_count: 0,
        priority: priority::DEFAULT,
        arg_value: None,
        ideal_pos: 0,
        id: sink.id,
    });
    heap.push(sink_key);

    let mut result = Vec::with_capacity(nodes.len());
    let mut visited: std::collections::HashSet<u64> = std::collections::HashSet::new();

    while let Some(order_key) = heap.pop() {
        let u_id = order_key.id;

        // Skip if already processed (can happen with diamond dependencies)
        if visited.contains(&u_id) {
            continue;
        }
        visited.insert(u_id);

        // Look up the UOp
        let u = match id_to_uop.get(&u_id) {
            Some(uop) => uop.clone(),
            None => continue,
        };

        result.push(u.clone());

        // Decrement out_degree for all sources
        for src in u.op().sources() {
            let src_key = UOpKey(src.clone());
            if let Some(deg) = out_degree.get_mut(&src_key) {
                *deg = deg.saturating_sub(1);
                if *deg == 0 && !visited.contains(&src.id) {
                    // All consumers processed, add to heap
                    if let Some(src_order_key) = priorities.get(&src_key) {
                        heap.push(src_order_key.clone());
                    }
                }
            }
        }
    }

    // Step 5: Reverse result (we built backwards from sink)
    result.reverse();

    // Step 6: Repair cross-block dominance for relocated pure scalars.
    //
    // The heap places a pure value at its earliest-program consumer's scope, which
    // in a multi-loop kernel may be *inside* one (sibling) loop while another
    // sibling loop also references it — the linear renderer then emits it into that
    // loop's basic block and it fails to dominate the sibling's uses (LLVM "does
    // not dominate all uses"). [`relocate_for_dominance`] moves ONLY such broken
    // nodes, and only up to the deepest loop scope that dominates all their uses
    // (a no-op for a node whose placement already dominates). Unlike the former
    // unconditional hoist of *every* range-independent node to the entry block, it
    // does not lengthen the live ranges of correctly-placed values, so the default
    // optimizer path keeps its register pressure.
    relocate_for_dominance(result)
}

/// Minimal dominance-preserving relocation (replaces the former unconditional
/// loop-invariant hoist). The linear renderer ([`crate::...llvm::text`]) emits one
/// basic block per RANGE/END, so a value lands in whatever loop body it is
/// scheduled into. A value is only *broken* if it is scheduled inside a loop it is
/// not in-scope of (a sibling sub-loop) yet consumed by an op outside that loop —
/// LLVM then reports "Instruction does not dominate all uses". This happens for
/// pure range-independent index subexpressions shared between sibling loops in
/// hand-built tile kernels (the svod-tk flash-attention masking/fill loops).
///
/// Unlike the old `hoist_loop_invariant`/`hoist_to_home_scope` (which lifted
/// **every** pure range-independent node to the entry block — long live ranges →
/// VGPR spills, +37% on the default-optimizer path), this moves a node only when
/// its current scope does not dominate all of its consumers, and only as far up as
/// the deepest loop scope that *does* (its in-scope range set ⊇ the move target).
/// A node whose placement already dominates its uses is left untouched, so normal
/// kernels are unaffected.
fn relocate_for_dominance(list: Vec<Arc<UOp>>) -> Vec<Arc<UOp>> {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::InScopeRangesProperty;

    let n = list.len();
    let pos: HashMap<u64, usize> = list.iter().enumerate().map(|(i, u)| (u.id, i)).collect();

    // open_after[i] = the stack of RANGE ids open *immediately after* position i
    // (i.e. the basic-block scope a node emitted right after i would belong to).
    // A RANGE opens its scope at its own position; an END closes it before its.
    let mut open_after: Vec<Vec<u64>> = Vec::with_capacity(n);
    let mut open: Vec<u64> = Vec::new();
    for u in &list {
        for ended in u.op().ended_ranges() {
            if matches!(ended.op(), Op::Range { .. })
                && let Some(p) = open.iter().rposition(|r| *r == ended.id)
            {
                open.remove(p);
            }
        }
        if matches!(u.op(), Op::Range { .. }) {
            open.push(u.id);
        }
        open_after.push(open.clone());
    }
    // The scope a node currently sits in = the open stack just *before* it (its
    // emit slot). open_before[i] = open stack as of position i, pre-RANGE-push.
    let open_before = |i: usize| -> Vec<u64> { if i == 0 { Vec::new() } else { open_after[i - 1].clone() } };

    // Consumer map (forward).
    let mut consumers: HashMap<u64, Vec<u64>> = HashMap::new();
    for u in &list {
        for s in u.op().sources() {
            consumers.entry(s.id).or_default().push(u.id);
        }
    }

    // A relocatable node is a pure scalar value op (no side effects, no loop role).
    let pure = |u: &Arc<UOp>| {
        matches!(u.op(), Op::Special { .. } | Op::Binary(..) | Op::Unary(..) | Op::Cast { .. } | Op::BitCast { .. })
    };

    // target[i] = the position to re-emit node i just *before* (its new slot), if
    // it must move. We move a node to land in the scope = intersection of (its own
    // in-scope range set) and (every consumer's open-loop scope) — the deepest loop
    // that is an ancestor of all uses. We realize that by inserting the node right
    // before the earliest consumer whose scope is an ancestor chain prefix; in
    // practice the entry block (move before the first RANGE) is the safe upper
    // bound for a range-independent node, and the innermost dominating loop for a
    // range-dependent one. We compute the precise insert marker below.
    let mut target: Vec<Option<usize>> = vec![None; n];

    for (i, u) in list.iter().enumerate() {
        if !pure(u) {
            continue;
        }
        let cur_scope = open_before(i);
        if cur_scope.is_empty() {
            continue; // already in the entry block — dominates everything
        }
        #[allow(clippy::mutable_key_type)]
        let own = InScopeRangesProperty::get(u);
        let own_has = |rid: u64| own.iter().any(|k| k.0.id == rid);

        // Find consumers not dominated by the current scope: a consumer is
        // dominated iff every loop currently open around the def is also open
        // around the consumer (the def's block is an ancestor of the use's block).
        let Some(cs) = consumers.get(&u.id) else { continue };
        let mut broken = false;
        for &c in cs {
            let Some(&cp) = pos.get(&c) else { continue };
            let use_scope = open_before(cp);
            if !cur_scope.iter().all(|r| use_scope.contains(r)) {
                broken = true;
                break;
            }
        }
        if !broken {
            continue;
        }

        // The node must move up to the deepest loop scope that is an ancestor of
        // ALL its consumers AND that it is in-scope of (so we never move it below a
        // range it depends on). Compute the surviving ancestor prefix of cur_scope.
        let mut keep_depth = cur_scope.len();
        for &c in cs {
            let Some(&cp) = pos.get(&c) else { continue };
            let use_scope = open_before(cp);
            // longest common prefix of cur_scope and use_scope, restricted to
            // ranges the node is genuinely in-scope of.
            let mut d = 0;
            while d < cur_scope.len() && d < use_scope.len() && cur_scope[d] == use_scope[d] && own_has(cur_scope[d]) {
                d += 1;
            }
            keep_depth = keep_depth.min(d);
        }
        // Re-emit just before the RANGE that opens scope level `keep_depth` (the
        // first loop the node must escape). If keep_depth == 0 that is the entry
        // block (before the first enclosing RANGE).
        let escape_range_id = cur_scope[keep_depth];
        if let Some(&marker) = pos.get(&escape_range_id) {
            target[i] = Some(marker);
        }
    }

    if target.iter().all(Option::is_none) {
        return list;
    }

    // SSA repair: moving node `i` before marker `m` strands any *pure* source `s`
    // of `i` that currently sits at a position `>= m` (it would be referenced
    // before its own emission). Pull such sources up to the same marker (or an
    // earlier one if they already move), iterating to a fixpoint over the topo
    // order. A non-pure source (a LOAD/etc. with loop role) cannot be after `m`:
    // `keep_depth` only escapes ranges the node is NOT in-scope of, and a non-pure
    // source the node depends on shares those in-scope ranges, so it is emitted in
    // a scope the marker does not escape — i.e. before `m`.
    loop {
        let mut changed = false;
        // Walk in reverse topo order so a consumer's target propagates to sources
        // in a single sweep where possible.
        for i in (0..n).rev() {
            let Some(m) = target[i] else { continue };
            for s in list[i].op().sources() {
                if !pure(&s) {
                    continue;
                }
                let Some(&sp) = pos.get(&s.id) else { continue };
                if sp < m {
                    continue; // source already emitted before the marker
                }
                let new_m = match target[sp] {
                    Some(existing) => existing.min(m),
                    None => m,
                };
                if target[sp] != Some(new_m) {
                    target[sp] = Some(new_m);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Re-emit each relocated node just before its target RANGE marker, preserving
    // topo order among nodes moved to the same marker (so a moved source still
    // precedes a moved consumer sharing the marker).
    let mut moved_before: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, t) in target.iter().enumerate() {
        if let Some(m) = t {
            moved_before.entry(*m).or_default().push(i);
        }
    }
    let mut out = Vec::with_capacity(n);
    for (i, u) in list.iter().enumerate() {
        if let Some(idxs) = moved_before.get(&i) {
            for &j in idxs {
                out.push(list[j].clone());
            }
        }
        if target[i].is_none() {
            out.push(u.clone());
        }
    }
    out
}

/// Compute the "run count" for a UOp based on its IN-SCOPE ranges.
///
/// The run count estimates how many times this operation executes,
/// based on the loop bounds of enclosing ranges that are CURRENTLY ACTIVE.
///
/// Thread ranges are EXCLUDED because they're pseudo-loops for codegen
/// structure, not actual loops. Instructions that depend on core_id
/// should still be placed in the entry block.
///
/// CFG predecessors are propagated via the `deps` field on `Op::Range`,
/// which makes `InScopeRangesProperty` accumulate parent loop ranges
/// naturally through `children()`. This matches Tinygrad's
/// `pm_add_control_flow` behavior.
///
/// This matches Tinygrad's linearizer where `run_count = prod([int(r.vmax)+1 for r in u.ranges])`
/// and `u.ranges` returns only ranges that haven't been ended yet at that point.
fn compute_run_count(uop: &Arc<UOp>) -> u64 {
    use svod_ir::uop::cached_property::CachedProperty;
    use svod_ir::uop::properties::InScopeRangesProperty;

    #[allow(clippy::mutable_key_type)]
    let in_scope = InScopeRangesProperty::get(uop);

    if in_scope.is_empty() {
        return 1;
    }

    // Tinygrad: run_count = prod([int(r.vmax)+1 for r in u.ranges])
    // ALL ranges contribute, including Thread ranges. No filtering.
    in_scope
        .iter()
        .map(|key| match key.0.vmax() {
            ConstValue::Int(v) => (v + 1) as u64,
            ConstValue::UInt(v) => v + 1,
            _ => 1,
        })
        .product()
}

/// Get priority and optional argument value for a UOp.
///
/// Note: Tinygrad uses `u.arg` for DEFINE_VAR ordering (the name tuple).
/// Svod uses `id` for tie-breaking since `arg_value` is numeric.
/// This gives deterministic ordering but not alphabetical by name.
fn get_priority(uop: &Arc<UOp>) -> (i32, Option<i64>) {
    match uop.op() {
        Op::Param { slot, device: None, .. } => (priority::PARAM, Some(*slot as i64)),
        Op::DefineVar { name, .. } => {
            // Use hash of name for stable ordering (Tinygrad: uses arg tuple for comparison)
            // This ensures consistent ordering across runs while approximating name-based sorting
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            name.hash(&mut hasher);
            (priority::DEFINE_VAR, Some(hasher.finish() as i64))
        }
        Op::DefineLocal(_) => (priority::DEFINE_LOCAL, None),
        Op::DefineReg { .. } => (priority::DEFINE_REG, None),
        Op::Const(_) | Op::VConst { .. } => (priority::DEFAULT, None),
        Op::End { .. } => (priority::END, None),
        Op::Load { .. } => (priority::LOAD, None),
        Op::Store { .. } => (priority::STORE, None),
        Op::Range { .. } => (priority::RANGE, None),
        _ => (priority::DEFAULT, None),
    }
}

#[cfg(test)]
#[path = "../test/unit/linearize/linearize_internal.rs"]
mod tests;
