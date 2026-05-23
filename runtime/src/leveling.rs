//! Shared topological leveling for runtime ops and schedule items.
//!
//! Single source of truth for the dependency-order and Kahn wave-leveling
//! algorithms used by BOTH the runtime executor ([`crate::execution_plan`]) and
//! the tensor memory planner (`svod-tensor`'s `memory_planner`). Keeping one
//! implementation guarantees the planner's level-interval reuse decisions match
//! the order kernels actually execute in.
//!
//! The graph is described abstractly by `node_ids` (one `u64` per node),
//! `callable_deps` (callable-ID dependencies, resolved against `node_ids` with
//! last-seen semantics for duplicate IDs), and optional `index_deps` (concrete
//! positional edges). All error paths — unresolved dep, self-loop, out-of-range
//! index dep, and cycles — return [`crate::error::Error::Execution`].

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

use crate::error::Result;

/// Resolved dependency graph over the input nodes, indexed positionally.
struct DependencyGraph {
    node_ids: Vec<u64>,
    in_degree: Vec<usize>,
    successors: Vec<Vec<usize>>,
}

/// Build the dependency graph from callable-ID deps plus optional index deps.
///
/// `callable_deps[i]` lists the callable IDs node `i` depends on (resolved
/// against `node_ids`, last-seen for duplicate IDs). `index_deps`, when present,
/// carries concrete positional edges (one `Vec` per node, length == `node_ids`).
fn build_topological_graph(
    node_ids: &[u64],
    callable_deps: &[Vec<u64>],
    index_deps: Option<&[Vec<usize>]>,
) -> Result<DependencyGraph> {
    let n = node_ids.len();
    if callable_deps.len() != n {
        return Err(crate::error::Error::Execution {
            reason: format!(
                "topological leveling dep table length mismatch: nodes={n}, callable_deps={}",
                callable_deps.len()
            ),
        });
    }
    if let Some(index_deps) = index_deps
        && index_deps.len() != n
    {
        return Err(crate::error::Error::Execution {
            reason: format!(
                "topological leveling index-dep table length mismatch: nodes={n}, index_deps={}",
                index_deps.len()
            ),
        });
    }

    let mut id_counts: HashMap<u64, usize> = HashMap::with_capacity(n);
    for &id in node_ids {
        *id_counts.entry(id).or_insert(0) += 1;
    }
    let has_duplicate_ids = id_counts.values().any(|&count| count > 1);

    let mut in_degree = vec![0usize; n];
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];

    if !has_duplicate_ids {
        let mut id_to_idx: HashMap<u64, usize> = HashMap::with_capacity(n);
        for (idx, &id) in node_ids.iter().enumerate() {
            id_to_idx.insert(id, idx);
        }

        for (idx, deps) in callable_deps.iter().enumerate() {
            for dep in deps {
                let Some(&dep_idx) = id_to_idx.get(dep) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!("node {} depends on unknown op id {}", node_ids[idx], dep),
                    });
                };
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }
        }
    } else {
        // Expanded schedules may contain repeated IDs for per-iteration items.
        // Resolve dependencies against the most recent prior node with that ID.
        let mut last_seen: HashMap<u64, usize> = HashMap::with_capacity(n);

        for (idx, deps) in callable_deps.iter().enumerate() {
            for dep in deps {
                let Some(&dep_idx) = last_seen.get(dep) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "node {} depends on unknown prior op id {} (duplicate-id schedule mode)",
                            node_ids[idx], dep
                        ),
                    });
                };
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }

            last_seen.insert(node_ids[idx], idx);
        }
    }

    if let Some(index_deps) = index_deps {
        for (idx, deps) in index_deps.iter().enumerate() {
            for &dep_idx in deps {
                if dep_idx >= n {
                    return Err(crate::error::Error::Execution {
                        reason: format!("node {} depends on unknown op index {}", node_ids[idx], dep_idx),
                    });
                }
                if dep_idx == idx {
                    return Err(crate::error::Error::Execution {
                        reason: format!("node {} cannot depend on itself by op index {}", node_ids[idx], dep_idx),
                    });
                }
                in_degree[idx] += 1;
                successors[dep_idx].push(idx);
            }
        }
    }

    Ok(DependencyGraph { node_ids: node_ids.to_vec(), in_degree, successors })
}

fn cycle_error(in_degree: &[usize], node_ids: &[u64]) -> crate::error::Error {
    let blocked: Vec<u64> = in_degree
        .iter()
        .enumerate()
        .filter_map(|(idx, &deg)| if deg > 0 { Some(node_ids[idx]) } else { None })
        .collect();
    crate::error::Error::Execution {
        reason: format!("cycle detected in prepared op dependencies: blocked_ids={blocked:?}"),
    }
}

/// Dependency-respecting linear order (deterministic min-index tie-break).
pub fn compute_topological_order(
    node_ids: &[u64],
    callable_deps: &[Vec<u64>],
    index_deps: Option<&[Vec<usize>]>,
) -> Result<Vec<usize>> {
    let DependencyGraph { node_ids, mut in_degree, successors } =
        build_topological_graph(node_ids, callable_deps, index_deps)?;

    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(idx));
        }
    }

    let mut order = Vec::with_capacity(node_ids.len());
    while let Some(Reverse(idx)) = ready.pop() {
        order.push(idx);
        for &succ in &successors[idx] {
            in_degree[succ] -= 1;
            if in_degree[succ] == 0 {
                ready.push(Reverse(succ));
            }
        }
    }

    if order.len() != node_ids.len() {
        return Err(cycle_error(&in_degree, &node_ids));
    }
    Ok(order)
}

/// Kahn wave-leveling: `levels[k]` is the set of node indices whose longest
/// dependency-path length is `k`. Within a level, indices are min-heap ordered.
/// Level assignment is independent of intra-level tie-break.
pub fn compute_topological_levels(
    node_ids: &[u64],
    callable_deps: &[Vec<u64>],
    index_deps: Option<&[Vec<usize>]>,
) -> Result<Vec<Vec<usize>>> {
    let DependencyGraph { node_ids, mut in_degree, successors } =
        build_topological_graph(node_ids, callable_deps, index_deps)?;

    let mut ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
    for (idx, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            ready.push(Reverse(idx));
        }
    }

    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut visited = 0usize;

    while !ready.is_empty() {
        let mut level: Vec<usize> = Vec::new();
        while let Some(Reverse(idx)) = ready.pop() {
            level.push(idx);
        }

        let mut next_ready: BinaryHeap<Reverse<usize>> = BinaryHeap::new();
        for &idx in &level {
            visited += 1;
            for &succ in &successors[idx] {
                in_degree[succ] -= 1;
                if in_degree[succ] == 0 {
                    next_ready.push(Reverse(succ));
                }
            }
        }

        levels.push(level);
        ready = next_ready;
    }

    if visited != node_ids.len() {
        return Err(cycle_error(&in_degree, &node_ids));
    }
    Ok(levels)
}
