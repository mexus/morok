//! Unit tests for the shared topological-leveling routines.

use crate::error::Error;
use crate::leveling::{compute_topological_levels, compute_topological_order};

fn is_execution_err<T: std::fmt::Debug>(r: Result<T, Error>) {
    match r {
        Err(Error::Execution { .. }) => {}
        other => panic!("expected Error::Execution, got {other:?}"),
    }
}

// ------------------------------------------------------------------ happy path

#[test]
fn levels_linear_chain() {
    // 10 → 11 → 12 : one node per level.
    let levels = compute_topological_levels(&[10, 11, 12], &[vec![], vec![10], vec![11]], None).unwrap();
    assert_eq!(levels, vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn levels_diamond() {
    // 0 → {1, 2} → 3. Level 1 holds both 1 and 2 (min-index ordered).
    let levels = compute_topological_levels(&[0, 1, 2, 3], &[vec![], vec![0], vec![0], vec![1, 2]], None).unwrap();
    assert_eq!(levels, vec![vec![0], vec![1, 2], vec![3]]);
}

#[test]
fn order_is_valid_topological() {
    let order = compute_topological_order(&[0, 1, 2, 3], &[vec![], vec![0], vec![0], vec![1, 2]], None).unwrap();
    // Each node appears after all its dependencies.
    let pos: std::collections::HashMap<usize, usize> = order.iter().enumerate().map(|(p, &n)| (n, p)).collect();
    assert!(pos[&0] < pos[&1] && pos[&0] < pos[&2] && pos[&1] < pos[&3] && pos[&2] < pos[&3]);
}

#[test]
fn index_deps_add_edges() {
    // No callable deps; an index edge 1→0 forces node 1 to a later level.
    let levels = compute_topological_levels(&[1, 2], &[vec![], vec![]], Some(&[vec![], vec![0]])).unwrap();
    assert_eq!(levels, vec![vec![0], vec![1]]);
}

#[test]
fn duplicate_ids_resolve_to_most_recent() {
    // Repeated id 7: the second node's dep on 7 resolves to the prior occurrence (index 0).
    let levels = compute_topological_levels(&[7, 7], &[vec![], vec![7]], None).unwrap();
    assert_eq!(levels, vec![vec![0], vec![1]]);
}

// ----------------------------------------------------------------- error paths

#[test]
fn errors_on_cycle() {
    is_execution_err(compute_topological_levels(&[1, 2], &[vec![2], vec![1]], None));
    is_execution_err(compute_topological_order(&[1, 2], &[vec![2], vec![1]], None));
}

#[test]
fn errors_on_unresolved_dep() {
    is_execution_err(compute_topological_levels(&[1], &[vec![99]], None));
}

#[test]
fn errors_on_callable_dep_len_mismatch() {
    is_execution_err(compute_topological_levels(&[1, 2], &[vec![]], None));
}

#[test]
fn errors_on_index_dep_len_mismatch() {
    is_execution_err(compute_topological_levels(&[1], &[vec![]], Some(&[vec![], vec![]])));
}

#[test]
fn errors_on_self_loop_index_dep() {
    is_execution_err(compute_topological_order(&[1], &[vec![]], Some(&[vec![0]])));
}

#[test]
fn errors_on_out_of_range_index_dep() {
    is_execution_err(compute_topological_levels(&[1], &[vec![]], Some(&[vec![5]])));
}
