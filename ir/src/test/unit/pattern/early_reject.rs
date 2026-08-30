//! Early reject: a compiled pattern is only dispatched when the root's direct children
//! carry every op kind its fixed-position sources demand.
//!
//! Tinygrad equivalent: `UPat.early_reject` (uop/ops.py:1349-1352) checked against
//! `UOp._src_ops` in `PatternMatcher.rewrite` (uop/ops.py:1480-1482).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use svod_macros::patterns;
use test_case::test_case;

use crate::op::OpMask;
use crate::op::pattern_derived::OpKey;
use crate::pattern::{RewriteResult, SimplifiedPatternMatcher, TypedPatternMatcher};
use crate::rewrite::engine::graph_rewrite;
use crate::types::{BinaryOp, TernaryOp, UnaryOp};
use crate::{ConstValue, Op, UOp};
use svod_dtype::DType;

fn int(value: i64) -> Arc<UOp> {
    UOp::const_(DType::Int32, ConstValue::Int(value))
}

fn var(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::Int32, 0, 1024)
}

fn bin(op: BinaryOp, lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    UOp::new(Op::Binary(op, lhs, rhs), DType::Int32)
}

fn mask(keys: &[OpKey]) -> OpMask {
    keys.iter().cloned().collect()
}

// =============================================================================
// (a) A node whose children lack a required op kind never reaches the closure
// =============================================================================

/// Root node kinds used by the dispatch table below.
#[derive(Debug, Clone, Copy)]
enum Node {
    /// `Add(Mul(1, 2), 3)` — children are MUL and CONST.
    AddMulConst,
    /// `Add(3, 4)` — both children are CONST.
    AddConstConst,
    /// `Add(x, y)` — both children are DEFINE_VAR.
    AddVarVar,
    /// `Neg(x)` — one DEFINE_VAR child.
    NegVar,
}

impl Node {
    fn build(self) -> Arc<UOp> {
        match self {
            Self::AddMulConst => bin(BinaryOp::Add, bin(BinaryOp::Mul, int(1), int(2)), int(3)),
            Self::AddConstConst => bin(BinaryOp::Add, int(3), int(4)),
            Self::AddVarVar => bin(BinaryOp::Add, var("a"), var("b")),
            Self::NegVar => UOp::new(Op::Unary(UnaryOp::Neg, var("a")), DType::Int32),
        }
    }

    fn key(self) -> OpKey {
        match self {
            Self::AddMulConst | Self::AddConstConst | Self::AddVarVar => OpKey::Binary(BinaryOp::Add),
            Self::NegVar => OpKey::Unary(UnaryOp::Neg),
        }
    }
}

/// Counting closure registered with `early_reject`; returns how many times it was entered.
fn dispatch_count(node: Node, early_reject: &[OpKey]) -> usize {
    let calls = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&calls);

    let mut matcher = SimplifiedPatternMatcher::<()>::new();
    matcher.add_rejecting(&[node.key()], early_reject, move |_uop, _ctx| {
        counter.fetch_add(1, Ordering::Relaxed);
        RewriteResult::NoMatch
    });
    matcher.rewrite(&node.build(), &mut ());
    calls.load(Ordering::Relaxed)
}

#[test_case(Node::AddMulConst, &[], 1; "no requirement always dispatches")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul)], 1; "required mul child present")]
#[test_case(Node::AddMulConst, &[OpKey::Const], 1; "required const child present")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul), OpKey::Const], 1; "both required present")]
#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Add)], 0; "required add child absent")]
#[test_case(Node::AddConstConst, &[OpKey::Binary(BinaryOp::Mul)], 0; "mul absent among consts")]
#[test_case(Node::AddConstConst, &[OpKey::Const], 1; "const present twice")]
#[test_case(Node::AddVarVar, &[OpKey::Const], 0; "const absent among vars")]
#[test_case(Node::AddVarVar, &[OpKey::DefineVar], 1; "define var present")]
#[test_case(Node::NegVar, &[OpKey::DefineVar], 1; "unary child present")]
#[test_case(Node::NegVar, &[OpKey::Const], 0; "unary child absent")]
fn dispatch_respects_early_reject(node: Node, early_reject: &[OpKey], expected: usize) {
    assert_eq!(dispatch_count(node, early_reject), expected);
}

/// One absent kind rejects even when the other required kinds are present.
#[test]
fn partial_overlap_still_rejects() {
    let requirement = &[OpKey::Binary(BinaryOp::Mul), OpKey::Binary(BinaryOp::Sub)];
    assert_eq!(dispatch_count(Node::AddMulConst, requirement), 0);
}

// =============================================================================
// (b) `patterns!` derives the requirement from the rule's fixed-position sources
// =============================================================================

/// Requirement the macro derived for the single rule of a one-rule matcher.
fn derived(matcher: &TypedPatternMatcher<()>, key: OpKey) -> OpMask {
    let rejects = matcher.early_rejects(&key);
    assert_eq!(rejects.len(), 1, "expected exactly one entry under {key:?}");
    rejects[0]
}

#[test]
fn derives_single_op_source() {
    let matcher = patterns! { Add(Mul(a, b), c) ~> a.mul(&b.add(c)) };
    assert_eq!(derived(&matcher, OpKey::Binary(BinaryOp::Add)), mask(&[OpKey::Binary(BinaryOp::Mul)]));
}

#[test]
fn derives_const_source() {
    let matcher = patterns! { Add(x, @zero) ~> x };
    assert_eq!(derived(&matcher, OpKey::Binary(BinaryOp::Add)), mask(&[OpKey::Const]));
}

#[test]
fn derives_union_over_sources() {
    let matcher = patterns! { Where(Lt(a, b), Cast { src: c, dtype: _d }, e) => Some(a.add(b).add(c).add(e)) };
    let expected = mask(&[OpKey::Binary(BinaryOp::Lt), OpKey::Cast]);
    assert_eq!(derived(&matcher, OpKey::Ternary(TernaryOp::Where)), expected);
}

#[test]
fn derives_from_struct_fields() {
    let matcher = patterns! { Reshape { src: Cast { src: inner, dtype: _d }, new_shape: _s } => Some(inner.clone()) };
    assert_eq!(derived(&matcher, OpKey::Reshape), mask(&[OpKey::Cast]));
}

/// Permuted sources demand the same set whichever ordering matches.
#[test]
fn derives_from_permutation() {
    let matcher = patterns! { Add[Mul(a, b), c] ~> a.mul(&b.add(c)) };
    assert_eq!(derived(&matcher, OpKey::Binary(BinaryOp::Add)), mask(&[OpKey::Binary(BinaryOp::Mul)]));
}

/// A source accepting several op kinds pins none of them — Tinygrad's `len(pp.op) == 1`.
/// `@anyconst` admits both CONST and VCONST, so it constrains the child set not at all.
#[test]
fn multi_kind_source_demands_nothing() {
    let matcher = patterns! { Add(x, c @anyconst(_vals)) => Some(x.add(c)) };
    assert!(derived(&matcher, OpKey::Binary(BinaryOp::Add)).is_empty());
}

/// A top-level alternative only demands what every branch demands.
#[test]
fn alternative_root_intersects_branches() {
    let matcher =
        patterns! { (Cast { src: Mul(a, b), dtype: _d } | BitCast { src: Mul(a, b), dtype: _e }) => Some(a.mul(b)) };
    let expected = mask(&[OpKey::Binary(BinaryOp::Mul)]);
    assert_eq!(derived(&matcher, OpKey::Cast), expected);
    assert_eq!(derived(&matcher, OpKey::BitCast), expected);
}

// =============================================================================
// (c) Wildcard sources are never early-rejected
// =============================================================================

#[test_case(Node::AddMulConst; "add over mul and const")]
#[test_case(Node::AddConstConst; "add over two consts")]
#[test_case(Node::AddVarVar; "add over two vars")]
#[test_case(Node::NegVar; "neg over var")]
fn wildcard_sources_are_never_rejected(node: Node) {
    // Bare-variable sources constrain nothing, so the rule stays dispatchable everywhere.
    let matcher = patterns! { Add(x, y) ~> y.add(x), Neg(x) ~> x.clone() };
    assert!(matcher.early_rejects(&node.key()).iter().all(|reject| reject.is_empty()));

    let node = node.build();
    assert!(!matches!(matcher.rewrite(&node, &mut ()), RewriteResult::NoMatch));
}

/// A wildcard rule has no root key at all and must run for every op, including leaves.
#[test]
fn wildcard_rule_runs_on_childless_node() {
    let matcher = patterns! { x if x.op().children().is_empty() => Some(int(7)) };
    assert_eq!(matcher.wildcard_count(), 1);

    for node in [int(1), var("a"), bin(BinaryOp::Add, int(1), int(2))] {
        let expected = node.op().children().is_empty();
        assert_eq!(!matches!(matcher.rewrite(&node, &mut ()), RewriteResult::NoMatch), expected);
    }
}

// =============================================================================
// (b bis) Equivalence pin: rejecting entries can never change a rewrite result
// =============================================================================

/// Rules exercising single-op, const, struct-field, permuted and wildcard sources.
fn pinned_matcher() -> TypedPatternMatcher<()> {
    patterns! {
        Add(x, @zero) ~> x,
        Mul(x, @one) ~> x,
        Add[Mul(a, b), Mul(c, d)] if Arc::ptr_eq(a, c) => Some(a.mul(&b.add(d))),
        Sub(Add(a, b), c) if Arc::ptr_eq(b, c) => Some(a.clone()),
        Neg(Neg(x)) ~> x,
        Cast { src: Cast { src: inner, dtype: _d }, dtype: outer } ~> inner.cast(outer.clone()),
        Where(Const(1), t, _f) ~> t,
    }
}

/// Graphs covering matching, non-matching and nested cases for the pinned rules.
fn pinned_graphs() -> Vec<Arc<UOp>> {
    let (x, y) = (var("x"), var("y"));
    let neg = |u: &Arc<UOp>| UOp::new(Op::Unary(UnaryOp::Neg, u.clone()), DType::Int32);
    vec![
        bin(BinaryOp::Add, x.clone(), int(0)),
        bin(BinaryOp::Mul, x.clone(), int(1)),
        bin(BinaryOp::Add, bin(BinaryOp::Mul, x.clone(), y.clone()), bin(BinaryOp::Mul, x.clone(), int(3))),
        bin(BinaryOp::Sub, bin(BinaryOp::Add, x.clone(), y.clone()), y.clone()),
        bin(BinaryOp::Sub, bin(BinaryOp::Add, x.clone(), y.clone()), int(5)),
        neg(&neg(&x)),
        neg(&x),
        x.clone().cast(DType::Int64).cast(DType::Float32),
        UOp::try_where(UOp::const_(DType::Bool, ConstValue::Bool(true)), x.clone(), y.clone())
            .expect("where over int32"),
        bin(BinaryOp::Add, bin(BinaryOp::Add, x.clone(), int(0)), bin(BinaryOp::Mul, y.clone(), int(1))),
        bin(BinaryOp::Add, x.clone(), y.clone()),
    ]
}

/// Every entry skipped by an early reject could not have matched, so clearing all rejects
/// must yield the pointer-identical graph — hash consing makes `ptr_eq` the exact check.
#[test]
fn early_reject_preserves_rewrite_results() {
    let matcher = pinned_matcher();
    let permissive = matcher.without_early_reject();
    assert!(permissive.early_rejects(&OpKey::Binary(BinaryOp::Add)).iter().all(|reject| reject.is_empty()));
    assert!(matcher.early_rejects(&OpKey::Binary(BinaryOp::Add)).iter().any(|m| !m.is_empty()));

    for graph in pinned_graphs() {
        let rejected = graph_rewrite(&matcher, graph.clone(), &mut ());
        let permissive = graph_rewrite(&permissive, graph.clone(), &mut ());
        assert!(Arc::ptr_eq(&rejected, &permissive), "diverged on {:?}", graph.op());
    }
}

// =============================================================================
// src_ops / OpMask
// =============================================================================

#[test_case(Node::AddMulConst, &[OpKey::Binary(BinaryOp::Mul), OpKey::Const]; "mul and const children")]
#[test_case(Node::AddConstConst, &[OpKey::Const]; "duplicate const children collapse")]
#[test_case(Node::AddVarVar, &[OpKey::DefineVar]; "define var children")]
#[test_case(Node::NegVar, &[OpKey::DefineVar]; "single unary child")]
fn src_ops_holds_direct_child_kinds(node: Node, expected: &[OpKey]) {
    assert_eq!(node.build().src_ops(), mask(expected));
}

/// Leaves carry the empty mask, which is a subset of everything and so rejects nothing.
#[test]
fn leaf_src_ops_is_empty() {
    assert!(int(1).src_ops().is_empty());
    assert!(OpMask::EMPTY.is_subset_of(int(1).src_ops()));
}

/// `src_ops` covers direct children only; grandchildren do not leak into the mask.
#[test]
fn src_ops_is_not_transitive() {
    let node = Node::AddMulConst.build();
    assert!(!mask(&[OpKey::Binary(BinaryOp::Add)]).is_subset_of(node.src_ops()));
    assert!(mask(&[OpKey::Const]).is_subset_of(node.src_ops()));
}

/// Every op kind gets its own bit — grouped ops included, so `Add` never masks `Mul`.
#[test]
fn op_keys_have_distinct_bits() {
    let keys = [
        OpKey::Const,
        OpKey::DefineVar,
        OpKey::Cast,
        OpKey::BitCast,
        OpKey::Reshape,
        OpKey::Binary(BinaryOp::Add),
        OpKey::Binary(BinaryOp::Mul),
        OpKey::Unary(UnaryOp::Neg),
        OpKey::Unary(UnaryOp::Sqrt),
        OpKey::Ternary(TernaryOp::Where),
        OpKey::Ternary(TernaryOp::MulAcc),
    ];
    for (i, a) in keys.iter().enumerate() {
        for (j, b) in keys.iter().enumerate() {
            assert_eq!(i == j, mask(std::slice::from_ref(a)).is_subset_of(mask(std::slice::from_ref(b))));
        }
    }
    assert!(keys.iter().all(|key| key.index() < crate::op::pattern_derived::OP_KEY_COUNT));
}
