mod devectorize_pin;
mod index_lowering;

use crate::{
    pattern::RewriteResult,
    rewrite::graph_rewrite,
    symbolic::patterns::{
        commutative_canonicalization, comparison_dsl_patterns, sym_phase3_patterns, term_combining_dsl_patterns,
        weak_float_values_are_committed,
    },
    symbolic::{sym, symbolic, symbolic_simple},
};
use smallvec::smallvec;
use std::{f32::consts::PI, sync::Arc};
use svod_dtype::DType;
use svod_ir::uop::cached_property::CachedProperty;
use svod_ir::uop::properties::HasWeakFloatProperty;
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};

fn assert_binary_sources(root: &Arc<UOp>, lhs: &Arc<UOp>, rhs: &Arc<UOp>) {
    let Op::Binary(_, actual_lhs, actual_rhs) = root.op() else {
        panic!("expected binary root, got {:?}", root.op());
    };
    assert!(Arc::ptr_eq(actual_lhs, lhs), "unexpected lhs: {:?}", root.op());
    assert!(Arc::ptr_eq(actual_rhs, rhs), "unexpected rhs: {:?}", root.op());
}

#[test]
fn commutative_index_ops_follow_tinygrad_structural_order() {
    let end = UOp::index_const(8);
    let special = UOp::special(end.clone(), "gidx0".to_string());
    let range = UOp::range(end, 0);

    for op in [BinaryOp::Add, BinaryOp::Mul, BinaryOp::And, BinaryOp::Or, BinaryOp::Xor, BinaryOp::Max] {
        let authored = UOp::new(Op::Binary(op, special.clone(), range.clone()), DType::WeakInt);
        let reversed = UOp::new(Op::Binary(op, range.clone(), special.clone()), DType::WeakInt);
        let authored = graph_rewrite(commutative_canonicalization(), authored, &mut ());
        let reversed = graph_rewrite(commutative_canonicalization(), reversed, &mut ());

        assert_binary_sources(&authored, &special, &range);
        assert_binary_sources(&reversed, &special, &range);
        assert!(Arc::ptr_eq(&authored, &reversed));
        assert_eq!(authored.content_hash, reversed.content_hash);
    }
}

#[test]
fn commutative_index_order_covers_constants_ranges_and_nested_trees() {
    let range0 = UOp::range_const(8, 0);
    let range1 = UOp::range_const(8, 1);
    let constant = UOp::index_const(3);

    let const_first = constant.try_add(&range0).unwrap();
    let const_first = graph_rewrite(commutative_canonicalization(), const_first, &mut ());
    assert_binary_sources(&const_first, &range0, &constant);

    let ranges_reversed = range1.try_add(&range0).unwrap();
    let ranges_reversed = graph_rewrite(commutative_canonicalization(), ranges_reversed, &mut ());
    assert_binary_sources(&ranges_reversed, &range0, &range1);

    let nested = range1.try_add(&range0).unwrap();
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let nested_first = nested.try_add(&special).unwrap();
    let nested_first = graph_rewrite(commutative_canonicalization(), nested_first, &mut ());
    let Op::Binary(BinaryOp::Add, actual_special, actual_nested) = nested_first.op() else {
        panic!("expected nested ADD, got {:?}", nested_first.op());
    };
    assert!(Arc::ptr_eq(actual_special, &special));
    assert_binary_sources(actual_nested, &range0, &range1);

    let nested_other_order = special.try_add(&range0.try_add(&range1).unwrap()).unwrap();
    let nested_other_order = graph_rewrite(commutative_canonicalization(), nested_other_order, &mut ());
    assert!(Arc::ptr_eq(&nested_first, &nested_other_order));

    let var_a = UOp::define_var("a".to_string(), 0, 8);
    let var_b = UOp::define_var("b".to_string(), 0, 8);
    let vars_reversed = var_b.try_add(&var_a).unwrap();
    let vars_reversed = graph_rewrite(commutative_canonicalization(), vars_reversed, &mut ());
    assert_binary_sources(&vars_reversed, &var_a, &var_b);
}

#[test]
fn commutative_order_is_weakint_only_and_preserves_tags() {
    for dtype in [DType::Index, DType::Int32, DType::Float32] {
        let lhs = UOp::const_(dtype.clone(), if dtype.is_float() { 4.0.into() } else { 4.into() });
        let rhs = UOp::const_(dtype.clone(), if dtype.is_float() { 3.0.into() } else { 3.into() });
        let authored = UOp::new(Op::Binary(BinaryOp::Add, lhs.clone(), rhs.clone()), dtype);
        let result = graph_rewrite(commutative_canonicalization(), authored.clone(), &mut ());
        assert!(Arc::ptr_eq(&result, &authored));
        assert_binary_sources(&result, &lhs, &rhs);
    }

    let range = UOp::range_const(8, 0);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let tagged = range.try_add(&special).unwrap().with_tag(smallvec![7]);
    let result = graph_rewrite(commutative_canonicalization(), tagged, &mut ());
    assert_binary_sources(&result, &special, &range);
    assert_eq!(result.tag().as_deref(), Some(&[7][..]));
}

#[test]
fn commutative_order_projects_vconst_as_tinygrad_stack() {
    let vconst = UOp::vconst(vec![ConstValue::Int(2), ConstValue::Int(1)], DType::WeakInt);
    let stack = UOp::stack(smallvec![UOp::range_const(8, 0), UOp::special(UOp::index_const(8), "gidx0".to_string()),]);
    let authored = UOp::new(Op::Binary(BinaryOp::Add, vconst.clone(), stack.clone()), vconst.dtype());

    let ordered = graph_rewrite(commutative_canonicalization(), authored, &mut ());
    assert_binary_sources(&ordered, &stack, &vconst);

    let left = UOp::stack(smallvec![UOp::range_const(8, 1), UOp::range_const(8, 0)]);
    let right = UOp::stack(smallvec![
        UOp::special(UOp::index_const(8), "gidx1".to_string()),
        UOp::special(UOp::index_const(8), "gidx0".to_string()),
    ]);
    let authored = UOp::new(Op::Binary(BinaryOp::Add, left.clone(), right.clone()), left.dtype());
    let ordered = graph_rewrite(commutative_canonicalization(), authored, &mut ());
    assert_binary_sources(&ordered, &right, &left);
}

#[test]
fn commutative_order_does_not_break_structural_ties_or_incomparable_args() {
    let base = UOp::index_const(7);
    let left = base.with_tag(smallvec![1]);
    let right = base.with_tag(smallvec![2]);
    let tied = UOp::new(Op::Binary(BinaryOp::Add, right.clone(), left.clone()), DType::WeakInt);
    let tied = graph_rewrite(commutative_canonicalization(), tied, &mut ());
    assert_binary_sources(&tied, &right, &left);

    let end = UOp::index_const(8);
    let weak = UOp::range_axis(end.clone(), svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Weak);
    let global = UOp::range_axis(end, svod_ir::AxisId::Renumbered(0), svod_ir::AxisType::Global);
    let incomparable = weak.try_add(&global).unwrap();
    let incomparable = graph_rewrite(commutative_canonicalization(), incomparable, &mut ());
    assert_binary_sources(&incomparable, &weak, &global);
}

#[test]
fn symbolic_boundary_applies_structural_commutative_order() {
    let range = UOp::range_const(8, 0);
    let special = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let reversed = range.try_add(&special).unwrap();
    let simple = graph_rewrite(symbolic_simple(), reversed.clone(), &mut ());
    assert_binary_sources(&simple, &range, &special);

    for matcher in [symbolic(), sym()] {
        let result = graph_rewrite(matcher, reversed.clone(), &mut ());
        assert_binary_sources(&result, &special, &range);
    }
}

#[test]
fn test_symbolic_simple_identity_folding() {
    let matcher = symbolic_simple();

    // Test: 5 + 0 -> 5
    let five = UOp::native_const(5i32);
    let zero = UOp::native_const(0i32);
    let add = five.try_add(&zero).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::sync::Arc::ptr_eq(&rewritten, &five));
    }

    // Test: 0 + 5 -> 5 (commutative)
    let add2 = zero.try_add(&five).unwrap();
    let result2 = matcher.rewrite(&add2, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(_)));

    // Test: 5 * 1 -> 5
    let one = UOp::native_const(1i32);
    let mul = five.try_mul(&one).unwrap();
    let result3 = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result3, RewriteResult::Rewritten(_)));

    // Test: 5 - 0 -> 5
    let sub = five.try_sub(&zero).unwrap();
    let result4 = graph_rewrite(matcher, sub, &mut ());
    assert!(Arc::ptr_eq(&result4, &five));

    // Test: 5 / 1 -> 5 (int division)
    let idiv = five.try_div(&one).unwrap();
    let result5 = matcher.rewrite(&idiv, &mut ());
    assert!(matches!(result5, RewriteResult::Rewritten(_)));

    // Test: 5.0 / 1.0 -> 5.0 (float division)
    let five_f = UOp::native_const(5.0f32);
    let one_f = UOp::native_const(1.0f32);
    let fdiv = five_f.try_div(&one_f).unwrap();
    let result6 = matcher.rewrite(&fdiv, &mut ());
    assert!(matches!(result6, RewriteResult::Rewritten(_)));

    // Test: 5 | 0 -> 5
    let or_op = five.try_or_op(&zero).unwrap();
    let result7 = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result7, RewriteResult::Rewritten(_)));

    // Test: 5 ^ 0 -> 5
    let xor_op = five.try_xor_op(&zero).unwrap();
    let result8 = matcher.rewrite(&xor_op, &mut ());
    assert!(matches!(result8, RewriteResult::Rewritten(_)));
}

#[test]
fn test_symbolic_add_sub_same_const_cancels() {
    use crate::rewrite::graph_rewrite;

    let x = UOp::define_var("x".to_string(), 0, 1024);
    let one = UOp::index_const(1);
    let expr = one.try_add(&x).unwrap().try_sub(&one).unwrap();

    let result = graph_rewrite(symbolic(), expr, &mut ());
    assert!(Arc::ptr_eq(&result, &x), "expected (1 + x) - 1 to simplify to x, got {:?}", result.op());
}

#[test]
fn test_symbolic_simple_zero_propagation() {
    let matcher = symbolic_simple();

    let five = UOp::native_const(5i32);
    let zero = UOp::native_const(0i32);

    // Test: 5 * 0 -> 0
    let mul = five.try_mul(&zero).unwrap();

    let result = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Check that the result is a zero constant (value-based)
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected Const op, got {:?}", rewritten.op());
        }
    }

    // Test: 0 * 5 -> 0 (commutative)
    let mul2 = zero.try_mul(&five).unwrap();
    let result2 = matcher.rewrite(&mul2, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(_)));

    // Test: 5 & 0 -> 0
    let and_op = five.try_and_op(&zero).unwrap();
    let result3 = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result3, RewriteResult::Rewritten(_)));

    // Test: 0 & 5 -> 0 (commutative)
    let and2 = zero.try_and_op(&five).unwrap();
    let result4 = matcher.rewrite(&and2, &mut ());
    assert!(matches!(result4, RewriteResult::Rewritten(_)));
}

#[test]
fn test_symbolic_simple_const_folding() {
    let matcher = symbolic_simple();

    // Test: 5 + 3 -> 8 (constant folding)
    let five = UOp::native_const(5i32);
    let three = UOp::native_const(3i32);
    let add = five.try_add(&three).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(8));
        } else {
            panic!("Expected Const(Int(8)), got {:?}", rewritten.op());
        }
    }

    // Test: 5 * 2 -> 10 (constant folding)
    let two = UOp::native_const(2i32);
    let mul = five.try_mul(&two).unwrap();

    let result2 = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result2, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result2 {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(10));
        } else {
            panic!("Expected Const(Int(10)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn reduced_float_folding_commits_result_before_comparison() {
    let matcher = symbolic_simple();
    let one = UOp::const_(DType::FP8E4M3, ConstValue::Float(1.0));
    let half_ulp = UOp::const_(DType::FP8E4M3, ConstValue::Float(0.0625));
    let add = one.try_add(&half_ulp).unwrap();
    let folded = graph_rewrite(&matcher, add, &mut ());
    assert!(matches!(folded.op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)));

    let rounded = UOp::const_(DType::Float32, ConstValue::Float(-3.2));
    let exact_grid_value = UOp::const_(DType::Float32, ConstValue::Float(-3.200000047683716));
    let comparison = rounded.try_cmpeq(&exact_grid_value).unwrap();
    let folded_comparison = graph_rewrite(symbolic(), comparison, &mut ());
    assert!(matches!(folded_comparison.op(), Op::Const(value) if value.0 == ConstValue::Bool(true)));
}

#[test]
fn reduced_float_vconst_folding_commits_each_result_lane() {
    let matcher = symbolic_simple();
    let values = UOp::vconst(vec![ConstValue::Float(1.0), ConstValue::Float(1.125)], DType::FP8E4M3);
    let increments = UOp::vconst(vec![ConstValue::Float(0.0625), ConstValue::Float(0.0625)], DType::FP8E4M3);
    let folded = graph_rewrite(&matcher, values.try_add(&increments).unwrap(), &mut ());
    assert!(matches!(folded.op(), Op::VConst { values }
        if values == &vec![ConstValue::Float(1.0), ConstValue::Float(1.25)]));
}

// ====== Tests for NEW patterns ======

#[test]
fn test_self_division() {
    // Test: x // x -> 1
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 1, 100);
    let div = x.try_div(&x).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(1));
        } else {
            panic!("Expected Const(1), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_division_by_neg_one() {
    // Test: x // -1 -> -x (which is MUL(x, -1) since neg() produces MUL)
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let neg_one = UOp::native_const(-1i32);
    let div = x.try_div(&neg_one).unwrap();

    let result = graph_rewrite(&matcher, div, &mut ());
    // x // -1 → neg(x) → MUL(x, -1)
    if let Op::Binary(svod_ir::BinaryOp::Mul, inner, c) = result.op() {
        assert!(std::sync::Arc::ptr_eq(inner, &x));
        assert!(matches!(c.op(), Op::Const(cv) if cv.0.is_neg_one()));
    } else {
        panic!("Expected MUL(x, -1), got {:?}", result.op());
    }
}

#[test]
fn test_idempotent_modulo() {
    // Test: (x % y) % y -> x % y
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 1, 10);

    // Build (x % y) % y
    let inner_mod = x.try_mod(&y).unwrap();
    let outer_mod = inner_mod.try_mod(&y).unwrap();

    let result = matcher.rewrite(&outer_mod, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be equivalent to inner_mod (x % y)
        if let Op::Binary(BinaryOp::FloorMod, a, b) = rewritten.op() {
            assert!(std::sync::Arc::ptr_eq(a, &x));
            assert!(std::sync::Arc::ptr_eq(b, &y));
        } else {
            panic!("Expected Binary(FloorMod, x, y), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_idempotent_and() {
    // Test: x & x -> x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let and_op = x.try_and_op(&x).unwrap();

    let result = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::sync::Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_idempotent_or() {
    // Test: x | x -> x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let or_op = x.try_or_op(&x).unwrap();

    let result = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::sync::Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_non_idempotent_and() {
    // Test: x & y (different variables) -> no match
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let y = UOp::var("y", DType::Int32, 0, i64::MAX);
    let and_op = x.try_and_op(&y).unwrap();

    let result = matcher.rewrite(&and_op, &mut ());
    // Should not match idempotent pattern
    // But might match other patterns (like zero propagation if one is zero)
    // For this test, we're using variables, so no simplification expected
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ====== Tests for ZERO FOLDING patterns ======

#[test]
fn test_self_comparison_lt() {
    // Test: x < x -> False
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let lt = x.try_cmplt(&x).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_self_modulo() {
    // Test: x % x -> 0
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 1, 100);
    let modulo = x.try_mod(&x).unwrap();

    let result = matcher.rewrite(&modulo, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected Const(0), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_self_inequality_int() {
    // Test: x != x -> False (for integers)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let ne = x.try_cmpne(&x).unwrap();

    let result = matcher.rewrite(&ne, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_self_inequality_float_no_fold() {
    // Test: x != x (for floats) -> no match (NaN != NaN is true)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);
    let ne = x.try_cmpne(&x).unwrap();

    let result = matcher.rewrite(&ne, &mut ());
    // Should not match because floats can have NaN
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ====== Tests for DIVISION patterns ======

#[test]
fn test_float_self_division() {
    // Zero is in range, so x/x may be NaN and must remain a division.
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);
    let div = x.try_div(&x).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_division_cancel_multiplication() {
    // Floating cancellation changes overflow, underflow, rounding, and y=0.
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);
    let y = UOp::var("y", DType::Float32, 0, i64::MAX);

    let mul = x.try_mul(&y).unwrap();
    let div = mul.try_div(&y).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_finite_nonzero_float_self_division() {
    let x = UOp::var("x", DType::Float32, 1, 10);
    let div = x.try_div(&x).unwrap();
    let RewriteResult::Rewritten(result) = symbolic_simple().rewrite(&div, &mut ()) else {
        panic!("finite non-zero x/x should fold")
    };
    assert!(matches!(result.op(), Op::Const(value) if value.0 == ConstValue::Float(1.0)));
}

#[test]
fn test_int_division_cancel_multiplication() {
    // Test: (x * y) // y -> x (integer division)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, -10, 10);
    let y = UOp::var("y", DType::Int32, 2, 3);

    let mul = x.try_mul(&y).unwrap();
    let div = mul.try_div(&y).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(std::sync::Arc::ptr_eq(&rewritten, &x));
    }
}

// ====== Tests for CAST OPTIMIZATION patterns ======

#[test]
fn test_cast_int_to_float_constant() {
    // Test: cast(int_const) -> float_const
    let matcher = crate::symbolic::pm_fold_cast_const();
    let int_val = UOp::native_const(42i32);
    let cast = int_val.cast(DType::Float32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Float(42.0));
        } else {
            panic!("Expected Const(Float(42.0)), got {:?}", rewritten.op());
        }
        assert_eq!(rewritten.dtype(), DType::Float32);
    }
}

#[test]
fn test_cast_float_to_int_constant() {
    // Test: cast(float_const) -> int_const
    let matcher = crate::symbolic::pm_fold_cast_const();
    let float_val = UOp::native_const(PI);
    let cast = float_val.cast(DType::Int32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(3));
        } else {
            panic!("Expected Const(Int(3)), got {:?}", rewritten.op());
        }
        assert_eq!(rewritten.dtype(), DType::Int32);
    }
}

#[test]
fn test_cast_bool_to_int_constant() {
    // Test: cast(bool_const) -> int_const
    let matcher = crate::symbolic::pm_fold_cast_const();
    let bool_val = UOp::native_const(true);
    let cast = bool_val.cast(DType::Int32);

    let result = matcher.rewrite(&cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(1));
        } else {
            panic!("Expected Const(Int(1)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_noop_cast_same_dtype() {
    // Test: x.cast(dtype) -> x if same dtype
    use crate::rewrite::graph_rewrite;

    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let cast = x.cast(DType::Int32);

    let result = graph_rewrite(&matcher, cast, &mut ());
    // The cast should be eliminated, returning x
    assert!(std::sync::Arc::ptr_eq(&result, &x), "Noop cast should be eliminated");
}

#[test]
fn test_double_cast_collapse_safe() {
    // Test: x.cast(Int32).cast(Int16) -> x.cast(Int16)
    // This is SAFE because Int32 can represent all Int16 values.
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int16, 0, i16::MAX as i64);

    // First cast: Int16 -> Int32 (widening, safe)
    let inner_cast = x.cast(DType::Int32);

    // Second cast: Int32 -> Int16 (narrowing back)
    let outer_cast = inner_cast.cast(DType::Int16);

    let result = matcher.rewrite(&outer_cast, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should simplify to just x (since x is already Int16 and intermediate was safe)
        assert!(std::sync::Arc::ptr_eq(&rewritten, &x), "Expected x, got {:?}", rewritten.op());
    }
}

#[test]
fn test_double_cast_no_collapse_unsafe() {
    // Test: x.cast(Float32).cast(Int32) should NOT collapse
    // This is UNSAFE because Float32 cannot exactly represent all Int32 values
    // (Float32 has only 23 mantissa bits, so integers > 2^24 may lose precision)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);

    // First cast: Int32 -> Float32 (potential precision loss for large integers)
    let inner_cast = x.cast(DType::Float32);

    // Second cast: Float32 -> Int32
    let outer_cast = inner_cast.cast(DType::Int32);

    let result = matcher.rewrite(&outer_cast, &mut ());
    // Should NOT be rewritten because the intermediate Float32 can't hold all Int32 values
    assert!(matches!(result, RewriteResult::NoMatch), "Unsafe double cast should NOT collapse: Int32->Float32->Int32");
}

#[test]
fn test_cast_non_constant_no_fold() {
    // Test: cast(variable) -> no constant folding (only dtype change)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let cast = x.cast(DType::Float32);

    let result = matcher.rewrite(&cast, &mut ());
    // Should not match constant folding pattern (not a constant)
    // Should not match noop cast (different dtypes)
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== Term Combining Tests ==========

#[test]
fn test_combine_identical_terms() {
    // Test: x + x → 2*x
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let add = x.try_add(&x).unwrap();

    let result = matcher.rewrite(&add, &mut ());

    // Debug: print the result if it doesn't match
    if !matches!(result, RewriteResult::Rewritten(_)) {
        eprintln!("Test failed: x + x didn't match. Result: {:?}", result);
        eprintln!("Add op: {:?}", add.op());
    }

    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be 2*x
        if let Op::Binary(BinaryOp::Mul, lhs, rhs) = rewritten.op() {
            let (c, var) = if matches!(lhs.op(), Op::Const(_)) {
                (lhs, rhs)
            } else if matches!(rhs.op(), Op::Const(_)) {
                (rhs, lhs)
            } else {
                panic!("Expected one Mul operand to be constant, got {:?}", rewritten.op());
            };

            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(2));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
            assert!(Arc::ptr_eq(var, &x));
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_combine_terms_with_coefficients() {
    // Test: (3 * x) + (5 * x) → 8 * x
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let term1 = c3.try_mul(&x).unwrap();
    let term2 = c5.try_mul(&x).unwrap();
    let add = term1.try_add(&term2).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x*8 (canonical form: var on left, const on right)
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_combine_terms_reversed_multiplication() {
    // Test: (x * 3) + (x * 5) → x * 8
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let term1 = x.try_mul(&c3).unwrap();
    let term2 = x.try_mul(&c5).unwrap();
    let add = term1.try_add(&term2).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x*8
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_no_combine_different_variables() {
    // Test: (3 * x) + (5 * y) → no rewrite (different variables)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let y = UOp::var("y", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let term1 = c3.try_mul(&x).unwrap();
    let term2 = c5.try_mul(&y).unwrap();
    let add = term1.try_add(&term2).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    // Should not combine different variables
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== ALU Folding Tests ==========

#[test]
fn test_alu_fold_addition_chain() {
    // Test: (x + 3) + 5 → x + 8
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let add1 = x.try_add(&c3).unwrap();
    let add2 = add1.try_add(&c5).unwrap();

    let result = matcher.rewrite(&add2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x + 8
        if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_multiplication_chain() {
    // Test: (x * 2) * 3 → x * 6
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c2 = UOp::native_const(2i32);
    let c3 = UOp::native_const(3i32);
    let mul1 = x.try_mul(&c2).unwrap();
    let mul2 = mul1.try_mul(&c3).unwrap();

    let result = matcher.rewrite(&mul2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x * 6
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(6));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_alu_fold_sub_then_add_positive() {
    // Test: (x - 3) + 5 → x + 2
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let sub = x.try_sub(&c3).unwrap();
    let add = sub.try_add(&c5).unwrap();

    let rewritten = graph_rewrite(matcher, add, &mut ());
    // Should be x + 2
    if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
        assert!(Arc::ptr_eq(var, &x));
        if let Op::Const(cv) = c.op() {
            assert_eq!(cv.0, ConstValue::Int(2));
        } else {
            panic!("Expected constant, got {:?}", c.op());
        }
    } else {
        panic!("Expected Add, got {:?}", rewritten.op());
    }
}

#[test]
fn test_alu_fold_sub_then_add_negative() {
    // Test: (x - 5) + 3 → x + (-2)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c5 = UOp::native_const(5i32);
    let c3 = UOp::native_const(3i32);
    let sub = x.try_sub(&c5).unwrap();
    let add = sub.try_add(&c3).unwrap();

    let rewritten = graph_rewrite(matcher, add, &mut ());
    // Should be x + (-2)
    if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
        assert!(Arc::ptr_eq(var, &x));
        if let Op::Const(cv) = c.op() {
            assert_eq!(cv.0, ConstValue::Int(-2));
        } else {
            panic!("Expected constant, got {:?}", c.op());
        }
    } else {
        panic!("Expected Add, got {:?}", rewritten.op());
    }
}

#[test]
fn test_alu_fold_add_then_sub_positive() {
    // Test: (x + 5) - 3 → x + 2
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c5 = UOp::native_const(5i32);
    let c3 = UOp::native_const(3i32);
    let add = x.try_add(&c5).unwrap();
    let sub = add.try_sub(&c3).unwrap();

    let rewritten = graph_rewrite(matcher, sub, &mut ());
    // Should be x + 2
    if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
        assert!(Arc::ptr_eq(var, &x));
        if let Op::Const(cv) = c.op() {
            assert_eq!(cv.0, ConstValue::Int(2));
        } else {
            panic!("Expected constant, got {:?}", c.op());
        }
    } else {
        panic!("Expected Add, got {:?}", rewritten.op());
    }
}

#[test]
fn test_alu_fold_add_then_sub_negative() {
    // Test: (x + 3) - 5 → x + (-2)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let add = x.try_add(&c3).unwrap();
    let sub = add.try_sub(&c5).unwrap();

    let rewritten = graph_rewrite(matcher, sub, &mut ());
    // Should be x + (-2)
    if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
        assert!(Arc::ptr_eq(var, &x));
        if let Op::Const(cv) = c.op() {
            assert_eq!(cv.0, ConstValue::Int(-2));
        } else {
            panic!("Expected constant, got {:?}", c.op());
        }
    } else {
        panic!("Expected Add, got {:?}", rewritten.op());
    }
}

// ========== Division Pattern Tests ==========

fn eval_closed_typed(expr: &Arc<UOp>) -> Option<ConstValue> {
    use svod_ir::uop::eval::{eval_binary_op_typed, eval_unary_op_typed};

    match expr.op() {
        Op::Const(value) => Some(value.0),
        Op::DefineVar { min_val, max_val, .. } if min_val == max_val => {
            ConstValue::Int(*min_val).cast(&expr.dtype().scalar_dtype())
        }
        Op::Binary(op, lhs, rhs) => {
            eval_binary_op_typed(*op, eval_closed_typed(lhs)?, eval_closed_typed(rhs)?, expr.dtype().base())
        }
        Op::Unary(op, src) => eval_unary_op_typed(*op, eval_closed_typed(src)?, expr.dtype().base()),
        _ => None,
    }
}

#[test]
fn typed_divmod_wrap_counterexamples_do_not_misrewrite() {
    let i8_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));

    let div = i8_const(100).mul(&i8_const(2)).add(&i8_const(1)).floor_div(&i8_const(2));
    let div_result = graph_rewrite(symbolic(), div.clone(), &mut ());
    assert_eq!(eval_closed_typed(&div), Some(ConstValue::Int(-28)));
    assert_eq!(eval_closed_typed(&div_result), eval_closed_typed(&div));
    assert!(!matches!(div_result.op(), Op::Const(value) if value.0 == ConstValue::Int(100)));

    let modulo = i8_const(100).mul(&i8_const(3)).add(&i8_const(1)).mod_(&i8_const(3));
    let mod_result = graph_rewrite(symbolic(), modulo.clone(), &mut ());
    assert_eq!(eval_closed_typed(&modulo), Some(ConstValue::Int(0)));
    assert_eq!(eval_closed_typed(&mod_result), eval_closed_typed(&modulo));
    assert!(!matches!(mod_result.op(), Op::Const(value) if value.0 == ConstValue::Int(1)));

    let x = UOp::var("wrap_x", DType::Int8, 100, 100);
    let y = UOp::var("wrap_y", DType::Int8, 2, 2);
    let cancellation = x.mul(&y).floor_div(&y);
    assert!(matches!(
        crate::symbolic::patterns::division_dsl_patterns().rewrite(&cancellation, &mut ()),
        RewriteResult::NoMatch
    ));
}

#[test]
fn typed_divmod_guards_cover_zero_and_integer_boundaries() {
    let zero = UOp::var("zero", DType::Int8, 0, 0);
    assert!(matches!(symbolic_simple().rewrite(&zero.floor_div(&zero), &mut ()), RewriteResult::NoMatch));
    assert!(matches!(symbolic_simple().rewrite(&zero.mod_(&zero), &mut ()), RewriteResult::NoMatch));

    let min = UOp::var("min", DType::Int8, i8::MIN as i64, i8::MIN as i64);
    let neg_one = UOp::const_(DType::Int8, ConstValue::Int(-1));
    assert!(matches!(symbolic_simple().rewrite(&min.floor_div(&neg_one), &mut ()), RewriteResult::NoMatch));

    let umax = UOp::var("umax", DType::UInt8, u8::MAX as i64, u8::MAX as i64);
    let two = UOp::const_(DType::UInt8, ConstValue::UInt(2));
    let wrapped = umax.mul(&two).floor_div(&two);
    assert!(matches!(
        crate::symbolic::patterns::division_dsl_patterns().rewrite(&wrapped, &mut ()),
        RewriteResult::NoMatch
    ));
}

#[test]
fn typed_division_cancellation_still_fires_when_product_is_exact() {
    for dtype in [
        DType::Int8,
        DType::UInt8,
        DType::WeakInt,
        DType::Index,
        DType::Int8.vec(4).unwrap(),
        DType::UInt8.vec(4).unwrap(),
    ] {
        let x = UOp::var("safe_x", dtype.clone(), 2, 10);
        let y = UOp::var("safe_y", dtype, 2, 3);
        let expression = x.mul(&y).floor_div(&y);
        let RewriteResult::Rewritten(result) =
            crate::symbolic::patterns::division_dsl_patterns().rewrite(&expression, &mut ())
        else {
            panic!("safe typed cancellation did not fire for {}", expression.tree());
        };
        assert!(Arc::ptr_eq(&result, &x));
    }
}

#[test]
fn qr_affine_divmod_congruence_folds_when_typed_arithmetic_is_exact() {
    let x = UOp::var("qr_index", DType::WeakInt, 0, 2);
    let five = UOp::const_(DType::WeakInt, ConstValue::Int(5));
    let numerator = x.mul(&x.const_like(6)).add(&x.const_like(2));

    let modulo = graph_rewrite(symbolic(), numerator.mod_(&five), &mut ());
    assert!(
        matches!(modulo.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
        if (Arc::ptr_eq(lhs, &x)
            && matches!(rhs.op(), Op::Const(value) if value.0 == ConstValue::Int(2)))
            || (Arc::ptr_eq(rhs, &x)
                && matches!(lhs.op(), Op::Const(value) if value.0 == ConstValue::Int(2)))),
        "unexpected modulo replacement: {}",
        modulo.tree()
    );

    let quotient = graph_rewrite(symbolic(), numerator.floor_div(&five), &mut ());
    assert!(Arc::ptr_eq(&quotient, &x), "unexpected quotient replacement: {}", quotient.tree());
}

#[test]
fn qr_affine_divmod_congruence_rejects_wrapping_source() {
    let x = UOp::var("qr_wrapping_index", DType::Int8, 20, 21);
    let five = UOp::const_(DType::Int8, ConstValue::Int(5));
    let numerator = x.mul(&x.const_like(6)).add(&x.const_like(2));

    for expression in [numerator.mod_(&five), numerator.floor_div(&five)] {
        assert!(
            matches!(
                crate::symbolic::patterns::advanced_division_dsl_patterns().rewrite(&expression, &mut ()),
                RewriteResult::NoMatch
            ),
            "wrapping congruence source was rewritten: {}",
            expression.tree()
        );
    }
}

#[test]
fn affine_divmod_congruence_rejects_hardware_vectors_without_dropping_terms() {
    let dtype = DType::Int8.vec(4).unwrap();
    let x = UOp::var("vector_x", dtype.clone(), 0, 1);
    let y = UOp::var("vector_y", dtype.clone(), 0, 1);
    let divisor = UOp::const_(dtype.clone(), ConstValue::Int(5));

    let vector_const = |value| UOp::const_(dtype.clone(), ConstValue::Int(value));
    let modulo = x.mul(&vector_const(6)).add(&y.mul(&vector_const(2))).mod_(&divisor);
    let quotient = x.mul(&vector_const(11)).add(&y.mul(&vector_const(6))).floor_div(&divisor);
    for expression in [modulo, quotient] {
        assert!(matches!(
            crate::symbolic::patterns::advanced_division_dsl_patterns().rewrite(&expression, &mut ()),
            RewriteResult::NoMatch
        ));
    }
}

#[test]
fn affine_divmod_congruence_preserves_broadcast_shape() {
    let scalar = |name| UOp::var(name, DType::Int8, 0, 1);
    let constant = |value| UOp::const_(DType::Int8, ConstValue::Int(value));
    let a = scalar("shape_a");
    let d = scalar("shape_d");
    let b = UOp::stack(vec![scalar("shape_b0"), scalar("shape_b1")].into());
    let numerator = a.mul(&constant(6)).add(&b.mul(&constant(5))).add(&d.mul(&constant(11)));
    let expression = numerator.mod_(&constant(5));
    assert_eq!(expression.shape().unwrap().unwrap().len(), 1);
    assert!(matches!(
        crate::symbolic::patterns::advanced_division_dsl_patterns().rewrite(&expression, &mut ()),
        RewriteResult::NoMatch
    ));
}

#[test]
fn divmod_guards_do_not_overflow_host_arithmetic() {
    let dtype = DType::WeakInt;
    let x = UOp::var("x", dtype.clone(), 0, 1);
    let c1 = UOp::const_(dtype.clone(), ConstValue::Int(i64::MAX));
    let c2 = UOp::const_(dtype.clone(), ConstValue::Int(i64::MAX));
    let c3 = UOp::const_(dtype, ConstValue::Int(1));
    let expression = x.mod_(&c1).mul(&c2).add(&x.floor_div(&c1).mul(&c3));
    let _ = crate::symbolic::patterns::div_mod_recombine_dsl_patterns().rewrite(&expression, &mut ());
}

#[test]
fn signed_floor_division_rewrites_keep_negative_cases_exact() {
    let i8_const = |value| UOp::const_(DType::Int8, ConstValue::Int(value));

    let x = UOp::var("comparison_x", DType::Int8, -1, 1);
    let comparison = x.floor_div(&i8_const(3)).lt(&i8_const(0));
    let RewriteResult::Rewritten(lifted) = comparison_dsl_patterns().rewrite(&comparison, &mut ()) else {
        panic!("positive-divisor comparison should lift");
    };
    assert!(matches!(lifted.op(), Op::Binary(BinaryOp::Lt, lhs, rhs)
        if Arc::ptr_eq(lhs, &x) && matches!(rhs.op(), Op::Const(value) if value.0 == ConstValue::Int(0))));

    // `(-128 // -9) // -2` has a single-bucket quotient, so tinygrad's
    // cancel_divmod (`uop/divandmod.py:13`) folds it to the exact constant. The
    // unsound `(a//b)//c -> a//(b*c)` reassociation stays rejected for c < 0.
    let nested = i8_const(-128).floor_div(&i8_const(-9)).floor_div(&i8_const(-2));
    let RewriteResult::Rewritten(folded) =
        crate::symbolic::patterns::advanced_division_dsl_patterns().rewrite(&nested, &mut ())
    else {
        panic!("single-bucket quotient should fold");
    };
    assert_eq!(eval_closed_typed(&nested), Some(ConstValue::Int(-7)));
    assert!(matches!(folded.op(), Op::Const(value) if value.0 == ConstValue::Int(-7)));

    let recombine = i8_const(-20)
        .floor_div(&i8_const(-9))
        .mod_(&i8_const(-2))
        .add(&i8_const(-20).floor_div(&i8_const(18)).mul(&i8_const(-2)));
    assert!(matches!(
        crate::symbolic::patterns::div_mod_recombine_dsl_patterns().rewrite(&recombine, &mut ()),
        RewriteResult::NoMatch
    ));
    assert_eq!(eval_closed_typed(&recombine), Some(ConstValue::Int(4)));
}

#[test]
fn exact_division_probe_does_not_panic_on_signed_min_over_neg_one() {
    let min = UOp::const_(DType::Int64, ConstValue::Int(i64::MIN));
    let zero = UOp::var("zero", DType::Int64, 0, 0);
    let expression = min.add(&zero).floor_div(&UOp::const_(DType::Int64, ConstValue::Int(-1)));
    assert!(matches!(
        crate::symbolic::patterns::advanced_division_dsl_patterns().rewrite(&expression, &mut ()),
        RewriteResult::NoMatch
    ));
}

#[test]
fn test_division_cancel_with_multiplication() {
    // Test: (a * b) // b → a
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Int32, -100, 100);
    let b = UOp::var("b", DType::Int32, 1, 100);
    let mul = a.try_mul(&b).unwrap();
    let div = mul.try_div(&b).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be just 'a'
        assert!(Arc::ptr_eq(&rewritten, &a));
    }
}

#[test]
fn test_division_chain_folding() {
    // Test: (a // 2) // 3 → a // 6
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c2 = UOp::native_const(2i32);
    let c3 = UOp::native_const(3i32);
    let div1 = a.try_div(&c2).unwrap();
    let div2 = div1.try_div(&c3).unwrap();

    let result = matcher.rewrite(&div2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be a // 6
        if let Op::Binary(BinaryOp::FloorDiv, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(6));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
        } else {
            panic!("Expected FloorDiv, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_exact_division_with_divides_helper() {
    // Test: (12 * x) // 3 → 4 * x (using divides helper)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let c12 = UOp::native_const(12i32);
    let c3 = UOp::native_const(3i32);
    let mul = c12.try_mul(&x).unwrap();
    let div = mul.try_div(&c3).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be 4 * x
        if let Op::Binary(BinaryOp::Mul, c, var) = rewritten.op() {
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(4));
            } else {
                panic!("Expected constant, got {:?}", c.op());
            }
            assert!(Arc::ptr_eq(var, &x));
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_modulo_with_divisible_left_operand() {
    // Test: (6 * x + y) % 3 → y % 3 (since 6*x is divisible by 3)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let c6 = UOp::native_const(6i32);
    let c3 = UOp::native_const(3i32);
    let mul = c6.try_mul(&x).unwrap();
    let add = mul.try_add(&y).unwrap();
    let modulo = add.try_mod(&c3).unwrap();

    let result = matcher.rewrite(&modulo, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be y % 3
        if let Op::Binary(BinaryOp::FloorMod, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &y));
            assert!(Arc::ptr_eq(c, &c3));
        } else {
            panic!("Expected FloorMod, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_modulo_with_divisible_right_operand() {
    // Test: (x + 9 * y) % 3 → x % 3 (since 9*y is divisible by 3)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let c9 = UOp::native_const(9i32);
    let c3 = UOp::native_const(3i32);
    let mul = c9.try_mul(&y).unwrap();
    let add = x.try_add(&mul).unwrap();
    let modulo = add.try_mod(&c3).unwrap();

    let result = matcher.rewrite(&modulo, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be x % 3
        if let Op::Binary(BinaryOp::FloorMod, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &x));
            assert!(Arc::ptr_eq(c, &c3));
        } else {
            panic!("Expected FloorMod, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_modulo_no_simplification() {
    // Test: (x + y) % 3 → no simplification (neither divisible by 3)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let c3 = UOp::native_const(3i32);
    let add = x.try_add(&y).unwrap();
    let modulo = add.try_mod(&c3).unwrap();

    let result = matcher.rewrite(&modulo, &mut ());
    // Should not simplify
    assert!(matches!(result, RewriteResult::NoMatch));
}

// ========== Distribution Pattern Tests ==========

#[test]
fn test_distribute_division_over_addition() {
    // Test: (6*x + 9*y) // 3 → (2*x) + (3*y)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let c6 = UOp::native_const(6i32);
    let c9 = UOp::native_const(9i32);
    let c3 = UOp::native_const(3i32);

    let term1 = c6.try_mul(&x).unwrap();
    let term2 = c9.try_mul(&y).unwrap();
    let add = term1.try_add(&term2).unwrap();
    let div = add.try_div(&c3).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (2*x) + (3*y)
        if let Op::Binary(BinaryOp::Add, left, right) = rewritten.op() {
            // Check left: 2*x
            if let Op::Binary(BinaryOp::Mul, c, var) = left.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(2));
                }
                assert!(Arc::ptr_eq(var, &x));
            } else {
                panic!("Expected Mul on left, got {:?}", left.op());
            }

            // Check right: 3*y
            if let Op::Binary(BinaryOp::Mul, c, var) = right.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(3));
                }
                assert!(Arc::ptr_eq(var, &y));
            } else {
                panic!("Expected Mul on right, got {:?}", right.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_distribute_division_over_subtraction() {
    // Test: (12*x - 6*y) // 3 → (4*x) - (2*y)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let c12 = UOp::native_const(12i32);
    let c6 = UOp::native_const(6i32);
    let c3 = UOp::native_const(3i32);

    let term1 = c12.try_mul(&x).unwrap();
    let term2 = c6.try_mul(&y).unwrap();
    let sub = term1.try_sub(&term2).unwrap();
    let div = sub.try_div(&c3).unwrap();

    let result = matcher.rewrite(&div, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be (4*x) + (-(2*y)) in Tinygrad-style subtraction form.
        if let Op::Binary(BinaryOp::Add, left, right) = rewritten.op() {
            // Check left: 4*x
            if let Op::Binary(BinaryOp::Mul, c, var) = left.op() {
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(4));
                }
                assert!(Arc::ptr_eq(var, &x));
            }

            // Check right: (2*y) * -1
            if let Op::Binary(BinaryOp::Mul, inner, neg_one) = right.op() {
                assert!(matches!(neg_one.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-1)));
                if let Op::Binary(BinaryOp::Mul, c, var) = inner.op() {
                    if let Op::Const(cv) = c.op() {
                        assert_eq!(cv.0, ConstValue::Int(2));
                    }
                    assert!(Arc::ptr_eq(var, &y));
                } else {
                    panic!("Expected inner Mul on right, got {:?}", inner.op());
                }
            } else {
                panic!("Expected negated Mul on right, got {:?}", right.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_distribute_multiplication_over_weak_addition_commutes() {
    let x = UOp::var("x", DType::WeakInt, 0, i64::MAX);
    let c5 = UOp::const_(DType::WeakInt, ConstValue::Int(5));
    let c2 = UOp::const_(DType::WeakInt, ConstValue::Int(2));
    for add in [x.try_add(&c5).unwrap(), c5.try_add(&x).unwrap()] {
        for mul in [c2.try_mul(&add).unwrap(), add.try_mul(&c2).unwrap()] {
            let RewriteResult::Rewritten(result) = term_combining_dsl_patterns().rewrite(&mul, &mut ()) else {
                panic!("expected weak multiplication distribution for {}", mul.tree());
            };
            assert!(matches!(result.op(), Op::Binary(BinaryOp::Add, ..)), "{}", result.tree());
        }
    }
}

#[test]
fn test_weak_distribution_preserves_negation_rule_priority() {
    let x = UOp::var("x", DType::WeakInt, -100, 100);
    let c5 = UOp::const_(DType::WeakInt, ConstValue::Int(5));
    let neg_one = UOp::const_(DType::WeakInt, ConstValue::Int(-1));
    let mul = neg_one.try_mul(&x.try_add(&c5).unwrap()).unwrap();

    let RewriteResult::Rewritten(result) = term_combining_dsl_patterns().rewrite(&mul, &mut ()) else {
        panic!("expected specific negation distribution");
    };
    let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() else { panic!("expected Add, got {}", result.tree()) };
    assert!(
        matches!(lhs.op(), Op::Binary(BinaryOp::Mul, value, c) if Arc::ptr_eq(value, &x) && matches!(c.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-1)))
    );
    assert!(matches!(rhs.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-5)));
}

#[test]
fn test_distribute_weak_addition_over_multiplication_commutes() {
    let x = UOp::var("x", DType::WeakInt, 0, i64::MAX);
    let y = UOp::var("y", DType::WeakInt, 0, i64::MAX);
    let c3 = UOp::const_(DType::WeakInt, ConstValue::Int(3));
    for add in [x.try_add(&y).unwrap(), y.try_add(&x).unwrap()] {
        for mul in [add.try_mul(&c3).unwrap(), c3.try_mul(&add).unwrap()] {
            let RewriteResult::Rewritten(result) = sym_phase3_patterns().rewrite(&mul, &mut ()) else {
                panic!("expected weak multiplication distribution for {}", mul.tree());
            };
            assert!(matches!(result.op(), Op::Binary(BinaryOp::Add, ..)), "{}", result.tree());
        }
    }
}

#[test]
fn test_multiplication_distribution_does_not_match_concrete_int() {
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let y = UOp::var("y", DType::Int32, 0, i64::MAX);
    let mul = x.try_add(&y).unwrap().try_mul(&UOp::native_const(3i32)).unwrap();

    assert!(matches!(sym_phase3_patterns().rewrite(&mul, &mut ()), RewriteResult::NoMatch));

    let add_const = x.try_add(&UOp::native_const(5i32)).unwrap();
    let mul_const = UOp::native_const(3i32).try_mul(&add_const).unwrap();
    assert!(matches!(term_combining_dsl_patterns().rewrite(&mul_const, &mut ()), RewriteResult::NoMatch));
}

// ========== Compositional Optimization Tests ==========

#[test]
#[ignore = "Distribution patterns conflict with compositional optimization"]
fn test_compositional_optimization_minimal_failure() {
    // Reproduces the exact failing case from the property test
    // Input: ((0 + var("a")) * 2) * 2
    // Expected: var("a") * 4
    // Direct optimization should give better or equal results to compositional
    //
    // NOTE: This test is ignored for the same reason as compositional_subexpr_optimization:
    // distribution patterns increase operation count but may enable other optimizations.

    use crate::rewrite::graph_rewrite;
    let matcher = symbolic_simple();

    // Build the expression: (0 + var("a")) * 2
    let a_var = UOp::var("a", DType::Int32, 0, 1);
    let zero = UOp::native_const(0i32);
    let two = UOp::native_const(2i32);
    let add = zero.try_add(&a_var).unwrap();
    let a = add.try_mul(&two).unwrap();
    let b = two.clone();

    // === DIRECT PATH ===
    // Build expression with un-optimized subexpressions and optimize
    let expr_unopt = a.try_mul(&b).unwrap();
    let direct_opt = graph_rewrite(&matcher, expr_unopt, &mut ());

    // === COMPOSITIONAL PATH ===
    // Optimize subexpressions first
    let opt_a = graph_rewrite(&matcher, a.clone(), &mut ());
    let opt_b = graph_rewrite(&matcher, b.clone(), &mut ());

    // Build expression with optimized subexpressions
    let expr_opt_subs = opt_a.try_mul(&opt_b).unwrap();

    // Optimize the composed expression
    let final_opt = graph_rewrite(&matcher, expr_opt_subs, &mut ());

    // Count operations
    fn count_ops(uop: &Arc<UOp>) -> usize {
        match uop.op() {
            Op::Binary(_, left, right) => 1 + count_ops(left) + count_ops(right),
            Op::Unary(_, src) => 1 + count_ops(src),
            Op::Ternary(_, a, b, c) => 1 + count_ops(a) + count_ops(b) + count_ops(c),
            _ => 0,
        }
    }

    let direct_count = count_ops(&direct_opt);
    let final_count = count_ops(&final_opt);

    println!("=== COMPOSITIONAL OPTIMIZATION DEBUG ===");
    println!("Original a: (0 + var(\"a\")) * 2");
    println!("Original b: 2");
    println!("Full expr: ((0 + var(\"a\")) * 2) * 2");
    println!();
    println!("Optimized a: {:?}", opt_a.op());
    println!("Optimized b: {:?}", opt_b.op());
    println!();
    println!("Direct optimization: {} ops -> {:?}", direct_count, direct_opt.op());
    println!("Compositional optimization: {} ops -> {:?}", final_count, final_opt.op());
    println!();

    // The compositional approach should be nearly as good as direct
    // EXPECTED: Both should optimize to var("a") * 4 (1 operation)
    // ACTUAL: Compositional gives worse results
    assert!(
        final_count <= direct_count + 1,
        "Compositional optimization ({} ops) should be nearly as good as direct ({} ops)",
        final_count,
        direct_count
    );
}

#[test]
fn test_multiplication_chain_folding() {
    // Test: (var("a") * 2) * 2 → var("a") * 4
    // This is the simplified version of the failing case

    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 100);
    let c2 = UOp::native_const(2i32);

    // Build (a * 2) * 2
    let mul1 = a.try_mul(&c2).unwrap();
    let mul2 = mul1.try_mul(&c2).unwrap();

    let result = matcher.rewrite(&mul2, &mut ());

    println!("=== MULTIPLICATION CHAIN TEST ===");
    println!("Input: (var(\"a\") * 2) * 2");
    match &result {
        crate::pattern::RewriteResult::Rewritten(r) => {
            println!("Result: {:?}", r.op());
        }
        _ => {
            println!("Result: No rewrite");
        }
    }

    assert!(matches!(result, crate::pattern::RewriteResult::Rewritten(_)));

    if let crate::pattern::RewriteResult::Rewritten(rewritten) = result {
        // Should be a * 4
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a), "Variable should be unchanged");
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(4), "Constant should be folded to 4");
            } else {
                panic!("Expected constant 4, got {:?}", c.op());
            }
        } else {
            panic!("Expected Binary(Mul, a, 4), got {:?}", rewritten.op());
        }
    }
}

// ====== Tests for BOOLEAN patterns (boolean_dsl_patterns) ======

#[test]
fn test_double_not_elimination() {
    // !!x → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let not_x = x.not();
    let not_not_x = not_x.not();

    let result = matcher.rewrite(&not_not_x, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_double_not_int() {
    // !!x → x (for integers - bitwise NOT)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let not_x = x.not();
    let not_not_x = not_x.not();

    let result = matcher.rewrite(&not_not_x, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_xor_self_cancellation() {
    // x ^ x → 0
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let xor_self = x.try_xor_op(&x).unwrap();

    let result = matcher.rewrite(&xor_self, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(0));
        } else {
            panic!("Expected constant 0");
        }
    }
}

// ====== Tests for NEGATION patterns (negation_dsl_patterns) ======

#[test]
fn test_double_neg_elimination() {
    // -(-x) → x
    // neg() produces MUL(x, -1), so -(-x) = MUL(MUL(x, -1), -1).
    // Folds via two-stage ALU (in symbolic tier 2): MUL(MUL(x, c1), c2) → MUL(x, c1*c2) = MUL(x, 1) → x.
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let neg_x = x.neg();
    let neg_neg_x = neg_x.neg();

    let result = graph_rewrite(&matcher, neg_neg_x, &mut ());
    assert!(Arc::ptr_eq(&result, &x), "double neg should simplify back to x, got: {}", result.tree());
}

#[test]
fn test_double_neg_float() {
    // -(-x) → x (for floats)
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);
    let neg_x = x.neg();
    let neg_neg_x = neg_x.neg();

    let result = graph_rewrite(&matcher, neg_neg_x, &mut ());
    assert!(Arc::ptr_eq(&result, &x), "double neg should simplify back to x, got: {}", result.tree());
}

// ====== Test: propagate_invalid through neg (MUL) ======

#[test]
fn test_propagate_invalid_through_neg() {
    // Core regression test for the Neg→MUL(-1) change.
    // neg(WHERE(cond, x, Invalid)) = MUL(WHERE(cond, x, Invalid), -1)
    // propagate_invalid pushes Binary through WHERE-Invalid:
    // → WHERE(cond, MUL(x, -1), Invalid)
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::patterns::propagate_invalid;
    let matcher = propagate_invalid();

    let cond = UOp::var("c", DType::Bool, 0, 1);
    let x = UOp::var("x", DType::Index, 0, 100);
    let invalid = UOp::invalid_marker();
    let gated = UOp::try_where(cond.clone(), x.clone(), invalid.clone()).unwrap();
    let negated = gated.neg(); // MUL(WHERE(cond, x, Invalid), -1)

    let result = graph_rewrite(&matcher, negated, &mut ());

    // Should be WHERE(cond, MUL(x, -1), Invalid)
    let Op::Ternary(svod_ir::TernaryOp::Where, c, inner, inv) = result.op() else {
        panic!("Expected WHERE, got: {}", result.tree());
    };
    assert!(Arc::ptr_eq(c, &cond), "condition should be preserved");
    assert!(UOp::is_invalid_marker(inv), "false branch should be Invalid");
    assert!(
        matches!(inner.op(), Op::Binary(svod_ir::BinaryOp::Mul, _, _)),
        "true branch should be MUL(x, -1), got: {}",
        inner.tree()
    );
}

#[test]
fn test_propagate_invalid_through_cast_preserves_gate() {
    use crate::symbolic::patterns::propagate_invalid;

    let cond = UOp::var("c", DType::Bool, 0, 1);
    let x = UOp::var("x", DType::Float16, 0, 100);
    let gated = UOp::try_where(cond.clone(), x, UOp::invalid_marker()).unwrap();
    let result = graph_rewrite(propagate_invalid(), gated.cast(DType::Float32), &mut ());

    let Op::Ternary(TernaryOp::Where, result_cond, value, invalid) = result.op() else {
        panic!("expected gated cast, got: {}", result.tree());
    };
    assert!(Arc::ptr_eq(result_cond, &cond));
    assert!(matches!(value.op(), Op::Cast { .. }));
    assert!(UOp::is_invalid_marker(invalid));
    assert_eq!(invalid.dtype(), DType::Bool);
}

#[test]
fn test_propagate_invalid_through_comparison_preserves_gate() {
    use crate::symbolic::patterns::propagate_invalid;

    let cond = UOp::var("c", DType::Bool, 0, 1);
    let x = UOp::var("x", DType::Float32, 0, 100);
    let gated = UOp::try_where(cond.clone(), x, UOp::invalid_marker()).unwrap();
    let result = graph_rewrite(propagate_invalid(), gated.lt(&UOp::native_const(1.0f32)), &mut ());

    let Op::Ternary(TernaryOp::Where, result_cond, value, invalid) = result.op() else {
        panic!("expected gated comparison, got: {}", result.tree());
    };
    assert!(Arc::ptr_eq(result_cond, &cond));
    assert!(matches!(value.op(), Op::Binary(BinaryOp::Lt, _, _)));
    assert!(UOp::is_invalid_marker(invalid));
    assert_eq!(invalid.dtype(), DType::Bool);

    // A bare Invalid poisons a non-comparison binary from either side, but a
    // comparison keeps it as an operand (tinygrad uop/symbolic.py:75-77).
    // Invalid only reaches an operand slot through source reconstruction, so build
    // the poisoned nodes directly rather than through the promoting constructors.
    let index = UOp::var("i", DType::Index, 0, 100);
    let marker = UOp::invalid_marker();
    let binary = |op, lhs: &Arc<UOp>, rhs: &Arc<UOp>| UOp::new(Op::Binary(op, lhs.clone(), rhs.clone()), DType::Index);
    for poisoned in [binary(BinaryOp::Sub, &index, &marker), binary(BinaryOp::Sub, &marker, &index)] {
        assert!(UOp::is_invalid_marker(&graph_rewrite(propagate_invalid(), poisoned, &mut ())));
    }
    let compared = UOp::new(Op::Binary(BinaryOp::Lt, index, marker), DType::Bool);
    assert!(matches!(graph_rewrite(propagate_invalid(), compared, &mut ()).op(), Op::Binary(BinaryOp::Lt, _, _)));
}

#[test]
fn test_remove_typed_invalid_lanes_at_final_cleanup() {
    use crate::symbolic::patterns::pm_remove_invalid;

    let invalid = UOp::invalid_marker();
    let one = UOp::const_(DType::Float16, ConstValue::Float(1.0));
    let result = graph_rewrite(pm_remove_invalid(), UOp::stack(vec![invalid, one].into()), &mut ());

    assert!(!result.any_in_subtree(UOp::is_invalid_marker));
    let Op::Stack { sources: elements } = result.op() else { panic!("expected VECTORIZE, got: {}", result.tree()) };
    assert!(matches!(elements[0].op(), Op::Const(cv) if cv.0 == ConstValue::Float(0.0)));
}

#[test]
fn single_valued_bounds_collapse_products_but_not_sums() {
    use crate::symbolic::patterns::vmin_vmax_collapse_patterns;

    let a = UOp::var("a", DType::Int32, 2, 2);
    let b = UOp::var("b", DType::Int32, 3, 3);
    let collapse = |root| graph_rewrite(vmin_vmax_collapse_patterns(), root, &mut ());

    let product = collapse(a.try_mul(&b).unwrap());
    assert!(matches!(product.op(), Op::Const(value) if value.0 == ConstValue::Int(6)));

    // Add/Sub/Max stay: collapsing them replicates the trivial-RANGE collapse that
    // breaks a hand-built kernel's trip-1 loop carry (tinygrad uop/symbolic.py:248).
    for sum in [a.try_add(&b).unwrap(), a.try_sub(&b).unwrap(), a.try_max(&b).unwrap()] {
        assert!(matches!(collapse(sum).op(), Op::Binary(..)));
    }
}

// ====== Tests for MINMAX patterns (minmax_dsl_patterns) ======

#[test]
fn test_max_self_identity() {
    // max(x, x) → x
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let max_self = x.try_max(&x).unwrap();

    let result = matcher.rewrite(&max_self, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_max_self_float() {
    // max(x, x) → x (for floats)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);
    let max_self = x.try_max(&x).unwrap();

    let result = matcher.rewrite(&max_self, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

// ====== Tests for POWER patterns (power_dsl_patterns) ======

#[test]
fn test_pow_zero_is_one() {
    // x ** 0 → 1
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let zero = UOp::native_const(0i32);
    let pow = x.try_pow(&zero).unwrap();

    let result = matcher.rewrite(&pow, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Int(1));
        } else {
            panic!("Expected constant 1");
        }
    }
}

#[test]
fn test_pow_one_is_identity() {
    // x ** 1 → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let one = UOp::native_const(1i32);
    let pow = x.try_pow(&one).unwrap();

    let result = matcher.rewrite(&pow, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_pow_float_zero() {
    // x ** 0.0 → 1.0
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Float32, 0, 100);
    let zero = UOp::native_const(0.0f32);
    let pow = x.try_pow(&zero).unwrap();

    let result = matcher.rewrite(&pow, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Float(1.0));
        } else {
            panic!("Expected constant 1.0");
        }
    }
}

// ====== Tests for WHERE/DCE patterns (dce_dsl_patterns) ======

#[test]
fn test_where_same_branches() {
    // where(cond, x, x) → x
    let matcher = symbolic_simple();
    let cond = UOp::var("cond", DType::Bool, 0, 1);
    let x = UOp::var("x", DType::Int32, 0, 100);
    let where_op = UOp::try_where(cond, Arc::clone(&x), Arc::clone(&x)).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_where_bool_true_false() {
    // where(x, true, false) → x (for bool x)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let true_val = UOp::native_const(true);
    let false_val = UOp::native_const(false);
    let where_op = UOp::try_where(Arc::clone(&x), true_val, false_val).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_where_bool_false_true() {
    // where(x, false, true) → !x (for bool x)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let false_val = UOp::native_const(false);
    let true_val = UOp::native_const(true);
    let where_op = UOp::try_where(Arc::clone(&x), false_val, true_val).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be Not(x)
        if let Op::Unary(UnaryOp::Not, inner) = rewritten.op() {
            assert!(Arc::ptr_eq(inner, &x));
        } else {
            panic!("Expected Not(x)");
        }
    }
}

#[test]
fn test_where_negated_condition() {
    // where(!cond, t, f) → where(cond, f, t)
    let matcher = symbolic();
    let cond = UOp::var("cond", DType::Bool, 0, 1);
    let not_cond = cond.not();
    let t = UOp::var("t", DType::Int32, 0, 100);
    let f = UOp::var("f", DType::Int32, 0, 100);
    let where_op = UOp::try_where(not_cond, Arc::clone(&t), Arc::clone(&f)).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        // Should be Where(cond, f, t) - branches swapped
        if let Op::Ternary(TernaryOp::Where, new_cond, new_t, new_f) = rewritten.op() {
            assert!(Arc::ptr_eq(new_cond, &cond));
            assert!(Arc::ptr_eq(new_t, &f)); // swapped
            assert!(Arc::ptr_eq(new_f, &t)); // swapped
        } else {
            panic!("Expected Where with swapped branches");
        }
    }
}

#[test]
fn test_where_const_true_condition() {
    // where(true, t, f) → t
    let matcher = symbolic_simple();
    let true_cond = UOp::native_const(true);
    let t = UOp::var("t", DType::Int32, 0, 100);
    let f = UOp::var("f", DType::Int32, 0, 100);
    let where_op = UOp::try_where(true_cond, Arc::clone(&t), f).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &t));
    }
}

#[test]
fn test_where_const_false_condition() {
    // where(false, t, f) → f
    let matcher = symbolic_simple();
    let false_cond = UOp::native_const(false);
    let t = UOp::var("t", DType::Int32, 0, 100);
    let f = UOp::var("f", DType::Int32, 0, 100);
    let where_op = UOp::try_where(false_cond, t, Arc::clone(&f)).unwrap();

    let result = matcher.rewrite(&where_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &f));
    }
}

// ========== Phase 1.1: Bounds-Based Comparison Tests ==========
//
// These tests verify that the ComparisonAnalyzer correctly simplifies
// comparisons based on known variable ranges.

#[test]
fn test_lt_bounds_always_true() {
    // a(0,8) < 77 → true (since max(a)=8 < 77)
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 8); // range [0, 8]
    let c77 = UOp::native_const(77i32);
    let lt = a.try_cmplt(&c77).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(true));
        } else {
            panic!("Expected Const(Bool(true)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_lt_bounds_always_true_edge() {
    // a(0,8) < 9 → true (since max(a)=8 < 9)
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 8);
    let c9 = UOp::native_const(9i32);
    let lt = a.try_cmplt(&c9).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(true));
        } else {
            panic!("Expected Const(Bool(true)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_lt_bounds_indeterminate() {
    // a(0,8) < 5 → indeterminate (could be 0 < 5 = true or 8 < 5 = false)
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Int32, 0, 8);
    let c5 = UOp::native_const(5i32);
    let lt = a.try_cmplt(&c5).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    // Should NOT be rewritten since the result is indeterminate
    assert!(matches!(result, RewriteResult::NoMatch));
}

#[test]
fn test_lt_bounds_always_false() {
    // a(0,8) < 0 → false (since min(a)=0 is not < 0)
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 8);
    let c0 = UOp::native_const(0i32);
    let lt = a.try_cmplt(&c0).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_lt_two_vars_always_true() {
    // a(0,4) < b(5,10) → true (since max(a)=4 < min(b)=5)
    // We create b(5,10) as b(0,5) + 5
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 4); // range [0, 4]
    let b_base = UOp::var("b", DType::Int32, 0, 5); // range [0, 5]
    let c5 = UOp::native_const(5i32);
    let b = b_base.try_add(&c5).unwrap(); // range [5, 10]

    let lt = a.try_cmplt(&b).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(true));
        } else {
            panic!("Expected Const(Bool(true)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_lt_two_vars_always_false() {
    // a(5,10) < b(0,4) → false (since min(a)=5 >= max(b)=4, so 5 < 4 is false)
    // We create a(5,10) as a(0,5) + 5
    let matcher = symbolic();
    let a_base = UOp::var("a", DType::Int32, 0, 5); // range [0, 5]
    let c5 = UOp::native_const(5i32);
    let a = a_base.try_add(&c5).unwrap(); // range [5, 10]
    let b = UOp::var("b", DType::Int32, 0, 4); // range [0, 4]

    let lt = a.try_cmplt(&b).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_ge_bounds_always_true() {
    // a(3,8) >= 3 → true (since min(a)=3 >= 3)
    // We create a(3,8) as a(0,5) + 3
    let matcher = symbolic();
    let a_base = UOp::var("a", DType::Int32, 0, 5); // range [0, 5]
    let c3 = UOp::native_const(3i32);
    let a = a_base.try_add(&c3).unwrap(); // range [3, 8]

    // a >= 3 is equivalent to !(a < 3), but we test via constants
    // Since there's no cmpge, we test a < 3 and expect false
    let lt = a.try_cmplt(&c3).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            // a(3,8) < 3 should be false (min(a)=3 is not < 3)
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_eq_bounds_always_false() {
    // a(0,4) == b(10,20) → false (non-overlapping ranges)
    // We create b(10,20) as b(0,10) + 10
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 4); // range [0, 4]
    let b_base = UOp::var("b", DType::Int32, 0, 10); // range [0, 10]
    let c10 = UOp::native_const(10i32);
    let b = b_base.try_add(&c10).unwrap(); // range [10, 20]

    let eq = a.try_cmpeq(&b).unwrap();

    let result = matcher.rewrite(&eq, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_ne_bounds_always_true() {
    // a(0,4) != b(10,20) → true (non-overlapping ranges)
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 4); // range [0, 4]
    let b_base = UOp::var("b", DType::Int32, 0, 10);
    let c10 = UOp::native_const(10i32);
    let b = b_base.try_add(&c10).unwrap(); // range [10, 20]

    let ne = a.try_cmpne(&b).unwrap();

    let result = matcher.rewrite(&ne, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(true));
        } else {
            panic!("Expected Const(Bool(true)), got {:?}", rewritten.op());
        }
    }
}

// ========== Phase 1.2: Nested Operation Tests ==========
//
// These tests verify that nested operations are correctly simplified
// using existing patterns.

#[test]
fn test_nested_div_div() {
    // (a // 10) // 9 → a // 90
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c10 = UOp::native_const(10i32);
    let c9 = UOp::native_const(9i32);
    let div1 = a.try_div(&c10).unwrap();
    let div2 = div1.try_div(&c9).unwrap();

    let result = matcher.rewrite(&div2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::FloorDiv, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(90));
            } else {
                panic!("Expected constant 90, got {:?}", c.op());
            }
        } else {
            panic!("Expected FloorDiv, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_nested_mul_mul() {
    // (a * 10) * 9 → a * 90
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c10 = UOp::native_const(10i32);
    let c9 = UOp::native_const(9i32);
    let mul1 = a.try_mul(&c10).unwrap();
    let mul2 = mul1.try_mul(&c9).unwrap();

    let result = matcher.rewrite(&mul2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(90));
            } else {
                panic!("Expected constant 90, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_nested_mod_mod_same_divisor() {
    // (a % 5) % 5 → a % 5 (idempotent modulo)
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c5 = UOp::native_const(5i32);
    let mod1 = a.try_mod(&c5).unwrap();
    let mod2 = mod1.try_mod(&c5).unwrap();

    let result = matcher.rewrite(&mod2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::FloorMod, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            assert!(Arc::ptr_eq(c, &c5));
        } else {
            panic!("Expected FloorMod(a, 5), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_nested_add_add() {
    // (a + 3) + 5 → a + 8
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let add1 = a.try_add(&c3).unwrap();
    let add2 = add1.try_add(&c5).unwrap();

    let result = matcher.rewrite(&add2, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(8));
            } else {
                panic!("Expected constant 8, got {:?}", c.op());
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_nested_sub_sub() {
    // (a - 3) - 5 → a + (-8)
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let sub1 = a.try_sub(&c3).unwrap();
    let sub2 = sub1.try_sub(&c5).unwrap();

    let rewritten = graph_rewrite(matcher, sub2, &mut ());
    if let Op::Binary(BinaryOp::Add, var, c) = rewritten.op() {
        assert!(Arc::ptr_eq(var, &a));
        if let Op::Const(cv) = c.op() {
            assert_eq!(cv.0, ConstValue::Int(-8));
        } else {
            panic!("Expected constant -8, got {:?}", c.op());
        }
    } else {
        panic!("Expected Add, got {:?}", rewritten.op());
    }
}

// ========== Phase 2: Comparison & Boolean Patterns ==========
//
// Tests for new comparison and boolean patterns.

#[test]
fn test_bool_or_not_tautology() {
    // x | !x → true (for bool type)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Bool, 0, 1); // bool variable
    let not_x = x.not();
    let or_op = x.try_or_op(&not_x).unwrap();

    let result = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(true));
        } else {
            panic!("Expected Const(Bool(true)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_and_not_contradiction() {
    // x & !x → false (for bool type)
    let matcher = symbolic();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let not_x = x.not();
    let and_op = x.try_and_op(&not_x).unwrap();

    let result = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_or_true_absorb() {
    // true | x → true
    let matcher = symbolic();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let true_const = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let or_op = true_const.try_or_op(&x).unwrap();

    let result = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(true));
        } else {
            panic!("Expected Const(Bool(true)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_and_false_absorb() {
    // false & x → false
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let false_const = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let and_op = false_const.try_and_op(&x).unwrap();

    let result = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Const(cv) = rewritten.op() {
            assert_eq!(cv.0, ConstValue::Bool(false));
        } else {
            panic!("Expected Const(Bool(false)), got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_and_true_identity() {
    // true & x → x
    let matcher = symbolic();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let true_const = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let and_op = true_const.try_and_op(&x).unwrap();

    let result = matcher.rewrite(&and_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_bool_or_false_identity() {
    // false | x → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let false_const = UOp::const_(DType::Bool, ConstValue::Bool(false));
    let or_op = false_const.try_or_op(&x).unwrap();

    let result = matcher.rewrite(&or_op, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_lt_const_offset() {
    // (a + 2) < 5 → a < 3
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c2 = UOp::native_const(2i32);
    let c5 = UOp::native_const(5i32);
    let add = a.try_add(&c2).unwrap();
    let lt = add.try_cmplt(&c5).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Lt, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(3)); // 5 - 2 = 3
            } else {
                panic!("Expected constant 3, got {:?}", c.op());
            }
        } else {
            panic!("Expected Lt, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_lt_const_offset_negative() {
    // (a + 10) < 5 → a < -5
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c10 = UOp::native_const(10i32);
    let c5 = UOp::native_const(5i32);
    let add = a.try_add(&c10).unwrap();
    let lt = add.try_cmplt(&c5).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Lt, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &a));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(-5)); // 5 - 10 = -5
            } else {
                panic!("Expected constant -5, got {:?}", c.op());
            }
        } else {
            panic!("Expected Lt, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_lt_negation_flip() {
    // -a < -b → b < a
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let b = UOp::var("b", DType::Int32, 0, i64::MAX);
    let neg_a = a.neg();
    let neg_b = b.neg();
    let lt = neg_a.try_cmplt(&neg_b).unwrap();

    let result = matcher.rewrite(&lt, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Lt, lhs, rhs) = rewritten.op() {
            // Should become b < a
            assert!(Arc::ptr_eq(lhs, &b));
            assert!(Arc::ptr_eq(rhs, &a));
        } else {
            panic!("Expected Lt(b, a), got {:?}", rewritten.op());
        }
    }
}

// ===== Phase 3: Division/Modulo Recombination Tests =====

#[test]
fn test_div_mod_recombine() {
    // x%n + (x//n)*n → x
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let n = UOp::native_const(4i32);

    // Build: x % 4 + (x // 4) * 4
    let mod_part = x.try_mod(&n).unwrap();
    let div_part = x.try_div(&n).unwrap();
    let mul_part = div_part.try_mul(&n).unwrap();
    let add = mod_part.try_add(&mul_part).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_div_mod_recombine_commutative() {
    // (x//n)*n + x%n → x (commutative form)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let n = UOp::native_const(4i32);

    // Build: (x // 4) * 4 + x % 4
    let div_part = x.try_div(&n).unwrap();
    let mul_part = div_part.try_mul(&n).unwrap();
    let mod_part = x.try_mod(&n).unwrap();
    let add = mul_part.try_add(&mod_part).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x));
    }
}

#[test]
fn test_nested_div_const() {
    // (a//2 + 1) // 2 → (a + 2) // 4
    // Pattern is in symbolic tier 2 (advanced_division), not symbolic_simple,
    // to avoid infinite loop with fast_division_patterns in Stage 18-19.
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, 100);
    let c2 = UOp::native_const(2i32);
    let c1 = UOp::native_const(1i32);

    // Build: (a // 2 + 1) // 2
    let div_inner = a.try_div(&c2).unwrap();
    let add = div_inner.try_add(&c1).unwrap();
    let div_outer = add.try_div(&c2).unwrap();

    let result = matcher.rewrite(&div_outer, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should become (a + 2) // 4
        if let Op::Binary(BinaryOp::FloorDiv, lhs, rhs) = rewritten.op() {
            // lhs should be a + 2
            if let Op::Binary(BinaryOp::Add, var, c) = lhs.op() {
                assert!(Arc::ptr_eq(var, &a));
                if let Op::Const(cv) = c.op() {
                    assert_eq!(cv.0, ConstValue::Int(2)); // c1 * c2 = 1 * 2 = 2
                } else {
                    panic!("Expected constant 2, got {:?}", c.op());
                }
            } else {
                panic!("Expected Add, got {:?}", lhs.op());
            }
            // rhs should be 4
            if let Op::Const(cv) = rhs.op() {
                assert_eq!(cv.0, ConstValue::Int(4)); // c1 * c3 = 2 * 2 = 4
            } else {
                panic!("Expected constant 4, got {:?}", rhs.op());
            }
        } else {
            panic!("Expected FloorDiv, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_nested_div_const_larger() {
    // (a//3 + 5) // 4 → (a + 15) // 12
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic();
    let a = UOp::var("a", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let c5 = UOp::native_const(5i32);
    let c4 = UOp::native_const(4i32);

    // Build: (a // 3 + 5) // 4
    let div_inner = a.try_div(&c3).unwrap();
    let add = div_inner.try_add(&c5).unwrap();
    let div_outer = add.try_div(&c4).unwrap();

    let result = graph_rewrite(&matcher, div_outer, &mut ());

    // graph_rewrite with full symbolic may simplify further (e.g. fold_divmod_congruence).
    // Verify the result is equivalent: either (a+15)//12 or a further-simplified form.
    // The key property: the nested division should not survive.
    assert!(
        !matches!(result.op(), Op::Binary(BinaryOp::FloorDiv, lhs, _) if matches!(lhs.op(), Op::Binary(BinaryOp::Add, inner, _) if matches!(inner.op(), Op::Binary(BinaryOp::FloorDiv, _, _)))),
        "Nested (a//c1 + c2) // c3 should be simplified, got: {}",
        result.tree()
    );
}

#[test]
fn test_div_mod_recombine_different_n() {
    // x%4 + (x//5)*4 should NOT simplify (different divisors)
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let n4 = UOp::native_const(4i32);
    let n5 = UOp::native_const(5i32);

    // Build: x % 4 + (x // 5) * 4
    let mod_part = x.try_mod(&n4).unwrap();
    let div_part = x.try_div(&n5).unwrap();
    let mul_part = div_part.try_mul(&n4).unwrap();
    let add = mod_part.try_add(&mul_part).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    // Should NOT rewrite to x because divisors don't match
    assert!(!matches!(result, RewriteResult::Rewritten(ref r) if Arc::ptr_eq(r, &x)));
}

#[test]
fn test_div_mod_property_identity() {
    // For any x, n > 0: x%n + (x//n)*n == x
    // This is a quick property spot-check with concrete values
    let x_val = 17i32;
    let n_val = 5i32;

    let mod_result = x_val % n_val; // 2
    let div_result = x_val / n_val; // 3
    let recombined = mod_result + div_result * n_val; // 2 + 15 = 17

    assert_eq!(recombined, x_val);
}

// ===== Phase 4: Where/Branch Pattern Tests =====

#[test]
fn test_where_merge_branches() {
    // where(a, where(b, c, d), d) → where(a & b, c, d)
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Bool, 0, 1);
    let b = UOp::var("b", DType::Bool, 0, 1);
    let c = UOp::var("c", DType::Int32, 0, i64::MAX);
    let d = UOp::var("d", DType::Int32, 0, i64::MAX);

    // Build: where(a, where(b, c, d), d)
    let inner_where = UOp::try_where(b.clone(), c.clone(), d.clone()).unwrap();
    let outer_where = UOp::try_where(a.clone(), inner_where, d.clone()).unwrap();

    let result = matcher.rewrite(&outer_where, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should become where(a & b, c, d)
        if let Op::Ternary(TernaryOp::Where, cond, true_val, false_val) = rewritten.op() {
            // Check condition is a & b
            if let Op::Binary(BinaryOp::And, lhs, rhs) = cond.op() {
                assert!(Arc::ptr_eq(lhs, &a) || Arc::ptr_eq(lhs, &b));
                assert!(Arc::ptr_eq(rhs, &a) || Arc::ptr_eq(rhs, &b));
            } else {
                panic!("Expected And condition, got {:?}", cond.op());
            }
            // True branch should be c
            assert!(Arc::ptr_eq(true_val, &c));
            // False branch should be d
            assert!(Arc::ptr_eq(false_val, &d));
        } else {
            panic!("Expected Where, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_where_merge_branches_no_match() {
    // where(a, where(b, c, d), e) should NOT simplify (d != e)
    let matcher = symbolic_simple();
    let a = UOp::var("a", DType::Bool, 0, 1);
    let b = UOp::var("b", DType::Bool, 0, 1);
    let c = UOp::var("c", DType::Int32, 0, i64::MAX);
    let d = UOp::var("d", DType::Int32, 0, i64::MAX);
    let e = UOp::var("e", DType::Int32, 0, i64::MAX);

    // Build: where(a, where(b, c, d), e)
    let inner_where = UOp::try_where(b.clone(), c.clone(), d.clone()).unwrap();
    let outer_where = UOp::try_where(a.clone(), inner_where.clone(), e.clone()).unwrap();

    let result = matcher.rewrite(&outer_where, &mut ());
    // May or may not rewrite, but if it does, should NOT be where(a&b, c, _)
    if let RewriteResult::Rewritten(rewritten) = &result
        && let Op::Ternary(TernaryOp::Where, cond, _, _) = rewritten.op()
    {
        // If rewritten, condition should NOT be And(a, b)
        if let Op::Binary(BinaryOp::And, _, _) = cond.op() {
            panic!("Should not merge branches when false values differ");
        }
    }
}

#[test]
fn test_cast_where_push() {
    // where(s, a, b).cast(f32) → where(s, a.cast(f32), b.cast(f32))
    let matcher = sym();
    let s = UOp::var("s", DType::Bool, 0, 1);
    let a = UOp::native_const(1i32);
    let b = UOp::native_const(0i32);

    // Build: cast(where(s, a, b), f32)
    let where_op = UOp::try_where(s.clone(), a.clone(), b.clone()).unwrap();
    let cast_where = where_op.cast(DType::Float32);

    let result = matcher.rewrite(&cast_where, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        // Should become where(s, cast(a, f32), cast(b, f32))
        if let Op::Ternary(TernaryOp::Where, cond, true_val, false_val) = rewritten.op() {
            assert!(Arc::ptr_eq(cond, &s));
            // True branch should be cast
            assert!(matches!(true_val.op(), Op::Cast { .. }));
            // False branch should be cast
            assert!(matches!(false_val.op(), Op::Cast { .. }));
        } else {
            panic!("Expected Where, got {:?}", rewritten.op());
        }
    }
}

// ========== Batch A+B: New Pattern Tests ==========

// --- A1: vmin==vmax collapse ---

// --- A3: Bool arithmetic ---

#[test]
fn test_bool_mul_is_and() {
    // Bool * Bool → AND
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let y = UOp::var("y", DType::Bool, 0, 1);
    let mul = x.try_mul(&y).unwrap();

    let result = matcher.rewrite(&mul, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::And, lhs, rhs) = rewritten.op() {
            assert!(Arc::ptr_eq(lhs, &x));
            assert!(Arc::ptr_eq(rhs, &y));
        } else {
            panic!("Expected And, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_add_is_or() {
    // Bool + Bool → OR
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let y = UOp::var("y", DType::Bool, 0, 1);
    let add = x.try_add(&y).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Or, lhs, rhs) = rewritten.op() {
            assert!(Arc::ptr_eq(lhs, &x));
            assert!(Arc::ptr_eq(rhs, &y));
        } else {
            panic!("Expected Or, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_max_is_or() {
    // Bool max Bool → OR
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Bool, 0, 1);
    let y = UOp::var("y", DType::Bool, 0, 1);
    let max = x.try_max(&y).unwrap();

    let result = matcher.rewrite(&max, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Or, lhs, rhs) = rewritten.op() {
            assert!(Arc::ptr_eq(lhs, &x));
            assert!(Arc::ptr_eq(rhs, &y));
        } else {
            panic!("Expected Or, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_bool_mul_non_bool_no_match() {
    // Int * Int should NOT become AND
    let matcher = symbolic_simple();
    let x = UOp::var("x", DType::Int32, 0, 100);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let mul = x.try_mul(&y).unwrap();

    let result = matcher.rewrite(&mul, &mut ());
    if let RewriteResult::Rewritten(rewritten) = &result {
        assert!(!matches!(rewritten.op(), Op::Binary(BinaryOp::And, ..)));
    }
}

// --- A2: Term combining new variants ---

#[test]
fn test_term_combine_x_plus_xc() {
    // x + x*3 → x*4
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let c3 = UOp::native_const(3i32);
    let xc = x.try_mul(&c3).unwrap();
    let add = x.try_add(&xc).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Mul, var, c) = rewritten.op() {
            assert!(Arc::ptr_eq(var, &x));
            if let Op::Const(cv) = c.op() {
                assert_eq!(cv.0, ConstValue::Int(4));
            } else {
                panic!("Expected const 4, got {:?}", c.op());
            }
        } else {
            panic!("Expected Mul, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_term_combine_y_plus_x_plus_x() {
    // (y + x) + x → y + x*2
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let y = UOp::var("y", DType::Int32, 0, i64::MAX);
    let yx = y.try_add(&x).unwrap();
    let add = yx.try_add(&x).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Binary(BinaryOp::Add, lhs, rhs) = rewritten.op() {
            assert!(Arc::ptr_eq(lhs, &y));
            if let Op::Binary(BinaryOp::Mul, _, c) = rhs.op()
                && let Op::Const(cv) = c.op()
            {
                assert_eq!(cv.0, ConstValue::Int(2));
            }
        } else {
            panic!("Expected Add, got {:?}", rewritten.op());
        }
    }
}

// --- A4: Negation distribution (const version) ---

#[test]
fn test_neg_one_times_x_plus_const() {
    // (-1) * (x + 3) should be distributed into a sum.
    // Multiple patterns can fire: the const negation pattern produces neg(x) + (-3),
    // while the general distribution pattern produces (-1*x) + (-1*3).
    // Both are valid simplifications; we verify the result is an Add of two terms.
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic();
    let x = UOp::var("x", DType::Int32, 0, i64::MAX);
    let neg_one = UOp::native_const(-1i32);
    let c3 = UOp::native_const(3i32);
    let add = x.try_add(&c3).unwrap();
    let mul = neg_one.try_mul(&add).unwrap();

    // Use graph_rewrite to apply all rewrites (including constant folding of -1*3 → -3)
    let result = graph_rewrite(&matcher, mul, &mut ());
    // After full rewriting: should be Neg(x) + (-3) or similar
    if let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() {
        // lhs should be either Neg(x) or Mul(-1, x)
        match lhs.op() {
            Op::Unary(UnaryOp::Neg, inner) => assert!(Arc::ptr_eq(inner, &x)),
            Op::Binary(BinaryOp::Mul, _, _) => { /* distribution form, also valid */ }
            _ => panic!("Expected Neg(x) or Mul(-1, x), got {:?}", lhs.op()),
        }
        // rhs should be -3 (after constant folding)
        if let Op::Const(cv) = rhs.op() {
            assert_eq!(cv.0, ConstValue::Int(-3));
        } else {
            panic!("Expected const -3, got {:?}", rhs.op());
        }
    } else {
        panic!("Expected Add after full rewrite, got {:?}", result.op());
    }
}

// --- A5: Range%end / Range//end ---

#[test]
fn test_range_mod_end() {
    // Range(end) % end → Range(end)
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic();
    let end = UOp::index_const(8);
    let range = UOp::range(end.clone(), 0);
    let modulo = range.try_mod(&end).unwrap();

    let result = graph_rewrite(&matcher, modulo, &mut ());
    assert!(matches!(result.op(), Op::Range { .. }) || matches!(result.op(), Op::Const(_)));
}

#[test]
fn test_range_div_end() {
    // Range(end) // end → 0
    use crate::rewrite::graph_rewrite;
    let matcher = symbolic();
    let end = UOp::index_const(8);
    let range = UOp::range(end.clone(), 0);
    let div = range.try_div(&end).unwrap();

    let result = graph_rewrite(&matcher, div, &mut ());
    if let Op::Const(cv) = result.op() {
        assert_eq!(cv.0, ConstValue::Int(0));
    } else {
        panic!("Expected const 0, got {:?}", result.op());
    }
}

// --- B1: c0*x < c1 ---

#[test]
fn test_weak_mul_lt_ceil_div() {
    let x = UOp::var("x", DType::WeakInt, -100, 100);
    let c3 = UOp::const_(DType::WeakInt, ConstValue::Int(3));
    let c10 = UOp::const_(DType::WeakInt, ConstValue::Int(10));
    let lt = c3.try_mul(&x).unwrap().try_cmplt(&c10).unwrap();

    let RewriteResult::Rewritten(result) = comparison_dsl_patterns().rewrite(&lt, &mut ()) else {
        panic!("expected weak ceil-div comparison simplification");
    };
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected Lt, got {}", result.tree()) };
    assert!(Arc::ptr_eq(lhs, &x));
    assert_eq!(rhs.dtype(), DType::WeakInt);
    assert!(matches!(rhs.op(), Op::Const(cv) if cv.0 == ConstValue::Int(4)));
}

#[test]
fn test_weak_negative_mul_lt_ceil_div() {
    let x = UOp::var("x", DType::WeakInt, -100, 100);
    let c0 = UOp::const_(DType::WeakInt, ConstValue::Int(-3));
    let c1 = UOp::const_(DType::WeakInt, ConstValue::Int(10));
    let lt = c0.try_mul(&x).unwrap().try_cmplt(&c1).unwrap();

    let RewriteResult::Rewritten(result) = comparison_dsl_patterns().rewrite(&lt, &mut ()) else {
        panic!("expected signed weak ceil-div comparison simplification");
    };
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected Lt, got {}", result.tree()) };
    assert!(
        matches!(lhs.op(), Op::Binary(BinaryOp::Mul, value, c) if Arc::ptr_eq(value, &x) && matches!(c.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-1)))
    );
    assert!(matches!(rhs.op(), Op::Const(cv) if cv.0 == ConstValue::Int(4)));
}

#[test]
fn test_weak_mul_lt_negative_bound_and_commuted_coefficient() {
    let x = UOp::var("x", DType::WeakInt, -100, 100);
    let c0 = UOp::const_(DType::WeakInt, ConstValue::Int(-3));
    let c1 = UOp::const_(DType::WeakInt, ConstValue::Int(-10));
    let lt = x.try_mul(&c0).unwrap().try_cmplt(&c1).unwrap();

    let RewriteResult::Rewritten(result) = comparison_dsl_patterns().rewrite(&lt, &mut ()) else {
        panic!("expected signed weak ceil-div comparison simplification");
    };
    let Op::Binary(BinaryOp::Lt, lhs, rhs) = result.op() else { panic!("expected Lt, got {}", result.tree()) };
    assert!(
        matches!(lhs.op(), Op::Binary(BinaryOp::Mul, value, c) if Arc::ptr_eq(value, &x) && matches!(c.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-1)))
    );
    assert!(matches!(rhs.op(), Op::Const(cv) if cv.0 == ConstValue::Int(-3)));
}

#[test]
fn test_mul_lt_ceil_div_matches_only_target_weak_coefficients() {
    for c0 in [-1, 1] {
        let x = UOp::var("x", DType::WeakInt, -100, 100);
        let coefficient = UOp::const_(DType::WeakInt, ConstValue::Int(c0));
        let bound = UOp::const_(DType::WeakInt, ConstValue::Int(10));
        let lt = coefficient.try_mul(&x).unwrap().try_cmplt(&bound).unwrap();
        assert!(matches!(comparison_dsl_patterns().rewrite(&lt, &mut ()), RewriteResult::NoMatch));
    }

    let x = UOp::var("x", DType::Int32, -100, 100);
    let lt = UOp::native_const(3i32).try_mul(&x).unwrap().try_cmplt(&UOp::native_const(10i32)).unwrap();
    assert!(matches!(comparison_dsl_patterns().rewrite(&lt, &mut ()), RewriteResult::NoMatch));
}

// --- WHERE ALU combining ---

#[test]
fn test_where_alu_combine_add() {
    // Add(WHERE(c, 1, b), WHERE(c, 2, e)) → WHERE(c, 3, Add(b,e))
    // Tinygrad requires at least one branch pair to be const
    let matcher = symbolic();
    let c = UOp::var("c", DType::Bool, 0, 1);
    let t1 = UOp::native_const(1i32);
    let b = UOp::var("b", DType::Int32, 0, 100);
    let t2 = UOp::native_const(2i32);
    let e = UOp::var("e", DType::Int32, 0, 100);

    let w1 = UOp::try_where(c.clone(), t1, b.clone()).unwrap();
    let w2 = UOp::try_where(c.clone(), t2, e.clone()).unwrap();
    let add = w1.try_add(&w2).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));
    if let RewriteResult::Rewritten(rewritten) = result {
        if let Op::Ternary(TernaryOp::Where, cond, _true_br, false_br) = rewritten.op() {
            assert!(Arc::ptr_eq(cond, &c));
            // false branch should be Add(b, e)
            assert!(matches!(false_br.op(), Op::Binary(BinaryOp::Add, ..)));
        } else {
            panic!("Expected Where, got {:?}", rewritten.op());
        }
    }
}

#[test]
fn test_where_alu_combine_associative_add() {
    // (y + WHERE(c, 1, b)) + WHERE(c, 2, e) → y + WHERE(c, 3, Add(b,e))
    // Tinygrad symbolic.py:207-208: associative variation for Add chains
    use crate::rewrite::graph_rewrite;

    let c = UOp::var("c", DType::Bool, 0, 1);
    let y = UOp::var("y", DType::Int32, 0, 100);
    let t1 = UOp::native_const(1i32);
    let b = UOp::var("b", DType::Int32, 0, 100);
    let t2 = UOp::native_const(2i32);
    let e = UOp::var("e", DType::Int32, 0, 100);

    let w1 = UOp::try_where(c.clone(), t1, b.clone()).unwrap();
    let w2 = UOp::try_where(c.clone(), t2, e.clone()).unwrap();
    // (y + w1) + w2
    let inner_add = y.try_add(&w1).unwrap();
    let outer_add = inner_add.try_add(&w2).unwrap();

    let result = graph_rewrite(&symbolic(), outer_add.clone(), &mut ());

    // Result should contain a WHERE with combined const true branches (1+2=3)
    // and the y term outside: y + WHERE(c, 3, b+e)
    // The key assertion: the two WHERE nodes should be merged into one
    let where_count = result.toposort().iter().filter(|n| matches!(n.op(), Op::Ternary(TernaryOp::Where, ..))).count();
    assert!(where_count <= 1, "Expected WHERE nodes to be combined, got {where_count}");
}

#[test]
fn test_where_alu_combine_different_cond_no_match() {
    // Add(WHERE(c1, a, b), WHERE(c2, d, e)) should NOT combine (different conditions)
    let matcher = symbolic_simple();
    let c1 = UOp::var("c1", DType::Bool, 0, 1);
    let c2 = UOp::var("c2", DType::Bool, 0, 1);
    let a = UOp::var("a", DType::Int32, 0, 100);
    let b = UOp::var("b", DType::Int32, 0, 100);
    let d = UOp::var("d", DType::Int32, 0, 100);
    let e = UOp::var("e", DType::Int32, 0, 100);

    let w1 = UOp::try_where(c1, a, b).unwrap();
    let w2 = UOp::try_where(c2, d, e).unwrap();
    let add = w1.try_add(&w2).unwrap();

    let result = matcher.rewrite(&add, &mut ());
    if let RewriteResult::Rewritten(rewritten) = &result {
        assert!(!matches!(rewritten.op(), Op::Ternary(TernaryOp::Where, ..)));
    }
}

// ============================================================================
// F5: valid_simplification tests
// ============================================================================

#[test]
fn test_simplify_valid_redundant_upper_bounds() {
    // x < 10 AND x < 5: simplify_valid may or may not simplify depending on
    // whether symbolic_simple can collapse fake_var < 10 → true.
    // What we CAN test: the function at least processes without panic and
    // returns either None or a valid result.
    use crate::symbolic::valid_simplification::simplify_valid;

    let x = UOp::range_const(20, 0);
    let c10 = UOp::index_const(10);
    let c5 = UOp::index_const(5);
    let lt10 = x.lt(&c10);
    let lt5 = x.lt(&c5);
    let combined = lt10.and_(&lt5);

    let result = simplify_valid(&combined);
    if let Some(simplified) = result {
        // If simplified, result should be no larger than original
        assert!(simplified.node_count() <= combined.node_count(), "Simplified result should not be larger");
    }
    // Either way, function should not panic — this is the key test
}

#[test]
fn lower_bound_clauses_use_the_bounds_minimum() {
    use crate::symbolic::valid_simplification::parse_valid;

    let range = UOp::range_const(20, 0);
    let begin = UOp::var("begin", DType::WeakInt, 2, 9);
    let ne_form = range.lt(&begin).ne(&UOp::native_const(true));
    let not_form = range.lt(&begin).not();

    for clause in [ne_form, not_form] {
        assert_eq!(parse_valid(&clause).map(|(_, upper, bound)| (upper, bound)), Some((false, 2)));
    }
}

#[test]
fn test_simplify_valid_no_parseable_clauses() {
    use crate::symbolic::valid_simplification::simplify_valid;

    let a = UOp::native_const(true);
    let b = UOp::native_const(true);
    let combined = a.and_(&b);

    let result = simplify_valid(&combined);
    assert!(result.is_some_and(|result| Arc::ptr_eq(&result, &a)), "duplicate clauses should be deduplicated");
}

#[test]
fn test_uop_given_valid_does_not_leak_fake_params() {
    use crate::symbolic::valid_simplification::uop_given_valid;

    let x = UOp::var("x", DType::Int32, 0, 100);
    let valid = x.lt(&UOp::native_const(10i32));
    let expression = x.try_add(&UOp::native_const(1i32)).unwrap();
    let result = uop_given_valid(&valid, &expression, false);

    assert!(!result.toposort().iter().any(|node| {
        matches!(node.op(), Op::Param { arg, .. } if arg.name.as_deref().is_some_and(|name| name.starts_with("fake")))
    }));
}

#[test]
fn test_drop_and_clauses_irrelevant_removed() {
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::valid_simplification::pm_drop_and_clauses;

    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(20, 1);
    let c5 = UOp::index_const(5);
    let c15 = UOp::index_const(15);

    let clause1 = r0.lt(&c5);
    let clause2 = r1.lt(&c15);
    let combined_cond = clause1.try_and_op(&clause2).unwrap();

    let expr = r0.try_add(&UOp::index_const(1)).unwrap();
    let invalid = UOp::invalid_marker();
    let gated = UOp::try_where(combined_cond, expr, invalid).unwrap();

    let matcher = pm_drop_and_clauses();
    let result = graph_rewrite(matcher, gated.clone(), &mut ());

    assert!(!Arc::ptr_eq(&result, &gated), "Expected clause dropping");
    assert!(matches!(result.op(), Op::Ternary(TernaryOp::Where, ..)));
}

#[test]
fn test_drop_and_clauses_all_relevant_kept() {
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::valid_simplification::pm_drop_and_clauses;

    let r0 = UOp::range_const(10, 0);
    let c5 = UOp::index_const(5);
    let c8 = UOp::index_const(8);

    let clause1 = r0.lt(&c5);
    let clause2 = r0.lt(&c8);
    let combined = clause1.try_and_op(&clause2).unwrap();

    let expr = r0.try_add(&UOp::index_const(1)).unwrap();
    let invalid = UOp::invalid_marker();
    let gated = UOp::try_where(combined, expr, invalid).unwrap();

    let matcher = pm_drop_and_clauses();
    let result = graph_rewrite(matcher, gated.clone(), &mut ());

    assert!(Arc::ptr_eq(&result, &gated), "Both clauses relevant, should not change");
}

#[test]
fn test_drop_and_clauses_single_clause_no_change() {
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::valid_simplification::pm_drop_and_clauses;

    let r0 = UOp::range_const(10, 0);
    let c5 = UOp::index_const(5);
    let clause = r0.lt(&c5);

    let expr = r0.try_add(&UOp::index_const(1)).unwrap();
    let invalid = UOp::invalid_marker();
    let gated = UOp::try_where(clause, expr, invalid).unwrap();

    let matcher = pm_drop_and_clauses();
    let result = graph_rewrite(matcher, gated.clone(), &mut ());

    assert!(Arc::ptr_eq(&result, &gated), "Single clause should not change");
}

// ============================================================================
// F6: compute_sound_vmin_vmax tests
// ============================================================================

#[test]
fn test_sound_vmin_vmax_const() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let c = UOp::native_const(42i32);
    let result = compute_sound_vmin_vmax(&c);
    assert_eq!(result, Some((ConstValue::Int(42), ConstValue::Int(42))));
}

#[test]
fn test_sound_vmin_vmax_range() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let r = UOp::range_const(10, 0);
    let result = compute_sound_vmin_vmax(&r);
    assert_eq!(result, Some((ConstValue::Int(0), ConstValue::Int(9))));
}

#[test]
fn test_sound_vmin_vmax_add() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let r = UOp::range_const(10, 0);
    let c = UOp::index_const(5);
    let sum = r.try_add(&c).unwrap();
    let result = compute_sound_vmin_vmax(&sum);
    assert_eq!(result, Some((ConstValue::Int(5), ConstValue::Int(14))));
}

#[test]
fn test_sound_vmin_vmax_and_const_mask() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let r = UOp::range_const(100, 0);
    let mask = UOp::native_const(7i32);
    let r_int = r.cast(DType::Scalar(svod_dtype::ScalarDType::Int32));
    let result_node = r_int.and_(&mask);
    let result = compute_sound_vmin_vmax(&result_node);
    assert_eq!(result, Some((ConstValue::Int(0), ConstValue::Int(7))));
}

#[test]
fn test_sound_vmin_vmax_and_variable_mask_unsound() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let r1 = UOp::range_const(100, 0);
    let r2 = UOp::range_const(50, 1);
    let r1_int = r1.cast(DType::Scalar(svod_dtype::ScalarDType::Int32));
    let r2_int = r2.cast(DType::Scalar(svod_dtype::ScalarDType::Int32));
    let result_node = r1_int.and_(&r2_int);
    let result = compute_sound_vmin_vmax(&result_node);
    assert!(result.is_none(), "AND with variable mask should be unsound");
}

#[test]
fn test_sound_vmin_vmax_load_unsound() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let buf = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 100, DType::Scalar(svod_dtype::ScalarDType::Float32));
    let idx = UOp::index_const(0);
    let index = UOp::index().buffer(buf.clone()).indices(vec![idx]).call().unwrap();
    let load = UOp::load().index(index).call();
    let result = compute_sound_vmin_vmax(&load);
    assert!(result.is_none(), "LOAD should be unsound");
}

#[test]
fn test_sound_vmin_vmax_nested_sound() {
    use svod_ir::uop::range_eval::compute_sound_vmin_vmax;

    let c = UOp::index_const(3);
    let r = UOp::range_const(10, 0);
    let sum = c.try_add(&r).unwrap();
    let result = compute_sound_vmin_vmax(&sum);
    assert_eq!(result, Some((ConstValue::Int(3), ConstValue::Int(12))));
}

fn unknown_f32() -> Arc<UOp> {
    let buffer = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 1, DType::Float32);
    let index = UOp::index().buffer(buffer).indices(vec![UOp::index_const(0)]).call().unwrap();
    UOp::load().index(index).call()
}

#[test]
fn unknown_float_max_finite_limit_does_not_fold() {
    use crate::rewrite::graph_rewrite;

    let value = unknown_f32();
    let finite_max = UOp::const_(DType::Float32, ConstValue::Float(f32::MAX as f64));
    let maximum = value.try_max(&finite_max).unwrap();
    let result = graph_rewrite(symbolic(), maximum, &mut ());
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Max, ..)), "unknown f32 MAX was folded: {}", result.tree());
}

#[test]
fn unknown_float_comparison_and_where_do_not_fold() {
    use crate::rewrite::graph_rewrite;

    let value = unknown_f32();
    let finite_max = UOp::const_(DType::Float32, ConstValue::Float(f32::MAX as f64));
    let condition = value.try_cmplt(&finite_max).unwrap();
    let comparison = graph_rewrite(symbolic(), condition.clone(), &mut ());
    assert!(matches!(comparison.op(), Op::Binary(BinaryOp::Lt, ..)));

    let selected = UOp::try_where(condition, value.const_like(1.0), value.const_like(2.0)).unwrap();
    let result = graph_rewrite(symbolic(), selected, &mut ());
    assert!(
        matches!(result.op(), Op::Ternary(TernaryOp::Where, ..)),
        "unknown float WHERE was folded: {}",
        result.tree()
    );
}

#[test]
fn unknown_float_self_comparison_does_not_assume_no_nan() {
    use crate::rewrite::graph_rewrite;

    let value = unknown_f32();
    let result = graph_rewrite(symbolic_simple(), value.try_cmplt(&value).unwrap(), &mut ());
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Lt, ..)), "unknown float x<x was folded: {}", result.tree());
}

#[test]
fn float_max_tie_does_not_discard_signed_zero() {
    use crate::rewrite::graph_rewrite;

    let index = UOp::var("i", DType::Int32, 0, 1);
    let condition = index.try_cmplt(&UOp::native_const(1i32)).unwrap();
    let negative_zero = UOp::const_(DType::Float32, ConstValue::Float(-0.0));
    let positive_zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let selected = UOp::try_where(condition, negative_zero, positive_zero.clone()).unwrap();
    let maximum = selected.try_max(&positive_zero).unwrap();

    let result = graph_rewrite(symbolic(), maximum, &mut ());
    assert!(matches!(result.op(), Op::Binary(BinaryOp::Max, ..)), "float MAX tie was folded: {}", result.tree());
}

#[test]
fn explicitly_bounded_float_comparison_can_fold() {
    use crate::rewrite::graph_rewrite;

    let value = UOp::var("bounded", DType::Float32, -1, 1);
    let condition = value.try_cmplt(&value.const_like(2.0)).unwrap();
    let result = graph_rewrite(symbolic(), condition, &mut ());
    assert!(matches!(result.op(), Op::Const(value) if value.0 == ConstValue::Bool(true)));
}

#[test]
fn unknown_float_division_power_and_reciprocal_are_not_algebraically_rewritten() {
    use crate::rewrite::graph_rewrite;

    let value = unknown_f32();
    let self_div = value.try_div(&value).unwrap();
    assert!(matches!(graph_rewrite(sym(), self_div, &mut ()).op(), Op::Binary(BinaryOp::Fdiv, ..)));

    let square = value.try_pow(&value.const_like(2.0)).unwrap();
    assert!(matches!(graph_rewrite(sym(), square, &mut ()).op(), Op::Binary(BinaryOp::Pow, ..)));

    let product = value.try_mul(&value).unwrap();
    let reciprocal = UOp::try_reciprocal(&product).unwrap();
    assert!(matches!(graph_rewrite(sym(), reciprocal, &mut ()).op(), Op::Unary(UnaryOp::Reciprocal, ..)));
}

#[test]
fn float_add_zero_preserves_signed_zero_semantics() {
    use crate::rewrite::graph_rewrite;

    let value = unknown_f32();
    let positive_zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let negative_zero = UOp::const_(DType::Float32, ConstValue::Float(-0.0));

    let positive = graph_rewrite(symbolic_simple(), value.try_add(&positive_zero).unwrap(), &mut ());
    assert!(matches!(positive.op(), Op::Binary(BinaryOp::Add, ..)));
    let negative = graph_rewrite(symbolic_simple(), value.try_add(&negative_zero).unwrap(), &mut ());
    assert!(Arc::ptr_eq(&negative, &value));
}

// ============================================================================
// F7: Missing pattern group tests
// ============================================================================

#[test]
fn test_sym_phase3_neg_distribution() {
    use crate::rewrite::graph_rewrite;
    use crate::symbolic::sym;

    let x = UOp::range_const(10, 0);
    let y = UOp::range_const(20, 1);
    let x_int = x.cast(DType::Scalar(svod_dtype::ScalarDType::Int32));
    let y_int = y.cast(DType::Scalar(svod_dtype::ScalarDType::Int32));
    let sum = x_int.try_add(&y_int).unwrap();
    let neg_one = UOp::native_const(-1i32);
    let product = neg_one.try_mul(&sum).unwrap();

    // (-1) * (x + y) → neg(x) + neg(y) is in sym_phase3_patterns (sym tier)
    let result = graph_rewrite(sym(), product.clone(), &mut ());

    assert!(!matches!(result.op(), Op::Binary(BinaryOp::Mul, ..)), "Expected negation distribution, got Mul");
}

#[test]
fn test_substitute_gated_skips_irrelevant_subtrees() {
    use std::collections::HashMap;
    use svod_ir::UOpKey;

    let r0 = UOp::range_const(10, 0);
    let r1 = UOp::range_const(20, 1);
    let replacement = UOp::index_const(42);

    let sum = r0.try_add(&r1).unwrap();

    #[allow(clippy::mutable_key_type)]
    let map = HashMap::from([(UOpKey(r0.clone()), replacement.clone())]);
    let result = sum.substitute_gated(&map);

    if let Op::Binary(BinaryOp::Add, lhs, rhs) = result.op() {
        assert!(Arc::ptr_eq(lhs, &replacement) || Arc::ptr_eq(rhs, &replacement), "Expected replacement in result");
        assert!(Arc::ptr_eq(lhs, &r1) || Arc::ptr_eq(rhs, &r1), "Expected r1 preserved in result");
    } else {
        panic!("Expected Add, got {:?}", std::mem::discriminant(result.op()));
    }
}

#[test]
fn test_substitute_gated_empty_map() {
    use std::collections::HashMap;

    let r0 = UOp::range_const(10, 0);
    #[allow(clippy::mutable_key_type)]
    let map: HashMap<svod_ir::UOpKey, Arc<UOp>> = HashMap::new();
    let result = r0.substitute_gated(&map);
    assert!(Arc::ptr_eq(&result, &r0), "Empty map should return original");
}

/// The value-sensitive guard runs on every pattern attempt, so it must be a
/// memoised per-node property rather than a graph walk (previously O(n^2)).
#[test_case::test_case(DType::Float32, true ; "committed float chain")]
#[test_case::test_case(DType::WeakFloat, false ; "weak float leaf")]
fn weak_float_guard_is_memoized_per_node(leaf_dtype: DType, committed: bool) {
    const DEPTH: usize = 64;
    let mut root = UOp::const_(leaf_dtype, ConstValue::Float(0.5));
    for _ in 0..DEPTH {
        root = UOp::new(Op::Unary(UnaryOp::Sqrt, root), DType::Float32);
    }

    assert_eq!(weak_float_values_are_committed(&root), committed);

    let nodes = root.toposort();
    assert_eq!(nodes.len(), DEPTH + 1);
    for node in &nodes {
        assert!(HasWeakFloatProperty::cache(node).get().is_some(), "one evaluation per node, cached in place");
    }
    assert!(std::ptr::eq(HasWeakFloatProperty::get(&root), HasWeakFloatProperty::get(&root)));
}

// =========================================================================
// uint64 pack/unpack cancellation (tinygrad uop/symbolic.py:170-173)
// =========================================================================

/// `(hi.cast(u64) << shift) | lo.cast(u64)` — the THREEFRY packing idiom.
fn packed_u64(hi: &Arc<UOp>, lo: &Arc<UOp>, shift: i64) -> Arc<UOp> {
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(shift));
    hi.cast(DType::UInt64).shl(&amount).or_(&lo.cast(DType::UInt64))
}

fn u32_var(name: &str) -> Arc<UOp> {
    UOp::var(name, DType::UInt32, 0, u32::MAX as i64)
}

#[test_case::test_case(32, true; "shift of thirty two cancels")]
#[test_case::test_case(16, false; "shift of sixteen must not cancel")]
fn uint64_pack_low_half_cancels_only_at_thirty_two(shift: i64, folds: bool) {
    let (hi, lo) = (u32_var("hi"), u32_var("lo"));
    let expr = packed_u64(&hi, &lo, shift).cast(DType::UInt32);
    let folded = graph_rewrite(symbolic_simple(), expr.clone(), &mut ());

    assert_eq!(Arc::ptr_eq(&folded, &lo), folds, "got {}", folded.tree());
}

#[test_case::test_case(32, true; "shift of thirty two cancels")]
#[test_case::test_case(31, false; "shift of thirty one must not cancel")]
fn uint64_pack_high_half_cancels_only_at_thirty_two(shift: i64, folds: bool) {
    let (hi, lo) = (u32_var("hi"), u32_var("lo"));
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(shift));
    let expr = packed_u64(&hi, &lo, shift).shr(&amount);
    let folded = graph_rewrite(symbolic_simple(), expr.clone(), &mut ());

    assert_eq!(Arc::ptr_eq(&folded, &hi.cast(DType::UInt64)), folds, "got {}", folded.tree());
}

#[test]
fn uint64_pack_high_half_needs_a_narrow_low_arm() {
    // A wide low arm can carry bits into the high half, so `>> 32` is not `hi`.
    let hi = u32_var("hi");
    let wide = UOp::var("wide", DType::UInt64, 0, i64::MAX);
    let amount = UOp::const_(DType::UInt64, ConstValue::Int(32));
    let expr = hi.cast(DType::UInt64).shl(&amount).or_(&wide).shr(&amount);
    let folded = graph_rewrite(symbolic_simple(), expr.clone(), &mut ());

    assert!(!Arc::ptr_eq(&folded, &hi.cast(DType::UInt64)), "must not cancel: {}", folded.tree());
}

// =========================================================================
// weak-dtype constant folding (tinygrad uop/symbolic.py:31-33, registered :139-142)
// =========================================================================

/// `fold_const_alu` folds `exec_alu(a.op, a.dtype, vals, False)` for every dtype —
/// weak included — and returns `a.const_like(...)` at the node's own dtype.
fn weak_int(value: i64) -> Arc<UOp> {
    UOp::const_(DType::WeakInt, ConstValue::Int(value))
}

fn raw_add(lhs: Arc<UOp>, rhs: Arc<UOp>) -> Arc<UOp> {
    let dtype = lhs.dtype();
    UOp::new(Op::Binary(BinaryOp::Add, lhs, rhs), dtype)
}

#[test_case::test_case(BinaryOp::Add, 1, 14, 15 ; "add")]
#[test_case::test_case(BinaryOp::Mul, 7, 28, 196 ; "mul")]
#[test_case::test_case(BinaryOp::Sub, 1, 14, -13 ; "sub")]
fn weak_int_constant_operands_fold(op: BinaryOp, lhs: i64, rhs: i64, expect: i64) {
    let expr = UOp::new(Op::Binary(op, weak_int(lhs), weak_int(rhs)), DType::WeakInt);
    let folded = graph_rewrite(symbolic_simple(), expr, &mut ());

    assert_eq!(folded.dtype(), DType::WeakInt, "the fold stays at the weak dtype: {}", folded.tree());
    assert_eq!(
        crate::rangeify::indexing::get_const_value(&folded),
        Some(ConstValue::Int(expect)),
        "got {}",
        folded.tree()
    );
}

#[test]
fn weak_int_constants_cancel_across_an_index_sum() {
    // The resnet50 `r_16_32_7_7_512_3_3` index shape: `((1+14) + R*196) + (-15)`.
    let range = UOp::range_const(512, 0);
    let scaled = UOp::new(Op::Binary(BinaryOp::Mul, range, weak_int(196)), DType::WeakInt);
    let expr = raw_add(raw_add(raw_add(weak_int(1), weak_int(14)), scaled.clone()), weak_int(-15));

    let folded = graph_rewrite(symbolic(), expr, &mut ());

    assert!(Arc::ptr_eq(&folded, &scaled), "the constants must cancel, got {}", folded.tree());
}
