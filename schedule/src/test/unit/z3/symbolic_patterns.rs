//! Z3/SMT equivalence pins for `symbolic_simple`.
//!
//! What the rewrites produce structurally is covered by `test::property::symbolic_props`;
//! what is pinned here is that each rewrite preserves semantics under an SMT solver, and
//! that expressions the optimizer leaves alone still mean what we think they mean.

use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::UOp;
use test_case::test_case;

use crate::rewrite::graph_rewrite;
use crate::symbolic::symbolic_simple;
use crate::z3::verify::verify_equivalence;

fn var(name: &str, min: i64, max: i64) -> Arc<UOp> {
    UOp::var(name, DType::Int32, min, max)
}

/// The expression under test and, where the rewrite is not expected to reach it on its own,
/// the value it must nevertheless be equivalent to.
fn expression(case: &str) -> (Arc<UOp>, Option<Arc<UOp>>) {
    let (x, zero) = (var("x", 0, 100), UOp::native_const(0i32));
    // Divisors are kept away from zero so the solver is not asked about undefined division.
    let nonzero = var("x", 1, 100);
    let konst = |value: i32| UOp::native_const(value);

    match case {
        "add_zero" => (x.try_add(&zero).unwrap(), None),
        "sub_zero" => (x.try_sub(&zero).unwrap(), None),
        "mul_one" => (x.try_mul(&konst(1)).unwrap(), None),
        "div_one" => (x.try_div(&konst(1)).unwrap(), None),
        "mul_zero" => (x.try_mul(&zero).unwrap(), None),
        "mod_one" => (x.try_mod(&konst(1)).unwrap(), Some(zero)),
        "zero_numerator" => (zero.try_div(&nonzero).unwrap(), Some(zero)),
        "self_sub" => (x.try_sub(&x).unwrap(), Some(zero)),
        "self_div" => (nonzero.try_div(&nonzero).unwrap(), Some(konst(1))),
        "self_mod" => (nonzero.try_mod(&nonzero).unwrap(), Some(zero)),
        "self_add" => (x.try_add(&x).unwrap(), Some(konst(2).try_mul(&x).unwrap())),
        "div_cancels_mul" => {
            let (a, b) = (var("a", 0, 100), var("b", 1, 100));
            (a.try_mul(&b).unwrap().try_div(&b).unwrap(), Some(a))
        }
        "div_chain" => {
            let (a, b, c) = (var("a", 0, 100), var("b", 1, 10), var("c", 1, 10));
            (a.try_div(&b).unwrap().try_div(&c).unwrap(), None)
        }
        "div_gcd_factor" => {
            let (a, b, six) = (var("a", 0, 60), var("b", 0, 10), konst(6));
            (a.try_mul(&six).unwrap().try_div(&b.try_mul(&six).unwrap()).unwrap(), None)
        }
        "combine_coefficients" => {
            let sum = konst(2).try_mul(&x).unwrap().try_add(&konst(3).try_mul(&x).unwrap()).unwrap();
            (sum, Some(konst(5).try_mul(&x).unwrap()))
        }
        "fold_add_constants" => {
            (x.try_add(&konst(3)).unwrap().try_add(&konst(5)).unwrap(), Some(x.try_add(&konst(8)).unwrap()))
        }
        "fold_mul_constants" => {
            (x.try_mul(&konst(2)).unwrap().try_mul(&konst(3)).unwrap(), Some(x.try_mul(&konst(6)).unwrap()))
        }
        other => panic!("unknown case {other}"),
    }
}

#[test_case("add_zero"; "adding zero")]
#[test_case("sub_zero"; "subtracting zero")]
#[test_case("mul_one"; "multiplying by one")]
#[test_case("div_one"; "dividing by one")]
#[test_case("mul_zero"; "multiplying by zero")]
#[test_case("mod_one"; "modulo one")]
#[test_case("zero_numerator"; "zero numerator")]
#[test_case("self_sub"; "subtracting itself")]
#[test_case("self_div"; "dividing by itself")]
#[test_case("self_mod"; "modulo itself")]
#[test_case("self_add"; "adding itself")]
#[test_case("div_cancels_mul"; "division cancelling a multiplication")]
#[test_case("div_chain"; "chained division")]
#[test_case("div_gcd_factor"; "common factor in a division")]
#[test_case("combine_coefficients"; "combining coefficients")]
#[test_case("fold_add_constants"; "folding added constants")]
#[test_case("fold_mul_constants"; "folding multiplied constants")]
fn symbolic_simple_rewrites_preserve_semantics(case: &str) {
    let (expr, reference) = expression(case);

    let simplified = graph_rewrite(symbolic_simple(), expr.clone(), &mut ());
    verify_equivalence(&expr, &simplified).expect("the rewrite must preserve semantics");

    if let Some(reference) = reference {
        verify_equivalence(&simplified, &reference).expect("the result must equal its reference value");
    }
}
