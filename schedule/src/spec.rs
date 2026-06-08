//! UOp spec verification — Rust port of tinygrad's `tinygrad/uop/spec.py`.
//!
//! tinygrad expresses each kernel invariant as a `(UPat, predicate)` rule in a
//! `PatternMatcher` and runs `type_verify(ast, spec)` to check every uop
//! against it (`spec.py:31`): a uop is valid only if the first matching rule's
//! predicate returns `True`, and a uop matching **no** rule also fails — the
//! spec is a *whitelist* (`spec.py:38`, `ret is not True → raise`). Because
//! Python is untyped, tinygrad reuses its rewriter `PatternMatcher` for this; in
//! Rust we mirror the design with a dedicated [`Spec`] of validity rules and a
//! [`type_verify`] runner.
//!
//! Verification is gated by `SVOD_SPEC` (default on, like tinygrad's `SPEC=1`)
//! so it can be disabled for perf. It turns a malformed kernel — a movement op
//! that should have been lowered to index arithmetic, a `<N x float>` leaking
//! into a memory index, a surviving `PtrCat` — into a recoverable `Err` *before*
//! the renderer turns it into a panic / malformed IR / GPU fault, so beam search
//! skips the offending candidate cleanly.
//!
//! [`spec_program`] is a whitelist (port of tinygrad's `spec_program` +
//! `spec_shared`): a lowered, pre-render kernel may contain only the ops below.
//! Architectural divergences from tinygrad — Svod renders `Reduce`/`Wmma`/
//! `Contract`/`Unroll`/`VConst` directly rather than expanding them before
//! linearize, and wraps the body in `Op::Linear` — are accepted explicitly.
//! [`spec_tensor`] stays in assertion mode (no enforced call site yet).

use std::sync::Arc;

use snafu::Snafu;
use svod_dtype::DType;
use svod_ir::{AddrSpace, BinaryOp, ConstValue, Op, TernaryOp, UOp};

#[derive(Debug, Clone, PartialEq, Eq, Snafu)]
#[snafu(visibility(pub))]
pub enum SpecError {
    #[snafu(display("UOp verification failed at {index} on {op} (dtype {dtype}): {reason}"))]
    Verification { index: usize, op: String, dtype: String, reason: &'static str },
}

/// A single spec rule (mirrors one `(UPat, predicate)` entry in tinygrad).
///
/// Returns `None` if the rule does not apply to `u`, `Some(Ok(()))` if it
/// applies and `u` is valid, or `Some(Err(reason))` if it applies and `u`
/// violates the invariant.
type SpecRule = Box<dyn Fn(&Arc<UOp>) -> Option<Result<(), &'static str>> + Send + Sync>;

/// An ordered set of validity rules (mirrors a tinygrad spec `PatternMatcher`).
pub struct Spec {
    rules: Vec<SpecRule>,
    /// Whitelist mode (tinygrad `spec_program`/`spec_tensor`): a uop matching no
    /// rule fails. Assertion mode (`false`): unmatched uops pass — used while a
    /// spec set is still being filled in.
    whitelist: bool,
}

impl Spec {
    /// Verdict of the first applicable rule, or `None` if no rule applies
    /// (tinygrad: the first matching pattern wins).
    fn check(&self, u: &Arc<UOp>) -> Option<Result<(), &'static str>> {
        self.rules.iter().find_map(|rule| rule(u))
    }
}

/// Whether spec verification runs. Mirrors tinygrad's `SPEC` ContextVar
/// (default 1 = on); disable with `SVOD_SPEC=0`.
pub fn spec_enabled() -> bool {
    !matches!(std::env::var("SVOD_SPEC").as_deref(), Ok("0"))
}

/// Check every uop reachable from `root` against `spec` (port of `type_verify`,
/// `spec.py:31`). In whitelist mode a uop matching no rule fails, exactly as
/// tinygrad's `ret is not True → raise`.
pub fn type_verify(root: &Arc<UOp>, spec: &Spec) -> Result<(), SpecError> {
    let nodes = root.toposort();
    for (index, u) in nodes.iter().enumerate() {
        let reason = match spec.check(u) {
            Some(Ok(())) => continue,
            Some(Err(reason)) => reason,
            None if spec.whitelist => "op not allowed in this spec (no matching rule)",
            None => continue,
        };
        // tinygrad prints the linearized uops on failure when DEBUG>=3
        // (`spec.py:39`); mirror that with a gated dump for diagnosis.
        if std::env::var_os("SVOD_SPEC_DEBUG").is_some() {
            eprintln!("[SPEC] reject #{index} {} (dtype {:?}): {reason}", u.op().as_ref(), u.dtype());
        }
        return VerificationSnafu { index, op: u.op().as_ref().to_string(), dtype: format!("{:?}", u.dtype()), reason }
            .fail();
    }
    Ok(())
}

// ============================================================================
// Helpers
// ============================================================================

#[inline]
fn ok_if(valid: bool, reason: &'static str) -> Result<(), &'static str> {
    if valid { Ok(()) } else { Err(reason) }
}

/// `true` if `dt` is a pointer in the given address space (port of tinygrad's
/// `isinstance(x.dtype, PtrDType) and x.dtype.addrspace == ...`).
fn is_ptr_in(dt: &DType, want: AddrSpace) -> bool {
    matches!(dt, DType::Ptr { addrspace, .. } if *addrspace == want)
}

// ============================================================================
// spec_shared — valid in both the tensor graph and lowered programs
// (port of `spec_shared`, spec.py:45)
// ============================================================================

/// `spec.py:52` — a `CONST`'s value type must match its dtype.
fn rule_const() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Const(cvh) => {
            let valid = match cvh.0 {
                ConstValue::Bool(_) => u.dtype().is_bool(),
                ConstValue::Float(_) => u.dtype().is_float(),
                ConstValue::Int(_) | ConstValue::UInt(_) => u.dtype().is_int(),
            };
            Some(ok_if(valid, "CONST value type does not match its dtype"))
        }
        _ => None,
    })
}

/// `spec.py:56-61` — ALU dtype invariants. WHERE/CMP/SHL-SHR/CDIV-CMOD are
/// special-cased exactly as tinygrad; every other ALU shares one base dtype.
fn rule_alu() -> SpecRule {
    Box::new(|u| {
        let result_base = u.dtype().base();
        match u.op() {
            // Unary preserves dtype.
            Op::Unary(_, x) => Some(ok_if(x.dtype().base() == result_base, "unary operand dtype mismatch")),

            // WHERE: bool condition, matching value/result dtypes (spec.py:56).
            Op::Ternary(TernaryOp::Where, c, x, y) => Some(ok_if(
                c.dtype().is_bool() && x.dtype() == y.dtype() && u.dtype() == x.dtype(),
                "WHERE condition must be bool with matching value/result dtypes",
            )),
            // MULACC: a*b+c, all sharing the result base.
            Op::Ternary(TernaryOp::MulAcc, a, b, c) => {
                Some(ok_if([a, b, c].iter().all(|s| s.dtype().base() == result_base), "MULACC operand dtype mismatch"))
            }

            Op::Binary(op, x, y) => {
                let (xb, yb) = (x.dtype().base(), y.dtype().base());
                let valid = if op.is_comparison() {
                    // CMPLT/CMPNE/CMPEQ: bool result, operands share base (spec.py:57).
                    u.dtype().is_bool() && xb == yb
                } else if matches!(op, BinaryOp::Shl | BinaryOp::Shr) {
                    // Shift distance may be a different int width (spec.py:59).
                    u.dtype() == x.dtype() && y.dtype().is_int()
                } else if matches!(op, BinaryOp::Idiv | BinaryOp::Mod) {
                    // C-style int div/mod must be integer (spec.py:60).
                    u.dtype().is_int() && xb == result_base && yb == result_base
                } else if matches!(op, BinaryOp::Threefry) {
                    // PRNG mixes uint widths; not a uniform-base ALU.
                    true
                } else {
                    xb == result_base && yb == result_base
                };
                Some(ok_if(valid, "binary operand/result dtype mismatch"))
            }
            _ => None,
        }
    })
}

/// `spec.py:67` — RANGE dtype matches its bound's dtype.
fn rule_range() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Range { end, .. } => Some(ok_if(u.dtype() == end.dtype(), "RANGE dtype must match its bound dtype")),
        _ => None,
    })
}

/// `spec.py:70` — every `INDEX`/`POINTER_INDEX` address operand must be integer.
fn rule_index_integer() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Index { indices, .. } => Some(ok_if(
            indices.iter().all(|idx| idx.dtype().is_int()),
            "non-integer value reached a memory INDEX operand",
        )),
        Op::PointerIndex { offset, .. } => Some(ok_if(offset.dtype().is_int(), "non-integer offset in POINTER_INDEX")),
        _ => None,
    })
}

/// `spec.py:71` — every range an END closes must be a `RANGE` (or, after
/// gpudims has replaced ranges with SPECIAL on the GPU path, a `SPECIAL`).
fn rule_end() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::End { ranges, .. } => Some(ok_if(
            ranges.iter().all(|r| matches!(r.op(), Op::Range { .. } | Op::Special { .. })),
            "END closes a non-RANGE/SPECIAL operand",
        )),
        _ => None,
    })
}

/// `spec.py:74` — PARAM (DEFINE_GLOBAL) is a global-address-space pointer.
fn rule_param() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Param { .. } => Some(ok_if(is_ptr_in(&u.dtype(), AddrSpace::Global), "PARAM must be a GLOBAL pointer")),
        _ => None,
    })
}

/// `spec.py:78` — a GROUP holds stores, groups, or noops.
fn rule_group() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Group { sources } => Some(ok_if(
            sources.iter().all(|s| matches!(s.op(), Op::Store { .. } | Op::Group { .. } | Op::Noop)),
            "GROUP may only hold STORE/GROUP/NOOP",
        )),
        _ => None,
    })
}

/// `spec.py:81` — DEFINE_LOCAL is a local-address-space pointer.
fn rule_define_local() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::DefineLocal(_) => {
            Some(ok_if(is_ptr_in(&u.dtype(), AddrSpace::Local), "DEFINE_LOCAL must be a LOCAL pointer"))
        }
        _ => None,
    })
}

/// `spec.py:96` — SPECIAL is indexed by an integer bound and named.
fn rule_special() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Special { end, .. } => Some(ok_if(end.dtype().is_int(), "SPECIAL bound must be integer")),
        _ => None,
    })
}

/// Structural ops with no further dtype invariant — accepted as in `spec_shared`
/// (SINK/NOOP/CAST/BITCAST/LOAD/STORE/WMMA/BARRIER/AFTER/CUSTOM/DEFINE_VAR/
/// DEFINE_REG/BIND). `Call` is shared (`spec_tensor` CALL, spec.py:143).
fn rule_shared_structural() -> SpecRule {
    Box::new(|u| {
        matches!(
            u.op(),
            Op::Sink { .. }
                | Op::Noop
                | Op::Cast { .. }
                | Op::BitCast { .. }
                | Op::Load { .. }
                | Op::Store { .. }
                | Op::Wmma { .. }
                | Op::Barrier { .. }
                | Op::After { .. }
                | Op::Custom { .. }
                | Op::CustomI { .. }
                | Op::Call { .. }
                | Op::DefineVar { .. }
                | Op::DefineReg { .. }
                | Op::Bind { .. }
        )
        .then_some(Ok(()))
    })
}

fn spec_shared() -> Vec<SpecRule> {
    vec![
        rule_const(),
        rule_alu(),
        rule_range(),
        rule_index_integer(),
        rule_end(),
        rule_param(),
        rule_group(),
        rule_define_local(),
        rule_special(),
        rule_shared_structural(),
    ]
}

// ============================================================================
// spec_program — additionally valid in lowered programs (port of `spec_program`,
// spec.py:200) plus Svod architectural ops
// ============================================================================

/// `spec.py:205` — movement ops are lowered to index arithmetic before a kernel
/// is linearized; a surviving one is a lowering bug.
fn rule_no_movement() -> SpecRule {
    Box::new(|u| u.op().is_movement().then_some(Err("movement op must be lowered away before a program")))
}

/// `spec.py:208` — Svod models `Invalid` as its own op (vs tinygrad's
/// `CONST(arg=Invalid)`); it must be folded out before a program.
fn rule_no_invalid() -> SpecRule {
    Box::new(|u| matches!(u.op(), Op::Invalid).then_some(Err("Invalid op must be folded out before a program")))
}

/// `PtrCat`/`Cat` (tinygrad PTRCAT/VCAT) must be distributed into scalar
/// loads/stores by the devectorizer; they have no rendering (spec.py:250 — "need
/// to be deleted").
fn rule_no_ptrcat_cat() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::PtrCat { .. } => Some(Err("PtrCat survived devectorization; must be distributed into scalar accesses")),
        Op::Cat { .. } => Some(Err("Cat (VCAT) survived devectorization; must be distributed into scalar accesses")),
        _ => None,
    })
}

/// `spec.py:215` (STACK) — VECTORIZE collects exactly `vcount` lane values, each
/// the scalar form of the result dtype.
fn rule_vectorize() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Vectorize { elements } => {
            let scalar = u.dtype().scalar_dtype();
            Some(ok_if(
                elements.len() == u.dtype().vcount() && elements.iter().all(|e| e.dtype() == scalar),
                "VECTORIZE lane count/dtype does not match its vector dtype",
            ))
        }
        _ => None,
    })
}

/// `spec.py:216,242` — GEP's output lane count matches its index count (covers
/// both single-lane scalar extracts and multi-lane shuffles).
fn rule_gep() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::Gep { indices, .. } => {
            Some(ok_if(u.dtype().vcount() == indices.len(), "GEP lane count does not match its index count"))
        }
        _ => None,
    })
}

/// `spec.py:219-220` — IF has a bool gate; ENDIF closes an IF.
fn rule_if() -> SpecRule {
    Box::new(|u| match u.op() {
        Op::If { condition, .. } => Some(ok_if(condition.dtype().is_bool(), "IF condition must be bool")),
        Op::EndIf { .. } => Some(Ok(())),
        _ => None,
    })
}

/// Svod architectural ops that legitimately survive into a linearized kernel,
/// unlike tinygrad which expands them earlier: the `Op::Linear` body wrapper,
/// vector constants, accumulator reductions, and the WMMA expander ops. Accepted
/// structurally (their internal invariants are enforced where they are built).
fn rule_program_structural() -> SpecRule {
    Box::new(|u| {
        matches!(
            u.op(),
            Op::Linear { .. } | Op::VConst { .. } | Op::Reduce { .. } | Op::Contract { .. } | Op::Unroll { .. }
        )
        .then_some(Ok(()))
    })
}

/// Spec for a lowered, pre-render kernel (`spec_program`, spec.py:200) — a
/// whitelist: program-only rules first, then the shared rules.
pub fn spec_program() -> Spec {
    let mut rules = vec![
        rule_no_movement(),
        rule_no_invalid(),
        rule_no_ptrcat_cat(),
        rule_vectorize(),
        rule_gep(),
        rule_if(),
        rule_program_structural(),
    ];
    rules.extend(spec_shared());
    Spec { rules, whitelist: true }
}

/// Spec for the tensor graph (`spec_tensor`, spec.py:116). Currently the shared
/// rules in assertion mode — there is no enforced tensor-graph call site yet.
pub fn spec_tensor() -> Spec {
    Spec { rules: spec_shared(), whitelist: false }
}
