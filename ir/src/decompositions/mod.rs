//! UOp decomposition framework.
//!
//! This module provides conditional decomposition of complex operations into
//! simpler primitives that all backends can handle. Backends that don't support
//! certain transcendental operations can use the pattern-based decompositor
//! to transform them into equivalent primitive operations.
//!
//! # Architecture
//!
//! 1. **Backend provides decomposition patterns** via `Renderer::decompositor()`
//! 2. **Decomposition pass** uses `graph_rewrite_bottom_up` to apply patterns
//! 3. **Each pattern** transforms one op into a subtree of primitive ops
//!
//! # Example
//!
//! ```ignore
//! // In tensor realization, before rendering:
//! if let Some(decompositor) = renderer.decompositor() {
//!     let ast = decompose_with(&kernel.ast, &decompositor);
//! }
//! let rendered = renderer.render(&ast)?;
//! ```

pub mod helpers;
pub mod ptrcat;
pub mod transcendentals;

use std::sync::Arc;

use crate::pattern::TypedPatternMatcher;
use crate::rewrite::graph_rewrite_bottom_up;
use crate::uop::UOp;
use svod_macros::patterns;

use transcendentals::{xcos, xerf, xexp, xexp2, xlog, xlog2, xpow, xrsqrt, xsin, xsqrt, xtan};

/// Vector-of-pointer decomposition for MLIR backend.
///
/// MLIR's LLVM dialect doesn't support `vector<N x ptr>` types. This pattern
/// eliminates VECTORIZE and PtrCat operations on pointer types that weren't
/// consumed by LOAD/STORE patterns during devectorization.
///
/// # Example
///
/// ```ignore
/// impl Renderer for MlirRenderer {
///     fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
///         Some(ptrcat_decomposition_patterns())
///     }
/// }
/// ```
pub fn ptrcat_decomposition_patterns() -> TypedPatternMatcher<()> {
    use crate::DType;

    patterns! {
        // Eliminate VECTORIZE on pointers by returning first element
        // (VECTORIZE on pointers that isn't consumed by GEP is dead code)
        Vectorize { elements } if matches!(elements[0].dtype(), DType::Ptr { .. }) ~> |elements| elements[0].clone(),

        // Eliminate bare PtrCat by returning first pointer
        // (PtrCat not consumed by LOAD/STORE is dead code)
        PtrCat { sources } ~> |sources| sources[0].clone(),
    }
}

/// All decomposition patterns for transcendental operations.
///
/// Returns a `TypedPatternMatcher` that decomposes:
/// - Unary: Exp2, Log2, Exp, Log, Sin, Cos, Tan, Sqrt, Rsqrt, Erf
/// - Binary: Pow
///
/// Backends that don't support these operations natively can use this
/// matcher with `decompose_with()` to decompose them into primitives.
///
/// # Example
///
/// ```ignore
/// impl Renderer for CpuRenderer {
///     fn decompositor(&self) -> Option<TypedPatternMatcher<()>> {
///         Some(all_decomposition_patterns())
///     }
/// }
/// ```
pub fn all_decomposition_patterns() -> TypedPatternMatcher<()> {
    patterns! {
        // Transcendental unary ops
        Exp2(src) ~> |src| xexp2(src),
        Log2(src) ~> |src| xlog2(src),
        Exp(src)  ~> |src| xexp(src),
        Log(src)  ~> |src| xlog(src),
        Sin(src)  ~> |src| xsin(src),
        Cos(src)  ~> |src| xcos(src),
        Tan(src)  ~> |src| xtan(src),
        Sqrt(src) ~> |src| xsqrt(src),
        Rsqrt(src) ~> |src| xrsqrt(src),
        Erf(src)  ~> |src| xerf(src),

        // Binary pow: x^y = exp2(y * log2(x))
        Pow(base, exp) ~> |base, exp| xpow(base, exp),
    }
}

/// Decomposition patterns for the AMD backend.
///
/// AMD's hardware `v_exp_f32`/`v_log_f32` (emitted as `@llvm.exp2`/`@llvm.log2`)
/// are lower precision than CPU libm, so the exp/log/trig family is routed
/// through the SLEEF `~1 ULP` polynomials instead. This mirrors tinygrad's
/// `TRANSCENDENTAL=2` force mode (`uop/decompositions.py`), and uses the same
/// coefficients (`transcendentals.rs`).
///
/// `Sqrt`/`Rsqrt` are deliberately **omitted** — AMD's `@llvm.sqrt` is
/// IEEE-correct (~0.5 ULP), better than the polynomial, and tinygrad likewise
/// keeps `SQRT` native in `AMDLLVMRenderer.code_for_op`.
///
/// Every pattern is guarded to `f16`/`f32`/`f64` (tinygrad's
/// `TRANSCENDENTAL_DTYPES`): the polynomials are only defined for those, and
/// integer `Pow` (ONNX `test_pow_types_*`) / `bf16` / `fp8` must keep their
/// native lowering.
pub fn amd_decomposition_patterns() -> TypedPatternMatcher<()> {
    use crate::DType;
    fn transc(d: &DType) -> bool {
        use svod_dtype::ScalarDType::{Float16, Float32, Float64};
        matches!(d.base(), Float16 | Float32 | Float64)
    }
    patterns! {
        Exp2(src) if transc(&src.dtype()) ~> |src| xexp2(src),
        Log2(src) if transc(&src.dtype()) ~> |src| xlog2(src),
        Exp(src)  if transc(&src.dtype()) ~> |src| xexp(src),
        Log(src)  if transc(&src.dtype()) ~> |src| xlog(src),
        Sin(src)  if transc(&src.dtype()) ~> |src| xsin(src),
        Cos(src)  if transc(&src.dtype()) ~> |src| xcos(src),
        Tan(src)  if transc(&src.dtype()) ~> |src| xtan(src),
        Erf(src)  if transc(&src.dtype()) ~> |src| xerf(src),

        // Binary pow: x^y = exp2(y * log2(x))
        Pow(base, exp) if transc(&base.dtype()) ~> |base, exp| xpow(base, exp),

        // bf16/fp8/int fall back to f32 then cast back (tinygrad's cast arm).
        // Int `Pow` would otherwise hit `@llvm.pow.f64`, which amdgcn can't
        // select; bf16/fp8 transcendentals have no native intrinsic either.
        Exp2(src) ~> |src| xexp2(&src.cast(DType::Float32)).cast(src.dtype()),
        Log2(src) ~> |src| xlog2(&src.cast(DType::Float32)).cast(src.dtype()),
        Exp(src)  ~> |src| xexp(&src.cast(DType::Float32)).cast(src.dtype()),
        Log(src)  ~> |src| xlog(&src.cast(DType::Float32)).cast(src.dtype()),
        Sin(src)  ~> |src| xsin(&src.cast(DType::Float32)).cast(src.dtype()),
        Cos(src)  ~> |src| xcos(&src.cast(DType::Float32)).cast(src.dtype()),
        Tan(src)  ~> |src| xtan(&src.cast(DType::Float32)).cast(src.dtype()),
        Erf(src)  ~> |src| xerf(&src.cast(DType::Float32)).cast(src.dtype()),
        Pow(base, exp) ~> |base, exp| xpow(&base.cast(DType::Float32), &exp.cast(DType::Float32)).cast(base.dtype()),
    }
}

/// Apply decomposition to a UOp graph using the provided pattern matcher.
///
/// Uses `graph_rewrite_bottom_up` to traverse the graph and apply decomposition
/// patterns. This ensures children are processed before parents, which is
/// important for recursive decomposition (e.g., when a decomposition result
/// contains more operations that need decomposition).
///
/// # Arguments
///
/// * `root` - The root UOp of the graph to decompose
/// * `matcher` - The pattern matcher containing decomposition rules
///
/// # Returns
///
/// A new UOp graph with matched operations replaced by their decompositions.
///
/// # Example
///
/// ```ignore
/// let matcher = all_decomposition_patterns();
/// let decomposed = decompose_with(&kernel.ast, &matcher);
/// ```
pub fn decompose_with(root: &Arc<UOp>, matcher: &TypedPatternMatcher<()>) -> Arc<UOp> {
    graph_rewrite_bottom_up(matcher, root.clone(), &mut ())
}
