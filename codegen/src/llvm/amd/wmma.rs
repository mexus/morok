//! AMD WMMA / MFMA intrinsic dispatch by gfx family.
//!
//! Tinygrad references:
//! - RDNA3/RDNA4 `wmma.*` mangling: `renderer/amd/wmma.py`, `renderer/llvmir.py:259-289`
//! - CDNA `mfma.*` mangling: same files
//!
//! The IR-level matrix shape is encoded in `WmmaMetadata::dims` as `(N, M, K)`.
//! For each (arch, in_dtype, acc_dtype, dims) tuple we map to one of the
//! `@llvm.amdgcn.{wmma|mfma}.*` intrinsics, packing inputs as needed.

use std::sync::Arc;

use svod_dtype::{AmdArch, ScalarDType};
use svod_ir::{WmmaMetadata, prelude::*};

use crate::llvm::common::{RenderContext, ldt};

/// Render a WMMA UOp for the AMD target. Returns `None` if the (arch, dtype,
/// shape) combination has no direct intrinsic; in that case the caller
/// surfaces an `InvalidGraph` error and the optimizer must decompose it
/// upstream.
#[allow(clippy::too_many_arguments)]
pub fn render_wmma_amd(
    uop: &Arc<UOp>,
    a: &Arc<UOp>,
    b: &Arc<UOp>,
    c: &Arc<UOp>,
    metadata: &WmmaMetadata,
    arch: AmdArch,
    ctx: &mut RenderContext,
    kernel: &mut Vec<String>,
) -> Option<()> {
    let dst = ctx.name(uop);
    let a_name = ctx.get(a).to_string();
    let b_name = ctx.get(b).to_string();
    let c_name = ctx.get(c).to_string();
    let a_ty = ldt(&a.dtype());
    let b_ty = ldt(&b.dtype());
    let c_ty = ldt(&c.dtype());
    let acc_ty = ldt(&uop.dtype());

    let (n, m, k) = metadata.dims;
    // WMMA operands are vectors (e.g. `<16 x half>` for inputs, `<8 x float>`
    // for accumulators). `DType::scalar()` returns `None` for `Vector{..}` —
    // we need `base()`, which unwraps both `Scalar` and `Vector` to the inner
    // ScalarDType. Tinygrad's `.scalar()` works on both because of how their
    // DType class is defined (`renderer/llvmir.py:57-62`); svod's `.scalar()`
    // is stricter so we use `.base()` here. Wrapping in `Some` keeps the
    // downstream API uniform.
    let in_scalar = Some(a.dtype().base());
    let acc_scalar = Some(uop.dtype().base());

    let intrinsic = match resolve_intrinsic(arch, in_scalar, acc_scalar, (n, m, k)) {
        Some(s) => s,
        None => {
            ctx.set_invalid_graph(format!(
                "AMD renderer: no WMMA/MFMA intrinsic for arch={arch} in={in_scalar:?} \
                 acc={acc_scalar:?} dims=({n},{m},{k})"
            ));
            return None;
        }
    };

    if arch.is_cdna() {
        // MFMA signature: 4 trailing constant ints (cbsz, abid, blgp).
        kernel.push(format!(
            "  {dst} = call {acc_ty} @{intrinsic}({a_ty} {a_name}, {b_ty} {b_name}, \
             {c_ty} {c_name}, i32 0, i32 0, i32 0)"
        ));
    } else {
        // WMMA signature: any non-f32 accumulator (f16/bf16/int) takes a trailing
        // `i1 false` (the clamp/opsel bit); f32-accumulating WMMAs take (A,B,C)
        // only. Mirrors tinygrad `dtype.scalar() != float` (`llvmir.py:62`).
        let needs_tail = !matches!(acc_scalar, Some(ScalarDType::Float32));
        let tail = if needs_tail { ", i1 false" } else { "" };
        kernel.push(format!(
            "  {dst} = call {acc_ty} @{intrinsic}({a_ty} {a_name}, {b_ty} {b_name}, {c_ty} {c_name}{tail})"
        ));
    }
    Some(())
}

/// Pick an amdgcn intrinsic name for a given (arch, dtype, shape) tuple.
///
/// Returns `None` for shapes/dtypes the renderer doesn't natively support
/// (the optimizer is expected to decompose those upstream).
///
/// Naming scheme:
/// - RDNA3/RDNA4: `llvm.amdgcn.wmma.<acc>.16x16x16.<in>` (with optional
///   `.tied` for vec(8) accumulators on gfx1100/1151 — we leave that to a
///   future pre-rewrite pass).
/// - CDNA: `llvm.amdgcn.mfma.<acc>.<N>x<M>x<K><in>`.
/// - RDNA2 and other non-matrix-core arches: `None` — the optimizer must
///   decompose WMMA UOps to scalar/vector loops before rendering.
fn resolve_intrinsic(
    arch: AmdArch,
    in_dt: Option<ScalarDType>,
    acc_dt: Option<ScalarDType>,
    dims: (usize, usize, usize),
) -> Option<String> {
    if !arch.has_matrix_cores() {
        return None;
    }

    let (n, m, k) = dims;
    let in_dt = in_dt?;
    let acc_dt = acc_dt?;

    if arch.is_cdna() {
        // CDNA fp8/bf8 carry their own leading dot; bf16/f16 use `bf16.1k`/`f16`
        // for K=16 and the dotted `.bf16`/`.f16` forms for K=32 (tinygrad
        // `llvmir.py:51-56`). Other suffixes append directly after `{k}`.
        let in_suffix = match (in_dt, k) {
            (ScalarDType::Float16, 32) => ".f16",
            (ScalarDType::BFloat16, 32) => ".bf16",
            (ScalarDType::Float16, _) => "f16",
            (ScalarDType::BFloat16, _) => "bf16.1k",
            (ScalarDType::Float32, _) => "f32",
            (ScalarDType::FP8E4M3, _) => ".fp8.fp8",
            (ScalarDType::FP8E5M2, _) => ".bf8.bf8",
            _ => return None,
        };
        let acc_suffix = match acc_dt {
            ScalarDType::Float32 => "f32",
            ScalarDType::Float64 => "f64",
            ScalarDType::Int32 => "i32",
            _ => return None,
        };
        return Some(format!("llvm.amdgcn.mfma.{acc_suffix}.{n}x{m}x{k}{in_suffix}"));
    }

    // RDNA3 / RDNA4 WMMA — both families use 16x16x16 matmul; differ in input
    // dtype packing (handled by upstream pre-rewrites at the renderer level
    // when present; here we just name the intrinsic).
    let in_suffix = match in_dt {
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Int8 => "iu8",
        ScalarDType::FP8E4M3 => "fp8.fp8",
        ScalarDType::FP8E5M2 => "bf8.bf8",
        _ => return None,
    };
    let acc_suffix = match acc_dt {
        ScalarDType::Float32 => "f32",
        ScalarDType::Float16 => "f16",
        ScalarDType::BFloat16 => "bf16",
        ScalarDType::Int32 => "i32",
        _ => return None,
    };
    Some(format!("llvm.amdgcn.wmma.{acc_suffix}.{n}x{m}x{k}.{in_suffix}"))
}

#[cfg(test)]
#[path = "../../test/unit/llvm_amd_wmma.rs"]
mod tests;
