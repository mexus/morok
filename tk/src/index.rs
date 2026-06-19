//! Flat addressing for tile buffers.
//!
//! Tiles carry a *logical* (multi-dimensional) shape but back onto a flat 1-D
//! pointer. [`flat_offset`] collapses multi-dim indices into a single
//! `Index`-typed offset (folding the all-constant part), and [`flat_index`] /
//! [`load_at`] turn that into the INDEX / LOAD the renderer expects.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, Op, UOp};

/// An index component: a compile-time constant or a runtime `Index`-typed UOp
/// (a loop range, `Special`, or derived lane arithmetic).
#[derive(Clone)]
pub enum Idx {
    Const(i64),
    Uop(Arc<UOp>),
}

impl Idx {
    /// Materialize this index component as an `Index`-typed UOp (a constant
    /// folds to a `cidx`, a dynamic component passes through).
    pub fn to_uop(&self) -> Arc<UOp> {
        match self {
            Idx::Const(c) => cidx(*c),
            Idx::Uop(u) => u.clone(),
        }
    }
}

impl From<i64> for Idx {
    fn from(v: i64) -> Self {
        Idx::Const(v)
    }
}
impl From<usize> for Idx {
    fn from(v: usize) -> Self {
        Idx::Const(v as i64)
    }
}
impl From<i32> for Idx {
    fn from(v: i32) -> Self {
        Idx::Const(v as i64)
    }
}
impl From<Arc<UOp>> for Idx {
    fn from(u: Arc<UOp>) -> Self {
        Idx::Uop(u)
    }
}
impl From<&Arc<UOp>> for Idx {
    fn from(u: &Arc<UOp>) -> Self {
        Idx::Uop(u.clone())
    }
}

/// A constant `Index`-typed UOp.
pub(crate) fn cidx(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

/// Row-major strides for `shape`.
pub fn strides(shape: &[usize]) -> Vec<i64> {
    let mut s = vec![1i64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1] as i64;
    }
    s
}

/// Collapse multi-dim `idxs` into a single `Index`-typed offset UOp, folding the
/// all-constant contribution into one constant and chaining only the dynamic
/// terms with `try_add`/`try_mul`.
pub fn flat_offset(shape: &[usize], idxs: &[Idx]) -> Arc<UOp> {
    assert_eq!(shape.len(), idxs.len(), "flat_offset: rank mismatch (shape {} vs idx {})", shape.len(), idxs.len());
    let st = strides(shape);
    let mut konst: i64 = 0;
    let mut dynamic: Option<Arc<UOp>> = None;
    for (i, idx) in idxs.iter().enumerate() {
        match idx {
            Idx::Const(c) => konst += c * st[i],
            Idx::Uop(u) => {
                let term =
                    if st[i] == 1 { u.clone() } else { u.try_mul(&cidx(st[i])).expect("flat_offset: stride mul") };
                dynamic = Some(match dynamic {
                    Some(a) => a.try_add(&term).expect("flat_offset: term add"),
                    None => term,
                });
            }
        }
    }
    match dynamic {
        None => cidx(konst),
        Some(a) if konst == 0 => a,
        Some(a) => a.try_add(&cidx(konst)).expect("flat_offset: const add"),
    }
}

/// Unwrap a `custom_kernel` placeholder (`PARAM` or `RESHAPE(PARAM)`) to its flat
/// 1-D pointer buffer plus element dtype. Hand-built kernels index the flat
/// PARAM directly rather than the multi-dim reshape view.
pub fn flat_ptr(placeholder: &Arc<UOp>) -> (Arc<UOp>, DType) {
    let buf = match placeholder.op() {
        Op::Reshape { src, .. } => src.clone(),
        _ => placeholder.clone(),
    };
    let elem = match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    };
    (buf, elem)
}

/// INDEX (ptr=true) into `buf` at the flattened offset — usable as both a STORE
/// target and a LOAD source.
pub fn flat_index(buf: &Arc<UOp>, shape: &[usize], idxs: &[Idx]) -> Arc<UOp> {
    let off = flat_offset(shape, idxs);
    UOp::index().buffer(buf.clone()).indices(vec![off]).ptr(true).call().expect("flat_index: INDEX construction")
}

/// LOAD from `buf` at the flattened offset (element dtype inferred from the
/// buffer's pointer base).
pub fn load_at(buf: &Arc<UOp>, shape: &[usize], idxs: &[Idx]) -> Arc<UOp> {
    let idx = flat_index(buf, shape, idxs);
    UOp::load().buffer(buf.clone()).index(idx).call()
}

/// INDEX (ptr=true) into `buf` at an already-flattened element `offset` — the
/// 1-D form used for flat GLOBAL buffer access (`srcf[src_i]` in tinygrad).
pub fn index_off(buf: &Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buf.clone()).indices(vec![offset]).ptr(true).call().expect("index_off: INDEX construction")
}

/// LOAD from `buf` at an already-flattened element `offset`.
pub fn load_off(buf: &Arc<UOp>, offset: Arc<UOp>) -> Arc<UOp> {
    let idx = index_off(buf, offset);
    UOp::load().buffer(buf.clone()).index(idx).call()
}

/// Wide LOAD of `lanes` contiguous elements from `buf` starting at element
/// `offset` — a single `<lanes × elem>` vector load (the 128-bit coalesced
/// GLOBAL fill: `bf16` × 8 = `global_load_dwordx4`). The INDEX points at the
/// first element; the vector LOAD reads the `lanes`-wide contiguous run. The
/// caller must guarantee `offset` is `lanes`-aligned (16-byte for `vec8` bf16).
pub fn load_vec(buf: &Arc<UOp>, offset: Arc<UOp>, lanes: usize) -> Arc<UOp> {
    let idx = index_off(buf, offset);
    let elem = match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    };
    UOp::load().buffer(buf.clone()).index(idx).dtype(elem.vec(lanes)).call()
}
