//! Buffer-bound tile wrappers — the eager tiles that bind a concrete UOp buffer
//! plus a *logical* shape to a [`Kernel`], mirroring tinygrad `tiles.py`'s
//! `GL`/`ST`/`RT`/`RV`. Each wrapper carries its backing `Arc<UOp>` (a flat 1-D
//! pointer), the multi-dim logical shape used for [`crate::index::flat_index`]
//! addressing, its layout/base-shape descriptor, and a borrow of the building
//! [`Kernel`].
//!
//! Unlike tinygrad's autowrapping proxy, these are plain structs (no `Deref` to
//! `Arc<UOp>`); a [`rewrap`](GL::rewrap) swaps the backing buffer after a store
//! (the tinykittens `ruop`) so later reads depend on the store via its
//! `After([END(STORE)])` node.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::DType;
use svod_ir::UOp;

use crate::Kernel;
use crate::tiles::{RTBaseShape, STBaseShape, TileLayout, VecLayout};

/// A register-backed tile ([`RT`] or [`RV`]) that the elementwise [`map`] /
/// math / reduce ops manipulate uniformly: a flat buffer, a logical shape, an
/// element dtype, and a `rewrap` to swap the backing buffer after a store.
pub trait RegTile<'k>: Clone {
    fn uop(&self) -> &Arc<UOp>;
    fn shape(&self) -> &[usize];
    fn elem(&self) -> &DType;
    fn layout(&self) -> TileLayout;
    fn rewrap(&self, new_uop: Arc<UOp>) -> Self;

    /// Rewrap with an extra ordering dependency (tinygrad `tile.after(dep)`),
    /// e.g. a write-after-read edge that forces this tile's next read to observe
    /// `deps` first.
    fn after(&self, deps: SmallVec<[Arc<UOp>; 4]>) -> Self {
        self.rewrap(self.uop().after(deps))
    }
}

impl<'k> RegTile<'k> for RT<'k> {
    fn uop(&self) -> &Arc<UOp> {
        &self.buf
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn elem(&self) -> &DType {
        &self.elem
    }
    fn layout(&self) -> TileLayout {
        self.layout
    }
    fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RT::rewrap(self, new_uop)
    }
}

impl<'k> RegTile<'k> for RV<'k> {
    fn uop(&self) -> &Arc<UOp> {
        &self.buf
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn elem(&self) -> &DType {
        &self.elem
    }
    /// An RV is logically a column of values; treat it as `Row` for the generic
    /// ops (it never broadcasts another vector into itself).
    fn layout(&self) -> TileLayout {
        TileLayout::Row
    }
    fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RV::rewrap(self, new_uop)
    }
}

/// Element (scalar) dtype backing a pointer tile buffer.
fn elem_of(buf: &Arc<UOp>) -> DType {
    match buf.dtype() {
        DType::Ptr { base, .. } => (*base).clone(),
        dt => dt,
    }
}

/// Macro for the shared tile accessors (`uop`/`shape`/`elem`/`ker`).
macro_rules! tile_accessors {
    () => {
        /// The backing flat 1-D pointer buffer (or its `After` re-wrap).
        pub fn uop(&self) -> &Arc<UOp> {
            &self.buf
        }
        /// The multi-dim logical shape used for flat addressing.
        pub fn shape(&self) -> &[usize] {
            &self.shape
        }
        /// The element (scalar) dtype.
        pub fn elem(&self) -> &DType {
            &self.elem
        }
        /// The building kernel.
        pub fn ker(&self) -> &'k Kernel {
            self.ker
        }
    };
}

/// A global-memory tile: the next bound buffer placeholder (`Param`) plus its
/// logical shape (e.g. `[1, 1, N, N]`). Accessed flat in load/store.
#[derive(Clone)]
pub struct GL<'k> {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    elem: DType,
    ker: &'k Kernel,
}

impl<'k> GL<'k> {
    tile_accessors!();
    /// Swap the backing buffer (after a store) keeping shape/dtype.
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        GL { buf: new_uop, shape: self.shape.clone(), elem: self.elem.clone(), ker: self.ker }
    }
}

/// A shared-memory (LDS) tile: a grid of [`STBaseShape`] fragments. Logical
/// shape is `[.., height, width, base.rows, base.cols]`.
#[derive(Clone)]
pub struct ST<'k> {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    pub rows: usize,
    pub cols: usize,
    pub layout: TileLayout,
    pub base: STBaseShape,
    elem: DType,
    ker: &'k Kernel,
    /// Optional additive flat-element offset into the backing buffer — the
    /// software double-buffer parity select (`tile % 2 * half_elems`). `None`
    /// for an ordinary single-buffered tile (the common case). When `Some`, every
    /// LDS access adds it to the computed flat offset, selecting one half of a
    /// `st_db` (2×-size) buffer at runtime.
    base_offset: Option<Arc<UOp>>,
}

impl<'k> ST<'k> {
    tile_accessors!();
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        ST {
            buf: new_uop,
            shape: self.shape.clone(),
            rows: self.rows,
            cols: self.cols,
            layout: self.layout,
            base: self.base,
            elem: self.elem.clone(),
            ker: self.ker,
            base_offset: self.base_offset.clone(),
        }
    }

    /// The per-half flat element count (the full single-half tile size); a
    /// [`Kernel::st_db`] buffer holds two of these. Used to form parity offsets.
    pub fn half_elems(&self) -> usize {
        self.shape.iter().product()
    }
    /// This tile viewing one half of a double buffer: every LDS access adds
    /// `off` (an `Index`-typed element offset, typically `parity * half_elems()`)
    /// to its flat address. Clones the wrapper (shares the backing buffer).
    pub fn with_base_offset(&self, off: Arc<UOp>) -> ST<'k> {
        let mut t = self.rewrap(self.buf.clone());
        t.base_offset = Some(off);
        t
    }
    /// The parity base offset, if this is a double-buffer half view.
    pub fn base_offset(&self) -> Option<&Arc<UOp>> {
        self.base_offset.as_ref()
    }
}

/// A register (per-lane) tile: a grid of [`RTBaseShape`] fragments. Logical
/// shape is `[height, width, base.elements_per_thread]`.
#[derive(Clone)]
pub struct RT<'k> {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    pub layout: TileLayout,
    pub base: RTBaseShape,
    elem: DType,
    ker: &'k Kernel,
}

impl<'k> RT<'k> {
    tile_accessors!();
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RT {
            buf: new_uop,
            shape: self.shape.clone(),
            layout: self.layout,
            base: self.base,
            elem: self.elem.clone(),
            ker: self.ker,
        }
    }
}

/// A register vector: logical shape `[outer_dim, inner_dim]` (`[tiles, 1]` for
/// the ortho layout).
#[derive(Clone)]
pub struct RV<'k> {
    buf: Arc<UOp>,
    shape: Vec<usize>,
    pub length: usize,
    pub layout: VecLayout,
    pub base: RTBaseShape,
    elem: DType,
    ker: &'k Kernel,
}

impl<'k> RV<'k> {
    tile_accessors!();
    pub fn rewrap(&self, new_uop: Arc<UOp>) -> Self {
        RV {
            buf: new_uop,
            shape: self.shape.clone(),
            length: self.length,
            layout: self.layout,
            base: self.base,
            elem: self.elem.clone(),
            ker: self.ker,
        }
    }
}

/// A type-erased tile, used by the addrspace-dispatching `Group::load`/`store`
/// (the svod analog of tinygrad's `isinstance` + `addrspace` checks). Concrete
/// tiles convert in via `From`; results extract out via [`Tile::st`] etc.
#[derive(Clone)]
pub enum Tile<'k> {
    Gl(GL<'k>),
    St(ST<'k>),
    Rt(RT<'k>),
    Rv(RV<'k>),
}

impl<'k> From<GL<'k>> for Tile<'k> {
    fn from(t: GL<'k>) -> Self {
        Tile::Gl(t)
    }
}
impl<'k> From<ST<'k>> for Tile<'k> {
    fn from(t: ST<'k>) -> Self {
        Tile::St(t)
    }
}
impl<'k> From<RT<'k>> for Tile<'k> {
    fn from(t: RT<'k>) -> Self {
        Tile::Rt(t)
    }
}
impl<'k> From<RV<'k>> for Tile<'k> {
    fn from(t: RV<'k>) -> Self {
        Tile::Rv(t)
    }
}

impl<'k> Tile<'k> {
    /// Extract a [`GL`] (panics on mismatch).
    pub fn gl(self) -> GL<'k> {
        match self {
            Tile::Gl(t) => t,
            _ => panic!("Tile::gl: not a GL tile"),
        }
    }
    /// Extract an [`ST`] (panics on mismatch).
    pub fn st(self) -> ST<'k> {
        match self {
            Tile::St(t) => t,
            _ => panic!("Tile::st: not an ST tile"),
        }
    }
    /// Extract an [`RT`] (panics on mismatch).
    pub fn rt(self) -> RT<'k> {
        match self {
            Tile::Rt(t) => t,
            _ => panic!("Tile::rt: not an RT tile"),
        }
    }
}

impl Kernel {
    /// Bind the next declared buffer as a [`GL`] tile (tinygrad `ker.gl`). The
    /// element dtype is taken from the bound buffer; `dtype` is accepted for API
    /// parity but the concrete buffer's dtype governs.
    pub fn gl(&self, shape: &[usize], _dtype: DType) -> GL<'_> {
        let buf = self.next_global();
        let elem = elem_of(&buf);
        GL { buf, shape: shape.to_vec(), elem, ker: self }
    }

    /// Allocate a shared-memory [`ST`] tile (tinygrad `ker.st`). `dims` is the
    /// `(rows, cols)` block size; the LDS buffer is `height×width×base` flat.
    pub fn st(&self, dims: (usize, usize), dtype: DType, layout: TileLayout, base: STBaseShape) -> ST<'_> {
        let (rows, cols) = dims;
        assert_eq!(rows % base.base.rows, 0, "ST rows {rows} not a multiple of base {}", base.base.rows);
        assert_eq!(cols % base.base.cols, 0, "ST cols {cols} not a multiple of base {}", base.base.cols);
        assert_eq!(cols % base.base.elements_per_thread(), 0, "ST cols {cols} not a multiple of elements_per_thread");
        let height = rows / base.base.rows;
        let width = cols / base.base.cols;
        let shape = vec![height, width, base.base.rows, base.base.cols];
        let flat = shape.iter().product();
        let buf = self.alloc_local(flat, dtype.clone());
        ST { buf, shape, rows, cols, layout, base, elem: dtype, ker: self, base_offset: None }
    }

    /// Allocate a **double-buffered** shared-memory [`ST`] tile: identical logical
    /// shape and addressing to [`Kernel::st`], but the backing LDS buffer is
    /// **2× the flat size** so the two halves can hold consecutive K-tiles for a
    /// software-pipelined K-loop. The returned tile has `base_offset = None`
    /// (it addresses half 0); the caller forms the two half-views with
    /// [`ST::with_base_offset`]`(parity * `[`ST::half_elems`]`())`.
    pub fn st_db(&self, dims: (usize, usize), dtype: DType, layout: TileLayout, base: STBaseShape) -> ST<'_> {
        let (rows, cols) = dims;
        assert_eq!(rows % base.base.rows, 0, "ST rows {rows} not a multiple of base {}", base.base.rows);
        assert_eq!(cols % base.base.cols, 0, "ST cols {cols} not a multiple of base {}", base.base.cols);
        assert_eq!(cols % base.base.elements_per_thread(), 0, "ST cols {cols} not a multiple of elements_per_thread");
        let height = rows / base.base.rows;
        let width = cols / base.base.cols;
        let shape = vec![height, width, base.base.rows, base.base.cols];
        let half: usize = shape.iter().product();
        let buf = self.alloc_local(2 * half, dtype.clone());
        ST { buf, shape, rows, cols, layout, base, elem: dtype, ker: self, base_offset: None }
    }

    /// Allocate a register [`RT`] tile (tinygrad `ker.rt`). `dims` is the
    /// `(rows, cols)` block size; the per-lane buffer is
    /// `height×width×elements_per_thread` flat.
    pub fn rt(&self, dims: (usize, usize), dtype: DType, layout: TileLayout, base: RTBaseShape) -> RT<'_> {
        let (rows, cols) = dims;
        assert_eq!(rows % base.base.rows, 0, "RT rows {rows} not a multiple of base {}", base.base.rows);
        assert_eq!(cols % base.base.cols, 0, "RT cols {cols} not a multiple of base {}", base.base.cols);
        let height = rows / base.base.rows;
        let width = cols / base.base.cols;
        let ept = base.base.elements_per_thread();
        let shape = vec![height, width, ept];
        let flat = shape.iter().product();
        let buf = self.alloc_reg(flat, dtype.clone());
        RT { buf, shape, layout, base, elem: dtype, ker: self }
    }

    /// Allocate a register vector [`RV`] tile (tinygrad `ker.rv`).
    pub fn rv(&self, length: usize, dtype: DType, layout: VecLayout, base: RTBaseShape) -> RV<'_> {
        let tiles = length / base.base.rows;
        let (outer, inner) = match layout {
            VecLayout::Ortho => (tiles, 1usize),
        };
        let shape = vec![outer, inner];
        let buf = self.alloc_reg(outer * inner, dtype.clone());
        RV { buf, shape, length, layout, base, elem: dtype, ker: self }
    }
}
