//! Per-lane register-tile elementwise ops: constant fills (`clear`/`zero`/
//! `ones`/`neg_inf` and their [`RV`] analogs), `copy`, `transpose`, and the
//! generic `map`. Each is wave-safe (every wave operates on its own register
//! tile) and lowers through [`Group::elementwise`](super::Group).

use std::sync::Arc;

use svod_ir::{ConstValue, UOp};

use super::Group;
use crate::index::{Idx, flat_index, load_at};
use crate::tile::{RT, RV, RegTile};

impl<'k> Group<'k> {
    /// Fill a register tile with `value` (tinygrad `clear`).
    fn clear(&self, reg: RT<'k>, value: f64) -> RT<'k> {
        // Per-lane register fill: identical on every wave (each clears its own RT).
        let (buf, shape, is_float, elem) =
            (reg.uop().clone(), reg.shape().to_vec(), reg.elem().is_float(), reg.elem().clone());
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let cv = if is_float { ConstValue::Float(value) } else { ConstValue::Int(value as i64) };
            flat_index(&buf, &shape, idxs).store(UOp::const_(elem.clone(), cv))
        });
        self.finalize_reg(reg, ended)
    }

    /// Zero a register tile.
    pub fn zero(&self, reg: RT<'k>) -> RT<'k> {
        self.clear(reg, 0.0)
    }
    /// Fill a register tile with `1` (tinygrad `ones`).
    pub fn ones(&self, reg: RT<'k>) -> RT<'k> {
        self.clear(reg, 1.0)
    }
    /// Fill a register tile with `-∞` (tinygrad `neg_inf`).
    pub fn neg_inf(&self, reg: RT<'k>) -> RT<'k> {
        self.clear(reg, f64::NEG_INFINITY)
    }
    /// Fill a register *vector* with `value` (the [`RV`] analog of [`clear`]).
    ///
    /// # Panics
    /// Panics if the group has more than one warp.
    pub fn clear_rv(&self, rv: RV<'k>, value: f64) -> RV<'k> {
        assert_eq!(self.warps, 1, "clear_rv is a single-warp op");
        let (buf, shape, is_float, elem) =
            (rv.uop().clone(), rv.shape().to_vec(), rv.elem().is_float(), rv.elem().clone());
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let cv = if is_float { ConstValue::Float(value) } else { ConstValue::Int(value as i64) };
            flat_index(&buf, &shape, idxs).store(UOp::const_(elem.clone(), cv))
        });
        self.finalize_tile(rv, ended)
    }
    /// Zero a register vector.
    ///
    /// # Panics
    /// Panics if the group has more than one warp.
    pub fn zero_rv(&self, rv: RV<'k>) -> RV<'k> {
        self.clear_rv(rv, 0.0)
    }
    /// Fill a register vector with `1`.
    ///
    /// # Panics
    /// Panics if the group has more than one warp.
    pub fn ones_rv(&self, rv: RV<'k>) -> RV<'k> {
        self.clear_rv(rv, 1.0)
    }
    /// Fill a register vector with `-∞`.
    ///
    /// # Panics
    /// Panics if the group has more than one warp.
    pub fn neg_inf_rv(&self, rv: RV<'k>) -> RV<'k> {
        self.clear_rv(rv, f64::NEG_INFINITY)
    }

    /// Copy `src` into `dst` element-wise (tinygrad `copy`), casting on a dtype
    /// mismatch. Generic over [`RT`]/[`RV`] (softmax copies a register vector).
    ///
    /// # Panics
    /// Panics if `dst` and `src` have different shapes.
    pub fn copy<T: RegTile<'k>>(&self, dst: T, src: &T) -> T {
        // Per-lane register op: wave-safe (each wave copies its own RT).
        assert_eq!(dst.shape(), src.shape(), "copy: shape mismatch");
        let (sbuf, sshape, selem) = (self.anchor(src.uop()), src.shape().to_vec(), src.elem().clone());
        let (dbuf, dshape, delem) = (dst.uop().clone(), dst.shape().to_vec(), dst.elem().clone());
        let ended = self.elementwise(&dshape.clone(), move |idxs| {
            let mut load = load_at(&sbuf, &sshape, idxs);
            if selem != delem {
                load = load.cast(delem.clone());
            }
            flat_index(&dbuf, &dshape, idxs).store(load)
        });
        self.finalize_tile(dst, ended)
    }

    /// Transpose `src` into `dst` element-wise (tinygrad `transpose`): write
    /// `src[h, w, inner]` to `dst[w, h, inner]`, casting on a dtype mismatch.
    /// Used by FA to swap a register fragment's height/width before a WMMA.
    ///
    /// # Panics
    /// Panics if the tile rank is less than 3 (it permutes the leading
    /// `[height, width, inner]` axes).
    pub fn transpose(&self, dst: RT<'k>, src: &RT<'k>) -> RT<'k> {
        // Per-lane register op: wave-safe (each wave transposes its own RT).
        let (sbuf, sshape, selem) = (self.anchor(src.uop()), src.shape().to_vec(), src.elem().clone());
        let (dbuf, dshape, delem) = (dst.uop().clone(), dst.shape().to_vec(), dst.elem().clone());
        // Iterate the source `[height, width, inner]`; write `dst[width, height, inner]`.
        let ended = self.elementwise(&sshape.clone(), move |idxs| {
            let mut load = load_at(&sbuf, &sshape, idxs);
            if selem != delem {
                load = load.cast(delem.clone());
            }
            flat_index(&dbuf, &dshape, &[idxs[1].clone(), idxs[0].clone(), idxs[2].clone()]).store(load)
        });
        self.finalize_reg(dst, ended)
    }

    /// Apply `op` to every element of a register tile (tinygrad `Group.map`):
    /// open a loop per logical dim, load the element, store `op(value, idx)`
    /// back, and rewrap the tile after the store. `idx` lets `op` index *other*
    /// tiles at the same position (the RV-broadcast and FA causal-mask path).
    pub fn map<T, F>(&self, a: T, op: F) -> T
    where
        T: RegTile<'k>,
        F: Fn(&Arc<UOp>, &[Idx]) -> Arc<UOp>,
    {
        // Per-lane register op: wave-safe (each wave maps its own RT).
        let (buf, shape) = (a.uop().clone(), a.shape().to_vec());
        let rbuf = self.anchor(&buf); // anchored read; store to the raw buffer
        let ended = self.elementwise(&shape.clone(), move |idxs| {
            let val = load_at(&rbuf, &shape, idxs);
            flat_index(&buf, &shape, idxs).store(op(&val, idxs))
        });
        self.finalize_tile(a, ended)
    }
}
