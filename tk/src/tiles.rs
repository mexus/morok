//! Tile shape descriptors and layouts.
//!
//! These are the pure, data-only building blocks shared by every tile kind. A
//! [`BaseShape`] is one WMMA-sized fragment (e.g. 16×16); a full tile is a grid
//! of base shapes. The concrete tile wrappers (GL/ST/RT/RV) that bind a buffer
//! and a [`crate::Kernel`] live alongside the builder.

use crate::WARP_THREADS;
use crate::swizzle::Swizzle;

/// Register-tile element layout within a warp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TileLayout {
    Row,
    Col,
}

/// Register-vector layout.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum VecLayout {
    Ortho,
}

/// A WMMA-sized base fragment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BaseShape {
    pub rows: usize,
    pub cols: usize,
}

impl BaseShape {
    pub const fn num_elements(&self) -> usize {
        self.rows * self.cols
    }
    /// Elements each thread (lane) holds for one base fragment.
    pub const fn elements_per_thread(&self) -> usize {
        self.num_elements() / WARP_THREADS
    }
}

/// Shared-tile base fragment: a [`BaseShape`] plus its LDS [`Swizzle`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct STBaseShape {
    pub base: BaseShape,
    pub swizzle: Swizzle,
}

/// Register-tile base fragment: a [`BaseShape`] plus the per-lane stride (the
/// wave64 fragment stride).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RTBaseShape {
    pub base: BaseShape,
    pub stride: usize,
}

impl RTBaseShape {
    pub const fn elements_per_thread(&self) -> usize {
        self.base.elements_per_thread()
    }
    pub const fn num_strides(&self) -> usize {
        self.elements_per_thread() / self.stride
    }
}

// Predefined shared-tile base shapes.
pub const ST_16X16: STBaseShape = STBaseShape { base: BaseShape { rows: 16, cols: 16 }, swizzle: Swizzle::Identity };
pub const ST_16X16_SWIZZLED: STBaseShape =
    STBaseShape { base: BaseShape { rows: 16, cols: 16 }, swizzle: Swizzle::Sw16x16 };
pub const ST_32X32: STBaseShape = STBaseShape { base: BaseShape { rows: 32, cols: 32 }, swizzle: Swizzle::Sw32x32 };
pub const ST_16X32: STBaseShape = STBaseShape { base: BaseShape { rows: 16, cols: 32 }, swizzle: Swizzle::Sw16x32 };
pub const ST_32X16: STBaseShape = STBaseShape { base: BaseShape { rows: 32, cols: 16 }, swizzle: Swizzle::Sw32x16 };

// Predefined register-tile base shapes.
pub const RT_16X16: RTBaseShape = RTBaseShape { base: BaseShape { rows: 16, cols: 16 }, stride: 4 };
pub const RT_32X32: RTBaseShape = RTBaseShape { base: BaseShape { rows: 32, cols: 32 }, stride: 4 };
pub const RT_16X32: RTBaseShape = RTBaseShape { base: BaseShape { rows: 16, cols: 32 }, stride: 8 };
pub const RT_32X16: RTBaseShape = RTBaseShape { base: BaseShape { rows: 32, cols: 16 }, stride: 8 };
