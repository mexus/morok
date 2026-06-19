//! Operator sugar on register tiles — the Rust analog of tinykittens'
//! `TileMathMixin`/`ElementwiseMixin`. With these impls a kernel body reads like
//! the math (`(att - &max_vec).exp2()`, `o_reg * &scale_vec`, `q * 0.5`) instead
//! of `warp.exp2(warp.sub_rv(att, &max_vec))`.
//!
//! Each operator routes through the kernel's warp [`Group`](crate::Group) math
//! (so the actual lowering stays in one place). Operators consume the left tile
//! by value — tiles are cheap `Arc`-backed wrappers — and return a fresh tile,
//! matching the `x = x <op> y` reassignment flow tile ops already use.
//!
//! Three right-hand flavors, mirroring `math.rs`:
//! - same-shape tile (`T <op> &T`),
//! - register-vector broadcast into an `RT` (`RT <op> &RV`),
//! - scalar (`T <op> f64`).

use core::ops::{Add, Div, Mul, Sub};

use crate::tile::{RT, RV};

/// `T <op> &T` (same-shape tile) → `Group::<op>`.
macro_rules! binop_same {
    ($T:ident, $Tr:ident, $m:ident, $g:ident) => {
        impl<'k> $Tr<&$T<'k>> for $T<'k> {
            type Output = $T<'k>;
            fn $m(self, rhs: &$T<'k>) -> $T<'k> {
                self.ker().warp().$g(self, rhs)
            }
        }
    };
}

/// `RT <op> &RV` (register-vector broadcast) → `Group::<op>_rv`.
macro_rules! binop_rt_rv {
    ($Tr:ident, $m:ident, $g:ident) => {
        impl<'k> $Tr<&RV<'k>> for RT<'k> {
            type Output = RT<'k>;
            fn $m(self, rhs: &RV<'k>) -> RT<'k> {
                self.ker().warp().$g(self, rhs)
            }
        }
    };
}

/// `T <op> f64` (scalar) → `Group::<op>_scalar`.
macro_rules! binop_scalar {
    ($T:ident, $Tr:ident, $m:ident, $g:ident) => {
        impl<'k> $Tr<f64> for $T<'k> {
            type Output = $T<'k>;
            fn $m(self, rhs: f64) -> $T<'k> {
                self.ker().warp().$g(self, rhs)
            }
        }
    };
}

binop_same!(RT, Add, add, add);
binop_same!(RT, Sub, sub, sub);
binop_same!(RT, Mul, mul, mul);
binop_same!(RT, Div, div, div);
binop_same!(RV, Add, add, add);
binop_same!(RV, Sub, sub, sub);
binop_same!(RV, Mul, mul, mul);
binop_same!(RV, Div, div, div);

binop_rt_rv!(Add, add, add_rv);
binop_rt_rv!(Sub, sub, sub_rv);
binop_rt_rv!(Mul, mul, mul_rv);
binop_rt_rv!(Div, div, div_rv);

binop_scalar!(RT, Add, add, add_scalar);
binop_scalar!(RT, Sub, sub, sub_scalar);
binop_scalar!(RT, Mul, mul, mul_scalar);
binop_scalar!(RT, Div, div, div_scalar);
binop_scalar!(RV, Add, add, add_scalar);
binop_scalar!(RV, Sub, sub, sub_scalar);
binop_scalar!(RV, Mul, mul, mul_scalar);
binop_scalar!(RV, Div, div, div_scalar);

impl<'k> RT<'k> {
    /// `exp2(self)` element-wise.
    pub fn exp2(self) -> Self {
        self.ker().warp().exp2(self)
    }
    /// `max(self, other)` element-wise.
    ///
    /// # Panics
    /// Panics if `self` and `other` have different shapes.
    pub fn maximum(self, other: &Self) -> Self {
        self.ker().warp().maximum(self, other)
    }
}

impl<'k> RV<'k> {
    /// `exp2(self)` element-wise.
    pub fn exp2(self) -> Self {
        self.ker().warp().exp2(self)
    }
    /// `max(self, other)` element-wise.
    ///
    /// # Panics
    /// Panics if `self` and `other` have different shapes.
    pub fn maximum(self, other: &Self) -> Self {
        self.ker().warp().maximum(self, other)
    }
}
