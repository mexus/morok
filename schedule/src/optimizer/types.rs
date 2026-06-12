//! Type definitions for the kernel optimization layer.
//!
//! The optimization directive data types (`OptOps`, `OptArg`, `Opt`) now live in
//! `svod_ir` (so [`svod_ir::KernelInfo`] can carry an author-supplied
//! `opts_to_apply` list without `ir` depending on `schedule`). They are
//! re-exported here so existing `svod_schedule::optimizer::{Opt, OptArg,
//! OptOps}` paths keep resolving. The `OptError`-returning accessor methods stay
//! schedule-side via the [`OptArgExt`] extension trait, keeping `OptError` out of
//! `ir`.
//!
//! Note: `AxisType` is likewise re-exported from `svod_ir`.
pub use svod_ir::{AxisType, Opt, OptArg, OptOps};

use super::error::*;

/// `OptError`-returning accessors for [`OptArg`].
///
/// `OptArg` is defined in `svod_ir`, which cannot reference `schedule`'s
/// `OptError`; these fallible accessors therefore live here as an extension
/// trait. Bring it into scope (`use crate::optimizer::OptArgExt;`) wherever
/// `opt.arg.int()? / .tc()? / .swap()?` is used.
pub trait OptArgExt {
    /// Extract integer value, returning an error if not an `Int` variant.
    fn int(&self) -> Result<usize, OptError>;
    /// Extract tensor core configuration, returning an error if not a `TensorCore` variant.
    fn tc(&self) -> Result<(i32, usize, usize), OptError>;
    /// Extract swap configuration, returning an error if not a `Swap` variant.
    fn swap(&self) -> Result<usize, OptError>;
}

impl OptArgExt for OptArg {
    fn int(&self) -> Result<usize, OptError> {
        match self {
            OptArg::Int(v) => Ok(*v),
            _ => InvalidArgTypeSnafu { expected: "Int", found: self.type_name() }.fail(),
        }
    }

    fn tc(&self) -> Result<(i32, usize, usize), OptError> {
        match self {
            OptArg::TensorCore { tc_select, opt_level, use_tc } => Ok((*tc_select, *opt_level, *use_tc)),
            _ => InvalidArgTypeSnafu { expected: "TensorCore", found: self.type_name() }.fail(),
        }
    }

    fn swap(&self) -> Result<usize, OptError> {
        match self {
            OptArg::Swap { other_axis } => Ok(*other_axis),
            _ => InvalidArgTypeSnafu { expected: "Swap", found: self.type_name() }.fail(),
        }
    }
}
