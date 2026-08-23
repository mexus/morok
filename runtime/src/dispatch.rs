//! Kernel dispatch via libffi.
//!
//! `KernelCif` wraps libffi's `Cif` with Send+Sync so it can be stored on
//! kernel structs and shared across rayon threads. A thread-local buffer
//! avoids per-call allocation for the packed u64 args.

use std::cell::RefCell;

use libffi::low::CodePtr;
use libffi::middle::{self, Cif, Type};
use smallvec::SmallVec;

/// Send+Sync wrapper for libffi Cif.
///
/// # Safety
///
/// `Cif` is `!Send + !Sync` due to raw pointer fields (conservative auto-trait).
/// Once prepared, a CIF is immutable — `Cif::call(&self)` only reads the
/// descriptor and `ffi_call` does not mutate it for non-closure calls.
/// All our CIFs describe stateless kernel signatures (N × u64 → void).
pub(crate) struct KernelCif {
    cif: Cif,
    arg_count: usize,
    abi: Vec<svod_device::device::AbiParamKind>,
}

unsafe impl Send for KernelCif {}
unsafe impl Sync for KernelCif {}

impl KernelCif {
    pub fn from_abi(abi: &[svod_device::device::AbiParamDescriptor]) -> Self {
        let types =
            abi.iter().map(|arg| if arg.is_storage() { Type::pointer() } else { Type::i32() }).collect::<Vec<_>>();
        Self {
            cif: Cif::new(types, Type::void()),
            arg_count: abi.len(),
            abi: abi.iter().map(|arg| arg.kind.clone()).collect(),
        }
    }

    /// Call the kernel, packing buffers + vals as u64 args.
    ///
    /// Uses a thread-local buffer for the packed args — zero allocation
    /// after warmup. The `SmallVec<[Arg; 32]>` avoids heap allocation for
    /// kernels with ≤32 arguments (the common case); kernels above that cap
    /// fall back to a heap allocation per dispatch.
    ///
    /// `var_patch`: if `Some((var_idx, value))`, patches
    /// `vals[var_idx]` to `value` before calling.
    #[inline]
    pub unsafe fn dispatch(
        &self,
        fn_ptr: *const (),
        buffers: &[*mut u8],
        vals: &[i64],
        var_patch: Option<(usize, usize)>,
    ) -> svod_device::Result<()> {
        let expected_buffers =
            self.abi.iter().filter(|kind| matches!(kind, svod_device::device::AbiParamKind::Storage(_))).count();
        let expected_vals = self.arg_count - expected_buffers;
        if buffers.len() != expected_buffers || vals.len() != expected_vals {
            return Err(svod_device::Error::ProgramAbiMismatch {
                reason: format!(
                    "kernel dispatch expected {expected_buffers} buffers/{expected_vals} scalars, got {}/{}",
                    buffers.len(),
                    vals.len()
                ),
            });
        }

        thread_local! {
            static PACKED: RefCell<SmallVec<[u64; 32]>> = RefCell::new(SmallVec::new());
        }

        PACKED.with_borrow_mut(|packed| {
            if packed.len() != self.arg_count {
                packed.resize(self.arg_count, 0);
            }

            let (mut buffer_idx, mut var_idx) = (0usize, 0usize);
            for (arg_idx, kind) in self.abi.iter().enumerate() {
                match kind {
                    svod_device::device::AbiParamKind::Storage(_) => {
                        packed[arg_idx] = buffers[buffer_idx] as u64;
                        buffer_idx += 1;
                    }
                    svod_device::device::AbiParamKind::Scalar => {
                        packed[arg_idx] = vals[var_idx] as u64;
                        var_idx += 1;
                    }
                }
            }

            if let Some((var_idx, value)) = var_patch {
                let Some(arg_idx) = self
                    .abi
                    .iter()
                    .enumerate()
                    .filter(|(_, kind)| matches!(kind, svod_device::device::AbiParamKind::Scalar))
                    .nth(var_idx)
                    .map(|(idx, _)| idx)
                else {
                    return;
                };
                packed[arg_idx] = value as u64;
            }

            let mut ffi_args: SmallVec<[middle::Arg; 32]> = SmallVec::with_capacity(self.arg_count);
            for value in packed.iter() {
                ffi_args.push(middle::arg(value));
            }

            unsafe {
                self.cif.call::<()>(CodePtr(fn_ptr as *mut _), &ffi_args);
            }
        });
        if let Some((var_idx, _)) = var_patch
            && var_idx >= expected_vals
        {
            return Err(svod_device::Error::ProgramAbiMismatch {
                reason: format!(
                    "kernel dispatch scalar patch index {var_idx} out of range for {expected_vals} scalars"
                ),
            });
        }
        Ok(())
    }
}
