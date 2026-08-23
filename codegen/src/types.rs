//! Types for code generation.

use svod_dtype::DType;

// Re-export new unified types from device crate
pub use svod_device::device::{ProgramSpec, Variable};
pub use svod_dtype::DeviceSpec;

/// A rendered kernel ready for compilation and execution.
#[derive(Debug, Clone)]
pub struct RenderedKernel {
    /// The generated code (LLVM IR, CUDA C, etc.)
    pub code: String,

    /// Kernel name (used as entry point and for debugging/caching).
    pub name: String,

    /// Buffer argument information.
    pub buffer_args: Vec<BufferArg>,

    /// Variable names in order (for populating vars array at runtime).
    pub var_names: Vec<String>,

    /// Complete PARAM ABI in generated source-signature order.
    pub abi: Vec<svod_device::device::AbiParamDescriptor>,

    /// Per-UOp source bindings emitted by structured text renderers.
    ///
    /// This is diagnostic/verifier metadata only. It lets safety tests tie an
    /// emitted branch, load, phi, or store back to the exact LINEAR operation
    /// without parsing unrelated instructions in the complete source string.
    pub operations: Vec<RenderedOperation>,
}

/// Generated source owned by one LINEAR UOp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedOperation {
    pub uop_id: u64,
    pub op: String,
    pub source_ids: Vec<u64>,
    pub result: Option<String>,
    pub lines: Vec<String>,
}

/// Information about a buffer argument to the kernel.
#[derive(Debug, Clone)]
pub struct BufferArg {
    /// Argument index.
    pub index: usize,

    /// Buffer name.
    pub name: String,

    /// Data type.
    pub dtype: DType,

    /// Whether this is an output buffer.
    pub is_output: bool,
}

impl RenderedKernel {
    /// Create a new rendered kernel.
    pub fn new(code: String, name: String) -> Self {
        Self { code, name, buffer_args: Vec::new(), var_names: Vec::new(), abi: Vec::new(), operations: Vec::new() }
    }

    /// Add a buffer argument.
    pub fn add_buffer_arg(&mut self, arg: BufferArg) {
        self.buffer_args.push(arg);
    }
}
