use morok_dtype::DType;
use snafu::Snafu;

mod recurrent;
pub use recurrent::{JitRecurrent, LstmState, RecurrentJit, StepTiming};

/// Shape + dtype descriptor for a single JIT input. Used by
/// `jit_wrapper!`-generated `prepare()` calls to allocate zero-initialized
/// placeholder buffers internally — callers no longer construct fake
/// `Tensor::zeros(..).realize()` placeholders.
#[derive(Clone, Debug)]
pub struct InputSpec {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl InputSpec {
    pub fn new(shape: &[usize], dtype: DType) -> Self {
        Self { shape: shape.to_vec(), dtype }
    }

    pub fn f32(shape: &[usize]) -> Self {
        Self::new(shape, DType::Float32)
    }

    pub fn i32(shape: &[usize]) -> Self {
        Self::new(shape, DType::Int32)
    }

    pub fn i64(shape: &[usize]) -> Self {
        Self::new(shape, DType::Int64)
    }
}

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum JitError {
    #[snafu(display("JIT not prepared: call prepare() first"))]
    NotPrepared,

    #[snafu(display("input buffer not found: {name}"))]
    InputBufferNotFound { name: &'static str },

    #[snafu(display("duplicate JIT input buffer: {name} aliases {duplicate_of} with {buffer_id:?}"))]
    DuplicateInputBuffer { name: &'static str, duplicate_of: &'static str, buffer_id: morok_device::BufferId },

    /// Wraps the user-supplied error type returned by a `jit_wrapper!` build
    /// closure. Genuine `Box<dyn>` because the closure's `E` is arbitrary.
    #[snafu(display("{source}"))]
    Build { source: Box<dyn std::error::Error + Send + Sync> },

    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(morok_tensor::error::Error, Box::new)))]
        source: Box<morok_tensor::error::Error>,
    },

    #[snafu(display("{source}"))]
    Device {
        #[snafu(source(from(morok_device::error::Error, Box::new)))]
        source: Box<morok_device::error::Error>,
    },

    /// `JitRecurrent::new` rejected a JIT whose output element count does not
    /// match the declared `head_len + |h| + |c|`. Typically means the `build`
    /// closure was changed and now emits a different layout than the wrapper
    /// expects.
    #[snafu(display(
        "JIT output layout mismatch: declared {declared_head} head + {declared_state} state elements \
         ({}), actual {actual} elements. Check that the `build` closure returns `cat([head, h, c], -1)` \
         with the declared shapes.",
        declared_head + declared_state
    ))]
    OutputLayoutMismatch { declared_head: usize, declared_state: usize, actual: usize },

    #[snafu(display("{source}"))]
    Runtime { source: morok_runtime::Error },
}

pub type Result<T> = std::result::Result<T, JitError>;
