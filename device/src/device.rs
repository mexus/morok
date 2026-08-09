//! Device abstraction.
//!
//! This module provides a unified Device abstraction that owns:
//! - **Renderer**: Transforms UOp graphs into source code (ProgramSpec)
//! - **Compiler**: Transforms source code into executable bytes
//! - **Runtime**: Creates executable Programs from compiled bytes
//! - **Allocator**: Manages memory allocation for buffers
//!
//! This design allows multiple backends (LLVM, CUDA, Metal, WebGPU) to coexist
//! and share compiled kernels via the method cache.

use std::collections::HashMap;
use std::sync::Arc;

use svod_dtype::DeviceSpec;
use svod_ir::{BinaryOp, ConstValue, Op, TernaryOp, UOp, UnaryOp};

use crate::allocator::Allocator;
use snafu::OptionExt;

use crate::error::{Error, Result, VarOutOfBoundsSnafu, WrongStageSnafu};

/// A compiled, executable kernel program.
///
/// This trait abstracts over different backend executors (LLVM JIT, CUDA, Metal, etc.).
/// Each backend implements this to provide unified execution interface.
///
/// Implementations must be stateless and reentrant from the host perspective.
/// The runtime caches and shares programs across execution plans, and may invoke
/// the same program from multiple host threads when dependency analysis proves
/// the buffer accesses are independent.
///
/// # Calling convention
///
/// Variable values are passed as a positional array (`vals`) rather than a
/// named HashMap. The order matches `var_names` in `CompiledSpec`.
pub trait Program: Send + Sync {
    /// Execute the kernel with given buffers and variable values.
    ///
    /// # Arguments
    ///
    /// * `buffers` - Raw pointers to buffer data (input and output buffers)
    /// * `vals` - Variable values in positional order (matches `var_names` in CompiledSpec)
    /// * `global_size` - Global work size (for GPU backends, None for CPU)
    /// * `local_size` - Local work size (for GPU backends, None for CPU)
    /// * `wait` - Block until this dispatch completes before returning. GPU
    ///   backends submit asynchronously and rely on the device timeline for
    ///   ordering, so `wait=false` returns right after submit. Pass `true`
    ///   only when the caller needs completion *without* a subsequent synchronizing read
    ///   (e.g. benchmark timing). Synchronous backends (CPU) ignore it.
    ///
    /// # Safety
    ///
    /// This is unsafe because:
    /// - Buffer pointers must be valid and properly aligned
    /// - Buffer sizes must match what the kernel expects
    /// - Caller must ensure no data races during execution
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        wait: bool,
    ) -> Result<()>;

    /// Get the kernel name (for debugging/profiling).
    fn name(&self) -> &str;

    /// Downcast hook so a backend graph factory can recover its concrete
    /// program type (to read backend-specific fields) when pre-building dispatch
    /// packets. Default returns nothing graphable.
    fn as_any(&self) -> &dyn std::any::Any {
        &()
    }

    /// Mint a reusable per-plan execution context, or `None` for per-call
    /// dispatch. An `ExecutionPlan` calls this once on its first kernel's
    /// program and reuses the returned context for every dispatch, so all the
    /// plan's kernels share one queue (distinct plans → distinct queues for
    /// cross-plan parallelism). Default `None`: the backend's `execute` is
    /// already self-contained (CPU), so the plan dispatches per-call.
    fn new_exec_context(&self) -> Result<Option<Box<dyn PlanContext>>> {
        Ok(None)
    }

    /// Static GPU resource usage (VGPR/SGPR/LDS/scratch) decoded from the
    /// compiled kernel descriptor, for profiling/occupancy. Default `None`:
    /// backends without a descriptor (CPU) report nothing.
    fn resource_usage(&self) -> Option<crate::profile::KernelResources> {
        None
    }
}

/// One graphable kernel: a program plus its fixed buffer pointers and launch
/// dims, captured once. Buffer pointers are plan-owned and stable across calls
/// (`ExecutionPlan::buffers`), so a graph can bake them in and only re-patch
/// variable-derived launch dims + `vals` on replay. Bundles the device,
/// program, buffers, and launch vars of a single graph call.
pub struct GraphKernel<'a> {
    pub program: &'a dyn Program,
    pub buffers: Vec<*mut u8>,
    pub vals: Vec<i64>,
    pub global_size: Option<[usize; 3]>,
    pub local_size: Option<[usize; 3]>,
    /// Emission-order indices of the producer kernels this kernel must wait on
    /// (RAW/WAR/WAW hazards over resolved buffer GVAs). Computed by the host
    /// hazard analysis in the same flatten order the kernels are emitted, so a
    /// DAG-aware backend can strip the per-dispatch BARRIER bit and gate only on
    /// these producers' completion signals. Empty = no producer in this graph.
    pub deps: Vec<usize>,
}

/// A pre-captured kernel chain replayed with one submit. Backends that can
/// pre-build their dispatch packets implement this so repeated inference pays
/// per-graph, not per-kernel, launch cost. Replay is equivalent to running every
/// captured kernel in order.
pub trait Graph: Send + Sync {
    /// Re-dispatch the captured chain. `vals` are positional updated launch
    /// vars (same order as capture); empty replays the baked-in values.
    fn replay(&self, vals: &[i64]) -> Result<()>;
}

/// A reusable per-plan execution context: a lease of device-level dispatch
/// resources (e.g. an AMD queue from the pool) held by an `ExecutionPlan` for
/// its lifetime so every kernel in the plan dispatches onto the same queue.
/// Minted by [`Program::new_exec_context`]; backends with no reusable context
/// (CPU) return `None` and the plan falls back to per-call [`Program::execute`].
pub trait PlanContext: Send + Sync {
    /// Dispatch one kernel of the plan onto this context. `program` belongs to
    /// the same plan and therefore the same backend that minted this context
    /// (a plan is single-device) — a construction invariant, not a runtime
    /// check. Submits asynchronously like [`Program::execute`] with `wait=false`.
    ///
    /// `profile` requests a per-dispatch HW timestamp handle (`None` otherwise,
    /// e.g. CPU). **The caller MUST retain the returned handle until after
    /// [`PlanContext::synchronize`]**: a profiling backend may bracket the async
    /// dispatch with GPU-clock probes that write into scratch the handle owns, so
    /// dropping it early frees that scratch while the GPU is still writing. Pass
    /// `false` on the fire-and-forget path that discards the handle.
    ///
    /// # Safety
    ///
    /// Same contract as [`Program::execute`]: buffer pointers must be valid and
    /// correctly sized, and the caller must avoid data races.
    unsafe fn dispatch(
        &self,
        program: &dyn Program,
        buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        local_size: Option<[usize; 3]>,
        profile: bool,
    ) -> Result<Option<Arc<dyn crate::DispatchTimestamps>>>;

    /// Drain this context's in-flight work (profiled-timestamp harvest).
    fn synchronize(&self) -> Result<()>;

    /// Arm hardware performance counters for subsequent profiling dispatches on
    /// this context (empty disables). Default no-op: backends without PMC ignore
    /// it. Counters are reported via [`DispatchTimestamps::counters`].
    fn set_pmc(&self, _counters: &[crate::profile::PmcCounter]) {}

    /// Whether hardware counter collection is currently available on this
    /// context (backend supports PMC and the GPU is in a stable power state).
    /// Default `false`.
    fn pmc_available(&self) -> bool {
        false
    }
}

/// Compilation result carrying source (JIT) or bytes (AOT).
///
/// Different backends need different information:
/// - LLVM JIT: needs source code to compile during runtime
/// - CUDA: needs PTX/CUBIN bytes to load
/// - Metal: needs metallib bytes to load
///
/// This design allows the RuntimeFactory to access whatever it needs
/// without requiring separate code paths for JIT vs AOT backends.
#[derive(Debug, Clone)]
pub struct CompiledSpec {
    /// Entry point function name
    pub name: String,

    /// Source code (for JIT backends like LLVM)
    /// Set to Some(...) for LLVM JIT, None for AOT backends
    pub src: Option<String>,

    /// Compiled bytes (for AOT backends like CUDA/Metal)
    /// Empty for LLVM JIT, populated for AOT backends
    pub bytes: Vec<u8>,

    /// Original AST for cache key construction via hash consing
    pub ast: Arc<UOp>,

    /// Variable names in order for populating vars array at runtime.
    /// Includes runtime variables such as core_id.
    pub var_names: Vec<String>,

    /// Symbolic global work size for dispatch.
    pub global_size: [Arc<UOp>; 3],

    /// Symbolic local work size for dispatch. None means direct global-id execution.
    pub local_size: Option<[Arc<UOp>; 3]>,

    /// Number of buffer arguments (for CIF construction at compile time).
    pub buf_count: usize,
}

impl CompiledSpec {
    /// Create a new CompiledSpec for JIT backends (source-based).
    pub fn from_source(name: String, src: String, ast: Arc<UOp>, buf_count: usize) -> Self {
        Self {
            name,
            src: Some(src),
            bytes: Vec::new(),
            ast,
            var_names: Vec::new(),
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
            buf_count,
        }
    }

    /// Create a new CompiledSpec for AOT backends (bytecode-based).
    pub fn from_bytes(name: String, bytes: Vec<u8>, ast: Arc<UOp>) -> Self {
        Self {
            name,
            src: None,
            bytes,
            ast,
            var_names: Vec::new(),
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
            buf_count: 0,
        }
    }

    /// Create a new CompiledSpec with work sizes for JIT backends.
    pub fn from_source_with_sizes(
        name: String,
        src: String,
        ast: Arc<UOp>,
        global_size: [usize; 3],
        local_size: Option<[usize; 3]>,
        buf_count: usize,
    ) -> Self {
        Self {
            name,
            src: Some(src),
            bytes: Vec::new(),
            ast,
            var_names: Vec::new(),
            global_size: concrete_launch_size(global_size),
            local_size: local_size.map(concrete_launch_size),
            buf_count,
        }
    }
}

/// Concrete launch dimensions passed to backend runtimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConcreteLaunchDims {
    pub global_size: [usize; 3],
    pub local_size: Option<[usize; 3]>,
}

fn default_launch_size() -> [Arc<UOp>; 3] {
    [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)]
}

fn concrete_launch_size(size: [usize; 3]) -> [Arc<UOp>; 3] {
    [UOp::index_const(size[0] as i64), UOp::index_const(size[1] as i64), UOp::index_const(size[2] as i64)]
}

fn const_value_to_i64(value: ConstValue) -> Result<i64> {
    match value {
        ConstValue::Invalid => Err(Error::Runtime { message: "Invalid reached launch-size evaluation".to_string() }),
        ConstValue::Int(v) => Ok(v),
        ConstValue::UInt(v) => i64::try_from(v)
            .map_err(|_| Error::Runtime { message: format!("launch-size constant {v} does not fit i64") }),
        ConstValue::Bool(v) => Ok(i64::from(v)),
        ConstValue::Float(v) => {
            Err(Error::Runtime { message: format!("launch-size expression must be integer, got float constant {v}") })
        }
    }
}

fn validate_var_bound(name: &str, value: i64, min_val: i64, max_val: i64) -> Result<()> {
    snafu::ensure!(
        value >= min_val && value <= max_val,
        VarOutOfBoundsSnafu { name, value, min: min_val, max: max_val }
    );
    Ok(())
}

fn checked_launch_binary(op: BinaryOp, lhs: i64, rhs: i64) -> Result<i64> {
    // EXHAUSTIVE match — no `_` catch-all. When a new `BinaryOp` variant is
    // added to `svod_ir::types::BinaryOp`, the compiler will fail this match
    // and force an explicit decision, instead of silently producing wrong
    // launch dims at runtime. A symbolic evaluator codegen'd from the renderer
    // pipeline would inherit every operator automatically; until svod unifies
    // the two evaluators, this exhaustive match is the belt-and-suspenders
    // that approximates the same guarantee.
    let value: Option<i64> = match op {
        // Integer arithmetic — checked for overflow.
        BinaryOp::Add => lhs.checked_add(rhs),
        BinaryOp::Sub => lhs.checked_sub(rhs),
        BinaryOp::Mul => lhs.checked_mul(rhs),
        BinaryOp::Idiv => (rhs != 0).then(|| lhs.checked_div(rhs)).flatten(),
        BinaryOp::Mod => (rhs != 0).then(|| lhs.checked_rem(rhs)).flatten(),
        BinaryOp::Max => Some(lhs.max(rhs)),
        // Integer power: only support non-negative exponents that fit in u32.
        BinaryOp::Pow => u32::try_from(rhs).ok().and_then(|e| lhs.checked_pow(e)),
        // Bitwise / shift. Negative shifts and shifts ≥ 64 are rejected.
        BinaryOp::Shl => u32::try_from(rhs).ok().filter(|&r| r < 64).and_then(|r| lhs.checked_shl(r)),
        BinaryOp::Shr => u32::try_from(rhs).ok().filter(|&r| r < 64).and_then(|r| lhs.checked_shr(r)),
        BinaryOp::And => Some(lhs & rhs),
        BinaryOp::Or => Some(lhs | rhs),
        BinaryOp::Xor => Some(lhs ^ rhs),
        // Comparisons — fold to 0/1 (consistent with IR's symbolic rewrite
        // pipeline where Bool is i1 and may participate in arithmetic via
        // CAST).
        BinaryOp::Lt => Some(i64::from(lhs < rhs)),
        BinaryOp::Le => Some(i64::from(lhs <= rhs)),
        BinaryOp::Eq => Some(i64::from(lhs == rhs)),
        BinaryOp::Ne => Some(i64::from(lhs != rhs)),
        BinaryOp::Gt => Some(i64::from(lhs > rhs)),
        BinaryOp::Ge => Some(i64::from(lhs >= rhs)),
        // Float-only / nonsense for launch dims.
        BinaryOp::Fdiv => {
            return Err(Error::Runtime {
                message: "Fdiv (float division) in launch-size expression — launch dims must be integer".into(),
            });
        }
        BinaryOp::Threefry => {
            return Err(Error::Runtime {
                message: "Threefry (PRNG) in launch-size expression — this is almost certainly a scheduler bug".into(),
            });
        }
    };

    value.ok_or_else(|| Error::Runtime { message: format!("invalid launch-size arithmetic: {lhs} {op:?} {rhs}") })
}

/// Evaluate a ternary op in a launch-size expression. `MulAcc(a, b, c) = a*b+c`
/// (overflow-checked); `Where(cond, t, f)` selects on a nonzero predicate.
/// EXHAUSTIVE match so a new `TernaryOp` variant fails the compile, not silently
/// at runtime.
fn checked_launch_ternary(op: TernaryOp, a: i64, b: i64, c: i64) -> Result<i64> {
    let value = match op {
        TernaryOp::MulAcc => a.checked_mul(b).and_then(|ab| ab.checked_add(c)),
        TernaryOp::Where => Some(if a != 0 { b } else { c }),
    };
    value.ok_or_else(|| Error::Runtime { message: format!("invalid launch-size ternary: {op:?}({a}, {b}, {c})") })
}

fn eval_launch_expr(expr: &Arc<UOp>, vars: &HashMap<&str, i64>) -> Result<i64> {
    match expr.op() {
        Op::Const(value) => const_value_to_i64(value.0),
        Op::DefineVar { name, min_val, max_val } => {
            let value = vars.get(name.as_str()).copied().ok_or_else(|| Error::Runtime {
                message: format!("missing runtime value for launch-size variable {name}"),
            })?;
            validate_var_bound(name, value, *min_val, *max_val)?;
            Ok(value)
        }
        Op::Bind { var, value } => {
            let bound = eval_launch_expr(value, vars)?;
            if let Op::DefineVar { name, min_val, max_val } = var.op() {
                validate_var_bound(name, bound, *min_val, *max_val)?;
            }
            Ok(bound)
        }
        Op::Binary(op, lhs, rhs) => {
            checked_launch_binary(*op, eval_launch_expr(lhs, vars)?, eval_launch_expr(rhs, vars)?)
        }
        // The symbolic simplifier fuses `a*b + c` (e.g. `16*ts − 1` from a
        // reshaped symbolic sequence axis) into a single MulAcc; `Where` can
        // appear when a launch dim is gated on a symbolic predicate. Evaluate
        // both rather than rejecting — the alternative is the scheduler
        // silently never fusing, which it does.
        Op::Ternary(op, a, b, c) => checked_launch_ternary(
            *op,
            eval_launch_expr(a, vars)?,
            eval_launch_expr(b, vars)?,
            eval_launch_expr(c, vars)?,
        ),
        Op::Unary(op, src) => checked_launch_unary(*op, eval_launch_expr(src, vars)?),
        Op::Cast { src, .. } | Op::BitCast { src, .. } | Op::After { passthrough: src, .. } => {
            eval_launch_expr(src, vars)
        }
        other => Err(Error::Runtime { message: format!("unsupported launch-size expression op: {other:?}") }),
    }
}

fn checked_launch_unary(op: UnaryOp, src: i64) -> Result<i64> {
    // EXHAUSTIVE match (no `_` catch-all) so a new `UnaryOp` variant fails the
    // build instead of silently corrupting launch dims. See the analogous
    // comment on `checked_launch_binary` for the longer rationale.
    let value: Option<i64> = match op {
        UnaryOp::Neg => src.checked_neg(),
        UnaryOp::Abs => src.checked_abs(),
        UnaryOp::Not => Some(!src),
        UnaryOp::Sign => Some(src.signum()),
        UnaryOp::Square => src.checked_mul(src),
        // For integer launch dims `trunc/floor/ceil/round` are identity since
        // the input is already an integer. A symbolic evaluator would collapse
        // these via the rewrite engine; the explicit arms here are the same
        // outcome.
        UnaryOp::Trunc | UnaryOp::Floor | UnaryOp::Ceil | UnaryOp::Round => Some(src),
        // Float-only — these have no meaning on integer launch dims and would
        // never reach here from a correct schedule.
        UnaryOp::Sqrt
        | UnaryOp::Rsqrt
        | UnaryOp::Exp
        | UnaryOp::Exp2
        | UnaryOp::Log
        | UnaryOp::Log2
        | UnaryOp::Sin
        | UnaryOp::Cos
        | UnaryOp::Tan
        | UnaryOp::Reciprocal
        | UnaryOp::Erf => {
            return Err(Error::Runtime {
                message: format!("float-only unary op {op:?} in launch-size expression — schedule bug"),
            });
        }
    };

    value.ok_or_else(|| Error::Runtime { message: format!("invalid launch-size unary arithmetic: {op:?} {src}") })
}

fn eval_launch_size(size: &[Arc<UOp>; 3], vars: &HashMap<&str, i64>) -> Result<[usize; 3]> {
    let mut out = [1usize; 3];
    for (idx, expr) in size.iter().enumerate() {
        let value = eval_launch_expr(expr, vars)?;
        if value <= 0 {
            return Err(Error::Runtime {
                message: format!("launch dimension {idx} evaluated to non-positive value {value}"),
            });
        }
        out[idx] = usize::try_from(value).map_err(|_| Error::Runtime {
            message: format!("launch dimension {idx} value {value} does not fit usize"),
        })?;
    }
    Ok(out)
}

/// A compiler that transforms source code into a compiled specification.
///
/// This trait abstracts over different compilation backends:
/// - LLVM: IR validation (JIT compiles at runtime)
/// - CUDA: CUDA C -> PTX/CUBIN
/// - Metal: Metal Shading Language -> metallib
/// - WebGPU: WGSL -> SPIR-V
pub trait Compiler: Send + Sync {
    /// Compile a program specification into executable form.
    ///
    /// # Arguments
    ///
    /// * `spec` - The program specification containing source code and metadata
    ///
    /// # Returns
    ///
    /// A CompiledSpec containing:
    /// - For JIT backends (LLVM): source code in `src` field, empty `bytes`
    /// - For AOT backends (CUDA/Metal): compiled bytes in `bytes` field, no `src`
    ///
    /// # Examples
    ///
    /// JIT backend (LLVM):
    /// ```ignore
    /// let compiled = compiler.compile(&spec)?;
    /// assert!(compiled.src.is_some());
    /// assert!(compiled.bytes.is_empty());
    /// ```
    ///
    /// AOT backend (CUDA):
    /// ```ignore
    /// let compiled = compiler.compile(&spec)?;
    /// assert!(compiled.src.is_none());
    /// assert!(!compiled.bytes.is_empty());
    /// ```
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec>;

    /// Cache key identifying this compiler backend.
    ///
    /// Used to differentiate compiled artifacts when the same device type
    /// can have multiple compiler backends (e.g., clang vs llvm-jit).
    fn cache_key(&self) -> &'static str;
}

/// A renderer that transforms UOp graphs into source code.
///
/// This trait abstracts over different code generation backends:
/// - LLVM IR generator
/// - CUDA C generator
/// - Metal Shading Language generator
/// - WGSL generator
pub trait Renderer: Send + Sync {
    /// Render a UOp graph into source code.
    ///
    /// # Arguments
    ///
    /// * `ast` - The kernel AST (UOp graph rooted at a CALL body such as SINK/PROGRAM)
    /// * `name` - Optional kernel name for debugging (e.g., "r_g16l16R32u4").
    ///   Falls back to "kernel" if None.
    ///
    /// # Returns
    ///
    /// A ProgramSpec containing:
    /// - Generated source code
    /// - Entry point name
    /// - Variable list
    /// - Work sizes (for GPU backends)
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec>;

    /// Get the device spec for this renderer.
    ///
    /// This is used for cache key construction and device selection.
    fn device(&self) -> &DeviceSpec;

    /// The GPU architecture this renderer targets, if any. Arch is a hardware
    /// property of the opened device (not the `DeviceSpec`), surfaced here so
    /// the scheduler can pick the matching optimizer profile (wave size, matrix
    /// cores, …). CPU and backends without an arch distinction return `None`.
    fn gpu_arch(&self) -> Option<svod_dtype::GpuArch> {
        None
    }

    /// Returns decomposition patterns for operations this backend doesn't support.
    ///
    /// This is used by the realization pass to decompose complex operations
    /// into simpler primitives before rendering.
    ///
    /// # Default Implementation
    ///
    /// Returns `None`, meaning no decomposition is needed (backend supports all ops).
    /// Backends that don't support certain operations (e.g., transcendentals)
    /// should override this to return appropriate patterns.
    fn decompositor(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        None
    }
}

/// A factory function that creates executable Programs from a compiled specification.
///
/// This is a function pointer that wraps the backend-specific loader:
/// - LLVM: Extract source from CompiledSpec and JIT compile
/// - CUDA: Extract bytes from CompiledSpec and call cuModuleLoadData + cuModuleGetFunction
/// - Metal: Extract bytes from CompiledSpec and call newLibraryWithData + newFunctionWithName
/// - WebGPU: Extract bytes from CompiledSpec and call createShaderModule
///
/// The CompiledSpec contains either source (for JIT) or bytes (for AOT),
/// allowing each backend to access what it needs.
pub type RuntimeFactory = Arc<dyn Fn(&CompiledSpec) -> Result<Box<dyn Program>> + Send + Sync>;

/// Builds a replayable graph from a captured kernel chain. Returns `Ok(None)`
/// when this backend can't graph the chain (then callers fall back to per-call
/// dispatch). A graphing backend pre-builds its dispatch packets; CPU has no
/// factory.
pub type GraphFactory = Arc<dyn Fn(&[GraphKernel<'_>]) -> Result<Option<Box<dyn Graph>>> + Send + Sync>;

/// A (Renderer, Compiler) pair for a specific backend.
///
/// Devices can have multiple compiler pairs (e.g., different optimization levels).
pub type CompilerPair = (Arc<dyn Renderer>, Arc<dyn Compiler>);

/// A device that owns renderer, compiler, runtime, and allocator.
///
/// A Device is a complete compilation + execution unit for a specific backend.
///
/// # Example
///
/// ```ignore
/// let cpu_device = create_cpu_device()?;
/// let spec = cpu_device.renderer.render(&kernel_ast, Some("E_L3"))?;
/// let compiled = cpu_device.compiler.compile(&spec)?;
/// let program = (cpu_device.runtime)(&compiled)?;
/// unsafe { program.execute(&buffers, &vals, None, None, /*wait=*/ true)?; }
/// ```
pub struct Device {
    /// Device specification
    pub device: DeviceSpec,

    /// Memory allocator for this device
    pub allocator: Arc<dyn Allocator>,

    /// Available (renderer, compiler) pairs for this device
    ///
    /// Most devices have one pair, but some may have multiple
    /// (e.g., different optimization levels or compilation modes).
    pub compilers: Vec<CompilerPair>,

    /// Primary renderer for this device
    ///
    /// This is typically `compilers[0].0`, stored separately for convenience.
    pub renderer: Arc<dyn Renderer>,

    /// Primary compiler for this device
    ///
    /// This is typically `compilers[0].1`, stored separately for convenience.
    pub compiler: Arc<dyn Compiler>,

    /// Runtime factory for creating executable programs
    ///
    /// Takes (entry_point, compiled_bytes) and returns a Program.
    pub runtime: RuntimeFactory,

    /// Optional graph factory for capture/replay. `None` means per-call
    /// dispatch only (CPU); a graphing backend installs one to capture/replay
    /// kernel chains.
    pub graph: Option<GraphFactory>,
}

impl Device {
    /// Create a new device with a single compiler pair.
    ///
    /// This is a convenience constructor for the common case where
    /// a device has only one renderer/compiler combination.
    pub fn new(
        device: DeviceSpec,
        allocator: Arc<dyn Allocator>,
        renderer: Arc<dyn Renderer>,
        compiler: Arc<dyn Compiler>,
        runtime: RuntimeFactory,
    ) -> Self {
        let compilers = vec![(renderer.clone(), compiler.clone())];
        Self { device, allocator, compilers, renderer, compiler, runtime, graph: None }
    }

    /// Install a graph factory (capture/replay). Builder-style for backends
    /// that can pre-build dispatch packets.
    pub fn with_graph(mut self, factory: GraphFactory) -> Self {
        self.graph = Some(factory);
        self
    }

    /// Get the base device key (strips device ID).
    ///
    /// Used for compiled byte cache sharing across device instances.
    /// Examples:
    /// - DeviceSpec::Cpu -> "CPU"
    /// - DeviceSpec::Cuda { device_id: 0 } -> "CUDA"
    /// - DeviceSpec::Cuda { device_id: 1 } -> "CUDA"
    /// - DeviceSpec::Metal { device_id: 0 } -> "Metal"
    ///
    /// This allows compiled CUDA kernels to be reused across CUDA:0 and CUDA:1.
    pub fn base_device_key(&self) -> &'static str {
        self.device.base_type()
    }
}

/// Program specification containing source code and metadata.
///
/// This is returned by Renderer::render() and consumed by Compiler::compile().
/// It bridges the gap between UOp graphs and compiled executables.
///
/// # Buffer metadata
///
/// - `globals`: Buffer indices from PARAM ops
/// - `outs`: Output buffer indices (written by STORE ops)
/// - `ins`: Input buffer indices (read by LOAD ops)
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    /// Kernel name (for debugging/profiling)
    pub name: String,

    /// Generated source code (LLVM IR, CUDA C, Metal, WGSL, etc.)
    pub src: String,

    /// Device specification
    pub device: DeviceSpec,

    /// Original AST (for cache key construction via hash consing)
    pub ast: Arc<UOp>,

    /// Symbolic global work size.
    pub global_size: [Arc<UOp>; 3],

    /// Symbolic local work size. None means direct global-id execution.
    pub local_size: Option<[Arc<UOp>; 3]>,

    /// Variable list (for symbolic shapes/strides)
    pub vars: Vec<Variable>,

    /// Variable names in order for populating vars array at runtime.
    /// Includes runtime variables such as core_id.
    pub var_names: Vec<String>,

    /// Global buffer indices (from PARAM slot values).
    pub globals: Vec<usize>,

    /// Output buffer indices (written by STORE ops).
    pub outs: Vec<usize>,

    /// Input buffer indices (read by LOAD ops, excluding outputs).
    pub ins: Vec<usize>,

    /// Number of buffer arguments (for CIF construction at compile time).
    pub buf_count: usize,
}

#[derive(Debug)]
struct DerivedProgramMetadata {
    vars: Vec<Variable>,
    var_names: Vec<String>,
    globals: Vec<usize>,
    outs: Vec<usize>,
    ins: Vec<usize>,
    global_size: [Arc<UOp>; 3],
    local_size: Option<[Arc<UOp>; 3]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LaunchDimKind {
    Global,
    Local,
    DirectGlobal,
}

impl ProgramSpec {
    /// Create a new program specification.
    pub fn new(name: String, src: String, device: DeviceSpec, ast: Arc<UOp>) -> Self {
        Self {
            name,
            src,
            device,
            ast,
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
            vars: Vec::new(),
            var_names: Vec::new(),
            globals: Vec::new(),
            outs: Vec::new(),
            ins: Vec::new(),
            buf_count: 0,
        }
    }

    /// Add a variable to the program.
    pub fn add_var(&mut self, var: Variable) {
        self.vars.push(var);
    }

    /// Set work sizes for GPU execution.
    pub fn set_work_sizes(&mut self, global: [usize; 3], local: [usize; 3]) {
        self.global_size = concrete_launch_size(global);
        self.local_size = Some(concrete_launch_size(local));
    }

    /// Set symbolic work sizes for replay with runtime variables.
    pub fn set_launch_dims(&mut self, global: [Arc<UOp>; 3], local: Option<[Arc<UOp>; 3]>) {
        self.global_size = global;
        self.local_size = local;
    }

    /// Evaluate symbolic launch dimensions using runtime variable values.
    pub fn launch_dims(&self, var_vals: &HashMap<&str, i64>) -> Result<ConcreteLaunchDims> {
        Self::resolve_launch_dims(&self.global_size, self.local_size.as_ref(), var_vals)
    }

    /// Evaluate launch dimensions stored outside a full ProgramSpec.
    pub fn resolve_launch_dims(
        global_size: &[Arc<UOp>; 3],
        local_size: Option<&[Arc<UOp>; 3]>,
        var_vals: &HashMap<&str, i64>,
    ) -> Result<ConcreteLaunchDims> {
        Ok(ConcreteLaunchDims {
            global_size: eval_launch_size(global_size, var_vals)?,
            local_size: local_size.map(|local| eval_launch_size(local, var_vals)).transpose()?,
        })
    }

    /// Set variable names for populating vars array at runtime.
    pub fn set_var_names(&mut self, var_names: Vec<String>) {
        self.var_names = var_names;
    }

    /// Set buffer metadata (globals, outs, ins).
    pub fn set_buffer_metadata(&mut self, globals: Vec<usize>, outs: Vec<usize>, ins: Vec<usize>) {
        self.globals = globals;
        self.outs = outs;
        self.ins = ins;
    }

    /// Derive and apply metadata from `self.ast`.
    ///
    /// Extracts program metadata from the kernel graph and keeps renderer
    /// wrappers aligned on one metadata path.
    pub fn apply_derived_metadata_from_ast(&mut self) {
        let derived = Self::derive_metadata_from_sink(&self.ast);
        self.globals = derived.globals;
        self.outs = derived.outs;
        self.ins = derived.ins;
        if self.vars.is_empty() {
            self.vars = derived.vars;
        }
        if self.var_names.is_empty() {
            self.var_names = derived.var_names;
        }
        if self.buf_count == 0 {
            self.buf_count = self.globals.len();
        }
        self.global_size = derived.global_size;
        self.local_size = derived.local_size;
    }

    fn special_launch_axis(name: &str) -> Option<(LaunchDimKind, usize)> {
        let kind = match name.chars().next()? {
            'g' => LaunchDimKind::Global,
            'l' => LaunchDimKind::Local,
            'i' => LaunchDimKind::DirectGlobal,
            _ => return None,
        };
        let suffix_start = name.rfind(|ch: char| !ch.is_ascii_digit()).map(|idx| idx + 1).unwrap_or(0);
        if suffix_start == name.len() {
            return None;
        }
        let axis = name[suffix_start..].parse::<usize>().ok()?;
        (axis < 3).then_some((kind, axis))
    }

    fn extract_param_slot_from_index(index: &Arc<UOp>) -> Option<usize> {
        /// Walk an arbitrary UOp expression chasing the underlying `Op::Param` slot.
        /// Handles passthroughs (`Cast`, `Bitcast`), and vector wrappers
        /// (`Vectorize` with all-identical `Param` elements, `Gep` into such a
        /// vector). Devectorize doesn't always eliminate vectorized PARAM
        /// pointers (e.g. scatter stores at `cpu/ops.rs:15-18`), so a kernel's
        /// output `Op::Store` may have an `index` whose `buffer` is a
        /// `Vectorize` of N copies of the same `Op::Param`, so the metadata
        /// derivation looks through these wrappers too.
        fn walk(uop: &Arc<UOp>) -> Option<usize> {
            match uop.op() {
                Op::Param { slot, device: None, .. } => Some(*slot),
                Op::Index { buffer, .. } => walk(buffer),
                Op::Cast { src, .. } | Op::BitCast { src, .. } => walk(src),
                Op::Gep { vector, .. } => walk(vector),
                Op::Vectorize { elements } => {
                    let first = walk(elements.first()?)?;
                    elements.iter().skip(1).all(|e| walk(e) == Some(first)).then_some(first)
                }
                _ => None,
            }
        }
        walk(index)
    }

    fn derive_metadata_from_sink(sink: &Arc<UOp>) -> DerivedProgramMetadata {
        let mut vars = Vec::new();
        let mut globals = Vec::new();
        let mut outs = Vec::new();
        let mut ins = Vec::new();
        let mut global_size = default_launch_size();
        let mut local_size = Some(default_launch_size());

        for node in sink.toposort() {
            match node.op() {
                Op::DefineVar { name, min_val, max_val } => {
                    vars.push(Variable::new(name.clone(), *min_val, *max_val));
                    if name == "core_id" {
                        global_size[0] = UOp::index_const(max_val.saturating_add(1));
                    }
                }
                Op::Param { slot, device: None, .. } => {
                    globals.push(*slot);
                }
                Op::Special { end, name } => {
                    if let Some((kind, axis)) = Self::special_launch_axis(name) {
                        match kind {
                            LaunchDimKind::Global => global_size[axis] = end.clone(),
                            LaunchDimKind::Local => {
                                local_size.get_or_insert_with(default_launch_size)[axis] = end.clone()
                            }
                            LaunchDimKind::DirectGlobal => {
                                global_size[axis] = end.clone();
                                local_size = None;
                            }
                        }
                    }
                }
                Op::Store { index, .. } => {
                    if let Some(slot) = Self::extract_param_slot_from_index(index) {
                        outs.push(slot);
                    }
                }
                Op::Load { index, .. } => {
                    if let Some(slot) = Self::extract_param_slot_from_index(index) {
                        ins.push(slot);
                    }
                }
                _ => {}
            }
        }

        vars.sort_by(|a, b| a.name.cmp(&b.name));
        vars.dedup_by(|a, b| a.name == b.name);
        let var_names = vars.iter().map(|v| v.name.clone()).collect();

        globals.sort_unstable();
        globals.dedup();

        outs.sort_unstable();
        outs.dedup();

        ins.sort_unstable();
        ins.dedup();

        DerivedProgramMetadata { vars, var_names, globals, outs, ins, global_size, local_size }
    }

    /// Build a ProgramSpec from a PROGRAM UOp state.
    ///
    /// Validates PROGRAM stage shape and derives metadata from PROGRAM itself.
    pub fn from_uop(program: &Arc<UOp>) -> Result<Self> {
        let Op::Program { sink, device, linear, source, binary } = program.op() else {
            return WrongStageSnafu { expected: "PROGRAM", got: format!("{:?}", program.op()) }.fail();
        };

        snafu::ensure!(
            matches!(sink.op(), Op::Sink { .. }),
            WrongStageSnafu { expected: "PROGRAM sink stage SINK", got: format!("{:?}", sink.op()) }
        );

        let device_spec = match device.op() {
            Op::Device(spec) => spec.clone(),
            other => return WrongStageSnafu { expected: "PROGRAM device DEVICE", got: format!("{other:?}") }.fail(),
        };

        let linear = linear
            .as_ref()
            .context(WrongStageSnafu { expected: "PROGRAM LINEAR stage", got: "missing".to_string() })?;
        snafu::ensure!(
            matches!(linear.op(), Op::Linear { .. }),
            WrongStageSnafu { expected: "PROGRAM linear stage LINEAR", got: format!("{:?}", linear.op()) }
        );

        let source = source
            .as_ref()
            .context(WrongStageSnafu { expected: "PROGRAM SOURCE stage", got: "missing".to_string() })?;
        let source_code = match source.op() {
            Op::Source { code } => code.clone(),
            other => {
                return WrongStageSnafu { expected: "PROGRAM source stage SOURCE", got: format!("{other:?}") }.fail();
            }
        };

        if let Some(binary) = binary {
            snafu::ensure!(
                matches!(binary.op(), Op::ProgramBinary { .. }),
                WrongStageSnafu { expected: "PROGRAM binary stage ProgramBinary", got: format!("{:?}", binary.op()) }
            );
        }

        let derived = Self::derive_metadata_from_sink(sink);
        let meta = program.metadata::<ProgramSpec>();

        let name = meta.as_ref().map(|m| m.name.clone()).unwrap_or_else(|| "kernel".to_string());

        let mut spec = Self::new(name, source_code, device_spec, sink.clone());
        spec.vars = meta.as_ref().map(|m| m.vars.clone()).filter(|vars| !vars.is_empty()).unwrap_or(derived.vars);
        spec.var_names =
            meta.as_ref().map(|m| m.var_names.clone()).filter(|names| !names.is_empty()).unwrap_or(derived.var_names);
        spec.globals =
            meta.as_ref().map(|m| m.globals.clone()).filter(|globals| !globals.is_empty()).unwrap_or(derived.globals);
        spec.outs = meta.as_ref().map(|m| m.outs.clone()).filter(|outs| !outs.is_empty()).unwrap_or(derived.outs);
        spec.ins = meta.as_ref().map(|m| m.ins.clone()).filter(|ins| !ins.is_empty()).unwrap_or(derived.ins);
        spec.buf_count = meta.as_ref().map(|m| m.buf_count).filter(|count| *count > 0).unwrap_or(spec.globals.len());
        // Launch dims: always derive from the SPECIAL UOps in the SINK,
        // ignoring any upstream `meta` value (iterate SPECIAL and set
        // `special_size[name_suffix] = end`). Keeping a meta override
        // here causes a dispatch-vs-IR mismatch: the meta-supplied
        // `global_size` is in kernel-name positional order ([g_x_size,
        // g_y_size]) but the SPECIAL UOps (post-gpudims `reverse=true`) may
        // assign axes to gidx in the opposite order, so the GPU dispatch
        // packet ends up with grid_size_x grid_size_y swapped relative to
        // what the LLVM IR's `workgroup.id.x` / `workgroup.id.y` use —
        // manifests as a 21× OOB on `r_g1375g64...` (`Phase 10/11`).
        spec.global_size = derived.global_size;
        spec.local_size = derived.local_size;

        Ok(spec)
    }
}

/// A variable in the kernel (for symbolic shapes/strides).
///
/// Variables represent symbolic values that are bound at kernel execution time.
/// Examples:
/// - Shape dimensions that vary per input
/// - Stride values computed from shapes
/// - Loop bounds determined by input sizes
#[derive(Debug, Clone)]
pub struct Variable {
    /// Variable name (must be unique within the kernel)
    pub name: String,

    /// Minimum value (for range validation)
    pub min: i64,

    /// Maximum value (for range validation)
    pub max: i64,
}

impl Variable {
    /// Create a new variable.
    pub fn new(name: String, min: i64, max: i64) -> Self {
        Self { name, min, max }
    }
}
