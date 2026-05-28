//! Pre-compiled execution plan for kernel execution.
//!
//! `ExecutionPlan` separates one-time preparation (kernel compilation, buffer
//! allocation) from fast repeated execution.
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────┐
//! │              PREPARATION (one-time)                      │
//! │  Schedule → instantiate → compile_kernels → build()     │
//! │                       ↓                                  │
//! │                ExecutionPlan                             │
//! └─────────────────────────────────────────────────────────┘
//!                         ↓
//! ┌─────────────────────────────────────────────────────────┐
//! │              EXECUTION (fast path)                       │
//! │  dependency-ordered PreparedOp execution                 │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! let plan = tensor.prepare()?;
//! plan.execute()?;
//! let output = plan.output_buffer();
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use smallvec::SmallVec;
use svod_device::device::ProgramSpec;
use svod_device::{Buffer, BufferId};
use svod_dtype::DeviceSpec;
use svod_ir::{CustomFunctionKind, Op, UOp};

use crate::error::Result;
use crate::kernel_cache::CachedKernel;
use crate::profiler::KernelProfile;

type RuntimeLaunchSizes = (Option<[usize; 3]>, Option<[usize; 3]>);

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// Variable values are stored as positional `vals: Vec<i64>` rather than a named
/// HashMap, matching Tinygrad's `vals: tuple[int, ...]` parameter style.
#[derive(Clone)]
pub struct PreparedKernel {
    /// Unique identifier (from original AST).
    pub id: u64,

    pub ast: Arc<UOp>,

    /// Compiled kernel program (Arc-shared from cache).
    pub kernel: Arc<CachedKernel>,

    /// Device this kernel executes on.
    pub device: DeviceSpec,

    /// Indices into `ExecutionPlan::buffers` for this kernel's buffers.
    /// Ordered as expected by the kernel (matches codegen buffer order).
    pub buffer_indices: Vec<usize>,

    /// Indices of output buffers within `buffer_indices`.
    pub output_indices: Vec<usize>,

    /// Variable values in positional order (matches `var_names` in CachedKernel).
    pub vals: Vec<i64>,

    /// Fixed variable bindings captured at prepare time.
    ///
    /// These mirror Tinygrad's `fixedvars` semantics: values fixed by scheduling
    /// (for example from bound ranges) are not overridden by `execute_with_vars`.
    pub fixedvars: HashMap<String, i64>,

    /// Kernel IDs that must complete before this one (dependencies).
    pub dependencies: Vec<u64>,

    /// Pre-computed raw buffer addresses for low-allocation execution.
    /// Computed once during prepare(), stable for the lifetime of ExecutionPlan.
    /// SAFETY: Pointers are valid as long as ExecutionPlan owns the buffers.
    pub buffer_ptrs: Vec<usize>,

    /// Pre-computed buffer IDs for dependency tracking.
    pub buffer_ids: Vec<BufferId>,

    /// Cached `(name, min_val, max_val)` triples for every `DefineVar` reachable
    /// from `ast`. Populated at construction so `validate_runtime_var_bounds`
    /// doesn't re-toposort on every execute call.
    pub runtime_vars: Vec<RuntimeVar>,
}

/// Bound description for one `DefineVar` consumed by a kernel.
#[derive(Clone, Debug)]
pub struct RuntimeVar {
    pub name: String,
    pub min_val: i64,
    pub max_val: i64,
}

/// Walk `root` and collect bounds for every reachable `DefineVar`.
pub fn collect_runtime_vars(root: &Arc<UOp>) -> Vec<RuntimeVar> {
    let mut vars = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for node in root.toposort() {
        if let Op::DefineVar { name, min_val, max_val } = node.op()
            && seen.insert(name.clone())
        {
            vars.push(RuntimeVar { name: name.clone(), min_val: *min_val, max_val: *max_val });
        }
    }
    vars
}

/// Prepared buffer-to-buffer copy operation.
#[derive(Clone, Debug)]
pub struct PreparedCopy {
    /// Unique operation identifier.
    pub id: u64,

    /// Buffer indices in ExecutionPlan order: [dst, src].
    pub buffer_indices: Vec<usize>,

    /// Operation IDs that must complete before this copy.
    pub dependencies: Vec<u64>,
}

/// Prepared zero-copy buffer view operation.
#[derive(Clone, Debug)]
pub struct PreparedBufferView {
    /// Unique operation identifier.
    pub id: u64,

    /// Output and base buffer indices in ExecutionPlan order.
    /// `buffer_indices[0]` is output view, `buffer_indices[1]` is base source.
    pub buffer_indices: Vec<usize>,

    /// Expected byte offset into base for the view.
    pub byte_offset: usize,

    /// Expected byte size of the view.
    pub byte_size: usize,

    /// Operation IDs that must complete before this view is consumed.
    pub dependencies: Vec<u64>,
}

/// Prepared custom runtime function operation.
#[derive(Clone, Debug)]
pub struct PreparedCustomFunction {
    /// Unique operation identifier.
    pub id: u64,

    /// Explicit custom function kind (for example: `EncDec`).
    pub kind: CustomFunctionKind,

    /// Runtime descriptor attributes encoded by the IR body.
    pub attrs: SmallVec<[Arc<UOp>; 4]>,

    /// Buffer indices in ExecutionPlan order.
    pub buffer_indices: Vec<usize>,

    /// Bound variable values for this operation.
    pub fixedvars: HashMap<String, i64>,

    /// Operation IDs that must complete before this custom function runs.
    pub dependencies: Vec<u64>,

    /// Cached `(name, min_val, max_val)` triples for every `DefineVar`
    /// reachable from `attrs`. Populated at construction so
    /// `validate_runtime_var_bounds` doesn't re-toposort on every execute call.
    pub runtime_vars: Vec<RuntimeVar>,
}

/// Prepared execution item.
#[derive(Clone, Debug)]
pub enum PreparedOp {
    /// Compiled kernel/program operation.
    CompiledProgram(PreparedKernel),

    /// Direct buffer copy operation.
    BufferCopy(PreparedCopy),

    /// Zero-copy view aliasing operation.
    BufferView(PreparedBufferView),

    /// Runtime custom function operation.
    CustomFunction(PreparedCustomFunction),
}

fn op_identity(op: &PreparedOp) -> (u64, Vec<u64>) {
    match op {
        PreparedOp::CompiledProgram(kernel) => (kernel.id, kernel.dependencies.clone()),
        PreparedOp::BufferCopy(copy) => (copy.id, copy.dependencies.clone()),
        PreparedOp::BufferView(view) => (view.id, view.dependencies.clone()),
        PreparedOp::CustomFunction(custom) => (custom.id, custom.dependencies.clone()),
    }
}

fn validate_var_bound(name: &str, value: i64, min_val: i64, max_val: i64) -> Result<()> {
    if value < min_val || value > max_val {
        return Err(crate::error::Error::Execution {
            reason: format!("variable {name}={value} is outside bounds [{min_val}, {max_val}]"),
        });
    }
    Ok(())
}

/// Extract `(node_ids, callable_deps)` from prepared ops for the shared
/// topological-leveling routines in [`crate::leveling`].
fn op_graph_inputs(ops: &[PreparedOp]) -> (Vec<u64>, Vec<Vec<u64>>) {
    ops.iter().map(op_identity).unzip()
}

#[cfg(test)]
fn compute_mixed_op_order(ops: &[PreparedOp]) -> Result<Vec<usize>> {
    compute_mixed_op_order_with_instance_dependencies(ops, &[])
}

fn compute_mixed_op_order_with_instance_dependencies(
    ops: &[PreparedOp],
    instance_deps_per_op: &[Vec<usize>],
) -> Result<Vec<usize>> {
    let (node_ids, callable_deps) = op_graph_inputs(ops);
    let index_deps = (!instance_deps_per_op.is_empty()).then_some(instance_deps_per_op);
    crate::leveling::compute_topological_order(&node_ids, &callable_deps, index_deps)
}

#[cfg(test)]
fn compute_execution_levels(ops: &[PreparedOp]) -> Result<Vec<Vec<usize>>> {
    compute_execution_levels_with_instance_dependencies(ops, &[])
}

fn compute_execution_levels_with_instance_dependencies(
    ops: &[PreparedOp],
    instance_deps_per_op: &[Vec<usize>],
) -> Result<Vec<Vec<usize>>> {
    let (node_ids, callable_deps) = op_graph_inputs(ops);
    let index_deps = (!instance_deps_per_op.is_empty()).then_some(instance_deps_per_op);
    crate::leveling::compute_topological_levels(&node_ids, &callable_deps, index_deps)
}

/// Pre-compiled execution plan for a computation graph.
///
/// Created once via `prepare()`, then executed multiple times.
/// The plan owns all its buffers and compiled kernels.
pub struct ExecutionPlan {
    /// Prepared operations in schedule order.
    ops: Vec<PreparedOp>,

    /// Concrete op-index dependencies parallel to `ops`.
    op_instance_dependencies: Vec<Vec<usize>>,

    /// Precomputed dependency-safe operation order.
    op_order: Vec<usize>,

    /// Topological levels of dependency-independent operations. Preserved as
    /// the execution-iteration order (each level flushed before the next) for
    /// consistency with pre-Step-6 plan semantics — some downstream kernel
    /// algorithms (e.g. iterative QR) are sensitive to within-level
    /// scheduling order vs. a single flat topological linearization.
    op_levels: Vec<Vec<usize>>,

    /// ALL buffers owned by this plan (inputs, intermediates, outputs).
    buffers: Vec<Buffer>,

    /// Mapping: AST id → buffer index (for kernel buffer binding).
    ast_to_buffer: HashMap<u64, usize>,

    /// Indices of output buffers in `buffers` (matches SINK source order).
    output_buffer_indices: Vec<usize>,

    /// Primary device for this plan.
    device: DeviceSpec,

    /// Last dynamic variable bindings supplied through `execute_with_vars`.
    runtime_var_vals: HashMap<String, i64>,

    /// Additional UOp IDs registered as aliases that need cleanup.
    alias_ids: Vec<u64>,

    /// Captured replayable graph, built lazily on first `execute()`. `Some(None)`
    /// means the chain isn't graphable (mixed ops / non-graph device) → per-call
    /// dispatch. Replaces N packet-builds + N doorbells with one submit; see
    /// `svod_device::Graph` (AMD indirect buffer).
    graph: std::sync::OnceLock<Option<Box<dyn svod_device::Graph>>>,

    /// Per-plan AMD connector — owns this plan's scratch, timeline, dispatch
    /// lock. Lazy-init on the first AMD kernel dispatch from the program's
    /// `Arc<AmdDeviceCore>` (cheap: the core is process-cached via
    /// `DEVICE_CACHE` and we Arc::clone it). Decouples this plan's dispatch
    /// state from other plans' on the same physical AMD:N — Step 4 of the
    /// connector refactor (`snug-honking-robin`).
    ///
    /// Non-AMD plans never touch this field; CPU programs continue to use
    /// `Program::execute(...)`.
    #[cfg(target_os = "linux")]
    amd_connector: std::sync::OnceLock<std::sync::Arc<svod_device::amd::AmdConnector>>,
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
    fn kernel_launch_sizes(kernel: &PreparedKernel) -> Result<RuntimeLaunchSizes> {
        let mut vars: HashMap<&str, i64> =
            HashMap::with_capacity(kernel.kernel.var_names.len() + kernel.fixedvars.len());
        for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
            let value = kernel.vals.get(idx).copied().ok_or_else(|| crate::error::Error::Execution {
                reason: format!(
                    "Kernel {} has {} var names but only {} values",
                    kernel.id,
                    kernel.kernel.var_names.len(),
                    kernel.vals.len()
                ),
            })?;
            vars.insert(name.as_str(), value);
        }
        for (name, value) in &kernel.fixedvars {
            vars.insert(name.as_str(), *value);
        }

        let dims =
            ProgramSpec::resolve_launch_dims(&kernel.kernel.global_size, kernel.kernel.local_size.as_ref(), &vars)
                .map_err(|e| crate::error::Error::Execution {
                    reason: format!("Kernel {} launch dimensions failed: {e}", kernel.id),
                })?;
        Ok((Some(dims.global_size), dims.local_size))
    }

    /// Lazily capture all kernels into a backend replay graph. Only AMD
    /// installs a graph factory; everything else (and any non-graphable chain)
    /// returns `None` → per-call dispatch. Gated to chains that are *all*
    /// compiled kernels with no runtime vars: copies/views/custom or dynamic
    /// launch dims keep the host in the loop and aren't graphed.
    fn graph(&self) -> &Option<Box<dyn svod_device::Graph>> {
        self.graph.get_or_init(|| self.build_graph().unwrap_or(None))
    }

    fn build_graph(&self) -> Result<Option<Box<dyn svod_device::Graph>>> {
        // Opt-in until replay is validated against the live AMD timeline/kernarg
        // lifetime (NotPresent faults on freed intermediate VAs). Per-call is the
        // safe default; SVOD_JIT_GRAPH=1 enables capture for benchmarking.
        if std::env::var_os("SVOD_JIT_GRAPH").is_none() {
            return Ok(None);
        }
        let all_static_kernels =
            self.ops.iter().all(|op| matches!(op, PreparedOp::CompiledProgram(k) if k.runtime_vars.is_empty()));
        if !all_static_kernels || self.ops.is_empty() {
            return Ok(None);
        }
        let dev = crate::device_registry::DEVICE_FACTORIES
            .device(&self.device, svod_device::registry::registry())
            .map_err(|e| crate::error::Error::Execution { reason: format!("device lookup: {e}") })?;
        let Some(factory) = dev.graph.clone() else { return Ok(None) };
        let mut kernels = Vec::with_capacity(self.op_order.len());
        for &idx in &self.op_order {
            let PreparedOp::CompiledProgram(k) = &self.ops[idx] else { return Ok(None) };
            let (global_size, local_size) = Self::kernel_launch_sizes(k)?;
            kernels.push(svod_device::GraphKernel {
                program: k.kernel.program.as_ref(),
                buffers: k.buffer_ptrs.iter().map(|&p| p as *mut u8).collect(),
                vals: k.vals.clone(),
                global_size,
                local_size,
            });
        }
        factory(&kernels).map_err(|e| crate::error::Error::Execution { reason: format!("graph capture: {e}") })
    }

    /// Lazy-init the plan's own AMD connector and return it. Cheap to call
    /// repeatedly — the connector lives for the plan's lifetime. Built from
    /// the program's `Arc<AmdDeviceCore>` (process-cached via `DEVICE_CACHE`)
    /// and seeded with a timeline signal acquired from the program's signal
    /// pool. Step 4 of the connector refactor — gives this plan a private
    /// scratch + timeline so cross-plan dispatches no longer contend on the
    /// shared device default connector.
    #[cfg(target_os = "linux")]
    fn amd_connector_for(
        &self,
        prog: &svod_device::amd::AmdProgram,
    ) -> Result<std::sync::Arc<svod_device::amd::AmdConnector>> {
        if let Some(c) = self.amd_connector.get() {
            return Ok(std::sync::Arc::clone(c));
        }
        // Build outside `OnceLock::get_or_init` so we can propagate errors
        // (KFD queue creation, signal-pool acquire). One-shot init race: if
        // two threads see empty, both build a connector; only one wins set().
        // The loser is dropped; its `AmdConnector::Drop` synchronises and
        // returns the timeline signal to the pool.
        // Recover the AMD device_id from the plan's DeviceSpec. The program
        // was loaded against the same device_id, so the allocator created
        // here shares its underlying `Arc<AmdDeviceCore>` via DEVICE_CACHE.
        let device_id = match &self.device {
            DeviceSpec::Amd { device_id } => *device_id,
            _ => {
                return Err(crate::error::Error::Execution {
                    reason: format!("amd_connector_for called on non-AMD plan (device={:?})", self.device),
                });
            }
        };
        let alloc = svod_device::amd::AmdAllocator::new(device_id)
            .map_err(|e| crate::error::Error::Execution { reason: format!("plan allocator: {e}") })?;
        let new_conn =
            svod_device::amd::AmdConnector::new_with_resources(std::sync::Arc::clone(prog.device().core()), &alloc)
                .map_err(|e| crate::error::Error::Execution {
                    reason: format!("AmdConnector::new_with_resources: {e}"),
                })?;
        let _ = self.amd_connector.set(new_conn);
        Ok(std::sync::Arc::clone(self.amd_connector.get().expect("connector set above")))
    }

    #[inline]
    fn execute_kernel(&self, kernel: &PreparedKernel) -> Result<()> {
        let buffer_ptrs: SmallVec<[*mut u8; 8]> = kernel.buffer_ptrs.iter().map(|&ptr| ptr as *mut u8).collect();
        let (global_size, local_size) = Self::kernel_launch_sizes(kernel)?;
        // Fast path for AMD: downcast and dispatch via `execute_on` with the
        // plan's own connector. Step 4 of the connector refactor — keeps each
        // plan's scratch/timeline/dispatch-lock state isolated.
        #[cfg(target_os = "linux")]
        if let Some(amd) = kernel.kernel.program.as_any().downcast_ref::<svod_device::amd::AmdProgram>() {
            let conn = self.amd_connector_for(amd)?;
            // Grow this connector's scratch to fit the program. Mirrors
            // tinygrad's `_ensure_has_local_memory` at program load
            // (`ops_amd.py:589-590`) but applied per-connector.
            conn.ensure_has_local_memory(amd.private_segment_size())
                .map_err(|e| crate::error::Error::Execution { reason: format!("scratch grow: {e}") })?;
            return unsafe {
                amd.execute_on(&conn, &buffer_ptrs, &kernel.vals, global_size, local_size, /*wait=*/ false).map_err(
                    |e| crate::error::Error::Execution { reason: format!("Kernel {} failed: {}", kernel.id, e) },
                )
            };
        }
        unsafe {
            kernel
                .kernel
                .program
                // wait=false: async submit. GPU ordering is enforced by the
                // device timeline; host reads (copyout / as_*) synchronize.
                .execute(&buffer_ptrs, &kernel.vals, global_size, local_size, /*wait=*/ false)
                .map_err(|e| crate::error::Error::Execution { reason: format!("Kernel {} failed: {}", kernel.id, e) })
        }
    }

    fn validate_runtime_var_bounds(&self, var_vals: &[(&str, i64)]) -> Result<()> {
        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for op in &self.ops {
            match op {
                PreparedOp::CompiledProgram(kernel) => {
                    for var in &kernel.runtime_vars {
                        if kernel.fixedvars.contains_key(&var.name) || var.name == "core_id" {
                            continue;
                        }
                        if let Some(&value) = vals_map.get(var.name.as_str()) {
                            validate_var_bound(&var.name, value, var.min_val, var.max_val)?;
                        }
                    }
                }
                PreparedOp::CustomFunction(custom) => {
                    for var in &custom.runtime_vars {
                        if custom.fixedvars.contains_key(&var.name) || var.name == "core_id" {
                            continue;
                        }
                        if let Some(&value) = vals_map.get(var.name.as_str()) {
                            validate_var_bound(&var.name, value, var.min_val, var.max_val)?;
                        }
                    }
                }
                PreparedOp::BufferCopy(_) | PreparedOp::BufferView(_) => {}
            }
        }
        Ok(())
    }

    fn update_runtime_var_vals(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        self.validate_runtime_var_bounds(var_vals)?;

        let vals_map: HashMap<&str, i64> = var_vals.iter().copied().collect();
        for &(name, value) in var_vals {
            if name == "core_id" {
                continue;
            }
            self.runtime_var_vals.insert(name.to_string(), value);
        }
        for op in &mut self.ops {
            if let PreparedOp::CompiledProgram(kernel) = op {
                for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
                    if kernel.fixedvars.contains_key(name) || name == "core_id" {
                        continue;
                    }
                    if let Some(&v) = vals_map.get(name.as_str()) {
                        let Some(slot) = kernel.vals.get_mut(idx) else {
                            return Err(crate::error::Error::Execution {
                                reason: format!(
                                    "Kernel {} has {} var names but only {} values",
                                    kernel.id,
                                    kernel.kernel.var_names.len(),
                                    kernel.vals.len()
                                ),
                            });
                        };
                        *slot = v;
                    }
                }
            }
        }
        Ok(())
    }

    #[inline]
    fn execute_copy(&self, copy: &PreparedCopy) -> Result<()> {
        if copy.buffer_indices.len() < 2 {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "Copy op {} requires at least two buffer indices (dst, src), got {}",
                    copy.id,
                    copy.buffer_indices.len()
                ),
            });
        }
        let dst_idx = copy.buffer_indices[0];
        let src_idx = copy.buffer_indices[1];

        if dst_idx >= self.buffers.len() || src_idx >= self.buffers.len() {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "Copy op {} buffer index out of range: dst={}, src={}, total_buffers={}",
                    copy.id,
                    dst_idx,
                    src_idx,
                    self.buffers.len()
                ),
            });
        }

        let mut dst = self.buffers[dst_idx].clone();
        let src = &self.buffers[src_idx];
        dst.copy_from(src)
            .map_err(|e| crate::error::Error::Execution { reason: format!("Copy op {} failed: {}", copy.id, e) })
    }

    #[inline]
    fn execute_buffer_view(&self, view: &PreparedBufferView) -> Result<()> {
        if view.buffer_indices.len() < 2 {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "BufferView op {} requires at least two buffer indices (out, base), got {}",
                    view.id,
                    view.buffer_indices.len()
                ),
            });
        }
        let out_idx = view.buffer_indices[0];
        let base_idx = view.buffer_indices[1];

        if out_idx >= self.buffers.len() || base_idx >= self.buffers.len() {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "BufferView op {} buffer index out of range: out={}, base={}, total_buffers={}",
                    view.id,
                    out_idx,
                    base_idx,
                    self.buffers.len()
                ),
            });
        }

        let out = &self.buffers[out_idx];
        let base = &self.buffers[base_idx];
        let expected_offset = base.offset() + view.byte_offset;

        if out.storage_id() != base.storage_id() || out.offset() != expected_offset || out.size() != view.byte_size {
            return Err(crate::error::Error::Execution {
                reason: format!(
                    "BufferView op {} mismatch: out(storage={:?},off={},size={}) base(storage={:?},off={}) expected(off={},size={})",
                    view.id,
                    out.storage_id(),
                    out.offset(),
                    out.size(),
                    base.storage_id(),
                    base.offset(),
                    expected_offset,
                    view.byte_size,
                ),
            });
        }
        Ok(())
    }

    #[inline]
    fn execute_custom_function(&self, custom: &PreparedCustomFunction) -> Result<()> {
        let mut buffers = Vec::with_capacity(custom.buffer_indices.len());
        for &idx in &custom.buffer_indices {
            let Some(buffer) = self.buffers.get(idx) else {
                return Err(crate::error::Error::Execution {
                    reason: format!(
                        "Custom function op {} ({:?}) buffer index out of range: idx={}, total_buffers={}",
                        custom.id,
                        custom.kind,
                        idx,
                        self.buffers.len()
                    ),
                });
            };
            buffers.push(buffer.clone());
        }

        let mut vars = self.runtime_var_vals.clone();
        vars.extend(custom.fixedvars.iter().map(|(k, v)| (k.clone(), *v)));

        crate::custom_function::run_custom_function(&custom.kind, &custom.attrs, &mut buffers, &vars).map_err(|e| {
            // Pass typed `Unsupported` errors through unchanged so callers can match on `kind`.
            // Other errors are wrapped with op context for debugging.
            match e {
                crate::error::Error::Unsupported { .. } => e,
                other => crate::error::Error::Execution {
                    reason: format!("Custom function op {} ({:?}) failed: {other}", custom.id, custom.kind),
                },
            }
        })
    }

    #[inline]
    fn execute_op(&self, op: &PreparedOp) -> Result<()> {
        match op {
            PreparedOp::CompiledProgram(kernel) => self.execute_kernel(kernel),
            PreparedOp::BufferCopy(copy) => self.execute_copy(copy),
            PreparedOp::BufferView(view) => self.execute_buffer_view(view),
            PreparedOp::CustomFunction(custom) => self.execute_custom_function(custom),
        }
    }

    /// Get the first (or only) output buffer after execution.
    ///
    /// Returns `None` for plans with no output buffers (for example, plans
    /// constructed before `set_output_buffer*` is called).
    pub fn output_buffer(&self) -> Option<&Buffer> {
        self.output_buffer_indices.first().and_then(|&i| self.buffers.get(i))
    }

    /// Get output buffer by position (matches SINK source order for batch).
    ///
    /// Returns `None` if `position` is out of range.
    pub fn output_buffer_at(&self, position: usize) -> Option<&Buffer> {
        self.output_buffer_indices.get(position).and_then(|&i| self.buffers.get(i))
    }

    /// Get all output buffers.
    pub fn output_buffers(&self) -> Vec<&Buffer> {
        self.output_buffer_indices.iter().map(|&i| &self.buffers[i]).collect()
    }

    /// Number of outputs in this plan.
    pub fn num_outputs(&self) -> usize {
        self.output_buffer_indices.len()
    }

    /// Get a buffer by AST id (for reading intermediate results).
    pub fn buffer(&self, ast_id: u64) -> Option<&Buffer> {
        self.ast_to_buffer.get(&ast_id).map(|&idx| &self.buffers[idx])
    }

    /// Get a mutable buffer by AST id (for `copyin()` on input buffers).
    pub fn buffer_mut_by_id(&mut self, ast_id: u64) -> Option<&mut Buffer> {
        self.ast_to_buffer.get(&ast_id).copied().map(|idx| &mut self.buffers[idx])
    }

    /// Get the primary device for this plan.
    pub fn device(&self) -> &DeviceSpec {
        &self.device
    }

    /// Get all buffers owned by this plan.
    pub fn buffers(&self) -> &[Buffer] {
        &self.buffers
    }

    /// Get mutable access to all buffers owned by this plan.
    pub fn buffers_mut(&mut self) -> &mut [Buffer] {
        &mut self.buffers
    }

    /// Get a mutable buffer by its index in the buffers array.
    pub fn buffer_at_mut(&mut self, index: usize) -> Option<&mut Buffer> {
        self.buffers.get_mut(index)
    }

    /// Get all prepared kernels.
    pub fn prepared_kernels(&self) -> Vec<&PreparedKernel> {
        self.ops
            .iter()
            .filter_map(|op| match op {
                PreparedOp::CompiledProgram(kernel) => Some(kernel),
                _ => None,
            })
            .collect()
    }

    /// Get all prepared operations in schedule order.
    pub fn prepared_ops(&self) -> &[PreparedOp] {
        &self.ops
    }

    /// Iterate over compiled kernels (for inspecting generated source code).
    pub fn kernels(&self) -> impl Iterator<Item = &CachedKernel> {
        self.ops.iter().filter_map(|op| match op {
            PreparedOp::CompiledProgram(kernel) => Some(kernel.kernel.as_ref()),
            _ => None,
        })
    }

    /// Execute the plan.
    ///
    /// Walks `op_levels` level-by-level and runs each op in the level in order.
    /// **Step 6 of the connector refactor** (`snug-honking-robin`) deleted the
    /// previous rayon-driven intra-level parallelism: on AMD the ring is
    /// fundamentally serial (per-connector `Mutex<QueueInner>`); on CPU the
    /// previous overlap was already cancelled by the kernel-thread guard.
    /// Per-plan ownership (Step 4) means multi-plan concurrency comes from
    /// distinct plans running on distinct connectors, which is what BEAM
    /// search relies on — not intra-plan rayon. We keep the level iteration
    /// order (rather than a flat `op_order` topological linearization)
    /// because iterative kernels (QR, etc.) are sensitive to within-level
    /// scheduling order.
    pub fn execute(&self) -> Result<()> {
        // Fast path: one captured indirect-buffer submit instead of per-kernel
        // packet build + doorbell. Built once, then every call just replays.
        if let Some(graph) = self.graph().as_deref() {
            return graph
                .replay(&[])
                .map_err(|e| crate::error::Error::Execution { reason: format!("graph replay failed: {e}") });
        }
        for level in &self.op_levels {
            for &idx in level {
                self.execute_op(&self.ops[idx])?;
            }
        }
        Ok(())
    }

    /// Execute the plan with per-kernel timing.
    ///
    /// Returns a [`KernelProfile`] for each kernel in execution order.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let plan = tensor.prepare()?;
    /// let profiles = plan.execute_profiled()?;
    ///
    /// // Sort by time descending
    /// let mut sorted = profiles;
    /// sorted.sort_by(|a, b| b.elapsed.cmp(&a.elapsed));
    /// for p in &sorted[..10.min(sorted.len())] {
    ///     println!("{:>8.3}ms  {}", p.elapsed.as_secs_f64() * 1000.0, p.kernel.entry_point);
    /// }
    /// ```
    pub fn execute_profiled(&self) -> Result<Vec<KernelProfile>> {
        let mut profiles = Vec::with_capacity(self.op_order.len());
        for level in &self.op_levels {
            for &idx in level {
                match &self.ops[idx] {
                    PreparedOp::CompiledProgram(kernel) => {
                        let start = Instant::now();
                        self.execute_kernel(kernel)?;
                        profiles.push(KernelProfile {
                            kernel: Arc::clone(&kernel.kernel),
                            device: kernel.device.clone(),
                            num_buffers: kernel.buffer_ptrs.len(),
                            elapsed: start.elapsed(),
                        });
                    }
                    PreparedOp::BufferCopy(copy) => self.execute_copy(copy)?,
                    PreparedOp::BufferView(view) => self.execute_buffer_view(view)?,
                    PreparedOp::CustomFunction(custom) => self.execute_custom_function(custom)?,
                }
            }
        }
        Ok(profiles)
    }

    /// Re-execute the plan with different variable bindings.
    ///
    /// The kernel code is NOT recompiled; only the `vals` passed to each kernel
    /// are updated. Buffers must be allocated to max variable values (which is
    /// the default when using `Variable::bind()`).
    ///
    /// # Safety contract
    ///
    /// Variable values **must** fall within `[min_val, max_val]` bounds defined
    /// at `Variable::new()` time. Exceeding `max_val` causes out-of-bounds buffer
    /// access (buffers are allocated to `max_val`). Use `Variable::bind()` to
    /// validate bounds before calling this method.
    ///
    /// Variables not present in `var_vals` keep their existing values from
    /// `prepare()` (or the previous `execute_with_vars` call). Internal
    /// variables like `core_id` are left untouched.
    pub fn execute_with_vars(&mut self, var_vals: &[(&str, i64)]) -> Result<()> {
        self.update_runtime_var_vals(var_vals)?;
        self.execute()
    }

    /// Re-execute the plan with different variable bindings and per-kernel timing.
    ///
    /// Updates kernel `vals` the same way as [`Self::execute_with_vars`] and then
    /// executes via [`Self::execute_profiled`].
    pub fn execute_with_vars_profiled(&mut self, var_vals: &[(&str, i64)]) -> Result<Vec<KernelProfile>> {
        self.update_runtime_var_vals(var_vals)?;
        self.execute_profiled()
    }

    /// Get the first output buffer index.
    pub fn output_buffer_idx(&self) -> usize {
        self.output_buffer_indices[0]
    }

    /// Get the AST ID to buffer index mapping.
    pub fn ast_to_buffer_map(&self) -> &HashMap<u64, usize> {
        &self.ast_to_buffer
    }

    /// Release intermediate buffers from the global buffer registry.
    ///
    /// Call this after you're done executing the plan to free intermediate
    /// buffers from the global registry. The output buffer is preserved.
    pub fn release_intermediate_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, true);
    }

    /// Release ALL buffers from the global registry, including the output.
    pub fn release_all_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, false);
    }

    fn release_buffers_impl<F>(&self, remove_fn: F, skip_output: bool)
    where
        F: Fn(u64),
    {
        let output_buf_ids: std::collections::HashSet<u64> = if skip_output {
            self.output_buffer_indices.iter().filter_map(|&idx| self.buffers.get(idx).map(|b| b.id().0)).collect()
        } else {
            std::collections::HashSet::new()
        };

        for (&ast_id, &buf_idx) in &self.ast_to_buffer {
            if skip_output && output_buf_ids.contains(&self.buffers[buf_idx].id().0) {
                continue;
            }
            remove_fn(ast_id);
        }

        for &alias_id in &self.alias_ids {
            remove_fn(alias_id);
        }
    }
}

impl std::fmt::Debug for ExecutionPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let kernel_count = self.ops.iter().filter(|op| matches!(op, PreparedOp::CompiledProgram(_))).count();
        f.debug_struct("ExecutionPlan")
            .field("ops", &self.ops.len())
            .field("op_instance_dependencies", &self.op_instance_dependencies.len())
            .field("op_order", &self.op_order.len())
            .field("kernels", &kernel_count)
            .field("buffers", &self.buffers.len())
            .field("device", &self.device)
            .finish()
    }
}

impl std::fmt::Debug for PreparedKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreparedKernel")
            .field("id", &self.id)
            .field("device", &self.device)
            .field("buffer_indices", &self.buffer_indices)
            .field("output_indices", &self.output_indices)
            .field("vals", &self.vals)
            .field("fixedvars", &self.fixedvars)
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

// ============================================================================
// Builder for ExecutionPlan
// ============================================================================

/// Builder for creating ExecutionPlan from schedule data.
pub struct ExecutionPlanBuilder {
    ops: Vec<PreparedOp>,
    op_instance_dependencies: Vec<Vec<usize>>,
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>,
    output_buffer_indices: Vec<usize>,
    device: DeviceSpec,
    alias_ids: Vec<u64>,
}

impl ExecutionPlanBuilder {
    /// Create a new builder.
    pub fn new(device: DeviceSpec) -> Self {
        Self {
            ops: Vec::new(),
            op_instance_dependencies: Vec::new(),
            buffers: Vec::new(),
            ast_to_buffer: HashMap::new(),
            output_buffer_indices: Vec::new(),
            device,
            alias_ids: Vec::new(),
        }
    }

    /// Add alias IDs that need cleanup.
    pub fn add_alias_ids(&mut self, ids: impl IntoIterator<Item = u64>) {
        self.alias_ids.extend(ids);
    }

    /// Add a buffer to the plan. Returns the buffer index.
    pub fn add_buffer(&mut self, ast_id: u64, buffer: Buffer) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buffer);
        self.ast_to_buffer.insert(ast_id, idx);
        idx
    }

    /// Map an additional AST/buffer UOp ID to an existing buffer index.
    pub fn map_buffer(&mut self, ast_id: u64, idx: usize) {
        self.ast_to_buffer.insert(ast_id, idx);
    }

    /// Replace a buffer at the given index (for BUFFER_VIEW sub-buffer views).
    pub fn replace_buffer(&mut self, idx: usize, buffer: Buffer) {
        self.buffers[idx] = buffer;
    }

    /// Set single output buffer index.
    pub fn set_output_buffer(&mut self, idx: usize) {
        self.output_buffer_indices = vec![idx];
    }

    /// Set multiple output buffer indices (batch scheduling).
    pub fn set_output_buffers(&mut self, indices: Vec<usize>) {
        self.output_buffer_indices = indices;
    }

    /// Compatibility helper: add a compiled kernel as a prepared operation.
    ///
    /// The canonical builder path is `add_op(PreparedOp::...)`.
    pub fn add_kernel(&mut self, kernel: PreparedKernel) {
        self.add_op(PreparedOp::CompiledProgram(kernel));
    }

    /// Add a prepared operation in schedule order.
    pub fn add_op(&mut self, op: PreparedOp) {
        self.add_op_with_instance_dependencies(op, Vec::new());
    }

    /// Add a prepared operation with concrete op-index dependencies.
    pub fn add_op_with_instance_dependencies(&mut self, op: PreparedOp, instance_dependencies: Vec<usize>) {
        self.ops.push(op);
        self.op_instance_dependencies.push(instance_dependencies);
    }

    /// Number of prepared ops added so far. Callers use this to assert 1:1
    /// emission against their source schedule.
    pub fn op_count(&self) -> usize {
        self.ops.len()
    }

    /// Build the ExecutionPlan.
    ///
    /// Finalizes by computing pre-allocated buffer pointers and buffer IDs
    /// for zero-allocation execution.
    pub fn build(mut self) -> Result<ExecutionPlan> {
        for op in &mut self.ops {
            let PreparedOp::CompiledProgram(kernel) = op else {
                continue;
            };

            if kernel.output_indices.is_empty() {
                return Err(crate::error::Error::Execution {
                    reason: format!("CompiledProgram {} has no output indices", kernel.id),
                });
            }
            for &out_idx in &kernel.output_indices {
                if out_idx >= kernel.buffer_indices.len() {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} output index out of range: output_idx={}, kernel_buffers={}",
                            kernel.id,
                            out_idx,
                            kernel.buffer_indices.len()
                        ),
                    });
                }
            }

            let mut buffer_ptrs = Vec::with_capacity(kernel.buffer_indices.len());
            let mut buffer_ids = Vec::with_capacity(kernel.buffer_indices.len());

            for &idx in &kernel.buffer_indices {
                let Some(buffer) = self.buffers.get(idx) else {
                    return Err(crate::error::Error::Execution {
                        reason: format!(
                            "CompiledProgram {} buffer index out of range: idx={}, total_buffers={}",
                            kernel.id,
                            idx,
                            self.buffers.len()
                        ),
                    });
                };
                buffer_ptrs.push(unsafe { buffer.as_raw_ptr() } as usize);
                buffer_ids.push(buffer.id());
            }

            kernel.buffer_ptrs = buffer_ptrs;
            kernel.buffer_ids = buffer_ids;
        }

        if self.output_buffer_indices.is_empty() && !self.buffers.is_empty() {
            return Err(crate::error::Error::Execution {
                reason: "execution plan output buffers must be set explicitly".to_string(),
            });
        }

        let op_order = compute_mixed_op_order_with_instance_dependencies(&self.ops, &self.op_instance_dependencies)?;
        let op_levels = compute_execution_levels_with_instance_dependencies(&self.ops, &self.op_instance_dependencies)?;

        Ok(ExecutionPlan {
            ops: self.ops,
            op_instance_dependencies: self.op_instance_dependencies,
            op_order,
            op_levels,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_indices: self.output_buffer_indices,
            device: self.device,
            runtime_var_vals: HashMap::new(),
            alias_ids: self.alias_ids,
            graph: std::sync::OnceLock::new(),
            #[cfg(target_os = "linux")]
            amd_connector: std::sync::OnceLock::new(),
        })
    }
}

#[cfg(test)]
#[path = "test/unit/execution_plan.rs"]
mod tests;
