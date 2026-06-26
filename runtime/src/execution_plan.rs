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
use snafu::ResultExt;
use svod_device::device::ProgramSpec;
use svod_device::{Buffer, BufferId};
use svod_dtype::DeviceSpec;
use svod_ir::{CustomFunctionKind, Op, UOp};

use crate::error::{ExecSnafu, Result};
use crate::kernel_cache::CachedKernel;
use crate::profiler::{KernelProfile, KernelStaticInfo, ProfileOptions, RunProfile, StageProfile};

type RuntimeLaunchSizes = (Option<[usize; 3]>, Option<[usize; 3]>);

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// Variable values are stored as positional `vals: Vec<i64>` rather than a named
/// HashMap.
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
    /// Values fixed by scheduling (for example from bound ranges) are not
    /// overridden by `execute_with_vars`.
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
    /// dispatch. Replaces N per-kernel submits with one; see
    /// `svod_device::Graph`.
    graph: std::sync::OnceLock<Option<Box<dyn svod_device::Graph>>>,

    /// Reusable per-plan execution context, minted lazily from the first
    /// kernel's program (`Program::new_exec_context`) and held for the plan's
    /// lifetime so every kernel dispatches onto the same backend queue (distinct
    /// plans → distinct queues for cross-plan parallelism). `Some(None)` means
    /// the backend has no reusable context (CPU) → per-call `Program::execute`.
    plan_ctx: std::sync::OnceLock<Option<Box<dyn svod_device::PlanContext>>>,
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
                .context(ExecSnafu { context: format!("kernel {} launch dimensions", kernel.id) })?;
        Ok((Some(dims.global_size), dims.local_size))
    }

    /// Lazily capture all kernels into a backend replay graph. Only backends
    /// that provide a graph factory install one; everything else (and any
    /// non-graphable chain) returns `None` → per-call dispatch. Gated to chains
    /// that are *all* compiled kernels with no runtime vars: copies/views/custom
    /// or dynamic launch dims keep the host in the loop and aren't graphed.
    fn graph(&self) -> &Option<Box<dyn svod_device::Graph>> {
        self.graph.get_or_init(|| self.build_graph().unwrap_or(None))
    }

    fn build_graph(&self) -> Result<Option<Box<dyn svod_device::Graph>>> {
        // Graph capture is on by default: an all-static compiled-kernel plan on a
        // graphable device replays the whole chain as one backend submit, instead
        // of the per-kernel dispatch round-trip. Validated against per-call across
        // the tensor suite (incl. multi-kernel decompositions). Capture walks
        // `op_levels` execution order, NOT the flat `op_order` topological sort
        // (below). Non-graphable plans (runtime vars, no graph factory, chains the
        // backend declines to capture, mixed devices) fall back to per-call via
        // the `Ok(None)` returns below.
        let all_static_kernels =
            self.ops.iter().all(|op| matches!(op, PreparedOp::CompiledProgram(k) if k.runtime_vars.is_empty()));
        if !all_static_kernels || self.ops.is_empty() {
            tracing::debug!(
                target: "svod_runtime::graph",
                ops = self.ops.len(),
                compiled = self.ops.iter().filter(|o| matches!(o, PreparedOp::CompiledProgram(_))).count(),
                with_runtime_vars =
                    self.ops.iter().filter(|o| matches!(o, PreparedOp::CompiledProgram(k) if !k.runtime_vars.is_empty())).count(),
                custom = self.ops.iter().filter(|o| matches!(o, PreparedOp::CustomFunction(_))).count(),
                copies = self.ops.iter().filter(|o| matches!(o, PreparedOp::BufferCopy(_) | PreparedOp::BufferView(_))).count(),
                "graph: per-call fallback (not all-static-compiled)"
            );
            return Ok(None);
        }
        let dev = crate::device_registry::DEVICE_FACTORIES.device(&self.device, svod_device::registry::registry())?;
        let Some(factory) = dev.graph.clone() else { return Ok(None) };
        // Capture in the SAME order `execute` runs the kernels — flatten
        // `op_levels` (level-by-level, intra-level in index order), NOT the flat
        // `op_order` topological sort. The two can differ, and a captured graph
        // replays its packets in strict queue (FIFO) order; using `op_order`
        // would dispatch a different sequence than the per-call path, corrupting
        // results whenever a reused buffer's ordering relies on the level walk
        // (e.g. multi-kernel decompositions like QR).
        // Walk the emission order (level-by-level, intra-level index order) once,
        // building the GraphKernel list AND a parallel hazard-dependency list in
        // lock-step. Hazards are keyed on the RESOLVED buffer GVA (`buffer_ptrs`),
        // not buffer ids: the memory planner aliases distinct logical buffers onto
        // one GVA, so a GVA-keyed walk catches the WAR/WAW the logical
        // `dependencies` field misses. For each emitted kernel `e`:
        //   reads  = buffer_ptrs[j] for j NOT in output_indices
        //   writes = buffer_ptrs[j] for j     in output_indices
        //   deps   = last_writer[read]  (RAW)
        //          ∪ last_writer[write] (WAW) ∪ readers[write] (WAR)
        // then update: readers[read].push(e); for each write set last_writer=e and
        // clear readers (a fresh writer; future readers depend on it via RAW).
        //
        // Soundness rests on `output_indices` being the COMPLETE write-set: a
        // missed write would leave no last_writer and (BARRIER stripped) race a
        // later reader. That holds here by construction — `output_indices` is
        // derived from the kernel's STORE targets (`ProgramSpec.outs`) and a
        // compiled kernel writes only via STOREs, and this walk only processes
        // `CompiledProgram` ops (the `else { return Ok(None) }` below). Custom
        // functions / copies are not graphed, so the invariant is not relied on
        // for them.
        let mut kernels = Vec::with_capacity(self.ops.len());
        let mut last_writer: HashMap<usize, usize> = HashMap::new();
        let mut readers: HashMap<usize, Vec<usize>> = HashMap::new();
        for level in &self.op_levels {
            for &idx in level {
                let PreparedOp::CompiledProgram(k) = &self.ops[idx] else { return Ok(None) };
                let (global_size, local_size) = Self::kernel_launch_sizes(k)?;
                let e = kernels.len();

                let write_pos: std::collections::HashSet<usize> = k.output_indices.iter().copied().collect();
                let writes: Vec<usize> =
                    k.output_indices.iter().filter_map(|&j| k.buffer_ptrs.get(j).copied()).collect();
                let reads: Vec<usize> =
                    (0..k.buffer_ptrs.len()).filter(|j| !write_pos.contains(j)).map(|j| k.buffer_ptrs[j]).collect();

                let mut deps: std::collections::HashSet<usize> = std::collections::HashSet::new();
                for &b in &reads {
                    if let Some(&w) = last_writer.get(&b) {
                        deps.insert(w); // RAW
                    }
                }
                for &b in &writes {
                    if let Some(&w) = last_writer.get(&b) {
                        deps.insert(w); // WAW
                    }
                    if let Some(rs) = readers.get(&b) {
                        deps.extend(rs.iter().copied()); // WAR
                    }
                }
                deps.remove(&e);
                let mut deps: Vec<usize> = deps.into_iter().collect();
                deps.sort_unstable();

                // Commit this kernel's effect on the hazard state.
                for &b in &reads {
                    readers.entry(b).or_default().push(e);
                }
                for &b in &writes {
                    last_writer.insert(b, e);
                    readers.insert(b, Vec::new());
                }

                kernels.push(svod_device::GraphKernel {
                    program: k.kernel.program.as_ref(),
                    buffers: k.buffer_ptrs.iter().map(|&p| p as *mut u8).collect(),
                    vals: k.vals.clone(),
                    global_size,
                    local_size,
                    deps,
                });
            }
        }
        let result = factory(&kernels).context(ExecSnafu { context: "graph capture" })?;
        tracing::debug!(target: "svod_runtime::graph", kernels = kernels.len(), captured = result.is_some(), "graph: capture result");
        Ok(result)
    }

    /// Lazily mint (once) the plan's execution context from `program` and cache
    /// it for the plan's lifetime. `None` ⇒ the backend has no reusable context
    /// (CPU) and the caller dispatches per-call via `Program::execute`. The
    /// context binds the plan to a shared queue; distinct plans spread onto
    /// distinct queues for cross-plan parallelism.
    fn plan_ctx(&self, program: &dyn svod_device::Program) -> Result<Option<&dyn svod_device::PlanContext>> {
        if let Some(slot) = self.plan_ctx.get() {
            return Ok(slot.as_deref());
        }
        let ctx = program.new_exec_context().context(ExecSnafu { context: "mint plan exec context" })?;
        // One-shot init race: if two threads see empty, both mint; only one wins
        // `set()`. The loser's context drops here harmlessly (its `Arc` over the
        // shared queue just decrements).
        let _ = self.plan_ctx.set(ctx);
        Ok(self.plan_ctx.get().expect("set above").as_deref())
    }

    /// Submit one kernel. When `profile` is set and the backend stamps
    /// dispatches, returns the dispatch's HW timestamp handle (`None` otherwise,
    /// e.g. CPU); the caller must hold it until after `synchronize`. The
    /// non-profiled `execute` path passes `false` and drops the handle.
    #[inline]
    fn execute_kernel(
        &self,
        kernel: &PreparedKernel,
        profile: bool,
    ) -> Result<Option<Arc<dyn svod_device::DispatchTimestamps>>> {
        let buffer_ptrs: SmallVec<[*mut u8; 8]> = kernel.buffer_ptrs.iter().map(|&ptr| ptr as *mut u8).collect();
        let (global_size, local_size) = Self::kernel_launch_sizes(kernel)?;
        let program = kernel.kernel.program.as_ref();
        // Backends that expose a reusable context dispatch through it so all the
        // plan's kernels share one queue. Others (CPU) return `None` and fall
        // back to per-call `Program::execute`.
        if let Some(ctx) = self.plan_ctx(program)? {
            return unsafe { ctx.dispatch(program, &buffer_ptrs, &kernel.vals, global_size, local_size, profile) }
                .context(ExecSnafu { context: format!("dispatch kernel {}", kernel.id) });
        }
        unsafe {
            program
                // wait=false: async submit. GPU ordering is enforced by the
                // device timeline; host reads (copyout / as_*) synchronize.
                .execute(&buffer_ptrs, &kernel.vals, global_size, local_size, /*wait=*/ false)
                .map(|_| None)
                .context(ExecSnafu { context: format!("execute kernel {}", kernel.id) })
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
        dst.copy_from(src).context(ExecSnafu { context: format!("copy op {}", copy.id) })
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
            PreparedOp::CompiledProgram(kernel) => self.execute_kernel(kernel, /*profile=*/ false).map(|_| ()),
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

    /// Copy `len` bytes from output `out_pos` (`src_off`) into the plan buffer
    /// at `dst_index` (`dst_off`) — both owned by this plan, so the borrow is
    /// split internally. The transfer stays on-device (SDMA when either side
    /// is device-local), letting recurrent state recycle output→input without
    /// a host round-trip.
    pub fn copy_output_region_to_buffer(
        &mut self,
        out_pos: usize,
        dst_index: usize,
        dst_off: usize,
        src_off: usize,
        len: usize,
    ) -> Result<()> {
        let src_index = *self.output_buffer_indices.get(out_pos).ok_or_else(|| crate::error::Error::Execution {
            reason: format!("copy_output_region_to_buffer: output {out_pos} out of range"),
        })?;
        if src_index == dst_index {
            return Err(crate::error::Error::Execution {
                reason: "copy_output_region_to_buffer: output aliases destination".into(),
            });
        }
        let (dst, src) = if dst_index < src_index {
            let (a, b) = self.buffers.split_at_mut(src_index);
            (&mut a[dst_index], &b[0])
        } else {
            let (a, b) = self.buffers.split_at_mut(dst_index);
            (&mut b[0], &a[src_index])
        };
        dst.copy_region_from(dst_off, src, src_off, len).context(ExecSnafu { context: "on-device state copy" })
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
    /// Walks `op_levels` level-by-level and runs each op within a level in
    /// builder-insertion order. Multi-plan concurrency comes from distinct
    /// `ExecutionPlan`s (e.g. BEAM search candidates) spread onto distinct
    /// backend execution contexts from the device — not from rayon inside one
    /// plan. The level-by-level iteration (vs. a flat `op_order` topological
    /// linearization) is load-bearing for iterative CPU kernels (QR, etc.)
    /// whose codegen is sensitive to within-level scheduling order — see
    /// `test_execute_walks_op_levels_in_level_order`.
    pub fn execute(&self) -> Result<()> {
        // Fast path: one captured graph submit instead of per-kernel dispatch.
        // Built once, then every call just replays.
        if let Some(graph) = self.graph().as_deref() {
            return graph.replay(&[]).context(ExecSnafu { context: "graph replay" });
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
    /// sorted.sort_by(|a, b| b.wall.cmp(&a.wall));
    /// for p in &sorted[..10.min(sorted.len())] {
    ///     println!("{:>8.3}ms  {}", p.wall.as_secs_f64() * 1000.0, p.kernel.entry_point);
    /// }
    /// ```
    /// Always dispatches per-kernel (never the captured graph): a graph replay
    /// has one signal per batch, so per-dispatch stamps don't exist there.
    /// Profiled timings reflect per-dispatch execution, not graph replay.
    pub fn execute_profiled(&self) -> Result<Vec<KernelProfile>> {
        let mut profiles = Vec::with_capacity(self.op_order.len());
        // Per-dispatch HW timestamp handles, harvested after the drain below
        // (the GPU stamps a dispatch's signal only on retirement).
        let mut handles: Vec<Option<Arc<dyn svod_device::DispatchTimestamps>>> =
            Vec::with_capacity(self.op_order.len());
        for level in &self.op_levels {
            for &idx in level {
                match &self.ops[idx] {
                    PreparedOp::CompiledProgram(kernel) => {
                        let start = Instant::now();
                        let handle = self.execute_kernel(kernel, /*profile=*/ true)?;
                        handles.push(handle);
                        profiles.push(KernelProfile {
                            kernel: Arc::clone(&kernel.kernel),
                            device: kernel.device.clone(),
                            num_buffers: kernel.buffer_ptrs.len(),
                            wall: start.elapsed(),
                            gpu_start_ns: None,
                            gpu_end_ns: None,
                            static_info: None,
                            counters: None,
                        });
                    }
                    PreparedOp::BufferCopy(copy) => self.execute_copy(copy)?,
                    PreparedOp::BufferView(view) => self.execute_buffer_view(view)?,
                    PreparedOp::CustomFunction(custom) => self.execute_custom_function(custom)?,
                }
            }
        }
        if handles.iter().any(Option::is_some) {
            // Handles exist only when a backend stamps dispatches, which means a
            // context was minted; drain it so the GPU has written back the
            // per-dispatch timestamps before we read them.
            if let Some(ctx) = self.plan_ctx.get().and_then(|s| s.as_deref()) {
                ctx.synchronize().context(ExecSnafu { context: "profiled drain" })?;
            }
            for (profile, handle) in profiles.iter_mut().zip(&handles) {
                if let Some((start, end)) = handle.as_ref().and_then(|h| h.timestamps_ns()) {
                    profile.gpu_start_ns = Some(start);
                    profile.gpu_end_ns = Some(end);
                }
                profile.counters = handle.as_ref().and_then(|h| h.counters());
            }
        }
        Ok(profiles)
    }

    /// Profile the plan: run the per-dispatch path `opts.iters` times, keeping
    /// each kernel's minimum device time (robust to outliers). Returns a
    /// single-stage [`RunProfile`]; render it with [`RunProfile::render_table`].
    ///
    /// Tier-2/3 static analysis (`opts.static_analysis`) and Tier-4 hardware
    /// counters (`opts.counters`) attach to each [`KernelProfile`] when enabled.
    /// Tier-4 is gated: it requires `pmc_available()` and a stable power state;
    /// otherwise it degrades gracefully to timing-only with a one-line note.
    pub fn profile(&self, opts: &ProfileOptions) -> Result<RunProfile> {
        let start = Instant::now();
        // Tier-4: arm hardware counters on the plan's context when requested and
        // the backend supports it in a stable power state. Degrade gracefully
        // (no counters, a one-line note) rather than failing the run.
        let counters = opts.counters.counters();
        let armed_ctx = if counters.is_empty() {
            None
        } else {
            let first_program = self.op_levels.iter().flatten().find_map(|&idx| match &self.ops[idx] {
                PreparedOp::CompiledProgram(k) => Some(k.kernel.program.as_ref()),
                _ => None,
            });
            match first_program.and_then(|p| self.plan_ctx(p).ok().flatten()) {
                Some(ctx) if ctx.pmc_available() => {
                    ctx.set_pmc(&counters);
                    Some(ctx)
                }
                Some(_) => {
                    eprintln!(
                        "SVOD_PMC: hardware counters unavailable (needs a profile_standard \
                         power state — run `amd-smi set -l stable_std`); reporting timing only"
                    );
                    None
                }
                None => None,
            }
        };
        // Each pass is one "profile" stage; merge passes by per-kernel min time.
        let run = |kernels| RunProfile { stages: vec![StageProfile::gpu("profile", start.elapsed(), kernels)] };
        let mut report = run(self.execute_profiled()?);
        for _ in 1..opts.iters {
            report.merge_min(run(self.execute_profiled()?));
        }
        // Disarm so later non-profiled executions on this context don't pay for
        // (or perturb from) counter programming.
        if let Some(ctx) = armed_ctx {
            ctx.set_pmc(&[]);
        }
        if opts.static_analysis {
            // Profiles are in dispatch order; the compiled kernels in op_levels
            // order line up one-to-one, so zip attaches each kernel's analysis.
            let kernels = self.op_levels.iter().flatten().filter_map(|&idx| match &self.ops[idx] {
                PreparedOp::CompiledProgram(k) => Some(k),
                _ => None,
            });
            for (profile, pk) in report.stages[0].kernels.iter_mut().zip(kernels) {
                profile.static_info = Some(self.kernel_static_info(pk));
            }
        }
        Ok(report)
    }

    /// Tier-2/3 static analysis for one kernel: AST flop estimate, compulsory
    /// byte traffic (each distinct buffer counted once), and decoded GPU
    /// resources when the backend exposes them.
    fn kernel_static_info(&self, pk: &PreparedKernel) -> KernelStaticInfo {
        // The AST walk saturates to u64::MAX when a range/special has an
        // unbounded symbolic end (common in hand-built kernels) — treat that as
        // "no reliable count" rather than reporting a garbage roofline.
        let raw_flops = svod_ir::compute_ops_estimate(&pk.ast);
        let est_flops = (raw_flops != u64::MAX).then_some(raw_flops);
        let mut seen = std::collections::HashSet::new();
        let est_bytes =
            pk.buffer_indices.iter().filter(|&&i| seen.insert(i)).map(|&i| self.buffers[i].size() as u64).sum();
        let resources = pk.kernel.program.resource_usage();
        KernelStaticInfo { est_flops, est_bytes, resources }
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

    /// Get the first output buffer index, or `None` for an output-less plan
    /// (mirrors [`Self::output_buffer`], which also returns `Option`).
    pub fn output_buffer_idx(&self) -> Option<usize> {
        self.output_buffer_indices.first().copied()
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

// No explicit `Drop for ExecutionPlan`: the plan's `plan_ctx`
// (`OnceLock<Option<Box<dyn PlanContext>>>`) just holds an `Arc` over a
// backend-shared queue/context. On plan drop the `Arc` decrements; the
// underlying queue stays in the backend's pool (freed only at device close), so
// plan churn never tears down backend queues.

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
            plan_ctx: std::sync::OnceLock::new(),
        })
    }
}

#[cfg(test)]
#[path = "test/unit/execution_plan.rs"]
mod tests;
