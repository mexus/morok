//! Pre-compiled execution plan for parallel kernel execution.
//!
//! This module provides `ExecutionPlan` - a structure that separates
//! one-time preparation (kernel compilation, buffer allocation, parallel
//! group computation) from fast repeated execution.
//!
//! # Design
//!
//! Like Python's code objects or PyTorch's traced graphs, `ExecutionPlan`
//! captures all the work needed to execute a computation graph:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    PREPARATION (one-time)                        │
//! │  Schedule → expand → compile_kernels → compute_parallel_groups  │
//! │                           ↓                                      │
//! │                    ExecutionPlan                                 │
//! └─────────────────────────────────────────────────────────────────┘
//!                               ↓
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    EXECUTION (fast path)                         │
//! │  for group in parallel_groups:                                   │
//! │      if single_kernel → execute_kernel()                         │
//! │      else → execute_parallel_group()                             │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```ignore
//! // One-time preparation (compiles kernels, allocates buffers)
//! let plan = ExecutionPlan::prepare(&schedule)?;
//!
//! // Fast execution (can be called many times)
//! plan.execute(&mut executor)?;
//!
//! // Get results
//! let output = plan.output_buffer();
//! ```

use std::collections::HashMap;
use std::sync::Arc;

use morok_device::{Buffer, BufferId};
use morok_dtype::DeviceSpec;
use morok_ir::UOp;

use crate::error::Result;
use crate::executor::UnifiedExecutor;
use crate::kernel_cache::CachedKernel;

// ============================================================================
// Core Structures
// ============================================================================

/// A pre-compiled kernel ready for execution.
///
/// This is the "lowered" form of a `ScheduleItem` - all compilation is complete,
/// only buffer binding and execution remain.
///
/// # Tinygrad Alignment
///
/// Variable values are stored as positional `vals: Vec<i64>` rather than a named
/// HashMap, matching Tinygrad's `vals: tuple[int, ...]` parameter style.
/// The order matches `var_names` in the `CachedKernel`.
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
    /// Used for dependency tracking.
    pub output_indices: Vec<usize>,

    /// Variable values in positional order (matches `var_names` in CachedKernel).
    /// Pre-expanded from bound_ranges during preparation.
    pub vals: Vec<i64>,

    /// Kernel IDs that must complete before this one (dependencies).
    pub dependencies: Vec<u64>,

    /// Pre-computed raw buffer pointers for zero-allocation execution.
    /// Computed once during prepare(), stable for the lifetime of ExecutionPlan.
    /// Initialize as empty; populated by `ExecutionPlanBuilder::build()`.
    /// SAFETY: Pointers are valid as long as ExecutionPlan owns the buffers.
    pub buffer_ptrs: Vec<*mut u8>,

    /// Pre-computed buffer IDs for dependency tracking.
    /// Avoids looking up Buffer objects during execution.
    /// Initialize as empty; populated by `ExecutionPlanBuilder::build()`.
    pub buffer_ids: Vec<BufferId>,
}

/// A group of kernels that can execute in parallel.
///
/// Within a group, kernels have no buffer conflicts and can be
/// executed together using `rayon::scope`.
#[derive(Debug, Clone)]
pub struct ParallelGroup {
    /// Indices into `ExecutionPlan::kernels` for kernels in this group.
    pub kernel_indices: Vec<usize>,
}

/// Pre-compiled execution plan for a computation graph.
///
/// Created once via `ExecutionPlan::prepare()`, then executed multiple times
/// with the same buffers. The plan owns all its buffers.
pub struct ExecutionPlan {
    /// Pre-compiled kernels in topological order.
    kernels: Vec<PreparedKernel>,

    /// Parallel groups for execution.
    /// Each group contains kernel indices that can run in parallel.
    parallel_groups: Vec<ParallelGroup>,

    /// ALL buffers owned by this plan (inputs, intermediates, outputs).
    /// Allocated during prepare(), reused across execute() calls.
    buffers: Vec<Buffer>,

    /// Mapping: AST id → buffer index (for kernel buffer binding).
    ast_to_buffer: HashMap<u64, usize>,

    /// Indices of output buffers in `buffers` (matches SINK source order).
    output_buffer_indices: Vec<usize>,

    /// Primary device for this plan.
    device: DeviceSpec,

    /// Additional UOp IDs registered as aliases that need cleanup.
    alias_ids: Vec<u64>,
}

// ============================================================================
// ExecutionPlan Implementation
// ============================================================================

impl ExecutionPlan {
    /// Get the first (or only) output buffer after execution.
    pub fn output_buffer(&self) -> &Buffer {
        &self.buffers[self.output_buffer_indices[0]]
    }

    /// Get output buffer by position (matches SINK source order for batch).
    pub fn output_buffer_at(&self, position: usize) -> &Buffer {
        &self.buffers[self.output_buffer_indices[position]]
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
    ///
    /// Used in the trace-and-rerun pattern: after `prepare()`, use this to write
    /// new input data into the plan's pre-allocated buffers, then call `execute()`.
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

    /// Get all prepared kernels.
    pub fn prepared_kernels(&self) -> &[PreparedKernel] {
        &self.kernels
    }

    /// Iterate over compiled kernels (for inspecting generated source code).
    ///
    /// Each `CachedKernel` contains:
    /// - `code`: Generated source code (LLVM IR, etc.)
    /// - `device`: Target device string
    /// - `entry_point`: Kernel entry function name
    ///
    /// # Example
    ///
    /// ```ignore
    /// for kernel in plan.kernels() {
    ///     println!("=== {} ===", kernel.entry_point);
    ///     println!("{}", kernel.code);
    /// }
    /// ```
    pub fn kernels(&self) -> impl Iterator<Item = &CachedKernel> {
        self.kernels.iter().map(|pk| pk.kernel.as_ref())
    }

    /// Get all parallel groups.
    pub fn parallel_groups(&self) -> &[ParallelGroup] {
        &self.parallel_groups
    }

    /// Execute the plan. All buffers already allocated during prepare().
    ///
    /// This is the fast path - no compilation, no allocation, just execution.
    /// Uses `rayon::scope` for zero-allocation parallel execution.
    ///
    /// # Arguments
    ///
    /// * `executor` - The UnifiedExecutor for dependency tracking
    ///
    /// # Returns
    ///
    /// Ok(()) on success, error if any kernel fails.
    pub fn execute(&self, executor: &mut UnifiedExecutor) -> Result<()> {
        for group in &self.parallel_groups {
            if !group.kernel_indices.is_empty() {
                executor.execute_kernels_by_indices(&self.kernels, &group.kernel_indices, &self.buffers)?;
            }
        }
        Ok(())
    }

    /// Re-execute the plan with different variable bindings.
    ///
    /// The kernel code is NOT recompiled; only the `vals` passed to each kernel
    /// are updated. Buffers must be allocated to max variable values (which is
    /// the default when using `Variable::bind()`).
    ///
    /// Use with `BoundVariable::bind()` for typed, bounds-checked rebinding:
    /// ```ignore
    /// let batch = model.variables["batch"].bind(new_size)?;
    /// plan.execute_with_vars(&mut executor, &[(&batch)])?;
    /// ```
    ///
    /// Variables not present in `var_vals` keep their existing values from
    /// `prepare()` (or the previous `execute_with_vars` call). This allows
    /// internal variables like `thread_id` (injected by CPU codegen for
    /// parallel dispatch) to remain untouched by user code.
    pub fn execute_with_vars(&mut self, executor: &mut UnifiedExecutor, var_vals: &[(&str, i64)]) -> Result<()> {
        for kernel in &mut self.kernels {
            for (idx, name) in kernel.kernel.var_names.iter().enumerate() {
                if let Some((_, v)) = var_vals.iter().find(|(n, _)| *n == name.as_str()) {
                    kernel.vals[idx] = *v;
                } else if name != "thread_id" {
                    // User variable not provided — keep existing value but warn
                    tracing::warn!(variable = %name, "execute_with_vars: variable not in var_vals, using existing value");
                }
                // thread_id: internal codegen variable, patched per-thread at dispatch time
            }
        }
        self.execute(executor)
    }

    /// Get the first output buffer index.
    pub fn output_buffer_idx(&self) -> usize {
        self.output_buffer_indices[0]
    }

    /// Get the AST ID to buffer index mapping.
    ///
    /// This returns UOp IDs mapped to their buffer indices in this plan.
    /// Used for buffer registry cleanup.
    pub fn ast_to_buffer_map(&self) -> &HashMap<u64, usize> {
        &self.ast_to_buffer
    }

    /// Release intermediate buffers from the global buffer registry.
    ///
    /// Call this after you're done executing the plan to free intermediate
    /// buffers from the global registry. The output buffer is preserved.
    ///
    /// This is useful for the `prepare()` + `execute()` pattern where you
    /// want to clean up after the final execution. For `realize()`, this
    /// cleanup happens automatically.
    ///
    /// # Arguments
    ///
    /// * `remove_fn` - Function to remove a buffer from the registry by AST ID.
    ///   Typically `morok_tensor::buffer_registry::remove_buffer`.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let plan = tensor.prepare()?;
    /// for _ in 0..100 {
    ///     plan.execute(&mut executor)?;
    /// }
    /// // Clean up intermediate buffers after final execution
    /// plan.release_intermediate_buffers(morok_tensor::buffer_registry::remove_buffer);
    /// ```
    pub fn release_intermediate_buffers<F>(&self, remove_fn: F)
    where
        F: Fn(u64),
    {
        self.release_buffers_impl(remove_fn, true);
    }

    /// Release ALL buffers from the global registry, including the output.
    ///
    /// Use this when you're done with the plan and want to clean up everything.
    /// The buffers themselves remain valid (owned by ExecutionPlan) but the
    /// registry entries are removed.
    ///
    /// For `realize()`, use this method since the output buffer will be
    /// re-registered under a new ID anyway.
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
        f.debug_struct("ExecutionPlan")
            .field("kernels", &self.kernels.len())
            .field("parallel_groups", &self.parallel_groups.len())
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
            .field("dependencies", &self.dependencies)
            .finish()
    }
}

// ============================================================================
// Builder for ExecutionPlan
// ============================================================================

/// Builder for creating ExecutionPlan from schedule data.
///
/// This is used by the tensor crate to construct ExecutionPlan
/// since it has access to Schedule and related types.
pub struct ExecutionPlanBuilder {
    kernels: Vec<PreparedKernel>,
    parallel_groups: Vec<ParallelGroup>,
    buffers: Vec<Buffer>,
    ast_to_buffer: HashMap<u64, usize>,
    output_buffer_indices: Vec<usize>,
    device: DeviceSpec,
    /// Additional UOp IDs registered as aliases in buffer_registry.
    /// These need to be cleaned up but don't have their own buffer index.
    alias_ids: Vec<u64>,
}

impl ExecutionPlanBuilder {
    /// Create a new builder.
    pub fn new(device: DeviceSpec) -> Self {
        Self {
            kernels: Vec::new(),
            parallel_groups: Vec::new(),
            buffers: Vec::new(),
            ast_to_buffer: HashMap::new(),
            output_buffer_indices: Vec::new(),
            device,
            alias_ids: Vec::new(),
        }
    }

    /// Add alias IDs that need cleanup.
    ///
    /// These are UOp IDs where a buffer was registered under an alternate key
    /// for lookup convenience. They don't have their own buffer but need to be
    /// removed from the registry during cleanup.
    pub fn add_alias_ids(&mut self, ids: impl IntoIterator<Item = u64>) {
        self.alias_ids.extend(ids);
    }

    /// Add a buffer to the plan.
    ///
    /// Returns the buffer index.
    pub fn add_buffer(&mut self, ast_id: u64, buffer: Buffer) -> usize {
        let idx = self.buffers.len();
        self.buffers.push(buffer);
        self.ast_to_buffer.insert(ast_id, idx);
        idx
    }

    /// Set single output buffer index (backwards compat).
    pub fn set_output_buffer(&mut self, idx: usize) {
        self.output_buffer_indices = vec![idx];
    }

    /// Set multiple output buffer indices (batch scheduling).
    pub fn set_output_buffers(&mut self, indices: Vec<usize>) {
        self.output_buffer_indices = indices;
    }

    /// Add a prepared kernel.
    pub fn add_kernel(&mut self, kernel: PreparedKernel) {
        self.kernels.push(kernel);
    }

    /// Set parallel groups.
    pub fn set_parallel_groups(&mut self, groups: Vec<ParallelGroup>) {
        self.parallel_groups = groups;
    }

    /// Build the ExecutionPlan.
    ///
    /// This finalizes the plan by computing pre-allocated buffer pointers
    /// and buffer IDs for zero-allocation execution.
    pub fn build(mut self) -> ExecutionPlan {
        // Compute buffer_ptrs and buffer_ids for each kernel
        for kernel in &mut self.kernels {
            kernel.buffer_ptrs = kernel
                .buffer_indices
                .iter()
                .map(|&idx| {
                    // SAFETY: buffers are allocated and owned by ExecutionPlan
                    unsafe { self.buffers[idx].as_raw_ptr() }
                })
                .collect();

            // Pre-compute buffer IDs for dependency tracking
            kernel.buffer_ids = kernel.buffer_indices.iter().map(|&idx| self.buffers[idx].id()).collect();
        }

        // Fallback: if no output was set, default to index 0
        if self.output_buffer_indices.is_empty() && !self.buffers.is_empty() {
            self.output_buffer_indices = vec![0];
        }

        ExecutionPlan {
            kernels: self.kernels,
            parallel_groups: self.parallel_groups,
            buffers: self.buffers,
            ast_to_buffer: self.ast_to_buffer,
            output_buffer_indices: self.output_buffer_indices,
            device: self.device,
            alias_ids: self.alias_ids,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let plan = builder.build();

        assert!(plan.kernels.is_empty());
        assert!(plan.parallel_groups.is_empty());
        assert!(plan.buffers.is_empty());
        assert_eq!(plan.device, DeviceSpec::Cpu);
    }

    #[test]
    fn test_parallel_group_debug() {
        let group = ParallelGroup { kernel_indices: vec![0, 1, 2] };

        let debug_str = format!("{:?}", group);
        assert!(debug_str.contains("kernel_indices"));
    }
}
