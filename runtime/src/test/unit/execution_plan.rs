use super::*;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use svod_device::allocator::{Allocator, BufferSpec, CpuAllocator, RawBuffer};
use svod_device::device::Program;
use svod_device::device::{CopyEndpoint, NativeReplayDecline, NativeReplayOutcome, PlanCall, PlanContext};
use svod_dtype::DType;
use svod_ir::{CustomFunctionKind, UOp};

fn default_launch_size() -> [Arc<UOp>; 3] {
    [UOp::index_const(1), UOp::index_const(1), UOp::index_const(1)]
}

#[test]
fn test_builder_basic() {
    let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let plan = builder.build().expect("build plan");

    assert!(plan.prepared_kernels().is_empty());
    assert!(plan.buffers.is_empty());
    assert_eq!(plan.device, DeviceSpec::Cpu);
}

#[test]
fn test_empty_plan_output_buffer_returns_none() {
    let builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let plan = builder.build().expect("build plan");

    assert!(plan.output_buffer().is_none(), "empty plan must not expose an output buffer");
    assert!(plan.output_buffer_at(0).is_none(), "empty plan output_buffer_at must be None");
    assert!(plan.output_buffer_at(7).is_none(), "out-of-range output_buffer_at must be None");
}

#[test]
fn test_copy_output_region_to_buffer() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let mut output = Buffer::new(alloc.clone(), DType::UInt8, vec![8], Default::default());
    let mut destination = Buffer::new(alloc, DType::UInt8, vec![8], Default::default());
    output.copyin(&[0, 1, 2, 3, 4, 5, 6, 7]).unwrap();
    destination.copyin(&[9; 8]).unwrap();

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let output_idx = builder.add_buffer(1, output);
    let destination_idx = builder.add_buffer(2, destination);
    builder.set_output_buffer(output_idx);
    let mut plan = builder.build().unwrap();

    plan.copy_output_region_to_buffer(0, destination_idx, 2, 3, 3).unwrap();
    let mut actual = [0; 8];
    plan.buffers()[destination_idx].copyout(&mut actual).unwrap();
    assert_eq!(actual, [9, 9, 3, 4, 5, 9, 9, 9]);

    assert!(plan.copy_output_region_to_buffer(1, destination_idx, 0, 0, 1).is_err());
    assert!(plan.copy_output_region_to_buffer(0, 99, 0, 0, 1).is_err());
    assert!(plan.copy_output_region_to_buffer(0, output_idx, 0, 0, 1).is_err());
}

#[test]
fn test_builder_map_buffer_alias() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let buf = Buffer::new(alloc, svod_dtype::DType::Float32, vec![8], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let idx = builder.add_buffer(10, buf);
    builder.map_buffer(11, idx);
    builder.set_output_buffer(idx);
    let plan = builder.build().expect("build plan");

    assert_eq!(plan.ast_to_buffer_map().get(&10), Some(&idx));
    assert_eq!(plan.ast_to_buffer_map().get(&11), Some(&idx));
    assert_eq!(plan.buffers().len(), 1);
}

#[test]
fn test_builder_requires_explicit_output_indices() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let buf = Buffer::new(alloc, svod_dtype::DType::Float32, vec![8], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    builder.add_buffer(10, buf);

    let err = builder.build().expect_err("build should fail when outputs are not set");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("output buffers must be set explicitly"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_buffer_copy_op() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1, dst);
    let src_idx = builder.add_buffer(2, src);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 99,
        buffer_indices: vec![dst_idx, src_idx],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute copy op");

    let mut output_data = vec![0.0f32; 4];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("dst copyout");

    assert_eq!(output_data, input_data);
}

#[test]
fn test_execute_custom_function_op_returns_unsupported() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(201, dst);
    let src_idx = builder.add_buffer(202, src);
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 200,
        kind: CustomFunctionKind::EncDec,
        attrs: smallvec::smallvec![svod_ir::UOp::index_const(3)],
        buffer_indices: vec![dst_idx, src_idx],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("EncDec runtime should be explicit unsupported");
    match err {
        crate::error::Error::Unsupported { kind, reason } => {
            assert_eq!(kind, "EncDec");
            assert!(reason.contains("attrs=1"), "unexpected reason: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    assert!(
        matches!(plan.execute(), Err(crate::error::Error::PlanPoisoned { .. })),
        "a callback failure after epoch reservation must reject immediate retry"
    );
}

#[test]
fn test_execution_plan_runs_host_allreduce_numerically() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let make = |values: &[f32]| {
        let mut buffer = Buffer::new(alloc.clone(), DType::Float32, vec![values.len()], Default::default());
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        buffer.copyin(&bytes).unwrap();
        buffer
    };
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let output = builder.add_buffer(301, make(&[0.0, 0.0]));
    let shard0 = builder.add_buffer(302, make(&[4.0, 7.0]));
    let shard1 = builder.add_buffer(303, make(&[11.0, 7.0]));
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 300,
        kind: CustomFunctionKind::AllReduce { reduce_op: svod_ir::ReduceOp::Add },
        attrs: smallvec::smallvec![],
        buffer_indices: vec![output, shard0, shard1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
    }));
    builder.set_output_buffer(output);
    let plan = builder.build().unwrap();

    plan.execute().unwrap();
    let mut bytes = vec![0; 2 * std::mem::size_of::<f32>()];
    plan.output_buffer().unwrap().copyout(&mut bytes).unwrap();
    let values = bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect::<Vec<_>>();
    assert_eq!(values, vec![15.0, 14.0]);
}

#[derive(Debug)]
struct Copy4F32Program {
    calls: Arc<AtomicUsize>,
}

impl Program for Copy4F32Program {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        let bytes = 4 * std::mem::size_of::<f32>();
        unsafe {
            std::ptr::copy_nonoverlapping(buffers[1], buffers[0], bytes);
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "copy4f32"
    }
}

#[derive(Debug)]
struct FixedGraphTimestamps {
    start: u64,
    end: u64,
    drops: Arc<AtomicUsize>,
}

impl svod_device::DispatchTimestamps for FixedGraphTimestamps {
    fn timestamps_ns(&self) -> Option<(u64, u64)> {
        Some((self.start, self.end))
    }
}

impl Drop for FixedGraphTimestamps {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::SeqCst);
    }
}

struct ProfileReplayGraph {
    replays: Arc<AtomicUsize>,
    timestamp_drops: Arc<AtomicUsize>,
}

impl svod_device::Graph for ProfileReplayGraph {
    fn replay(&self, _buffers: &[u64], _vals: &[i64]) -> svod_device::Result<()> {
        self.replays.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn replay_profiled(
        &self,
        _buffers: &[u64],
        _vals: &[i64],
    ) -> svod_device::Result<Option<Vec<Arc<dyn svod_device::DispatchTimestamps>>>> {
        self.replays.fetch_add(1, Ordering::SeqCst);
        Ok(Some(vec![Arc::new(FixedGraphTimestamps {
            start: 100,
            end: 140,
            drops: Arc::clone(&self.timestamp_drops),
        })]))
    }
}

#[test]
fn profiled_execution_uses_graph_replay_timestamps_when_available() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let buffer = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    let calls = Arc::new(AtomicUsize::new(0));
    let replays = Arc::new(AtomicUsize::new(0));
    let timestamp_drops = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let buffer_idx = builder.add_buffer(42, buffer);
    builder.add_kernel(PreparedKernel {
        id: 42,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: Arc::clone(&calls) }),
            device: "CPU".into(),
            code: String::new(),
            entry_point: "profile_graph".into(),
            var_names: Vec::new(),
            globals: vec![0],
            outs: vec![0],
            ins: Vec::new(),
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![buffer_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(buffer_idx);
    let plan = builder.build().unwrap();
    plan.graph
        .set(Some(Box::new(ProfileReplayGraph {
            replays: Arc::clone(&replays),
            timestamp_drops: Arc::clone(&timestamp_drops),
        })))
        .map_err(|_| ())
        .unwrap();

    let profiles = plan.execute_profiled().unwrap();
    assert_eq!(profiles.len(), 1);
    assert_eq!((profiles[0].gpu_start_ns, profiles[0].gpu_end_ns), (Some(100), Some(140)));
    assert_eq!(replays.load(Ordering::SeqCst), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 0, "profiled graph must not redispatch per-call");
    assert_eq!(timestamp_drops.load(Ordering::SeqCst), 1, "finalizer must release handles after collection");
}

#[test]
fn test_builder_rejects_invalid_compiled_output_index() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let a_idx = builder.add_buffer(700, a);
    let b_idx = builder.add_buffer(701, b);
    builder.add_kernel(PreparedKernel {
        id: 77,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: Arc::new(AtomicUsize::new(0)) }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![2],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![a_idx, b_idx],
        output_indices: vec![2],
        input_indices: Vec::new(),
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(a_idx);

    let err = builder.build().expect_err("invalid compiled output index should fail build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("output index out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[derive(Debug)]
struct RecordLaunchProgram {
    calls: Arc<AtomicUsize>,
    global_x: Arc<AtomicUsize>,
    first_val: Arc<AtomicUsize>,
}

#[derive(Clone)]
struct RecordLaunchCounters {
    calls: Arc<AtomicUsize>,
    global_x: Arc<AtomicUsize>,
    first_val: Arc<AtomicUsize>,
}

impl Program for RecordLaunchProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        vals: &[i64],
        global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.global_x.store(global_size.map(|size| size[0]).unwrap_or(0), Ordering::SeqCst);
        self.first_val.store(vals.first().copied().unwrap_or(0) as usize, Ordering::SeqCst);
        Ok(())
    }

    fn name(&self) -> &str {
        "record_launch"
    }
}

fn add_record_launch_kernel(
    builder: &mut ExecutionPlanBuilder,
    buffer_idx: usize,
    var: Arc<UOp>,
    global_expr: Arc<UOp>,
    counters: RecordLaunchCounters,
    initial_val: i64,
) {
    builder.add_kernel(PreparedKernel {
        id: 8500,
        ast: UOp::sink(vec![var.clone(), global_expr.clone()]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(RecordLaunchProgram {
                calls: counters.calls,
                global_x: counters.global_x,
                first_val: counters.first_val,
            }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "record_launch".to_string(),
            var_names: vec![match var.op() {
                svod_ir::Op::DefineVar { name, .. } => name.clone(),
                _ => "N".to_string(),
            }],
            globals: vec![0],
            outs: vec![0],
            ins: Vec::new(),
            global_size: [global_expr, UOp::index_const(1), UOp::index_const(1)],
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![buffer_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: vec![initial_val],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
}

#[test]
fn test_execute_mixed_ops_compiled_copy_view_in_order() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut mid = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut copy_dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let input_data = [1.0f32, 2.0, 3.0, 4.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    mid.copyin(zero_bytes).expect("mid init");
    copy_dst.copyin(zero_bytes).expect("copy_dst init");

    let byte_offset = std::mem::size_of::<f32>();
    let byte_size = 3 * std::mem::size_of::<f32>();
    let view = copy_dst.view(byte_offset, byte_size).expect("create output view");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);

    let src_idx = builder.add_buffer(10, src);
    let mid_idx = builder.add_buffer(11, mid);
    let copy_idx = builder.add_buffer(12, copy_dst);
    let out_idx = builder.add_buffer(13, view);

    let prepared_kernel = PreparedKernel {
        id: 1,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![mid_idx, src_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    };
    builder.add_kernel(prepared_kernel);

    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 2,
        buffer_indices: vec![copy_idx, mid_idx],
        dependencies: vec![1],
    }));

    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute mixed op plan");

    assert_eq!(calls.load(Ordering::Relaxed), 1, "compiled op should run exactly once");

    let mut output_data = vec![0.0f32; 3];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("output copyout");
    assert_eq!(output_data, vec![2.0, 3.0, 4.0]);
}

#[test]
fn test_execute_mixed_ops_respects_dependencies_not_insertion_order() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut mid = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut out = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let input_data = [9.0f32, 8.0, 7.0, 6.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    mid.copyin(zero_bytes).expect("mid init");
    out.copyin(zero_bytes).expect("out init");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);

    let src_idx = builder.add_buffer(300, src);
    let mid_idx = builder.add_buffer(301, mid);
    let out_idx = builder.add_buffer(302, out);

    // Insert compiled kernel first, but make it depend on copy op id=2.
    // Mixed-op execution must honor deps and run copy before this kernel.
    builder.add_kernel(PreparedKernel {
        id: 3,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32_out".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![out_idx, mid_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: vec![2],
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });

    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 2,
        buffer_indices: vec![mid_idx, src_idx],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute dependency-ordered mixed ops");

    assert_eq!(calls.load(Ordering::Relaxed), 1, "compiled op should run exactly once");

    let mut output_data = vec![0.0f32; 4];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("output copyout");
    assert_eq!(output_data, input_data);
}

#[test]
fn test_execute_mixed_ops_missing_dependency_errors() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(400, dst);
    let src_idx = builder.add_buffer(401, src);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 10,
        buffer_indices: vec![dst_idx, src_idx],
        dependencies: vec![999],
    }));
    builder.set_output_buffer(dst_idx);

    let err = builder.build().expect_err("missing dependency should fail during build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("unknown op id"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_mixed_ops_cycle_errors() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let a = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let b = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let a_idx = builder.add_buffer(500, a);
    let b_idx = builder.add_buffer(501, b);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 1,
        buffer_indices: vec![a_idx, b_idx],
        dependencies: vec![2],
    }));
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 2,
        buffer_indices: vec![b_idx, a_idx],
        dependencies: vec![1],
    }));
    builder.set_output_buffer(a_idx);

    let err = builder.build().expect_err("cyclic deps should fail during build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("cycle detected"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_mixed_ops_allows_duplicate_ids_in_expanded_schedule_order() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mid = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let out = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let input_data = [3.0f32, 1.0, 4.0, 1.0];
    let input_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(input_data.as_ptr() as *const u8, input_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(input_bytes).expect("src copyin");

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let src_idx = builder.add_buffer(800, src);
    let mid_idx = builder.add_buffer(801, mid);
    let out_idx = builder.add_buffer(802, out);

    // Expanded schedules can produce repeated op ids for per-iteration items.
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 42,
        buffer_indices: vec![mid_idx, src_idx],
        dependencies: Vec::new(),
    }));
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 42,
        buffer_indices: vec![out_idx, mid_idx],
        dependencies: vec![42],
    }));
    builder.set_output_buffer(out_idx);

    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute duplicate-id schedule");

    let mut output_data = vec![0.0f32; 4];
    let output_bytes: &mut [u8] = unsafe {
        std::slice::from_raw_parts_mut(
            output_data.as_mut_ptr() as *mut u8,
            output_data.len() * std::mem::size_of::<f32>(),
        )
    };
    plan.output_buffer().expect("plan has output").copyout(output_bytes).expect("output copyout");
    assert_eq!(output_data, input_data);
}

#[test]
fn test_execute_copy_invalid_indices_errors() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(600, dst);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 55,
        buffer_indices: vec![dst_idx, dst_idx + 1],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("invalid copy indices should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_build_compiled_program_invalid_buffer_indices_errors() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = {
        let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
        dst.ensure_allocated().expect("dst allocation");
        dst
    };

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(860, dst);
    builder.add_kernel(PreparedKernel {
        id: 861,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: Arc::new(AtomicUsize::new(0)) }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "invalid_indices".to_string(),
            var_names: Vec::new(),
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, dst_idx + 1],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: Vec::new(),
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let err = builder.build().expect_err("invalid compiled-program buffer indices should fail during build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("buffer index out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_custom_function_invalid_indices_errors() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(880, dst);
    builder.add_op(PreparedOp::CustomFunction(PreparedCustomFunction {
        id: 881,
        kind: CustomFunctionKind::EncDec,
        attrs: smallvec::smallvec![],
        buffer_indices: vec![dst_idx, dst_idx + 1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        runtime_vars: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let plan = builder.build().expect("build plan");
    let err = plan.execute().expect_err("invalid custom function indices should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("buffer index out of range"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_with_vars_does_not_override_fixedvars() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(zero_bytes).expect("src init");
    dst.copyin(zero_bytes).expect("dst init");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(900, dst);
    let src_idx = builder.add_buffer(901, src);
    builder.add_kernel(PreparedKernel {
        id: 900,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32_fixedvars".to_string(),
            var_names: vec!["N".to_string()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, src_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: vec![7],
        fixedvars: HashMap::from([(String::from("N"), 7)]),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("N", 42)]).expect("execute with vars");

    let kernels = plan.prepared_kernels();
    assert_eq!(kernels[0].vals.as_slice(), &[7], "fixedvars should win over execute_with_vars overrides");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "kernel should execute exactly once");
}

#[test]
fn test_execute_with_vars_updates_non_fixed_vars() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let mut src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let mut dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    let zero_data = [0.0f32; 4];
    let zero_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(zero_data.as_ptr() as *const u8, zero_data.len() * std::mem::size_of::<f32>())
    };
    src.copyin(zero_bytes).expect("src init");
    dst.copyin(zero_bytes).expect("dst init");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(910, dst);
    let src_idx = builder.add_buffer(911, src);
    builder.add_kernel(PreparedKernel {
        id: 910,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "copy4f32_dynamicvars".to_string(),
            var_names: vec!["N".to_string()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, src_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: vec![1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("N", 42)]).expect("execute with vars");

    let kernels = plan.prepared_kernels();
    assert_eq!(kernels[0].vals.as_slice(), &[42], "execute_with_vars should update non-fixed variable values");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "kernel should execute exactly once");
}

#[test]
fn test_execute_with_vars_updates_symbolic_global_size_without_recompile() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(0));
    let counters =
        RecordLaunchCounters { calls: calls.clone(), global_x: global_x.clone(), first_val: first_val.clone() };
    let n = UOp::define_var("N".to_string(), 1, 8);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8500, dst);
    add_record_launch_kernel(&mut builder, dst_idx, n.clone(), n, counters, 1);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("N", 5)]).expect("execute with dynamic launch size");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(global_x.load(Ordering::SeqCst), 5);
    assert_eq!(first_val.load(Ordering::SeqCst), 5);
}

#[test]
fn test_execute_with_vars_rejects_out_of_bounds_launch_var_before_dispatch() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(0));
    let counters = RecordLaunchCounters { calls: calls.clone(), global_x, first_val };
    let n = UOp::define_var("N".to_string(), 1, 4);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8510, dst);
    add_record_launch_kernel(&mut builder, dst_idx, n.clone(), n, counters, 1);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    let err = plan.execute_with_vars(&[("N", 5)]).expect_err("out-of-bounds launch var should fail");
    // The bound is enforced at launch-dim resolution (device side), surfaced as
    // `Exec` carrying the underlying `svod_device` error as its source.
    match err {
        crate::error::Error::Exec { source, .. } => {
            assert!(source.to_string().contains("outside bounds"), "unexpected error: {source}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
    assert_eq!(calls.load(Ordering::SeqCst), 0, "kernel must not dispatch after launch-var bounds failure");
}

#[test]
fn test_execute_with_vars_profiled_updates_symbolic_global_size() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(0));
    let counters = RecordLaunchCounters { calls: calls.clone(), global_x: global_x.clone(), first_val };
    let n = UOp::define_var("N".to_string(), 1, 8);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8520, dst);
    add_record_launch_kernel(&mut builder, dst_idx, n.clone(), n, counters, 1);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    let profiles = plan.execute_with_vars_profiled(&[("N", 6)]).expect("execute profiled dynamic launch size");

    assert_eq!(profiles.len(), 1);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(global_x.load(Ordering::SeqCst), 6);
}

#[test]
fn test_execute_with_vars_does_not_override_core_id_runtime_var() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let global_x = Arc::new(AtomicUsize::new(0));
    let first_val = Arc::new(AtomicUsize::new(usize::MAX));
    let counters = RecordLaunchCounters { calls: calls.clone(), global_x, first_val: first_val.clone() };
    let core_id = UOp::define_var("core_id".to_string(), 0, 3);

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(8530, dst);
    add_record_launch_kernel(&mut builder, dst_idx, core_id, UOp::index_const(1), counters, 0);
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    plan.execute_with_vars(&[("core_id", 2)]).expect("execute with ignored core_id override");

    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(first_val.load(Ordering::SeqCst), 0, "core_id is a runtime var and must not be user-overridden");
}

#[test]
fn test_compute_execution_levels_duplicate_ids_is_deterministic() {
    let ops = vec![
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![0, 1], dependencies: vec![] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 9, buffer_indices: vec![2, 3], dependencies: vec![42] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![4, 5], dependencies: vec![9] }),
    ];

    let order = compute_mixed_op_order(&ops).expect("dependency order");
    let levels = compute_execution_levels(&ops).expect("dependency levels");
    assert_eq!(order, vec![0, 1, 2]);
    assert_eq!(levels, vec![vec![0], vec![1], vec![2]]);
}

#[test]
fn test_instance_dependencies_target_exact_duplicate_id_instance() {
    let ops = vec![
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![0, 1], dependencies: vec![] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 9, buffer_indices: vec![2, 3], dependencies: vec![] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![4, 5], dependencies: vec![9] }),
        PreparedOp::BufferCopy(PreparedCopy { id: 77, buffer_indices: vec![6, 7], dependencies: vec![] }),
    ];
    let instance_deps = vec![vec![], vec![], vec![], vec![0]];

    let levels = compute_execution_levels_with_instance_dependencies(&ops, &instance_deps).expect("dependency levels");
    assert_eq!(levels, vec![vec![0, 1], vec![2, 3]]);
}

#[test]
fn test_instance_dependencies_reject_unknown_op_index() {
    let ops = vec![PreparedOp::BufferCopy(PreparedCopy { id: 42, buffer_indices: vec![0, 1], dependencies: vec![] })];
    let instance_deps = vec![vec![1]];

    let err = compute_execution_levels_with_instance_dependencies(&ops, &instance_deps)
        .expect_err("unknown op-index dependency should fail");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("unknown op index"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_execute_with_vars_profiled_updates_non_fixed_vars() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");

    let src = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let dst = Buffer::new(alloc, DType::Float32, vec![4], Default::default());
    src.ensure_allocated().expect("allocate src");
    dst.ensure_allocated().expect("allocate dst");

    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1300, dst);
    let src_idx = builder.add_buffer(1301, src);
    builder.add_kernel(PreparedKernel {
        id: 1300,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(Copy4F32Program { calls: calls.clone() }),
            device: "CPU".to_string(),
            code: String::new(),
            entry_point: "profiled_var_update".to_string(),
            var_names: vec!["N".to_string()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![dst_idx, src_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: vec![1],
        fixedvars: HashMap::new(),
        dependencies: Vec::new(),
        buffer_ptrs: Vec::new(),
        buffer_ids: Vec::new(),
        runtime_vars: Vec::new(),
    });
    builder.set_output_buffer(dst_idx);

    let mut plan = builder.build().expect("build plan");
    let profiles = plan.execute_with_vars_profiled(&[("N", 42)]).expect("execute with vars profiled");

    assert_eq!(profiles.len(), 1, "profile should include the compiled kernel");
    let kernels = plan.prepared_kernels();
    assert_eq!(kernels[0].vals.as_slice(), &[42], "execute_with_vars_profiled should update non-fixed variables");
    assert_eq!(calls.load(Ordering::Relaxed), 1, "kernel should execute exactly once");
}

/// Pins that `ExecutionPlan::execute()` walks `op_levels` (level-by-level)
/// rather than a flat topological linearization. Regression guard for the
/// fix shipped in commit fcbb725 (Step 6 of the connector refactor): QR
/// decomposition and other iterative CPU kernels are sensitive to within-
/// level ordering, and a future refactor that switches back to flat
/// `op_order` would silently regress them.
///
/// Construction: 4 ops, deps `A → C`, `B → D`. Level structure is
/// `[[A,B], [C,D]]`. Any valid topological order respects A<C and B<D;
/// only a level-by-level walk guarantees `{A,B}` both before `{C,D}` both.
#[derive(Debug)]
struct OrderRecorderProgram {
    id: u64,
    sink: Arc<parking_lot::Mutex<Vec<u64>>>,
}

impl Program for OrderRecorderProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.sink.lock().push(self.id);
        Ok(())
    }

    fn name(&self) -> &str {
        "order_recorder"
    }
}

#[test]
fn test_execute_walks_op_levels_in_level_order() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let sink = Arc::new(parking_lot::Mutex::new(Vec::<u64>::new()));

    fn record_kernel(id: u64, sink: &Arc<parking_lot::Mutex<Vec<u64>>>, deps: Vec<u64>) -> PreparedKernel {
        PreparedKernel {
            id,
            ast: UOp::sink(vec![]),
            kernel: Arc::new(CachedKernel {
                program: Box::new(OrderRecorderProgram { id, sink: Arc::clone(sink) }),
                device: "CPU".to_string(),
                code: String::new(),
                entry_point: format!("op{id}"),
                var_names: Vec::new(),
                globals: vec![0],
                outs: vec![0],
                ins: Vec::new(),
                global_size: default_launch_size(),
                local_size: Some(default_launch_size()),
            }),
            device: DeviceSpec::Cpu,
            buffer_indices: vec![0],
            output_indices: vec![0],
            input_indices: Vec::new(),
            vals: Vec::new(),
            fixedvars: HashMap::new(),
            dependencies: deps,
            buffer_ptrs: Vec::new(),
            buffer_ids: Vec::new(),
            runtime_vars: Vec::new(),
        }
    }

    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let out = Buffer::new(alloc, DType::Float32, vec![1], Default::default());
    out.ensure_allocated().expect("out alloc");
    let out_idx = builder.add_buffer(900, out);
    builder.set_output_buffer(out_idx);
    // Level 0: ids 1, 2 (no deps).
    // Level 1: ids 3 (deps [1]), 4 (deps [2]).
    builder.add_op(PreparedOp::CompiledProgram(record_kernel(1, &sink, Vec::new())));
    builder.add_op(PreparedOp::CompiledProgram(record_kernel(2, &sink, Vec::new())));
    builder.add_op(PreparedOp::CompiledProgram(record_kernel(3, &sink, vec![1])));
    builder.add_op(PreparedOp::CompiledProgram(record_kernel(4, &sink, vec![2])));
    let plan = builder.build().expect("build plan");
    plan.execute().expect("execute");

    let order = sink.lock().clone();
    assert_eq!(order.len(), 4, "expected 4 ops to run, got {order:?}");
    // Level boundary: every level-0 id (1, 2) must precede every level-1 id (3, 4).
    let pos = |id: u64| order.iter().position(|&x| x == id).expect("id not recorded");
    let last_level0 = pos(1).max(pos(2));
    let first_level1 = pos(3).min(pos(4));
    assert!(
        last_level0 < first_level1,
        "level-1 op ran before a level-0 op (order={order:?}); execute() must walk op_levels, not flat op_order"
    );
}

fn hcq_access(storage: u64) -> BufferAccess {
    BufferAccess { storage: BufferId(storage), owner: DeviceSpec::Cpu, start: 0, end: 64 }
}

fn hcq_op(operation: usize, queue: svod_device::hcq::QueueKind, reads: &[u64], writes: &[u64]) -> HcqPreparedOperation {
    HcqPreparedOperation {
        operation,
        device: DeviceSpec::Cpu,
        queue,
        reads: reads.iter().map(|&id| hcq_access(id)).collect(),
        writes: writes.iter().map(|&id| hcq_access(id)).collect(),
        is_copy: matches!(queue, svod_device::hcq::QueueKind::Copy(_)),
    }
}

fn operation_submission(plan: &HcqLinkedPlan, operation: usize) -> &svod_device::hcq::LaneSubmission {
    plan.semantic
        .lanes()
        .iter()
        .find(|submission| submission.commands.iter().any(|command| command.operation == operation))
        .expect("operation submission")
}

fn hcq_plan(operations: &[HcqPreparedOperation]) -> HcqLinkedPlan {
    HcqLinkedPlan::capture(operations.to_vec()).unwrap()
}

#[test]
fn hcq_independent_compute_and_copy_can_overlap() {
    use svod_device::hcq::QueueKind;
    let plan = hcq_plan(&[hcq_op(0, QueueKind::Compute(0), &[1], &[2]), hcq_op(1, QueueKind::Copy(0), &[3], &[4])]);
    assert!(operation_submission(&plan, 1).waits.is_empty());
}

#[test]
fn hcq_raw_dependency_waits_for_cross_queue_writer() {
    use svod_device::hcq::QueueKind;
    let plan = hcq_plan(&[hcq_op(0, QueueKind::Compute(0), &[], &[1]), hcq_op(1, QueueKind::Copy(0), &[1], &[2])]);
    assert_eq!(operation_submission(&plan, 1).waits[0].lane.queue, QueueKind::Compute(0));
}

#[test]
fn hcq_war_dependency_waits_for_cross_queue_reader() {
    use svod_device::hcq::QueueKind;
    let plan = hcq_plan(&[hcq_op(0, QueueKind::Compute(0), &[1], &[]), hcq_op(1, QueueKind::Copy(0), &[], &[1])]);
    assert_eq!(operation_submission(&plan, 1).waits[0].lane.queue, QueueKind::Compute(0));
}

#[test]
fn hcq_waw_dependency_waits_for_cross_queue_writer() {
    use svod_device::hcq::QueueKind;
    let plan = hcq_plan(&[hcq_op(0, QueueKind::Copy(0), &[], &[1]), hcq_op(1, QueueKind::Compute(0), &[], &[1])]);
    assert_eq!(operation_submission(&plan, 1).waits[0].lane.queue, QueueKind::Copy(0));
}

#[test]
fn hcq_compute_to_copy_and_copy_to_compute_use_queue_timelines() {
    use svod_device::hcq::QueueKind;
    let plan = hcq_plan(&[
        hcq_op(0, QueueKind::Compute(0), &[], &[1]),
        hcq_op(1, QueueKind::Copy(0), &[1], &[2]),
        hcq_op(2, QueueKind::Compute(0), &[2], &[3]),
    ]);
    assert_eq!(operation_submission(&plan, 1).waits[0].lane.queue, QueueKind::Compute(0));
    assert_eq!(operation_submission(&plan, 2).waits[0].lane.queue, QueueKind::Copy(0));
}

#[derive(Debug)]
struct ReplayRecorderProgram {
    calls: Arc<parking_lot::Mutex<Vec<(usize, usize, i64)>>>,
}

impl Program for ReplayRecorderProgram {
    unsafe fn execute(
        &self,
        buffers: &[*mut u8],
        vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.lock().push((buffers[0] as usize, buffers[1] as usize, vals[0]));
        unsafe { std::ptr::copy_nonoverlapping(buffers[1], buffers[0], 4) };
        Ok(())
    }

    fn name(&self) -> &str {
        "replay_recorder"
    }
}

#[test]
fn repeated_normal_execution_repatches_vars_buffers_and_mixed_copy_plan() {
    let alloc = svod_device::registry::cpu().unwrap();
    let mut source = Buffer::new(alloc.clone(), DType::UInt8, vec![4], Default::default());
    source.copyin(&[1, 2, 3, 4]).unwrap();
    let middle = Buffer::new(alloc.clone(), DType::UInt8, vec![4], Default::default());
    let output = Buffer::new(alloc.clone(), DType::UInt8, vec![4], Default::default());
    let calls = Arc::new(parking_lot::Mutex::new(Vec::new()));
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let source_idx = builder.add_buffer(20_001, source);
    let middle_idx = builder.add_buffer(20_002, middle);
    let output_idx = builder.add_buffer(20_003, output);
    builder.add_kernel(PreparedKernel {
        id: 20_010,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(ReplayRecorderProgram { calls: Arc::clone(&calls) }),
            device: "CPU".into(),
            code: String::new(),
            entry_point: "replay_recorder".into(),
            var_names: vec!["N".into()],
            globals: vec![0, 1],
            outs: vec![0],
            ins: vec![1],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: DeviceSpec::Cpu,
        buffer_indices: vec![middle_idx, source_idx],
        output_indices: vec![0],
        input_indices: Vec::new(),
        vals: vec![1],
        fixedvars: HashMap::new(),
        dependencies: vec![],
        buffer_ptrs: vec![],
        buffer_ids: vec![],
        runtime_vars: vec![],
    });
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 20_011,
        buffer_indices: vec![output_idx, middle_idx],
        dependencies: vec![20_010],
    }));
    builder.set_output_buffer(output_idx);
    let mut plan = builder.build().unwrap();
    let static_lanes = plan.hcq_linked.get().unwrap().semantic.lanes().as_ptr();

    plan.execute_with_vars(&[("N", 3)]).unwrap();
    let mut first = [0; 4];
    plan.output_buffer().unwrap().copyout(&mut first).unwrap();
    assert_eq!(first, [1, 2, 3, 4]);

    let mut replacement = Buffer::new(alloc, DType::UInt8, vec![4], Default::default());
    replacement.copyin(&[9, 8, 7, 6]).unwrap();
    *plan.buffer_at_mut(source_idx).unwrap() = replacement;
    plan.execute_with_vars(&[("N", 7)]).unwrap();
    let mut second = [0; 4];
    plan.output_buffer().unwrap().copyout(&mut second).unwrap();
    assert_eq!(second, [9, 8, 7, 6]);

    let calls = calls.lock();
    assert_eq!(calls.len(), 2);
    assert_eq!((calls[0].2, calls[1].2), (3, 7));
    assert_ne!(calls[0].1, calls[1].1, "replacement buffer address must be patched on replay");
    assert_eq!(plan.hcq_linked.get().unwrap().semantic.lanes().as_ptr(), static_lanes);
}

#[derive(Debug)]
struct TaggedCpuAllocator(DeviceSpec);

impl Allocator for TaggedCpuAllocator {
    fn _alloc(&self, size: usize, options: &BufferSpec, zero: bool) -> svod_device::Result<RawBuffer> {
        CpuAllocator._alloc(size, options, zero)
    }

    fn name(&self) -> &str {
        "tagged-cpu"
    }

    fn _copyin(&self, dest: &RawBuffer, dest_off: usize, src: &[u8]) -> svod_device::Result<()> {
        CpuAllocator._copyin(dest, dest_off, src)
    }

    fn _copyout(&self, dest: &mut [u8], src: &RawBuffer, src_off: usize) -> svod_device::Result<()> {
        CpuAllocator._copyout(dest, src, src_off)
    }

    fn device_spec(&self) -> DeviceSpec {
        self.0.clone()
    }
}

#[derive(Debug)]
struct NativeReplayProgram {
    replays: Arc<AtomicUsize>,
    fail: bool,
}

#[derive(Debug)]
struct RejectDispatchProgram {
    calls: Arc<AtomicUsize>,
}

impl Program for RejectDispatchProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Err(svod_device::Error::Runtime { message: "semantic fallback reached".into() })
    }

    fn name(&self) -> &str {
        "reject_dispatch"
    }
}

impl Program for NativeReplayProgram {
    unsafe fn execute(
        &self,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _wait: bool,
    ) -> svod_device::Result<()> {
        panic!("native replay test must not use per-operation dispatch")
    }

    fn name(&self) -> &str {
        "native_replay_recorder"
    }

    fn new_exec_context(&self) -> svod_device::Result<Option<Box<dyn PlanContext>>> {
        Ok(Some(Box::new(NativeReplayContext { replays: Arc::clone(&self.replays), fail: self.fail })))
    }
}

#[derive(Debug)]
struct NativeReplayContext {
    replays: Arc<AtomicUsize>,
    fail: bool,
}

impl PlanContext for NativeReplayContext {
    unsafe fn dispatch(
        &self,
        _program: &dyn Program,
        _buffers: &[*mut u8],
        _vals: &[i64],
        _global_size: Option<[usize; 3]>,
        _local_size: Option<[usize; 3]>,
        _profile: bool,
    ) -> svod_device::Result<Option<Arc<dyn svod_device::DispatchTimestamps>>> {
        panic!("native replay test must not dispatch individual kernels")
    }

    fn replay_linked_plan(
        &self,
        _plan: &svod_device::hcq::SemanticLinkedPlan,
        _calls: &[PlanCall<'_>],
    ) -> svod_device::Result<NativeReplayOutcome> {
        self.replays.fetch_add(1, Ordering::SeqCst);
        if self.fail {
            return Err(svod_device::Error::Runtime { message: "native submit rejected".into() });
        }
        Ok(NativeReplayOutcome::Executed)
    }

    fn synchronize(&self) -> svod_device::Result<()> {
        Ok(())
    }
}

fn tagged_buffer(device: DeviceSpec) -> Buffer {
    Buffer::new(Arc::new(TaggedCpuAllocator(device)), DType::UInt8, vec![4], Default::default())
}

#[test]
fn staged_copy_uses_fresh_host_storage_each_epoch() {
    use svod_device::hcq::CopyLeg;

    let mut source = tagged_buffer(DeviceSpec::Amd { device_id: 0 });
    source.copyin(&[1, 2, 3, 4]).unwrap();
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst = builder.add_buffer(20_101, tagged_buffer(DeviceSpec::Cpu));
    let src = builder.add_buffer(20_102, source);
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 20_103,
        buffer_indices: vec![dst, src],
        dependencies: vec![],
    }));
    builder.set_output_buffer(dst);
    let mut plan = builder.build().unwrap();
    let legs = plan
        .hcq_linked
        .get()
        .unwrap()
        .semantic
        .lanes()
        .iter()
        .map(|lane| lane.commands[0].copy_leg.unwrap())
        .collect::<Vec<_>>();
    assert_eq!(legs, [CopyLeg::ToHost, CopyLeg::FromHost]);

    plan.execute().unwrap();
    let mut first = [0; 4];
    plan.output_buffer().unwrap().copyout(&mut first).unwrap();
    assert_eq!(first, [1, 2, 3, 4]);

    plan.buffer_at_mut(src).unwrap().copyin(&[9, 8, 7, 6]).unwrap();
    plan.execute().unwrap();
    let mut second = [0; 4];
    plan.output_buffer().unwrap().copyout(&mut second).unwrap();
    assert_eq!(second, [9, 8, 7, 6]);
}

#[test]
fn graph_replay_rejects_forged_amd_allocation_ownership() {
    let calls = Arc::new(AtomicUsize::new(0));
    let replays = Arc::new(AtomicUsize::new(0));
    let amd = DeviceSpec::Amd { device_id: 0 };
    let mut builder = ExecutionPlanBuilder::new(amd.clone());
    let buffer_idx = builder.add_buffer(20_001, tagged_buffer(amd.clone()));
    builder.add_kernel(PreparedKernel {
        id: 20_010,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(RejectDispatchProgram { calls: Arc::clone(&calls) }),
            device: "CPU".into(),
            code: String::new(),
            entry_point: "graph_endpoint_guard".into(),
            var_names: vec![],
            globals: vec![0],
            outs: vec![0],
            ins: vec![],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: amd.clone(),
        buffer_indices: vec![buffer_idx],
        output_indices: vec![0],
        input_indices: vec![],
        vals: vec![],
        fixedvars: HashMap::new(),
        dependencies: vec![],
        buffer_ptrs: vec![],
        buffer_ids: vec![],
        runtime_vars: vec![],
    });
    builder.set_output_buffer(buffer_idx);
    let mut plan = builder.build().unwrap();
    plan.graph
        .set(Some(Box::new(ProfileReplayGraph {
            replays: Arc::clone(&replays),
            timestamp_drops: Arc::new(AtomicUsize::new(0)),
        })))
        .map_err(|_| ())
        .unwrap();

    *plan.buffer_at_mut(buffer_idx).unwrap() = tagged_buffer(amd);
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::IncompatibleProgramAllocation {
            operation: 20_010,
            argument: 0,
            expected: DeviceSpec::Amd { device_id: 0 },
        })
    );
    let error = plan.execute().expect_err("forged AMD ownership must use semantic fallback");
    assert!(error.to_string().contains("semantic fallback reached"));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(replays.load(Ordering::SeqCst), 0, "forged endpoint reached graph backend");
}

fn native_copy_plan_with_source(
    source_device: DeviceSpec,
    fail_native: bool,
) -> (ExecutionPlan, Arc<AtomicUsize>, usize, usize, usize) {
    let owner = DeviceSpec::Cpu;
    let replays = Arc::new(AtomicUsize::new(0));
    let mut builder = ExecutionPlanBuilder::new(owner.clone());
    let kernel_idx = builder.add_buffer(21_001, tagged_buffer(owner.clone()));
    let dst_idx = builder.add_buffer(21_002, tagged_buffer(owner.clone()));
    let src_idx = builder.add_buffer(21_003, tagged_buffer(source_device));
    builder.add_kernel(PreparedKernel {
        id: 21_010,
        ast: UOp::sink(vec![]),
        kernel: Arc::new(CachedKernel {
            program: Box::new(NativeReplayProgram { replays: Arc::clone(&replays), fail: fail_native }),
            device: "AMD:0".into(),
            code: String::new(),
            entry_point: "native_replay_recorder".into(),
            var_names: vec![],
            globals: vec![0],
            outs: vec![0],
            ins: vec![],
            global_size: default_launch_size(),
            local_size: Some(default_launch_size()),
        }),
        device: owner,
        buffer_indices: vec![kernel_idx],
        output_indices: vec![0],
        input_indices: vec![],
        vals: vec![],
        fixedvars: HashMap::new(),
        dependencies: vec![],
        buffer_ptrs: vec![],
        buffer_ids: vec![],
        runtime_vars: vec![],
    });
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 21_011,
        buffer_indices: vec![dst_idx, src_idx],
        dependencies: vec![21_010],
    }));
    builder.set_output_buffer(dst_idx);
    (builder.build().unwrap(), replays, kernel_idx, dst_idx, src_idx)
}

fn native_copy_plan() -> (ExecutionPlan, Arc<AtomicUsize>, usize, usize, usize) {
    native_copy_plan_with_source(DeviceSpec::Cpu, false)
}

#[test]
fn native_replay_rejects_staged_semantic_copy() {
    let (plan, replays, _, _, _) = native_copy_plan_with_source(DeviceSpec::Amd { device_id: 0 }, false);
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::StagedCopy { operation: 1 })
    );
    assert_eq!(replays.load(Ordering::SeqCst), 0);
}

#[test]
fn native_replay_requires_copy_endpoints_on_context_device() {
    let (mut plan, replays, _kernel_idx, dst_idx, src_idx) = native_copy_plan();
    assert_eq!(plan.replay_native_linked_plan().unwrap(), NativeReplayOutcome::Executed);
    assert_eq!(replays.load(Ordering::SeqCst), 1);

    *plan.buffer_at_mut(dst_idx).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id: 1 });
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::ForeignCopyEndpoint {
            operation: 21_011,
            endpoint: CopyEndpoint::Destination,
            expected: DeviceSpec::Cpu,
            actual: DeviceSpec::Amd { device_id: 1 },
        })
    );
    assert_eq!(replays.load(Ordering::SeqCst), 1, "foreign GPU endpoint reached native context");

    *plan.buffer_at_mut(dst_idx).unwrap() = tagged_buffer(DeviceSpec::Cpu);
    *plan.buffer_at_mut(src_idx).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id: 0 });
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::ForeignCopyEndpoint {
            operation: 21_011,
            endpoint: CopyEndpoint::Source,
            expected: DeviceSpec::Cpu,
            actual: DeviceSpec::Amd { device_id: 0 },
        })
    );
    assert_eq!(replays.load(Ordering::SeqCst), 1, "CPU endpoint reached native context");
}

#[test]
fn native_replay_requires_program_endpoints_on_context_device() {
    let (mut plan, replays, kernel_idx, _dst_idx, _src_idx) = native_copy_plan();
    *plan.buffer_at_mut(kernel_idx).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id: 1 });
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::ForeignProgramEndpoint {
            operation: 21_010,
            argument: 0,
            expected: DeviceSpec::Cpu,
            actual: DeviceSpec::Amd { device_id: 1 },
        })
    );
    assert_eq!(replays.load(Ordering::SeqCst), 0, "foreign PROGRAM endpoint reached native context");

    *plan.buffer_at_mut(kernel_idx).unwrap() = tagged_buffer(DeviceSpec::Amd { device_id: 0 });
    assert_eq!(
        plan.replay_native_linked_plan().unwrap(),
        NativeReplayOutcome::Declined(NativeReplayDecline::ForeignProgramEndpoint {
            operation: 21_010,
            argument: 0,
            expected: DeviceSpec::Cpu,
            actual: DeviceSpec::Amd { device_id: 0 },
        })
    );
    assert_eq!(replays.load(Ordering::SeqCst), 0, "CPU PROGRAM endpoint reached native context");
}

#[test]
fn concurrent_execution_plans_keep_linked_context_timelines_isolated() {
    fn copy_plan(seed: u8) -> ExecutionPlan {
        let alloc = svod_device::registry::cpu().unwrap();
        let mut source = Buffer::new(alloc.clone(), DType::UInt8, vec![4], Default::default());
        source.copyin(&[seed; 4]).unwrap();
        let output = Buffer::new(alloc, DType::UInt8, vec![4], Default::default());
        let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
        let output_idx = builder.add_buffer(seed as u64 + 30_000, output);
        let source_idx = builder.add_buffer(seed as u64 + 31_000, source);
        builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
            id: seed as u64 + 32_000,
            buffer_indices: vec![output_idx, source_idx],
            dependencies: vec![],
        }));
        builder.set_output_buffer(output_idx);
        builder.build().unwrap()
    }

    let left = Arc::new(copy_plan(4));
    let right = Arc::new(copy_plan(7));
    let left_signal = left.hcq_linked.get().unwrap().semantic.bindings()[0].point.signal_address;
    let right_signal = right.hcq_linked.get().unwrap().semantic.bindings()[0].point.signal_address;
    assert_ne!(left_signal, right_signal);
    let a = Arc::clone(&left);
    let b = Arc::clone(&right);
    std::thread::scope(|scope| {
        scope.spawn(move || {
            for _ in 0..20 {
                a.execute().unwrap();
            }
        });
        scope.spawn(move || {
            for _ in 0..20 {
                b.execute().unwrap();
            }
        });
    });
    let mut left_out = [0; 4];
    let mut right_out = [0; 4];
    left.output_buffer().unwrap().copyout(&mut left_out).unwrap();
    right.output_buffer().unwrap().copyout(&mut right_out).unwrap();
    assert_eq!(left_out, [4; 4]);
    assert_eq!(right_out, [7; 4]);
}

#[test]
fn hcq_same_queue_dependencies_are_fifo_elided() {
    use svod_device::hcq::QueueKind;
    let plan = hcq_plan(&[hcq_op(0, QueueKind::Compute(0), &[], &[1]), hcq_op(1, QueueKind::Compute(0), &[1], &[2])]);
    assert!(operation_submission(&plan, 1).waits.is_empty());
}

#[test]
fn build_rejects_buffer_copy_without_source_and_destination() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let mut builder = ExecutionPlanBuilder::new(DeviceSpec::Cpu);
    let dst_idx = builder.add_buffer(1, Buffer::new(alloc, DType::Float32, vec![4], Default::default()));
    builder.add_op(PreparedOp::BufferCopy(PreparedCopy {
        id: 77,
        buffer_indices: vec![dst_idx],
        dependencies: Vec::new(),
    }));
    builder.set_output_buffer(dst_idx);

    let err = builder.build().expect_err("one-endpoint copy must not build");
    match err {
        crate::error::Error::Execution { reason } => {
            assert!(reason.contains("requires two buffer indices"), "unexpected error: {reason}");
        }
        other => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn failed_native_replay_poisons_the_plan() {
    let (plan, replays, _, _, _) = native_copy_plan_with_source(DeviceSpec::Cpu, true);

    let first = plan.execute().expect_err("failing native submit must surface");
    assert!(matches!(first, crate::error::Error::Exec { .. }), "{first:?}");
    assert_eq!(replays.load(Ordering::SeqCst), 1);

    let second = plan.execute().expect_err("a failed native submit must not stay retryable");
    assert!(matches!(second, crate::error::Error::PlanPoisoned { .. }), "{second:?}");
    assert_eq!(replays.load(Ordering::SeqCst), 1, "poisoned plan must not resubmit");
}
