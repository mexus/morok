use super::test_support::{MockAmdIface, amd_alloc_or_skip, require_multi_xcc, require_single_xcc};
use crate::amd::AmdAllocator;
use crate::amd::program::*;
use crate::error::Error;
use std::sync::Arc;

/// Serializes the `#[ignore]` PM4-graph probes that toggle the per-device
/// `pm4_graph` flag. The flag lives on the process-global (`DEVICE_CACHE`-backed)
/// `AmdDeviceCore`, so two probes running concurrently would observe each other's
/// writes; holding this lock for each probe's duration makes the save/restore in
/// [`Pm4GraphOverride`] well-defined regardless of `--test-threads`.
static PM4_GRAPH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn global_f32_buffer_abi() -> [crate::device::AbiParamDescriptor; 1] {
    [crate::device::AbiParamDescriptor {
        slot: 0,
        kind: crate::device::AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
        dtype: svod_dtype::DType::Float32,
        name: None,
    }]
}

/// Scoped enable of the per-device `pm4_graph` capture flag: records the previous
/// value and restores it on drop, so a probe's mutation of the shared core flag
/// never leaks into a later test in the same process. Acquire
/// [`PM4_GRAPH_TEST_LOCK`] first so the save/restore window is exclusive.
struct Pm4GraphOverride<'a> {
    core: &'a crate::amd::device::AmdDeviceCore,
    prev: bool,
}

impl<'a> Pm4GraphOverride<'a> {
    fn enable(core: &'a crate::amd::device::AmdDeviceCore) -> Self {
        let prev = core.pm4_graph();
        core.set_pm4_graph(true);
        Self { core, prev }
    }
}

impl Drop for Pm4GraphOverride<'_> {
    fn drop(&mut self) {
        self.core.set_pm4_graph(self.prev);
    }
}

/// Compile a trivial amdgcn kernel via Phase 2, then parse it back and
/// verify the kernel descriptor round-trips. Skipped when host clang
/// lacks AMDGPU target.
#[test]
fn parse_kernel_descriptor_from_compiled_elf() {
    // We can't pull svod-runtime here (dependency would cycle), so we
    // shell out to clang ourselves with the same flags as
    // `runtime::amd::compile`. Lighter than wiring a dev-dep.
    let ir = r#"; ModuleID = 'p6_smoke'
source_filename = "p6_smoke"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @p6_smoke(ptr noalias %buf0) #0 {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}

attributes #0 = { alwaysinline nounwind "no-builtins" "amdgpu-flat-work-group-size"="1,64" "no-trapping-math"="true" }
"#;
    let out = match std::process::Command::new("clang")
        .args([
            "-x",
            "ir",
            "-c",
            "-O2",
            "--target=amdgcn-amd-amdhsa",
            "-mcpu=gfx1100",
            "-mcumode",
            "-nogpulib",
            "-nogpuinc",
            "-Wno-override-module",
            "-",
            "-o",
            "-",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => {
            eprintln!("skipping: clang not available");
            return;
        }
    };
    use std::io::Write;
    let mut out = out;
    out.stdin.take().unwrap().write_all(ir.as_bytes()).unwrap();
    let output = out.wait_with_output().unwrap();
    if !output.status.success() {
        eprintln!("skipping: clang amdgcn compile failed (target may be unavailable)");
        return;
    }
    let bytes = output.stdout;
    let parsed = parse_kernel(&bytes, "p6_smoke").expect("parse");
    // Sanity: kernarg_size is at least one ptr (8 bytes), aligned.
    let kernarg_size = parsed.kd.kernarg_size;
    assert!(kernarg_size >= 8, "kernarg_size {} should hold at least one pointer", kernarg_size);
    // Sanity: descriptor offset is inside the image.
    assert!((parsed.kd_offset as usize) < parsed.image.len());
}

/// Shell out to `clang` to compile amdgcn IR → code object (ELF). Mirrors
/// `runtime::amd::compile` but avoids the dep cycle (cf. the test above).
/// Returns `None` if clang is missing or the target is unavailable.
fn clang_amdgcn(ir: &str, mcpu: &str) -> Option<Vec<u8>> {
    use std::io::Write;
    let child = std::process::Command::new("clang")
        .args(["-x", "ir", "-c", "-O2", "--target=amdgcn-amd-amdhsa"])
        .arg(format!("-mcpu={mcpu}"))
        .args(["-mcumode", "-nogpulib", "-nogpuinc", "-Wno-override-module", "-", "-o", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .ok()?;
    child.stdin.as_ref()?.write_all(ir.as_bytes()).ok()?;
    let out = child.wait_with_output().ok()?;
    if !out.status.success() || out.stdout.len() < 4 || &out.stdout[..4] != b"\x7fELF" {
        return None;
    }
    Some(out.stdout)
}

#[test]
fn mock_program_code_allocation_is_balanced_on_success_and_failure() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @mock_program() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };

    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    let program = AmdProgram::load(device, &allocator, &bytes, "mock_program", &[]).expect("program load");
    assert_eq!((iface.allocation_count(), iface.live_handle_count()), (1, 1));
    drop(program);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (1, 0));

    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    iface.script_alloc(Err(Error::Runtime { message: "scripted code allocation".into() }));
    assert!(AmdProgram::load(device, &allocator, &bytes, "mock_program", &[]).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (0, 0, 0));
}

#[test]
fn mock_program_post_allocation_validation_reclaims_code() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
declare ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
define amdgpu_kernel void @dispatch_ptr_program() #0 {
entry:
  %dispatch = call ptr addrspace(4) @llvm.amdgcn.dispatch.ptr()
  %value = load volatile i8, ptr addrspace(4) %dispatch, align 1
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    assert!(matches!(
        AmdProgram::load(device, &allocator, &bytes, "dispatch_ptr_program", &[]),
        Err(Error::Runtime { message }) if message.contains("ENABLE_SGPR_DISPATCH_PTR")
    ));
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (1, 1, 0));
}

#[test]
fn mock_graph_capture_storage_unwinds_each_post_lane_allocation() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @graph_program() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };

    for fail_at in 6..=8 {
        let iface = Arc::new(MockAmdIface::default());
        let device = iface.device();
        let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
        device.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).expect("signal pool"));
        device.core().set_pm4_graph(true);
        let program = AmdProgram::load(Arc::clone(&device), &allocator, &bytes, "graph_program", &[]).unwrap();
        let allocations_before = iface.allocation_count();
        let frees_before = iface.free_count();
        for _ in 0..fail_at {
            iface.script_alloc(Ok(()));
        }
        iface.script_alloc(Err(Error::Runtime { message: "scripted graph allocation".into() }));
        let kernels = [crate::device::GraphKernel {
            program: &program,
            buffers: Vec::new(),
            vals: Vec::new(),
            global_size: Some([1, 1, 1]),
            local_size: Some([1, 1, 1]),
            deps: Vec::new(),
        }];
        assert!(crate::amd::graph::AmdGraph::capture(&allocator, &kernels).is_err(), "fail_at={fail_at}");
        assert_eq!(iface.allocation_count() - allocations_before, fail_at, "fail_at={fail_at}");
        assert_eq!(iface.free_count() - frees_before, fail_at - 6, "fail_at={fail_at}");
        assert!(iface.free_issues().is_empty());
    }
}

#[test]
fn mock_graph_success_drop_frees_kernarg_and_both_resident_streams_once() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @graph_program() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    device.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).unwrap());
    device.core().set_pm4_graph(true);
    let program = AmdProgram::load(Arc::clone(&device), &allocator, &bytes, "graph_program", &[]).unwrap();
    let kernels = [crate::device::GraphKernel {
        program: &program,
        buffers: Vec::new(),
        vals: Vec::new(),
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
        deps: Vec::new(),
    }];
    let allocations_before = iface.allocation_count();
    let graph = crate::amd::graph::AmdGraph::capture(&allocator, &kernels).unwrap().expect("graph");
    assert_eq!(iface.allocation_count() - allocations_before, 9);
    drop(graph);
    assert_eq!(iface.free_count(), 3);
    assert!(iface.free_issues().is_empty());
}

#[test]
fn mock_graph_replay_skips_repacking_unchanged_kernargs() {
    use crate::device::{AbiParamDescriptor, AbiParamKind};
    use svod_dtype::{AddrSpace, DType};

    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @graph_program(ptr addrspace(1) %out) #0 {
entry:
  store float 0.0, ptr addrspace(1) %out, align 4
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    device.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).unwrap());
    device.core().set_pm4_graph(true);
    let abi = [AbiParamDescriptor {
        slot: 0,
        kind: AbiParamKind::Storage(AddrSpace::Global),
        dtype: DType::Float32,
        name: None,
    }];
    let program = AmdProgram::load(Arc::clone(&device), &allocator, &bytes, "graph_program", &abi).unwrap();
    let kernels = [crate::device::GraphKernel {
        program: &program,
        buffers: vec![0x1000 as *mut u8],
        vals: Vec::new(),
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
        deps: Vec::new(),
    }];
    let graph = crate::amd::graph::AmdGraph::capture_amd(&allocator, &kernels).unwrap().expect("graph");

    assert_eq!(graph.kernarg_pack_probe(&[0x2000], &[]).unwrap(), 1);
    assert_eq!(graph.kernarg_pack_probe(&[0x2000], &[]).unwrap(), 1, "identical arguments must not repack");
    assert_eq!(graph.kernarg_pack_probe(&[0x3000], &[]).unwrap(), 2, "changed arguments repack");
}

#[test]
fn mock_graph_capture_emits_one_memory_barrier_for_the_whole_chain() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @graph_program() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    device.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).unwrap());
    device.core().set_pm4_graph(true);
    let program = AmdProgram::load(Arc::clone(&device), &allocator, &bytes, "graph_program", &[]).unwrap();
    let kernels = std::array::from_fn::<_, 3, _>(|_| crate::device::GraphKernel {
        program: &program,
        buffers: Vec::new(),
        vals: Vec::new(),
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
        deps: Vec::new(),
    });
    let graph = crate::amd::graph::AmdGraph::capture_amd(&allocator, &kernels).unwrap().expect("graph");
    let dwords = graph
        .linked_bytes()
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().unwrap()))
        .collect::<Vec<_>>();

    // One HDP flush + full acquire for the whole graph (tinygrad graph/hcq.py:157)...
    let barrier = crate::amd::sys::pm4::hdp_flush();
    assert_eq!(dwords.windows(barrier.len()).filter(|run| **run == barrier).count(), 1);
    // ...while every dispatch keeps its own CS_PARTIAL_FLUSH.
    let flush = crate::amd::sys::pm4::event_write(
        crate::amd::sys::pm4::CS_PARTIAL_FLUSH,
        crate::amd::sys::pm4::EVENT_INDEX_PARTIAL_FLUSH,
    );
    assert_eq!(dwords.windows(flush.len()).filter(|run| **run == flush).count(), 3);
}

#[test]
fn mock_graph_failed_drain_quarantines_graph_program_and_queue_storage() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @graph_program() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    device.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).unwrap());
    device.core().set_pm4_graph(true);
    let program = AmdProgram::load(Arc::clone(&device), &allocator, &bytes, "graph_program", &[]).unwrap();
    let kernels = [crate::device::GraphKernel {
        program: &program,
        buffers: Vec::new(),
        vals: Vec::new(),
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
        deps: Vec::new(),
    }];
    let graph = crate::amd::graph::AmdGraph::capture(&allocator, &kernels).unwrap().expect("graph");
    graph.replay(&[], &[]).expect("mock publication");
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "mock graph drain", errno: 5 }));
    let allocations = iface.allocation_count();
    drop(graph);
    drop(program);
    assert!(device.is_poisoned());
    assert_eq!(iface.allocation_count(), allocations);
    assert_eq!(iface.free_count(), 0);
    assert_eq!(iface.live_handle_count(), allocations);
}

#[test]
fn mock_linked_plan_capture_and_transactional_publication_own_storage() {
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @linked_program() #0 {
entry:
  ret void
}
attributes #0 = { nounwind "amdgpu-flat-work-group-size"="1,1" }
"#;
    let Some(bytes) = clang_amdgcn(ir, "gfx1100") else {
        eprintln!("skipping: clang amdgcn target unavailable");
        return;
    };
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    device.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).unwrap());
    let program = AmdProgram::load(Arc::clone(&device), &allocator, &bytes, "linked_program", &[]).unwrap();
    let pool = crate::amd::connector::PoolQueue::new_with_resources(Arc::clone(device.core()), &allocator).unwrap();
    let owner = crate::amd::connector::OwnerCtx::new(Arc::clone(device.core()), allocator.clone());
    let semantic = crate::hcq::SemanticLinkedPlan::from_lane_submissions(
        vec![crate::hcq::LaneSubmission {
            lane: crate::hcq::DeviceQueue {
                device: svod_dtype::DeviceSpec::Amd { device_id: 0 },
                queue: crate::hcq::QueueKind::Compute(0),
            },
            waits: Vec::new(),
            commands: vec![crate::hcq::TopologyCommand { operation: 0, copy_leg: None }],
            signal_value: 1,
        }],
        |_| [0x1000, 0x1008],
    )
    .unwrap();
    let calls = [crate::device::PlanCall::Program {
        program: &program,
        buffers: &[],
        vals: &[],
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
    }];

    let allocations = iface.allocation_count();
    iface.script_alloc(Err(Error::Runtime { message: "scripted linked kernarg allocation".into() }));
    assert!(crate::amd::AmdLinkedPlan::capture(&owner, &pool, &semantic, &calls).is_err());
    assert_eq!(iface.allocation_count(), allocations);

    let mut linked =
        crate::amd::AmdLinkedPlan::capture(&owner, &pool, &semantic, &calls).unwrap().expect("linked plan");
    assert_eq!(iface.allocation_count(), allocations + 1);

    iface.script_publication(Err(Error::Runtime { message: "scripted linked after reservation".into() }));
    let failure = linked.replay(&owner, &pool, &calls).unwrap_err();
    assert!(!failure.published);
    assert!(!device.is_poisoned());

    iface.script_publication(Ok(()));
    iface.script_publication(Err(Error::Runtime { message: "scripted linked before doorbell".into() }));
    let failure = linked.replay(&owner, &pool, &calls).unwrap_err();
    assert!(!failure.published);
    assert!(!device.is_poisoned());

    iface.script_publication(Ok(()));
    iface.script_publication(Ok(()));
    iface.script_publication(Err(Error::Runtime { message: "scripted linked after doorbell".into() }));
    let failure = linked.replay(&owner, &pool, &calls).unwrap_err();
    assert!(failure.published);
    assert!(device.is_poisoned());
    drop(linked);
    assert_eq!(iface.free_count(), 0, "linked kernargs and all device storage must be quarantined");
}

/// PHASE-2 GATE (manual hardware probe; `#[ignore]`). Captures a static
/// kernel chain via [`AmdGraph`] and replays it on the device's SHARED queue
/// (single-queue mode — no multi-queue dependency), verifying the kernel runs
/// on each replay. This is the first real validation of graph capture/replay
/// (it was "opt-in until validated" before). A wrong batch/completion would
/// show as a stale buffer or a synchronize timeout.
///
/// Run: cargo test -p svod-device --lib aql_graph_capture_replay -- --ignored --nocapture
#[test]
#[ignore = "manual hardware probe; needs a real gfx942 AMD GPU + clang"]
fn aql_graph_capture_replay() {
    use crate::allocator::RawBuffer;
    use crate::amd::AmdGraph;
    use crate::device::{GraphKernel, Program};

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if !require_multi_xcc(&alloc) {
        return;
    }
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }

    let ir = r#"; ModuleID = 'amd_graph_probe'
source_filename = "amd_graph_probe"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @amd_graph_probe(ptr noalias %buf0) #0 {
entry:
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}

attributes #0 = { alwaysinline nounwind "no-builtins" "amdgpu-flat-work-group-size"="1,64" "no-trapping-math"="true" }
"#;
    let bytes = match clang_amdgcn(ir, "gfx942") {
        Some(b) => b,
        None => {
            eprintln!("PROBE skipped: clang amdgcn (gfx942) unavailable.");
            return;
        }
    };
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "amd_graph_probe", &global_f32_buffer_abi())
        .expect("load program");

    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let (out_gpu, out_host) = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("output buffer must be host-visible"),
    };

    // Capture a one-kernel static chain.
    let kernels = vec![GraphKernel {
        program: &prog as &dyn Program,
        buffers: vec![out_gpu as *mut u8],
        vals: vec![],
        global_size: Some([1, 1, 1]),
        local_size: Some([1, 1, 1]),
        deps: vec![],
    }];
    let graph = match AmdGraph::capture(&alloc, &kernels).expect("capture") {
        Some(g) => g,
        None => {
            eprintln!("PROBE skipped: chain not graphable on this device.");
            return;
        }
    };

    // Replay twice with a fresh sentinel each time; the kernel must overwrite it
    // with 0.0 and `synchronize_all` must drain the graph's completion signal.
    for trial in 0..2 {
        let sentinel: f32 = -7.5 * (trial as f32 + 1.0);
        // SAFETY: out_host is the host-visible output buffer.
        unsafe { std::ptr::write_volatile(out_host.as_ptr() as *mut f32, sentinel) };
        graph.replay(&[], &[]).expect("graph replay");
        core.synchronize_all().expect("synchronize_all");
        let v = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };
        assert_eq!(v, 0.0, "graph replay #{trial}: kernel store must land ({sentinel} -> {v})");
    }
    eprintln!("PROBE graph capture+replay: 2 replays on the shared queue ran the kernel correctly.");

    drop(graph); // synchronize + free kernargs before the output buffer drops.
    out_buf.free_amd_device_in_place();
}

/// DEBUG: minimal 2-kernel data-dependency graph. Kernel A stores 5.0 to buf[0];
/// kernel B reads buf[0], adds 1.0, writes back → expect 6.0. Tests whether a
/// batched graph honours the inter-kernel read-after-write (the QR multi-kernel
/// failure mode), with full control over the chain.
#[test]
#[ignore = "manual hardware probe; needs a real gfx942 AMD GPU + clang"]
fn aql_graph_two_kernel_raw_dependency() {
    use crate::allocator::RawBuffer;
    use crate::amd::AmdGraph;
    use crate::device::{GraphKernel, Program};

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if !require_multi_xcc(&alloc) {
        return;
    }
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }

    let ir_set = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k_set(ptr noalias %buf0) #0 {
  store float 5.0, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let ir_inc = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k_inc(ptr noalias %buf0) #0 {
  %v = load float, ptr %buf0
  %r = fadd float %v, 1.0
  store float %r, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let (bytes_set, bytes_inc) = match (clang_amdgcn(ir_set, "gfx942"), clang_amdgcn(ir_inc, "gfx942")) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("PROBE skipped: clang unavailable.");
            return;
        }
    };
    let prog_set =
        AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_set, "k_set", &global_f32_buffer_abi()).expect("load k_set");
    let prog_inc =
        AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_inc, "k_inc", &global_f32_buffer_abi()).expect("load k_inc");

    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let (out_gpu, out_host) = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("host-visible"),
    };
    // SAFETY: seed a sentinel.
    unsafe { std::ptr::write_volatile(out_host.as_ptr() as *mut f32, -99.0) };

    let kernels = vec![
        GraphKernel {
            program: &prog_set as &dyn Program,
            buffers: vec![out_gpu as *mut u8],
            vals: vec![],
            global_size: Some([1, 1, 1]),
            local_size: Some([1, 1, 1]),
            deps: vec![],
        },
        // Kernel B reads+writes the buffer A wrote: a true RAW (and WAW) on A.
        // Declaring dep `0` makes DAG dispatch gate B's launch on A's completion
        // signal via a `barrier_and`, the exact path this probe validates.
        GraphKernel {
            program: &prog_inc as &dyn Program,
            buffers: vec![out_gpu as *mut u8],
            vals: vec![],
            global_size: Some([1, 1, 1]),
            local_size: Some([1, 1, 1]),
            deps: vec![0],
        },
    ];
    let graph = match AmdGraph::capture(&alloc, &kernels).expect("capture") {
        Some(g) => g,
        None => {
            eprintln!("PROBE skipped: not graphable.");
            return;
        }
    };
    graph.replay(&[], &[]).expect("replay");
    core.synchronize_all().expect("sync");
    let v = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };
    eprintln!(
        "PROBE 2-kernel RAW: buf -99 -> {v} (A=5.0 then B=+1.0; expect 6.0; 5.0 ⇒ B ran before A's write was visible)"
    );
    assert_eq!(v, 6.0, "kernel B must observe kernel A's write in the batch");

    drop(graph);
    out_buf.free_amd_device_in_place();
}

/// PM4 DISPATCH TIMESTAMP PROBE (manual hardware probe; `#[ignore]`). Dispatch
/// one trivial kernel through the per-plan `PlanContext` with `profile=true`,
/// drain, and read back the GPU-clock `start`/`end` the two `release_mem_timestamp`
/// probes wrote. Validates the single-XCC PM4 timestamp round-trip end to end
/// (the path with no AQL `ENABLE_PROFILING` auto-stamp).
///
/// Run: SVOD_DEVICE=AMD:0 cargo test -p svod-device --lib pm4_dispatch_timestamp_probe -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "manual hardware probe; needs a real single-XCC AMD GPU + clang"]
fn pm4_dispatch_timestamp_probe() {
    use crate::allocator::RawBuffer;
    use crate::device::Program;

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if !require_single_xcc(&alloc) {
        return;
    }
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    let mcpu = alloc.dev.arch.mcpu();
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @pm4_ts_probe(ptr noalias %buf0) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let bytes = match clang_amdgcn(ir, mcpu) {
        Some(b) => b,
        None => {
            eprintln!("PROBE skipped: clang amdgcn ({mcpu}) unavailable.");
            return;
        }
    };
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "pm4_ts_probe", &global_f32_buffer_abi())
        .expect("load program");

    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let out_gpu = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, .. } => *gpu_addr,
        _ => panic!("output buffer must be host-visible"),
    };

    let ctx = prog.new_exec_context().expect("exec context").expect("AMD yields a plan context");
    // profile=true: we hold `handle` across `synchronize`, so arming the probes
    // is safe (this is exactly the invariant the `profile` flag enforces).
    let handle = unsafe {
        ctx.dispatch(&prog as &dyn Program, &[out_gpu as *mut u8], &[], Some([1, 1, 1]), Some([1, 1, 1]), true)
    }
    .expect("dispatch");
    ctx.synchronize().expect("synchronize");

    let (start, end) =
        handle.expect("a profiled dispatch yields a timestamp handle").timestamps_ns().expect("gpu-clock timestamps");
    assert!(end > start, "end ts ({end}) must exceed start ts ({start})");
    let dur = end - start;
    assert!(dur < 1_000_000_000, "a trivial kernel should run in < 1s, got {dur} ns");
    eprintln!("PROBE PM4 dispatch timestamp: start={start} end={end} dur={dur} ns");

    out_buf.free_amd_device_in_place();
}

/// Forced-AQL timeline stress on any supported AMD GPU. Multi-XCC hardware uses
/// AQL by default; single-XCC hardware must set `SVOD_AMD_AQL=1`.
///
/// Run: `SVOD_DEVICE=AMD:0 SVOD_AMD_AQL=1 cargo test -p svod-device --lib
/// aql_timeline_stress_probe -- --ignored --nocapture --test-threads=1`
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU + clang and an AQL queue"]
fn aql_timeline_stress_probe() {
    use crate::allocator::RawBuffer;
    use crate::device::Program;

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if crate::amd::queue::AmdComputeQueue::will_use_pm4(core) {
        eprintln!("PROBE skipped: set SVOD_AMD_AQL=1 on single-XCC hardware.");
        return;
    }
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    let mcpu = alloc.dev.arch.mcpu();
    let ir = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @aql_stress(ptr noalias %buf0) #0 {
  store float 1.0, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let bytes = match clang_amdgcn(ir, mcpu) {
        Some(bytes) => bytes,
        None => {
            eprintln!("PROBE skipped: clang amdgcn ({mcpu}) unavailable.");
            return;
        }
    };
    let program =
        AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "aql_stress", &global_f32_buffer_abi()).expect("load");
    let output = alloc.alloc_uncached(64).expect("output");
    let (output_gpu, output_host) = match &output {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(host), .. } => (*gpu_addr, *host),
        _ => panic!("output buffer must be host-visible"),
    };
    let context = program.new_exec_context().expect("context").expect("AMD context");

    for _ in 0..2_000 {
        unsafe { context.dispatch(&program, &[output_gpu as *mut u8], &[], Some([1, 1, 1]), Some([1, 1, 1]), false) }
            .expect("asynchronous AQL dispatch");
    }
    context.synchronize().expect("asynchronous AQL drain");

    for _ in 0..128 {
        unsafe { context.dispatch(&program, &[output_gpu as *mut u8], &[], Some([1, 1, 1]), Some([1, 1, 1]), false) }
            .expect("synchronous AQL dispatch");
        context.synchronize().expect("synchronous AQL drain");
    }

    for _ in 0..4 {
        let mut timestamps = Vec::new();
        for _ in 0..32 {
            timestamps.push(
                unsafe {
                    context.dispatch(&program, &[output_gpu as *mut u8], &[], Some([1, 1, 1]), Some([1, 1, 1]), true)
                }
                .expect("profiled AQL dispatch")
                .expect("timestamp handle"),
            );
        }
        context.synchronize().expect("profiled AQL drain");
        for timestamp in timestamps {
            let (start, end) = timestamp.timestamps_ns().expect("AQL PM4 timestamps");
            assert!(end > start, "end ts ({end}) must exceed start ts ({start})");
        }
    }

    let value = unsafe { std::ptr::read_volatile(output_host.as_ptr() as *const f32) };
    assert_eq!(value, 1.0);
    eprintln!("PROBE AQL timeline stress: 2000 async, 128 sync, 128 profiled dispatches passed.");
    output.free_amd_device_in_place();
}

/// PM4 GRAPH PROBE (manual hardware probe; `#[ignore]`). Single-XCC (RDNA, e.g.
/// gfx1151) analogue of `aql_graph_capture_replay_probe`: capture a 12-kernel
/// static chain into a resident PM4 indirect buffer and replay it twice via
/// the PM4 branch of `AmdGraph::capture`. The
/// kernel's `store f32 0.0` must reach the host-visible buffer on each replay,
/// and `synchronize_all` must drain the wrapping PM4 counter.
///
/// Run: SVOD_DEVICE=AMD:0 cargo test -p svod-device --lib pm4_graph_capture_replay_probe -- --ignored --nocapture --test-threads=1
/// (the probe forces the per-device `pm4_graph` flag on — capture is opt-in by default).
#[test]
#[ignore = "manual hardware probe; needs a real single-XCC AMD GPU + clang"]
fn pm4_graph_capture_replay_probe() {
    use crate::allocator::RawBuffer;
    use crate::amd::AmdGraph;
    use crate::device::{GraphKernel, Program};

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if !require_single_xcc(&alloc) {
        return;
    }
    // PM4 graph capture is opt-in (default per-call — it regresses on gfx1151);
    // force it on so this probe exercises the capture path. Serialize + restore:
    // the flag lives on the shared device core, so we hold the test lock for the
    // probe's duration and restore the prior value on drop (no leak, no race).
    let _serial = PM4_GRAPH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _pm4 = Pm4GraphOverride::enable(core);
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    let mcpu = alloc.dev.arch.mcpu();

    let ir = r#"target triple = "amdgcn-amd-amdhsa"
declare i32 @llvm.amdgcn.workitem.id.x()
define amdgpu_kernel void @pm4_graph_probe(ptr noalias %buf0) #0 {
  %tid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %tid_ext = zext i32 %tid to i64
  %p = getelementptr inbounds float, ptr %buf0, i64 %tid_ext
  store float 0.0, ptr %p
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let bytes = match clang_amdgcn(ir, mcpu) {
        Some(b) => b,
        None => {
            eprintln!("PROBE skipped: clang amdgcn ({mcpu}) unavailable.");
            return;
        }
    };
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "pm4_graph_probe", &global_f32_buffer_abi())
        .expect("load program");

    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let (out_gpu, out_host) = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("output buffer must be host-visible"),
    };

    let kernels = (0..12)
        .map(|index| GraphKernel {
            program: &prog as &dyn Program,
            buffers: vec![out_gpu as *mut u8],
            vals: vec![],
            global_size: Some([1, 1, 1]),
            local_size: Some([1, 1, 1]),
            deps: (index > 0).then(|| vec![index - 1]).unwrap_or_default(),
        })
        .collect::<Vec<_>>();
    let graph = match AmdGraph::capture(&alloc, &kernels).expect("capture") {
        Some(g) => g,
        None => {
            eprintln!("PROBE skipped: chain not graphable on this device.");
            return;
        }
    };

    for trial in 0..2 {
        let sentinel: f32 = -7.5 * (trial as f32 + 1.0);
        // SAFETY: out_host is the host-visible output buffer.
        unsafe { std::ptr::write_volatile(out_host.as_ptr() as *mut f32, sentinel) };
        graph.replay(&[], &[]).expect("graph replay");
        core.synchronize_all().expect("synchronize_all");
        let v = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };
        assert_eq!(v, 0.0, "PM4 graph replay #{trial}: kernel store must land ({sentinel} -> {v})");
    }
    let timestamps = graph.replay_profiled(&[], &[]).expect("profiled graph replay").expect("profile support");
    assert_eq!(timestamps.len(), kernels.len());
    for timestamp in timestamps {
        let (start, end) = timestamp.timestamps_ns().expect("graph GPU timestamps");
        assert!(end > start, "graph end ts ({end}) must exceed start ts ({start})");
    }
    eprintln!("PROBE graph capture+replay: 12 kernels, 2 normal and 1 profiled replay passed.");

    drop(graph);
    out_buf.free_amd_device_in_place();
}

/// PM4 GRAPH RAW PROBE (manual hardware probe; `#[ignore]`). Single-XCC analogue
/// of `aql_graph_two_kernel_raw_dependency`: a 2-kernel RAW chain in one PM4
/// indirect buffer. Kernel A stores 5.0; kernel B loads, adds 1.0, stores back →
/// expect 6.0. Validates the per-kernel `hdp_flush + acquire_mem` hazard barrier
/// makes A's write visible to B inside the single captured IB (a `5.0` result
/// means B ran before A's store was visible — barrier missing).
///
/// Run: SVOD_DEVICE=AMD:0 cargo test -p svod-device --lib pm4_graph_two_kernel_raw_dependency -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "manual hardware probe; needs a real single-XCC AMD GPU + clang"]
fn pm4_graph_two_kernel_raw_dependency() {
    use crate::allocator::RawBuffer;
    use crate::amd::AmdGraph;
    use crate::device::{GraphKernel, Program};

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    if !require_single_xcc(&alloc) {
        return;
    }
    // PM4 graph capture is opt-in (default per-call); force it on for this probe.
    // Serialize + restore the shared per-device flag (no leak, no race).
    let _serial = PM4_GRAPH_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let _pm4 = Pm4GraphOverride::enable(core);
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    let mcpu = alloc.dev.arch.mcpu();

    let ir_set = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k_set(ptr noalias %buf0) #0 {
  store float 5.0, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let ir_inc = r#"target triple = "amdgcn-amd-amdhsa"
define amdgpu_kernel void @k_inc(ptr noalias %buf0) #0 {
  %v = load float, ptr %buf0
  %r = fadd float %v, 1.0
  store float %r, ptr %buf0
  ret void
}
attributes #0 = { alwaysinline nounwind "amdgpu-flat-work-group-size"="1,64" }
"#;
    let (bytes_set, bytes_inc) = match (clang_amdgcn(ir_set, mcpu), clang_amdgcn(ir_inc, mcpu)) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            eprintln!("PROBE skipped: clang unavailable.");
            return;
        }
    };
    let prog_set =
        AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_set, "k_set", &global_f32_buffer_abi()).expect("load k_set");
    let prog_inc =
        AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_inc, "k_inc", &global_f32_buffer_abi()).expect("load k_inc");

    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let (out_gpu, out_host) = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("host-visible"),
    };
    // SAFETY: seed a sentinel.
    unsafe { std::ptr::write_volatile(out_host.as_ptr() as *mut f32, -99.0) };

    let kernels = vec![
        GraphKernel {
            program: &prog_set as &dyn Program,
            buffers: vec![out_gpu as *mut u8],
            vals: vec![],
            global_size: Some([1, 1, 1]),
            local_size: Some([1, 1, 1]),
            deps: vec![],
        },
        GraphKernel {
            program: &prog_inc as &dyn Program,
            buffers: vec![out_gpu as *mut u8],
            vals: vec![],
            global_size: Some([1, 1, 1]),
            local_size: Some([1, 1, 1]),
            deps: vec![0],
        },
    ];
    let graph = match AmdGraph::capture(&alloc, &kernels).expect("capture") {
        Some(g) => g,
        None => {
            eprintln!("PROBE skipped: not graphable.");
            return;
        }
    };
    graph.replay(&[], &[]).expect("replay");
    core.synchronize_all().expect("sync");
    let v = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };
    eprintln!(
        "PROBE PM4 2-kernel RAW: buf -99 -> {v} (A=5.0 then B=+1.0; expect 6.0; 5.0 ⇒ B ran before A's write was visible)"
    );
    assert_eq!(v, 6.0, "kernel B must observe kernel A's write in the captured IB");

    drop(graph);
    out_buf.free_amd_device_in_place();
}
