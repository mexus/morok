use super::test_support::{amd_alloc_or_skip, require_multi_xcc, require_single_xcc};
use crate::amd::program::*;
use crate::amd::queue::build_dispatch_packet;

/// Serializes the `#[ignore]` PM4-graph probes that toggle the per-device
/// `pm4_graph` flag. The flag lives on the process-global (`DEVICE_CACHE`-backed)
/// `AmdDeviceCore`, so two probes running concurrently would observe each other's
/// writes; holding this lock for each probe's duration makes the save/restore in
/// [`Pm4GraphOverride`] well-defined regardless of `--test-threads`.
static PM4_GRAPH_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

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

/// PHASE-0 GATE (manual hardware probe; `#[ignore]`). Dispatches a REAL kernel
/// with a native `amd_signal_t` completion_signal — no `RELEASE_MEM`, no
/// `PRED_EXEC`, no timeline — via [`AmdComputeQueue::dispatch_aql_native`], and
/// reports two things the shared-queue redesign hinges on:
///   1. **Completion fires** (and once vs once-per-XCC): the signal's `value`
///      goes 1 -> 0 (once) or 1 -> -(xccs-1) (per-XCC); 1 (timeout) = STOP.
///   2. **Coherence without the manual prologue**: the kernel's `store f32 0.0`
///      reaches the host-visible output buffer purely via the AQL packet's
///      system-scope release fence (we emit no `hdp_flush`/`acquire_mem`).
///
/// Run: cargo test -p svod-device --lib aql_native_kernel_dispatch_probe -- --ignored --nocapture
#[test]
#[ignore = "manual hardware probe; needs a real gfx942 AMD GPU + clang"]
fn aql_native_kernel_dispatch_probe() {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use crate::allocator::RawBuffer;
    use crate::amd::sys::hsa::{amd_signal_kind_t_AMD_SIGNAL_KIND_USER, amd_signal_t};

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    let xccs = alloc.dev.node.num_xcc.max(1);
    if !require_multi_xcc(&alloc) {
        return;
    }
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }

    // One-buffer kernel: workitem 0 stores 0.0 into buf0[0].
    let ir = r#"; ModuleID = 'amd_native_probe'
source_filename = "amd_native_probe"
target triple = "amdgcn-amd-amdhsa"

declare i32 @llvm.amdgcn.workitem.id.x()

define amdgpu_kernel void @amd_native_probe(ptr noalias %buf0) #0 {
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
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "amd_native_probe", 1, 0).expect("load program");
    let pool = crate::amd::connector::PoolQueue::new_with_resources(Arc::clone(core), &alloc).expect("pool queue");
    let conn = crate::amd::connector::OwnerCtx::new(Arc::clone(&pool));
    assert!(!conn.pool().queue().is_pm4(), "multi-XCC device must use an AQL queue");

    // Output buffer (host-visible GTT), seeded with a sentinel so the kernel's
    // store-of-0.0 is observable.
    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let (out_gpu, out_host) = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("output buffer must be host-visible"),
    };
    let sentinel: f32 = -123.5;
    // SAFETY: out_host is the 64-byte buffer we just allocated.
    unsafe { std::ptr::write_volatile(out_host.as_ptr() as *mut f32, sentinel) };

    // Native completion signal: amd_signal_t { kind=USER, value=1 }, no mailbox.
    let sig_buf = alloc.alloc_uncached(64).expect("signal buffer");
    let (sig_gpu, sig_host) = match &sig_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("signal buffer must be host-visible"),
    };
    let value_off = std::mem::offset_of!(amd_signal_t, __bindgen_anon_1);
    // SAFETY: amd_signal_t is POD; all-zero is a valid INVALID signal.
    let mut sig: amd_signal_t = unsafe { std::mem::zeroed() };
    sig.kind = amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64;
    sig.__bindgen_anon_1.value = 1;
    // SAFETY: sig_host points at the 64-byte signal buffer.
    unsafe { std::ptr::copy_nonoverlapping(&sig as *const _ as *const u8, sig_host.as_ptr(), 64) };
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

    // Kernargs: a single 64-bit pointer = the output buffer GPU VA.
    let off = conn.pool().arena().bump(prog.kernarg_size(), 16).expect("kernarg bump");
    // SAFETY: fresh slot, sole writer.
    let host_base = unsafe { conn.pool().arena().host_at(off) };
    unsafe { std::ptr::copy_nonoverlapping(out_gpu.to_le_bytes().as_ptr(), host_base, 8) };
    let kernarg_gpu = conn.pool().arena().gpu_at(off);

    // Packed-field copies (cf. execute_on) before passing by value.
    let priv_seg = prog.kd.private_segment_fixed_size;
    let group_seg = prog.kd.group_segment_fixed_size;
    let packet = build_dispatch_packet(
        [1, 1, 1],
        [1, 1, 1],
        priv_seg,
        group_seg,
        prog.aql_prog_addr,
        kernarg_gpu,
        /*completion_signal=*/ sig_gpu,
    );
    conn.pool().queue().dispatch_aql_native(&packet).expect("native dispatch");

    // Poll completion: value drops below the initial 1, or time out.
    // SAFETY: coherent GTT slot; firmware writes `value` at sig_gpu+value_off.
    let value_ptr = unsafe { sig_host.as_ptr().add(value_off) as *const i64 };
    let start = Instant::now();
    let final_value = loop {
        let v = unsafe { std::ptr::read_volatile(value_ptr) };
        if v <= 0 || start.elapsed() > Duration::from_secs(5) {
            break v;
        }
        std::hint::spin_loop();
    };
    let final_kind = unsafe { std::ptr::read_volatile(sig_host.as_ptr() as *const i64) };
    let out_val = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };

    eprintln!(
        "PROBE native KERNEL dispatch (num_xcc={xccs}): completion 1 -> {final_value} in {:?}; kind={final_kind}",
        start.elapsed()
    );
    eprintln!(
        "  completion: 0 = once | <0 = per-XCC ({} decrements) | 1 = NEVER FIRED (timeout){}",
        if final_value < 0 { (1 - final_value).to_string() } else { "n/a".into() },
        if final_kind != amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64 {
            " | WARNING: kind moved — handle treated as &value, not struct base"
        } else {
            ""
        }
    );
    eprintln!(
        "  coherence: out buffer {sentinel} -> {out_val} (expect 0.0 ⇒ release fence flushed without manual prologue)"
    );

    sig_buf.free_amd_device_in_place();
    out_buf.free_amd_device_in_place();
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
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "amd_graph_probe", 1, 0).expect("load program");

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
        graph.replay(&[]).expect("graph replay");
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
    let prog_set = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_set, "k_set", 1, 0).expect("load k_set");
    let prog_inc = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_inc, "k_inc", 1, 0).expect("load k_inc");

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
    graph.replay(&[]).expect("replay");
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
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "pm4_ts_probe", 1, 0).expect("load program");

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

/// PM4 GRAPH PROBE (manual hardware probe; `#[ignore]`). Single-XCC (RDNA, e.g.
/// gfx1151) analogue of `aql_graph_capture_replay_probe`: capture a ONE-kernel
/// static chain into a PM4 indirect buffer and replay it twice via
/// [`AmdGraphPm4`] (the `will_use_pm4` branch of `AmdGraph::capture`). The
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
    let prog = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes, "pm4_graph_probe", 1, 0).expect("load program");

    let out_buf = alloc.alloc_uncached(64).expect("output buffer");
    let (out_gpu, out_host) = match &out_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("output buffer must be host-visible"),
    };

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

    for trial in 0..2 {
        let sentinel: f32 = -7.5 * (trial as f32 + 1.0);
        // SAFETY: out_host is the host-visible output buffer.
        unsafe { std::ptr::write_volatile(out_host.as_ptr() as *mut f32, sentinel) };
        graph.replay(&[]).expect("graph replay");
        core.synchronize_all().expect("synchronize_all");
        let v = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };
        assert_eq!(v, 0.0, "PM4 graph replay #{trial}: kernel store must land ({sentinel} -> {v})");
    }
    eprintln!("PROBE PM4 graph capture+replay: 2 replays via one indirect buffer ran the kernel correctly.");

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
    let prog_set = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_set, "k_set", 1, 0).expect("load k_set");
    let prog_inc = AmdProgram::load(alloc.dev.clone(), &alloc, &bytes_inc, "k_inc", 1, 0).expect("load k_inc");

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
    graph.replay(&[]).expect("replay");
    core.synchronize_all().expect("sync");
    let v = unsafe { std::ptr::read_volatile(out_host.as_ptr() as *const f32) };
    eprintln!(
        "PROBE PM4 2-kernel RAW: buf -99 -> {v} (A=5.0 then B=+1.0; expect 6.0; 5.0 ⇒ B ran before A's write was visible)"
    );
    assert_eq!(v, 6.0, "kernel B must observe kernel A's write in the captured IB");

    drop(graph);
    out_buf.free_amd_device_in_place();
}
