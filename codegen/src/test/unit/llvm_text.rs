use smallvec::SmallVec;
use svod_dtype::{AddrSpace, AmdArch, DType};
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, ReduceOp, RendererDevice, WmmaMetadata, WmmaUpcastAxes};

use super::*;
use crate::Renderer;
use crate::llvm::LlvmTextRenderer;

#[test]
fn test_simple_add() {
    let a = UOp::param(0, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
    let b = UOp::param(1, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
    let out = UOp::param(2, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);

    let idx = UOp::index_const(0);
    let a_idx = UOp::index().buffer(a.clone()).indices(vec![idx.clone()]).call().unwrap();
    let b_idx = UOp::index().buffer(b.clone()).indices(vec![idx.clone()]).call().unwrap();
    let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();

    let a_load = UOp::load().buffer(a.clone()).index(a_idx).call();
    let b_load = UOp::load().buffer(b.clone()).index(b_idx).call();

    let add = UOp::new(Op::Binary(BinaryOp::Add, a_load, b_load), DType::Float32);

    let store = out_idx.store(add);
    let sink = UOp::sink(vec![store]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());

    let result = render(&linear, Some("test_add")).unwrap();
    println!("{}", result.code);

    assert!(result.code.contains("define void @test_add("));
    assert!(result.code.contains("noalias align 32"));
    assert!(!result.code.contains("_inner"));
    assert!(!result.code.contains("ptr %args"));
    assert!(result.code.contains("fadd"));
    assert!(result.code.contains("load"));
    assert!(result.code.contains("store"));
}

// ── AMD target tests ───────────────────────────────────────────────────────
//
// These exercise the AMDLLVMRenderer codegen path without invoking clang.
// We assert on the emitted IR strings; downstream clang-amdgcn invocation is
// gated on Phase 2 and verified separately.

fn render_amd_linearized(root: &std::sync::Arc<svod_ir::UOp>, arch: AmdArch, name: &str) -> crate::RenderedKernel {
    let linear = svod_ir::UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    LlvmTextRenderer::amd(arch).render(&linear, Some(name)).expect("AMD render")
}

#[test]
fn amd_emits_kernel_abi_and_target_triple() {
    let p = UOp::param(0, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
    let idx = UOp::index_const(0);
    let p_idx = UOp::index().buffer(p.clone()).indices(vec![idx]).call().unwrap();
    let store = p_idx.store(UOp::native_const(1.0f32));
    let sink = UOp::sink(vec![store]);

    let result = render_amd_linearized(&sink, AmdArch::Gfx1100, "amd_smoke");
    println!("{}", result.code);

    assert!(result.code.contains("target triple = \"amdgcn-amd-amdhsa\""), "missing triple:\n{}", result.code);
    assert!(result.code.contains("define amdgpu_kernel void @amd_smoke("), "missing kernel ABI:\n{}", result.code);
    assert!(
        result.code.contains("\"amdgpu-flat-work-group-size\"=\"1,1\""),
        "missing flat-work-group-size attr:\n{}",
        result.code
    );
    assert!(result.code.contains("alwaysinline"), "missing alwaysinline attr:\n{}", result.code);
}

#[test]
fn amd_special_emits_workgroup_workitem_intrinsics() {
    // y[gidx0] = x[lidx0]
    let x = UOp::param(0, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);
    let y = UOp::param(1, 1, DType::Float32.ptr(Some(1), AddrSpace::Global), None);

    let g = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let l = UOp::special(UOp::index_const(4), "lidx0".to_string());

    let x_idx = UOp::index().buffer(x.clone()).indices(vec![l]).call().unwrap();
    let load = UOp::load().buffer(x.clone()).index(x_idx).call();

    let y_idx = UOp::index().buffer(y.clone()).indices(vec![g]).call().unwrap();
    let store = y_idx.store(load);
    let sink = UOp::sink(vec![store]);

    let result = render_amd_linearized(&sink, AmdArch::Gfx1100, "amd_special");
    println!("{}", result.code);

    assert!(
        result.code.contains("call i32 @llvm.amdgcn.workgroup.id.x()"),
        "missing workgroup.id intrinsic:\n{}",
        result.code
    );
    assert!(
        result.code.contains("call i32 @llvm.amdgcn.workitem.id.x()"),
        "missing workitem.id intrinsic:\n{}",
        result.code
    );
    assert!(
        result.code.contains("declare i32 @llvm.amdgcn.workgroup.id.x()"),
        "missing workgroup.id declare:\n{}",
        result.code
    );
    assert!(
        result.code.contains("\"amdgpu-flat-work-group-size\"=\"1,4\""),
        "wrong flat-work-group-size (expected upper bound 4):\n{}",
        result.code
    );
}

#[test]
fn amd_barrier_emits_fence_and_s_barrier() {
    let noop = UOp::noop();
    let barrier = noop.barrier(smallvec::SmallVec::new());
    let sink = UOp::sink(vec![barrier]);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1100, "amd_barrier");
    println!("{}", result.code);

    assert!(result.code.contains("fence syncscope(\"workgroup\") release"), "missing release fence:\n{}", result.code);
    assert!(result.code.contains("call void @llvm.amdgcn.s.barrier()"), "missing s.barrier:\n{}", result.code);
    assert!(result.code.contains("fence syncscope(\"workgroup\") acquire"), "missing acquire fence:\n{}", result.code);
    assert!(
        result.code.contains("declare void @llvm.amdgcn.s.barrier()"),
        "missing s.barrier declare:\n{}",
        result.code
    );
}

#[test]
fn amd_define_local_emits_addrspace3_module_global() {
    // DefineLocal with size=16, base=f32 → @local<id> addrspace(3) global
    let local = UOp::new(Op::DefineLocal(42), DType::Float32.ptr(Some(16), AddrSpace::Local));
    let sink = UOp::sink(vec![local]);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1100, "amd_lds");
    println!("{}", result.code);

    assert!(
        result.code.contains("@local42 = internal unnamed_addr addrspace(3) global [16 x float] undef"),
        "missing addrspace(3) LDS global:\n{}",
        result.code
    );
}

// ── Reduce / WMMA emission (parity with tinygrad's AMDLLVMRenderer) ──────────

fn reduce_sum_sink() -> std::sync::Arc<svod_ir::UOp> {
    // sum of 5.0 over the range 0..10.
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let range =
        UOp::range_axis(UOp::const_(DType::Index, ConstValue::Int(10)), AxisId::Renumbered(0), AxisType::Reduce);
    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    UOp::sink(vec![reduce.end(ranges)])
}

#[test]
fn amd_reduce_accumulator_uses_addrspace5() {
    // AMDGPU rejects addrspace(0) allocas (clang: "alloca on amdgpu must be in
    // addrspace(5)"), so the reduce accumulator allocates in addrspace(5) and
    // addrspacecasts to a generic `ptr` — same idiom as DEFINE_REG. (tinygrad
    // can keep addrspace(0) allocas because it feeds the triple to the LLVM
    // C-API TargetMachine; svod emits the triple into the IR text and compiles
    // via the clang CLI, which applies the amdgcn datalayout at parse time.)
    let result = render_amd_linearized(&reduce_sum_sink(), AmdArch::Gfx1151, "amd_reduce");
    println!("{}", result.code);

    assert!(result.code.contains("define amdgpu_kernel void @amd_reduce("), "missing kernel ABI:\n{}", result.code);
    assert!(
        result.code.contains("alloca float, addrspace(5)"),
        "reduce accumulator must alloca in addrspace(5):\n{}",
        result.code
    );
    assert!(
        result.code.contains("addrspacecast ptr addrspace(5)"),
        "reduce accumulator must addrspacecast to a generic ptr:\n{}",
        result.code
    );
}

#[test]
fn amd_reduce_ir_assembles_with_llvm_as() {
    // Smoke-test that the emitted AMD IR actually verifies, by piping it through
    // `llvm-as` (skipped when no such tool is on PATH). This is the regression
    // guard that caught the addrspace(0) reduce-accumulator bug.
    let result = render_amd_linearized(&reduce_sum_sink(), AmdArch::Gfx1151, "amd_reduce_asm");
    assert_llvm_ir_assembles(&result.code);
}

/// f16×f16→f32 WMMA metadata for an RDNA3 16×16×16 tile (`<16 x half>` inputs,
/// `<8 x float>` accumulator).
fn wmma_f16_f32_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_16_half_float".to_string(),
        dims: (16, 16, 16),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: RendererDevice::AppleAmx, // unused by the AMD path (keyed on `arch`)
        threads: 32,
        upcast_axes: WmmaUpcastAxes { a: vec![(2, 16)], b: vec![(2, 16)], c: vec![(2, 8)] },
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

fn wmma_f16_f32_sink() -> std::sync::Arc<svod_ir::UOp> {
    let a = UOp::const_(DType::Float16, ConstValue::Float(0.0)).broadcast(16);
    let b = UOp::const_(DType::Float16, ConstValue::Float(0.0)).broadcast(16);
    let c = UOp::const_(DType::Float32, ConstValue::Float(0.0)).broadcast(8);
    UOp::sink(vec![UOp::wmma(a, b, c, wmma_f16_f32_metadata())])
}

#[test]
fn amd_wmma_emits_intrinsic_without_amx_scratch() {
    // AMD lowers WMMA to `llvm.amdgcn.wmma.*` over SSA vectors, so the CPU/AMX
    // scratch allocas must NOT be emitted on the AMD path (they were, before
    // the LlvmTarget::Cpu gate).
    let result = render_amd_linearized(&wmma_f16_f32_sink(), AmdArch::Gfx1100, "amd_wmma");
    println!("{}", result.code);

    assert!(
        result.code.contains("@llvm.amdgcn.wmma.f32.16x16x16.f16"),
        "missing WMMA intrinsic call:\n{}",
        result.code
    );
    assert!(!result.code.contains("_amx"), "AMD WMMA must not emit AMX scratch allocas:\n{}", result.code);
    // NB: no llvm-as smoke here. These const operands splat into inline vector
    // literals (`<16 x half> <half 0, ...>`), and the lightweight intrinsic-
    // declaration synthesizer comma-splits inside them. Real WMMA operands are
    // SSA values (loads/contracts) with no internal commas, so the path that
    // breaks here is never produced in practice.
}

#[test]
fn cpu_wmma_still_emits_amx_scratch() {
    // Regression guard for the other side of the gate: the CPU/AMX path must
    // keep preallocating its `_amx` scratch slots.
    let a = UOp::const_(DType::Float32, ConstValue::Float(0.0)).broadcast(16);
    let b = UOp::const_(DType::Float32, ConstValue::Float(0.0)).broadcast(16);
    let c = UOp::const_(DType::Float32, ConstValue::Float(0.0)).broadcast(256);
    let wmma = UOp::wmma(a, b, c, cpu_amx_f32_metadata());
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(UOp::sink(vec![wmma])).into());
    let result = render(&linear, Some("cpu_wmma")).expect("CPU render");
    assert!(result.code.contains("_amx"), "CPU WMMA must emit AMX scratch:\n{}", result.code);
}

fn cpu_amx_f32_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_float_float".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: RendererDevice::AppleAmx,
        threads: 1,
        upcast_axes: WmmaUpcastAxes { a: vec![(2, 256)], b: vec![(2, 256)], c: vec![(2, 256)] },
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

/// Pipe `ir` through an `llvm-as` on PATH and assert it parses. Skips (returns)
/// when no `llvm-as` is installed, so the test is a no-op on machines without
/// LLVM tools.
fn assert_llvm_ir_assembles(ir: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let tool = ["llvm-as", "llvm-as-19", "llvm-as-18", "llvm-as-17", "llvm-as-16"].into_iter().find(|t| {
        Command::new(t)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    });
    let Some(tool) = tool else {
        eprintln!("skipping llvm-as smoke test: no llvm-as on PATH");
        return;
    };

    let mut child = Command::new(tool)
        .args(["-o", "/dev/null", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn llvm-as");
    child.stdin.take().unwrap().write_all(ir.as_bytes()).expect("write IR to llvm-as");
    let out = child.wait_with_output().expect("wait for llvm-as");
    assert!(
        out.status.success(),
        "llvm-as rejected the emitted AMD IR:\n{ir}\n--- llvm-as stderr ---\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
