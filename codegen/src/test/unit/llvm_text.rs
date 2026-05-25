use super::*;
use svod_dtype::{AddrSpace, AmdArch, DType};
use svod_ir::{BinaryOp, Op};

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
