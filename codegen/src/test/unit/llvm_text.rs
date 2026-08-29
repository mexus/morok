use svod_dtype::{AddrSpace, AmdArch, DType};
use svod_ir::{AxisId, AxisType, BinaryOp, ConstValue, Op, RendererDevice, WmmaMetadata, WmmaUpcastAxes};

use super::*;
use crate::Renderer;
use crate::llvm::LlvmTextRenderer;

fn slotted_var(name: &str, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param { shape, arg } = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param { shape: shape.clone(), arg }, DType::Int32)
}

#[test]
fn llvm_signature_uses_canonical_mixed_param_slot_order() {
    let sink = UOp::sink(vec![
        slotted_var("high", 3),
        UOp::param(2, 1, DType::Float32, None),
        slotted_var("low", 1),
        UOp::param(0, 1, DType::Float32, None),
    ]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());
    let rendered = render(&linear, Some("mixed_abi")).expect("render mixed ABI");
    assert_eq!(rendered.buffer_args.iter().map(|arg| arg.index).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(rendered.var_names, vec!["low", "high"]);
    assert!(
        rendered.code.contains(
            "define void @mixed_abi(ptr noalias align 32 %data0, i32 %data1, ptr noalias align 32 %data2, i32 %data3)"
        ),
        "{}",
        rendered.code
    );
}

#[test]
fn test_simple_add() {
    let a = UOp::param(0, 1, DType::Float32, None);
    let b = UOp::param(1, 1, DType::Float32, None);
    let out = UOp::param(2, 1, DType::Float32, None);

    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let a_idx = UOp::index().buffer(a.clone()).indices(vec![idx.clone()]).call().unwrap();
    let b_idx = UOp::index().buffer(b.clone()).indices(vec![idx.clone()]).call().unwrap();
    let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();

    let a_load = UOp::load().index(a_idx).call();
    let b_load = UOp::load().index(b_idx).call();

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

#[test]
fn llvm_float_comparisons_use_tinygrad_nan_predicates() {
    let lhs = UOp::param(0, 1, DType::Float32, None);
    let rhs = UOp::param(1, 1, DType::Float32, None);
    let index = UOp::const_(DType::Int32, ConstValue::Int(0));
    let lhs = UOp::load().index(UOp::index().buffer(lhs).indices(vec![index.clone()]).call().unwrap()).call();
    let rhs = UOp::load().index(UOp::index().buffer(rhs).indices(vec![index]).call().unwrap()).call();
    let sink = UOp::sink(vec![
        lhs.try_cmplt(&rhs).unwrap(),
        lhs.try_cmple(&rhs).unwrap(),
        lhs.try_cmpgt(&rhs).unwrap(),
        lhs.try_cmpge(&rhs).unwrap(),
        lhs.try_cmpeq(&rhs).unwrap(),
        lhs.try_cmpne(&rhs).unwrap(),
    ]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());

    for rendered in [
        render(&linear, Some("cpu_float_cmp")).expect("CPU LLVM render"),
        LlvmTextRenderer::amd(AmdArch::Gfx1151).render(&linear, Some("amd_float_cmp")).expect("AMD LLVM render"),
    ] {
        for predicate in ["olt", "ole", "ogt", "oge", "oeq", "une"] {
            assert!(
                rendered.code.lines().any(|line| line.contains("fcmp ") && line.contains(predicate)),
                "{predicate}:\n{}",
                rendered.code
            );
        }
        for predicate in ["ult", "ule", "ugt", "uge"] {
            assert!(
                !rendered.code.lines().any(|line| line.contains("fcmp ") && line.contains(predicate)),
                "unordered relational predicate {predicate}:\n{}",
                rendered.code
            );
        }
    }
}

#[test]
fn llvm_rejects_end_for_non_innermost_range() {
    let bound = UOp::const_(DType::Int32, ConstValue::Int(2));
    let outer = UOp::range_axis_dtype(bound.clone(), AxisId::Renumbered(0), AxisType::Loop, DType::Int32);
    let inner = UOp::range_axis_dtype(bound, AxisId::Renumbered(1), AxisType::Loop, DType::Int32);
    let zero = UOp::native_const(0i32);
    let end_outer = zero.end(smallvec::smallvec![outer.clone()]);
    let linear = UOp::linear(smallvec::smallvec![outer.op().sources()[0].clone(), zero, outer, inner, end_outer]);

    let err = render(&linear, Some("bad_end_order")).expect_err("mismatched END must fail");
    assert!(format!("{err}").contains("innermost open range"), "unexpected error: {err}");
}

#[test]
fn llvm_rejects_unclosed_range() {
    let range = UOp::range_axis_dtype(
        UOp::const_(DType::Int32, ConstValue::Int(2)),
        AxisId::Renumbered(0),
        AxisType::Loop,
        DType::Int32,
    );
    let linear = UOp::linear(smallvec::smallvec![range.op().sources()[0].clone(), range]);

    let err = render(&linear, Some("unclosed_range")).expect_err("unclosed RANGE must fail");
    assert!(format!("{err}").contains("unclosed LLVM ranges"), "unexpected error: {err}");
}

// ── AMD target tests ───────────────────────────────────────────────────────
//
// These exercise the AMDLLVMRenderer codegen path without invoking clang.
// We assert on the emitted IR strings; downstream clang-amdgcn invocation is
// gated on Phase 2 and verified separately.

fn render_amd_linearized(root: &std::sync::Arc<svod_ir::UOp>, arch: AmdArch, name: &str) -> crate::RenderedKernel {
    let code_renderer = LlvmTextRenderer::amd(arch);
    let optimizer_renderer = svod_schedule::OptimizerRenderer::for_amd_arch(arch).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        code_renderer.decompositor(),
        Some(crate::llvm::amd_extra_matcher()),
    );
    let lowered = svod_schedule::apply_post_optimization_with_renderer(root.clone(), &optimizer_renderer)
        .expect("post optimization");
    let linear = svod_ir::UOp::linear(svod_schedule::linearize_with_cfg(lowered).into());
    code_renderer.render(&linear, Some(name)).expect("AMD render")
}

#[test]
fn amd_emits_kernel_abi_and_target_triple() {
    let p = UOp::param(0, 1, DType::Float32, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let p_idx = UOp::index().buffer(p.clone()).indices(vec![idx]).call().unwrap();
    let store = p_idx.store(UOp::const_(DType::Float32, ConstValue::Float(1.0)));
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

/// The AMDGPU backend selects `@llvm.exp2` for f16/f32 but has no f64 lowering
/// ("no libcall available for fexp2"), so only f64 stays on ROCm's `__ocml_*`.
#[test_case::test_case(DType::Float16, "@llvm.exp2.f16", "half")]
#[test_case::test_case(DType::Float32, "@llvm.exp2.f32", "float")]
#[test_case::test_case(DType::Float64, "@__ocml_exp2_f64", "double")]
fn amd_uses_llvm_intrinsics_and_contract_only_float_flags(dtype: DType, callee: &str, llvm_type: &str) {
    let input = UOp::param(0, 1, dtype.clone(), None);
    let output = UOp::param(1, 1, dtype, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let input_idx = UOp::index().buffer(input).indices(vec![idx.clone()]).call().unwrap();
    let output_idx = UOp::index().buffer(output).indices(vec![idx]).call().unwrap();
    let value = UOp::load().index(input_idx).call();
    let result = value.try_exp2().unwrap().try_div(&value).unwrap();
    let sink = UOp::sink(vec![output_idx.store(result)]);

    let rendered = render_amd_linearized(&sink, AmdArch::Gfx1151, "amd_float_flags");

    assert!(
        rendered.code.contains(&format!("declare {llvm_type} {callee}({llvm_type})")),
        "missing declaration for {callee}:\n{}",
        rendered.code
    );
    assert!(rendered.code.contains(&format!("call {llvm_type} {callee}")), "missing {callee} call:\n{}", rendered.code);
    assert!(
        rendered.code.contains(&format!("fdiv contract {llvm_type}")),
        "missing contract-only fdiv:\n{}",
        rendered.code
    );
    assert!(!rendered.code.contains(" arcp "), "AMD must not permit approximate reciprocal:\n{}", rendered.code);
}

#[test]
fn amd_cdna_ordinary_fp8_alu_widens_without_changing_storage() {
    for arch in [AmdArch::Gfx942, AmdArch::Gfx950] {
        for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
            let input = UOp::param(0, 2, dtype.clone(), None);
            let output = UOp::param(1, 1, dtype.clone(), None);
            let first = UOp::index().buffer(input.clone()).indices(vec![UOp::native_const(0i32)]).call().unwrap();
            let second = UOp::index().buffer(input).indices(vec![UOp::native_const(1i32)]).call().unwrap();
            let output = UOp::index().buffer(output).indices(vec![UOp::native_const(0i32)]).call().unwrap();
            let first = UOp::load().index(first).call();
            let second = UOp::load().index(second).call();
            let value = first.try_add(&second).unwrap().try_mul(&second).unwrap();
            let rendered = render_amd_linearized(&UOp::sink(vec![output.store(value)]), arch, "amd_fp8_alu");

            assert!(rendered.code.lines().any(|line| line.contains("fadd contract float")), "{}", rendered.code);
            assert!(rendered.code.lines().any(|line| line.contains("fmul contract float")), "{}", rendered.code);
            assert!(!rendered.code.contains("fadd contract i8"), "{}", rendered.code);
            assert!(!rendered.code.contains("fmul contract i8"), "{}", rendered.code);
            assert!(rendered.code.contains("load <2 x i8>") && rendered.code.contains("store i8"), "{}", rendered.code);
            assert_amd_ir_compiles(&rendered.code, arch.mcpu());
        }
    }
}

#[test]
fn amd_special_emits_workgroup_workitem_intrinsics() {
    // y[gidx0] = x[lidx0]
    let x = UOp::param(0, 1, DType::Float32, None);
    let y = UOp::param(1, 1, DType::Float32, None);

    let g = UOp::special(UOp::const_(DType::Int32, ConstValue::Int(8)), "gidx0".to_string());
    let l = UOp::special(UOp::const_(DType::Int32, ConstValue::Int(4)), "lidx0".to_string());

    let x_idx = UOp::index().buffer(x.clone()).indices(vec![l]).call().unwrap();
    let load = UOp::load().index(x_idx).call();

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
    // LOCAL BUFFER with size=16, base=f32 -> @local<slot> addrspace(3) global
    let local = UOp::buffer(42, 16, DType::Float32, AddrSpace::Local, None);
    let sink = UOp::sink(vec![local]);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1100, "amd_lds");
    println!("{}", result.code);

    assert!(
        result.code.contains("@local42 = internal unnamed_addr addrspace(3) global [16 x float] undef"),
        "missing addrspace(3) LDS global:\n{}",
        result.code
    );
}

// ── WMMA emission (parity with tinygrad's AMDLLVMRenderer) ──────────────────

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
        upcast_axes: Some(WmmaUpcastAxes {
            a: vec![(svod_ir::AxisId::Renumbered(2), 16)],
            b: vec![(svod_ir::AxisId::Renumbered(2), 16)],
            c: vec![(svod_ir::AxisId::Renumbered(2), 8)],
        }),
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
    // This test only guards the AMX-scratch gate, not IR validity: const
    // operands splat into inline VConst literals that exercise a different
    // naming path. The `assert_llvm_ir_assembles` coverage for the WMMA call
    // and declaration lives in the SSA-operand tests below.
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
        upcast_axes: Some(WmmaUpcastAxes {
            a: vec![(svod_ir::AxisId::Renumbered(2), 256)],
            b: vec![(svod_ir::AxisId::Renumbered(2), 256)],
            c: vec![(svod_ir::AxisId::Renumbered(2), 256)],
        }),
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

// ── SSA-operand WMMA assemble tests ──────────────────────────────────────────
//
// Real WMMA operands are SSA vector values, not the const splats above. These
// build operands from buffer loads (broadcast of a non-const load lowers to a
// VECTORIZE → `%vN`), then assert the emitted IR assembles under `llvm-as` —
// the regression guard for declaration synthesis + the bf16→i16 / fp8→i64
// operand bitcasts.

fn wmma_buf_load(slot: usize, dt: DType) -> std::sync::Arc<svod_ir::UOp> {
    let p = UOp::param(slot, 1, dt, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    // PARAM and INDEX both carry the storage dtype; INDEX renders as an address.
    let p_idx = UOp::index().buffer(p.clone()).indices(vec![idx]).call().unwrap();
    UOp::load().index(p_idx).call()
}

fn wmma_meta(dims: (usize, usize, usize), in_dt: DType, out_dt: DType, c_count: usize) -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_test".to_string(),
        dims,
        dtype_in: in_dt,
        dtype_out: out_dt,
        device: RendererDevice::AppleAmx, // unused by the AMD path (keyed on `arch`)
        threads: 32,
        upcast_axes: Some(WmmaUpcastAxes {
            a: vec![(svod_ir::AxisId::Renumbered(2), 16)],
            b: vec![(svod_ir::AxisId::Renumbered(2), 16)],
            c: vec![(svod_ir::AxisId::Renumbered(2), c_count)],
        }),
        reduce_axes: vec![],
        tile_grid: (1, 1),
    }
}

/// WMMA over SSA operands: A/B are `<in_count x in_dt>`, C is `<out_count x out_dt>`.
fn wmma_ssa_sink(
    in_dt: DType,
    in_count: usize,
    out_dt: DType,
    out_count: usize,
    meta: WmmaMetadata,
) -> std::sync::Arc<svod_ir::UOp> {
    let a = wmma_buf_load(0, in_dt.clone()).broadcast(in_count);
    let b = wmma_buf_load(1, in_dt).broadcast(in_count);
    let c = wmma_buf_load(2, out_dt).broadcast(out_count);
    UOp::sink(vec![UOp::wmma(a, b, c, meta)])
}

#[test]
fn amd_wmma_f16_f32_ssa_assembles() {
    let meta = wmma_meta((16, 16, 16), DType::Float16, DType::Float32, 8);
    let sink = wmma_ssa_sink(DType::Float16, 16, DType::Float32, 8, meta);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1151, "amd_wmma_f16");
    println!("{}", result.code);
    assert!(result.code.contains("<16 x half> %"), "operands must be SSA vectors:\n{}", result.code);
    assert!(
        result
            .code
            .contains("declare <8 x float> @llvm.amdgcn.wmma.f32.16x16x16.f16(<16 x half>, <16 x half>, <8 x float>)"),
        "garbled WMMA declaration (the `<16` truncation bug):\n{}",
        result.code
    );
    assert_llvm_ir_assembles(&result.code);
}

#[test]
fn amd_wmma_bf16_f32_ssa_assembles() {
    // gfx1151 (the live target) declares a bf16→f32 tensor core. bf16 operands
    // must reach the intrinsic as `<16 x i16>`, not `<16 x bfloat>`.
    let meta = wmma_meta((16, 16, 16), DType::BFloat16, DType::Float32, 8);
    let sink = wmma_ssa_sink(DType::BFloat16, 16, DType::Float32, 8, meta);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1151, "amd_wmma_bf16");
    println!("{}", result.code);
    assert!(result.code.contains("bitcast <16 x bfloat>"), "missing bf16→i16 operand bitcast:\n{}", result.code);
    assert!(
        result.code.contains("@llvm.amdgcn.wmma.f32.16x16x16.bf16(<16 x i16>, <16 x i16>, <8 x float>)"),
        "bf16 WMMA must use i16 wire types:\n{}",
        result.code
    );
    assert_llvm_ir_assembles(&result.code);
}

#[test]
fn amd_wmma_int8_i32_packs_signed_operands() {
    let meta = wmma_meta((16, 16, 16), DType::Int8, DType::Int32, 8);
    let sink = wmma_ssa_sink(DType::Int8, 16, DType::Int32, 8, meta);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1151, "amd_wmma_int8");

    assert_eq!(result.code.matches("bitcast <16 x i8>").count(), 2, "{}", result.code);
    assert_eq!(result.code.matches("to <4 x i32>").count(), 2, "{}", result.code);
    assert!(
        result.code.contains("@llvm.amdgcn.wmma.i32.16x16x16.iu8(i1 true, <4 x i32>")
            && result.code.contains(", i1 true, <4 x i32>")
            && result.code.contains(", <8 x i32>")
            && result.code.contains(", i1 false)"),
        "RDNA int8 WMMA signature must carry packed operands and signedness flags:\n{}",
        result.code
    );
    assert_llvm_ir_assembles(&result.code);
    assert_amd_ir_compiles(&result.code, AmdArch::Gfx1151.mcpu());
}

#[test]
fn amd_gfx12_f16_wmma_compiles() {
    let meta = wmma_meta((16, 16, 16), DType::Float16, DType::Float32, 8);
    let sink = wmma_ssa_sink(DType::Float16, 8, DType::Float32, 8, meta);
    for arch in [AmdArch::Gfx1200, AmdArch::Gfx1201] {
        let result = render_amd_linearized(&sink, arch, "amd_gfx12_wmma_f16");
        assert!(
            result.code.contains("@llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16(<8 x half>, <8 x half>, <8 x float>)"),
            "gfx12 WMMA must use LLVM's overloaded vector suffixes:\n{}",
            result.code
        );
        assert_llvm_ir_assembles(&result.code);
        assert_amd_ir_compiles(&result.code, arch.mcpu());
    }
}

#[test]
fn amd_gfx1201_direct_fp8_wmma_is_rejected() {
    let meta = wmma_meta((16, 16, 16), DType::FP8E4M3, DType::Float32, 8);
    let sink = wmma_ssa_sink(DType::FP8E4M3, 8, DType::Float32, 8, meta);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());
    let err = svod_codegen_renderer(&linear, AmdArch::Gfx1201)
        .expect_err("pinned Tinygrad RDNA4 table has no native FP8 WMMA");
    assert!(err.to_string().contains("no WMMA/MFMA intrinsic for arch=gfx1201"), "unexpected error: {err}");
}

fn svod_codegen_renderer(linear: &std::sync::Arc<svod_ir::UOp>, arch: AmdArch) -> crate::Result<crate::RenderedKernel> {
    use crate::Renderer;
    crate::llvm::LlvmTextRenderer::amd(arch).render(linear, Some("amd_direct_fp8"))
}

#[test]
fn amd_wmma_bf16_bf16_bitcasts_result() {
    // bf16 accumulator: all three operands go as i16 and the i16 result is
    // bitcast back to bf16 (tinygrad llvmir.py:292-294).
    let meta = wmma_meta((16, 16, 16), DType::BFloat16, DType::BFloat16, 16);
    let sink = wmma_ssa_sink(DType::BFloat16, 16, DType::BFloat16, 16, meta);
    let result = render_amd_linearized(&sink, AmdArch::Gfx1100, "amd_wmma_bf16bf16");
    println!("{}", result.code);
    assert!(
        result.code.contains("@llvm.amdgcn.wmma.bf16.16x16x16.bf16(<16 x i16>, <16 x i16>, <16 x i16>, i1)"),
        "bf16→bf16 declaration wrong:\n{}",
        result.code
    );
    assert!(
        result.code.contains("bitcast <16 x i16>") && result.code.contains("to <16 x bfloat>"),
        "missing i16→bf16 result bitcast:\n{}",
        result.code
    );
    assert_llvm_ir_assembles(&result.code);
}

#[test]
fn amd_wmma_bf16_k32_uses_native_bfloat_on_gfx950() {
    // The CDNA4 (gfx950) K=32 `.bf16` double-rate MFMA takes native `<N x bfloat>`
    // operands — passing them as `<N x i16>` fails LLVM's verifier outright
    // (verified: `llc -mcpu=gfx950` accepts `<8 x bfloat>`, rejects `<8 x i16>`).
    // So unlike the K=16 `bf16.1k` form, this path must NOT bitcast bf16 → i16.
    let meta = wmma_meta((16, 16, 32), DType::BFloat16, DType::Float32, 4);
    let sink = wmma_ssa_sink(DType::BFloat16, 8, DType::Float32, 4, meta);
    let result = render_amd_linearized(&sink, AmdArch::Gfx950, "amd_wmma_bf16_k32");
    println!("{}", result.code);
    assert!(
        result.code.contains("@llvm.amdgcn.mfma.f32.16x16x32.bf16(<8 x bfloat>, <8 x bfloat>, <4 x float>"),
        "bf16-K32 MFMA must take native bfloat operands:\n{}",
        result.code
    );
    assert!(
        !result.code.contains("to <8 x i16>") && !result.code.contains("(<8 x i16>"),
        "bf16-K32 operands must NOT be bitcast to i16:\n{}",
        result.code
    );
    assert_llvm_ir_assembles(&result.code);
}

#[test]
fn amd_wmma_fp8_packs_operands_to_i64() {
    // CDNA fp8: the 8 fp8 lanes pack into one i64 (tinygrad bitcasts fp8.vec(8)
    // → uint64). MFMA carries the trailing cbsz/abid/blgp immediates.
    let meta = wmma_meta((16, 16, 32), DType::FP8E4M3, DType::Float32, 4);
    let sink = wmma_ssa_sink(DType::FP8E4M3, 8, DType::Float32, 4, meta);
    for arch in [AmdArch::Gfx942, AmdArch::Gfx950] {
        let result = render_amd_linearized(&sink, arch, "amd_wmma_fp8");
        println!("{}", result.code);
        assert!(
            result.code.contains("bitcast <8 x i8>") && result.code.contains(" to i64"),
            "fp8 operands must pack into i64 on {arch}:\n{}",
            result.code
        );
        assert!(
            result.code.contains("(i64, i64, <4 x float>, i32, i32, i32)"),
            "fp8 MFMA declaration wrong on {arch}:\n{}",
            result.code
        );
        assert_llvm_ir_assembles(&result.code);
        assert_amd_ir_compiles(&result.code, arch.mcpu());
    }
}

#[test]
fn amd_wmma_gfx950_scaled_fp8_uses_i32_vectors_and_scale_immediates() {
    let meta = wmma_meta((16, 16, 128), DType::FP8E5M2, DType::Float32, 4);
    let sink = wmma_ssa_sink(DType::FP8E5M2, 32, DType::Float32, 4, meta);
    let result = render_amd_linearized(&sink, AmdArch::Gfx950, "amd_wmma_scaled_fp8");
    println!("{}", result.code);
    assert!(
        result.code.contains("bitcast <32 x i8>") && result.code.contains("to <8 x i32>"),
        "scaled FP8 operands must pack as eight i32 words:\n{}",
        result.code
    );
    assert!(
        result.code.contains("@llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4(")
            && result.code.contains("i32 1, i32 1, i32 0, i32 127, i32 0, i32 127)"),
        "scaled BF8 MFMA signature/immediates are wrong:\n{}",
        result.code
    );
    assert_llvm_ir_assembles(&result.code);
    assert_amd_ir_compiles(&result.code, AmdArch::Gfx950.mcpu());
}

/// Pipe `ir` through an `llvm-as` on PATH and assert it parses. Skips (returns)
/// when no `llvm-as` is installed, so the test is a no-op on machines without
/// LLVM tools.
fn assert_llvm_ir_assembles(ir: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    // Prefer the newest llvm-as available (versioned names first); a bare
    // `llvm-as` on PATH is often an old system build.
    let tool = ["llvm-as-20", "llvm-as-19", "llvm-as-18", "llvm-as-17", "llvm-as-16", "llvm-as-15", "llvm-as"]
        .into_iter()
        .find(|t| {
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

    // Pipe `src` through `tool` and return its exit status + stderr.
    let run = |src: &str| -> (bool, String) {
        let mut child = Command::new(tool)
            .args(["-o", "/dev/null", "-"])
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn llvm-as");
        child.stdin.take().unwrap().write_all(src.as_bytes()).expect("write IR to llvm-as");
        let out = child.wait_with_output().expect("wait for llvm-as");
        (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
    };

    // Our IR uses opaque pointers (`ptr`), which LLVM defaults to from v15 on.
    // A tool that can't parse them natively is LLVM ≤ 14 — which also predates
    // the gfx94x/gfx950 MFMA + fp8-conversion intrinsics these tests exercise,
    // so its verdict is unreliable. Skip rather than emit a false failure.
    let probe = "define i32 @p(ptr %x) {\n  %v = load i32, ptr %x\n  ret i32 %v\n}\n";
    if !run(probe).0 {
        eprintln!("skipping llvm-as smoke test: {tool} is too old for opaque-pointer / gfx94x IR");
        return;
    }

    let (ok, stderr) = run(ir);
    assert!(ok, "llvm-as rejected the emitted AMD IR:\n{ir}\n--- llvm-as stderr ---\n{stderr}");
}

fn assert_amd_ir_compiles(ir: &str, arch: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let has_target =
        Command::new("clang").arg("--print-targets").output().ok().filter(|out| out.status.success()).is_some_and(
            |out| String::from_utf8_lossy(&out.stdout).lines().any(|line| line.trim_start().starts_with("amdgcn")),
        );
    if !has_target {
        eprintln!("skipping gfx12 compile test: clang has no AMDGPU target");
        return;
    }

    let mcpu = format!("-mcpu={arch}");
    let mut child = Command::new("clang")
        .args([
            "-x",
            "ir",
            "-c",
            "-O2",
            "--target=amdgcn-amd-amdhsa",
            &mcpu,
            "-nogpulib",
            "-nogpuinc",
            "-Wno-override-module",
            "-",
            "-o",
            "/dev/null",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn clang");
    child.stdin.take().unwrap().write_all(ir.as_bytes()).expect("write AMD IR");
    let output = child.wait_with_output().expect("wait for clang");
    assert!(
        output.status.success(),
        "clang rejected emitted {arch} IR:\n{}\n{ir}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn test_custom_typed_statement_emits_ssa_assignment() {
    // A typed CUSTOM renders the RHS of an SSA assignment (`%v = <rhs>`); the
    // LLVM type lives in the RHS, so the template is a full instruction RHS.
    let a = UOp::param(0, 1, DType::Float32, None);
    let out = UOp::param(1, 1, DType::Float32, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let a_idx = UOp::index().buffer(a.clone()).indices(vec![idx.clone()]).call().unwrap();
    let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();
    let a_load = UOp::load().index(a_idx).call();
    let custom = UOp::custom(smallvec::smallvec![a_load], "fmul float {0}, 2.0".to_string(), DType::Float32);
    let store = out_idx.store(custom);
    let sink = UOp::sink(vec![store]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());

    let result = render(&linear, Some("custom_typed")).unwrap();
    println!("{}", result.code);
    assert!(result.code.contains("= fmul float"), "typed CUSTOM should emit an fmul assignment:\n{}", result.code);
    assert!(result.code.contains(", 2.0"), "template literal should survive:\n{}", result.code);
}

#[test]
fn test_customi_inline_is_substituted_into_consumer() {
    // CUSTOMI registers its formatted text as an operand and is inlined into
    // consumers rather than emitted as its own instruction.
    let out = UOp::param(0, 1, DType::Float32, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();
    let inline = UOp::customi(smallvec::SmallVec::new(), "4.0".to_string(), DType::Float32);
    let store = out_idx.store(inline);
    let sink = UOp::sink(vec![store]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink).into());

    let result = render(&linear, Some("customi_inline")).unwrap();
    println!("{}", result.code);
    assert!(result.code.contains("store float 4.0"), "CUSTOMI text should be inlined into the store:\n{}", result.code);
}

#[test]
fn test_custom_void_hoists_declare_to_module_prefix_amd() {
    // A Void CUSTOM emits raw IR lines; any `declare` is hoisted (deduplicated)
    // to the module prefix so custom bodies can reference arbitrary intrinsics.
    let out = UOp::param(0, 1, DType::Float32, None);
    let idx = UOp::const_(DType::Int32, ConstValue::Int(0));
    let out_idx = UOp::index().buffer(out.clone()).indices(vec![idx.clone()]).call().unwrap();
    let one = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let store = out_idx.store(one);
    let custom = UOp::custom(
        smallvec::SmallVec::new(),
        "declare void @llvm.amdgcn.s.barrier()\ncall void @llvm.amdgcn.s.barrier()".to_string(),
        DType::Void,
    );
    let sink = UOp::sink(vec![store, custom]);
    let result = render_amd_linearized(&sink, AmdArch::Gfx942, "custom_void");
    println!("{}", result.code);

    assert!(
        result.code.contains("declare void @llvm.amdgcn.s.barrier()"),
        "declare should be hoisted to the module prefix:\n{}",
        result.code
    );
    assert!(
        result.code.contains("call void @llvm.amdgcn.s.barrier()"),
        "the call body line should be emitted:\n{}",
        result.code
    );
    // The declare must appear exactly once even though it was inside the body.
    assert_eq!(
        result.code.matches("declare void @llvm.amdgcn.s.barrier()").count(),
        1,
        "declare should be deduplicated:\n{}",
        result.code
    );
}
