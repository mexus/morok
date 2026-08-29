//! C renderer tests for code generation verification.

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, ReduceOp, UOp, WmmaMetadata, WmmaUpcastAxes};

use crate::c::render;
use crate::c::types::c_const;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

fn concrete_range(end: i64, axis_type: AxisType) -> std::sync::Arc<UOp> {
    let end = UOp::const_(DType::Int32, ConstValue::Int(end));
    UOp::new(Op::Range { end, axis_id: AxisId::Renumbered(0), axis_type, deps: SmallVec::new() }, DType::Int32)
}

fn slotted_var(name: &str, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param { shape, arg } = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param { shape: shape.clone(), arg }, DType::Int32)
}

fn volatile_param(slot: usize, size: usize) -> std::sync::Arc<UOp> {
    let mut arg = ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None);
    arg.volatile = true;
    UOp::new(Op::Param { shape: UOp::index_const(size as i64), arg }, DType::Float32)
}

#[test]
fn c_signature_uses_canonical_mixed_param_slot_order() {
    let sink = UOp::sink(vec![
        slotted_var("high", 3),
        UOp::param(2, 1, DType::Float32, None),
        slotted_var("low", 1),
        UOp::param(0, 1, DType::Float32, None),
    ]);
    let rendered = render_linearized(&sink, Some("mixed_abi")).expect("render mixed ABI");
    assert_eq!(rendered.buffer_args.iter().map(|arg| arg.index).collect::<Vec<_>>(), vec![0, 2]);
    assert_eq!(rendered.var_names, vec!["low", "high"]);
    assert!(
        rendered
            .code
            .contains("void mixed_abi(float* restrict data0, const int data1, float* restrict data2, const int data3)"),
        "{}",
        rendered.code
    );
}

#[test]
fn c_qualifies_only_volatile_buffer_parameters() {
    let sink = UOp::sink(vec![volatile_param(0, 1), UOp::param(1, 1, DType::Float32, None)]);
    let rendered = render_linearized(&sink, Some("volatile_params")).expect("render volatile C ABI");
    assert!(
        rendered.code.contains("void volatile_params(volatile float* restrict data0, float* restrict data1)"),
        "{}",
        rendered.code
    );
}

#[test]
fn test_render_linear_input_succeeds() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());

    let rendered = render(&linear, Some("test_linear")).expect("C codegen from LINEAR should succeed");
    assert!(rendered.code.contains("test_linear"));
}

#[test]
fn grouped_shrink_renders_single_vector_load_and_store() {
    let shrink = |src| {
        UOp::new(
            Op::Shrink {
                src,
                offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
                sizes: UOp::const_(DType::Int32, ConstValue::Int(4)),
            },
            DType::Float32,
        )
    };
    let output = shrink(UOp::param(0, 8, DType::Float32, None));
    let input = shrink(UOp::param(1, 8, DType::Float32, None));
    let sink = UOp::sink(vec![output.store(UOp::load().index(input).call())]);

    let rendered = render_linearized(&sink, Some("grouped_memory")).expect("render grouped C memory");
    assert_eq!(rendered.code.matches("float4").count(), 4, "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn grouped_half_shrink_declares_shape_width_vector() {
    let shrink = |src| {
        UOp::new(
            Op::Shrink {
                src,
                offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
                sizes: UOp::const_(DType::Int32, ConstValue::Int(4)),
            },
            DType::Float16,
        )
    };
    let output = shrink(UOp::param(0, 8, DType::Float16, None));
    let input = shrink(UOp::param(1, 8, DType::Float16, None));
    let sink = UOp::sink(vec![output.store(UOp::load().index(input).call())]);

    let rendered = render_linearized(&sink, Some("grouped_half_memory")).expect("render grouped half C memory");
    assert!(rendered.code.contains("typedef _Float16 half4"), "{}", rendered.code);
    assert_eq!(rendered.code.matches("half4").count(), 4, "{}", rendered.code);
    assert!(!rendered.code.contains("void4"), "{}", rendered.code);
}

#[test]
fn clang_materializes_tinygrad_ssa_boundaries() {
    let out = UOp::param(0, 1, DType::Int32, None);
    let out_index =
        UOp::index().buffer(out).indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))]).call().unwrap();
    let selected = UOp::try_where(
        UOp::const_(DType::Bool, ConstValue::Bool(true)),
        UOp::const_(DType::Int32, ConstValue::Int(2)),
        UOp::const_(DType::Int32, ConstValue::Int(3)),
    )
    .unwrap();
    let rendered = render_linearized(&UOp::sink(vec![out_index.store(selected)]), Some("materialize_where"))
        .expect("render WHERE materialization");
    assert!(rendered.code.contains("int alu0 = (1 ? 2 : 3);"), "{}", rendered.code);
    assert!(rendered.code.contains("= alu0;"), "{}", rendered.code);

    let input = UOp::param(0, 1, DType::Float32, None);
    let output = UOp::param(1, 1, DType::Float32, None);
    let zero = UOp::const_(DType::Index, ConstValue::Int(0));
    let input_index = UOp::index().buffer(input).indices(vec![zero.clone()]).call().unwrap();
    let output_index = UOp::index().buffer(output).indices(vec![zero]).call().unwrap();
    let rendered = render_linearized(
        &UOp::sink(vec![output_index.store(UOp::load().index(input_index).call())]),
        Some("materialize_load"),
    )
    .expect("render LOAD materialization");
    assert!(rendered.code.contains("float val0 = *(data0 + 0LL);"), "{}", rendered.code);
    assert!(rendered.code.contains("= val0;"), "{}", rendered.code);

    let vector = UOp::vconst(vec![ConstValue::UInt(1); 4], DType::UInt32);
    let cast = vector.cast(DType::UInt32.vec(4).unwrap()).cast(DType::Int32.vec(4).unwrap());
    let rendered = render_linearized(&UOp::sink(vec![cast]), Some("materialize_vector_cast"))
        .expect("render vector CAST materialization");
    assert!(rendered.code.contains("int4 cast0 = __builtin_convertvector"), "{}", rendered.code);
}

#[test]
fn clang_preserves_shape_width_across_materialized_values_and_store_aliases() {
    let shaped = UOp::stack((0..4).map(|value| UOp::const_(DType::UInt32, ConstValue::UInt(value))).collect());
    let cast = shaped.cast(DType::Float32);
    let rendered = render_linearized(&UOp::sink(vec![cast]), Some("materialize_shaped_cast"))
        .expect("render scalar-dtype shaped CAST");
    assert!(rendered.code.contains("typedef float float4"), "{}", rendered.code);
    assert!(rendered.code.contains("float4 cast0 = __builtin_convertvector"), "{}", rendered.code);

    let output = UOp::new(
        Op::Shrink {
            src: UOp::param(0, 4, DType::Int32, None),
            offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
            sizes: UOp::const_(DType::Int32, ConstValue::Int(4)),
        },
        DType::Int32,
    );
    let value = UOp::stack((0..4).map(|value| UOp::const_(DType::Int32, ConstValue::Int(value))).collect()).detach();
    let rendered = render_linearized(&UOp::sink(vec![output.store(value)]), Some("store_shaped_alias"))
        .expect("render shaped alias STORE");
    assert!(rendered.code.contains("*((int4*)(data0 + 0)) ="), "{}", rendered.code);
}

#[test]
fn clang_stack_dereferences_address_carrying_index_lanes() {
    let shrink = UOp::new(
        Op::Shrink {
            src: UOp::param(0, 8, DType::Float32, None),
            offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
            sizes: UOp::const_(DType::Int32, ConstValue::Int(4)),
        },
        DType::Float32,
    );
    let lanes = (0..4)
        .map(|lane| {
            UOp::index()
                .buffer(shrink.clone())
                .indices(vec![UOp::const_(DType::Index, ConstValue::Int(lane))])
                .call()
                .unwrap()
        })
        .collect();
    let output = UOp::new(
        Op::Shrink {
            src: UOp::param(1, 8, DType::Float32, None),
            offsets: UOp::const_(DType::Int32, ConstValue::Int(0)),
            sizes: UOp::const_(DType::Int32, ConstValue::Int(4)),
        },
        DType::Float32,
    );
    let rendered =
        render_linearized(&UOp::sink(vec![output.store(UOp::stack(lanes).detach())]), Some("stack_index_lanes"))
            .expect("render STACK of address INDEX lanes");
    assert!(rendered.code.contains("(float4){*("), "{}", rendered.code);
    assert_c_compiles(&rendered.code);
}

#[test]
fn clang_preserves_address_casts_as_pointers() {
    let index = UOp::index()
        .buffer(UOp::param(0, 1, DType::Float32, None))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let address = index.cast(DType::Int32);
    let output = UOp::index()
        .buffer(UOp::param(1, 1, DType::Int32, None))
        .indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))])
        .call()
        .unwrap();
    let rendered =
        render_linearized(&UOp::sink(vec![output.store(UOp::load().index(address).call())]), Some("address_cast"))
            .expect("render address CAST");
    assert!(rendered.code.contains("((int*)(data0 + 0LL))"), "{}", rendered.code);
    assert!(!rendered.code.contains("__builtin_convertvector"), "{}", rendered.code);
}

#[test]
fn clang_vector_alignment_rounds_down_like_tinygrad() {
    let vector = UOp::const_(DType::Float32.vec(3).unwrap(), ConstValue::Float(0.0));
    let rendered = render_linearized(&UOp::sink(vec![vector]), Some("float3_alignment")).expect("render float3");

    assert!(
        rendered.code.contains("typedef float float3 __attribute__((aligned(8),ext_vector_type(3)))"),
        "{}",
        rendered.code
    );
}

#[test]
fn test_render_rejects_non_linear_inputs() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let program = UOp::program(sink.clone(), info, None, None, None);

    let err = render(&program, Some("test_program_input")).expect_err("PROGRAM input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");

    let err = render(&sink, Some("test_sink_input")).expect_err("SINK input must fail");
    assert!(format!("{err}").contains("expects LINEAR input"), "unexpected error: {err:?}");
}

#[test]
fn test_getaddr_must_be_resolved_before_codegen() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::UInt8);
    let address = buffer.getaddr(None);
    let linear = UOp::linear(vec![buffer, address].into());
    let err = render(&linear, Some("getaddr")).expect_err("GETADDR is an HCQ runtime op, not a kernel op");
    assert!(format!("{err}").contains("GetAddr"), "unexpected error: {err:?}");
}

#[test]
fn test_render_rejects_fnuz_without_fallback() {
    let constant = UOp::const_(DType::FP8E5M2FNUZ, ConstValue::Float(1.0));
    let linear = UOp::linear(vec![constant].into());
    let err = render(&linear, Some("fnuz")).expect_err("FNUZ rendering must fail");
    let message = format!("{err}");
    assert!(message.contains("does not support FP8E5M2FNUZ"), "unexpected error: {message}");
    assert!(message.contains("cannot use OCP FP8 decomposition or raw-byte fallback"), "unexpected error: {message}");
}

#[test]
fn c_constants_consume_committed_values_and_fp8_bits() {
    let f32_value = UOp::const_(DType::Float32, ConstValue::Float(-3.2));
    let Op::Const(f32_value) = f32_value.op() else { unreachable!() };
    assert_eq!(c_const(&f32_value.0, &DType::Float32), "-3.2e0f");

    let fp8 = UOp::const_(DType::FP8E4M3, ConstValue::Float(1.1875));
    let Op::Const(fp8) = fp8.op() else { unreachable!() };
    assert_eq!(c_const(&fp8.0, &DType::FP8E4M3), "58");
}

#[test]
fn test_range_end_basic() {
    let range = concrete_range(10, AxisType::Loop);
    let noop = UOp::noop();
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = noop.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render_linearized(&sink, Some("test_loop")).expect("C codegen failed");

    assert!(result.code.contains("for"), "Missing for loop:\n{}", result.code);
    assert!(result.code.contains("ridx0"), "Missing loop variable:\n{}", result.code);
    assert!(result.code.contains("< 10"), "Missing loop bound:\n{}", result.code);
}

#[test]
fn test_reduce_add_basic() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(5.0));
    let range = concrete_range(10, AxisType::Reduce);

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Add);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render_linearized(&sink, Some("test_reduce")).expect("C codegen failed");

    assert!(result.code.contains("acc"), "Missing accumulator:\n{}", result.code);
    assert!(result.code.contains("for"), "Missing for loop:\n{}", result.code);
    assert!(result.code.contains("+="), "Missing accumulation:\n{}", result.code);
    assert!(result.code.contains("0.0f"), "Missing identity value:\n{}", result.code);
}

#[test]
fn test_reduce_max() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(3.0));
    let range = concrete_range(5, AxisType::Reduce);

    let reduce = const_val.reduce(smallvec::smallvec![range.clone()], ReduceOp::Max);
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = reduce.end(ranges);
    let sink = UOp::sink(vec![end_op]);

    let result = render_linearized(&sink, Some("test_reduce_max")).expect("C codegen failed");

    assert!(result.code.contains("fmaxf"), "Missing fmaxf:\n{}", result.code);
}

#[test]
fn test_reduce_empty_ranges() {
    let const_val = UOp::const_(DType::Float32, ConstValue::Float(42.0));
    let reduce = const_val.reduce(smallvec::smallvec![], ReduceOp::Add);
    let sink = UOp::sink(vec![reduce]);

    let result = render_linearized(&sink, Some("test_reduce_empty"));
    assert!(result.is_ok(), "C codegen failed: {:?}", result.err());
}

#[test]
fn test_multi_index_requires_linearization() {
    let buffer = UOp::param(0, 1024, DType::Float32, None);
    let i = UOp::const_(DType::Index, ConstValue::Int(1));
    let j = UOp::const_(DType::Index, ConstValue::Int(2));
    let index = UOp::index().buffer(buffer).indices(vec![i, j]).call().unwrap();
    let sink = UOp::sink(vec![index]);

    let linear = UOp::linear(sink.toposort().into());
    let err = render(&linear, Some("test_multi_index_requires_linearization"))
        .expect_err("multi-index INDEX must surface as InvalidGraph");
    assert!(
        matches!(&err, crate::Error::InvalidGraph { reason } if reason.contains("linearized INDEX")),
        "expected InvalidGraph(linearized INDEX), got {err:?}",
    );
}

#[test]
fn test_gated_load_emits_conditional_dereference() {
    let buffer = UOp::param(0, 1024, DType::Float32, None);
    let out = UOp::param(1, 1024, DType::Float32, None);
    let idx = UOp::const_(DType::Index, ConstValue::Int(1));
    let gate = UOp::const_(DType::Bool, ConstValue::Bool(true));
    let index = UOp::index().buffer(buffer).indices(vec![idx]).call().unwrap();
    let alt = UOp::const_(DType::Float32, ConstValue::Float(7.0));
    let load = UOp::load().index(index).alt(alt).gate(gate).call();
    let out_idx = UOp::index().buffer(out).indices(vec![UOp::const_(DType::Index, ConstValue::Int(0))]).call().unwrap();
    let sink = UOp::sink(vec![out_idx.store(load)]);

    let rendered = render_linearized(&sink, Some("test_gated_load_emits_conditional_dereference"))
        .expect("C backend should render gated load");
    assert!(
        rendered.code.contains("1 ? *(data0 + 1LL) : 7.0f"),
        "gated load should conditionally evaluate the dereference:\n{}",
        rendered.code
    );
}

/// Helper to create AMX float32 WMMA metadata matching the APPLE_AMX TcConfig.
fn amx_f32_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_float_float".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: svod_ir::RendererDevice::AppleAmx,
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

/// Helper to create AMX mixed-precision (f16→f32) WMMA metadata.
fn amx_f16_to_f32_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_half_float".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float16,
        dtype_out: DType::Float32,
        device: svod_ir::RendererDevice::AppleAmx,
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

/// Helper to create AMX WMMA metadata with 2×2 tile grid.
fn amx_tile_grid_metadata() -> WmmaMetadata {
    WmmaMetadata {
        name: "WMMA_16_16_1_float_float_tile2x2".to_string(),
        dims: (16, 16, 1),
        dtype_in: DType::Float32,
        dtype_out: DType::Float32,
        device: svod_ir::RendererDevice::AppleAmx,
        threads: 1,
        upcast_axes: Some(WmmaUpcastAxes {
            a: vec![(svod_ir::AxisId::Renumbered(2), 256)],
            b: vec![(svod_ir::AxisId::Renumbered(2), 256)],
            c: vec![(svod_ir::AxisId::Renumbered(2), 256)],
        }),
        reduce_axes: vec![],
        tile_grid: (2, 2),
    }
}

#[test]
fn test_wmma_preamble_macros() {
    // Construct a minimal WMMA node: a(float16) × b(float16) + c(float256) → float256
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify AMX macros are emitted
    assert!(result.code.contains("#define AMX_SET"), "Missing AMX_SET macro:\n{}", result.code);
    assert!(result.code.contains("#define AMX("), "Missing AMX macro:\n{}", result.code);
}

#[test]
fn test_wmma_preamble_static_function() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify static wrapper function is emitted with correct signature
    assert!(
        result
            .code
            .contains("static float256 __WMMA_16_16_1_float_float(float16 data1, float16 data2, float256 data0)"),
        "Missing or incorrect static WMMA function signature:\n{}",
        result.code,
    );
    // Verify AMX instructions inside the static function
    assert!(result.code.contains("AMX_SET(0)"), "Missing AMX_SET(0) init:\n{}", result.code);
    assert!(result.code.contains("AMX_SET(1)"), "Missing AMX_SET(1) finalize:\n{}", result.code);
    assert!(result.code.contains("AMX(12,"), "Missing fma32 instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(0,"), "Missing ldx instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(1,"), "Missing ldy instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(4,"), "Missing ldz instruction:\n{}", result.code);
    assert!(result.code.contains("AMX(5,"), "Missing stz instruction:\n{}", result.code);
}

#[test]
fn test_wmma_function_call() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify the kernel body contains a WMMA function call
    assert!(result.code.contains("__WMMA_16_16_1_float_float("), "Missing WMMA function call:\n{}", result.code);
    assert!(
        result.code.lines().any(|line| line.trim_start().starts_with("float256 wmma") && line.contains(" = __WMMA")),
        "WMMA result width was lost:\n{}",
        result.code
    );
}

#[test]
fn test_wmma_vector_typedefs() {
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma")).expect("C codegen failed");

    // Verify vector typedefs for float16 and float256 are emitted
    assert!(result.code.contains("typedef float float16"), "Missing float16 typedef:\n{}", result.code);
    assert!(result.code.contains("typedef float float256"), "Missing float256 typedef:\n{}", result.code);
}

#[test]
fn test_wmma_mixed_precision_flag() {
    // f16 × f16 → f32 requires bit 62 set in FMA encoding
    let zero = UOp::const_(DType::Float16, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = UOp::const_(DType::Float32, ConstValue::Float(0.0)).broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_f16_to_f32_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma_mixed")).expect("C codegen failed");

    // Verify fma16 opcode (15) is used
    assert!(result.code.contains("AMX(15,"), "Missing fma16 opcode:\n{}", result.code);

    // Verify bit 62 is set: 1<<62 = 4611686018427387904 in decimal
    assert!(
        result.code.contains("4611686018427387904ull"),
        "Missing mixed-precision bit 62 flag in FMA encoding:\n{}",
        result.code
    );
}

#[test]
fn test_wmma_tile_grid_load_pair() {
    // 2×2 tile grid should enable load-pair mode on LDX/LDY
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_tile_grid_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma_tile")).expect("C codegen failed");

    // Verify load-pair bit (bit 62) is set on LDX and LDY
    // 1<<62 = 4611686018427387904 in decimal
    assert!(
        result.code.contains("AMX(0, (int *)(&data2), 4611686018427387904ull)"),
        "Missing load-pair bit on LDX (opcode 0):\n{}",
        result.code
    );
    assert!(
        result.code.contains("AMX(1, (int *)(&data1), 4611686018427387904ull)"),
        "Missing load-pair bit on LDY (opcode 1):\n{}",
        result.code
    );
}

#[test]
fn test_wmma_tile_grid_multiple_fma() {
    // 2×2 tile grid should emit 4 FMAs with proper encoding
    let zero = UOp::const_(DType::Float32, ConstValue::Float(0.0));
    let a = zero.broadcast(16);
    let b = zero.broadcast(16);
    let c = zero.broadcast(256);

    let wmma = UOp::wmma(a, b, c, amx_tile_grid_metadata());
    let sink = UOp::sink(vec![wmma]);

    let result = render_linearized(&sink, Some("test_wmma_multi_fma")).expect("C codegen failed");

    // Count FMA calls - should be 4 for a 2x2 tile grid
    let fma_count = result.code.matches("AMX(12,").count();
    assert_eq!(fma_count, 4, "Expected 4 FMAs for 2x2 tile grid, got {}:\n{}", fma_count, result.code);

    // Verify FMA encodings for each tile position
    // encoding = fma_flags | (z_row << 20) | (x_off << 10) | y_off
    // where x_off = tx * 64, y_off = ty * 64

    // Tile (0,0): z_row=0, x_off=0, y_off=0 → encoding = 0
    assert!(result.code.contains("AMX(12, 0, 0ull);"), "Missing FMA for tile (0,0):\n{}", result.code);

    // Tile (0,1): z_row=1, x_off=64, y_off=0 → encoding = (1<<20) | (64<<10) | 0 = 1114112
    assert!(result.code.contains("AMX(12, 0, 1114112ull);"), "Missing FMA for tile (0,1):\n{}", result.code);

    // Tile (1,0): z_row=2, x_off=0, y_off=64 → encoding = (2<<20) | (0<<10) | 64 = 2097216
    assert!(result.code.contains("AMX(12, 0, 2097216ull);"), "Missing FMA for tile (1,0):\n{}", result.code);

    // Tile (1,1): z_row=3, x_off=64, y_off=64 → encoding = (3<<20) | (64<<10) | 64 = 3211328
    assert!(result.code.contains("AMX(12, 0, 3211328ull);"), "Missing FMA for tile (1,1):\n{}", result.code);
}

#[test]
fn test_custom_statement_is_materialized() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let expr = UOp::custom(smallvec::smallvec![one], "({0} + 3)".to_string(), DType::Int32);
    let stmt = UOp::custom(smallvec::smallvec![expr], "sink({0})".to_string(), DType::Void);
    let sink = UOp::sink(vec![stmt]);

    let result = render_linearized(&sink, Some("test_custom_stmt")).expect("C codegen failed");

    assert!(
        result.code.contains("int custom0 = (1 + 3);"),
        "CUSTOM should materialize to a statement:\n{}",
        result.code
    );
    assert!(
        result.code.contains("sink(custom0);"),
        "CUSTOM consumer should reference materialized value:\n{}",
        result.code
    );
}

#[test]
fn test_customi_is_inline_and_formats_placeholders() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c = UOp::const_(DType::Int32, ConstValue::Int(3));
    let inline = UOp::customi(smallvec::smallvec![a, b, c], "{0} + {2} + {1}".to_string(), DType::Int32);
    let stmt = UOp::custom(smallvec::smallvec![inline], "emit({0})".to_string(), DType::Void);
    let sink = UOp::sink(vec![stmt]);

    let result = render_linearized(&sink, Some("test_customi_inline")).expect("C codegen failed");

    assert!(
        result.code.contains("emit(1 + 3 + 2);"),
        "CUSTOMI should stay inline and substitute placeholders in-order:\n{}",
        result.code
    );
    assert!(!result.code.contains("custom0 ="), "CUSTOMI must not create temp statements:\n{}", result.code);
}

#[test]
fn test_custom_template_rejects_out_of_bounds_placeholder() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let bad = UOp::custom(smallvec::smallvec![one], "emit({1})".to_string(), DType::Void);
    let sink = UOp::sink(vec![bad]);

    let err = render_linearized(&sink, Some("test_custom_bad_index")).expect_err("out-of-bounds placeholder must fail");
    assert!(format!("{err}").contains("out of bounds"), "unexpected error: {err}");
}

#[test]
fn test_custom_template_rejects_unmatched_brace() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let bad = UOp::custom(smallvec::smallvec![one], "emit({0".to_string(), DType::Void);
    let sink = UOp::sink(vec![bad]);

    let err = render_linearized(&sink, Some("test_custom_unmatched_brace")).expect_err("unmatched braces must fail");
    assert!(format!("{err}").contains("unmatched"), "unexpected error: {err}");
}

#[test]
fn test_custom_template_rejects_mixed_auto_and_manual_placeholders() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let bad = UOp::customi(smallvec::smallvec![a, b], "{} + {1}".to_string(), DType::Int32);
    let sink = UOp::sink(vec![bad]);

    let err = render_linearized(&sink, Some("test_custom_mixed_placeholders"))
        .expect_err("mixed placeholder modes must fail");
    assert!(format!("{err}").contains("mixes automatic"), "unexpected error: {err}");
}

/// Pipe `src` through `clang -fsyntax-only` and assert it parses. Skips when no
/// clang is on PATH, so the test is a no-op on machines without a C compiler.
/// Mirrors `assert_llvm_ir_assembles` in `llvm_text.rs`.
fn assert_c_compiles(src: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};

    let Ok(mut child) = Command::new("clang")
        .args(["-fsyntax-only", "-Wno-unused-value", "-x", "c", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
    else {
        eprintln!("skipping C compile check: no clang on PATH");
        return;
    };
    child.stdin.take().unwrap().write_all(src.as_bytes()).expect("write C source to clang");
    let output = child.wait_with_output().expect("wait for clang");
    assert!(
        output.status.success(),
        "clang rejected the emitted C:\n{src}\n--- clang stderr ---\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
