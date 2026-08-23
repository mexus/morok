//! LLVM renderer tests for loop and reduction codegen.

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, Op, ParamArg, UOp};

use crate::llvm::common::lconst;
use crate::llvm::text::render;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

fn volatile_param(slot: usize, size: usize) -> std::sync::Arc<UOp> {
    let mut arg = ParamArg::buffer(slot, DType::Float32, AddrSpace::Global, None);
    arg.volatile = true;
    UOp::new(Op::Param { shape: UOp::native_const(size as i32), arg }, DType::Float32)
}

#[test]
fn test_render_linear_input_succeeds() {
    let sink = UOp::sink(vec![UOp::const_(DType::Float32, ConstValue::Float(1.0))]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());

    let rendered = render(&linear, Some("test_linear")).expect("LLVM codegen from LINEAR should succeed");
    assert!(rendered.code.contains("test_linear"));
}

#[test]
fn llvm_constants_use_committed_storage_bits() {
    let half = UOp::const_(DType::Float16, ConstValue::Float(1.0 / 123_008.0));
    let Op::Const(half) = half.op() else { unreachable!() };
    assert_eq!(lconst(&half.0, &DType::Float16), "0xH0088");

    let fp8 = UOp::const_(DType::FP8E4M3, ConstValue::Float(1.1875));
    let Op::Const(fp8) = fp8.op() else { unreachable!() };
    assert_eq!(lconst(&fp8.0, &DType::FP8E4M3), "58");
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

    let rendered = render_linearized(&sink, Some("grouped_memory")).expect("render grouped LLVM memory");
    assert_eq!(rendered.code.matches("load <4 x float>").count(), 1, "{}", rendered.code);
    assert_eq!(rendered.code.matches("store <4 x float>").count(), 1, "{}", rendered.code);
}

#[test]
fn volatile_scalar_and_grouped_memory_accesses_render_explicitly() {
    let index = UOp::native_const(0i32);
    let scalar_input = UOp::index().buffer(volatile_param(1, 1)).indices(vec![index.clone()]).call().unwrap();
    let scalar_output = UOp::index().buffer(volatile_param(0, 1)).indices(vec![index]).call().unwrap();
    let scalar = UOp::sink(vec![scalar_output.store(UOp::load().index(scalar_input).call())]);
    let scalar = render_linearized(&scalar, Some("volatile_scalar")).expect("render volatile scalar LLVM");
    assert!(scalar.code.contains("load volatile float"), "{}", scalar.code);
    assert!(scalar.code.contains("store volatile float"), "{}", scalar.code);

    let shrink = |src| {
        UOp::new(Op::Shrink { src, offsets: UOp::native_const(0i32), sizes: UOp::native_const(4i32) }, DType::Float32)
    };
    let grouped_input = shrink(volatile_param(1, 8));
    let grouped_output = shrink(volatile_param(0, 8));
    let grouped = UOp::sink(vec![grouped_output.store(UOp::load().index(grouped_input).call())]);
    let grouped = render_linearized(&grouped, Some("volatile_grouped")).expect("render volatile grouped LLVM");
    assert!(grouped.code.contains("load volatile <4 x float>"), "{}", grouped.code);
    assert!(grouped.code.contains("store volatile <4 x float>"), "{}", grouped.code);
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

/// Test basic RANGE/END loop codegen.
///
/// Creates the equivalent of:
/// ```c
/// for (int i = 0; i < 10; i++) {
///     // empty body
/// }
/// ```
#[test]
fn test_range_end_basic() {
    // Create range: for i in 0..10
    let end = UOp::const_(DType::Int64, ConstValue::Int(10));
    let range = UOp::new(
        Op::Range { end, axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop, deps: SmallVec::new() },
        DType::Int64,
    );

    // Create a NOOP as the computation (empty loop body)
    let noop = UOp::noop();

    // End the loop - END wraps computation and references the range
    let ranges: SmallVec<[_; 4]> = smallvec::smallvec![range];
    let end_op = noop.end(ranges);

    // Wrap in SINK
    let sink = UOp::sink(vec![end_op]);

    // Render to LLVM IR
    let result = render_linearized(&sink, Some("test_loop"));
    if let Err(ref e) = result {
        eprintln!("Codegen failed: {:?}", e);
    }
    assert!(result.is_ok(), "Codegen failed: {:?}", result.err());

    let kernel = result.unwrap();
    let ir = &kernel.code;

    // Verify loop structure in generated IR (Tinygrad-style: entry/latch/body/footer/exit)
    // Block names use axis_id which varies, so just check for the patterns
    assert!(ir.contains("loop_entry_"), "Missing entry block:\n{}", ir);
    assert!(ir.contains("loop_latch_"), "Missing latch block:\n{}", ir);
    assert!(ir.contains("loop_body_"), "Missing body block:\n{}", ir);
    assert!(ir.contains("loop_footer_"), "Missing footer block:\n{}", ir);
    assert!(ir.contains("loop_exit_"), "Missing exit block:\n{}", ir);
    assert!(ir.contains("phi i64"), "Missing PHI node:\n{}", ir);
}

/// Test basic REDUCE codegen with sum operation.
///
/// Creates the equivalent of:
/// ```c
/// float acc = 0.0;
/// for (int i = 0; i < 10; i++) {
///     acc += 5.0;  // constant value
/// }
/// return acc;  // should be 50.0
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
fn shaped_reg_index_renders_as_memory_load() {
    let reg = UOp::buffer(0, 4, DType::Float32, AddrSpace::Reg, None);
    let index = UOp::index().buffer(reg).indices(vec![UOp::const_(DType::Int32, ConstValue::Int(2))]).call().unwrap();
    let sink = UOp::sink(vec![UOp::load().index(index).call()]);

    let result = render_linearized(&sink, Some("shaped_reg_load")).unwrap();
    assert!(result.code.contains("getelementptr inbounds float, ptr %reg0, i32 2"), "{}", result.code);
    assert!(result.code.contains("load float, ptr"), "{}", result.code);
    assert!(!result.code.contains("extractelement <4 x float> %reg0"), "{}", result.code);
}

#[test]
fn shaped_reg_after_index_renders_as_memory_load() {
    let reg = UOp::buffer(0, 4, DType::Float32, AddrSpace::Reg, None);
    let after = reg.after(smallvec::smallvec![UOp::noop()]);
    assert_eq!(after.addrspace(), Some(AddrSpace::Reg));
    let index = UOp::index().buffer(after).indices(vec![UOp::const_(DType::Int32, ConstValue::Int(2))]).call().unwrap();
    let sink = UOp::sink(vec![UOp::load().index(index).call()]);

    let result = render_linearized(&sink, Some("shaped_reg_after_load")).unwrap();
    assert!(result.code.contains("getelementptr inbounds float, ptr %reg0, i32 2"), "{}", result.code);
    assert!(result.code.contains("load float, ptr"), "{}", result.code);
    assert!(!result.code.contains("extractelement <4 x float> %reg0"), "{}", result.code);
}

#[test]
fn test_custom_renders_typed_statement_in_llvm_backend() {
    let one = UOp::const_(DType::Int32, ConstValue::Int(1));
    let custom = UOp::custom(smallvec::smallvec![one], "add i32 {0}, 3".to_string(), DType::Int32);
    let sink = UOp::sink(vec![custom]);

    let result = render_linearized(&sink, Some("test_custom")).expect("LLVM backend should render CUSTOM");
    assert!(result.code.contains("= add i32 1, 3"), "typed CUSTOM should render its RHS:\n{}", result.code);
}

#[test]
fn test_customi_inlines_into_consumer_in_llvm_backend() {
    let a = UOp::const_(DType::Int32, ConstValue::Int(1));
    let b = UOp::const_(DType::Int32, ConstValue::Int(2));
    let c = UOp::const_(DType::Int32, ConstValue::Int(3));
    // `{2}` selects the third dep (const 3); CUSTOMI is inlined as the operand
    // string "3" into the consuming CUSTOM rather than emitting its own line.
    let inline = UOp::customi(smallvec::smallvec![a, b, c], "{2}".to_string(), DType::Int32);
    let custom = UOp::custom(smallvec::smallvec![inline], "add i32 {0}, 10".to_string(), DType::Int32);
    let sink = UOp::sink(vec![custom]);

    let result = render_linearized(&sink, Some("test_customi")).expect("LLVM backend should render CUSTOMI");
    assert!(
        result.code.contains("= add i32 3, 10"),
        "CUSTOMI operand should be inlined into the consumer:\n{}",
        result.code
    );
}
