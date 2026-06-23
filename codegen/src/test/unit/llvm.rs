//! LLVM renderer tests for loop and reduction codegen.

use smallvec::SmallVec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{AxisId, AxisType, ConstValue, ReduceOp, UOp};

use crate::llvm::text::render;

fn render_linearized(root: &std::sync::Arc<UOp>, name: Option<&str>) -> crate::Result<crate::RenderedKernel> {
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(root.clone()).into());
    render(&linear, name)
}

#[test]
fn test_render_linear_input_succeeds() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());

    let rendered = render(&linear, Some("test_linear")).expect("LLVM codegen from LINEAR should succeed");
    assert!(rendered.code.contains("test_linear"));
}

#[test]
fn test_render_rejects_non_linear_inputs() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let program = UOp::program(sink.clone(), UOp::device(DeviceSpec::Cpu), None, None, None);

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
    let end = UOp::const_(DType::Index, ConstValue::Int(10));
    let range = UOp::range_axis(end, AxisId::Renumbered(0), AxisType::Loop);

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
    let ptr_dtype = DType::Float32.ptr(None, svod_dtype::AddrSpace::Global).unwrap();
    let buffer = UOp::param(0, 1024, ptr_dtype, None);
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
