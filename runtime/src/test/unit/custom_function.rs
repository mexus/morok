use super::*;

use svod_dtype::DType;
use svod_ir::CustomFunctionKind;

fn buffer(dtype: DType, shape: Vec<usize>) -> Buffer {
    Buffer::new(svod_device::registry::cpu().expect("cpu allocator"), dtype, shape, Default::default())
}

fn f32_buffer(values: &[f32]) -> Buffer {
    let mut buffer = buffer(DType::Float32, vec![values.len()]);
    buffer.copyin(&values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>()).unwrap();
    buffer
}

fn read_f32(buffer: &Buffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.size()];
    buffer.copyout(&mut bytes).unwrap();
    bytes.as_chunks::<4>().0.iter().copied().map(f32::from_le_bytes).collect()
}

fn allreduce(op: svod_ir::ReduceOp, buffers: &mut [Buffer]) -> crate::Result<()> {
    run_custom_function(&CustomFunctionKind::AllReduce { reduce_op: op }, &[], buffers, &HashMap::new())
}

/// The reserved kinds report a typed `Unsupported` carrying their shape, and do
/// so before touching the buffer list — a bufferless call must not fail first.
#[test_case::test_case(CustomFunctionKind::EncDec, 1, 2, "EncDec", "attrs=1, buffers=2"; "encdec with attrs and buffers")]
#[test_case::test_case(CustomFunctionKind::EncDec, 0, 0, "EncDec", "attrs=0, buffers=0"; "encdec with neither")]
#[test_case::test_case(CustomFunctionKind::Graph, 0, 0, "Graph", "attrs=0, buffers=0"; "graph")]
fn reserved_kinds_report_typed_unsupported(
    kind: CustomFunctionKind,
    attrs: usize,
    buffers: usize,
    expected_kind: &str,
    expected_reason: &str,
) {
    let attrs = (0..attrs).map(|i| svod_ir::UOp::index_const(i as i64)).collect::<Vec<_>>();
    let mut buffers = (0..buffers).map(|_| buffer(DType::Float32, vec![4])).collect::<Vec<_>>();

    let err = run_custom_function(&kind, &attrs, &mut buffers, &HashMap::new())
        .expect_err("reserved kinds report unsupported runtime behavior");

    match err {
        crate::Error::Unsupported { kind, reason } => {
            assert_eq!(kind, expected_kind);
            assert!(reason.contains(expected_reason), "unexpected reason: {reason}");
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
}

#[test_case::test_case(svod_ir::ReduceOp::Add, vec![5.0, 3.0, 7.0]; "sum")]
#[test_case::test_case(svod_ir::ReduceOp::Max, vec![4.0, 5.0, 9.0]; "max")]
fn host_allreduce_reduces_shards_numerically(op: svod_ir::ReduceOp, expected: Vec<f32>) {
    let mut buffers = vec![f32_buffer(&[0.0; 3]), f32_buffer(&[1.0, 5.0, -2.0]), f32_buffer(&[4.0, -2.0, 9.0])];
    allreduce(op, &mut buffers).unwrap();
    assert_eq!(read_f32(&buffers[0]), expected);
}

/// Sub-word floats accumulate on their own storage grid, not a widened one:
/// f16 rounds each partial sum to f16, and bf16 saturates to +inf rather than
/// staying finite the way an f32 accumulator would.
#[test]
fn host_allreduce_accumulates_on_the_storage_dtype_grid() {
    let f16 = |values: &[f64]| {
        let mut buffer = buffer(DType::Float16, vec![values.len()]);
        let bytes = values
            .iter()
            .flat_map(|value| {
                (svod_dtype::cast::committed_float_bits(*value, svod_dtype::ScalarDType::Float16).unwrap() as u16)
                    .to_le_bytes()
            })
            .collect::<Vec<_>>();
        buffer.copyin(&bytes).unwrap();
        buffer
    };
    let mut buffers = vec![f16(&[0.0, 0.0]), f16(&[1.5, -2.0]), f16(&[2.25, 5.0])];
    allreduce(svod_ir::ReduceOp::Add, &mut buffers).unwrap();
    let mut bytes = vec![0; buffers[0].size()];
    buffers[0].copyout(&mut bytes).unwrap();
    let values = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| svod_dtype::cast::f16_bits_to_float(u16::from_le_bytes(*chunk)))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![3.75, 3.0]);

    let bf16 = |bits: u16| {
        let mut buffer = buffer(DType::BFloat16, vec![1]);
        buffer.copyin(&bits.to_le_bytes()).unwrap();
        buffer
    };
    const BF16_MAX_FINITE: u16 = 0x7f7f;
    const BF16_INF: u16 = 0x7f80;
    let mut buffers = vec![bf16(0), bf16(BF16_MAX_FINITE), bf16(BF16_MAX_FINITE)];
    allreduce(svod_ir::ReduceOp::Add, &mut buffers).unwrap();
    let mut bytes = [0; 2];
    buffers[0].copyout(&mut bytes).unwrap();
    assert_eq!(u16::from_le_bytes(bytes), BF16_INF);
}

#[test]
fn host_allreduce_rejects_shape_and_element_alignment_mismatches() {
    let mut shape_mismatch =
        vec![buffer(DType::Float32, vec![2]), buffer(DType::Float32, vec![1, 2]), buffer(DType::Float32, vec![2])];
    let err = allreduce(svod_ir::ReduceOp::Add, &mut shape_mismatch).unwrap_err();
    assert!(matches!(err, crate::Error::Execution { reason } if reason.contains("identical dtype, shape")));

    let base = buffer(DType::Float32, vec![1]);
    let odd = || base.view(0, 3).unwrap();
    let mut unaligned = vec![odd(), odd(), odd()];
    let err = allreduce(svod_ir::ReduceOp::Add, &mut unaligned).unwrap_err();
    assert!(matches!(err, crate::Error::Execution { reason } if reason.contains("not aligned")));
}
