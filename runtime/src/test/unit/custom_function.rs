use super::*;

use svod_dtype::DType;
use svod_ir::CustomFunctionKind;

#[test]
fn test_encdec_returns_typed_unsupported() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let dst = Buffer::new(alloc.clone(), DType::Float32, vec![4], Default::default());
    let src = Buffer::new(alloc, DType::Float32, vec![4], Default::default());

    let attr = svod_ir::UOp::index_const(7);
    let mut bufs = vec![dst, src];
    let err = run_custom_function(&CustomFunctionKind::EncDec, &[attr], &mut bufs, &HashMap::new())
        .expect_err("encdec should report unsupported runtime behavior");

    match err {
        crate::Error::Unsupported { kind, reason } => {
            assert_eq!(kind, "EncDec");
            assert!(reason.contains("attrs=1"), "unexpected reason: {reason}");
        }
        other => panic!("expected Error::Unsupported, got {other:?}"),
    }
}

#[test]
fn test_graph_returns_typed_unsupported() {
    let mut no_buffers = Vec::<Buffer>::new();
    let err = run_custom_function(&CustomFunctionKind::Graph, &[], &mut no_buffers, &HashMap::new())
        .expect_err("graph should report unsupported runtime behavior");
    assert!(matches!(err, crate::Error::Unsupported { kind, .. } if kind == "Graph"));
}

#[test]
fn test_encdec_unsupported_does_not_require_buffers_first() {
    let mut no_buffers = Vec::<Buffer>::new();
    let err = run_custom_function(&CustomFunctionKind::EncDec, &[], &mut no_buffers, &HashMap::new())
        .expect_err("encdec should fail as unsupported");
    assert!(matches!(err, crate::Error::Unsupported { .. }));
}

fn f32_buffer(values: &[f32]) -> Buffer {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let mut buffer = Buffer::new(alloc, DType::Float32, vec![values.len()], Default::default());
    let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
    buffer.copyin(&bytes).unwrap();
    buffer
}

fn read_f32(buffer: &Buffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.size()];
    buffer.copyout(&mut bytes).unwrap();
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap())).collect()
}

#[test]
fn host_allreduce_executes_sum_and_max_numerically() {
    for (op, expected) in [(svod_ir::ReduceOp::Add, vec![5.0, 3.0, 7.0]), (svod_ir::ReduceOp::Max, vec![4.0, 5.0, 9.0])]
    {
        let mut buffers = vec![f32_buffer(&[0.0; 3]), f32_buffer(&[1.0, 5.0, -2.0]), f32_buffer(&[4.0, -2.0, 9.0])];
        run_custom_function(&CustomFunctionKind::AllReduce { reduce_op: op }, &[], &mut buffers, &HashMap::new())
            .unwrap();
        assert_eq!(read_f32(&buffers[0]), expected);
    }
}

#[test]
fn host_allreduce_executes_float16_sum_on_storage_grid() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let make = |values: &[f64]| {
        let mut buffer = Buffer::new(alloc.clone(), DType::Float16, vec![values.len()], Default::default());
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
    let mut buffers = vec![make(&[0.0, 0.0]), make(&[1.5, -2.0]), make(&[2.25, 5.0])];
    run_custom_function(
        &CustomFunctionKind::AllReduce { reduce_op: svod_ir::ReduceOp::Add },
        &[],
        &mut buffers,
        &HashMap::new(),
    )
    .unwrap();
    let mut bytes = vec![0; buffers[0].size()];
    buffers[0].copyout(&mut bytes).unwrap();
    let values = bytes
        .chunks_exact(2)
        .map(|chunk| svod_dtype::cast::f16_bits_to_float(u16::from_le_bytes(chunk.try_into().unwrap())))
        .collect::<Vec<_>>();
    assert_eq!(values, vec![3.75, 3.0]);
}

#[test]
fn host_allreduce_bfloat16_sum_overflows_on_float32_grid() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let make = |bits: u16| {
        let mut buffer = Buffer::new(alloc.clone(), DType::BFloat16, vec![1], Default::default());
        buffer.copyin(&bits.to_le_bytes()).unwrap();
        buffer
    };
    let max_finite = 0x7f7f;
    let mut buffers = vec![make(0), make(max_finite), make(max_finite)];

    run_custom_function(
        &CustomFunctionKind::AllReduce { reduce_op: svod_ir::ReduceOp::Add },
        &[],
        &mut buffers,
        &HashMap::new(),
    )
    .unwrap();

    let mut bytes = [0; 2];
    buffers[0].copyout(&mut bytes).unwrap();
    assert_eq!(u16::from_le_bytes(bytes), 0x7f80);
}

#[test]
fn host_allreduce_rejects_shape_and_element_alignment_mismatches() {
    let alloc = svod_device::registry::cpu().expect("cpu allocator");
    let make = |shape: &[usize]| Buffer::new(alloc.clone(), DType::Float32, shape.to_vec(), Default::default());

    let mut shape_mismatch = vec![make(&[2]), make(&[1, 2]), make(&[2])];
    let err = run_custom_function(
        &CustomFunctionKind::AllReduce { reduce_op: svod_ir::ReduceOp::Add },
        &[],
        &mut shape_mismatch,
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(matches!(err, crate::Error::Execution { reason } if reason.contains("identical dtype, shape")));

    let base = make(&[1]);
    let odd = || base.view(0, 3).unwrap();
    let mut unaligned = vec![odd(), odd(), odd()];
    let err = run_custom_function(
        &CustomFunctionKind::AllReduce { reduce_op: svod_ir::ReduceOp::Add },
        &[],
        &mut unaligned,
        &HashMap::new(),
    )
    .unwrap_err();
    assert!(matches!(err, crate::Error::Execution { reason } if reason.contains("not aligned")));
}
