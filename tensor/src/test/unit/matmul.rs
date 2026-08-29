use crate::*;
use ndarray::{Array2, array};
use svod_dtype::DType;
use svod_schedule::{
    BeamConfig, HeuristicsConfig, OptStrategy, OptimizerConfig, TcOptLevel, TcSelect, testing::setup_test_tracing,
};

fn prep_config(optimizer: OptimizerConfig) -> PrepareConfig {
    optimizer.into()
}
fn env_config() -> PrepareConfig {
    PrepareConfig::from_env()
}

/// Helper to compare svod result against ndarray reference with tolerance.
fn assert_matmul_close(actual: &[f32], expected: &Array2<f32>, tol: f32) {
    let expected_flat: Vec<f32> = expected.iter().copied().collect();
    assert_eq!(actual.len(), expected_flat.len(), "Length mismatch: {} != {}", actual.len(), expected_flat.len());

    for (i, (a, e)) in actual.iter().zip(expected_flat.iter()).enumerate() {
        assert!((a - e).abs() < tol, "Mismatch at index {}: svod={} vs ndarray={} (diff: {})", i, a, e, (a - e).abs());
    }
}

/// Helper to run validated square matmul test for a given size.
fn run_validated_square_matmul(size: usize, tol: f32) {
    // Use prime modulos to create varied but reproducible data
    let a_data: Vec<f32> = (0..size * size).map(|x| ((x % 31) as f32) * 0.05 - 0.8).collect();
    let b_data: Vec<f32> = (0..size * size).map(|x| ((x % 37) as f32) * 0.04 - 0.7).collect();

    let a_nd = Array2::from_shape_vec((size, size), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((size, size), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, tol);
}

/// Helper to run validated non-square matmul test.
fn run_validated_matmul(m: usize, k: usize, n: usize, tol: f32) {
    let a_data: Vec<f32> = (0..m * k).map(|x| ((x % 41) as f32) * 0.04 - 0.8).collect();
    let b_data: Vec<f32> = (0..k * n).map(|x| ((x % 43) as f32) * 0.035 - 0.7).collect();

    let a_nd = Array2::from_shape_vec((m, k), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((k, n), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape[0].as_const().unwrap(), m, "Output shape mismatch");
    assert_eq!(c_shape[1].as_const().unwrap(), n, "Output shape mismatch");

    let expected = a_nd.dot(&b_nd);

    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, tol);
}

// =========================================================================
// Validated matmul tests (codegen required)
// =========================================================================

crate::codegen_tests! {
    fn test_matmul_validated_2x2(config) {
        // Simple 2x2 matmul with known values
        let a_nd = Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let b_nd = Array2::from_shape_vec((2, 2), vec![5.0f32, 6.0, 7.0, 8.0]).unwrap();

        // Compute with svod
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        // Compute reference with ndarray
        let expected = a_nd.dot(&b_nd);

        // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    fn test_matmul_int8_returns_narrow_dtype(config) {
        // int8·int8 must return int8 (the promoted operand dtype), not the widened
        // int32 sum accumulator. [[1,2],[3,4]]·[[5,6],[7,8]] = [[19,22],[43,50]] (fit i8).
        let a = Tensor::from_ndarray(&Array2::from_shape_vec((2, 2), vec![1.0f32, 2.0, 3.0, 4.0]).unwrap())
            .cast(DType::Int8)
            .unwrap();
        let b = Tensor::from_ndarray(&Array2::from_shape_vec((2, 2), vec![5.0f32, 6.0, 7.0, 8.0]).unwrap())
            .cast(DType::Int8)
            .unwrap();
        let mut c = a.matmul(&b).unwrap();
        assert_eq!(c.uop().dtype(), DType::Int8, "int8 matmul must return int8, not the int32 accumulator");
        c.realize_with(&config).unwrap();
        assert_eq!(c.as_vec::<i8>().unwrap(), vec![19i8, 22, 43, 50]);
    }

    fn test_matmul_validated_3x3(config) {
        // 3x3 matmul with sequential values
        let a_data: Vec<f32> = (1..=9).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (10..=18).map(|x| x as f32).collect();

        let a_nd = Array2::from_shape_vec((3, 3), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((3, 3), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-4);
    }

    fn test_matmul_validated_2x3_3x4(config) {
        // [2, 3] @ [3, 4] -> [2, 4]
        let a_data: Vec<f32> = (1..=6).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

        let a_nd = Array2::from_shape_vec((2, 3), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((3, 4), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-4);
    }

    fn test_matmul_validated_tall_wide(config) {
        // [4, 2] @ [2, 5] -> [4, 5]
        let a_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.5).collect();
        let b_data: Vec<f32> = (1..=10).map(|x| x as f32 * 0.3).collect();

        let a_nd = Array2::from_shape_vec((4, 2), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((2, 5), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }

    fn test_matmul_validated_16x16(config) {
        // Larger matrix to test vectorization paths
        const SIZE: usize = 16;
        let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.1).collect();
        let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| (x as f32) * 0.05 + 1.0).collect();

        let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-3);
    }

    fn test_matmul_validated_32x32(config) {
        // Test with 32x32 to exercise more optimization paths
        const SIZE: usize = 32;
        let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 17) as f32) * 0.1 - 0.8).collect();
        let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x % 13) as f32) * 0.15 - 0.5).collect();

        let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
        let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd);
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-2);
    }

    fn test_dot_product_validated(config) {
        // 1D @ 1D dot product
        let a_data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b_data = [2.0f32, 3.0, 4.0, 5.0, 6.0];

        let a = Tensor::from_slice(a_data);
        let b = Tensor::from_slice(b_data);
        let mut c = a.dot(&b).unwrap();
        c.realize_with(&config).unwrap();

        // Expected: 1*2 + 2*3 + 3*4 + 4*5 + 5*6 = 2 + 6 + 12 + 20 + 30 = 70
        let expected: f32 = a_data.iter().zip(b_data.iter()).map(|(a, b)| a * b).sum();

        assert_eq!(c.shape().unwrap().len(), 0, "Dot product should be scalar");
        let result = c.as_vec::<f32>().unwrap();
        assert!((result[0] - expected).abs() < 1e-5, "Expected {}, got {}", expected, result[0]);
    }

    fn test_vector_matrix_validated(config) {
        // [4] @ [4, 3] -> [3]
        let v_data = [1.0f32, 2.0, 3.0, 4.0];
        let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();

        let v = Tensor::from_slice(v_data);
        let m_nd = Array2::from_shape_vec((4, 3), m_data).unwrap();
        let m = Tensor::from_ndarray(&m_nd);
        let mut c = v.dot(&m).unwrap();
        c.realize_with(&config).unwrap();

        // ndarray: need to treat vector as [1, 4] @ [4, 3] -> [1, 3], then squeeze
        let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
        let expected = v_nd.dot(&m_nd);

        assert_eq!(c.shape().unwrap()[0].as_const().unwrap(), 3);
        let svod_result = c.as_vec::<f32>().unwrap();
        for (i, (a, e)) in svod_result.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
        }
    }

    fn test_matrix_vector_validated(config) {
        // [3, 4] @ [4] -> [3]
        let m_data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let v_data = [1.0f32, 2.0, 3.0, 4.0];

        let m_nd = Array2::from_shape_vec((3, 4), m_data).unwrap();
        let m = Tensor::from_ndarray(&m_nd);
        let v = Tensor::from_slice(v_data);
        let mut c = m.dot(&v).unwrap();
        c.realize_with(&config).unwrap();

        let v_nd = ndarray::Array1::from_vec(v_data.to_vec());
        let expected = m_nd.dot(&v_nd);

        assert_eq!(c.shape().unwrap()[0].as_const().unwrap(), 3);
        let svod_result = c.as_vec::<f32>().unwrap();
        for (i, (a, e)) in svod_result.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, a, e);
        }
    }

    fn test_matmul_identity_validated(config) {
        // A @ I = A
        let a_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let identity_data = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0];

        let a_nd = Array2::from_shape_vec((4, 4), a_data.clone()).unwrap();
        let i_nd = Array2::from_shape_vec((4, 4), identity_data).unwrap();
        let a = Tensor::from_ndarray(&a_nd);
        let i = Tensor::from_ndarray(&i_nd);
        let mut c = a.matmul(&i).unwrap();
        c.realize_with(&config).unwrap();
        let svod_result = c.as_vec::<f32>().unwrap();

        // Result should equal original A
        for (i, (actual, expected)) in svod_result.iter().zip(a_data.iter()).enumerate() {
            assert!((actual - expected).abs() < 1e-5, "Mismatch at index {}: {} != {}", i, actual, expected);
        }
    }

    fn test_matmul_negative_values_validated(config) {
        // Test with negative values to ensure sign handling
        let a_nd = Array2::from_shape_vec((2, 3), vec![-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0]).unwrap();
        let b_nd = Array2::from_shape_vec((3, 2), vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0]).unwrap();

        let a = Tensor::from_ndarray(&a_nd);
        let b = Tensor::from_ndarray(&b_nd).try_transpose(0, 1).unwrap();
        let b = b.try_transpose(0, 1).unwrap(); // Back to [3, 2] but contiguous
        let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

        let expected = a_nd.dot(&b_nd);

        assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-5);
    }
}

// ========== Basic 2D x 2D Tests ==========

#[test]
fn test_matmul_2d_basic() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 2);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_matmul_2d_non_square() {
    // [2, 3] @ [3, 4] → [2, 4]
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((3, 4)));
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 2);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 4);
}

#[test]
fn test_matmul_alias() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    // Test that matmul is an alias for dot
    let c1 = a.dot(&b).unwrap();
    let c2 = a.matmul(&b).unwrap();

    assert_eq!(c1.shape().unwrap().len(), c2.shape().unwrap().len());
}

// ========== 1D x 1D Tests (Dot Product) ==========

#[test]
fn test_dot_product_1d() {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = a.dot(&b).unwrap();

    // Result should be scalar (0D tensor)
    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 0);
}

#[test]
fn test_dot_product_orthogonal() {
    let a = Tensor::from_slice([1.0f32, 0.0, 0.0]);
    let b = Tensor::from_slice([0.0f32, 1.0, 0.0]);
    let c = a.dot(&b).unwrap();

    // Orthogonal vectors → dot product = 0
    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 0);
}

// ========== 1D x 2D and 2D x 1D Tests ==========

#[test]
fn test_vector_matrix() {
    // [3] @ [3, 4] → [4]
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((3, 4)));
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 1);
    assert_eq!(c_shape[0].as_const().unwrap(), 4);
}

#[test]
fn test_matrix_vector() {
    // [2, 3] @ [3] → [2]
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
    let b = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 1);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
}

// ========== Batched Matmul Tests ==========

#[test]
fn test_batched_matmul_3d() {
    // [2, 3, 4] @ [2, 4, 5] → [2, 3, 5]
    let a = Tensor::from_ndarray(&ndarray::Array3::<f32>::ones((2, 3, 4)));
    let b = Tensor::from_ndarray(&ndarray::Array3::<f32>::ones((2, 4, 5)));
    let c = a.dot(&b).unwrap();

    let c_shape = c.shape().unwrap();
    assert_eq!(c_shape.len(), 3);
    assert_eq!(c_shape[0].as_const().unwrap(), 2);
    assert_eq!(c_shape[1].as_const().unwrap(), 3);
    assert_eq!(c_shape[2].as_const().unwrap(), 5);
}

// ========== Edge Cases ==========

#[test]
fn test_matmul_error_0d() {
    let scalar = Tensor::from_ndarray(&ndarray::Array0::<f32>::from_elem((), 1.0));
    let vector = Tensor::from_slice([1.0f32, 2.0, 3.0]);

    // 0D tensors not supported
    assert!(scalar.dot(&vector).is_err());
    assert!(vector.dot(&scalar).is_err());
}

#[test]
fn test_matmul_error_shape_mismatch() {
    // [2, 3] @ [4, 5] - inner dimensions don't match
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((4, 5)));

    let result = a.dot(&b);
    assert!(result.is_err());
}

#[test]
fn test_matmul_identity() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let identity = Tensor::from_ndarray(&array![[1.0f32, 0.0], [0.0, 1.0]]);

    let result = a.dot(&identity).unwrap();

    // Result shape should match input
    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 2);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

// ========== Dtype Tests ==========

#[test]
fn test_matmul_dtype_promotion() {
    let a = Tensor::from_ndarray(&array![[1i32, 2], [3, 4]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    let c = a.dot(&b).unwrap();
    // Result should be promoted to float32
    assert_eq!(c.uop().dtype(), DType::Float32);
}

#[test]
fn test_matmul_explicit_dtype() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);

    // Use float64 accumulation
    let c = a.matmul_with().other(&b).dtype(DType::Float64).call().unwrap();
    assert_eq!(c.uop().dtype(), DType::Float64);
}

/// RDNA4 follows Tinygrad 8c8b43de's tensor-core table: FP8 storage is
/// emulated as bytes, arithmetic is widened to f16, and the resulting matmul
/// uses the f16->f32 gfx12 WMMA. This is compile-only and never opens a GPU.
#[test]
fn test_matmul_fp8_gfx1201_decomposes_to_f16_wmma_compile_only() {
    use svod_dtype::{AmdArch, DeviceSpec, ScalarDType};
    use svod_ir::{Op, RendererDevice};
    use svod_schedule::{OptimizerRenderer, optimize_kernel_with_config};

    for dtype in [DType::FP8E4M3, DType::FP8E5M2, DType::FP8E4M3FNUZ, DType::FP8E5M2FNUZ] {
        let a = Tensor::empty(&[16, 16], dtype.clone());
        let b = Tensor::empty(&[16, 16], dtype.clone());
        let c = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("tensor FP8 matmul");

        let rangeified = svod_schedule::rangeify_with_map(svod_ir::UOp::sink(vec![c.uop().contiguous()]))
            .expect("rangeify FP8 tensor matmul");
        let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("split FP8 kernels");
        let pre = crate::schedule::create_pre_schedule(kernel_graph).expect("prepare FP8 tensor schedule");
        assert_eq!(pre.items.len(), 1, "FP8 matmul must remain one tensor kernel");

        let heuristics = HeuristicsConfig::builder().tc_select(TcSelect::Index(0)).matvec_enabled(false).build();
        let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(AmdArch::Gfx1201);
        let opt_renderer = OptimizerRenderer::for_amd_arch(AmdArch::Gfx1201).with_rewrite_capabilities(
            svod_ir::RendererOps::all(),
            svod_codegen::traits::Renderer::decompositor(&renderer),
            None,
        );
        let optimized = optimize_kernel_with_config(pre.items[0].ast.clone(), &opt_renderer, &config)
            .expect("gfx1201 FP8 decomposition and TC optimization");

        let nodes = optimized.toposort();
        let wmma = nodes
            .iter()
            .find_map(|u| match u.op() {
                Op::Wmma { metadata, .. } => Some(metadata),
                _ => None,
            })
            .expect("decomposed FP8 matmul must select f16 WMMA");
        assert_eq!((wmma.dtype_in.clone(), wmma.dtype_out.clone()), (DType::Float16, DType::Float32));
        assert_eq!((wmma.device, wmma.threads), (RendererDevice::AmdRdna4, 32));
        assert!(
            !nodes.iter().any(|u| u.dtype().base() == dtype.base()),
            "{dtype:?} arithmetic must be fully decomposed"
        );
        assert!(
            nodes.iter().any(|u| matches!(u.op(), Op::Param { arg, .. } if arg.dtype.base() == ScalarDType::UInt8)),
            "{dtype:?} storage must remain byte-addressed"
        );

        let program = svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Amd { device_id: 0 })
            .expect("final target graph");
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize gfx1201 FP8 matmul");
        let linear =
            linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR stage");
        let rendered = svod_codegen::traits::Renderer::render(&renderer, &linear, Some("matmul_fp8_gfx1201"))
            .expect("render decomposed gfx1201 matmul");
        assert!(
            rendered.code.contains("llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v8f16"),
            "{dtype:?} must select gfx12 f16 WMMA"
        );
        assert!(!rendered.code.contains("16x16x16.fp8"), "{dtype:?} must not claim native FP8 WMMA");
        assert!(!rendered.code.contains("16x16x16.bf8"), "{dtype:?} must not alias E5M2 to native BF8 WMMA");
        if svod_runtime::amd::compile::has_amdgpu_target() {
            let object = svod_runtime::amd::compile::compile_ir_to_amd_object(&rendered.code, AmdArch::Gfx1201)
                .expect("compile decomposed gfx1201 matmul");
            assert_eq!(&object[..4], b"\x7fELF");
        }
    }
}

/// Positive control for native OCP FP8: gfx950 keeps E4M3/E5M2 operands and
/// selects its scaled K=128 MFMA. Compile-only; no hardware is opened.
#[test]
fn test_matmul_ocp_fp8_gfx950_native_mfma_compile_only() {
    use svod_dtype::{AmdArch, DeviceSpec};
    use svod_ir::Op;
    use svod_schedule::{OptimizerRenderer, optimize_kernel_with_config};

    for dtype in [DType::FP8E4M3, DType::FP8E5M2] {
        let a = Tensor::empty(&[16, 128], dtype.clone());
        let b = Tensor::empty(&[128, 16], dtype.clone());
        let c = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("native OCP FP8 tensor matmul");
        let rangeified = svod_schedule::rangeify_with_map(svod_ir::UOp::sink(vec![c.uop().contiguous()]))
            .expect("rangeify native FP8 tensor matmul");
        let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("split native FP8 kernels");
        let pre = crate::schedule::create_pre_schedule(kernel_graph).expect("prepare native FP8 schedule");

        let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(AmdArch::Gfx950);
        let opt_renderer = OptimizerRenderer::for_amd_arch(AmdArch::Gfx950).with_rewrite_capabilities(
            svod_ir::RendererOps::all(),
            svod_codegen::traits::Renderer::decompositor(&renderer),
            None,
        );
        let tc_index = opt_renderer
            .tensor_cores
            .iter()
            .position(|tc| tc.dtype_in == dtype && tc.dtype_out == DType::Float32 && tc.dims == (16, 16, 128))
            .expect("gfx950 scaled FP8 tensor core");
        let heuristics = HeuristicsConfig::builder().tc_select(TcSelect::Index(tc_index)).matvec_enabled(false).build();
        let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
        let optimized = optimize_kernel_with_config(pre.items[0].ast.clone(), &opt_renderer, &config)
            .expect("gfx950 native FP8 TC optimization");
        let metadata = optimized
            .toposort()
            .into_iter()
            .find_map(|u| match u.op() {
                Op::Wmma { metadata, .. } => Some(metadata.clone()),
                _ => None,
            })
            .expect("native FP8 matmul must emit MFMA");
        assert_eq!(
            (metadata.dims, metadata.dtype_in, metadata.dtype_out),
            ((16, 16, 128), dtype.clone(), DType::Float32)
        );

        let program = svod_codegen::program_pipeline::program_from_sink(optimized, DeviceSpec::Amd { device_id: 0 })
            .expect("final target graph");
        let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize gfx950 FP8 matmul");
        let linear =
            linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR stage");
        let rendered = svod_codegen::traits::Renderer::render(&renderer, &linear, Some("matmul_fp8_gfx950"))
            .expect("render native gfx950 FP8 matmul");
        assert!(rendered.code.contains("llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4"));
        let format = if dtype == DType::FP8E5M2 { "i32 1, i32 1" } else { "i32 0, i32 0" };
        assert!(rendered.code.contains(format), "wrong scaled format selectors for {dtype:?}");
        if svod_runtime::amd::compile::has_amdgpu_target() {
            let object = svod_runtime::amd::compile::compile_ir_to_amd_object(&rendered.code, AmdArch::Gfx950)
                .expect("compile native gfx950 FP8 matmul");
            assert_eq!(&object[..4], b"\x7fELF");
        }
    }
}

/// Compile-only regression for tinygrad 8c8b43de's padded tensor-core path.
/// M=5 is padded to gfx1151's 16-row WMMA tile; no GPU is opened or submitted.
#[test]
fn test_matmul_m5_gfx1151_padded_wmma_compile_only() {
    use svod_dtype::{AmdArch, DeviceSpec};
    use svod_ir::{BinaryOp, ConstValue, Op, RendererDevice};
    use svod_schedule::{OptimizerRenderer, optimize_kernel_with_config};

    let a = Tensor::empty(&[5, 16], DType::Float16);
    let b = Tensor::empty(&[16, 16], DType::Float16);
    let c = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("tensor matmul");

    let rangeified = svod_schedule::rangeify_with_map(svod_ir::UOp::sink(vec![c.uop().contiguous()]))
        .expect("rangeify tensor matmul");
    let (kernel_graph, _) = svod_schedule::try_get_kernel_graph(rangeified.sink).expect("split kernels");
    let pre = crate::schedule::create_pre_schedule(kernel_graph).expect("prepare tensor schedule");
    assert_eq!(pre.items.len(), 1, "matmul must remain one tensor kernel");

    let heuristics = HeuristicsConfig::builder()
        .tc_opt(TcOptLevel::Padded)
        .tc_select(TcSelect::Index(0))
        .matvec_enabled(false)
        .build();
    let config = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
    let opt_renderer = OptimizerRenderer::for_amd_arch(AmdArch::Gfx1151).with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        None,
        None,
    );
    let optimized = optimize_kernel_with_config(pre.items[0].ast.clone(), &opt_renderer, &config)
        .expect("gfx1151 padded tensor-core optimization");

    let nodes = optimized.toposort();
    assert!(
        nodes.iter().all(|u| !matches!(u.op(), Op::Reduce { .. })),
        "pre-coalescing M=5 WMMA must not retain an operand-side range as a residual REDUCE",
    );
    let params = nodes
        .iter()
        .filter_map(|u| match u.op() {
            Op::Param { shape, arg } => Some((arg.slot, u.dtype(), shape.vmax().try_int())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        params,
        vec![(0, DType::Float32, Some(80)), (1, DType::Float16, Some(80)), (2, DType::Float16, Some(256))],
        "kernel ABI must be C[5x16], A[5x16], B[16x16]",
    );

    let (wmma_node, wmma) = nodes
        .iter()
        .find_map(|u| match u.op() {
            Op::Wmma { metadata, .. } => Some((u, metadata)),
            _ => None,
        })
        .expect("pinned TC must emit WMMA");
    assert_eq!(wmma.dims, (16, 16, 16));
    assert_eq!((wmma.dtype_in.clone(), wmma.dtype_out.clone()), (DType::Float16, DType::Float32));
    assert_eq!((wmma.device, wmma.threads), (RendererDevice::AmdRdna3, 32));
    assert!(wmma.upcast_axes.is_none(), "expander must consume WMMA axis metadata");
    let Op::Wmma { c: accumulator, .. } = wmma_node.op() else { unreachable!() };
    assert_eq!(
        accumulator.shape().unwrap().unwrap().last().and_then(|extent| extent.as_const()),
        Some(8),
        "gfx1151 must retain its eight-register hardware accumulator fragment",
    );

    fn eval_lane(u: &std::sync::Arc<svod_ir::UOp>, lane: i64) -> i64 {
        match u.op() {
            Op::Const(value) => value.0.try_int().expect("integer constant"),
            Op::Special { name, .. } if name == "lidx0" => lane,
            Op::Cast { src, .. } => eval_lane(src, lane),
            Op::Binary(op, a, b) => {
                let (a, b) = (eval_lane(a, lane), eval_lane(b, lane));
                match op {
                    BinaryOp::Add => a + b,
                    BinaryOp::Mul => a * b,
                    BinaryOp::And => a & b,
                    BinaryOp::Shl => a << b,
                    BinaryOp::Shr => a >> b,
                    _ => panic!("unexpected lane-index operation {op:?}"),
                }
            }
            Op::Ternary(svod_ir::TernaryOp::MulAcc, a, b, c) => {
                eval_lane(a, lane) * eval_lane(b, lane) + eval_lane(c, lane)
            }
            _ => panic!("unexpected lane-index node {}", u.op().as_ref()),
        }
    }
    fn eval_gate(u: &std::sync::Arc<svod_ir::UOp>, lane: i64) -> bool {
        match u.op() {
            Op::Binary(BinaryOp::Lt, lhs, rhs) => eval_lane(lhs, lane) < eval_lane(rhs, lane),
            _ => panic!("unexpected validity gate {}", u.op().as_ref()),
        }
    }
    fn memory_param_slot(index: &std::sync::Arc<svod_ir::UOp>) -> Option<usize> {
        let buffer = match index.op() {
            Op::Index { buffer, .. } => buffer,
            Op::Shrink { src, .. } => src,
            _ => return None,
        };
        match buffer.op() {
            Op::Param { arg, .. } => Some(arg.slot),
            _ => None,
        }
    }
    let is_zero =
        |u: &std::sync::Arc<svod_ir::UOp>| matches!(u.op(), Op::Const(value) if value.0 == ConstValue::Float(0.0));
    let is_zero_stack = |u: &std::sync::Arc<svod_ir::UOp>, lanes: usize| matches!(u.op(), Op::Stack { sources } if sources.len() == lanes && sources.iter().all(is_zero));
    let is_lidx_lt = |gate: &std::sync::Arc<svod_ir::UOp>, bound: i64| {
        matches!(gate.op(), Op::Binary(BinaryOp::Lt, lhs, rhs)
            if rhs.vmin().try_int() == Some(bound)
                && rhs.vmax().try_int() == Some(bound)
                && lhs.toposort().iter().any(|u| matches!(u.op(), Op::Special { end, name }
                    if name == "lidx0" && end.vmax().try_int() == Some(32))))
    };

    let mut a_loads = Vec::new();
    let mut b_loads = Vec::new();
    let mut output_stores = Vec::new();
    for u in &nodes {
        if let Op::Load { index, alt, gate } = u.op() {
            match memory_param_slot(index) {
                Some(1) => {
                    let Op::Shrink { offsets, sizes, .. } = index.op() else {
                        panic!("every A load must use a shaped SHRINK address: {}", u.tree())
                    };
                    assert_eq!(
                        u.shape().unwrap().unwrap().iter().filter_map(|extent| extent.as_const()).collect::<Vec<_>>(),
                        [4]
                    );
                    assert_eq!(eval_lane(sizes, 0), 4, "each pinned A access must contain four shaped lanes");
                    let alt = alt.as_ref().expect("padded A loads require a shaped zero alternative");
                    let gate = gate.as_ref().expect("padded A loads require a validity gate");
                    assert!(is_zero_stack(alt, 4), "every invalid padded A lane must contribute zero: {}", alt.tree());
                    assert!(
                        is_lidx_lt(gate, 5),
                        "A loads must be guarded by the padded row < M predicate: {}",
                        gate.tree()
                    );
                    a_loads.push((offsets.clone(), gate.clone()));
                }
                Some(2) => {
                    let Op::Index { indices, .. } = index.op() else {
                        panic!("every B load must use a scalar INDEX address: {}", u.tree())
                    };
                    assert_eq!(indices.len(), 1, "B loads must have one address expression");
                    assert!(alt.is_none() && gate.is_none(), "unpadded B fragment loads must remain ungated");
                    assert!(u.shape().unwrap().unwrap().is_empty(), "B loads must remain scalar");
                    b_loads.push(indices[0].clone());
                }
                slot => panic!("unexpected or malformed load in pinned WMMA graph (slot {slot:?}): {}", u.tree()),
            }
        }
        if let Op::Store { index, value, gate } = u.op() {
            assert_eq!(memory_param_slot(index), Some(0), "only C stores are permitted in this fixture: {}", u.tree());
            let Op::Index { indices, .. } = index.op() else { panic!("C stores must use scalar INDEX addresses") };
            assert_eq!(indices.len(), 1);
            let Op::Index { buffer, indices: value_indices } = value.op() else {
                panic!("C store value must index the WMMA result: {}", value.tree())
            };
            assert!(std::sync::Arc::ptr_eq(buffer, wmma_node), "C store must consume this graph's WMMA accumulator");
            assert_eq!(value_indices.len(), 1);
            output_stores.push((indices[0].clone(), gate.clone(), value_indices[0].clone()));
        }
    }
    assert_eq!(a_loads.len(), 4, "pinned WMMA A fragment must contain four shaped loads");
    let mut loaded_a = std::collections::BTreeSet::new();
    let mut padded_a = std::collections::BTreeSet::new();
    for (offset, gate) in &a_loads {
        for lane in 0..32 {
            for shaped_lane in 0..4 {
                let index = eval_lane(offset, lane) + shaped_lane;
                assert!((0..256).contains(&index), "raw padded A tile index must stay within 16x16");
                if eval_gate(gate, lane) {
                    assert!((0..80).contains(&index), "enabled A load must stay within the real 5x16 allocation");
                    loaded_a.insert(index);
                } else {
                    assert!((80..256).contains(&index), "only physical padded-tail A lanes may be disabled");
                    padded_a.insert(index);
                }
            }
        }
    }
    assert_eq!(loaded_a, (0..80).collect(), "enabled A loads must cover exactly the real allocation");
    assert_eq!(padded_a, (80..256).collect(), "disabled zero-fill lanes must cover exactly the padded A tail");

    assert_eq!(b_loads.len(), 16, "WMMA B fragment must retain its 16 scalar column loads");
    let mut loaded_b = std::collections::BTreeMap::new();
    for index in &b_loads {
        for lane in 0..32 {
            let index = eval_lane(index, lane);
            assert!((0..256).contains(&index), "B fragment address must stay within B[0..256]");
            *loaded_b.entry(index).or_insert(0usize) += 1;
        }
    }
    assert_eq!(loaded_b.keys().copied().collect::<std::collections::BTreeSet<_>>(), (0..256).collect());
    assert!(
        loaded_b.values().all(|count| *count == 2),
        "wave32 duplicates each B value across its two WMMA lane halves"
    );

    assert_eq!(output_stores.len(), 3, "WMMA must cover C with three per-lane output fragments");
    assert_eq!(output_stores.iter().filter(|(_, gate, _)| gate.is_some()).count(), 1);
    let mut stored_c = std::collections::BTreeSet::new();
    let mut padded_c = std::collections::BTreeSet::new();
    let mut result_lanes = std::collections::BTreeSet::new();
    for (index, gate, value_index) in &output_stores {
        result_lanes.insert(eval_lane(value_index, 0));
        if let Some(gate) = gate {
            assert!(is_lidx_lt(gate, 16), "the partial C fragment must be guarded by lane < 16: {}", gate.tree());
        }
        for lane in 0..32 {
            let index = eval_lane(index, lane);
            if gate.as_ref().is_none_or(|gate| eval_gate(gate, lane)) {
                assert!((0..80).contains(&index), "enabled C store must stay within the 5x16 allocation");
                assert!(stored_c.insert(index), "C index {index} must be stored exactly once");
            } else {
                assert!((80..96).contains(&index), "only the final C fragment tail may be disabled");
                padded_c.insert(index);
            }
        }
    }
    assert_eq!(stored_c, (0..80).collect(), "stores must cover exactly C[0..80]");
    assert_eq!(padded_c, (80..96).collect(), "the partial C fragment must gate exactly C[80..96]");
    assert_eq!(result_lanes, [0, 1, 2].into_iter().collect(), "C stores must consume three WMMA result lanes");

    let program =
        svod_codegen::program_pipeline::program_from_sink(optimized.clone(), DeviceSpec::Amd { device_id: 0 })
            .expect("final target graph");
    let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize gfx1151 program");
    let linear = linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR stage");
    let Op::Linear { ops } = linear.op() else { unreachable!() };
    let linear_wmma = ops.iter().find(|u| matches!(u.op(), Op::Wmma { .. })).expect("LINEAR WMMA");
    let mut linear_a_loads = Vec::new();
    let mut linear_b_loads = Vec::new();
    let mut linear_output_stores = Vec::new();
    let mut linear_ifs = Vec::new();
    let mut linear_endifs = Vec::new();
    for (position, u) in ops.iter().enumerate() {
        match u.op() {
            Op::Load { index, alt, gate } => match memory_param_slot(index) {
                Some(1) => {
                    let Op::Shrink { offsets, sizes, .. } = index.op() else {
                        panic!("LINEAR A load must retain shaped SHRINK address: {}", u.tree())
                    };
                    assert_eq!(eval_lane(sizes, 0), 4);
                    let alt = alt.as_ref().expect("LINEAR A load zero alternate");
                    let gate = gate.as_ref().expect("LINEAR A load row gate");
                    assert!(is_zero_stack(alt, 4), "LINEAR A alternate must remain four zeros");
                    assert!(is_lidx_lt(gate, 5), "LINEAR A load must retain row<5 gate");
                    linear_a_loads.push((u, index, offsets, gate, alt));
                }
                Some(2) => {
                    let Op::Index { indices, .. } = index.op() else {
                        panic!("LINEAR B load must use scalar INDEX: {}", u.tree())
                    };
                    assert_eq!(indices.len(), 1);
                    assert!(alt.is_none() && gate.is_none(), "LINEAR B loads must remain ungated");
                    assert!(u.shape().unwrap().unwrap().is_empty(), "LINEAR B loads must remain scalar");
                    linear_b_loads.push((u, &indices[0]));
                }
                slot => panic!("unexpected or malformed LINEAR load (slot {slot:?}): {}", u.tree()),
            },
            Op::Store { index, value, gate } => {
                assert_eq!(memory_param_slot(index), Some(0), "only C stores are permitted in LINEAR: {}", u.tree());
                assert!(gate.is_none(), "LINEAR cleanup must move C gates to IF/ENDIF");
                let Op::Index { indices, .. } = index.op() else { panic!("LINEAR C store must use scalar INDEX") };
                assert_eq!(indices.len(), 1);
                let Op::Index { buffer, indices: value_indices } = value.op() else {
                    panic!("LINEAR C value must index WMMA result: {}", value.tree())
                };
                assert!(std::sync::Arc::ptr_eq(buffer, linear_wmma), "LINEAR C value must consume LINEAR WMMA result");
                assert_eq!(value_indices.len(), 1);
                linear_output_stores.push((position, u, index, &indices[0], &value_indices[0]));
            }
            Op::If { condition, body } => linear_ifs.push((position, u, condition, body)),
            Op::EndIf { if_op } => linear_endifs.push((position, u, if_op)),
            _ => {}
        }
    }
    assert_eq!(linear_a_loads.len(), 4, "all four shaped A accesses must retain row<5 zero-fill gating");
    let mut linear_loaded_a = std::collections::BTreeSet::new();
    let mut linear_padded_a = std::collections::BTreeSet::new();
    for (_, _, offsets, gate, _) in &linear_a_loads {
        for lane in 0..32 {
            for shaped_lane in 0..4 {
                let address = eval_lane(offsets, lane) + shaped_lane;
                if eval_gate(gate, lane) {
                    assert!((0..80).contains(&address), "LINEAR true gate reached padded A address {address}");
                    linear_loaded_a.insert(address);
                } else {
                    assert!((80..256).contains(&address), "LINEAR false A gate must own only padded addresses");
                    linear_padded_a.insert(address);
                }
            }
        }
    }
    assert_eq!(linear_loaded_a, (0..80).collect(), "LINEAR A loads must still cover exactly A[0..80]");
    assert_eq!(linear_padded_a, (80..256).collect(), "LINEAR A zero-fill must cover exactly padded A[80..256]");
    assert_eq!(linear_b_loads.len(), 16, "LINEAR must enumerate all sixteen scalar B loads");
    let mut linear_loaded_b = std::collections::BTreeMap::new();
    for (_, index) in &linear_b_loads {
        for lane in 0..32 {
            *linear_loaded_b.entry(eval_lane(index, lane)).or_insert(0usize) += 1;
        }
    }
    assert_eq!(linear_loaded_b.keys().copied().collect::<std::collections::BTreeSet<_>>(), (0..256).collect());
    assert!(linear_loaded_b.values().all(|count| *count == 2), "LINEAR B values require wave32 two-way replication");

    assert_eq!(linear_output_stores.len(), 3, "post-coalescing M=5 WMMA must keep Tinygrad's three distinct C stores");
    assert_eq!(linear_ifs.len(), 1, "LINEAR must contain exactly the partial C store IF");
    assert_eq!(linear_endifs.len(), 1, "LINEAR must contain exactly the partial C store ENDIF");
    let (if_position, if_node, if_condition, if_body) = linear_ifs[0];
    let (endif_position, endif_node, endif_owner) = linear_endifs[0];
    assert!(is_lidx_lt(if_condition, 16), "partial C store IF must be lane<16");
    assert!(
        std::sync::Arc::ptr_eq(endif_owner, if_node),
        "ENDIF must reference the partial-store IF by source identity"
    );
    assert_eq!(if_body.len(), 1, "partial-store IF must own exactly one address dependency");
    let guarded_store = linear_output_stores
        .iter()
        .find(|(position, _, _, _, _)| *position == if_position + 1)
        .expect("partial C store must immediately follow IF");
    assert_eq!(endif_position, if_position + 2, "ENDIF must immediately follow the partial C store");
    assert!(
        std::sync::Arc::ptr_eq(&if_body[0], guarded_store.2),
        "IF body must own the partial store address by identity"
    );
    assert_eq!(
        ops[if_position + 1..endif_position].iter().filter(|u| matches!(u.op(), Op::Store { .. })).count(),
        1,
        "partial-store IF must not enclose an unrelated store",
    );

    let mut linear_stored_c = std::collections::BTreeSet::new();
    let mut linear_padded_c = std::collections::BTreeSet::new();
    let mut linear_result_lanes = std::collections::BTreeSet::new();
    for (position, _, _, index, value_index) in &linear_output_stores {
        let gate = (*position == guarded_store.0).then_some(if_condition);
        linear_result_lanes.insert(eval_lane(value_index, 0));
        for lane in 0..32 {
            let address = eval_lane(index, lane);
            if gate.is_none_or(|gate| eval_gate(gate, lane)) {
                assert!((0..80).contains(&address), "LINEAR enabled C store escaped C allocation");
                assert!(linear_stored_c.insert(address), "LINEAR C address {address} stored more than once");
            } else {
                assert!((80..96).contains(&address), "LINEAR disabled C address must be final padded tail");
                linear_padded_c.insert(address);
            }
        }
    }
    assert_eq!(linear_stored_c, (0..80).collect(), "LINEAR stores must cover exactly C[0..80]");
    assert_eq!(linear_padded_c, (80..96).collect(), "LINEAR guard must disable exactly C[80..96]");
    assert_eq!(linear_result_lanes, [0, 1, 2].into_iter().collect(), "LINEAR stores must consume WMMA lanes 0,1,2");

    let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(AmdArch::Gfx1151);
    let rendered = svod_codegen::traits::Renderer::render(&renderer, &linear, Some("matmul_m5_gfx1151"))
        .expect("render gfx1151 LLVM");
    assert!(rendered.code.contains("llvm.amdgcn.wmma.f32.16x16x16.f16"), "must select gfx11 f16 WMMA");
    assert!(!rendered.code.contains("mfma"), "gfx1151 must not select CDNA MFMA");
    let rendered_op =
        |id| rendered.operations.iter().find(|operation| operation.uop_id == id).expect("rendered UOp metadata");
    for (load, index, _, gate, alt) in &linear_a_loads {
        let load_render = rendered_op(load.id);
        let index_render = rendered_op(index.id);
        let gate_render = rendered_op(gate.id);
        let alt_render = rendered_op(alt.id);
        let address_name = index_render.result.as_ref().expect("rendered A address result");
        let gate_name = gate_render.result.as_ref().expect("rendered A gate result");
        let alt_name = alt_render.result.as_ref().expect("rendered A zero result");
        assert_eq!(index_render.lines.len(), 1, "each A address must render one GEP");
        assert!(
            index_render.lines[0].contains("getelementptr") && index_render.lines[0].contains("half"),
            "A address metadata must own its half-vector GEP: {:?}",
            index_render.lines
        );
        assert!(load_render.lines.iter().any(|line| line.contains("br i1") && line.contains(gate_name)));
        assert!(load_render.lines.iter().any(|line| line.contains("load <4 x half>") && line.contains(address_name)));
        assert!(load_render.lines.iter().any(|line| line.contains("phi <4 x half>") && line.contains(alt_name)));
        assert_eq!(
            load_render.source_ids,
            vec![index.id, alt.id, gate.id],
            "rendered A load sources must preserve ownership"
        );
    }
    let if_render = rendered_op(if_node.id);
    let guarded_store_render = rendered_op(guarded_store.1.id);
    let guarded_address_render = rendered_op(guarded_store.2.id);
    let endif_render = rendered_op(endif_node.id);
    let condition_name = rendered_op(if_condition.id).result.as_ref().expect("rendered C condition");
    let address_name = guarded_address_render.result.as_ref().expect("rendered C address");
    let Op::Store { value: guarded_value, .. } = guarded_store.1.op() else { unreachable!() };
    let value_name = rendered_op(guarded_value.id).result.as_ref().expect("rendered C value");
    assert_eq!(if_render.source_ids, vec![if_condition.id, guarded_store.2.id]);
    assert!(if_render.lines.iter().any(|line| line.contains("br i1") && line.contains(condition_name)));
    assert!(guarded_address_render.lines.iter().any(|line| line.contains("getelementptr")));
    assert!(
        guarded_store_render
            .lines
            .iter()
            .any(|line| { line.contains("store float") && line.contains(address_name) && line.contains(value_name) })
    );
    assert_eq!(guarded_store_render.source_ids, vec![guarded_store.2.id, guarded_value.id]);
    assert_eq!(endif_render.source_ids, vec![if_node.id]);
    assert!(endif_render.lines.iter().any(|line| line.contains("br label") && line.contains("if_end_")));
    if svod_runtime::amd::compile::has_amdgpu_target() {
        let object = svod_runtime::amd::compile::compile_ir_to_amd_object(&rendered.code, AmdArch::Gfx1151)
            .expect("compile padded gfx1151 WMMA matmul");
        assert_eq!(&object[..4], b"\x7fELF");
    }
}

/// Hardware acceptance for the compile-only regression above. This test exits
/// before dispatch unless the selected device is exactly AMD:0 on gfx1151.
///
/// Run once after a clean boot:
/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tensor test_matmul_m5_gfx1151_padded_wmma_amd -- --ignored --nocapture --test-threads=1`.
#[test]
#[ignore = "requires AMD:0 with gfx1151; dispatches a real padded WMMA kernel"]
fn test_matmul_m5_gfx1151_padded_wmma_amd() {
    use svod_dtype::{AmdArch, DeviceSpec};

    setup_test_tracing();
    let device = DeviceSpec::Amd { device_id: 0 };
    assert_eq!(svod_device::registry::resolve_amd_arch_from_topology(0).expect("AMD:0 topology"), AmdArch::Gfx1151);

    let a_data = (0..5 * 16).map(|i| (i as f32 % 11.0 - 5.0) * 0.125).collect::<Vec<_>>();
    let b_data = (0..16 * 16).map(|i| (i as f32 % 13.0 - 6.0) * 0.0625).collect::<Vec<_>>();
    let expected = (0..5)
        .flat_map(|m| {
            let a_data = &a_data;
            let b_data = &b_data;
            (0..16).map(move |n| (0..16).map(|k| a_data[m * 16 + k] * b_data[k * 16 + n]).sum::<f32>())
        })
        .collect::<Vec<_>>();

    let a = Tensor::from_slice(&a_data).try_reshape([5, 16]).unwrap().cast(DType::Float16).unwrap();
    let b = Tensor::from_slice(&b_data).try_reshape([16, 16]).unwrap().cast(DType::Float16).unwrap();
    assert_eq!(a.device(), device, "set SVOD_DEVICE=AMD:0; refusing to dispatch elsewhere");
    assert_eq!(b.device(), device, "set SVOD_DEVICE=AMD:0; refusing to dispatch elsewhere");

    let heuristics = HeuristicsConfig::builder()
        .tc_opt(TcOptLevel::Padded)
        .tc_select(TcSelect::Index(0))
        .matvec_enabled(false)
        .build();
    let optimizer = OptimizerConfig::builder().strategy(OptStrategy::Heuristic).heuristics(heuristics).build();
    let config = prep_config(optimizer);
    let mut c = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("tensor matmul");
    assert_eq!(c.device(), device);

    let plan = c.prepare_with(&config).expect("prepare padded WMMA on AMD:0");
    assert!(
        plan.kernels().any(|kernel| kernel.code.contains("llvm.amdgcn.wmma.f32.16x16x16.f16")),
        "prepared plan must contain gfx11 f16-to-f32 WMMA before execution"
    );

    plan.execute().expect("dispatch padded WMMA");
    let output = plan.output_buffer().expect("matmul output buffer");
    output.synchronize().expect("synchronize padded WMMA immediately after dispatch");
    let mut actual = vec![0.0f32; 5 * 16];
    output
        .copyout(unsafe {
            std::slice::from_raw_parts_mut(actual.as_mut_ptr().cast::<u8>(), actual.len() * std::mem::size_of::<f32>())
        })
        .expect("copy padded WMMA output to host");

    for (i, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            (actual - expected).abs() <= 2e-2,
            "mismatch at output {i}: GPU={actual}, CPU={expected}, diff={}",
            (actual - expected).abs()
        );
    }
}

#[test]
#[ignore] // Run with: cargo test -p svod-tensor test_print_matmul_ir -- --ignored --nocapture
fn test_print_matmul_ir() {
    // Create 4x4 matmul to see generated IR
    let a = Tensor::from_ndarray(&Array2::from_shape_vec((4, 4), (0..16).map(|i| i as f32).collect()).unwrap());
    let b = Tensor::from_ndarray(&Array2::from_shape_vec((4, 4), (0..16).map(|i| i as f32).collect()).unwrap());
    let mut c = a.matmul(&b).unwrap();

    let plan = c.prepare().expect("prepare should succeed");

    println!("\n=== Generated Kernels ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

#[test]
#[ignore] // Run with: cargo test -p svod-tensor test_print_matmul_512x512_ir -- --ignored --nocapture
fn test_print_matmul_512x512_ir() {
    const SIZE: usize = 512;
    let a = Tensor::from_ndarray(
        &Array2::from_shape_vec((SIZE, SIZE), (0..SIZE * SIZE).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let b = Tensor::from_ndarray(
        &Array2::from_shape_vec((SIZE, SIZE), (0..SIZE * SIZE).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let mut c = a.matmul(&b).unwrap();

    // Use Heuristic strategy (Beam has a pre-existing bug with horizontal reduction)
    let config = prep_config(OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build());
    let plan = c.prepare_with(&config).expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 with output upcast) ===\n");
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

#[test]
fn test_beam_search_matmul() {
    // Test beam search optimization for matmul - reproduces float vector index bug
    let size = 512; // Original size that triggered the bug
    let a = Tensor::from_ndarray(
        &Array2::from_shape_vec((size, size), (0..size * size).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let b = Tensor::from_ndarray(
        &Array2::from_shape_vec((size, size), (0..size * size).map(|i| (i as f32) * 0.01).collect()).unwrap(),
    );
    let mut c = a.matmul(&b).unwrap();

    // Use width=2 for reasonable test time. Disable disk cache to avoid stale results
    // from previous runs affecting correctness (beam cache is keyed by AST hash, but
    // the post-optimization pipeline may have changed).
    let beam_config = prep_config(
        OptimizerConfig::builder()
            .strategy(OptStrategy::Beam { width: 2 })
            .beam(BeamConfig::builder().disable_cache(true).build())
            .build(),
    );

    let plan = c.prepare_with(&beam_config).expect("beam search prepare should succeed");

    println!("\n=== Beam Search Kernels ({}x{}) ===\n", size, size);
    for kernel in plan.kernels() {
        println!("--- {} ({}) ---", kernel.entry_point, kernel.device);
        println!("{}", kernel.code);
        println!();
    }
}

// ========== Linear Layer Tests ==========

#[test]
fn test_linear_basic() {
    // input: [1, 3], weight: [2, 3], bias: [2]
    let input = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    let weight = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);
    let bias = Tensor::from_slice([0.1f32, 0.2]);

    let result = input.linear().weight(&weight).bias(&bias).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_no_bias() {
    let input = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0]]);
    let weight = Tensor::from_ndarray(&array![[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]]);

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 1);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_batched() {
    // input: [4, 3], weight: [2, 3] → output: [4, 2]
    let input = Tensor::from_ndarray(&Array2::<f32>::ones((4, 3)));
    let weight = Tensor::from_ndarray(&Array2::<f32>::ones((2, 3)));

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 2);
    assert_eq!(result_shape[0].as_const().unwrap(), 4);
    assert_eq!(result_shape[1].as_const().unwrap(), 2);
}

#[test]
fn test_linear_1d_weight() {
    // Test 1D weight case (element-wise multiply)
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let weight = Tensor::from_slice([2.0f32, 3.0, 4.0]);

    let result = input.linear().weight(&weight).call().unwrap();

    let result_shape = result.shape().unwrap();
    assert_eq!(result_shape.len(), 1);
    assert_eq!(result_shape[0].as_const().unwrap(), 3);
}

// ========== Minimal VECTORIZE Normalization Test ==========

#[test]
fn test_vectorize_normalize_minimal() {
    // Test 64x64 matmul with vectorization enabled
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((64, 64)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((64, 64)));
    let mut c = a.matmul(&b).unwrap();

    // Explicit config to avoid test pollution from shared global state
    let config = prep_config(OptimizerConfig::builder().strategy(OptStrategy::Heuristic).build());
    let result = c.realize_with(&config);
    assert!(result.is_ok(), "realize failed: {:?}", result.err());
}

// ========== 512x512 Stackd Test (for UPCAST debugging) ==========

#[test]
fn test_matmul_512x512_vectorized() {
    // Create 512x512 matrices filled with 1.0
    const SIZE: usize = 512;
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let mut c = a.matmul(&b).unwrap();

    // Use from_env() to respect SVOD_OUTPUT_UPCAST and other env vars
    // Note: Beam search has a pre-existing bug with horizontal reduction, using Heuristic
    let config = env_config();
    c.realize_with(&config).unwrap();
    let result = c.as_vec::<f32>().unwrap();

    // Each element should be 512 (sum of 512 ones)
    assert_eq!(result.len(), SIZE * SIZE);
    assert!((result[0] - SIZE as f32).abs() < 0.01, "Expected {}, got {}", SIZE, result[0]);
}

#[test]
fn test_matmul_64x64_vectorized() {
    // Create 64x64 matrices filled with 1.0
    const SIZE: usize = 64;
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let mut c = a.matmul(&b).unwrap();

    let config = env_config();
    c.realize_with(&config).unwrap();
    let result = c.as_vec::<f32>().unwrap();

    // Each element should be 64 (sum of 64 ones)
    assert_eq!(result.len(), SIZE * SIZE);
    assert!((result[0] - SIZE as f32).abs() < 0.01, "Expected {}, got {}", SIZE, result[0]);
}

#[test]
#[ignore] // Run with: cargo test -p svod-tensor test_print_matmul_64x64_ir -- --ignored --nocapture
fn test_print_matmul_64x64_ir() {
    const SIZE: usize = 64;
    let a = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let b = Tensor::from_ndarray(&Array2::<f32>::ones((SIZE, SIZE)));
    let mut c = a.matmul(&b).unwrap();

    let config = env_config();
    let plan = c.prepare_with(&config).expect("prepare should succeed");

    println!("\n=== Generated Kernels (64x64 matmul) ===\n");
    for kernel in plan.prepared_kernels() {
        println!("--- {} ({}) ---", kernel.kernel.entry_point, kernel.kernel.device);
        println!("{}", kernel.ast.tree());
        println!("{}", kernel.kernel.code);
        println!();
    }
}

/// gfx942 (CDNA3) MFMA tensor-core matmul: end-to-end proof + numerical
/// validation, parameterized over the low-precision input dtype. A
/// `in_dtype·in_dtype` matmul accumulating in f32 matches a cdna3 tensor core,
/// so BEAM must lower it to `intrinsic`. Inputs are small integers (−3..3 /
/// −2..2, all exact in bf16/f16/fp8e4m3), so the MFMA result — accumulated
/// across the residual K-tile loop and fanned out across the M/N output tiles —
/// must equal the f32 reference exactly. Guards both the reduce-loop lowering
/// and the per-tile expansion. `tol` stays small because the inputs round-trip
/// losslessly and the accumulation is in f32.
fn validate_mfma_square(size: usize, in_dtype: DType, intrinsic: &str, tol: f32) {
    let beam = OptimizerConfig::builder()
        .strategy(OptStrategy::Beam { width: 2 })
        .beam(BeamConfig::builder().disable_cache(true).build())
        .build();
    validate_mfma_square_with(beam, size, in_dtype, intrinsic, tol);
}

fn validate_mfma_square_with(opt: OptimizerConfig, size: usize, in_dtype: DType, intrinsic: &str, tol: f32) {
    let a_data: Vec<f32> = (0..size * size).map(|x| ((x % 7) as f32) - 3.0).collect();
    let b_data: Vec<f32> = (0..size * size).map(|x| ((x % 5) as f32) - 2.0).collect();
    let a_nd = Array2::from_shape_vec((size, size), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((size, size), b_data).unwrap();

    let beam = prep_config(opt);
    let build = || {
        let a = Tensor::from_ndarray(&a_nd).cast(in_dtype.clone()).unwrap();
        let b = Tensor::from_ndarray(&b_nd).cast(in_dtype.clone()).unwrap();
        a.matmul_with().other(&b).dtype(DType::Float32).call().unwrap()
    };

    // 1) The selected kernel must actually use the expected MFMA (not a fallback).
    let mut probe = build();
    let plan = probe.prepare_with(&beam).expect("prepare should succeed");
    let saw_mfma = plan.prepared_kernels().iter().any(|k| k.kernel.code.contains(intrinsic));
    assert!(saw_mfma, "BEAM did not select {intrinsic} for a {in_dtype:?} {size}x{size} matmul on gfx942");

    // 2) The MFMA result must match the f32 reference (exact for integer inputs).
    let mut c = build();
    c.realize_with(&beam).unwrap();
    let expected = a_nd.dot(&b_nd);
    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, tol);
}

/// Hardware-gated: `SVOD_DEVICE=AMD:0 cargo test -p svod-tensor test_matmul_bf16_mfma_validated -- --ignored --nocapture`.
#[test]
#[ignore]
fn test_matmul_bf16_mfma_validated() {
    validate_mfma_square(512, DType::BFloat16, "llvm.amdgcn.mfma.f32.16x16x16bf16.1k", 1.0);
}

/// gfx942 f16 16×16×16 MFMA (the `f16` plain form).
#[test]
#[ignore]
fn test_matmul_f16_mfma_validated() {
    validate_mfma_square(512, DType::Float16, "llvm.amdgcn.mfma.f32.16x16x16f16", 1.0);
}

/// gfx942 fp8 (e4m3) 16×16×32 MFMA. The cdna3 fp8 tensor core is K=32, so this
/// also exercises the K=32 reduce-tile lowering and i64 operand packing. Uses a
/// smaller matrix: the fp8 path compiles many BEAM candidates (each with the
/// fp8-conversion prelude), so 512² BEAM is ~8min; 128² keeps it tractable while
/// still tiling into the 16×16×32 core.
#[test]
#[ignore]
fn test_matmul_fp8_mfma_validated() {
    validate_mfma_square(128, DType::FP8E4M3, "llvm.amdgcn.mfma.f32.16x16x32.fp8.fp8", 1.0);
}

// ========== Validated Matmul Tests (64x64 with env_config) ==========

#[test]
fn test_matmul_validated_64x64() {
    // 64x64 test with varied data
    const SIZE: usize = 64;
    let a_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.01).sin()).collect();
    let b_data: Vec<f32> = (0..SIZE * SIZE).map(|x| ((x as f32) * 0.02).cos()).collect();

    let a_nd = Array2::from_shape_vec((SIZE, SIZE), a_data).unwrap();
    let b_nd = Array2::from_shape_vec((SIZE, SIZE), b_data).unwrap();
    let a = Tensor::from_ndarray(&a_nd);
    let b = Tensor::from_ndarray(&b_nd);

    let config = env_config();
    let mut c = a.matmul(&b).unwrap();
    c.realize_with(&config).unwrap();

    let expected = a_nd.dot(&b_nd);

    // Larger tolerance for accumulated floating point error
    assert_matmul_close(&c.as_vec::<f32>().unwrap(), &expected, 1e-1);
}

// ========== Large Dimension Validated Tests ==========

use test_case::test_case;

// Square matrix tests with increasing sizes
#[test_case(128, 0.5; "128x128")]
#[test_case(256, 1.0; "256x256")]
#[test_case(500, 1.5; "500x500 non-power-of-2")]
#[test_case(512, 2.0; "512x512")]
#[test_case(1024, 3.0; "1024x1024")]
fn test_matmul_validated_square(size: usize, tol: f32) {
    setup_test_tracing();
    run_validated_square_matmul(size, tol);
}

// Non-square matrix tests
#[test_case(512, 256, 384, 2.0; "512x256 @ 256x384")]
#[test_case(1024, 64, 128, 1.0; "1024x64 @ 64x128 tall-skinny")]
#[test_case(64, 512, 64, 1.5; "64x512 @ 512x64 wide")]
#[test_case(256, 1024, 256, 2.5; "256x1024 @ 1024x256 large-K")]
fn test_matmul_validated_non_square(m: usize, k: usize, n: usize, tol: f32) {
    run_validated_matmul(m, k, n, tol);
}
