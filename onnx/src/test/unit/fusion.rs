//! ONNX kernel fusion tests — loads real ONNX subgraphs and validates kernel counts.
//!
//! Test models in test/data/ are minimal subgraphs extracted from GigaAM RNNT encoder
//! with zeroed weights (small file sizes). Kernel counts verified against Tinygrad.

use std::path::Path;

use morok_dtype::{DType, ScalarDType};
use morok_tensor::{CpuBackend, PrepareConfig, Tensor};

use crate::importer::OnnxImporter;

fn count_kernels(model_path: &Path) -> usize {
    let mut importer = OnnxImporter::new();
    let result = importer.import(model_path, &[]).unwrap();

    for (name, input_tensor) in &result.inputs {
        let shape: Vec<usize> = input_tensor
            .shape()
            .unwrap_or_else(|e| panic!("input '{name}' shape: {e}"))
            .iter()
            .map(|d| d.as_const().unwrap_or_else(|| panic!("dynamic dim in '{name}'")))
            .collect();
        let n: usize = shape.iter().product();
        let data = vec![0u8; n * 4]; // zeros, float32
        let real = Tensor::from_raw_bytes(&data, &shape, DType::Scalar(ScalarDType::Float32))
            .unwrap_or_else(|e| panic!("input '{name}': {e}"));
        input_tensor.assign(&real);
    }

    let config = PrepareConfig::for_cpu_backend(CpuBackend::Clang);
    let mut outputs: Vec<Tensor> = result.outputs.values().cloned().collect();
    let plan = Tensor::prepare_batch_with(outputs.iter_mut(), &config).unwrap();
    plan.prepared_kernels().len()
}

/// Conformer feed-forward block with SiLU gate (static shapes):
///   LayerNorm → MatMul+Add → Sigmoid → Mul ← MatMul+Add → MatMul+Add
///
/// Tinygrad: 5 kernels
#[test]
fn test_conformer_ff1_kernel_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/test/data/conformer_ff1.onnx");
    assert!(path.exists(), "test model not found: {}", path.display());
    let n = count_kernels(&path);
    assert_eq!(n, 5, "conformer_ff1 should produce 5 kernels (tinygrad), got {n}");
}

/// Same FF1 block but with symbolic batch_size and seq_len dimensions.
/// The real GigaAM model uses symbolic dims — this tests that variable
/// simplification doesn't cause extra kernels.
///
/// Tinygrad: 5 kernels (same as static — variables are bound to concrete values)
#[test]
fn test_conformer_ff1_dynamic_kernel_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/test/data/conformer_ff1_dynamic.onnx");
    assert!(path.exists(), "test model not found: {}", path.display());

    let mut importer = OnnxImporter::new();
    let result = importer.import(&path, &[("batch_size", 6), ("seq_len", 512)]).unwrap();

    for (name, input_tensor) in &result.inputs {
        let shape: Vec<usize> = input_tensor
            .shape()
            .unwrap()
            .iter()
            .map(|d| d.as_const().unwrap_or_else(|| panic!("dynamic dim in '{name}'")))
            .collect();
        let data = vec![0u8; shape.iter().product::<usize>() * 4];
        let real = Tensor::from_raw_bytes(&data, &shape, DType::Scalar(ScalarDType::Float32)).unwrap();
        input_tensor.assign(&real);
    }

    let config = PrepareConfig::for_cpu_backend(CpuBackend::Clang);
    let mut outputs: Vec<Tensor> = result.outputs.values().cloned().collect();
    let plan = Tensor::prepare_batch_with(outputs.iter_mut(), &config).unwrap();
    let n = plan.prepared_kernels().len();
    assert_eq!(n, 5, "conformer_ff1_dynamic should produce 5 kernels (tinygrad), got {n}");
}

/// Conv block: LN → 1×1 Conv → GLU → depthwise Conv → BN → SiLU → 1×1 Conv
///
/// Tinygrad: 9 kernels
#[test]
fn test_conformer_conv_kernel_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/test/data/conformer_conv.onnx");
    assert!(path.exists(), "test model not found: {}", path.display());
    let n = count_kernels(&path);
    assert_eq!(n, 9, "conformer_conv should produce 9 kernels (tinygrad), got {n}");
}

/// Attention with rotary position encoding (Conformer-style):
///   Q/K projections → reshape → RoPE (slice+neg+concat+mul+add) → attn → output proj
///
/// Tinygrad: 8 kernels (same as plain attention — RoPE fuses with Q/K projections)
#[test]
fn test_conformer_attn_rope_kernel_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/test/data/conformer_attn_rope.onnx");
    assert!(path.exists(), "test model not found: {}", path.display());
    let n = count_kernels(&path);
    assert_eq!(n, 8, "attn_rope should produce 8 kernels (tinygrad), got {n}");
}

/// Full conformer layer: FF1 + Attention + Conv + FF2 with residuals and LayerNorms
///
/// Tinygrad: 36 kernels
#[test]
fn test_conformer_full_layer_kernel_count() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/test/data/conformer_full_layer.onnx");
    assert!(path.exists(), "test model not found: {}", path.display());
    let n = count_kernels(&path);
    assert_eq!(n, 36, "conformer_full_layer should produce 36 kernels (tinygrad), got {n}");
}
