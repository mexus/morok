//! Kernel fusion tests — validates that operations fuse correctly into minimal kernels.
//!
//! Each test's expected kernel count is verified against Tinygrad (CPU=1, Tensor.empty).

use crate::Tensor;

fn kernel_count(tensor: &mut Tensor) -> usize {
    let plan = tensor.prepare().expect("prepare should succeed");
    plan.prepared_kernels().len()
}

fn e(shape: &[usize]) -> Tensor {
    Tensor::empty(shape, morok_dtype::DType::Float32)
}

// =========================================================================
// Basic ops (Tinygrad: all verified with CPU=1, Tensor.empty)
// =========================================================================

/// Tinygrad: 1 kernel
#[test]
fn test_matmul_add_fusion() {
    let (x, w, b) = (e(&[6, 32, 768]), e(&[768, 768]), e(&[768]));
    let mut out = &x.matmul(&w).unwrap() + &b;
    assert_eq!(kernel_count(&mut out), 1);
}

/// Tinygrad: 1 kernel
#[test]
fn test_matmul_one_kernel() {
    let mut out = e(&[6, 32, 768]).matmul(&e(&[768, 768])).unwrap();
    assert_eq!(kernel_count(&mut out), 1);
}

/// Tinygrad: 2 kernels (split_reduceop)
#[test]
fn test_sum_kernels() {
    let mut out = e(&[6, 32, 768]).sum(()).unwrap();
    assert_eq!(kernel_count(&mut out), 2);
}

/// Tinygrad: 1 kernel
#[test]
fn test_elementwise_chain() {
    let (x, y, z) = (e(&[6, 32, 768]), e(&[6, 32, 768]), e(&[6, 32, 768]));
    let mut out = &(&x + &y) * &z;
    assert_eq!(kernel_count(&mut out), 1);
}

/// Tinygrad: 1 kernel
#[test]
fn test_silu_fusion() {
    let x = e(&[6, 32, 768]);
    let mut out = &x.sigmoid().unwrap() * &x;
    assert_eq!(kernel_count(&mut out), 1);
}

/// Tinygrad: 1 kernel — both matmuls + bias + sigmoid + mul all fused
#[test]
fn test_silu_gate_block() {
    let (x, w1, b1, w2, b2) = (e(&[6, 32, 768]), e(&[768, 3072]), e(&[3072]), e(&[768, 3072]), e(&[3072]));
    let gate = (&x.matmul(&w1).unwrap() + &b1).sigmoid().unwrap();
    let value = &x.matmul(&w2).unwrap() + &b2;
    let mut out = &gate * &value;
    assert_eq!(kernel_count(&mut out), 1);
}

// =========================================================================
// Neural network patterns (Tinygrad-verified)
// =========================================================================

/// Tinygrad: 3 kernels (mean reduce, variance reduce, normalize+scale+shift)
#[test]
fn test_layernorm_kernels() {
    let x = e(&[6, 512, 768]);
    let gamma = e(&[768]);
    let beta = e(&[768]);
    let normed = x.layernorm(-1, 1e-5).unwrap();
    let mut out = &(&normed * &gamma) + &beta;
    assert_eq!(kernel_count(&mut out), 3);
}

/// Tinygrad: 3 kernels (max reduce, exp+sum reduce, div)
#[test]
fn test_softmax_kernels() {
    let mut out = e(&[6, 16, 512, 48]).softmax(-1isize).unwrap();
    assert_eq!(kernel_count(&mut out), 3);
}

/// Tinygrad: 8 kernels (3 QKV matmuls + attn matmul + softmax(3) + output matmul)
#[test]
fn test_attention_kernels() {
    let (x, wq, wk, wv, wo) = (e(&[6, 512, 768]), e(&[768, 768]), e(&[768, 768]), e(&[768, 768]), e(&[768, 768]));
    let q = x.matmul(&wq).unwrap().try_reshape([6, 512, 16, 48]).unwrap().try_permute(&[0, 2, 1, 3]).unwrap();
    let k = x.matmul(&wk).unwrap().try_reshape([6, 512, 16, 48]).unwrap().try_permute(&[0, 2, 1, 3]).unwrap();
    let v = x.matmul(&wv).unwrap().try_reshape([6, 512, 16, 48]).unwrap().try_permute(&[0, 2, 1, 3]).unwrap();
    let kt = k.try_permute(&[0, 1, 3, 2]).unwrap();
    let scale = Tensor::const_(1.0f32 / 48.0f32.sqrt(), morok_dtype::DType::Float32);
    let attn = &q.matmul(&kt).unwrap() * &scale;
    let attn_w = attn.softmax(-1isize).unwrap();
    let attn_out = attn_w.matmul(&v).unwrap().try_permute(&[0, 2, 1, 3]).unwrap().try_reshape([6, 512, 768]).unwrap();
    let mut out = attn_out.matmul(&wo).unwrap();
    assert_eq!(kernel_count(&mut out), 8);
}

/// Tinygrad: 2 kernels (fused gate matmul + down projection matmul)
#[test]
fn test_ff_silu_gate_with_down_proj() {
    let (x, w_up, b_up, w_gate, b_gate, w_down, b_down) =
        (e(&[6, 512, 768]), e(&[768, 3072]), e(&[3072]), e(&[768, 3072]), e(&[3072]), e(&[3072, 768]), e(&[768]));
    let ff = &(&x.matmul(&w_gate).unwrap() + &b_gate).sigmoid().unwrap() * &(&x.matmul(&w_up).unwrap() + &b_up);
    let mut out = &ff.matmul(&w_down).unwrap() + &b_down;
    assert_eq!(kernel_count(&mut out), 2);
}
