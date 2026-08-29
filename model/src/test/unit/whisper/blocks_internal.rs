//! Whisper block internals: FP16 LayerNorm and Linear must keep their affine
//! and bias epilogues in FP32 and round only once, at the final cast.

use crate::whisper::blocks::{LayerNormWeights, linear_with_bias};
use svod_dtype::DType;
use svod_tensor::Tensor;

fn realized_f32(tensor: Tensor) -> Vec<f32> {
    let mut tensor = tensor.cast(DType::Float32).unwrap();
    tensor.realize().unwrap();
    tensor.as_vec::<f32>().unwrap()
}

#[test]
fn fp16_layernorm_keeps_affine_in_fp32_until_final_cast() {
    let x = Tensor::from_slice([0.1013f32, -0.2037, 0.3071, 1.913])
        .try_reshape([1usize, 4])
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let layer = LayerNormWeights {
        weight: Tensor::from_slice([17.25f32, -31.5, 47.75, -63.0]).cast(DType::Float16).unwrap(),
        bias: Tensor::from_slice([0.03125f32, -0.0625, 0.09375, -0.125]).cast(DType::Float16).unwrap(),
        eps: 1e-5,
    };

    let reference = x
        .cast(DType::Float32)
        .unwrap()
        .layernorm(-1, layer.eps)
        .unwrap()
        .try_mul(&layer.weight.cast(DType::Float32).unwrap())
        .unwrap()
        .try_add(&layer.bias.cast(DType::Float32).unwrap())
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let legacy = x.layernorm(-1, layer.eps).unwrap().try_mul(&layer.weight).unwrap().try_add(&layer.bias).unwrap();
    let actual = realized_f32(layer.apply(&x).unwrap());
    let reference = realized_f32(reference);
    let legacy = realized_f32(legacy);

    assert_eq!(actual, reference, "LayerNorm must round only after the FP32 affine epilogue");
    assert_ne!(legacy, reference, "fixture must detect rounding before the affine epilogue");
}

#[test]
fn fp16_linear_keeps_bias_epilogue_in_fp32_until_final_cast() {
    let x = Tensor::from_slice([0.3333f32, -0.1428, 0.0909, 0.0769])
        .try_reshape([1usize, 4])
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let weight = Tensor::from_slice([3.0f32, -5.0, 7.0, -11.0, -2.0, 4.0, -6.0, 8.0])
        .try_reshape([2usize, 4])
        .unwrap()
        .cast(DType::Float16)
        .unwrap();
    let bias = Tensor::from_slice([-2.5f32, 2.0]).cast(DType::Float16).unwrap();

    let matmul_f32 = x.linear().weight(&weight).dtype(DType::Float32).call().unwrap();
    let reference = matmul_f32.try_add(&bias.cast(DType::Float32).unwrap()).unwrap().cast(DType::Float16).unwrap();
    let legacy = x.linear().weight(&weight).bias(&bias).call().unwrap();
    let actual = realized_f32(linear_with_bias(&x, &weight, &bias).unwrap());
    let reference = realized_f32(reference);
    let legacy = realized_f32(legacy);

    assert_eq!(actual, reference, "linear bias must be added to the FP32 accumulator");
    assert_ne!(legacy, reference, "fixture must detect rounding before bias addition");
}
