# morok-tensor

High-level tensor API with lazy evaluation.

## Basic Example

```rust
use morok_tensor::Tensor;

let a = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
let b = Tensor::from_slice(&[4.0f32, 5.0, 6.0]);
let c = (&a + &b)?;

// Extract as Vec
let result = c.to_vec::<f32>()?;
assert_eq!(result, vec![5.0, 7.0, 9.0]);
```

## ndarray Interop

```rust
use morok_tensor::Tensor;
use ndarray::array;

// Zero-copy from ndarray (fast path for C-contiguous arrays)
let input = array![[1.0f32, 2.0], [3.0, 4.0]];
let t = Tensor::from_ndarray(&input);

// Compute and extract back as ndarray
let result = (t * 2.0)?.to_ndarray::<f32>()?;
assert_eq!(result, array![[2.0, 4.0], [6.0, 8.0]].into_dyn());
```

### Zero-Copy View

For realized tensors on CPU, `array_view` returns a borrowed ndarray view
without copying data:

```rust
let t = Tensor::from_slice(&[1.0f32, 2.0, 3.0]).realize()?;
let view = t.array_view::<f32>()?;  // no copy, lifetime tied to tensor
assert_eq!(view.len(), 3);
```

## Prepare/Execute Infrastructure

For repeated kernel executions (e.g., benchmarks, inference loops), separate preparation from execution:

```rust
use morok_tensor::Tensor;
use morok_runtime::global_executor;

let a = Tensor::from_slice(&data_a);
let b = Tensor::from_slice(&data_b);
let result = a.matmul(&b)?;

// One-time preparation (compiles kernels, allocates buffers)
let plan = result.prepare()?;

// Fast repeated execution
let mut executor = global_executor();
for _ in 0..1000 {
    plan.execute(&mut executor)?;
}
```

## Features

**Supported:**

- Lazy tensor construction
- Arithmetic: add, sub, mul, div, pow
- Math: sqrt, exp, log, sin, cos
- Reduction: sum, mean, max, min, argmax, argmin
- Shape: reshape, transpose, permute, expand, squeeze
- Activation: relu, sigmoid, tanh, softmax, gelu
- Matrix: matmul, dot, linear

## Testing

```bash
cargo test -p morok-tensor
```
