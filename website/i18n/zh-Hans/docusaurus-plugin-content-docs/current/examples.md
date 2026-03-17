---
sidebar_label: 实践示例
---

# 实践：从 Tensor 到模型

本章通过渐进式示例教你使用 Morok。你将从基本的张量操作开始，逐步构建出一个完整的神经网络分类器。

**你将学到：**
- 创建和操作张量
- 形状操作（reshape、transpose、broadcast）
- 矩阵乘法
- 构建可复用的层
- 组装完整的模型

**前置条件：**
- 基本的 Rust 知识
- 在 `Cargo.toml` 中添加 `morok_tensor`

**核心模式：** Morok 使用*惰性求值*。操作只构建计算图，不会立即执行。调用 `realize()` 时才会一次性编译和运行所有操作。

---

## 示例 1：Hello Tensor

创建张量、执行操作并获取结果。

```rust
use morok_tensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create tensors from slices
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);
    let b = Tensor::from_slice([10.0f32, 20.0, 30.0, 40.0]);

    // Lazy operations (no execution yet)
    let sum = &a + &b;
    let scaled = &sum * &Tensor::from_slice([0.1f32]);

    // Execute and get results
    let result = scaled.realize()?;
    let data = result.to_ndarray::<f32>()?;
    println!("Result: {:?}", data);
    // Output: [1.1, 2.2, 3.3, 4.4]

    Ok(())
}
```

**发生了什么：**

1. `Tensor::from_slice()` 从 Rust slice 创建张量。`f32` 后缀告诉 Rust 元素类型。

2. `&a + &b` 不会执行任何计算，它返回一个*表示*加法操作的新 `Tensor`。`&` 借用张量以便后续复用。

3. `realize()` 是关键所在。Morok 会：
   - 分析计算图
   - 尽可能融合操作
   - 生成优化后的代码
   - 在目标设备上执行

4. `to_ndarray()` 将结果提取为 `ndarray::ArrayD` 以供查看。

**试试看：** 去掉 `realize()` 调用。代码仍能运行，但 `data` 会是空的——什么都没有被计算。

---

## 示例 2：形状变换

神经网络不断地重塑数据。来掌握基础操作。

```rust
fn shape_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a 1D tensor with 6 elements
    let data = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    println!("Original shape: {:?}", data.shape());  // [6]

    // Reshape to a 2x3 matrix
    let matrix = data.try_reshape(&[2, 3])?;
    println!("Matrix shape: {:?}", matrix.shape());  // [2, 3]
    // [[1, 2, 3],
    //  [4, 5, 6]]

    // Transpose to 3x2
    let transposed = matrix.try_transpose(0, 1)?;
    println!("Transposed shape: {:?}", transposed.shape());  // [3, 2]
    // [[1, 4],
    //  [2, 5],
    //  [3, 6]]

    // Broadcasting: add a row vector to every row
    // [3, 2] + [1, 2] → [3, 2]
    let bias = Tensor::from_slice([100.0f32, 200.0])
        .try_reshape(&[1, 2])?;
    let biased = &transposed + &bias;

    let result = biased.realize()?;
    println!("{:?}", result.to_ndarray::<f32>()?);
    // [[101, 204],
    //  [102, 205],
    //  [103, 206]]

    Ok(())
}
```

**核心操作：**

| 操作 | 功能说明 |
|-----------|--------------|
| `try_reshape(&[2, 3])` | 改变形状（总元素数不变） |
| `try_reshape(&[-1, 3])` | 根据总大小自动推断维度 |
| `try_transpose(0, 1)` | 交换第 0 和第 1 维 |
| `try_squeeze(dim)` | 移除大小为 1 的维度 |
| `try_unsqueeze(dim)` | 添加大小为 1 的维度 |

**广播规则**（与 NumPy/PyTorch 相同）：
- 形状从右侧对齐
- 每个维度必须匹配或为 1
- 大小为 1 的维度会被"拉伸"以匹配

```text
[3, 2] + [1, 2] → [3, 2]  ✓ (1 broadcasts to 3)
[3, 2] + [2]    → [3, 2]  ✓ (implicit [1, 2])
[3, 2] + [3]    → error   ✗ (2 ≠ 3)
```

---

## 示例 3：矩阵乘法

矩阵乘法是神经网络的核心运算，每一层都会用到它。

```rust
fn matmul_example() -> Result<(), Box<dyn std::error::Error>> {
    use ndarray::array;

    // Input: 4 samples, 3 features each → shape [4, 3]
    let input = Tensor::from_ndarray(&array![
        [1.0f32, 2.0, 3.0],    // sample 0
        [4.0, 5.0, 6.0],       // sample 1
        [7.0, 8.0, 9.0],       // sample 2
        [10.0, 11.0, 12.0],    // sample 3
    ]);

    // Weights: 3 inputs → 2 outputs → shape [3, 2]
    let weights = Tensor::from_ndarray(&array![
        [0.1f32, 0.2],  // feature 0 → outputs
        [0.3, 0.4],     // feature 1 → outputs
        [0.5, 0.6],     // feature 2 → outputs
    ]);

    // Matrix multiply: [4, 3] @ [3, 2] → [4, 2]
    let output = input.dot(&weights)?;

    let result = output.realize()?;
    println!("Output shape: {:?}", result.shape());  // [4, 2]
    println!("{:?}", result.to_ndarray::<f32>()?);
    // Each row: weighted sum of that sample's features

    Ok(())
}
```

**`dot()` 的形状规则：**

| 左操作数 | 右操作数 | 结果 |
|------|-------|--------|
| `[M, K]` | `[K, N]` | `[M, N]` |
| `[K]` | `[K, N]` | `[N]`（向量-矩阵） |
| `[M, K]` | `[K]` | `[M]`（矩阵-向量） |
| `[B, M, K]` | `[B, K, N]` | `[B, M, N]`（批量） |

内部维度必须匹配（即 `K`）。可以这样理解："对左矩阵的每一行，与右矩阵的每一列做点积。"

---

## 示例 4：构建线性层

线性层计算 `y = x @ W.T + b`。我们从头构建一个。

```rust
use morok_tensor::{Tensor, Error};

struct Linear {
    weight: Tensor,  // shape: [out_features, in_features]
    bias: Tensor,    // shape: [out_features]
}

impl Linear {
    fn new(in_features: usize, out_features: usize) -> Self {
        // Simple initialization (real code would use proper random init)
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|i| (i as f32 * 0.1).sin() * 0.1)
            .collect();
        let bias_data = vec![0.0f32; out_features];

        Self {
            weight: Tensor::from_slice(weight_data)
                .try_reshape(&[out_features as isize, in_features as isize])
                .expect("reshape failed"),
            bias: Tensor::from_slice(bias_data),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // y = x @ W.T + b
        let weight_t = self.weight.try_transpose(0, 1)?;
        let out = x.dot(&weight_t)?;
        Ok(&out + &self.bias)
    }
}

fn linear_example() -> Result<(), Box<dyn std::error::Error>> {
    // Create a layer: 4 inputs → 2 outputs
    let layer = Linear::new(4, 2);

    // Single sample with 4 features
    let input = Tensor::from_slice([1.0f32, 2.0, 3.0, 4.0]);

    // Forward pass
    let output = layer.forward(&input)?;

    let result = output.realize()?;
    println!("Output: {:?}", result.to_ndarray::<f32>()?);

    Ok(())
}
```

**为什么要转置权重？**

PyTorch 惯例将权重存储为 `[out_features, in_features]`。对于一个 4 → 2 的映射层：
- 权重形状：`[2, 4]`
- 输入形状：`[4]` 或 `[batch, 4]`
- 我们需要：`input @ weight.T` = `[batch, 4] @ [4, 2]` = `[batch, 2]`

这种惯例使权重矩阵易于阅读：第 `i` 行包含所有输入到第 `i` 个输出的权重。

---

## 示例 5：MNIST 分类器

构建一个完整的神经网络，可以对手写数字进行分类。

```rust
/// Two-layer neural network for MNIST
/// Architecture: 784 (28×28 pixels) → 128 (hidden) → 10 (digits)
struct MnistNet {
    fc1: Linear,
    fc2: Linear,
}

impl MnistNet {
    fn new() -> Self {
        Self {
            fc1: Linear::new(784, 128),
            fc2: Linear::new(128, 10),
        }
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        // Layer 1: linear + ReLU activation
        let x = self.fc1.forward(x)?;
        let x = x.relu()?;

        // Layer 2: linear (no activation — raw logits)
        self.fc2.forward(&x)
    }

    fn predict(&self, x: &Tensor) -> Result<Tensor, Error> {
        let logits = self.forward(x)?;
        // Convert logits to probabilities
        logits.softmax(-1)
    }
}

fn mnist_example() -> Result<(), Box<dyn std::error::Error>> {
    let model = MnistNet::new();

    // Simulate a 28×28 grayscale image (flattened to 784)
    let fake_image: Vec<f32> = (0..784)
        .map(|i| (i as f32) / 784.0)
        .collect();
    let input = Tensor::from_slice(fake_image)
        .try_reshape(&[1, 784])?;  // batch size 1

    // Forward pass
    let logits = model.forward(&input)?;
    let probs = logits.softmax(-1)?;

    // Get results
    let probs_result = probs.realize()?;
    println!("Probabilities: {:?}", probs_result.to_ndarray::<f32>()?);

    // Get predicted class
    let prediction = logits.argmax(Some(-1))?;
    let pred_result = prediction.realize()?;
    println!("Predicted digit: {:?}", pred_result.to_ndarray::<i32>()?);

    Ok(())
}
```

**核心概念：**

1. **ReLU 激活函数：** `x.relu()` 返回 `max(0, x)`。它引入非线性——没有它的话，堆叠线性层只相当于一个大的线性层。

2. **Logits 与概率：** 最后一层的原始输出（logits）可以是任意实数。`softmax()` 将它们转换为总和为 1 的概率。

3. **argmax：** 返回最大值的索引——即预测的类别。

4. **批维度：** 单张图像使用形状 `[1, 784]`。如果有 32 张图像，使用 `[32, 784]`。模型会自动处理批次。

---

## 示例 6：深入内部

想看看 Morok 生成了什么？以下是如何查看 IR 和生成的代码。

```rust
fn inspect_compilation() -> Result<(), Box<dyn std::error::Error>> {
    let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
    let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
    let c = &a + &b;

    // Print the computation graph (before compilation)
    println!("=== IR Graph ===");
    println!("{}", c.uop().tree());

    // Compile and execute
    let result = c.realize()?;

    // Inspect generated kernels
    println!("\n=== Generated Kernels ===");
    for (i, kernel) in result.kernels().iter().enumerate() {
        println!("Kernel {}: {}", i, kernel.name);
        println!("Backend: {}", kernel.backend);
        println!("Code:\n{}\n", kernel.code);
    }

    Ok(())
}
```

**你会看到：**

1. **IR 图：** UOp 树展示了 `BUFFER`、`LOAD`、`ADD`、`STORE` 等操作。这是 Morok 在优化之前的中间表示。

2. **生成的代码：** 实际运行的 LLVM IR 或 GPU 代码。注意 Morok 是如何将 load 和 add 融合到一个 kernel 中的——无需中间缓冲区。

**调试技巧：** 如果某些操作看起来慢或不对，打印 IR 树。注意检查：
- 意外的操作（冗余的 reshape、多余的拷贝）
- 缺少融合（本可以用一个 kernel 完成的地方却用了多个）
- 形状不匹配（通常是错误的根本原因）

---

## 总结

你已经学会了使用 Morok 的核心模式：

| 任务 | 代码 |
|------|------|
| 创建张量 | `Tensor::from_slice([1.0f32, 2.0])` |
| 算术运算 | `&a + &b`, `&a * &b`, `-&a` |
| 重塑形状 | `t.try_reshape(&[2, 3])?` |
| 转置 | `t.try_transpose(0, 1)?` |
| 矩阵乘法 | `a.dot(&b)?` |
| 激活函数 | `t.relu()?`, `t.softmax(-1)?` |
| 执行 | `t.realize()?` |
| 提取数据 | `result.to_ndarray::<f32>()?` |

**惰性求值模式：**

1. 用各种操作构建计算图
2. 最后调用一次 `realize()`
3. Morok 统一优化并执行所有操作

**下一步：**

- [Op 手册](./architecture/op-bestiary) — IR 操作参考
- [执行流水线](./architecture/pipeline) — 编译过程详解
- [优化系统](./architecture/optimizations) — 基于模式的重写
