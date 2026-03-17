---
sidebar_label: ONNX 推理
---

# ONNX 模型推理

Morok 的 ONNX 导入器是运行模型推理的推荐方式。它加载标准的 `.onnx` 文件，将算子分解为 Morok 的惰性张量操作，并通过完整的优化流水线编译执行——无需 C++ 运行时。

**当前状态：**

| 能力 | 状态 |
|------|------|
| 前向推理 | 已支持 |
| 162 / 200 个 ONNX 算子 | [算子对齐详情](https://github.com/patsak/morok/blob/main/onnx/PARITY.md) |
| CNN 架构（ResNet、DenseNet、VGG 等） | 已验证 9 个模型 |
| Microsoft 扩展（Attention、RotaryEmbedding） | 已支持 |
| 动态批大小 | 计划在下一版本中支持 |
| 训练 / 反向传播 | 不支持 |

**与其他框架的比较**

在纯 Rust 框架中，Morok 的 ONNX 算子覆盖面最广——162 个算子，双后端（Clang + LLVM）上通过 1361 项一致性测试。`candle` 和 `burn` 支持的算子更少，也没有同等规模的测试套件。如果需要与生产环境 ONNX 模型的最大兼容性，用 `ort`——C++ ONNX Runtime 的 Rust 封装，覆盖完整的 ONNX 规范。

---

## 快速开始

在你的 `Cargo.toml` 中添加 `morok-onnx` 和 `morok-tensor`：

```toml
[dependencies]
morok-onnx = { git = "https://github.com/patsak/morok" }
morok-tensor = { git = "https://github.com/patsak/morok" }
```

### 简单用法：全初始化器模型

对于所有输入都内嵌在文件中（无运行时输入）的模型：

```rust
use morok_onnx::OnnxImporter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let outputs = importer.import_path("model.onnx")?;

    // Each output is a lazy Tensor — realize to get data
    for (name, tensor) in &outputs {
        let result = tensor.realize()?;
        println!("{name}: {:?}", result.to_ndarray::<f32>()?);
    }
    Ok(())
}
```

### 两阶段用法：带运行时输入的模型

大多数模型需要运行时数据（图像、token、音频）。两阶段 API 将图构建与执行分离：

```rust
use morok_onnx::OnnxImporter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let importer = OnnxImporter::new();

    // Phase 1: Parse structure (no execution)
    let graph = importer.prepare(model_proto)?;

    // Inspect what the model needs
    for (name, spec) in &graph.inputs {
        println!("{name}: shape={:?}, dtype={:?}", spec.shape, spec.dtype);
    }

    // Phase 2: Build lazy computation graph
    let (inputs, outputs) = importer.trace(&graph)?;

    // Execute
    let result = outputs["output"].realize()?;
    Ok(())
}
```

`prepare()` / `trace()` 的分离设计是因为图结构是静态的——你只需解析一次，就可以使用不同的维度绑定或输入数据多次 trace。

---

## 架构

### 两阶段设计

导入器分两个阶段处理 ONNX 模型：

**阶段一 — `prepare()`：** 提取图拓扑结构，不执行任何计算。解析 protobuf，将初始化器（权重）与运行时输入分离，记录 opset 版本，并预解析控制流子图。返回 `OnnxGraph`——一个轻量级结构，可以在执行前检查。

**阶段二 — `trace()`：** 按拓扑顺序遍历图，将每个 ONNX 节点分派给对应的 Tensor 实现。这将构建 Morok 的惰性计算 DAG——此时不会进行实际计算。结果是一组 `Tensor` 句柄，当调用 `realize()` 时，才会编译并通过完整流水线执行。

```text
model.onnx → prepare() → OnnxGraph → trace() → lazy Tensors → realize() → results
                 │                        │
                 │ structure only         │ builds computation DAG
                 │ (no execution)         │ (no execution)
                 ▼                        ▼
          Inspect inputs/outputs    Pass to optimizer/codegen
```

这样做有几个好处：
- **执行前检查：** 在分配任何资源之前检查输入形状和 DType
- **多次 trace：** 使用不同的动态维度绑定重新 trace
- **外部权重：** 单独加载权重（适用于带有外部数据文件的模型）

### 算子分解

每个 ONNX 算子都会分解为 Morok Tensor 操作，复杂程度不一：

**直接映射** — 约 60 个算子与 tensor 方法一一对应：

```rust
// In the registry:
"Add" => x.try_add(y)?
"Relu" => x.relu()?
"Sigmoid" => x.sigmoid()?
"Equal" => x.try_eq(y)?
```

**Builder 模式** — 带有多个可选参数的复杂算子使用流式 API：

```rust
// Conv with optional bias, padding, dilation, groups
x.conv()
    .weight(w)
    .maybe_bias(bias)
    .auto_pad(AutoPad::SameLower)
    .group(32)
    .maybe_dilations(Some(&[2, 2]))
    .call()?
```

**多步分解** — BatchNormalization、Attention 和 Mod 等算子需要中间计算。例如，Python 风格的整数 `Mod` 被分解为截断取模 + 符号调整：

```rust
let trunc_mod = x.try_mod(y)?;
let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
```

### 属性验证

`Attrs` 辅助工具使用弹出式提取——每次调用 `attrs.int("axis", -1)` 或 `attrs.float("epsilon", 1e-5)` 都会从映射中移除该属性。算子处理完成后，`attrs.done()` 断言映射为空。任何剩余属性都会触发错误，在 trace 时捕获不完整的算子实现，而不是产生静默的错误结果。

### Opset 版本管理

ONNX 模型按域声明 opset 导入。导入器跟踪这些信息并将版本传递给每个算子处理器。算子根据版本切换行为——例如，`Softmax` 的默认轴从 `1`（opset < 13）变为 `-1`（opset >= 13），而 `ReduceSum` 在 opset 13 时将其轴从属性移至输入张量。

---

## 使用模型

### 动态维度

ONNX 输入可以有符号维度，如 `"batch_size"` 或 `"sequence_length"`。在 trace 时绑定它们：

```rust
let graph = importer.prepare(model)?;

// Bind symbolic dims to concrete values
let (inputs, outputs) = importer.trace_with_dims(
    &graph,
    &[("batch_size", 1), ("sequence_length", 512)],
)?;
```

未绑定的动态维度会在 trace 时产生明确的错误。你可以通过 `InputSpec::shape` 检查哪些维度是动态的：

```rust
for (name, spec) in &graph.inputs {
    for dim in &spec.shape {
        match dim {
            DimValue::Static(n) => print!("{n} "),
            DimValue::Dynamic(name) => print!("{name}? "),
        }
    }
}
```

### 外部权重

一些 ONNX 模型将权重存储在单独的文件中。使用 `trace_external()` 来提供它们：

```rust
let (inputs, outputs) = importer.trace_external(
    &graph,
    external_weights,  // HashMap<String, Tensor>
)?;
```

### Microsoft 扩展

导入器支持多个 `com.microsoft` 贡献算子，这些算子常见于从 ONNX Runtime 导出的 transformer 模型中：

| 扩展 | 功能说明 |
|------|---------|
| `Attention` | 打包的 QKV 投影，支持掩码和历史 KV cache |
| `RotaryEmbedding` | 旋转位置编码（交错/非交错） |
| `SkipLayerNormalization` | 融合的残差 + LayerNorm + 缩放 |
| `EmbedLayerNormalization` | Token + 位置 + 段落嵌入 → LayerNorm |

标准 ONNX transformer 算子（ai.onnx 域的 `Attention`）同样支持，包括分组查询注意力（GQA）、因果掩码、历史 KV cache 和 softcap。

---

## 控制流与局限性

### 语义 If：两个分支始终执行

ONNX 的 `If` 算子具有数据依赖的控制流——条件决定执行哪个分支。Morok 的惰性求值模型与此从根本上不兼容：由于 trace 时不执行任何计算，条件值是未知的。

**Morok 的解决方案：** 同时 trace *两个*分支，然后使用 `Tensor::where_()` 合并结果：

```text
ONNX:    if condition { then_branch } else { else_branch }
Morok:   then_result.where_(&condition, &else_result)
```

这实现了**一次 trace，多次运行**——编译后的图在运行时可以处理任何条件值。但它有一个硬性约束：**两个分支必须产生相同的输出形状和 DType。** 形状多态的模型（即 then 分支产生 `[3, 4]` 而 else 分支产生 `[5, 6]`）无法 trace。

在实践中，大多数带有 `If` 节点的 ONNX 模型都满足此约束，因为它们使用条件逻辑进行值选择，而非改变形状的控制流。

### 不支持 Loop 和 Scan

迭代控制流（`Loop`、`Scan`）尚未实现。这些算子需要重复 trace 或展开，这与单次 trace 架构冲突。使用循环模式的模型通常通过展开的算子工作（LSTM、GRU、RNN 已作为原生算子实现）。

### 不支持批处理（暂时）

动态批处理——同时对多个输入运行推理——计划在下一版本中支持。目前，批维度必须通过 `trace_with_dims()` 在 trace 时绑定到固定值。

### 不支持训练

导入器仅支持推理。没有反向传播、梯度计算或优化器支持。

### 缺失的算子类别

| 类别 | 示例 | 原因 |
|------|------|------|
| 量化 | DequantizeLinear、QuantizeLinear | 需要 IR 中的量化 DType 支持 |
| 序列操作 | SequenceConstruct、SequenceAt | 非张量类型不在 Morok 的类型系统中 |
| 随机数 | RandomNormal、RandomUniform | 有状态 RNG 尚未实现 |
| 信号处理 | DFT、STFT、MelWeightMatrix | 低优先级；小众用例 |
| 文本 | StringNormalizer、TfIdfVectorizer | 不支持字符串类型 |

用到这些算子的模型，可以用 `ort`（ONNX Runtime 封装），它覆盖完整规范。

---

## 调试

### 逐节点输出追踪

设置 trace 日志级别以输出中间结果：

```bash
RUST_LOG=morok_onnx::importer=trace cargo run
```

这会逐个 realize 每个节点的输出并打印前 5 个值——模型输出有误时可以用来做数值二分。注意这会破坏内核融合（每个节点单独运行），纯粹是调试用途。

### 检查图结构

trace 之前，用 `OnnxGraph` 看看模型需要什么：

```rust
let graph = importer.prepare(model)?;

println!("Inputs:");
for (name, spec) in &graph.inputs {
    println!("  {name}: {:?} {:?}", spec.shape, spec.dtype);
}

println!("Outputs: {:?}", graph.output_names());
println!("Nodes: {}", graph.nodes.len());
println!("Initializers: {}", graph.initializers.len());
```

---

## 总结

| 方面 | 详情 |
|------|------|
| **入口点** | `OnnxImporter::new()` |
| **简单导入** | `importer.import_path("model.onnx")?` |
| **两阶段** | `prepare()` → `trace()` / `trace_with_dims()` |
| **算子** | 162 / 200（[完整对齐表](https://github.com/patsak/morok/blob/main/onnx/PARITY.md)） |
| **已验证模型** | ResNet50、DenseNet121、VGG19、Inception、AlexNet、ShuffleNet、SqueezeNet、ZFNet |
| **后端** | Clang + LLVM（结果一致） |
| **扩展** | com.microsoft Attention、RotaryEmbedding、SkipLayerNorm、EmbedLayerNorm |
| **局限性** | 不支持训练、不支持批处理（暂时）、不支持 Loop/Scan、形状多态的 If |

**下一步：** [实践示例](./examples)——张量基础，或 [执行流水线](./architecture/pipeline)——了解编译是怎么跑的。
