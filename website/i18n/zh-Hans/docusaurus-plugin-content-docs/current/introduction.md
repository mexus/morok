---
sidebar_label: 简介
---

# Morok

> ⚠️ **Pre-alpha 阶段软件。** API 尚不稳定，可能随时更改。不建议用于生产环境。 🚧💀

Morok 是一个基于 Rust 的 ML 编译器，灵感来自 [Tinygrad](https://github.com/tinygrad/tinygrad)。它具有基于 UOp 的 IR 惰性张量求值、模式驱动优化和多后端代码生成。

## 亮点

| 特性 | 描述 |
|---------|-------------|
| **声明式优化** | `patterns!` DSL 实现图重写，通过 Z3 验证正确性 |
| **惰性求值** | Tensor 构建计算图，仅在 `realize()` 时编译执行 |
| **CUDA 支持** | 统一内存、D2D 拷贝、LRU 缓冲区缓存 |
| **溯源追踪** | `#[track_caller]` 将每个 UOp 追溯到源码位置 |
| **80+ IR 操作** | 算术、内存、控制流、WMMA tensor core |
| **20+ 优化** | 常量折叠、tensor core、向量化、循环展开 |

## 快速示例

```rust
use morok_tensor::Tensor;

// Build lazy computation graph
let a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
let b = Tensor::from_slice([4.0f32, 5.0, 6.0]);
let c = (&a + &b).sum();

// Compile and execute
let result = c.realize()?;
```

## 许可证

MIT
