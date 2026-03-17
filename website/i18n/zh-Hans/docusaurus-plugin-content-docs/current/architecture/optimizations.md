---
sidebar_label: 优化系统
---

# 基于模式的优化

打开任何一个生产级 ML 编译器，你会发现几十个优化 pass：常量折叠、死代码消除、算子融合、循环分块、向量化、内存布局优化。每个 pass 都有自己的数据结构、遍历逻辑和 bug。

Morok 采用了不同的方案：**一种机制搞定一切**。

```text
Traditional Compiler:              Morok:
┌─────────────────────────┐       ┌─────────────────────────┐
│  Constant Folding       │       │                         │
│  Dead Code Elimination  │       │   patterns! {           │
│  Loop Unrolling         │       │       Add[x, @zero] ~> x│
│  Operator Fusion        │       │       Mul[x, @zero] ~> 0│
│  Vectorization          │       │       // ...more        │
│  Memory Planning        │       │   }                     │
│  ...20 more passes      │       │                         │
└─────────────────────────┘       │   graph_rewrite(...)    │
     Custom logic each            └─────────────────────────┘
                                       One mechanism
```

Morok 中的每一项优化都表达为一个**模式**："当你看到这种结构，就替换为那种结构。"同一个 `graph_rewrite()` 函数负责常量折叠、将变换操作转换为循环、优化内存访问模式，以及降级到硬件原语。

本章解释基于模式的优化如何工作以及为什么它很强大。

---

## `patterns!` DSL

Morok 提供了一种领域特定语言来编写优化模式。它长这样：

```rust
patterns! {
    // Identity folding: x + 0 → x
    Add[x, @zero] ~> |x| x.clone(),

    // Constant folding: 3 + 4 → 7
    Add(a @const(a_val), b @const(b_val))
        => |a, a_val, b_val| eval_add(a_val, b_val).map(|r| UOp::const_(a.dtype(), r)),

    // Self-folding: x / x → 1
    Idiv(x, x) ~> |x| UOp::one(x.dtype()),

    // Dead code elimination: if(true) { t } else { f } → t
    Where(@true, t, _f) ~> |t| t.clone(),
}
```

宏将这些模式编译为高效的 Rust 代码。语法详解：

| 语法 | 含义 | 示例 |
|--------|---------|---------|
| `(x, y)` | **有序。** 按精确顺序匹配。 | `Sub(x, @zero) ~> x` |
| `[x, y]` | **交换律。** 尝试两种顺序。 | `Add[x, @zero] ~> x` |
| `@zero` | **零常量。** 匹配 0 或 0.0。 | `Mul[_, z @ @zero] ~> z` |
| `@one` | **一常量。** 匹配 1 或 1.0。 | `Mul[x, @one] ~> x` |
| `@const(val)` | **提取常量。** 绑定值。 | `Add(@const(a), @const(b))` |
| `x, x` | **同一操作数。** 自动生成 ptr_eq 检查。 | `Idiv(x, x) ~> UOp::one(...)` |
| `~>` | **不会失败。** 总是成功，返回 `Arc<UOp>`。 | `Add[x, @zero] ~> x` |
| `=>` | **可能失败。** 返回 `Option<Arc<UOp>>`。 | `=> eval(...).map(...)` |
| `for op in binary [...]` | **模板。** 为多个操作生成模式。 | 见下文 |
| `@context Type` | **有状态。** 在模式中访问可变上下文。 | 见下文 |

### 模板展开

不必为每个二元操作写同样的模式，用 for 循环：

```rust
patterns! {
    for op in binary [Add, Mul, Sub, Idiv, Fdiv, Max] {
        op(a @const(a_val), b @const(b_val))
            => |a, a_val, b_val| eval_binary(op, a_val, b_val)
                .map(|r| UOp::const_(a.dtype(), r))
    }
}
```

这在编译期展开为六个独立的模式——每个操作一个。

### 有状态模式

某些优化需要上下文（比如当前在哪个 kernel、哪些范围是活跃的）。声明一个上下文类型：

```rust
patterns! {
    @context KernelContext;

    ReduceAxis { src } => |reduce, src, ctx| {
        ctx.record_reduction(reduce);
        transform_reduce(reduce, src, ctx)
    }
}
```

上下文作为最后一个参数传递给模式闭包。

---

## 模式匹配的工作原理

`patterns!` 宏生成一个 `SimplifiedPatternMatcher`，可以在 **O(1)** 时间内分派模式。

### OpKey 索引

每个 UOp 都有一个操作类型（Add、Mul、Load 等）。`#[derive(PatternEnum)]` 宏生成一个 `OpKey` enum，将操作映射为可哈希的键：

```rust
pub enum OpKey {
    Binary(BinaryOp),    // Add, Mul, Sub, ...
    Unary(UnaryOp),      // Neg, Sqrt, Exp, ...
    Ternary(TernaryOp),  // Where, MulAcc
    Const,
    Load,
    Store,
    // ... one variant per operation category
}
```

### Matcher 结构

```rust
pub struct SimplifiedPatternMatcher<C = ()> {
    indexed: HashMap<OpKey, Vec<PatternClosure<C>>>,  // O(1) lookup
    wildcards: Vec<PatternClosure<C>>,                 // patterns matching any op
}
```

匹配一个 UOp 时：

1. 从 UOp 的操作中**提取 OpKey**
2. 在 HashMap 中**查找**——O(1)
3. **逐一尝试闭包**直到有一个匹配
4. 如果没有索引模式匹配，**回退**到通配符

这比线性扫描所有模式快 5-10 倍。

### 交换律处理

对于 `Add[x, @zero]` 这样的模式，宏生成尝试两种顺序的代码：

```rust
// Try (x, @zero)
if let Some(result) = try_match_ordered(&children[0], &children[1]) {
    return result;
}
// Try (@zero, x)
if let Some(result) = try_match_ordered(&children[1], &children[0]) {
    return result;
}
```

### 重复检测

当你写 `Idiv(x, x)` 时，模式应该只在两个操作数是*同一个* UOp（指针相等，不是结构相等）时匹配。宏自动生成这个检查：

```rust
// Generated code for Idiv(x, x)
let x = &children[0];
let x_dup = &children[1];
if !Arc::ptr_eq(x, x_dup) {
    return NoMatch;
}
// ... rest of pattern
```

这利用了 hash consing——相同的子表达式共享同一个指针。

---

## 重写引擎：两阶段算法

仅有模式匹配还不够。考虑这个表达式：

```text
WHERE(Lt(3, 5), t, f)
```

要化简它，需要两步：
1. `Lt(3, 5)` → `true`（常量折叠）
2. `WHERE(true, t, f)` → `t`（死代码消除）

但 `WHERE` 模式在子节点被化简之前不会匹配。重写引擎通过**两阶段算法**解决这个问题。

### 阶段 0：模式应用

```rust
fn rewrite_stage0(&mut self, uop: &Arc<UOp>) -> RewriteResult {
    match self.matcher.try_match(uop) {
        Some(replacement) => RewriteResult::Rewritten(replacement),
        None => RewriteResult::Gate(uop.clone()),  // process children
    }
}
```

如果没有模式匹配，返回 `Gate`——表示先处理子节点的信号。

### 阶段 1：源重建

子节点被重写后，用新的子节点重建节点，再次尝试模式：

```rust
fn rewrite_stage1(&mut self, uop: &Arc<UOp>, new_children: Vec<Arc<UOp>>) {
    // Rebuild with optimized children
    let rebuilt = uop.with_sources(new_children);

    // Try patterns again—might match now!
    match self.matcher.try_match(&rebuilt) {
        Some(replacement) => replacement,
        None => rebuilt,
    }
}
```

### 关键：级联优化

```text
Stage 0: WHERE(Lt(3, 5), t, f)     → Gate (no match, process children)
         └── Lt(3, 5)              → true (constant folding matches!)

Stage 1: WHERE(true, t, f)         → t (dead code elimination matches!)
```

重建阶段重新应用模式，使多步优化在一次遍历中完成。

### 安全限制

为防止无限循环，引擎有限制：
- 每个节点最多 **1000 次迭代**
- 总计最多 **100,000 次迭代**
- 超限时 panic 并输出诊断信息

在实践中，良好的模式能快速收敛。

---

## 完整优化流水线

模式匹配是更大流水线的一部分。当你调用 `tensor.realize()` 时，以下是发生的过程：

```text
Tensor.realize()
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  RANGEIFY                                               │
│  Convert movement ops (RESHAPE, PERMUTE, EXPAND)        │
│  into explicit RANGE loops with INDEX operations        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  KERNEL SPLITTING                                       │
│  Split computation graph at STORE boundaries            │
│  Each STORE becomes a separate kernel                   │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  FOR EACH KERNEL:                                       │
│                                                         │
│  1. Symbolic Simplification (algebraic patterns)        │
│                                                         │
│  2. Scheduler Creation                                  │
│     └── Convert LOOP → GLOBAL for GPU parallelization   │
│                                                         │
│  3. Kernel Optimization (heuristic OR beam search)      │
│     ├── Tensor Cores (WMMA) for matmul                  │
│     ├── Vectorization (UPCAST)                          │
│     ├── Loop Unrolling (UNROLL)                         │
│     ├── GPU Local Memory (LOCAL)                        │
│     ├── Grouped Reductions (GROUP)                      │
│     └── Threading (THREAD) for CPU                      │
│                                                         │
│  4. Post-Optimization Passes                            │
│     ├── Devectorize (memory coalescing)                 │
│     ├── Expand (UNROLL → vector operations)             │
│     ├── FMA Decomposition (a*b+c → MulAcc)              │
│     └── Bool Storage (cast bool↔uint8 for memory)       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  CODE GENERATION                                        │
│  Render optimized AST to LLVM IR, compile, execute      │
└─────────────────────────────────────────────────────────┘
```

每个框都使用基于模式的重写。不同之处在于应用哪些模式：

- **Rangeify**：变换操作 → BUFFERIZE + INDEX 模式
- **符号化简**：代数化简模式
- **后优化**：内存访问优化模式

---

## Kernel 优化：启发式 vs Beam 搜索

符号化简之后，每个 kernel 需要*调度决策*：如何分块循环、在哪里并行化、是否使用 tensor core。Morok 提供两种策略。

### 启发式（默认）

启发式优化器按固定顺序应用优化：

```rust
pub fn hand_coded_optimizations(scheduler: &mut Scheduler) {
    // 1. Tensor cores (if matmul pattern detected)
    if let Some(tc) = detect_tensor_core_pattern(scheduler) {
        apply_tensor_core(scheduler, tc);
        return;  // TC handles everything
    }

    // 2. Grouped reductions (two-stage for large reductions)
    apply_grouped_reduction_if_needed(scheduler);

    // 3. Vectorization (UPCAST output dimensions)
    apply_upcast(scheduler, 4);

    // 4. GPU local memory (workgroup dimensions)
    apply_local_dims(scheduler);

    // 5. CPU threading
    apply_threading(scheduler);
}
```

**优点**：快（每个 kernel 约 50ms）、可预测、无需硬件测量。

**缺点**：可能错过优化机会，固定启发式不能适应不同工作负载。

### Beam 搜索（可选）

对于生产工作负载，beam 搜索能找到更好的调度方案：

```rust
pub fn beam_search(scheduler: Scheduler, config: BeamConfig) -> Scheduler {
    let mut beam = vec![scheduler];

    for iteration in 0..config.max_iterations {
        let mut candidates = vec![];

        for state in &beam {
            // Generate all valid next actions
            for action in generate_actions(state) {
                if let Ok(next) = state.apply(action) {
                    candidates.push(next);
                }
            }
        }

        // Compile and time each candidate
        let timed: Vec<_> = candidates.par_iter()
            .map(|c| (c, measure_kernel_time(c)))
            .collect();

        // Keep top K by execution time
        beam = timed.into_iter()
            .sorted_by_key(|(_, time)| *time)
            .take(config.beam_width)
            .map(|(c, _)| c)
            .collect();
    }

    beam.into_iter().next().unwrap()
}
```

动作空间包含约 500 个预定义动作：
- `UPCAST(axis, amount)` — 向量化输出维度
- `UNROLL(axis, amount)` — 展开规约循环
- `LOCAL(axis, amount)` — 使用 GPU 共享内存
- `GROUP(axis, amount)` — 两阶段规约
- `THREAD(axis, amount)` — CPU 并行化
- `SWAP(axis1, axis2)` — 重排全局维度

**优点**：找到接近最优的调度方案，能适应硬件。

**缺点**：每个 kernel 需要几分钟（但结果按 AST 哈希缓存）。

### 配置

```bash
# Disable optimization (debugging)
MOROK_NOOPT=1 cargo run

# Enable beam search with width 8
MOROK_BEAM=8 cargo run
```

或通过代码配置：

```rust
let config = OptimizerConfig::builder()
    .strategy(OptStrategy::Beam { width: 8 })
    .build();

tensor.realize_with(config)?;
```

---

## 对比：其他编译器如何优化

不同的 ML 编译器采用不同的优化方式：

| 方面 | XLA | TVM/Ansor | Triton | **Morok** |
|--------|-----|-----------|--------|-----------|
| **理念** | 固定启发式 | 基于搜索 | 程序员引导 | 基于模式 |
| **融合** | 保守规则 | Tile-and-fuse | 块级别 | 图重写 |
| **自动调优** | 无 | 进化算法 + 代价模型 | 网格搜索 | Beam 搜索 |
| **调优成本** | 0 | 数小时 | 数分钟 | 数分钟（有缓存） |
| **灵活性** | 低 | 高 | 中 | 高 |
| **透明度** | 低（C++ pass） | 中（Python） | 中（DSL） | 高（patterns!） |

### XLA — 生产级保守

XLA 使用固定启发式做融合决策。安全可预测，但会损失一些性能。融合规则硬编码在 C++ 中——扩展它们需要深入的编译器知识。

### TVM/Ansor — 极致自动调优

TVM 将*计算什么*和*如何计算*分离。Ansor 使用进化搜索配合学习的代价模型来探索调度空间。可以达到业界最佳性能，但每个模型调优需要数小时。

### Triton — 程序员引导

Triton 提供一个类 Python 的 DSL，让你显式编写分块算法。编译器处理寄存器分配和内存管理。在控制和自动化之间取得了良好平衡，但需要 GPU 编程专业知识。

### Morok — 模式组合

Morok 的洞察：将优化表达为可组合的模式。每个模式是局部的、可验证的。复杂优化通过组合涌现。Beam 搜索在需要时提供自动调优，结果缓存可复用。

---

## 为什么这很重要：实际好处

基于模式的优化对开发者有具体的优势：

**调试是直接的。** 模式是可读的代码。在任何模式中加一个 `println!` 来追踪何时触发：

```rust
patterns! {
    Add[x, @zero] ~> |x| {
        println!("Folding add-zero: {:?}", x);
        x.clone()
    }
}
```

**扩展很容易。** 添加自定义优化只需两行：

```rust
patterns! {
    // Your domain-specific optimization
    MyOp(x, y) if is_special_case(x, y) ~> transform(x, y)
}
```

不需要理解编译器内部实现、编写 visitor 或修改 pass 管理器。

**正确性是局部的。** 每个模式都是一个小定理："如果出现这种结构，用那种结构替换可以保持语义。"可以独立验证每个模式。正确模式的组合产生正确的程序。

**性能可调。** O(1) 模式分派默认就很快。对生产工作负载启用 beam 搜索。按 AST 哈希缓存结果——调优一次，永久受益。

---

## 更深层的洞察

模式匹配用通用性换取了可组合性。

通用优化 pass 可以做任何事——但这恰恰是问题所在。它难以验证、难以扩展、难以与其他 pass 组合。顺序很重要。交互很微妙。

模式是受约束的：它匹配特定结构，产生特定替换。但约束使组合成为可能。以任何顺序运行模式——结果收敛到同一个不动点。添加新模式不会破坏现有的。删除模式不会产生级联故障。

每个模式都是关于语义等价的定理。重写引擎是定理证明器，从输入到优化输出寻找推导路径。正确性来自每一步的正确性。

这是 Unix 哲学在编译器中的应用：小的、专注的工具进行组合。基于模式的优化不能解决所有问题——但对于它能解决的问题，它解决得很优雅。
