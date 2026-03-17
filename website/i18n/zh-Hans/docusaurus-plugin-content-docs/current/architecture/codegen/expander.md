---
sidebar_label: 阶段 2 — Expander
---

# 阶段 2：Expander

**目标**：将优化原语（UNROLL/UPCAST）转换为显式操作。

---

## Stage 8：优化后符号化简

> **阶段速览**
>
> **目标**：优化之后的符号化简
> **关键模式**：WHERE 移动、常量折叠
> **影响**：改善 load 合并和向量化

**做了什么**：优化之后的符号化简，外加 WHERE 移动。

**为什么重要**：WHERE 操作类似于 `if` 语句。这个阶段将 `if` 检查从 load 之后移到 load 之前。硬件可以在条件为 false 时跳过加载，节省内存带宽。

**模式**：`sym + pm_move_where_on_load`

```text
// Before: WHERE guards a load
WHERE(valid, LOAD(index), alt)

// After: validity moved to INDEX
LOAD(INDEX(ptr, idx, valid=valid), alt)
```

将 validity 移入 INDEX 可以改善 load 合并和向量化。

**注意**：该模式仅在替代值为 `0` 时匹配。变换涉及复杂的子句分析：重复检测、range 依赖检查和数据依赖 load 验证。

**注意**：Morok 实现使用 `gate=` 而不是 `valid=`（Index 结构体有一个 `gate` 字段）。概念完全相同。

**Morok**：`symbolic/patterns.rs` 中的 `pm_move_where_on_load()`

---

## Stage 9：Expander

> **阶段速览**
>
> **目标**：将 UNROLL/UPCAST 转换为显式操作
> **关键概念**：UNROLL、CONTRACT、模式顺序
> **影响**：使向量化变得显式，为硬件做好准备

**做了什么**：将 UNROLL/UPCAST 优化原语转换为显式操作。

**为什么重要**：UPCAST 和 UNROLL 标记的是意图——我们想做什么。这个阶段将意图变为现实，让硬件能够实际执行。

**模式**：`symbolic_simple() + pm_pre_expander + pm_group_for_reduce + expander`

注意：Morok 在此阶段使用 `symbolic_simple()`（不是 `sym`），因为 `symbolic()` 已在 Stage 4 运行过。Tinygrad 使用 `sym`，包含额外的模式。

⚠️ **重要：模式优先级**

这些模式被组合在一起运行直到不动点。顺序会影响当多个模式都能匹配时哪个先尝试：
1. `sym` 优先（符号化简）
2. `pm_pre_expander` 其次（转换 UPCAST/UNROLL range）
3. `pm_group_for_reduce` 第三（处理 GROUP_REDUCE 轴）
4. `expander` 最后（主展开）

错误的优先级可能导致向量化或规约作用域不正确。

**UNROLL 和 CONTRACT**：

UNROLL 和 CONTRACT 协同工作：

```text
UNROLL: "Take this one thing and make N copies for different positions"
Example:  x → [x_0, x_1, x_2, x_3]

CONTRACT: "Take these N things and combine them back"
Example:  [a, b, c, d] → one vector containing all four
```

二者配合：UPCAST 标记向量化意图 → UNROLL 展开 → CONTRACT 组合。

**UPCAST range → VECTORIZE**：
```text
// Before: UPCAST marks vectorization intent
RANGE(end=4, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL with constant indices
UNROLL(VCONST([0, 1, 2, 3]))
      ↓ [expander]
// Step 2: Expand operations with UNROLL sources
// Operations now have unrolled sources
      ↓ [CONTRACT or implicit]
// After: explicit VECTORIZE
VECTORIZE(op[0], op[1], op[2], op[3])
```

**UNROLL range → 重复操作**：

当我们说"操作被复制"时，听起来像是复制粘贴。但实际上不是。编译器创建的是单条 SIMD 指令，同时处理所有 N 个元素。把 SIMD 寄存器想象成一个装着 4 个数字的盒子；两个盒子相加就是 8 个数字同时相加。

```text
// Before: UPCAST marks vectorization intent
RANGE(end=3, UPCAST)
      ↓ [pm_pre_expander]
// Step 1: Convert to UNROLL
UNROLL(VCONST([0, 1, 2]))
      ↓ [expander]
// Step 2: Operations expand to handle all positions
// After: operations processed together (not duplicated)
UNROLL([op_at_0, op_at_1, op_at_2])
```

**UNROLL/END/CONTRACT 交互**：
```text
Before: END(STORE(...), [RANGE(UPCAST)])
             ↓ [pm_pre_expander]
Step 1: END(STORE(...), [UNROLL(VCONST([0,1,2,3]))])
             ↓ [expander]
Step 2: END(CONTRACT(STORE(...×4)), [])
```

**穿过 AFTER/END 的广播**：
```text
// Broadcast VECTORIZE (all elements identical)
AFTER(VECTORIZE([x, x, x, x]), deps) → VECTORIZE([AFTER(x, deps), AFTER(x, deps), ...])
```

**GROUP_REDUCE 处理**（`pm_group_for_reduce`）：

GROUP_REDUCE 是张量核心规约的特殊轴类型：

```text
// Before: REDUCE with GROUP_REDUCE ranges
REDUCE(src, [range(GROUP_REDUCE)])
           ↓ [pm_group_for_reduce]
// After: Shared memory reduction pattern
1. Track upstream LOCAL ranges
2. BUFFERIZE result with group ranges (AddrSpace.LOCAL)
3. INDEX into buffer with transformed ranges
4. Final REDUCE with axes (range_id+100, AxisType.REDUCE)
```

这实现了通过共享内存进行高效的张量核心累加。

**Morok**：`expand.rs`

---

## Stage 10：添加本地 Buffer

> **阶段速览**
>
> **目标**：为快速内存（共享 / L1）准备 buffer
> **关键模式**：带 locals 的 bufferize、提取提示
> **影响**：频繁访问的数据留在快速内存中

**做了什么**：为本地内存使用准备 buffer，并应用代码生成特定的清理。

**为什么重要**：**本地 buffer** = 靠近计算单元的快速内存：
- GPU：共享内存（LDS）——比全局内存快 100 倍
- CPU：L1 缓存——比主内存快 10 倍

编译器将频繁访问的数据移到本地 buffer，就像把重要文件放在桌面而不是网络驱动器上一样。

**模式**：`pm_add_buffers_local + rangeify_codegen`

| 变换 | 用途 |
|------|------|
| `bufferize_to_store` | 转换 `allow_locals=true` 的 BUFFERIZE |
| 移除 CONTIGUOUS 包装器 | 在代码生成前移除优化提示 |
| 移除 NOOP | 清理无操作 |

**Morok**：`rangeify/patterns.rs`、`rangeify/transforms.rs`、`optimizer/mod.rs`
