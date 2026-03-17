---
sidebar_label: 阶段 4 — Linearizer
---

# 阶段 4：Linearizer

**目标**：将 DAG 转换为线性指令序列。

---

## Stage 16：索引降低后符号化简

> **阶段速览**
>
> **目标**：索引降低之后的完整符号化简
> **关键模式**：全部符号规则（140+）
> **影响**：序列化前的最终清理

**做了什么**：索引降低之后的完整符号化简。

**为什么重要**：现在索引已经是具体整数（i32/i64），算术可以充分化简。这是线性化之前清理表达式的最后机会。

**模式**：`symbolic`

包含 GEP 推送模式——将地址计算穿过算术运算：
```text
Before:  GEP(ADD(arr_a, arr_b), idx)
              ↓ [Push GEP through ADD]
After:   ADD(GEP(arr_a, idx), GEP(arr_b, idx))
```
*为什么？* 使 GEP 可以并行计算，并可能促进下游向量化。（注意：该模式仅在 GEP 的 dtype 和 ALU 的 dtype 都不是指针时适用。）

---

## Stage 17：Pre-Matcher（可选）

> **阶段速览**
>
> **目标**：分解之前的后端特定模式
> **关键模式**：Renderer 特定
> **影响**：硬件特定优化

**做了什么**：在分解之前应用 renderer 特定的模式。

**为什么重要**：每个后端可以添加自己的模式。例如，DSP 后端用这一步将通用模式替换为 DSP 特定的 SIMD 内联函数。这样无需修改通用流水线就能实现硬件特定优化。

**模式**：`renderer.pre_matcher`

大多数后端（CPU、GPU）不需要这一步。只有专用硬件使用它。

**注意**：Morok 目前未实现此阶段。`Renderer` trait 有 `render()`、`backend_name()` 和 `decompositor()` 方法，但还没有 `pre_matcher` 支持。这是未来为 DSP 等专用后端预留的增强。

---

## Stage 18：分解

> **阶段速览**
>
> **目标**：重写目标不支持的操作
> **关键模式**：2 的幂、超越函数近似
> **影响**：将高层操作映射到硬件指令

**做了什么**：对目标不支持的操作进行后期重写。

**为什么重要**：硬件并不支持所有操作。例如，大多数 CPU 没有直接的 `sin` 指令。我们用已有的操作（加法、乘法等）来近似它。

**模式**：`symbolic_simple() + get_late_rewrite_patterns()`

注意：`pm_render()` 和 `pm_split_ends()` 不属于此组合 pass——它们在 Stage 19 中单独运行。

| 模式 | 示例 | 使用场景 |
|------|------|----------|
| `MOD → AND` | `x % 8 → x & 7` | 2 的幂除数 |
| `MUL → SHL` | `x * 16 → x << 4` | 2 的幂乘数 |
| `DIV → SHR` | `x // 8 → x >> 3` | 2 的幂除数 |
| `FDIV → MUL` | `x / 2.0 → x * 0.5` | 浮点常量除数 |
| `NEG` | `x * -1 → NEG(x)` | 当支持 NEG 时 |
| `MULACC` | `a * b + c → MULACC(a, b, c)` | 当支持 FMA 时 |
| 快速整数除法 | `x // 7 → (x * M) >> S` | 非 2 的幂除数 |
| 德摩根定律 | `(!x) & (!y) → !(x \| y)` | 布尔化简（双向） |
| 比较取反 | `!(x < c) → (c-1) < x` | 整数比较 |

超越函数近似（SIN、EXP、LOG 等）通过 `decompositor()` 路径实现（参见 `ir/src/decompositions/transcendentals.rs`）。

**Morok**：`optimizer/mod.rs`

---

## Stage 19：最终重写

> **阶段速览**
>
> **目标**：为线性化做准备
> **关键模式**：CONST 向量化、GEP 解析、END 分割
> **影响**：为线性化准备好干净的表示

**做了什么**：为线性化做准备。

**为什么重要**：有些模式在分解之后更容易应用。这个阶段在转换为线性序列之前做最后的清理。

**模式**：`symbolic_simple() + get_late_rewrite_patterns() + pm_render()`

注意：`extra_matcher` 和 `pm_split_ends` 单独运行，不作为此组合 pass 的一部分。

**CONST 向量化**：
```text
// Make vector constants explicit
CONST(1.0) used as vec4 → VECTORIZE(1.0, 1.0, 1.0, 1.0)
```

**CAT 到 VECTORIZE**（通过 `pm_render`）：
```text
CAT(a, b, c, d) → VECTORIZE(a, b, c, d)
```
CAT 无法直接渲染；代码生成需要显式的 VECTORIZE。

**GEP 解析**：转换剩余的 GEP 操作。

**分割多 range END**：
```text
// Before: END closing multiple ranges
END(op, [range_a, range_b])

// After: nested single ENDs
END(END(op, range_a), range_b)
```

**extra_matcher**：每个后端可以添加自己的最终模式。这样无需修改通用流水线就能实现硬件特定优化。

**Morok**：`devectorize.rs`、`linearize/mod.rs`、`optimizer/mod.rs`

---

## Stage 20：添加控制流

> **阶段速览**
>
> **目标**：构建控制流图并添加 range 依赖
> **关键概念**：三种关系类型（嵌套、依赖、独立）
> **影响**：正确的指令排序

**做了什么**：构建控制流图并添加 range 依赖。

**为什么重要**：操作必须按有效顺序执行。如果一个 load 使用了 RANGE 的值，那么 RANGE 必须先执行。这个阶段跟踪并强制执行这些依赖。

**模式**：`pm_add_control_flow`（自底向上）

```text
// Analyze which END operations depend on which
END(computation, [RANGE_A]) and END(other_computation, [RANGE_B]) are siblings
→ Creates edge: RANGE_B.src += END(computation)

// Add explicit dependency
RANGE_B waits for RANGE_A to complete
```

**三种关系类型**：

| 关系 | 示例 | 含义 |
|------|------|------|
| 嵌套 | RANGE_A 在 RANGE_B 内部 | A 必须在 B 开始前完成 |
| 依赖 | END_A 和 END_B 是兄弟 | END_B 必须等待 END_A（兄弟依赖） |
| 独立 | RANGE_X 和 RANGE_Y 无交互 | 可以并行运行 |

自底向上遍历确保依赖从叶到根正确传播。

**Morok**：`schedule/src/linearize/mod.rs`

---

## Stage 21：线性化

> **阶段速览**
>
> **目标**：将 DAG 转换为线性指令序列
> **关键算法**：优先级感知的拓扑排序
> **影响**：有效的执行顺序

**做了什么**：通过优先级感知的拓扑排序将 DAG 转换为线性指令序列。

**为什么重要**：图结构不指定执行顺序。我们需要在尊重依赖的前提下将其展平。优先级确保合理的排序（定义在使用之前、load 在计算之前、store 在最后）。

**函数**：`linearize(sink)`

| 操作 | 优先级 | 原因 |
|------|--------|------|
| DEFINE_GLOBAL | -20 | 参数必须先定义 |
| DEFINE_VAR | -19 | 变量必须先定义 |
| DEFINE_LOCAL | -18 | 分配优先 |
| DEFINE_REG | -17 | 寄存器优先 |
| CONST | -10 | 常量尽早放置以便复用（Morok 扩展；Tinygrad 默认为 0） |
| LOAD | -1 | Load 在使用之前 |
| END | -5 | 关闭 range |
| STORE | +1 | Store 在计算之后 |
| RANGE | +5 | Range 在内容之前打开 |

优先级越低 = 序列中越靠前。这确保了：
- 定义最先
- Load 在计算之前
- Store 最后
- Range 在其内容之前打开，之后关闭

**run_count 排序**：操作主要按执行频率（run_count）排序，然后按优先级。执行频率较低的操作（外层循环之外）先调度，而内层循环中的操作（run_count 更高）后调度。例如：执行 100 次的 CONST 出现在执行 100 万次的 CONST 之前。

**run_count 计算**：
```text
run_count = prod(int(r.vmax) + 1 for r in u.ranges)
```
这根据包围的 range 计算操作执行多少次。

**Morok**：`schedule/src/linearize/mod.rs`

---

## Stage 22：清理 IF/ENDIF

> **阶段速览**
>
> **目标**：线性指令列表的最终清理
> **关键变换**：门控 INDEX → IF/STORE/ENDIF
> **影响**：处理不支持谓词写入的硬件

**做了什么**：线性指令列表的最终清理。

**为什么重要**：某些硬件（现代 GPU）支持"谓词写入"——仅在条件为真时写入内存。较老的硬件不支持。对于那些硬件，我们将 store 包装在 IF 语句中。此阶段**仅**在硬件不支持谓词写入时运行。

**模式**：`pm_linearize_cleanups`（通过 `line_rewrite`，而非 `graph_rewrite`）

```text
// Gated INDEX in STORE becomes conditional store
STORE(INDEX(ptr, idx, valid=cond), value)
→ IF(cond) { STORE(INDEX(ptr, idx), value) } ENDIF
```

**注意**：此阶段使用 `line_rewrite` 而非 `graph_rewrite`，因为它操作的是已线性化的指令列表而非 DAG。

到此为止，指令列表已准备好进行代码生成。

**Morok**：`schedule/src/linearize/mod.rs`（谓词写入路径）
