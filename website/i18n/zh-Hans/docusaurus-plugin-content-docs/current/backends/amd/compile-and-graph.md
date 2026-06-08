---
sidebar_label: 编译与图
---

# 编译与图

本页跟随一个内核从已渲染的 LLVM IR 走到一次运行中的调度，然后
介绍如何把整条内核链捕获成单一可重放的 PM4
图。它所依托的调度机制——环、connector、timeline——
在 [队列与调度](./queues-and-dispatch.md) 中描述。

---

## 从 IR 到一个已加载的程序

编译路径是 **AMD LLVM IR 文本 → `clang` → ELF code object → VRAM 内
加载**。三个 crate 协作，在
`runtime/src/devices/amd.rs` 中接线在一起：

```text
  UOp IR
    │  LlvmTextRenderer::amd(arch)         (svod-codegen)
    ▼
  AMD LLVM IR (text)
    │  compile_ir_to_amd_object            (svod-runtime)
    ▼
  AMDGPU ELF code object
    │  AmdProgram::load                    (svod-device)
    ▼
  resident in VRAM, kernel descriptor decoded
```

### 渲染

`AmdRendererWrapper::render` 使用 `LlvmTextRenderer::amd(arch)` 发出 AMD LLVM
IR。它还安装了一个 AMD 特定的分解 pass
（`amd_decomposition_patterns`），将 `exp`/`log`/三角函数经由 SLEEF
多项式路由，因为硬件的 `exp2`/`log2` 比 CPU
libm 精度更低（`sqrt` 保持原生）。

### 编译

`compile_ir_to_amd_object`（`runtime/src/amd/compile.rs`）外部调用 `clang`，
在 stdin 上灌入 IR，在 stdout 上读回 ELF——没有临时文件，
与 [CPU JIT 加载器](../jit-loader.md) 相同的内存内风格：

```text
clang -x ir -c -O3 --target=amdgcn-amd-amdhsa -mcpu=<arch> \
      -mcumode -nogpulib -nogpuinc -Wno-override-module -fno-math-errno - -o -
```

`clang` 在内部为单个翻译单元调用 `lld`，因此输出是
一个可直接加载的 AMDGPU ELF——没有独立的链接步骤。一个被缓存的
`has_amdgpu_target()` 探测（针对 `amdgcn` 的 `clang --print-targets`）会把一个
缺少 AMDGPU target 的 clang 变成一个干净的 `JitCompilation` 错误，而非
崩溃。设置 `SVOD_DUMP_AMD_IR=<dir>` 会转储每个内核的 `.ll` 供
检视。

### 加载与描述符解析

`AmdProgram::load`（`device/src/amd/program.rs`）用 `object` crate 解析 ELF，
并按 tinygrad 的 `elf_loader` 那样布置镜像：
带有非零地址的 `SHF_ALLOC` section 放在其地址处；地址为 0 的
section 对齐追加。它校验 ELF64-LE + `EM_AMDGPU`，应用 clang 发出的
`R_AMDGPU_ABS64` / `R_AMDGPU_REL64` / `R_AMDGPU_REL32` 重定位
（其他任何东西都是干净的错误，绝不会静默地写零），并解析
内核描述符符号 **`<name>.kd`**。

从 64 字节的 `AmdHsaKernelDescriptor` 中，它推导出调度所需的一切：

| 推导出的 | 来自 |
|---|---|
| `aql_prog_addr` | `code_gpu + kd_offset`（即 AQL 的 `kernel_object`） |
| `pm4_prog_addr` | `aql_prog_addr + kernel_code_entry_byte_offset`（着色器入口；LO/HI 寄存器携带 `>> 8`） |
| `rsrc1 / rsrc2 / rsrc3` | `compute_pgm_rsrc{1,2,3}`，已打上 gfx11 cwsr-priv 位与 LDS-size 字段的补丁 |
| `wave32` | `kernel_code_properties & 0x400`（RDNA3/4 默认） |
| `target_major` | 9 / 11 / 12，来自设备 arch |
| kernarg / scratch / group 尺寸 | `kernarg_size`、`private_segment_fixed_size`、`group_segment_fixed_size` |

加载时会发生两项安全检查：一个过大的 group（LDS）段会以
`GroupSegmentTooLarge` 快速失败，而一个设置了 `ENABLE_SGPR_DISPATCH_PTR`
（它会需要在 kernargs 旁边再带一个 HSA 调度数据包——尚未接线）的
内核会被拒绝。code object 被复制进一个宿主可见的 `nolru` VRAM 缓冲区，
在程序的整个生命周期中持有。

---

## 调度一个内核

`AmdProgram::execute_on(conn, buffers, vals, global, local, wait)` 是 plan 与图
使用的 connector 范围的调度路径（`Program::execute` trait
方法租用一个 connector 并委托到这里）。它会：

1. **校验**针对内核的缓冲区与标量计数，并检查 kernarg
   布局是否容得下：`buf_count*8 + var_count*4 ≤ kernarg_size`。
2. 通过 bump connector 的 arena **填充一个 kernarg 槽**，将每个
   缓冲区 VA 写为 8 字节，将每个标量写为 4 字节的 `i32`。这种 `i32` 打包
   是刻意的——渲染器将 `Index → i32` 降低，因此描述符的
   `kernarg_size` 反映 4 字节的 var；打包 8 字节会溢出进
   下一个槽。
3. 用 kernarg 指针**构建 `USER_DATA`**。可选的 4-dword scratch
   描述符在 `dispatch_pm4` *内部*被前置，与 `COMPUTE_DISPATCH_SCRATCH_BASE`
   寄存器在同一时刻从实时的
   `scratch_gpu_va()` 读取——这样一次并发的 scratch 重分配就不会让描述符与
   寄存器不一致。
4. **调度**——`queue.dispatch_pm4(...)`（PM4 路径）或
   `queue.dispatch_aql(...)` 配一个 `build_dispatch_packet`（AQL 路径）。
5. 若 `wait`，则调用 `conn.synchronize()`。

---

## 图捕获与重放：`AmdGraph`

当同一条内核链反复运行时（流式推理），把
每内核的 `wait → barrier → exec → signal → doorbell` 往返付出 N 次是
浪费。`AmdGraph`（`device/src/amd/graph.rs`）——tinygrad 的
`HCQGraph` 的 1:1 移植——把整条链捕获进**一个 PM4 命令流**，
将其绑定进一个宿主可见的页，并用**一个 doorbell** 重放它。

### 结构

图是一个设备 timeline 步：

```text
preamble:  memory_barrier
           wait(virt_timeline, timeline-1)
           wait(kick, kickoff)
           signal(self, kickoff)
per kernel: exec()            ← no inter-kernel signal/wait; same-queue ordering
                                 is the acquire_mem + CS_PARTIAL_FLUSH in exec
final:     signal(virt_timeline, timeline)   ← advances the real timeline by +1
```

`virt_timeline` 的地址与值是**符号**（`Sym::VirtTimelineSigAddr`、
`Sym::VirtTimelineVal`、`Sym::Kickoff`），在重放时解析为 connector 的
真实信号地址和 `timeline_value() - 1`，因此图能与普通的每调用
调度和 `synchronize` 组合。捕获在一个专用页中为每个内核布置一个固定的
kernarg 槽——拥有那个页（而不是共享滚动的 kernarg arena，并发的
每调用调度可能套圈进入陈旧的 VA）正是让重放安全的东西。

重放（`Graph::replay`）递增 kickoff 计数器，等待上一次重放的
timeline target，预留本步的值，解析符号，并用单个
`submit_dwords` doorbell 提交已绑定的 IB——然后通过设置 kick 信号释放
已暂存的 IB。它异步返回；反压是*下一次*重放的 wait。

### 捕获何时发生

捕获以若干方式设门，若有任何一项失败则回退到每调用调度
（`Ok(None)`）：

- **必须设置 `SVOD_JIT_GRAPH`。** `ExecutionPlan::build_graph`
  （`runtime/src/execution_plan.rs`）否则返回 `None`——每调用调度
  是安全的默认；图路径是为 benchmark 而选择启用的。
- 该链必须是**全部已编译的内核且没有运行时 var**——复制、
  view 和动态 launch 维度会让宿主留在回路中。
- 设备必须处于**多队列模式**。在默认的单队列模式下，
  `AmdGraph::capture` 返回 `Ok(None)`，因为图保有自己的
  connector 与环（用一个 doorbell 重放），而单队列调度
  锁并不覆盖它们。
- 该链必须是**单设备、单 XCC 的 PM4**——AQL（多 XCC）和
  跨设备的链不在范围之内。

:::caution 图捕获是双重设门的
要得到一个真正的 `AmdGraph`，你需要**同时**设置 `SVOD_JIT_GRAPH`（设成任何值即可）
**以及** `SVOD_AMD_SINGLE_QUEUE=0`。在默认的单队列模式下，捕获始终
返回 `None` 且调度保持每调用——这是正确而安全的，只是没有被
图加速而已。
:::

---

## 为什么这很重要

编译就是一个 `clang` 子进程加一次进程内 ELF 加载——没有 ROCm，没有
临时文件，与 CPU 路径相同的极简主义。调度复用了来自
[队列与调度](./queues-and-dispatch.md) 的整套 connector/timeline 机制，
因此 [JIT 图](../../architecture/jit-graphs.md) 层的"编译一次 / 重放多次"承诺
在 AMD 上每次重放只用一个 doorbell 即可落地——一旦图路径被启用。
