---
sidebar_label: AM 驱动
---

# AM 驱动（用户态）

**AM** 驱动是第二个 [`AmdIface`](./overview.md) 后端，它直接驱动
GPU 的 PCI BAR，彻底绕过内核的 `amdgpu`/KFD 驱动。它
是 tinygrad 用户态 AM 驱动的移植。其动机很具体：
无锁的 [多队列调度](./queues-and-dispatch.md) 路径会在繁重的并发负载下令
内核的 MES/runlist 调度器过载，并可能让内核崩溃。如果我们拥有
GPU——页表、固件、调度——内核就永不处于调度路径中，
也就无法被过载。

:::caution 开发中——尚不可选
本页同时记录当下存在什么，以及其余部分的路线图。
**`SVOD_AMD_BACKEND=am` 目前会返回错误**（`device.rs` 只接受
`kfd`）。已实现的是不需特权、不需 GPU 的*逻辑*；
特权硬件启动被推迟。下面各节明确标注每个部件的
状态。
:::

代码位于 `device/src/amd/am/` 之下。它在 Linux 上无条件编译
（纯逻辑，无额外依赖），因此始终被类型检查、lint
和单元测试——后端在*运行时*选择，绝不藏在一个
可能腐烂的 cargo feature 之后。

---

## 内核曾为我们做的事

KFD 给了后端三样东西。AM 必须各自自己提供：

| KFD 提供 | AM 必须做 |
|---|---|
| VRAM 分配 + GPU 页表映射 | 一个 GMMU：VA 分配器 + 4 级页表 walker + PTE 编码 |
| 队列创建（MES/HQD 设置） | 写 MQD、使能 MEC、映射 doorbell |
| 内存 + 固件启动 | PCI BAR 映射、IP discovery、PSP 固件加载 |

关键在于，接缝*之上*的一切——PM4/AQL 数据包构建器、
环、信号、kernarg arena、timeline 和反压——都**原封不动**
沿用过来。AM 只替换那五个 `AmdIface` 方法。

---

## 当下存在什么（已构建并测试）

纯逻辑那一半已实现并在无 GPU 的情况下单元测试，页表由一个
可注入的 `PhysMem` trait 后备（测试中是一个普通缓冲区，
真实驱动中是 BAR 映射的 VRAM）。

| 模块 | 它实现什么 | 状态 |
|---|---|---|
| `am/mm/tlsf.rs` | TLSF（Two-Level Segregated Fit）分配器——tinygrad 的 `TLSFAllocator` 的移植 | **完成** + 单元测试 + 一个 proptest |
| `am/mm/pagetable.rs` | GMMU 几何 + PTE/PDE 位编码 | **gfx11 完成** + 测试 |
| `am/mm/manager.rs` | `MemoryManager`：VA 分配、4 级页表遍历、大页选择、表回收、`valloc`/`vfree` | **完成** + 针对伪造 VRAM 的测试 |
| `am/regs.rs` | `RegDef`/`RegField` 类型 + `select(prefix, ip_ver)` 解析器 | **完成** + 测试 |
| `am/regs_gen.rs` | Vendored 寄存器表（`GC_11_5_0`、`MMHUB_3_3_0`、`MP_14_0_2`、…） | **已生成并提交** |

### GMMU

页表几何是 **4 级 / 48 位**（`va_shifts = [12, 21, 30, 39]`），
一种**跨 gfx9/11/12 共享**的形状——因此几何本身不针对 arch 分支。
只有叶 PTE 编码（尤其是 MTYPE 内存类型字段）才是
arch 特定的。`MemoryManager` 运行三个 TLSF 子分配器（VA 空间、
物理 VRAM、页表池），并以 `Inspect` / `Create` /
`Free` 模式遍历表，在对齐允许处选择大页，并在 unmap 时回收空
表。

### 寄存器表是生成一次，然后 vendored

tinygrad 是一个有时缺席的子模块，因此构建绝不能依赖它。
取而代之，`device/tools/gen_am_regs.py` 在添加或更新一个 arch 时
被**手动**运行：它解析 tinygrad 的 `autogen/am/regs.py` 并发出已提交的
`am/regs_gen.rs`。`regs.rs` 只是 `include!` 它。在启动时，正确的表由
发现到的 `ip_ver` 选定（`select` 挑选共享同一 major 的最大版本 `≤ ip_ver`
——tinygrad 的 `import_module` 规则）。添加一个 arch 就是
拓宽生成器的模块列表并重新运行它——没有构建或运行时
逻辑改动。

---

## 推迟了什么（尚不在代码树中）

特权启动需要 root/caps（unbind `amdgpu`、`mmap` PCI BAR、
mode-1 reset），尚未出现在源码中：

- **AMDev 编排器**（BAR 映射、boot 序列）；
- **PCI/BAR** 访问与 **IP discovery** 解析；
- **PSP 固件加载**（风险最高的子系统——一次版本特定的
  握手）；
- **IP-block** 模块（SOC / GMC / IH / PSP / SMU / GFX / SDMA）；
- 把这一切系到接缝上的 **`AmIface`** 实现者。

在*已实现*的页表模块内，只有 **gfx11/RDNA3** 是活跃的：
gfx9（VG10）和 gfx12 的 PTE 编码路径是刻意的 `unimplemented!` panic，
各由一个断言其 panic 的测试守护——常量已捕获但
尚未经过硬件验证。

---

## 目标硬件与 arch 参数化

寄存器与页表的目标是 **gfx1151——「Strix Halo」APU**（它
报告 GC 11.5.1 → `gc_11_5_0` 表）。该驱动按 tinygrad 的方式参数化：
通过 **启动时从 IP discovery 读取的 `ip_ver` 元组**，而非一个
手工维护的 arch 枚举。arch 差异本应是共享模块内部小而内联的
`if ip_ver >= (X, Y, Z)` 分支，加上版本键控的
寄存器表——这样 gfx12 大致就成了一次数据添加，而 gfx9 虽已
纳入考虑但被推迟。

:::note 为什么在这台机器上推迟启动
实际硬件是一块 Strix Halo APU，它同时也是**主显示 GPU**。
AM 必须 unbind `amdgpu` 并取得独占所有权，这会让显示熄灭；
而 tinygrad 的 AM 白名单的是独立 RDNA3/4 的设备 ID，不是这块 APU。所以在
这台机器上没有可对照验证的可工作 AM 参照物。特权
启动（下面的各阶段）被推迟到一块外置/独立 GPU 上，在那里 tinygrad
AM 是一个经过验证的参考。与此同时，[单队列 KFD 模式](./queues-and-dispatch.md)
已经修复了催生 AM 的那次内核崩溃，因此在此期间没有什么被阻塞。
:::

---

## 路线图

每个阶段都可独立测试，在同一张卡上用 tinygrad AM 作为
每阶段的参照物：

| 阶段 | 里程碑 |
|---|---|
| **A** | PCI + discovery（只读）：unbind `amdgpu`、映射 BAR、解析 IP discovery、把每个值与 tinygrad 比对 |
| **B** | regs + GMC 页表：`valloc` + 映射一个缓冲区、读回 PTE、让数据经由 BAR 往返 |
| **C** | PSP 固件加载（风险关卡）：sOS bootloader 握手、TMR、每 IP 的固件加载——与一份 tinygrad 抄本逐 dword 比对 |
| **D** | GFX MEC + `setup_ring`：写 v11 compute MQD、使能 MEC、映射 doorbell（`CP_HQD_ACTIVE == 1`） |
| **E** | 调度一个内核：在 AM 后备的 core 上复用整个现有的上半部 |
| **F** | 并发 + 去桩：真实中断处理器、最高时钟、跑那个让 KFD 崩溃的工作负载——它现在不会崩溃，因为内核被绕过了 |

一旦阶段 F 能干净地跑那个诱发崩溃的并发，AM 就成为
多队列/流式工作负载的推荐模式，而 KFD（单队列）
仍作为可移植的回退。gfx12/RDNA4 随后就是一次廉价的后续工作（拓宽
寄存器表 + 添加 `gc >= (12,0,0)` 分支）；gfx9/CDNA 则是一项更大、
更晚的工程。

---

## 为什么这很重要

AM 驱动是对那个内核过载问题的真正答案，而
[单队列模式](./queues-and-dispatch.md) 只是绕开了它。那些昂贵、
不需 GPU 的部分——GMMU 和寄存器表——已经构建并测试，
因此剩下的工作就是特权启动，它被卡在硬件上而非
卡在设计上。而且因为它接在同样的五方法
[接缝](./overview.md) 之后，当它落地时，调度、编译或图机制
没有一样需要改动。
