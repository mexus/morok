---
sidebar_label: 队列与调度
---

# 队列与调度

调度一个内核意味着将命令数据包写入一个环形缓冲区，并
敲响 doorbell。本页介绍环机制（`AmdComputeQueue`）、包装它的
每所有者捆绑（`AmdConnector`）、两种调度策略（单队列与多队列）、
完成原语（`Timeline`），以及配置后端的每一个
环境变量。

这个设计的形态来自一个事实：**tinygrad 是 GIL 串行化的**——
每个设备一个 compute 队列，由 Python 的 GIL 让每次调度成为原子的。
Svod 移除 GIL 以获得真正的并发，因此 GIL 所提供的不变量
必须显式地重建。其结果是一条可以做到无锁的调度路径。

---

## 命令环：`AmdComputeQueue`

`device/src/amd/queue.rs` 定义了 `AmdComputeQueue`，它拥有：

- 一个 **16 MiB 宿主可见环**（`COMPUTE_RING_BYTES`）——命令数据包直接由 CPU
  写入其中；
- 一个 **doorbell**（`*mut u64` MMIO）——通过在此写入新的 write-index，
  告知 GPU 的命令处理器"有新工作"；
- GART 常驻的 **write/read dispatch-id** 槽——KFD 除了 doorbell 之外还会读取
  write 指针，所以它会被先发布。

### PM4 与 AQL

存在两种线上数据包格式，在队列创建时根据
设备的 XCC 计数一次性选定：

```text
will_use_pm4(core) = !SVOD_AMD_AQL && num_xcc == 1
```

- **PM4**（单 XCC：gfx11/12 的默认）——原始 PM4 dword 直接写入
  环中（`KFD_IOC_QUEUE_TYPE_COMPUTE`）。doorbell 用
  下一个 dword 索引敲响。
- **AQL**（多 XCC 的 CDNA）——64 字节 AQL 数据包
  （`KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`），其中 PM4 辅助包被包裹在 AQL
  vendor-IB 数据包内部。doorbell 用最后完成的槽
  （`write_idx - 1`）敲响。

一次 PM4 调度是一个固定序列，对应 tinygrad 的
`hcq.py:371-378`：

```text
wait(timeline, prev)  →  hdp_flush  →  acquire_mem  →  exec  →  release_mem(timeline, next)
```

`exec` 是加载着色器地址、`RSRC1/2/3`
寄存器、scratch 描述符和 `TMPRING_SIZE`、
`USER_DATA` SGPR、launch 维度的 `SET_SH_REG` 流，随后是
`DISPATCH_DIRECT`，再接一个 `CS_PARTIAL_FLUSH`。末尾的 `release_mem` 会在
GPU 完成时将该调度的 timeline 值写入 connector 的信号槽。

### 无锁内部可变性

`AmdComputeQueue.inner` 是一个 `UnsafeCell<QueueInner>`，而非 `Mutex`——调度
通过 `&self` 无锁地修改它。这之所以可靠，是因为一个
**单所有者不变量**：在一个 `ConnectorLease` 的生命周期内，恰好有一个
线程对该队列发出顺序的、非重入的调度（与 `RawBuffer` 在
`device/src/allocator.rs` 中使用的模式相同）。共享的排空线程从不
触碰队列——它只读取 timeline（见下文）。不同 connector 的
队列由 GPU 的硬件调度器（MES——微引擎调度器）交错，而非 CPU 锁。

### 环反压

一个跑 `wait=false` 比 GPU 排空更快的宿主会套圈 16 MiB 环
并覆盖尚未消费的数据包。`wait_dispatch_headroom` 通过
将未退役调度的数量限制到 `RING_MAX_INFLIGHT`（环的一半）来防止这一点，
在达到该上限时在 **timeline 信号**上阻塞：

```rust
let last_reserved = conn.timeline_value().saturating_sub(1);
if last_reserved > RING_MAX_INFLIGHT {
    let target = last_reserved - RING_MAX_INFLIGHT;
    conn.timeline_signal().wait_signal_value(target, 30_000)?;
}
```

它在 timeline 信号——这个经过验证的完成原语——上设门，而不是在
PM4 read 指针上，后者的 COMPUTE 队列语义不可靠，会让一次自旋
死锁。

---

## 每所有者捆绑：`AmdConnector`

单有一个队列还不足以调度。`AmdConnector`
（`device/src/amd/connector.rs`）把一个独立调用方所需的一切捆在一起：

| 字段 | 它是什么 |
|---|---|
| `queue: Box<AmdComputeQueue>` | 环 + doorbell + GART（唯一所有者 → 无锁） |
| `arena: Box<KernargArena>` | 一个 16 MiB GTT kernarg bump arena |
| `scratch_state: Mutex<ScratchState>` | 寄存器溢出的 scratch 后备，按需增长 |
| `timeline: Arc<Timeline>` | 单调计数器 + 完成信号 |

每个 `ExecutionPlan` 和每个 `AmdGraph` 都拥有自己的 connector。队列与 arena 上的
`Box`（而非 `Arc`）是承重的：它证明没有第二个
句柄别名了那个 `UnsafeCell`，而这正是让无锁调度可靠的东西。
arena 是每 connector 的，因此它的 bump 游标与该 connector 的 timeline 共享
同一套排序——一个被绕回复用的 kernarg 槽在那个 timeline
排空后就可证明是空闲的。

`ensure_has_local_memory(private_segment_size)` 会在一个
刚加载的内核需要比当前已分配更多的每线程字节数时增长 scratch 缓冲区
（分配新的 → 交换 → 排空 → 释放旧的）。scratch 是仅 GPU 的 VRAM，会被动态
重新分配，也是 `NotPresent` 故障在历史上的来源——见
[调试](./debugging.md)。

---

## 两种调度策略

每设备的 `Dispatcher` 枚举（`device/src/amd/device.rs`）选择所有者如何
获得 connector，以及调度是否被串行化。它在
设备打开时根据 `SVOD_AMD_SINGLE_QUEUE` 一次性构建：

### 单队列（默认）

```text
SVOD_AMD_SINGLE_QUEUE unset or ≠ 0
```

每个物理设备上的每个所有者共享**一个** connector，且调度 +
scratch 重分配通过 `exec_guard()` 取得的 `Mutex<()>` 串行化。
于是内核对每个 GPU 只看到一个 compute 队列——tinygrad 的模型。

这是 **KFD 安全**模式，并且它之所以是默认有一个具体原因：
繁重的并发多队列调度会**令内核的 MES/runlist
调度器过载，并可能让内核崩溃**。一个 GPU 只有一个命令处理器，反正也是
顺序运行调度；多队列只是重叠了 CPU 侧的数据包
组装，而那正是把 KFD 推入坏路径的东西。单队列消除了那次
崩溃。

### 多队列（选择启用）

```text
SVOD_AMD_SINGLE_QUEUE=0
```

每个所有者从一个空闲池中租用一个**独占拥有**的 connector（上限为
`CONNECTOR_POOL_CAP = 4`）；MES 交错这 N 个队列，因此调度
不需要 CPU 锁，且 `exec_guard()` 返回 `None`。租约的独占且
不可别名正是阻止两个调度者共享同一个 KFD 队列的东西。

:::caution 内核过载的告诫
多队列是无锁、最大并发的路径，但它也正是那条在
负载下令 KFD 过载的路径。正因如此它需要选择启用。真正的修复——拥有
GPU 使得内核永不处于调度路径中——是
[用户态 AM 驱动](./am-driver.md)。
:::

### `ConnectorLease`

`lease_connector` 返回一个 `ConnectorLease`——一个非 `Clone` 的句柄，它
`Deref` 为 `&AmdConnector`，因此调用方与模式无关。在 drop 时，
`return_connector` 会做与模式相应的事情：重新入池（多队列，直到
上限）或什么都不做（单队列——共享 connector 继续存活于 core 上）。
它在 drop 时**不**同步；connector 的 `Timeline` 保持注册
状态，因此设备级排空仍会覆盖它。

---

## 完成原语：`Timeline`

`Timeline`（`device/src/amd/signal.rs`）是一个单调的 `AtomicU64` 计数器外加
GPU 在调度完成时写入的 GTT 一致的 `AmdSignal` 槽。它是
**唯一一个跨所有者的原语**：

- 一个 connector 对它*调度*——`next()` 做 `fetch_add(1)` 以预留
  其 `release_mem` 数据包将要写入的值；
- 任何线程都能对它*排空*——`drain()` 读取该原子并轮询信号
  槽，**从不触碰队列**。

那种解耦正是让调度保持无锁的东西。设备 core
（`AmdDeviceCore`）为每个 connector 持有 `Weak<Timeline>`——而非
`Weak<AmdConnector>`——因此 `synchronize_all`（任何宿主读取或
缓冲区释放之前的栅栏）纯粹通过这些原子来排空所有在途工作：

```text
AmdDeviceCore.synchronize_all():
   for each live Timeline:  timeline.drain(30s)   // atomics + signal slot only
```

`AmdSignal::wait_signal_value` 分层轮询——紧凑自旋 → `yield_now` → 200 ms 之后
KFD `WAIT_EVENTS` 睡眠——这样一次漫长或停滞的等待不会烧掉一个 CPU，
而等待期间的 GPU 故障会立即浮现，而不是阻塞
整整 30 秒的超时。

:::note 2³² 回绕
PM4 `WAIT_REG_MEM`/`RELEASE_MEM` 比较信号槽的低 32 位，因此
计数器必须保持在 2³² 以下。`ensure_timeline_headroom` 在预留每个值之前，
在一个 2³¹ 水位线（`TIMELINE_WRAP_WATERMARK`）处排空并重置，因此一个
漫长的 `wait=false` 循环无法爬过 2³² 而产生一次假超时。
:::

---

## 接缝

队列层所需的全部四种内核操作都路由经过设备 core 上的
[`AmdIface`](./overview.md) trait：

```rust
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    fn alloc_raw(&self, size, kind, tag, cpu_access, zero) -> Result<AllocResult>;
    fn free_raw(&self, gpu_va, size, handle);
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    fn teardown_ring(&self, queue_id: u32);
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;
}
```

注意 trait 中*没有*什么：环、GART、EOP 和 ctx-save 缓冲区
全都在接缝之上（经由 `alloc_raw`）在 `create_queue` 内部分配。trait
只**激活**队列——`setup_ring` 发出 `CREATE_QUEUE`，并在一个上半部已经拥有的
环之上 `mmap` doorbell。`KfdIface` 是当今唯一的
实现者。

---

## 配置参考

每一个影响 AMD 后端的环境变量：

| 变量 | 默认 | 效果 |
|---|---|---|
| `SVOD_DEVICE` | `CPU` | 为张量选择默认设备。设 `SVOD_DEVICE=AMD:0` 以在第一块 AMD GPU 上运行 |
| `SVOD_AMD_BACKEND` | `kfd` | 后端选择。如今只接受 `kfd`；`am` 是未来的接缝（若设置则报错） |
| `SVOD_AMD_SINGLE_QUEUE` | `1`（开） | `=0` 选择启用无锁多队列调度；否则为 KFD 安全的单队列模式 |
| `SVOD_AMD_AQL` | `0`（关） | `=1` 即便在单 XCC 硬件上也强制 AQL 数据包——用于二分排查 PM4 与 AQL 问题 |
| `SVOD_JIT_GRAPH` | 未设 | 启用 PM4 图捕获/重放（还要求多队列模式）。见 [编译与图](./compile-and-graph.md) |
| `SVOD_KFD_TOPOLOGY` | sysfs 路径 | 覆盖拓扑根，用于无硬件测试 |
| `SVOD_DEBUG_DISPATCH` | 未设 | 转储每次调度的内核 / grid / kernarg / scratch / 缓冲区 VA。见 [调试](./debugging.md) |
| `SVOD_DUMP_AMD_IR` | 未设 | 若设为一个目录，则将每个内核的 AMD LLVM IR 转储到那里 |

:::caution 不存在 `SVOD_AMD_MAX_QUEUES`
多队列空闲池大小是 `device.rs` 中的编译期常量
`CONNECTOR_POOL_CAP = 4`，而非一个环境变量。
:::

---

## 为什么这很重要

GIL 免费给了 tinygrad 一个原子的调度临界区。Svod 用三种方式
重建那个保证：环的**单所有者归属**（无调度
锁）、用于排空的**共享 timeline 信号**（原子，从不动队列），以及
**显式的环反压**。单队列默认在今天保证内核
安全；无锁的多队列路径已经就绪，待
[AM 驱动](./am-driver.md) 把内核彻底移出回路时即可使用。
