---
sidebar_label: KFD 绑定
---

# KFD 绑定

后端通过对 `/dev/kfd` 的一小组固定 `ioctl` 调用与内核对话。
本页介绍这些调用如何绑定到 Rust、后端实际使用其中的哪些、GPU 节点如何被
发现，以及将一个 `ioctl` 变成已映射 GPU 缓冲区的分配流程。关于后端为什么是
KFD 直连而非基于 HIP 的*原因*，见 [概览](./overview.md)。

---

## 绑定是如何生成的

KFD 的 ABI 是一个 C 头文件 `kfd_ioctl.h`，从内核原样 vendored 进
`device/include/kfd_ioctl.h`（即上游 AMD 文件，连同其完整的 ABI
版本历史）。Rust 绑定由 `bindgen` 在构建时从它生成：

- `device/build.rs` **仅在 Linux 上**运行 `bindgen`，精确地 allow-list
  后端所需的 KFD 类型与常量：

  ```text
  allowlist_type:  kfd_ioctl_.*_args, kfd_event_data,
                   kfd_hsa_memory_exception_data, kfd_hsa_hw_exception_data,
                   kfd_memory_exception_failure, __u\d+, __s\d+, …
  allowlist_var:   KFD_IOC_.*, AMDKFD_IOC_.*, KFD_MAX_QUEUE_PERCENTAGE, …
  ```

  并带有 `.derive_default(true).layout_tests(false).generate_comments(false)`。
  输出被写入 `$OUT_DIR/kfd_sys.rs`。

- 在**非 Linux** 宿主上，`build.rs` 改为写入一个空桩，使该模块
  始终能编译（届时 AMD 路径在运行时返回 `Err(NoAmdGpu)`）。

- `device/src/amd/sys/kfd.rs` 是一行 `include!` 生成文件的代码。

:::note 为什么手写 ioctl 宏
`bindgen` 发出参数*结构体*但不发出 `_IOWR` ioctl 号宏。
那些宏在 `device/src/amd/sys/ioctl.rs` 中使用
`nix::ioctl_readwrite!` 手工声明，类型码为 `KFD_IOCTL_BASE = b'K'`。即便头文件写的是
`_IOR`/`_IOW`，每个 ioctl 也都声明为 `readwrite`——KFD
把参数结构体当作输入/输出，内核两个方向都容忍。
:::

---

## 后端使用的 ioctl

这些 `(group, opcode, args)` 三元组直接来自 `kfd_ioctl.h`。下面是
带有真实调用点的那些：

| 包装器 | Op | 用于 |
|---|---|---|
| `kfd_get_version` | `0x01` | 读取 KFD ABI 版本（控制 `RUNTIME_ENABLE`） |
| `kfd_create_queue` | `0x02` | `setup_ring` — 创建一个 compute/SDMA 队列 |
| `kfd_destroy_queue` | `0x03` | `teardown_ring` |
| `kfd_create_event` | `0x08` | 队列信号、内存故障与 hw-exception 事件；绑定事件页 |
| `kfd_wait_events` | `0x0C` | `wait_events` — 在完成/故障事件上阻塞 |
| `kfd_acquire_vm` | `0x15` | 将 DRM render fd 注册为本进程对该 GPU 的 VM |
| `kfd_alloc_memory_of_gpu` | `0x16` | `alloc_raw` — 分配 VRAM/GTT |
| `kfd_free_memory_of_gpu` | `0x17` | `free_raw` |
| `kfd_map_memory_to_gpu` | `0x18` | 将一个分配绑定进 GPU 页表 |
| `kfd_unmap_memory_from_gpu` | `0x19` | `free_raw` |
| `kfd_runtime_enable` | `0x25` | 启用运行时（仅 KFD ABI ≥ 1.14） |

还有少数几个（`set_memory_policy`、`get_clock_counters`、
`get_process_apertures`、`update_queue`、`destroy_event`、`set_event`、
`reset_event`）为完整性而声明，但目前未被调用。

### 设备启动序列

`KfdIface::open`（`device/src/amd/iface.rs`）按顺序发出这些调用，
对应 tinygrad 的 `ops_amd.py`：

```text
open /dev/kfd  (process-shared, one fd)
open /dev/dri/renderD<minor>  (per node — the DRM render fd)
   │
   ├─ GET_VERSION            → capture ABI version
   ├─ ACQUIRE_VM(drm_fd)     → register this fd as the process VM for the GPU
   ├─ RUNTIME_ENABLE         → only if ABI ≥ 1.14
   ├─ (event page: alloc + bind once per process, map per device)
   └─ CREATE_EVENT × 3       → queue-signal, memory-fault, hw-exception
```

DRM render fd 很有意思：这里**没有任何 DRM ioctl**。`drm_fd` 仅以
两种方式使用——*按编号*传入 `ACQUIRE_VM`，以及作为宿主可见映射的
`mmap` fd。相比之下，doorbell 则是从 KFD fd `mmap` 出来的。

---

## 拓扑：找到 GPU

GPU 节点是从 sysfs 枚举的，而不是通过 ioctl。
`device/src/amd/topology.rs` 读取
`/sys/devices/virtual/kfd/kfd/topology/nodes/<N>/properties`——每行一个
`key value` 对——并返回一个 `Vec<AmdNode>`，跳过 CPU 节点
（`gpu_id == 0`）。它从不 panic：没有 `/dev/kfd` 的宿主会产生一个空
向量，设备工厂将其转化为干净的 `Err(NoAmdGpu)`。

每个 `AmdNode` 携带后端其余部分所需的字段：
`gpu_id`、`drm_render_minor`、`gfx_target_version`（如 `110000` → gfx1100）、
`simd_count`、`simd_per_cu`、`max_waves_per_simd`、`num_xcc`、`lds_size_in_kb`、
`max_slots_scratch_cu` 等等——这些用于 scratch 尺寸计算以及 PM4 与
AQL 的抉择。

:::tip 无硬件测试
sysfs 根目录可用 **`SVOD_KFD_TOPOLOGY`** 覆盖，因此解析器可针对一个
没有 GPU 的伪造 nodes 目录进行单元测试。
:::

---

## 分配流程

每个缓冲区都遵循同样的四步路径，在
`KfdIface::alloc_raw` 中实现一次：

```text
1. reserve_va(size)                     mmap(PROT_NONE, …) — reserve host VA
2. ALLOC_MEMORY_OF_GPU(va, size, flags) → returns handle + mmap_offset
3. if host-visible:                     mmap(va, …, MAP_FIXED, drm_fd, offset)
4. MAP_MEMORY_TO_GPU(handle)            bind into the GPU page table
```

宿主 VA 先用一个匿名的 `PROT_NONE` 映射预留，使得第 3 步中宿主可见的
`mmap` 能恰好落在那个地址（`MAP_FIXED`）。
释放则反向进行：`UNMAP_MEMORY_FROM_GPU` → `munmap` → `FREE_MEMORY_OF_GPU`。

### 分配种类

`alloc_raw` 接收一个 `AllocKind`，它选定 KFD 标志集——这些标志被组装的
唯一位置：

| `AllocKind` | 标志 | 用于 |
|---|---|---|
| `DeviceVram { executable }` | `VRAM \| WRITABLE \| NO_SUBSTITUTE`（代码额外加 `EXECUTABLE`，宿主可见时额外加 `PUBLIC`） | 张量数据、code object、scratch |
| `UncachedGtt` | `GTT \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE \| PUBLIC \| COHERENT \| UNCACHED` | 命令环、GART 页、信号槽、事件页 |

`UNCACHED | COHERENT` 的这种 GTT 变体很关键：命令环和信号
槽必须在 CPU 与 GPU 之间立即可见，否则宿主会永远自旋
等待一个卡在 GPU L2 中的完成值。KFD 会以 `EINVAL`
拒绝对一个纯 VRAM 环执行 `CREATE_QUEUE`。

### 处处宿主可见

由于没有 SDMA 队列，分配器（`device/src/amd/allocator.rs`）
对每个缓冲区强制 `cpu_access = true`：`has_sdma_queue()` 始终为
`false`，因此 `_alloc` 会把它 OR 进去。于是复制（`_copyin`/`_copyout`/`_transfer`）
就是在一次 `synchronize()` 之后的普通宿主 `memmove`。通用的
`LruAllocator`（`device/src/allocator.rs`）按
`(size, BufferSpec)` 池化已释放的缓冲区；`nolru` spec 对 code object、
scratch 和队列基础设施绕过该池。

:::note 进程共享状态
`/dev/kfd` 每进程只打开一次，并由所有设备共享（事件
通过 id 针对该 fd 寻址）。0x8000 字节的 KFD **事件页**同样
每进程分配并绑定一次；后续设备只是将其 `MAP_MEMORY_TO_GPU`
进它们各自的 `gpu_id`。两者都对应 tinygrad 的每进程模型。
:::

---

## 为什么这很重要

整个面向内核的接口面就是**一个 vendored 头文件、十一个 ioctl，以及一个
sysfs 解析器**。这正是后端能够避开 ROCm
用户态栈的全部原因：内核 ABI 小而稳定，因此直接绑定它比
集成 HIP 要少写代码——而且它让
[后端接缝](./overview.md) 可以自由地用用户态
[AM 驱动](./am-driver.md) 替换掉 KFD，而无需触碰其上的任何东西。
