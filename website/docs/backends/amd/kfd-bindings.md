---
sidebar_label: KFD Bindings
---

# KFD Bindings

The backend speaks to the kernel through a small, fixed set of `ioctl` calls on
`/dev/kfd`. This page covers how those calls are bound to Rust, which ones the
backend actually uses, how GPU nodes are discovered, and the allocation flow
that turns an `ioctl` into a mapped GPU buffer. For *why* the backend is
KFD-direct rather than HIP-based, see the [Overview](./overview.md).

---

## How the bindings are generated

KFD's ABI is a C header, `kfd_ioctl.h`, vendored verbatim from the kernel into
`device/include/kfd_ioctl.h` (the upstream AMD file, complete with its ABI
version history). Rust bindings are generated from it at build time by
`bindgen`:

- `device/build.rs` runs `bindgen` **only on Linux**, allow-listing exactly the
  KFD types and constants the backend needs:

  ```text
  allowlist_type:  kfd_ioctl_.*_args, kfd_event_data,
                   kfd_hsa_memory_exception_data, kfd_hsa_hw_exception_data,
                   kfd_memory_exception_failure, __u\d+, __s\d+, …
  allowlist_var:   KFD_IOC_.*, AMDKFD_IOC_.*, KFD_MAX_QUEUE_PERCENTAGE, …
  ```

  with `.derive_default(true).layout_tests(false).generate_comments(false)`. The
  output is written to `$OUT_DIR/kfd_sys.rs`.

- On **non-Linux** hosts `build.rs` writes an empty stub instead, so the module
  always compiles (the AMD path then returns `Err(NoAmdGpu)` at runtime).

- `device/src/amd/sys/kfd.rs` is a one-liner that `include!`s the generated
  file.

:::note Why hand-written ioctl macros
`bindgen` emits the argument *structs* but not the `_IOWR` ioctl-number macros.
Those are declared by hand in `device/src/amd/sys/ioctl.rs` using
`nix::ioctl_readwrite!`, with the type code `KFD_IOCTL_BASE = b'K'`. Every
ioctl is declared `readwrite` even where the header says `_IOR`/`_IOW` — KFD
treats the argument struct as in/out, and the kernel tolerates both directions.
:::

---

## The ioctls the backend uses

The `(group, opcode, args)` triples come straight from `kfd_ioctl.h`. These are
the ones with live call sites:

| Wrapper | Op | Used for |
|---|---|---|
| `kfd_get_version` | `0x01` | Read the KFD ABI version (gates `RUNTIME_ENABLE`) |
| `kfd_create_queue` | `0x02` | `setup_ring` — create a compute/SDMA queue |
| `kfd_destroy_queue` | `0x03` | `teardown_ring` |
| `kfd_create_event` | `0x08` | The queue-signal, memory-fault, and hw-exception events; binding the event page |
| `kfd_wait_events` | `0x0C` | `wait_events` — block on completion / fault events |
| `kfd_acquire_vm` | `0x15` | Register the DRM render fd as this process's VM for the GPU |
| `kfd_alloc_memory_of_gpu` | `0x16` | `alloc_raw` — allocate VRAM/GTT |
| `kfd_free_memory_of_gpu` | `0x17` | `free_raw` |
| `kfd_map_memory_to_gpu` | `0x18` | Bind an allocation into the GPU page table |
| `kfd_unmap_memory_from_gpu` | `0x19` | `free_raw` |
| `kfd_runtime_enable` | `0x25` | Enable the runtime (KFD ABI ≥ 1.14 only) |

A handful more (`set_memory_policy`, `get_clock_counters`,
`get_process_apertures`, `update_queue`, `destroy_event`, `set_event`,
`reset_event`) are declared for completeness but not currently called.

### Device bring-up sequence

`KfdIface::open` (`device/src/amd/iface.rs`) issues these in order, mirroring
tinygrad's `ops_amd.py`:

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

The DRM render fd is interesting: there are **no DRM ioctls**. The `drm_fd` is
used only two ways — passed *by number* into `ACQUIRE_VM`, and as the `mmap` fd
for host-visible mappings. The doorbell, by contrast, is `mmap`ped from the KFD
fd.

---

## Topology: finding the GPU

GPU nodes are enumerated from sysfs, not via an ioctl.
`device/src/amd/topology.rs` reads
`/sys/devices/virtual/kfd/kfd/topology/nodes/<N>/properties` — one
`key value` pair per line — and returns a `Vec<AmdNode>`, skipping CPU nodes
(`gpu_id == 0`). It never panics: a host with no `/dev/kfd` yields an empty
vector, which the device factory turns into a clean `Err(NoAmdGpu)`.

Each `AmdNode` carries the fields the rest of the backend needs:
`gpu_id`, `drm_render_minor`, `gfx_target_version` (e.g. `110000` → gfx1100),
`simd_count`, `simd_per_cu`, `max_waves_per_simd`, `num_xcc`, `lds_size_in_kb`,
`max_slots_scratch_cu`, and friends — these feed scratch sizing and the PM4-vs-
AQL decision.

:::tip Testing without hardware
The sysfs root is overridable with **`SVOD_KFD_TOPOLOGY`**, so the parser is
unit-tested against a fabricated nodes directory with no GPU present.
:::

---

## The allocation flow

Every buffer follows the same four-step path, implemented once in
`KfdIface::alloc_raw`:

```text
1. reserve_va(size)                     mmap(PROT_NONE, …) — reserve host VA
2. ALLOC_MEMORY_OF_GPU(va, size, flags) → returns handle + mmap_offset
3. if host-visible:                     mmap(va, …, MAP_FIXED, drm_fd, offset)
4. MAP_MEMORY_TO_GPU(handle)            bind into the GPU page table
```

The host VA is reserved first with an anonymous `PROT_NONE` mapping so the
host-visible `mmap` in step 3 can land at exactly that address (`MAP_FIXED`).
Freeing reverses it: `UNMAP_MEMORY_FROM_GPU` → `munmap` → `FREE_MEMORY_OF_GPU`.

### Allocation flavors

`alloc_raw` takes an `AllocKind` that selects the KFD flag set — the single
place those flags are composed:

| `AllocKind` | Flags | Used for |
|---|---|---|
| `DeviceVram { executable }` | `VRAM \| WRITABLE \| NO_SUBSTITUTE` (+ `EXECUTABLE` for code, + `PUBLIC` when host-visible) | Tensor data, code objects, scratch |
| `UncachedGtt` | `GTT \| WRITABLE \| EXECUTABLE \| NO_SUBSTITUTE \| PUBLIC \| COHERENT \| UNCACHED` | Command rings, GART pages, signal slots, the event page |

The `UNCACHED | COHERENT` GTT flavor matters: the command ring and the signal
slots must be immediately visible between CPU and GPU, or the host spins forever
waiting on a completion value stuck in GPU L2. KFD rejects `CREATE_QUEUE` on a
plain-VRAM ring with `EINVAL`.

### Host-visible everywhere

Because there is no SDMA queue, the allocator (`device/src/amd/allocator.rs`)
forces `cpu_access = true` on every buffer: `has_sdma_queue()` is always
`false`, so `_alloc` ORs it in. Copies (`_copyin`/`_copyout`/`_transfer`) are
therefore plain host `memmove` after a `synchronize()`. The generic
`LruAllocator` (`device/src/allocator.rs`) pools freed buffers by
`(size, BufferSpec)`; the `nolru` spec bypasses the pool for code objects,
scratch, and queue infrastructure.

:::note Process-shared state
`/dev/kfd` is opened once per process and shared by all devices (events are
addressed by id against that fd). The 0x8000-byte KFD **event page** is likewise
allocated and bound once per process; subsequent devices only `MAP_MEMORY_TO_GPU`
it into their own `gpu_id`. Both mirror tinygrad's per-process model.
:::

---

## Why this matters

The entire kernel-facing surface is **one vendored header, eleven ioctls, and a
sysfs parser**. That is the whole reason the backend can avoid the ROCm
userspace stack: the kernel ABI is small and stable, so binding it directly is
less code than integrating HIP would be — and it leaves the
[backend seam](./overview.md) free to swap KFD out for the userspace
[AM driver](./am-driver.md) without touching anything above it.
