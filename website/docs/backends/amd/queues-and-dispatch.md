---
sidebar_label: Queues & Dispatch
---

# Queues & Dispatch

Dispatching a kernel means writing command packets into a ring buffer and
ringing a doorbell. This page covers the ring machinery (`AmdComputeQueue`), the
per-owner bundle that wraps it (`AmdConnector`), the two dispatch strategies
(single-queue vs multi-queue), the completion primitive (`Timeline`), and every
environment variable that configures the backend.

The shape of this design comes from one fact: **tinygrad is GIL-serialized** —
one compute queue per device, with Python's GIL making each dispatch atomic.
Svod removes the GIL to get real concurrency, so the invariants the GIL provided
have to be rebuilt explicitly. The result is a dispatch path that can be
lock-free.

---

## The command ring: `AmdComputeQueue`

`device/src/amd/queue.rs` defines `AmdComputeQueue`, which owns:

- a **16 MiB host-visible ring** (`COMPUTE_RING_BYTES`) — command packets are
  written straight into it from the CPU;
- a **doorbell** (`*mut u64` MMIO) — the GPU's command processor is told "new
  work" by writing the new write-index here;
- GART-resident **write/read dispatch-id** slots — KFD reads the write pointer
  in addition to the doorbell, so it's published first.

### PM4 vs AQL

There are two on-wire packet formats, chosen once at queue creation from the
device's XCC count:

```text
will_use_pm4(core) = !SVOD_AMD_AQL && num_xcc == 1
```

- **PM4** (single-XCC: the gfx11/12 default) — raw PM4 dwords written directly
  into the ring (`KFD_IOC_QUEUE_TYPE_COMPUTE`). The doorbell is rung with the
  next dword index.
- **AQL** (multi-XCC CDNA) — 64-byte AQL packets
  (`KFD_IOC_QUEUE_TYPE_COMPUTE_AQL`), with PM4 helpers wrapped inside AQL
  vendor-IB packets. The doorbell is rung with the last-completed slot
  (`write_idx - 1`).

A single PM4 dispatch is a fixed sequence, mirroring tinygrad's
`hcq.py:371-378`:

```text
wait(timeline, prev)  →  hdp_flush  →  acquire_mem  →  exec  →  release_mem(timeline, next)
```

`exec` is the `SET_SH_REG` stream that loads the shader address, the
`RSRC1/2/3` registers, the scratch descriptor and `TMPRING_SIZE`, the
`USER_DATA` SGPRs, the launch dims, then `DISPATCH_DIRECT` followed by a
`CS_PARTIAL_FLUSH`. The `release_mem` at the end writes the dispatch's timeline
value to the connector's signal slot when the GPU finishes.

### Lock-free interior mutability

`AmdComputeQueue.inner` is an `UnsafeCell<QueueInner>`, not a `Mutex` — dispatch
mutates it through `&self` with no lock. This is sound because of a
**single-owner invariant**: for the lifetime of a `ConnectorLease`, exactly one
thread issues sequential, non-reentrant dispatch against the queue (the same
pattern `RawBuffer` uses in `device/src/allocator.rs`). The shared drainer never
touches the queue — it reads only the timeline (see below). Distinct connectors'
queues are interleaved by the GPU's hardware scheduler (MES — the MicroEngine
Scheduler), not a CPU lock.

### Ring back-pressure

A host running `wait=false` faster than the GPU drains would lap the 16 MiB ring
and overwrite unconsumed packets. `wait_dispatch_headroom` prevents this by
bounding the number of un-retired dispatches to `RING_MAX_INFLIGHT` (half the
ring), blocking on the **timeline signal** when the bound is hit:

```rust
let last_reserved = conn.timeline_value().saturating_sub(1);
if last_reserved > RING_MAX_INFLIGHT {
    let target = last_reserved - RING_MAX_INFLIGHT;
    conn.timeline_signal().wait_signal_value(target, 30_000)?;
}
```

It gates on the timeline signal — the proven completion primitive — rather than
the PM4 read pointer, whose COMPUTE-queue semantics are unreliable and would
deadlock a spin.

---

## The per-owner bundle: `AmdConnector`

A queue alone isn't enough to dispatch. `AmdConnector`
(`device/src/amd/connector.rs`) bundles everything one independent caller needs:

| Field | What it is |
|---|---|
| `queue: Box<AmdComputeQueue>` | The ring + doorbell + GART (sole owner → lock-free) |
| `arena: Box<KernargArena>` | A 16 MiB GTT kernarg bump arena |
| `scratch_state: Mutex<ScratchState>` | Register-spill scratch backing, grown on demand |
| `timeline: Arc<Timeline>` | The monotonic counter + completion signal |

Every `ExecutionPlan` and every `AmdGraph` owns its own connector. The `Box`
(not `Arc`) on the queue and arena is load-bearing: it proves there is no second
handle aliasing the `UnsafeCell`, which is what makes lock-free dispatch sound.
The arena is per-connector so its bump cursor and the connector's timeline share
one ordering — a wrapped kernarg slot is provably free once that timeline
drains.

`ensure_has_local_memory(private_segment_size)` grows the scratch buffer when a
freshly-loaded kernel needs more bytes-per-thread than currently allocated
(alloc new → swap → drain → free old). Scratch is GPU-only VRAM, dynamically
realloc'd, and the historical source of `NotPresent` faults — see
[Debugging](./debugging.md).

---

## Two dispatch strategies

The per-device `Dispatcher` enum (`device/src/amd/device.rs`) chooses how owners
get a connector and whether dispatch is serialized. It is built once at
device-open from `SVOD_AMD_SINGLE_QUEUE`:

### Single-queue (default)

```text
SVOD_AMD_SINGLE_QUEUE unset or ≠ 0
```

Every owner shares **one** connector per physical device, and dispatch +
scratch-realloc are serialized behind a `Mutex<()>` taken via `exec_guard()`.
The kernel then only ever sees one compute queue per GPU — tinygrad's model.

This is the **KFD-safe** mode, and it is the default for a concrete reason:
heavy concurrent multi-queue dispatch **overloads the kernel's MES/runlist
scheduler and can crash the kernel**. One GPU has one command processor and runs
dispatches sequentially anyway; multi-queue only overlapped CPU-side packet
assembly, which is what drove KFD into the bad path. Single-queue removes that
crash.

### Multi-queue (opt-in)

```text
SVOD_AMD_SINGLE_QUEUE=0
```

Each owner leases an **exclusively-owned** connector from an idle pool (bounded
by `CONNECTOR_POOL_CAP = 4`); the MES interleaves the N queues, so dispatch
needs no CPU lock and `exec_guard()` returns `None`. The lease being exclusive
and un-aliasable is what stops two dispatchers from sharing one KFD queue.

:::caution The kernel-overload caveat
Multi-queue is the lock-free, maximally-concurrent path, but it is the one that
overloads KFD under load. It is opt-in for that reason. The real fix — owning
the GPU so the kernel is never in the dispatch path — is the
[userspace AM driver](./am-driver.md).
:::

### `ConnectorLease`

`lease_connector` returns a `ConnectorLease` — a non-`Clone` handle that
`Deref`s to `&AmdConnector`, so callers are mode-agnostic. On drop,
`return_connector` does the mode-appropriate thing: re-pool it (multi-queue, up
to the cap) or nothing (single-queue — the shared connector lives on the core).
It does **not** synchronize on drop; the connector's `Timeline` stays registered
so the device-wide drain still covers it.

---

## The completion primitive: `Timeline`

`Timeline` (`device/src/amd/signal.rs`) is a monotonic `AtomicU64` counter plus
the GTT-coherent `AmdSignal` slot the GPU writes on dispatch completion. It is
**the one primitive that crosses owners**:

- a connector *dispatches* against it — `next()` does `fetch_add(1)` to reserve
  the value its `release_mem` packet will write;
- any thread can *drain* it — `drain()` reads the atomic and polls the signal
  slot, **never touching the queue**.

That decoupling is what keeps dispatch lock-free. The device core
(`AmdDeviceCore`) holds `Weak<Timeline>` for every connector — not
`Weak<AmdConnector>` — so `synchronize_all` (the fence before any host read or
buffer free) drains all in-flight work purely through these atomics:

```text
AmdDeviceCore.synchronize_all():
   for each live Timeline:  timeline.drain(30s)   // atomics + signal slot only
```

`AmdSignal::wait_signal_value` polls in tiers — tight spin → `yield_now` → KFD
`WAIT_EVENTS` sleep after 200 ms — so a long or stalled wait doesn't burn a CPU,
and a GPU fault during the wait surfaces immediately instead of blocking the
full 30 s timeout.

:::note 2³² wraparound
PM4 `WAIT_REG_MEM`/`RELEASE_MEM` compare the low 32 bits of the signal slot, so
the counter must stay below 2³². `ensure_timeline_headroom` drains and resets at
a 2³¹ watermark (`TIMELINE_WRAP_WATERMARK`) before reserving each value, so a
long `wait=false` loop can't climb past 2³² and produce a false timeout.
:::

---

## The seam

All four kernel operations the queue layer needs route through the
[`AmdIface`](./overview.md) trait on the device core:

```rust
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    fn alloc_raw(&self, size, kind, tag, cpu_access, zero) -> Result<AllocResult>;
    fn free_raw(&self, gpu_va, size, handle);
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    fn teardown_ring(&self, queue_id: u32);
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;
}
```

Note what is *not* in the trait: the ring, GART, EOP and ctx-save buffers are
all allocated above the seam (via `alloc_raw`) inside `create_queue`. The trait
only **activates** the queue — `setup_ring` issues `CREATE_QUEUE` and `mmap`s
the doorbell over a ring the upper half already owns. `KfdIface` is the sole
implementor today.

---

## Configuration reference

Every environment variable that affects the AMD backend:

| Variable | Default | Effect |
|---|---|---|
| `SVOD_DEVICE` | `CPU` | Selects the default device for tensors. Set `SVOD_DEVICE=AMD:0` to run on the first AMD GPU |
| `SVOD_AMD_BACKEND` | `kfd` | Backend selection. Only `kfd` is accepted today; `am` is the future seam (errors if set) |
| `SVOD_AMD_SINGLE_QUEUE` | `1` (on) | `=0` opts into lock-free multi-queue dispatch; otherwise the KFD-safe single-queue mode |
| `SVOD_AMD_AQL` | `0` (off) | `=1` forces AQL packets even on single-XCC hardware — for bisecting PM4 vs AQL issues |
| `SVOD_JIT_GRAPH` | unset | Enables PM4 graph capture/replay (also requires multi-queue mode). See [Compile & Graph](./compile-and-graph.md) |
| `SVOD_KFD_TOPOLOGY` | sysfs path | Overrides the topology root, for testing without hardware |
| `SVOD_DEBUG_DISPATCH` | unset | Dumps per-dispatch kernel / grid / kernarg / scratch / buffer VAs. See [Debugging](./debugging.md) |
| `SVOD_DUMP_AMD_IR` | unset | If set to a directory, dumps each kernel's AMD LLVM IR there |

:::caution There is no `SVOD_AMD_MAX_QUEUES`
The multi-queue idle-pool size is the compile-time constant
`CONNECTOR_POOL_CAP = 4` in `device.rs`, not an environment variable.
:::

---

## Why this matters

The GIL gave tinygrad an atomic dispatch critical section for free. Svod rebuilds
that guarantee three ways: **single-owner ownership** for the ring (no dispatch
lock), a **shared timeline signal** for drains (atomics, never the queue), and
**explicit ring back-pressure**. The single-queue default keeps the kernel safe
today; the lock-free multi-queue path is ready for when the
[AM driver](./am-driver.md) takes the kernel out of the loop entirely.
