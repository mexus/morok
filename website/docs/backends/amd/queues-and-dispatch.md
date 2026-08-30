---
sidebar_label: Queues & Dispatch
---

# Queues & Dispatch

The AMD backend preserves Tinygrad's validated PM4, AQL, and SDMA packet
semantics, but uses Rust ownership for queue scheduling and failure handling.
The central rule is simple: **one non-clone lease is the only publication
authority for a compute lane**.

## Compute lanes

`AmdDeviceCore` owns a bounded `QueuePool`. Its slots are fixed `OnceLock`s and
queues are created lazily up to `SVOD_AMD_HW_QUEUES` (default 4, maximum 64).
An atomic bitset tracks leases:

- claiming an initialized idle lane is an atomic compare-exchange;
- queue creation is a cold serialized path;
- when every lane is leased, callers park on a condition variable;
- dropping `QueueLease` clears the bit and wakes one waiter;
- queues never co-tenant host publishers.

The `QueueLease` is deliberately not stored in programs or graph templates.
`OwnerCtx` contains logical plan state: completion, profiling configuration,
and an optional linked replay template.

Direct semantic fallback keeps one lease across all kernels in a replay epoch,
then `PlanContext::finish_replay` releases it. A later epoch waits the prior
finalizer before acquiring another lane, because a different queue would not
inherit the old queue's FIFO ordering. Graph and native linked replay already
wait before reusing their mutable kernarg/control storage and lease a lane for
each publication epoch.

## Native queues

`AmdComputeQueue` owns a 16 MiB host-visible ring, GART read/write pointers, a
doorbell mapping, and KFD queue backing. Packet format is selected once:

```text
PM4 = num_xcc == 1 && SVOD_AMD_AQL != 1
AQL = otherwise
```

- PM4 queues publish raw dwords and ring the next dword index.
- AQL queues publish 64-byte packets and ring the last completed packet index.
- AQL kernel `completion_signal` remains zero. Vendor-IB PM4 waits/stores own
  timeline completion, with XCC0 `PRED_EXEC` on multi-XCC hardware.

The lane lease eliminates compute co-tenancy. `AmdComputeQueue.inner` still uses
a mutex as a Rust aliasing guard; it is uncontended on the normal compute path.
The singleton SDMA queue is independently mutex-protected because copies from
different plans may share it.

## Publication

Submission is split into preparation and publication:

1. Validate program identity, concrete buffer ownership, ABI, launch geometry,
   patch tables, and hardware stream limits.
2. Reserve and write kernargs/control data.
3. Acquire ring headroom.
4. Register a prepared finalizer when device-wide drains need to observe a
   plan-owned timeline.
5. Publish packets and doorbells.
6. Mark the finalizer published.

If an error unwinds after registration, the prepared finalizer becomes failed.
A concurrent drain wakes and fails immediately rather than waiting for a
terminal store that was never published. The physical device is then poisoned,
so the lane cannot be reused and hardware-referenced allocations are
quarantined.

PM4, AQL, and SDMA publication all check monotonically increasing KFD read
pointers before wrapping their rings. Ordinary dispatch additionally bounds
in-flight timeline values. PM4 timeline values drain and reset at the 2^31
watermark because hardware wait/store packets compare the low 32 bits.

## Resource lifetime

Every direct submission finalizer retains its code object. Graphs and linked
plans retain all code objects they link. Persistent kernarg, resident command,
control, timestamp, and PMC allocations remain owned until their exact replay
completion is retired.

Queue lifecycle is explicit:

```text
Constructing -> Active
Constructing -> Destroyed | Quarantined
Active -> Destroyed
Active -> Quarantined
```

Orderly compute teardown is drain, KFD `DESTROY_QUEUE`, scratch release, then
ring/GART/context release. A failed drain or destroy poisons the physical device
and leaves all potentially referenced backing mapped. Doorbell unmap failure
after successful queue destruction is reported as a host mapping leak, but does
not unnecessarily quarantine safe GPU backing.

If `CREATE_QUEUE` succeeds but doorbell mapping and rollback destruction both
fail, `setup_ring` returns `AmdQueueStillActive`. The caller poisons the device
before allocation guards unwind, preventing a live KFD queue from observing
freed ring memory.

Panic abandonment also poisons the device. Signal slots are not returned to the
pool while panicking or after poison, so a caught panic cannot recycle a slot
that an abandoned queue may still target.

## Device-wide drains

Each lane owns a queue timeline and a FIFO of non-queue finalizers. The device
core keeps weak references to every initialized lane. Host reads, host writes,
and destructive frees call `synchronize_all`, which snapshots those lanes and
waits their timelines without taking publication locks.

Native replay additionally validates every current PROGRAM and COPY endpoint.
For AMD buffers this compares the actual `RawBuffer::AmdDevice` core with the
selected physical device; an allocator merely reporting `AMD:N` is not enough.

## Backend seam

KFD operations are isolated behind `AmdIface`:

```rust
pub trait AmdIface: Send + Sync + std::fmt::Debug {
    fn alloc_raw(/* ... */) -> Result<AllocResult>;
    fn free_raw(&self, gpu_va: u64, size: usize, handle: u64);
    fn setup_ring(&self, desc: &RingDesc) -> Result<QueueHandle>;
    fn teardown_ring(
        &self,
        queue_id: u32,
        doorbell_base: NonNull<u8>,
    ) -> Result<QueueTeardown>;
    fn wait_events(&self, timeout_ms: u32) -> Result<Option<Error>>;
}
```

Ring, GART, EOP, context-save, and inactive-signal buffers are allocated above
this seam. `setup_ring` activates those resources and maps the doorbell.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `SVOD_DEVICE` | `CPU` | Select default tensor device, for example `AMD:0` |
| `SVOD_AMD_BACKEND` | `kfd` | AMD backend; only `kfd` is currently accepted |
| `SVOD_AMD_HW_QUEUES` | `4` | Bounded compute-lane count, clamped to 1 through 64 |
| `SVOD_AMD_AQL` | unset | Nonzero forces AQL on single-XCC hardware |
| `SVOD_PM4_GRAPH` | unset | `=1` enables PM4 graph capture |
| `SVOD_KFD_TOPOLOGY` | sysfs | Override KFD topology root for tests |
| `SVOD_DEBUG_DISPATCH` | unset | Print dispatch grid, kernarg, scratch, and buffer addresses |
| `SVOD_DUMP_AMD_IR` | unset | Directory for generated AMD LLVM IR |

There is no `SVOD_AMD_SINGLE_QUEUE`. Set `SVOD_AMD_HW_QUEUES=1` when a single
hardware lane is required.
