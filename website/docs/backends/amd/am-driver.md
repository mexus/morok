---
sidebar_label: AM Driver
---

# The AM Driver (Userspace)

The **AM** driver is a second [`AmdIface`](./overview.md) backend that drives the
GPU's PCI BARs directly, bypassing the kernel `amdgpu`/KFD driver entirely. It
is a port of tinygrad's userspace AM driver. The motivation is concrete: the
lock-free [multi-queue dispatch](./queues-and-dispatch.md) path overloads the
kernel's MES/runlist scheduler under heavy concurrent load and can crash the
kernel. If we own the GPU — page tables, firmware, scheduling — the kernel is
never in the dispatch path and can't be overloaded.

:::caution Work in progress — not yet selectable
This page documents both what exists today and the roadmap for the rest.
**`SVOD_AMD_BACKEND=am` currently returns an error** (`device.rs` accepts only
`kfd`). What is implemented is the unprivileged, GPU-free *logic*; the
privileged hardware bring-up is deferred. The sections below mark each piece's
status explicitly.
:::

The code lives under `device/src/amd/am/`. It compiles unconditionally on Linux
(pure logic, no extra dependencies), so it is always type-checked, linted, and
unit-tested — the backend is chosen at *runtime*, never behind a cargo feature
that could rot.

---

## What the kernel did for us

KFD gave the backend three things. AM has to provide each itself:

| KFD provided | AM must do |
|---|---|
| VRAM allocation + GPU page-table mapping | A GMMU: VA allocator + 4-level page-table walker + PTE encoding |
| Queue creation (MES/HQD setup) | Write the MQD, enable the MEC, map the doorbell |
| Memory + firmware bring-up | PCI BAR mapping, IP discovery, PSP firmware load |

Crucially, everything *above* the seam — the PM4/AQL packet builders, the
ring, signals, kernarg arena, timeline, and back-pressure — carries over
**unchanged**. AM only replaces the five `AmdIface` methods.

---

## What exists today (built & tested)

The pure-logic half is implemented and unit-tested without a GPU, backing the
page tables with an injectable `PhysMem` trait (a plain buffer in tests,
BAR-mapped VRAM in the real driver).

| Module | What it implements | Status |
|---|---|---|
| `am/mm/tlsf.rs` | TLSF (Two-Level Segregated Fit) allocator — port of tinygrad's `TLSFAllocator` | **Done** + unit tests + a proptest |
| `am/mm/pagetable.rs` | GMMU geometry + PTE/PDE bit encoding | **Done for gfx11** + tests |
| `am/mm/manager.rs` | `MemoryManager`: VA alloc, 4-level page-table walk, huge-page selection, table reclaim, `valloc`/`vfree` | **Done** + tests against a fake VRAM |
| `am/regs.rs` | `RegDef`/`RegField` types + the `select(prefix, ip_ver)` resolver | **Done** + tests |
| `am/regs_gen.rs` | Vendored register tables (`GC_11_5_0`, `MMHUB_3_3_0`, `MP_14_0_2`, …) | **Generated & committed** |

### The GMMU

The page-table geometry is **4-level / 48-bit** (`va_shifts = [12, 21, 30, 39]`),
a shape **shared across gfx9/11/12** — so the geometry itself does not branch on
arch. Only the leaf PTE encoding (notably the MTYPE memory-type field) is
arch-specific. The `MemoryManager` runs three TLSF sub-allocators (VA space,
physical VRAM, page-table pool) and walks the table in `Inspect` / `Create` /
`Free` modes, selecting huge pages where alignment allows and reclaiming empty
tables on unmap.

### Register tables are generated-once, then vendored

tinygrad is a sometimes-absent submodule, so the build must never depend on it.
Instead `device/tools/gen_am_regs.py` is run **manually** when adding or updating
an arch: it parses tinygrad's `autogen/am/regs.py` and emits the committed
`am/regs_gen.rs`. `regs.rs` just `include!`s it. At boot the right table is
chosen by the discovered `ip_ver` (`select` picks the greatest version `≤ ip_ver`
sharing the same major — tinygrad's `import_module` rule). Adding an arch is
widening the generator's module list and re-running it — no build or runtime
logic change.

---

## What is deferred (not yet in the tree)

The privileged bring-up needs root/caps (unbind `amdgpu`, `mmap` the PCI BARs,
mode-1 reset) and is not present in the source yet:

- the **AMDev orchestrator** (BAR map, boot sequence);
- **PCI/BAR** access and **IP discovery** parsing;
- **PSP firmware load** (the highest-risk subsystem — a version-specific
  handshake);
- the **IP-block** modules (SOC / GMC / IH / PSP / SMU / GFX / SDMA);
- the **`AmIface`** implementor that ties it all to the seam.

Within the *implemented* page-table module, only **gfx11/RDNA3** is live: the
gfx9 (VG10) and gfx12 PTE-encoding paths are deliberate `unimplemented!` panics,
each guarded by a test asserting it panics — the constants are captured but not
yet hardware-validated.

---

## Target hardware & arch parametrization

The register and page-table target is **gfx1151 — the "Strix Halo" APU** (which
reports GC 11.5.1 → the `gc_11_5_0` table). The driver is parametrized the way
tinygrad's is: by **`ip_ver` tuples read from IP discovery at boot**, not a
hand-maintained arch enum. Arch differences are meant to be small inline
`if ip_ver >= (X, Y, Z)` branches inside shared modules plus version-keyed
register tables — so gfx12 becomes mostly a data addition and gfx9 is
accommodated but deferred.

:::note Why bring-up is deferred on this machine
The actual hardware is a Strix Halo APU that is also the **primary display GPU**.
AM has to unbind `amdgpu` and take exclusive ownership, which kills the display;
and tinygrad's AM whitelists discrete RDNA3/4 device IDs, not this APU. So there
is no working AM oracle on this machine to validate against. The privileged
bring-up (phases below) is deferred to an external/discrete GPU where tinygrad
AM is a proven reference. Meanwhile, the [single-queue KFD mode](./queues-and-dispatch.md)
already fixed the kernel crash that motivated AM, so nothing is blocked in the
interim.
:::

---

## Roadmap

Each phase is independently testable, with tinygrad AM on the same card as the
per-phase oracle:

| Phase | Milestone |
|---|---|
| **A** | PCI + discovery (read-only): unbind `amdgpu`, map BARs, parse IP discovery, diff every value against tinygrad |
| **B** | regs + GMC page tables: `valloc` + map a buffer, read back the PTE, round-trip data through the BAR |
| **C** | PSP firmware load (the risk gate): the sOS bootloader handshake, TMR, per-IP firmware load — diffed dword-for-dword against a tinygrad transcript |
| **D** | GFX MEC + `setup_ring`: write the v11 compute MQD, enable the MEC, map the doorbell (`CP_HQD_ACTIVE == 1`) |
| **E** | dispatch one kernel: reuse the entire existing upper half on the AM-backed core |
| **F** | concurrency + de-stub: real interrupt handler, max clocks, run the workload that crashed KFD — which can't crash now, the kernel is bypassed |

Once Phase F runs the crash-inducing concurrency cleanly, AM becomes the
recommended mode for multi-queue/streaming workloads, with KFD (single-queue)
remaining the portable fallback. gfx12/RDNA4 is then a cheap follow-on (widen the
register tables + add the `gc >= (12,0,0)` branches); gfx9/CDNA is a larger,
later effort.

---

## Why this matters

The AM driver is the real answer to the kernel-overload problem that the
[single-queue mode](./queues-and-dispatch.md) only sidesteps. The expensive,
GPU-free parts — the GMMU and the register tables — are already built and tested,
so the remaining work is the privileged bring-up, which is gated on hardware
rather than on design. And because it slots in behind the same five-method
[seam](./overview.md), none of the dispatch, compile, or graph machinery has to
change when it lands.
