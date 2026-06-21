---
sidebar_label: Wave32 vs Wave64
---

# Keeping One Kernel Correct on Two Architectures

Here is a bug that doesn't exist on NVIDIA. You write a tile kernel, test it on a CDNA
datacenter GPU, and it's perfect. You run the *same* kernel on an RDNA laptop APU and the
numbers are garbage — no crash, no error, just wrong. Nothing in the code looks different.

[What Tiling Is](./tiling) introduced fragments and role-based selection; this chapter explains
why that indirection has to exist. The culprit is the **wavefront size**, and dealing with it
cleanly is what separates a tile library that works on one chip from one that's actually portable.

---

## The 32-vs-64 split

A wavefront (AMD's "warp") is the group of lanes that execute in lockstep. On AMD there are two
sizes, and Svod targets both:

| Architecture | Example | Matrix op | Wavefront |
|--------------|---------|-----------|-----------|
| **CDNA** | gfx942 (datacenter) | MFMA | **wave64** — 64 lanes |
| **RDNA** | gfx1151 (RDNA3.5) | WMMA | **wave32** — 32 lanes |

That single number ripples through everything. A `16×16` tile has 256 elements. Spread across
64 lanes, that's 4 elements per lane; across 32 lanes, it's 8. Different lanes own different
elements. So:

- the **register layout** of a tile differs,
- the **operand layout** the matrix instruction expects differs (RDNA even *replicates* some
  operands across lanes),
- and any **cross-lane reduction** — the heart of softmax and layernorm — has a different
  number of steps and a different sibling pattern.

A kernel that hardcodes "there are 64 lanes, reduce by xor-ing lanes 16, 32, 48" computes a
*partial* reduction on a 32-lane machine and silently returns wrong values.

---

## The fix: ask for a role, not a shape

`tk`'s answer is a layer of indirection. A kernel never writes down a concrete fragment shape
like "16×16, 4 elements per lane." Instead it asks for a **role**, and lets the architecture
capabilities resolve it:

```text
   kernel says:  "I need an accumulator fragment"   (FragRole::Accumulator)
                          │
                          ▼
   ArchCaps::frag(role)   ── on CDNA ──▶  the wave64 16×16 shape
                          └─ on RDNA ──▶  the wave32 16×16 shape (8 ept, replicated operands)
```

The roles are `FragRole::{Accumulator, Operand, AccumulatorT}` and the resolver is
`ArchCaps::frag(role)` in `tk/src/arch.rs`. The kernel author writes "accumulator" and "operand";
the *physical* layout — element count per lane, the interleave map, replication — is filled in
for the target wave size. Write once, run on both.

This is the same lesson HipKittens learned (see [tk vs HipKittens vs CuTile](./comparison)): it
ships two parallel backends, `cdna4` (wave64) and `udna1` (wave32), keyed off a single
`WARP_THREADS` constant so the tile types recompile correctly for each. `tk` collapses that into
one runtime-resolved `ArchCaps`.

---

## A bug this actually caught

The reason this indirection exists isn't theoretical. An early `tk` cross-lane all-reduce —
the `shuffle_xor` primitive used to sum a value across a wave — was written with a hardcoded
wave64 reduction tree. On RDNA's 32-lane waves it reduced over lanes that don't participate,
producing wrong sums for exactly the softmax-style reductions attention depends on. The fix was
to drive the reduction off `caps.wave_size` and the role-resolved fragment instead of a
constant. The shuffle primitives in `tk/src/group.rs` now read the wave size; the bug class is
designed out.

:::tip For GPU experts
Two capability methods on `ArchCaps` (`tk/src/arch.rs`) carry most of the wave-specific weight:

- **`reduce_tree()`** returns the cross-lane sibling-fold offsets. On wave64 that's
  `[16, 32, 48]` (three xor steps to fold 4 sub-fragments); on wave32 it's `[16]` (one step).
  Cross-lane reductions iterate over this list — get it from `caps`, never hardcode it.
- **`acc_reusable_as_input()`** answers: "can a matrix accumulator be fed straight back in as an
  operand to the next multiply?" On CDNA it's `true` — the layouts match, so it's a free register
  copy. On RDNA it's `false` — the accumulator and operand layouts differ, so the value makes a
  round-trip through LDS to be relaid out. [Flash Attention](./flash-attention) handles this split
  between its two matmuls.

The `ept` field on `BaseShape` (from [What Tiling Is](./tiling)) exists for the same
reason: on RDNA, operands are replicated across lanes, so elements-per-thread isn't
`element_count / wave_size` and must be stored explicitly.
:::

---

## Why this matters

Portability across wave sizes is the AMD-specific tax on hand-written kernels, and it's why a
naive port of an NVIDIA tile library doesn't just work. `tk` pays the tax once, in the
`ArchCaps` abstraction, so individual kernels stay readable: they speak in *roles* and let the
hardware table sort out the lanes. [Flash Attention](./flash-attention) is where you see this
pay off in a real kernel.
