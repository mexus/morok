---
sidebar_label: Overview
---

# Why Hand-Written Kernels?

Svod's whole personality is automation. You build a lazy graph, call `realize()`, and the
optimizer decides how to tile, vectorize, and parallelize every loop — and if you turn on
[beam search](../architecture/optimizations/kernel-search), it will compile and *time*
hundreds of candidate schedules to find a fast one. You never write a loop.

So why does Svod ship a crate — `tk` — whose entire job is letting you write GPU kernels
*by hand*?

Because some kernels can't be discovered by searching over loop transformations. The
optimizer's action space is "take this reduction and tile it, unroll it, put it in shared
memory." That's enough for matmul, for a fused feed-forward block, for a layernorm. It is
**not** enough for Flash Attention, whose math is a *recurrence*: each block of keys updates
a running maximum and a running sum, rescaling the accumulator as it goes. There is no single
`REDUCE` to tile — the loop body depends on the result of the previous iteration. No amount
of axis-shuffling produces it.

For kernels like that, you need to write the algorithm. `tk` is how you do it, without
leaving the compiler.

---

## `tk` is a builder, not a backend

The temptation, when you need a hand-written kernel, is to bolt on a second code path: a
little GPU DSL that emits its own assembly and gets launched separately from everything else.
Now you have two compilers, two debuggers, two mental models.

`tk` refuses that. It is, in its own words, *"a thin eager builder, not a backend."* When you
author a kernel with `tk`, it doesn't emit machine code — it emits the **same UOp IR** that
the rest of Svod already speaks: explicit `RANGE` loops, `INDEX`/`STORE` memory ops, `WMMA`
matrix-core ops. The exact intermediate representation described in the
[IR Design chapter](../architecture/ir-design).

That means a hand-written `tk` kernel and an autotuned graph kernel are the *same kind of
object* — two subgraphs in one UOp DAG, rendered by one renderer, run by one runtime. The
[next chapters](./lowering) show exactly how that works.

---

## The three faces of `tk`

Depending on who you are and what you're doing, `tk` presents one of three interfaces (all
re-exported from `tk/src/lib.rs`):

| Face | You are… | What you touch |
|------|----------|----------------|
| **USE** | an application author who just wants a fast kernel | `matmul`, `flash_attention`, `flash_attention_with` — they return lazy `Tensor`s, no kernel knowledge required |
| **AUTHOR** | writing a new tile kernel | the `Kernel` / `Group` builder, `ArchCaps`, the tile types (`GL`/`ST`/`RT`/`RV`), `Swizzle`, `graph_launch` |
| **DEBUG** | testing or benchmarking a kernel in isolation | `compile`, `launch`, `run_kernel`, `CompiledLaunch`, and structural `KernelFingerprint`s |

The USE face is the important one for most readers: `flash_attention(q, k, v)` gives you back
an ordinary `Tensor` that participates in the lazy graph like any other. You never see a tile.
The [tiling chapter](./tiling) opens up the AUTHOR face; the
[debugging chapter](./debugging) covers DEBUG.

---

## When to hand-write, and when to let BEAM do it

The decision rule is simple, and it's worth stating plainly because it's the reason `tk` is
small:

| Kernel | How Svod builds it | Why |
|--------|--------------------|-----|
| matmul / GEMM | **graph-native + BEAM** | A single reduction over the contraction axis. BEAM's `TC`/`UPCAST`/`LOCAL` actions tile it well; a hand kernel buys little. |
| feed-forward / elementwise | **graph-native + BEAM** | Plain fusible graph ops. |
| **Flash Attention** | **hand-authored in `tk`** | Online-softmax recurrence — not expressible as one schedulable `REDUCE`. |

`tk` *does* also ship a hand-written `matmul`, but it earns its keep as a performance canary
for the DSL itself, not as the production matmul. Production matmul goes through the graph.

In other words: **the hand-written surface is deliberately tiny.** It exists for the kernels
the search can't reach, and nothing more.

:::tip For GPU experts
The structural difference between a hand-authored kernel and a BEAM-tuned one is a *single
field*. Every kernel in Svod is a `SINK` UOp carrying a `KernelInfo`. A graph kernel leaves
`opts_to_apply: None`, which tells the optimizer "you choose the schedule (heuristics or
beam)." A `tk` kernel sets `opts_to_apply: Some(vec![])` — "this body is already lowered;
apply *zero* further optimizations." Same IR, same pipeline, one marker. The
[lowering chapter](./lowering) traces this end to end.
:::

---

## Where this section goes

The rest of this section builds up from the hardware problem to the design comparison:

1. **[Where the FLOPS Hide](./where-flops-hide)** — why a matrix core is hard to saturate,
   and the handful of bottlenecks every fast kernel has to beat.
2. **[What Tiling Is](./tiling)** — the abstraction that answers those bottlenecks, and how
   `tk` represents tiles in the type system.
3. **[Authoring into the IR](./lowering)** — how a `tk` kernel becomes UOps and joins the
   lazy graph.
4. **[Wave32 vs Wave64](./wave-portability)** — keeping one kernel correct across two AMD
   architectures.
5. **[Flash Attention](./flash-attention)** — the worked example that motivated all of this.
6. **[Debugging](./debugging)** — running and verifying kernels by hand.
7. **[tk vs HipKittens vs CuTile](./comparison)** — where this design sits in the landscape.
