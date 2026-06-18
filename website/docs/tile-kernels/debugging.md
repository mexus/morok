---
sidebar_label: Debugging
---

# Debugging and Verifying Kernels

A hand-written kernel is only as trustworthy as your ability to check it. The USE face hands
you a lazy `Tensor` that fuses into a big graph — convenient, but a bad place to ask "is this
one kernel correct, and how fast is it?" `tk`'s **DEBUG face** exists for exactly that: run a
single kernel against concrete buffers, read the result back, time it, and prove that a
refactor didn't change its behavior.

---

## Direct dispatch: run one kernel, see the bytes

The direct-launch API (`tk/src/launch.rs`) bypasses the tensor scheduler entirely. You give it
a finished `Kernel` and real input buffers; it renders, compiles, and dispatches, writing the
result into an output buffer you can read back:

```rust
// Conceptual — the DEBUG face from tk/src/lib.rs
let out = run_kernel(&kernel, &[&input_a, &input_b])?;
let values = out.as_vec::<f32>()?;   // read the GPU result straight back
assert_eq!(values, expected);
```

Because this skips scheduling, fusion, and dependency tracking, what you measure is *just your
kernel* — not a graph that happens to contain it. That isolation is the point: when a number is
wrong, you want to know it's wrong *here*, not somewhere in a fused pipeline.

A note on the path: the direct route runs one rewrite the normal pipeline would otherwise apply
later (lowering `Index` arithmetic to the target's integer dtype), because it deliberately
skips the optimizer stage that usually does it. You get correct code without the scheduler.

---

## Timing on real hardware

For performance work, `CompiledLaunch` (from `compile` / `compile_kernel`) exposes hardware
timestamps rather than wall-clock guesses:

```rust
let launch = compile_kernel(&kernel, device)?;
launch.dispatch(&buffers)?;
let ns = launch.dispatch_gpu_ns();   // device-measured dispatch time
```

`dispatch_gpu_ns()` reads the GPU's own timestamp counters around the dispatch, so you're
measuring time on the device, not the round-trip latency of launching it. This is what the
criterion benches in `tk/benches/kernels.rs` use to compare a `tk` kernel against the
graph-native baseline.

---

## Fingerprints: proving a refactor is behavior-preserving

The subtle risk with hand-written kernels: you "clean up" the builder code, the kernel still
compiles and still produces plausible numbers, but the *generated IR* changed in a way that
only shows up on some shape or some architecture later.

`KernelFingerprint` (`tk/src/fingerprint.rs`) guards against this. It computes a deterministic,
structural hash of a kernel's UOp graph — the shape of the SINK, not the pointer identities. You
snapshot the fingerprint as a golden value, and a refactor that's meant to be purely cosmetic
must reproduce it:

```rust
let fp = kernel_fingerprint(&sink);
assert_eq!(fp, GOLDEN_MATMUL_FINGERPRINT);  // structure unchanged ⇒ behavior unchanged
```

If the fingerprint moves, you changed the emitted IR — intentionally or not — and the golden
test makes you look. The unit tests under `tk/src/test/unit/golden` use exactly this to lock
the matmul and Flash-Attention graphs.

---

## Which tool for which question

| You're asking… | Use |
|----------------|-----|
| "Does this kernel produce the right numbers?" | `run_kernel` + `as_vec`, compare against a reference |
| "How fast is it on this GPU?" | `compile_kernel` + `dispatch_gpu_ns` |
| "Did my refactor change the emitted IR?" | `KernelFingerprint` golden test |
| "Is the *device/driver layer* misbehaving?" | the [AMD backend debugging guide](../backends/amd/debugging) |

That last row matters: this chapter is about debugging *kernels* — the IR you authored and the
numbers it produces. When the problem is below that — queue dispatch, memory faults, the driver —
the [AMD Backend → Debugging](../backends/amd/debugging) chapter is the right place.

---

## Why this matters

Hand-authoring trades the optimizer's safety net for control. The DEBUG face is how you make
that trade safely: isolation to localize correctness bugs, hardware timestamps to make
performance claims you can defend, and structural fingerprints so that "I just tidied the code"
can't silently become "I changed the kernel." With those three, a hand-written kernel is as
verifiable as an autotuned one.
