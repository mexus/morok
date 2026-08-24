# Tinygrad Compiler Adoption Plan

## Purpose

This file is the durable execution record for replacing Svod's mixed-generation
compiler with the compiler and UOp architecture from Tinygrad commit
`8c8b43de62515abe6c820b1de5aa26b30f48e43a`.

The migration ends at Tinygrad's final `PROGRAM` representation. Svod keeps its
Rust runtime and backend infrastructure below the `PROGRAM -> ExecutionPlan`
boundary.

This plan is intentionally detailed because the work will span multiple context
compactions. Update the status tables and checkpoint section at the end of every
coherent migration slice.

## Pinned Baseline

| Item | Value |
|---|---|
| Working branch | `compiler/tinygrad-uop-adoption` |
| Target Tinygrad checkout | `submodules/new_new_tinygrad` |
| Target Tinygrad commit | `8c8b43de62515abe6c820b1de5aa26b30f48e43a` |
| Previous Svod Tinygrad baseline | `1f8b24a6b95148202946e4e787030842ef6e4359` |
| Former failing workload | Whisper `decoder_slots=5` on AMD `gfx1151`; memory-safe, functional acceptance open |
| Current canonical mismatch ledger | `scripts/CANONICAL_KNOWN_GAPS.txt` |
| Isolated audio input | `/tmp/opencode/whisper-30-60.wav` |

Before using the reference checkout, verify the pin:

```bash
git -C submodules/new_new_tinygrad rev-parse HEAD
```

Do not silently update the target commit during this migration. A target update
requires a plan change that identifies the new commit and audits all already
ported phases for relevant upstream changes.

## Architectural Contract

### Adopt From Tinygrad

The compiler side must adopt the target revision's behavior and ordering for:

- Weak scalar dtypes, promotion, and dtype commitment.
- `dtype_from_uop` and source-sensitive dtype reconstruction.
- Canonical bool `Invalid` constants and polymorphic dtype matching.
- Scalar UOp storage with shaped values represented by `STACK` and shape data.
- Current pointer, image, parameter, memory, `INDEX`, `LOAD`, and `STORE` forms.
- Shared, tensor, kernel-graph, and program specs.
- Symbolic simplification, invalid propagation, and movement rewrites.
- Graph-native scheduling, rangeify, memory planning, and callification.
- Current optimizer ordering and post-range transforms.
- Validity-aware memory coalescing and late gate materialization.
- Devectorization, expansion, GPU dimension lowering, and linearization.
- Final `PROGRAM` graph structure and renderer-facing contracts.

### Retain From Svod

The following are not replaced by Tinygrad Python runtime implementations:

- `Arc<UOp>` ownership and hash consing.
- Rust typed errors where an operation can fail recoverably.
- AMD driver, queues, allocators, virtual memory, graph capture, and profiling.
- LLVM and C renderer/backend implementations.
- Runtime kernel cache, dispatch, JIT loading, and `ExecutionPlan` execution.
- Svod tensor/model APIs unless a compiler contract requires an adapter.
- Svod-specific architecture optimizations after parity is established.

### Prohibited Transitional End State

The migration is not complete if any of these remain in the compiler pipeline:

- Typed `Invalid` nodes or a dedicated `Op::Invalid`.
- Compiler dependence on legacy `DType::Index` semantics.
- Compiler vectors represented only by `DType::Vector` instead of shaped UOps.
- Gates stored on old `INDEX` nodes after the late-gater cutover.
- The combined legacy devectorization/gating pass as the authoritative path.
- A pass order assembled from mixed Tinygrad generations.
- Silent fallback to an old dtype when current Tinygrad has a production rule.
- Runtime or backend changes that bypass malformed compiler output instead of
  correcting the compiler representation.

## Definition Of Done

The adoption is complete only when all statements below are true:

1. Svod can serialize equivalent UOp stages from both implementations into a
   canonical form and compare them without allocation IDs or Python/Rust syntax.
2. Shared, tensor, kernel-graph, and program specs agree on the parity fixture
   corpus.
3. Current Tinygrad pass order is represented explicitly in Svod and no legacy
   pass runs in parallel with its replacement.
4. A logical `M=5` padded WMMA kernel compiles without an out-of-bounds address,
   retains validity through coalescing, and materializes correct late gates.
5. Final program IR contains no weak dtype, `Invalid`, legacy index gate,
   `PTRCAT`, unresolved shaped vector form, or unsupported movement op.
6. Existing CPU, LLVM, C, and AMD compiler tests pass.
7. Whisper `decoder_slots=5` runs on `gfx1151` without a memory fault and produces
   a valid transcript. Search parity uses identical model, language, strategy,
   beam width, fallback, and audio settings: greedy compares slots 1 and 5;
   beam-5 compares capacity 5 with a larger capacity. Empty or degenerate
   transcripts on a model appropriate for the input are functional failures.
8. Long-audio timestamp seek behavior is either implemented and tested or
   tracked as a separate model-layer task; it must not be conflated with compiler
   acceptance.

## Current Status

| Phase | State | Notes |
|---|---|---|
| 0. Reference and parity infrastructure | Complete for cutover | Canonical Rust/Python serializers, named tensor/rangeify/kernel/optimizer/program capture, three strict scalar fixtures, and two asserted memory-divergence fixtures are implemented. Add fixtures as each later phase needs them. |
| 1. Dtype and canonical Invalid foundation | Complete | Weak commitment, reduced-precision constants, Invalid, float bounds, and structural ordering are covered. |
| 2. Scalar shape and memory representation | Complete | Structured parameters, scalar/empty/multi shape arguments, and buffer ownership are implemented. |
| 3. UOp operation set and specs | Complete | Kernel-graph and final PROGRAM boundaries reject unsupported and legacy forms with typed errors. |
| 4. Symbolic and movement parity | Complete | Direct fixtures and every production stage from `tensor` through `linearized` strictly match the pinned reference; the mismatch manifest is empty. |
| 5. Graph scheduling and rangeify | Complete for supported subset | Pre-rangeify multi placement, tensor-form REDUCE, DEVICE splitting, host-staged SUM/MAX all-reduce, and non-overridable CALL lane binding are executable on host/mock. |
| 6. Optimizer and post-range | Complete for production corpus | Renderer semantics, optimizer defaults, extent-only names, and cache identity match the pinned production fixture. |
| 7. Late memory pipeline | Complete for production corpus | Expanded, coalesced, gated, PROGRAM, and linearized captures are strict matches. |
| 8. Renderer and runtime boundary | Complete on host | C/LLVM/AMD semantic fixes, persistent objects, and host/mock HCQ execution are implemented; hardware gates remain. |
| 9. Svod extension restoration | In progress | Planner modes and overlap instrumentation are restored conservatively; hardware measurements remain. |
| 10. Compile-only and GPU acceptance | Complete | CPU and AMD ONNX acceptance passed, including generated nodes and light models. |
| 11. Whisper and long-audio acceptance | In progress | The `gfx1151` workload is memory-safe and produces non-repeating text after the Tinygrad-exact late-gater `vconst_like` fix. Meaningful functional acceptance is pending medium/large-v3 runs on the Russian input; earlier tiny-model output is not evidence of a decoder-slot defect. |

## Work Protocol

### Slice Size

Each implementation slice must be independently reviewable and leave the
workspace buildable. Prefer one representation cutover or one pass family per
commit. Do not combine backend refactoring with compiler parity work unless the
new compiler contract makes the adapter unavoidable.

### Porting Method

For every Tinygrad function or pattern family:

1. Read the complete target file and its direct helpers before editing Rust.
2. Record Python-to-Rust operation and argument mappings in tests or this plan.
3. Port semantics and pass ordering, not Python syntax.
4. Preserve `Arc<UOp>` hash-cons identity and typed Rust errors.
5. Add canonical parity fixtures before deleting the old implementation.
6. Switch the production call site atomically to the new implementation.
7. Delete the superseded path in the same slice when practical.
8. Run focused tests, then crate tests, then workspace checks.

### No Compatibility By Default

Do not retain old compiler representations merely to keep internal callers
unchanged. Adapt callers as part of the cutover. Compatibility code is allowed
only for persisted data, public external consumers, or a documented runtime
boundary.

### GPU Safety Gate

Do not execute the known padded WMMA or Whisper slots-5 workload until all of the
following compile-only checks pass:

- Every padded lane has an explicit validity path in canonical stage output.
- Coalescing does not widen an access beyond the validity represented by the
  original lanes.
- Late gating applies to each resulting `LOAD` and `STORE` as required.
- Final linearized addresses for logical rows `5..15` cannot reach backing
  storage for logical `M=5`.
- Program spec and renderer preflight both pass.

The machine may need a reboot after prior GPU faults. Reboot only after a
candidate passes these structural gates.

## Canonical Parity Infrastructure

### Canonical Serialization Format

Create a serializer usable at named compiler stages. It must produce stable text
or JSON with:

- Nodes in deterministic topological order.
- Stable local node numbers instead of Rust IDs or Python object IDs.
- Operation name.
- Canonical dtype including weak/scalar/pointer/image metadata.
- Canonical shape and axis metadata.
- Canonical argument data with sorted maps and deterministic tuples.
- Source node numbers in source order.
- Buffer/parameter slots without physical addresses.
- Kernel and renderer metadata required to explain generated memory access.
- No provenance paths, allocation addresses, hash values, or debug-only tags by
  default.

Provide an optional verbose mode for provenance and backend metadata, but never
use verbose output as the parity oracle.

### Required Stage Names

Both implementations should emit at least these stages:

| Stage | Meaning |
|---|---|
| `tensor` | Tensor UOp graph before scheduling. |
| `scheduled` | Graph-native scheduled kernel graph. |
| `rangeified` | Ranges and memory operations made explicit. |
| `kernel_ast` | Kernel selected and callified. |
| `optimized` | Kernel optimization complete. |
| `postrange` | PADTO and other post-range transforms complete. |
| `expanded` | Expansion/devectorization prerequisites complete. |
| `coalesced` | Memory accesses coalesced with validity preserved. |
| `gated` | Late load/store gates materialized. |
| `linearized` | Ordered renderer input. |
| `program` | Final program graph and launch metadata. |

### Initial Fixture Corpus

Add small fixtures before broad model captures:

- Scalar weak-int plus strong integer promotion.
- Weak-float transcendental commitment.
- Bool comparison and `WHERE` with canonical Invalid.
- Scalar and shaped-vector load/store.
- Gated scalar load with alternate value.
- Gated vector load with mixed valid lanes.
- Reduction with padded extent.
- Logical `M=5` matrix multiply padded to WMMA tile size.
- Local/shared-memory staging around WMMA.
- Multi-output kernel and callified kernel graph.
- Variable-bound range and launch dimensions.

Store small expected fixtures in the relevant Rust test directories. Store large
diagnostic dumps under `/tmp/opencode`; do not commit model-sized dumps.

## Source Mapping

| Tinygrad target | Primary Svod destination | Migration purpose |
|---|---|---|
| `tinygrad/dtype.py` | `dtype/src/lib.rs`, `dtype/src/cast.rs` | Dtype identity, weakness, promotion, pointers, images. |
| `tinygrad/uop/ops.py` | `ir/src/op.rs`, `ir/src/types.rs`, `ir/src/uop/`, `ir/src/dtype_rule.rs` | UOp model, source layout, dtype derivation, reconstruction. |
| `tinygrad/uop/weak.py` | `ir/` and `schedule/src/symbolic/` | Weak index lowering and weak commitment. |
| `tinygrad/uop/spec.py` | `schedule/src/spec.rs` | Shared/tensor/kernel/program invariants. |
| `tinygrad/uop/symbolic.py` | `schedule/src/symbolic/` | Symbolic and Invalid rewrites. |
| `tinygrad/uop/movement.py` | `ir/src/uop/constructors/shape.rs`, schedule rewrites | Movement and shape semantics. |
| `tinygrad/schedule/__init__.py` | `schedule/src/rangeify/`, scheduler entry points | Graph-native schedule construction. |
| `tinygrad/schedule/rangeify.py` | `schedule/src/rangeify/` | Rangeification and kernel formation. |
| `tinygrad/schedule/indexing.py` | `schedule/src/rangeify/indexing.rs` | Current index and memory lowering. |
| `tinygrad/schedule/memory.py` | `schedule/src/rangeify/`, runtime adapter | Bufferization and memory planning. |
| `tinygrad/codegen/__init__.py` | `schedule/src/optimizer/mod.rs`, `codegen/src/program_pipeline.rs` | Authoritative compiler pass order. |
| `tinygrad/codegen/opt/__init__.py` | `schedule/src/optimizer/` | Kernel optimizer and axis model. |
| `tinygrad/codegen/opt/tc.py` | `schedule/src/optimizer/tc.rs` | Tensor-core selection and lowering. |
| `tinygrad/codegen/opt/postrange.py` | `schedule/src/optimizer/opts.rs` or new focused module | PADTO and post-range behavior. |
| `tinygrad/codegen/gpudims.py` | `schedule/src/gpudims.rs` | Global/local dimension lowering. |
| `tinygrad/codegen/late/coalesce.py` | New focused schedule late module | Validity-aware memory coalescing. |
| `tinygrad/codegen/late/gater.py` | New focused schedule late module | Late `LOAD`/`STORE` gate creation. |
| `tinygrad/codegen/late/linearizer.py` | `schedule/src/linearize/` | Final control/data ordering. |
| `tinygrad/renderer/cstyle.py` | `codegen/src/c/` | Renderer contract adaptation. |
| `tinygrad/renderer/llvmir.py` | `codegen/src/llvm/` | LLVM renderer contract adaptation. |
| `tinygrad/engine/realize.py` | `runtime/src/execution_plan.rs`, dispatch adapter | Final program-to-runtime boundary only. |

## Phase 0: Reference And Parity Harness

### Objectives

Make semantic differences observable before replacing foundational
representations.

### Tasks

- Add the target pin and Python-to-Rust mapping to repository documentation.
- Implement canonical Rust stage serialization.
- Add a reference-side Python serializer at the pinned checkout or in a small
  external harness under `/tmp/opencode`.
- Add fixture runners that compile the same logical graph in Python and Rust.
- Normalize operation naming differences explicitly in one mapping table.
- Add structural diff output that reports the first differing node and traces
  its source dependencies.
- Capture the logical `M=5` padded matmul stages without executing a GPU kernel.
- Record exact commands for regenerating each fixture.

### Exit Gates

- At least five scalar/memory fixtures serialize deterministically across two
  runs.
- A deliberately changed dtype or source order produces a focused diff.
- The padded `M=5` fixture reaches the current Svod post-range stage without GPU
  execution and can be archived under `/tmp/opencode`.

## Phase 1: Dtype, Dtype Production, And Invalid

### Completed In Initial Checkpoint

- Added `WeakInt` and `WeakFloat` scalar variants.
- Ported the target promotion lattice and lossless-cast behavior.
- Added strong/weak conversion and `least_upper_float` helpers.
- Added centralized `dtype_from_op` production rules.
- Re-derived result dtype when rewritten sources change dtype.
- Replaced dedicated/typed Invalid operations with canonical
  `CONST(Invalid): bool`.
- Made Invalid polymorphic for ALU, `WHERE`, vector lanes, index consumers, and
  shared specs.
- Rejected weak dtypes and surviving Invalid constants in renderers.
- Added focused parity and canonical identity tests.

### Remaining Tasks

- Compare every target `dtype_from_uop` arm with `dtype_from_op`; remove `None`
  cases as operation metadata is modernized.
- Port `tinygrad/uop/weak.py`, including weak index lowering and commitment pass
  placement.
- Remove transitional assumptions that materialized constants begin as Rust
  native strong dtypes when Tinygrad keeps them weak.
- Replace `DType::Index` with the target integer/index lowering semantics after
  shape and memory representation is ready.
- Audit serialization discriminants if dtypes cross any persisted boundary.

### Exit Gates

- Target weak-dtype unit fixture corpus passes in Rust.
- No renderer receives a weak dtype.
- No UOp constructor creates typed Invalid.
- No `Op::Invalid` reference exists.
- `dtype_from_op` has an explicit documented reason for every remaining `None`.

## Phase 2: Scalar Shape And Memory Representation

### Rationale

This is the blocking cutover. Direct ports of modern Tinygrad passes are unsafe
while Svod still encodes vectors, pointers, index values, and memory gates using
older forms.

### Tasks

- Model current scalar dtype identity separately from UOp shape.
- Add the target shaped-value representation and `STACK` semantics.
- Replace compiler use of `DType::Vector`; keep backend vector types only in the
  renderer adapter if required.
- Port current pointer and image dtype metadata.
- Port structured parameter metadata and operation arguments.
- Change `PARAM`, `BUFFER`, local memory, and register definitions to target
  source/argument layouts.
- Port current `INDEX` representation.
- Port current `LOAD` and `STORE` source layouts, including alternate values,
  dependencies, and memory metadata.
- Remove the dedicated gate field from `INDEX`; validity is encoded as
  `WHERE(index, Invalid)` until the late gater materializes `LOAD`/`STORE` gate
  operands.
- Replace `VECTORIZE`, `CAT`, `PTRCAT`, and legacy pointer-vector assumptions
  according to target `STACK` and shaped memory rules.
- Update hashing, equality, reconstruction, children traversal, shape inference,
  pretty printing, serializers, and property generators atomically.
- Add adapters at renderer boundaries rather than preserving old compiler forms.

### Tests

- Scalar, vector, pointer, and image construction parity.
- Hash-cons identity after source reconstruction.
- `STACK` shape and lane extraction.
- Scalar and vector load/store source ordering.
- Parameter and local-memory metadata round trips.
- Invalid lanes in shaped values.
- Compile-fail/spec tests for old index gates and old vector dtypes.

### Exit Gates

- Compiler passes no longer inspect `DType::Vector` or legacy pointer lane count.
- `INDEX` has no gate field.
- All memory UOps serialize with target-compatible source ordering.
- Workspace builds with old memory constructors removed or unreachable.
- Phase 0 fixtures agree through the tensor stage.

## Phase 3: UOp Operation Set And Specs

### Tasks

- Audit the complete target `Ops` enum against `ir/src/op.rs`.
- Add missing operations and remove superseded operations as one schema cutover.
- Align operation groups used by pattern matching.
- Align `src`, `arg`, tuple/function/call, program, and control-flow semantics.
- Complete `dtype_from_op` for the modernized operation set.
- Port `spec_shared`, `spec_tensor`, kernel-graph specs, and `spec_program` in
  target order.
- Preserve Svod-only operations only behind explicit extension spec rules.
- Ensure `matches_dtype` follows canonical Invalid and shape behavior.
- Run specs at the same compiler boundaries as Tinygrad.
- Improve spec errors with node serialization and dependency context.

### Exit Gates

- Every operation is accepted or rejected by the corresponding target spec.
- No broad structural whitelist hides missing dtype/source invariants.
- Malformed memory, shape, call, and program fixtures fail at the expected stage.
- Phase 0 fixtures agree through shared and tensor specs.

## Phase 4: Weak, Symbolic, And Movement Rewrites

### Tasks

- Port weak commitment and weak index lowering in target order.
- Port target symbolic pattern groups as separate named matchers.
- Port canonical Invalid poisoning and gate normalization exactly.
- Port comparison, div/mod, range, and validity simplification behavior.
- Port movement and shape rewrites against shaped UOps.
- Remove transitional typed-Invalid and early-gate cleanup patterns.
- Ensure rewrite reconstruction always uses modern dtype production.
- Compare fixed-point behavior and matcher ordering with target Tinygrad.
- Add iteration-limit diagnostics for rewrite cycles.

### Exit Gates

- Symbolic parity fixtures match canonical output.
- Invalid remains bool and survives only where target Tinygrad retains it.
- No early rewrite converts Invalid into a typed zero before the target cleanup.
- Movement operations satisfy tensor specs before scheduling.

## Phase 5: Graph-Native Scheduling, Rangeify, And Callification

### Tasks

- Port target schedule construction from `tinygrad/schedule/__init__.py`.
- Port memory scheduling and bufferization from `schedule/memory.py`.
- Port current indexing and rangeify representation.
- Port multi-device/all-reduce scheduling only where Svod supports the same
  semantics; otherwise reject explicitly.
- Port kernel extraction and graph-native dependencies.
- Port tuple/function/call construction and callification.
- Preserve Svod allocator/runtime objects outside the compiler graph.
- Remove legacy schedule forms when their target replacement reaches parity.
- Add deterministic kernel ordering and serializer coverage.

### Exit Gates

- Tensor-to-kernel fixtures agree through every captured stage from `tensor` to
  `linearized`.
- Multi-output and callified fixtures pass kernel-graph specs.
- No runtime allocation address participates in compiler graph identity.
- Existing model compile tests reach kernel optimization through the new path.

## Phase 6: Optimizer And Post-Range Pipeline

### Authoritative Order

Use `tinygrad/codegen/__init__.py` at the pinned commit as the source of truth.
Represent order explicitly in one Rust orchestration function and add an order
test. Do not infer ordering from module names or previous Svod behavior.

### Tasks

- Port optimizer axis model and option validation.
- Port heuristic option selection.
- Port tensor-core eligibility and lowering from `opt/tc.py`.
- Port expansion, contraction, unroll, local/group, and upcast behavior.
- Port post-range patterns from `opt/postrange.py`.
- Replace current PADTO implementation with target behavior on modern memory IR.
- Verify non-ADD reduction padding behavior via validity, not old unsafe-op bans.
- Remove current investigative PADTO code once parity replacement is active.
- Add optimizer state serialization before and after each option.

### Padded WMMA Assertions

- Logical `M=5` may be physically padded to a supported tile.
- Padded compute lanes must carry Invalid/validity according to target semantics.
- Store-side validity must exclude rows beyond logical `M`.
- Load-side validity must survive tensor-core staging and later coalescing.
- Work expansion limit must match the pinned Tinygrad condition exactly.

### Exit Gates

- Optimizer option traces match Tinygrad for the fixture corpus.
- Post-range canonical graphs match for logical `M=5`.
- No physical address is generated or executed during this phase.

## Phase 7: Late Memory Pipeline

### 7A. Expansion And Devectorization

- Split the legacy combined `schedule/src/devectorize.rs` responsibilities.
- Port target expansion/devectorization prerequisites in target order.
- Eliminate transitional `PTRCAT`/`CAT` forms at the same stage as Tinygrad.
- Preserve shaped lane validity rather than introducing index gates.

### 7B. Coalescing

- Port `tinygrad/codegen/late/coalesce.py` directly against modern memory UOps.
- Treat value and validity lanes as one transformation unit.
- Add fixtures for contiguous lanes, gaps, mixed validity, alternate values,
  alignment boundaries, and local/global address spaces.
- Assert that widening cannot create an access outside the union of valid source
  lanes unless the widened access itself is safely gated.

### 7C. Late Gating

- Port `tinygrad/codegen/late/gater.py` after coalescing.
- Materialize `LOAD` alternate values and `STORE` predicates from validity data.
- Normalize conditions only after memory grouping is final.
- Reject any remaining canonical Invalid at program verification.

### 7D. Linearization

- Port `tinygrad/codegen/late/linearizer.py` control and data ordering.
- Align range endings, barriers, IF/ENDIF, accumulator placement, and dependency
  ordering.
- Remove the old linearizer path once parity fixtures pass.

### Exit Gates

- Canonical output agrees at `expanded`, `coalesced`, `gated`, and `linearized`.
- Every memory operation has valid final address/gate/alternate semantics.
- Final program spec rejects all unresolved Invalid or shaped intermediate forms.
- Logical `M=5` address analysis proves no access at `+0x12000`, `+0x1b000`, or
  any other padded-tail offset beyond allocation bounds.

## Phase 8: Renderer And Runtime Boundary

### Tasks

- Define one Rust `PROGRAM -> ExecutionPlan` adapter.
- Map modern parameters, buffers, launch dimensions, globals, locals, and binary
  metadata into existing Svod runtime types.
- Update C, LLVM, and AMD renderers to consume final program contracts.
- Keep renderer-specific vector and pointer types inside renderer lowering.
- Keep compiler UOps independent of runtime physical addresses.
- Preserve kernel cache keys while versioning them for the new canonical program
  representation.
- Preserve typed runtime errors and backend diagnostics.
- Add renderer preflight that runs `spec_program` before code generation.

### Exit Gates

- Equivalent final program fixtures render deterministically.
- CPU and AMD backends consume the same compiler-level `PROGRAM` graph.
- Existing allocator, graph capture, profiling, and dispatch tests pass unchanged
  except for explicit adapter updates.
- No backend contains a PADTO/Invalid workaround.

## Phase 9: Restore Svod Extensions

Restore extensions one at a time after target parity, each with before/after
canonical fixtures:

- AMD-specific WMMA/MFMA selection and metadata.
- LLVM backend-only operations.
- CPU queue and thread dispatch behavior.
- Z3 verification hooks.
- Svod graph capture and profiling metadata.
- Architecture-level fused kernels and model-specific optimizations.

An extension must not alter shared Tinygrad semantics. If it does, document the
intentional divergence, add a focused test, and keep the divergence at a named
extension boundary.

## Phase 10: Validation And GPU Acceptance

### Static And Unit Validation

Run after each coherent slice:

```bash
cargo fmt --all -- --check
git diff --check
cargo check --workspace --all-targets
cargo test -p svod-dtype
cargo test -p svod-ir --lib
cargo test -p svod-schedule --lib
cargo test -p svod-codegen --lib
```

Run broader feature suites at phase boundaries where dependencies are available:

```bash
cargo test --workspace
cargo test --workspace --features proptest
cargo test --workspace --features z3,proptest
```

### Compile-Only Padded WMMA Validation

1. Compile logical `M=5` without submitting a queue packet.
2. Capture all canonical stages.
3. Diff against pinned Tinygrad.
4. Inspect final scalar addresses and gates.
5. Confirm allocation extents and element widths.
6. Confirm no padded tail can access physical memory.
7. Archive the passing dump under `/tmp/opencode` with target and Svod commit IDs.

### GPU Validation

Only after compile-only validation:

1. Reboot if the AMD device remains affected by prior memory faults.
2. Run a minimal deterministic padded WMMA kernel.
3. Synchronize immediately and inspect driver/runtime errors.
4. Compare output to CPU/reference results.
5. Run repeated iterations with allocator reuse.
6. Run greedy Whisper with `decoder_slots=1` as the compiler/memory baseline.
7. Repeat the identical greedy configuration with `decoder_slots=5`.
8. Run beam-5 with `decoder_slots=5`, then repeat it with a larger capacity;
   never compare beam-1 or greedy output to beam-5 as slot parity evidence.
9. Compare tokens/logits or transcript and timing on medium or large-v3 for the
   Russian input, keeping language, fallback, and every decode option identical.
10. Run graph capture and profiling variants.

Stop GPU testing immediately on a memory fault. Preserve compiler dumps, launch
metadata, allocation sizes, and fault offsets before rebooting.

## Phase 11: Whisper And Long-Audio Acceptance

### Compiler Acceptance

- `decoder_slots=5` must compile and run without out-of-bounds faults.
- Slot batching must preserve output ordering and cache ownership.
- Mixed precision and cross-attention cache behavior must remain correct.
- Repeated runs must not depend on stale allocator contents.

### Separate Model-Layer Work

Long audio still requires Whisper behavior beyond compiler correctness:

- Timestamp-driven seek advancement.
- Replay of unfinished tails.
- Prompt/token carry-over between windows.
- Correct no-speech and timestamp boundary handling.

Track this separately after slots-5 compiler acceptance. Do not weaken compiler
validation to make long-form inference appear functional.

## Risk Register

| Risk | Signal | Mitigation |
|---|---|---|
| Mixed old/new memory forms | Repeated adapter branches or typed Invalid reappears | Stop and complete phase 2 atomically. |
| Incorrect pass order | Individual tests pass but parity diverges late | Add explicit pass-order snapshot and stage fixtures. |
| Hash-cons reconstruction error | Equivalent UOps fail pointer identity or stale dtype persists | Test reconstruction and dtype derivation on every schema cutover. |
| Invalid removed too early | PADTO tail becomes an unconditional memory access | Track validity in canonical stage diffs through late gater. |
| Coalescer widens beyond gate | Fault offset aligns with padded row/tile | Compare lane validity before and after every coalesce rewrite. |
| Renderer hides malformed IR | Backend-specific workaround appears | Enforce `spec_program` before every renderer. |
| Runtime regression during compiler work | Allocator/dispatch tests change unexpectedly | Keep one narrow `PROGRAM -> ExecutionPlan` adapter. |
| Target checkout drifts | Reference output changes without Svod edit | Verify exact target commit before parity generation. |
| Large model dumps enter git | Repository size grows or fixtures become unstable | Keep large captures under `/tmp/opencode`. |
| GPU remains poisoned after fault | Unrelated kernels fail after compiler fix | Complete compile-only validation, then reboot once. |

## Commit Strategy

Recommended checkpoint commits:

1. `compiler: add weak dtypes and canonical invalid`
2. `test: add tinygrad stage parity harness`
3. `ir: adopt shaped uops and current memory forms`
4. `schedule: port current specs and symbolic pipeline`
5. `schedule: adopt graph-native rangeify and callification`
6. `schedule: port current optimizer and postrange pipeline`
7. `schedule: port late coalescing and gating`
8. `schedule: port current linearizer`
9. `codegen: adapt renderers to current program uops`
10. `runtime: adapt current program to execution plan`
11. `compiler: restore svod backend extensions`
12. `test: validate padded wmma and whisper slots`

Do not commit reference checkouts, audio files, GPU dumps, or unrelated worktree
changes with these checkpoints.

## Compaction Handoff Template

At the end of every substantial slice, update this section or provide an
equivalent conversation checkpoint containing:

```text
Target Tinygrad commit:
Current Svod branch and commit:
Completed phase/task:
Files changed:
Behavioral decisions made:
Parity fixtures added/updated:
Commands run and results:
Known failures:
Large artifacts under /tmp/opencode:
Worktree changes intentionally not owned:
Exact next task:
GPU safe to run: yes/no, with reason:
```

## Latest Checkpoint

Target Tinygrad commit:
`8c8b43de62515abe6c820b1de5aa26b30f48e43a`

Current branch:
`compiler/tinygrad-uop-adoption`

Completed:
Host-closeable IR, spec, codegen, HCQ lifecycle, object-cache, planner,
pre-rangeify multi-device, and host-staged SUM/MAX all-reduce remediation is
committed. Tensor-form REDUCE matches the pinned direct representation, aligned
MSTACK CALLs execute per lane with fixed `_device_num`, and the canonical
mismatch manifest is empty. Exact compiler identity flows through schedule,
optimized-kernel, and persistent object caches. AMD
publication, drain failure, allocation ownership, and teardown are covered by a
scripted mock fault matrix. Canonical production parity is strict through every
captured stage from `tensor` to `linearized`.

Validated:

- `svod-ir`: 587 tests passed in the latest full library run.
- `svod-schedule`: 1,173 passed and 4 ignored after tensor-form REDUCE and all-reduce integration.
- `svod-codegen`: 125 tests passed; program pipeline 41 passed after persistent-object integration.
- `svod-runtime`: 103 tests passed, including validated host all-reduce, corruption recovery, eviction, and fresh-process object reuse.
- `svod-device`: 215 host-only tests passed and 7 ignored; all targets check passed.
- `svod-tensor`: full host-only library suite passed with 1,451 tests and 16 ignored.
- Strict canonical evidence matches every production stage through `linearized`.
- DenseNet121: Clang and LLVM passed in release mode.
- Full serial release `svod-onnx`: 3,098 passed, 636 ignored, 0 failed.
- AMD hand-written ONNX references: 97 passed.
- AMD generated ONNX nodes: 1,361 passed, 317 ignored, 0 failed, serially.
- AMD light models: all 9 passed, including AlexNet, DenseNet121, ResNet50,
  and VGG19.
- The linked-HCQ AlexNet fault reproducer passes after the Tinygrad-aligned
  kernarg, SDMA fence, ring-size, and unread-space fixes.
- Bool Bitcast lowering and AMD floating-point flags now match the pinned
  Tinygrad paths; the previously failing Bitcast and Mish references pass.
- Whisper tiny on the isolated 30-second input is memory-safe on `gfx1151`.
  Late gating now matches pinned Tinygrad by using `vconst_like(0)` for shaped
  load alternates, producing a post-movement `STACK` instead of reintroducing
  `EXPAND` at the final PROGRAM boundary. The earlier tiny-model transcripts on
  Russian audio do not establish a decoder-slot correctness issue: the compared
  runs also changed decoding strategy/beam width, and tiny is not a meaningful
  quality oracle for this input. Functional acceptance is pending comparable
  medium and large-v3 results.
- `cargo check --workspace` and `cargo test --workspace --no-run` passed.
- `git diff --check` passed.
- `cargo fmt --all -- --check` passed.

Intentionally uncommitted artifacts:

- `submodules/new_new_tinygrad/` reference checkout.
- `submodules/transformers/` and `submodules/whisper.cpp/` reference checkouts.
- Audio files in the repository root.
- Missing legacy `submodules/tinygrad` worktree entry.

Exact next task:
Perform hardware-backed planner/lane measurements and multi-XCC/multi-GPU
safety acceptance. Model runs resume after those gates; long-audio seek remains
a separate model-layer task.

GPU safe to run:
PM4, SDMA, WMMA, profiling, lane, graph, and model validation are safe serially
on gfx1151. Forced AQL must remain isolated with `SVOD_AMD_HW_QUEUES=1` and one
test thread; multi-XCC and physical multi-GPU acceptance require suitable
hardware.
