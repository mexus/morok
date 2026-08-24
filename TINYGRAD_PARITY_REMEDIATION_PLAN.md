# Tinygrad Parity Remediation Plan

## Purpose

This document is the action plan for eliminating the correctness, safety,
coverage, and performance gaps found during the audit of Svod's compiler and
runtime migration to Tinygrad commit
`8c8b43de62515abe6c820b1de5aa26b30f48e43a`.

The historical migration record remains in
`TINYGRAD_COMPILER_MIGRATION_PLAN.md`. This file is intentionally separate: it
is an executable remediation backlog with dependencies, implementation steps,
tests, and acceptance gates.

The reference checkout is:

```text
submodules/new_new_tinygrad
```

Before generating or accepting parity evidence, verify the pin:

```bash
git -C submodules/new_new_tinygrad rev-parse HEAD
```

Expected output:

```text
8c8b43de62515abe6c820b1de5aa26b30f48e43a
```

## Current Readiness

The host-closeable weak-dtype, canonical Invalid, shaped STACK, structured
PARAM, rangeify, codegen, object-cache, planner, and HCQ lifecycle work is
implemented. The default core crates compile and their focused suites pass.

The host remediation and ACCEPT-01 matrix are complete. The overall migration
is not ready to be declared complete because:

- Canonical direct and production corpora are strict with an empty mismatch
  manifest.
- Supported multi-device CALLs expand aligned lanes and bind a non-overridable
  per-lane `_device_num`; public Tensor ownership still represents one output
  buffer and cannot publish a sharded output directly.
- Multi-XCC AQL and physical multi-GPU replay are implemented but still require
  validation on suitable hardware.
- Hardware-backed PERF-04 overlap and memory measurements, then model
  acceptance, remain outstanding.

No GPU acceptance result may replace a failing structural gate. Hardware tests
must be the final validation step, not the debugging mechanism for malformed
IR or command streams.

## Definition Of Done

The remediation is complete only when all of the following are true:

- The Tinygrad checkout is still at the pinned commit.
- Canonical fixtures execute equivalent production stages in both
  implementations and compare schema-compatible output.
- Canonical PROGRAM output includes all semantic `ProgramInfo` fields.
- Reduced-precision constants are committed to the same dtype grids as
  Tinygrad before folding or rendering.
- Every final PARAM has a unique canonical slot and matches Tinygrad ordering.
- Shared, tensor, kernel-graph, and program specs run at equivalent boundaries.
- Supported multi-device graphs run `multi_pm` before rangeification and lower
  reduced shard axes to executable all-reduce work.
- Unsupported multi-device forms fail with typed errors at the earliest
  boundary instead of surviving to codegen.
- C, LLVM, and AMD LLVM pass targeted semantic and compile tests.
- RDNA int8 WMMA and CDNA3 FP8 behavior match the target capability model.
- Final PROGRAM graphs contain no weak dtype, canonical Invalid, legacy Index,
  or unsupported movement form.
- Multi-XCC AQL completion uses a validated terminal completion discipline.
- Native linked replay proves ownership/accessibility for every kernel and copy
  endpoint.
- Failed queue drains never unmap allocations that hardware might still use.
- Queue publication cannot leave an unaccounted timeline reservation.
- Compiler and BEAM caches include every behavior-affecting identity.
- The padded `M=5` compile-only test reaches final generated code and proves all
  enabled addresses are in bounds.
- Required host suites pass, followed by single-XCC, multi-XCC, and multi-GPU
  hardware suites where applicable.

## Work Rules

1. Keep each semantic fix independently reviewable.
2. Add or repair the failing parity fixture before changing production code.
3. Port semantics and pass placement from the pinned source, not from current
   Tinygrad main or memory.
4. Do not hide malformed compiler output in a renderer or runtime adapter.
5. Prefer typed errors at public boundaries over assertions and panics.
6. Do not retain old behavior merely to keep a stale test passing.
7. Run focused tests first, then crate tests, then workspace checks.
8. Do not run known-risk GPU tests until their compile-only and packet-level
   gates pass.
9. Record every intentional divergence with a focused test and a short rationale
   in this document.
10. Update the status table after every merged remediation slice.

## Status And Dependency Table

| ID | Priority | Area | Depends On | State |
|---|---|---|---|---|
| EVID-01A | P0 | Canonical schema and operational stage harness | None | Complete |
| EVID-01B | P0 | Strict production-stage and corpus parity | All compiler items | Complete |
| EVID-02 | P0 | Padded M=5 acceptance fixture | EVID-01A | Complete |
| IR-01 | P0 | Reduced-precision constant commitment | EVID-01A | Complete |
| IR-02 | P0 | PARAM slot allocation | EVID-01A | Complete |
| IR-03 | P1 | FUNCTION result-shape substitution | EVID-01A | Complete |
| IR-04 | P1 | Float analysis bounds | IR-01 | Complete |
| IR-05 | P1 | Structural commutative ordering | EVID-01A | Complete |
| SPEC-01 | P1 | Kernel-graph specification boundary | EVID-01A | Complete |
| SPEC-02 | P1 | Final legacy Index rejection | SPEC-01 | Complete |
| MULTI-01 | P0 | Pre-rangeify multi-device rewrite placement | EVID-01A | Complete |
| MULTI-02 | P0 | Sharded reduction and all-reduce lowering | MULTI-01 | Complete on host; hardware collective optimization excluded |
| MULTI-03 | P0 | DEVICE range kernel splitting | MULTI-01 | Complete on host/mock; physical lanes pending |
| MULTI-04 | P1 | Unsupported multi-device rejection | MULTI-01 | Complete |
| CG-01 | P0 | RDNA int8 WMMA ABI packing | IR-01 | Complete |
| CG-02 | P1 | LLVM floating comparison semantics | IR-01 | Complete |
| CG-03 | P1 | CDNA3 ordinary FP8 capability | IR-01 | Complete; hardware execution pending |
| CG-04 | P1 | Volatile memory rendering | SPEC-01 | Complete |
| CG-05 | P1 | Fast integer division default and control | EVID-01A | Complete |
| CG-06 | P1 | Expander cleanup parity | EVID-01A | Complete |
| CG-07 | P1 | Shift-amount dtype parity | EVID-01A | Complete |
| CG-08 | P1 | Renderer-safe generated kernel names | EVID-01A | Complete |
| CG-09 | P1 | Remove unsupported MLIR backend | None | Complete |
| HCQ-01 | P0 | Multi-XCC AQL completion | None | Implemented; hardware pending |
| HCQ-02 | P0 | Native copy endpoint ownership | None | Implemented; hardware pending |
| HCQ-03 | P0 | Failed-drain allocation lifetime | None | Complete |
| HCQ-04 | P1 | Transactional timeline publication | HCQ-01 | Complete |
| HCQ-05 | P1 | PM4/AQL graph size limits | HCQ-01 | Implemented; boundary stress pending |
| HCQ-06 | P1 | AMD allocation ownership and teardown | HCQ-03 | Complete |
| HCQ-07 | P1 | Executed multi-device lane topology | HCQ-02 | Complete on host/mock; hardware pending |
| PERF-01 | P1 | Optimizer defaults and cache identity | EVID-01A | Complete |
| PERF-02 | P1 | BEAM compile/search efficiency | PERF-01 | Complete on host; hardware benchmark pending |
| PERF-03 | P2 | Persistent compiled-object cache | PERF-01 | Complete |
| PERF-04 | P2 | Memory planner and execution overlap | HCQ-07 | Instrumented; hardware measurement pending |
| ACCEPT-01 | P0 | Host acceptance matrix | All compiler items | Complete |
| ACCEPT-02 | P0 | AMD hardware safety matrix | All HCQ items | Open |
| ACCEPT-03 | P1 | Model acceptance and performance | ACCEPT-01, ACCEPT-02 | Open |

## Phase 0: Restore Trustworthy Evidence

### EVID-01A: Make The Canonical Schema And Harness Operational

#### Problem

Rust emits canonical schema version 6 in `ir/src/uop/canonical.rs:15-25`, while
the reference serializer emits version 5 in
`scripts/tinygrad-canonical.py:96-121`. The shell driver compares whole JSON
documents in `scripts/check-canonical-parity.sh:8-18`, so it exits before
checking the first graph.

The current fixtures are hand-built UOps from
`ir/examples/canonical_fixture.rs:1-46` and
`scripts/tinygrad-canonical.py:124-146`. They do not execute the production
tensor, rangeify, kernel, optimizer, coalescing, gating, linearization, or
PROGRAM pipelines. The serializer also lacks complete pointer, image,
callable, kernel, and PROGRAM metadata mappings.

The required stage contract is documented in
`TINYGRAD_COMPILER_MIGRATION_PLAN.md:170-227`.

#### Tinygrad Sources

- `submodules/new_new_tinygrad/tinygrad/uop/ops.py:193-207`
- `submodules/new_new_tinygrad/tinygrad/uop/render.py:117-119`
- `submodules/new_new_tinygrad/tinygrad/schedule/rangeify.py`
- `submodules/new_new_tinygrad/tinygrad/codegen/__init__.py`
- `submodules/new_new_tinygrad/tinygrad/codegen/late/coalesce.py`
- `submodules/new_new_tinygrad/tinygrad/codegen/late/gater.py`
- `submodules/new_new_tinygrad/tinygrad/codegen/late/linearizer.py`

#### Actions

- Define one versioned canonical schema document shared by the Rust and Python
  serializers.
- Update the Python serializer to schema 6 rather than weakening the Rust
  serializer to schema 5.
- Serialize every semantic dtype component: scalar identity, weakness, vector
  count where still backend-only, pointer target, address space, size, image
  shape, and device identity.
- Serialize every semantic operation argument used by the target compiler,
  including `ParamArg`, range axis paths, stage metadata, WMMA metadata,
  `CallInfo`, `FunctionInfo`, `ProgramInfo`, sink metadata, and devices.
- Add explicit normalization only for known representation differences. Do not
  normalize away dtype, source ordering, shape, validity, ABI, launch, or target
  differences.
- Add a first-difference reporter that prints the stage, node, field, and source
  chain rather than dumping two complete JSON documents.
- Add production fixture entry points on both sides that run equivalent logical
  graphs through named stages.
- Keep allocation addresses, Rust UOp IDs, Python object IDs, and provenance out
  of the default comparison.
- Add a `--verbose` diagnostic mode for provenance and backend-only metadata.
- Fail immediately if the Tinygrad checkout does not match the pinned commit.

#### Required Fixtures

- Weak integer promotion.
- Weak float commitment.
- Canonical Invalid through WHERE.
- Scalar and shaped STACK.
- Structured PARAM and BUFFER.
- Scalar load/store.
- Gated scalar load with alternate.
- Mixed-validity shaped load.
- Symbolic FUNCTION result shape.
- Copy and all-reduce metadata.
- Multi-output callified graph.
- Variable-bound launch dimensions.
- Padded reduction.
- Padded `M=5` WMMA.
- Final PROGRAM with non-default name, target, global/local sizes, variables,
  globals, inputs, and outputs.

#### Exit Gate

```bash
CANONICAL_RECORD_KNOWN_GAPS=1 ./scripts/check-canonical-parity.sh
```

Evidence mode must pass twice in clean temporary directories, require the exact
checked-in mismatch manifest, and fail with a focused diagnostic when a
fixture's dtype, source order, gate, launch dimension, or expected-failure
signature is deliberately changed. Ordinary mode must remain a strict gate and
must fail while any production stage differs.

#### Current EVID-01A Evidence (2026-08-22)

The cumulative production runners, strict Rust capture path, schema validator,
aligned canonical diff, normalized full-document SHA-256 signatures, exact
expected-gap manifest, typed PAD normalization, strict local/shared WMMA
fixture, production multi-output callification capture, and rich PROGRAM fixture
are implemented. The scheduled boundary now serializes Tinygrad's actual LINEAR
and Svod's actual pre-schedule descriptors/invocations as a common ordered
schedule artifact, including complete ASTs, buffer/global and output slots,
dependencies, and bindings. Authored BUFFER slots are preserved without stage or
high-bit inference. Public constructors derive weak promotion, weak commitment,
Invalid-WHERE, and modern gated-LOAD dtypes/layouts. Symbolic FUNCTION shape
substitution is exact expected-failing evidence for IR-03; padded reduction is
exact expected-failing evidence for REDUCE-01 (`REDUCE_AXIS` versus pinned
`REDUCE`). The direct corpus and exact implemented/unsupported coverage are
documented in `scripts/CANONICAL_SCHEMA_V6.md`. EVID-01A is **Complete** after
focused self-tests, two clean evidence runs, strict-failure verification, crate
tests, schedule tests/example, workspace all-target check, formatting, diff, and
pinned-reference checks passed.

### EVID-01B: Close Strict Production-Stage And Corpus Parity

**Status: Complete.** The production multi-output fixture and every captured
stage from `tensor` through `linearized` compare strictly. The known-gap manifest
contains no production entry.

#### Problem

The operational harness exposed compiler differences in call bindings, schedule
slots, renderer capabilities, generated names, and GROUP shape. Those owning
boundaries are repaired rather than normalized away.

#### Actions

- Resolve the manifest entries through their owning compiler items; do not add
  canonical normalization for semantic dtype, source, validity, optimization,
  launch, ABI, or target differences.
- Promote each repaired production fixture/stage to strict parity by removing
  its exact manifest section.
- Close IR-03 and REDUCE-01 expected failures only when their exact fixtures
  pass without a divergent substitute mismatch.
- Keep `CANONICAL_RECORD_KNOWN_GAPS=1` as evidence recording, never as the
  ordinary CI success contract.

#### Exit Gate

```bash
./scripts/check-canonical-parity.sh
```

Ordinary mode must pass with no production mismatches and no production entries
in `scripts/CANONICAL_KNOWN_GAPS.txt`. EVID-01B is the final compiler parity gate,
after the compiler remediation items, not a prerequisite for beginning them.

### EVID-02: Repair The Padded M=5 WMMA Acceptance Fixture

#### Problem

`tensor/src/test/unit/matmul.rs:587-785` expects 16 scalar A-fragment loads at
line 713. The current Svod graph and pinned Tinygrad graph use four gated shaped
loads at that stage. The stale assertion prevents all later checks at lines
741-784 from running.

This is currently a test defect, not proof that padded WMMA lowering is wrong.
It nevertheless means the most important compiler safety gate is not being
executed.

#### Actions

- Capture the pinned Tinygrad graph at the corresponding optimized,
  devectorized, coalesced, gated, and linearized stages.
- Update the pre-linearization assertion to validate four shaped accesses and
  their lane validity rather than requiring a legacy scalar representation.
- Evaluate every shaped lane and prove that a true gate addresses only A
  elements `0..80`.
- Verify invalid lanes use a zero alternate before WMMA consumption.
- Verify output stores cover C elements `0..80` exactly once.
- Continue through PROGRAM construction, linearization, LLVM rendering, and
  optional AMD object compilation.
- Assert no generated address can reach the padded physical tail.
- Keep the hardware test ignored until the compile-only gate passes.

#### Exit Gate

```bash
cargo test -p svod-tensor --lib test_matmul_m5_gfx1151_padded_wmma_compile_only -- --nocapture
```

The test must reach and pass final LLVM assertions. The canonical stage outputs
must also match the pinned reference for validity and address expressions.

#### Current EVID-02 Evidence (2026-08-22)

Schema v2 in `scripts/EVID02_SAFETY_SCHEMA.md` captures complete typed source
graphs at the shared `late-final-rewrite` and `linearized` boundaries from both
Svod and pinned Tinygrad. Producers emit no safety summary or synthesized
order. The strict validator independently enumerates operations, evaluates the
serialized expression ASTs for all lanes, derives all access/coverage/order/IF
semantics, proves exact A/B LOAD ancestry into the corresponding WMMA operands,
validates the float16/float32 accumulator and result dataflow plus strict
operation-level dtypes, and runs eleven adversarial graph mutations on each
artifact. It runs
before broader EVID-01B handling in `check-canonical-parity.sh`; full graph
identity remains an explicit EVID-01B gap.

The Rust compile-only test rejects malformed or extra A/B loads and C stores at
both final rewrite and LINEAR, re-evaluates `A[0..80]`, padded `A[80..256]`,
twice-covered `B[0..256]`, `C[0..80]`, and disabled `C[80..96]`, and proves the
sole IF owns only the partial store by operation order and source identity.
Each C store indexes lanes 0, 1, and 2 of the same WMMA result. LLVM renderer
metadata ties every A GEP/load/branch/zero-fill phi and the partial C
branch/GEP/store to its originating LINEAR UOp. Available-target gfx1151 object
compilation succeeds.

The v2 focused padded-WMMA test passes in debug and release, all 91 codegen unit
tests pass, and a release LLVM WMMA assembly test passes. Strict source-graph
safety comparison passes twice; canonical evidence mode regenerates and
validates both producers twice and verifies the unchanged EVID-01B known-gap
manifest. `cargo check --workspace --all-targets`, formatting, diff checking,
and exact clean pin checks pass. No hardware test was run in that slice. CG-08
was subsequently closed at the PROGRAM boundary. EVID-02 is **Complete**.

## Phase 1: Repair Core IR Semantics

### IR-01: Commit Constants To Their Declared DType

#### Problem

Svod stores floating constants as `f64`. `ConstValue::cast` preserves the
original value for f16, bf16, f32, and f64, while FP8 conversion is unsupported
in `ir/src/types.rs:153-168`. `truncate` only narrows integer values at
`ir/src/types.rs:337-356`. The constant constructor can therefore retain a
value not representable by its declared dtype in
`ir/src/uop/constructors/data.rs:26-38`.

Tinygrad commits constants through dtype-specific conversion functions:

- `submodules/new_new_tinygrad/tinygrad/dtype.py:70-83`
- `submodules/new_new_tinygrad/tinygrad/dtype.py:219-275`
- `submodules/new_new_tinygrad/tinygrad/dtype.py:290-294`
- `submodules/new_new_tinygrad/test/null/test_const_folding.py:54-63`

Incorrect commitment affects constant folding, comparisons, range analysis,
LLVM constant bits, and the relationship between compiled constants and values
stored in memory.

#### Actions

- Implement round-to-nearest-even conversion for f16, bf16, and f32.
- Implement the pinned Tinygrad grids for FP8 E4M3, E5M2, E4M3FNUZ, and
  E5M2FNUZ.
- Define exact behavior for NaN payloads, signed zero, infinities, overflow,
  underflow, and subnormals to match the pin.
- Make typed constant construction fail rather than silently preserving the
  source value when conversion is unsupported.
- Ensure vector/shaped constants commit every lane identically.
- Remove renderer-side truncation assumptions once IR constants are canonical.
- Audit all constant-folding paths in `ir/src/uop/eval.rs` and schedule symbolic
  rewrites for use of committed operands and committed results.

#### Tests

- Port Tinygrad's dtype-grid constant tests.
- Add midpoint and neighboring-value tests for every reduced float format.
- Add signed-zero, NaN, infinity, overflow, underflow, and subnormal tests.
- Compare folded arithmetic and comparison results against emitted backend
  constants.
- Add canonical serialization checks using exact float bits.

#### Exit Gate

For every floating dtype, constructing a typed constant in Svod and Tinygrad
must produce the same semantic value and canonical bits. LLVM and C must
consume the committed value without a second, divergent quantization step.

#### Current IR-01 Evidence (2026-08-22)

Constant commitment now lives in `svod-dtype` and matches the pinned conversion
grids for f16, bf16, f32, f64, E4M3, E5M2, E4M3FNUZ, and E5M2FNUZ. Tests pin
all 256 decoded values for each FP8 format by semantic-bit hash and cover RNE
midpoint neighbors, signed zero, canonical NaN, infinities, finite overflow,
underflow, subnormals, and saturation. Scalar and VConst constructors commit
every value, checked constructors return `ConstantConversion`, and infallible
wrappers no longer retain unsupported source values.

Typed scalar/vector folding commits results before later comparisons. Canonical
serialization records the committed semantic f64 bits, while C and LLVM
constant paths consume those values and use the shared storage-bit encoder for
low-precision formats. Debug and release C/LLVM rendering tests pass.

Focused dtype, IR, schedule, codegen, and tensor-math tests pass. Canonical
evidence mode passes twice with the exact existing known-gap manifest; strict
mode retains the accepted EVID-01B status. No known-gap hash or production gap
changed. Workspace all-target checking, formatting, diff checking, and the
clean Tinygrad pin at `8c8b43de62515abe6c820b1de5aa26b30f48e43a` pass.

The exact pinned `weak.py` and codegen composition were traced with direct
weak-float constant lanes. The follow-up production-pipeline ordering defects
are closed by a composable value-sensitive matcher guard: rewrites that fold,
compare bounds, combine coefficients, decompose powers, or reassociate float
operations defer while their root contains weak-float values. Weak integers
retain the pre-lowering symbolic/index rules because selecting i32/i64 is
value-preserving. `pm_lower_index_dtype` remains in the pinned source order and
removes the guard condition before final symbolic rewriting.

Production-stage regressions cover the exact term-combining, WHERE-bound, and
power cases, immediate f32 midpoint neighbors, scalar and VConst arithmetic and
comparison, mechanical STACK lanes, and Invalid preservation. The focused
lowering suite passes 10/10; full `svod-schedule --lib` passes 1094 with four
pre-existing ignored tests; `svod-dtype` passes 18 tests and three doctests;
`svod-ir --lib` passes 550; and `svod-codegen --lib` passes 93. Canonical
evidence mode passes twice with the exact unchanged known-gap manifest, while
strict mode exits 1 as required. Workspace all-target checking, formatting,
diff checking, and the clean exact Tinygrad pin at
`8c8b43de62515abe6c820b1de5aa26b30f48e43a` pass. IR-01 is **Complete**.

### IR-02: Make PARAM Slot Assignment Collision-Free

#### Problem

`codegen/src/program_pipeline.rs:93-113` starts unassigned PARAM numbering at
zero and ignores already occupied slots. Tinygrad starts after the count of
assigned parameters in
`submodules/new_new_tinygrad/tinygrad/codegen/__init__.py:382-384`.

A normal symbolic tensor probe produced a storage PARAM at slot 0 and a scalar
PARAM at slot 0. Buffer and scalar signatures are often collected separately,
which hides the collision in simple tests, but `ProgramInfo`, canonical ABI
ordering, cache identity, and future unified consumers require globally stable
slots.

#### Actions

- Match the pinned numbering rule exactly.
- Validate uniqueness after numbering and return a typed codegen error on any
  duplicate slot.
- Document whether authored non-dense slots are accepted. If accepted, reserve
  occupied values rather than relying only on a count.
- Ensure CFG insertion still occurs before final numbering.
- Ensure renderer argument collection and runtime buffer/variable binding use
  the same canonical order.
- Include slots in canonical PROGRAM fixtures.

#### Tests

- Assigned storage slot 0 plus one unassigned scalar.
- Assigned slots `0..N-1` plus multiple unassigned scalars.
- Sparse authored slots plus unassigned parameters.
- Mixed global, local/register, and scalar parameters.
- Symbolic tensor execution through PROGRAM, render, compile, and runtime bind.
- Duplicate authored slot rejection.

#### Exit Gate

All PARAM slots in final PROGRAM are unique and deterministic, and the ordered
ABI matches the pinned Tinygrad fixture.

#### Current IR-02 Evidence (2026-08-22)

Svod now follows pinned Tinygrad's post-CFG `pm_number_params` traversal and
assigned-PARAM count for normal dense kernels. Sparse authored slots are
preserved and reserved: numbering starts at the assigned count and skips an
occupied value rather than renumbering authored slots or allowing a collision.
Every `PARAM` class shares this slot identity; local/register `BUFFER` scratch
allocations remain in their separate internal namespace. Duplicate authored or
final slots, unassigned final slots, malformed `ProgramInfo`, and renderer ABI
disagreement return typed program-pipeline errors.

`AbiParamDescriptor` is the single ordered ABI representation carried by
C/LLVM renderer metadata, `ProgramSpec`, `CompiledSpec`, CPU libffi, and AMD
direct/graph/linked-plan kernarg packing. It is sorted by globally unique PARAM
slot and preserves scalar name/dtype plus storage address-space/kind. Source
signatures and runtime values are interleaved in that exact order; there is no
storage-then-scalars compatibility path. The public AMD program loader accepts
only the complete descriptor sequence and retains interleaving exactly; its
buffer-only hardware probes also supply explicit descriptors. `ProgramInfo`
keeps Tinygrad's globals/vars projections but is validated against the complete
descriptor list.

Numbering seeds its counter from the pinned full authored topological PARAM
count, then traverses and substitutes only the outer executable graph while
preserving CALL/FUNCTION bodies. Opaque formals remain unnumbered and excluded;
an identity-shared formal escaping into the outer graph is a typed error.
Prebuilt PROGRAM inputs receive the same final validation, including target,
slot, name, dtype, kind, order, launch dimensions, and globals/vars/ins/outs.
Malformed ProgramInfo entries, unnamed scalar PARAMs, duplicate/unassigned
slots, target mismatch, renderer/compiler disagreement, leaked opaque formals,
and runtime arity mismatch all return typed errors rather than panicking.
`ProgramInfo.vars` validation compares the complete canonical PARAM semantics
before ABI descriptor projection, including bounds, `multiple_of`, axis,
address space, device, volatility, dtype, shape, slot, and name. This rejects
descriptor-equivalent forged metadata while accepting distinct UOp allocations
with identical semantic content. `globals`, `outs`, and `ins` are compared in
their exact canonical sink-derived order by the same shared device-layer
validator used by codegen and `ProgramSpec`.

Focused tests cover exact mixed and sparse C/LLVM signatures, dense production
numbering, duplicate versus reused PARAM identity, deterministic reconstruction,
malformed prebuilt PROGRAM/ProgramInfo values, opaque boundaries, reversed
renderer descriptors, interleaved HCQ packing, and real hardware-free CPU and
tensor symbolic rebind execution through both Clang and LLVM. The canonical
production fixture asserts that it enters PROGRAM with an unassigned PARAM and
leaves with every outer PARAM numbered.

The mixed storage/scalar `program_info` fixture passes pinned cross-language
canonical parity. Two independent evidence-mode runs verify the exact known-gap
manifest; strict mode reports only the remaining documented gaps. IR-02 removes
the production PROGRAM scalar-slot difference (`0` versus pinned `2`) and the
resulting five LINEAR PARAM-order differences. Revalidation on 2026-08-22 passes
109 codegen, 159 device (6 hardware probes ignored), 550 IR, 1,094 schedule (4
ignored), and 84 runtime library tests, plus 3 prepare/execute-loop and 35
symbolic tensor tests. `cargo check --workspace --all-targets`, `cargo fmt --all
-- --check`, `git diff --check`, and pinned-reference cleanliness also pass. The
known-gap manifest is unchanged, including the existing program/linearized gaps.
Clang, LLVM, and ELF-JIT convenience methods delegate to descriptor-derived
libffi; none casts generated individual arguments to aggregate arrays. Final blocker
revalidation passes 111 codegen, 162 device (6 hardware probes ignored), and 85
runtime tests, two independent canonical evidence-mode runs, strict-mode
expected status 1, workspace all-target checking, formatting/diff checks, and
the clean exact Tinygrad pin.

Final blocker closure carries renderer/compiler provenance as semantic
`Op::Source`/`Op::ProgramBinary` data rather than UOp metadata. The versioned
SOURCE identity contains an explicit ordered IR-neutral ABI descriptor sequence,
target, entry name, SHA-256 of canonical-v6 LINEAR JSON under the fixed
`source-stage-linear-v1` label, and SHA-256 of exact source bytes. The versioned
BINARY identity embeds that exact SOURCE identity and adds the compiler cache key
plus SHA-256 of exact binary bytes. These fields participate in `Op`, `UOpKey`,
content hash, parent PROGRAM interning, typed constructors, and serde. Raw
SOURCE/BINARY constructors produce explicit unproven identities and executable
PROGRAM extraction rejects them. Canonical v6 rejects SOURCE/BINARY in
non-verbose oracle documents rather than erasing the post-v6 identity.

`do_render`, `do_compile`, and `get_program` derive and validate semantic
identities before and after PROGRAM reconstruction. `ProgramSpec::from_uop`
independently recomputes sink ABI, target/name, LINEAR digest, source digest, and
BINARY source/payload correspondence; exact compiler-key reuse is enforced by
`do_compile`. `Device::new` wraps every runtime loader with target, compiler,
LINEAR, source, ABI, and binary identity validation. Pre-interned identity-free
or differently identified children and parent PROGRAMs cannot replace newly
proven stages. Missing/tampered source, compiler-key mismatch, and transplanted
binary identities return typed `ProgramStageMismatch` before backend loading.

Compact buffer arrays are now indexed only by storage-descriptor ordinal. Sparse
slots `[0,5]` and interleaved storage/scalar slots are covered in tensor prepare,
TK launch planning, HCQ PROGRAM packing, CPU libffi execution, and AMD kernarg
packing. Exact compact arity is checked before indexing while `globals`, `ins`,
and `outs` continue to use slots for metadata and alias semantics.

`validate_abi_descriptors` is the shared descriptor boundary for
`CompiledSpec`, `ProgramSpec`, CPU/AMD runtime factories, direct Clang/LLVM JIT
loaders, and `AmdProgram::load`. It rejects non-ascending/duplicate/sentinel
slots, unsupported storage/scalar dtypes and names, duplicate scalar names, and
count/name projection mismatches. Public source/byte constructors now require a
complete descriptor sequence and derive counts; descriptorless argument-bearing
runtime specs fail before backend loading.

Independent final exit-gate revalidation on 2026-08-22 passes 115 codegen, 167
device (6 manual hardware probes ignored), 87 runtime, and 551 IR tests. Two
complete canonical evidence-mode runs each verify two independent producer
captures and the unchanged exact known-gap manifest; strict mode returns expected
status 1. `cargo check --workspace --all-targets`, `cargo fmt --all -- --check`,
`git diff --check`, and the clean exact Tinygrad pin
`8c8b43de62515abe6c820b1de5aa26b30f48e43a` pass. IR-02 is **Complete**.

### IR-03: Substitute Call Arguments In FUNCTION Result Shapes

#### Problem

Svod derives FUNCTION/GETTUPLE shapes directly from the body in
`ir/src/shape.rs:715-735`. Formal PARAM dimensions can therefore escape into the
call result. Tinygrad substitutes call arguments during shape inference in
`submodules/new_new_tinygrad/tinygrad/uop/ops.py:360-369`, with coverage in
`submodules/new_new_tinygrad/test/unit/test_call.py:109-128`.

#### Actions

- Build a formal-PARAM to call-argument substitution map at shape inference.
- Apply substitution recursively to every symbolic output dimension.
- Handle tuple outputs and expression-valued dimensions.
- Keep opaque callable bodies opaque except for their declared output shape.
- Ensure substituted shapes participate in hash identity and schedule-cache
  normalization.

#### Exit Gate

Port Tinygrad's direct and expression-valued call-shape tests. No result shape
may contain a formal PARAM that is not present in the actual caller graph.

#### Implementation Evidence

Svod now separates whole-FUNCTION execution validation from pure selected-output
shape substitution. Execution excludes Tinygrad's free slot `-1`, accepts sparse
slots and unused actuals, and requires exact `max_shape`, dtype, and axis parity.
Shape inference gathers only PARAMs reachable from the selected `inner_shape`,
preserves pinned Python indexing semantics by mapping slot `-1` to the last
actual argument, and does not inspect unselected outputs. Missing slots,
incompatible actuals, void symbolic replacements, and dangling selected-shape
formals return typed errors. Substitution is single-pass and preserves nested
CALL/FUNCTION/PROGRAM bodies, while schedule-cache normalization traverses call
arguments without entering opaque implementations.

FUNCTION remains a void, shapeless tuple wrapper; void CALL remains shapeless;
typed instruction CALL is scalar. GETTUPLE recursively substitutes direct,
repeated, nested-expression, and tuple/multi-output dimensions. Rangeify now
extracts only direct `GETTUPLE(TUPLE(...), i)`: precompiled FUNCTIONs remain
opaque with their actual arguments and nested FUNCTION bodies intact, while
non-precompiled FUNCTIONs still inline. Callable body and actual-argument
children participate in structural/content hashes, and focused identity/cache
tests prove distinct symbolic actuals cannot collide. Production constructors
cover logical-shape PARAMs, positional scalar PARAMs, exact fallible FUNCTION
validation, and typed CALLs.

The `symbolic_function` canonical fixture uses those constructors with two tuple
outputs and an arithmetic symbolic dimension. It matches the pinned Tinygrad
graph exactly in two independent evidence runs and is now a strict fixture; its
IR-03 expected-failure manifest section was removed. REDUCE-01 is now the sole
expected-failure section; strict mode returns the expected status 1 for the
remaining documented EVID-01B gaps. Revalidation on 2026-08-22 passes 561 IR,
1,097 schedule (4 ignored), and 115 codegen library tests. All 22 focused tensor
call/schedule tests, 10 relevant custom-kernel tests (2 hardware tests ignored),
and 24 schedule-cache tests pass. Two independent canonical evidence runs match
the `symbolic_function` fixture and exact known-gap manifest; strict mode fails
only for the remaining documented EVID-01B/REDUCE-01 gaps. `cargo check
--workspace --all-targets`, `cargo fmt --all -- --check`, `git diff --check`, and
the clean exact Tinygrad pin
`8c8b43de62515abe6c820b1de5aa26b30f48e43a` pass. IR-03 is **Complete**.

### IR-04: Use Conservative Floating Analysis Bounds

#### Problem

Svod's fallback float bounds are finite representable extrema in
`dtype/src/lib.rs:283-325` and are consumed through
`ir/src/uop/range_eval.rs:637-640`. Tinygrad uses `[-inf, +inf]` in
`submodules/new_new_tinygrad/tinygrad/dtype.py:70-83`.

The sound range evaluator already declines unsupported proofs, reducing the
blast radius, but ordinary bound consumers can still mis-handle infinity,
overflow, or NaN-sensitive transforms.

#### Actions

- Separate numeric format limits from conservative analysis bounds.
- Make unknown floating values use `[-inf, +inf]` for compiler proofs.
- Audit bound-driven MAX, comparison, and WHERE simplifications.
- Require sound bounds for any rewrite that changes observable behavior.
- Keep finite `finfo` limits available for user-facing dtype metadata.

#### Exit Gate

`max(unknown_f32, f32::MAX)` must not fold to the finite constant. Add matching
tests for infinity, overflow-to-infinity, NaN, and WHERE conditions.

#### Implementation And Evidence

Numeric format limits remain finite for dtype metadata, while conservative
analysis bounds for every floating scalar, vector, and shaped dtype are now
`[-inf, +inf]`. Unknown PARAM and LOAD values use those analysis bounds. Range
evaluation covers finite arithmetic, overflow to infinity, division by zero,
NaN-producing operations, casts, comparison domains, and WHERE unions without
using unsupported results as proofs.

Integer range arithmetic is evaluated in `i128` and committed to the declared
dtype before becoming proof input. Exactly representable ranges remain narrow;
overflow, underflow, wrapping casts, invalid division or shifts, and negation of
the signed minimum conservatively fall back to the full typed domain. Non-constant
MULACC declines sound proof rather than deriving an endpoint-only interval.
Sampled int8/uint8 binary and cast regressions verify runtime values remain
enclosed by every returned sound range.

Behavior-changing float rewrites now require sound, NaN-aware bounds. In
particular, `x < x` remains observable when `x` may be NaN, bound-driven MAX
selection requires strict range separation, and range ordering preserves
distinct signed-zero endpoints. This prevents both
`max(unknown_f32, f32::MAX)` and signed-zero MAX ties from folding while still
allowing comparisons over explicitly bounded finite values. Floating reduction
identities use infinities rather than finite format extrema.

Typed-integer div/mod rewrites use reusable checked no-wrap proofs over
`SoundVminVmaxProperty`. They enforce exact signed/unsigned 8/16/32/64 plus
WeakInt/Index bounds and prove source and replacement dtype, logical shape, and
every changed Add/Sub/Mul/Neg/FloorDiv/FloorMod intermediate. Factor-out,
cancellation, exact-division, nested-division, bucket, comparison-lift, and
recombination rules reject uncertain ranges, zero divisors, signed-minimum
overflow, wrapping intermediates, and invalid host arithmetic. General
decomposition remains disabled; the scalar affine congruence subset required by
QR is enabled only for positive divisors and nonnegative sound ranges. Hardware
vectors reject rather than constructing partial candidates, and factor/GCD
helpers handle `i64::MIN` conservatively.

Regressions cover Int8 factor-div/mod and cancellation counterexamples,
positive and negative floor-division composition, zero divisors, Int8/UInt8 and
`i64::MIN` boundaries, affine congruence, hardware vectors, shaped broadcasts,
and sampled original-versus-rewritten execution. Revalidation on 2026-08-23
passes 19 dtype tests plus 3 doc tests, 580 IR tests, 1,117 schedule tests with 4
ignored, 115 codegen tests, 66 tensor math tests, and 203 tensor reduction tests.
Two canonical evidence-mode runs reproduce the exact known-gap manifest; strict
mode returns the required status 1 only for documented EVID-01B/REDUCE-01 gaps.
`cargo check --workspace --all-targets`, `cargo fmt --all -- --check`,
`git diff --check`, and the clean exact Tinygrad pin
`8c8b43de62515abe6c820b1de5aa26b30f48e43a` pass. Independent adversarial
review accepts the final implementation with no unresolved findings. IR-04 is
**Complete**.

### IR-05: Match Structural Commutative Ordering

#### Problem

Pinned Tinygrad applies a narrow index-expression commutative flip in
`submodules/new_new_tinygrad/tinygrad/uop/symbolic.py:205-218`. Svod recognizes
commutative operators in `ir/src/types.rs:1114-1128` and helper decomposition,
but does not yet prove the same structural source ordering at the symbolic
boundary. Equivalent expressions can therefore retain different source order,
changing hash identity, rewrite eligibility, and canonical stage topology.

#### Actions

- Port the pinned ordering predicate and its index-only scope exactly.
- Do not globally sort commutative operands or erase authored source order.
- Apply ordering at the equivalent symbolic pass boundary and include it in
  hash-consing/cache identity tests.

#### Tests

- Compare both source orders for ADD, MUL, AND, OR, XOR, and MAX where covered
  by the pinned matcher.
- Cover constants, RANGE expressions, nested associative trees, and non-index
  expressions that must remain unflipped.
- Require canonical optimized-stage parity and stable hashes across repeats.

#### Exit Gate

Every expression in the focused corpus has the pinned structural source order,
while non-target commutative expressions preserve their original order.

#### Implementation And Evidence

The pinned structural projection and comparison are implemented at the symbolic
boundary for Add, Mul, Max, Eq, Ne, And, Or, and Xor over scalar or shaped
WeakInt. Concrete integer and hardware-vector dtypes remain outside the matcher.
The comparison recursively projects `(op, arg, dtype, sources...)`, including
VConst as a STACK of constants and the exact WMMA metadata tuple. Exact
projection ties preserve authored order; hash-consing continues to use ordered
child identities while external content hashes remain allocation-independent.

Register-stack scalarization preserves dependencies, proves elimination after
the optimizer pass, and merges only RANGE closures that feed register reads.
Focused regressions cover both operand orders, constants, RANGE expressions,
nested associative trees, projection ties, WMMA ordering, register END
isolation, bool storage, and scalar/empty/1D/2D `nonzero` behavior. Independent
review found no remaining structural-ordering, WMMA, register-stack, CFG, or C
renderer findings.

Acceptance exposed two unrelated host-path blockers that previously hid the
full gate. The C renderer now matches Tinygrad's SSA materialization boundaries
for WHERE, non-register LOAD, shaped CAST, and WMMA; derives grouped access
width without recursively probing every UOp's address space; and preserves
shaped declarations, stores, bitcasts, and address casts. The uncached tensor
schedule path now uses the same normalize/rangeify/restore pipeline as a cache
miss, preventing tagged buffer clones from becoming zero-filled replacements.

Revalidation on 2026-08-23 passes 581 IR tests, 1,130 schedule tests with 4
ignored, 118 codegen tests, all focused tensor modules, all 24 schedule-cache
tests, and the complete tensor library suite with 1,431 passed and 16 ignored.
The formerly stalled Clang Threefry distribution regression passes in 28.90
seconds. Two independent canonical runs produced byte-identical diagnostics;
evidence mode verifies the exact checked-in manifest, with strict parity at
`tensor` and `rangeified`. Strict mode returns status 1 only for the documented
EVID-01B production gaps, while REDUCE-01 remains the sole expected-failure
fixture. `cargo check --workspace --all-targets`, formatting, `git diff
--check`, and the clean pinned Tinygrad commit
`8c8b43de62515abe6c820b1de5aa26b30f48e43a` pass. Hardware-only validation
remains deferred. IR-05 is **Complete**.

## Phase 2: Restore Specification Boundaries

### SPEC-01: Port And Enforce `spec_kernel_graph`

#### Problem

Tinygrad verifies the callified graph after kernel splitting:

- `submodules/new_new_tinygrad/tinygrad/uop/spec.py:256-282`
- `submodules/new_new_tinygrad/tinygrad/schedule/rangeify.py:599-603`

Svod returns after `fix_assign` in `schedule/src/rangeify/kernel.rs:465-475`.
The tensor spec runs later on individual kernel ASTs, and the program spec runs
at final codegen. Neither validates the outer CALL/AFTER/buffer topology at its
production boundary.

#### Actions

- Port the pinned kernel-graph whitelist and source/dtype invariants.
- Run it after `fix_assign` and before constructing `PreSchedule` items.
- Preserve call boundaries while verifying the outer graph.
- Include structured MSELECT/MSTACK, CALL, AFTER, COPY, buffer, and tuple rules.
- Return contextual typed errors containing the failing node and source path.

#### Tests

- Valid multi-output call graph.
- Invalid CALL body and argument ordering.
- Invalid MSELECT index and MSTACK layout.
- Invalid AFTER dependency shape.
- Invalid tuple/device metadata.
- Cross-device COPY accepted only in the supported outer form.

#### Exit Gate

Equivalent valid and invalid fixtures must pass or fail at the same boundary in
Svod and Tinygrad.

### SPEC-02: Reject Legacy Index DType At The Final Boundary

#### Problem

`ScalarDType::Index` remains in `dtype/src/lib.rs:191-195`. Normal index lowering
should remove it, but `spec_program` rejects weak dtypes without explicitly
rejecting legacy Index in `schedule/src/spec.rs:449-528`. Backends then disagree
about how to handle it.

#### Actions

- Add an explicit final PROGRAM rejection for legacy Index.
- Audit `dtype_from_op` fallback arms that can preserve a stale Index dtype.
- Add a post-index-lowering invariant check before late decomposition.
- Keep backend assertions only as defense in depth, not as the primary gate.

#### Exit Gate

No valid final PROGRAM fixture contains Index. A deliberately injected Index
must fail `spec_program` with a typed error before rendering.

## Phase 3: Complete Supported Multi-Device Scheduling

### MULTI-01: Move Supported `multi_pm` Rewrites Before Rangeification

#### Problem

Tinygrad runs `multi_pm` before range assignment in
`submodules/new_new_tinygrad/tinygrad/schedule/rangeify.py:575-584`. Svod enters
rangeification in `schedule/src/rangeify/transforms.rs:293-337`, splits kernels,
and only applies `multi_pm` later in
`schedule/src/optimizer/mod.rs:947-979`.

Svod's supported exact subset is implemented in `schedule/src/multi.rs:119-153`.
Running it after splitting can materialize or separate graphs before shard-aware
movement and selection are resolved.

#### Actions

- Add the exact supported `multi_pm` matcher at the beginning of
  `rangeify_with_map`, matching the pinned placement.
- Decide whether a second per-kernel application is required by the reference.
  Do not retain it by default without a parity fixture proving the need.
- Stage-capture the graph immediately before and after multi rewriting.
- Reject unsupported unresolved multi forms before ordinary rangeification can
  erase their intent.

#### Exit Gate

End-to-end MSELECT, same-axis ALU, and supported movement fixtures must match the
reference before and after rangeification.

### MULTI-02: Lower Reductions Across Sharded Axes

**Status: Complete on host.** The representable single-axis subset emits local
reductions followed by explicit host-staged SUM/MAX execution. Reduced-precision
rounding and overflow, buffer shape/alignment, dependency edges,
unresolved-collective rejection, and typed subset validation are covered.
Native hardware collective strategies remain excluded.

#### Problem

Svod leaves a reduction unchanged when it crosses the shard axis in
`schedule/src/multi.rs:57-64`. Tinygrad performs the local reduction and emits
ALLREDUCE in `submodules/new_new_tinygrad/tinygrad/schedule/multi.py:103-119`,
then lowers all-reduce work through:

- `submodules/new_new_tinygrad/tinygrad/schedule/rangeify.py:127-138`
- `submodules/new_new_tinygrad/tinygrad/schedule/allreduce.py:6-58`
- `submodules/new_new_tinygrad/test/unit/test_allreduce.py`

Svod's `Op::AllReduce` now lowers to a correctness-first executable host
strategy; optimized native collectives remain a hardware follow-up.

#### Actions

- Emit local reduction plus `Op::AllReduce` for the representable single-axis
  shard model.
- Define executable lowering for the initial supported strategy. Start with a
  correctness-first naive strategy if ring/all-to-all cannot be ported in the
  same slice.
- Port reduced-precision cast behavior where Tinygrad applies
  `ALLREDUCE_CAST`.
- Add explicit runtime schedule items and dependency edges for collective work.
- Ensure `AllReduce` never reaches `spec_program` unresolved.
- Reject multi-axis or unrepresentable collective forms with typed errors.

#### Exit Gate

Single-axis sharded SUM and MAX fixtures must produce correct numeric output on
the host/mock multi-device executor. Hardware strategies require separate
acceptance before being enabled by default.

### MULTI-03: Permit Open DEVICE Ranges At Kernel Boundaries

**Status: Complete on host/mock.** Aligned MSTACK CALL sources expand into one
schedule item per lane, DEVICE extent is checked against lane cardinality, and
`_device_num` is fixed per lane and cannot be overridden at execution. Expanded
consumers conservatively depend on all emitted producer instances. A real CPU
plan executes both lanes with distinct bound values.

#### Problem

Svod rejects a STORE whenever any range remains open in
`schedule/src/rangeify/kernel.rs:249-264`. Tinygrad permits open DEVICE ranges in
`submodules/new_new_tinygrad/tinygrad/schedule/rangeify.py:541-550`. Svod only
lowers DEVICE ranges later in `schedule/src/gpudims.rs:43-73`, so an incorrectly
rejected store cannot be recovered.

#### Actions

- Reject open computational ranges but permit open DEVICE ranges at the split
  boundary.
- Preserve DEVICE range identity into the kernel call.
- Lower DEVICE ranges to `_device_num` consistently for CPU/mock and GPU paths.
- Do not make DEVICE lowering conditional on unrelated renderer capabilities.

#### Exit Gate

A STORE with only an open DEVICE range must split into a valid kernel call and
receive a correctly bounded `_device_num` parameter.

### MULTI-04: Make The Supported Subset Explicit

#### Problem

Svod's `Op::Multi` intentionally lacks Tinygrad's tuple-device and shard-range
metadata. This prevents exact representation of multi-axis UNSHARD, resharding,
and several movement forms. The limitation is documented in
`schedule/src/multi.rs:1-5`, but some unsupported forms currently survive
unchanged.

#### Actions

- Publish one capability table for supported multi-device operations.
- Reject unsupported reshape, reshard, mismatched-axis ALU, multi-axis shard,
  and out-of-range selection at the tensor or kernel-graph spec boundary.
- Do not silently leave unsupported multi nodes for ordinary codegen.
- Add issue references for intentionally deferred multi-axis representation.

#### Exit Gate

Every multi-device operation either lowers to an executable supported form or
fails with a specific typed error before kernel optimization.

## Phase 4: Correct Backend Semantics

### CG-01: Pack RDNA Int8 WMMA Operands To The Intrinsic ABI

#### Problem

Svod advertises RDNA3 int8-to-int32 tensor cores in
`schedule/src/optimizer/renderer.rs:921-928` and selects the int8 intrinsic in
`codegen/src/llvm/amd/wmma.rs:223-244`. Its wire conversion only packs bf16 and
FP8 in `codegen/src/llvm/amd/wmma.rs:123-133`.

Tinygrad bitcasts each `<16 x i8>` operand to `<4 x i32>` before rendering in
`submodules/new_new_tinygrad/tinygrad/renderer/llvmir.py:294-298`.

#### Actions

- Add the architecture-specific pre-render bitcast for both signed and unsigned
  int8 operands.
- Verify exact intrinsic signatures for gfx1100 and gfx1151.
- Preserve accumulator and result lane layout.
- Validate signedness/immediate selectors.

#### Exit Gate

Render and compile an int8-to-int32 RDNA WMMA kernel. LLVM verification must
accept the intrinsic, and execution must match a scalar reference on supported
hardware.

### CG-02: Match LLVM Floating Comparison Predicates

#### Problem

Svod emits unordered relational comparisons in
`codegen/src/llvm/cpu/ops.rs:777-811`. Tinygrad's primitive floating comparison
uses ordered `olt` in
`submodules/new_new_tinygrad/tinygrad/renderer/llvmir.py:69-77`.

The difference is observable for NaN: unordered less-than is true when either
operand is NaN.

#### Actions

- Use ordered predicates for `<`, `<=`, `>`, and `>=` unless a specific IR op
  explicitly requests unordered behavior.
- Keep ordered equality and unordered inequality aligned with the pin.
- Audit fast-math flags to ensure they do not invalidate required NaN behavior.

#### Exit Gate

CPU LLVM and AMD LLVM tests must show all relational comparisons with NaN are
false, equality is false, and inequality is true, matching Tinygrad and C.

### CG-03: Separate FP8 Matrix Capability From Ordinary FP8 ALU Capability

#### Problem

Svod marks CDNA3/gfx942 FP8 as generally supported in
`schedule/src/optimizer/renderer.rs:382-403`. Tinygrad supports gfx942 FP8 matrix
operations but does not advertise ordinary OCP FP8 ALU in
`submodules/new_new_tinygrad/tinygrad/renderer/llvmir.py:244-248,320-321`.

#### Actions

- Split storage, conversion, matrix-operand, and ordinary-ALU capability checks.
- Keep valid gfx942 MFMA selection.
- Force non-WMMA FP8 arithmetic through target-supported decomposition.
- Verify gfx950 native/scaled FP8 behavior remains unchanged.

#### Exit Gate

Ordinary gfx942 FP8 add/multiply must not emit floating ALU over LLVM `i8`.
gfx942 MFMA and gfx950 scaled MFMA tests must continue to compile.

### CG-04: Render Volatile Memory Semantics

#### Problem

`ParamArg.volatile` is retained in `ir/src/types.rs:364-390`, but C and LLVM
ignore it. Tinygrad consumes volatile metadata in:

- `submodules/new_new_tinygrad/tinygrad/renderer/cstyle.py:149-155`
- `submodules/new_new_tinygrad/tinygrad/renderer/llvmir.py:87-101`

#### Actions

- Add volatile pointer qualification or access syntax to C-like backends.
- Emit LLVM `volatile` loads and stores after tracing the address to its PARAM.
- Cover gated, grouped, local, and global accesses.

#### Exit Gate

Renderer tests must assert volatile syntax/flags for volatile buffers and their
absence for ordinary buffers.

### CG-05: Match Fast Integer Division Defaults

#### Problem

Tinygrad defaults `DISABLE_FAST_IDIV=1` in
`submodules/new_new_tinygrad/tinygrad/helpers.py:244-245` and conditionally
installs magic non-power-of-two division in
`submodules/new_new_tinygrad/tinygrad/codegen/decomp/op.py:89-110`.

Svod installs `fast_division_patterns` and `pm_mod_to_idiv` whenever shifts are
supported in `schedule/src/optimizer/mod.rs:892-923`.

#### Actions

- Add an explicit optimizer configuration field for fast integer division.
- Default it to disabled to match the pin.
- Include the setting in optimizer and BEAM cache identities.
- Preserve always-safe power-of-two transformations separately.
- Port Tinygrad's counterexamples and add randomized signed/unsigned tests.

#### Exit Gate

Default optimized IR must match Tinygrad for non-power-of-two division. Enabling
the option must pass exhaustive narrow-width and property-based equivalence
tests before use in indexing.

### CG-06: Match Expander Cleanup Placement

#### Problem

Pinned Tinygrad composes `expander2` with `pm_flatten_range` and `mop_cleanup` in
`submodules/new_new_tinygrad/tinygrad/codegen/__init__.py:84-90`, then runs
`mop_cleanup` again with reduction lowering at lines 317-320. Svod's
`schedule/src/expand.rs:165-180` runs the expander alone and first composes
movement cleanup with reduction lowering in
`schedule/src/optimizer/mod.rs:431-441`. The stage delta shows cleanup-visible
structure already differs at `expanded`.

#### Actions

- Compose the pinned flattening and movement cleanup rules at the expander
  boundary in the same order.
- Retain the second cleanup before local reduction only if the pinned pass does.
- Audit fixpoint behavior so cleanup does not consume validity or shaped lanes
  required by late memory passes.

#### Tests

- Add focused EXPAND/RESHAPE/PAD/SHRINK cleanup graphs around RANGE and REDUCE.
- Cover direct WMMA expansion and grouped reduction staging.
- Require strict `expanded` capture parity before and after reduction removal.

#### Exit Gate

The focused expanded-stage corpus matches pinned Tinygrad structurally and no
cleanup-required movement op survives to reduction lowering.

### CG-07: Match Shift-Amount DType Semantics

#### Problem

Pinned Tinygrad derives a shift result from the left operand while committing
the shift amount to the left dtype during weak commitment in
`submodules/new_new_tinygrad/tinygrad/uop/weak.py:49-58`; final spec permits the
renderer-specific `uint32` count exception in
`submodules/new_new_tinygrad/tinygrad/uop/spec.py:64-72`. Svod preserves the
left result dtype in `ir/src/dtype_rule.rs:83` but its constructors in
`ir/src/uop/constructors/compute.rs:71-84` only validate that both operands are
integer and do not establish the pinned count dtype.

#### Actions

- Commit weak shift amounts to the left operand dtype at the equivalent pass.
- Preserve only the pinned renderer-lowered `uint32` exception at final spec.
- Audit decomposition helpers that currently construct shift constants directly
  in the value dtype.

#### Tests

- Cover signed/unsigned 8/16/32/64-bit values with weak and strong counts.
- Cover renderer-lowered `uint32` counts and reject unrelated mixed count dtypes.
- Compare canonical dtype/source fields before and after weak commitment and at
  final PROGRAM validation.

#### Exit Gate

All shift-count dtypes match the pin before rendering, and final spec accepts no
mixed shift form beyond the explicit `uint32` renderer exception.

### CG-08: Sanitize Generated Kernel Names At The PROGRAM Boundary

#### Problem

Broad EVID-02 validation exposed generated symbolic-shape names such as
`E_L?n6`. `codegen/src/program_pipeline.rs:142-145` forwards a structured SINK
name directly while only the optimizer-metadata fallback uses
`KernelInfo::function_name()`. C and LLVM therefore receive an invalid `?` in
the function identifier and clang rejects otherwise valid symbolic-batch
kernels. This is independent of padded WMMA lowering.

#### Actions

- Apply Tinygrad-compatible `to_function_name` sanitization exactly once to the
  selected PROGRAM entry point, including structured SINK names.
- Keep human-readable diagnostic names separate from renderer identifiers.
- Cover symbolic extents and punctuation in both C and LLVM renderers without
  changing explicit valid names.

#### Exit Gate

The `symbolic_batch` tensor selection passes for Clang and LLVM, generated C and
LLVM contain valid identical entry-point identifiers, and canonical PROGRAM
name evidence matches the pinned sanitization rule.

## Phase 5: Keep The Supported Backend Surface Testable

### CG-09: Remove The Unsupported MLIR Backend

#### Decision

The optional MLIR backend was disabled by default, required an exact external
LLVM 21 toolchain unavailable to normal workspace validation, duplicated the
working C and LLVM CPU paths, and accumulated compile and semantic drift. Svod
has no committed workload or deployment that depends on it. Maintaining the
feature therefore imposed a backend-wide tax without a validated user path.

#### Actions

- Remove the MLIR renderer, ExecutionEngine runtime, CPU backend selection, and
  backend-specific tests.
- Remove Melior dependencies, propagated crate features, lockfile entries, and
  the Melior git submodule.
- Remove MLIR-specific error variants, documentation, and acceptance gates.
- Preserve generic references to MLIR only where they describe external
  compiler architecture or a rewrite algorithm rather than a Svod backend.

#### Exit Gate

No `mlir` Cargo feature, Melior dependency, Svod MLIR module, runtime selector,
or backend claim remains. Workspace all-target checking and the affected C/LLVM
host suites pass. CG-09 is **Complete**.

#### Evidence (2026-08-23)

The renderer, ExecutionEngine runtime, CPU selector, feature propagation,
backend-specific tests and errors, Nix toolchain variables, Cargo dependencies,
lockfile records, and Melior submodule are removed. User-facing backend tables
now list only Clang and LLVM; generic references to external MLIR architectures
and MLIR-style rewrite algorithms remain intentionally intact. Cargo metadata
contains no `mlir` feature and `Cargo.lock` contains no Melior or `mlir-sys`
package.

`cargo check --workspace --all-targets`, 118 codegen tests, 86 runtime tests,
and the complete tensor suite with 1,431 passed and 16 ignored pass. The ONNX
library test target compiles; its broad execution suite retains unrelated
pre-existing attention-expansion, affine-grid, and light-model failures.
Formatting and diff checks pass, and independent review found no remaining
backend-removal issue. Nix evaluation was unavailable because `nix` is not
installed in this environment. The workspace-wide all-features probe proceeds
without MLIR and remains blocked by unrelated CUDA API drift and missing Z3
headers.

## Phase 6: Make HCQ And AMD Failure-Safe

### HCQ-01: Use A Validated Multi-XCC AQL Completion Discipline

#### Problem

Multi-XCC devices select AQL in `device/src/amd/queue.rs:1016-1040`. Direct
dispatch puts completion directly on kernel packets and waits for it in
`device/src/amd/queue.rs:1099-1116`. The same file documents that this mechanism
can strand on multi-XCC and provides an unused trailing barrier workaround at
`device/src/amd/queue.rs:1305-1337`.

Tinygrad leaves kernel completion unset and finalizes through a PM4 timeline
store in:

- `submodules/new_new_tinygrad/tinygrad/runtime/support/hcq.py:372-380`
- `submodules/new_new_tinygrad/tinygrad/runtime/ops_amd.py:385-445`

#### Actions

- Make completion queue-owned rather than per-kernel for ordinary AQL dispatch.
- Use a trailing barrier/control sequence that produces one terminal finalizer.
- Ensure multi-XCC PM4 release stores are predicated to the required XCC.
- Redesign per-kernel profiling timestamps so correctness does not depend on the
  unreliable completion mechanism.
- Apply the same ownership rules to direct dispatch, graph replay, and linked
  plan replay.
- Remove or wire the currently unused workaround helper.

#### Host Tests

- Packet goldens for kernel packets with no native completion signal.
- Terminal barrier/control-store completion packet.
- XCC predication around final stores.
- Signal-pool ownership and release after finalizer drop.
- Profiled and unprofiled packet sequences.

#### Current HCQ-01 Evidence (2026-08-23)

Direct AQL dispatch now uses the pinned Tinygrad completion discipline:
`Wait -> MemoryBarrier -> [Timestamp] -> Compute -> [Timestamp] -> Store` is
split into barriered PM4 vendor IBs around a native dispatch whose
`completion_signal` remains zero. The queue's monotonic timeline owns ordinary
completion; no native decrement-to-zero signal is allocated or waited. Direct
control storage comes from the queue-owned host-visible arena and remains alive
until timeline retirement.

AQL graph replay now uses the same linked PM4 control-store finalizer instead of
a terminal `BARRIER_AND`. Profiled direct and graph paths use explicit PM4 GPU
clock stores, so timestamp collection and synchronization no longer depend on
native kernel completion. Linked-plan AQL stores retain their existing typed
device/compute/copy timelines. Every multi-XCC PM4 timestamp and timeline store
inside an AQL vendor IB is wrapped in `PRED_EXEC(xcc_mask=1)`; forced AQL on a
single-XCC device emits no predication. Store/timestamp patch offsets account
for the predicate prefix.

The obsolete barrier-signal dispatch helper, native AQL finalizer variant,
native signal wait/reclamation path, and AQL completion patch site are removed.
Host packet tests pin unprofiled and profiled direct sequences, zero completion
fields, terminal control stores, XCC0 predication, control-only finalizers, and
single-XCC behavior. `cargo test -p svod-device --lib --no-fail-fast` passes
with 171 passed and 7 hardware tests ignored; workspace all-target checking
passes.

Forced AQL on the available single-XCC gfx1151 passes 2,000 asynchronous, 128
synchronous, and 128 profiled direct dispatches with valid explicit PM4
timestamps. Normal and profiled graph replay, a two-kernel RAW graph, and three
prepared linked-plan tensor fixtures also pass with `SVOD_AMD_AQL=1`. This
was followed by the complete forced-AQL tensor suite: 1,431 passed and 16
ignored in 215.36 seconds. Forced-AQL ONNX add and AlexNet fixtures also pass.
This validates AQL vendor-IB execution and timeline ownership broadly, but not
multi-XCC `PRED_EXEC`; HCQ-01 remains **hardware pending** until the stress gate
below runs on gfx942 or gfx950.

#### Hardware Exit Gate

On gfx942 or gfx950, run thousands of synchronous, asynchronous, and profiled
AQL submissions under signal-pool pressure. No timeout, stranded signal, or
missing timestamp is acceptable.

### HCQ-02: Validate Every Native Copy Endpoint

**Status: Implemented; two-GPU hardware pending.** Native linked replay now
requires both current copy endpoints to match the selected context's exact
`DeviceSpec` before any address is resolved or any backend context is called.
The check runs on every replay, so replacing a captured endpoint with AMD:1 or
CPU safely declines to semantic host staging. Declines are typed with operation,
endpoint, expected owner, and actual owner. A host mock proves local AMD:0
endpoints enter native replay while post-capture AMD:1 and CPU replacements do
not. Verified peer mappings remain intentionally unsupported.

#### Problem

Native replay eligibility checks compiled-program devices in
`runtime/src/execution_plan.rs:608-622` but not copy endpoint ownership. It then
passes raw addresses into the first program's AMD context at lines 652-685.
`device/src/amd/linked_plan.rs:354-395` submits all copies through that owner's
SDMA queue.

Tinygrad chooses a concrete copy device, maps endpoints, and schedules per-device
queues in `submodules/new_new_tinygrad/tinygrad/runtime/graph/hcq.py:86-113,198-218`.

#### Actions

- Include source and destination physical devices in native replay eligibility.
- Revalidate eligibility on replay when replaceable buffers can change device.
- Require a verified bidirectional accessibility policy for direct peer copies.
- Decline native replay and use host staging when ownership/accessibility cannot
  be proven.
- Never submit CPU or foreign-GPU VAs through a selected GPU's SDMA queue.
- Return a typed reason for native replay decline for diagnostics.

#### Exit Gate

Host/mock tests must prove that AMD:0 compute plus AMD:0-to-AMD:1 or AMD-to-CPU
copy cannot enter one-context native replay. A two-GPU hardware test must preserve
sentinels without a KFD fault.

### HCQ-03: Never Free Hardware-Referenced Memory After A Failed Drain

**Status: Complete on host/mock.** AMD drain failures
now poison the physical device and destructive buffer release centrally
quarantines allocations on poisoned devices or during panic unwind. Allocator,
graph, linked-plan, program, PMC, queue, kernarg, scratch, and SDMA teardown paths
no longer free or overwrite potentially live storage after a failed wait.
Scratch growth now drains before allocation/state publication and leaves the old
descriptor/backing untouched on failure. Kernarg wrap propagates a failed drain
without resetting its cursor. Scripted mock wait failures and free-call audits
cover scratch, graph, program, kernarg, PMC, queue, signal, and user buffers.

#### Problem

Scratch growth and teardown continue freeing old backing after drain failures in
`device/src/amd/connector.rs:370-410`. Similar teardown risks exist in allocator,
graph, and linked-plan drops. A timeout does not prove hardware stopped using the
allocation.

Tinygrad aborts before freeing mapped buffers when synchronization raises in
`submodules/new_new_tinygrad/tinygrad/runtime/support/hcq.py:566-570`.

#### Actions

- Propagate synchronization failures from explicit resize/free operations.
- On Drop, quarantine or intentionally leak potentially live allocations after
  a failed drain.
- Poison the owning device/queue so subsequent work fails immediately.
- Do not publish new scratch state and free old state as one partially fallible
  sequence.
- Add RAII guards for allocation construction paths.
- Centralize the policy for teardown after synchronization failure.

#### Exit Gate

Fault-injection tests must show that a failed drain returns an error, poisons the
owner, and does not call unmap/free on the old scratch, graph, program, kernarg,
or user-buffer allocation.

### HCQ-04: Make Timeline Reservation And Publication Transactional

**Status: Complete on host/mock.** Direct PM4/AQL and
graph paths finish fallible lowering, patching, resident copies, and size checks
before reservation where practical. AMD linked replay snapshots its timelines,
completes all validation/patching and aggregate PM4/AQL/SDMA capacity waits, and
holds compute/copy queue guards before registering a prepared finalizer. The
remaining publication section only copies prepared bytes and rings doorbells.
Pre-publication errors restore the timeline snapshot; partial publication or
panic poisons the physical device and resolves the prepared finalizer as failed.
Semantic linked replay latches typed `PlanPoisoned` after committed-epoch
failure. Ordinary drains no longer reset generations, and PM4/AQL/SDMA rollover
reset is serialized under publication authority. Host fault injection covers
reservation, pre-doorbell, post-doorbell, panic, linked replay, and rollback.

#### Problem

Several paths reserve timeline points before fallible allocation, patching, or
publication. If no command stores the reserved value, a later submission can
wait forever.

Relevant paths include:

- `device/src/amd/queue.rs:1133-1189`
- `device/src/amd/queue.rs:1233-1246`
- `device/src/amd/linked_plan.rs:334-354`
- `runtime/src/execution_plan.rs:231-260`

#### Actions

- Complete validation, allocation, and patch planning before reservation.
- After reservation, make publication infallible where practical.
- Otherwise introduce an explicit reservation guard that either commits a
  signal-producing submission or poisons the timeline.
- Do not retry a plan with an uncommitted prior reservation.
- Preserve callback failure semantics without allowing silent timeline reuse.

#### Exit Gate

Inject failures after every pre-publication stage. Every case must either leave
the timeline unchanged or poison it with an immediate typed failure; no future
wait may time out on a value that was never published.

### HCQ-05: Bound Or Indirect Large Graph Submissions

**Status: Implemented; boundary stress pending.** PM4 graph bodies now live in
graph-owned host-visible resident storage and each replay publishes only one
four-dword `PACKET3_INDIRECT_BUFFER`. AQL publication validates packet count and
waits for exact read-pointer headroom. PM4 ring, PM4 IB, AQL ring, and SDMA ring
limits are release-build checks returning typed `CommandStreamTooLarge` errors;
linked-plan capture preflights each native stream, and replay rejects aggregate
transactions that cannot fit atomically (including SDMA wrap padding). Host
tests pin PM4/AQL/SDMA/IB boundaries in release mode. A 12-kernel normal/profiled graph passes on gfx1151
through both native PM4 and forced AQL; larger near-capacity stress remains.

#### Problem

PM4 graph capture concatenates complete command streams in
`device/src/amd/graph.rs:126-205` and publishes them through
`device/src/amd/queue.rs:1224-1247`. `push_pm4` has only a debug assertion for a
1024-dword submission at `device/src/amd/queue.rs:946-969`. AQL graph submission
also needs a total ring-capacity check.

Tinygrad stores PM4 graph bodies in resident memory and publishes a small IB
packet in `submodules/new_new_tinygrad/tinygrad/runtime/ops_amd.py:396-420`.

#### Actions

- Prefer resident indirect-buffer replay for large PM4 graphs.
- Enforce command, dword, packet, ring, and hardware IB-field limits in release
  builds.
- Split or reject oversized AQL graph submissions before ring writes.
- Include limits in graph eligibility and cache identity.
- Preserve immutable linked bytes and private replay patch state.

#### Exit Gate

Graphs below, at, and above each boundary must either replay correctly or return
a typed size error. No debug-only safety check is acceptable.

### HCQ-06: Complete AMD Allocation Ownership

**Status: Complete on host/mock.** The shared-owner queue model
has been replaced by a bounded atomic lane pool. A non-clone `QueueLease` is the
only compute publication authority; graph and linked replay lease per epoch,
while direct fallback releases its epoch lease through `finish_replay`. The
backend-local queue mutex remains as a Rust aliasing guard and is uncontended by
compute publishers.

Program code is reference-counted and retained by direct finalizers, graphs, and
linked plans. Queue teardown uses explicit `Active`, `Destroyed`, and
`Quarantined` states with drain -> destroy -> scratch/backing order. Failed
doorbell-map rollback returns typed `AmdQueueStillActive` and poisons before
construction guards unwind; destroy failure quarantines, while a doorbell-only
unmap leak does not suppress safe backing release. Panic abandonment poisons and
prevents signal-slot reuse. Graph/linked storage, inactive signals, copy staging,
PMC readback, signal-pool construction, device-zero failure, and LRU teardown all
have explicit ownership paths. Mock allocation/free counters, process event-page
unwind, and exhaustive per-step injection close the host gate. No parallel GPU
suite was run after two earlier forced-AQL attempts wedged gfx1151 MES and
restarted the OS.

#### Problem

`RawBuffer::AmdDevice` intentionally does not free automatically. Several owners
do not implement balanced teardown, including `AmdProgram` code buffers. Other
reported candidates include AQL inactive buffers, copy staging buffers, PMC
readback buffers, and construction failures after allocation.

#### Actions

- Inventory every direct `RawBuffer` owner and record its freeing authority.
- Add RAII construction guards so partial initialization cannot leak.
- Add safe Drop paths that obey HCQ-03 after failed synchronization.
- Keep process-lifetime executable caching explicit rather than relying on
  accidental leaks.
- Add allocation/free counters to the mock AMD interface.

#### Exit Gate

Successful create/drop, failed construction at every step, cache loser, graph
capture failure, and profiling teardown must balance allocations unless the
allocation is deliberately quarantined after a simulated hardware timeout.

### HCQ-07: Execute The Device-Aware Lane Topology

#### Problem

`HcqLinkedPlan::capture` computes device-aware `LaneSubmission`s with staging
legs and cross-lane waits in `runtime/src/execution_plan.rs:97-133`, but normal
execution consumes separately constructed semantic submissions. The retained
topology is currently diagnostic rather than authoritative.

#### Actions

- Make `LaneSubmission` or an equivalent per-device form the executable source
  of truth.
- Key timelines by concrete device and queue, not only generic compute/copy
  kind.
- Preserve two-leg host staging and cross-device waits.
- Ensure native and semantic replay consume the same ownership decisions.
- Remove duplicate scheduling state once the executable topology is established.

#### Exit Gate

Two-device null/mock tests must execute, not merely inspect, direct-peer and
host-staged plans with compute-to-copy, copy-to-compute, RAW, WAR, WAW, and merge
boundaries.

## Phase 7: Restore Performance And Cache Parity

### PERF-01: Align Optimizer Defaults And Cache Identity

**Status: Complete.** Strict tensor-core and fast-division defaults are typed.
Optimized-kernel and BEAM keys include optimizer behavior, exact renderer
capabilities, exact compiler identity, target, limits, and schema.

#### Problem

Svod defaults normal tensor-core matching to padded mode in
`schedule/src/optimizer/config.rs:91-113,335-352`. Tinygrad defaults `TC_OPT=0`
outside BEAM. Other behavior-affecting values such as fast division, image
rewrites, exact renderer capabilities, AMX selection, and decomposition policy
are not uniformly represented in cache keys.

The persistent BEAM key in `schedule/src/optimizer/beam.rs:578-631` lacks exact
backend/compiler identity and a schema/compiler revision. Tinygrad includes
renderer identity and cache versioning in:

- `submodules/new_new_tinygrad/tinygrad/codegen/opt/search.py:115-121`
- `submodules/new_new_tinygrad/tinygrad/helpers.py:396-435`

#### Actions

- Match pinned heuristic defaults, including strict non-BEAM tensor-core
  selection.
- Parse and model all supported optimizer controls explicitly.
- Hash every behavior-affecting optimizer value.
- Include exact renderer capability fingerprint, compiler backend, architecture,
  and cache schema version in optimized-kernel and BEAM keys.
- Invalidate old cache entries rather than attempting compatibility with a
  semantically different key.
- Add cache-cold/cache-warm differential tests.

#### Exit Gate

Switching CPU Clang/LLVM, AMX, tensor-core policy, image mode, fast division, or
renderer capabilities must never reuse an incompatible optimized result.

### PERF-02: Parallelize And De-Duplicate BEAM Work

**Status: Complete on host; hardware benchmark pending.** Candidate generation,
structural filtering, bounded compilation, exact binary/source de-duplication,
and serialized timing are separate stages with counts and timings. Deterministic
fakes and CPU BEAM numerics preserve the winner across cold and warm searches.

#### Problem

Svod evaluates candidates serially in
`schedule/src/optimizer/beam.rs:448-474`. Compilation and benchmarking occur
before duplicate and excessive-compute filtering in
`tensor/src/realize.rs:1797-1893`.

Tinygrad compiles candidates in parallel and filters duplicate libraries and
excessive compute before timing in
`submodules/new_new_tinygrad/tinygrad/codegen/opt/search.py:126-159`.

#### Actions

- Separate candidate generation, structural de-duplication, compilation,
  binary/source de-duplication, and timing.
- Reject structurally duplicate and excessive-compute candidates before compile.
- Compile independent candidates in a bounded worker pool.
- Reject duplicate binaries before device timing.
- Keep device benchmarking serialized where required by backend safety.
- Record counts and timings for each stage.

#### Exit Gate

For representative matmul and reduction searches, report generated, unique IR,
compiled, unique binary, and benchmarked candidate counts. Cold BEAM latency
must improve without changing the winning result or numeric output.

### PERF-03: Add A Persistent Compiled-Object Cache

**Status: Complete.** CPU C, CPU LLVM-text, and AMD Clang compilation now emit
validated reusable object bytes through a content-addressed cache. Exact
toolchain, target, flags, ABI, object format, and schema identity participate in
the key. Atomic writes, process locking, corruption recovery, bounded eviction,
and fresh-process warm reuse are covered by host tests.

#### Problem

Svod's optimized and executable caches are process-local. AMD invokes clang
again after every process restart through `runtime/src/amd/compile.rs:54-97`.
Tinygrad persists source-keyed compiler output in
`submodules/new_new_tinygrad/tinygrad/device.py:303-311`.

#### Actions

- Define a content-addressed persistent object key containing source/IR hash,
  target architecture, compiler identity/version, flags, ABI version, and cache
  schema version.
- Store compiled objects atomically.
- Validate object headers and target compatibility before load.
- Add bounded eviction and corruption recovery.
- Keep loaded `Program` instances device-specific even when object bytes are
  shared.

#### Exit Gate

The second fresh process compiling an unchanged model must invoke no compiler.
Changing architecture, compiler flags, ABI, or schema must produce a miss.

### PERF-04: Measure Memory Planning Versus Concurrency

**Status: Instrumented; hardware measurement pending.** Explicit Disabled,
Remap, and Arena planner modes report source/arena/peak bytes, exclusions, and
reuse. Semantic lane execution reports makespan, busy, wait, and overlap while
retaining the conservative execution-level reuse policy.

#### Problem

Svod plans by execution level and excludes transfer-associated storage in
`tensor/src/memory_planner/mod.rs:188-329`. Tinygrad uses linear positions and a
separate copy lane in
`submodules/new_new_tinygrad/tinygrad/schedule/memory.py:24-59`.

Svod's approach is conservative and may be intentional, but it can consume more
memory for wide or transfer-heavy plans.

#### Actions

- Add instrumentation for original bytes, arena bytes, peak resident bytes,
  reused allocations, and makespan.
- Build fork/join and alternating copy/compute benchmarks.
- Compare current level-based planning with a lane-aware lifetime model.
- Do not trade correctness or overlap for memory reduction without explicit
  measurements.

#### Exit Gate

Document the selected policy with measured memory and execution tradeoffs. Add
regressions for alias, view, copy, output, and same-level parallel safety.

## Phase 8: Acceptance And Rollout

### ACCEPT-01: Host Compiler And Runtime Matrix

**Status: Complete.** The pinned reference check, formatting, global diff check,
workspace all-target check, strict canonical gate, and required host-only crate
suites pass. Final counts include 587 IR, 1,173 schedule with 4 ignored, 125
codegen, 215 device with 7 ignored, 103 runtime, and 1,451 tensor with 16 ignored.

Run after all compiler remediation items:

```bash
git diff --check
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo test -p svod-dtype
cargo test -p svod-ir --lib
cargo test -p svod-macros --lib
cargo test -p svod-schedule --lib
cargo test -p svod-codegen --lib
SVOD_DISABLE_AMD=1 cargo test -p svod-device --lib --no-fail-fast
SVOD_DISABLE_AMD=1 cargo test -p svod-runtime --lib --no-fail-fast
SVOD_DISABLE_AMD=1 cargo test -p svod-tensor --lib --no-fail-fast
./scripts/check-canonical-parity.sh
```

Run broader configurations when dependencies are available:

```bash
cargo test --workspace
cargo test --workspace --features proptest
cargo test --workspace --features z3,proptest
```

Required focused gates:

- Reduced-precision constant grid.
- PARAM uniqueness and ABI ordering.
- FUNCTION symbolic output shape.
- Kernel-graph spec corpus.
- Multi-device pre-rangeify rewrite.
- Sharded all-reduce scheduling.
- Open DEVICE range splitting.
- Fast division disabled-by-default behavior.
- LLVM NaN comparisons.
- RDNA int8 WMMA object compilation.
- gfx942 ordinary FP8 decomposition.
- Volatile C/LLVM output.
- Padded `M=5` final address proof.

### ACCEPT-02: AMD Hardware Safety Matrix

Run only after packet-level and compile-only gates pass.

#### Single-XCC

- PM4 direct dispatch.
- PM4 linked-plan replay.
- PM4 graph replay below and above previous stream-size thresholds.
- SDMA copy and rollover.
- Scratch growth and allocator reuse.
- Profiled and unprofiled execution.

#### Multi-XCC

- Ordinary AQL direct dispatch.
- Linked AQL waits, barriers, control stores, and finalizers.
- Profiled graph replay with per-kernel timestamps.
- Signal-pool pressure and timeline rollover.
- Repeated allocation/free and scratch growth.

#### Multi-GPU

- Same-device native copy.
- Verified direct peer copy where capability is explicit.
- Forced host staging where peer access is absent.
- AMD-to-CPU and CPU-to-AMD copies.
- Cross-device RAW, WAR, and WAW dependencies.
- Buffer replacement that changes endpoint ownership.

Stop immediately on a KFD fault, timeout, or poisoned device. Preserve command
streams, timeline values, addresses, allocation extents, ownership metadata,
and the exact test seed before rebooting.

### ACCEPT-03: Model Correctness And Performance

Run only after ACCEPT-01 and ACCEPT-02 are green.

- ONNX reference and generated-node suites.
- AMD light-model suite.
- GigaAM encoder with profiling.
- Whisper greedy with identical settings for decoder slots 1 and 5.
- Whisper beam-5 with capacity 5 and a larger capacity.
- Medium and large-v3 on the same audio, language, strategy, fallback, and
  precision settings.
- Graph and non-graph execution.
- Profiled and unprofiled execution.
- Cold process and warm persistent-cache execution.

Record:

- Transcript or token/logit parity.
- Kernel count and generated source identity.
- Compile and BEAM latency.
- Per-kernel GPU time.
- Copy/compute overlap.
- Peak device and host-pinned memory.
- Cache hit/miss counts.

Long-audio timestamp seek remains a separate model-layer acceptance item and
must not be used to weaken compiler parity criteria.

## Recommended Commit Sequence

1. `test: make canonical schema and evidence harness operational`
2. `test: update padded m5 wmma structural acceptance`
3. `ir: commit reduced precision constants to dtype grids`
4. `codegen: assign globally unique parameter slots`
5. `ir: substitute callable result shapes and fix float bounds`
6. `schedule: add kernel graph verification boundary`
7. `schedule: move supported multi rewrites before rangeify`
8. `schedule: lower sharded reductions and device ranges`
9. `codegen: fix llvm comparisons and rdna int8 wmma abi`
10. `codegen: separate fp8 capabilities and preserve volatile access`
11. `schedule: align fast idiv and tensor core defaults`
12. `codegen: remove unsupported mlir backend`
13. `device: make aql completion multi-xcc safe`
14. `runtime: validate native copy endpoint ownership`
15. `device: make drain failure and timeline publication transactional`
16. `device: bound graph submissions and close allocation leaks`
17. `runtime: execute device-aware hcq lane topology`
18. `schedule: version optimizer and beam cache identity`
19. `schedule: parallelize and deduplicate beam compilation`
20. `runtime: add persistent compiled object cache`
21. `test: close strict production-stage canonical parity`
22. `test: run host, hardware, and model acceptance matrix`

## Progress Update Template

Use this block when closing an item:

```text
Item ID:
Target Tinygrad commit:
Svod commit:
Behavior changed:
Svod files:
Tinygrad references:
Parity fixtures:
Focused tests:
Crate/workspace tests:
Hardware tested:
Known residual risks:
Cache invalidation required:
GPU safe to test next:
Next unblocked item:
```

## Known Refuted Or Reclassified Audit Findings

These items should not be reopened without new evidence:

- Assignment WAR dependencies are not absent. Svod inserts them during kernel
  splitting in `schedule/src/rangeify/kernel.rs:325-407`; tensor scheduling
  later consumes the resulting AFTER dependencies.
- AMD and CPU executable identities now include exact target, toolchain, flags,
  ABI, and object format. Persistent objects are content-addressed, validated
  before load, and recovered after corruption.
- The current padded `M=5` graph is not known to have lost validity. The observed
  failure is a stale scalar-load assertion that prevents the actual safety
  checks from running.
- Production PROGRAM identity does contain `ProgramInfo`. The confirmed gap is
  canonical parity serialization omitting that metadata, not production
  hash-consing.
- Svod's single-axis Multi representation is intentionally narrower than
  Tinygrad's UNSHARD model. The remediation requirement is to make supported
  behavior correct and unsupported behavior explicit, not to silently claim
  complete multi-axis parity.
- `DeviceSpec` cannot encode Tinygrad's tuple-valued collective destination.
  The supported host all-reduce subset therefore returns one in-place result on
  shard zero after reading every shard; replicated outputs and native ring or
  all-to-all strategies remain explicit hardware/runtime follow-ups.
- The low-level scheduler and execution plan retain every expanded lane output,
  but public `Tensor` and `TensorEntry` ownership still hold one buffer and
  `buf_uop()` selects lane zero from MSTACK. Sharded public output publication
  requires a distinct multi-buffer Tensor representation and is not inferred by
  the lane scheduler.

## Immediate Next Actions

Perform hardware-backed PERF-04 and ACCEPT-02 on suitable single-XCC,
multi-XCC, and multi-GPU systems.

Run hardware-backed PERF-04 measurements and ACCEPT-02 only after the host
compiler gates are clean. Keep multi-XCC and physical multi-GPU tests reserved
for suitable hardware; forced AQL on gfx1151 remains isolated and serial.
