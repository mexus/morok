# PR #142 Review Remediation Plan

Action plan for the 111 confirmed/plausible findings from the `/code-review --max`
of `compiler/tinygrad-uop-adoption` vs `model_whisper_continous_batching`.
Every item was re-verified by a subsystem agent against the tinygrad checkout
and the current morok code; disputes against the original review are recorded
inline as **Review correction**.

Companion docs: `TINYGRAD_PARITY_REMEDIATION_PLAN.md` (host-parity backlog, pins
`submodules/new_new_tinygrad @ 8c8b43de`), `HCQ_PORT_LEDGER.md` (AMD HCQ intent).

## Ground rules

- **Adopt, don't patch.** Where tinygrad has the construct, port its shape
  (file:line cited per item). Where morok deliberately diverges and the
  divergence is justified, the action is *keep + document* (one comment line
  naming the tinygrad location and the reason) — not a silent difference.
- **Tests cost as little code as possible.** Add `#[test_case]` rows to an
  existing fn before writing a new fn; write a new fn before a new file; reuse
  the helpers listed in §Shared test infrastructure. No test in-place
  `#[cfg(test)] mod` (CLAUDE.md).
- **One concern per commit**, ordered as in each phase. Gate: `cargo fmt`,
  `cargo clippy --workspace --all-targets` (0 errors), `cargo test`.
- Effort: S ≤ 1 h, M ≤ half day, L > half day. Risk = blast radius on
  generated kernels / runtime behaviour.

## Phase map

| Phase | Scope | Items | Must precede |
|---|---|---|---|
| P0 | Gating + reference pin | CV1 CV4 SS2 CV2 | everything (CV1 unblocks clippy) |
| P1 | IR constants & dtypes | IB1 IB2 IC2 IC3 IB3 IB4 SM2 | P2, P3, P6 |
| P2 | Symbolic | XB6 SA1 SA2 SC2 EC5 XB3 SB6 SB2 | P3 |
| P3 | Rangeify | RB2 RB4 RB5/RC3 RA1 RA2 RB1 RC1 RC2 RA4 IC1 DA3 | — |
| P4 | gpudims / spec / linearize | DC5 SB1 SA4 SA5 DB4 SB3 SA3 SB4 EC3 CC2 EC1 DB2 | — |
| P5 | devectorize / optimizer | DA2 DA1 DA4 DC2 DC1 DC3 DC4 EC2 DB1 DB5 DC6 | — |
| P6 | C / LLVM codegen | CA1 CA5 CB2 CA2 CA4 CA6 CB1 CB3 CA7 XB5 CA3 | — |
| P7 | tensor / model / onnx | TA2 TB1 TC1 TA3 TC2 TA8 TA4 XB4 TC5 TC3 TC4 | — |
| P8 | runtime | UC2 UA5 UC3 UA3 UA2 UA4 UB1 UC1 UB2 UA1 UA6 UC4 ES1 RS1 ES4 TA6 TA5 CV3 | — |
| P9 | AMD device | VA5 VA3 VA2 VC3 VB3 VB2 VB5 VC2 VA6 VB6 VA1 VA4 VC4 ES2 VC5 VC6 | — |

P1→P2→P3 is the only hard chain. P4–P9 are independent of each other and can
be parallelised by subsystem.

## Decisions needed before starting (owner: reviewer)

| # | Decision | Recommendation |
|---|---|---|
| D1 | Which tinygrad is the reference? `submodules/tinygrad @ 1f8b24a6b` (2025-11-21, all citations below) vs the parity plan's `new_new_tinygrad @ 8c8b43de` (absent on disk, gitlink deleted). | Restore ONE tracked gitlink; if 8c8b43de, re-check every "Tinygrad:" citation whose file moved. |
| D2 | RC1: delete the gate-based range narrowing (tinygrad has none) or repair it to scan all indices. | Delete; re-add only with a benchmark showing the loss. |
| D3 | VB3: stop panic-unwind from poisoning the process-global AMD core (tinygrad only poisons on drain timeout / fault). Weakens HCQ-03/04 latch. | Adopt tinygrad; quarantine already isolates the lane. |
| D4 | UC1: emit `@llvm.*` intrinsics on AMD (tinygrad) so `-nogpulib` returns, leaving f64 transcendentals on ocml. | Adopt intrinsics. |
| D5 | UC4(b): `CompilerProcess`/`start_compile_process` is implemented on 3 devices but has zero callers — wire into BEAM compile wave or delete. | Delete unless BEAM needs it this quarter. |
| D6 | TC2: restore the whisper `decoder.token_embedding.weight` checkpoint-dtype exemption (only observable at FP8). | Restore. |
| D7 | VB6: restore only the single-queue half (1 MemoryBarrier per graph) now; defer cross-queue `opt_deps` narrowing until multi-queue graphs are captured. | Yes. |
| D8 | CA7: keep ordered fcmp predicates (matches morok's C backend) instead of tinygrad's unordered. | Keep + document + rename test. |

---

## P0 — Gating and reference pin

**CV1** clippy hard error blocks linting of the whole workspace (`svod-ir` fails, so
nothing downstream is linted). `cargo clippy --workspace --all-targets 2>&1 | grep -c '^error'` = 3.
- Fix: `ir/src/decompositions/transcendentals.rs:494` → `use std::f64::consts::FRAC_1_PI`, replace the literal. Also clear the 3 free warnings surfaced: `shape.rs:359` needless lifetime, `types.rs:720` redundant closure, `uop/hash_consing.rs:102` `large_enum_variant` (box `ProgramInfo`).
- Test: clippy count 0. S/low. **Land first, alone.**

**CV4** `.tokeignore` unchanged while 3020 lines of parity harness were added.
- Fix: append `scripts/`, `ir/examples/`, `tensor/examples/`.
- Test: `tokei` delta. S/low. Ride with CV1.

**SS2** parity harness can't run from a clean checkout (4 sub-defects, all verified).
1. `scripts/check-canonical-parity.sh:7-13` pins `submodules/new_new_tinygrad` (untracked, absent) → repoint `REFERENCE` at the tracked submodule (per D1).
2. `submodules/tinygrad` gitlink deleted but `.gitmodules` stanza remains → `git add submodules/tinygrad` at the chosen pin, or drop the stanza (the branch did the consistent thing for `melior`).
3. `scripts/CANONICAL_KNOWN_GAPS.txt` is 0 bytes → populate or make the script treat absence explicitly.
4. Harness isn't invoked anywhere → opt-in CI target once 1–3 land.
- Test: `bash scripts/check-canonical-parity.sh` passes line 13 in a fresh clone. M/med.

**CV2** 7 in-place `#[cfg(test)] mod` in 6 files (review said 5; `implicit_barriers.rs` was missed). Moves:
- `schedule/src/optimizer/mod.rs:145,257` (14 tests) → `schedule/src/test/unit/optimizer/mod_internal.rs` (needs `pub(crate)` on `LocalBufferContext`, `extra_symbolic_patterns`, `lower_index_patterns`)
- `schedule/src/optimizer/implicit_barriers.rs:107` (10) → `test/unit/optimizer/implicit_barriers.rs`
- `device/src/amd/connector.rs:748` (2), `linked_plan.rs:575` (3), `program.rs:494` (3) → `device/src/test/unit/amd/{connector,linked_plan,program_abi}.rs`
- `runtime/src/object_cache.rs:399` (4) → `runtime/src/test/unit/object_cache.rs`
- Test: same pass count (36). M/low. **Land last in the whole plan** — it conflicts with every `optimizer/mod.rs` edit.

---

## P1 — IR constants and dtypes

**IB1 + IB2 + IC2 + IC3 — one changeset, in this order.**

**IB1** bf16 VCONST folding aborts on f32-range overflow (`ConstValue::truncate` expects on `commit_float` → `None`).
- Tinygrad: `dtype.py:126-131` `as_const` is total; `float_to_bf16` (`dtype.py:244`) only runs at render/tensor-creation, never in const folding.
- Fix: make the bf16 arm of `commit_float` (`dtype/src/cast.rs:23-25`) total — finite values at/over the `0x47ef_ffff_f000_0000` midpoint return `±inf`; delete the now-redundant special case in `commit_eval_result` (`ir/src/uop/eval.rs:562-570`). `truncate` becomes total for all scalar dtypes.
- Test: flip the two pins in `dtype/src/test/dtype_parity.rs:155` to `Some(INFINITY)`; one row in `ir/src/test/unit/vector.rs`: bf16 `VCONST[3e38;2]` squared → `[INF;2]`. S/low.

**IB2** `UOp::const_` panics on `CAST(CONST(f64,1e300))→bf16` fold. Subsumed by IB1 (`try_const_` then only fails for pointer/Void = tinygrad's contract).
- Test: `ir/src/test/unit/uop.rs:44` — replace the bf16 error row with a value row asserting `Float(INFINITY)`. S/low.

**IC2** `const_like` drops the receiver's vector count → every `x.const_like(v)` on a Vector dtype fails promotion (`neg()`, `rne`, `f2f_clamp`, `cast_float_to_bf16` on vectors abort).
- Tinygrad: `uop/ops.py:376-378` `const_like` = `UOp.const(self.dtype, b)`.
- Fix: `ir/src/uop/constructors/data.rs:83` pass `self.dtype()` not `.scalar_dtype()`; `try_const_` already normalises against a Vector's scalar.
- Test: `ir/src/test/unit/uop.rs:74` +2 lines: `UInt16.vec(4)` const → `const_like(1).dtype() == v.dtype()`, `v.neg().dtype() == v.dtype()`. S/med.

**IC3** `vconst_like` expects a bounded shape; gater calls it unconditionally.
- Tinygrad: `codegen/late/devectorizer.py:268-272` uses plain `x.const_like(0)`.
- Fix: after IC2, delete `vconst_like` (`data.rs:102`) and switch `schedule/src/late/gater.rs:35/67/95/146` to `const_like`.
- Test: `schedule/src/test/unit/late_coalesce.rs` — gated LOAD off an unshaped INDEX rewrites to `Load{alt: Some(Const)}`. S/low.

**IB3** `shape_to_uop` emits `CAST(CONST)` lanes for mixed const/symbolic shapes; `SInt::simplify` can't unwrap → `[Const(3), Symbolic]` reads back fully symbolic.
- Tinygrad: `uop/ops.py:1253` `sint_to_uop(x, dtype)` builds `UOp.const(dtype, x)` at the target dtype.
- Fix: `ir/src/shape.rs:504-524` compute the lane dtype once, materialise every dim via `dim.to_uop(lane_dtype)`; also let `SInt::simplify` (`ir/src/sint.rs:195`) look through `Cast{src: Const}`.
- Test: `ir/src/test/unit/shape.rs:209` +3 lines with `UOp::var("n",Int32,1,8)`; assert `[0] == SInt::Const(3)`. S/low.

**IB4** INDEX dtype guard is symmetric weak-collapse → Float32 over Float16 buffer accepted.
- Tinygrad: `uop/ops.py:366-367` dtype is exactly the buffer's base dtype.
- Fix: `ir/src/uop/constructors/memory.rs:81` → `result == inferred || (result.is_weak() && result.weak_dtype() == inferred.weak_dtype())`.
- Test: `ir/src/test/unit/dtype_rule.rs:100` +2 lines: Float16 buffer + `.dtype(Float32)` → `is_err()`. S/low.

**SM2** `AxisId` manual `Ord` says `Unrenumbered(3) == UnrenumberedPath([3])` while derived `PartialEq`/`Hash` disagree; `tc.rs:554` sorts-then-dedups.
- Fix: `ir/src/types.rs:1041` — manual `PartialEq`/`Hash` over the same `(is_renumbered(), path())` key.
- Test: `ir/src/test/unit/canonical.rs` +3 lines: `cmp==Equal ⇔ ==`, equal hashes. S/low.

---

## P2 — Symbolic

Order: XB6 → (SA1, SA2, SC2) → EC5, XB3, SB6 → `UOp::gcd` → SB2.

**XB6** `CAST(CONST)` folding moved out of `symbolic_simple` into standalone `pm_fold_cast_const`, so devectorize/pm_add_loads/pm_add_images no longer fold it.
- Tinygrad: `uop/symbolic.py:96` inside `symbolic_simple`.
- Fix: move the rule into `cast_dsl_patterns()` (composed by `symbolic_simple_base`, `schedule/src/symbolic/patterns.rs:574`); delete `pm_fold_cast_const` and its 8 `+ pm_fold_cast_const()` sites (`optimizer/mod.rs:249/1071/1209`, `rangeify/transforms.rs:560/1305/1397`, `rangeify/patterns.rs:1835`, 3 test helpers).
- Test: `schedule/src/test/unit/symbolic/mod.rs:543` repoint the 3 matcher bindings at `symbolic_simple()`; +1 row: devectorize output has no `Cast{src: Const}`. M/med — run full schedule suite.

**SA1** `parse_valid` `NOT(X<c)` arm uses `c.vmax` → unsound `x >= begin.vmax` for symbolic pad begin.
- Tinygrad: `uop/symbolic.py:267-269` uses `vmin`.
- Fix: `schedule/src/symbolic/valid_simplification.rs:41` `c_vmax` → `c_vmin`.
- Test: `symbolic/mod.rs` +3 lines: `var b∈[2,9]`, `parse_valid(NOT(r<b)).2 == 2`. S/low.

**SA2** `propagate_invalid` wraps comparisons in WHERE-Invalid; tinygrad drops the gate for comparisons.
- Tinygrad: `uop/symbolic.py:33-35`.
- Fix: `patterns.rs:447,:460` restrict to the non-comparison list already at `:474`; add a sibling rule over `[Lt,Le,Eq,Ne,Gt,Ge]` returning the bare binary. Add tinygrad's Index-dtype guard on the `invalid ⊕ y → invalid` rule.
- Test: +3 lines in the existing propagate_invalid fn: `WHERE(c,r,INV) < k` → `Binary(Lt)`. S/low.

**SC2** `pm_remove_invalid` rewrites Index-dtype `WHERE(c, idx, INV)` to `WHERE(c, idx, 0)` → unconditional access to element 0 when the gater skipped it.
- Tinygrad: `uop/ops.py:1271` `pm_lower_index_dtype` is the only sanctioned lowering; no `→0` for addresses.
- Fix: `patterns.rs:488` guard the WHERE and STACK arms on `!dtype.is_index()` (the doc already claims this).
- Test: `symbolic/index_lowering.rs` +2 lines: Index-dtype WHERE is returned `ptr_eq`. S/low.

**EC5** `simplify_valid_load` runs `graph_rewrite(symbolic())` before the `ptr_eq` early-out.
- Tinygrad: `devectorizer.py:14-15` returns on `idx is start_idx` first.
- Fix: `schedule/src/late/coalesce.rs:132-135` hoist the pointer check. Existing tests cover. S/low.

**XB3** `vmin_vmax_collapse` Add/Sub/Max exclusion is intentional but untested (test deleted).
- Tinygrad: `symbolic.py:210` collapses every ALU — morok's exclusion is a documented divergence.
- Test: `symbolic/mod.rs` one fn: `a[2,2]*b[3,3]` → Const, `a+b` stays Add. S/low.

**SB6** reciprocal distribution absent from `sym()`. **Review correction:** justified divergence — all six rules are IEEE-inexact and `unknown_float_division_power_and_reciprocal_are_not_algebraically_rewritten` (`symbolic/mod.rs:3285`) pins the non-rewrite.
- Fix: delete "reciprocal distribution" from the `sym()` doc (`patterns.rs:667`), add one sentence naming `symbolic.py:394-399` and the pinning test. S/low.

**SB2** `fold_divmod_general` is a dead stub; `(N*i+j)//N` with symbolic N no longer folds.
- Tinygrad: `uop/divandmod.py:8-96` (cancel_divmod, remove_nested_mod, fold_binary_numerator, congruence, gcd_with_remainder, nest_div_by_smallest_factor, divide_by_gcd, factor_remainder) wired at `:107`; negative-normalisation rules `:99-101,109-110`.
- Prereq commit: symbolic multi-arg `UOp::gcd` (`uop/ops.py:715-718`) in `ir/src/uop/helpers.rs` next to `divide_exact`.
- Fix: port into `schedule/src/symbolic/divmod.rs`, keeping `fold_divmod_congruence` as its congruence branch and `exact_integer_rewrite` as morok's no-wrap proof; replace the ad-hoc const-divisor rules at `patterns.rs:760-900` with one `(Idiv|Mod) → fold_divmod_general` rule + the 4 normalisation rules.
- Test: `symbolic/fast_div_internal.rs` `#[test_case]` rows: `(N*i+j)//N → i` (N var 1..64), `x//d → -(x//-d)` for `d.vmax<0`, `%0` errors. L/med — last in P2; run symbolic proptests + z3 suite.

---

## P3 — Rangeify

Order: RB2 → RB4+RB5 → RA1 → RA2 → RB1 → RC1 → RC2 → image guards (RA4, IC1, DA3).

**RB2** `STAGE(COPY)` became removable → dead-axis pruning shrinks the copy destination (`[4]→expand([4,8])→to(Amd)` allocates 4 elements).
- Tinygrad: `rangeify.py:120-124` `ALWAYS_RUN_OPS = {CONTIGUOUS, COPY, ASSIGN, NOOP}`; `indexing.py:9-11` COPY ∈ `ALWAYS_CONTIGUOUS`.
- Fix: `schedule/src/rangeify/patterns.rs:297` use the existing `is_always_run_op(op) || After` predicate (shared with `:398`); `indexing.rs:341` add `Op::Copy` to `is_always_contiguous`.
- Test: `test/unit/rangeify/patterns.rs` near :208, one `#[test_case]` row: STAGE over `copy_to_device(Amd)` with a dead range → `NoMatch`. S/low.

**RB4** SINK and BUFFER_VIEW realize rules deleted; `rangeify(sink([a*b]))` yields 0 stages.
- Tinygrad: `indexing.py:26,28,30`.
- Fix: `pm_generate_realize_map` (`indexing.rs:291`) add the SINK row (`mark_realize_all` on each source base not `is_always_contiguous`), add `Op::Slice` to the always-realize set, re-add the REDUCE-on-OUTER row if missing; order SINK last.
- Test: `test/unit/rangeify/pipeline.rs` +2 lines: `run_rangeify(sink([a*b]))` contains a `Stage`. S/med — expect kernel-count churn in `kernel_count.rs`.

**RB5 / RC3** PAD/ReduceAxis leak post-condition removed and `spec_tensor` whitelists PAD.
- Tinygrad: no recovery loop — the invariant is that neither survives `pm_apply_rangeify`.
- Fix: do NOT reinstate the retry loop; add the cheap post-condition at the end of `run_rangeify` (`indexing.rs:215`) returning `UnsupportedOperation`, and drop `Op::Pad`/`Op::ReduceAxis` from the whitelist in `spec.rs:744`. Same commit as RB4. RC3 is a duplicate — range-map fallbacks are already tinygrad-shaped.
- Test: `#[test_case]` rows over pad shapes on an existing pad-pipeline fn: no `Pad|ReduceAxis` in output. S/low.

**RA1** `push_op_through_after` tag placement inverted → tagged `AFTER(RESHAPE(buf),[STORE])` outputs come back unreshaped.
- Tinygrad: `rangeify.py:22-23` inner AFTER `tag=None`, outer movement node `tag=a.tag`.
- Fix: `transforms.rs:667-676` swap: `after.with_sources(..).rtag(None)`; `r.with_sources(..).rtag(after.tag().clone())`.
- Test: `test/unit/rangeify/movement_patterns.rs` one fn: out is `Reshape` with the tag, inner AFTER untagged. S/low.

**RA2** same-device COPY elision returns the source verbatim, dropping the COPY's tag.
- Tinygrad: `rangeify.py:103` `x.f(Ops.NOOP, tag=copy.tag)`.
- Fix: morok's `Noop` is nullary (used ~90 places) so an exact port is invasive; return `src.rtag(merge(src.tag(), copy.tag()))` in `early_rewrites` (`patterns.rs:91`) and document why (barrier role is covered by RB2's `is_always_run_op(Copy)`).
- Test: `test/unit/rangeify/patterns.rs` +3 lines: tagged same-device COPY → not Copy, tag preserved. S/low.

**RB1** per-CALL device uniformity check deleted; `sink(contiguous(cpu*amd))` compiles for Cpu with an AMD pointer.
- Tinygrad: `rangeify.py:505-506` raises in `split_store`.
- Fix: re-add `validate_normal_kernel_devices(&after_split)?` in `try_get_kernel_graph` before `fix_assign` (body recoverable verbatim from `git show 954efea1^:schedule/src/rangeify/kernel.rs` L330-357); returns the still-defined `KernelSplitMixedDevices`.
- Test: `test/unit/rangeify/device_semantics.rs` one fn: Cpu×Amd buffers → `Err`. S/low.

**RC1** gate-based range narrowing in `pm_simplify_ranges` only inspects `indices[0]` → ranges in `indices[1..]` get shrunk by another INDEX's tighter bound.
- Tinygrad: `codegen/simplify.py:39-40` `pm_simplify_ranges` is only `simplify_merge_adjacent` — no gate collection exists.
- Fix (D2): delete `mark_gated`/`protect_reduce_ranges`/`substitute_simplified_ranges`/`SimplifyRangesContext` (`transforms.rs:1322-1400`) keeping the two `simplify_merge_adjacent` rows; drop `bounded_load_narrows_range`, `conflicting_gates_choose_largest_bound`, `reduce_range_is_protected`. Fallback if perf forbids: harvest every index and pin every index's ranges.
- Test: `test/unit/rangeify/simplify_ranges.rs` one fn (reuse `buffer()`, `narrowed_end`, `simplify`): narrow INDEX on r + wide 2-index INDEX using r → `narrowed_end(r) == 16`. M/med.

**RC2** `pm_split_ranges` lost image-STORE range protection.
- Tinygrad: `simplify.py:56-63` `dont_sub_ranges_for_image` sets `ctx[r] = None`; `do_substitute` skips `None`.
- Fix: `SplitRangesContext.marked_ranges: HashMap<u64, Option<i64>>` (`transforms.rs:420`); add a `Store` row inserting `None` for image-buffer index ranges; skip `None` in the substitute.
- Test: `test/unit/rangeify/split_ranges.rs` one fn with a `DType::Image` Noop buffer: ranges unchanged. S/low.

**Image guards (one commit):**
- **RA4** `linearize_static_indices` (`transforms.rs:787`) — add `if buffer.dtype().is_image() { return None; }` (tinygrad `devectorizer.py:176-191` never flattens image coords). Test: `test/unit/rangeify/indexing.rs` via `transform_single_source` → 2-index INDEX. S/low.
- **IC1** `gater.rs` `move_image_load/store` hard-code Float32 with no image guard → panic on non-f32 2-index INDEX. Fix: `image_gate` returns `None` unless `buffer.dtype().is_image()`; rebuild INDEX with `index.dtype()` like `move_load` (`gater.rs:100`). Test: `test/unit/devectorize/late_gater.rs` Int32 2-index gated load → `NoMatch`. S/low.
- **DA3** `apply_image_upcasts` keys off shape. **Review correction:** `image_valid_dims` / `IMAGE` env don't exist at this pin — only the `ImageDType` guard is a divergence. Fix: `heuristics.rs:222` `if !buffer.dtype().is_image() { continue; }`; check `unrollable` membership before the fallback arm (`heuristic.py:51-60`). Test: `test/unit/optimizer/heuristics.rs` negative row: rank-3 Float32 → `false`. S/low.

---

## P4 — gpudims / spec / linearize / coalesce

Order: batch A (DC5-swap, SB1, SA4, SA5, DB4) → SB3 → SA3 → SB4 → EC3 → CC2 → EC1 → DB2 → DC5 const/var arms (separate commit).

**DC5** `priority()` gives Local −17 / everything-else −18 (inverted), and is missing `DefineVar=-19`, `Const=-10`.
- Tinygrad: `codegen/late/linearizer.py:29-38`.
- Fix: `linearize/linearize.rs:788` Local→−18, add `Reg→−17`. Separate commit: `DefineVar→−19`, `Const→−10` (reorders every prologue; churns rendered goldens).
- Test: `test/unit/linearize/linearize_internal.rs:203` flip the two asserts; +1 line `priority(const) == (-10, None)`. S/low (const arm med).

**SB1** `add_gpudims` lost the `has_threads` guard and `i < all_idxs.len()` bound → panics on Renderer::cpu with >1 global.
- Tinygrad: `gpudims.py:68` THREAD folds into the ordinary global path; CPU renders `gidx0` as `core_id` (`cstyle.py:214`).
- Fix: restore `return None` (+ `debug_assert!`, `tracing::warn`) when `global_dims.len()!=1 || !local_dims.is_empty()` or no concrete bound; restore the bound check. Follow-up (renderer): delete the branch and render `core_id` like tinygrad.
- Test: `test/unit/gpudims_internal.rs` +2 lines: two thread ranges on `Renderer::cpu()` → `None`. S/low.

**SA4** `global_prod_max` cap divides by a per-axis local extent that can be 0; `hw_local` empty when locals are div/mod expressions.
- Tinygrad: no product cap (`gpudims.py:84`; AMD `cstyle.py:432` flat per-axis). Keep as a documented AMD divergence (dispatch grid is in work-items).
- Fix: `gpudims.rs:201` divide by `local.max(1)`; derive `hw_local` from the `Special` leaves of the local index expressions (toposort, `lidx*` by name).
- Test: `gpudims_internal.rs` rows on `test_global_product_cap_accounts_for_local_extent`: local 0 no panic; contracted `[64,64,64,4]` on `amd_cdna3` still caps. S/low.

**SA5** `memory_coalescing` release `assert!`s on gated load/store.
- Tinygrad: `devectorizer.py:130-174` carries gated srcs through untouched.
- Fix: `late/coalesce.rs:314/318` → `debug_assert!` + `continue` (skip coalescing that access); same for "multiple stores"; keep the `:397` expect (provably unreachable).
- Test: `test/unit/late_coalesce.rs` +2 lines: gated LOAD survives `target_coalesce` as 1 load. S/low.

**DB4** AMX `[16,8,4,2]` fold widths dropped.
- Tinygrad: `devectorizer.py:140-152`.
- Fix: `coalesce.rs:355-365` push `[16,8,4,2]` when `ctx.device.is_apple_amx()` (predicate exists at `renderer.rs:391`). DSP arm: keep + document (no DSP renderer).
- Test: mirror `shaped_width_eight_load_uses_two_scalar_dtype_width_four_accesses` with 16 offsets on `Renderer::apple_amx()` → 1 load, shape `[16]`. S/low.

**SB3** `compute_store_masks` global-store test is one-level; misses `Stack([Param])` INDEX buffers.
- Tinygrad: `uop/ops.py:616-624` `buf_target()` recurses AFTER|INDEX|STORE|LOAD|VECTORIZE.
- Fix: one shared `fn buf_target(&Arc<UOp>) -> Option<&ParamArg>` in spec.rs (hoisted from `rule_getaddr`, `spec.rs:387-394`), used by `gpudims.rs:262`.
- Test: `gpudims_internal.rs` `#[test_case]` row on `test_missing_group_reduce_masks_structured_global_param_store` with `UOp::stack([param,param])` → mask present. S/low.

**SA3** `rule_end` second arm dead (RANGE is never Void); Bool/Void backedge END from `split_end_with_tag` rejected by `spec_program`.
- Tinygrad: `spec.py:193` program spec only `END(x, RANGE)`; `linearizer.py:93-96` drops non-RANGE srcs when splitting.
- Fix: replace the dead arm with `End{ranges} if ranges.iter().all(|r| matches!(r.dtype(), Void|Bool))`; fix the "or SPECIAL" doc.
- Test: `test/unit/spec.rs` +2 lines: `noop.end([range]).end([Bool const])` verifies under `spec_program()`. S/low.

**SB4** `get_grouped_dims` lost identity/contraction/expansion branches → symbolic divisor leaves permanent `FloorDiv/FloorMod`.
- Tinygrad: `gpudims.py:38-57` four exits; identity early-return at :57.
- Fix: restore `if dims_eq(&limited, dims) { return raw_idxs }` and the `limited.len() > dims.len()` expansion arm; keep the shared flatten helper for the two decomposing branches.
- Test: `gpudims_internal.rs` +2 lines: `get_grouped_dims("gidx", [var_n, 8], None, true)` all `Special`. M/med — gpudims goldens move.

**EC3** `type_verify_call_aware` allocates path/children Vecs per node for a failure-only diagnostic.
- Tinygrad: `spec.py:262-270` plain toposort list.
- Fix: plain `Vec<Arc<UOp>>` walk; rebuild `source_path` in a second walk only on failure. Also produce `pub fn type_verify_list(&[Arc<UOp>], &Spec)` for CC2.
- Test: existing `preoptimization_rejects_malformed_dtype_before_rewrites` still asserts the same `source_path`. M/low.

**CC2** `type_verify` moved off the linearized list → IF/ENDIF from `line_rewrite_cleanups` and isel rewrites unverified; `rule_if` (`spec.rs:580`) dead.
- Tinygrad: `codegen/__init__.py:138-141` verifies `lst` after `line_rewrite`.
- Fix: call `type_verify_list` in `do_linearize` right after `line_rewrite_cleanups`; keep the pre-linearize gate.
- Test: `codegen/src/test/unit/program_pipeline.rs`: gated-store program → Ok; hand-built LINEAR with bare `If` → Err. M/med — last of the spec work.

**EC1** `tinygrad_tuplize_cmp` memoises nothing across calls.
- Tinygrad: `uop/ops.py:187-189` `cached_property tuplize`.
- Fix: `thread_local! HashMap<(u64,u64), Option<Ordering>>` with size cap (ids are monotonic, `hash_consing.rs:52`). Follow-up: cached `TuplizeKey` on UOp in `svod-ir`.
- Test: existing ordering tests. S/low.

**DB2** `compare_tuplize` unbounded recursion (8 MiB stack overflow ~20-30k depth).
- Fix: iterative worklist `Vec<(Arc<TuplizeKey>, Arc<TuplizeKey>)>`; keep the memo; do NOT reinstate `MAX_KEY_LEN=128` (non-total order).
- Test: `linearize_internal.rs:216` reuse `test_tuplize_is_recursive_past_128_elements` with 40 000 and a 2 MiB thread. M/low.

---

## P5 — Devectorize / optimizer

Order: DA2 → DA1 → DA4 → DC2 → DC1 → DC3 → DC4 → EC2 → (DB1, DB5, DC6).

**DA1 / DA2** 64-bit SHL/SHR word-split wrong for shifts ≥ 32. **Review correction:** tinygrad has no 64→32 split (`decompositions.py:338-343` is int MUL/IDIV→shift); `pm_long_decomp` is Svod-only for no-native-i64 targets — document as such. Bugs are real.
- Fix: `devectorize.rs:1108-1110` (Shl) `word==1` branch selects `low = a0 << n` (n = b0&31) not raw `a0`; use `n` in `high`. `:1131-1133` (Shr) `long_bin(Shr, a1, n)` not `b0`.
- Test: new `schedule/src/test/property/long_shift.rs` proptest `(x: u64, s: 0..64)` evaluating both words via a local `eval_word` (copy of `symbolic_props.rs:641` + Where/Cast/BitCast arms) against native `<<`/`>>` (Int64 and UInt64). Smoke rows `#[test_case(0|31|32|33|63)]` in `devectorize/fp8_decomp.rs`. S/low.

**DA4** `stack_with_shape` `chunks(0)` panic on trailing-zero shapes.
- Fix: `devectorize.rs:1366` `if chunk == 0 || count*chunk != len { return None }`; make `pub(crate)`.
- Test: `devectorize/edge_cases.rs` +2 lines: `(vec![], [4,0]) → None`, `(4 elems, [2,2]) → Some`. S/low.

**DC2** `pm_wmma_add` folds via panicking `c.add(add)`.
- Tinygrad: `devectorizer.py:313-315` guard-free, dtype assert inside `alu`.
- Fix: `try_add(..).ok()?` in all three arms; no explicit pre-check.
- Test: `devectorize/new_patterns.rs`: mismatched Add via `with_sources` → no panic, WMMA unfused. S/low.

**DC1** `apply_padto` lost the ADD-reduce / UnsafePad guard → BEAM_PADTO/TC_OPT≥2 pads MAX/MUL reduces.
- Tinygrad: `postrange.py:192-193`; `GroupOp.UnsafePad = {RECIPROCAL, LOG2, EXP2, IDIV, POW}` (`uop/__init__.py:131`) — note the base's guard also rejected `Lt`, which tinygrad allows.
- Fix: in `opts.rs` after the axis-type check: for Reduce|GroupReduce require `reduce_op == Add` and no `Unary(Reciprocal|Log2|Exp2) | Binary(Idiv|Pow)` in `r.backward_slice()`.
- Test: `optimizer/opts_validation.rs` `#[test_case(Add,true)] #[test_case(Max,false)]` + an `Exp2`-above-reduce row → false. M/low.

**DC3** BEAM cache key omits the action space (BEAM_PADTO/TC/TC_OPT). Tinygrad (`search.py:121`) has the same hazard — this is a justified go-beyond.
- Fix: `action_space_hash(&BEAM_ACTIONS)` as a `CacheKey` field; append in `to_bytes`; bump schema 7→8.
- Test: `optimizer/beam_internal.rs` +2 lines: hash differs for `BEAM_ACTIONS[1..]`; key bytes differ. S/low.

**DC4** `DISABLE_FAST_IDIV` defaults to 1 (tinygrad `helpers.py:176` → 0); `beam_search_cached` hardcodes fingerprint 0.
- Fix: `config.rs:517` `false`; env parses `.unwrap_or(0) != 0`; route `beam_search_cached` (`beam.rs:978`) through the `_with_behavior` variant or mark it test-only.
- Test: `optimizer/config_internal.rs:98` flip; +2 rows: `x/3` rewritten away from raw `CDiv` when enabled. S/med — run symbolic proptests + z3.

**EC2** `devectorize()` wraps the combined matcher in an outer fixed-point loop.
- Tinygrad: `codegen/__init__.py:80` single `graph_rewrite`.
- Fix: `devectorize.rs:258-268` one call; if a second pass is load-bearing, that is a missing pattern.
- Test: `devectorize/pipeline.rs` +2 lines idempotence: `apply(apply(x)) == apply(x)`. S/low.

**DB1** `Op::Special` hand-lowered bypass removed (aligned). Finish: delete `symbolic_no_dead_loop` (`symbolic/patterns.rs:657`, re-export `mod.rs:15`), fix 3 stale comments (`tk/src/kernel.rs:217`, `tk/src/launch.rs:448`, `tk/src/test/unit/fa.rs:121`). Gate on a GPU run of the `#[ignore]`d tk/tk2 suite. Test: `optimizer/opts_to_apply.rs` Special + empty opts → succeeds. M/med.

**DB5** cdna3 fp32 16x16x4 MFMA removal (aligned, `tc.py:126`). Finish: delete `AMD_CDNA_16164`, `TcConfig::build_beam_only`, `heuristic_pick`, the `heuristic_pick=false` branch (`heuristics.rs:1023-1040`), `f32_mfma_config` + its 2 ignored tests (`tensor/src/test/unit/matmul.rs:1331+`). Test: `renderer_internal.rs` +2 lines: 4 cores, none Float32. M/low.

**DC6** `tc_opt` default Padded→Strict = tinygrad `TC_OPT=0`. Keep + document on `HeuristicsConfig::tc_opt` (`config.rs:308`); set explicitly in GEMM benches. Test: `test_heuristics_config_default` +1 line. S/low.

---

## P6 — C / LLVM codegen

Order: shared helpers → CA1 → CA5 → CB2 → LLVM cluster → CA7, XB5, CA3 → upstream follow-up.

**Shared (first commit):** move `access_width`/`value_width`/`shaped_dtype` from `codegen/src/c/types.rs:200-232` to `codegen/src/common.rs`; add `assert_c_compiles(src)` in `codegen/src/test/unit/c.rs` modelled on `assert_llvm_ir_assembles` (`llvm_text.rs:597-647`): `clang -fsyntax-only -x c -`, skip if clang absent (~15 lines).

**CA1** C STACK lane-deref uses `is_storage_source`, disagreeing with INDEX's `addrspace()` decision → `STACK(INDEX(SHRINK(param),i))` renders pointers in a float4 literal.
- Tinygrad: `cstyle.py:46-47` INDEX is always an address; lane reads are GEP.
- Fix: `c/ops.rs:429` `if matches!(op, Index) && source.addrspace().is_some() { "*({v})" }`; delete `is_storage_source`.
- Test: `c.rs` Stack of 4 `Index(Shrink(param0,0,4), i)` → contains `(float4){*(`; `assert_c_compiles`. S/low.

**CA5** C STORE derives width only from the index → `*(data0+0) = (float4){..}`.
- Fix: `c/ops.rs:281` `width = value_width(value).max(access_width(index))`; same rule LOAD uses at `:245`.
- Test: `Index(param,0).store(VConst[1.0;4])` → `*((float4*)(data0 + 0`; compiles. S/low.

**CB2** Index/Shrink/address-Cast `ctx.register` raw strings → scope-escape hoist no longer covers addresses (`use of undeclared identifier ridx0`). Only mitigation for the resnet50/densenet121/inception_v2/shufflenet class.
- Tinygrad: `cstyle.py:189` inlines INDEX unconditionally — safe there only because its linearizer never places a node inside a range consumed outside it.
- Fix: `CContext::emit_address(uop, expr, kernel)` — register unless `scope_escaping.contains(id)`, else hoist `{elem}* bidx{n};` and assign at depth. Call from `c/ops.rs:200,218,225,330`.
- Test: hand-built `UOp::linear([p0,p1,range,idx,load,end,store])`: no `ridx0` after the closing `}`, `float* bidx0;` hoisted; compiles. M/med.

**LLVM cluster — CA2 CA4 CA6 CB1 CB3, one commit.** The LLVM renderer never adopted lane-count-in-shape / INDEX-is-element-dtype.
- Tinygrad: `llvmir.py:76-77` (INDEX = gep, never extractelement), `:82-84` (gated LOAD single dtype for load/alt/phi), `:189-190` (ptr CAST is an alias), `:91-96` (VECTORIZE inserts values), `:105-108` (ALU type = src dtype).
- Fix: (1) `fn lshaped(u) = ldt(&shaped_dtype(u))`; delete `memory_access_dtype`. (2) use `lshaped` for every value operand: Binary `:181-186`, Unary `:226`, Where `:328`, MulAcc `:344`, Cast `:353-369`, BitCast `:377`, LOAD/STORE, `render_vectorize` vec type. (3) INDEX mirrors C: empty indices → `ctx.alias` (CA6); branch on `buffer.addrspace()` — `Some` → `getelementptr inbounds {ldt(uop.dtype)}`, `None` → `extractelement {lshaped(buffer)}` (CA2); drop the `vector_count` block. (4) gated LOAD: type load/phi with `lshaped(uop)`, validate `shaped_dtype(alt)==shaped_dtype(uop)`, render a splat literal for a scalar Const alt via a `shaped_operand` helper (CA4). (5) STACK: per element with `addrspace().is_some()` emit `load {scalar}, ptr {val}` then insert (CB1).
- Test: `llvm_text.rs` one `#[test_case]`-driven `llvm_shaped_values_assemble(build: fn()->Arc<UOp>)` reusing `assert_llvm_ir_assembles`, 6 rows: `Stack([Index(param,i);4])`, `Cast(Stack([u32;4])→f32)`, `Index(Shrink(param0,0,4),2)`, gated Load with shape-[4] alt, `Index(param,[])`, `Store(Index(param,i), VConst<4>)`. L/med.

**Upstream follow-up (after both renderers are green):** `devectorize.rs:1632` `stack_index` and `:1666` `stack_wmma_sources` build `Stack(Index(..))` where tinygrad's devectorizer builds `VECTORIZE(LOAD(INDEX(..)))`. Wrap in `UOp::load()` when the source carries an address space, then delete the two compensating deref rules (CA1, CB1-5). Test: the same 6-row table stays green.

**CA7** ordered vs unordered fcmp (D8): keep; rename test `llvm_float_comparisons_use_ordered_predicates_matching_c`; +1 line `!contains("fcmp ... ult")`; 2-line comment at `llvm/cpu/ops.rs:797` citing `llvmir.py:70`. S/low.

**XB5** gated-LOAD alt/gate pairing guard has no negative test. **Review correction:** the MLIR half is moot (backend removed). Test: `c.rs` + `llvm.rs` `#[test_case((true,false))] #[test_case((false,true))]` → `InvalidGraph` containing "alt and gate"; one positive row (`" ? "` / `"= phi "`); one non-Bool-gate row. S/low.

**CA3** MFMA suffix catch-alls accept any K; `scale.` keyed on k==128. Fix: `(Float16,16)`/`(BFloat16,16)` exhaustive; `scale` iff suffix `.f8f6f4`. Test: `llvm_amd_wmma.rs` `#[test_case(Float16)] #[test_case(BFloat16)]` at (16,16,128) on Gfx950 → Err, no `mfma.scale.`. S/low.

---

## P7 — Tensor / model / ONNX

Order: TA2, TB1, TC1 (independent) → TA3+TC2 (same file) → TA8 → TA4 → XB4 → TC5, TC3 → TC4.

**TA2** `nonzero()` skips the modulo when `dims[axis]==1` — only valid for axis 0 (`[2,1,3]` hit at (1,0,0) → `[1,1,0]`). No tinygrad analogue.
- Fix: `tensor/src/indexing.rs:498` guard `axis != 0`.
- Test: `tensor/src/test/unit/indexing.rs` `test_nonzero_interior_singleton` next to `test_nonzero_2d`; `#[test_case]` rows `[2,1,3]`, `[1,3]`, `[3,1,1]`. S/low.

**TB1** ONNX `Mod fmod=1` int path calls raw `try_cmod`, bypassing broadcast.
- Tinygrad: `nn/onnx.py:646` Tensor-level both arms.
- Fix: add `try_cmod => try_cmod,` to `impl_tensor_ops!` (`tensor/src/arithmetic.rs:55`); `onnx/src/registry/mod.rs:106` → `x.try_cmod(y)?`.
- Test: `onnx/src/test/unit/arithmetic.rs` `#[test_case]` `(1,[3,4],[4])`, `(1,[3,4],[1,4])`, `(0,[3,4],[4])` → shape `[3,4]`, host `%`. S/low.

**TC1** GigaAM encoder coercion exempts only `.weight_scale`; MHSA/conv scales are `*_weight_scale` (`encoder.rs:346-358`).
- Fix: `model/src/gigaam/model.rs:275` `!key.ends_with("weight_scale")` (matches `prepare_scaled_weights` L33-37).
- Test: `model/src/test/unit/remap.rs`: `q_weight_scale` + `ffn1.linear1.weight_scale` stay Float32 under `Float16` encoder dtype. S/low.

**TA3** Whisper `weight_scale` `[out]` multiplied into `[out,in]` without reshape → square projections scale the input axis, `[4d,d]` fails to broadcast. **Review correction:** upgrade PLAUSIBLE→CONFIRMED (unconditional shape mismatch).
- Fix: `model/src/whisper/loader.rs:36-41` mirror GigaAM (`gigaam/model.rs:70-78`): `[out,1,..]` `try_reshape` before `try_mul`.
- Test: `model/src/test/unit/whisper/`: synthetic `mlp.0.weight [8,2]` + scale `[8]` loads; row 0 == `w[0,:]*scale[0]`. S/low.

**TC2** token-embedding exemption removed (D6): restore `|| key == "decoder.token_embedding.weight"` at `loader.rs:94`, revert the docstring sentence. Test: `#[test_case(FP8E4M3)] #[test_case(Float16)]` embedding dtype == checkpoint. S/low.

**TA8** symbolic `arange` ceildiv is positive-step only.
- Tinygrad: `helpers.py:49` `ceildiv = -(num // -amt)`.
- Fix: `tensor/src/lib.rs:421` `diff.floor_div(&step.neg()).neg()`.
- Test: shape-only assertion with a bound Variable and `step=-1` (realization still ICEs in `_cumalu`). S/low.

**TA4** memory-planner gated-STORE exclusion deleted with its 2 tests; tinygrad's TLSF path (`memory.py:17,70`) has no equivalent but morok's arena mode packs over a prior tenant.
- Fix: cannot restore verbatim — `Op::Index` lost `gate`; scan `Op::Store{gate: Some(_)}` in `item.ast.toposort()`, resolve via `item.buffer_uop_ids` (`tensor/src/schedule.rs:708`), add `PlannerExclusionReason::GatedStore`.
- Test: restore `test_memory_planner_skips_masked_store_outputs` (+ wrapped) from `git show origin/model_whisper_continous_batching:tensor/src/test/unit/memory_planner.rs`, retargeted to `UOp::store_gated` (`memory.rs:217`). M/med.

**XB4** patterns! Option-field tests. **Review correction:** 4 tests, not 3, and they lived in `schedule/src/test/unit/pattern/proc_macro_dsl.rs` (base L1136, 1619, 1647, 1675), not `macros/`; all matched `Index{gate}` which no longer exists — verbatim restore is NOT enough.
- Fix: retarget to `Load{alt: None, gate: None}`, `Load{gate: Some(g)} ~> g`, bare-`gate` binding on `Store`, using `UOp::store`/`store_gated`. Covers `OptionNone/OptionSome` codegen (`macros/src/patterns/codegen.rs:639,651`) used by `coalesce.rs:462`, `symbolic/patterns.rs:540,545`. M/low.

**TC5** `PrepareConfig::default()` / `From<OptimizerConfig>` hardcode `PlannerMode::Arena`. Fix: `mode_from_env()` at `config.rs:89,148-157`. Test: `memory_planner.rs` `#[test_case("disabled"|"remap"|"")]` with the env-guard helper. S/low.

**TC3** `initial_kernel_var_values` errors for any unbound `Variable::new(name,1,N)`, breaking documented prepare-once flow (`variable.rs:100`). Fix: placeholder `bounds.min.max(0)`; move the real bounds check to `execute_with_vars` (`execution_plan.rs:1485`). Test: `tensor/src/test/unit/variable.rs` near :327: unbound prepare Ok; `("N",4)` Ok; `("N",9)` Err. S/low.

**TC4** `tk/src/launch.rs:203-207` rustdoc. **Review correction:** behavioural half disputed — compact ordinal binding is the deliberate contract, pinned by `tk/src/test/unit/kernel_probe.rs:46-52`. Doc-only rewrite. S/low.

---

## P8 — Runtime

Order: UC2 → UA5+UC3 → UA3 → UA2 → UA4 → UC1 → UB1 → UB2 → RS1 → ES4 → (UA1, UA6, UC4a housekeeping) → ES1 → TC5/TC3 (P7) → TA6 → TA5 → CV3 last.

**UC2** `JitKernel::load_object_with_abi` validates with `validate_c_object` (expects `Dynamic` under `dlopen-fallback`) → all JitKernel loads fail under that feature (2 tests verified failing).
- Fix: `jit_loader.rs:52` → `validate_relocatable_object`; `:30` use `jit_loader::c_object_flags()` so the object is relocatable in both configs.
- Test: existing `runtime/src/test/unit/cpu.rs:209/234`; add `--features dlopen-fallback` to CI for the crate. S/low.

**UA5 + UC3** (one commit, adjacent lines).
- UA5: `execute_profiled` `unreachable!()` on a degraded `BufferCopy` (`execution_plan.rs:1364`) where `execute()` returns `Error::Execution`. Fix: `PreparedOp::BufferCopy(c) => self.execute_copy(c)?`; validate two `buffer_indices` in `ExecutionPlanBuilder::build` (`:1675`). Test: one-buffer BufferCopy → `build()` Err.
- UC3: `graph_endpoints_match_device()?` / `replay_native_linked_plan()?` run outside the poison closure (`:1197-1199`) → failed native submit is silently retryable. Fix: move both inside the closure. Test: failing native replay → `execute()` Err, second call returns the poisoned error. S/low.

**UA3** hazard read-set narrowed to `ProgramSpec.ins` in `build_graph` (`:763-765`) AND production `hcq_operations` (`:599-601`).
- Tinygrad: `jit.py:118-133` reads = every non-written rawbuf.
- Fix: `reads = (0..n).filter(!outputs)` in both; keep `input_indices` as a `debug_assert!` subset check; fix the stale comment at `:728`.
- Test: `runtime/src/test/unit/execution_plan.rs` two-kernel plan whose consumer omits a shared index → dep still present. S/low.

**UA2** object-cache I/O errors fatal at both ends.
- Tinygrad: `helpers.py:315-324` `diskcache_get` swallows errors.
- Fix: `ObjectCache::from_env` → `Ok(None)` + warn on open failure; `get_or_compile` (`object_cache.rs:145`) demote write/evict to warn and return the validated bytes.
- Test: (after CV2 move) read-only dir still returns bytes; `SVOD_OBJECT_CACHE_DIR=/proc/x` → `Ok(None)`. S/low.

**UA4** `LockFile` spins forever, trusts recycled PIDs, unlinks other owners' locks.
- Fix: `flock`-based advisory lock (`fd-lock`/`fs4`, `try_lock_exclusive` bounded retry); on timeout compile without publishing (cache is advisory after UA2); drop the global `eviction.lock` from the miss path — evict opportunistically under `try_lock`.
- Test: pre-created `.lock` with bogus pid → `get_or_compile` returns within a bound. M/med.

**UC1** `-nogpulib` removal (D4).
- Tinygrad: `llvmir.py:207,214-234` `@llvm.{sqrt,log2,exp2}`, f64 log2/exp2 → `xlog2`/`xexp2`, SIN via transcendental expansion; `compiler_amd.py:111-123` links no device libs.
- Fix: `codegen/src/llvm/amd/ops.rs:42-67` render `@llvm.{sqrt,log2,exp2,sin}.{half,float,double}`; keep `@__ocml_*` only for f64 log2/exp2/sin; add `declare`s in `llvm/text/mod.rs:365-372`; `amd_object_flags(ir, arch)` pushes `-nogpulib` unless `ir.contains("@__ocml_")` (flags are in `CompilerIdentity.flags`, so the cache key stays sound).
- Test: `llvm_text.rs:175` retarget as `#[test_case(Float32 => "@llvm.exp2.float")] #[test_case(Float64 => "@__ocml_exp2_f64")]`; `runtime/src/test/unit/amd_compile.rs` row: flags contain `-nogpulib` for ocml-free IR. M/med. Do before UB1 (both touch `CompilerIdentity`).

**UB1** `-march=native` probe cached across heterogeneous hosts (shared cache dir → foreign-ISA object).
- Tinygrad: `compiler_cpu.py:31` bakes live `processor`/`feats` into the key.
- Fix: when a flag contains `native`, extend `probe_input` (`clang.rs:77`) with a host fingerprint (digest of `clang -march=native -E -dM -x c /dev/null`); flow the resolved arch into `CompilerIdentity::target_architecture`.
- Test: `runtime/src/test/unit/clang.rs` `#[test_case("-march=native" => false)] #[test_case("-march=x86-64" => true)]` shared-probe-path with differing injected fingerprints. M/med.

**UB2** scalar kernel args declared `i32` but passed as `&u64` slots → 0 on big-endian. Confirmed (`device.rs:524`, `dispatch.rs:33`).
- Fix: split the thread-local into `ptrs: SmallVec<[*mut u8;16]>` and `scalars: SmallVec<[i32;16]>`; build `ffi_args` from the typed slots.
- Test: `cpu.rs` `(float* out, int n)` kernel `#[test_case(1|-7|i32::MAX)]` → `out[0]==n`. S/low.

**RS1** kernarg packing duplicated 4× (alignments 128/16/16/128). **Review correction:** tinygrad `hcq.py:315` aligns to 8, not 16.
- Fix: one `pub fn kernarg_offsets(sizes, align) -> (Vec<usize>, usize)` in `device/src/hcq.rs`; standardise on 16; call from `linked_plan.rs:150`, `graph.rs:181`, `program.rs:735`, `PlaceholderPacking::pack`. Shrinks linked-plan kernarg allocation up to 8×.
- Test: `device/src/test/unit/hcq.rs` `#[test_case(&[8,12,4],16 => vec![0,16,32])]`. S/low. Before ES4.

**ES4** `CommandBufferCache::default()` at all 5 production sites → 0% hit rate, still pays deep-clone+hash. **Correctness trap:** `link()` hardcodes `(0, DeviceSpec::Cpu)` — a shared cache must use `link_for_context`.
- Fix: one cache in per-device pool/lane state; thread `&mut`; switch all sites to `link_for_context`.
- Test: `device/src/test/unit/amd/queue.rs:816` second `link_for_context` → `Arc::ptr_eq`; differing context → distinct. M/med.

**Housekeeping commit:**
- **UA1** `CpuQueueExecutor.trace` (`device/src/hcq.rs:585`) grows unbounded on every execute; only read by `device/src/test/unit/hcq.rs:134`. Delete the field/push/getter (removes a `Command::clone` per command from the hot path); move the assertion onto `NullHcq`. Test: compile-time (no `trace()` method). S/low.
- **UA6** `has_amdgpu_target_with` forks `clang --print-targets` per compile; the `OnceLock` variant is dead. Fix: cache keyed on `toolchain.executable` (or `ObjectCache::get_or_create_probe("clang-targets")`, `clang.rs:156`); delete the unused fn. S/low.
- **UC4(a)** `spawn_compile_process` PID+seq `create_dir` (`clang.rs:169`) → `tempfile::Builder::new().prefix("svod-beam-").tempdir()`; deletes 3 manual unwind paths. **UC4(b)** (D5): **Review correction** — `spawn_compile_process` is wired into `Device::start_compile_process` on cpu/amd; the *trait method* has zero callers. Wire into BEAM wave (`realize.rs:1782`) or delete `CompilerProcess`/`CompilerProcessTask` + 3 impls. S/M.

**ES1** `execute()` re-walks every buffer through `matches_native_device` (global `DEVICE_CACHE` mutex per buffer), twice per execute.
- Fix: resolve the expected `Arc<AmdCore>` once per execute; `Buffer::matches_native_core(&Arc<AmdCore>)` = `ensure_allocated` + `Arc::ptr_eq`; merge the two walks.
- Test: counting stub allocator, O(buffers) over 100 executes. M/low.

**TA6** `WorkerPool::run` checks timeout before `try_recv` → on-deadline responses dropped, healthy worker SIGKILLed.
- Fix: `beam_worker.rs:489-498` `try_recv` first; evaluate `timed_out` only in the `Empty` arm.
- Test: new `tensor/src/test/unit/beam_worker.rs` with a fake `SpawnedWorker` answering at the deadline → `completed` fires once, no reset. S/low.

**TA5** BEAM helper resolution: `CARGO_BIN_EXE_` never set for the lib → nested `cargo build` on every path, profile dir guessed, failures cached forever.
- Fix: `--message-format=json-render-diagnostics`, take the last `"executable"`; cache successes only (`Mutex<Option<PathBuf>>`).
- Test: `beam_worker.rs` bad `SVOD_BEAM_WORKER` → Err; then valid → Ok (failure not sticky). M/med.

**CV3** `Result<_, String>` in 9 `beam_worker.rs` signatures. Fix: snafu `BeamWorker` enum (`SpawnHelper{source}`, `HelperUnavailable{reason}`, `Frame{source}`, `ProtocolMismatch{expected,actual}`, `WorkerMisorder{got,expected}`) + `BeamWorker{source}` in `tensor/src/error.rs`; `worker_main` may keep a boxed error. Test: one variant `matches!`. M/low — **last in P8** (rewrites the signatures TA5/TA6 touch).

---

## P9 — AMD device

Order: VA5+VA3 → VA2, VC3 → VB3+VB2 → VB5 → VC2 → VA6 → VB6 (cheap half) → VA1/VB1, VA4 → VC4, ES2 → VC5, VC6 (deferred/epic).

**VA5 + VA3** (one commit). `TimelineReservation`/`CopyTimelineReservation` stash `*mut QueueInner` from a `&mut` and write through it in `Drop` (stacked-borrows UB); `copy_fenced` records the rollback index *after* pushing copy packets (`queue.rs:2197`), defeating rollback.
- Tinygrad: n/a — submission is single-threaded and infallible after ring writes (`ops_amd.py:496-531`); morok's rollback is an addition and must be right.
- Fix: `struct RingRollback<'a> { idx: &'a mut u64, saved: u64, committed: bool }` with `commit(self)` and `Drop` restoring when uncommitted; create immediately after locking, before any push, at all four publication sites; delete `track_write_idx`/`write_idx` from both reservations. Borrowck makes the "before the pushes" placement structural.
- Test: `device/src/test/unit/amd/queue.rs` +3 lines: dropped uncommitted restores 40; committed keeps 47. M/low.

**VA2** zero-byte `Command::Copy` never marks `CopySrc/CopyDst` consumed → linked-plan capture hard-fails (CPU path guards `bytes != 0`).
- Tinygrad: `ops_amd.py:450-460` emits 0 packets, no arity check.
- Fix: `lower_hcq_sdma_command_buffer` (`queue.rs:811`) hoist the two `command_binding` lookups + `consumed.insert` above the chunk loop.
- Test: `Copy{bytes:0}` → `bytes.is_empty() && patches.runtime.is_empty()`, no Err. S/low.

**VC3** `Device::new` doc example permanently `ProgramStageMismatch` (`device.rs:1141`). Rewrite via `do_compile`/`bind_program_stage`; optionally promote to a compiled doctest. S/low.

**VB3 + VB2** "teardown policy" commit.
- VB3 (D3): panic unwind through `PoolQueue::drop` (`connector.rs:526`), `ActivatedQueueGuard::drop`, `AmdCopyQueue::drop` poisons the process-global core. Tinygrad `hcq.py:378-393` sets `error_state` only on drain timeout, per device. Fix: remove `core.poison` from the three unwind arms, keep warn + `quarantine()`. Test: `device/src/test/unit/amd/device.rs` `catch_unwind` inside a lease → `!core.is_poisoned()`. S/med — note in `HCQ_PORT_LEDGER.md`.
- VB2 **Review correction (partial):** tinygrad never destroys queues (`__del__` frees only `hw_page`), so the leaked id matches the reference. The real hazard is the pairing: `close` refuses once quarantined but `QueueInner::drop` (`queue.rs:1024-1046`) still unmaps ring/GART/EOP under a live CP. Fix: `quarantined: bool` on `QueueInner`; `drop` leaks (warn) when set. Test: `MockAmdIface` poison + drop lane → `live_allocation_count` unchanged. S/low.

**VB5** signal-pool exhaustion is a hard failure; panic-unwound signals leak slots.
- Tinygrad: `hcq.py:404-409` `new_signal` **grows** a page on exhaustion; `HCQSignal.__del__` returns unconditionally.
- Fix: `SignalPool` holds `Vec<RawBuffer>` pages, allocates another on empty `free_slots`; delete the FIFO/back-pressure question; drop the `panicking()` skip in `AmdSignal::drop`, keep the `is_poisoned()` skip (documented divergence — a wedged CP may still write).
- Test: `device/src/test/unit/amd/signal.rs` acquire `capacity()+1` → Ok, `live_allocation_count()==2`. M/low.

**VC2** one 16 MiB kernarg arena per lane (×4) vs tinygrad's one per device (`hcq.py:375-376`; `ops_amd.py:716-718` warns about resizable BAR).
- Fix: move `KernargArena` to `AmdDeviceCore`, `Arc` per `PoolQueue`; the wrap path already drains via `core.synchronize_all()`, so the invariant holds; update the module doc.
- Test: `device/src/test/unit/amd/kernarg.rs` two lanes → `alloc_count_for_tag(Kernarg)==1`. M/low.

**VA6** `Command::Store` drops `INT_SEL` and the KFD event mailbox write.
- Tinygrad: `ops_amd.py:368-377` interrupt-after-write + mailbox `release_mem`; `AMDSignal._sleep` (`:44-46`) blocks in `WAIT_EVENTS`; SDMA `:466-469` mailbox fence + `SDMA_OP_TRAP`.
- Fix: `queue.rs:504` `interrupt = true` (`sys/pm4.rs:243`); emit the mailbox `release_mem` for the queue-timeline store when the KFD event exists; same for SDMA.
- Test: golden dword `dwords[2] & INT_SEL_MASK`. GPU: `SVOD_AMD_AQL=0 cargo test -p svod-device --lib amd -- --ignored`. S/med — before VB6 to share one hardware session.

**VB6** (D7) per-kernel HDP-flush + full `acquire_mem` + `barrier:true` in both graph paths; pre-replay `synchronize()`.
- Tinygrad: `graph/hcq.py:146` one `memory_barrier()` per device queue at graph head; per-dispatch coherence is `acquire_mem(gli=0,gl2=0)` + `CS_PARTIAL_FLUSH` (`ops_amd.py:309,354`) — which `build_exec_pm4` already emits (`queue.rs:1999,2040`); `opt_deps` (`:100-119`) only elides cross-queue waits.
- Fix now: one `MemoryBarrier` at graph head (`graph.rs:213`); drop the pre-replay `synchronize()` if the linked timeline wait covers it. Defer: cross-queue narrowing.
- Test: 3-kernel capture → exactly 1 `MemoryBarrier`. GPU: 12-kernel gfx1151 graph rerun from the ledger. S/med.

**VA1 / VB1** `SubmissionFinalizer::wait` parks untimed while `Prepared`. Tinygrad: single 30 s `HCQDEV_WAIT_TIMEOUT_MS` (`hcq.py:250-262`). Fix: `wait_for(remaining)` → `TimelineTimeout{what:"AMD submission publication"}`; pass the remainder to `wait_signal_value_with_progress`; poll `poison_error()` on wake. Test: never-published finalizer `wait(50)` → `TimelineTimeout`. S/low.

**VA4** `QueuePool::acquire` parks untimed when all lanes leased. Fix: `wait_for(30 s)` in the retry loop; after N expiries `TimelineTimeout{what:"AMD lane acquisition"}`. Test: `MockAmdIface`, lease all, scoped thread → Err (once timeout injectable). S/low.

**VC4** dead duplicate scratch re-check (`connector.rs:495`) + 30 s drain under the session mutex. Tinygrad `ops_amd.py:990-1013` single check, no lock. Fix: delete the re-check; hoist `ensure_has_local_memory` before locking `session` in `OwnerCtx::dispatch` (the lease is the publication authority). Test: idempotence — `scratch_alloc_count()==1`. S/low.

**ES2** `AmdGraph::replay` repacks every slot + rebuilds `ClikeKernargLayout` per replay. Tinygrad `graph/hcq.py:203-204` + `_prev_resolved_syms` skip. Fix: layout built once at capture in `KernargSlot`; cache last `(buffers, vals)` and skip `pack` when unchanged. Test: `#[cfg(test)]` write counter unchanged on second identical replay. S/low.

**VC5** compute guard held while acquiring the SDMA guard (`linked_plan.rs:507`), up to 60 s stall. Tinygrad completes compute `_submit` before copy `_submit`. Fix: split `prepare_linked_publication` into `wait_publication_headroom` (releases between polls) + `lock_publication`; wait both, then lock both back-to-back with a cheap re-verify. GPU-only test. M/med.

**VC6** linked SDMA publication never advances the copy-queue timeline; `Drop`s unreachable due to a pre-existing `Arc` cycle. Tinygrad has one `timeline_signal` per device (`hcq.py:371`). Fix: register the SDMA finalizer with the copy queue's inflight list; drain in `AmdCopyQueue::drop`. **Epic candidate:** VC5+VC6 both dissolve under tinygrad's single per-device timeline — scope them there rather than patching plan-local timelines twice. M/med.

---

## Shared test infrastructure (add once, reuse everywhere)

| Helper | Location | Used by |
|---|---|---|
| `assert_c_compiles(src)` (clang `-fsyntax-only`, skip if absent) | `codegen/src/test/unit/c.rs` | CA1 CA5 CB2 XB5 |
| `llvm_shaped_values_assemble(build)` `#[test_case]` table over `assert_llvm_ir_assembles` | `codegen/src/test/unit/llvm_text.rs` | LLVM cluster, follow-up |
| `eval_word` (typed evaluator + Where/Cast/BitCast) | `schedule/src/test/property/long_shift.rs` | DA1 DA2 |
| `buf_target()` (shared with `rule_getaddr`) | `schedule/src/spec.rs` | SB3 |
| `RingRollback` unit + `MockAmdIface` counters (`live_allocation_count`, `alloc_count_for_tag`, `scratch_alloc_count`) | `device/src/test/unit/amd/` | VA5 VB2 VB5 VC2 VC4 |
| fake `SpawnedWorker` | `tensor/src/test/unit/beam_worker.rs` | TA5 TA6 CV3 |
| env-guard helper (existing) | `tensor/src/test/unit/memory_planner.rs` | TC5 |

Every other item is ≤ 5 lines added to an existing test fn or a `#[test_case]` row.

## Exit gate

1. `cargo fmt --check`; `cargo clippy --workspace --all-targets` → 0 errors (CV1); `cargo test --workspace`; `cargo test -p svod-runtime --features dlopen-fallback` (UC2).
2. Symbolic proptests + z3 suite green after XB6, SB2, DC4.
3. `schedule/src/test/unit/rangeify/kernel_count.rs` re-baselined after RB4 with the delta explained per kernel.
4. GPU session (one, shared): `#[ignore]`d tk/tk2 suite (DB1), `SVOD_AMD_AQL=0 … --ignored` (VA6, VB6, VC5, VC6), 12-kernel gfx1151 graph rerun.
5. ONNX light tests for resnet50 / densenet121 / inception_v2 / shufflenet pass on both C and LLVM CPU backends (CB2 + LLVM cluster).
6. `bash scripts/check-canonical-parity.sh` runs from a clean clone (SS2) against the pin chosen in D1.
