# EVID-02 Safety Evidence Schema Version 2

This typed JSON contract is the strict independent-equivalent safety gate for
the `M=5, K=16, N=16`, `float16 -> float32`, gfx1151 padded-WMMA fixture at the
exact Tinygrad pin `8c8b43de62515abe6c820b1de5aa26b30f48e43a`.

## Shared Boundaries

The required stages are `late-final-rewrite` and `linearized`.

- Tinygrad `late-final-rewrite` is the result of the `graph_rewrite` named
  `final rewrite` inside `tinygrad.codegen.full_rewrite_to_sink`, captured
  before `pm_implicit_barriers`, `pm_add_control_flow`, and `pm_number_params`.
- Svod `late-final-rewrite` is captured by
  `optimize_kernel_with_config_and_final_rewrite` immediately after the final
  matcher and `pm_remove_invalid`, before `pm_implicit_barriers`. Production
  optimization still continues through implicit barriers unchanged.
- Tinygrad `linearized` is the `LINEAR` returned by
  `tinygrad.codegen.do_linearize`; Svod `linearized` is the `LINEAR` returned by
  `svod_codegen::program_pipeline::do_linearize`.

Thus the first stage is the same late-final-rewrite boundary in both
implementations, before CFG/control-flow insertion and PARAM numbering. The
second stage is actual renderer order after line cleanup.

## Source Graphs

Each stage contains exactly `name`, `root`, and `nodes`. `nodes` is the complete
root-reachable typed source graph. Every node contains:

- `id`: contiguous source-topological identity
- `op`: typed operation name
- `dtype`: one exact scalar name from `void`, `bool`, `int`, `int32`, `int64`,
  `float16`, or `float32`
- `shape`: constant extents or `null`
- `src`: ordered source identities
- `arg`: typed CONST, PARAM, SPECIAL, or WMMA metadata, otherwise `null`

The first root is `SINK`. The second root is `LINEAR`, whose ordered `src` list
is the actual operation order. Producers do not emit lane maps, access lists,
coverage summaries, guard summaries, or synthesized/sorted operation order.

## Independent Validation

`scripts/evid02-safety-diff.py` independently traverses each graph and rejects
missing, unreachable, disconnected, or non-topological nodes. From source
edges and expression ASTs it:

- enumerates every PARAM, LOAD, STORE, WMMA, IF, ENDIF, INDEX/SHRINK,
  alternate, gate, address expression, and WMMA-result index;
- evaluates integer and boolean expression ASTs for all 32 lanes;
- derives widths, zero alternates, gates, enabled/disabled address sets,
  multiplicity, and C result-lane ownership;
- requires exactly one WMMA and exactly the fixture A/B/C PARAM ABI;
- proves that the WMMA A operand's LOAD ancestors are all and only the four
  padded A loads and that its B operand's LOAD ancestors are all and only the
  sixteen B loads; operation presence or LINEAR ordering alone is insufficient;
- requires 16-lane `float16` A/B operands, an eight-lane `float32` accumulator
  containing only `float32` zero constants, a `float32` WMMA result, and every C
  STORE value to INDEX that exact WMMA node rather than another fragment-like
  node;
- validates operation-level dtype semantics: A/B PARAMs, indices, SHRINKs,
  LOADs, alternates, fragment STACKs, and WMMA inputs are `float16`; the C PARAM,
  C indices and STORE values, accumulator, and WMMA result are `float32`; gates
  and comparisons are `bool`; statements are `void`; and every address, shape,
  index, shift, and integer-expression operand has an allowed integer dtype;
- rejects every unaccounted A/B/C access and any extra control flow;
- derives order solely from `LINEAR.src`, requires all loads before WMMA and
  stores after it, and derives IF/store/ENDIF ownership from positions and
  source identity;
- requires one non-nested IF immediately owning only the partial C store and
  one ENDIF whose source is that exact IF;
- proves `A[0..80]`, zero-filled padded `A[80..256]`, twice-covered
  `B[0..256]`, exactly-once `C[0..80]`, and disabled `C[80..96]`.

Cross-language comparison uses only these independently derived normalized
safety semantics. Each normalized access includes its operation, PARAM,
address, gate, alternate, result, and result-index dtype semantics. The only
dtype normalization is the independently validated integer category shared by
Tinygrad `int` and Svod `int32`/`int64`; float widths, bool, and void remain
exact. Full source graphs may differ, including commutation of the two
equivalent unguarded C stores.

Every strict comparison runs eleven adversarial mutations. In addition to a
forged lane source, omitted LINEAR access, extra IF, changed LINEAR order,
second WMMA, and changed expression AST, it replaces both WMMA data operands
with unrelated constants while retaining every LOAD, changes an A LOAD to
`float32`, changes the accumulator and result dtypes independently, and reroutes
one C STORE through a different WMMA-shaped fragment. All mutations must fail
source-graph validation. Unknown or producer-authored summary fields are
rejected by the exact schema, so a coordinated forged summary/order cannot
override the graph.
