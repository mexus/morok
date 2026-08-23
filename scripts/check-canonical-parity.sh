#!/usr/bin/env bash
set -euo pipefail
export NO_COLOR=1
export CARGO_TERM_COLOR=never

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN="8c8b43de62515abe6c820b1de5aa26b30f48e43a"
REFERENCE="$ROOT/submodules/new_new_tinygrad"
ACTUAL="$(git -C "$REFERENCE" rev-parse HEAD)"
if [[ "$ACTUAL" != "$PIN" ]]; then
  printf 'Tinygrad reference is %s, expected %s\n' "$ACTUAL" "$PIN" >&2
  exit 1
fi

mkdir -p "${TMPDIR:-/tmp}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/svod-canonical-parity.XXXXXX")"
if [[ -z "${CANONICAL_KEEP_TMP:-}" ]]; then
  trap 'rm -rf "$TMP"' EXIT
else
  printf 'canonical artifacts: %s\n' "$TMP"
fi

"$ROOT/scripts/canonical-diff.py" --self-test
"$ROOT/scripts/evid02-safety-diff.py" --self-test
(cd "$REFERENCE" && uv run python ../../scripts/tinygrad-canonical.py --self-test)

FIRST="$TMP/first"
SECOND="$TMP/second"
mkdir "$FIRST" "$SECOND"

# EVID-02 is a strict independent-equivalent safety gate even while broader
# EVID-01B production graph identity remains recorded below as known gaps.
for artifacts in "$FIRST" "$SECOND"; do
  cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-tensor --example evid02_safety \
    >"$artifacts/evid02-safety-rust.json"
  (cd "$REFERENCE" && uv run python ../../scripts/tinygrad-canonical.py --evid02-safety) \
    >"$artifacts/evid02-safety-python.json"
done
"$ROOT/scripts/evid02-safety-diff.py" "$FIRST/evid02-safety-rust.json" "$SECOND/evid02-safety-rust.json" \
  --left-name svod-first --right-name svod-second
"$ROOT/scripts/evid02-safety-diff.py" "$FIRST/evid02-safety-python.json" "$SECOND/evid02-safety-python.json" \
  --left-name tinygrad-first --right-name tinygrad-second
"$ROOT/scripts/evid02-safety-diff.py" "$FIRST/evid02-safety-rust.json" "$FIRST/evid02-safety-python.json"
"$ROOT/scripts/evid02-safety-diff.py" "$SECOND/evid02-safety-rust.json" "$SECOND/evid02-safety-python.json"
printf 'EVID-02 padded-WMMA safety stages: strict parity ok\n'

for fixture in invalid_where scalar_stack shaped_stack buffer scalar_load gated_load \
               scalar_store mixed_valid_load copy allreduce multi_output_call local_wmma_staging \
               range_split_outer range_split_inner range_split_nested program_info symbolic_function; do
  if [[ "$fixture" == "program_info" ]]; then stage_args=(--stage program); else stage_args=(); fi
  for artifacts in "$FIRST" "$SECOND"; do
    cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-ir --example canonical_fixture -- "$fixture" \
      >"$artifacts/$fixture-rust.json"
    (cd "$REFERENCE" && uv run python ../../scripts/tinygrad-canonical.py "$fixture" "${stage_args[@]}") \
      >"$artifacts/$fixture-python.json"
  done
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$fixture-rust.json" "$SECOND/$fixture-rust.json" \
    --left-name rust-first --right-name rust-second
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$fixture-python.json" "$SECOND/$fixture-python.json" \
    --left-name python-first --right-name python-second
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$fixture-rust.json" "$FIRST/$fixture-python.json"
  "$ROOT/scripts/canonical-diff.py" "$SECOND/$fixture-rust.json" "$SECOND/$fixture-python.json"
  printf 'canonical fixture: %s: ok\n' "$fixture"
done

MANIFEST="$ROOT/scripts/CANONICAL_KNOWN_GAPS.txt"
GENERATED_MANIFEST="$TMP/known-gaps.txt"
PRODUCTION_GAPS=()

record_mismatch() {
  local category="$1" name="$2" issue="$3" first_left="$4" first_right="$5" second_left="$6" second_right="$7"
  local first_diff="$TMP/$category-$name-first.diff" second_diff="$TMP/$category-$name-second.diff"
  local first_status second_status
  set +e
  "$ROOT/scripts/canonical-diff.py" "$first_left" "$first_right" >"$first_diff" 2>&1
  first_status=$?
  "$ROOT/scripts/canonical-diff.py" "$second_left" "$second_right" >"$second_diff" 2>&1
  second_status=$?
  set -e
  if [[ $first_status -eq 0 || $second_status -eq 0 ]]; then
    printf '%s %s unexpectedly passed; remove its expected mismatch evidence\n' "$category" "$name" >&2
    exit 1
  fi
  if [[ $first_status -ne 1 || $second_status -ne 1 ]]; then
    printf '%s %s comparison failed with non-mismatch status %d/%d\n' "$category" "$name" "$first_status" "$second_status" >&2
    cat "$first_diff" >&2
    cat "$second_diff" >&2
    exit 1
  fi
  if ! cmp -s "$first_diff" "$second_diff"; then
    printf '%s %s mismatch diagnostic is not deterministic\n' "$category" "$name" >&2
    diff -u "$first_diff" "$second_diff" >&2 || true
    exit 1
  fi
  printf '[%s %s issue=%s]\n' "$category" "$name" "$issue" >>"$GENERATED_MANIFEST"
  cat "$first_diff" >>"$GENERATED_MANIFEST"
  printf '\n' >>"$GENERATED_MANIFEST"
}

for fixture in weak_int_add weak_float_neg_zero; do
  for artifacts in "$FIRST" "$SECOND"; do
    cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-ir --example canonical_fixture -- "$fixture" \
      >"$artifacts/$fixture-rust.json"
    (cd "$REFERENCE" && uv run python ../../scripts/tinygrad-canonical.py "$fixture") \
      >"$artifacts/$fixture-python.json"
  done
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$fixture-rust.json" "$SECOND/$fixture-rust.json" \
    --left-name rust-first --right-name rust-second
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$fixture-python.json" "$SECOND/$fixture-python.json" \
    --left-name python-first --right-name python-second
  record_mismatch direct-fixture "$fixture" EVID-01B \
    "$FIRST/$fixture-rust.json" "$FIRST/$fixture-python.json" \
    "$SECOND/$fixture-rust.json" "$SECOND/$fixture-python.json"
  PRODUCTION_GAPS+=("$fixture")
  printf 'canonical derived promotion fixture: %s: deterministic mismatch\n' "$fixture"
done

for fixture in padded_reduction; do
  for artifacts in "$FIRST" "$SECOND"; do
    cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-ir --example canonical_fixture -- "$fixture" \
      >"$artifacts/$fixture-rust.json"
    (cd "$REFERENCE" && uv run python ../../scripts/tinygrad-canonical.py "$fixture") \
      >"$artifacts/$fixture-python.json"
  done
  record_mismatch expected-failure "$fixture" REDUCE-01 \
    "$FIRST/$fixture-rust.json" "$FIRST/$fixture-python.json" \
    "$SECOND/$fixture-rust.json" "$SECOND/$fixture-python.json"
  printf 'canonical expected failure: %s (REDUCE-01): exact mismatch ok\n' "$fixture"
done

for artifacts in "$FIRST" "$SECOND"; do
  SVOD_CAPTURE_CANONICAL_STAGE=kernel_ast \
  SVOD_CAPTURE_CANONICAL_LABEL=kernel_ast \
  SVOD_CAPTURE_CANONICAL_PATH="$artifacts/multi_output_callified-rust.json" \
    cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-tensor --example canonical_stages -- multi_output_callified \
    >/dev/null
  (cd "$REFERENCE" && NO_COLOR=1 uv run python ../../scripts/tinygrad-canonical.py --production-multi-output) \
    >"$artifacts/multi_output_callified-python.json"
done
"$ROOT/scripts/canonical-diff.py" "$FIRST/multi_output_callified-rust.json" "$SECOND/multi_output_callified-rust.json" \
  --left-name rust-first --right-name rust-second
"$ROOT/scripts/canonical-diff.py" "$FIRST/multi_output_callified-python.json" "$SECOND/multi_output_callified-python.json" \
  --left-name python-first --right-name python-second
record_mismatch production-fixture multi_output_callified EVID-01B \
  "$FIRST/multi_output_callified-rust.json" "$FIRST/multi_output_callified-python.json" \
  "$SECOND/multi_output_callified-rust.json" "$SECOND/multi_output_callified-python.json"
PRODUCTION_GAPS+=(multi_output_callified)
printf 'canonical production fixture: multi_output_callified: deterministic mismatch\n'

PRODUCTION_STAGES=(tensor rangeified kernel_ast scheduled optimized postrange expanded coalesced gated program linearized)
for stage in "${PRODUCTION_STAGES[@]}"; do
  for artifacts in "$FIRST" "$SECOND"; do
    SVOD_CAPTURE_CANONICAL_STAGE="$stage" \
    SVOD_CAPTURE_CANONICAL_LABEL="$stage" \
    SVOD_CAPTURE_CANONICAL_PATH="$artifacts/$stage-rust.json" \
      cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-tensor --example canonical_stages -- "$stage" \
      >/dev/null
    (cd "$REFERENCE" && uv run python ../../scripts/tinygrad-canonical.py --production-stage "$stage") \
      >"$artifacts/$stage-python.json"
  done
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$stage-rust.json" "$SECOND/$stage-rust.json" \
    --left-name rust-first --right-name rust-second
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$stage-python.json" "$SECOND/$stage-python.json" \
    --left-name python-first --right-name python-second
  first_diff="$TMP/production-stage-$stage-first.diff"
  set +e
  "$ROOT/scripts/canonical-diff.py" "$FIRST/$stage-rust.json" "$FIRST/$stage-python.json" >"$first_diff" 2>&1
  status=$?
  set -e
  if [[ $status -eq 0 ]]; then
    "$ROOT/scripts/canonical-diff.py" "$SECOND/$stage-rust.json" "$SECOND/$stage-python.json"
    printf 'canonical production stage: %s: strict parity ok\n' "$stage"
  elif [[ $status -eq 1 ]]; then
    record_mismatch production-stage "$stage" EVID-01B \
      "$FIRST/$stage-rust.json" "$FIRST/$stage-python.json" \
      "$SECOND/$stage-rust.json" "$SECOND/$stage-python.json"
    PRODUCTION_GAPS+=("$stage")
    printf 'canonical production stage: %s: deterministic mismatch\n' "$stage"
  else
    cat "$first_diff" >&2
    exit "$status"
  fi
done

# record_mismatch separates sections with a blank line; keep a normal single
# trailing newline so the checked-in text manifest is editor-friendly.
truncate -s -1 "$GENERATED_MANIFEST"

if [[ ! -f "$MANIFEST" ]] || ! cmp -s "$MANIFEST" "$GENERATED_MANIFEST"; then
  printf 'canonical mismatch manifest differs from %s\n' "$MANIFEST" >&2
  if [[ -f "$MANIFEST" ]]; then diff -u "$MANIFEST" "$GENERATED_MANIFEST" >&2 || true; else cat "$GENERATED_MANIFEST" >&2; fi
  exit 1
fi

if ((${#PRODUCTION_GAPS[@]})); then
  if [[ "${CANONICAL_RECORD_KNOWN_GAPS:-0}" == "1" ]]; then
    printf 'canonical evidence mode: exact known-gap manifest verified (%s)\n' "${PRODUCTION_GAPS[*]}"
  else
    printf 'canonical strict gate failed; production mismatches: %s\n' "${PRODUCTION_GAPS[*]}" >&2
    printf 'Use CANONICAL_RECORD_KNOWN_GAPS=1 only to record checked-in deterministic evidence.\n' >&2
    exit 1
  fi
fi
