#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="${TMPDIR:-/tmp}/svod-canonical-parity"
mkdir -p "$TMP"

for fixture in weak_int_add weak_float_neg_zero invalid_where; do
  cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-ir --example canonical_fixture -- "$fixture" >"$TMP/$fixture-rust.json"
  "$ROOT/scripts/tinygrad-canonical.py" "$fixture" >"$TMP/$fixture-python.json"
  python - "$TMP/$fixture-rust.json" "$TMP/$fixture-python.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: rust = json.load(f)
with open(sys.argv[2]) as f: python = json.load(f)
if rust != python:
  print(json.dumps({"rust": rust, "python": python}, indent=2))
  raise SystemExit(1)
PY
  printf 'canonical parity: %s: ok\n' "$fixture"
done
