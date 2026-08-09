#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="${TMPDIR:-/tmp}/svod-canonical-parity"
mkdir -p "$TMP"

for fixture in weak_int_add weak_float_neg_zero invalid_where scalar_load gated_load; do
  cargo run --quiet --manifest-path "$ROOT/Cargo.toml" -p svod-ir --example canonical_fixture -- "$fixture" >"$TMP/$fixture-rust.json"
  "$ROOT/scripts/tinygrad-canonical.py" "$fixture" >"$TMP/$fixture-python.json"
  python - "$TMP/$fixture-rust.json" "$TMP/$fixture-python.json" <<'PY'
import json, sys
with open(sys.argv[1]) as f: rust = json.load(f)
with open(sys.argv[2]) as f: python = json.load(f)
expect_equal = sys.argv[1].rsplit('/', 1)[-1].split('-rust.json')[0] in {"weak_int_add", "weak_float_neg_zero", "invalid_where"}
if expect_equal and rust != python:
  print(json.dumps({"rust": rust, "python": python}, indent=2))
  raise SystemExit(1)
if not expect_equal and rust == python:
  print("known representation divergence disappeared; promote this fixture to strict parity")
  raise SystemExit(1)
if not expect_equal:
  fixture = sys.argv[1].rsplit('/', 1)[-1].split('-rust.json')[0]
  signatures = {
    "scalar_load": {
      "rust": [("PARAM", 0), ("CONST", 0), ("INDEX", 2), ("LOAD", 2)],
      "python": [("CONST", 0), ("PARAM", 1), ("CONST", 0), ("INDEX", 2), ("LOAD", 1)],
    },
    "gated_load": {
      "rust": [("PARAM", 0), ("CONST", 0), ("CONST", 0), ("CMPLT", 2), ("INDEX", 3), ("CONST", 0), ("LOAD", 3)],
      "python": [("CONST", 0), ("PARAM", 1), ("CONST", 0), ("CONST", 0), ("CMPLT", 2), ("INDEX", 3), ("CONST", 0), ("LOAD", 2)],
    },
  }
  actual = {
    "rust": [(node["op"], len(node["src"])) for node in rust["nodes"]],
    "python": [(node["op"], len(node["src"])) for node in python["nodes"]],
  }
  if actual != signatures[fixture]:
    print(json.dumps({"expected": signatures[fixture], "actual": actual}, indent=2))
    raise SystemExit(1)
  rust_param = next(node for node in rust["nodes"] if node["op"] == "PARAM")
  python_param = next(node for node in python["nodes"] if node["op"] == "PARAM")
  assert rust_param["dtype"]["kind"] == "pointer" and python_param["dtype"]["kind"] == "scalar"
  assert rust["nodes"][-1]["src"][0] == rust_param["id"] and python["nodes"][-1]["src"][0] != python_param["id"]
  print("known representation divergence: PARAM shape/pointer encoding and redundant LOAD buffer source")
PY
  printf 'canonical fixture: %s: ok\n' "$fixture"
done
