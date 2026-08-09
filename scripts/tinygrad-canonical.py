#!/usr/bin/env python3
"""Emit canonical UOp JSON from the pinned Tinygrad reference checkout."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
TINYGRAD = ROOT / "submodules" / "new_new_tinygrad"
TARGET_COMMIT = "8c8b43de62515abe6c820b1de5aa26b30f48e43a"
sys.path.insert(0, str(TINYGRAD))

from tinygrad.dtype import Invalid, InvalidType, dtypes  # noqa: E402
from tinygrad.uop import Ops  # noqa: E402
from tinygrad.uop.ops import UOp  # noqa: E402


def verify_target() -> None:
  actual = subprocess.check_output(["git", "-C", str(TINYGRAD), "rev-parse", "HEAD"], text=True).strip()
  if actual != TARGET_COMMIT:
    raise RuntimeError(f"Tinygrad reference is {actual}, expected {TARGET_COMMIT}")


def canonical_dtype(dtype) -> dict[str, Any]:
  names = {
    dtypes.void: "void", dtypes.weakint: "weakint", dtypes.bool: "bool",
    dtypes.int8: "int8", dtypes.uint8: "uint8", dtypes.int16: "int16", dtypes.uint16: "uint16",
    dtypes.int32: "int32", dtypes.uint32: "uint32", dtypes.int64: "int64", dtypes.uint64: "uint64",
    dtypes.weakfloat: "weakfloat", dtypes.fp8e4m3: "fp8e4m3", dtypes.fp8e5m2: "fp8e5m2",
    dtypes.float16: "float16", dtypes.bfloat16: "bfloat16", dtypes.float32: "float32", dtypes.float64: "float64",
  }
  if dtype not in names: raise TypeError(f"unsupported dtype in canonical serializer: {dtype!r}")
  return {"kind": "scalar", "name": names[dtype]}


def canonical_const(value: Any, dtype) -> dict[str, Any]:
  if isinstance(value, InvalidType): return {"kind": "invalid"}
  if isinstance(value, bool): return {"kind": "bool", "value": value}
  if isinstance(value, float):
    bits = struct.unpack("<Q", struct.pack("<d", float(value)))[0]
    return {"kind": "float", "bits": f"0x{bits:016x}"}
  if isinstance(value, int):
    kind = "uint" if dtypes.is_unsigned(dtype) else "int"
    return {"kind": kind, "value": value}
  raise TypeError(f"unsupported constant in canonical serializer: {value!r}")


def canonical_value(value: Any) -> Any:
  if value is None or isinstance(value, (bool, int, str)): return value
  if isinstance(value, float):
    bits = struct.unpack("<Q", struct.pack("<d", value))[0]
    return {"float_bits": f"0x{bits:016x}"}
  if isinstance(value, (tuple, list)): return [canonical_value(item) for item in value]
  if hasattr(value, "name"): return value.name
  return repr(value)


def canonical_arg(node: UOp) -> dict[str, Any]:
  if node.op is Ops.CONST: return {"kind": "const", "value": canonical_const(node.arg, node.dtype)}
  if node.arg is None: return {"kind": "none"}
  # Unsupported target metadata remains explicit rather than falling back to a
  # Python repr silently. Add a typed mapping when a parity fixture reaches it.
  return {"kind": "python", "value": canonical_value(node.arg)}


def canonical_graph(stage: str, roots: Iterable[UOp]) -> dict[str, Any]:
  roots = tuple(roots)
  topo: list[UOp] = []
  seen: set[UOp] = set()
  for root in roots:
    for node in root.toposort():
      if node not in seen:
        seen.add(node)
        topo.append(node)
  ids = {node: index for index, node in enumerate(topo)}

  nodes = []
  for node_id, node in enumerate(topo):
    shape = []
    for dim in node.shape:
      shape.append({"kind": "symbolic", "node": ids.get(dim)} if isinstance(dim, UOp) else {"kind": "const", "value": dim})
    nodes.append({
      "id": node_id,
      "op": node.op.name,
      "dtype": canonical_dtype(node.dtype),
      "shape": shape,
      "arg": canonical_arg(node),
      "src": [ids[source] for source in node.src],
    })

  return {"schema_version": 1, "stage": stage, "roots": [ids[root] for root in roots], "nodes": nodes}


def fixture(name: str) -> UOp:
  if name == "weak_int_add": return UOp(Ops.ADD, src=(UOp.const(7), UOp.const(2)))
  if name == "weak_float_neg_zero": return UOp.const(-0.0)
  if name == "invalid_where":
    return UOp(Ops.WHERE, src=(UOp.const(True), UOp.const(1.0, dtypes.float16), UOp.const(Invalid)))
  raise ValueError(f"unknown fixture {name!r}")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("fixture", choices=("weak_int_add", "weak_float_neg_zero", "invalid_where"))
  parser.add_argument("--stage", default="tensor")
  args = parser.parse_args()
  verify_target()
  json.dump(canonical_graph(args.stage, (fixture(args.fixture),)), sys.stdout, indent=2)
  print()


if __name__ == "__main__": main()
