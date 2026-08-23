#!/usr/bin/env python3
"""Validate EVID-02 directly from serialized source graphs and compare semantics."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


class EvidenceError(ValueError): pass


ALLOWED_DTYPES = {"void", "bool", "int", "int32", "int64", "float16", "float32"}
INTEGER_DTYPES = {"int", "int32", "int64"}


def require(condition: bool, path: str, message: str) -> None:
  if not condition: raise EvidenceError(f"{path}: {message}")


def exact(value: Any, keys: set[str], path: str) -> dict[str, Any]:
  require(isinstance(value, dict), path, "expected object")
  require(set(value) == keys, path, f"fields must be {sorted(keys)}, got {sorted(value)}")
  return value


def integer(value: Any, path: str, minimum: int = 0) -> int:
  require(isinstance(value, int) and not isinstance(value, bool) and value >= minimum, path, f"expected integer >= {minimum}")
  return value


def array(value: Any, path: str, length: int | None = None) -> list[Any]:
  require(isinstance(value, list), path, "expected array")
  if length is not None: require(len(value) == length, path, f"expected {length} entries")
  return value


def parse_graph(value: Any, path: str, expected_name: str) -> tuple[dict[int, dict[str, Any]], int]:
  graph = exact(value, {"name", "root", "nodes"}, path)
  require(graph["name"] == expected_name, path + ".name", f"expected {expected_name}")
  raw_nodes = array(graph["nodes"], path + ".nodes")
  nodes: dict[int, dict[str, Any]] = {}
  for position,raw in enumerate(raw_nodes):
    node = exact(raw, {"id", "op", "dtype", "shape", "src", "arg"}, f"{path}.nodes[{position}]")
    node_id = integer(node["id"], f"{path}.nodes[{position}].id")
    require(node_id == position, f"{path}.nodes[{position}].id", "node IDs must be contiguous source-topological positions")
    require(isinstance(node["op"], str) and node["op"], f"{path}.nodes[{position}].op", "expected operation name")
    require(isinstance(node["dtype"], str) and node["dtype"] in ALLOWED_DTYPES, f"{path}.nodes[{position}].dtype",
            f"dtype must be one of {sorted(ALLOWED_DTYPES)}")
    if node["shape"] is not None:
      for index,extent in enumerate(array(node["shape"], f"{path}.nodes[{position}].shape")):
        integer(extent, f"{path}.nodes[{position}].shape[{index}]")
    for index,source in enumerate(array(node["src"], f"{path}.nodes[{position}].src")):
      source = integer(source, f"{path}.nodes[{position}].src[{index}]")
      require(source < node_id, f"{path}.nodes[{position}].src[{index}]", "source must precede consumer")
    nodes[node_id] = node
  root = integer(graph["root"], path + ".root")
  require(root in nodes, path + ".root", "unknown root")
  reachable: set[int] = set()
  stack = [root]
  while stack:
    node_id = stack.pop()
    if node_id in reachable: continue
    reachable.add(node_id); stack.extend(nodes[node_id]["src"])
  require(reachable == set(nodes), path + ".nodes", "node table must contain exactly the complete root-reachable source graph")
  return nodes, root


def const(node: dict[str, Any], path: str) -> int | float | bool:
  require(node["op"] == "CONST", path, "expected CONST")
  arg = exact(node["arg"], {"kind", "value"}, path + ".arg")
  require(arg["kind"] in {"int", "float", "bool"}, path + ".arg.kind", "unsupported constant kind")
  expected = {"int": int, "float": (int, float), "bool": bool}[arg["kind"]]
  require(isinstance(arg["value"], expected) and not (arg["kind"] != "bool" and isinstance(arg["value"], bool)),
          path + ".arg.value", "constant value has wrong type")
  expected_dtypes = {"int": INTEGER_DTYPES, "float": {"float16", "float32"}, "bool": {"bool"}}[arg["kind"]]
  require(node["dtype"] in expected_dtypes, path + ".dtype", f"{arg['kind']} constant has incompatible dtype")
  return arg["value"]


def eval_int(nodes: dict[int, dict[str, Any]], node_id: int, lane: int, path: str) -> int:
  node = nodes[node_id]; op = node["op"]; src = node["src"]
  require(node["dtype"] in INTEGER_DTYPES, f"{path}.nodes[{node_id}].dtype", "integer expression must have integer dtype")
  if op == "CONST":
    value = const(node, f"{path}.nodes[{node_id}]")
    require(isinstance(value, int) and not isinstance(value, bool), path, "integer expression requires integer CONST")
    return value
  if op == "SPECIAL":
    arg = exact(node["arg"], {"name"}, f"{path}.nodes[{node_id}].arg")
    require(arg["name"] == "lidx0", path, "only lidx0 is valid in EVID-02 expressions")
    return lane
  if op == "CAST":
    require(len(src) == 1, path, "CAST expression arity")
    return eval_int(nodes, src[0], lane, path)
  if op in {"ADD", "MUL", "AND", "SHL", "SHR"}:
    require(len(src) == 2, path, f"{op} expression arity")
    left,right = (eval_int(nodes, source, lane, path) for source in src)
    if op == "ADD": return left + right
    if op == "MUL": return left * right
    if op == "AND": return left & right
    if op == "SHL": return left << right
    return left >> right
  if op == "MULACC":
    require(len(src) == 3, path, "MULACC expression arity")
    a,b,c = (eval_int(nodes, source, lane, path) for source in src)
    return a * b + c
  raise EvidenceError(f"{path}.nodes[{node_id}]: unsupported integer expression operation {op}")


def eval_bool(nodes: dict[int, dict[str, Any]], node_id: int, lane: int, path: str) -> bool:
  node = nodes[node_id]
  require(node["op"] in {"LT", "CMPLT"} and len(node["src"]) == 2, path, "boolean expression must be CMPLT")
  require(node["dtype"] == "bool", f"{path}.nodes[{node_id}].dtype", "gate must have bool dtype")
  return eval_int(nodes, node["src"][0], lane, path) < eval_int(nodes, node["src"][1], lane, path)


def lane_map(nodes: dict[int, dict[str, Any]], node_id: int, path: str) -> tuple[int, ...]:
  return tuple(eval_int(nodes, node_id, lane, path) for lane in range(32))


def lane_mask(nodes: dict[int, dict[str, Any]], node_id: int, path: str) -> tuple[bool, ...]:
  return tuple(eval_bool(nodes, node_id, lane, path) for lane in range(32))


def width(node: dict[str, Any], path: str) -> int:
  shape = node["shape"]
  require(shape is not None, path + ".shape", "memory operation must have a shape")
  result = 1
  for extent in shape: result *= extent
  return max(result, 1)


def zero_alternate(nodes: dict[int, dict[str, Any]], node_id: int, expected: int, path: str) -> None:
  node = nodes[node_id]
  require(node["dtype"] == "float16", path + ".dtype", "A alternate must have float16 dtype")
  values = node["src"] if node["op"] == "STACK" else [node_id]
  require(len(values) == expected, path, "alternate width mismatch")
  for source in values:
    require(nodes[source]["dtype"] == "float16", f"{path}.nodes[{source}].dtype", "A alternate lane must be float16")
    value = const(nodes[source], f"{path}.nodes[{source}]")
    require(not isinstance(value, bool) and value == 0, path, "alternate must be all zero")


def memory(nodes: dict[int, dict[str, Any]], node_id: int, path: str) -> tuple[int, int, str]:
  node = nodes[node_id]
  require(node["op"] in {"INDEX", "SHRINK"}, path, "memory access must use INDEX/SHRINK")
  require(len(node["src"]) == (2 if node["op"] == "INDEX" else 3), path, "memory index must be one-dimensional")
  param = nodes[node["src"][0]]
  require(param["op"] == "PARAM", path, "memory access must resolve directly to PARAM")
  require(node["dtype"] == param["dtype"], path + ".dtype", "memory index dtype must match PARAM element dtype")
  require(nodes[node["src"][1]]["dtype"] in INTEGER_DTYPES, path + ".address.dtype", "address must have integer dtype")
  if node["op"] == "SHRINK":
    require(nodes[node["src"][2]]["dtype"] in INTEGER_DTYPES, path + ".size.dtype", "SHRINK size must have integer dtype")
  arg = exact(param["arg"], {"slot"}, path + ".param.arg")
  return integer(arg["slot"], path + ".param.arg.slot"), node["src"][1], param["dtype"]


def access_semantics(nodes: dict[int, dict[str, Any]], node_id: int, guard: int | None, wmma_id: int,
                     path: str) -> tuple[dict[str, Any], dict[int, int]]:
  node = nodes[node_id]; store = node["op"] == "STORE"; src = node["src"]
  require(node["op"] in {"LOAD", "STORE"}, path, "expected memory operation")
  if store: require(len(src) in {2, 3}, path, "STORE arity")
  else: require(len(src) in {1, 3}, path, "LOAD arity")
  slot,address_id,param_dtype = memory(nodes, src[0], path + ".index")
  role = {0: "C", 1: "A", 2: "B"}.get(slot)
  require(role is not None and (role == "C") == store, path, "unaccounted A/B/C access")
  expected_dtype = "float32" if store else "float16"
  require(param_dtype == expected_dtype, path, f"{role} PARAM must be {expected_dtype}")
  require(node["dtype"] == ("void" if store else "float16"), path + ".dtype", "memory operation dtype mismatch")
  access_width = width(node, path)
  address = lane_map(nodes, address_id, path)
  direct_gate = src[2] if len(src) == 3 else None
  require(not (direct_gate is not None and guard is not None), path, "STORE cannot have direct gate and IF guard")
  gate_id = direct_gate if direct_gate is not None else guard
  if gate_id is not None: require(nodes[gate_id]["dtype"] == "bool", path + ".gate.dtype", "memory gate must be bool")
  mask = (True,) * 32 if gate_id is None else lane_mask(nodes, gate_id, path)
  alternate = False
  if not store:
    require((len(src) == 3) == (direct_gate is not None), path, "LOAD gate/alternate pairing")
    if len(src) == 3:
      zero_alternate(nodes, src[1], access_width, path + ".alternate"); alternate = True
  enabled: set[int] = set(); disabled: set[int] = set(); counts: dict[int, int] = {}
  for base,active in zip(address, mask):
    for offset in range(access_width):
      target = base + offset
      (enabled if active else disabled).add(target)
      if active: counts[target] = counts.get(target, 0) + 1
  result_lane = None
  result_dtype = None
  if store:
    value = nodes[src[1]]
    require(value["dtype"] == "float32", path + ".value.dtype", "C STORE value must be float32")
    require(value["op"] == "INDEX" and len(value["src"]) == 2 and value["src"][0] == wmma_id,
            path, "C store value must index this stage's WMMA by source identity")
    require(nodes[value["src"][1]]["dtype"] in INTEGER_DTYPES, path + ".value.index.dtype",
            "WMMA result index must have integer dtype")
    lanes = lane_map(nodes, value["src"][1], path)
    require(len(set(lanes)) == 1, path, "WMMA result lane must be scalar")
    result_lane = lanes[0]; result_dtype = value["dtype"]
  semantic = {"role": role, "width": access_width, "address": address, "gate": mask if gate_id is not None else None,
              "alternate": alternate, "enabled": tuple(sorted(enabled)), "disabled": tuple(sorted(disabled)), "result_lane": result_lane,
              "dtypes": {"operation": node["dtype"], "parameter": param_dtype, "address": "integer",
                         "gate": "bool" if gate_id is not None else None,
                         "alternate": "float16" if alternate else None, "result": result_dtype,
                         "result_index": "integer" if store else None}}
  return semantic, counts


def ancestors(nodes: dict[int, dict[str, Any]], node_id: int) -> set[int]:
  result: set[int] = set(); stack = list(nodes[node_id]["src"])
  while stack:
    source = stack.pop()
    if source in result: continue
    result.add(source); stack.extend(nodes[source]["src"])
  return result


def validate_dtype_semantics(nodes: dict[int, dict[str, Any]], path: str) -> None:
  integer_ops = {"ADD", "MUL", "AND", "SHL", "SHR", "MULACC", "CAST", "SPECIAL"}
  float_ops = {"PARAM", "INDEX", "SHRINK", "LOAD", "STACK", "WHERE", "WMMA"}
  void_ops = {"SINK", "LINEAR", "GROUP", "IF", "ENDIF", "STORE"}
  allowed_ops = integer_ops | float_ops | void_ops | {"CONST", "LT", "CMPLT"}
  for node_id,node in nodes.items():
    node_path = f"{path}.nodes[{node_id}]"; op = node["op"]; src = node["src"]
    require(op in allowed_ops, node_path + ".op", f"unsupported EVID-02 operation {op}")
    if op == "CONST": const(node, node_path)
    elif op in integer_ops:
      require(node["dtype"] in INTEGER_DTYPES, node_path + ".dtype", f"{op} must have integer dtype")
    elif op in {"LT", "CMPLT"}:
      require(node["dtype"] == "bool" and len(src) == 2, node_path, "comparison must produce bool from two operands")
      require(all(nodes[source]["dtype"] in INTEGER_DTYPES for source in src), node_path, "comparison operands must be integers")
    elif op in float_ops:
      require(node["dtype"] in {"float16", "float32"}, node_path + ".dtype", f"{op} must have fixture float dtype")
    else:
      require(node["dtype"] == "void", node_path + ".dtype", f"{op} must have void dtype")

    if op in {"ADD", "MUL", "AND", "SHL", "SHR"}:
      require(len(src) == 2 and all(nodes[source]["dtype"] in INTEGER_DTYPES for source in src), node_path,
              f"{op} must consume two integer operands")
    elif op == "MULACC":
      require(len(src) == 3 and all(nodes[source]["dtype"] in INTEGER_DTYPES for source in src), node_path,
              "integer MULACC must consume three integer operands")
    elif op == "CAST":
      require(len(src) == 1 and nodes[src[0]]["dtype"] in INTEGER_DTYPES, node_path, "index CAST must consume one integer")
    elif op == "SPECIAL":
      require(len(src) == 1 and nodes[src[0]]["dtype"] in INTEGER_DTYPES, node_path, "SPECIAL bound must be integer")
    elif op == "PARAM":
      require(src and all(nodes[source]["dtype"] in INTEGER_DTYPES for source in src), node_path, "PARAM shape must be integer")
    elif op == "INDEX":
      require(len(src) == 2 and nodes[src[0]]["dtype"] == node["dtype"] and nodes[src[1]]["dtype"] in INTEGER_DTYPES,
              node_path, "INDEX must preserve value dtype and use an integer index")
    elif op == "SHRINK":
      require(len(src) == 3 and nodes[src[0]]["dtype"] == node["dtype"]
              and all(nodes[source]["dtype"] in INTEGER_DTYPES for source in src[1:]), node_path,
              "SHRINK must preserve value dtype and use integer offset/size")
    elif op == "LOAD":
      require(len(src) in {1, 3} and nodes[src[0]]["dtype"] == node["dtype"], node_path,
              "LOAD index and result dtypes must match")
      if len(src) == 3:
        require(nodes[src[1]]["dtype"] == node["dtype"] and nodes[src[2]]["dtype"] == "bool", node_path,
                "gated LOAD alternate/result dtypes must match and gate must be bool")
    elif op == "STORE":
      require(len(src) in {2, 3} and nodes[src[0]]["dtype"] == nodes[src[1]]["dtype"] == "float32", node_path,
              "C STORE address and value must be float32")
      if len(src) == 3: require(nodes[src[2]]["dtype"] == "bool", node_path, "STORE gate must be bool")
    elif op == "STACK":
      require(src and all(nodes[source]["dtype"] == node["dtype"] for source in src), node_path,
              "STACK sources must match result dtype")
    elif op == "WHERE":
      require(len(src) == 3 and nodes[src[0]]["dtype"] == "bool"
              and nodes[src[1]]["dtype"] == nodes[src[2]]["dtype"] == node["dtype"], node_path,
              "WHERE must consume bool and matching float values")
    elif op == "IF":
      require(len(src) == 2 and nodes[src[0]]["dtype"] == "bool", node_path, "IF condition must be bool")
    elif op == "ENDIF":
      require(len(src) == 1 and nodes[src[0]]["op"] == "IF", node_path, "ENDIF source must be IF")


def derive_stage(value: Any, path: str, expected_name: str) -> dict[str, Any]:
  nodes,root = parse_graph(value, path, expected_name)
  validate_dtype_semantics(nodes, path)
  expected_root = "LINEAR" if expected_name == "linearized" else "SINK"
  require(nodes[root]["op"] == expected_root, path + ".root", f"{expected_name} root must be {expected_root}")
  relevant = {op: [node_id for node_id,node in nodes.items() if node["op"] == op]
              for op in ("PARAM", "LOAD", "STORE", "WMMA", "IF", "ENDIF")}
  require(len(relevant["WMMA"]) == 1, path, "requires exactly one WMMA")
  wmma_id = relevant["WMMA"][0]; wmma = nodes[wmma_id]
  arg = exact(wmma["arg"], {"dims", "input_dtype", "device", "threads", "upcast_axes"}, path + ".wmma.arg")
  require(arg["dims"] == [16, 16, 16] and arg["input_dtype"] == "float16" and arg["threads"] == 32,
          path + ".wmma", "WMMA metadata mismatch")
  require(arg["device"] in {"AMD", "AMD_RDNA3"} and arg["upcast_axes"] is None, path + ".wmma", "WMMA target/upcast mismatch")
  require(wmma["dtype"] == "float32" and width(wmma, path + ".wmma") == 8 and len(wmma["src"]) == 3,
          path + ".wmma", "WMMA accumulator mismatch")
  a_operand,b_operand,accumulator = (nodes[source] for source in wmma["src"])
  require(a_operand["dtype"] == b_operand["dtype"] == "float16"
          and width(a_operand, path + ".wmma.a") == width(b_operand, path + ".wmma.b") == 16,
          path + ".wmma", "WMMA A/B operands must be 16-lane float16 fragments")
  require(accumulator["dtype"] == "float32" and width(accumulator, path + ".wmma.accumulator") == 8
          and accumulator["op"] == "STACK" and len(accumulator["src"]) == 8,
          path + ".wmma.accumulator", "WMMA accumulator must be an eight-lane float32 zero stack")
  for source in accumulator["src"]:
    require(nodes[source]["dtype"] == "float32" and const(nodes[source], f"{path}.nodes[{source}]") == 0.0,
            path + ".wmma.accumulator", "every accumulator lane must be a float32 zero")
  require(not any(nodes[source]["op"] in {"LOAD", "WMMA"} for source in ancestors(nodes, wmma["src"][2])),
          path + ".wmma.accumulator", "initial accumulator cannot depend on LOAD or WMMA")

  params = {}
  for node_id in relevant["PARAM"]:
    node = nodes[node_id]; slot = integer(exact(node["arg"], {"slot"}, f"{path}.nodes[{node_id}].arg")["slot"], path)
    require(slot not in params, path, "duplicate PARAM slot"); params[slot] = (node["dtype"], width(node, path))
  require(params == {0: ("float32", 80), 1: ("float16", 80), 2: ("float16", 256)}, path, "fixture PARAM ABI mismatch")

  guards: dict[int, int] = {}
  order_signature: list[str] = []
  if expected_name == "linearized":
    line = nodes[root]["src"]
    require(len(line) == len(set(line)), path + ".LINEAR", "LINEAR contains duplicate node identity")
    positions = {node_id: position for position,node_id in enumerate(line)}
    for op in ("LOAD", "STORE", "WMMA", "IF", "ENDIF"):
      require(set(relevant[op]).issubset(positions), path + ".LINEAR", f"every {op} must occur in actual LINEAR order")
    require(len(relevant["IF"]) == len(relevant["ENDIF"]) == 1, path, "LINEAR requires exactly one IF/ENDIF")
    if_id,endif_id = relevant["IF"][0],relevant["ENDIF"][0]
    if_node,endif_node = nodes[if_id],nodes[endif_id]
    require(len(if_node["src"]) == 2 and endif_node["src"] == [if_id], path, "ENDIF must own IF by source identity")
    if_position,endif_position = positions[if_id],positions[endif_id]
    require(endif_position == if_position + 2, path, "IF must immediately enclose exactly one operation")
    store_id = line[if_position + 1]
    require(store_id in relevant["STORE"] and nodes[store_id]["src"][0] == if_node["src"][1], path,
            "IF must own one C store address by source identity")
    guards[store_id] = if_node["src"][0]
    require(lane_mask(nodes, if_node["src"][0],  path) == tuple(lane < 16 for lane in range(32)), path, "partial C IF must be lane < 16")
    wmma_position = positions[wmma_id]
    require(all(positions[node_id] < wmma_position for node_id in relevant["LOAD"]), path, "all loads must precede WMMA in actual LINEAR order")
    require(all(positions[node_id] > wmma_position for node_id in relevant["STORE"]), path, "all stores must follow WMMA in actual LINEAR order")
  else:
    require(not relevant["IF"] and not relevant["ENDIF"], path, "late-final-rewrite cannot contain control flow")

  aggregate = {key: set() for key in ("a_enabled", "a_disabled", "b_enabled", "b_disabled", "c_enabled", "c_disabled")}
  b_counts: dict[int, int] = {}; c_counts: dict[int, int] = {}; accesses = {}
  for node_id in relevant["LOAD"] + relevant["STORE"]:
    semantic,counts = access_semantics(nodes, node_id, guards.get(node_id), wmma_id, f"{path}.nodes[{node_id}]")
    role = semantic["role"].lower(); aggregate[f"{role}_enabled"].update(semantic["enabled"]); aggregate[f"{role}_disabled"].update(semantic["disabled"])
    target_counts = b_counts if semantic["role"] == "B" else c_counts if semantic["role"] == "C" else None
    if target_counts is not None:
      for address,count in counts.items(): target_counts[address] = target_counts.get(address, 0) + count
    accesses[node_id] = semantic
  loads = [accesses[node_id] for node_id in relevant["LOAD"]]; stores = [accesses[node_id] for node_id in relevant["STORE"]]
  require(len(loads) == 20 and sum(access["role"] == "A" for access in loads) == 4, path, "expected four A and sixteen B loads")
  require(all(access["width"] == 4 and access["gate"] is not None and access["alternate"] for access in loads if access["role"] == "A"), path, "A loads must be gated width-four zero-fill")
  require(all(access["width"] == 1 and access["gate"] is None and not access["alternate"] for access in loads if access["role"] == "B"), path, "B loads must be ungated scalars")
  require(len(stores) == 3 and sorted(access["result_lane"] for access in stores) == [0, 1, 2], path, "C stores must consume WMMA lanes 0,1,2")
  a_load_ids = {node_id for node_id in relevant["LOAD"] if accesses[node_id]["role"] == "A"}
  b_load_ids = {node_id for node_id in relevant["LOAD"] if accesses[node_id]["role"] == "B"}
  a_operand_loads = {node_id for node_id in ancestors(nodes, wmma["src"][0]) if nodes[node_id]["op"] == "LOAD"}
  b_operand_loads = {node_id for node_id in ancestors(nodes, wmma["src"][1]) if nodes[node_id]["op"] == "LOAD"}
  require(a_operand_loads == a_load_ids, path + ".wmma.a",
          "WMMA A operand must depend on all and only the four padded A LOADs")
  require(b_operand_loads == b_load_ids, path + ".wmma.b",
          "WMMA B operand must depend on all and only the sixteen B LOADs")
  require(sum(access["gate"] is not None for access in stores) == 1, path, "exactly one C store must be gated/IF-owned")
  require(aggregate["a_enabled"] == set(range(80)) and aggregate["a_disabled"] == set(range(80, 256)), path, "A safety coverage mismatch")
  require(aggregate["b_enabled"] == set(range(256)) and not aggregate["b_disabled"], path, "B safety coverage mismatch")
  require(b_counts == {address: 2 for address in range(256)}, path, "B multiplicity mismatch")
  require(aggregate["c_enabled"] == set(range(80)) and aggregate["c_disabled"] == set(range(80, 96)), path, "C safety coverage mismatch")
  require(c_counts == {address: 1 for address in range(80)}, path, "C enabled addresses must be stored exactly once")
  if expected_name == "late-final-rewrite":
    require(sum(len(nodes[node_id]["src"]) == 3 for node_id in relevant["STORE"]) == 1, path, "late-final-rewrite needs one directly gated store")
  else:
    require(all(len(nodes[node_id]["src"]) == 2 for node_id in relevant["STORE"]), path, "LINEAR stores cannot retain direct gates")
    unguarded_c_stores = 0
    for node_id in nodes[root]["src"]:
      if node_id in accesses:
        access = accesses[node_id]; order_signature.append(f"{nodes[node_id]['op']}:{access['role']}:{access['result_lane']}")
        if nodes[node_id]["op"] == "STORE" and node_id not in guards:
          order_signature.pop(); unguarded_c_stores += 1
      elif node_id == wmma_id: order_signature.append("WMMA")
      elif nodes[node_id]["op"] in {"IF", "ENDIF"}: order_signature.append(nodes[node_id]["op"])
  # Source identity and positions above establish the actual order and IF
  # ownership. The two independent implementations may commute equivalent
  # unguarded C stores around the guarded region, so compare their validated
  # count separately while retaining the IF-owned store's actual order.
  order_signature = [entry.rsplit(":", 1)[0] if entry.startswith("STORE:C:") else entry for entry in order_signature]
  if expected_name == "linearized": order_signature.append(f"UNGUARDED_C_STORES:{unguarded_c_stores}")
  normalized_accesses = sorted(accesses.values(), key=lambda value: json.dumps(value, sort_keys=True))
  return {"name": expected_name, "wmma": {**arg, "device": "AMD", "dtype": wmma["dtype"], "width": 8},
          "accesses": normalized_accesses, "coverage": {key: tuple(sorted(value)) for key,value in aggregate.items()},
          "b_multiplicity": tuple(sorted(b_counts.items())), "actual_relevant_order": tuple(order_signature)}


def validate(document: Any, name: str) -> list[dict[str, Any]]:
  path = f"{name}:$"
  document = exact(document, {"schema_version", "evidence", "reference", "fixture", "stages"}, path)
  require(document["schema_version"] == 2 and document["evidence"] == "EVID-02", path, "wrong evidence schema")
  require(document["reference"] == "8c8b43de62515abe6c820b1de5aa26b30f48e43a", path, "wrong Tinygrad pin")
  fixture = exact(document["fixture"], {"m", "k", "n", "input_dtype", "accumulator_dtype", "target"}, path + ".fixture")
  require(fixture == {"m": 5, "k": 16, "n": 16, "input_dtype": "float16", "accumulator_dtype": "float32", "target": "gfx1151"}, path, "wrong fixture")
  stages = array(document["stages"], path + ".stages", 2)
  return [derive_stage(stages[0], path + ".stages[0]", "late-final-rewrite"),
          derive_stage(stages[1], path + ".stages[1]", "linearized")]


def adversarial_test(document: Any, name: str) -> None:
  def rejected(label: str, mutate) -> None:
    forged = copy.deepcopy(document); mutate(forged)
    try: validate(forged, f"{name}-{label}")
    except EvidenceError: return
    raise AssertionError(f"adversarial mutation was accepted: {label}")
  def linear(doc): return doc["stages"][1]
  def nodes(doc): return linear(doc)["nodes"]
  def root(doc): return nodes(doc)[linear(doc)["root"]]
  rejected("lane-map", lambda doc: next(node for node in nodes(doc) if node["op"] == "SPECIAL")["arg"].update(name="forged_lane"))
  rejected("omit-access", lambda doc: root(doc)["src"].remove(next(node["id"] for node in nodes(doc) if node["op"] == "LOAD")))
  def extra_control(doc):
    candidate = next(nodes(doc)[node_id] for node_id in root(doc)["src"] if nodes(doc)[node_id]["op"] not in {"LOAD", "STORE", "WMMA", "IF", "ENDIF"} and len(nodes(doc)[node_id]["src"]) >= 2)
    candidate["op"] = "IF"; candidate["src"] = candidate["src"][:2]; candidate["arg"] = None
  rejected("extra-if", extra_control)
  def reorder(doc):
    line = root(doc)["src"]; load = next(i for i,node_id in enumerate(line) if nodes(doc)[node_id]["op"] == "LOAD"); wmma = next(i for i,node_id in enumerate(line) if nodes(doc)[node_id]["op"] == "WMMA")
    line[load],line[wmma] = line[wmma],line[load]
  rejected("actual-order", reorder)
  def second_wmma(doc):
    original = next(node for node in nodes(doc) if node["op"] == "WMMA")
    candidate = next(nodes(doc)[node_id] for node_id in root(doc)["src"]
                     if node_id > original["id"] and nodes(doc)[node_id]["op"] not in {"LOAD", "STORE", "WMMA", "IF", "ENDIF"})
    candidate.update(op="WMMA", dtype=original["dtype"], shape=original["shape"], src=original["src"], arg=copy.deepcopy(original["arg"]))
  rejected("second-wmma", second_wmma)
  def expression_ast(doc):
    candidate = next(node for node in nodes(doc) if node["op"] == "ADD" and len(node["src"]) == 2)
    candidate["op"] = "MUL"
  rejected("expression-ast", expression_ast)
  def replace_wmma_data(doc):
    table = nodes(doc); wmma = next(node for node in table if node["op"] == "WMMA")
    zero = next(node["id"] for node in table if node["op"] == "CONST" and node["dtype"] == "float16")
    replacement_id = wmma["src"][1]; replacement = table[replacement_id]
    if replacement["op"] != "STACK" or width(replacement, "self-test") != 16:
      raise AssertionError("self-test fixture has no 16-lane B fragment")
    replacement["src"] = [zero] * 16
    wmma["src"][:2] = [replacement_id, replacement_id]
  rejected("wmma-data-operands", replace_wmma_data)
  def mutate_a_load_dtype(doc):
    table = nodes(doc)
    for load in (node for node in table if node["op"] == "LOAD"):
      index = table[load["src"][0]]; param = table[index["src"][0]]
      if param["op"] == "PARAM" and param["arg"] == {"slot": 1}:
        load["dtype"] = "float32"; return
    raise AssertionError("self-test fixture has no A LOAD")
  rejected("a-load-float32", mutate_a_load_dtype)
  def mutate_accumulator_dtype(doc):
    table = nodes(doc); wmma = next(node for node in table if node["op"] == "WMMA")
    table[wmma["src"][2]]["dtype"] = "float16"
  rejected("accumulator-dtype", mutate_accumulator_dtype)
  rejected("result-dtype", lambda doc: next(node for node in nodes(doc) if node["op"] == "WMMA").update(dtype="float16"))
  def reroute_c_store(doc):
    table = nodes(doc); wmma = next(node for node in table if node["op"] == "WMMA")
    store = next(node for node in table if node["op"] == "STORE")
    table[store["src"][1]]["src"][0] = wmma["src"][2]
  rejected("c-store-other-wmma-fragment", reroute_c_store)


def self_test() -> None:
  try: validate({"schema_version": 2, "evidence": "EVID-02"}, "self-test")
  except EvidenceError: pass
  else: raise AssertionError("malformed EVID-02 document was accepted")


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("left", nargs="?", type=Path); parser.add_argument("right", nargs="?", type=Path)
  parser.add_argument("--left-name", default="svod"); parser.add_argument("--right-name", default="tinygrad")
  parser.add_argument("--self-test", action="store_true")
  args = parser.parse_args()
  if args.self_test: self_test(); return
  if args.left is None or args.right is None: parser.error("left and right evidence files are required")
  left,right = json.loads(args.left.read_text()),json.loads(args.right.read_text())
  try:
    left_semantics = validate(left, args.left_name); right_semantics = validate(right, args.right_name)
    adversarial_test(left, args.left_name); adversarial_test(right, args.right_name)
  except EvidenceError as error: print(f"invalid EVID-02 evidence: {error}"); raise SystemExit(2) from error
  if left_semantics != right_semantics:
    print("EVID-02 independently derived safety semantics mismatch")
    raise SystemExit(1)
  print("EVID-02 source-graph safety parity: strict match (adversarial mutations rejected)")


if __name__ == "__main__": main()
