---
sidebar_label: ONNX Inference
---

# ONNX Model Inference

Morok's ONNX importer is the recommended way to run model inference. It loads standard `.onnx` files, decomposes operators into Morok's lazy tensor operations, and compiles them through the full optimization pipeline — no C++ runtime required.

**Current status:**

| Capability | Status |
|------------|--------|
| Forward inference | Supported |
| 162 / 200 ONNX operators | [Parity details](https://github.com/patsak/morok/blob/main/onnx/PARITY.md) |
| CNN architectures (ResNet, DenseNet, VGG, ...) | 9 models validated |
| Microsoft extensions (Attention, RotaryEmbedding) | Supported |
| Dynamic batch size | Planned for next release |
| Training / backward pass | Not supported |

**How does Morok compare to other Rust ML frameworks?**

Among pure-Rust frameworks, Morok offers the broadest ONNX operator coverage — 162 operators with 1361 passing conformance tests across dual backends (Clang + LLVM). `candle` and `burn` each support fewer operators and lack conformance test suites of comparable scope. That said, if you need maximum compatibility with production ONNX models, use `ort` — a Rust wrapper around the C++ ONNX Runtime — which covers the full ONNX spec.

---

## Quick Start

Add `morok-onnx` and `morok-tensor` to your `Cargo.toml`:

```toml
[dependencies]
morok-onnx = { git = "https://github.com/patsak/morok" }
morok-tensor = { git = "https://github.com/patsak/morok" }
```

### Simple: All-Initializer Models

For models where all inputs are baked into the file (no runtime inputs):

```rust
use morok_onnx::OnnxImporter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut importer = OnnxImporter::new();
    let outputs = importer.import_path("model.onnx")?;

    // Each output is a lazy Tensor — realize to get data
    for (name, tensor) in &outputs {
        let result = tensor.realize()?;
        println!("{name}: {:?}", result.to_ndarray::<f32>()?);
    }
    Ok(())
}
```

### Two-Phase: Models with Runtime Inputs

Most models need runtime data (images, tokens, audio). The two-phase API separates graph preparation from execution:

```rust
use morok_onnx::OnnxImporter;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let importer = OnnxImporter::new();

    // Phase 1: Parse structure (no execution)
    let graph = importer.prepare(model_proto)?;

    // Inspect what the model needs
    for (name, spec) in &graph.inputs {
        println!("{name}: shape={:?}, dtype={:?}", spec.shape, spec.dtype);
    }

    // Phase 2: Build lazy computation graph
    let (inputs, outputs) = importer.trace(&graph)?;

    // Execute
    let result = outputs["output"].realize()?;
    Ok(())
}
```

The `prepare()` / `trace()` split exists because graph structure is static — you parse it once and can trace multiple times with different dimension bindings or input data.

---

## Architecture

### Two-Phase Design

The importer processes ONNX models in two distinct phases:

**Phase 1 — `prepare()`:** Extracts graph topology without executing anything. Parses the protobuf, separates initializers (weights) from runtime inputs, records opset versions, and pre-parses control flow subgraphs. Returns an `OnnxGraph` — a lightweight structure you can inspect before committing to execution.

**Phase 2 — `trace()`:** Walks the graph in topological order, dispatching each ONNX node to its Tensor implementation. This builds Morok's lazy computation DAG — no actual math happens yet. The result is a set of `Tensor` handles that, when `realize()`'d, compile and execute through the full pipeline.

```text
model.onnx → prepare() → OnnxGraph → trace() → lazy Tensors → realize() → results
                 │                        │
                 │ structure only         │ builds computation DAG
                 │ (no execution)         │ (no execution)
                 ▼                        ▼
          Inspect inputs/outputs    Pass to optimizer/codegen
```

This separation enables several patterns:
- **Inspect before executing:** Check input shapes and dtypes before allocating anything
- **Multiple traces:** Re-trace with different dynamic dimension bindings
- **External weights:** Load weights separately (useful for models with external data files)

### Operator Decomposition

Every ONNX operator is decomposed into Morok Tensor operations. The complexity varies:

**Direct mappings** — about 60 operators map 1:1 to a tensor method:

```rust
// In the registry:
"Add" => x.try_add(y)?
"Relu" => x.relu()?
"Sigmoid" => x.sigmoid()?
"Equal" => x.try_eq(y)?
```

**Builder patterns** — complex operators with many optional parameters use fluent APIs:

```rust
// Conv with optional bias, padding, dilation, groups
x.conv()
    .weight(w)
    .maybe_bias(bias)
    .auto_pad(AutoPad::SameLower)
    .group(32)
    .maybe_dilations(Some(&[2, 2]))
    .call()?
```

**Multi-step decompositions** — operators like BatchNormalization, Attention, and Mod require intermediate computations. For example, Python-style integer `Mod` decomposes into truncation mod + sign adjustment:

```rust
let trunc_mod = x.try_mod(y)?;
let signs_differ = trunc_mod.bitwise_xor(y)?.try_lt(&zero)?;
let needs_adj = mod_ne_zero.bitwise_and(&signs_differ)?;
trunc_mod.try_add(&y.where_(&needs_adj, &zero)?)?
```

### Attribute Validation

The `Attrs` helper uses pop-based extraction — each call to `attrs.int("axis", -1)` or `attrs.float("epsilon", 1e-5)` removes the attribute from the map. After the operator finishes, `attrs.done()` asserts the map is empty. Any leftover attributes trigger an error, catching incomplete operator implementations at trace time rather than producing silent wrong results.

### Opset Versioning

ONNX models declare opset imports per domain. The importer tracks these and passes the version to each operator handler. Operators switch behavior based on version — for example, `Softmax`'s default axis changed from `1` (opset < 13) to `-1` (opset >= 13), and `ReduceSum` moved its axes from an attribute to an input tensor at opset 13.

---

## Working with Models

### Dynamic Dimensions

ONNX inputs can have symbolic dimensions like `"batch_size"` or `"sequence_length"`. Bind them at trace time:

```rust
let graph = importer.prepare(model)?;

// Bind symbolic dims to concrete values
let (inputs, outputs) = importer.trace_with_dims(
    &graph,
    &[("batch_size", 1), ("sequence_length", 512)],
)?;
```

Unbound dynamic dimensions cause a clear error at trace time. You can inspect which dimensions are dynamic via `InputSpec::shape`:

```rust
for (name, spec) in &graph.inputs {
    for dim in &spec.shape {
        match dim {
            DimValue::Static(n) => print!("{n} "),
            DimValue::Dynamic(name) => print!("{name}? "),
        }
    }
}
```

### External Weights

Some ONNX models store weights in separate files. Use `trace_external()` to provide them:

```rust
let (inputs, outputs) = importer.trace_external(
    &graph,
    external_weights,  // HashMap<String, Tensor>
)?;
```

### Microsoft Extensions

The importer supports several `com.microsoft` contrib operators commonly found in transformer models exported from ONNX Runtime:

| Extension | What it does |
|-----------|-------------|
| `Attention` | Packed QKV projection with masking, past KV cache |
| `RotaryEmbedding` | Rotary positional embeddings (interleaved/non-interleaved) |
| `SkipLayerNormalization` | Fused residual + LayerNorm + scale |
| `EmbedLayerNormalization` | Token + position + segment embeddings → LayerNorm |

Standard ONNX transformer operators (`Attention` from the ai.onnx domain) are also supported with grouped query attention (GQA), causal masking, past KV caching, and softcap.

---

## Control Flow and Limitations

### Semantic If: Both Branches Always Execute

ONNX's `If` operator has data-dependent control flow — the condition determines which branch runs. Morok's lazy evaluation model is fundamentally incompatible with this: since nothing executes at trace time, the condition value is unknown.

**Morok's solution:** Trace *both* branches, then merge results with `Tensor::where_()`:

```text
ONNX:    if condition { then_branch } else { else_branch }
Morok:   then_result.where_(&condition, &else_result)
```

This enables **trace-once, run-many** — the compiled graph handles any condition value at runtime. But it has a hard constraint: **both branches must produce identical output shapes and dtypes.** Models with shape-polymorphic branches (where the then-branch produces `[3, 4]` and the else-branch produces `[5, 6]`) cannot be traced.

In practice, most ONNX models with `If` nodes satisfy this constraint because they use conditional logic for value selection, not shape-changing control flow.

### No Loop or Scan

Iterative control flow (`Loop`, `Scan`) is not implemented. These operators require repeated tracing or unrolling, which conflicts with the single-trace architecture. Models using recurrent patterns typically work via unrolled operators (LSTM, GRU, RNN are implemented as native ops).

### No Batching (Yet)

Dynamic batching — running inference on multiple inputs simultaneously — is planned for the next release. Currently, batch dimensions must be bound to a fixed value at trace time via `trace_with_dims()`.

### No Training

The importer is inference-only. There is no backward pass, gradient computation, or optimizer support.

### Missing Operator Categories

| Category | Examples | Why |
|----------|----------|-----|
| Quantization | DequantizeLinear, QuantizeLinear | Requires quantized dtype support in IR |
| Sequence ops | SequenceConstruct, SequenceAt | Non-tensor types not in Morok's type system |
| Random | RandomNormal, RandomUniform | Stateful RNG not yet implemented |
| Signal processing | DFT, STFT, MelWeightMatrix | Low priority; niche use cases |
| Text | StringNormalizer, TfIdfVectorizer | String types not supported |

For models using these operators, consider `ort` (ONNX Runtime wrapper) which covers the full spec.

---

## Debugging

### Per-Node Output Tracing

Set the trace log level to dump intermediate outputs:

```bash
RUST_LOG=morok_onnx::importer=trace cargo run
```

This realizes each node's output individually and prints the first 5 values — useful for numerical bisection when a model produces wrong results. Note that this breaks kernel fusion (each node runs separately), so it's purely a debugging tool.

### Inspecting the Graph

Use the `OnnxGraph` structure to understand what a model needs before tracing:

```rust
let graph = importer.prepare(model)?;

println!("Inputs:");
for (name, spec) in &graph.inputs {
    println!("  {name}: {:?} {:?}", spec.shape, spec.dtype);
}

println!("Outputs: {:?}", graph.output_names());
println!("Nodes: {}", graph.nodes.len());
println!("Initializers: {}", graph.initializers.len());
```

---

## Summary

| Aspect | Detail |
|--------|--------|
| **Entry point** | `OnnxImporter::new()` |
| **Simple import** | `importer.import_path("model.onnx")?` |
| **Two-phase** | `prepare()` → `trace()` / `trace_with_dims()` |
| **Operators** | 162 / 200 ([full parity table](https://github.com/patsak/morok/blob/main/onnx/PARITY.md)) |
| **Validated models** | ResNet50, DenseNet121, VGG19, Inception, AlexNet, ShuffleNet, SqueezeNet, ZFNet |
| **Backends** | Clang + LLVM (identical results) |
| **Extensions** | com.microsoft Attention, RotaryEmbedding, SkipLayerNorm, EmbedLayerNorm |
| **Limitations** | No training, no batching (yet), no Loop/Scan, shape-polymorphic If |

**Next:** [Hands-On Examples](./examples) for tensor basics, or [Execution Pipeline](./architecture/pipeline) for how compilation works under the hood.
