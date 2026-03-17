# morok-codegen

Backend code generation from optimized UOp graphs.

## Example

```rust
use morok_codegen::{Renderer, render};

let code = render(&kernel_graph, backend)?;
```

## Backends

| Backend | Output | Feature | Default |
|---------|--------|---------|---------|
| **Clang** | C source → `clang -shared -O2` → `.so` | always | yes |
| **LLVM JIT** | LLVM IR text → Inkwell ExecutionEngine | always | no |
| **MLIR** | MLIR (arith/scf/llvm dialects) → MLIR ExecutionEngine | `mlir` | no |

Select at runtime via `MOROK_CPU_BACKEND` env var (`clang`, `llvm`, `mlir`).

**Planned:**

- PTX renderer (CUDA)
- Metal renderer
- WebGPU (WGSL) renderer

## Testing

```bash
cargo test -p morok-codegen
```
