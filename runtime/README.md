# morok-runtime

Kernel execution interface bridging codegen to hardware.

## Example

```rust
use morok_runtime::CompiledKernel;

let kernel = compile(code)?;
kernel.execute(&[buf_a.ptr(), buf_b.ptr(), buf_out.ptr()])?;
```

## Backends

| Backend | How it works | Feature |
|---------|-------------|---------|
| **Clang** (default) | Compiles C to `.so` via `clang -shared -O2`, loads with `dlopen` | always |
| **LLVM JIT** | JIT compiles LLVM IR via Inkwell ExecutionEngine | always |
| **MLIR** | Lowers MLIR dialects to LLVM, JIT via MLIR ExecutionEngine | `mlir` |

Select at runtime: `MOROK_CPU_BACKEND=clang|llvm|mlir`

**Planned:**

- CUDA kernel execution
- Metal kernel execution

## Testing

```bash
cargo test -p morok-runtime
```
