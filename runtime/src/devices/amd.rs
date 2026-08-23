//! AMD GPU device factory.
//!
//! Wires together:
//! - `svod_codegen::llvm::LlvmTextRenderer::amd(arch)` for IR emission.
//! - `svod_runtime::amd::compile_ir_to_amd_object` for clang amdgcn compile.
//! - `svod_device::amd::AmdProgram` for ELF load + AQL dispatch.
//!
//! Construction returns `Err(NoAmdGpu)` cleanly on hosts that don't have a
//! supported AMD GPU; never panics.

use std::sync::Arc;

use svod_codegen::llvm::LlvmTextRenderer;
use svod_device::Result;
use svod_device::amd::{AmdAllocator, AmdCopyQueue, AmdGraph, AmdProgram, SignalPool};
use svod_device::device::{
    CompiledSpec, Compiler, Device, Graph, GraphFactory, GraphKernel, Program, ProgramSpec, Renderer, RuntimeFactory,
};
use svod_device::registry::DeviceRegistry;
use svod_dtype::{AmdArch, DeviceSpec};
use svod_ir::UOp;

/// Create an `AMD:N` device end-to-end (allocator + renderer + compiler +
/// runtime). The arch is queried from KFD topology at device-open time and
/// stored on the opened `AmdDevice` (NOT in the `DeviceSpec`). The
/// `arch` parameter is the cache-key hint — kept so the compiler can emit
/// the right `-mcpu`.
pub fn create_amd_device(registry: &DeviceRegistry, device_id: usize, arch: AmdArch) -> Result<Device> {
    let spec = DeviceSpec::Amd { device_id };
    let allocator = registry.get(&spec)?;
    let renderer = Arc::new(AmdRendererWrapper { device: spec.clone(), arch });
    let compiler = Arc::new(AmdCompiler { arch });
    // Build the per-device process-shared state: the signal pool (singleton
    // per physical AMD:N, lives on AmdDeviceCore). Each `ExecutionPlan` /
    // `AmdGraph` / per-call `Program::execute` leases or builds its OWN
    // connector (own KFD ring + kernarg arena + scratch + timeline), so no
    // compute-queue or arena is pre-built here.
    let amd_alloc = AmdAllocator::new(device_id)?;
    let device_handle = Arc::clone(&amd_alloc.dev);
    // Signal-pool sizing: per-op AQL dispatch needs only a few slots, but a
    // captured DAG graph reserves one slot per kernel (low hundreds) for its
    // lifetime, across several concurrent owners. 1024 slots (64 KiB GTT) covers
    // that with headroom; the pool rounds up to whole 64-slot pages.
    // Sized for the worst combination: a captured graph reserves a slot per
    // kernel for its lifetime while a profiled execution holds every
    // dispatch's signal until harvest. Slots are 64 B each — 4096 is 256 KiB
    // of GTT, cheap insurance against `SignalPool exhausted`.
    let signal_pool = SignalPool::new(&amd_alloc, 4096)?;
    // Seed the pool onto the device core so `PoolQueue::new_with_resources`
    // can acquire its PM4 counter signal.
    device_handle.core().install_signal_pool(signal_pool);
    // Bring up the SDMA copy queue so host↔device staging works — this is what
    // lets buffers be device-local (non-host-visible). A creation failure
    // cleanly leaves has_sdma_queue=false, so buffers stay host-visible and use
    // the memmove copy path (today's behaviour). Must run before any _alloc,
    // which reads has_sdma_queue to decide cpu_access.
    match AmdCopyQueue::create(&amd_alloc) {
        Ok(copy_queue) => {
            device_handle.core().install_copy_queue(copy_queue);
            device_handle.core().set_has_sdma_queue(true);
        }
        Err(e) => {
            tracing::warn!(error = %e, "SDMA copy queue unavailable; AMD buffers stay host-visible");
        }
    }
    // PM4 graph capture is opt-in via `SVOD_PM4_GRAPH=1` (default OFF — it
    // regresses on gfx1151). Parse the env ONCE here into the per-device flag so
    // the capture path reads a plain bool: only "1" enables; "0"/empty/unset stay
    // OFF (presence alone no longer enables it), and there is no per-capture env
    // lookup to race with test toggles.
    if std::env::var("SVOD_PM4_GRAPH").as_deref() == Ok("1") {
        device_handle.core().set_pm4_graph(true);
    }
    // No default connector: every dispatcher leases/owns its own connector
    // (`Program::execute` leases per call; plans/graphs hold one for their
    // lifetime). The pool starts empty and warms on first lease.
    let runtime: RuntimeFactory = Arc::new(move |compiled: &CompiledSpec| -> Result<Box<dyn Program>> {
        svod_device::device::validate_abi_descriptors(&compiled.abi, compiled.buf_count, &compiled.var_names)?;
        // `CompiledSpec.bytes` is the clang-produced amdgcn ELF.
        if compiled.bytes.is_empty() {
            return Err(svod_device::Error::Runtime {
                message: "AMD RuntimeFactory: CompiledSpec has empty ELF bytes".into(),
            });
        }
        // We need an AmdAllocator inside the closure for AmdProgram::load
        // (it allocates the code-object VRAM buffer). Constructing a fresh
        // one is cheap — the shared DEVICE_CACHE returns the same
        // Arc<AmdDevice>, so no kernel ioctls re-execute.
        let alloc = AmdAllocator::new(device_id)?;
        let prg = AmdProgram::load(Arc::clone(&device_handle), &alloc, &compiled.bytes, &compiled.name, &compiled.abi)?;
        Ok(Box::new(prg) as Box<dyn Program>)
    });

    // Graph factory: pre-build a PM4 indirect buffer for a captured kernel
    // chain and replay it with one doorbell (`svod_device::amd::AmdGraph`).
    // Returns `Ok(None)` when the chain isn't graphable (AQL queue, non-AMD
    // program), so the caller falls back to per-call dispatch. A fresh
    // AmdAllocator shares the cached `Arc<AmdDevice>`, so capture allocates the
    // IB page through the same KFD VM with no extra device open.
    let graph: GraphFactory = Arc::new(move |kernels: &[GraphKernel]| -> Result<Option<Box<dyn Graph>>> {
        let alloc = AmdAllocator::new(device_id)?;
        AmdGraph::capture(&alloc, kernels)
    });

    Ok(Device::new(spec, allocator, renderer, compiler, runtime).with_graph(graph))
}

struct AmdRendererWrapper {
    device: DeviceSpec,
    arch: AmdArch,
}

impl Renderer for AmdRendererWrapper {
    fn render(&self, ast: &Arc<UOp>, name: Option<&str>) -> Result<ProgramSpec> {
        let renderer = LlvmTextRenderer::amd(self.arch);
        let rendered = svod_codegen::Renderer::render(&renderer, ast, name.or(Some("kernel")))
            .map_err(|e| svod_device::Error::Runtime { message: format!("AMD IR rendering failed: {e}") })?;
        let mut spec = ProgramSpec::new(rendered.name.clone(), rendered.code.clone(), self.device.clone(), ast.clone());
        spec.set_var_names(rendered.var_names.clone());
        spec.abi = rendered.abi.clone();
        if spec.buf_count == 0 {
            spec.buf_count = rendered.buffer_args.len();
        }
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn gpu_arch(&self) -> Option<svod_dtype::GpuArch> {
        Some(svod_dtype::GpuArch::Amd(self.arch))
    }

    fn supported_ops(&self) -> svod_ir::RendererOps {
        let mut ops = svod_ir::RendererOps::all();
        ops.binary.remove(&svod_ir::BinaryOp::Threefry);
        ops.binary.remove(&svod_ir::BinaryOp::Pow);
        ops.binary.remove(&svod_ir::BinaryOp::Max);
        for op in [
            svod_ir::UnaryOp::Exp,
            svod_ir::UnaryOp::Log,
            svod_ir::UnaryOp::Cos,
            svod_ir::UnaryOp::Tan,
            svod_ir::UnaryOp::Erf,
        ] {
            ops.unary.remove(&op);
        }
        ops
    }

    fn decompositor(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        // Target Exp2/Log2/Sin/Sqrt selection is centralized in the scheduler.
        // This matcher only handles Morok's additional transcendental ops.
        Some(svod_ir::decompositions::amd_decomposition_patterns())
    }
}

struct AmdCompiler {
    arch: AmdArch,
}

impl Compiler for AmdCompiler {
    fn compile(&self, spec: &ProgramSpec) -> Result<CompiledSpec> {
        let bytes = crate::amd::compile_ir_to_amd_object(&spec.src, self.arch)
            .map_err(|e| svod_device::Error::Runtime { message: format!("AMD clang compile failed: {e}") })?;
        let mut compiled = CompiledSpec::from_bytes(spec.name.clone(), bytes, spec.ast.clone(), spec.abi.clone())?;
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &'static str {
        "amd-clang"
    }
}
