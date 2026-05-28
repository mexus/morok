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
use svod_device::amd::{AmdAllocator, AmdComputeQueue, AmdGraph, AmdProgram, KernargArena, SignalPool};
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
    // Build the per-device runtime state ONCE: queue, kernarg arena, signal
    // pool. AmdProgram::execute reuses these across dispatches. The
    // AmdAllocator we instantiate here shares its underlying `Arc<AmdDevice>`
    // with the one cached by the registry (see DEVICE_CACHE in
    // device.rs), so we don't double-open `/dev/kfd` or ACQUIRE_VM twice.
    let amd_alloc = AmdAllocator::new(device_id)?;
    let device_handle = Arc::clone(&amd_alloc.dev);
    let queue = AmdComputeQueue::create(&amd_alloc)?;
    let arena = KernargArena::new(&amd_alloc, device_handle.core())?;
    let signal_pool = SignalPool::new(&amd_alloc)?;
    // Seed the pool onto the device core so any future `AmdConnector` built
    // directly against the core (e.g. graph/plan connectors in Commit B)
    // can acquire its timeline signal without reaching back through an
    // `AmdProgram`. Idempotent — only the first call wins.
    device_handle.core().install_signal_pool(Arc::clone(&signal_pool));
    // Install the device-global timeline signal. Mirrors tinygrad
    // `HCQCompiled.__init__` (`hcq.py:415`): one signal owned by the device,
    // reused across all submits + waits. `AmdAllocator::free` synchronizes
    // against this before unmapping, which fixes the page-aligned NotPresent
    // faults caused by tearing down a buffer mapping while the GPU still has
    // pending references to its VA.
    device_handle.init_timeline(Arc::new(signal_pool.acquire()?));
    let runtime: RuntimeFactory = Arc::new(move |compiled: &CompiledSpec| -> Result<Box<dyn Program>> {
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
        let prg = AmdProgram::load(
            Arc::clone(&device_handle),
            &alloc,
            Arc::clone(&queue),
            Arc::clone(&arena),
            Arc::clone(&signal_pool),
            &compiled.bytes,
            &compiled.name,
            compiled.buf_count,
            compiled.var_names.len(),
        )?;
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
        spec.apply_derived_metadata_from_ast();
        if spec.buf_count == 0 {
            spec.buf_count = rendered.buffer_args.len();
        }
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn amd_arch(&self) -> Option<AmdArch> {
        Some(self.arch)
    }

    fn decompositor(&self) -> Option<svod_ir::pattern::TypedPatternMatcher<()>> {
        // AMD's hardware exp2/log2 are lower precision than CPU libm; route the
        // exp/log/trig family through the SLEEF polynomial pass (sqrt stays
        // native). See `amd_decomposition_patterns` for the tinygrad rationale.
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
        let mut compiled = CompiledSpec::from_bytes(spec.name.clone(), bytes, spec.ast.clone());
        compiled.var_names = spec.var_names.clone();
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        compiled.buf_count = spec.buf_count;
        Ok(compiled)
    }

    fn cache_key(&self) -> &'static str {
        "amd-clang"
    }
}
