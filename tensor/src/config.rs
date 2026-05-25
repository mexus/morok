use std::sync::Arc;

use snafu::ResultExt;
use svod_device::device::Device;
use svod_device::registry::DeviceRegistry;
use svod_ir::DeviceSpec;
use svod_runtime::CpuBackend;
use svod_schedule::OptimizerConfig;

use crate::error::{DeviceFactorySnafu, DeviceSnafu};

/// Resolves a `DeviceSpec` into a concrete `Device` for compilation.
///
/// Implementations control which codegen backend is used for each device type.
/// This enables per-call backend selection instead of relying on the
/// `DEVICE_FACTORIES` singleton (which bakes one backend per device spec).
pub(crate) trait DeviceResolver: Send + Sync {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>>;
}

/// Default resolver: delegates to `DEVICE_FACTORIES` singleton (reads env vars
/// like `SVOD_CPU_BACKEND` at first device creation, then caches).
struct EnvResolver;

impl DeviceResolver for EnvResolver {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        svod_runtime::DEVICE_FACTORIES.device(spec, registry).context(DeviceFactorySnafu)
    }
}

/// Creates CPU devices with a specific backend; delegates other device types
/// to `DEVICE_FACTORIES`. This is the resolver used by `PrepareConfig::for_cpu_backend()`.
struct CpuBackendResolver(CpuBackend);

impl DeviceResolver for CpuBackendResolver {
    fn resolve(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        match spec {
            DeviceSpec::Cpu => {
                Ok(Arc::new(svod_runtime::create_cpu_device_with_backend(registry, self.0).context(DeviceSnafu)?))
            }
            _ => svod_runtime::DEVICE_FACTORIES.device(spec, registry).context(DeviceFactorySnafu),
        }
    }
}

/// Configuration for `prepare()`/`realize()` that bundles optimizer settings
/// with device resolution (codegen backend selection).
///
/// Instead of relying on the `SVOD_CPU_BACKEND` env var (global mutable state),
/// the backend is selected per-call via a [`DeviceResolver`].
#[allow(rustdoc::private_intra_doc_links)]
#[derive(Clone)]
pub struct PrepareConfig {
    pub optimizer: OptimizerConfig,
    pub(crate) resolver: Arc<dyn DeviceResolver>,
    /// When `true`, force the cache-cold rangeify/scheduling path even if
    /// `SVOD_DISABLE_SCHEDULE_CACHE` is unset. Primarily useful in tests
    /// that need to compare cache-warm vs cache-cold outputs without mutating
    /// process-global env state.
    pub disable_schedule_cache: bool,
}

impl std::fmt::Debug for PrepareConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrepareConfig")
            .field("optimizer", &self.optimizer)
            .field("disable_schedule_cache", &self.disable_schedule_cache)
            .finish_non_exhaustive()
    }
}

impl Default for PrepareConfig {
    fn default() -> Self {
        Self { optimizer: OptimizerConfig::default(), resolver: Arc::new(EnvResolver), disable_schedule_cache: false }
    }
}

impl PrepareConfig {
    /// Read both `SVOD_CPU_BACKEND` and optimizer env vars.
    pub fn from_env() -> Self {
        // Re-resolve `SVOD_DEVICE` through the registry parser so all
        // consumers (tensor construction, schedule rangeify, runtime factory)
        // see the same canonical DeviceSpec value.
        normalize_default_device_from_env();
        Self { optimizer: OptimizerConfig::from_env(), resolver: Arc::new(EnvResolver), disable_schedule_cache: false }
    }

    /// Convenience constructor: specific CPU backend with optimizer settings
    /// resolved from env (`BEAM`, `SVOD_NOOPT`, `IGNORE_BEAM_CACHE`,
    /// `BEAM_*`, `SVOD_*`). Used by the `codegen_tests!` macro so a single
    /// `BEAM=4 cargo test` flips every codegen-test target to BEAM
    /// without changing test bodies.
    pub fn for_cpu_backend(backend: CpuBackend) -> Self {
        Self {
            optimizer: OptimizerConfig::from_env(),
            resolver: Arc::new(CpuBackendResolver(backend)),
            disable_schedule_cache: false,
        }
    }

    /// AMD variant for the `codegen_tests!` macro: returns `Some(_)` only
    /// when this host has a [supported](svod_dtype::AmdArch) AMD GPU
    /// (RDNA3 + CDNA). On other hosts the macro's `amd::*` tests skip with
    /// a clear message.
    ///
    /// **Status**: the AMD realize pipeline (CPU→VRAM staging + dispatch +
    /// result copy-back) is not yet wired in `realize.rs`; until it is, this
    /// function returns `None` even on supported hardware. The macro
    /// scaffold is in place so that flipping the pipeline integration is a
    /// one-line change here.
    #[cfg(target_os = "linux")]
    pub fn for_amd_if_available() -> Option<Self> {
        // Detect supported AMD device. Returns None when the host has no
        // /dev/kfd, no GPU nodes, or only unsupported gfx targets.
        let _arch = amd_test_arch()?;
        // TODO(phase 7.1): swap to an AmdBackendResolver once realize.rs
        // supports cross-device buffer staging. Returning None for now means
        // the codegen_tests!::amd variant always skips on this host — by
        // design, not a bug.
        None
    }

    #[cfg(not(target_os = "linux"))]
    pub fn for_amd_if_available() -> Option<Self> {
        None
    }
}

/// Detect a supported AMD GPU on this host. Returns the gfx-family arch of
/// device 0 when (a) `/dev/kfd` exists, (b) KFD topology has a GPU node, and
/// (c) the gfx target maps to one of `AmdArch`'s supported families
/// (RDNA3 + CDNA).
#[cfg(target_os = "linux")]
pub fn amd_test_arch() -> Option<svod_dtype::AmdArch> {
    let nodes = svod_device::amd::topology::enumerate();
    nodes.into_iter().find_map(|n| svod_dtype::AmdArch::from_gfx_target_version(n.gfx_target_version))
}

#[cfg(not(target_os = "linux"))]
pub fn amd_test_arch() -> Option<svod_dtype::AmdArch> {
    None
}

/// If `SVOD_DEVICE` is set and svod-dtype parsed it with a placeholder
/// arch, re-parse it via the registry (which queries KFD topology) and
/// override the thread-local default device. Idempotent — safe to call from
/// multiple `PrepareConfig::from_env` sites.
fn normalize_default_device_from_env() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use svod_device::registry::DeviceSpecExt;
    static DONE: AtomicBool = AtomicBool::new(false);
    if DONE.swap(true, Ordering::AcqRel) {
        return;
    }
    let Ok(raw) = std::env::var("SVOD_DEVICE") else {
        return;
    };
    let Ok(normalized) = <DeviceSpec as DeviceSpecExt>::parse(raw.trim()) else {
        return;
    };
    svod_dtype::default_device::set_default_device(normalized);
}

impl PrepareConfig {
    /// Resolve a `DeviceSpec` into a `Device` using this config's resolver.
    pub(crate) fn resolve_device(&self, spec: &DeviceSpec, registry: &DeviceRegistry) -> crate::Result<Arc<Device>> {
        self.resolver.resolve(spec, registry)
    }
}

impl From<OptimizerConfig> for PrepareConfig {
    fn from(optimizer: OptimizerConfig) -> Self {
        Self { optimizer, resolver: Arc::new(EnvResolver), disable_schedule_cache: false }
    }
}

/// Generate one test per codegen backend (Clang, LLVM) from a single test body.
///
/// Supports three forms:
///
/// **Simple test** (config only, no extra params):
/// ```ignore
/// codegen_tests! {
///     fn test_add(config) {
///         let mut a = Tensor::from_slice([1.0f32, 2.0, 3.0]);
///         a.realize_with(&config).unwrap();
///         let result: Vec<f32> = a.as_vec().unwrap();
///     }
/// }
/// // Generates: test_add::clang, test_add::llvm
/// ```
///
/// **Parameterized test** (extra typed params, use with `#[test_case]`):
/// ```ignore
/// codegen_tests! {
///     #[test_case(128, 0.5; "128x128")]
///     fn test_matmul(config, size: usize, tol: f32) {
///         let mut result = run_matmul(size);
///         result.realize_with(&config).unwrap();
///         assert_close(&result, tol);
///     }
/// }
/// // Generates: test_matmul::clang::test_matmul, test_matmul::llvm::test_matmul
/// ```
///
/// **Proptest** (property-based, params use `in` syntax):
/// ```ignore
/// codegen_tests! {
///     #[proptest_config(ProptestConfig::with_cases(50))]
///     fn test_sort_random(config, data in proptest::collection::vec(-100.0f32..100.0, 1..=16)) {
///         let mut t = Tensor::from_slice(&data);
///         let (sorted, _) = t.sort(-1, false).unwrap();
///         // ...
///     }
/// }
/// // Generates: test_sort_random::clang, test_sort_random::llvm
/// ```
#[macro_export]
macro_rules! codegen_tests {
    // Base case
    () => {};

    // Simple test (config only, no extra params)
    ($(#[$meta:meta])* fn $name:ident($config:ident) $body:block $($rest:tt)*) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            $(#[$meta])*
            fn clang() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                $body
            }

            #[test]
            $(#[$meta])*
            fn llvm() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                $body
            }

            /// AMD variant — runs only when a supported AMD GPU is detected
            /// on this host (RDNA3 + CDNA). On unsupported hardware or hosts
            /// without `/dev/kfd` this test exits with a skip message rather
            /// than a failure, so the unified test suite still runs on any
            /// CI runner.
            #[test]
            $(#[$meta])*
            fn amd() {
                ::svod_schedule::testing::setup_test_tracing();
                let $config = match $crate::PrepareConfig::for_amd_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("amd codegen_tests variant: skipped (no supported AMD GPU)");
                        return;
                    }
                };
                $body
            }
        }
        $crate::codegen_tests!($($rest)*);
    };

    // Proptest with config: #[proptest_config(...)] fn name(config, param in strategy) { body }
    (#[proptest_config($($pc:tt)*)] $(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident in $strategy:expr),+ $(,)?) $body:block $($rest:tt)*) => {
        $crate::codegen_tests!(@proptest $name, $config, [$($param in $strategy),+], $body,
            ::proptest::test_runner::TestRunner::new($($pc)*), [$(#[$meta])*]);
        $crate::codegen_tests!($($rest)*);
    };

    // Proptest with default config: fn name(config, param in strategy) { body }
    ($(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident in $strategy:expr),+ $(,)?) $body:block $($rest:tt)*) => {
        $crate::codegen_tests!(@proptest $name, $config, [$($param in $strategy),+], $body,
            ::proptest::test_runner::TestRunner::default(), [$(#[$meta])*]);
        $crate::codegen_tests!($($rest)*);
    };

    // Internal: proptest code generation (uses TestRunner API directly)
    (@proptest $name:ident, $config:ident, [$($param:ident in $strategy:expr),+], $body:block, $runner:expr, [$(#[$meta:meta])*]) => {
        mod $name {
            #[allow(unused_imports)]
            use super::*;

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn clang() {
                ::svod_schedule::testing::setup_test_tracing();
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn llvm() {
                ::svod_schedule::testing::setup_test_tracing();
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                    $body
                    Ok(())
                }).unwrap();
            }

            #[test]
            #[allow(unused_parens)]
            $(#[$meta])*
            fn amd() {
                ::svod_schedule::testing::setup_test_tracing();
                let amd_cfg = match $crate::PrepareConfig::for_amd_if_available() {
                    Some(cfg) => cfg,
                    None => {
                        eprintln!("amd codegen_tests variant: skipped (no supported AMD GPU)");
                        return;
                    }
                };
                let mut runner = $runner;
                runner.run(&($($strategy),+), |($($param),+)| {
                    let $config = amd_cfg.clone();
                    $body
                    Ok(())
                }).unwrap();
            }
        }
    };

    // Parameterized test (extra typed params — test_case attrs expected, no #[test])
    ($(#[$meta:meta])* fn $name:ident($config:ident, $($param:ident: $ty:ty),+ $(,)?) $body:block $($rest:tt)*) => {
        mod $name {
            mod clang {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Clang);
                    $body
                }
            }
            mod llvm {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = $crate::PrepareConfig::for_cpu_backend($crate::CpuBackend::Llvm);
                    $body
                }
            }
            mod amd {
                #[allow(unused_imports)]
                use super::super::*;
                use ::test_case::test_case;

                $(#[$meta])*
                fn $name($($param: $ty),+) {
                    ::svod_schedule::testing::setup_test_tracing();
                    let $config = match $crate::PrepareConfig::for_amd_if_available() {
                        Some(cfg) => cfg,
                        None => {
                            eprintln!("amd codegen_tests variant: skipped (no supported AMD GPU)");
                            return;
                        }
                    };
                    $body
                }
            }
        }
        $crate::codegen_tests!($($rest)*);
    };
}
