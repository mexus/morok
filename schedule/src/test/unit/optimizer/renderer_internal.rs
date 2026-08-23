use super::*;
use svod_ir::RendererDevice;

#[test]
fn test_renderer_cpu() {
    let r = Renderer::cpu();
    assert_eq!(r.device, RendererDevice::Cpu);
    assert!(!r.has_local);
    assert!(r.has_threads);
    assert_eq!(r.tensor_cores.len(), 0);
}

#[test]
fn test_renderer_cuda() {
    let r = Renderer::cuda();
    assert_eq!(r.device, RendererDevice::CudaSm80); // Default is SM80/Ampere
    assert!(r.has_local);
    assert!(r.has_shared);
    assert!(!r.has_threads);
    assert!(r.shared_max > 0);
    assert!(!r.tensor_cores.is_empty());
}

#[test]
fn test_for_amd_arch_maps_each_family() {
    use svod_dtype::AmdArch;
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx942).device, RendererDevice::AmdCdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx950).device, RendererDevice::AmdCdna4);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1100).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1151).device, RendererDevice::AmdRdna3);
    assert_eq!(Renderer::for_amd_arch(AmdArch::Gfx1201).device, RendererDevice::AmdRdna4);
}

#[test]
fn test_renderer_fingerprint_tracks_exact_target_and_capabilities() {
    use svod_dtype::AmdArch;

    let gfx1100 = Renderer::for_amd_arch(AmdArch::Gfx1100);
    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_ne!(gfx1100.cache_fingerprint(), gfx1151.cache_fingerprint());

    let mut constrained = gfx1151.clone();
    constrained.upcast_max -= 1;
    assert_ne!(gfx1151.cache_fingerprint(), constrained.cache_fingerprint());

    constrained = gfx1151.clone();
    constrained.tensor_cores.clear();
    assert_ne!(gfx1151.cache_fingerprint(), constrained.cache_fingerprint());

    let all_ops = gfx1151.clone().with_rewrite_capabilities(svod_ir::RendererOps::all(), None, None);
    let mut fewer_ops = svod_ir::RendererOps::all();
    fewer_ops.binary.remove(&svod_ir::BinaryOp::Threefry);
    let fewer_ops = gfx1151.with_rewrite_capabilities(fewer_ops, None, None);
    assert_ne!(all_ops.cache_fingerprint(), fewer_ops.cache_fingerprint());
}

#[test]
fn test_amd_fp8_dtype_capabilities_are_arch_specific() {
    use svod_dtype::{AmdArch, ScalarDType};

    for arch in [AmdArch::Gfx942, AmdArch::Gfx950] {
        let renderer = Renderer::for_amd_arch(arch);
        assert!(renderer.supports_storage_dtype(ScalarDType::FP8E4M3), "{arch} must keep OCP FP8 storage");
        assert!(renderer.supports_storage_dtype(ScalarDType::FP8E5M2), "{arch} must keep OCP BF8 storage");
        assert!(renderer.supports_conversion_dtype(ScalarDType::FP8E4M3), "{arch} must keep OCP FP8 conversion");
        assert!(renderer.supports_conversion_dtype(ScalarDType::FP8E5M2), "{arch} must keep OCP BF8 conversion");
        assert!(!renderer.supports_alu_dtype(ScalarDType::FP8E4M3), "{arch} must widen ordinary FP8 ALU");
        assert!(!renderer.supports_alu_dtype(ScalarDType::FP8E5M2), "{arch} must widen ordinary BF8 ALU");
        assert!(renderer.supports_matrix_dtype(ScalarDType::FP8E4M3), "{arch} must keep FP8 matrix operands");
        assert!(renderer.supports_matrix_dtype(ScalarDType::FP8E5M2), "{arch} must keep BF8 matrix operands");
        assert!(!renderer.supports_dtype(ScalarDType::FP8E4M3FNUZ), "{arch} must decompose FNUZ FP8");
        assert!(!renderer.supports_dtype(ScalarDType::FP8E5M2FNUZ), "{arch} must decompose FNUZ BF8");
    }

    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2, ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
        assert!(!gfx1151.supports_dtype(dtype), "gfx1151 must decompose {dtype:?}");
    }

    let gfx1201 = Renderer::for_amd_arch(AmdArch::Gfx1201);
    for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2, ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
        assert!(!gfx1201.supports_dtype(dtype), "gfx1201 must decompose {dtype:?} to f16");
    }
}

#[test]
fn test_amd_tensor_core_tables_match_architecture() {
    use svod_dtype::AmdArch;

    let gfx942 = Renderer::for_amd_arch(AmdArch::Gfx942);
    assert_eq!(gfx942.tensor_cores.len(), 4);
    assert!(gfx942.tensor_cores.iter().any(|tc| tc.dims == (16, 16, 32) && tc.dtype_in == DType::FP8E4M3));

    let gfx950 = Renderer::for_amd_arch(AmdArch::Gfx950);
    assert_eq!(gfx950.tensor_cores.len(), 8);
    assert!(gfx950.tensor_cores.iter().any(|tc| tc.dims == (16, 16, 128) && tc.dtype_in == DType::FP8E4M3));

    let gfx1151 = Renderer::for_amd_arch(AmdArch::Gfx1151);
    assert_eq!(gfx1151.tensor_cores.len(), 4);
    assert!(!gfx1151.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(gfx1151.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Int8 && tc.dtype_out == DType::Int32));

    let gfx1201 = Renderer::for_amd_arch(AmdArch::Gfx1201);
    assert_eq!(gfx1201.tensor_cores.len(), 4);
    assert!(!gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in.scalar_dtype().is_fp8()));
    assert!(gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in == DType::Float16 && tc.dtype_out == DType::Float32));
    assert!(gfx1201.tensor_cores.iter().any(|tc| tc.dtype_in == DType::BFloat16 && tc.dtype_out == DType::BFloat16));
}

#[test]
fn test_local_max_axes_match_renderer_capabilities() {
    assert_eq!(Renderer::cuda().local_max_axes(), Some([1024, 1024, 64]));
    assert_eq!(Renderer::webgpu().local_max_axes(), Some([256, 256, 64]));
    assert_eq!(Renderer::amd_cdna3().local_max_axes(), None);
    assert_eq!(Renderer::cpu().local_max_axes(), None);
    assert_eq!(Renderer::metal().local_max_axes(), None);
}

#[test]
fn test_tensor_core_cuda() {
    let tc = CUDA_81616.build(DType::Float16, DType::Float32);
    assert_eq!(tc.dims, (8, 16, 16));
    assert_eq!(tc.threads, 32);
    assert_eq!(tc.dtype_in, DType::Float16);
    assert_eq!(tc.dtype_out, DType::Float32);
    assert!(!tc.opts.is_empty());
}
