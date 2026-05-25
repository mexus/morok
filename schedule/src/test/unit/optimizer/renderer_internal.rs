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
fn test_tensor_core_cuda() {
    let tc = CUDA_81616.build(DType::Float16, DType::Float32);
    assert_eq!(tc.dims, (8, 16, 16));
    assert_eq!(tc.threads, 32);
    assert_eq!(tc.dtype_in, DType::Float16);
    assert_eq!(tc.dtype_out, DType::Float32);
    assert!(!tc.opts.is_empty());
}
