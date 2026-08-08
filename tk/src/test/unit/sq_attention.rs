use std::sync::Arc;

use svod_dtype::{AmdArch, DType, DeviceSpec};
use svod_ir::{Op, UOp};
use svod_tensor::Tensor;

use crate::kernels::sq_attention::{
    SQ_ATTENTION_SUPPORTED_ARCHS, SqAttentionOpts, build_single_query_attention, build_single_query_attention_merge,
    build_single_query_attention_partial,
};
use crate::{ArchCaps, Kernel};

fn buffers(b: usize, n: usize, h: usize, d: usize, masked: bool) -> Vec<Arc<UOp>> {
    let mut bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
    ];
    if masked {
        bufs.push(UOp::new_buffer(DeviceSpec::Cpu, b, DType::Int32));
    }
    bufs
}

fn sink(caps: ArchCaps, masked: bool) -> Arc<UOp> {
    let (b, n, h, d) = (2, 5, 3, 64);
    let ker =
        Kernel::new("sq_attention", [h as i64, b as i64, 1], caps.wave_size as i64, buffers(b, n, h, d, masked), caps);
    build_single_query_attention(&ker, b, n, h, d, masked, masked);
    ker.finish(1)
}

fn split_sinks(caps: ArchCaps, splits: usize) -> (Arc<UOp>, Arc<UOp>) {
    let (b, n, h, d) = (2, 20, 3, 64);
    let partial_buffers = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * 2, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::Float32),
    ];
    let partial = Kernel::new(
        "sq_attention_partial",
        [h as i64, b as i64, splits as i64],
        caps.wave_size as i64,
        partial_buffers,
        caps,
    );
    build_single_query_attention_partial(&partial, b, n, h, d, splits);
    let partial = partial.finish(2);

    let merge_buffers = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * d, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, b * splits * h * 2, DType::Float32),
    ];
    let merge = Kernel::new("sq_attention_merge", [h as i64, b as i64, 1], caps.wave_size as i64, merge_buffers, caps);
    build_single_query_attention_merge(&merge, b, h, d, splits);
    (partial, merge.finish(1))
}

#[test]
fn sq_attention_graph_shape_both_arches() {
    for caps in [ArchCaps::GFX942, ArchCaps::for_arch(AmdArch::Gfx1151)] {
        for masked in [false, true] {
            let topo = sink(caps, masked).toposort();
            let shuffles = topo.iter().filter(|u| matches!(u.op(), Op::Custom { .. })).count();
            assert_eq!(shuffles, caps.wave_size.ilog2() as usize, "{:?}: one XOR reduction", caps.arch);
            assert!(
                topo.iter().any(|u| matches!(u.op(), Op::Unary(svod_ir::UnaryOp::Exp2, ..))),
                "{:?}: exp2",
                caps.arch
            );
            assert!(topo.iter().any(|u| matches!(u.op(), Op::Range { .. })), "{:?}: streamed N loop", caps.arch);
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "{:?}: no LDS", caps.arch);
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })), "{:?}: no MFMA/WMMA", caps.arch);
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })), "{:?}: no barrier", caps.arch);
        }
        let (partial, merge) = split_sinks(caps, 4);
        for (name, graph) in [("partial", partial), ("merge", merge)] {
            let topo = graph.toposort();
            assert!(topo.iter().any(|u| matches!(u.op(), Op::Range { .. })), "{:?}: split {name} loop", caps.arch);
            assert!(!topo.iter().any(|u| matches!(u.op(), Op::DefineLocal(_))), "{:?}: split {name} no LDS", caps.arch);
            assert!(
                !topo.iter().any(|u| matches!(u.op(), Op::Wmma { .. })),
                "{:?}: split {name} no MFMA/WMMA",
                caps.arch
            );
            assert!(
                !topo.iter().any(|u| matches!(u.op(), Op::Barrier { .. })),
                "{:?}: split {name} no barrier",
                caps.arch
            );
        }
    }
}

#[test]
fn sq_attention_renders_both_arches() {
    for arch in [AmdArch::Gfx942, AmdArch::Gfx1151] {
        let caps = ArchCaps::for_arch(arch);
        let (partial, merge) = split_sinks(caps, 4);
        for (name, graph, shuffle) in [
            ("sq_attention", sink(caps, true), true),
            ("sq_attention_partial", partial, true),
            ("sq_attention_merge", merge, false),
        ] {
            let lowered =
                svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), graph, &mut ());
            let program = svod_codegen::program_pipeline::program_from_sink(lowered, DeviceSpec::Cpu);
            let linearized = svod_codegen::program_pipeline::do_linearize(&program).expect("linearize");
            let linear =
                linearized.toposort().into_iter().find(|u| matches!(u.op(), Op::Linear { .. })).expect("LINEAR");
            let renderer = svod_codegen::llvm::LlvmTextRenderer::amd(arch);
            let code = svod_codegen::traits::Renderer::render(&renderer, &linear, Some(name)).expect("render").code;
            assert_eq!(code.contains("llvm.amdgcn.ds.bpermute"), shuffle, "{arch:?}: {name} shuffle rendering");
            assert!(!code.contains("@local"), "{arch:?}: {name} no LDS allocation");
        }
    }
}

fn supported_device() -> bool {
    crate::target::check_target(&Tensor::empty(&[1], DType::Float32).device(), SQ_ATTENTION_SUPPORTED_ARCHS).is_ok()
}

fn cpu_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    dims: (usize, usize, usize, usize),
    lens: Option<&[i32]>,
) -> Vec<f32> {
    let (b, n, h, d) = dims;
    let mut out = vec![0.0; b * h * d];
    for bi in 0..b {
        for hi in 0..h {
            let valid = |ni: usize| lens.is_none_or(|ls| ni < ls[bi] as usize || ni + 1 == n);
            let mut scores = vec![f32::NEG_INFINITY; n];
            for (ni, score) in scores.iter_mut().enumerate().filter(|(ni, _)| valid(*ni)) {
                let mut dot = 0.0;
                for di in 0..d {
                    dot += q[(bi * h + hi) * d + di] * k[((bi * n + ni) * h + hi) * d + di];
                }
                *score = dot / (d as f32).sqrt();
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|x| (*x - max).exp()).collect();
            let norm: f32 = weights.iter().sum();
            for di in 0..d {
                for ni in 0..n {
                    out[(bi * h + hi) * d + di] += weights[ni] * v[((bi * n + ni) * h + hi) * d + di] / norm;
                }
            }
        }
    }
    out
}

#[test]
#[ignore]
fn sq_attention_numerical_amd() {
    if !supported_device() {
        eprintln!("skip sq_attention_numerical_amd: unsupported device/toolchain");
        return;
    }
    let (b, n, h, d) = (2, 20, 3, 64);
    let mut q = Tensor::randn(&[b, 1, h, d]).expect("q");
    let mut k = Tensor::randn(&[b, n, h, d]).expect("k");
    let mut v = Tensor::randn(&[b, n, h, d]).expect("v");
    q.realize().expect("realize q");
    k.realize().expect("realize k");
    v.realize().expect("realize v");
    let qv = q.as_vec::<f32>().expect("q vec");
    let kv = k.as_vec::<f32>().expect("k vec");
    let vv = v.as_vec::<f32>().expect("v vec");

    for (lens, splits) in [(None, vec![1, 2, 4]), (Some(vec![7i32, 13]), vec![1])] {
        for split in splits {
            let mut lens_t = lens.as_ref().map(|x| Tensor::from_slice(x.as_slice()));
            if let Some(t) = &mut lens_t {
                t.realize().expect("realize lens");
            }
            let opts = SqAttentionOpts { key_lens: lens_t.as_ref(), include_last: lens.is_some(), split };
            let mut got = crate::single_query_attention(&q, &k, &v, opts).expect("sq attention").expect("supported");
            got.realize().expect("realize output");
            let got = got.as_vec::<f32>().expect("output vec");
            let expected = cpu_reference(&qv, &kv, &vv, (b, n, h, d), lens.as_deref());
            let max_abs = got.iter().zip(&expected).map(|(a, e)| (a - e).abs()).fold(0.0f32, f32::max);
            assert!(max_abs < 2e-4, "split {split} max abs error {max_abs}");
        }
    }
}
