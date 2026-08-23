use svod_device::device::{CompiledSpec, Compiler, ProgramSpec, Renderer};
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{BinaryOp, InsArg, Op, ParamArg, RewriteResult, TypedPatternMatcher, UOp};

fn committed_sink(sources: Vec<std::sync::Arc<UOp>>) -> std::sync::Arc<UOp> {
    let sink = UOp::sink(sources);
    let sink = svod_schedule::graph_rewrite(&svod_schedule::symbolic::pm_lower_index_dtype(), sink, &mut ());
    assert!(sink.toposort().iter().all(|u| !u.dtype().is_weak()), "end-to-end fixture must commit weak dtypes");
    sink
}

fn program(
    sink: std::sync::Arc<UOp>,
    target: DeviceSpec,
    linear: Option<std::sync::Arc<UOp>>,
    source: Option<std::sync::Arc<UOp>>,
    binary: Option<std::sync::Arc<UOp>>,
) -> std::sync::Arc<UOp> {
    let info = svod_ir::ProgramInfo::from_sink(&sink, target);
    UOp::program(sink, info, linear, source, binary)
}

fn scalar_param(name: &str, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), 0, 16, DType::Int32);
    let Op::Param { shape, arg } = var.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param { shape: shape.clone(), arg }, DType::Int32)
}

fn param_slot(root: &std::sync::Arc<UOp>, name: &str) -> usize {
    root.toposort()
        .into_iter()
        .find_map(|u| match u.op() {
            Op::Param { arg, .. } if arg.name.as_deref() == Some(name) => Some(arg.slot),
            _ => None,
        })
        .unwrap_or_else(|| panic!("missing PARAM {name}"))
}

struct MockRenderer {
    device: DeviceSpec,
}

struct CAbiRenderer;

struct LlvmAbiRenderer;

impl Renderer for CAbiRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let rendered = crate::c::render(ast, name)
            .map_err(|error| svod_device::Error::Runtime { message: format!("C rendering failed: {error}") })?;
        let mut spec = ProgramSpec::new(rendered.name, rendered.code, DeviceSpec::Cpu, ast.clone());
        spec.var_names = rendered.var_names;
        spec.buf_count = rendered.buffer_args.len();
        spec.abi = rendered.abi;
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        static DEVICE: DeviceSpec = DeviceSpec::Cpu;
        &DEVICE
    }
}

impl Renderer for LlvmAbiRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let rendered = crate::llvm::text::render(ast, name)
            .map_err(|error| svod_device::Error::Runtime { message: format!("LLVM rendering failed: {error}") })?;
        let mut spec = ProgramSpec::new(rendered.name, rendered.code, DeviceSpec::Cpu, ast.clone());
        spec.var_names = rendered.var_names;
        spec.buf_count = rendered.buffer_args.len();
        spec.abi = rendered.abi;
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        static DEVICE: DeviceSpec = DeviceSpec::Cpu;
        &DEVICE
    }
}

#[derive(Clone)]
struct MockIsaRenderer {
    device: DeviceSpec,
    events: std::sync::Arc<std::sync::Mutex<Vec<String>>>,
}

impl Renderer for MockIsaRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let Op::Linear { ops } = ast.op() else { panic!("ISA renderer must receive LINEAR") };
        let source = ops
            .iter()
            .filter_map(|u| match u.op() {
                Op::Ins { arg, .. } => Some(arg.opcode.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ProgramSpec::new(name.unwrap_or("kernel").to_string(), source, self.device.clone(), ast.clone()))
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }

    fn pre_isel_matcher(&self) -> Option<TypedPatternMatcher<svod_device::isa::PreIselContext>> {
        let events = self.events.clone();
        let mut matcher: TypedPatternMatcher<svod_device::isa::PreIselContext> = TypedPatternMatcher::new();
        matcher.add(&[svod_ir::op::pattern_derived::OpKey::Const], move |u, ctx| {
            let Op::Const(value) = u.op() else { return RewriteResult::NoMatch };
            let temp = ctx.next_temp();
            events.lock().unwrap().push(format!("pre:{:?}:{temp}", value.0));
            RewriteResult::Rewritten(UOp::ins([], u.dtype(), InsArg::new(format!("imm.{:?}", value.0))))
        });
        Some(matcher)
    }

    fn isel_matcher(&self) -> Option<TypedPatternMatcher<svod_device::isa::IselContext>> {
        let events = self.events.clone();
        let mut matcher: TypedPatternMatcher<svod_device::isa::IselContext> = TypedPatternMatcher::new();
        matcher.add(&[svod_ir::op::pattern_derived::OpKey::Binary(BinaryOp::Add)], move |u, ctx| {
            let Op::Binary(BinaryOp::Add, lhs, rhs) = u.op() else { return RewriteResult::NoMatch };
            assert!(matches!(lhs.op(), Op::Ins { .. }) && matches!(rhs.op(), Op::Ins { .. }));
            assert_eq!(ctx.uses(lhs).len(), 1);
            assert_eq!(ctx.uses(rhs).len(), 1);
            let vreg = ctx.next_vreg();
            events.lock().unwrap().push(format!("isel:add:v{vreg}"));
            RewriteResult::Rewritten(UOp::ins(
                [lhs.clone(), rhs.clone()],
                u.dtype(),
                InsArg::with_attributes("mock.add", vec![("dst".into(), format!("v{vreg}"))]),
            ))
        });
        Some(matcher)
    }
}

#[test]
fn isa_selection_is_bottom_up_after_program_info_and_renders_program_source() {
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let renderer = MockIsaRenderer { device: DeviceSpec::Cpu, events: events.clone() };
    let sink = committed_sink(vec![UOp::const_(DType::Int32, 1.into()).add(&UOp::const_(DType::Int32, 2.into()))]);
    let expected_info =
        svod_ir::ProgramInfo::from_sink(&svod_schedule::add_control_flow(sink.clone()), DeviceSpec::Cpu);

    let program = crate::program_pipeline::program_from_sink_with_renderer(sink, &renderer).expect("ISA PROGRAM");
    let Op::Program { sink: selected, info, .. } = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(info, &expected_info, "ProgramInfo must be discovered before instruction selection");
    assert!(matches!(selected.op(), Op::Sink { .. }));
    assert!(selected.toposort().iter().any(|u| matches!(u.op(), Op::Ins { arg, .. } if arg.opcode == "mock.add")));
    svod_schedule::spec::type_verify(selected, &svod_schedule::spec::spec_program()).expect("INS is target-spec legal");
    assert_eq!(
        *events.lock().unwrap(),
        vec!["pre:Int(1):-1", "pre:Int(2):-2", "isel:add:v0"],
        "both ISA passes must walk children before parents",
    );

    let (rendered, spec) =
        crate::program_pipeline::do_render(&program, &renderer).expect("render selected instructions");
    assert_eq!(spec.src, "imm.Int(1)\nimm.Int(2)\nmock.add");
    let Op::Program { source: Some(source), .. } = rendered.op() else { panic!("expected PROGRAM SOURCE") };
    assert!(matches!(source.op(), Op::Source { code, .. } if code == &spec.src));
}

#[test]
fn assigned_storage_zero_reserves_scalar_slot_one() {
    let global = UOp::param(0, 16, DType::Float32, None);
    let scalar = UOp::variable("n".into(), 0, 16, DType::Int32);
    let sink = committed_sink(vec![global, scalar]);

    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    let Op::Program { sink, info, .. } = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(param_slot(sink, "n"), 1);
    assert_eq!(info.globals, vec![0]);
    assert_eq!(info.vars.iter().map(|var| param_slot(var, "n")).collect::<Vec<_>>(), vec![1]);
}

#[test]
fn dense_assigned_params_precede_multiple_unassigned_scalars_in_walk_order() {
    let globals = [UOp::param(0, 16, DType::Float32, None), UOp::param(1, 16, DType::Float32, None)];
    let first = UOp::variable("z_first".into(), 0, 16, DType::Int32);
    let second = UOp::variable("a_second".into(), 0, 16, DType::Int32);
    let sink = committed_sink(vec![globals[0].clone(), globals[1].clone(), first.add(&second)]);

    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    let Op::Program { sink, info, .. } = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(param_slot(sink, "z_first"), 2);
    assert_eq!(param_slot(sink, "a_second"), 3);
    assert_eq!(info.globals, vec![0, 1]);
    assert_eq!(
        info.vars
            .iter()
            .map(|var| match var.op() {
                Op::Param { arg, .. } => arg.slot,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
}

#[test]
fn sparse_authored_slots_are_preserved_and_skipped() {
    let zero = UOp::param(0, 16, DType::Float32, None);
    let five = UOp::param(5, 16, DType::Float32, None);
    let first = UOp::variable("first".into(), 0, 16, DType::Int32);
    let second = UOp::variable("second".into(), 0, 16, DType::Int32);
    let sink = committed_sink(vec![zero, five, first.add(&second)]);

    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("sparse slots");
    let Op::Program { sink, info, .. } = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(param_slot(sink, "first"), 2);
    assert_eq!(param_slot(sink, "second"), 3);
    assert_eq!(info.globals, vec![0, 5]);

    let linear = UOp::linear(svod_schedule::linearize(sink.clone()).into());
    let c = crate::c::render(&linear, Some("sparse_abi")).expect("sparse C ABI");
    assert!(
        c.code.contains(
            "void sparse_abi(float* restrict data0, const int data2, const int data3, float* restrict data5)"
        ),
        "{}",
        c.code
    );
    let llvm = crate::llvm::text::render(&linear, Some("sparse_abi")).expect("sparse LLVM ABI");
    assert!(
        llvm.code.contains(
            "define void @sparse_abi(ptr noalias align 32 %data0, i32 %data2, i32 %data3, ptr noalias align 32 %data5)"
        ),
        "{}",
        llvm.code
    );
}

fn storage_param(slot: usize, addrspace: svod_ir::AddrSpace) -> std::sync::Arc<UOp> {
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![16usize.into()]);
    UOp::new(Op::Param { shape, arg: ParamArg::buffer(slot, DType::Float32, addrspace, None) }, DType::Float32)
}

#[test]
fn param_abi_namespace_excludes_local_and_register_buffer_scratch() {
    let global = storage_param(0, svod_ir::AddrSpace::Global);
    let local_param = storage_param(1, svod_ir::AddrSpace::Local);
    let reg_param = storage_param(2, svod_ir::AddrSpace::Reg);
    let local_scratch = UOp::buffer(0, 16, DType::Float32, svod_ir::AddrSpace::Local, None);
    let reg_scratch = UOp::buffer(1, 16, DType::Float32, svod_ir::AddrSpace::Reg, None);
    let scalar = UOp::variable("n".into(), 0, 16, DType::Int32);
    let sink = committed_sink(vec![global, local_param, reg_param, local_scratch, reg_scratch, scalar]);

    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    let Op::Program { sink, info, .. } = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(param_slot(sink, "n"), 3);
    assert_eq!(info.globals, vec![0, 1, 2]);
    assert_eq!(info.vars.len(), 1);
}

#[test]
fn duplicate_authored_slots_are_a_typed_program_error() {
    let global = UOp::param(0, 16, DType::Float32, None);
    let scalar = scalar_param("n", 0);
    let err = crate::program_pipeline::program_from_sink(committed_sink(vec![global, scalar]), DeviceSpec::Cpu)
        .expect_err("duplicate authored ABI slots must fail");
    assert!(matches!(err, svod_device::Error::DuplicateProgramParamSlot { slot: 0, .. }), "{err:?}");
}

#[test]
fn reused_param_is_one_abi_argument() {
    let global = UOp::param(0, 16, DType::Float32, None);
    let program =
        crate::program_pipeline::program_from_sink(committed_sink(vec![global.clone(), global]), DeviceSpec::Cpu)
            .expect("the same PARAM reused is not a duplicate definition");
    let Op::Program { info, .. } = program.op() else { panic!("PROGRAM") };
    assert_eq!(info.globals, vec![0]);
}

#[test]
fn malformed_prebuilt_program_rejects_duplicate_and_unassigned_slots() {
    for sink in [
        committed_sink(vec![UOp::param(0, 1, DType::Float32, None), scalar_param("n", 0)]),
        committed_sink(vec![UOp::variable("n".into(), 0, 16, DType::Int32)]),
    ] {
        let prebuilt = program(sink, DeviceSpec::Cpu, None, None, None);
        let err = crate::program_pipeline::do_linearize(&prebuilt).expect_err("malformed prebuilt PROGRAM must fail");
        assert!(
            matches!(
                err,
                svod_device::Error::DuplicateProgramParamSlot { .. }
                    | svod_device::Error::UnassignedProgramParam { .. }
            ),
            "{err:?}"
        );
    }
}

#[test]
fn unnamed_scalar_and_non_param_program_info_var_are_typed_errors() {
    let mut arg = ParamArg::variable("n".into(), DType::Int32, 0, 16);
    arg.name = None;
    let unnamed = UOp::new(Op::Param { shape: UOp::index_const(1), arg }, DType::Int32);
    let err = crate::program_pipeline::program_from_sink(committed_sink(vec![unnamed]), DeviceSpec::Cpu)
        .expect_err("unnamed scalar must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");

    let sink = committed_sink(vec![UOp::const_(DType::Int32, 1.into())]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    info.vars.push(UOp::const_(DType::Int32, 2.into()));
    let malformed = UOp::program(sink, info, None, None, None);
    let err = crate::program_pipeline::do_linearize(&malformed).expect_err("non-PARAM ProgramInfo var must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

#[test]
fn prebuilt_program_rejects_descriptor_equivalent_var_semantic_forgery() {
    let sink = committed_sink(vec![scalar_param("n", 0)]);
    let renderer = MockRenderer { device: DeviceSpec::Cpu };

    for mutation in ["bounds", "multiple_of", "axis"] {
        let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
        let Op::Param { shape, arg } = info.vars[0].op() else { unreachable!() };
        let mut forged_arg = arg.clone();
        match mutation {
            "bounds" => {
                forged_arg.vmin_vmax = Some((
                    svod_ir::ConstValueHash(svod_ir::ConstValue::Int(-1000)),
                    svod_ir::ConstValueHash(svod_ir::ConstValue::Int(1000)),
                ));
            }
            "multiple_of" => forged_arg.multiple_of = Some(8),
            "axis" => forged_arg.axis = Some(2),
            _ => unreachable!(),
        }
        info.vars[0] = UOp::new(Op::Param { shape: shape.clone(), arg: forged_arg }, DType::Int32);

        let prebuilt = UOp::program(sink.clone(), info.clone(), None, None, None);
        for err in [
            crate::program_pipeline::do_linearize(&prebuilt)
                .expect_err("do_linearize must reject forged ProgramInfo.vars"),
            crate::program_pipeline::get_program(
                &prebuilt,
                &renderer,
                &MockCompiler,
                crate::program_pipeline::ProgramTarget::Linear,
            )
            .expect_err("get_program must reject forged ProgramInfo.vars"),
        ] {
            match err {
                svod_device::Error::ProgramAbiMismatch { reason } => {
                    assert!(reason.contains("ProgramInfo.vars"), "{mutation}: {reason}");
                }
                other => panic!("{mutation}: expected ProgramAbiMismatch, got {other:?}"),
            }
        }

        let staged = UOp::program(
            sink.clone(),
            info,
            Some(UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into())),
            Some(UOp::source("// forged metadata".to_string())),
            None,
        );
        let err = ProgramSpec::from_uop(&staged).expect_err("ProgramSpec must reject forged ProgramInfo.vars");
        match err {
            svod_device::Error::ProgramAbiMismatch { reason } => {
                assert!(reason.contains("ProgramInfo.vars"), "{mutation}: {reason}");
            }
            other => panic!("{mutation}: expected ProgramAbiMismatch, got {other:?}"),
        }
    }
}

#[test]
fn prebuilt_program_accepts_semantically_identical_nonidentical_var() {
    let sink = committed_sink(vec![scalar_param("n", 0)]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let sink_var = info.vars[0].clone();
    let reconstructed = UOp::new(sink_var.op().clone(), sink_var.dtype()).with_metadata("detached variable");
    assert!(!std::sync::Arc::ptr_eq(&sink_var, &reconstructed));
    assert_eq!(sink_var.content_hash, reconstructed.content_hash);
    info.vars[0] = reconstructed;

    let prebuilt = UOp::program(sink, info, None, None, None);
    crate::program_pipeline::do_linearize(&prebuilt)
        .expect("validation must compare PARAM value semantics rather than allocation identity");
}

#[test]
fn opaque_function_formal_stays_unassigned_and_outside_outer_abi() {
    let formal = UOp::variable("formal".into(), 0, 16, DType::Int32);
    let outer = UOp::variable("outer".into(), 0, 16, DType::Int32);
    let call = UOp::sink(vec![formal]).call(smallvec::smallvec![outer], Default::default());
    let sink = crate::program_pipeline::number_params(svod_schedule::add_control_flow(committed_sink(vec![call])))
        .expect("opaque formal must not enter PROGRAM ABI");
    let info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    assert_eq!(param_slot(&sink, "outer"), 0);
    assert_eq!(info.vars.len(), 1);
    let formal_slot = sink
        .toposort_call_aware(true)
        .into_iter()
        .find_map(|node| match node.op() {
            Op::Param { arg, .. } if arg.name.as_deref() == Some("formal") => Some(arg.slot),
            _ => None,
        })
        .expect("formal PARAM remains in opaque body");
    assert_eq!(formal_slot, usize::MAX);
}

#[test]
fn opaque_formal_leaking_into_executable_graph_is_rejected() {
    let formal = UOp::variable("formal".into(), 0, 16, DType::Int32);
    let call = UOp::sink(vec![formal.clone()]).call(smallvec::smallvec![UOp::index_const(1)], Default::default());
    let sink = svod_schedule::add_control_flow(committed_sink(vec![call, formal]));
    let err = crate::program_pipeline::executable_params(&sink).expect_err("leaked opaque formal must fail");
    assert!(matches!(err, svod_device::Error::LeakedOpaqueProgramParam { .. }), "{err:?}");
}

#[test]
fn repeated_program_construction_has_identical_slots_and_identity() {
    let global = UOp::param(0, 16, DType::Float32, None);
    let first = UOp::variable("first".into(), 0, 16, DType::Int32);
    let second = UOp::variable("second".into(), 0, 16, DType::Int32);
    let sink = committed_sink(vec![global, first.add(&second)]);
    let a = crate::program_pipeline::program_from_sink(sink.clone(), DeviceSpec::Cpu).expect("first PROGRAM");
    let b = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("second PROGRAM");
    let Op::Program { info: ai, .. } = a.op() else { unreachable!() };
    let Op::Program { info: bi, .. } = b.op() else { unreachable!() };
    assert_eq!(ai, bi);
    assert_eq!(a.content_hash, b.content_hash);
}

#[test]
fn symbolic_program_render_compile_and_runtime_binding_share_canonical_abi() {
    let output = UOp::param(0, 1, DType::Float32, None);
    let n = UOp::variable("n".into(), 1, 16, DType::Int32);
    let index = UOp::index().buffer(output).indices(vec![UOp::index_const(0)]).call().expect("output index");
    let sink = committed_sink(vec![index.store(n.cast(DType::Float32))]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("PROGRAM");
    let Op::Program { info, .. } = program.op() else { unreachable!() };
    assert_eq!(info.globals, vec![0]);
    assert_eq!(param_slot(&program, "n"), 1);

    let (rendered, spec) = crate::program_pipeline::do_render(&program, &CAbiRenderer).expect("C render");
    assert_eq!(spec.globals, vec![0]);
    assert_eq!(spec.var_names, vec!["n"]);
    assert_eq!(spec.buf_count, 1);
    assert!(spec.src.contains("void test(float* restrict data0, const int data1)"), "{}", spec.src);

    let (compiled_program, compiled) = crate::program_pipeline::do_compile(&rendered, &MockCompiler).expect("compile");
    assert_eq!(compiled.buf_count, 1);
    assert_eq!(compiled.var_names, vec!["n"]);
    assert!(matches!(compiled_program.op(), Op::Program { binary: Some(_), .. }));

    let mut kernargs = [0u8; 12];
    let written = svod_device::hcq::ClikeKernargLayout::pack_program(
        info,
        &spec.abi,
        &mut kernargs,
        &[0x1122_3344_5566_7788],
        &[7],
    )
    .expect("runtime kernarg binding");
    assert_eq!(written, 12);
    assert_eq!(&kernargs[..8], &0x1122_3344_5566_7788u64.to_le_bytes());
    assert_eq!(&kernargs[8..], &7i32.to_le_bytes());
}

#[test]
fn number_params_uses_walk_after_control_flow_insertion() {
    let first = UOp::variable("first".into(), 0, 16, DType::Int32);
    let second = UOp::variable("second".into(), 0, 16, DType::Int32);
    let r0 = UOp::range(UOp::index_const(4), 0);
    let r1 = UOp::range(UOp::index_const(4), 1);
    let end0 = first.add(&r0.cast(DType::Int32)).end(smallvec::smallvec![r0]);
    let end1 = second.add(&r1.cast(DType::Int32)).end(smallvec::smallvec![r1]);
    let raw = committed_sink(vec![end0, end1]);
    let premature_slots: Vec<usize> = svod_ir::ProgramInfo::from_sink(&raw, DeviceSpec::Cpu)
        .vars
        .iter()
        .map(|u| match u.op() {
            Op::Param { arg, .. } => arg.slot,
            _ => unreachable!(),
        })
        .collect();
    let prepared = svod_schedule::add_control_flow(raw.clone());

    assert!(prepared.toposort().iter().any(|u| matches!(u.op(), Op::Range { deps, .. } if !deps.is_empty())));
    let expected_names: Vec<String> = prepared
        .toposort()
        .into_iter()
        .filter_map(|u| match u.op() {
            Op::Param { arg, .. } if arg.slot == usize::MAX && arg.addrspace.is_none() => arg.name.clone(),
            _ => None,
        })
        .collect();

    let program = crate::program_pipeline::program_from_sink(raw, DeviceSpec::Cpu).expect("final target graph");
    let Op::Program { sink, info, .. } = program.op() else { panic!("expected PROGRAM") };
    let actual_names: Vec<String> = info
        .vars
        .iter()
        .map(|u| match u.op() {
            Op::Param { arg, .. } => arg.name.clone().unwrap(),
            _ => unreachable!(),
        })
        .collect();
    assert_eq!(premature_slots, vec![usize::MAX, usize::MAX], "fixture must expose premature ProgramInfo slots");
    assert_eq!(actual_names, expected_names);
    assert_eq!(
        info.vars
            .iter()
            .map(|u| match u.op() {
                Op::Param { arg, .. } => arg.slot,
                _ => unreachable!(),
            })
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(
        sink.toposort()
            .iter()
            .all(|u| !matches!(u.op(), Op::Param { arg, .. } if arg.addrspace.is_none() && arg.slot == usize::MAX))
    );
}

struct LinearOnlyRenderer {
    device: DeviceSpec,
}

struct WrongAbiRenderer;

struct ReversedStorageRenderer;

impl Renderer for WrongAbiRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let mut spec = ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// wrong ABI".to_string(),
            DeviceSpec::Cpu,
            ast.clone(),
        );
        spec.buf_count = 1;
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        static DEVICE: DeviceSpec = DeviceSpec::Cpu;
        &DEVICE
    }
}

impl Renderer for ReversedStorageRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let rendered = crate::c::render(ast, name)
            .map_err(|error| svod_device::Error::Runtime { message: format!("C rendering failed: {error}") })?;
        let mut spec = ProgramSpec::new(rendered.name, rendered.code, DeviceSpec::Cpu, ast.clone());
        spec.abi = rendered.abi;
        spec.abi.reverse();
        spec.buf_count = rendered.buffer_args.len();
        spec.var_names = rendered.var_names;
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        static DEVICE: DeviceSpec = DeviceSpec::Cpu;
        &DEVICE
    }
}

impl Renderer for LinearOnlyRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        assert!(matches!(ast.op(), Op::Linear { .. }), "renderer should receive LINEAR stage");
        let spec = ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// linear source".to_string(),
            self.device.clone(),
            ast.clone(),
        );
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

impl Renderer for MockRenderer {
    fn supported_ops(&self) -> svod_ir::RendererOps {
        svod_ir::RendererOps::all()
    }

    fn render(&self, ast: &std::sync::Arc<UOp>, name: Option<&str>) -> svod_device::Result<ProgramSpec> {
        let spec = ProgramSpec::new(
            name.unwrap_or("kernel").to_string(),
            "// mock source".to_string(),
            self.device.clone(),
            ast.clone(),
        );
        Ok(spec)
    }

    fn device(&self) -> &DeviceSpec {
        &self.device
    }
}

struct MockCompiler;

impl Compiler for MockCompiler {
    fn compile(&self, spec: &ProgramSpec) -> svod_device::Result<CompiledSpec> {
        let mut compiled =
            CompiledSpec::from_bytes(spec.name.clone(), vec![1, 2, 3], spec.ast.clone(), spec.abi.clone())?;
        compiled.global_size = spec.global_size.clone();
        compiled.local_size = spec.local_size.clone();
        Ok(compiled)
    }

    fn cache_key(&self) -> &'static str {
        "mock"
    }
}

struct PanicCompiler;

impl Compiler for PanicCompiler {
    fn compile(&self, _spec: &ProgramSpec) -> svod_device::Result<CompiledSpec> {
        panic!("compiler should not be invoked when PROGRAM already has BINARY")
    }

    fn cache_key(&self) -> &'static str {
        "mock"
    }
}

struct OtherCompiler;

impl Compiler for OtherCompiler {
    fn compile(&self, _spec: &ProgramSpec) -> svod_device::Result<CompiledSpec> {
        panic!("compiler-key mismatch must be rejected before compilation")
    }

    fn cache_key(&self) -> &'static str {
        "other"
    }
}

#[test]
fn test_program_pipeline_sets_all_stages() {
    let sink = committed_sink(vec![UOp::native_const(1.0f32)]);
    let renderer = MockRenderer { device: DeviceSpec::Cpu };
    let compiler = MockCompiler;

    let program =
        crate::program_pipeline::program_from_sink(sink.clone(), DeviceSpec::Cpu).expect("final target graph");
    let (program, rendered_spec) = crate::program_pipeline::do_render(&program, &renderer).expect("render stage");
    let (program, compiled) = crate::program_pipeline::do_compile(&program, &compiler).expect("compile stage");
    let spec = ProgramSpec::from_uop(&program).expect("ProgramSpec::from_uop");

    match program.op() {
        Op::Program { linear, source, binary, .. } => {
            assert!(linear.is_some(), "LINEAR stage missing");
            assert!(source.is_some(), "SOURCE stage missing");
            assert!(binary.is_some(), "BINARY stage missing");
        }
        other => panic!("expected PROGRAM op, got {other:?}"),
    }
    let sources = program.op().children();
    assert_eq!(sources.len(), 4, "Tinygrad PROGRAM sources are SINK, LINEAR, SOURCE, BINARY");
    assert!(matches!(sources[0].op(), Op::Sink { .. }));
    assert!(matches!(sources[1].op(), Op::Linear { .. }));
    assert!(matches!(sources[2].op(), Op::Source { .. }));
    assert!(matches!(sources[3].op(), Op::ProgramBinary { .. }));

    assert_eq!(rendered_spec.name, "test");
    assert_eq!(spec.name, "test");
    assert_eq!(spec.src, "// mock source");
    assert_eq!(spec.ast.id, sink.id);
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
    assert_eq!(compiled.name, "test");
    assert!(spec.globals.is_empty());
    assert!(spec.outs.is_empty());
    assert!(spec.ins.is_empty());
}

#[test]
fn test_do_compile_requires_source_stage() {
    let sink = committed_sink(vec![UOp::native_const(1.0f32)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    let compiler = MockCompiler;

    let err = crate::program_pipeline::do_compile(&program, &compiler).expect_err("compile should fail without source");
    assert!(format!("{err}").contains("PROGRAM has no SOURCE stage"));
}

#[test]
fn test_do_compile_reuses_existing_binary_stage() {
    // Launch dims come from the SPECIAL UOps in the SINK (ProgramSpec::from_uop
    // ignores meta work sizes by design), so seed a `gidx0` with bound 4 to get
    // global_size == [4, 1, 1].
    let sink = committed_sink(vec![UOp::native_const(2.0f32), UOp::special(UOp::index_const(4), "gidx0".to_string())]);
    let renderer = MockRenderer { device: DeviceSpec::Cpu };
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).unwrap();
    let (program, _) = crate::program_pipeline::do_render(&program, &renderer).unwrap();
    let (program, _) = crate::program_pipeline::do_compile(&program, &MockCompiler).unwrap();

    let (compiled_program, compiled) =
        crate::program_pipeline::do_compile(&program, &PanicCompiler).expect("binary stage should be reused");

    assert!(std::sync::Arc::ptr_eq(&compiled_program, &program));
    assert_eq!(compiled.name, "test");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
    assert_eq!(compiled.src.as_deref(), Some("// mock source"));
    assert!(compiled.var_names.is_empty());
    assert_eq!(compiled.buf_count, 0);
    let vars = std::collections::HashMap::new();
    let launch = ProgramSpec::resolve_launch_dims(&compiled.global_size, compiled.local_size.as_ref(), &vars)
        .expect("resolve launch dims");
    assert_eq!(launch.global_size, [4, 1, 1]);

    let rebuilt =
        ProgramSpec::from_uop(&compiled_program).expect("ProgramSpec::from_uop should support binary+metadata");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.src, "// mock source");
}

#[test]
fn existing_binary_requires_exact_compiler_cache_key() {
    let program =
        crate::program_pipeline::program_from_sink(committed_sink(vec![UOp::native_const(2.0f32)]), DeviceSpec::Cpu)
            .unwrap();
    let (program, _) = crate::program_pipeline::do_render(&program, &MockRenderer { device: DeviceSpec::Cpu }).unwrap();
    let (program, _) = crate::program_pipeline::do_compile(&program, &MockCompiler).unwrap();
    let err = crate::program_pipeline::do_compile(&program, &OtherCompiler)
        .expect_err("binary from another compiler key must not be reused");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

#[test]
fn semantic_stage_identity_defeats_preinterned_children_and_parent_programs() {
    let initial =
        crate::program_pipeline::program_from_sink(committed_sink(vec![UOp::native_const(11.0f32)]), DeviceSpec::Cpu)
            .unwrap();
    let linearized = crate::program_pipeline::do_linearize(&initial).unwrap();
    let Op::Program { sink, info, linear: Some(linear), .. } = linearized.op() else { unreachable!() };

    let raw_source = UOp::source("// mock source".into());
    let raw_source_parent =
        UOp::program(sink.clone(), info.clone(), Some(linear.clone()), Some(raw_source.clone()), None);
    let abi = ProgramSpec::validate_program_param_abi(sink, info).unwrap();
    let valid_identity = svod_device::device::source_stage_identity(info, &abi, linear, "// mock source").unwrap();
    let different_source = UOp::source_with_identity(
        "// mock source".into(),
        svod_ir::SourceStageIdentity { entry_name: "different".into(), ..valid_identity },
    );
    let different_source_parent =
        UOp::program(sink.clone(), info.clone(), Some(linear.clone()), Some(different_source.clone()), None);

    let (rendered, _) =
        crate::program_pipeline::do_render(&initial, &MockRenderer { device: DeviceSpec::Cpu }).unwrap();
    let Op::Program { source: Some(rendered_source), .. } = rendered.op() else { unreachable!() };
    assert!(!std::sync::Arc::ptr_eq(rendered_source, &raw_source));
    assert!(!std::sync::Arc::ptr_eq(rendered_source, &different_source));
    assert!(!std::sync::Arc::ptr_eq(&rendered, &raw_source_parent));
    assert!(!std::sync::Arc::ptr_eq(&rendered, &different_source_parent));
    assert!(matches!(rendered_source.op(), Op::Source { identity: Some(_), .. }));

    let raw_binary = UOp::binary(vec![1, 2, 3]);
    let raw_binary_parent = UOp::program(
        sink.clone(),
        info.clone(),
        Some(linear.clone()),
        Some(rendered_source.clone()),
        Some(raw_binary.clone()),
    );
    let Op::Source { identity: Some(source_identity), .. } = rendered_source.op() else { unreachable!() };
    let different_binary = UOp::binary_with_identity(
        vec![1, 2, 3],
        svod_device::device::binary_stage_identity(source_identity.clone(), "other", &[1, 2, 3]),
    );
    let different_binary_parent = UOp::program(
        sink.clone(),
        info.clone(),
        Some(linear.clone()),
        Some(rendered_source.clone()),
        Some(different_binary.clone()),
    );

    let (compiled, _) = crate::program_pipeline::do_compile(&rendered, &MockCompiler).unwrap();
    let Op::Program { binary: Some(compiled_binary), .. } = compiled.op() else { unreachable!() };
    assert!(!std::sync::Arc::ptr_eq(compiled_binary, &raw_binary));
    assert!(!std::sync::Arc::ptr_eq(compiled_binary, &different_binary));
    assert!(!std::sync::Arc::ptr_eq(&compiled, &raw_binary_parent));
    assert!(!std::sync::Arc::ptr_eq(&compiled, &different_binary_parent));
    assert!(matches!(compiled_binary.op(), Op::ProgramBinary { identity: Some(_), .. }));
}

#[test]
fn test_do_render_uses_linear_stage_input() {
    let sink = committed_sink(vec![UOp::native_const(5.0f32)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    let renderer = LinearOnlyRenderer { device: DeviceSpec::Cpu };

    let (rendered_program, spec) =
        crate::program_pipeline::do_render(&program, &renderer).expect("render stage should succeed");

    assert_eq!(spec.name, "test");
    assert!(matches!(spec.ast.op(), Op::Sink { .. }));
    match rendered_program.op() {
        Op::Program { linear, source, .. } => {
            assert!(linear.is_some(), "LINEAR stage should be present");
            assert!(source.is_some(), "SOURCE stage should be present");
        }
        other => panic!("expected PROGRAM op, got {other:?}"),
    }
}

#[test]
fn do_render_rejects_renderer_program_info_abi_disagreement_with_typed_error() {
    let sink = committed_sink(vec![UOp::native_const(5.0f32)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");
    let err =
        crate::program_pipeline::do_render(&program, &WrongAbiRenderer).expect_err("wrong renderer ABI must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

#[test]
fn renderer_reversing_storage_descriptors_is_rejected() {
    let sink = committed_sink(vec![UOp::param(0, 1, DType::Float32, None), UOp::param(1, 1, DType::Int32, None)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("PROGRAM");
    let err = crate::program_pipeline::do_render(&program, &ReversedStorageRenderer)
        .expect_err("same-count reversed storage ABI must fail");
    assert!(matches!(err, svod_device::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

#[test]
fn prebuilt_program_target_must_match_renderer() {
    let sink = committed_sink(vec![UOp::native_const(1i32)]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("PROGRAM");
    let renderer = LinearOnlyRenderer { device: DeviceSpec::Amd { device_id: 0 } };
    let err = crate::program_pipeline::do_render(&program, &renderer).expect_err("target mismatch must fail");
    assert!(matches!(err, svod_device::Error::ProgramTargetMismatch { .. }), "{err:?}");
}

#[test]
fn final_sink_verification_precedes_line_cleanup() {
    let valid = committed_sink(vec![UOp::native_const(5.0f32)]);
    let malformed = UOp::new(valid.op().clone(), DType::Float32);

    let err = crate::program_pipeline::program_from_sink(malformed, DeviceSpec::Cpu)
        .expect_err("the original final SINK dtype must be checked");
    assert!(format!("{err}").contains("SINK must be void"), "unexpected error: {err:?}");
}

#[test]
fn linear_only_program_cannot_bypass_final_sink_verification() {
    let valid = committed_sink(vec![UOp::native_const(5.0f32)]);
    let malformed = UOp::new(valid.op().clone(), DType::Float32);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(valid).into());
    let staged = program(malformed, DeviceSpec::Cpu, Some(linear), None, None);

    let err = crate::program_pipeline::get_program(
        &staged,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        crate::program_pipeline::ProgramTarget::Linear,
    )
    .expect_err("retaining an existing LINEAR stage must still verify its final SINK");
    assert!(format!("{err}").contains("SINK must be void"), "unexpected error: {err:?}");
}

#[test]
fn test_do_render_rejects_unproven_existing_source_stage() {
    let sink = committed_sink(vec![UOp::native_const(8.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(UOp::source("// stale source".to_string())), None);
    let renderer = MockRenderer { device: DeviceSpec::Cpu };

    let err = crate::program_pipeline::do_render(&program, &renderer)
        .expect_err("render should reject SOURCE stages without renderer identity");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");
}

#[test]
fn test_do_render_rejects_program_with_existing_binary_stage() {
    let sink = committed_sink(vec![UOp::native_const(9.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), None, Some(UOp::binary(vec![1, 2, 3])));
    let renderer = MockRenderer { device: DeviceSpec::Cpu };

    let err = crate::program_pipeline::do_render(&program, &renderer)
        .expect_err("render should reject programs that already have BINARY stage");
    let msg = format!("{err}");
    assert!(msg.contains("stages must be SINK"), "unexpected error: {msg}");
}

#[test]
fn test_do_compile_rejects_malformed_binary_stage() {
    let sink = committed_sink(vec![UOp::native_const(6.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());
    let source = UOp::source("// source".to_string());
    let malformed_binary = UOp::const_(DType::Float32, svod_ir::ConstValue::Float(1.0));
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), Some(malformed_binary));

    let err = crate::program_pipeline::do_compile(&program, &MockCompiler)
        .expect_err("compile should fail when binary stage is not ProgramBinary");
    assert!(format!("{err}").contains("ProgramBinary"));
}

#[test]
fn test_do_compile_rejects_empty_source_stage() {
    let sink = committed_sink(vec![UOp::native_const(7.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());
    let mut meta = ProgramSpec::new("empty_source".to_string(), String::new(), DeviceSpec::Cpu, sink.clone());
    meta.set_var_names(vec!["N".to_string()]);
    meta.buf_count = 1;

    let program =
        program(sink, DeviceSpec::Cpu, Some(linear), Some(UOp::source(String::new())), None).with_metadata(meta);

    let err = crate::program_pipeline::do_compile(&program, &MockCompiler)
        .expect_err("compile should fail when SOURCE stage is empty");
    assert!(format!("{err}").contains("empty SOURCE stage"));
}

#[test]
fn test_do_linearize_emits_cleaned_linear_stage() {
    let out = UOp::param(0, 16, DType::Float32, None);
    let idx = UOp::index_const(0);
    let gate = UOp::native_const(true);
    let out_index = UOp::index().buffer(out).indices(vec![idx]).call().expect("index");
    let store = out_index.store_gated(UOp::native_const(1.0f32), gate);
    let sink = committed_sink(vec![store]);
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("final target graph");

    let linearized = crate::program_pipeline::do_linearize(&program).expect("linearize stage should succeed");

    let Op::Program { linear: Some(linear), .. } = linearized.op() else {
        panic!("expected PROGRAM with LINEAR stage");
    };
    let Op::Linear { ops } = linear.op() else {
        panic!("expected LINEAR payload");
    };

    assert!(ops.iter().any(|u| matches!(u.op(), Op::If { .. })), "expected IF from cleanup");
    assert!(ops.iter().any(|u| matches!(u.op(), Op::EndIf { .. })), "expected ENDIF from cleanup");
    assert!(
        ops.iter().any(|u| {
            matches!(u.op(), Op::Store { index, gate: None, .. } if matches!(index.op(), Op::Index { .. }))
        })
    );
}

#[test]
fn test_hand_lowered_final_rewrite_stays_invalid_free_through_linearize() {
    let lane = UOp::special(UOp::index_const(4), "gidx0".to_string());
    let valid = lane.try_cmplt(&UOp::index_const(3)).expect("validity condition");
    let guarded_index = UOp::try_where(valid, lane.clone(), UOp::invalid_marker()).expect("guarded index");
    let input = UOp::param(1, 4, DType::Float32, None);
    let output = UOp::param(0, 4, DType::Float32, None);
    let load_index = UOp::index().buffer(input).indices(vec![guarded_index]).call().expect("input index");
    let value = UOp::load().index(load_index).call();
    let store_index = UOp::index().buffer(output).indices(vec![lane]).call().expect("output index");
    let sink = UOp::sink_with_info(
        vec![store_index.store(value)],
        svod_ir::KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() },
    );
    assert!(sink.toposort().iter().any(UOp::is_invalid_marker), "fixture must contain index validity");

    let optimizer_renderer = svod_schedule::OptimizerRenderer::amd_rdna3().with_rewrite_capabilities(
        svod_ir::RendererOps::all(),
        None,
        None,
    );
    let optimized = svod_schedule::optimize_kernel_with_config(
        sink,
        &optimizer_renderer,
        &svod_schedule::OptimizerConfig::default(),
    )
    .expect("hand-lowered final rewrite");
    assert!(
        optimized.toposort().iter().all(|u| !UOp::is_invalid_marker(u)),
        "mandatory final rewrite must remove Invalid from hand-lowered kernels"
    );

    let program = crate::program_pipeline::program_from_sink(optimized, DeviceSpec::Amd { device_id: 0 })
        .expect("final target graph");
    let linearized = crate::program_pipeline::do_linearize(&program).expect("PROGRAM -> LINEAR");
    let Op::Program { linear: Some(linear), .. } = linearized.op() else { panic!("expected LINEAR stage") };
    let Op::Linear { ops } = linear.op() else { panic!("expected LINEAR op") };
    assert!(
        ops.iter().all(|u| !UOp::is_invalid_marker(u)),
        "stage-20-clean input must remain Invalid-free through PROGRAM -> LINEAR"
    );
}

#[test]
fn test_structured_custom_name_wins_over_optimizer_shape_name() {
    let sink = UOp::sink_with_info(
        vec![UOp::noop()],
        svod_ir::KernelInfo {
            name: Some("flash_attention".to_string()),
            opts_to_apply: Some(vec![]),
            ..Default::default()
        },
    )
    .with_metadata(svod_schedule::optimizer::KernelInfo::new("E_L2L48", vec![], false));

    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).expect("program");
    let Op::Program { info, .. } = program.op() else { panic!("expected PROGRAM") };
    assert_eq!(info.name, "flash_attention");
}

#[test]
fn test_structured_symbolic_name_is_sanitized_at_renderer_boundary() {
    let sink = UOp::sink_with_info(
        vec![UOp::noop()],
        svod_ir::KernelInfo {
            name: Some("E_\x1b[31mL?\x1b[0mn6".to_string()),
            opts_to_apply: Some(vec![]),
            ..Default::default()
        },
    );

    for renderer in [&CAbiRenderer as &dyn Renderer, &LlvmAbiRenderer as &dyn Renderer] {
        let program =
            crate::program_pipeline::program_from_sink_with_renderer(sink.clone(), renderer).expect("program");
        let Op::Program { info, .. } = program.op() else { panic!("expected PROGRAM") };
        assert_eq!(info.name, "E_\x1b[31mL?\x1b[0mn6");
        assert_eq!(info.function_name(), "E_L3Fn6");

        let (_, spec) = crate::program_pipeline::do_render(&program, renderer).expect("render sanitized PROGRAM");
        assert_eq!(spec.name, "E_L3Fn6");
        assert!(spec.src.contains("E_L3Fn6"), "{}", spec.src);
        assert!(!spec.src.contains('?'), "{}", spec.src);
        assert!(!spec.src.contains('\x1b'), "{}", spec.src);
    }
}

#[test]
fn test_get_program_progresses_from_stage1_to_binary() {
    let sink = committed_sink(vec![UOp::native_const(3.0f32)]);
    let linear = UOp::linear(svod_schedule::linearize_with_cfg(sink.clone()).into());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), None, None);

    let advanced = crate::program_pipeline::get_program(
        &program,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect("stage-1 PROGRAM should advance to BINARY");

    match advanced.op() {
        Op::Program { linear, source, binary, .. } => {
            assert!(linear.is_some());
            assert!(source.is_some());
            assert!(binary.is_some());
        }
        other => panic!("expected PROGRAM op, got {other:?}"),
    }
}

#[test]
fn test_get_program_progresses_from_stage2_to_binary() {
    let sink = committed_sink(vec![UOp::native_const(4.0f32)]);
    let renderer = MockRenderer { device: DeviceSpec::Cpu };
    let program = crate::program_pipeline::program_from_sink(sink, DeviceSpec::Cpu).unwrap();
    let (program, _) = crate::program_pipeline::do_render(&program, &renderer).unwrap();

    let advanced = crate::program_pipeline::get_program(
        &program,
        &renderer,
        &MockCompiler,
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect("stage-2 PROGRAM should advance to BINARY");

    let (_, compiled) = crate::program_pipeline::do_compile(&advanced, &PanicCompiler)
        .expect("binary stage should be reusable after get_program");
    assert_eq!(compiled.bytes, vec![1, 2, 3]);
}

#[test]
fn staged_source_with_same_program_info_but_tampered_payload_is_rejected() {
    let renderer = MockRenderer { device: DeviceSpec::Cpu };
    let initial =
        crate::program_pipeline::program_from_sink(committed_sink(vec![UOp::native_const(4.0f32)]), DeviceSpec::Cpu)
            .unwrap();
    let (rendered, _) = crate::program_pipeline::do_render(&initial, &renderer).unwrap();
    let Op::Program { sink, info, linear, .. } = rendered.op() else { unreachable!() };
    let tampered = UOp::program(
        sink.clone(),
        info.clone(),
        linear.clone(),
        Some(UOp::source("// attacker-controlled source".into())),
        None,
    );

    let err = crate::program_pipeline::do_compile(&tampered, &MockCompiler)
        .expect_err("same ProgramInfo must not authenticate arbitrary source");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");
}

#[test]
fn staged_binary_from_different_signature_is_rejected() {
    let first = crate::program_pipeline::program_from_sink(
        committed_sink(vec![UOp::param(0, 4, DType::Float32, None)]),
        DeviceSpec::Cpu,
    )
    .unwrap();
    let second = crate::program_pipeline::program_from_sink(
        committed_sink(vec![UOp::param(0, 4, DType::Float32, None), UOp::param(5, 4, DType::Float32, None)]),
        DeviceSpec::Cpu,
    )
    .unwrap();
    let (first, _) = crate::program_pipeline::do_render(&first, &CAbiRenderer).unwrap();
    let (second, _) = crate::program_pipeline::do_render(&second, &CAbiRenderer).unwrap();
    let (second, _) = crate::program_pipeline::do_compile(&second, &MockCompiler).unwrap();
    let Op::Program { binary: Some(other_binary), .. } = second.op() else { unreachable!() };
    let Op::Program { sink, info, linear, source, .. } = first.op() else { unreachable!() };
    let mismatched =
        UOp::program(sink.clone(), info.clone(), linear.clone(), source.clone(), Some(other_binary.clone()));

    let err = crate::program_pipeline::do_compile(&mismatched, &MockCompiler)
        .expect_err("binary identity from another signature must not be reusable");
    assert!(matches!(err, svod_device::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

#[test]
fn test_get_program_rejects_malformed_staged_program() {
    let sink = committed_sink(vec![UOp::native_const(5.0f32)]);
    let malformed =
        program(sink, DeviceSpec::Cpu, None, Some(UOp::source("// source without linear".to_string())), None);

    let err = crate::program_pipeline::get_program(
        &malformed,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect_err("malformed PROGRAM state must be rejected");

    assert!(format!("{err}").contains("malformed PROGRAM state"));
}

#[test]
fn test_get_program_rejects_sink_input() {
    let sink = committed_sink(vec![UOp::native_const(1.0f32)]);

    let err = crate::program_pipeline::get_program(
        &sink,
        &MockRenderer { device: DeviceSpec::Cpu },
        &MockCompiler,
        crate::program_pipeline::ProgramTarget::Binary,
    )
    .expect_err("SINK input should be rejected by strict staged PROGRAM pipeline");

    assert!(format!("{err}").contains("expected PROGRAM input"));
}
