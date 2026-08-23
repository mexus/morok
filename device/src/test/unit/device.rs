use crate::{DeviceSpec, registry::DeviceSpecExt};
use svod_dtype::DType;
use svod_ir::{Op, UOp};

fn program(
    sink: std::sync::Arc<UOp>,
    target: DeviceSpec,
    linear: Option<std::sync::Arc<UOp>>,
    source: Option<std::sync::Arc<UOp>>,
    binary: Option<std::sync::Arc<UOp>>,
) -> std::sync::Arc<UOp> {
    let info = svod_ir::ProgramInfo::from_sink(&sink, target);
    let source = match (&linear, source) {
        (Some(linear), Some(source)) => match source.op() {
            Op::Source { code, identity: None } => crate::device::ProgramSpec::validate_program_param_abi(&sink, &info)
                .ok()
                .and_then(|abi| crate::device::source_stage_identity(&info, &abi, linear, code).ok())
                .map_or_else(
                    || Some(source.clone()),
                    |identity| Some(UOp::source_with_identity(code.clone(), identity)),
                ),
            _ => Some(source),
        },
        (_, source) => source,
    };
    let binary = match (&source, binary) {
        (Some(source), Some(binary)) => match (source.op(), binary.op()) {
            (Op::Source { identity: Some(source_identity), .. }, Op::ProgramBinary { bytes, identity: None }) => {
                Some(UOp::binary_with_identity(
                    bytes.clone(),
                    crate::device::binary_stage_identity(source_identity.clone(), "device-test", bytes),
                ))
            }
            _ => Some(binary),
        },
        (_, binary) => binary,
    };
    UOp::program(sink, info, linear, source, binary)
}

fn slotted_var(name: &str, min: i64, max: i64, slot: usize) -> std::sync::Arc<UOp> {
    let var = UOp::variable(name.to_string(), min, max, DType::Int32);
    let Op::Param { shape, arg } = var.op() else { panic!("variable PARAM") };
    let mut arg = arg.clone();
    arg.slot = slot;
    UOp::new(Op::Param { shape: shape.clone(), arg }, DType::Int32)
}

#[test]
fn compiled_spec_requires_complete_descriptor_abi() {
    use crate::device::{AbiParamDescriptor, AbiParamKind, CompiledSpec, validate_abi_descriptors};
    use svod_dtype::AddrSpace;

    let abi = vec![
        AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor { slot: 5, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("n".into()) },
    ];
    let spec = CompiledSpec::from_source("k".into(), "void k(float*, int) {}".into(), UOp::sink(vec![]), abi)
        .expect("descriptor ABI");
    assert_eq!(spec.buf_count, 1);
    assert_eq!(spec.var_names, ["n"]);

    let err = validate_abi_descriptors(&[], 1, &[]).expect_err("descriptorless argument-bearing ABI must fail");
    assert!(matches!(err, crate::Error::ProgramAbiMismatch { .. }), "{err:?}");
}

#[test]
fn test_device_spec_parse() {
    assert_eq!(DeviceSpec::parse("CPU").unwrap(), DeviceSpec::Cpu);
    assert_eq!(DeviceSpec::parse("cpu").unwrap(), DeviceSpec::Cpu);

    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSpec::parse("CUDA:0").unwrap(), DeviceSpec::Cuda { device_id: 0 });
        assert_eq!(DeviceSpec::parse("cuda").unwrap(), DeviceSpec::Cuda { device_id: 0 });
        assert_eq!(DeviceSpec::parse("GPU:2").unwrap(), DeviceSpec::Cuda { device_id: 2 });
    }
}

#[test]
fn test_device_spec_parse_amd() {
    assert_eq!(DeviceSpec::parse("AMD").unwrap(), DeviceSpec::Amd { device_id: 0 });
    assert_eq!(DeviceSpec::parse("AMD:1").unwrap(), DeviceSpec::Amd { device_id: 1 });
    assert_eq!(DeviceSpec::parse("hip:2").unwrap(), DeviceSpec::Amd { device_id: 2 });
}

#[test]
fn test_device_spec_canonicalize() {
    assert_eq!(DeviceSpec::Cpu.canonicalize(), "CPU");

    #[cfg(feature = "cuda")]
    {
        assert_eq!(DeviceSpec::Cuda { device_id: 1 }.canonicalize(), "CUDA:1");
    }

    assert_eq!(DeviceSpec::Amd { device_id: 0 }.canonicalize(), "AMD:0");
    assert_eq!(DeviceSpec::Amd { device_id: 2 }.canonicalize(), "AMD:2");
}

#[test]
fn test_amd_device_open_returns_clean_result() {
    // On hosts without AMD GPU: NoAmdGpu.
    // On hosts with an unsupported arch (e.g. RDNA2/gfx1036): AmdAllocFailed.
    // On hosts with a supported gfx target: Ok. Never panics; that's the
    // load-bearing assertion.
    use crate::error::Error;
    match crate::registry::get_device("AMD:0") {
        Ok(_)
        | Err(Error::NoAmdGpu { .. })
        | Err(Error::AmdAllocFailed { .. })
        | Err(Error::AmdIoctl { .. })
        | Err(Error::DeviceUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error variant: {other:?}"),
    }
}

#[test]
fn test_program_spec_from_uop_ignores_metadata_overrides() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// test kernel".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let mut spec =
        crate::device::ProgramSpec::new("k_test".to_string(), "// old src".to_string(), DeviceSpec::Cpu, sink.clone());
    spec.buf_count = 2;

    let program = program.with_metadata(spec.clone());
    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("program spec from uop");

    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.src, "// test kernel");
    assert_eq!(rebuilt.device, DeviceSpec::Cpu);
    assert_eq!(rebuilt.ast.id, sink.id);
    assert!(rebuilt.var_names.is_empty());
    assert!(rebuilt.globals.is_empty());
    assert!(rebuilt.outs.is_empty());
    assert!(rebuilt.ins.is_empty());
    assert_eq!(rebuilt.buf_count, 0);
}

#[test]
fn test_program_spec_from_uop_ignores_metadata_io() {
    let sink = UOp::sink(vec![UOp::native_const(1.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// test kernel".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let mut spec =
        crate::device::ProgramSpec::new("k_test".to_string(), "// old src".to_string(), DeviceSpec::Cpu, sink);
    spec.set_buffer_metadata(vec![1, 0], vec![1], vec![0]);
    spec.buf_count = 2;

    let rebuilt = crate::device::ProgramSpec::from_uop(&program.with_metadata(spec)).expect("program spec from uop");
    assert!(rebuilt.globals.is_empty());
    assert!(rebuilt.outs.is_empty());
    assert!(rebuilt.ins.is_empty());
    assert_eq!(rebuilt.buf_count, 0);
}

#[test]
fn test_program_spec_from_uop_without_metadata_derives_name_and_vars() {
    let var = slotted_var("N", 1, 8, 0);
    let sink = UOp::sink(vec![var]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void var_kernel(float* data0) {}".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should succeed");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.var_names, vec!["N".to_string()]);
    assert_eq!(rebuilt.vars.len(), 1);
}

#[test]
fn test_program_spec_derives_launch_dims_from_specials() {
    let g = UOp::special(UOp::index_const(8), "gidx0".to_string());
    let l = UOp::special(UOp::index_const(4), "lidx0".to_string());
    let sink = UOp::sink(vec![g, l]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void launch_kernel() {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from specials");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [8, 1, 1]);
    assert_eq!(launch.local_size, Some([4, 1, 1]));
}

#[test]
fn test_program_spec_launch_dims_resolves_mulacc_extent() {
    // The symbolic simplifier fuses `16*ts − 1` (from a reshaped `16·ts`
    // sequence axis) into a single MulAcc launch extent. The launch-size
    // evaluator must compute `ts*16 − 1` rather than reject the op.
    let ts = slotted_var("ts", 1, 8, 0);
    let sixteen = UOp::const_(DType::Int32, 16.into());
    let minus_one = UOp::const_(DType::Int32, (-1).into());
    let extent = UOp::try_mulacc(ts, sixteen, minus_one).expect("build MulAcc extent");
    let g = UOp::special(extent, "gidx0".to_string());
    let sink = UOp::sink(vec![g]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void mulacc_kernel() {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from mulacc special");
    let vars = std::collections::HashMap::from([("ts", 8i64)]);
    let launch = spec.launch_dims(&vars).expect("resolve launch dims with MulAcc extent");
    assert_eq!(launch.global_size, [127, 1, 1], "ts*16 - 1 = 8*16 - 1 = 127");
}

#[test]
fn test_program_spec_direct_global_special_disables_local_size() {
    let idx = UOp::special(UOp::index_const(16), "idx0".to_string());
    let sink = UOp::sink(vec![idx]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void direct_global_kernel() {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from idx special");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [16, 1, 1]);
    assert_eq!(launch.local_size, None);
}

#[test]
fn test_program_spec_core_id_sets_cpu_global_size() {
    let core_id = slotted_var("core_id", 0, 7, 0);
    let sink = UOp::sink(vec![core_id]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void core_kernel(int core_id) {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let spec = crate::device::ProgramSpec::from_uop(&program).expect("program spec from core_id");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [8, 1, 1]);
}

#[test]
fn test_program_spec_metadata_launch_dims_do_not_hide_program_info_core_id() {
    let core_id = slotted_var("core_id", 0, 3, 0);
    let sink = UOp::sink(vec![core_id]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void core_kernel(int core_id) {}".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);
    let meta = crate::device::ProgramSpec::new("core".to_string(), "// old".to_string(), DeviceSpec::Cpu, sink);

    let spec = crate::device::ProgramSpec::from_uop(&program.with_metadata(meta)).expect("program spec from metadata");
    let vars = std::collections::HashMap::new();
    let launch = spec.launch_dims(&vars).expect("resolve launch dims");
    assert_eq!(launch.global_size, [4, 1, 1]);
}

#[test]
fn test_program_spec_from_uop_without_metadata_derives_buf_count_and_io() {
    let param = UOp::param(0, 16, DType::Float32, None);
    let idx = UOp::index_const(0);
    let load_idx = UOp::index().buffer(param.clone()).indices(vec![idx.clone()]).call().expect("load index");
    let load = UOp::load().index(load_idx).call();
    let store_idx = UOp::index().buffer(param).indices(vec![idx]).call().expect("store index");
    let sink = UOp::sink(vec![store_idx.store(load)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void io_kernel(float* data0) {}".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should derive I/O");
    assert_eq!(rebuilt.globals, vec![0]);
    assert_eq!(rebuilt.outs, vec![0]);
    assert_eq!(rebuilt.ins, vec![0]);
    assert_eq!(rebuilt.buf_count, 1);
}

#[test]
fn program_spec_rejects_duplicate_storage_scalar_slots_with_typed_error() {
    let global = UOp::param(0, 1, DType::Float32, None);
    let scalar = UOp::variable("n".to_string(), 1, 8, DType::Int32);
    let Op::Param { shape, arg } = scalar.op() else { unreachable!() };
    let mut arg = arg.clone();
    arg.slot = 0;
    let scalar = UOp::new(Op::Param { shape: shape.clone(), arg }, DType::Int32);
    let sink = UOp::sink(vec![global, scalar]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void duplicate(float* data0, int n) {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let err = crate::device::ProgramSpec::from_uop(&program).expect_err("duplicate slot must fail");
    assert!(matches!(err, crate::Error::DuplicateProgramParamSlot { slot: 0, .. }), "{err:?}");
}

#[test]
fn program_spec_rejects_descriptor_equivalent_var_semantic_forgery() {
    let scalar = slotted_var("n", 0, 16, 0);
    let sink = UOp::sink(vec![scalar]);

    for mutation in ["bounds", "multiple_of"] {
        let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
        let Op::Param { shape, arg } = info.vars[0].op() else { unreachable!() };
        let mut forged_arg = arg.clone();
        if mutation == "bounds" {
            forged_arg.vmin_vmax = Some((
                svod_ir::ConstValueHash(svod_ir::ConstValue::Int(-1000)),
                svod_ir::ConstValueHash(svod_ir::ConstValue::Int(1000)),
            ));
        } else {
            forged_arg.multiple_of = Some(8);
        }
        info.vars[0] = UOp::new(Op::Param { shape: shape.clone(), arg: forged_arg }, DType::Int32);
        let staged = UOp::program(
            sink.clone(),
            info,
            Some(UOp::linear(sink.toposort().into())),
            Some(UOp::source("void forged(int n) {}".to_string())),
            None,
        );

        let err = crate::device::ProgramSpec::from_uop(&staged).expect_err("semantic forgery must fail");
        match err {
            crate::Error::ProgramAbiMismatch { reason } => {
                assert!(reason.contains("ProgramInfo.vars"), "{mutation}: {reason}");
            }
            other => panic!("{mutation}: expected ProgramAbiMismatch, got {other:?}"),
        }
    }
}

#[test]
fn program_spec_accepts_semantically_identical_nonidentical_var() {
    let scalar = slotted_var("n", 0, 16, 0);
    let sink = UOp::sink(vec![scalar]);
    let mut info = svod_ir::ProgramInfo::from_sink(&sink, DeviceSpec::Cpu);
    let sink_var = info.vars[0].clone();
    let reconstructed = UOp::new(sink_var.op().clone(), sink_var.dtype()).with_metadata("detached variable");
    assert!(!std::sync::Arc::ptr_eq(&sink_var, &reconstructed));
    info.vars[0] = reconstructed;
    let staged = UOp::program(sink.clone(), info, Some(UOp::linear(sink.toposort().into())), None, None);
    let Op::Program { info, linear: Some(linear), .. } = staged.op() else { unreachable!() };
    let abi = crate::device::ProgramSpec::validate_program_param_abi(&sink, info).unwrap();
    let code = "void accepted(int n) {}".to_string();
    let identity = crate::device::source_stage_identity(info, &abi, linear, &code).unwrap();
    let staged = UOp::program(
        sink.clone(),
        info.clone(),
        Some(linear.clone()),
        Some(UOp::source_with_identity(code, identity)),
        None,
    );

    crate::device::ProgramSpec::from_uop(&staged)
        .expect("validation must compare PARAM value semantics rather than allocation identity");
}

#[test]
fn test_program_spec_from_uop_requires_program_source() {
    let sink = UOp::sink(vec![UOp::native_const(3.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let program_without_source = program(sink.clone(), DeviceSpec::Cpu, Some(linear), None, None);
    assert!(crate::device::ProgramSpec::from_uop(&program_without_source).is_err());

    let non_program = UOp::native_const(1.0f32);
    assert!(crate::device::ProgramSpec::from_uop(&non_program).is_err());

    let bad_source = UOp::native_const(1.0f32);
    let linear = UOp::linear(sink.toposort().into());
    let bad_program = program(sink, DeviceSpec::Cpu, Some(linear), Some(bad_source), None);
    assert!(crate::device::ProgramSpec::from_uop(&bad_program).is_err());

    let raw_sink = UOp::sink(vec![]);
    let raw_linear = UOp::linear(raw_sink.toposort().into());
    let raw = UOp::program(
        raw_sink.clone(),
        svod_ir::ProgramInfo::from_sink(&raw_sink, DeviceSpec::Cpu),
        Some(raw_linear),
        Some(UOp::source("unproven".into())),
        None,
    );
    let err = crate::device::ProgramSpec::from_uop(&raw).expect_err("identity-less SOURCE must be rejected");
    assert!(matches!(err, crate::Error::ProgramStageMismatch { stage: "SOURCE", .. }), "{err:?}");

    if let Op::Program { .. } = bad_program.op() {
        // ensure we exercised Program path in this test
    } else {
        panic!("expected PROGRAM op");
    }
}

#[test]
fn test_program_spec_from_uop_binary_stage_ignores_metadata() {
    let sink = UOp::sink(vec![UOp::native_const(4.0f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("// binary source".to_string());
    let program = program(sink.clone(), DeviceSpec::Cpu, Some(linear), Some(source), Some(UOp::binary(vec![1, 2, 3])));

    let mut spec =
        crate::device::ProgramSpec::new("precompiled".to_string(), "// cached src".to_string(), DeviceSpec::Cpu, sink);
    spec.set_var_names(vec!["N".to_string()]);
    spec.buf_count = 3;

    let program = program.with_metadata(spec);
    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("program spec from binary+metadata");
    assert_eq!(rebuilt.name, "test");
    assert_eq!(rebuilt.src, "// binary source");
    assert!(rebuilt.var_names.is_empty());
    assert_eq!(rebuilt.buf_count, 0);
}

#[test]
fn program_spec_rejects_empty_binary_compiler_key() {
    let sink = UOp::sink(vec![]);
    let linear = UOp::linear(sink.toposort().into());
    let staged = program(
        sink,
        DeviceSpec::Cpu,
        Some(linear),
        Some(UOp::source("source".into())),
        Some(UOp::binary(vec![1, 2, 3])),
    );
    let Op::Program { sink, info, linear, source, binary: Some(binary) } = staged.op() else { unreachable!() };
    let Op::ProgramBinary { bytes, identity: Some(identity) } = binary.op() else { unreachable!() };
    let malformed = UOp::binary_with_identity(
        bytes.clone(),
        svod_ir::BinaryStageIdentity { compiler_key: String::new(), ..identity.clone() },
    );
    let staged = UOp::program(sink.clone(), info.clone(), linear.clone(), source.clone(), Some(malformed));
    let err = crate::device::ProgramSpec::from_uop(&staged).expect_err("empty compiler key must be rejected");
    assert!(matches!(err, crate::Error::ProgramStageMismatch { stage: "BINARY", .. }), "{err:?}");
}

#[test]
fn test_program_spec_from_uop_without_metadata_defaults_name_to_kernel() {
    let sink = UOp::sink(vec![UOp::native_const(4.5f32)]);
    let linear = UOp::linear(sink.toposort().into());
    let source = UOp::source("void default_name_kernel() {}".to_string());
    let program = program(sink, DeviceSpec::Cpu, Some(linear), Some(source), None);

    let rebuilt = crate::device::ProgramSpec::from_uop(&program).expect("metadata-free from_uop should succeed");
    assert_eq!(rebuilt.name, "test");
}
