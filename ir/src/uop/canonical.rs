//! Stable, allocation-independent serialization for UOp DAGs.
//!
//! Directly serializing [`UOp`] would encode recursive `Arc`s, duplicate shared
//! nodes, and expose runtime IDs and caches. This module instead emits a
//! dependency-first node table with graph-local source IDs.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use serde::Serialize;
use svod_dtype::{AddrSpace, DType, ImageKind, ScalarDType};

use crate::{BinaryOp, ConstValue, Op, SInt, TernaryOp, UOp};

/// Version of the canonical graph schema.
pub const CANONICAL_SCHEMA_VERSION: u32 = 1;

/// Canonical graph representation used by cross-implementation parity tests.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalGraph {
    pub schema_version: u32,
    pub stage: String,
    pub roots: Vec<usize>,
    pub nodes: Vec<CanonicalNode>,
}

/// One UOp in a canonical dependency-first node table.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CanonicalNode {
    pub id: usize,
    pub op: String,
    pub dtype: CanonicalDType,
    pub shape: Option<Vec<CanonicalShapeDim>>,
    pub arg: CanonicalArg,
    pub src: Vec<usize>,
}

/// Language-neutral dtype representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalDType {
    Scalar { name: String },
    Vector { scalar: String, count: usize },
    Pointer { base: Box<CanonicalDType>, address_space: String, size: Option<usize>, count: usize },
    Image { image_kind: String, shape: Vec<usize> },
}

/// Stable shape dimension. Symbolic dimensions refer to a graph-local node ID
/// when that expression is part of the serialized DAG.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalShapeDim {
    Const { value: usize },
    Symbolic { node: Option<usize> },
    Infer,
}

/// Constants use float bit patterns so NaN, infinities, and signed zero remain
/// deterministic and valid JSON.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalConst {
    Invalid,
    Int { value: i64 },
    UInt { value: u64 },
    Float { bits: String },
    Bool { value: bool },
}

/// Operation metadata with all UOp sources removed. Sources are represented by
/// [`CanonicalNode::src`], keeping the Serde schema acyclic.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CanonicalArg {
    None,
    Const {
        value: CanonicalConst,
    },
    Id {
        value: usize,
    },
    Device {
        name: String,
    },
    Sink {
        name: Option<String>,
        opts_to_apply: Option<Vec<crate::Opt>>,
    },
    DType {
        value: CanonicalDType,
    },
    Index {
        value: usize,
    },
    Name {
        value: String,
    },
    Param {
        slot: usize,
        size: usize,
    },
    Size {
        value: usize,
    },
    View {
        size: usize,
        offset: usize,
    },
    Bufferize {
        device: Option<String>,
        address_space: String,
        removable: bool,
    },
    Axes {
        values: Vec<usize>,
    },
    BoolAxes {
        values: Vec<bool>,
    },
    Reduce {
        op: String,
        axes: Option<Vec<usize>>,
    },
    Range {
        axis: usize,
        renumbered: bool,
        axis_type: String,
    },
    Constants {
        values: Vec<CanonicalConst>,
    },
    DefineVar {
        name: String,
        min: i64,
        max: i64,
    },
    DefineReg {
        size: usize,
        id: usize,
    },
    AxisPairs {
        values: Vec<(usize, usize)>,
    },
    Call {
        grad_tag: Option<String>,
        metadata: Vec<String>,
        name: Option<String>,
        precompile: bool,
        precompile_backward: bool,
    },
    Wmma {
        name: String,
        dims: (usize, usize, usize),
        dtype_in: CanonicalDType,
        dtype_out: CanonicalDType,
        device: String,
        threads: usize,
        upcast_a: Vec<(usize, usize)>,
        upcast_b: Vec<(usize, usize)>,
        upcast_c: Vec<(usize, usize)>,
        reduce_axes: Vec<usize>,
        tile_grid: (usize, usize),
    },
    Source {
        code: String,
    },
    Binary {
        length: usize,
        xxh64: String,
    },
    Hints {
        values: Vec<crate::ContiguousHint>,
    },
    Code {
        value: String,
    },
    CustomFunction {
        kind_name: String,
    },
}

impl CanonicalGraph {
    /// Serialize one root and all of its call/program bodies.
    pub fn from_root(stage: impl Into<String>, root: &Arc<UOp>) -> crate::Result<Self> {
        Self::from_roots(stage, std::slice::from_ref(root))
    }

    /// Serialize multiple ordered roots into one deduplicated node table.
    pub fn from_roots(stage: impl Into<String>, roots: &[Arc<UOp>]) -> crate::Result<Self> {
        let mut seen = HashSet::new();
        let mut topo = Vec::new();
        for root in roots {
            for node in root.toposort() {
                if seen.insert(node.id) {
                    topo.push(node);
                }
            }
        }

        let ids: HashMap<u64, usize> = topo.iter().enumerate().map(|(id, node)| (node.id, id)).collect();
        let nodes =
            topo.iter().enumerate().map(|(id, node)| canonical_node(id, node, &ids)).collect::<crate::Result<_>>()?;
        let roots = roots.iter().map(|root| ids[&root.id]).collect();

        Ok(Self { schema_version: CANONICAL_SCHEMA_VERSION, stage: stage.into(), roots, nodes })
    }

    /// Render canonical JSON with deterministic field and node ordering.
    pub fn to_pretty_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(self)
    }
}

/// Emit one canonical graph to stderr when its stage matches
/// `SVOD_DUMP_CANONICAL_STAGE` by prefix.
pub fn dump_canonical_stage(stage: &str, root: &Arc<UOp>) {
    let Ok(prefix) = std::env::var("SVOD_DUMP_CANONICAL_STAGE") else {
        return;
    };
    if !stage.starts_with(&prefix) {
        return;
    }

    eprintln!("[dump-canonical] {stage} :");
    match CanonicalGraph::from_root(stage, root) {
        Ok(graph) => match graph.to_pretty_json() {
            Ok(json) => eprintln!("{json}"),
            Err(error) => eprintln!("[dump-canonical] {stage} : JSON error: {error}"),
        },
        Err(error) => eprintln!("[dump-canonical] {stage} : graph error: {error}"),
    }
    eprintln!("[dump-canonical] {stage} : end");
}

fn canonical_node(id: usize, node: &Arc<UOp>, ids: &HashMap<u64, usize>) -> crate::Result<CanonicalNode> {
    let shape = node.shape()?.map(|shape| {
        shape
            .iter()
            .map(|dim| match dim {
                SInt::Const(value) => CanonicalShapeDim::Const { value: *value },
                SInt::Symbolic(expr) => CanonicalShapeDim::Symbolic { node: ids.get(&expr.id).copied() },
                SInt::Infer => CanonicalShapeDim::Infer,
            })
            .collect()
    });
    let src = node.op().children().into_iter().map(|source| ids[&source.id]).collect();

    Ok(CanonicalNode {
        id,
        op: canonical_op_name(node.op()),
        dtype: canonical_dtype(&node.dtype()),
        shape,
        arg: canonical_arg(node.op()),
        src,
    })
}

fn scalar_name(dtype: ScalarDType) -> String {
    match dtype {
        ScalarDType::Bool => "bool",
        ScalarDType::WeakInt => "weakint",
        ScalarDType::Int8 => "int8",
        ScalarDType::UInt8 => "uint8",
        ScalarDType::Int16 => "int16",
        ScalarDType::UInt16 => "uint16",
        ScalarDType::Int32 => "int32",
        ScalarDType::UInt32 => "uint32",
        ScalarDType::Int64 => "int64",
        ScalarDType::UInt64 => "uint64",
        ScalarDType::WeakFloat => "weakfloat",
        ScalarDType::FP8E4M3 => "fp8e4m3",
        ScalarDType::FP8E5M2 => "fp8e5m2",
        ScalarDType::Float16 => "float16",
        ScalarDType::BFloat16 => "bfloat16",
        ScalarDType::Float32 => "float32",
        ScalarDType::Float64 => "float64",
        ScalarDType::Void => "void",
        ScalarDType::Index => "index",
    }
    .to_string()
}

fn canonical_dtype(dtype: &DType) -> CanonicalDType {
    match dtype {
        DType::Scalar(scalar) => CanonicalDType::Scalar { name: scalar_name(*scalar) },
        DType::Vector { scalar, count } => CanonicalDType::Vector { scalar: scalar_name(*scalar), count: *count },
        DType::Ptr { base, addrspace, size, vcount } => CanonicalDType::Pointer {
            base: Box::new(canonical_dtype(base)),
            address_space: match addrspace {
                AddrSpace::Global => "global",
                AddrSpace::Local => "local",
                AddrSpace::Reg => "register",
            }
            .to_string(),
            size: *size,
            count: *vcount,
        },
        DType::Image { kind, shape } => CanonicalDType::Image {
            image_kind: match kind {
                ImageKind::Half => "half",
                ImageKind::Float => "float",
            }
            .to_string(),
            shape: shape.clone(),
        },
    }
}

fn canonical_const(value: ConstValue) -> CanonicalConst {
    match value {
        ConstValue::Invalid => CanonicalConst::Invalid,
        ConstValue::Int(value) => CanonicalConst::Int { value },
        ConstValue::UInt(value) => CanonicalConst::UInt { value },
        ConstValue::Float(value) => CanonicalConst::Float { bits: format!("0x{:016x}", value.to_bits()) },
        ConstValue::Bool(value) => CanonicalConst::Bool { value },
    }
}

fn address_space_name(address_space: AddrSpace) -> String {
    match address_space {
        AddrSpace::Global => "global",
        AddrSpace::Local => "local",
        AddrSpace::Reg => "register",
    }
    .to_string()
}

fn canonical_arg(op: &Op) -> CanonicalArg {
    match op {
        Op::Const(constant) => CanonicalArg::Const { value: canonical_const(constant.0) },
        Op::Unique(id) | Op::LUnique(id) | Op::DefineLocal(id) => CanonicalArg::Id { value: *id },
        Op::Device(device) => CanonicalArg::Device { name: device.canonicalize() },
        Op::Sink { info: None, .. } => CanonicalArg::None,
        Op::Sink { info: Some(info), .. } => {
            CanonicalArg::Sink { name: info.name.clone(), opts_to_apply: info.opts_to_apply.clone() }
        }
        Op::Cast { dtype, .. } | Op::BitCast { dtype, .. } => CanonicalArg::DType { value: canonical_dtype(dtype) },
        Op::MSelect { device_index, .. } => CanonicalArg::Index { value: *device_index },
        Op::Special { name, .. } => CanonicalArg::Name { value: name.clone() },
        Op::Param { slot, size, device: _ } => CanonicalArg::Param { slot: *slot, size: *size },
        Op::Buffer { size, .. } => CanonicalArg::Size { value: *size },
        Op::BufferView { size, offset, .. } => CanonicalArg::View { size: *size, offset: *offset },
        Op::Bufferize { opts, .. } => CanonicalArg::Bufferize {
            device: opts.device.as_ref().map(|device| device.canonicalize()),
            address_space: address_space_name(opts.addrspace),
            removable: opts.removable,
        },
        Op::Permute { axes, .. } => CanonicalArg::Axes { values: axes.clone() },
        Op::Flip { axes, .. } => CanonicalArg::BoolAxes { values: axes.clone() },
        Op::Multi { axis, .. } => CanonicalArg::Index { value: *axis },
        Op::ReduceAxis { reduce_op, axes, .. } => {
            CanonicalArg::Reduce { op: reduce_name(*reduce_op).to_string(), axes: Some(axes.clone()) }
        }
        Op::Reduce { reduce_op, .. } | Op::AllReduce { reduce_op, .. } => {
            CanonicalArg::Reduce { op: reduce_name(*reduce_op).to_string(), axes: None }
        }
        Op::Range { axis_id, axis_type, .. } => CanonicalArg::Range {
            axis: axis_id.value(),
            renumbered: axis_id.is_renumbered(),
            axis_type: format!("{axis_type:?}").to_ascii_uppercase(),
        },
        Op::Gep { indices, .. } => CanonicalArg::Axes { values: indices.clone() },
        Op::VConst { values } => {
            CanonicalArg::Constants { values: values.iter().copied().map(canonical_const).collect() }
        }
        Op::DefineVar { name, min_val, max_val } => {
            CanonicalArg::DefineVar { name: name.clone(), min: *min_val, max: *max_val }
        }
        Op::DefineReg { size, id } => CanonicalArg::DefineReg { size: *size, id: *id },
        Op::Wmma { metadata, .. } => CanonicalArg::Wmma {
            name: metadata.name.clone(),
            dims: metadata.dims,
            dtype_in: canonical_dtype(&metadata.dtype_in),
            dtype_out: canonical_dtype(&metadata.dtype_out),
            device: metadata.device.canonical().to_string(),
            threads: metadata.threads,
            upcast_a: metadata.upcast_axes.a.clone(),
            upcast_b: metadata.upcast_axes.b.clone(),
            upcast_c: metadata.upcast_axes.c.clone(),
            reduce_axes: metadata.reduce_axes.clone(),
            tile_grid: metadata.tile_grid,
        },
        Op::Contract { upcast_ranges, .. } => CanonicalArg::AxisPairs { values: upcast_ranges.clone() },
        Op::Unroll { unroll_axes, .. } => CanonicalArg::AxisPairs { values: unroll_axes.clone() },
        Op::Call { info, .. } | Op::Function { info, .. } => CanonicalArg::Call {
            grad_tag: info.grad_tag.clone(),
            metadata: info.metadata.clone(),
            name: info.name.clone(),
            precompile: info.precompile,
            precompile_backward: info.precompile_backward,
        },
        Op::GetTuple { index, .. } => CanonicalArg::Index { value: *index },
        Op::Source { code } => CanonicalArg::Source { code: code.clone() },
        Op::ProgramBinary { bytes } => CanonicalArg::Binary {
            length: bytes.len(),
            xxh64: format!("0x{:016x}", xxhash_rust::xxh64::xxh64(bytes, 0)),
        },
        Op::Contiguous { opts, .. } => CanonicalArg::Hints { values: opts.to_vec() },
        Op::Custom { code, .. } | Op::CustomI { code, .. } => CanonicalArg::Code { value: code.clone() },
        Op::CustomFunction { kind, .. } => CanonicalArg::CustomFunction { kind_name: format!("{kind:?}") },
        Op::Noop
        | Op::Unary(..)
        | Op::Binary(..)
        | Op::Ternary(..)
        | Op::Group { .. }
        | Op::Index { .. }
        | Op::PointerIndex { .. }
        | Op::Copy { .. }
        | Op::MStack { .. }
        | Op::Stack { .. }
        | Op::Reshape { .. }
        | Op::Expand { .. }
        | Op::Pad { .. }
        | Op::Shrink { .. }
        | Op::If { .. }
        | Op::EndIf { .. }
        | Op::End { .. }
        | Op::Barrier { .. }
        | Op::Vectorize { .. }
        | Op::Cat { .. }
        | Op::PtrCat { .. }
        | Op::Bind { .. }
        | Op::Tuple { .. }
        | Op::Program { .. }
        | Op::Linear { .. }
        | Op::Detach { .. }
        | Op::ContiguousBackward { .. }
        | Op::After { .. }
        | Op::Precast { .. }
        | Op::Load { .. }
        | Op::Store { .. } => CanonicalArg::None,
    }
}

fn canonical_op_name(op: &Op) -> String {
    match op {
        Op::Unary(kind, _) => kind.as_ref().to_ascii_uppercase(),
        Op::Binary(kind, _, _) => binary_name(*kind).to_string(),
        Op::Ternary(kind, _, _, _) => ternary_name(*kind).to_string(),
        _ => match op {
            Op::Const(_) => "CONST",
            Op::Unique(_) => "UNIQUE",
            Op::LUnique(_) => "LUNIQUE",
            Op::Device(_) => "DEVICE",
            Op::Noop => "NOOP",
            Op::DefineLocal(_) => "DEFINE_LOCAL",
            Op::Sink { .. } => "SINK",
            Op::Group { .. } => "GROUP",
            Op::Cast { .. } => "CAST",
            Op::BitCast { .. } => "BITCAST",
            Op::MSelect { .. } => "MSELECT",
            Op::Special { .. } => "SPECIAL",
            Op::Param { .. } => "PARAM",
            Op::Buffer { .. } => "BUFFER",
            Op::BufferView { .. } => "BUFFER_VIEW",
            Op::Bufferize { .. } => "BUFFERIZE",
            Op::Index { .. } => "INDEX",
            Op::PointerIndex { .. } => "POINTER_INDEX",
            Op::Copy { .. } => "COPY",
            Op::MStack { .. } => "MSTACK",
            Op::Stack { .. } => "STACK",
            Op::Reshape { .. } => "RESHAPE",
            Op::Permute { .. } => "PERMUTE",
            Op::Expand { .. } => "EXPAND",
            Op::Pad { .. } => "PAD",
            Op::Shrink { .. } => "SHRINK",
            Op::Flip { .. } => "FLIP",
            Op::Multi { .. } => "MULTI",
            Op::ReduceAxis { .. } => "REDUCE_AXIS",
            Op::Reduce { .. } => "REDUCE",
            Op::AllReduce { .. } => "ALLREDUCE",
            Op::If { .. } => "IF",
            Op::EndIf { .. } => "ENDIF",
            Op::Range { .. } => "RANGE",
            Op::End { .. } => "END",
            Op::Barrier { .. } => "BARRIER",
            Op::Vectorize { .. } => "VECTORIZE",
            Op::Gep { .. } => "GEP",
            Op::VConst { .. } => "VCONST",
            Op::Cat { .. } => "CAT",
            Op::PtrCat { .. } => "PTRCAT",
            Op::DefineVar { .. } => "DEFINE_VAR",
            Op::Bind { .. } => "BIND",
            Op::DefineReg { .. } => "DEFINE_REG",
            Op::Wmma { .. } => "WMMA",
            Op::Contract { .. } => "CONTRACT",
            Op::Unroll { .. } => "UNROLL",
            Op::Call { .. } => "CALL",
            Op::Function { .. } => "FUNCTION",
            Op::Tuple { .. } => "TUPLE",
            Op::GetTuple { .. } => "GETTUPLE",
            Op::Program { .. } => "PROGRAM",
            Op::Linear { .. } => "LINEAR",
            Op::Source { .. } => "SOURCE",
            Op::ProgramBinary { .. } => "BINARY",
            Op::Detach { .. } => "DETACH",
            Op::Contiguous { .. } => "CONTIGUOUS",
            Op::ContiguousBackward { .. } => "CONTIGUOUS_BACKWARD",
            Op::After { .. } => "AFTER",
            Op::Precast { .. } => "PRECAST",
            Op::Custom { .. } => "CUSTOM",
            Op::CustomFunction { .. } => "CUSTOM_FUNCTION",
            Op::CustomI { .. } => "CUSTOMI",
            Op::Load { .. } => "LOAD",
            Op::Store { .. } => "STORE",
            Op::Unary(..) | Op::Binary(..) | Op::Ternary(..) => unreachable!(),
        }
        .to_string(),
    }
}

fn binary_name(op: BinaryOp) -> &'static str {
    match op {
        BinaryOp::Add => "ADD",
        BinaryOp::Mul => "MUL",
        BinaryOp::Sub => "SUB",
        BinaryOp::Mod => "MOD",
        BinaryOp::Max => "MAX",
        BinaryOp::Pow => "POW",
        BinaryOp::Idiv => "IDIV",
        BinaryOp::Fdiv => "FDIV",
        BinaryOp::Lt => "CMPLT",
        BinaryOp::Le => "CMPLE",
        BinaryOp::Eq => "CMPEQ",
        BinaryOp::Ne => "CMPNE",
        BinaryOp::Gt => "CMPGT",
        BinaryOp::Ge => "CMPGE",
        BinaryOp::And => "AND",
        BinaryOp::Or => "OR",
        BinaryOp::Xor => "XOR",
        BinaryOp::Shl => "SHL",
        BinaryOp::Shr => "SHR",
        BinaryOp::Threefry => "THREEFRY",
    }
}

fn ternary_name(op: TernaryOp) -> &'static str {
    match op {
        TernaryOp::Where => "WHERE",
        TernaryOp::MulAcc => "MULACC",
    }
}

fn reduce_name(op: crate::ReduceOp) -> &'static str {
    match op {
        crate::ReduceOp::Add => "ADD",
        crate::ReduceOp::Mul => "MUL",
        crate::ReduceOp::Max => "MAX",
        crate::ReduceOp::Min => "MIN",
    }
}
