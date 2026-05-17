//! Loader for pyannote's `pytorch_model.bin` checkpoint format.
//!
//! pyannote saves checkpoints as
//! `torch.save({"state_dict": <OrderedDict>, "pytorch-lightning_version": "..."}, path)`
//! which is a ZIP archive containing a top-level pickle that wraps the inner
//! state dict in another dict. `repugnant-pickle::torch::RepugnantTorchTensors`
//! only walks a single top-level OrderedDict / Dict, so we use its lower-level
//! `parse_ops` + `evaluate` directly and descend one level into the
//! `state_dict` key before extracting tensors.

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use morok_dtype::DType;
use morok_tensor::Tensor;
use repugnant_pickle::ops::PickleOp;
use repugnant_pickle::{SequenceType, Value, evaluate, parse_ops};
use snafu::{ResultExt, Snafu};
use zip::ZipArchive;
use zip::result::ZipError;

use crate::state::StateDict;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("io: {source}"))]
    Io { source: std::io::Error },
    #[snafu(display("zip: {source}"))]
    Zip { source: ZipError },
    #[snafu(display("pickle parse: {message}"))]
    Parse { message: String },
    #[snafu(display("data.pkl not found in archive"))]
    NoDataPkl,
    #[snafu(display("unexpected pickle structure: {context}"))]
    Structure { context: String },
    #[snafu(display("unsupported tensor dtype: {dtype}"))]
    UnsupportedDtype { dtype: String },
    #[snafu(display("compressed storage is not supported"))]
    CompressedStorage,
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(morok_tensor::error::Error, Box::new)))]
        source: Box<morok_tensor::error::Error>,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Load a PyTorch-Lightning checkpoint of the shape
/// `{"state_dict": <OrderedDict>, "pytorch-lightning_version": "..."}` and
/// return its `state_dict` as a [`StateDict`], stripping `strip_prefix` from
/// each key (use `""` to keep keys verbatim).
pub fn load_pyannote_pytorch_bin(path: &Path, strip_prefix: &str) -> Result<StateDict> {
    let raw_file = File::open(path).context(IoSnafu)?;
    let mut zp = ZipArchive::new(raw_file.try_clone().context(IoSnafu)?).context(ZipSnafu)?;

    // 1. Find the `*/data.pkl` member.
    let data_pkl_name =
        zp.file_names().find(|s| s.ends_with("/data.pkl")).map(str::to_owned).ok_or(Error::NoDataPkl)?;
    let pfx = data_pkl_name.rsplit_once('/').map(|(p, _)| p.to_owned()).unwrap_or_default();

    // 2. Read its bytes.
    let mut pkl_bytes = Vec::new();
    zp.by_name(&data_pkl_name).context(ZipSnafu)?.read_to_end(&mut pkl_bytes).context(IoSnafu)?;

    // 3. Parse + evaluate.
    let (_remain, ops) = parse_ops::<nom::error::VerboseError<&[u8]>>(&pkl_bytes)
        .map_err(|e| Error::Parse { message: format!("{e:?}") })?;
    let (vals, _memo) = evaluate(&ops, true).map_err(|e| Error::Parse { message: format!("{e:?}") })?;

    // 4. Toplevel for pyannote checkpoints (`torch.save({"state_dict": ...,
    //    "pyannote.audio": ..., "pytorch-lightning_version": ...})`) parses
    //    via repugnant-pickle as `Seq(Dict, [Seq(Tuple, [(k,v), (k,v), ...])])`
    //    — a single-item Dict whose lone slot is a Tuple of (k, v) sub-tuples.
    //    Walk through it, find the `state_dict` entry, and unwrap the
    //    `Build(Global(OrderedDict, [_, items_tuple]), _state)` it points to.
    let toplevel = vals.first().ok_or_else(|| Error::Structure { context: "no toplevel value".into() })?;
    let top_items = unwrap_outer_dict(toplevel)
        .ok_or_else(|| Error::Structure { context: "toplevel not recognised as a dict-of-pairs".into() })?;
    let state_dict_value = top_items
        .iter()
        .find_map(|item| match item {
            Value::Seq(SequenceType::Tuple, seq)
                if seq.len() == 2 && matches!(&seq[0], Value::String("state_dict")) =>
            {
                Some(&seq[1])
            }
            _ => None,
        })
        .ok_or_else(|| Error::Structure { context: "no 'state_dict' key at toplevel".into() })?;
    let sd_items = unwrap_ordered_dict_items(state_dict_value)
        .ok_or_else(|| Error::Structure { context: "state_dict value is not an OrderedDict".into() })?;

    // 5. For each entry, decode `_rebuild_tensor_v2` args and read bytes from
    //    the right place in the zip. Cache each storage member's data_start so
    //    tensors sharing storage don't re-resolve.
    let mut data_starts: HashMap<String, u64> = HashMap::new();
    let mut raw_file = raw_file;
    let mut sd = StateDict::new();

    for di in sd_items {
        let (k, v) = match di {
            Value::Seq(SequenceType::Tuple, seq) if seq.len() == 2 => (&seq[0], &seq[1]),
            _ => continue,
        };
        let name = match k {
            Value::String(s) => *s,
            _ => continue,
        };
        let args = match v {
            Value::Global(g, seq)
                if g.as_ref() == &Value::Raw(Cow::Owned(PickleOp::GLOBAL("torch._utils", "_rebuild_tensor_v2"))) =>
            {
                seq
            }
            _ => continue,
        };

        let (pidval, offs_elems, shape_vals) = match args.as_slice() {
            [Value::Seq(SequenceType::Tuple, seq)] => match seq.as_slice() {
                [
                    Value::PersId(pidval),
                    Value::Int(offs),
                    Value::Seq(SequenceType::Tuple, shape),
                    Value::Seq(SequenceType::Tuple, _stride),
                    Value::Bool(_grad),
                    ..,
                ] => (pidval.as_ref(), *offs as u64, shape),
                _ => {
                    return Err(Error::Structure { context: format!("unexpected _rebuild_tensor_v2 args for {name}") });
                }
            },
            _ => return Err(Error::Structure { context: format!("unexpected outer Seq for {name}") }),
        };

        let shape: Vec<usize> = shape_vals
            .iter()
            .map(|x| match x {
                Value::Int(n) => Ok(*n as usize),
                _ => Err(Error::Structure { context: "non-int shape dim".into() }),
            })
            .collect::<Result<Vec<_>>>()?;

        let (stype_str, sfile_id) = match pidval {
            Value::Seq(SequenceType::Tuple, seq) => match seq.as_slice() {
                [
                    Value::String("storage"),
                    Value::Raw(op),
                    Value::String(sfile),
                    Value::String(_sdev),
                    Value::Int(_slen),
                ] => match op.as_ref() {
                    PickleOp::GLOBAL("torch", styp) if styp.ends_with("Storage") => {
                        (&styp[..styp.len() - "Storage".len()], *sfile)
                    }
                    _ => return Err(Error::Structure { context: "unexpected storage type".into() }),
                },
                _ => return Err(Error::Structure { context: "unexpected PID seq".into() }),
            },
            _ => return Err(Error::Structure { context: "unexpected PID".into() }),
        };

        let dtype = parse_dtype(stype_str)?;
        let elem_size = dtype.bytes();
        let sfile = format!("{pfx}/data/{sfile_id}");

        let data_start = if let Some(start) = data_starts.get(&sfile) {
            *start
        } else {
            let zf = zp.by_name(&sfile).context(ZipSnafu)?;
            if zf.compression() != zip::CompressionMethod::STORE {
                return Err(Error::CompressedStorage);
            }
            let start = zf.data_start();
            drop(zf);
            data_starts.insert(sfile.clone(), start);
            start
        };

        let element_count: usize = shape.iter().product();
        let byte_offset = offs_elems * elem_size as u64;
        let byte_length = element_count * elem_size;

        let abs = data_start + byte_offset;
        raw_file.seek(SeekFrom::Start(abs)).context(IoSnafu)?;
        let mut buf = vec![0u8; byte_length];
        raw_file.read_exact(&mut buf).context(IoSnafu)?;

        let tensor = Tensor::from_raw_bytes(&buf, &shape, dtype).context(TensorSnafu)?;
        let final_name = name.strip_prefix(strip_prefix).unwrap_or(name).to_string();
        sd.insert(final_name, tensor);
    }

    Ok(sd)
}

/// Accept the various shapes that repugnant-pickle hands us for a top-level
/// Python dict: either `Seq(Dict, [pair, pair, ...])` (paired), or the wrapper
/// form `Seq(Dict, [Seq(Tuple, [pair, pair, ...])])` that pyannote/lightning
/// checkpoints surface as. Returns the inner Vec of (key, value) sub-tuples.
fn unwrap_outer_dict<'b, 'a: 'b>(v: &'b Value<'a>) -> Option<&'b Vec<Value<'a>>> {
    match v {
        Value::Seq(SequenceType::Dict, items) => {
            // Heuristic: torch.save of a plain Python dict surfaces here as a
            // single-slot `Seq(Dict, [Seq(Tuple, [pair, pair, ...])])`. Peel
            // that wrapper iff each child is itself a 2-element Tuple (a key,
            // value pair); otherwise treat items as already paired.
            if items.len() == 1
                && let Value::Seq(SequenceType::Tuple, inner) = &items[0]
                && inner.iter().all(|x| matches!(x, Value::Seq(SequenceType::Tuple, kv) if kv.len() == 2))
            {
                return Some(inner);
            }
            Some(items)
        }
        Value::Build(body, _state) => unwrap_outer_dict(body.as_ref()),
        _ => None,
    }
}

/// Unwrap the OrderedDict-as-value pattern emitted by torch.save: the dict's
/// value is `Build(Global(OrderedDict, [empty_tuple, items_tuple]), _state)`.
/// Returns the `items_tuple` contents (list of `Seq(Tuple, [k, v])` pairs).
fn unwrap_ordered_dict_items<'b, 'a: 'b>(v: &'b Value<'a>) -> Option<&'b Vec<Value<'a>>> {
    let inner = match v {
        Value::Build(body, _state) => body.as_ref(),
        _ => v,
    };
    match inner {
        Value::Global(g, args) => match g.as_ref() {
            Value::Raw(rv) if **rv == PickleOp::GLOBAL("collections", "OrderedDict") => match args.as_slice() {
                [_, Value::Seq(SequenceType::Tuple, items), ..] => Some(items),
                _ => None,
            },
            _ => None,
        },
        Value::Seq(SequenceType::Dict, items) => Some(items),
        _ => None,
    }
}

fn parse_dtype(s: &str) -> Result<DType> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "float64" | "double" => DType::Float64,
        "float32" | "float" => DType::Float32,
        "float16" | "half" => DType::Float16,
        "bfloat16" => DType::BFloat16,
        "int64" | "long" => DType::Int64,
        "int32" | "int" => DType::Int32,
        "int16" | "short" => DType::Int16,
        "int8" | "char" => DType::Int8,
        "uint8" | "byte" => DType::UInt8,
        other => return Err(Error::UnsupportedDtype { dtype: other.into() }),
    })
}
