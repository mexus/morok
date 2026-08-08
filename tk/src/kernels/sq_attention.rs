//! FP32 single-query attention for decoder inference.
//!
//! One wave owns one `(batch, head)`. Q stays resident in registers while K/V
//! stream over N; lane `l` owns dimensions `l + j*wave_size`. Dot products use
//! XOR-shuffle all-reduces and a one-pass stable online softmax. There is no LDS,
//! MFMA, or split-K.

use std::sync::Arc;

use smallvec::smallvec;
use snafu::ensure;
use svod_dtype::{AmdArch, DType};
use svod_ir::{ConstValue, UOp};
use svod_tensor::Tensor;

use crate::Kernel;
use crate::index::{Idx, flat_index, load_at};
use crate::scaffold::GlSpec;

/// Architectures on which the scalar shuffle implementation is supported.
pub const SQ_ATTENTION_SUPPORTED_ARCHS: &[AmdArch] = &[AmdArch::Gfx942, AmdArch::Gfx1151];

/// Compile-time masking options for [`single_query_attention`].
#[derive(Clone, Copy, Default)]
pub struct SqAttentionOpts<'a> {
    /// Optional `[B]` i32 valid-key counts. Keys `0..key_lens[b]` are valid.
    pub key_lens: Option<&'a Tensor>,
    /// Also include key `N-1`. Required when `key_lens` is present; this is the
    /// Whisper self-cache layout where the current token occupies the final slot.
    pub include_last: bool,
}

fn cidx(v: i64) -> Arc<UOp> {
    UOp::const_(DType::Index, ConstValue::Int(v))
}

fn f32c(v: f64) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(v))
}

/// Build the one-wave single-query attention kernel.
///
/// ABI is `out, q, k, v, [key_lens]`, with sequence-major `[B,S,H,D]` globals.
pub(crate) fn build_single_query_attention(
    ker: &Kernel,
    b: usize,
    n: usize,
    h: usize,
    d: usize,
    masked: bool,
    include_last: bool,
) {
    let wave = ker.caps.wave_size;
    Kernel::assert_divisible(d, wave, "single-query attention D");
    assert!(n > 0, "single-query attention N must be > 0");
    let ept = d / wave;
    let warp = ker.warp();
    let f32 = DType::Float32;

    let (outs, ins) = ker.bind_abi(
        &[GlSpec::new(&[b, 1, h, d], f32.clone())],
        &[
            GlSpec::new(&[b, 1, h, d], f32.clone()),
            GlSpec::new(&[b, n, h, d], f32.clone()),
            GlSpec::new(&[b, n, h, d], f32.clone()),
        ],
    );
    let (out, q, k, v) = (outs[0].clone(), ins[0].clone(), ins[1].clone(), ins[2].clone());
    let batch = ker.grid_y();
    let head = ker.grid_x();
    let lane = ker.laneid();
    let prefix = masked.then(|| {
        let lens = ker.gl(&[b], DType::Int32);
        load_at(lens.uop(), lens.shape(), &[Idx::from(&batch)]).cast(DType::Index)
    });

    let q_reg = ker.alloc_reg(ept, f32.clone());
    let o_reg = ker.alloc_reg(ept, f32.clone());
    let max_reg = ker.alloc_reg(1, f32.clone());
    let norm_reg = ker.alloc_reg(1, f32.clone());
    let scale = f32c(std::f64::consts::LOG2_E / (d as f64).sqrt());

    let mut init = Vec::with_capacity(2 * ept + 2);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let qv = load_at(q.uop(), q.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
            .mul(&scale);
        init.push(flat_index(&q_reg, &[ept], &[Idx::Const(j as i64)]).store(qv));
        init.push(flat_index(&o_reg, &[ept], &[Idx::Const(j as i64)]).store(f32c(0.0)));
    }
    init.push(flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(f32c(f64::NEG_INFINITY)));
    init.push(flat_index(&norm_reg, &[1], &[Idx::Const(0)]).store(f32c(0.0)));
    let initialized = UOp::group(init);
    let q_reg = q_reg.after(smallvec![initialized.clone()]);
    let o_reg = o_reg.after(smallvec![initialized.clone()]);
    let max_reg = max_reg.after(smallvec![initialized.clone()]);
    let norm_reg = norm_reg.after(smallvec![initialized]);

    let lp = ker.loop_static(n as i64);
    let key = lp.index().clone();
    let q_loop = q_reg.after(smallvec![key.clone()]);
    let o_loop = o_reg.after(smallvec![key.clone()]);
    let max_loop = max_reg.after(smallvec![key.clone()]);
    let norm_loop = norm_reg.after(smallvec![key.clone()]);

    let mut dot = f32c(0.0);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let qv = load_at(&q_loop, &[ept], &[Idx::Const(j as i64)]);
        let kv = load_at(k.uop(), k.shape(), &[Idx::from(&batch), Idx::from(&key), Idx::from(&head), Idx::from(dim)]);
        dot = dot.add(&qv.mul(&kv));
    }
    let score = warp.wave_reduce_scalar(dot, |a, p| a.add(p));
    let valid = prefix.as_ref().map(|len| {
        let in_prefix = key.lt(len);
        if include_last { in_prefix.or_(&key.eq(&cidx(n as i64 - 1))) } else { in_prefix }
    });

    let old_max = load_at(&max_loop, &[1], &[Idx::Const(0)]);
    let old_norm = load_at(&norm_loop, &[1], &[Idx::Const(0)]);
    let next_max = old_max.max(&score);
    let alpha = old_max.sub(&next_max).try_exp2().expect("exp2 alpha");
    let beta = score.sub(&next_max).try_exp2().expect("exp2 beta");
    let candidate_norm = old_norm.mul(&alpha).add(&beta);
    let select = |candidate: Arc<UOp>, old: Arc<UOp>| match &valid {
        Some(pred) => UOp::try_where(pred.clone(), candidate, old).expect("mask select"),
        None => candidate,
    };
    let new_max = select(next_max, old_max);
    let new_norm = select(candidate_norm, old_norm);

    let max_store = flat_index(&max_reg, &[1], &[Idx::Const(0)]).store(new_max);
    let norm_store = flat_index(&norm_reg.after(smallvec![max_store.clone()]), &[1], &[Idx::Const(0)]).store(new_norm);
    let mut output_stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let old_o = load_at(&o_loop, &[ept], &[Idx::Const(j as i64)]);
        let vv = load_at(v.uop(), v.shape(), &[Idx::from(&batch), Idx::from(&key), Idx::from(&head), Idx::from(dim)]);
        let candidate = old_o.mul(&alpha).add(&vv.mul(&beta));
        let new_o = select(candidate, old_o);
        output_stores.push(
            flat_index(&o_reg.after(smallvec![norm_store.clone()]), &[ept], &[Idx::Const(j as i64)]).store(new_o),
        );
    }
    let output_group = UOp::group(output_stores);
    ker.push_store(output_group, o_reg.clone());
    let ended = lp.close();

    let final_o = o_reg.after(smallvec![ended.clone()]);
    let final_norm = norm_reg.after(smallvec![ended]);
    let denom = load_at(&final_norm, &[1], &[Idx::Const(0)]);
    let mut stores = Vec::with_capacity(ept);
    for j in 0..ept {
        let dim = lane.add(&cidx((j * wave) as i64));
        let value = load_at(&final_o, &[ept], &[Idx::Const(j as i64)]).try_div(&denom).expect("normalize");
        stores.push(
            flat_index(out.uop(), out.shape(), &[Idx::from(&batch), Idx::Const(0), Idx::from(&head), Idx::from(dim)])
                .store(value),
        );
    }
    ker.push_store(UOp::group(stores), out.uop().clone());
}

/// Graph-native FP32 single-query attention.
///
/// Q is `[B,1,H,D]`, K/V are `[B,N,H,D]`, and output is `[B,1,H,D]`.
/// Returns `Ok(None)` when the target is not gfx942/gfx1151. No generic SDPA
/// fallback is performed here.
pub fn single_query_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    opts: SqAttentionOpts<'_>,
) -> crate::LaunchResult<Option<Tensor>> {
    let qd = crate::launch::concrete_dims(q, "single-query attention", "q", 4)?;
    let kd = crate::launch::concrete_dims(k, "single-query attention", "k", 4)?;
    let vd = crate::launch::concrete_dims(v, "single-query attention", "v", 4)?;
    let (b, n, h, d) = (qd[0], kd[1], qd[2], qd[3]);
    let dtype = q.uop().dtype();
    let masked = opts.key_lens.is_some();

    ensure!(
        qd[1] == 1,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "Q sequence",
            value: qd[1],
            multiple: 1usize
        }
    );
    ensure!(
        kd[0] == b,
        crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim: "K batch B", a: kd[0], b }
    );
    ensure!(
        kd[2] == h,
        crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim: "K heads H", a: kd[2], b: h }
    );
    ensure!(
        kd[3] == d,
        crate::launch::OperandDimMismatchSnafu {
            kernel: "single-query attention",
            dim: "K head dim D",
            a: kd[3],
            b: d
        }
    );
    for (dim, a, expected) in
        [("V batch B", vd[0], b), ("V sequence N", vd[1], n), ("V heads H", vd[2], h), ("V head dim D", vd[3], d)]
    {
        ensure!(
            a == expected,
            crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim, a, b: expected }
        );
    }
    ensure!(
        n > 0,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "N (> 0)",
            value: n,
            multiple: 1usize
        }
    );
    ensure!(
        !masked || opts.include_last,
        crate::launch::DimMultipleSnafu {
            kernel: "single-query attention",
            dim: "include_last (required with key_lens)",
            value: opts.include_last as usize,
            multiple: 1usize
        }
    );
    if let Some(lens) = opts.key_lens {
        let ld = crate::launch::concrete_dims(lens, "single-query attention", "key_lens", 1)?;
        ensure!(
            ld == [b],
            crate::launch::OperandDimMismatchSnafu { kernel: "single-query attention", dim: "key_lens B", a: ld[0], b }
        );
        ensure!(
            lens.uop().dtype() == DType::Int32,
            crate::launch::DtypeSnafu { kernel: "single-query attention", got: lens.uop().dtype(), expected: "i32" }
        );
    }

    crate::launch_custom(
        &q.device(),
        SQ_ATTENTION_SUPPORTED_ARCHS,
        move |arch| {
            let wave = arch.wave_size() as usize;
            ensure!(
                dtype == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: dtype.clone(), expected: "f32" }
            );
            ensure!(
                k.uop().dtype() == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: k.uop().dtype(), expected: "f32" }
            );
            ensure!(
                v.uop().dtype() == DType::Float32,
                crate::launch::DtypeSnafu { kernel: "single-query attention", got: v.uop().dtype(), expected: "f32" }
            );
            ensure!(
                d.is_multiple_of(wave),
                crate::launch::DimMultipleSnafu {
                    kernel: "single-query attention",
                    dim: "D",
                    value: d,
                    multiple: wave
                }
            );
            Ok(())
        },
        true,
        move |arch| {
            let caps = crate::ArchCaps::for_arch(arch);
            let out = Tensor::empty(&[b, 1, h, d], DType::Float32);
            let mut inputs = vec![q, k, v];
            if let Some(lens) = opts.key_lens {
                inputs.push(lens);
            }
            crate::graph_launch(
                "sq_attention",
                [h as i64, b as i64, 1],
                caps.wave_size as i64,
                out,
                &inputs,
                caps,
                move |ker| {
                    build_single_query_attention(ker, b, n, h, d, masked, opts.include_last);
                    ker.finish(1)
                },
            )
        },
    )
}
