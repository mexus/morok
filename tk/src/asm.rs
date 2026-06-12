//! Raw gfx942 scheduling/synchronization primitives, injected as `Op::Custom`
//! Void side-effects (M2 K-loop pipeline). These carry no data — each takes a
//! `dep` purely so the linearizer's toposort sequences it *after* the prior
//! cluster and a consumer can `.after([..])` it to sequence the next cluster
//! *after* it (and keep it live through DCE).
//!
//! `s_setprio`/`s_waitcnt` ride `call … asm sideeffect` (a scheduling boundary
//! the AMDGPU backend cannot reorder across); `sched_barrier` rides the
//! `@llvm.amdgcn.sched.barrier` intrinsic (its `declare` is auto-hoisted to the
//! module prefix by the CUSTOM renderer). The `dep` is referenced only for
//! ordering — the emitted text has no `{N}` placeholder, which the strict
//! template validator allows (it only bounds-checks present placeholders).

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::UOp;

/// `s_setprio N` (N ∈ 0..=3): raise/lower this wave's issue priority around an
/// MFMA burst so the scheduler keeps the systolic array fed (`GEMM` cluster
/// `s_setprio(1)` before the MFMAs, `s_setprio(0)` after).
pub fn s_setprio(prio: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(smallvec![dep], format!("call void asm sideeffect \"s_setprio {prio}\", \"\"()"), DType::Void)
}

/// `s_waitcnt lgkmcnt(n)`: drain outstanding LDS (`ds_read`/`ds_write`) traffic
/// down to `n` before proceeding — the deferred-wait the register-staged
/// prefetch relies on (issue the next tile's loads, then wait only at the
/// consuming cluster).
pub fn s_waitcnt_lgkmcnt(n: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(smallvec![dep], format!("call void asm sideeffect \"s_waitcnt lgkmcnt({n})\", \"\"()"), DType::Void)
}

/// `@llvm.amdgcn.sched.barrier(mask)`: a hard instruction-scheduling fence.
/// `mask = 0` forbids *any* instruction from moving across it, pinning each
/// cluster's loads/MFMAs/`ds_write` into their program-order region so the
/// pipeline structure survives the AMDGPU machine scheduler.
pub fn sched_barrier(mask: i64, dep: Arc<UOp>) -> Arc<UOp> {
    UOp::custom(
        smallvec![dep],
        format!(
            "declare void @llvm.amdgcn.sched.barrier(i32)\n\
             call void @llvm.amdgcn.sched.barrier(i32 {mask})"
        ),
        DType::Void,
    )
}
