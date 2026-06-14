//! Golden structural fingerprints of the production kernel builders — the committed
//! regression oracle. Because the LLVM render is non-deterministic
//! ([`crate::fingerprint`]), behavior preservation is checked on the build-time UOp
//! graph: a refactor that changes a kernel's graph changes its digest. Update an
//! `expected` const ONLY for an intentional graph change — the failure message
//! prints the new value to paste.

use std::sync::Arc;

use svod_dtype::{DType, DeviceSpec};
use svod_ir::UOp;

use crate::kernels::fa::{FaConfig, build_fa_mw_rdb};
use crate::kernels::matmul::{M1_CFG, build_matmul_cfg, build_matmul_db};
use crate::{ArchCaps, Kernel, kernel_fingerprint};

fn matmul_sink() -> Arc<UOp> {
    let n = 512usize;
    let bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
    ];
    let ker =
        Kernel::new("matmul_cfg", M1_CFG.grid_dims(n), M1_CFG.threads(crate::WARP_THREADS), bufs, ArchCaps::GFX942);
    build_matmul_cfg(&ker, n, M1_CFG);
    ker.finish(M1_CFG.n_accum)
}

fn matmul_db_sink() -> Arc<UOp> {
    let n = 512usize;
    let bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::Float32),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, n * n, DType::BFloat16),
    ];
    let ker =
        Kernel::new("matmul_db", M1_CFG.grid_dims(n), M1_CFG.threads(crate::WARP_THREADS), bufs, ArchCaps::GFX942);
    build_matmul_db(&ker, n);
    ker.finish(M1_CFG.n_accum)
}

fn fa_sink() -> Arc<UOp> {
    let (b, h, h_kv, d, n) = (1usize, 2usize, 2usize, 64usize, 128usize);
    let bufs = vec![
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_kv * d, DType::BFloat16),
        UOp::new_buffer(DeviceSpec::Cpu, b * n * h_kv * d, DType::BFloat16),
    ];
    let ker = Kernel::new("fa_mw_rdb", [h as i64, (n / 16 / 8) as i64, b as i64], 8 * 64, bufs, ArchCaps::GFX942);
    build_fa_mw_rdb(
        &ker,
        b,
        n,
        h,
        h_kv,
        d,
        FaConfig { q_blk: 16, kv_blk: 16, ..Default::default() },
        DType::BFloat16,
        false,
    );
    ker.finish(1)
}

// Committed gfx942 golden digests. Update ONLY for an intentional graph change.
// Re-baked for 4c (subtile): the per-warp band moved from a folded address term
// into the tile's base_offset — same address VALUE, one extra add node. Validated
// numerically on gfx942 (matmul *_amd) + gfx1151/395.
const MATMUL_DIGEST: u128 = 0x99eb_67e7_9598_ec54_0000_0000_0000_0000;
const MATMUL_NODES: usize = 483;
// Re-baked for the online-softmax rescale rewrite: `scale_vec` dropped its scratch
// buffer and is now the `(max_vec_last - &max_vec).exp2()` same-shape vec−vec op
// (reusing `max_vec_last`'s dead buffer) instead of a hand-rolled `load_at` merge —
// 3 fewer nodes, same numerics. Validated on gfx942 (fa *_amd) + gfx1151/395.
const FA_DIGEST: u128 = 0x97cf_434e_4785_1a9f_0000_0000_0000_0000;
const FA_NODES: usize = 878;
const MATMUL_DB_DIGEST: u128 = 0x1bde_e499_c405_69fb_0000_0000_0000_0000;
const MATMUL_DB_NODES: usize = 643;

fn check(name: &str, sink: Arc<UOp>, digest: u128, nodes: usize) {
    let fp = kernel_fingerprint(&sink);
    assert_eq!(
        (fp.digest, fp.node_count),
        (digest, nodes),
        "{name} graph changed. If intentional, set the const to:\n  \
         DIGEST = 0x{:032x}; NODES = {};\nop_counts = {:#?}",
        fp.digest,
        fp.node_count,
        fp.op_counts
    );
}

#[test]
fn golden_matmul_cfg() {
    check("matmul_cfg", matmul_sink(), MATMUL_DIGEST, MATMUL_NODES);
}

#[test]
fn golden_fa_mw_rdb() {
    check("fa_mw_rdb", fa_sink(), FA_DIGEST, FA_NODES);
}

#[test]
fn golden_matmul_db() {
    check("matmul_db", matmul_db_sink(), MATMUL_DB_DIGEST, MATMUL_DB_NODES);
}

/// The fingerprint is invariant to the global id counter: building the same kernel
/// twice in one process (fresh ids each time) yields the same digest.
#[test]
fn fingerprint_is_build_deterministic() {
    assert_eq!(kernel_fingerprint(&matmul_sink()).digest, kernel_fingerprint(&matmul_sink()).digest);
    assert_eq!(kernel_fingerprint(&fa_sink()).digest, kernel_fingerprint(&fa_sink()).digest);
}
