//! Hardware-gated gfx942 (MI300X) throughput benchmark for the svod-tk matmul
//! and flash-attention forward kernels — measured on **GPU device time**.
//!
//! Each kernel is **compiled once** ([`crate::compile_kernel`]) so the timed
//! region is pure dispatch — render/compile is excluded, matching how a JIT'd
//! kernel is actually replayed. We warm up (stabilize clocks + page-in buffers),
//! validate correctness against svod's own Tensor reference op at every size,
//! then time `N` dispatches.
//!
//! The reported number is **GPU device time**: the HW-stamped per-dispatch CP
//! timestamps ([`CompiledLaunch::dispatch_gpu_ns`], 10 ns/tick) give true
//! on-device kernel time, free of host launch/submit overhead — at these tiny
//! kernels (tens of µs) wall-clock is dominated by dispatch overhead and masks
//! kernel-level wins. We keep a wall-clock column (`dispatch(true)`, the real
//! replay path) so `wall − gpu` shows exactly how much was launch overhead.
//!
//! The reference (`Tensor::matmul` / `Tensor::scaled_dot_product_attention`) is
//! prepared once into an [`svod_runtime::ExecutionPlan`] and replayed with
//! `execute_profiled`, whose per-kernel `gpu_{start,end}_ns` we sum for an
//! apples-to-apples GPU-time comparison.
//!
//! Run:
//! `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bench -- --ignored --nocapture`

use std::panic::AssertUnwindSafe;
use std::time::Instant;

use svod_dtype::DType;
use svod_tensor::Tensor;

use crate::CompiledLaunch;
use crate::kernels::fa::{FaConfig, build_fa, build_fa_kv, build_fa_mw, build_fa_mw_db, build_fa_mw_rdb};
use crate::kernels::matmul::build_matmul;

/// MI300X dense bf16 matrix-engine peak throughput (TFLOP/s). AMD's CDNA3 spec
/// sheet lists ~1307.4 TFLOP/s peak bf16 matrix (non-sparse) for the MI300X.
const MI300X_BF16_PEAK_TFLOPS: f64 = 1307.0;

// ── timing primitives ───────────────────────────────────────────────────────

fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 }
}

/// GPU device-time + wall-clock samples (microseconds) from a timed run.
struct Timing {
    /// HW-stamped on-device kernel time per dispatch (µs). Empty on backends
    /// that don't stamp (CPU) — then `gpu_med` falls back to wall.
    gpu_us: Vec<f64>,
    /// Host wall-clock per completion-synced dispatch (µs).
    wall_us: Vec<f64>,
}

impl Timing {
    /// Median GPU device time (µs), or median wall-clock when unstamped.
    fn gpu_med(&self) -> f64 {
        if self.gpu_us.is_empty() { median(&self.wall_us) } else { median(&self.gpu_us) }
    }

    /// Median host wall-clock (µs).
    fn wall_med(&self) -> f64 {
        median(&self.wall_us)
    }
}

/// Warm up, then time `iters` svod-tk dispatches twice: a **wall pass**
/// (`dispatch(true)`, the real replay path — so `wall − gpu` is pure host
/// launch/submit/sync overhead) and a **GPU pass** ([`CompiledLaunch::dispatch_gpu_ns`],
/// HW-stamped on-device kernel time). Returns both sample sets in µs.
fn time_launch(launch: &CompiledLaunch, warmup: usize, iters: usize) -> Timing {
    for _ in 0..warmup {
        // SAFETY: bound buffers stay allocated for `launch`'s lifetime.
        unsafe { launch.dispatch(true).expect("warmup dispatch") };
    }
    let mut wall_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        unsafe { launch.dispatch(true).expect("timed wall dispatch") };
        wall_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let mut gpu_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        if let Some(ns) = launch.dispatch_gpu_ns().expect("timed gpu dispatch") {
            gpu_us.push(ns as f64 / 1e3);
        }
    }
    Timing { gpu_us, wall_us }
}

/// Warm up then time `iters` reference-plan replays. `execute_profiled` drains
/// the device timeline and exposes per-kernel HW GPU stamps; we sum them across
/// the plan's kernels (a reference SDPA/GEMM may lower to several) for the GPU
/// device time per replay, and wrap each replay in wall-clock too.
fn time_plan(plan: &svod_runtime::ExecutionPlan, warmup: usize, iters: usize) -> Timing {
    for _ in 0..warmup {
        plan.execute_profiled().expect("warmup reference");
    }
    let mut gpu_us = Vec::with_capacity(iters);
    let mut wall_us = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let profiles = plan.execute_profiled().expect("timed reference");
        wall_us.push(t.elapsed().as_secs_f64() * 1e6);
        let gpu_ns: u64 = profiles
            .iter()
            .filter_map(|p| match (p.gpu_start_ns, p.gpu_end_ns) {
                (Some(s), Some(e)) => Some(e - s),
                _ => None,
            })
            .sum();
        if gpu_ns > 0 {
            gpu_us.push(gpu_ns as f64 / 1e3);
        }
    }
    Timing { gpu_us, wall_us }
}

/// One benchmarked row (a kernel + its reference at a given size). All times µs.
struct Row {
    n: usize,
    correct: bool,
    max_abs: f32,
    /// svod-tk GPU device time (median µs) and host wall-clock (median µs).
    tk_gpu_us: f64,
    tk_wall_us: f64,
    /// TFLOPS derived from GPU device time.
    tflops: f64,
    pct_peak: f64,
    /// Reference GPU device time + wall (median µs).
    ref_gpu_us: f64,
    ref_wall_us: f64,
    /// ref_gpu / tk_gpu — speedup on GPU device time (>1 = tk faster).
    speedup: f64,
}

// ── matmul ────────────────────────────────────────────────────────────────

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bench_matmul_amd -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_matmul_amd() {
    // 8-wave (2×4) 256×256-block / 512-thread kernel; per-N iteration counts
    // keep the whole sweep within a few minutes.
    let sizes: &[(usize, usize)] = &[(512, 50), (1024, 50), (2048, 30), (4096, 12)];
    let mut rows = Vec::new();
    for &(n, iters) in sizes {
        match std::panic::catch_unwind(AssertUnwindSafe(|| run_matmul(n, iters))) {
            Ok(row) => rows.push(row),
            Err(_) => println!("matmul N={n}: SKIPPED (panic — likely OOM or compile/dispatch failure)"),
        }
    }
    print_matmul_table(&rows);
}

fn run_matmul(n: usize, iters: usize) -> Row {
    use crate::kernels::matmul::M1_CFG;

    // bf16 inputs realized once so kernel + reference see identical rounded
    // values; f32 output (the kernel accumulates in f32 like MFMA).
    let mut a = Tensor::rand(&[n, n]).expect("rand a").cast(DType::BFloat16).expect("a→bf16");
    let mut b = Tensor::rand(&[n, n]).expect("rand b").cast(DType::BFloat16).expect("b→bf16");
    a.realize().expect("realize a");
    b.realize().expect("realize b");
    let mut c = Tensor::empty(&[n, n], DType::Float32);

    // Compile the svod-tk kernel ONCE (render+compile excluded from timing).
    let launch = crate::compile_kernel(
        "simple_matmul",
        M1_CFG.grid_dims(n),
        M1_CFG.threads(),
        &mut [&mut c],
        &[&a, &b],
        |ker| {
            build_matmul(ker, n);
            ker.finish(M1_CFG.n_accum)
        },
    )
    .expect("compile matmul");

    // Correctness: dispatch once, compare against the f32 ground-truth matmul of
    // the same bf16-rounded operands (max abs err < 5e-2 for bf16 inputs).
    unsafe { launch.dispatch(true).expect("matmul correctness dispatch") };
    let got = c.as_vec::<f32>().expect("read c");
    let af = a.cast(DType::Float32).expect("a→f32");
    let bf = b.cast(DType::Float32).expect("b→f32");
    let mut refc = af.matmul(&bf).expect("reference matmul");
    refc.realize().expect("realize reference");
    let expected = refc.as_vec::<f32>().expect("read reference");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    let correct = max_abs < 5e-2;

    // svod-tk timing — GPU device time.
    let tk = time_launch(&launch, 5, iters);
    let tk_gpu_us = tk.gpu_med();
    let flops = 2.0 * (n as f64).powi(3);
    let tflops = flops / (tk_gpu_us / 1e6) / 1e12;

    // Reference timing: svod's own bf16→f32 GEMM (bf16 inputs, f32 accumulate)
    // — the matmul a user would write for this kernel. Fresh plan; `prepare`
    // wraps the output in CONTIGUOUS so every replay recomputes on the GPU.
    let mut reft = a.matmul_with().other(&b).dtype(DType::Float32).call().expect("reference matmul (timing)");
    let ref_plan = reft.prepare().expect("prepare reference plan");
    let rf = time_plan(&ref_plan, 5, iters);
    let ref_gpu_us = rf.gpu_med();

    Row {
        n,
        correct,
        max_abs,
        tk_gpu_us,
        tk_wall_us: tk.wall_med(),
        tflops,
        pct_peak: tflops / MI300X_BF16_PEAK_TFLOPS * 100.0,
        ref_gpu_us,
        ref_wall_us: rf.wall_med(),
        speedup: if tk_gpu_us > 0.0 { ref_gpu_us / tk_gpu_us } else { 0.0 },
    }
}

fn print_matmul_table(rows: &[Row]) {
    println!("\n=== svod-tk MATMUL (bf16→f32) on gfx942 / MI300X — GPU device time ===");
    println!(
        "FLOPs = 2·N³ (TFLOPS from GPU time);  peak bf16 matrix = {MI300X_BF16_PEAK_TFLOPS} TFLOP/s (MI300X dense)"
    );
    println!(
        "{:>6}  {:>10}  {:>10}  {:>9}  {:>8}  {:>11}  {:>11}  {:>10}  {:>8}",
        "N", "tk gpu µs", "tk wall µs", "TFLOPS", "% peak", "ref gpu µs", "ref wall µs", "tk vs ref", "correct"
    );
    for r in rows {
        println!(
            "{:>6}  {:>10.2}  {:>10.2}  {:>9.2}  {:>7.2}%  {:>11.2}  {:>11.2}  {:>9.2}x  {:>8}  (max|err|={:.2e})",
            r.n,
            r.tk_gpu_us,
            r.tk_wall_us,
            r.tflops,
            r.pct_peak,
            r.ref_gpu_us,
            r.ref_wall_us,
            r.speedup,
            if r.correct { "OK" } else { "INCORRECT" },
            r.max_abs,
        );
    }
}

// ── flash-attention ─────────────────────────────────────────────────────────

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bench_fa_amd -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_fa_amd() {
    // B=1, D=64, H=H_KV=2 (svod has no GQA); sweep sequence length.
    let (b, h, d) = (1usize, 2usize, 64usize);
    let sizes: &[(usize, usize)] = &[(128, 50), (512, 50), (1024, 40), (2048, 20)];
    let mut rows = Vec::new();
    for &(n, iters) in sizes {
        match std::panic::catch_unwind(AssertUnwindSafe(|| run_fa(b, n, h, d, iters))) {
            Ok(row) => rows.push(row),
            Err(_) => println!("fa N={n}: SKIPPED (panic — likely OOM or compile/dispatch failure)"),
        }
    }
    print_fa_table(&rows, b, h, d);
    fa_block_skip_ab();
}

fn run_fa(b: usize, n: usize, h: usize, d: usize, iters: usize) -> Row {
    let h_kv = h;
    let grid = [h as i64, (n / 16) as i64, b as i64];

    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::BFloat16).expect("→bf16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());
    let mut o = Tensor::empty(&[b, n, h, d], DType::BFloat16);

    // Compile the FA kernel ONCE.
    let launch = crate::compile_kernel("fa", grid, 64, &mut [&mut o], &[&q, &k, &v], |ker| {
        build_fa(ker, b, n, h, h_kv, d);
        ker.finish(1)
    })
    .expect("compile fa");

    // Correctness vs causal SDPA over identical bf16 operands ([B,N,H,D] →
    // [B,H,N,D], SDPA, back). bf16 tolerance ~3e-2.
    unsafe { launch.dispatch(true).expect("fa correctness dispatch") };
    let mut of = o.cast(DType::Float32).expect("o→f32");
    of.realize().expect("realize o→f32");
    let got = of.as_vec::<f32>().expect("read o");

    let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("permute");
    let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
    let ref_bhnd = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
    let mut refp = ref_bhnd.try_permute(&[0, 2, 1, 3]).expect("permute back");
    refp.realize().expect("realize reference");
    let expected = refp.as_vec::<f32>().expect("read reference");
    let max_abs = got.iter().zip(&expected).map(|(g, e)| (g - e).abs()).fold(0.0f32, f32::max);
    let correct = max_abs < 3e-2;

    // svod-tk timing — GPU device time.
    let tk = time_launch(&launch, 5, iters);
    let tk_gpu_us = tk.gpu_med();
    // Causal useful FLOPs: non-causal QKᵀ+A·V ≈ 4·B·H·N²·D; causal ≈ half.
    let flops = 2.0 * (b * h * d) as f64 * (n as f64).powi(2);
    let tflops = flops / (tk_gpu_us / 1e6) / 1e12;

    // Reference SDPA plan (fresh, recomputes on each replay).
    let (qp2, kp2, vp2) = (perm(&q), perm(&k), perm(&v));
    let ref2 = qp2.scaled_dot_product_attention().key(&kp2).value(&vp2).is_causal(true).call().expect("sdpa timing");
    let mut ref2 = ref2.try_permute(&[0, 2, 1, 3]).expect("permute back (timing)");
    let ref_plan = ref2.prepare().expect("prepare reference plan");
    let rf = time_plan(&ref_plan, 5, iters);
    let ref_gpu_us = rf.gpu_med();

    Row {
        n,
        correct,
        max_abs,
        tk_gpu_us,
        tk_wall_us: tk.wall_med(),
        tflops,
        pct_peak: tflops / MI300X_BF16_PEAK_TFLOPS * 100.0,
        ref_gpu_us,
        ref_wall_us: rf.wall_med(),
        speedup: if tk_gpu_us > 0.0 { ref_gpu_us / tk_gpu_us } else { 0.0 },
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --release --lib bench_fa_graph_amd -- --ignored --nocapture`
///
/// **Benchmark-as-normal-code:** the graph-native `flash_attention` (a `custom_kernel`
/// / `Op::Call` node) timed through the SAME `prepare()` → `time_plan` path as the
/// SDPA reference — no bespoke `CompiledLaunch` harness. Confirms a graph-node tk
/// kernel composes + profiles like any tensor op, and reports it vs SDPA on GPU
/// device time. `sdpa/fa = sdpa_gpu / fa_gpu` (>1 ⇒ tk faster).
#[test]
#[ignore]
fn bench_fa_graph_amd() {
    let d = 64usize;
    let shapes: &[(&str, usize, usize)] = &[("B=1 H=2", 1, 2), ("B=1 H=16 (inference)", 1, 16), ("B=8 H=16", 8, 16)];
    let ns: &[(usize, usize)] = &[(512, 40), (1024, 25), (2048, 15)];
    println!("\n=== svod-tk GRAPH FA (custom_kernel node, prepare+time_plan) vs SDPA — GPU device time, D={d} ===");
    println!("causal useful FLOPs = 2·B·H·N²·D;  prepared like any tensor op.  sdpa/fa = sdpa_gpu / fa_gpu");
    for &(label, b, h) in shapes {
        println!("\n-- {label} --");
        println!("{:>6} | {:>9} {:>9} | {:>9} {:>8}", "N", "fa µs", "fa TF", "sdpa µs", "sdpa/fa");
        for &(n, iters) in ns {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let flops = 2.0 * (b * h * d) as f64 * (n as f64).powi(2);
                let mk = || {
                    let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::BFloat16).expect("→bf16");
                    t.realize().expect("realize");
                    t
                };
                let (q, k, v) = (mk(), mk(), mk());
                // Graph FA: lazy custom_kernel Tensor → prepare → ExecutionPlan.
                let mut fa = crate::kernels::fa::flash_attention(&q, &k, &v).expect("graph fa");
                let fa_plan = fa.prepare().expect("prepare fa");
                let fa_t = time_plan(&fa_plan, 5, iters);
                // SDPA reference plan (same prepare/time_plan path).
                let perm = |t: &Tensor| t.cast(DType::Float32).expect("→f32").try_permute(&[0, 2, 1, 3]).expect("perm");
                let (qp, kp, vp) = (perm(&q), perm(&k), perm(&v));
                let refb = qp.scaled_dot_product_attention().key(&kp).value(&vp).is_causal(true).call().expect("sdpa");
                let mut refp = refb.try_permute(&[0, 2, 1, 3]).expect("perm back");
                let ref_plan = refp.prepare().expect("prepare ref");
                let ref_t = time_plan(&ref_plan, 5, iters);
                (flops, fa_t.gpu_med(), ref_t.gpu_med())
            }));
            match r {
                Ok((flops, fau, refu)) => {
                    let tf = |us: f64| if us > 0.0 { flops / (us / 1e6) / 1e12 } else { 0.0 };
                    let sp = if fau > 0.0 && refu > 0.0 { refu / fau } else { 0.0 };
                    println!("{n:>6} | {fau:9.2} {:9.2} | {refu:9.2} {sp:7.2}x", tf(fau));
                }
                Err(_) => println!("{n:>6} | SKIPPED (compile/dispatch failure)"),
            }
        }
    }
}

fn print_fa_table(rows: &[Row], b: usize, h: usize, d: usize) {
    println!("\n=== svod-tk FLASH-ATTENTION fwd (causal, bf16) on gfx942 / MI300X — GPU device time ===");
    println!(
        "B={b} H=H_KV={h} D={d};  causal useful FLOPs = 2·B·H·N²·D \
         (non-causal 4·B·H·N²·D, ≈half masked)"
    );
    println!(
        "{:>6}  {:>10}  {:>10}  {:>9}  {:>11}  {:>11}  {:>10}  {:>8}",
        "N", "tk gpu µs", "tk wall µs", "TFLOPS", "ref gpu µs", "ref wall µs", "tk vs ref", "correct"
    );
    for r in rows {
        println!(
            "{:>6}  {:>10.2}  {:>10.2}  {:>9.2}  {:>11.2}  {:>11.2}  {:>9.2}x  {:>8}  (max|err|={:.2e})",
            r.n,
            r.tk_gpu_us,
            r.tk_wall_us,
            r.tflops,
            r.ref_gpu_us,
            r.ref_wall_us,
            r.speedup,
            if r.correct { "OK" } else { "INCORRECT" },
            r.max_abs,
        );
    }
}

// ── single-warp vs multi-wave (8-warp) FA — GPU device time ──────────────────

/// Which FA kernel a launch builds.
#[derive(Clone, Copy)]
enum FaKind {
    /// Single-warp (`block 64`, grid dim1 `n/16`).
    Single,
    /// Multi-wave 8-warp (`block 512`, grid dim1 `n/16/8`).
    Mw,
    /// Double-buffered multi-wave (`pipelined` toggles stage 1 vs 2), with the
    /// FA-4 per-warp tile heights `(q_blk, kv_blk)` (grid dim1 `n/q_blk/8`).
    MwDb { pipelined: bool, q_blk: usize, kv_blk: usize },
    /// Rolled double-buffered multi-wave (one FaScratch set), per-warp tiles
    /// `(q_blk, kv_blk)`; `unroll` = fully-flat (unrolled) compute body.
    Rdb { q_blk: usize, kv_blk: usize, unroll: bool },
}

/// Compile an FA launch of the requested [`FaKind`] over fresh random bf16
/// operands. Returns the launch + its output tensor (held so its buffer outlives
/// the launch).
fn compile_fa_launch(b: usize, n: usize, h: usize, d: usize, kind: FaKind) -> (CompiledLaunch, Tensor) {
    let h_kv = h;
    // Multi-wave Q-tile height: BLK(16) except for the FA-4 `MwDb` tile configs.
    let q_blk = match kind {
        FaKind::MwDb { q_blk, .. } | FaKind::Rdb { q_blk, .. } => q_blk,
        _ => 16,
    };
    let (block, gd1) = match kind {
        FaKind::Single => (64, (n / 16) as i64),
        _ => (8 * 64, (n / q_blk / 8) as i64),
    };
    let grid = [h as i64, gd1, b as i64];
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::BFloat16).expect("→bf16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());
    let mut o = Tensor::empty(&[b, n, h, d], DType::BFloat16);
    let name = match kind {
        FaKind::Single => "fa",
        FaKind::Mw => "fa_mw",
        FaKind::MwDb { .. } => "fa_mw_db",
        FaKind::Rdb { .. } => "fa_mw_rdb",
    };
    let launch = crate::compile_kernel(name, grid, block, &mut [&mut o], &[&q, &k, &v], |ker| {
        match kind {
            FaKind::Single => build_fa(ker, b, n, h, h_kv, d),
            FaKind::Mw => build_fa_mw(ker, b, n, h, h_kv, d),
            FaKind::MwDb { pipelined, q_blk, kv_blk } => {
                build_fa_mw_db(ker, b, n, h, h_kv, d, FaConfig { q_blk, kv_blk, pipelined, ..Default::default() })
            }
            FaKind::Rdb { q_blk, kv_blk, unroll } => build_fa_mw_rdb(
                ker,
                b,
                n,
                h,
                h_kv,
                d,
                FaConfig { q_blk, kv_blk, unroll, ..Default::default() },
                q.uop().dtype(),
                false,
            ),
        }
        ker.finish(1)
    })
    .expect("compile fa");
    (launch, o)
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bench_fa_rdb_amd -- --ignored --nocapture`
///
/// FA unroll-by-2 double buffer (`db`, two FaScratch sets) vs the **rolled**
/// double buffer (`rdb`, one FaScratch set) on GPU device time, INTERLEAVED per
/// sample-pair (VF clock variance). `rdb/db = db_gpu / rdb_gpu` (>1 ⇒ rdb faster).
#[test]
#[ignore]
fn bench_fa_rdb_amd() {
    let d = 64usize;
    let shapes: &[(&str, usize, usize)] =
        &[("B=1 H=2  (tiny)", 1, 2), ("B=1 H=16 (inference)", 1, 16), ("B=8 H=16 (occ-bound)", 8, 16)];
    let ns: &[(usize, usize)] = &[(512, 60), (1024, 40), (2048, 25)];
    println!("\n=== svod-tk FA unroll-by-2 (db) vs rolled double-buffer (rdb) — GPU device time (TFLOPS), D={d} ===");
    println!(
        "causal useful FLOPs = 2·B·H·N²·D;  interleaved db,rdb per sample.  rdb/db = db_gpu/rdb_gpu (>1 ⇒ rdb faster)"
    );
    for &(label, b, h) in shapes {
        println!("\n-- {label} --");
        println!("{:>6} | {:>9} {:>9} | {:>9} {:>9} {:>7}", "N", "db µs", "db TF", "rdb µs", "rdb TF", "rdb/db");
        for &(n, iters) in ns {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let flops = 2.0 * b as f64 * h as f64 * (n as f64).powi(2) * d as f64;
                let (db, _od) = compile_fa_launch(b, n, h, d, FaKind::MwDb { pipelined: true, q_blk: 16, kv_blk: 16 });
                let (rdb, _or) = compile_fa_launch(b, n, h, d, FaKind::Rdb { q_blk: 16, kv_blk: 16, unroll: false });
                for _ in 0..5 {
                    // SAFETY: output buffers held by `_od`/`_or` for the launches' lifetime.
                    unsafe {
                        db.dispatch(true).expect("warm db");
                        rdb.dispatch(true).expect("warm rdb");
                    }
                }
                let (mut gd, mut gr) = (Vec::with_capacity(iters), Vec::with_capacity(iters));
                for _ in 0..iters {
                    if let Some(ns) = db.dispatch_gpu_ns().expect("db gpu") {
                        gd.push(ns as f64 / 1e3);
                    }
                    if let Some(ns) = rdb.dispatch_gpu_ns().expect("rdb gpu") {
                        gr.push(ns as f64 / 1e3);
                    }
                }
                (flops, median(&gd), median(&gr))
            }));
            match r {
                Ok((flops, dbu, rdbu)) => {
                    let tf = |us: f64| if us > 0.0 { flops / (us / 1e6) / 1e12 } else { 0.0 };
                    let sp = if rdbu > 0.0 && dbu > 0.0 { dbu / rdbu } else { 0.0 };
                    println!("{n:>6} | {dbu:9.2} {:9.2} | {rdbu:9.2} {:9.2} {sp:6.2}x", tf(dbu), tf(rdbu));
                }
                Err(_) => println!("{n:>6} | SKIPPED (compile/dispatch failure)"),
            }
        }
    }
}

/// `SVOD_DEVICE=AMD:0 cargo test -p svod-tk --lib bench_fa_mw_amd -- --ignored --nocapture`
///
/// Single-warp vs multi-wave (8-warp) FA on **GPU device time** at production
/// single-batch shapes (B=1,H=2) and an occupancy-bound batched shape
/// (B=8,H=16), across N where `n/16` is a multiple of 8. Both run the same
/// causal useful FLOPs, so `speedup = sw_gpu / mw_gpu`.
#[test]
#[ignore]
fn bench_fa_mw_amd() {
    let d = 64usize;
    let shapes: &[(&str, usize, usize)] =
        &[("B=1 H=2  (tiny)", 1, 2), ("B=1 H=16 (inference)", 1, 16), ("B=8 H=16 (occ-bound)", 8, 16)];
    let ns: &[(usize, usize)] = &[(512, 40), (1024, 30), (2048, 15)];

    println!("\n=== svod-tk FA single-warp vs multi-wave vs double-buffer — GPU device time (TFLOPS), D={d} ===");
    println!(
        "causal useful FLOPs = 2·B·H·N²·D;  db=double-buffered unroll-by-2 (stage ii).  \
         dbp_vs_sw = sw_gpu/dbp_gpu (>1 ⇒ pipelined db beats single-warp)"
    );
    for &(label, b, h) in shapes {
        println!("\n-- {label} --");
        println!(
            "{:>6} | {:>8} {:>8} {:>9} {:>9} | {:>9} {:>9} {:>9} {:>9}",
            "N", "sw TF", "mw TF", "db_naiv", "db_pipe", "mw/sw", "dbn/sw", "dbp/sw", "dbp/mw"
        );
        for &(n, iters) in ns {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let (sw, _osw) = compile_fa_launch(b, n, h, d, FaKind::Single);
                let (mw, _omw) = compile_fa_launch(b, n, h, d, FaKind::Mw);
                let (dbn, _odbn) =
                    compile_fa_launch(b, n, h, d, FaKind::MwDb { pipelined: false, q_blk: 16, kv_blk: 16 });
                let (dbp, _odbp) =
                    compile_fa_launch(b, n, h, d, FaKind::MwDb { pipelined: true, q_blk: 16, kv_blk: 16 });
                let sw_gpu = time_launch(&sw, 5, iters).gpu_med();
                let mw_gpu = time_launch(&mw, 5, iters).gpu_med();
                let dbn_gpu = time_launch(&dbn, 5, iters).gpu_med();
                let dbp_gpu = time_launch(&dbp, 5, iters).gpu_med();
                (sw_gpu, mw_gpu, dbn_gpu, dbp_gpu)
            }));
            match r {
                Ok((sw_gpu, mw_gpu, dbn_gpu, dbp_gpu)) => {
                    let flops = 2.0 * (b * h * d) as f64 * (n as f64).powi(2);
                    let tf = |us: f64| flops / (us / 1e6) / 1e12;
                    let ratio = |fast: f64| if fast > 0.0 { sw_gpu / fast } else { 0.0 };
                    let dbp_vs_mw = if dbp_gpu > 0.0 { mw_gpu / dbp_gpu } else { 0.0 };
                    println!(
                        "{n:>6} | {:>8.2} {:>8.2} {:>9.2} {:>9.2} | {:>8.2}x {:>8.2}x {:>8.2}x {:>8.2}x",
                        tf(sw_gpu),
                        tf(mw_gpu),
                        tf(dbn_gpu),
                        tf(dbp_gpu),
                        ratio(mw_gpu),
                        ratio(dbn_gpu),
                        ratio(dbp_gpu),
                        dbp_vs_mw,
                    );
                }
                Err(_) => println!("{n:>6}  SKIPPED (panic — likely OOM or dispatch failure)"),
            }
        }
    }
}

// ── causal block-skip A/B (GPU device time) ──────────────────────────────────

/// Compile an FA launch at the given shape with `causal_skip` toggling the KV
/// loop bound (`q_seq+1` dynamic vs. full `n/BLK`). Returns the launch and its
/// bound output tensor (held so its buffer outlives the launch).
fn compile_fa_kv_launch(b: usize, n: usize, h: usize, d: usize, causal_skip: bool) -> (CompiledLaunch, Tensor) {
    let h_kv = h;
    let grid = [h as i64, (n / 16) as i64, b as i64];
    let mk = || {
        let mut t = Tensor::randn(&[b, n, h, d]).expect("randn").cast(DType::BFloat16).expect("→bf16");
        t.realize().expect("realize");
        t
    };
    let (q, k, v) = (mk(), mk(), mk());
    let mut o = Tensor::empty(&[b, n, h, d], DType::BFloat16);
    let launch = crate::compile_kernel("fa_ab", grid, 64, &mut [&mut o], &[&q, &k, &v], |ker| {
        build_fa_kv(ker, b, n, h, h_kv, d, causal_skip);
        ker.finish(1)
    })
    .expect("compile fa A/B");
    (launch, o)
}

/// A/B the causal block-skip (skip vs full KV loop) on **GPU device time**, at a
/// small shape and an occupancy-bound shape across the N sweep. The ratio
/// `skip_gpu / full_gpu` (<1 ⇒ block-skip cuts on-device kernel time) shows
/// where the dynamic `q_seq+1` bound helps: when workgroups ≫ resident capacity
/// (occupancy-bound) the saved blocks free SIMDs for waiting workgroups, so the
/// effect shows in device time; at a tiny shape the GPU is under-subscribed and
/// the skip mostly trims a latency-bound tail.
fn fa_block_skip_ab() {
    let d = 64usize;
    // (label, B, H): #workgroups = B·H·(N/16). small = under-subscribed;
    // occ-bound = thousands of single-wave workgroups ≫ ~CU·waves resident.
    let shapes: &[(&str, usize, usize)] = &[("small  B=1 H=2", 1, 2), ("occ-bound B=8 H=16", 8, 16)];
    let ns: &[(usize, usize)] = &[(128, 50), (512, 30), (1024, 15), (2048, 8)];

    println!("\n=== svod-tk FA causal block-skip A/B (GPU device time, D={d}) ===");
    println!("ratio = skip_gpu / full_gpu  (<1 ⇒ block-skip cuts on-device kernel time)");
    for &(label, b, h) in shapes {
        println!("\n-- {label}  (#workgroups = B·H·N/16) --");
        println!("{:>6}  {:>11}  {:>11}  {:>8}  {:>10}", "N", "skip gpu µs", "full gpu µs", "ratio", "wgroups");
        for &(n, iters) in ns {
            let r = std::panic::catch_unwind(AssertUnwindSafe(|| {
                let (skip, _o_skip) = compile_fa_kv_launch(b, n, h, d, true);
                let (full, _o_full) = compile_fa_kv_launch(b, n, h, d, false);
                let skip_gpu = time_launch(&skip, 5, iters).gpu_med();
                let full_gpu = time_launch(&full, 5, iters).gpu_med();
                (skip_gpu, full_gpu, b * h * (n / 16))
            }));
            match r {
                Ok((skip_gpu, full_gpu, wg)) => {
                    let ratio = if full_gpu > 0.0 { skip_gpu / full_gpu } else { 0.0 };
                    println!("{n:>6}  {skip_gpu:>11.2}  {full_gpu:>11.2}  {ratio:>7.3}x  {wg:>10}");
                }
                Err(_) => println!("{n:>6}  SKIPPED (panic — likely OOM or dispatch failure)"),
            }
        }
    }
}
