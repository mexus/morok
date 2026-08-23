//! Beam search auto-tuning for kernel optimization.
//!
//! Implements a beam search algorithm that explores the optimization space
//! to find high-performance kernel configurations. This is slower than
//! heuristic-based optimization but can achieve ML-quality performance.
//!
//! # Algorithm
//!
//! 1. Start with base scheduler
//! 2. Generate all valid actions (OptOps applications)
//! 3. Compile and time each candidate
//! 4. Keep top K (beam width) by timing
//! 5. Repeat until no improvement or timeout
//!
//! # Caching
//!
//! Results are cached to disk using sled. The cache key is a hash of
//! (ast_hash, beam_width, device_name). Caching can be disabled via
//! the IGNORE_BEAM_CACHE environment variable.

use std::sync::Arc;
use std::time::Duration;

use once_cell::sync::Lazy;

use svod_ir::{AxisType, ConstValue, Op, UOp};

use super::Scheduler;
use super::config::BeamConfig;
use super::error::*;
use super::opts::apply_opt;
use super::types::{Opt, OptArg, OptOps};

/// Minimum measurable improvement before BEAM stops iterating.
///
/// Default 10 ns. With kernels timing in hundreds of µs this floor
/// effectively never fires; it exists to stop beam when improvements
/// drop into measurement noise. Override via `BEAM_MIN_PROGRESS`
/// (nanoseconds; `0` to disable).
fn beam_min_progress() -> Duration {
    static CACHED: Lazy<Duration> = Lazy::new(|| {
        let nanos: u64 = std::env::var("BEAM_MIN_PROGRESS").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
        Duration::from_nanos(nanos)
    });
    *CACHED
}

// ============================================================================
// ACTION SPACE
// ============================================================================

/// Thread-count amounts considered by beam search.
///
/// Static set `[2,3,4,5,8,12,16,24,32,64]` filtered by `max_threads`. We don't
/// pre-filter by divisor patterns — `apply_thread` enforces divisibility against
/// the chosen axis at apply time, and the true divisibility depends on
/// post-action shape.
fn thread_action_amounts(max_threads: usize) -> Vec<usize> {
    const AMOUNTS: [usize; 10] = [2, 3, 4, 5, 8, 12, 16, 24, 32, 64];
    AMOUNTS.iter().copied().filter(|&t| t <= max_threads).collect()
}

/// Pre-computed action space for beam search (~500 actions).
pub static BEAM_ACTIONS: Lazy<Vec<Opt>> = Lazy::new(|| {
    let mut actions = Vec::with_capacity(600);

    // UPCAST: axes 0-7, amounts [0, 2, 3, 4, 5, 7]
    // amount=0 means "full size" - handled specially in apply
    for axis in 0..8 {
        for &amt in &[0, 2, 3, 4, 5, 7] {
            actions.push(Opt::upcast(axis, amt));
        }
    }

    // UNROLL: axes 0-4, amounts [0, 4, 7]
    for axis in 0..5 {
        for &amt in &[0, 4, 7] {
            actions.push(Opt::unroll(axis, amt));
        }
    }

    // LOCAL: axes 0-5, amounts [2, 3, 4, 8, 13, 16, 29]
    for axis in 0..6 {
        for &amt in &[2, 3, 4, 8, 13, 16, 29] {
            actions.push(Opt::local(axis, amt));
        }
    }
    // Hand-tuned LOCAL extras outside the grid.
    actions.push(Opt::local(0, 32));
    actions.push(Opt::local(6, 2));

    // GROUPTOP: axes 0-2, amounts [13, 16, 28, 29, 32, 49, 64, 256]
    for axis in 0..3 {
        for &amt in &[13, 16, 28, 29, 32, 49, 64, 256] {
            actions.push(Opt::grouptop(axis, amt));
        }
    }

    // GROUP: axes 0-2, amounts [0, 4, 8, 16]
    for axis in 0..3 {
        for &amt in &[0, 4, 8, 16] {
            actions.push(Opt::group(axis, amt));
        }
    }

    // TC: tensor cores. 1 default-axis action + 9 axis variants = 10 actions.
    // Survivors after post-compile dedup are unchanged compared to a wider
    // brute-force enumeration because `seen_libs` collapses duplicate kernels.
    const TC_AXIS_CHOICES: usize = 9;
    const TC_OPT_DEFAULT: usize = 0;
    const TC_OPT_AXIS: usize = 2;
    actions.push(Opt::tc(Some(0), -1, TC_OPT_DEFAULT, 1));
    for axis_choice in 0..TC_AXIS_CHOICES {
        actions.push(Opt::tc(Some(axis_choice), -1, TC_OPT_AXIS, 1));
    }

    // SWAP: axis pairs
    for a0 in 0..5 {
        for a1 in (a0 + 1)..5 {
            actions.push(Opt::swap(a0, a1));
        }
    }

    // THREAD: CPU parallelization with smart divisor selection
    // Include thread counts that divide common tensor sizes (64, 128, 256, 512, 1024)
    let max_threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let thread_amounts = thread_action_amounts(max_threads);
    for axis in 0..3 {
        for &amt in &thread_amounts {
            actions.push(Opt::thread(axis, amt));
        }
    }

    // NOLOCALS — only when explicitly enabled via `SVOD_NOLOCALS`.
    if std::env::var("SVOD_NOLOCALS").is_ok() {
        actions.push(Opt::nolocals());
    }

    actions
});

// ============================================================================
// ACTION GENERATION & FILTERING
// ============================================================================

/// `(op, axis)` pairs that have an `arg=0` (full-axis) variant in
/// [`BEAM_ACTIONS`]. Used by [`passes_prefilter`] to dedup the explicit
/// `arg=axis_size` variants whenever the `arg=0` variant covers the same case.
static FULL_AXIS_VARIANTS: Lazy<std::collections::HashSet<(OptOps, usize)>> = Lazy::new(|| {
    BEAM_ACTIONS
        .iter()
        .filter_map(|opt| {
            let axis = opt.axis?;
            match opt.arg {
                OptArg::Int(0) => Some((opt.op, axis)),
                _ => None,
            }
        })
        .collect()
});

/// Pre-apply filter with two early-rejects:
///
/// 1. The action's logical axis can't be resolved (would always fail in
///    `apply_opt`). Skips the candidate clone+apply roundtrip.
/// 2. The action's `arg` already equals the axis's full size AND an `arg=0`
///    variant exists in `BEAM_ACTIONS` for the same `(op, axis)`. The two
///    actions produce the same kernel post-codegen, so we drop the explicit
///    one to halve dedup work.
fn passes_prefilter(scheduler: &Scheduler, action: &Opt) -> bool {
    // TC and NOLOCALS skip the filter — they have no logical axis.
    if action.op == OptOps::TC || action.axis.is_none() {
        return true;
    }
    // Resolve the logical axis to a real axis. Failure → action would fail
    // at apply time; skip now.
    let real_axis = match scheduler.real_axis(action.op, action.axis) {
        Ok(a) if a >= 0 => a as usize,
        _ => return false,
    };
    if real_axis >= scheduler.shape_len() {
        return false;
    }
    // Dedup: skip if `arg == full_shape[real_axis]` and an `arg=0` variant
    // covers the same case. Only `OptArg::Int` carries a comparable arg.
    if let OptArg::Int(arg) = action.arg
        && arg > 0
        && let Some(&size) = scheduler.full_shape().get(real_axis)
        && size as usize == arg
        && let Some(axis) = action.axis
        && FULL_AXIS_VARIANTS.contains(&(action.op, axis))
    {
        return false;
    }
    true
}

/// `BEAM_DEBUG=1` toggles eprintln! tracing of action survival across
/// the prefilter/apply/limit/time stages. Cheap when disabled (one env-cached
/// bool check per call); useful for diagnosing why an action class never wins.
fn beam_debug_enabled() -> bool {
    static CACHED: Lazy<bool> = Lazy::new(|| std::env::var("BEAM_DEBUG").is_ok());
    *CACHED
}

/// Per-stage candidate counts, broken out by [`OptOps`] kind. Aggregated by
/// [`generate_actions`] when [`beam_debug_enabled`] is on.
#[derive(Default, Debug)]
struct ActionStageCounts {
    attempted: std::collections::HashMap<OptOps, usize>,
    prefilter_dropped: std::collections::HashMap<OptOps, usize>,
    apply_dropped: std::collections::HashMap<OptOps, usize>,
    limit_dropped: std::collections::HashMap<OptOps, usize>,
    survived: std::collections::HashMap<OptOps, usize>,
}

/// Generate all valid next-states from the current scheduler.
///
/// Applies each action from `BEAM_ACTIONS` and filters to those that:
/// 1. Pass the cheap [`passes_prefilter`] gate (axis resolves, no arg-eq-size dup)
/// 2. Apply successfully (divisibility, bounds, etc.)
/// 3. Pass limit checks (upcast size, local size, UOp count)
fn generate_actions(scheduler: &Scheduler, config: &BeamConfig) -> Vec<Scheduler> {
    let debug = beam_debug_enabled();
    let mut counts = ActionStageCounts::default();
    let mut out = Vec::with_capacity(BEAM_ACTIONS.len());

    for action in BEAM_ACTIONS.iter() {
        if debug {
            *counts.attempted.entry(action.op).or_insert(0) += 1;
        }
        if !passes_prefilter(scheduler, action) {
            if debug {
                *counts.prefilter_dropped.entry(action.op).or_insert(0) += 1;
            }
            continue;
        }
        let mut candidate = scheduler.clone();
        match apply_opt(&mut candidate, action, true) {
            Ok(()) => {
                if !validate_limits(&candidate, config) {
                    if debug {
                        *counts.limit_dropped.entry(action.op).or_insert(0) += 1;
                    }
                    continue;
                }
                if debug {
                    *counts.survived.entry(action.op).or_insert(0) += 1;
                }
                out.push(candidate);
            }
            Err(_) => {
                if debug {
                    *counts.apply_dropped.entry(action.op).or_insert(0) += 1;
                }
            }
        }
    }

    if debug {
        let ops_in_order = [
            OptOps::TC,
            OptOps::UPCAST,
            OptOps::UNROLL,
            OptOps::LOCAL,
            OptOps::GROUP,
            OptOps::GROUPTOP,
            OptOps::THREAD,
            OptOps::SWAP,
            OptOps::PADTO,
            OptOps::NOLOCALS,
        ];
        eprintln!("[beam] generate_actions: {} survivors", out.len());
        // Print every action class, not only the ones with non-zero
        // `attempted`. A class with `attempted=0` means the BEAM_ACTIONS
        // static doesn't even contain that variant — useful for catching
        // missing actions vs. catastrophically high apply/limit drops.
        for op in ops_in_order {
            let a = counts.attempted.get(&op).copied().unwrap_or(0);
            let pf = counts.prefilter_dropped.get(&op).copied().unwrap_or(0);
            let ap = counts.apply_dropped.get(&op).copied().unwrap_or(0);
            let lim = counts.limit_dropped.get(&op).copied().unwrap_or(0);
            let s = counts.survived.get(&op).copied().unwrap_or(0);
            eprintln!("  {op:?}: attempted={a:3} prefilter={pf:3} apply_err={ap:3} limit={lim:3} survived={s:3}");
        }
    }

    out
}

/// Validate that a scheduler state is within configured limits.
///
/// Per-candidate filter: reject if `(up_axes_prod / tc_up) > max_upcast`
/// or `local_axes_prod > max_local`, where `tc_up = prod(tc.dims) /
/// tc.threads` if a TC is active else 1.
///
/// The `tc_up` divisor accounts for the TC tile's contribution to the
/// total UPCAST/UNROLL product — without it, applying TC immediately
/// saturates `max_upcast` (e.g. APPLE_AMX `prod((16,16,1))/1 = 256`),
/// blocking any post-TC UPCAST composition.
fn validate_limits(scheduler: &Scheduler, config: &BeamConfig) -> bool {
    let upcast_sz = product_of_axes(scheduler, &[AxisType::Upcast, AxisType::Unroll]);
    let local_sz = product_of_axes(scheduler, &[AxisType::Local, AxisType::Warp, AxisType::GroupReduce]);
    let tc_up = active_tc_upcast(scheduler);

    upcast_sz / tc_up <= config.max_upcast && local_sz <= config.max_local
}

/// Return `prod(tc.dims) / tc.threads` for the active TC, or 1 if none.
///
/// Uses `scheduler.selected_tc_index` (recorded by `apply_axis_choice_impl`)
/// rather than guessing from the renderer's TC list. For multi-TC renderers
/// (e.g. SM89 with f16+bf16+tf32 variants) this is the only correct
/// accounting.
fn active_tc_upcast(scheduler: &Scheduler) -> usize {
    let Some(idx) = scheduler.selected_tc_index else {
        return 1;
    };
    scheduler
        .ren
        .tensor_cores
        .get(idx)
        .map(|tc| {
            let prod = tc.dims.0 * tc.dims.1 * tc.dims.2;
            prod / tc.threads.max(1)
        })
        .unwrap_or(1)
}

/// Calculate product of dimension sizes for given axis types.
fn product_of_axes(scheduler: &Scheduler, types: &[AxisType]) -> usize {
    scheduler
        .rngs()
        .iter()
        .filter_map(|rng| {
            if let Op::Range { axis_type, end, .. } = rng.op()
                && types.contains(axis_type)
                && let Op::Const(cv) = end.op()
                && let ConstValue::Int(sz) = cv.0
            {
                Some(sz as usize)
            } else {
                None
            }
        })
        .product::<usize>()
        .max(1)
}

// ============================================================================
// BEAM SEARCH ALGORITHM
// ============================================================================

/// Beam search result containing optimized scheduler and timing.
pub struct BeamResult {
    /// Optimized scheduler state.
    pub scheduler: Scheduler,
    /// Best timing achieved.
    pub timing: Duration,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Total candidates evaluated.
    pub candidates_evaluated: usize,
}

/// Metrics returned by the `compile_and_time` closure for each candidate.
///
/// Timing drives ranking; the IR hash drives `seen_libs` dedup; the compute-op
/// count drives the `least_compute_ops*1000` filter.
#[derive(Debug, Clone, Copy)]
pub struct CandidateMetrics {
    /// Best execution timing across the run loop (`min(tms)`).
    pub timing: Duration,
    /// Hash of the post-codegen IR — kernels that lower to the same IR are
    /// guaranteed to compile to the same object, so we skip duplicates.
    pub ir_hash: u64,
    /// Cheap upper bound on the kernel's compute work; used by the
    /// `least_compute_ops*1000` filter to discard degenerate candidates.
    pub compute_ops: u64,
}

/// Hash a UOp tree to a `u64` for `seen_libs` dedup.
///
/// Uses the pre-computed `content_hash` field on `UOp` (see
/// `ir/src/uop/hash_consing.rs`), which is the same structural hash the
/// hash-consing cache and `schedule_cache` rely on. O(1) — read the cached
/// field instead of re-walking the graph.
pub fn hash_post_codegen_ir(uop: &Arc<UOp>) -> u64 {
    uop.content_hash
}

// The symbolic compute-ops estimate is an AST-only walk, so it lives in `ir`
// (shared with the runtime profiler's roofline). Re-exported here to keep the
// `schedule` BEAM call sites and public surface unchanged.
pub use svod_ir::compute_ops_estimate;

/// Run beam search optimization.
///
/// # Arguments
///
/// * `scheduler` - Initial scheduler state
/// * `config` - Beam search configuration
/// * `compile_and_time` - Function to compile and time a scheduler state
///
/// # Returns
///
/// `BeamResult` containing the best scheduler found and performance metrics.
///
/// # Example
///
/// ```ignore
/// let config = BeamConfig::default();
/// let compile_and_time = |s: &Scheduler, early_stop: Option<Duration>| {
///     let ast = s.get_optimized_ast(None);
///     let kernel = compile_kernel(&ast)?;
///     let bench = benchmark_kernel(&kernel, ..., early_stop)?;
///     Some(CandidateMetrics { timing: bench.min, ir_hash: ..., compute_ops: ... })
/// };
///
/// let result = beam_search(scheduler, &config, compile_and_time)?;
/// println!("Best time: {:?}", result.timing);
/// ```
pub fn beam_search<F>(scheduler: Scheduler, config: &BeamConfig, compile_and_time: F) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    let mut iterations = 0;
    let mut candidates_evaluated = 0;

    // Initialize beam with `Duration::MAX` so the first iteration has no
    // incumbent to beat. Avoids one wasted compile+time per `beam_search`
    // invocation (also charged on cache replay through `OPT_CACHE`).
    let mut beam: Vec<(Scheduler, Duration)> = vec![(scheduler.clone(), Duration::MAX)];

    // `seen_libs` and `least_compute_ops` persist across the entire beam
    // search. Identity-keyed dedup carries across iterations, so a kernel
    // produced at iter N and re-produced (via a different opt order) at
    // iter N+1 only gets compiled+timed once.
    let mut seen_libs: std::collections::HashSet<u64> = std::collections::HashSet::new();
    let mut least_compute_ops: u64 = u64::MAX;

    // No total search budget; terminates on empty candidate set, empty timed
    // list, `min_progress` floor, or sub-noise gain. Per-candidate compile
    // budgets live separately in `compile_and_time`'s thread+timeout wrapper.
    loop {
        iterations += 1;

        // 1. EXPAND: Generate all valid next states from current beam (sequential)
        // Note: Scheduler is not Sync due to OnceCell caches, so expansion is sequential
        let candidates: Vec<Scheduler> = beam.iter().flat_map(|(s, _)| generate_actions(s, config)).collect();

        if candidates.is_empty() {
            break;
        }

        // Reject any candidate whose first run already exceeds 3× the current beam best.
        let beam_best = beam.first().map(|(_, t)| *t);
        let early_stop = beam_best.and_then(|t| t.checked_mul(3));

        // 2. COMPILE & TIME: Evaluate performance
        let mut timed: Vec<(Scheduler, Duration)> = Vec::new();
        for s in candidates {
            let Some(metrics) = compile_and_time(&s, early_stop) else { continue };

            if !seen_libs.insert(metrics.ir_hash) {
                continue;
            }
            least_compute_ops = least_compute_ops.min(metrics.compute_ops);
            if least_compute_ops.saturating_mul(1000) < metrics.compute_ops {
                continue;
            }

            timed.push((s, metrics.timing));
        }

        candidates_evaluated += timed.len();

        if timed.is_empty() {
            break;
        }

        if beam_debug_enabled() {
            // Bucket survivors by the *last* applied opt (the one this iteration
            // just stacked on). Useful for spotting "TC compiled but lost on
            // timing vs UPCAST" vs "TC never survived to timing at all".
            let mut by_op: std::collections::HashMap<OptOps, (usize, Duration)> = std::collections::HashMap::new();
            for (s, t) in &timed {
                if let Some(opt) = s.applied_opts.last() {
                    let entry = by_op.entry(opt.op).or_insert((0, Duration::MAX));
                    entry.0 += 1;
                    if *t < entry.1 {
                        entry.1 = *t;
                    }
                }
            }
            eprintln!("[beam iter {iterations}] timed survivors by last-op (count, best):");
            let ops_in_order = [
                OptOps::TC,
                OptOps::UPCAST,
                OptOps::UNROLL,
                OptOps::LOCAL,
                OptOps::GROUP,
                OptOps::GROUPTOP,
                OptOps::THREAD,
            ];
            for op in ops_in_order {
                if let Some((cnt, best)) = by_op.get(&op) {
                    eprintln!("  {op:?}: count={cnt:3} best={best:?}");
                }
            }
        }

        // 3. SORT: Sort by timing (best first)
        let mut sorted = timed;
        sorted.sort_by_key(|(_, t)| *t);

        // 4. CHECK TERMINATION — exit when the new best is already below
        //    the progress floor (fast-enough kernel) OR when the gain over
        //    the incumbent is sub-noise. Sub-noise gains don't justify a
        //    next compile round.
        let best_new = sorted[0].1;
        let best_old = beam.first().map(|(_, t)| *t).unwrap_or(Duration::MAX);
        let min_progress = beam_min_progress();
        let absolute_floor = best_new < min_progress;
        let no_real_gain = best_old.saturating_sub(best_new) < min_progress;

        if absolute_floor || no_real_gain {
            // When exiting AND we did improve, pin the beam to the single
            // new winner so callers see it.
            if best_new < best_old {
                beam = sorted.into_iter().take(1).collect();
            }
            break;
        }

        // 5. PRUNE: Keep top K by timing
        beam = sorted.into_iter().take(config.beam_width).collect();
    }

    let (best_scheduler, best_timing) = beam.into_iter().next().unwrap_or((scheduler, Duration::MAX));

    Ok(BeamResult { scheduler: best_scheduler, timing: best_timing, iterations, candidates_evaluated })
}

// ============================================================================
// REPLAY
// ============================================================================

/// Replay a sequence of optimizations on a scheduler.
///
/// Used to restore cached beam search results.
pub fn replay_opts(mut scheduler: Scheduler, opts: &[Opt]) -> Result<Scheduler, OptError> {
    for opt in opts {
        apply_opt(&mut scheduler, opt, true)?;
    }
    Ok(scheduler)
}

/// Get the applied optimizations from a scheduler.
pub fn get_applied_opts(scheduler: &Scheduler) -> &[Opt] {
    &scheduler.applied_opts
}

// ============================================================================
// CACHING
// ============================================================================

/// Global sled database for beam search cache.
///
/// Lazy-initialized on first access. Returns None if cache directory
/// cannot be created or database cannot be opened.
static CACHE_DB: Lazy<Option<sled::Db>> = Lazy::new(|| {
    let cache_dir = dirs::cache_dir()?.join("svod");
    std::fs::create_dir_all(&cache_dir).ok()?;
    sled::open(cache_dir.join("beam_cache")).ok()
});

/// Cache key for beam search results.
///
/// Includes the limit configuration (max_upcast, max_local, max_uops) so that
/// changing caps invalidates cached entries: replaying opts produced under a
/// looser cap could reintroduce a kernel that no longer satisfies the new cap.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    /// On-disk key schema. Bump whenever replay semantics change.
    schema: u32,
    /// Hash of the AST structure.
    ast_hash: u64,
    /// Beam width used for search.
    beam_width: usize,
    /// Renderer/TC backend.
    device: svod_ir::RendererDevice,
    /// Full target/capability/rewrite identity.
    renderer_fingerprint: u64,
    /// Upcast/unroll product cap at search time.
    max_upcast: usize,
    /// Local/warp/group_reduce product cap at search time.
    max_local: usize,
    /// UOp count cap at search time.
    max_uops: usize,
    /// Post-optimization behavior not represented by BeamConfig.
    behavior_fingerprint: u64,
}

impl CacheKey {
    /// Create a cache key from a scheduler and config.
    fn from_scheduler(scheduler: &Scheduler, config: &BeamConfig, behavior_fingerprint: u64) -> Self {
        // Use structural hash for cross-run stability. The recursive Hash for UOp
        // traverses (dtype, op) of the entire DAG — same AST structure produces
        // the same hash regardless of process-local ids.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        scheduler.ast().hash(&mut hasher);
        let ast_hash = hasher.finish();

        Self {
            schema: 3,
            ast_hash,
            beam_width: config.beam_width,
            device: scheduler.ren.device,
            renderer_fingerprint: scheduler.ren.cache_fingerprint(),
            max_upcast: config.max_upcast,
            max_local: config.max_local,
            max_uops: config.max_uops,
            behavior_fingerprint,
        }
    }

    /// Convert to bytes for database key.
    fn to_bytes(&self) -> Vec<u8> {
        let device_str = self.device.canonical();
        let mut bytes = Vec::with_capacity(68 + device_str.len());
        bytes.extend_from_slice(&self.schema.to_le_bytes());
        bytes.extend_from_slice(&self.ast_hash.to_le_bytes());
        bytes.extend_from_slice(&self.renderer_fingerprint.to_le_bytes());
        bytes.extend_from_slice(&self.beam_width.to_le_bytes());
        bytes.extend_from_slice(&self.max_upcast.to_le_bytes());
        bytes.extend_from_slice(&self.max_local.to_le_bytes());
        bytes.extend_from_slice(&self.max_uops.to_le_bytes());
        bytes.extend_from_slice(&self.behavior_fingerprint.to_le_bytes());
        bytes.extend_from_slice(device_str.as_bytes());
        bytes
    }
}

/// Serialize applied opts to bytes for caching using bincode.
fn serialize_opts(opts: &[Opt]) -> Vec<u8> {
    bincode::serialize(opts).expect("Opt serialization should not fail")
}

/// Deserialize opts from cached bytes using bincode.
fn deserialize_opts(bytes: &[u8]) -> Option<Vec<Opt>> {
    bincode::deserialize(bytes).ok()
}

/// Get cached beam search result.
fn cache_get(key: &CacheKey) -> Option<Vec<Opt>> {
    let db = CACHE_DB.as_ref()?;
    let bytes = db.get(key.to_bytes()).ok()??;
    deserialize_opts(&bytes)
}

/// Store beam search result in cache.
fn cache_put(key: &CacheKey, opts: &[Opt]) {
    if let Some(db) = CACHE_DB.as_ref()
        && db.insert(key.to_bytes(), serialize_opts(opts)).is_ok()
    {
        // Flush to disk to ensure persistence across runs
        let _ = db.flush();
    }
}

/// Remove a stale cache entry.
fn cache_invalidate(key: &CacheKey) {
    if let Some(db) = CACHE_DB.as_ref() {
        let _ = db.remove(key.to_bytes());
        let _ = db.flush();
    }
}

/// Run beam search with disk caching.
///
/// Checks the cache before running beam search. If a cached result exists,
/// replays the optimizations instead of searching. Results are cached after
/// successful search.
///
/// # Arguments
///
/// * `scheduler` - Initial scheduler state
/// * `config` - Beam search configuration (includes disable_cache flag)
/// * `compile_and_time` - Function to compile and time a scheduler state
///
/// # Returns
///
/// `BeamResult` containing the best scheduler found.
pub fn beam_search_cached<F>(
    scheduler: Scheduler,
    config: &BeamConfig,
    compile_and_time: F,
) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    beam_search_cached_with_behavior(scheduler, config, 0, compile_and_time)
}

/// Run cached beam search with an explicit post-optimization behavior identity.
pub fn beam_search_cached_with_behavior<F>(
    scheduler: Scheduler,
    config: &BeamConfig,
    behavior_fingerprint: u64,
    compile_and_time: F,
) -> Result<BeamResult, OptError>
where
    F: Fn(&Scheduler, Option<Duration>) -> Option<CandidateMetrics> + Sync,
{
    let key = CacheKey::from_scheduler(&scheduler, config, behavior_fingerprint);

    // Check cache (unless disabled)
    if !config.disable_cache
        && let Some(cached_opts) = cache_get(&key)
    {
        // Replay cached optimizations. If replay fails (stale entry from code changes),
        // or the replayed scheduler exceeds the current limits (looser cap at search
        // time, tighter cap now), invalidate and fall through to fresh search.
        tracing::info!(opts_count = cached_opts.len(), "Beam cache HIT - replaying opts");
        match replay_opts(scheduler.clone(), &cached_opts) {
            Ok(replayed) if validate_limits(&replayed, config) => {
                let timing = compile_and_time(&replayed, None).map(|m| m.timing).unwrap_or(Duration::MAX);
                return Ok(BeamResult { scheduler: replayed, timing, iterations: 0, candidates_evaluated: 0 });
            }
            Ok(_) => {
                tracing::warn!("Beam cache replayed scheduler violates limits - invalidating");
                cache_invalidate(&key);
            }
            Err(e) => {
                tracing::warn!(?e, "Beam cache replay failed (stale entry?) - invalidating");
                cache_invalidate(&key);
            }
        }
    }

    tracing::info!("Beam cache MISS - running search");
    // Run beam search
    let result = beam_search(scheduler, config, compile_and_time)?;

    // Cache result (unless disabled)
    if !config.disable_cache {
        cache_put(&key, &result.scheduler.applied_opts);
    }

    Ok(result)
}

/// Clear the beam search cache.
///
/// Useful for testing or when invalidating cached results.
pub fn clear_cache() {
    if let Some(db) = CACHE_DB.as_ref() {
        let _ = db.clear();
    }
}

#[cfg(test)]
#[path = "../test/unit/optimizer/beam_internal.rs"]
mod tests;
