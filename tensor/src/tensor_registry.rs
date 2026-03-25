//! Global tensor registry for atomic graph substitution.
//!
//! This module implements Tinygrad's `all_tensors` pattern using papaya's lock-free HashMap.
//! When rangeify transforms a UOp (e.g., NEG → BUFFERIZE(NEG)), the `becomes_map` must be
//! applied to ALL tensors that reference it - not just the one being realized.
//!
//! Without this, diamond patterns (like argmin's NEG feeding both MAX and EQ) fail because
//! different consumers see different versions of the same producer.
//!
//! # Thread Safety
//!
//! All operations are lock-free and thread-safe. Uses papaya's epoch-based reclamation
//! for concurrent access and parking_lot::RwLock for interior UOp mutation.
//!
//! # Memory Management (Tinygrad-aligned)
//!
//! Tensors are stored as `Weak<TensorEntry>` in the registry. When all strong references
//! (held by `Tensor` structs) are dropped, the entry becomes eligible for cleanup.
//! Dead weak refs are cleaned lazily on access or via `gc_dead_refs()`.
//!
//! This matches Tinygrad's `weakref.WeakKeyDictionary` pattern - no manual cleanup required.
//!
//! # Buffer Storage
//!
//! Buffers are stored in a separate map (`BUFFERS`) indexed by UOp ID.
//! This is a lookup index for `collect_input_buffers()` during schedule creation.
//! - Key is UOp ID (unique per buffer via `Op::Unique` monotonic counter)
//! - Value is `Arc<Buffer>` (strong ref — kept alive for the duration of the computation)
//! - Stale entries cleaned via `gc_dead_refs()` when the UOp is no longer alive
//!
//! Unlike Tinygrad (which stores buffers inline in UOps), Morok uses a separate
//! index because UOps are immutable and hash-consed. Unique buffer IDs guarantee
//! entries never collide, so stale entries are harmless — only a memory concern.
//!
//! TensorEntry also caches the buffer for direct access via tensor.buffer().

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock, Weak};

use morok_device::Buffer;
use morok_ir::{Op, UOp, UOpKey};
use papaya::HashMap as PapayaMap;
use parking_lot::RwLock;

/// Atomic counter for unique tensor IDs.
static TENSOR_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

fn next_tensor_id() -> u64 {
    TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Entry in the global tensor registry.
///
/// Uses RwLock for interior mutability of the UOp during global substitution.
/// The RwLock allows concurrent reads (typical tensor operations) with exclusive
/// writes only during `apply_map_to_tensors`.
///
/// Buffer can be set at construction (input tensors) or later (realized tensors).
/// Uses OnceLock for thread-safe one-time initialization.
pub struct TensorEntry {
    /// Unique tensor ID (stable across UOp updates).
    pub id: u64,
    /// The computation graph (mutable for global substitution).
    pub uop: RwLock<Arc<UOp>>,
    /// The materialized buffer (can be set once via OnceLock).
    buffer: OnceLock<Arc<Buffer>>,
}

impl TensorEntry {
    /// Get the buffer if materialized.
    pub fn buffer(&self) -> Option<&Arc<Buffer>> {
        self.buffer.get()
    }

    /// Set the buffer (can only be called once, subsequent calls are no-ops).
    /// Returns true if buffer was set, false if already set.
    pub fn set_buffer(&self, buffer: Arc<Buffer>) -> bool {
        self.buffer.set(buffer).is_ok()
    }
}

// Global tensor registry using lock-free concurrent HashMap.
//
// Design: Stores Weak<TensorEntry> for automatic memory management (Tinygrad-aligned).
// - Tensor structs hold Arc<TensorEntry> (strong refs)
// - Registry holds Weak<TensorEntry> (weak refs)
// - When Tensor is dropped, TensorEntry can be cleaned up
// - Dead weak refs cleaned lazily on access or via gc_dead_refs()
static TENSORS: OnceLock<PapayaMap<u64, Weak<TensorEntry>>> = OnceLock::new();

// Direct buffer storage: UOp ID → Arc<Buffer>.
//
// Lookup index for collect_input_buffers() during schedule creation.
// Buffer UOp IDs are unique (Op::Unique monotonic counter), so entries never
// collide across tests. Stale entries cleaned via gc_dead_refs().
static BUFFERS: OnceLock<PapayaMap<u64, Arc<Buffer>>> = OnceLock::new();

/// Pending assigns: buffer_uop.id → list of ASSIGN UOps.
///
/// Matches Tinygrad's `_pending_assigns: dict[UOp, list[UOp]]`.
/// When `tensor.assign(value)` is called, the ASSIGN UOp is appended here.
/// During `realize()`, pending assigns for referenced BUFFERs are
/// side-realized (recursively for transitive deps) before the main computation.
///
/// Uses PapayaMap for lock-free concurrent access (consistent with TENSORS/BUFFERS).
/// Multiple assigns per buffer are supported (for partial/view assigns).
static PENDING_ASSIGNS: OnceLock<PapayaMap<u64, Vec<Arc<UOp>>>> = OnceLock::new();

fn tensors() -> &'static PapayaMap<u64, Weak<TensorEntry>> {
    TENSORS.get_or_init(PapayaMap::new)
}

fn buffers() -> &'static PapayaMap<u64, Arc<Buffer>> {
    BUFFERS.get_or_init(PapayaMap::new)
}

fn pending_assigns() -> &'static PapayaMap<u64, Vec<Arc<UOp>>> {
    PENDING_ASSIGNS.get_or_init(PapayaMap::new)
}

/// Track a pending assign for side-realization during realize().
///
/// Appends to the list for this buffer (supports multiple partial/view assigns).
/// Deduplicates: if the same UOp is already pending, it's replaced (matches Tinygrad line 322).
pub fn add_pending_assign(buffer_id: u64, assign_uop: Arc<UOp>) {
    let map = pending_assigns();
    let guard = map.guard();
    let mut list = map.get(&buffer_id, &guard).cloned().unwrap_or_default();
    list.retain(|existing| !Arc::ptr_eq(existing, &assign_uop));
    list.push(assign_uop);
    map.insert(buffer_id, list, &guard);
}

/// Take all pending assigns for a buffer (removes from tracking).
/// Returns None if no assigns pending. Matches Tinygrad's `_pending_assigns.pop(buf, [])`.
pub fn take_pending_assigns(buffer_id: u64) -> Option<Vec<Arc<UOp>>> {
    let map = pending_assigns();
    let guard = map.guard();
    let val = map.get(&buffer_id, &guard).cloned();
    if val.is_some() {
        map.remove(&buffer_id, &guard);
    }
    val.filter(|v| !v.is_empty())
}

/// Check if a specific buffer has pending assigns (for recursion guard).
pub fn has_pending_assign(buffer_id: u64) -> bool {
    let map = pending_assigns();
    let guard = map.guard();
    map.get(&buffer_id, &guard).is_some_and(|v| !v.is_empty())
}

/// Check if there are any pending assigns globally.
pub fn has_pending_assigns() -> bool {
    !pending_assigns().is_empty()
}

/// Substitute a becomes_map into all remaining pending assigns.
///
/// After side-realizing one pending assign, its becomes_map must propagate
/// into other pending assigns that may reference stale graphs.
/// Matches Tinygrad tensor.py:285-287.
pub fn substitute_pending_assigns(becomes_map: &HashMap<UOpKey, Arc<UOp>>) {
    if becomes_map.is_empty() {
        return;
    }
    let map = pending_assigns();
    let pinned = map.pin();
    let entries: Vec<(u64, Vec<Arc<UOp>>)> = pinned.iter().map(|(&k, v)| (k, v.clone())).collect();
    for (key, assign_list) in entries {
        let mut changed = false;
        let new_list: Vec<Arc<UOp>> = assign_list
            .iter()
            .map(|uop| {
                let new_uop = uop.substitute(becomes_map);
                if !Arc::ptr_eq(uop, &new_uop) {
                    changed = true;
                }
                new_uop
            })
            .collect();
        if changed {
            let guard = map.guard();
            map.insert(key, new_list, &guard);
        }
    }
}

/// Register a new tensor without buffer (for lazy computation graphs).
///
/// Thread-safe: each call creates a unique tensor ID.
/// The registry stores a weak reference; the caller holds the strong reference.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
///
/// # Returns
///
/// Arc to the registered TensorEntry (caller owns the strong reference)
pub fn register_tensor(uop: Arc<UOp>) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop), buffer: OnceLock::new() });

    // Store weak ref in registry - entry stays alive as long as caller holds Arc
    let guard = tensors().guard();
    tensors().insert(id, Arc::downgrade(&entry), &guard);

    entry
}

/// Register a new tensor with buffer (for input tensors and realized tensors).
///
/// Stores buffer in both:
/// 1. BUFFERS map (indexed by UOp ID) - for schedule buffer lookups
/// 2. TensorEntry.buffer - for direct tensor access
///
/// The registry stores a weak reference; the caller holds the strong reference.
///
/// # Arguments
///
/// * `uop` - The tensor's computation graph
/// * `buffer` - The materialized buffer
/// * `buffer_uop_id` - The UOp ID to index under (for lookups)
///
/// # Returns
///
/// Arc to the registered TensorEntry (caller owns the strong reference)
pub fn register_tensor_with_buffer(uop: Arc<UOp>, buffer: Arc<Buffer>, buffer_uop_id: u64) -> Arc<TensorEntry> {
    let id = next_tensor_id();
    let entry = Arc::new(TensorEntry { id, uop: RwLock::new(uop), buffer: OnceLock::from(buffer.clone()) });

    // Store weak ref in tensor registry
    let guard = tensors().guard();
    tensors().insert(id, Arc::downgrade(&entry), &guard);

    // Store buffer indexed by UOp ID (for collect_input_buffers lookups)
    let buf_guard = buffers().guard();
    buffers().insert(buffer_uop_id, buffer, &buf_guard);

    entry
}

/// Get buffer by UOp ID (cloned).
///
/// Direct lookup from BUFFERS map.
/// Used by collect_input_buffers() during schedule creation.
pub fn get_buffer(uop_id: u64) -> Option<Buffer> {
    let guard = buffers().guard();
    buffers().get(&uop_id, &guard).map(|arc_buf| (**arc_buf).clone())
}

/// Get buffer by UOp ID as Arc (shared reference, no clone).
///
/// Used by ensure_buffer() to attach a buffer without cloning.
pub fn get_buffer_arc(uop_id: u64) -> Option<Arc<Buffer>> {
    let guard = buffers().guard();
    buffers().get(&uop_id, &guard).cloned()
}

/// Remove buffer entry from the BUFFERS map.
///
/// Called during cleanup to eagerly remove stale entries.
pub fn remove_buffer(uop_id: u64) {
    let buf_guard = buffers().guard();
    buffers().remove(&uop_id, &buf_guard);
}

/// Get count of buffers in the registry (for testing/diagnostics).
pub fn buffer_count() -> usize {
    buffers().len()
}

/// Register a buffer for an existing tensor.
///
/// Used by realize() to associate output buffers with tensors for schedule lookups.
/// Stores buffer in both BUFFERS map and TensorEntry.
///
/// # Arguments
///
/// * `uop_id` - The UOp ID to index under (for lookups)
/// * `tensor_id` - The tensor ID that owns this buffer
/// * `buffer` - The materialized buffer
pub fn register_buffer(uop_id: u64, tensor_id: u64, buffer: Arc<Buffer>) {
    // Store buffer indexed by UOp ID (for collect_input_buffers lookups)
    let buf_guard = buffers().guard();
    buffers().insert(uop_id, buffer.clone(), &buf_guard);

    // Also set buffer on the TensorEntry for direct tensor access
    if let Some(entry) = get_tensor(tensor_id) {
        entry.set_buffer(buffer);
    }
}

/// Register a buffer by UOp ID only (no TensorEntry association).
///
/// Used for pending assign side-realization where the buffer belongs
/// to the computation graph, not a specific tensor.
pub fn register_buffer_by_uop_id(uop_id: u64, buffer: Arc<Buffer>) {
    let guard = buffers().guard();
    buffers().insert(uop_id, buffer, &guard);
}

/// Get a tensor entry by ID.
///
/// Thread-safe read operation. Returns None if tensor was dropped.
pub fn get_tensor(id: u64) -> Option<Arc<TensorEntry>> {
    let guard = tensors().guard();
    tensors().get(&id, &guard)?.upgrade()
}

/// Remove dead weak references and stale buffer entries from the registry.
///
/// Tensors: removes entries whose `Weak<TensorEntry>` can no longer be upgraded.
/// Buffers: removes entries whose UOp is no longer alive in the UOp cache.
///
/// This is optional — stale entries don't affect correctness (unique buffer IDs
/// prevent collisions). Call this to reclaim registry memory in long-running programs.
pub fn gc_dead_refs() {
    // Clean dead tensor weak refs
    let map = tensors();
    let guard = map.guard();
    let to_remove: Vec<u64> = map.iter(&guard).filter(|(_, weak)| weak.upgrade().is_none()).map(|(k, _)| *k).collect();
    for id in to_remove {
        map.remove(&id, &guard);
    }

    // Clean stale buffer entries (UOp no longer alive in the cache)
    let live_uop_ids = morok_ir::uop::live_uop_ids();
    let buf_map = buffers();
    let buf_guard = buf_map.guard();
    let stale_bufs: Vec<u64> =
        buf_map.iter(&buf_guard).filter(|(uop_id, _)| !live_uop_ids.contains(uop_id)).map(|(id, _)| *id).collect();
    for uop_id in stale_bufs {
        buf_map.remove(&uop_id, &buf_guard);
    }
}

/// Legacy alias for gc_dead_refs (for compatibility).
///
/// With weak references, tensors are automatically cleaned up when no longer
/// referenced. This function now just cleans up dead weak refs in the registry.
#[deprecated(note = "Tensor registry now uses weak refs - cleanup is automatic. Use gc_dead_refs() to clean registry.")]
pub fn gc_unused_tensors() {
    gc_dead_refs();
}

/// Apply a transformation map to ALL live tensors globally.
///
/// This is Morok's equivalent of Tinygrad's `_apply_map_to_tensors`.
/// When rangeify creates a becomes_map (old UOp → new UOp), this function
/// ensures ALL tensors see the same transformed versions.
///
/// # Arguments
///
/// * `becomes_map` - Mapping from original UOps to their transformed versions
///
/// # Thread Safety
///
/// This function acquires write locks on affected tensors during the update phase.
/// Other tensors can still be read/written concurrently.
#[allow(clippy::mutable_key_type)]
pub fn apply_map_to_tensors(becomes_map: &HashMap<UOpKey, Arc<UOp>>) {
    if becomes_map.is_empty() {
        return;
    }

    let map = tensors();
    let guard = map.guard();

    // Phase 1: Find affected tensors (read-only scan, skip dead weak refs)
    let affected: Vec<Arc<TensorEntry>> = map
        .iter(&guard)
        .filter_map(|(_, weak)| {
            let entry = weak.upgrade()?; // Skip dead entries
            let is_affected = {
                let uop = entry.uop.read();
                // Check if tensor's root UOp is in map
                if becomes_map.contains_key(&UOpKey(uop.clone())) {
                    true
                } else {
                    // Check if any node in the graph is in map
                    uop.toposort().iter().any(|n| becomes_map.contains_key(&UOpKey(n.clone())))
                }
            }; // uop lock dropped here
            if is_affected { Some(entry) } else { None }
        })
        .collect();

    if affected.is_empty() {
        return;
    }

    // Phase 2: Create SINK of affected tensor UOps
    let sources: Vec<Arc<UOp>> = affected.iter().map(|e| e.uop.read().clone()).collect();
    let sink = UOp::sink(sources.clone());

    // Phase 3: Atomic substitution across all affected UOps
    let new_sink = sink.substitute(becomes_map);

    // Phase 4: Update each tensor's UOp (acquires write locks)
    if let Op::Sink { sources: new_sources } = new_sink.op() {
        for (entry, (old, new)) in affected.iter().zip(sources.iter().zip(new_sources.iter())) {
            if !Arc::ptr_eq(old, new) {
                *entry.uop.write() = new.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morok_dtype::DType;
    use morok_ir::ConstValue;

    #[test]
    fn test_register_and_get() {
        crate::test::helpers::test_setup();

        let uop = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let entry = register_tensor(uop.clone());

        let retrieved = get_tensor(entry.id).expect("Should find tensor");
        assert_eq!(retrieved.id, entry.id);
        assert!(Arc::ptr_eq(&*retrieved.uop.read(), &uop));
    }

    #[test]
    fn test_apply_map_updates_tensors() {
        crate::test::helpers::test_setup();

        // Create two tensors sharing a common UOp
        let shared = UOp::const_(DType::Float32, ConstValue::Float(1.0));
        let t1_uop = shared.neg();
        let t2_uop = shared.neg(); // Same as t1_uop due to hash consing

        let t1 = register_tensor(t1_uop.clone());
        let t2 = register_tensor(t2_uop.clone());

        // Create a replacement for the shared const
        let replacement = UOp::const_(DType::Float32, ConstValue::Float(2.0));

        #[allow(clippy::mutable_key_type)]
        let mut becomes_map = HashMap::new();
        becomes_map.insert(UOpKey(shared.clone()), replacement.clone());

        // Apply the map
        apply_map_to_tensors(&becomes_map);

        // Both tensors should now reference the replacement
        let t1_new = t1.uop.read();
        let t2_new = t2.uop.read();

        // The root NEG should now have the replacement as its source
        assert!(!Arc::ptr_eq(&*t1_new, &t1_uop), "t1 should be updated");
        assert!(!Arc::ptr_eq(&*t2_new, &t2_uop), "t2 should be updated");
    }
}
