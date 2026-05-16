use parking_lot::Mutex;

/// Shared lock serializing every test that touches global RNG state
/// (`manual_seed`, `Tensor::rand`). All `rand` test modules grab this before
/// calling into the rand module. Separate per-file statics would not serialize
/// across files, causing parallel races on `GLOBAL_SEED` / `SEED_EPOCH`.
pub(super) static RAND_TEST_LOCK: Mutex<()> = Mutex::new(());

pub mod distributions;
pub mod dtype;
pub mod like;
pub mod reference;
pub mod smoke;
