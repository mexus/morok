//! Object-cache degradation: every store-side failure must report a miss
//! rather than fail the caller's compile (UA2).

use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};

fn identity() -> CompilerIdentity {
    CompilerIdentity {
        schema: OBJECT_CACHE_SCHEMA,
        backend: "fake".into(),
        target_architecture: "target-a".into(),
        toolchain: "fake 1.2.3".into(),
        flags: vec!["-first".into()],
        abi: "abi-v1".into(),
        object_format: "fake-object-v1".into(),
    }
}

#[test]
fn unwritable_store_still_serves_compiled_bytes() {
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::tempdir().unwrap();
    let cache = ObjectCache::open(dir.path(), 4096).unwrap();
    let warm = ObjectCacheKey::new(b"warm", identity());
    assert_eq!(cache.get_or_compile(&warm, |_| Ok(()), || Ok(b"warm-object".to_vec())).unwrap(), b"warm-object");

    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o555)).unwrap();
    let writable = std::fs::File::create(dir.path().join(".writable-probe")).is_ok();
    let result = (!writable).then(|| {
        let cold = ObjectCacheKey::new(b"cold", identity());
        (
            cache.get_or_compile(&warm, |_| Ok(()), || panic!("warm entry must still be readable")),
            cache.get_or_compile(&cold, |_| Ok(()), || Ok(b"cold-object".to_vec())),
        )
    });
    std::fs::set_permissions(dir.path(), std::fs::Permissions::from_mode(0o755)).unwrap();

    // Running as root defeats the mode bits; the assertion is then vacuous.
    let Some((warm_bytes, cold_bytes)) = result else { return };
    assert_eq!(warm_bytes.unwrap(), b"warm-object");
    assert_eq!(cold_bytes.unwrap(), b"cold-object", "an unpublishable entry must still be compiled and returned");
}

#[test]
fn uncreatable_cache_directory_disables_the_cache() {
    // procfs rejects directory creation for every uid.
    unsafe { std::env::set_var("SVOD_OBJECT_CACHE_DIR", "/proc/svod-object-cache-must-not-exist") };
    let cache = ObjectCache::from_env();
    unsafe { std::env::remove_var("SVOD_OBJECT_CACHE_DIR") };
    assert!(matches!(cache, Ok(None)), "an unopenable store must disable the cache, not fail: {cache:?}");
}
