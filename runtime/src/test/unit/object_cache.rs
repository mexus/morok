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

#[test]
fn abandoned_lock_file_does_not_stall_compilation() {
    // Pre-UA4 this spun for `STALE_LOCK_AGE` (120 s) because the lock was the
    // file's existence and the recorded pid was alive (it is this process).
    let dir = tempfile::tempdir().unwrap();
    let cache = ObjectCache::open(dir.path(), 4096).unwrap();
    let key = ObjectCacheKey::new(b"abandoned", identity());
    let digest = key.digest();
    let lock_path =
        dir.path().join(format!("{}.lock", digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()));
    std::fs::write(&lock_path, format!("{}\n", std::process::id())).unwrap();

    let start = std::time::Instant::now();
    let bytes = cache.get_or_compile(&key, |_| Ok(()), || Ok(b"object".to_vec())).unwrap();
    assert_eq!(bytes, b"object");
    assert!(start.elapsed() < std::time::Duration::from_secs(5), "took {:?}", start.elapsed());
    // The entry is published: an unheld lock file never blocked publication.
    assert_eq!(cache.get_or_compile(&key, |_| Ok(()), || panic!("must be a cache hit")).unwrap(), b"object");
}

/// Key derivation, hit/corruption recovery, concurrent publication and eviction
/// over a fully populated compiler identity. Reaches `ObjectCache::entry_path`,
/// so it keeps its own identity fixture rather than sharing the one above.
mod store_semantics {
    use crate::Error;
    use crate::object_cache::{CompilerIdentity, OBJECT_CACHE_SCHEMA, ObjectCache, ObjectCacheKey};
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    fn identity() -> CompilerIdentity {
        CompilerIdentity {
            schema: OBJECT_CACHE_SCHEMA,
            backend: "fake".into(),
            target_architecture: "target-a".into(),
            toolchain: "fake 1.2.3 sha256:abc".into(),
            flags: vec!["-first".into(), "-second".into()],
            abi: "abi-v1".into(),
            object_format: "fake-object-v1".into(),
        }
    }

    #[test]
    fn key_covers_every_identity_field_and_flag_order() {
        let base = ObjectCacheKey::new(b"source", identity()).digest();
        let mut variants = Vec::new();
        let mut changed = identity();
        changed.schema += 1;
        variants.push(changed);
        let mut changed = identity();
        changed.backend.push('x');
        variants.push(changed);
        let mut changed = identity();
        changed.target_architecture.push('x');
        variants.push(changed);
        let mut changed = identity();
        changed.toolchain.push('x');
        variants.push(changed);
        let mut changed = identity();
        changed.flags.swap(0, 1);
        variants.push(changed);
        let mut changed = identity();
        changed.abi.push('x');
        variants.push(changed);
        let mut changed = identity();
        changed.object_format.push('x');
        variants.push(changed);
        assert!(variants.into_iter().all(|identity| ObjectCacheKey::new(b"source", identity).digest() != base));
        assert_ne!(ObjectCacheKey::new(b"other source", identity()).digest(), base);
    }

    #[test]
    fn deterministic_hit_and_corruption_recovery() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ObjectCache::open(dir.path(), 4096).unwrap();
        let key = ObjectCacheKey::new(b"source", identity());
        let calls = std::cell::Cell::new(0);
        let validate = |bytes: &[u8]| {
            if bytes.starts_with(b"OBJ") {
                Ok(())
            } else {
                Err(Error::JitCompilation { reason: "bad fake object".into() })
            }
        };
        let first = cache
            .get_or_compile(&key, validate, || {
                calls.set(calls.get() + 1);
                Ok(b"OBJ-one".to_vec())
            })
            .unwrap();
        let second = cache.get_or_compile(&key, validate, || panic!("cache hit must not compile")).unwrap();
        assert_eq!(first, second);
        assert_eq!(calls.get(), 1);

        fs::write(cache.entry_path(&key.digest()), b"corrupt").unwrap();
        let recovered = cache
            .get_or_compile(&key, validate, || {
                calls.set(calls.get() + 1);
                Ok(b"OBJ-two".to_vec())
            })
            .unwrap();
        assert_eq!(recovered, b"OBJ-two");
        assert_eq!(calls.get(), 2);
    }

    #[test]
    fn concurrent_writers_compile_once() {
        let dir = tempfile::tempdir().unwrap();
        let cache = Arc::new(ObjectCache::open(dir.path(), 4096).unwrap());
        let key = Arc::new(ObjectCacheKey::new(b"shared", identity()));
        let barrier = Arc::new(Barrier::new(8));
        let calls = Arc::new(AtomicU64::new(0));
        let threads = (0..8)
            .map(|_| {
                let cache = Arc::clone(&cache);
                let key = Arc::clone(&key);
                let barrier = Arc::clone(&barrier);
                let calls = Arc::clone(&calls);
                std::thread::spawn(move || {
                    barrier.wait();
                    cache
                        .get_or_compile(
                            &key,
                            |_| Ok(()),
                            || {
                                calls.fetch_add(1, Ordering::SeqCst);
                                std::thread::sleep(Duration::from_millis(30));
                                Ok(b"object".to_vec())
                            },
                        )
                        .unwrap()
                })
            })
            .collect::<Vec<_>>();
        for thread in threads {
            assert_eq!(thread.join().unwrap(), b"object");
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert!(
            !fs::read_dir(dir.path()).unwrap().any(|entry| entry
                .unwrap()
                .path()
                .extension()
                .and_then(|ext| ext.to_str())
                == Some("tmp"))
        );
    }

    #[test]
    fn eviction_bounds_stored_object_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ObjectCache::open(dir.path(), 300).unwrap();
        for source in [b"one".as_slice(), b"two", b"three"] {
            let key = ObjectCacheKey::new(source, identity());
            cache.get_or_compile(&key, |_| Ok(()), || Ok(vec![source[0]; 100])).unwrap();
            std::thread::sleep(Duration::from_millis(2));
        }
        let total: u64 = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.path().extension()?.to_str()? != "obj" {
                    return None;
                }
                Some(entry.metadata().ok()?.len())
            })
            .sum();
        assert!(total <= 300, "stored {total} bytes");
    }
}
