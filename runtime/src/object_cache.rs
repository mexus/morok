//! Persistent, content-addressed compiled-object storage.
//!
//! Entries are intentionally schema-specific: old formats are cache misses, not
//! migration inputs. The store owns no process-global state, so callers can
//! disable it or drop it without affecting compiler/runtime lifetime.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

use crate::{Error, Result};

pub const OBJECT_CACHE_SCHEMA: u32 = 1;
const MAGIC: &[u8; 16] = b"SVODOBJCACHE\0\0\0\0";
const HEADER_LEN: usize = MAGIC.len() + 4 + 32 + 32 + 8;
const DEFAULT_MAX_BYTES: u64 = 1024 * 1024 * 1024;
const STALE_LOCK_AGE: Duration = Duration::from_secs(120);
static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// Every compiler property that can change emitted object bytes or their ABI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompilerIdentity {
    pub schema: u32,
    pub backend: String,
    pub target_architecture: String,
    pub toolchain: String,
    pub flags: Vec<String>,
    pub abi: String,
    pub object_format: String,
}

impl CompilerIdentity {
    pub fn cache_key(&self) -> String {
        let schema = self.schema.to_le_bytes();
        let mut fields = vec![schema.as_slice()];
        fields.extend(self.fields());
        format!("{}:{}", self.backend, hex(&digest_fields(fields)))
    }

    fn fields(&self) -> Vec<&[u8]> {
        let mut fields = vec![
            self.backend.as_bytes(),
            self.target_architecture.as_bytes(),
            self.toolchain.as_bytes(),
            self.abi.as_bytes(),
            self.object_format.as_bytes(),
        ];
        fields.extend(self.flags.iter().map(String::as_bytes));
        fields
    }
}

/// Content address for one compiler output. The source itself is never used as
/// a filename; only its digest participates in the canonical key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectCacheKey {
    pub source_digest: [u8; 32],
    pub compiler: CompilerIdentity,
}

impl ObjectCacheKey {
    pub fn new(source: &[u8], compiler: CompilerIdentity) -> Self {
        Self { source_digest: Sha256::digest(source).into(), compiler }
    }

    pub fn digest(&self) -> [u8; 32] {
        let schema = self.compiler.schema.to_le_bytes();
        let mut fields = vec![schema.as_slice(), self.source_digest.as_slice()];
        fields.extend(self.compiler.fields());
        digest_fields(fields)
    }
}

/// A host-owned cache handle. Dropping it closes all cache state; no worker or
/// global map survives the handle.
#[derive(Debug)]
pub struct ObjectCache {
    root: PathBuf,
    max_bytes: u64,
}

impl ObjectCache {
    pub fn open(root: impl Into<PathBuf>, max_bytes: u64) -> Result<Self> {
        let root = root.into();
        fs::create_dir_all(&root).map_err(|e| cache_io("create cache directory", e))?;
        Ok(Self { root, max_bytes })
    }

    /// Open the default cache. `SVOD_OBJECT_CACHE=0` is the explicit host-side
    /// off switch. `SVOD_OBJECT_CACHE_DIR` and `SVOD_OBJECT_CACHE_MAX_BYTES`
    /// override location and byte budget.
    pub fn from_env() -> Result<Option<Self>> {
        if std::env::var("SVOD_OBJECT_CACHE").as_deref() == Ok("0") {
            return Ok(None);
        }
        let root = if let Some(path) = std::env::var_os("SVOD_OBJECT_CACHE_DIR") {
            PathBuf::from(path)
        } else if let Some(path) = std::env::var_os("XDG_CACHE_HOME") {
            PathBuf::from(path).join("svod/objects")
        } else if let Some(path) = std::env::var_os("HOME") {
            PathBuf::from(path).join(".cache/svod/objects")
        } else {
            return Ok(None);
        };
        let max_bytes = match std::env::var("SVOD_OBJECT_CACHE_MAX_BYTES") {
            Ok(value) => value.parse().map_err(|_| Error::JitCompilation {
                reason: format!("invalid SVOD_OBJECT_CACHE_MAX_BYTES={value:?}"),
            })?,
            Err(_) => DEFAULT_MAX_BYTES,
        };
        Self::open(root, max_bytes).map(Some)
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Read or produce one validated object. Both cached and newly compiled
    /// bytes pass the backend validator before they can reach a runtime loader.
    pub fn get_or_compile<V, C>(&self, key: &ObjectCacheKey, validate: V, compile: C) -> Result<Vec<u8>>
    where
        V: Fn(&[u8]) -> Result<()>,
        C: FnOnce() -> Result<Vec<u8>>,
    {
        let digest = key.digest();
        let path = self.entry_path(&digest);
        if let Some(bytes) = self.read_validated(&path, &digest, &validate)? {
            return Ok(bytes);
        }

        let lock_path = path.with_extension("lock");
        let _lock = LockFile::acquire(&lock_path)?;
        if let Some(bytes) = self.read_validated(&path, &digest, &validate)? {
            return Ok(bytes);
        }

        let bytes = compile()?;
        validate(&bytes)?;
        let encoded = encode_entry(&digest, &bytes);
        if self.max_bytes > 0 && encoded.len() as u64 <= self.max_bytes {
            atomic_write(&path, &encoded)?;
        }
        self.evict_to_budget(path.file_name())?;
        Ok(bytes)
    }

    pub(crate) fn get_validated<V>(&self, key: &ObjectCacheKey, validate: V) -> Result<Option<Vec<u8>>>
    where
        V: Fn(&[u8]) -> Result<()>,
    {
        let digest = key.digest();
        self.read_validated(&self.entry_path(&digest), &digest, &validate)
    }

    pub(crate) fn publish_compiled<V>(&self, key: &ObjectCacheKey, bytes: Vec<u8>, validate: V) -> Result<Vec<u8>>
    where
        V: Fn(&[u8]) -> Result<()>,
    {
        self.get_or_compile(key, validate, || Ok(bytes))
    }

    /// Persist deterministic compiler probes separately from evictable object
    /// entries. This lets a warm process reconstruct a versioned object key
    /// without invoking the compiler just to run `--version` or `-###`.
    pub(crate) fn get_or_create_probe<C>(&self, namespace: &str, input: &[u8], create: C) -> Result<Vec<u8>>
    where
        C: FnOnce() -> Result<Vec<u8>>,
    {
        let digest = digest_fields([namespace.as_bytes(), input]);
        let path = self.root.join(format!("probe-{}-{}.data", sanitize(namespace), hex(&digest)));
        if let Some(bytes) = read_entry(&path, &digest)? {
            return Ok(bytes);
        }
        let _lock = LockFile::acquire(&path.with_extension("lock"))?;
        if let Some(bytes) = read_entry(&path, &digest)? {
            return Ok(bytes);
        }
        let bytes = create()?;
        if bytes.is_empty() {
            return Err(Error::JitCompilation { reason: format!("empty {namespace} compiler probe") });
        }
        atomic_write(&path, &encode_entry(&digest, &bytes))?;
        Ok(bytes)
    }

    fn entry_path(&self, digest: &[u8; 32]) -> PathBuf {
        self.root.join(format!("{}.obj", hex(digest)))
    }

    fn read_validated<V>(&self, path: &Path, digest: &[u8; 32], validate: &V) -> Result<Option<Vec<u8>>>
    where
        V: Fn(&[u8]) -> Result<()>,
    {
        let Some(bytes) = read_entry(path, digest)? else { return Ok(None) };
        if validate(&bytes).is_ok() {
            return Ok(Some(bytes));
        }
        let _ = fs::remove_file(path);
        Ok(None)
    }

    fn evict_to_budget(&self, protected: Option<&std::ffi::OsStr>) -> Result<()> {
        if self.max_bytes == 0 {
            return Ok(());
        }
        let _lock = LockFile::acquire(&self.root.join("eviction.lock"))?;
        let mut entries = Vec::new();
        let mut total = 0u64;
        for item in fs::read_dir(&self.root).map_err(|e| cache_io("scan cache for eviction", e))? {
            let item = item.map_err(|e| cache_io("read cache directory entry", e))?;
            let path = item.path();
            if path.extension().and_then(|value| value.to_str()) != Some("obj") {
                continue;
            }
            let metadata = match item.metadata() {
                Ok(metadata) if metadata.is_file() => metadata,
                _ => continue,
            };
            total = total.saturating_add(metadata.len());
            entries.push((metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH), metadata.len(), path));
        }
        entries.sort_by_key(|(modified, _, _)| *modified);
        for (_, size, path) in entries {
            if total <= self.max_bytes {
                break;
            }
            if protected.is_some_and(|name| path.file_name() == Some(name)) {
                continue;
            }
            match fs::remove_file(&path) {
                Ok(()) => total = total.saturating_sub(size),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => total = total.saturating_sub(size),
                Err(error) => return Err(cache_io("evict cache entry", error)),
            }
        }
        Ok(())
    }
}

struct LockFile {
    path: PathBuf,
    _file: File,
}

impl LockFile {
    fn acquire(path: &Path) -> Result<Self> {
        loop {
            match OpenOptions::new().write(true).create_new(true).open(path) {
                Ok(mut file) => {
                    let _ = writeln!(file, "{}", std::process::id());
                    return Ok(Self { path: path.to_path_buf(), _file: file });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    let stale = fs::metadata(path)
                        .and_then(|metadata| metadata.modified())
                        .ok()
                        .and_then(|modified| modified.elapsed().ok())
                        .is_some_and(|age| age > STALE_LOCK_AGE);
                    if stale && !lock_owner_is_alive(path) {
                        let _ = fs::remove_file(path);
                    } else {
                        std::thread::sleep(Duration::from_millis(10));
                    }
                }
                Err(error) => return Err(cache_io("acquire cache lock", error)),
            }
        }
    }
}

impl Drop for LockFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn read_entry(path: &Path, expected_key: &[u8; 32]) -> Result<Option<Vec<u8>>> {
    let mut file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(cache_io("open cache entry", error)),
    };
    let mut encoded = Vec::new();
    if file.read_to_end(&mut encoded).is_err() {
        let _ = fs::remove_file(path);
        return Ok(None);
    }
    let Some(payload) = decode_entry(&encoded, expected_key) else {
        let _ = fs::remove_file(path);
        return Ok(None);
    };
    Ok(Some(payload.to_vec()))
}

fn encode_entry(key: &[u8; 32], payload: &[u8]) -> Vec<u8> {
    let payload_digest: [u8; 32] = Sha256::digest(payload).into();
    let mut encoded = Vec::with_capacity(HEADER_LEN + payload.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&OBJECT_CACHE_SCHEMA.to_le_bytes());
    encoded.extend_from_slice(key);
    encoded.extend_from_slice(&payload_digest);
    encoded.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    encoded.extend_from_slice(payload);
    encoded
}

fn decode_entry<'a>(encoded: &'a [u8], expected_key: &[u8; 32]) -> Option<&'a [u8]> {
    if encoded.len() < HEADER_LEN || &encoded[..MAGIC.len()] != MAGIC {
        return None;
    }
    let schema = u32::from_le_bytes(encoded[16..20].try_into().ok()?);
    if schema != OBJECT_CACHE_SCHEMA || &encoded[20..52] != expected_key {
        return None;
    }
    let expected_payload_digest = &encoded[52..84];
    let len = usize::try_from(u64::from_le_bytes(encoded[84..92].try_into().ok()?)).ok()?;
    let payload = encoded.get(HEADER_LEN..HEADER_LEN.checked_add(len)?)?;
    if HEADER_LEN + len != encoded.len() || Sha256::digest(payload).as_slice() != expected_payload_digest {
        return None;
    }
    Some(payload)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::JitCompilation { reason: format!("cache path has no parent: {}", path.display()) })?;
    let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let temp = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name().and_then(|name| name.to_str()).unwrap_or("entry"),
        std::process::id(),
        sequence
    ));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .map_err(|e| cache_io("create cache temp", e))?;
        file.write_all(bytes).map_err(|e| cache_io("write cache temp", e))?;
        file.sync_all().map_err(|e| cache_io("sync cache temp", e))?;
        fs::rename(&temp, path).map_err(|e| cache_io("publish cache entry", e))?;
        File::open(parent).and_then(|directory| directory.sync_all()).map_err(|e| cache_io("sync cache directory", e))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

fn digest_fields<'a>(fields: impl IntoIterator<Item = &'a [u8]>) -> [u8; 32] {
    let mut digest = Sha256::new();
    for field in fields {
        digest.update((field.len() as u64).to_le_bytes());
        digest.update(field);
    }
    digest.finalize().into()
}

fn hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0xf) as usize] as char);
    }
    output
}

fn sanitize(value: &str) -> String {
    value.chars().map(|ch| if ch.is_ascii_alphanumeric() || ch == '-' { ch } else { '_' }).collect()
}

#[cfg(unix)]
fn lock_owner_is_alive(path: &Path) -> bool {
    let Ok(contents) = fs::read_to_string(path) else { return false };
    let Some(pid) = contents.trim().parse::<i32>().ok().filter(|pid| *pid > 0) else { return false };
    // Signal 0 performs existence/permission checking without delivering a
    // signal. EPERM still means the owner exists.
    let result = unsafe { libc::kill(pid, 0) };
    result == 0 || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

#[cfg(not(unix))]
fn lock_owner_is_alive(_path: &Path) -> bool {
    false
}

fn cache_io(action: &'static str, source: std::io::Error) -> Error {
    Error::Jit { source: Box::new(source.into()), context: action }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};

    use super::*;

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
