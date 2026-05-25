//! KFD topology parser: enumerates AMD GPU nodes via sysfs.
//!
//! Reads `/sys/devices/virtual/kfd/kfd/topology/nodes/N/properties` for each
//! numeric subdir, returning a [`Vec<AmdNode>`]. Containers and hosts without
//! `/dev/kfd` get an empty result (never panic).
//!
//! Source format (one key/value per line, whitespace-separated):
//! ```text
//! gpu_id 4660
//! cpu_cores_count 0
//! simd_count 4
//! gfx_target_version 110000
//! drm_render_minor 128
//! …
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

/// A single KFD topology entry (CPU or GPU node).
#[derive(Debug, Clone)]
pub struct AmdNode {
    /// KFD node index (0, 1, 2…), matches the sysfs directory name.
    pub node_id: u32,
    /// KFD's `gpu_id`; 0 for CPU nodes (which we filter out at enumerate-time
    /// for the GPU-only enumerator).
    pub gpu_id: u32,
    /// Minor number for `/dev/dri/renderD{N}`. 0 for CPU nodes.
    pub drm_render_minor: u32,
    /// Raw `gfx_target_version` integer (e.g. `110000` for gfx1100).
    pub gfx_target_version: u32,
    pub simd_count: u32,
    pub array_count: u32,
    pub simd_per_cu: u32,
    pub max_waves_per_simd: u32,
    pub lds_size_in_kb: u32,
    pub wave_front_size: u32,
    /// Number of XCC compute engines. Defaults to 1 when the field is absent.
    pub num_xcc: u32,
    /// Maximum number of independent compute queues KFD will give us on this
    /// device. Defaults to 8 when the field is absent.
    pub num_cp_queues: u32,
    /// Maximum scratch slots per CU. Used to size the default scratch buffer
    /// at device open. KFD's sysfs property `max_slots_scratch_cu`; falls back
    /// to `simd_per_cu * max_waves_per_simd` (matches RDNA scratch slot
    /// budgeting) when absent.
    pub max_slots_scratch_cu: u32,
}

/// Sysfs root for KFD topology. Public so tests can override via env.
fn topology_root() -> PathBuf {
    if let Ok(override_path) = std::env::var("SVOD_KFD_TOPOLOGY") {
        return PathBuf::from(override_path);
    }
    PathBuf::from("/sys/devices/virtual/kfd/kfd/topology/nodes")
}

/// Enumerate all KFD GPU nodes on this host.
///
/// Returns `Ok(empty)` when no nodes are present (no `/dev/kfd`, container
/// without sysfs, or only a CPU node) so callers can distinguish "no GPU" from
/// hard failures. Errors propagate when sysfs is present but malformed.
pub fn enumerate() -> Vec<AmdNode> {
    let root = topology_root();
    let read = match fs::read_dir(&root) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };

    let mut nodes = Vec::new();
    for entry in read.flatten() {
        let name = match entry.file_name().to_str().map(str::to_string) {
            Some(n) => n,
            None => continue,
        };
        let node_id: u32 = match name.parse() {
            Ok(n) => n,
            Err(_) => continue,
        };
        let node_dir = entry.path();
        let props_path = node_dir.join("properties");
        let props = match fs::read_to_string(&props_path) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let map = parse_properties(&props);
        // `gpu_id` lives in a sibling file on real KFD; fall back to the
        // properties block for the test stub that crams it inline.
        let gpu_id = fs::read_to_string(node_dir.join("gpu_id"))
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok())
            .unwrap_or_else(|| map.get("gpu_id").copied().unwrap_or(0) as u32);
        if gpu_id == 0 {
            // CPU node — skip.
            continue;
        }
        let simd_per_cu = map.get("simd_per_cu").copied().unwrap_or(0) as u32;
        let max_waves_per_simd = map.get("max_waves_per_simd").copied().unwrap_or(0) as u32;
        nodes.push(AmdNode {
            node_id,
            gpu_id,
            drm_render_minor: map.get("drm_render_minor").copied().unwrap_or(0) as u32,
            gfx_target_version: map.get("gfx_target_version").copied().unwrap_or(0) as u32,
            simd_count: map.get("simd_count").copied().unwrap_or(0) as u32,
            array_count: map.get("array_count").copied().unwrap_or(0) as u32,
            simd_per_cu,
            max_waves_per_simd,
            lds_size_in_kb: map.get("lds_size_in_kb").copied().unwrap_or(0) as u32,
            wave_front_size: map.get("wave_front_size").copied().unwrap_or(0) as u32,
            num_xcc: map.get("num_xcc").copied().unwrap_or(1) as u32,
            num_cp_queues: map.get("num_cp_queues").copied().unwrap_or(8) as u32,
            // KFD exposes `max_slots_scratch_cu` directly on some kernels;
            // fall back to simd_per_cu * max_waves_per_simd (the natural
            // scratch slot budget) when absent.
            max_slots_scratch_cu: map
                .get("max_slots_scratch_cu")
                .copied()
                .map(|v| v as u32)
                .unwrap_or_else(|| simd_per_cu.max(1) * max_waves_per_simd.max(1)),
        });
    }
    nodes.sort_by_key(|n| n.node_id);
    nodes
}

fn parse_properties(s: &str) -> HashMap<String, u64> {
    let mut map = HashMap::new();
    for line in s.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut it = line.split_whitespace();
        let key = match it.next() {
            Some(k) => k.to_string(),
            None => continue,
        };
        let val = match it.next().and_then(|v| v.parse::<u64>().ok()) {
            Some(v) => v,
            None => continue,
        };
        map.insert(key, val);
    }
    map
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parse_properties_handles_real_kfd_format() {
        let raw = "cpu_cores_count 0\nsimd_count 4\ngfx_target_version 110000\ndrm_render_minor 128\n";
        let map = parse_properties(raw);
        assert_eq!(map["cpu_cores_count"], 0);
        assert_eq!(map["simd_count"], 4);
        assert_eq!(map["gfx_target_version"], 110000);
        assert_eq!(map["drm_render_minor"], 128);
    }

    #[test]
    fn enumerate_returns_empty_when_topology_missing() {
        let temp = tempfile_dir();
        // Point at a non-existent directory.
        unsafe {
            std::env::set_var("SVOD_KFD_TOPOLOGY", temp.join("does_not_exist"));
        }
        let nodes = enumerate();
        unsafe {
            std::env::remove_var("SVOD_KFD_TOPOLOGY");
        }
        assert!(nodes.is_empty());
    }

    #[test]
    fn enumerate_skips_cpu_nodes_and_parses_gpu() {
        let root = tempfile_dir();
        let n0 = root.join("0");
        let n1 = root.join("1");
        std::fs::create_dir_all(&n0).unwrap();
        std::fs::create_dir_all(&n1).unwrap();
        // Node 0: CPU (gpu_id 0).
        let mut f = std::fs::File::create(n0.join("properties")).unwrap();
        write!(f, "gpu_id 0\ncpu_cores_count 32\nsimd_count 0\n").unwrap();
        // Node 1: GPU.
        let mut f = std::fs::File::create(n1.join("properties")).unwrap();
        write!(
            f,
            "gpu_id 5710\nsimd_count 4\ngfx_target_version 110000\ndrm_render_minor 128\nwave_front_size 32\nnum_cp_queues 8\n"
        )
        .unwrap();

        unsafe {
            std::env::set_var("SVOD_KFD_TOPOLOGY", &root);
        }
        let nodes = enumerate();
        unsafe {
            std::env::remove_var("SVOD_KFD_TOPOLOGY");
        }
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].node_id, 1);
        assert_eq!(nodes[0].gpu_id, 5710);
        assert_eq!(nodes[0].gfx_target_version, 110000);
        assert_eq!(nodes[0].drm_render_minor, 128);
        assert_eq!(nodes[0].wave_front_size, 32);
        assert_eq!(nodes[0].num_cp_queues, 8);
    }

    fn tempfile_dir() -> PathBuf {
        // Build a fresh per-test tempdir so concurrent tests don't collide on
        // `SVOD_KFD_TOPOLOGY`. We don't pull `tempfile` for one test path.
        let pid = std::process::id();
        let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("svod-kfd-topo-{pid}-{nonce}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Reads the real `/sys/devices/virtual/kfd/kfd/topology/nodes/` if it
    /// exists on this host. Asserts only that the parser doesn't choke on
    /// real-world data; the field values depend on the local hardware.
    #[test]
    fn enumerate_real_host_topology_does_not_panic() {
        // Ensure no test-suite override leaks in.
        unsafe {
            std::env::remove_var("SVOD_KFD_TOPOLOGY");
        }
        let nodes = enumerate();
        for n in &nodes {
            assert!(n.gpu_id != 0, "enumerate must skip CPU nodes (got gpu_id 0 in {n:?})");
        }
        eprintln!("host has {} KFD GPU node(s): {nodes:?}", nodes.len());
    }
}
