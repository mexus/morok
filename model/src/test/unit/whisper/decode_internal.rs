//! Whisper decode-seed internals: the prefill seed owns its own device caches
//! (never aliasing the prefill buffers) and seeds every scheduler row.

use crate::whisper::decode::{PrefillMetadata, build_decode_seed, clone_device_cache, copy_device_cache_row};
use std::sync::Arc;
use svod_device::{Buffer, BufferSpec, CpuAllocator};
use svod_dtype::DType;

fn cache(allocator: Arc<CpuAllocator>, values: &[f32]) -> Buffer {
    let mut buffer = Buffer::allocate(allocator, DType::Float32, vec![values.len()], BufferSpec::default()).unwrap();
    buffer.copyin(bytemuck::cast_slice(values)).unwrap();
    buffer
}

#[test]
fn scheduler_seed_owns_device_buffers_and_seeds_multiple_rows() {
    let allocator = Arc::new(CpuAllocator);
    let self_values = [1.0f32, 2.0, 3.0, 4.0];
    let cross_values = [5.0f32, 6.0, 7.0, 8.0, 9.0, 10.0];
    let self_k_source = cache(allocator.clone(), &self_values);
    let self_v_source = cache(allocator.clone(), &self_values);
    let cross_k_source = cache(allocator.clone(), &cross_values);
    let cross_v_source = cache(allocator.clone(), &cross_values);
    let metadata = PrefillMetadata {
        initial_tokens: vec![1, 2],
        sample_begin: 2,
        init_len: 2,
        suppress_tokens: Vec::new(),
        prefill_logits: vec![0.0; 4],
        no_speech_prob: f32::NAN,
        pos_embedding: Vec::new(),
        n_state: 0,
    };

    let seed = build_decode_seed(
        metadata,
        clone_device_cache(&self_k_source).unwrap(),
        clone_device_cache(&self_v_source).unwrap(),
        clone_device_cache(&cross_k_source).unwrap(),
        clone_device_cache(&cross_v_source).unwrap(),
    )
    .unwrap();

    assert_eq!(seed.metadata.initial_tokens, [1, 2]);
    assert_eq!(seed.per_pos_bytes, 2 * std::mem::size_of::<f32>());
    assert_eq!(seed.self_cache_bytes, std::mem::size_of_val(&self_values));
    assert_eq!(seed.cross_cache_bytes, std::mem::size_of_val(&cross_values));
    assert_eq!((seed.self_positions, seed.cross_positions), (2, 3));
    assert_ne!(seed.self_k_cache.storage_id(), self_k_source.storage_id());
    assert_ne!(seed.self_v_cache.storage_id(), self_v_source.storage_id());
    assert_ne!(seed.cross_k.storage_id(), cross_k_source.storage_id());
    assert_ne!(seed.cross_v.storage_id(), cross_v_source.storage_id());

    let self_stride = 4 * seed.per_pos_bytes;
    let mut self_rows = Buffer::allocate(
        allocator.clone(),
        DType::Float32,
        vec![2 * self_stride / std::mem::size_of::<f32>()],
        BufferSpec::default(),
    )
    .unwrap();
    copy_device_cache_row(&mut self_rows, 0, self_stride, &seed.self_k_cache).unwrap();
    copy_device_cache_row(&mut self_rows, 1, self_stride, &seed.self_k_cache).unwrap();
    let self_bytes = self_rows.as_host_bytes().unwrap();
    let expected_self: &[u8] = bytemuck::cast_slice(&self_values);
    assert_eq!(&self_bytes[..expected_self.len()], expected_self);
    assert_eq!(&self_bytes[self_stride..self_stride + expected_self.len()], expected_self);

    let mut cross_rows = Buffer::allocate(
        allocator,
        DType::Float32,
        vec![2 * seed.cross_cache_bytes / std::mem::size_of::<f32>()],
        BufferSpec::default(),
    )
    .unwrap();
    copy_device_cache_row(&mut cross_rows, 0, seed.cross_cache_bytes, &seed.cross_k).unwrap();
    copy_device_cache_row(&mut cross_rows, 1, seed.cross_cache_bytes, &seed.cross_k).unwrap();
    let cross_bytes = cross_rows.as_host_bytes().unwrap();
    let expected_cross: &[u8] = bytemuck::cast_slice(&cross_values);
    assert_eq!(&cross_bytes[..expected_cross.len()], expected_cross);
    assert_eq!(&cross_bytes[seed.cross_cache_bytes..], expected_cross);
}
