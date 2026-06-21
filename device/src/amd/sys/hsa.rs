//! HSA / AQL ABI bindings.
//!
//! The HSA runtime descriptors the KFD-direct AQL path lays into GART and the
//! ring — `hsa_signal_t`, `hsa_kernel_dispatch_packet_t`, `hsa_queue_t`, and the
//! 256-byte `amd_queue_t` — plus the packet-header / queue-property /
//! kernel-code-property bit enums are bindgen output from the ROCm headers
//! vendored under `device/include/hsa/` (see `build.rs`). bindgen's
//! `layout_tests` emit compile-time size/offset asserts for these, so a header
//! bump that shifts a field can no longer silently corrupt the descriptor.
//!
//! The lone hand-written type is [`AmdHsaKernelDescriptor`]: the 64-byte
//! AMDGPU code-object *kernel descriptor* is an LLVM AMDHSA ABI struct that no
//! ROCm header declares (the closest, `amd_kernel_code.h`'s `amd_kernel_code_t`,
//! is the older 256-byte v2 layout), so we read it by offset out of the loaded
//! ELF. Its `kernel_code_properties` field shares the
//! `amd_kernel_code_properties_t` bit layout, so callers test it with the
//! generated `AMD_KERNEL_CODE_PROPERTIES_*` constants.

#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]

use std::mem::{offset_of, size_of};

include!(concat!(env!("OUT_DIR"), "/hsa_sys.rs"));

/// Build the `header` field for a kernel-dispatch packet: the BARRIER bit plus
/// system-scope acquire+release fences — matching ROCclr's default user-kernel
/// header (`rocvirtual.cpp` `dispatchPacketHeader_ = KERNEL_DISPATCH |
/// barrierHBits | scope`), which is what actually dispatches dependent compute
/// kernels. (The barrier=0 header in ROCr's `amd_blit_kernel.cpp` is the
/// internal *blit* path: independent memcpy kernels whose only deps are
/// signals, ordered by explicit BARRIER_AND packets — not a dependent kernel
/// chain.) Two distinct guarantees are needed for a producer→consumer chain on
/// the in-order queue, and the fences alone do not give the first:
/// - **Execution ordering** (BARRIER bit): the packet processor will not launch
///   this dispatch until every preceding packet has *completed*. Without it the
///   next kernel launches while the previous is still running and reads a
///   half-written intermediate → NaN cascade. The acquire/release scope orders
///   *caches*, not *execution*, so it cannot substitute for the barrier.
/// - **Cache coherence** (SCACQUIRE/SCRELEASE): the prior packet's release
///   flushes its writes to the fence scope before the next packet's acquire
///   invalidates and reads. SYSTEM scope is the conservative choice (ROCclr uses
///   AGENT scope intra-device and upgrades to SYSTEM only at host-read
///   boundaries; this uses SYSTEM scope everywhere — conservatively correct,
///   marginally slower than AGENT scope).
///
/// Note: the earlier belief that BARRIER "wedges the queue on multi-XCC" was a
/// misdiagnosis of the scratch-realloc-on-live-queue wedge (fixed by
/// pre-reserving scratch); BARRIER is correct and required. Bit positions from
/// the generated `hsa_packet_*` enums; `c_uint` constants narrowed to `u16`.
pub const fn kernel_dispatch_header() -> u16 {
    (hsa_packet_type_t_HSA_PACKET_TYPE_KERNEL_DISPATCH as u16)
        | (1 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER as u16)
        | ((hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16)
            << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE as u16)
        | ((hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16)
            << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE as u16)
}

// amd_queue_t field offsets, derived from the generated struct so they track
// the header. KFD's create_queue expects rptr/wptr at the dispatch-id offsets;
// the AQL packet processor reads the scratch (private-segment) config from the
// others, patched post-create on every scratch (re)allocation (see
// `AmdComputeQueue::set_aql_scratch`).
pub const OFFSET_READ_DISPATCH_ID: usize = offset_of!(amd_queue_t, read_dispatch_id);
pub const OFFSET_WRITE_DISPATCH_ID: usize = offset_of!(amd_queue_t, write_dispatch_id);
pub const OFFSET_COMPUTE_TMPRING_SIZE: usize = offset_of!(amd_queue_t, compute_tmpring_size);
pub const OFFSET_SCRATCH_RESOURCE_DESCRIPTOR: usize = offset_of!(amd_queue_t, scratch_resource_descriptor);
pub const OFFSET_SCRATCH_BACKING_MEMORY_LOCATION: usize = offset_of!(amd_queue_t, scratch_backing_memory_location);
pub const OFFSET_SCRATCH_WAVE64_LANE_BYTE_SIZE: usize = offset_of!(amd_queue_t, scratch_wave64_lane_byte_size);
/// `amd_queue_t.queue_inactive_signal` — the CP trap handler writes an exception
/// code (e.g. `0x401` insufficient-scratch) into this signal's `value` and halts
/// the queue until the host resets it to 0 (see ROCr `DynamicQueueEventsHandler`).
pub const OFFSET_QUEUE_INACTIVE_SIGNAL: usize = offset_of!(amd_queue_t, queue_inactive_signal);

/// AMDGPU kernel descriptor (64 bytes), stored in the code object's `.rodata`
/// kernel-descriptor symbol. Layout per the LLVM AMDHSA ABI (v5); read by
/// offset out of the loaded ELF when constructing an `AmdProgram`. Hand-written
/// because no vendored ROCm header declares this struct (see module docs).
#[repr(C, packed)]
#[derive(Debug, Clone, Copy, Default)]
pub struct AmdHsaKernelDescriptor {
    pub group_segment_fixed_size: u32,      // off  0..4
    pub private_segment_fixed_size: u32,    // off  4..8
    pub kernarg_size: u32,                  // off  8..12
    pub _reserved0: [u8; 4],                // off 12..16
    pub kernel_code_entry_byte_offset: i64, // off 16..24
    pub _reserved1: [u8; 20],               // off 24..44
    pub compute_pgm_rsrc3: u32,             // off 44..48
    pub compute_pgm_rsrc1: u32,             // off 48..52
    pub compute_pgm_rsrc2: u32,             // off 52..56
    pub kernel_code_properties: u16,        // off 56..58 (bit 10 = wave32)
    pub _reserved2: [u8; 6],                // off 58..64
}

const _DESC_SIZE_IS_64: () = assert!(size_of::<AmdHsaKernelDescriptor>() == 64);
