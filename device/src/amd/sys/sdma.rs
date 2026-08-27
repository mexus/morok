//! SDMA (System DMA) packet builders for the AMD copy queue.
//!
//! Uses the `sdma_4_0_0` opcode set — identical across SDMA v4/v5/v6 for these
//! ops, and gfx942's SDMA IP is `(4,4,2)`. All values are `u32` dwords stored
//! little-endian in the ring; the engine consumes them as a packed command
//! stream. SDMA op `0` is NOP, so zero-fill doubles as ring padding.
//!
//! Completion is signalled with a bare value-[`fence`]: it writes the timeline
//! value straight into the GTT signal slot the host busy-polls
//! ([`AmdSignal::wait_signal_value`](crate::amd::signal::AmdSignal)). The
//! KFD interrupt path (a second fence to the event mailbox + `SDMA_OP_TRAP`)
//! is intentionally omitted — svod observes completion via the coherent GTT
//! slot, so the TRAP (the #1 hang source if the interrupt context bits are
//! wrong) is unnecessary.

/// `SDMA_OP_COPY` — linear/tiled copy (sub-op selects the variant).
const SDMA_OP_COPY: u32 = 1;
/// `SDMA_OP_FENCE` — write a 32-bit value to a memory address on completion.
const SDMA_OP_FENCE: u32 = 5;
/// `SDMA_OP_POLL_REGMEM` — block the engine until a mem/reg value satisfies a
/// comparison.
const SDMA_OP_POLL_REGMEM: u32 = 8;
/// Generated SDMA v4/v5/v6 definitions: linear memory write and global clock.
const SDMA_OP_WRITE: u32 = 2;
const SDMA_OP_TIMESTAMP: u32 = 13;
const SDMA_SUBOP_TIMESTAMP_GET_GLOBAL: u32 = 2;
/// Linear-copy sub-opcode (sits in header bits 8..16; for linear it is 0).
const SDMA_SUBOP_COPY_LINEAR: u32 = 0;
/// POLL_REGMEM comparison: value `>=` reference.
const WAIT_REG_MEM_FUNCTION_GEQ: u32 = 5;

/// Max bytes per linear-copy packet. The COUNT field is 22 bits on SDMA v4
/// (gfx942), so a single packet copies at most `0x40_0000` bytes — callers
/// chunk larger transfers. (svod's old inline builder assumed `0x4000_0000`,
/// a latent truncation bug.)
pub const SDMA_MAX_COPY_BYTES: usize = 0x0040_0000;

#[inline]
fn lo(addr: u64) -> u32 {
    addr as u32
}

#[inline]
fn hi(addr: u64) -> u32 {
    (addr >> 32) as u32
}

/// Linear copy of `size` bytes `src` → `dst`. 7 dwords. `size` must be in
/// `1..=SDMA_MAX_COPY_BYTES` (the COUNT field is `size - 1`, masked to 22 bits).
pub fn copy_linear(src: u64, dst: u64, size: usize) -> [u32; 7] {
    debug_assert!(size > 0 && size <= SDMA_MAX_COPY_BYTES, "SDMA copy size {size} out of range");
    let header = SDMA_OP_COPY | (SDMA_SUBOP_COPY_LINEAR << 8);
    [header, ((size - 1) as u32) & 0x003F_FFFF, 0, lo(src), hi(src), lo(dst), hi(dst)]
}

/// Write the 32-bit `value` to `addr` once the engine reaches this packet.
/// 4 dwords. gfx9 takes no MTYPE flags; newer targets use MTYPE(3), matching
/// Tinygrad's AMD copy-queue signal packet. Writing the
/// timeline value into the coherent GTT signal slot is what the host wait
/// observes.
pub fn fence(addr: u64, value: u32, target_major: u32) -> [u32; 4] {
    let mtype = if target_major == 9 { 0 } else { 3 << 16 };
    [SDMA_OP_FENCE | mtype, lo(addr), hi(addr), value]
}

/// Stall the engine until the 32-bit value at `addr` is `>= value`. 6 dwords.
/// Used to serialise a copy behind prior in-flight work referencing the same
/// timeline slot.
pub fn poll_regmem_geq(addr: u64, value: u32) -> [u32; 6] {
    let header = SDMA_OP_POLL_REGMEM | (WAIT_REG_MEM_FUNCTION_GEQ << 28) | (1 << 31);
    // DW5: poll interval 0x04, retry count 0xFFF.
    [header, lo(addr), hi(addr), value, 0xFFFF_FFFF, 0x04 | (0xFFF << 16)]
}

/// Write one 64-bit value. SDMA WRITE's DW3 is the dword count minus one.
pub fn write_u64(addr: u64, value: u64) -> [u32; 6] {
    [SDMA_OP_WRITE, lo(addr), hi(addr), 1, value as u32, (value >> 32) as u32]
}

/// Write the global GPU clock counter to memory (three dwords).
pub fn timestamp_global(addr: u64) -> [u32; 3] {
    [SDMA_OP_TIMESTAMP | (SDMA_SUBOP_TIMESTAMP_GET_GLOBAL << 8), lo(addr), hi(addr)]
}
