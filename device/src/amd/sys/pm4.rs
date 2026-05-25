//! PM4 packet builders for AMD GPU command processor.
//!
//! Direct port of tinygrad's `runtime/autogen/am/pm4_nv.py` (Navi family /
//! gfx10+). The PM4 opcodes/bitfields are HARDWARE constants — stable across
//! kernel versions and across gfx10/gfx11/gfx12 (CDNA gfx9 uses different
//! values, not covered here).
//!
//! Used by [`AmdComputeQueue`] to emit signal/barrier/wait packets through
//! the AQL vendor-specific indirect-buffer mechanism. The AQL kernel-dispatch
//! packet itself doesn't honor the HSA `completion_signal` field on AMD
//! hardware; tinygrad therefore wraps PM4 `RELEASE_MEM` in an AQL vendor IB
//! packet to set the completion signal — see `ops_amd.py:385-394, 433-435`.

#![allow(dead_code)]

// ── PACKET3 header ────────────────────────────────────────────────────────

pub const PACKET_TYPE3: u32 = 3;

/// Build a PACKET3 header dword.
/// `n` is the number of data dwords following the header (i.e. total dwords - 1).
pub const fn packet3(op: u32, n: u32) -> u32 {
    (PACKET_TYPE3 << 30) | ((op & 0xFF) << 8) | ((n & 0x3FFF) << 16)
}

// ── PACKET3 opcodes (gfx10+) ──────────────────────────────────────────────

pub const PACKET3_NOP: u32 = 0x10;
pub const PACKET3_WAIT_REG_MEM: u32 = 0x3C;
pub const PACKET3_INDIRECT_BUFFER: u32 = 0x3F;
pub const PACKET3_RELEASE_MEM: u32 = 0x49;
pub const PACKET3_ACQUIRE_MEM: u32 = 0x58;

/// `(1 << 23)` bit for the PACKET3_INDIRECT_BUFFER count dword, marking the
/// IB as valid for the CP to execute.
pub const INDIRECT_BUFFER_VALID: u32 = 1 << 23;

// ── RELEASE_MEM bitfields (DW1) ───────────────────────────────────────────

/// VGT event type for "cache flush + invalidate timestamp".
pub const CACHE_FLUSH_AND_INV_TS_EVENT: u32 = 20;

/// `event_index = end_of_pipe` for `RELEASE_MEM`.
pub const EVENT_INDEX_END_OF_PIPE: u32 = 5;

pub const fn release_mem_event_type(x: u32) -> u32 {
    x << 0
}
pub const fn release_mem_event_index(x: u32) -> u32 {
    x << 8
}

pub const RELEASE_MEM_GCR_GLM_WB: u32 = 1 << 12;
pub const RELEASE_MEM_GCR_GLM_INV: u32 = 1 << 13;
pub const RELEASE_MEM_GCR_GLV_INV: u32 = 1 << 14;
pub const RELEASE_MEM_GCR_GL1_INV: u32 = 1 << 15;
pub const RELEASE_MEM_GCR_GL2_INV: u32 = 1 << 20;
pub const RELEASE_MEM_GCR_GL2_WB: u32 = 1 << 21;
pub const RELEASE_MEM_GCR_SEQ: u32 = 1 << 22;

/// All cache-flush flags ORed — matches tinygrad's `cache_flush=True` path
/// at `ops_amd.py:112-114`.
pub const RELEASE_MEM_CACHE_FLUSH_ALL: u32 = RELEASE_MEM_GCR_GLV_INV
    | RELEASE_MEM_GCR_GL1_INV
    | RELEASE_MEM_GCR_GL2_INV
    | RELEASE_MEM_GCR_GLM_WB
    | RELEASE_MEM_GCR_GLM_INV
    | RELEASE_MEM_GCR_GL2_WB
    | RELEASE_MEM_GCR_SEQ;

// ── RELEASE_MEM bitfields (DW2) ───────────────────────────────────────────

pub const fn release_mem_dst_sel(x: u32) -> u32 {
    x << 16
}
pub const fn release_mem_int_sel(x: u32) -> u32 {
    x << 24
}
pub const fn release_mem_data_sel(x: u32) -> u32 {
    x << 29
}

/// `data_sel = send_32_bit_low` — write the low 32 bits of the value field
/// (DW5..6) to the target address. Used by tinygrad's `signal()` at
/// `ops_amd.py:388`.
pub const DATA_SEL_SEND_32_BIT_LOW: u32 = 1;
/// `data_sel = send_64_bit_data`.
pub const DATA_SEL_SEND_64_BIT_DATA: u32 = 2;
/// `int_sel = send_interrupt_after_write_confirm`.
pub const INT_SEL_INTERRUPT_AFTER_WRITE: u32 = 2;
/// `dst_sel = 0` for ordinary memory destinations.
pub const DST_SEL_MEMORY: u32 = 0;

// ── WAIT_REG_MEM bitfields (DW1: info) ────────────────────────────────────

pub const fn wait_reg_mem_function(x: u32) -> u32 {
    x << 0
}
pub const fn wait_reg_mem_mem_space(x: u32) -> u32 {
    x << 4
}
pub const fn wait_reg_mem_operation(x: u32) -> u32 {
    x << 6
}
pub const fn wait_reg_mem_engine(x: u32) -> u32 {
    x << 8
}

/// `function = >= comparison`. Tinygrad uses this for signal-value waits at
/// `ops_amd.py:31` and `:498`.
pub const WAIT_REG_MEM_FUNC_GEQ: u32 = 5;

// ── HDP flush register handshake addresses ────────────────────────────────
//
// Tinygrad `ops_amd.py:133-137` emits a register-space WAIT_REG_MEM against
// the BIF HDP flush register pair before `acquire_mem` in `memory_barrier`.
// Without this handshake, host writes to GTT (kernarg arena, ring) may not
// be visible to the GPU's command processor when it consumes the next
// packet — manifests as a hung dispatch with the signal never firing.
//
// The register pair address is **uniform across supported arches**:
// gfx9 (vega_offsets) and gfx10+ (navi_offsets) both define
// `NBIO_BASE_INST0_SEG2 = 0x0D20`, and every NBIO/NBIF dictionary in
// `regs.py` defines the polled register at `(offset=262, segment=2)` —
// only the symbolic name (PF0 vs PF1) differs across NBIO versions
// (Strix Point's nbio_7_11_0 calls it `PF1_GPU_HDP_FLUSH_REQ` while older
// nbios call the same physical register `PF0_GPU_HDP_FLUSH_REQ` —
// tinygrad's `pf = '0' if version[:2] != (7, 11) else '1'` is purely a
// name disambiguation, not an address change).
pub const HDP_FLUSH_REQ_ADDR: u32 = 0xD20 + 262;
pub const HDP_FLUSH_DONE_ADDR: u32 = 0xD20 + 263;

// ── PACKET3 opcodes & constants (PM4 dispatch path, single-XCC) ──────────
//
// Mirrors `submodules/new_new_tinygrad/tinygrad/runtime/autogen/am/pm4_nv.py`.
// These constants drive the raw-PM4 `AmdComputeQueue::exec_pm4` path that
// tinygrad uses when `xccs == 1` (the gfx11/gfx12 default at `ops_amd.py:989`).

pub const PACKET3_DISPATCH_DIRECT: u32 = 0x15; // pm4_nv.py:61
pub const PACKET3_SET_SH_REG: u32 = 0x76; // pm4_nv.py:395
pub const PACKET3_SET_SH_REG_START: u32 = 0x2c00; // pm4_nv.py:396
pub const PACKET3_EVENT_WRITE: u32 = 0x46; // pm4_nv.py:291

// ── SH-relative COMPUTE_* register offsets ────────────────────────────────
//
// Each constant is `(GC_BASE_INST0_SEG0 + field_offset) - PACKET3_SET_SH_REG_START`.
// The pair is uniform across the supported arch matrix because gfx9 uses
// `GC_BASE=0x2000` with `field_offset=0xe00`, while gfx11/12 use
// `GC_BASE=0x1260` with `field_offset=0x1ba0` — both sums equal `0x2e00`,
// yielding `0x200` once `PACKET3_SET_SH_REG_START=0x2c00` is subtracted.
//
// Source: `submodules/new_new_tinygrad/tinygrad/runtime/autogen/am/regs.py`
// (CDNA entry at line 351, RDNA3 at 1608, RDNA4 at 5576) and per-arch
// `*_offsets.py` (`vega_offsets.py:199`, `navi_offsets.py` for gfx11/12).

pub const COMPUTE_DISPATCH_INITIATOR: u32 = 0x200;
pub const COMPUTE_START_X: u32 = 0x204;
pub const COMPUTE_NUM_THREAD_X: u32 = 0x207;
pub const COMPUTE_PGM_LO: u32 = 0x20c;
pub const COMPUTE_DISPATCH_SCRATCH_BASE_LO: u32 = 0x210;
pub const COMPUTE_PGM_RSRC1: u32 = 0x212;
pub const COMPUTE_PGM_RSRC2: u32 = 0x213;
pub const COMPUTE_RESOURCE_LIMITS: u32 = 0x215;
pub const COMPUTE_TMPRING_SIZE: u32 = 0x218;
pub const COMPUTE_RESTART_X: u32 = 0x21b;
pub const COMPUTE_PGM_RSRC3: u32 = 0x228;
pub const COMPUTE_USER_DATA_0: u32 = 0x240;

// ── COMPUTE_DISPATCH_INITIATOR field bits ─────────────────────────────────
// Per `regs.py:1608` (gfx11) / `:5576` (gfx12). The `cs_w32_en` bit is only
// defined on gfx11/12; gfx9 (CDNA) ignores it.

pub const DISPATCH_INITIATOR_COMPUTE_SHADER_EN: u32 = 1 << 0;
pub const DISPATCH_INITIATOR_FORCE_START_AT_000: u32 = 1 << 2;
pub const DISPATCH_INITIATOR_CS_W32_EN: u32 = 1 << 15;

// ── Event constants ───────────────────────────────────────────────────────
// `EVENT_WRITE(CS_PARTIAL_FLUSH, EVENT_INDEX=4)` terminates a PM4 dispatch
// stream — flushes CS state so the next dispatch sees clean queue state
// (tinygrad `ops_amd.py:367`).

pub const CS_PARTIAL_FLUSH: u32 = 7;
pub const EVENT_INDEX_PARTIAL_FLUSH: u32 = 4;

// ── Builders ──────────────────────────────────────────────────────────────

/// Build a PM4 RELEASE_MEM packet that writes `value` (low 32 bits) to
/// `addr`. With `cache_flush = true`, also flushes/invalidates all GPU
/// caches before the write — required so prior kernel stores are visible
/// at the signal site. Matches `AMDComputeQueue.signal` /
/// `release_mem` at `ops_amd.py:110-131`.
///
/// Layout (8 dwords total):
/// ```text
/// dw0  PACKET3(RELEASE_MEM, 6)   header (count = body_len - 1)
/// dw1  event_dw                   event type + index + cache flush bits
/// dw2  memsel_dw                  DATA_SEL | INT_SEL | DST_SEL
/// dw3  addr lo
/// dw4  addr hi
/// dw5  value lo
/// dw6  value hi
/// dw7  ctxid                      always 0 in our usage
/// ```
pub fn release_mem(addr: u64, value: u32, cache_flush: bool) -> [u32; 8] {
    let cache = if cache_flush { RELEASE_MEM_CACHE_FLUSH_ALL } else { 0 };
    let event_dw =
        release_mem_event_type(CACHE_FLUSH_AND_INV_TS_EVENT) | release_mem_event_index(EVENT_INDEX_END_OF_PIPE) | cache;
    let memsel_dw = release_mem_data_sel(DATA_SEL_SEND_32_BIT_LOW)
        | release_mem_int_sel(INT_SEL_INTERRUPT_AFTER_WRITE)
        | release_mem_dst_sel(DST_SEL_MEMORY);
    [
        packet3(PACKET3_RELEASE_MEM, 6),
        event_dw,
        memsel_dw,
        addr as u32,
        (addr >> 32) as u32,
        value,
        0, // value_hi (unused for 32-bit data_sel)
        0, // ctxid
    ]
}

// ── ACQUIRE_MEM GCR_CNTL cache-flag bit positions (gfx10+) ────────────────
// Source: `submodules/new_new_tinygrad/tinygrad/runtime/autogen/am/pm4_nv.py:351-364`.
// Each constant is the lambda `(x) << N` shift positions; here we encode the
// `1`-set value directly.

pub const GCR_GLI_INV: u32 = 1 << 0;
pub const GCR_GL1_RANGE: u32 = 1 << 2; // not used; included for documentation
pub const GCR_GLM_WB: u32 = 1 << 4;
pub const GCR_GLM_INV: u32 = 1 << 5;
pub const GCR_GLK_WB: u32 = 1 << 6;
pub const GCR_GLK_INV: u32 = 1 << 7;
pub const GCR_GLV_INV: u32 = 1 << 8;
pub const GCR_GL1_INV: u32 = 1 << 9;
pub const GCR_GL2_INV: u32 = 1 << 14;
pub const GCR_GL2_WB: u32 = 1 << 15;

/// All cache levels invalidated/written-back. Equivalent to tinygrad's
/// `acquire_mem()` default (`gli=glm=glk=glv=gl1=gl2=1`). Yields 0xC3F1.
pub const GCR_FLAGS_ALL: u32 =
    GCR_GLI_INV | GCR_GLM_WB | GCR_GLM_INV | GCR_GLK_WB | GCR_GLK_INV | GCR_GLV_INV | GCR_GL1_INV | GCR_GL2_INV | GCR_GL2_WB;

/// Mild flush: skips GLI and GL2 invalidate/writeback. Equivalent to
/// tinygrad's `_exec` pre-dispatch `acquire_mem(gli=0, gl2=0)`. Yields 0x03F0.
pub const GCR_FLAGS_NO_GLI_GL2: u32 = GCR_GLM_WB | GCR_GLM_INV | GCR_GLK_WB | GCR_GLK_INV | GCR_GLV_INV | GCR_GL1_INV;

/// Build a PM4 ACQUIRE_MEM packet (cache invalidate). Tinygrad's
/// `memory_barrier()` at `ops_amd.py:92-100` issues this after a
/// WAIT_REG_MEM on the HDP-flush register. `cache_flags` selects which
/// cache levels to invalidate/write-back — use [`GCR_FLAGS_ALL`] for the
/// `memory_barrier` site and [`GCR_FLAGS_NO_GLI_GL2`] for the pre-dispatch
/// site in `exec_pm4` (tinygrad's `acquire_mem(gli=0, gl2=0)`).
///
/// Layout (8 dwords total):
/// ```text
/// dw0  PACKET3(ACQUIRE_MEM, 6)   header
/// dw1  0                          (CP_COHER_CNTL — unused on gfx10+)
/// dw2  size lo                    0xFFFFFFFF = full VA range
/// dw3  size hi                    0xFFFFFFFF
/// dw4  addr lo                    0
/// dw5  addr hi                    0
/// dw6  0                          poll interval (unused for ACQUIRE_MEM)
/// dw7  cache flags                GLI/GLM/GLK/GLV/GL1/GL2 inv + WBs
/// ```
pub fn acquire_mem_with(cache_flags: u32) -> [u32; 8] {
    [packet3(PACKET3_ACQUIRE_MEM, 6), 0, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0, cache_flags]
}

/// Shorthand for `acquire_mem_with(GCR_FLAGS_ALL)` — invalidates and writes
/// back every cache level. Used by `memory_barrier`.
pub fn acquire_mem() -> [u32; 8] {
    acquire_mem_with(GCR_FLAGS_ALL)
}

/// Build a PM4 WAIT_REG_MEM packet that polls memory at `addr` until
/// `(*addr & mask) >= value`. Used by tinygrad's AQL `wait()` to block on
/// a signal slot reaching the target value (`ops_amd.py:85-90`).
pub fn wait_reg_mem(addr: u64, value: u32, mask: u32) -> [u32; 7] {
    let info = wait_reg_mem_mem_space(1) // memory (not register)
        | wait_reg_mem_function(WAIT_REG_MEM_FUNC_GEQ)
        | wait_reg_mem_engine(0); // ME engine
    [
        packet3(PACKET3_WAIT_REG_MEM, 5),
        info,
        addr as u32,
        (addr >> 32) as u32,
        value,
        mask,
        4, // poll interval (4 dword units)
    ]
}

/// Build a register-space WAIT_REG_MEM packet that polls `reg_done` until it
/// matches the value the CP previously wrote to `reg_req`. This is the
/// "register handshake" variant: `mem_space = 0` (register), `operation = 1`
/// (signals CP to perform the write-then-poll handshake). Used by
/// tinygrad's `memory_barrier` (`ops_amd.py:133-137`) to flush the HDP
/// before subsequent cache invalidations — without this handshake, host
/// writes to GTT memory (kernarg arena, ring buffers) are not guaranteed
/// to be visible to the GPU's command processor.
///
/// Layout: 7 dwords (same shape as memory-space variant), but the two
/// address dwords carry `(reg_req, reg_done)` instead of `(addr_lo, addr_hi)`.
pub fn wait_reg_mem_register(reg_req: u32, reg_done: u32, value: u32, mask: u32) -> [u32; 7] {
    let info = wait_reg_mem_mem_space(0)            // 0 = register space
        | wait_reg_mem_operation(1)                  // 1 = REQ→DONE handshake
        | wait_reg_mem_function(WAIT_REG_MEM_FUNC_GEQ)
        | wait_reg_mem_engine(0);
    [packet3(PACKET3_WAIT_REG_MEM, 5), info, reg_req, reg_done, value, mask, 4]
}

/// Build the HDP flush handshake packet for `memory_barrier`. Polls the
/// BIF GPU_HDP_FLUSH_REQ/DONE register pair with `value = mask = 0xFFFF_FFFF`
/// (full handshake — wait for all engines). Tinygrad `ops_amd.py:135-136`.
pub fn hdp_flush() -> [u32; 7] {
    wait_reg_mem_register(HDP_FLUSH_REQ_ADDR, HDP_FLUSH_DONE_ADDR, 0xFFFF_FFFF, 0xFFFF_FFFF)
}

/// Build a `PACKET3_SET_SH_REG`. `reg_offset` is the SH-relative offset
/// (e.g. [`COMPUTE_PGM_LO`]). Successive `values` are written to consecutive
/// registers, matching tinygrad's `wreg(reg, *args)` semantics at
/// `ops_amd.py:69` (the same `wreg` call sets `PGM_LO` + `PGM_HI` in one
/// packet, etc.).
///
/// Output layout (2 + N dwords):
/// ```text
/// dw0  PACKET3(SET_SH_REG, N)        header (count = data dwords)
/// dw1  reg_offset                    SH-relative, bits 0..15
/// dw2..  values...                   N consecutive register payloads
/// ```
pub fn set_sh_reg(reg_offset: u32, values: &[u32]) -> Vec<u32> {
    let n = values.len() as u32;
    let mut v = Vec::with_capacity(2 + values.len());
    v.push(packet3(PACKET3_SET_SH_REG, n));
    v.push(reg_offset & 0xFFFF);
    v.extend_from_slice(values);
    v
}

/// Build a `PACKET3_DISPATCH_DIRECT` packet — 5 dwords total. The PM4 CP
/// reads `dim_x/y/z` and the `dispatch_initiator` value, then launches a
/// compute dispatch with the previously-set COMPUTE_* registers. Mirrors
/// the final packet emitted by `_exec` at `ops_amd.py:361-364`.
pub fn dispatch_direct(grid: [u32; 3], dispatch_initiator: u32) -> [u32; 5] {
    [packet3(PACKET3_DISPATCH_DIRECT, 3), grid[0], grid[1], grid[2], dispatch_initiator]
}

/// Build a `PACKET3_EVENT_WRITE` — 2 dwords. We use this to emit
/// `CS_PARTIAL_FLUSH` after a `DISPATCH_DIRECT` so back-to-back dispatches
/// see clean queue state (tinygrad `ops_amd.py:367`).
pub fn event_write(event_type: u32, event_index: u32) -> [u32; 2] {
    [packet3(PACKET3_EVENT_WRITE, 0), (event_type & 0x3f) | ((event_index & 0xf) << 8)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packet3_header_layout() {
        // PACKET3(RELEASE_MEM=0x49, 6) = (3 << 30) | (0x49 << 8) | (6 << 16)
        let hdr = packet3(PACKET3_RELEASE_MEM, 6);
        assert_eq!(hdr, (3 << 30) | (0x49 << 8) | (6 << 16));
    }

    #[test]
    fn release_mem_packet_shape() {
        let pkt = release_mem(0x1234_5678_0000_4000, 42, true);
        assert_eq!(pkt.len(), 8);
        assert_eq!(pkt[3], 0x0000_4000); // addr lo
        assert_eq!(pkt[4], 0x1234_5678); // addr hi
        assert_eq!(pkt[5], 42); // value
        // memsel_dw must have DATA_SEL=1 in bits 29-31 (value 1 << 29)
        assert_eq!(pkt[2] & (0b111 << 29), 1 << 29);
        // INT_SEL=2 in bits 24-26
        assert_eq!((pkt[2] >> 24) & 0b111, 2);
    }

    #[test]
    fn set_sh_reg_header_layout() {
        // Two values at COMPUTE_PGM_LO (= PGM_LO + PGM_HI in one packet).
        let pkt = set_sh_reg(COMPUTE_PGM_LO, &[0x1234_5678, 0x9abc_def0]);
        assert_eq!(pkt.len(), 4);
        // header: count = 2 (number of data dwords excluding header AND reg_offset)
        assert_eq!(pkt[0], packet3(PACKET3_SET_SH_REG, 2));
        // reg_offset takes the low 16 bits
        assert_eq!(pkt[1], COMPUTE_PGM_LO);
        assert_eq!(pkt[2], 0x1234_5678);
        assert_eq!(pkt[3], 0x9abc_def0);
    }

    #[test]
    fn dispatch_direct_layout() {
        let di =
            DISPATCH_INITIATOR_FORCE_START_AT_000 | DISPATCH_INITIATOR_COMPUTE_SHADER_EN | DISPATCH_INITIATOR_CS_W32_EN;
        let pkt = dispatch_direct([4, 8, 16], di);
        // header + grid_x + grid_y + grid_z + dispatch_initiator = 5 dwords
        assert_eq!(pkt.len(), 5);
        assert_eq!(pkt[0], packet3(PACKET3_DISPATCH_DIRECT, 3));
        assert_eq!(pkt[1], 4);
        assert_eq!(pkt[2], 8);
        assert_eq!(pkt[3], 16);
        assert_eq!(pkt[4], di);
    }

    #[test]
    fn event_write_partial_flush_shape() {
        let pkt = event_write(CS_PARTIAL_FLUSH, EVENT_INDEX_PARTIAL_FLUSH);
        // header + (event_type | (event_index << 8))
        assert_eq!(pkt[0], packet3(PACKET3_EVENT_WRITE, 0));
        assert_eq!(pkt[1], CS_PARTIAL_FLUSH | (EVENT_INDEX_PARTIAL_FLUSH << 8));
    }

    #[test]
    fn hdp_flush_register_handshake_shape() {
        let pkt = hdp_flush();
        // 7 dwords: header + info + reg_req + reg_done + value + mask + poll
        assert_eq!(pkt.len(), 7);
        assert_eq!(pkt[0], packet3(PACKET3_WAIT_REG_MEM, 5));
        // info: mem_space=0 (register), operation=1, function=GEQ, engine=0
        let expected_info = wait_reg_mem_operation(1) | wait_reg_mem_function(WAIT_REG_MEM_FUNC_GEQ);
        assert_eq!(pkt[1], expected_info);
        assert_eq!(pkt[2], HDP_FLUSH_REQ_ADDR);
        assert_eq!(pkt[3], HDP_FLUSH_DONE_ADDR);
        assert_eq!(pkt[4], 0xFFFF_FFFF);
        assert_eq!(pkt[5], 0xFFFF_FFFF);
        assert_eq!(pkt[6], 4);
    }
}
