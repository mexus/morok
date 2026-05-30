
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
