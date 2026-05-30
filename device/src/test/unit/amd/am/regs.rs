use super::*;

#[test]
fn selects_gfx1151_gc_module() {
    // Strix Halo reports GC 11.5.1 → the gc_11_5_0 table.
    let gc = select("gc", (11, 5, 1)).expect("gc module for gfx1151");
    assert!(!gc.is_empty());
    // A wrong major has no match (we don't vendor gfx9/12 gc here yet).
    assert!(select("gc", (9, 4, 2)).is_none());
    assert!(select("gc", (12, 0, 0)).is_none());
}

#[test]
fn resolves_known_register_and_fields() {
    let gc = select("gc", (11, 5, 1)).unwrap();
    let grbm = find(gc, "regGRBM_STATUS").expect("regGRBM_STATUS present");
    assert_eq!(grbm.offset, 3492);
    assert_eq!(grbm.segment, 0);
    let gui = grbm.field("gui_active").expect("gui_active field");
    assert_eq!((gui.lo, gui.hi), (31, 31));
}

#[test]
fn encode_decode_round_trip() {
    let gc = select("gc", (11, 5, 1)).unwrap();
    let grbm = find(gc, "regGRBM_STATUS").unwrap();
    let word = grbm.encode(&[("gui_active", 1), ("cp_busy", 1)]);
    assert_eq!(word, (1u64 << 31) | (1u64 << 29));
    assert_eq!(grbm.get(word, "gui_active"), 1);
    assert_eq!(grbm.get(word, "cp_busy"), 1);
    assert_eq!(grbm.get(word, "cb_busy"), 0);
}

#[test]
fn multi_bit_field_masks_correctly() {
    let gc = select("gc", (11, 5, 1)).unwrap();
    // regGRBM_STATUS.me0pipe0_cmdfifo_avail is a 4-bit field [0,3].
    let grbm = find(gc, "regGRBM_STATUS").unwrap();
    let word = grbm.encode(&[("me0pipe0_cmdfifo_avail", 0xF)]);
    assert_eq!(grbm.get(word, "me0pipe0_cmdfifo_avail"), 0xF);
    // High bits of an over-wide value don't bleed past the field on decode.
    assert_eq!(grbm.get(0xFF, "me0pipe0_cmdfifo_avail"), 0xF);
}
