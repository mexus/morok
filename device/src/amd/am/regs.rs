//! AM register access: vendored register tables + the resolver.
//!
//! The tables in `regs_gen.rs` are generated **once** (by
//! `device/tools/gen_am_regs.py`, from tinygrad's `autogen/am/regs.py`) and
//! committed — the cargo build never depends on the tinygrad submodule. At boot
//! the right module is chosen by the discovered `ip_ver` ([`select`]), mirroring
//! tinygrad's `import_asic_regs`. A `RegDef` carries the dword offset within its
//! IP segment plus its bitfields; the absolute MMIO address (segment base +
//! offset) is resolved later from the IP-discovery register bases.

/// One register field: an inclusive bit range `[lo, hi]`.
#[derive(Clone, Copy, Debug)]
pub struct RegField {
    pub name: &'static str,
    pub lo: u8,
    pub hi: u8,
}

/// A register definition: dword `offset` within IP `segment`, plus bitfields.
#[derive(Clone, Copy, Debug)]
pub struct RegDef {
    pub name: &'static str,
    pub offset: u32,
    pub segment: u8,
    pub fields: &'static [RegField],
}

/// A vendored register module: `(prefix, (maj, min, rev), table)`.
pub type RegModule = (&'static str, (u8, u8, u8), &'static [RegDef]);

include!("regs_gen.rs");

impl RegDef {
    /// The named field, if present.
    pub fn field(&self, name: &str) -> Option<&RegField> {
        self.fields.iter().find(|f| f.name == name)
    }

    /// OR together `value << field.lo` for each named field (port of
    /// `AMDReg.encode`). Panics on an unknown field name (a programming error).
    pub fn encode(&self, values: &[(&str, u32)]) -> u64 {
        let mut acc = 0u64;
        for (name, v) in values {
            let f = self.field(name).unwrap_or_else(|| panic!("register {} has no field {name}", self.name));
            acc |= (*v as u64) << f.lo;
        }
        acc
    }

    /// Extract a field's value from a register word (port of `AMDReg.decode`).
    pub fn get(&self, val: u64, name: &str) -> u32 {
        let f = self.field(name).unwrap_or_else(|| panic!("register {} has no field {name}", self.name));
        let width = (f.hi - f.lo + 1) as u32;
        let mask = if width >= 64 { u64::MAX } else { (1u64 << width) - 1 };
        ((val >> f.lo) & mask) as u32
    }
}

/// Find a register by name within a module's table (linear scan; not hot —
/// callers resolve a register once at setup).
pub fn find<'a>(regs: &'a [RegDef], name: &str) -> Option<&'a RegDef> {
    regs.iter().find(|r| r.name == name)
}

/// Select the register module for `prefix` (e.g. `"gc"`) whose version is the
/// greatest `<= ip_ver` sharing the same major — tinygrad's `import_module`
/// rule. `None` if no matching module is vendored (re-run the generator with a
/// wider module list to add one).
pub fn select(prefix: &str, ip_ver: (u8, u8, u8)) -> Option<&'static [RegDef]> {
    AM_REG_MODULES
        .iter()
        .filter(|(p, v, _)| *p == prefix && v.0 == ip_ver.0 && *v <= ip_ver)
        .max_by_key(|(_, v, _)| *v)
        .map(|(_, _, regs)| *regs)
}

#[cfg(test)]
mod tests {
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
}
