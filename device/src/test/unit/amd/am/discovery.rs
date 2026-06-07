use super::*;

fn wr16(b: &mut [u8], off: usize, v: u16) {
    b[off..off + 2].copy_from_slice(&v.to_le_bytes());
}
fn wr32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
fn wr64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}

/// Build a minimal MI300X-shaped table: 1 die, 8 GC instances (9,4,3) with
/// 64-bit bases, 1 OSSSYS (4,4,2), gc_info v2.
fn synthetic_table() -> Vec<u8> {
    let mut t = vec![0u8; TABLE_SIZE];
    wr32(&mut t, 0, BINARY_SIGNATURE);
    let (ihdr, gc) = (0x100usize, 0x800usize);
    wr16(&mut t, 12, ihdr as u16); // table_list[IP_DISCOVERY].offset
    wr16(&mut t, 12 + 8, gc as u16); // table_list[GC].offset

    wr32(&mut t, ihdr, DISCOVERY_TABLE_SIGNATURE);
    wr16(&mut t, ihdr + 12, 1); // num_dies
    let die = 0x200usize;
    wr16(&mut t, ihdr + 14 + 2, die as u16); // die_info[0].die_offset
    t[ihdr + 78] = 1; // base_addr_64_bit

    wr16(&mut t, die + 2, 9); // num_ips
    let mut ip = die + 4;
    for inst in 0..8u8 {
        wr16(&mut t, ip, 11); // GC hw_id
        t[ip + 2] = inst;
        t[ip + 3] = 2; // num_base_address
        (t[ip + 4], t[ip + 5], t[ip + 6]) = (9, 4, 3);
        wr64(&mut t, ip + 8, 0x2000 + inst as u64 * 0x100);
        wr64(&mut t, ip + 16, 0xA000 + inst as u64 * 0x100);
        ip += 8 + 16;
    }
    wr16(&mut t, ip, 40); // OSSSYS
    t[ip + 3] = 1;
    (t[ip + 4], t[ip + 5], t[ip + 6]) = (4, 4, 2);
    wr64(&mut t, ip + 8, 0x4280);

    wr16(&mut t, gc + 4, 2); // gc_info v2
    for (i, v) in [4u32, 19, 1, 0, 0, 0, 0, 0, 0, 0, 0, 64, 8, 32, 64].iter().enumerate() {
        wr32(&mut t, gc + 12 + i * 4, *v);
    }
    t
}

#[test]
fn parses_synthetic_mi300x_table() {
    let d = Discovery::parse(&synthetic_table(), 192 << 30).unwrap();
    assert_eq!(d.ip_ver[&GC_HWIP], (9, 4, 3));
    assert_eq!(d.ip_ver[&OSSSYS_HWIP], (4, 4, 2));
    assert_eq!(d.xccs(), 8);
    assert_eq!(d.regs_offset[&GC_HWIP][&3], vec![0x2300, 0xA300]);
    assert_eq!(d.reserved_vram_size, 384 << 20);
    assert_eq!(d.gc.num_se, 4);
    assert_eq!(d.gc.num_cu_per_sh, 19);
    assert_eq!(d.gc.wave_size, 64);
    assert_eq!(d.gc.max_waves_per_simd, 8);
    assert_eq!(d.gc.lds_size_kb, 64);
}

#[test]
fn rejects_bad_signature() {
    let mut t = synthetic_table();
    t[0] = 0;
    assert!(Discovery::parse(&t, 1 << 30).is_err());
}

#[test]
fn reg_addr_resolves_via_bases() {
    let d = Discovery::parse(&synthetic_table(), 192 << 30).unwrap();
    let gc = crate::amd::am::regs::select("gc", d.ip_ver[&GC_HWIP]).unwrap();
    let hqd = crate::amd::am::regs::find(gc, "regCP_HQD_ACTIVE").unwrap();
    let bases = &d.regs_offset[&GC_HWIP][&0];
    assert_eq!(hqd.dword_index(bases), 0x2000 + hqd.offset as usize);
}
