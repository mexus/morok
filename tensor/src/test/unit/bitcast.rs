//! Tests for `Tensor::bitcast` extension that handles size-changing dtype
//! reinterpretation (u32 ↔ u16, u32 ↔ u64). Round-trips must be byte-identical
//! to host memory's little-endian representation.

use morok_dtype::DType;

use crate::Tensor;

fn realize_u32(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<u32> {
    t.realize_with(config).expect("realize");
    t.as_vec::<u32>().expect("read")
}

fn realize_u16(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<u16> {
    t.realize_with(config).expect("realize");
    t.as_vec::<u16>().expect("read")
}

fn realize_u64(t: &mut Tensor, config: &crate::PrepareConfig) -> Vec<u64> {
    t.realize_with(config).expect("realize");
    t.as_vec::<u64>().expect("read")
}

crate::codegen_tests! {
    fn bitcast_identity_same_size(config) {
        let src = Tensor::from_slice([0x1234_5678u32, 0xDEAD_BEEF]);
        let mut as_u32 = src.bitcast(DType::UInt32).unwrap();
        assert_eq!(realize_u32(&mut as_u32, &config), vec![0x1234_5678, 0xDEAD_BEEF]);
    }

    fn bitcast_u32_to_u16_splits_little_endian(config) {
        let src = Tensor::from_slice([0x1234_5678u32, 0xDEAD_BEEF]);
        let mut as_u16 = src.bitcast(DType::UInt16).unwrap();
        // u32 word 0x12345678 is bytes [0x78, 0x56, 0x34, 0x12] in LE → u16s 0x5678, 0x1234.
        // u32 word 0xDEADBEEF                                       → u16s 0xBEEF, 0xDEAD.
        assert_eq!(realize_u16(&mut as_u16, &config), vec![0x5678, 0x1234, 0xBEEF, 0xDEAD]);
    }

    fn bitcast_u32_to_u64_combines_little_endian(config) {
        let src = Tensor::from_slice([0x1234_5678u32, 0xDEAD_BEEF, 0xCAFE_BABE, 0xF00D_FACE]);
        let mut as_u64 = src.bitcast(DType::UInt64).unwrap();
        // Pair (0x12345678, 0xDEADBEEF) → 0xDEADBEEF_12345678 (LE: low word first).
        // Pair (0xCAFEBABE, 0xF00DFACE) → 0xF00DFACE_CAFEBABE.
        assert_eq!(realize_u64(&mut as_u64, &config), vec![0xDEAD_BEEF_1234_5678, 0xF00D_FACE_CAFE_BABE]);
    }

    fn bitcast_u32_u16_u32_round_trip(config) {
        let src_values = [0u32, 1, u32::MAX, 0x1234_5678, 0xFFFF_0000, 0x0000_FFFF];
        let src = Tensor::from_slice(src_values);
        let intermediate = src.bitcast(DType::UInt16).unwrap();
        let mut back = intermediate.bitcast(DType::UInt32).unwrap();
        assert_eq!(realize_u32(&mut back, &config), src_values.to_vec());
    }

    fn bitcast_u32_u64_u32_round_trip(config) {
        // Even count so u32 → u64 doesn't drop anything.
        let src_values = [0u32, 1, u32::MAX, 0x1234_5678, 0xFFFF_0000, 0x0000_FFFF];
        let src = Tensor::from_slice(src_values);
        let intermediate = src.bitcast(DType::UInt64).unwrap();
        let mut back = intermediate.bitcast(DType::UInt32).unwrap();
        assert_eq!(realize_u32(&mut back, &config), src_values.to_vec());
    }
}

// Validation-only test — no realize, no codegen variants needed.
#[test]
fn bitcast_rejects_indivisible_last_dim() {
    // 3 u32 elements would need 1.5 u64s — not divisible.
    let src = Tensor::from_slice([1u32, 2, 3]);
    assert!(
        src.bitcast(DType::UInt64).is_err(),
        "bitcast should reject when (shape[-1] * src_size) is not divisible by dst_size"
    );
}
