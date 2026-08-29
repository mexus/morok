#![cfg(test)]

use crate::{
    DEFAULT_FLOAT, DEFAULT_INT, DType, ScalarDType,
    cast::{commit_float, committed_float_bits, float_to_fp8, fp8_to_float},
};

#[test]
fn tinygrad_weak_promotion_lattice() {
    assert_eq!(DType::least_upper_dtype(&[DType::WeakInt, DType::WeakInt]), Some(DType::WeakInt));
    assert_eq!(DType::least_upper_dtype(&[DType::Bool, DType::WeakInt]), Some(DType::WeakInt));
    assert_eq!(DType::least_upper_dtype(&[DType::WeakInt, DType::Int8]), Some(DType::Int8));
    assert_eq!(DType::least_upper_dtype(&[DType::Int64, DType::UInt64]), Some(DType::WeakFloat));
    assert_eq!(DType::least_upper_dtype(&[DType::WeakFloat, DType::Float16]), Some(DType::Float16));
    assert_eq!(DType::least_upper_dtype(&[DType::WeakFloat, DType::FP8E4M3FNUZ]), Some(DType::FP8E4M3FNUZ));
    assert_eq!(DType::least_upper_dtype(&[DType::FP8E4M3FNUZ, DType::FP8E5M2FNUZ]), Some(DType::Float16));
    assert_eq!(DType::least_upper_dtype(&[DType::FP8E4M3, DType::FP8E4M3FNUZ]), Some(DType::Float16));
    assert_eq!(DType::least_upper_float(DType::WeakInt), Some(DType::WeakFloat));
    assert_eq!(
        DType::least_upper_dtype(&[DType::WeakInt.vec(4).unwrap(), DType::Int16.vec(4).unwrap()]),
        DType::Int16.vec(4)
    );
}

#[test]
fn tinygrad_strong_and_weak_dtype() {
    assert_eq!(DType::WeakInt.strong_dtype(), DType::Int32);
    assert_eq!(DType::WeakFloat.strong_dtype(), DType::Float32);
    assert_eq!(DEFAULT_INT, DType::Int32);
    assert_eq!(DEFAULT_FLOAT, DType::Float32);
    assert_eq!(DType::Int16.weak_dtype(), DType::WeakInt);
    assert_eq!(DType::Float64.weak_dtype(), DType::WeakFloat);
    assert!(DType::WeakInt.is_int());
    assert!(DType::WeakFloat.is_float());
    assert!(DType::WeakInt.vec(4).unwrap().is_int());
    assert!(DType::WeakFloat.vec(8).unwrap().is_float());
    assert_eq!(DType::WeakInt.vec(4).unwrap().strong_dtype(), DType::Int32.vec(4).unwrap());
    assert_eq!(DType::WeakFloat.vec(8).unwrap().strong_dtype(), DType::Float32.vec(8).unwrap());
    assert_eq!(DType::Int16.vec(4).unwrap().weak_dtype(), DType::WeakInt.vec(4).unwrap());
    assert_eq!(DType::Float64.vec(8).unwrap().weak_dtype(), DType::WeakFloat.vec(8).unwrap());
    assert_eq!(ScalarDType::WeakInt.bytes(), 100);
    assert_eq!(ScalarDType::WeakFloat.bitsize(), 800);
}

#[test]
fn tinygrad_fnuz_metadata() {
    assert_eq!(ScalarDType::FP8E4M3FNUZ.bytes(), 1);
    assert_eq!(ScalarDType::FP8E5M2FNUZ.bytes(), 1);
    assert_eq!(ScalarDType::FP8E4M3FNUZ.finfo(), Some((4, 3)));
    assert_eq!(ScalarDType::FP8E5M2FNUZ.finfo(), Some((5, 2)));
    assert_eq!(ScalarDType::FP8E4M3FNUZ.c_style(), "float8_e4m3fnuz");
    assert_eq!(ScalarDType::FP8E5M2FNUZ.c_style(), "float8_e5m2fnuz");
    assert!(ScalarDType::FP8E4M3FNUZ.is_fp8_fnuz());
}

#[test]
fn finite_format_limits_are_separate_from_analysis_bounds() {
    let floats = [
        DType::WeakFloat,
        DType::FP8E4M3,
        DType::FP8E5M2,
        DType::FP8E4M3FNUZ,
        DType::FP8E5M2FNUZ,
        DType::Float16,
        DType::BFloat16,
        DType::Float32,
        DType::Float64,
    ];
    for dtype in floats {
        assert!(dtype.min_value().is_finite(), "{dtype:?} format minimum");
        assert!(dtype.max_value().is_finite(), "{dtype:?} format maximum");
        assert_eq!(dtype.analysis_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
        assert_eq!(dtype.vec(4).unwrap().analysis_bounds(), (f64::NEG_INFINITY, f64::INFINITY));
    }

    assert_eq!(DType::Int8.analysis_bounds(), (i8::MIN as f64, i8::MAX as f64));
    assert_eq!(DType::UInt16.analysis_bounds(), (0.0, u16::MAX as f64));
}

#[test]
fn tinygrad_lossless_cast_table() {
    assert!(!ScalarDType::Int8.can_safe_cast(ScalarDType::UInt64));
    assert!(!ScalarDType::Int32.can_safe_cast(ScalarDType::UInt32));
    assert!(ScalarDType::UInt8.can_safe_cast(ScalarDType::Int16));
    assert!(ScalarDType::UInt32.can_safe_cast(ScalarDType::Int64));
    assert!(!ScalarDType::Int32.can_safe_cast(ScalarDType::Float32));
    assert!(!ScalarDType::Int64.can_safe_cast(ScalarDType::Float64));
    assert!(ScalarDType::Int8.can_safe_cast(ScalarDType::Float16));
    assert!(!ScalarDType::Int8.can_safe_cast(ScalarDType::BFloat16));
}

fn next_down(value: f64) -> f64 {
    f64::from_bits(value.to_bits() - 1)
}

fn next_up(value: f64) -> f64 {
    f64::from_bits(value.to_bits() + 1)
}

#[test]
fn tinygrad_ieee_and_bfloat_midpoint_grids() {
    let cases = [
        (ScalarDType::Float16, 1.0 + 2f64.powi(-11), 1.0, 1.0 + 2f64.powi(-10)),
        (ScalarDType::Float32, 1.0 + 2f64.powi(-24), 1.0, 1.0 + 2f64.powi(-23)),
    ];
    for (dtype, midpoint, lower, upper) in cases {
        assert_eq!(commit_float(next_down(midpoint), dtype), Some(lower), "{dtype:?} midpoint - 1 ulp");
        assert_eq!(commit_float(midpoint, dtype), Some(lower), "{dtype:?} midpoint tie-to-even");
        assert_eq!(commit_float(next_up(midpoint), dtype), Some(upper), "{dtype:?} midpoint + 1 ulp");
    }
    let bf_midpoint = 1.0f32 + 2f32.powi(-8);
    assert_eq!(commit_float(f32::from_bits(bf_midpoint.to_bits() - 1) as f64, ScalarDType::BFloat16), Some(1.0));
    assert_eq!(commit_float(bf_midpoint as f64, ScalarDType::BFloat16), Some(1.0));
    assert_eq!(
        commit_float(f32::from_bits(bf_midpoint.to_bits() + 1) as f64, ScalarDType::BFloat16),
        Some(1.0 + 2f64.powi(-7))
    );
    assert_eq!(commit_float(std::f64::consts::PI, ScalarDType::Float64), Some(std::f64::consts::PI));
}

#[test]
fn tinygrad_reduced_float_edges_and_exact_storage_bits() {
    let nan_bits = f64::NAN.to_bits();
    for dtype in [ScalarDType::Float16, ScalarDType::BFloat16, ScalarDType::Float32, ScalarDType::Float64] {
        assert_eq!(commit_float(f64::from_bits(0xfff8_1234_5678_9abc), dtype).unwrap().to_bits(), nan_bits);
        assert!(commit_float(f64::INFINITY, dtype).unwrap().is_infinite());
        assert!(commit_float(f64::NEG_INFINITY, dtype).unwrap().is_sign_negative());
        assert_eq!(commit_float(-0.0, dtype).unwrap().to_bits(), (-0.0f64).to_bits());
    }

    assert_eq!(
        committed_float_bits(commit_float(1.0 + 2f64.powi(-11), ScalarDType::Float16).unwrap(), ScalarDType::Float16),
        Some(0x3c00)
    );
    assert_eq!(
        committed_float_bits(commit_float(1.0 + 2f64.powi(-8), ScalarDType::BFloat16).unwrap(), ScalarDType::BFloat16),
        Some(0x3f80)
    );
    assert_eq!(
        committed_float_bits(commit_float(1.0 + 2f64.powi(-24), ScalarDType::Float32).unwrap(), ScalarDType::Float32),
        Some(0x3f80_0000)
    );

    assert_eq!(commit_float(2f64.powi(-25), ScalarDType::Float16), Some(0.0));
    assert_eq!(commit_float(next_up(2f64.powi(-25)), ScalarDType::Float16), Some(2f64.powi(-24)));
    assert_eq!(commit_float(65_520.0, ScalarDType::Float16), Some(f64::INFINITY));
    assert_eq!(commit_float(1e300, ScalarDType::Float16), Some(f64::INFINITY));

    assert_eq!(commit_float(2f64.powi(-134), ScalarDType::BFloat16), Some(0.0));
    let bf_underflow_midpoint = f32::from_bits(0x0000_8000);
    assert_eq!(
        commit_float(f32::from_bits(bf_underflow_midpoint.to_bits() + 1) as f64, ScalarDType::BFloat16),
        Some(2f64.powi(-133))
    );
    // Divergence from tinygrad's `float_to_bf16`, which raises OverflowError past the
    // f32 rounding midpoint: morok saturates so `truncate` is total (IB1).
    assert_eq!(commit_float(1e300, ScalarDType::BFloat16), Some(f64::INFINITY));
    assert_eq!(commit_float(-1e300, ScalarDType::BFloat16), Some(f64::NEG_INFINITY));
    let f32_max = f32::MAX as f64;
    assert!(commit_float(next_up(f32_max), ScalarDType::BFloat16).unwrap().is_infinite());
    assert_eq!(commit_float(f64::from_bits(0x47ef_ffff_f000_0000), ScalarDType::BFloat16), Some(f64::INFINITY));
}

fn fp8_grid_hash(dtype: ScalarDType) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in 0..=u8::MAX {
        for part in fp8_to_float(byte, dtype).unwrap().to_bits().to_le_bytes() {
            hash ^= part as u64;
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }
    hash
}

#[test]
fn tinygrad_fp8_decode_grids_are_exact() {
    // FNV-1a over all 256 semantic f64 bit patterns, generated from the pinned
    // tinygrad fp8_to_float implementation at 8c8b43de.
    assert_eq!(fp8_grid_hash(ScalarDType::FP8E4M3), 0x4d4d_ed90_c4f5_d9b5);
    assert_eq!(fp8_grid_hash(ScalarDType::FP8E5M2), 0x0607_c8f7_b453_ef15);
    assert_eq!(fp8_grid_hash(ScalarDType::FP8E4M3FNUZ), 0x33a0_6c4d_1199_92b0);
    assert_eq!(fp8_grid_hash(ScalarDType::FP8E5M2FNUZ), 0x08d9_906c_c5ea_2650);
}

#[test]
fn tinygrad_fp8_midpoints_specials_and_saturation() {
    for dtype in [ScalarDType::FP8E4M3, ScalarDType::FP8E5M2, ScalarDType::FP8E4M3FNUZ, ScalarDType::FP8E5M2FNUZ] {
        for lower_bits in [0x00, 0x01, 0x08, 0x21, 0x3f] {
            let lower = fp8_to_float(lower_bits, dtype).unwrap();
            let upper = fp8_to_float(lower_bits + 1, dtype).unwrap();
            let midpoint = (lower + upper) / 2.0;
            let tie = if lower_bits & 1 == 0 { lower_bits } else { lower_bits + 1 };
            assert_eq!(float_to_fp8(next_down(midpoint), dtype), Some(lower_bits));
            assert_eq!(float_to_fp8(midpoint, dtype), Some(tie));
            assert_eq!(float_to_fp8(next_up(midpoint), dtype), Some(lower_bits + 1));
        }
    }

    assert_eq!(float_to_fp8(-0.0, ScalarDType::FP8E4M3), Some(0x80));
    assert_eq!(float_to_fp8(-0.0, ScalarDType::FP8E5M2), Some(0x80));
    assert_eq!(float_to_fp8(-0.0, ScalarDType::FP8E4M3FNUZ), Some(0x00));
    assert_eq!(float_to_fp8(-0.0, ScalarDType::FP8E5M2FNUZ), Some(0x00));
    assert_eq!(float_to_fp8(f64::INFINITY, ScalarDType::FP8E4M3), Some(0x7f));
    assert_eq!(float_to_fp8(f64::INFINITY, ScalarDType::FP8E5M2), Some(0x7c));
    assert_eq!(float_to_fp8(f64::INFINITY, ScalarDType::FP8E4M3FNUZ), Some(0x80));
    assert_eq!(float_to_fp8(f64::NAN, ScalarDType::FP8E5M2FNUZ), Some(0x80));
    assert_eq!(float_to_fp8(1e300, ScalarDType::FP8E4M3), Some(0x7e));
    assert_eq!(float_to_fp8(1e300, ScalarDType::FP8E5M2), Some(0x7b));
    assert_eq!(float_to_fp8(1e300, ScalarDType::FP8E4M3FNUZ), Some(0x7f));
    assert_eq!(float_to_fp8(1e300, ScalarDType::FP8E5M2FNUZ), Some(0x7f));
}
