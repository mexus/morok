use super::*;
use enumset::EnumSet;

/// Commit a floating-point value to a concrete Tinygrad dtype grid.
///
/// The returned `f64` is the semantic value carried by the IR, not the storage
/// bits. `None` matches pinned Tinygrad's unsupported conversion cases.
pub fn commit_float(value: f64, dtype: ScalarDType) -> Option<f64> {
    use ScalarDType::*;

    // Tinygrad canonicalizes NaN before applying the dtype conversion.
    let value = if value.is_nan() { f64::NAN } else { value };
    Some(match dtype {
        WeakFloat | Float64 => value,
        Float32 => (value as f32) as f64,
        Float16 => f16_bits_to_float(float_to_f16_bits(value)),
        BFloat16 => {
            if !value.is_finite() {
                value
            } else {
                // float_to_bf16 uses struct.pack('f'), which rejects finite
                // values outside the f32 range rather than producing infinity.
                if value.to_bits() & 0x7fff_ffff_ffff_ffff >= 0x47ef_ffff_f000_0000 {
                    return None;
                }
                let bits = (value as f32).to_bits();
                let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
                f32::from_bits(rounded) as f64
            }
        }
        FP8E4M3 | FP8E5M2 | FP8E4M3FNUZ | FP8E5M2FNUZ => fp8_to_float(float_to_fp8(value, dtype)?, dtype)?,
        _ => return None,
    })
}

/// Encode a value using pinned Tinygrad's CUDA-derived FP8 conversion.
pub fn float_to_fp8(value: f64, dtype: ScalarDType) -> Option<u8> {
    use ScalarDType::*;

    let value = if value.is_nan() { f64::NAN } else { value };
    let fnuz = matches!(dtype, FP8E4M3FNUZ | FP8E5M2FNUZ);
    if fnuz && !value.is_finite() {
        return Some(0x80);
    }
    if fnuz && value == 0.0 {
        return Some(0x00);
    }
    if dtype == FP8E4M3 && !value.is_finite() {
        return Some(if value.is_sign_positive() { 0x7f } else { 0xff });
    }
    if dtype == FP8E5M2 && !value.is_finite() {
        let sign = if value.is_sign_positive() { 0 } else { 0x80 };
        return Some(sign | if value.is_infinite() { 0x7c } else { 0x7f });
    }

    let (bias, sig_bits, mant_mask, min_denorm_half, ovf_threshold, max_norm, min_norm) = match dtype {
        FP8E4M3 => {
            (7i32, 4u32, 0x7u64, 0x3f50_0000_0000_0000u64, 0x407d_0000_0000_0000u64, 0x7eu8, 0x3f90_0000_0000_0000u64)
        }
        FP8E5M2 => (15, 3, 0x3, 0x3ee0_0000_0000_0000, 0x40ee_0000_0000_0000 - 1, 0x7b, 0x3f10_0000_0000_0000),
        FP8E4M3FNUZ => (8, 4, 0x7, 0x3f40_0000_0000_0000, 0x406f_0000_0000_0000 - 1, 0x7f, 0x3f80_0000_0000_0000),
        FP8E5M2FNUZ => (16, 3, 0x3, 0x3ed0_0000_0000_0000, 0x40ee_0000_0000_0000 - 1, 0x7f, 0x3f00_0000_0000_0000),
        _ => return None,
    };
    let bits = value.to_bits();
    let half_ulp = 1u64 << (52 - sig_bits);
    let sign = (((bits >> 63) & 1) << 7) as u8;
    let exp = ((bits >> 52) & 0x7ff) as i32 - 1023 + bias;
    let mut mantissa = (bits >> (53 - sig_bits)) & mant_mask;
    let abs = bits & 0x7fff_ffff_ffff_ffff;
    let mut result: u64;

    if abs <= min_denorm_half {
        result = 0;
    } else if abs > ovf_threshold {
        result = max_norm as u64;
    } else if abs >= min_norm {
        result = ((exp as u64) << (sig_bits - 1)) | mantissa;
        let round_bits = bits & ((half_ulp << 1) - 1);
        if round_bits > half_ulp || (round_bits == half_ulp && mantissa & 1 != 0) {
            result += 1;
        }
    } else {
        let shift = (1 - exp) as u32;
        mantissa |= 1 << (sig_bits - 1);
        result = mantissa >> shift;
        let half = half_ulp << shift;
        let round_bits = (bits | (1 << 52)) & ((half << 1) - 1);
        if round_bits > half || (round_bits == half && result & 1 != 0) {
            result += 1;
        }
    }
    if fnuz && result == 0 { Some(0) } else { Some((result as u8) | sign) }
}

/// Decode one FP8 storage byte to the semantic value used by constants.
pub fn fp8_to_float(value: u8, dtype: ScalarDType) -> Option<f64> {
    use ScalarDType::*;

    let (bias, sig_bits, fnuz) = match dtype {
        FP8E4M3 => (7, 4, false),
        FP8E5M2 => (15, 3, false),
        FP8E4M3FNUZ => (8, 4, true),
        FP8E5M2FNUZ => (16, 3, true),
        _ => return None,
    };
    if fnuz && value == 0x80 {
        return Some(f64::NAN);
    }
    if value & 0x7f == 0 {
        return Some(if value & 0x80 != 0 { -0.0 } else { 0.0 });
    }
    let mant_bits = sig_bits - 1;
    let exp_bits = 8 - sig_bits;
    let exp_max = (1u8 << exp_bits) - 1;
    let mant_max = (1u8 << mant_bits) - 1;
    let sign = value >> 7;
    let exp = (value >> mant_bits) & exp_max;
    let mantissa = value & mant_max;
    if !fnuz && exp == exp_max {
        if dtype == FP8E5M2 {
            let result = if mantissa == 0 { f64::INFINITY } else { f64::NAN };
            return Some(if sign != 0 { -result } else { result });
        }
        if mantissa == mant_max {
            return Some(f64::NAN);
        }
    }
    let value = if exp == 0 {
        (mantissa as f64 / (mant_max as f64 + 1.0)) * 2f64.powi(1 - bias)
    } else {
        (1.0 + mantissa as f64 / (mant_max as f64 + 1.0)) * 2f64.powi(exp as i32 - bias)
    };
    Some(if sign != 0 { -value } else { value })
}

/// Storage bits for a value already committed to its declared float dtype.
pub fn committed_float_bits(value: f64, dtype: ScalarDType) -> Option<u64> {
    use ScalarDType::*;
    Some(match dtype {
        Float16 => float_to_f16_bits(value) as u64,
        BFloat16 => ((value as f32).to_bits() >> 16) as u64,
        Float32 => (value as f32).to_bits() as u64,
        WeakFloat | Float64 => value.to_bits(),
        FP8E4M3 | FP8E5M2 | FP8E4M3FNUZ | FP8E5M2FNUZ => float_to_fp8(value, dtype)? as u64,
        _ => return None,
    })
}

/// Direct f64-to-binary16 RNE conversion matching Python `struct.pack('e')`.
pub fn float_to_f16_bits(value: f64) -> u16 {
    fn round_shift(value: u64, shift: u32) -> u64 {
        if shift >= 64 {
            return 0;
        }
        let quotient = value >> shift;
        let remainder = value & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        quotient + u64::from(remainder > half || (remainder == half && quotient & 1 != 0))
    }

    let bits = value.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let raw_exp = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    if raw_exp == 0x7ff {
        return sign | if fraction == 0 { 0x7c00 } else { 0x7e00 };
    }
    if raw_exp == 0 && fraction == 0 {
        return sign;
    }

    let exponent = raw_exp - 1023;
    let significand = if raw_exp == 0 { fraction } else { fraction | (1 << 52) };
    if exponent > 15 || (exponent == 15 && significand >= ((1u64 << 52) | (0x3ffu64 << 42) | (1 << 41))) {
        return sign | 0x7c00;
    }
    if exponent >= -14 {
        let rounded = round_shift(significand, 42);
        if rounded == 0x800 {
            return sign | (((exponent + 16) as u16) << 10);
        }
        return sign | (((exponent + 15) as u16) << 10) | (rounded as u16 & 0x03ff);
    }
    if exponent < -25 {
        return sign;
    }
    sign | round_shift(significand, (28 - exponent) as u32) as u16
}

fn f16_bits_to_float(bits: u16) -> f64 {
    let sign = bits & 0x8000 != 0;
    let exponent = (bits >> 10) & 0x1f;
    let mantissa = bits & 0x03ff;
    let value = if exponent == 0x1f {
        if mantissa == 0 { f64::INFINITY } else { f64::NAN }
    } else if exponent == 0 {
        mantissa as f64 * 2f64.powi(-24)
    } else {
        (1.0 + mantissa as f64 / 1024.0) * 2f64.powi(exponent as i32 - 15)
    };
    if sign { -value } else { value }
}

impl ScalarDType {
    const fn promotion_rank(self) -> u8 {
        use ScalarDType::*;
        match self {
            Bool => 0,
            WeakInt => 1,
            Int8 => 2,
            UInt8 => 3,
            Int16 => 4,
            UInt16 => 5,
            Int32 => 6,
            UInt32 => 7,
            Int64 => 8,
            UInt64 => 9,
            WeakFloat => 10,
            FP8E4M3 => 11,
            FP8E4M3FNUZ => 12,
            FP8E5M2 => 13,
            FP8E5M2FNUZ => 14,
            Float16 => 15,
            BFloat16 => 16,
            Float32 => 17,
            Float64 => 18,
            Void => 19,
            Index => 20,
        }
    }

    const fn promotion_lattice(self) -> &'static [Self] {
        use ScalarDType::*;
        match self {
            Bool => &[WeakInt],
            WeakInt => &[Int8, UInt8],
            Int8 => &[Int16],
            Int16 => &[Int32],
            Int32 => &[Int64],
            Int64 => &[WeakFloat],
            UInt8 => &[Int16, UInt16],
            UInt16 => &[Int32, UInt32],
            UInt32 => &[Int64, UInt64],
            UInt64 => &[WeakFloat],
            WeakFloat => &[FP8E4M3, FP8E4M3FNUZ, FP8E5M2, FP8E5M2FNUZ],
            FP8E5M2 => &[Float16, BFloat16],
            FP8E4M3 => &[Float16, BFloat16],
            FP8E4M3FNUZ => &[Float16, BFloat16],
            FP8E5M2FNUZ => &[Float16, BFloat16],
            Float16 => &[Float32],
            BFloat16 => &[Float32],
            Float32 => &[Float64],
            Float64 | Void | Index => &[],
        }
    }

    fn get_recursive_parents(self) -> EnumSet<Self> {
        self.promotion_lattice()
            .iter()
            .fold(EnumSet::only(self), |dtypes, &parent| dtypes.union(parent.get_recursive_parents()))
    }

    /// Check if casting from `from` to `to` is safe (preserves value).
    pub fn can_safe_cast(self, to: Self) -> bool {
        // Port of tinygrad's can_lossless_cast.
        if self == to || matches!(self, Self::Bool) {
            return true;
        }

        // Index type: can cast from any integer to Index
        if matches!(to, Self::Index) {
            return self.is_int();
        }

        use ScalarDType::*;
        match to {
            WeakInt => self.is_signed() || self.is_unsigned(),
            Float64 => matches!(
                self,
                Float32
                    | Float16
                    | BFloat16
                    | FP8E4M3
                    | FP8E4M3FNUZ
                    | FP8E5M2
                    | FP8E5M2FNUZ
                    | UInt32
                    | UInt16
                    | UInt8
                    | Int32
                    | Int16
                    | Int8
            ),
            Float32 => matches!(
                self,
                Float16 | BFloat16 | FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | UInt16 | UInt8 | Int16 | Int8
            ),
            Float16 => matches!(self, FP8E4M3 | FP8E4M3FNUZ | FP8E5M2 | FP8E5M2FNUZ | UInt8 | Int8),
            UInt64 => matches!(self, UInt32 | UInt16 | UInt8),
            UInt32 => matches!(self, UInt16 | UInt8),
            UInt16 => matches!(self, UInt8),
            Int64 => matches!(self, UInt32 | UInt16 | UInt8 | Int32 | Int16 | Int8),
            Int32 => matches!(self, UInt16 | UInt8 | Int16 | Int8),
            Int16 => matches!(self, UInt8 | Int8),
            _ => false,
        }
    }

    /// Return the weak scalar of the same numeric kind.
    pub const fn weak(self) -> Self {
        if self.is_int() {
            Self::WeakInt
        } else if self.is_float() {
            Self::WeakFloat
        } else {
            self
        }
    }
}

impl DType {
    /// Check if casting from `from` to `to` is safe (preserves value).
    pub fn can_safe_cast(from: Self, to: Self) -> bool {
        // Extract scalars
        let (Some(from_scalar), Some(to_scalar)) = (from.scalar(), to.scalar()) else {
            return false;
        };

        // Check scalar cast is safe
        if !from_scalar.can_safe_cast(to_scalar) {
            return false;
        }

        // Vector counts must match (or broadcast from scalar)
        from.count() == to.count() || from.count() == 1 || to.count() == 1
    }

    /// Find the least upper bound type for a set of dtypes.
    ///
    /// Returns the smallest type that all input types can be safely cast to.
    ///
    /// Type promotion rules:
    /// - Scalar + Scalar → promoted Scalar
    /// - `Ptr<T>` + `Ptr<T>` → `Ptr<T>` (same Ptr types)
    /// - `Ptr<T>` + `Scalar(T)` → `Scalar(T)` (Ptr will be auto-loaded in codegen)
    /// - `Ptr<T>` + `Scalar(U)` → promoted Scalar (if T and U are compatible)
    pub fn least_upper_dtype(dtypes: &[Self]) -> Option<Self> {
        if dtypes.is_empty() {
            return None;
        }

        // Check for ImageDType first (they always win in promotion)
        if let Some(img) = dtypes.iter().find(|d| matches!(d, DType::Image { .. })) {
            return Some(img.clone());
        }

        // Check if all types are identical Ptr types
        let first = &dtypes[0];
        if matches!(first, DType::Ptr { .. }) && dtypes.iter().all(|d| d == first) {
            return Some(first.clone());
        }

        // Vector is only a mechanical wrapper around the scalar promotion rule.
        // Mixed scalar/vector inputs or mismatched vector widths are not a new
        // promotion rule and must be handled by the caller's shape logic.
        let vector_count = match &dtypes[0] {
            DType::Vector { count, .. } => Some(*count),
            _ => None,
        };
        if !dtypes.iter().all(|dtype| match (vector_count, dtype) {
            (Some(count), DType::Vector { count: other, .. }) => count == *other,
            (None, DType::Vector { .. }) | (Some(_), _) => false,
            (None, _) => true,
        }) {
            return None;
        }

        // Find common scalar type via promotion lattice intersection
        // Use base() to extract scalar from Ptr types for promotion
        // This allows Ptr<Float32> + Float32 → Float32
        let scalar_result = dtypes
            .iter()
            .map(|d| d.base())
            .map(|s| s.get_recursive_parents())
            .reduce(|lhs, rhs| lhs.intersection(rhs))?
            .iter()
            .min_by_key(|dtype| dtype.promotion_rank())?;

        Some(match vector_count {
            Some(count) => DType::Vector { scalar: scalar_result, count },
            None => DType::Scalar(scalar_result),
        })
    }

    /// Return the floating-point computation dtype for a scalar input.
    pub fn least_upper_float(dtype: Self) -> Option<Self> {
        let count = dtype.vcount();
        let scalar = dtype.scalar_dtype();
        let result = if scalar == Self::WeakInt {
            Self::WeakFloat
        } else if scalar.is_float() {
            scalar
        } else {
            Self::least_upper_dtype(&[scalar, Self::default_float()])?
        };
        result.vec(count)
    }

    pub fn strong_dtype(&self) -> Self {
        let (base, count) = match self {
            Self::Scalar(base) => (*base, 1),
            Self::Vector { scalar, count } => (*scalar, *count),
            _ => return self.clone(),
        };
        let scalar = match base {
            ScalarDType::WeakInt => Self::default_int(),
            ScalarDType::WeakFloat => Self::default_float(),
            _ => return self.clone(),
        };
        scalar.vec(count).expect("default dtype is scalar")
    }

    pub fn weak_dtype(&self) -> Self {
        match self {
            Self::Scalar(base) => Self::Scalar(base.weak()),
            Self::Vector { scalar, count } => Self::Scalar(scalar.weak()).vec(*count).expect("weak dtype is scalar"),
            _ => self.clone(),
        }
    }

    pub const fn default_int() -> Self {
        DEFAULT_INT
    }

    pub const fn default_float() -> Self {
        DEFAULT_FLOAT
    }
}
