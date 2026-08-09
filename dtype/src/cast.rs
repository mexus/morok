use super::*;
use enumset::EnumSet;

impl ScalarDType {
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
            WeakFloat => &[FP8E4M3, FP8E5M2],
            FP8E5M2 => &[Float16, BFloat16],
            FP8E4M3 => &[Float16, BFloat16],
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
                Float32 | Float16 | BFloat16 | FP8E4M3 | FP8E5M2 | UInt32 | UInt16 | UInt8 | Int32 | Int16 | Int8
            ),
            Float32 => matches!(self, Float16 | BFloat16 | FP8E4M3 | FP8E5M2 | UInt16 | UInt8 | Int16 | Int8),
            Float16 => matches!(self, FP8E4M3 | FP8E5M2 | UInt8 | Int8),
            UInt64 => matches!(self, UInt32 | UInt16 | UInt8),
            UInt32 => matches!(self, UInt16 | UInt8),
            UInt16 => matches!(self, UInt8),
            Int64 => matches!(self, UInt32 | UInt16 | UInt8 | Int32 | Int16 | Int8),
            Int32 => matches!(self, UInt16 | UInt8 | Int16 | Int8),
            Int16 => matches!(self, UInt8 | Int8),
            _ => false,
        }
    }

    /// Commit a weak scalar to the current Tinygrad defaults.
    pub const fn strong(self) -> Self {
        match self {
            Self::WeakInt => Self::Int32,
            Self::WeakFloat => Self::Float32,
            _ => self,
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

        // Find common scalar type via promotion lattice intersection
        // Use base() to extract scalar from Ptr types for promotion
        // This allows Ptr<Float32> + Float32 → Float32
        let scalar_result = dtypes
            .iter()
            .map(|d| d.base())
            .map(|s| s.get_recursive_parents())
            .reduce(|lhs, rhs| lhs.intersection(rhs))?
            .iter()
            .min()?; // min by discriminant (= priority: lower = more specific)

        // Svod extension: preserve vector count if all inputs have the same vcount > 1.
        // Tinygrad's least_upper_dtype always returns scalar; we extend it to preserve
        // vector width when all operands agree, avoiding unnecessary devectorize/revectorize.
        let vcount = dtypes[0].vcount();
        if vcount > 1 && dtypes.iter().all(|d| d.vcount() == vcount) {
            Some(DType::Vector { scalar: scalar_result, count: vcount })
        } else {
            Some(DType::Scalar(scalar_result))
        }
    }

    /// Return the floating-point computation dtype for a scalar input.
    pub fn least_upper_float(dtype: Self) -> Option<Self> {
        if dtype == Self::WeakInt {
            Some(Self::WeakFloat)
        } else if dtype.is_float() || dtype == Self::WeakFloat {
            Some(dtype)
        } else {
            Self::least_upper_dtype(&[dtype, Self::Float32])
        }
    }

    pub fn strong_dtype(&self) -> Self {
        match self {
            Self::Scalar(s) => Self::Scalar(s.strong()),
            _ => self.clone(),
        }
    }

    pub fn weak_dtype(&self) -> Self {
        match self {
            Self::Scalar(s) => Self::Scalar(s.weak()),
            _ => self.clone(),
        }
    }
}
