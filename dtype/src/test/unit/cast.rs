use super::*;
use strum::{EnumCount, VariantArray};

/// The `RECURSIVE_PARENTS` table must agree with the recursive lattice fold it
/// replaced, for every variant — including the FP8 ones whose shared
/// Float16/BFloat16/Float32/Float64 tail is what made the fold expensive.
#[test]
fn recursive_parents_table_matches_the_recursive_fold() {
    for &dtype in ScalarDType::VARIANTS {
        assert_eq!(
            dtype.get_recursive_parents(),
            dtype.recursive_parents_oracle(),
            "promotion parents diverge for {dtype:?}"
        );
    }
}

/// Indexing the table by discriminant is only sound while the discriminants are
/// dense and within `COUNT`.
#[test]
fn scalar_dtype_discriminants_are_dense() {
    let mut seen = vec![false; ScalarDType::COUNT];
    for &dtype in ScalarDType::VARIANTS {
        let index = dtype as usize;
        assert!(index < ScalarDType::COUNT, "{dtype:?} discriminant {index} is out of table range");
        assert!(!std::mem::replace(&mut seen[index], true), "{dtype:?} shares discriminant {index}");
    }
    assert!(seen.into_iter().all(|hit| hit), "discriminants must cover 0..COUNT");
}
