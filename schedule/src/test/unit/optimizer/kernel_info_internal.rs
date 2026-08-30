use super::*;

/// `function_name` strips ANSI decoration and hex-escapes anything else that is not
/// a valid identifier character.
#[test_case::test_case("test_kernel", "test_kernel"; "already an identifier")]
#[test_case::test_case("r_g16l16R32u4", "r_g16l16R32u4"; "kernel name")]
#[test_case::test_case("r\x1b[34mg16\x1b[0m", "rg16"; "colour codes are dropped")]
#[test_case::test_case("E_\x1b[31mL?\x1b[0mn6\x1b[K", "E_L3Fn6"; "erase-line code is dropped, question mark escaped")]
#[test_case::test_case("test-kernel+v2", "test2Dkernel2Bv2"; "punctuation is hex escaped")]
fn function_name_is_a_valid_identifier(name: &str, expected: &str) {
    assert_eq!(KernelInfo::new(name, vec![], false).function_name(), expected);
}
