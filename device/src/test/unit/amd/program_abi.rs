//! ABI-retention preconditions for `AmdProgram::load`: the descriptor list is
//! kept verbatim (interleaved storage/scalar order), sparse storage slots pack
//! to compact kernarg ordinals, and malformed descriptors are rejected before
//! any ELF is touched.

use crate::amd::program::retain_program_abi;
use crate::device::{AbiParamDescriptor, AbiParamKind};
use svod_dtype::{AddrSpace, DType};
#[test]
fn loaded_program_retains_interleaved_abi_exactly() {
    let abi = vec![
        AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor { slot: 1, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("low".into()) },
        AbiParamDescriptor {
            slot: 2,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor { slot: 3, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("high".into()) },
    ];

    let (retained, buf_count, var_count) = retain_program_abi(&abi).unwrap();

    assert_eq!(retained, abi);
    assert_eq!((buf_count, var_count), (2, 2));
}

#[test]
fn amd_kernarg_packing_uses_compact_storage_ordinals_for_sparse_slots() {
    let abi = vec![
        AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
        AbiParamDescriptor { slot: 1, kind: AbiParamKind::Scalar, dtype: DType::Int32, name: Some("n".into()) },
        AbiParamDescriptor {
            slot: 5,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Float32,
            name: None,
        },
    ];
    let mut dst = [0u8; 24];
    crate::hcq::ClikeKernargLayout::from_abi(&abi).pack(&mut dst, &[0x1000, 0x5000], &[7]).unwrap();

    assert_eq!(&dst[..8], &0x1000u64.to_le_bytes());
    assert_eq!(&dst[8..12], &7i32.to_le_bytes());
    assert_eq!(&dst[16..24], &0x5000u64.to_le_bytes());
}

#[test]
fn loaded_program_rejects_malformed_descriptors_before_elf_loading() {
    let storage = |slot, name| AbiParamDescriptor {
        slot,
        kind: AbiParamKind::Storage(AddrSpace::Global),
        dtype: DType::Float32,
        name,
    };
    let scalar = |slot, dtype, name| AbiParamDescriptor { slot, kind: AbiParamKind::Scalar, dtype, name };
    let malformed = [
        vec![storage(1, None), storage(0, None)],
        vec![storage(0, None), storage(0, None)],
        vec![storage(usize::MAX, None)],
        vec![scalar(0, DType::Float32, Some("n".into()))],
        vec![scalar(0, DType::Int32, None)],
        vec![storage(0, Some("not_storage".into()))],
        vec![AbiParamDescriptor {
            slot: 0,
            kind: AbiParamKind::Storage(AddrSpace::Global),
            dtype: DType::Void,
            name: None,
        }],
    ];

    for abi in malformed {
        let err = retain_program_abi(&abi).expect_err("AmdProgram::load ABI precondition must reject malformed input");
        assert!(
            matches!(&err, crate::Error::ProgramAbiMismatch { .. } | crate::Error::UnassignedProgramParam { .. }),
            "{err:?}"
        );
    }
}
