use super::*;

fn abi(buffers: usize, vars: &[&str]) -> Vec<svod_device::device::AbiParamDescriptor> {
    use svod_device::device::{AbiParamDescriptor, AbiParamKind};
    (0..buffers)
        .map(|slot| AbiParamDescriptor {
            slot,
            kind: AbiParamKind::Storage(svod_dtype::AddrSpace::Global),
            dtype: svod_dtype::DType::Float32,
            name: None,
        })
        .chain(vars.iter().enumerate().map(|(index, name)| AbiParamDescriptor {
            slot: buffers + index,
            kind: AbiParamKind::Scalar,
            dtype: svod_dtype::DType::Int32,
            name: Some((*name).into()),
        }))
        .collect()
}

#[test]
fn test_jit_loader_noop() {
    let src = "void test_kernel(void) { }\n";
    let kernel = JitKernel::compile_with_abi(src, "test_kernel", vec![], &abi(0, &[])).unwrap();
    assert_eq!(kernel.name(), "test_kernel");
    unsafe {
        kernel.execute_with_vals(&[], &[]).unwrap();
    }
}

#[test]
fn test_jit_loader_add() {
    let src = r#"
void add_kernel(float* restrict a, float* restrict b, float* restrict out) {
    out[0] = a[0] + b[0];
}
"#;
    let kernel = JitKernel::compile_with_abi(src, "add_kernel", vec![], &abi(3, &[])).unwrap();

    let mut a = [1.0f32];
    let mut b = [2.0f32];
    let mut out = [0.0f32];

    let buffers = vec![a.as_mut_ptr() as *mut u8, b.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }

    assert_eq!(out[0], 3.0);
}

#[test]
fn test_jit_loader_math() {
    let src = r#"
void math_kernel(float* restrict in_buf, float* restrict out) {
    out[0] = __builtin_sqrtf(in_buf[0]);
}
"#;
    let kernel = JitKernel::compile_with_abi(src, "math_kernel", vec![], &abi(2, &[])).unwrap();

    let mut input = [9.0f32];
    let mut out = [0.0f32];

    let buffers = vec![input.as_mut_ptr() as *mut u8, out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[]).unwrap();
    }

    assert!((out[0] - 3.0).abs() < 1e-6);
}

#[test]
fn test_jit_loader_with_vars() {
    let src = r#"
void var_kernel(float* restrict out, const int N) {
    for (int i = 0; i < N; i++) {
        out[i] = (float)i;
    }
}
"#;
    let kernel = JitKernel::compile_with_abi(src, "var_kernel", vec!["N".to_string()], &abi(1, &["N"])).unwrap();

    let mut out = [0.0f32; 8];
    let buffers = vec![out.as_mut_ptr() as *mut u8];

    unsafe {
        kernel.execute_with_vals(&buffers, &[5]).unwrap();
    }

    assert_eq!(out[0], 0.0);
    assert_eq!(out[1], 1.0);
    assert_eq!(out[2], 2.0);
    assert_eq!(out[3], 3.0);
    assert_eq!(out[4], 4.0);
}

// ── x86-64 direct-branch range ──────────────────────────────────────────────

/// Build a scratch buffer holding `E8 <rel32>` at offset 3, followed by the
/// veneer pool. Returns (buffer, displacement offset, patch address).
#[cfg(target_arch = "x86_64")]
fn call_site(opcode: u8) -> (Vec<u8>, usize, usize) {
    let veneer_base = 16;
    let mut buf = vec![0u8; veneer_base + VENEER_SIZE];
    buf[3] = opcode;
    (buf, 4, veneer_base)
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_x86_64_direct_call_beyond_two_gib_routes_through_veneer() {
    let (mut buf, off, veneer_base) = call_site(0xE8);
    let base = buf.as_ptr() as u64;
    let patch = base + off as u64;
    // 3 GiB above the call site: the rel32 field cannot encode this.
    let target = patch + (3 << 30);
    let mut state = RelocState { veneers: VeneerPool::new(veneer_base), ..Default::default() };

    reloc_x86_64(&mut buf, off, patch, target, -4, object::elf::R_X86_64_PLT32, &mut state).unwrap();

    // The patched branch must land on the veneer, not on a truncated address.
    let disp = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as i64;
    assert_eq!(patch as i64 + 4 + disp, base as i64 + veneer_base as i64);

    // MOVABS $target, %r11 ; JMP *%r11
    assert_eq!(&buf[veneer_base..veneer_base + 2], &[0x49, 0xBB]);
    assert_eq!(u64::from_le_bytes(buf[veneer_base + 2..veneer_base + 10].try_into().unwrap()), target);
    assert_eq!(&buf[veneer_base + 10..veneer_base + 13], &[0x41, 0xFF, 0xE3]);
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_x86_64_far_calls_to_one_target_share_a_veneer() {
    let veneer_base = 32;
    let mut buf = vec![0u8; veneer_base + VENEER_SIZE];
    buf[3] = 0xE8;
    buf[11] = 0xE9;
    let base = buf.as_ptr() as u64;
    let mut state = RelocState { veneers: VeneerPool::new(veneer_base), ..Default::default() };
    let target = base + (3 << 30);

    for off in [4usize, 12] {
        reloc_x86_64(&mut buf, off, base + off as u64, target, -4, object::elf::R_X86_64_PLT32, &mut state).unwrap();
    }

    assert_eq!(state.veneers.next, veneer_base + VENEER_SIZE);
    for off in [4usize, 12] {
        let disp = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as i64;
        assert_eq!(base as i64 + off as i64 + 4 + disp, base as i64 + veneer_base as i64);
    }
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_x86_64_in_range_call_stays_direct() {
    let (mut buf, off, veneer_base) = call_site(0xE8);
    let base = buf.as_ptr() as u64;
    let patch = base + off as u64;
    let target = patch + 0x1000;
    let mut state = RelocState { veneers: VeneerPool::new(veneer_base), ..Default::default() };

    reloc_x86_64(&mut buf, off, patch, target, -4, object::elf::R_X86_64_PLT32, &mut state).unwrap();

    let disp = i32::from_le_bytes(buf[off..off + 4].try_into().unwrap()) as i64;
    assert_eq!(patch as i64 + 4 + disp, target as i64);
    assert_eq!(state.veneers.next, veneer_base, "no veneer needed for an in-range call");
}

#[cfg(target_arch = "x86_64")]
#[test]
fn test_x86_64_out_of_range_non_branch_reloc_is_an_error() {
    // A RIP-relative data reference has no trampoline escape: it must fail
    // loudly instead of silently truncating to a bogus address.
    let (mut buf, off, veneer_base) = call_site(0x8B);
    let base = buf.as_ptr() as u64;
    let patch = base + off as u64;
    let mut state = RelocState { veneers: VeneerPool::new(veneer_base), ..Default::default() };

    for r_type in [object::elf::R_X86_64_PC32, object::elf::R_X86_64_REX_GOTPCRELX] {
        let err = reloc_x86_64(&mut buf, off, patch, patch + (3 << 30), -4, r_type, &mut state).unwrap_err();
        assert!(err.to_string().contains("out of range"), "{err}");
    }
    assert_eq!(state.veneers.next, veneer_base);
}

/// A veneer is only useful if it actually runs: map two regions 4 GiB apart —
/// beyond the reach of `call rel32` — and check the relocated call reaches the
/// callee and returns through it.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn test_x86_64_veneer_executes_a_call_beyond_two_gib() {
    const LEN: usize = 4096;

    fn reserve(addr: usize) -> *mut u8 {
        let p = unsafe {
            libc::mmap(
                addr as *mut libc::c_void,
                LEN,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_FIXED_NOREPLACE,
                -1,
                0,
            )
        };
        assert_eq!(p as usize, addr, "reserve {addr:#x}");
        p.cast()
    }

    let caller = reserve(0x2000_0000_0000);
    let callee = reserve(0x2001_0000_0000);

    // callee: MOV $42, %eax ; RET
    unsafe { std::ptr::copy_nonoverlapping([0xB8, 0x2A, 0x00, 0x00, 0x00, 0xC3].as_ptr(), callee, 6) };

    // caller: CALL rel32 ; RET, with the veneer pool starting at offset 16.
    let veneer_base = 16;
    let code = unsafe { std::slice::from_raw_parts_mut(caller, LEN) };
    code[0] = 0xE8;
    code[5] = 0xC3;
    let mut state = RelocState { veneers: VeneerPool::new(veneer_base), ..Default::default() };
    reloc_x86_64(code, 1, caller as u64 + 1, callee as u64, -4, object::elf::R_X86_64_PLT32, &mut state).unwrap();

    for region in [caller, callee] {
        assert_eq!(unsafe { libc::mprotect(region.cast(), LEN, libc::PROT_READ | libc::PROT_EXEC) }, 0);
    }

    let entry: extern "C" fn() -> u32 = unsafe { std::mem::transmute(caller as *const ()) };
    let answer = entry();

    for region in [caller, callee] {
        unsafe { libc::munmap(region.cast(), LEN) };
    }
    assert_eq!(answer, 42);
}
