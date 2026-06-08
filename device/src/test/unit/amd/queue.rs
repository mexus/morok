use super::test_support::amd_alloc_or_skip;
use crate::amd::queue::*;
use crate::amd::sys::hsa::{
    hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM, hsa_kernel_dispatch_packet_t, kernel_dispatch_header,
};

#[test]
fn aql_packet_header_layout() {
    let h = kernel_dispatch_header();
    // AQL packet header = TYPE_KERNEL_DISPATCH | barrier | sys-acq | sys-rel
    let sys = hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16;
    let expected: u16 = 2 | (1 << 8) | (sys << 9) | (sys << 11);
    assert_eq!(h, expected);
}

#[test]
fn aql_packet_is_64_bytes() {
    assert_eq!(size_of::<hsa_kernel_dispatch_packet_t>(), AQL_PACKET_BYTES);
}

#[test]
fn build_dispatch_picks_correct_dims() {
    // `setup` (dims) is the high 16 bits of the header/setup union's full_header.
    let dims = |p: &hsa_kernel_dispatch_packet_t| (unsafe { p.__bindgen_anon_1.full_header } >> 16) & 0b11;
    let p1 = build_dispatch_packet([64, 1, 1], [1024, 1, 1], 0, 0, 0, 0, 0);
    assert_eq!(dims(&p1), 1);
    let p2 = build_dispatch_packet([8, 8, 1], [256, 256, 1], 0, 0, 0, 0, 0);
    assert_eq!(dims(&p2), 2);
    let p3 = build_dispatch_packet([4, 4, 4], [64, 64, 64], 0, 0, 0, 0, 0);
    assert_eq!(dims(&p3), 3);
}

#[test]
fn sdma_linear_copy_dwords_layout() {
    let dw = crate::amd::sys::sdma::copy_linear(0x1_0000_2000, 0x2_0000_3000, 4096);
    assert_eq!(dw[0], 0x01);
    assert_eq!(dw[1], 4095);
    assert_eq!(dw[3], 0x0000_2000);
    assert_eq!(dw[4], 0x0000_0001);
    assert_eq!(dw[5], 0x0000_3000);
    assert_eq!(dw[6], 0x0000_0002);
}

/// Live SDMA staging roundtrip: a device-local (host_ptr: None) buffer is
/// filled via `_copyin` and read back via `_copyout`, exercising the real SDMA
/// copy + fence + signal-wait path. Skipped without an AMD GPU. A wrong fence
/// fails via the 30 s copy timeout rather than hanging.
#[test]
fn sdma_device_local_roundtrip() {
    use crate::allocator::{Allocator, BufferSpec, RawBuffer};
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = alloc.dev.core();
    // Bring up signal pool + copy queue (the device factory normally does this);
    // both installers are idempotent, so this is safe if a factory already ran.
    if core.signal_pool().is_none() {
        core.install_signal_pool(crate::amd::signal::SignalPool::new(&alloc, 64).expect("signal pool"));
    }
    if core.copy_queue().is_none() {
        core.install_copy_queue(AmdCopyQueue::create(&alloc).expect("copy queue"));
        core.set_has_sdma_queue(true);
    }

    let spec = BufferSpec { cpu_access: false, ..Default::default() };
    // Span > staging size (4 MiB) to exercise multi-chunk staging.
    let n = 5 * 1024 * 1024usize;
    let buf = alloc._alloc(n, &spec, false).expect("device-local alloc");
    assert!(matches!(buf, RawBuffer::AmdDevice { host_ptr: None, .. }), "buffer must be device-only");

    let src: Vec<u8> = (0..n).map(|i| (i.wrapping_mul(2654435761) >> 13) as u8).collect();
    alloc._copyin(&buf, 0, &src).expect("copyin");
    let mut out = vec![0u8; n];
    alloc._copyout(&mut out, &buf, 0).expect("copyout");
    assert_eq!(src, out, "SDMA host↔device roundtrip must preserve bytes");

    // Device→device transfer into a second device-local buffer.
    let buf2 = alloc._alloc(n, &spec, false).expect("device-local alloc 2");
    alloc._transfer(&buf2, 0, &buf, 0, n).expect("transfer");
    let mut out2 = vec![0u8; n];
    alloc._copyout(&mut out2, &buf2, 0).expect("copyout 2");
    assert_eq!(src, out2, "SDMA device→device transfer must preserve bytes");

    alloc._free(buf, &spec);
    alloc._free(buf2, &spec);
}

/// Live compute queue creation (exercises the KFD CREATE_QUEUE path).
/// Skipped without a supported AMD GPU. A real dispatch needs the device
/// timeline wired up by the factory, so we only assert creation here.
#[test]
fn compute_queue_create_if_hw_supports() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let _q = AmdComputeQueue::create(&alloc).expect("create compute queue");
}

/// On real AQL hardware (multi-XCC CDNA), `set_aql_scratch` must land the
/// scratch descriptor at the right `amd_queue_t` offsets in the GART page the
/// firmware reads. Exercises the offsets + volatile writes end-to-end against a
/// live queue; on PM4 hardware the queue has no descriptor and the write is a
/// no-op (we skip the assertion there).
#[test]
fn set_aql_scratch_round_trips_through_gart() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let q = AmdComputeQueue::create(&alloc).expect("create compute queue");
    if q.is_pm4() {
        return; // PM4 queues program scratch via registers; no GART descriptor.
    }
    // A realistic descriptor, sized exactly as a 256-byte/thread scratch alloc
    // would be on this device.
    let (va, _size, tmpring, _rounded, _handle, desc) =
        crate::amd::device::alloc_scratch(alloc.dev.core().iface(), &alloc.dev.node, &alloc.dev.arch, 256)
            .expect("alloc scratch");
    assert_ne!(desc, crate::amd::device::AqlScratchDesc::default(), "CDNA must synthesize a descriptor");
    q.set_aql_scratch(&desc);
    assert_eq!(q.read_aql_scratch(), desc, "GART descriptor must match what we wrote");
    // Sanity: the descriptor points at the freshly allocated scratch buffer.
    assert_eq!(desc.backing_va, va);
    assert_eq!(desc.tmpring_size, tmpring);
    // Free the scratch we allocated for the test.
    alloc.dev.core().iface().free_raw(va, _size, _handle);
}

/// EXPERIMENT (manual hardware probe; not part of the normal suite — `#[ignore]`).
///
/// Does the AQL packet processor honor a *native* `completion_signal` on a
/// KFD-direct `COMPUTE_AQL` queue, with a plain busy-wait signal the host polls
/// (no event mailbox / interrupt)? And on multi-XCC (MI300 SPX), does it fire
/// once or once-per-XCC? This is the question gating the queue redesign (drop
/// the PM4-`RELEASE_MEM` monotonic timeline + per-owner queues in favor of
/// native AQL dispatch + countdown completion signals + a shared lock-free ring).
///
/// We submit a single AQL `barrier_and` packet (no shader → no scratch / no
/// `rsrc` config → cannot fault a CU) whose `completion_signal` points at a
/// 64-byte `amd_signal_t` we built ourselves, with `value` initialized to **1**.
/// The final value is self-interpreting:
///
/// - `0` → fired exactly once (countdown works; single completion op).
/// - `< 0` → fired `1 - value` times (once per command-processor pass, i.e.
///   per-XCC; this is the multi-XCC answer).
/// - `1` → never fired (timeout): the native path is unusable as-is here.
///
/// We also dump the `kind` word: if it moved instead of `value`, the firmware
/// treats `completion_signal.handle` as `&value` rather than the struct base.
///
/// Run with:
///   cargo test -p svod-device --lib aql_native_completion_signal_probe \
///       -- --ignored --nocapture
#[test]
#[ignore = "manual hardware probe; needs a real AMD GPU and prints findings"]
fn aql_native_completion_signal_probe() {
    use std::time::{Duration, Instant};

    use crate::allocator::RawBuffer;
    use crate::amd::sys::hsa::{
        amd_signal_kind_t_AMD_SIGNAL_KIND_USER, amd_signal_t, hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM,
        hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER, hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE,
        hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE, hsa_packet_type_t_HSA_PACKET_TYPE_BARRIER_AND,
    };

    let Some(alloc) = amd_alloc_or_skip() else { return };
    let xccs = alloc.dev.node.num_xcc.max(1);
    let q = AmdComputeQueue::create(&alloc).expect("create compute queue");
    if q.is_pm4() {
        eprintln!("PROBE skipped: single-XCC PM4 queue (the native-completion question is multi-XCC AQL).");
        return;
    }

    // 64-byte host-visible, uncached-coherent GTT buffer for the amd_signal_t —
    // same memory class as the real timeline signal slot, so the GPU's write is
    // visible to the host poll without an explicit invalidate.
    let sig_buf = alloc.alloc_uncached(64).expect("signal buffer");
    let (sig_gpu, sig_host) = match &sig_buf {
        RawBuffer::AmdDevice { gpu_addr, host_ptr: Some(h), .. } => (*gpu_addr, *h),
        _ => panic!("signal buffer must be host-visible"),
    };

    // Build a KIND_USER busy-wait signal with value=1 and NO event mailbox, and
    // lay it into the buffer. `value` lives in the first anonymous union (off 8).
    let value_off = std::mem::offset_of!(amd_signal_t, __bindgen_anon_1);
    // SAFETY: amd_signal_t is plain old data; all-zero is a valid INVALID signal.
    let mut sig: amd_signal_t = unsafe { std::mem::zeroed() };
    sig.kind = amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64;
    sig.__bindgen_anon_1.value = 1;
    // SAFETY: sig_host points at the 64-byte buffer we just allocated.
    unsafe { std::ptr::copy_nonoverlapping(&sig as *const _ as *const u8, sig_host.as_ptr(), 64) };
    std::sync::atomic::fence(std::sync::atomic::Ordering::Release);

    // AQL barrier_and packet (64 B): header at dw0 (low 16 bits), five dep_signal
    // slots left 0 (no deps → completes immediately), completion_signal.handle at
    // dw14/15 = the amd_signal_t base VA.
    let header: u16 = (hsa_packet_type_t_HSA_PACKET_TYPE_BARRIER_AND as u16)
        | (1u16 << hsa_packet_header_t_HSA_PACKET_HEADER_BARRIER as u16)
        | ((hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16)
            << hsa_packet_header_t_HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE as u16)
        | ((hsa_fence_scope_t_HSA_FENCE_SCOPE_SYSTEM as u16)
            << hsa_packet_header_t_HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE as u16);
    let mut pkt = [0u32; 16];
    pkt[0] = header as u32; // dw0: header in low 16, reserved0 = 0 in high 16
    pkt[14] = sig_gpu as u32;
    pkt[15] = (sig_gpu >> 32) as u32;

    q.submit_aql(&[pkt]).expect("submit barrier_and");

    // Poll the signal value until it drops below the initial 1, or time out.
    // SAFETY: coherent GTT slot; the GPU writes `value` at sig_gpu+value_off.
    let value_ptr = unsafe { sig_host.as_ptr().add(value_off) as *const i64 };
    let kind_ptr = sig_host.as_ptr() as *const i64;
    let start = Instant::now();
    let final_value = loop {
        let v = unsafe { std::ptr::read_volatile(value_ptr) };
        if v <= 0 || start.elapsed() > Duration::from_secs(5) {
            break v;
        }
        std::hint::spin_loop();
    };
    let final_kind = unsafe { std::ptr::read_volatile(kind_ptr) };

    eprintln!(
        "PROBE native AQL completion_signal (num_xcc={xccs}): value 1 -> {final_value} \
         after {:?}; kind now {final_kind}",
        start.elapsed()
    );
    eprintln!(
        "  interpretation: 0 = fired once | <0 = fired {} times (per-XCC) | 1 = never fired (timeout){}",
        if final_value < 0 { (1 - final_value).to_string() } else { "n/a".into() },
        if final_kind != amd_signal_kind_t_AMD_SIGNAL_KIND_USER as i64 {
            " | WARNING: kind word moved — handle treated as &value, not struct base"
        } else {
            ""
        }
    );

    // SAFETY: drained (value observed or timed out); free the probe buffer.
    sig_buf.free_amd_device_in_place();
}
