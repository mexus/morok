//! Empirical resolution of the store→reload-of-same-buffer ordering question
//! (the one risk the integration audit left open): does a REG accumulator —
//! init, then a reduce loop, then a post-loop read — order correctly through
//! the Svod schedule, and what `after`/`end` shape does the schedule accept?
//!
//! Findings (this file is the evidence):
//! - No dependency edge at all → the post-loop read races the loop (reads the
//!   init value). So an `after(store)` edge IS required.
//! - The canonical accumulator shape is exactly svod's OWN `reduce_to_acc`
//!   lowering (`schedule/src/devectorize.rs`) and `Kernel::endrange`: a single
//!   `acc.after([store.end([range])])` node that BOTH closes the reduce loop
//!   (the `END(STORE)`) AND carries the data edge for the post-loop read. The
//!   `END(STORE)` reaches the SINK *through* the out-store's dependency chain;
//!   it is never a separate SINK source (that double-stores the REG and breaks
//!   global-buffer extraction).
//! - These `After([END(STORE)])` / `After([RANGE])` edges are intra-kernel: they
//!   live inside an opaque marked-SINK CALL body, so the tensor scheduler walks
//!   them with `toposort_call_aware(false)` and never validates them as
//!   inter-kernel callable dependencies (`tensor/src/schedule.rs`).
//!
//! Each kernel computes `out[0] = sum(in[0..8]) = 28`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::{DType, DeviceSpec};
use svod_ir::{AxisType, ConstValue, KernelInfo, UOp};
use svod_tensor::{CpuBackend, Tensor};

use crate::index::{Idx, cidx, load_at};
use crate::launch;

fn f32c(v: f32) -> Arc<UOp> {
    UOp::const_(DType::Float32, ConstValue::Float(v as f64))
}
fn index0(buf: &Arc<UOp>) -> Arc<UOp> {
    UOp::index().buffer(buf.clone()).indices(vec![cidx(0)]).call().expect("INDEX[0]")
}
fn load0(buf: &Arc<UOp>) -> Arc<UOp> {
    UOp::load().index(index0(buf)).call()
}

const INPUT: [f32; 8] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];

#[test]
fn sparse_and_interleaved_program_slots_plan_compact_buffers() {
    assert_eq!(crate::launch::plan_compact_buffers(&[0, 5], 2).unwrap(), vec![(0, 0), (1, 5)]);
    assert_eq!(crate::launch::plan_compact_buffers(&[0, 2], 2).unwrap(), vec![(0, 0), (1, 2)]);
    assert!(matches!(
        crate::launch::plan_compact_buffers(&[0, 5], 1),
        Err(crate::launch::Error::BufferCount { expected: 2, supplied: 1, .. })
    ));
}

/// The canonical accumulator shape, identical to svod's own `reduce_to_acc`
/// lowering (`schedule/src/devectorize.rs`) and `Kernel::endrange`:
///
/// 1. `acc_init`  = `index(acc).store(0)` — a bare STORE that seeds the REG.
/// 2. `acc_loop`  = `index(acc.after([acc_init, range]))` read inside the loop;
///    the `After([bare STORE, RANGE])` orders the read after init and inside
///    the reduce range.
/// 3. `store_end` = `index(acc).store(acc_loop + in[i]).end([range])` — the
///    `END(STORE)` that closes the reduce loop.
/// 4. post-loop read = `index(acc.after([store_end]))` — the `After([END(STORE)])`
///    that both reads the final accumulator AND threads `store_end` into the
///    out-store's dependency chain so the loop reaches the SINK.
///
/// The `END(STORE)` is NOT a separate SINK source: it flows to the SINK only
/// through `out_store`, exactly as `reduce_to_acc` returns
/// `acc.after([store_end]).index(0)` as the single result value.
#[test]
fn test_accumulator_after_bare_store_cpu() {
    svod_dtype::default_device::with_default_device(DeviceSpec::Cpu, || {
        let n = INPUT.len();

        // Materialize concrete CPU buffers BEFORE building the kernel — the
        // direct-launch path (tinygrad `Tensor.realize(...)` → `sink.call(bufs)`
        // → `run_linear`), not the tensor scheduler. `from_slice` already backs
        // the input; `realize_buffer` allocates the empty output in place.
        let src = Tensor::from_slice(INPUT);
        let dst = Tensor::empty(&[1], DType::Float32);
        let src_buf = launch::realize_buffer(&src).expect("input buffer");
        let dst_buf = launch::realize_buffer(&dst).expect("output buffer");

        // Bind the Kernel to the concrete BUFFER UOps (output first, then input);
        // `gl()` hands them out as flat 1-D Params — no `flat_ptr` unwrap needed.
        let ker =
            crate::Kernel::new("acc", [1, 1, 1], 1, vec![dst.uop().base(), src.uop().base()], crate::ArchCaps::GFX942);
        let out_buf = ker.next_global();
        let in_buf = ker.next_global();

        let acc = ker.alloc_reg(1, DType::Float32);
        let init = index0(&acc).store(f32c(0.0)); // bare STORE — seeds the REG

        let i = ker.raw_range(n as i64, AxisType::Reduce);
        // Read the accumulator inside the loop: after init + inside the range.
        let acc_loop = load0(&acc.after(smallvec![init, i.clone()]));
        let in_v = load_at(&in_buf, &[n], &[Idx::from(&i)]);
        let sum = acc_loop.try_add(&in_v).expect("acc + in");
        // END(STORE) closes the reduce loop (the `reduce_to_acc` `store_end`).
        let store_end = index0(&acc).store(sum).end(smallvec![i]);
        // Post-loop read: after([END(STORE)]) — threads the loop into the SINK.
        let result = load0(&acc.after(smallvec![store_end]));

        let out_store = index0(&out_buf).store(result);
        let sink =
            UOp::sink_with_info(vec![out_store], KernelInfo { opts_to_apply: Some(vec![]), ..Default::default() });

        // Compile + dispatch directly against the concrete buffers (outputs first).
        let device = svod_runtime::create_cpu_device_with_backend(svod_device::registry::registry(), CpuBackend::Clang)
            .expect("clang device");
        let buffers = [dst_buf, src_buf];
        launch::launch(&device, sink, &buffers).expect("launch");

        assert_eq!(dst.as_vec::<f32>().expect("read"), vec![28.0]);
    });
}
