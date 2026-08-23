//! Tests for memory and buffer operations constructors.

use svod_dtype::DType;
use svod_dtype::DeviceSpec;

use crate::types::{AddrSpace, AxisId, AxisType, BufferizeOpts};
use crate::{Op, UOp};

#[test]
fn getaddr_is_scalar_uint64_with_exact_storage_metadata() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let address = buffer.getaddr(None);

    assert_eq!(address.dtype(), DType::UInt64);
    assert_eq!(address.shape().unwrap().unwrap().as_slice(), &[]);
    assert_eq!(address.addrspace(), None);
    assert!(address.tree().contains("GETADDR(CPU)"));
    let graph = crate::CanonicalGraph::from_root("hcq", &address).unwrap();
    let node = graph.nodes.last().unwrap();
    assert_eq!(node.op, "GETADDR");
    assert_eq!(node.src.len(), 1);
    assert_eq!(node.arg, crate::CanonicalArg::Device { name: "CPU".to_string() });
    match address.op() {
        Op::GetAddr { src, device } => {
            assert!(std::sync::Arc::ptr_eq(src, &buffer));
            assert_eq!(device, &DeviceSpec::Cpu);
            assert_eq!(src.addrspace(), Some(AddrSpace::Global));
        }
        op => panic!("expected GETADDR, got {op:?}"),
    }
}

#[test]
fn getaddr_hash_reconstruction_and_source_filter_match_target() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 4, DType::UInt8);
    let cpu = buffer.getaddr(None);
    let cuda = buffer.getaddr(Some(DeviceSpec::Cuda { device_id: 0 }));
    assert_ne!(cpu.id, cuda.id, "device argument participates in hash consing");
    assert!(std::sync::Arc::ptr_eq(&cpu.with_sources(vec![buffer.clone()]), &cpu));

    let scalar = UOp::native_const(1u64);
    assert!(std::sync::Arc::ptr_eq(&scalar.getaddr(Some(DeviceSpec::Cpu)), &scalar));
}

#[test]
fn test_bufferize() {
    let compute = UOp::native_const(1.0f32);
    let r1 = UOp::range_axis(UOp::native_const(10i32), AxisId::Renumbered(0), AxisType::Loop);
    let r2 = UOp::range_axis(UOp::native_const(20i32), AxisId::Renumbered(1), AxisType::Loop);

    let opts = BufferizeOpts::new(DeviceSpec::Cpu);
    let stage = UOp::stage(compute.clone(), vec![r1, r2], opts);

    // Should have same dtype as compute
    assert_eq!(stage.dtype(), DType::Float32);

    // Should be Stage op
    if let Op::Stage { compute: c, ranges, opts: o } = stage.op() {
        assert!(std::sync::Arc::ptr_eq(c, &compute));
        assert_eq!(ranges.len(), 2);
        assert_eq!(o.device, Some(DeviceSpec::Cpu));
        assert_eq!(o.addrspace, AddrSpace::Global);
    } else {
        panic!("Expected Stage op");
    }
}

#[test]
fn test_bufferize_local() {
    let compute = UOp::native_const(1.0f32);
    let r = UOp::range_axis(UOp::native_const(10i32), AxisId::Renumbered(0), AxisType::Loop);

    let opts = BufferizeOpts::local();
    let stage = UOp::stage(compute, vec![r], opts);

    if let Op::Stage { opts: o, .. } = stage.op() {
        assert_eq!(o.addrspace, AddrSpace::Local);
    } else {
        panic!("Expected Stage op");
    }
}

#[test]
fn test_load() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let offset = UOp::index_const(0);
    let index = UOp::index().buffer(buffer.clone()).indices(vec![offset]).call().unwrap();

    let load = UOp::load().index(index.clone()).call();

    // Should have same dtype as buffer
    assert_eq!(load.dtype(), DType::Float32);

    // Should be Load op
    if let Op::Load { index: i, .. } = load.op() {
        assert!(std::sync::Arc::ptr_eq(i, &index));
    } else {
        panic!("Expected Load op");
    }
}

#[test]
fn test_store() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 100, DType::Float32);
    let index_offset = UOp::index_const(0);
    let value = UOp::native_const(42.0f32);

    // Create INDEX op first (STORE's index field is an INDEX op)
    let index = UOp::index().buffer(buffer.clone()).indices(vec![index_offset]).call().unwrap();

    // Use store_value() on INDEX (preferred API)
    let store = index.store_value(value.clone());

    // Store should have Void dtype
    assert_eq!(store.dtype(), DType::Void);

    // Should be Store op with index pointing to buffer
    if let Op::Store { index: i, value: v, .. } = store.op() {
        assert!(std::sync::Arc::ptr_eq(i, &index));
        assert!(std::sync::Arc::ptr_eq(v, &value));
        // Verify buffer can be accessed via store_buffer()
        assert!(std::sync::Arc::ptr_eq(store.store_buffer().unwrap(), &buffer));
    } else {
        panic!("Expected Store op");
    }
}

#[test]
fn test_codegen_param() {
    // Per-kernel codegen PARAM: scalar storage dtype and global address metadata.
    let p = UOp::param(0, 1024, DType::Float32, None);

    assert_eq!(p.dtype(), DType::Float32);
    assert_eq!(p.shape().unwrap().unwrap()[0].as_const(), Some(1024));

    if let Op::Param { arg, .. } = p.op() {
        assert_eq!(arg.slot, 0);
        assert_eq!(arg.dtype, DType::Float32);
        assert_eq!(arg.addrspace, Some(svod_dtype::AddrSpace::Global));
        assert!(arg.device.is_none());
    } else {
        panic!("Expected Param op");
    }
}

#[test]
fn test_index_infers_buffer_dtype() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32);
    let offset = UOp::index_const(0);

    let inferred = UOp::index().buffer(buffer).indices(vec![offset]).call().unwrap();
    assert_eq!(inferred.dtype(), DType::Float32);
}

#[test]
fn test_shaped_index_and_load_keep_scalar_element_dtype() {
    let buffer = UOp::new_buffer(DeviceSpec::Cpu, 16, DType::Float32);
    let offsets = UOp::stack(smallvec::smallvec![UOp::index_const(0), UOp::index_const(1)]);
    let index = UOp::index().buffer(buffer).indices(vec![offsets]).call().unwrap();
    let load = UOp::load().index(index.clone()).call();

    assert_eq!(index.dtype(), DType::Float32);
    assert_eq!(load.dtype(), DType::Float32);
    assert_eq!(index.shape().unwrap().unwrap().as_slice(), &[crate::SInt::Const(2)]);
    assert_eq!(load.shape().unwrap().unwrap().as_slice(), &[crate::SInt::Const(2)]);
}

#[test]
fn test_local_buffer() {
    let dl = UOp::buffer(1, 4, DType::Int32, AddrSpace::Local, None);

    assert_eq!(dl.dtype(), DType::Int32);

    if let Op::Buffer { arg, .. } = dl.op() {
        assert_eq!(arg.slot, 1);
        assert_eq!(arg.addrspace, Some(AddrSpace::Local));
        assert_eq!(dl.shape().unwrap().unwrap().as_slice(), &[crate::SInt::Const(4)]);
    } else {
        panic!("Expected local Buffer op");
    }
}
