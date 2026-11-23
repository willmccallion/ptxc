//! Device buffers and host copies.

use crate::context;
use ptxc_core::{ByteOffset, ByteSize};
use ptxc_cuda::CudaError;

#[test]
fn bytes_round_trip_through_device_memory() {
    let mut buffer = context().alloc(ByteSize::new(5)).unwrap();
    let mut back = [0_u8; 5];

    buffer.copy_from_host(&[1, 2, 3, 4, 5]).unwrap();
    buffer.copy_to_host(&mut back).unwrap();

    assert_eq!(back, [1, 2, 3, 4, 5]);
    assert_eq!(buffer.size(), ByteSize::new(5));
}

#[test]
fn a_copy_of_the_wrong_size_is_rejected() {
    let mut buffer = context().alloc(ByteSize::new(4)).unwrap();

    let error = buffer.copy_from_host(&[0; 3]).unwrap_err();

    assert_eq!(error, CudaError::CopySize { host: ByteSize::new(3), device: ByteSize::new(4) });
}

#[test]
fn an_impossible_allocation_names_the_driver_error() {
    let error = context().alloc(ByteSize::new(1 << 50)).unwrap_err();

    let CudaError::Driver { call, name, .. } = &error else { panic!("{error}") };
    assert_eq!((*call, name.as_str()), ("cuMemAlloc", "CUDA_ERROR_OUT_OF_MEMORY"));
}

#[test]
fn handles_work_from_another_thread() {
    let mut buffer = context().alloc(ByteSize::new(2)).unwrap();
    buffer.copy_from_host(&[7, 9]).unwrap();

    let back = std::thread::spawn(move || {
        let mut back = [0_u8; 2];
        buffer.copy_to_host(&mut back).unwrap();
        back
    })
    .join()
    .unwrap();

    assert_eq!(back, [7, 9]);
}

#[test]
fn dropped_buffers_give_their_memory_back() {
    let context = context();
    let gibibyte = ByteSize::new(1 << 30);

    for _ in 0..16 {
        let buffer = context.alloc(gibibyte).unwrap();
        drop(buffer);
    }
}

#[test]
fn a_slice_must_lie_within_its_buffer() {
    let buffer = context().alloc(ByteSize::new(16)).unwrap();

    assert_eq!(
        buffer.slice(ByteOffset::new(8), ByteSize::new(8)).unwrap().size(),
        ByteSize::new(8)
    );
    let past = buffer.slice(ByteOffset::new(9), ByteSize::new(8)).unwrap_err();
    let overflowing = buffer.slice(ByteOffset::new(u64::MAX), ByteSize::new(2)).unwrap_err();

    assert!(matches!(past, CudaError::SliceBounds { .. }), "{past}");
    assert!(matches!(overflowing, CudaError::SliceBounds { .. }), "{overflowing}");
}

#[test]
fn a_stream_copies_between_slices_in_order() {
    let context = context();
    let mut buffer = context.alloc(ByteSize::new(8)).unwrap();
    buffer.copy_from_host(&[1, 2, 3, 4, 0, 0, 0, 0]).unwrap();
    let stream = context.stream().unwrap();
    let low = buffer.slice(ByteOffset::new(0), ByteSize::new(4)).unwrap();
    let high = buffer.slice(ByteOffset::new(4), ByteSize::new(4)).unwrap();

    stream.copy(low, high).unwrap();
    let mismatch = stream.copy(low, buffer.all()).unwrap_err();
    stream.synchronize().unwrap();

    let mut back = [0_u8; 8];
    buffer.copy_to_host(&mut back).unwrap();
    assert_eq!(back, [1, 2, 3, 4, 1, 2, 3, 4]);
    assert!(matches!(mismatch, CudaError::SliceSizes { .. }), "{mismatch}");
}
