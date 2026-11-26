//! Stream capture: work queued while a stream captures is recorded, not
//! run, and the instantiated graph replays it on the buffers' contents at
//! each launch.

use crate::{VECTOR_ADD, context, f32_bytes, from_f32_bytes};
use ptxc_core::{ByteSize, KernelName};
use ptxc_cuda::{BlockDim, DeviceBuffer, Function, GridDim, Stream};

const LENGTH: u32 = 1000;
const BLOCK: u32 = 256;

fn values(offset: f32) -> Vec<f32> {
    (0..LENGTH).map(|i| f32::from(u16::try_from(i).unwrap()).mul_add(0.5, offset)).collect()
}

/// Queues `out = a + b` on `stream`.
fn add(kernel: &Function, [a, b, out]: [&DeviceBuffer; 3], stream: &Stream) {
    kernel
        .launch()
        .grid(GridDim::new(LENGTH.div_ceil(BLOCK), 1, 1))
        .block(BlockDim::new(BLOCK, 1, 1))
        .arg(a)
        .arg(b)
        .arg(out)
        .arg(LENGTH)
        .launch(stream)
        .unwrap();
}

fn read(buffer: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0_u8; usize::try_from(LENGTH).unwrap() * 4];
    buffer.copy_to_host(&mut bytes).unwrap();
    from_f32_bytes(&bytes)
}

#[test]
fn a_captured_graph_replays_its_launches_and_copies_on_the_buffers_contents() {
    let context = context();
    let module = context.load_ptx(VECTOR_ADD).unwrap();
    let kernel = module.function(&KernelName::new("vector_add")).unwrap();
    let bytes = ByteSize::new(u64::from(LENGTH) * 4);
    let [mut a, mut b, sum, copy, mut out] = [(); 5].map(|()| context.alloc(bytes).unwrap());
    out.copy_from_host(&f32_bytes(&vec![0.0; 1000])).unwrap();
    let stream = context.stream().unwrap();

    let capture = stream.begin_capture().unwrap();
    add(&kernel, [&a, &b, &sum], capture.stream());
    capture.stream().copy(sum.all(), copy.all()).unwrap();
    add(&kernel, [&sum, &copy, &out], capture.stream());
    let graph = capture.end().unwrap().instantiate().unwrap();

    stream.synchronize().unwrap();
    assert_eq!(read(&out), vec![0.0; 1000], "a capture runs nothing");
    for offset in [1.0, -3.0, 0.25] {
        let (x, y) = (values(offset), values(2.0 * offset));
        a.copy_from_host(&f32_bytes(&x)).unwrap();
        b.copy_from_host(&f32_bytes(&y)).unwrap();

        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();

        let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| 2.0 * (x + y)).collect();
        assert_eq!(read(&out), expected);
    }
}

#[test]
fn a_capture_dropped_before_its_end_leaves_the_stream_running_work() {
    let context = context();
    let module = context.load_ptx(VECTOR_ADD).unwrap();
    let kernel = module.function(&KernelName::new("vector_add")).unwrap();
    let bytes = ByteSize::new(u64::from(LENGTH) * 4);
    let [mut a, mut b, out] = [(); 3].map(|()| context.alloc(bytes).unwrap());
    let (x, y) = (values(1.0), values(4.0));
    a.copy_from_host(&f32_bytes(&x)).unwrap();
    b.copy_from_host(&f32_bytes(&y)).unwrap();
    let stream = context.stream().unwrap();
    let capture = stream.begin_capture().unwrap();
    add(&kernel, [&a, &b, &out], capture.stream());

    drop(capture);
    add(&kernel, [&a, &b, &out], &stream);
    stream.synchronize().unwrap();

    let expected: Vec<f32> = x.iter().zip(&y).map(|(x, y)| x + y).collect();
    assert_eq!(read(&out), expected);
}

#[test]
fn a_synchronous_copy_while_capturing_fails_and_so_does_the_capture() {
    let context = context();
    let mut buffer = context.alloc(ByteSize::new(16)).unwrap();
    let stream = context.stream().unwrap();
    let capture = stream.begin_capture().unwrap();

    let copied = buffer.copy_from_host(&[0; 16]);

    assert!(copied.is_err(), "{copied:?}");
    assert!(capture.end().is_err());
    assert!(buffer.copy_from_host(&[0; 16]).is_ok());
}
