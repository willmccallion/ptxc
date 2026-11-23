//! Launching the vector add and timing it.

use crate::{SPIN, VECTOR_ADD, context, f32_bytes, from_f32_bytes};
use ptxc_core::{ByteSize, KernelName};
use ptxc_cuda::{BlockDim, Context, CudaError, Event, GridDim, Module, Stream};
use std::time::{Duration, Instant};

/// Not a multiple of the block size, so the bounds check matters.
const LENGTH: u32 = 1000;
const BLOCK: u32 = 256;

#[test]
fn the_vector_add_matches_the_host() {
    let context = context();
    let module = context.load_ptx(VECTOR_ADD).unwrap();
    let kernel = module.function(&KernelName::new("vector_add")).unwrap();
    let a: Vec<f32> = (0..LENGTH).map(|i| f32::from(u16::try_from(i).unwrap())).collect();
    let b: Vec<f32> = a.iter().map(|value| 0.5 - value * 2.0).collect();
    let bytes = ByteSize::new(u64::from(LENGTH) * 4);
    let (mut a_buffer, mut b_buffer) =
        (context.alloc(bytes).unwrap(), context.alloc(bytes).unwrap());
    a_buffer.copy_from_host(&f32_bytes(&a)).unwrap();
    b_buffer.copy_from_host(&f32_bytes(&b)).unwrap();
    let out = context.alloc(bytes).unwrap();
    let stream = context.stream().unwrap();

    kernel
        .launch()
        .grid(GridDim::new(LENGTH.div_ceil(BLOCK), 1, 1))
        .block(BlockDim::new(BLOCK, 1, 1))
        .arg(&a_buffer)
        .arg(&b_buffer)
        .arg(&out)
        .arg(LENGTH)
        .launch(&stream)
        .unwrap();
    stream.synchronize().unwrap();

    let mut result = vec![0_u8; f32_bytes(&a).len()];
    out.copy_to_host(&mut result).unwrap();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    assert_eq!(from_f32_bytes(&result), expected);
}

#[test]
fn arguments_that_do_not_match_the_parameters_are_rejected() {
    let context = context();
    let module = context.load_ptx(VECTOR_ADD).unwrap();
    let kernel = module.function(&KernelName::new("vector_add")).unwrap();
    let buffer = context.alloc(ByteSize::new(4)).unwrap();
    let stream = context.stream().unwrap();
    let launch = || kernel.launch().grid(GridDim::new(1, 1, 1)).block(BlockDim::new(1, 1, 1));

    let missing = launch().arg(&buffer).arg(&buffer).arg(&buffer).launch(&stream).unwrap_err();
    let narrow =
        launch().arg(&buffer).arg(&buffer).arg(&buffer).arg(1_u64).launch(&stream).unwrap_err();

    assert!(
        matches!(missing, CudaError::ArgumentCount { expected: 4, actual: 3, .. }),
        "{missing}"
    );
    assert!(matches!(narrow, CudaError::ArgumentSize { index: 3, .. }), "{narrow}");
}

/// At least 64 ms even at the RTX 4060 Ti's highest clock (3.1 GHz).
const SPIN_CYCLES: u64 = 200_000_000;

/// Below the spin's shortest possible time, so a wait that returns sooner
/// cannot have waited for the kernel.
const SPIN_LOWER_BOUND: Duration = Duration::from_millis(50);

/// Records `start`, a spin kernel and `end` on one stream, without waiting,
/// and returns when the launch was queued. The module is returned too:
/// unloading it would wait for the kernel.
fn spin_between(context: &Context) -> (Module, Stream, Event, Event, Instant) {
    let module = context.load_ptx(SPIN).unwrap();
    let kernel = module.function(&KernelName::new("spin")).unwrap();
    let stream = context.stream().unwrap();
    let (start, end) = (context.event().unwrap(), context.event().unwrap());
    let queued = Instant::now();
    start.record(&stream).unwrap();
    kernel
        .launch()
        .grid(GridDim::new(1, 1, 1))
        .block(BlockDim::new(1, 1, 1))
        .arg(SPIN_CYCLES)
        .launch(&stream)
        .unwrap();
    end.record(&stream).unwrap();
    drop(kernel);
    (module, stream, start, end, queued)
}

#[test]
fn an_event_waits_for_the_work_before_it() {
    let context = context();
    let (_module, _stream, _start, end, queued) = spin_between(&context);

    end.synchronize().unwrap();

    assert!(queued.elapsed() >= SPIN_LOWER_BOUND, "{:?}", queued.elapsed());
}

#[test]
fn a_stream_waits_for_its_queued_work() {
    let context = context();
    let (_module, stream, _start, _end, queued) = spin_between(&context);

    stream.synchronize().unwrap();

    assert!(queued.elapsed() >= SPIN_LOWER_BOUND, "{:?}", queued.elapsed());
}

#[test]
fn events_time_the_kernel_between_them() {
    let context = context();
    let (_module, stream, start, end, _queued) = spin_between(&context);
    stream.synchronize().unwrap();

    let elapsed = end.elapsed_since(&start).unwrap();

    assert!(elapsed.get() >= 50_000.0, "{elapsed:?}");
}

#[test]
fn events_time_the_work_between_them() {
    let context = context();
    let stream = context.stream().unwrap();
    let (start, end) = (context.event().unwrap(), context.event().unwrap());

    start.record(&stream).unwrap();
    end.record(&stream).unwrap();
    end.synchronize().unwrap();

    assert!(end.elapsed_since(&start).unwrap().get() >= 0.0);
}
