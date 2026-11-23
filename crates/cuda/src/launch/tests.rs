use super::*;

fn bytes(sizes: &[u64]) -> Vec<ByteSize> {
    sizes.iter().map(|&size| ByteSize::new(size)).collect()
}

fn kernel() -> KernelName {
    KernelName::new("add")
}

#[test]
fn matching_arguments_pass() {
    assert_eq!(check_args(&kernel(), &bytes(&[8, 8, 4]), &bytes(&[8, 8, 4])), Ok(()));
    assert_eq!(check_args(&kernel(), &[], &[]), Ok(()));
}

#[test]
fn a_missing_or_extra_argument_is_an_error() {
    for args in [bytes(&[8]), bytes(&[8, 4, 4])] {
        assert_eq!(
            check_args(&kernel(), &bytes(&[8, 4]), &args),
            Err(CudaError::ArgumentCount { kernel: kernel(), expected: 2, actual: args.len() })
        );
    }
}

#[test]
fn an_argument_of_the_wrong_size_is_an_error_naming_it() {
    assert_eq!(
        check_args(&kernel(), &bytes(&[8, 4]), &bytes(&[8, 8])),
        Err(CudaError::ArgumentSize {
            kernel: kernel(),
            index: 1,
            expected: ByteSize::new(4),
            actual: ByteSize::new(8),
        })
    );
}

#[test]
fn scalars_fill_their_slots_from_the_low_bytes() {
    assert_eq!(KernelArg::U32(0x0102_0304).slot().0, [4, 3, 2, 1, 0, 0, 0, 0]);
    assert_eq!(KernelArg::U64(u64::MAX).slot().0, [0xff; 8]);
    assert_eq!(KernelArg::F32(1.0).slot().0[..4], 1.0_f32.to_ne_bytes());
    assert_eq!(
        (KernelArg::U32(0).size(), KernelArg::F32(0.0).size()),
        (ByteSize::new(4), ByteSize::new(4))
    );
    assert_eq!(KernelArg::U64(0).size(), ByteSize::new(8));
}
