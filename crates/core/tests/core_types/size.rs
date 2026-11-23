use ptxc_core::{Alignment, ByteOffset, ByteSize, DType, ElementCount, SizeError};

fn align_up(offset: u64, alignment: u64) -> Result<u64, SizeError> {
    ByteOffset::new(offset).align_up(Alignment::new(alignment).unwrap()).map(ByteOffset::get)
}

#[test]
fn element_sizes_match_onnx_storage() {
    let sizes = [
        (DType::F32, 4),
        (DType::F16, 2),
        (DType::Bf16, 2),
        (DType::F8E4M3Fn, 1),
        (DType::F8E5M2, 1),
        (DType::I8, 1),
        (DType::U8, 1),
        (DType::I32, 4),
        (DType::I64, 8),
        (DType::Bool, 1),
    ];

    for (dtype, bytes) in sizes {
        assert_eq!(dtype.element_size(), ByteSize::new(bytes), "{dtype:?}");
    }
}

#[test]
fn in_bytes_multiplies_by_the_element_size() {
    let bytes = ElementCount::new(6).in_bytes(DType::F16).unwrap();

    assert_eq!(bytes, ByteSize::new(12));
}

#[test]
fn in_bytes_past_u64_max_is_rejected() {
    let count = ElementCount::new(u64::MAX / 4 + 1);

    let error = count.in_bytes(DType::F32).unwrap_err();

    assert_eq!(error, SizeError::ByteSizeOverflow { count, dtype: DType::F32 });
}

#[test]
fn alignment_must_be_a_power_of_two() {
    assert!(Alignment::new(1).is_ok());
    assert!(Alignment::new(256).is_ok());
    for bytes in [0, 3, 96] {
        assert_eq!(Alignment::new(bytes), Err(SizeError::NotPowerOfTwo(bytes)));
    }
}

#[test]
fn align_up_rounds_to_the_next_multiple() {
    assert_eq!(align_up(1, 16).unwrap(), 16);
    assert_eq!(align_up(17, 16).unwrap(), 32);
}

#[test]
fn align_up_keeps_aligned_offsets() {
    assert_eq!(align_up(0, 256).unwrap(), 0);
    assert_eq!(align_up(512, 256).unwrap(), 512);
}

#[test]
fn align_up_past_u64_max_is_rejected() {
    assert!(align_up(u64::MAX - 2, 4).is_err());
    assert_eq!(align_up(u64::MAX - 3, 4).unwrap(), u64::MAX - 3);
}

#[test]
fn alignment_get_returns_its_bytes() {
    assert_eq!(Alignment::new(64).unwrap().get(), 64);
}

#[test]
fn a_count_converts_to_the_nearest_f32() {
    assert_eq!(ElementCount::new(3).to_f32(), 3.0);
    assert_eq!(ElementCount::new(16_777_217).to_f32(), 16_777_216.0);
}

#[test]
fn an_offset_is_aligned_exactly_at_multiples() {
    let alignment = Alignment::CUDA_MALLOC;

    assert_eq!(alignment.get(), 256);
    assert!(ByteOffset::new(0).is_aligned(alignment));
    assert!(ByteOffset::new(512).is_aligned(alignment));
    assert!(!ByteOffset::new(255).is_aligned(alignment));
    assert!(!ByteOffset::new(257).is_aligned(alignment));
    assert!(!ByteOffset::new(2).is_aligned(alignment));
}
