//! The `mma.sync` variants a target supports.

use ptxc_core::{DType, Dim};
use ptxc_target::{MmaInstruction, MmaShape, SmVersion, preset};

const fn shape(m: u64, n: u64, k: u64) -> MmaShape {
    MmaShape { m: Dim::new(m), n: Dim::new(n), k: Dim::new(k) }
}

#[test]
fn sm_89_multiplies_every_tensor_core_type_into_a_wide_accumulator() {
    let target = preset(SmVersion::SM_89).unwrap();

    let listed: Vec<(MmaShape, DType, DType)> = target
        .mma()
        .iter()
        .map(|instruction| (instruction.shape, instruction.multiplicands, instruction.accumulator))
        .collect();

    let expected = [
        (shape(16, 8, 16), DType::F16, DType::F32),
        (shape(16, 8, 16), DType::Bf16, DType::F32),
        (shape(16, 8, 32), DType::F8E4M3Fn, DType::F32),
        (shape(16, 8, 32), DType::F8E5M2, DType::F32),
        (shape(16, 8, 32), DType::I8, DType::I32),
        (shape(16, 8, 32), DType::U8, DType::I32),
    ];
    assert_eq!(listed, expected);
}

#[test]
fn a_listed_variant_is_found_by_its_shape_and_types() {
    let target = preset(SmVersion::SM_89).unwrap();

    let found = target.mma_instruction(shape(16, 8, 16), DType::Bf16, DType::F32);

    let expected = MmaInstruction {
        shape: shape(16, 8, 16),
        multiplicands: DType::Bf16,
        accumulator: DType::F32,
    };
    assert_eq!(found, Some(expected));
}

#[test]
fn another_shape_or_type_pair_is_not() {
    let target = preset(SmVersion::SM_89).unwrap();
    let missing = |shape, multiplicands, accumulator| {
        target.mma_instruction(shape, multiplicands, accumulator).is_none()
    };

    assert!(missing(shape(16, 8, 16), DType::F16, DType::F16));
    assert!(missing(shape(16, 8, 8), DType::F16, DType::F32));
    assert!(missing(shape(16, 8, 16), DType::F8E4M3Fn, DType::F32));
    assert!(missing(shape(16, 8, 32), DType::I8, DType::F32));
    assert!(missing(shape(16, 8, 16), DType::F32, DType::F32));
}
