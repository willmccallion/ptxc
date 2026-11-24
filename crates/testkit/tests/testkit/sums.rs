use ptxc_core::{DType, Dim, ElementCount, HostTensor, Shape};
use ptxc_loop::{LoweredKernel, Schedule, TileOrder};
use ptxc_target::{MmaShape, SmVersion, Target, preset};
use ptxc_testkit::compare::Tolerance;
use ptxc_testkit::sums::SumRounding;
use ptxc_testkit::tensorized::{ONE_TILE, Operands, exact_inputs, product, tensorized};
use std::sync::Arc;

fn target() -> Target {
    preset(SmVersion::SM_89).unwrap()
}

/// A 16 by 8 by `depth` f16 product tensorized and serial, with its
/// inputs.
fn product_kernels(depth: u64) -> ([LoweredKernel; 2], Vec<HostTensor>) {
    let shape = MmaShape { m: Dim::new(16), n: Dim::new(8), k: Dim::new(16) };
    let instruction = target().mma_instruction(shape, DType::F16, DType::F32).unwrap();
    let extents = [16, 8, depth];
    let compute = product(instruction, extents, Operands::matmul(extents, TileOrder::RowMajor));
    let tiled = tensorized(&compute, ONE_TILE, instruction).validate(&target()).unwrap().lower();
    let serial = Schedule::new(Arc::clone(&compute)).validate(&target()).unwrap().lower();
    ([tiled, serial], exact_inputs(&compute))
}

#[test]
fn only_a_stage_multiplying_on_tensor_cores_rounds_its_sums_its_own_way() {
    let ([tiled, serial], inputs) = product_kernels(48);

    let untouched = SumRounding::Interpreted.after(&serial, &inputs);
    let tensor_cores = SumRounding::Interpreted.after(&tiled, &inputs);

    assert_eq!(untouched, SumRounding::Interpreted);
    assert_eq!(tensor_cores, SumRounding::TensorCores { longest: ElementCount::new(48) });
}

#[test]
fn tensor_core_stages_keep_the_longest_axis_of_any_of_their_arguments() {
    let ([short, _], short_inputs) = product_kernels(32);
    let ([long, _], long_inputs) = product_kernels(64);
    let ([_, serial], serial_inputs) = product_kernels(96);

    let rounding = SumRounding::Interpreted
        .after(&long, &long_inputs)
        .after(&short, &short_inputs)
        .after(&serial, &serial_inputs);

    assert_eq!(rounding, SumRounding::TensorCores { longest: ElementCount::new(64) });
}

#[test]
fn interpreted_sums_match_exactly_and_tensor_core_ones_within_a_widened_tolerance() {
    let longest = ElementCount::new(64);
    let tensor_cores = SumRounding::TensorCores { longest };

    assert_eq!(SumRounding::Interpreted.tolerance(DType::F16), Tolerance::Exact);
    for dtype in [DType::F16, DType::Bf16, DType::F32] {
        let widened = Tolerance::for_dtype(dtype).for_reduction(longest);
        assert_eq!(tensor_cores.tolerance(dtype), widened);
    }
    assert_eq!(tensor_cores.tolerance(DType::I32), Tolerance::Exact);
}

#[test]
fn an_argument_with_no_elements_still_counts_one() {
    let ([tiled, _], _) = product_kernels(16);
    let empty = HostTensor::new(Shape::new(vec![Dim::new(0)]).unwrap(), Vec::<f32>::new().into());

    let rounding = SumRounding::Interpreted.after(&tiled, &[empty.unwrap()]);

    assert_eq!(rounding, SumRounding::TensorCores { longest: ElementCount::new(1) });
}
