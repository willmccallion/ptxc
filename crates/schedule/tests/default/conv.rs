//! Convolutions tiled as implicit GEMMs, on CUDA cores or, where the
//! target has an `mma` for their types, on tensor cores: the schedule the
//! rule builds, and that it computes exactly what the serial schedule does,
//! in either channel layout; grouped ones stay with the reduction rule.

use crate::matmul::tiled_and_serial;
use proptest::prelude::*;
use ptxc_loop::print_schedule;
use ptxc_schedule::default_schedule;
use ptxc_target::{SmVersion, preset};
use ptxc_testkit::convolution::{Channels, Conv, ConvTypes, conv, conv_inputs, typed_conv};
use ptxc_testkit::golden;
use ptxc_testkit::tensorized::exact_inputs;
use std::sync::Arc;

const PADDED: Conv = Conv {
    batch: 2,
    size: 7,
    channels: 3,
    features: 5,
    kernel: 3,
    stride: 2,
    dilation: 1,
    pad: 1,
    groups: 1,
};

#[test]
fn a_convolution_reindexes_its_operands_and_tiles_its_fused_loops() {
    let compute = conv(PADDED, Channels::First);
    let target = preset(SmVersion::SM_89).unwrap();

    let schedule = default_schedule(Arc::clone(&compute), &target).unwrap();

    golden::check("conv_simt.txt", &print_schedule(&schedule)).unwrap();
    let [tiled, serial] = tiled_and_serial(&compute, &conv_inputs(&compute));
    assert_eq!(tiled, serial);
}

#[test]
fn an_f16_convolution_tiles_its_implicit_gemm_on_tensor_cores() {
    let target = preset(SmVersion::SM_89).unwrap();
    let instruction = *target.mma().first().unwrap();
    let compute = typed_conv(PADDED, Channels::Last, ConvTypes::of(instruction));

    let schedule = default_schedule(Arc::clone(&compute), &target).unwrap();

    assert!(schedule.lower().multiplies_on_tensor_cores());
    golden::check("conv_mma.txt", &print_schedule(&schedule)).unwrap();
    let [tiled, serial] = tiled_and_serial(&compute, &exact_inputs(&compute));
    assert_eq!(tiled, serial);
}

#[test]
fn a_convolution_of_one_output_position_still_multiplies_its_input_on_the_left() {
    // `n`, `p` and `q` each have one value, so the input's index reads none
    // of the tile's row loops, though its staged tile steps with them.
    let single = Conv { batch: 1, size: 1, channels: 1, features: 1, ..PADDED };
    let target = preset(SmVersion::SM_89).unwrap();
    let instruction = *target.mma().first().unwrap();
    let compute = typed_conv(single, Channels::First, ConvTypes::of(instruction));

    let kernel = default_schedule(Arc::clone(&compute), &target).unwrap().lower();

    assert!(kernel.multiplies_on_tensor_cores());
    let [tiled, serial] = tiled_and_serial(&compute, &exact_inputs(&compute));
    assert_eq!(tiled, serial);
}

fn conv_shape() -> impl Strategy<Value = Conv> {
    let channels = (1_u64..=2, 1_u64..=3, 1_u64..=3);
    let window = (1_u64..=3, 1_u64..=2, 1_u64..=2, 0_u64..=2);
    (1_u64..=2, 1_u64..=8, channels, window).prop_filter_map(
        "a window the padded input holds",
        |(batch, size, (groups, channels, per_group), (kernel, stride, dilation, pad))| {
            let geometry = Conv {
                batch,
                size,
                channels,
                features: groups * per_group,
                kernel,
                stride,
                dilation,
                pad,
                groups,
            };
            (size + 2 * pad > dilation * (kernel - 1)).then_some(geometry)
        },
    )
}

proptest! {
    #![proptest_config(crate::cases(64))]

    /// A one-group convolution reindexes into an implicit GEMM and a
    /// grouped one does not; either sums each output in the serial order,
    /// so the results match exactly.
    #[test]
    fn a_tiled_convolution_matches_the_serial_schedule_exactly(
        geometry in conv_shape(),
        channels_last in any::<bool>(),
    ) {
        let channels = if channels_last { Channels::Last } else { Channels::First };
        let compute = conv(geometry, channels);
        let target = preset(SmVersion::SM_89).unwrap();

        let schedule = default_schedule(Arc::clone(&compute), &target).unwrap();
        let [tiled, serial] = tiled_and_serial(&compute, &conv_inputs(&compute));

        let reindexed = print_schedule(&schedule).contains("reindex");
        prop_assert_eq!(reindexed, geometry.groups == 1);
        prop_assert_eq!(tiled, serial);
    }

    /// Every `mma` type multiplies a one-group convolution on tensor
    /// cores, exactly as the serial schedule sums it.
    #[test]
    fn a_tensorized_convolution_matches_the_serial_schedule_exactly(
        geometry in conv_shape().prop_map(|geometry| Conv { groups: 1, ..geometry }),
        channels_last in any::<bool>(),
        instruction in 0_usize..6,
    ) {
        let target = preset(SmVersion::SM_89).unwrap();
        let instruction = target.mma()[instruction % target.mma().len()];
        let channels = if channels_last { Channels::Last } else { Channels::First };
        let compute = typed_conv(geometry, channels, ConvTypes::of(instruction));

        let kernel = default_schedule(Arc::clone(&compute), &target).unwrap().lower();
        let [tiled, serial] = tiled_and_serial(&compute, &exact_inputs(&compute));

        prop_assert!(kernel.multiplies_on_tensor_cores());
        prop_assert_eq!(tiled, serial);
    }
}
