//! `Conv`: windows, groups and bias, on cases worked out by hand.

use crate::builders::{assert_close, f32s, op};

#[test]
fn a_kernel_slides_over_the_input() {
    let x = f32s(&[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let w = f32s(&[1, 1, 3], &[1.0, 0.0, -1.0]);

    assert_close(
        &op("conv<groups=1, axes=[[1, 1, 0, 0]]>", "f32[1, 1, 3]", &[x, w]),
        &[1, 1, 3],
        &[-2.0, -2.0, -2.0],
    );
}

#[test]
fn padding_stride_and_dilation_place_the_taps() {
    let x = f32s(&[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let w = f32s(&[1, 1, 2], &[1.0, 1.0]);

    let y = op("conv<groups=1, axes=[[2, 2, 1, 2]]>", "f32[1, 1, 3]", &[x, w]);

    assert_close(&y, &[1, 1, 3], &[2.0, 6.0, 4.0]);
}

#[test]
fn channels_sum_and_the_bias_adds_per_filter() {
    let x = f32s(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 0.0, 1.0, 1.0, 0.0]);
    let w = f32s(&[2, 2, 1], &[1.0, 1.0, 1.0, -1.0]);
    let bias = f32s(&[2], &[10.0, 20.0]);

    let y = op("conv<groups=1, axes=[[1, 1, 0, 0]]>", "f32[2, 2, 2]", &[x, w, bias]);

    assert_close(&y, &[2, 2, 2], &[14.0, 16.0, 18.0, 18.0, 11.0, 11.0, 19.0, 21.0]);
}

#[test]
fn groups_convolve_their_own_channels() {
    let x = f32s(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let w = f32s(&[2, 1, 1], &[2.0, 3.0]);

    let y = op("conv<groups=2, axes=[[1, 1, 0, 0]]>", "f32[1, 2, 3]", &[x, w]);

    assert_close(&y, &[1, 2, 3], &[2.0, 4.0, 6.0, 12.0, 15.0, 18.0]);
}

#[test]
fn two_spatial_axes_window_independently() {
    let x = f32s(&[1, 1, 3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let w = f32s(&[1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);

    let plain = op(
        "conv<groups=1, axes=[[1, 1, 0, 0], [1, 1, 0, 0]]>",
        "f32[1, 1, 2, 2]",
        &[x.clone(), w.clone()],
    );
    let padded =
        op("conv<groups=1, axes=[[1, 1, 1, 0], [2, 1, 0, 1]]>", "f32[1, 1, 3, 2]", &[x, w]);

    assert_close(&plain, &[1, 1, 2, 2], &[6.0, 8.0, 12.0, 14.0]);
    assert_close(&padded, &[1, 1, 3, 2], &[2.0, 0.0, 6.0, 3.0, 12.0, 6.0]);
}

#[test]
fn each_group_sums_its_own_several_channels() {
    let x = f32s(&[1, 4, 1], &[1.0, 2.0, 3.0, 4.0]);
    let w = f32s(&[2, 2, 1], &[1.0, 10.0, 1.0, 10.0]);

    let y = op("conv<groups=2, axes=[[1, 1, 0, 0]]>", "f32[1, 2, 1]", &[x, w]);

    assert_close(&y, &[1, 2, 1], &[21.0, 43.0]);
}

#[test]
fn every_channel_of_a_group_has_its_own_kernel() {
    let x = f32s(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let w = f32s(&[1, 2, 2], &[1.0, 10.0, 100.0, 1000.0]);

    assert_close(
        &op("conv<groups=1, axes=[[1, 1, 0, 0]]>", "f32[1, 1, 1]", &[x, w]),
        &[1, 1, 1],
        &[4321.0],
    );
}

#[test]
fn a_padded_tap_multiplies_zero_by_its_weight_as_onnx_pads() {
    // The first output's left tap reads padding: 0 * inf is NaN.
    let x = f32s(&[1, 1, 2], &[1.0, 2.0]);
    let w = f32s(&[1, 1, 2], &[f32::INFINITY, 1.0]);

    let y = op("conv<groups=1, axes=[[1, 1, 1, 0]]>", "f32[1, 1, 2]", &[x, w]);

    assert_close(&y, &[1, 1, 2], &[f64::NAN, f64::INFINITY]);
}
