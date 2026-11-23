//! `Resize`, against a Python port of ONNX's reference implementation.

use crate::builders::{assert_close, assert_exact, f32s, op, tensor};

fn x() -> ptxc_core::HostTensor {
    f32s(&[4], &[1.0, 2.0, 3.0, 4.0])
}

fn resize(attrs: &str, result: &str, input: ptxc_core::HostTensor) -> ptxc_core::HostTensor {
    op(&format!("resize<{attrs}>"), result, &[input])
}

#[test]
fn nearest_rounding_modes() {
    let nearest = |rounding: &str| {
        let attrs = format!(
            "mode=nearest, nearest={rounding}, coordinates=half_pixel, antialias=off, axes=[[6, 1.5]]"
        );
        resize(&attrs, "f32[6]", x())
    };

    assert_close(&nearest("round_prefer_floor"), &[6], &[1.0, 1.0, 2.0, 3.0, 3.0, 4.0]);
    assert_close(&nearest("round_prefer_ceil"), &[6], &[1.0, 2.0, 2.0, 3.0, 4.0, 4.0]);
    assert_close(&nearest("floor"), &[6], &[1.0, 1.0, 2.0, 2.0, 3.0, 4.0]);
    assert_close(&nearest("ceil"), &[6], &[1.0, 2.0, 3.0, 3.0, 4.0, 4.0]);
}

#[test]
fn linear_interpolation_under_each_coordinate_transform() {
    let linear = |coordinates: &str, size: u64, scale: &str| {
        let attrs = format!(
            "mode=linear, coordinates={coordinates}, antialias=off, axes=[[{size}, {scale}]]"
        );
        resize(&attrs, &format!("f32[{size}]"), x())
    };

    assert_close(
        &linear("half_pixel", 8, "2.0"),
        &[8],
        &[1.0, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.0],
    );
    assert_close(&linear("align_corners", 2, "0.6"), &[2], &[1.0, 3.142_857_142_857_143]);
    assert_close(
        &linear("asymmetric", 6, "1.5"),
        &[6],
        &[1.0, 1.666_666_666_666_666_5, 2.333_333_333_333_333, 3.0, 3.666_666_666_666_666_5, 4.0],
    );
    assert_close(&linear("pytorch_half_pixel", 1, "0.25"), &[1], &[1.0]);
    assert_close(
        &linear("half_pixel_symmetric", 2, "0.6"),
        &[2],
        &[1.666_666_666_666_666_5, 3.333_333_333_333_333],
    );
}

#[test]
fn cubic_interpolation_with_and_without_outside_taps() {
    let cubic = |a: &str, outside: &str| {
        let attrs = format!(
            "mode=cubic, cubic_a={a}, exclude_outside={outside}, coordinates=half_pixel, antialias=off, axes=[[8, 2.0]]"
        );
        resize(&attrs, "f32[8]", x())
    };

    assert_close(
        &cubic("-0.75", "include"),
        &[8],
        &[
            0.894_531_25,
            1.191_406_25,
            1.667_968_75,
            2.296_875,
            2.703_125,
            3.332_031_25,
            3.808_593_75,
            4.105_468_75,
        ],
    );
    assert_close(
        &cubic("-0.5", "exclude"),
        &[8],
        &[
            0.911_764_705_882_352_8,
            1.167_883_211_678_832,
            1.709_923_664_122_137_4,
            2.25,
            2.75,
            3.290_076_335_877_862_4,
            3.832_116_788_321_168,
            4.088_235_294_117_647,
        ],
    );
}

#[test]
fn crop_and_resize_extrapolates_outside_the_input() {
    let attrs = "mode=linear, coordinates=tf_crop_and_resize, roi=[[0.4, 1.6]], extrapolation=-9.0, antialias=off, axes=[[3, 1.0]]";

    assert_close(&resize(attrs, "f32[3]", x()), &[3], &[2.2, 3.4, -9.0]);
}

#[test]
fn antialias_stretches_the_filter_when_downsampling() {
    let eight = f32s(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let attrs = "mode=linear, coordinates=half_pixel, antialias=on, axes=[[4, 0.5]]";

    assert_close(&resize(attrs, "f32[4]", eight), &[4], &[1.625, 3.5, 5.5, 7.375]);
}

#[test]
fn every_axis_interpolates_as_a_tensor_product() {
    let square = f32s(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let attrs = "mode=linear, coordinates=half_pixel, antialias=off, axes=[[4, 2.0], [3, 1.5]]";

    assert_close(
        &resize(attrs, "f32[4, 3]", square),
        &[4, 3],
        &[1.0, 1.5, 2.0, 1.5, 2.0, 2.5, 2.5, 3.0, 3.5, 3.0, 3.5, 4.0],
    );
}

#[test]
fn integer_results_truncate_and_saturate() {
    let linear = "mode=linear, coordinates=half_pixel, antialias=off, axes=[[4, 2.0]]";
    let cubic = "mode=cubic, cubic_a=-0.75, exclude_outside=include, coordinates=half_pixel, antialias=off, axes=[[8, 2.0]]";

    assert_exact(&resize(linear, "u8[4]", tensor(&[2], vec![0_u8, 10])), &[4], &[0, 2, 7, 10]);
    assert_exact(
        &resize(cubic, "u8[8]", tensor(&[4], vec![0_u8, 100, 0, 100])),
        &[8],
        &[0, 26, 87, 84, 15, 12, 73, 110],
    );
}

#[test]
fn unit_width_outputs_and_upsampling_under_pytorch_half_pixel() {
    let linear = |coordinates: &str, size: u64, scale: &str| {
        let attrs = format!(
            "mode=linear, coordinates={coordinates}, antialias=off, axes=[[{size}, {scale}]]"
        );
        resize(&attrs, &format!("f32[{size}]"), x())
    };
    let crop = "mode=linear, coordinates=tf_crop_and_resize, roi=[[0.2, 0.8]], extrapolation=-9.0, antialias=off, axes=[[1, 0.25]]";

    assert_close(
        &linear("pytorch_half_pixel", 6, "1.5"),
        &[6],
        &[1.0, 1.5, 2.166_666_666_666_667, 2.833_333_333_333_333_5, 3.5, 4.0],
    );
    assert_close(&linear("align_corners", 1, "0.25"), &[1], &[1.0]);
    assert_close(&resize(crop, "f32[1]", x()), &[1], &[2.5]);
}

#[test]
fn downsampling_filters_with_and_without_antialias() {
    let eight = || f32s(&[8], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let cubic = |antialias: &str| {
        let attrs = format!(
            "mode=cubic, cubic_a=-0.75, exclude_outside=include, coordinates=half_pixel, antialias={antialias}, axes=[[4, 0.5]]"
        );
        resize(&attrs, "f32[4]", eight())
    };
    let linear = "mode=linear, coordinates=half_pixel, antialias=on, axes=[[6, 0.75]]";

    assert_close(&cubic("off"), &[4], &[1.406_25, 3.5, 5.5, 7.593_75]);
    assert_close(&cubic("on"), &[4], &[1.472_656_25, 3.482_421_875, 5.517_578_125, 7.527_343_75]);
    assert_close(
        &resize(linear, "f32[6]", eight()),
        &[6],
        &[
            1.272_727_272_727_272_7,
            2.5,
            3.818_181_818_181_819,
            5.181_818_181_818_182_5,
            6.5,
            7.727_272_727_272_727_5,
        ],
    );
}
