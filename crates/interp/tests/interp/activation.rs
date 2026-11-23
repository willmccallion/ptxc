//! Activations, `Clip` and `PRelu`, against Python's float64 evaluation of
//! ONNX's formulas.

use crate::builders::{assert_close, assert_exact, f32s, i64s, op, tensor};
use ptxc_core::float::F16;

fn x() -> ptxc_core::HostTensor {
    f32s(&[5], &[-2.0, -0.5, 0.0, 0.5, 3.0])
}

fn activation(name: &str) -> ptxc_core::HostTensor {
    op(name, "f32[5]", &[x()])
}

#[test]
fn parameterless_activations() {
    assert_close(&activation("relu"), &[5], &[0.0, 0.0, 0.0, 0.5, 3.0]);
    assert_close(
        &activation("sigmoid"),
        &[5],
        &[
            0.119_202_922_022_117_55,
            0.377_540_668_798_145_4,
            0.5,
            0.622_459_331_201_854_6,
            0.952_574_126_822_433_4,
        ],
    );
    assert_close(
        &activation("tanh"),
        &[5],
        &[
            -0.964_027_580_075_816_9,
            -0.462_117_157_260_009_74,
            0.0,
            0.462_117_157_260_009_74,
            0.995_054_753_686_730_5,
        ],
    );
    assert_close(
        &activation("softplus"),
        &[5],
        &[
            0.126_928_011_042_972_6,
            0.474_076_984_180_106_7,
            std::f64::consts::LN_2,
            0.974_076_984_180_106_7,
            3.048_587_351_573_742,
        ],
    );
    assert_close(&activation("softsign"), &[5], &[-2.0 / 3.0, -1.0 / 3.0, 0.0, 1.0 / 3.0, 0.75]);
    assert_close(
        &activation("hard_swish"),
        &[5],
        &[-1.0 / 3.0, -0.208_333_333_333_333_34, 0.0, 0.291_666_666_666_666_7, 3.0],
    );
    assert_close(
        &activation("mish"),
        &[5],
        &[
            -0.252_501_482_695_709_1,
            -0.220_743_774_651_73,
            0.0,
            0.375_245_211_304_895_13,
            2.986_535_004_967_957_5,
        ],
    );
}

#[test]
fn gelu_exactly_and_with_the_tanh_approximation() {
    assert_close(
        &activation("gelu<approximate=none>"),
        &[5],
        &[
            -0.045_500_263_896_358_42,
            -0.154_268_769_362_993_44,
            0.0,
            0.345_731_230_637_006_56,
            2.995_950_305_905_11,
        ],
    );
    assert_close(
        &activation("gelu<approximate=tanh>"),
        &[5],
        &[
            -0.045_402_305_912_224_94,
            -0.154_285_990_174_856_06,
            0.0,
            0.345_714_009_825_143_94,
            2.996_362_607_918_227,
        ],
    );
}

#[test]
fn parameterised_activations() {
    assert_close(&activation("leaky_relu<alpha=0.1>"), &[5], &[-0.2, -0.05, 0.0, 0.5, 3.0]);
    assert_close(
        &activation("elu<alpha=0.5>"),
        &[5],
        &[-0.432_332_358_381_693_65, -0.196_734_670_143_683_3, 0.0, 0.5, 3.0],
    );
    assert_close(
        &activation("celu<alpha=2.0>"),
        &[5],
        &[-1.264_241_117_657_115_3, -0.442_398_433_857_190_24, 0.0, 0.5, 3.0],
    );
    assert_close(&activation("thresholded_relu<alpha=0.5>"), &[5], &[0.0, 0.0, 0.0, 0.0, 3.0]);
    assert_close(
        &activation("selu<alpha=1.5, gamma=2.0>"),
        &[5],
        &[-2.593_994_150_290_162, -1.180_408_020_862_099_7, 0.0, 1.0, 6.0],
    );
    assert_close(
        &activation("hard_sigmoid<alpha=0.3, beta=0.4>"),
        &[5],
        &[0.0, 0.25, 0.4, 0.55, 1.0],
    );
}

#[test]
fn relu_passes_nan_and_softplus_does_not_overflow() {
    let extremes = || f32s(&[2], &[f32::NAN, 100.0]);

    assert_close(&op("relu", "f32[2]", &[extremes()]), &[2], &[f64::NAN, 100.0]);
    assert_close(&op("softplus", "f32[2]", &[extremes()]), &[2], &[f64::NAN, 100.0]);
    assert_close(
        &op("sigmoid", "f32[2]", &[f32s(&[2], &[-100.0, 100.0])]),
        &[2],
        &[3.720_075_976_020_836e-44, 1.0],
    );
}

#[test]
fn relu_takes_signed_integers() {
    let x = i64s(&[3], &[-3, 0, 5]);

    assert_exact(&op("relu", "i64[3]", &[x]), &[3], &[0, 0, 5]);
}

#[test]
fn celu_propagates_nan_as_numpys_maximum_and_minimum_do() {
    let nan = f32s(&[1], &[f32::NAN]);

    assert_close(&op("celu<alpha=1.0>", "f32[1]", &[nan]), &[1], &[f64::NAN]);
}

#[test]
fn narrow_inputs_widen_and_round_back() {
    let half = tensor(&[2], vec![F16::from_f32(-1.0), F16::from_f32(1.0)]);

    assert_close(&op("sigmoid", "f16[2]", &[half]), &[2], &[0.269_042_968_75, 0.730_957_031_25]);
}

#[test]
fn clip_takes_the_upper_bound_when_the_bounds_cross_and_passes_nan() {
    let x = f32s(&[4], &[-5.0, 0.5, 9.0, f32::NAN]);
    let bounds = |low: f32, high: f32| [x.clone(), f32s(&[], &[low]), f32s(&[], &[high])];

    assert_close(&op("clip", "f32[4]", &bounds(0.0, 1.0)), &[4], &[0.0, 0.5, 1.0, f64::NAN]);
    assert_close(&op("clip", "f32[4]", &bounds(2.0, 1.0)), &[4], &[1.0, 1.0, 1.0, f64::NAN]);
    let ints = [i64s(&[3], &[-5, 3, 9]), i64s(&[], &[0]), i64s(&[], &[4])];
    assert_exact(&op("clip", "i64[3]", &ints), &[3], &[0, 3, 4]);
}

#[test]
fn prelu_scales_negatives_by_a_broadcast_slope() {
    let x = f32s(&[2, 2], &[-1.0, 2.0, -3.0, 4.0]);
    let slope = f32s(&[2], &[0.5, 0.25]);

    assert_close(&op("prelu", "f32[2, 2]", &[x, slope]), &[2, 2], &[-0.5, 2.0, -1.5, 4.0]);
    let ints = [i64s(&[3], &[-4, 0, 5]), i64s(&[1], &[3])];
    assert_exact(&op("prelu", "i64[3]", &ints), &[3], &[-12, 0, 5]);
}

#[test]
fn bf16_results_round_to_bf16() {
    let brain = tensor(
        &[2],
        vec![ptxc_core::float::Bf16::from_f32(-1.0), ptxc_core::float::Bf16::from_f32(1.0)],
    );

    assert_close(&op("sigmoid", "bf16[2]", &[brain]), &[2], &[0.269_531_25, 0.730_468_75]);
}
