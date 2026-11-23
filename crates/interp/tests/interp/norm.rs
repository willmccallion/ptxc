//! Softmax and the normalisations, against Python's float64 evaluation of
//! ONNX's definitions.

use crate::builders::{assert_close, f32s, op, run};

/// `[[1, 2, 4], [-1, 0, 1]]`.
fn x() -> ptxc_core::HostTensor {
    f32s(&[2, 3], &[1.0, 2.0, 4.0, -1.0, 0.0, 1.0])
}

#[test]
fn softmax_normalises_along_its_axis() {
    let along_rows = op("softmax<axis=1>", "f32[2, 3]", &[x()]);
    let along_columns = op("softmax<axis=0>", "f32[2, 3]", &[x()]);
    let log = op("log_softmax<axis=1>", "f32[2, 3]", &[x()]);

    assert_close(
        &along_rows,
        &[2, 3],
        &[
            0.042_010_066_134_066_05,
            0.114_195_199_384_594_49,
            0.843_794_734_481_339_5,
            0.090_030_573_170_380_46,
            0.244_728_471_054_797_64,
            0.665_240_955_774_821_8,
        ],
    );
    assert_close(
        &along_columns,
        &[2, 3],
        &[
            0.880_797_077_977_882_3,
            0.880_797_077_977_882_3,
            0.952_574_126_822_433_4,
            0.119_202_922_022_117_55,
            0.119_202_922_022_117_55,
            0.047_425_873_177_566_79,
        ],
    );
    assert_close(
        &log,
        &[2, 3],
        &[
            -3.169_846_019_556_285_6,
            -2.169_846_019_556_285_6,
            -0.169_846_019_556_285_67,
            -2.407_605_964_444_38,
            -1.407_605_964_444_380_4,
            -0.407_605_964_444_380_46,
        ],
    );
}

#[test]
fn softmax_is_stable_for_large_inputs() {
    let large = f32s(&[2], &[1000.0, 1001.0]);

    assert_close(
        &op("softmax<axis=0>", "f32[2]", &[large]),
        &[2],
        &[0.268_941_421_369_995_1, 0.731_058_578_630_004_9],
    );
}

const LAYER: &str = "graph(%x: f32[2, 3], %s: f32[3], %b: f32[3]) {
  %y, %mean, %inv = layer_norm<axis=1, epsilon=1e-5, stash=f32, outputs=with_statistics> %x, %s, %b : f32[2, 3], f32[2, 1], f32[2, 1]
  return %y, %mean, %inv
}
";

#[test]
fn layer_norm_scales_shifts_and_reports_its_statistics() {
    let outputs = run(LAYER, &[x(), f32s(&[3], &[1.0, 2.0, 0.5]), f32s(&[3], &[0.0, 1.0, -1.0])]);

    assert_close(
        &outputs[0],
        &[2, 3],
        &[
            -1.069_041_531_450_297_7,
            0.465_479_234_274_851_05,
            -0.331_849_042_843_564_2,
            -1.224_735_685_908_390_2,
            1.0,
            -0.387_632_157_045_804_9,
        ],
    );
    assert_close(&outputs[1], &[2, 1], &[2.333_333_333_333_333_5, 0.0]);
    assert_close(&outputs[2], &[2, 1], &[0.801_781_148_587_723_1, 1.224_735_685_908_390_2]);
}

#[test]
fn layer_norm_over_every_axis_broadcasts_a_scalar_scale() {
    let normalized = op(
        "layer_norm<axis=0, epsilon=1e-5, stash=f32, outputs=normalized>",
        "f32[2, 3]",
        &[x(), f32s(&[], &[2.0])],
    );

    assert_close(
        &normalized,
        &[2, 3],
        &[
            -0.211_999_147_239_385_37,
            1.059_995_736_196_926_3,
            3.603_985_503_069_549_3,
            -2.755_988_914_112_009,
            -1.483_994_030_675_697_1,
            -0.211_999_147_239_385_37,
        ],
    );
}

#[test]
fn bf16_statistics_are_stored_as_bf16() {
    let text = LAYER
        .replace("stash=f32", "stash=bf16")
        .replace("f32[2, 1], f32[2, 1]", "bf16[2, 1], bf16[2, 1]");

    let outputs = run(&text, &[x(), f32s(&[3], &[1.0, 2.0, 0.5]), f32s(&[3], &[0.0, 1.0, -1.0])]);

    assert_close(&outputs[1], &[2, 1], &[2.328_125, 0.0]);
    assert_close(&outputs[2], &[2, 1], &[0.800_781_25, 1.226_562_5]);
}

#[test]
fn rms_norm_divides_by_the_root_mean_square() {
    let normalized = op(
        "rms_norm<axis=1, epsilon=0.5, stash=f32>",
        "f32[2, 3]",
        &[x(), f32s(&[3], &[1.0, 2.0, 0.5])],
    );

    assert_close(
        &normalized,
        &[2, 3],
        &[
            0.365_148_371_670_110_7,
            1.460_593_486_680_442_9,
            0.730_296_743_340_221_4,
            -0.925_820_099_772_551_5,
            0.0,
            0.462_910_049_886_275_77,
        ],
    );
}

#[test]
fn batch_norm_uses_the_given_statistics_per_channel() {
    let x = f32s(&[1, 2, 2], &[1.0, 3.0, 2.0, 6.0]);
    let channel = |values: [f32; 2]| f32s(&[2], &values);
    let operands =
        [x, channel([2.0, 0.5]), channel([1.0, -1.0]), channel([2.0, 4.0]), channel([1.0, 4.0])];

    let normalized = op("batch_norm<epsilon=1e-5>", "f32[1, 2, 2]", &operands);

    assert_close(
        &normalized,
        &[1, 2, 2],
        &[
            -0.999_990_000_074_999_5,
            2.999_990_000_074_999_5,
            -1.499_999_375_001_171_8,
            -0.500_000_624_998_828_1,
        ],
    );
}

#[test]
fn instance_norm_normalises_each_sample_and_channel() {
    let x = f32s(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 4.0, 10.0]);

    let normalized = op(
        "instance_norm<epsilon=1e-5>",
        "f32[1, 2, 3]",
        &[x, f32s(&[2], &[1.0, 2.0]), f32s(&[2], &[0.0, 1.0])],
    );

    assert_close(
        &normalized,
        &[1, 2, 3],
        &[
            -1.224_735_685_908_390_2,
            0.0,
            1.224_735_685_908_390_2,
            -0.414_212_678_490_447_2,
            -0.414_212_678_490_447_2,
            3.828_425_356_980_894_4,
        ],
    );
}
