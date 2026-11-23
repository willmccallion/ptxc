//! `Attention`, against a NumPy port of ONNX's reference implementation.
//! Q has two heads sharing K and V's one head.

use crate::builders::{assert_close, f32s, i64s, run, tensor};
use ptxc_core::HostTensor;

fn q() -> HostTensor {
    f32s(&[1, 2, 2, 2], &[-0.7, -0.33, 0.04, 0.41, 0.78, -0.55, -0.18, 0.19])
}

fn k() -> HostTensor {
    f32s(&[1, 1, 3, 2], &[-0.3, 0.07, 0.44, 0.81, -0.52, -0.15])
}

fn v() -> HostTensor {
    f32s(&[1, 1, 3, 2], &[0.1, 0.47, 0.84, -0.49, -0.12, 0.25])
}

/// The default scale, `1 / sqrt(2)`.
const SCALE: &str = "0.70710678118654752";

/// A one-op graph over operands `%o0`, `%o1`, … with `results`' types.
fn attention(attrs: &str, operands: &[HostTensor], results: &[&str]) -> Vec<HostTensor> {
    let params: Vec<String> = operands
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let ty = ptxc_graph::TensorType::new(
                t.dtype(),
                t.shape().clone(),
                ptxc_graph::Layout::RowMajor,
            );
            format!("%o{i}: {}", ptxc_graph::text::format_type(&ty))
        })
        .collect();
    let names: Vec<String> = (0..operands.len()).map(|i| format!("%o{i}")).collect();
    let outputs: Vec<String> = (0..results.len()).map(|i| format!("%r{i}")).collect();
    let text = format!(
        "graph({}) {{\n  {} = attention<{attrs}> {} : {}\n  return {}\n}}\n",
        params.join(", "),
        outputs.join(", "),
        names.join(", "),
        results.join(", "),
        outputs.join(", ")
    );
    run(&text, operands)
}

fn heads(mask: &str, rest: &str) -> String {
    format!(
        "layout=heads, mask={mask}, causality=full, past=absent, present=absent, nonpad_lengths=absent, {rest}"
    )
}

const Y: &str = "f32[1, 2, 2, 2]";

#[test]
fn grouped_heads_share_keys_and_values() {
    let y =
        attention(&heads("none", &format!("scale={SCALE}, precision=f32")), &[q(), k(), v()], &[Y]);

    assert_close(
        &y[0],
        &[1, 2, 2, 2],
        &[
            0.162_910_141_997_430_06,
            0.173_223_347_277_797_45,
            0.328_695_639_766_824_75,
            0.025_955_581_597_656_728,
            0.301_236_102_325_285_86,
            0.051_264_058_535_808_9,
            0.274_526_635_834_835_4,
            0.075_587_215_263_714_33,
        ],
    );
}

#[test]
fn softcap_bounds_the_scaled_scores() {
    let attrs = heads("none", "qk=softcapped, scale=2.0, softcap=0.3, precision=f32");

    let outputs = attention(&attrs, &[q(), k(), v()], &[Y, "f32[1, 2, 2, 3]"]);

    assert_close(
        &outputs[0],
        &[1, 2, 2, 2],
        &[
            0.174_653_505_333_294,
            0.171_616_990_711_452_14,
            0.348_357_384_041_624_5,
            0.013_399_900_044_515_384,
            0.294_827_168_802_682_67,
            0.055_700_775_897_466_5,
            0.276_067_931_607_331_45,
            0.074_198_001_810_196_92,
        ],
    );
    assert_close(
        &outputs[1],
        &[1, 2, 2, 3],
        &[
            0.254_147_450_284_861_96,
            -0.299_720_346_295_143_6,
            0.297_590_132_447_843_77,
            0.033_262_681_885_447_9,
            0.294_388_230_849_088_5,
            -0.149_856_071_524_933_77,
            -0.284_551_505_389_054_27,
            -0.177_845_420_079_344_84,
            -0.292_031_000_126_107_2,
            0.126_240_716_213_413_37,
            0.138_162_842_647_908_7,
            0.122_597_621_553_316_23,
        ],
    );
}

#[test]
fn qk_outputs_capture_the_scores_before_and_after_the_softmax() {
    let qk = |stage: &str| {
        attention(
            &heads("none", &format!("qk={stage}, scale={SCALE}, precision=f32")),
            &[q(), k(), v()],
            &[Y, "f32[1, 2, 2, 3]"],
        )
    };

    assert_close(
        &qk("matmul")[1],
        &[1, 2, 2, 3],
        &[
            0.132_158_257_403_765_77,
            -0.406_798_531_216_620_74,
            0.292_388_654_020_637_35,
            0.011_808_683_245_815_342,
            0.247_275_241_380_935_54,
            -0.058_194_888_091_652_8,
            -0.192_686_597_873_334_2,
            -0.072_337_023_715_383_61,
            -0.228_466_201_001_373_52,
            0.047_588_286_373_854_68,
            0.052_820_876_554_634_98,
            0.046_032_651_455_244_315,
        ],
    );
    assert_close(
        &qk("softmax")[1],
        &[1, 2, 2, 3],
        &[
            0.362_694_969_157_176_14,
            0.211_580_467_482_136_85,
            0.425_724_563_360_687_1,
            0.312_706_234_655_617_25,
            0.395_729_445_981_863_5,
            0.291_564_319_362_519_3,
            0.323_337_465_013_604_7,
            0.364_689_437_523_221_7,
            0.311_973_097_463_175_53,
            0.332_923_628_799_315_53,
            0.334_670_247_394_777,
            0.332_406_123_805_907_45,
        ],
    );
}

#[test]
fn a_short_bool_mask_hides_false_keys_and_every_key_past_its_end() {
    let mask = tensor(&[2, 2], vec![true, false, true, true]);
    let attrs = heads("boolean", &format!("qk=masked, scale={SCALE}, precision=f32"));

    let outputs = attention(&attrs, &[q(), k(), v(), mask], &[Y, "f32[1, 2, 2, 3]"]);

    let inf = f64::NEG_INFINITY;
    assert_close(
        &outputs[0],
        &[1, 2, 2, 2],
        &[
            0.1,
            0.47,
            0.513_361_153_355_614_8,
            -0.066_252_307_055_932_75,
            0.1,
            0.47,
            0.470_968_026_974_730_5,
            -0.011_255_818_778_028_838,
        ],
    );
    assert_close(
        &outputs[1],
        &[1, 2, 2, 3],
        &[
            0.132_158_257_403_765_77,
            inf,
            inf,
            0.011_808_683_245_815_342,
            0.247_275_241_380_935_54,
            inf,
            -0.192_686_597_873_334_2,
            inf,
            inf,
            0.047_588_286_373_854_68,
            0.052_820_876_554_634_98,
            inf,
        ],
    );
}

#[test]
fn an_additive_mask_adds_to_the_scores() {
    let mask = f32s(&[2, 3], &[0.5, -1.0, 0.0, 0.0, 0.0, -0.25]);

    let y = attention(
        &heads("additive", &format!("scale={SCALE}, precision=f32")),
        &[q(), k(), v(), mask],
        &[Y],
    );

    assert_close(
        &y[0],
        &[1, 2, 2, 2],
        &[
            0.067_263_494_973_524_73,
            0.317_139_897_787_000_8,
            0.359_628_717_958_530_46,
            0.010_509_959_000_440_383,
            0.131_295_678_522_506_98,
            0.268_382_687_472_361_27,
            0.305_837_612_886_041_55,
            0.061_745_222_849_795_23,
        ],
    );
}

#[test]
fn causal_masking_counts_positions_from_the_end_of_the_past() {
    let past_key = f32s(&[1, 1, 1, 2], &[-0.5, -0.13]);
    let past_value = f32s(&[1, 1, 1, 2], &[-0.1, 0.27]);
    let attrs = format!(
        "layout=heads, mask=none, causality=causal, past=present, present=present, nonpad_lengths=absent, scale={SCALE}, precision=f32"
    );

    let outputs = attention(
        &attrs,
        &[q(), k(), v(), past_key, past_value],
        &[Y, "f32[1, 1, 4, 2]", "f32[1, 1, 4, 2]"],
    );

    assert_close(
        &outputs[0],
        &[1, 2, 2, 2],
        &[
            -0.007_270_349_200_837_527,
            0.362_729_650_799_162_4,
            0.333_719_591_533_417_24,
            0.032_229_458_989_044_14,
            0.001_626_202_222_427_874_7,
            0.371_626_202_222_427_8,
            0.281_156_839_118_236_1,
            0.082_244_164_639_030_71,
        ],
    );
    assert_close(&outputs[1], &[1, 1, 4, 2], &[-0.5, -0.13, -0.3, 0.07, 0.44, 0.81, -0.52, -0.15]);
    assert_close(&outputs[2], &[1, 1, 4, 2], &[-0.1, 0.27, 0.1, 0.47, 0.84, -0.49, -0.12, 0.25]);
}

#[test]
fn non_pad_lengths_hide_each_samples_padding() {
    let attrs = format!(
        "layout=heads, mask=none, causality=full, past=absent, present=absent, nonpad_lengths=present, scale={SCALE}, precision=f32"
    );

    let y = attention(&attrs, &[q(), k(), v(), i64s(&[1], &[2])], &[Y]);

    assert_close(
        &y[0],
        &[1, 2, 2, 2],
        &[
            0.372_638_416_946_811_55,
            0.116_306_918_555_487_64,
            0.513_361_153_355_614_8,
            -0.066_252_307_055_932_75,
            0.492_237_836_590_611_16,
            -0.038_849_085_306_738_924,
            0.470_968_026_974_730_5,
            -0.011_255_818_778_028_838,
        ],
    );
}

#[test]
fn packed_inputs_regroup_into_heads_and_back() {
    let q = f32s(&[1, 2, 4], &[-0.7, -0.33, 0.78, -0.55, 0.04, 0.41, -0.18, 0.19]);
    let k = f32s(&[1, 3, 2], &[-0.3, 0.07, 0.44, 0.81, -0.52, -0.15]);
    let v = f32s(&[1, 3, 2], &[0.1, 0.47, 0.84, -0.49, -0.12, 0.25]);
    let attrs = format!(
        "layout=packed, q_heads=2, kv_heads=1, mask=none, causality=full, past=absent, present=absent, nonpad_lengths=absent, scale={SCALE}, precision=f32"
    );

    let y = attention(&attrs, &[q, k, v], &["f32[1, 2, 4]"]);

    assert_close(
        &y[0],
        &[1, 2, 4],
        &[
            0.162_910_141_997_430_06,
            0.173_223_347_277_797_45,
            0.301_236_102_325_285_86,
            0.051_264_058_535_808_9,
            0.328_695_639_766_824_75,
            0.025_955_581_597_656_728,
            0.274_526_635_834_835_4,
            0.075_587_215_263_714_33,
        ],
    );
}

fn q2() -> HostTensor {
    f32s(
        &[2, 2, 2, 2],
        &[
            -0.6, -0.23, 0.14, 0.51, 0.88, -0.45, -0.08, 0.29, 0.66, -0.67, -0.3, 0.07, 0.44, 0.81,
            -0.52, -0.15,
        ],
    )
}

fn k2() -> HostTensor {
    f32s(
        &[2, 1, 3, 2],
        &[-0.2, 0.17, 0.54, -0.79, -0.42, -0.05, 0.32, 0.69, -0.64, -0.27, 0.1, 0.47],
    )
}

fn v2() -> HostTensor {
    f32s(
        &[2, 1, 3, 2],
        &[0.2, 0.57, -0.76, -0.39, -0.02, 0.35, 0.72, -0.61, -0.24, 0.13, 0.5, 0.87],
    )
}

const BATCH_Y: [f64; 16] = [
    -0.157_694_894_474_825_93,
    0.212_305_105_525_174_12,
    -0.148_797_481_418_385_56,
    0.221_202_518_581_614_55,
    -0.355_693_880_293_093_46,
    0.014_306_119_706_906_515,
    -0.151_682_532_781_375_94,
    0.218_317_467_218_624_16,
    0.325_473_364_165_164_7,
    0.130_382_953_695_122_07,
    0.298_763_897_674_714_3,
    0.138_409_631_987_319_12,
    0.457_024_070_594_235_5,
    0.071_914_277_924_071_18,
    0.243_294_182_138_099_25,
    0.152_055_663_973_646_5,
];

#[test]
fn every_sample_of_a_batch_attends_to_its_own_keys() {
    let y = attention(
        &heads("none", &format!("scale={SCALE}, precision=f32")),
        &[q2(), k2(), v2()],
        &["f32[2, 2, 2, 2]"],
    );

    assert_close(&y[0], &[2, 2, 2, 2], &BATCH_Y);
}

#[test]
fn packed_batches_regroup_per_sample() {
    let attrs = format!(
        "layout=packed, q_heads=2, kv_heads=1, mask=none, causality=full, past=absent, present=absent, nonpad_lengths=absent, scale={SCALE}, precision=f32"
    );
    let packed_q = f32s(
        &[2, 2, 4],
        &[
            -0.6, -0.23, 0.88, -0.45, 0.14, 0.51, -0.08, 0.29, 0.66, -0.67, 0.44, 0.81, -0.3, 0.07,
            -0.52, -0.15,
        ],
    );
    let packed_k = f32s(
        &[2, 3, 2],
        &[-0.2, 0.17, 0.54, -0.79, -0.42, -0.05, 0.32, 0.69, -0.64, -0.27, 0.1, 0.47],
    );
    let packed_v = f32s(
        &[2, 3, 2],
        &[0.2, 0.57, -0.76, -0.39, -0.02, 0.35, 0.72, -0.61, -0.24, 0.13, 0.5, 0.87],
    );

    let y = attention(&attrs, &[packed_q, packed_k, packed_v], &["f32[2, 2, 4]"]);

    let order = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15];
    assert_close(&y[0], &[2, 2, 4], &order.map(|i| BATCH_Y[i]));
}

#[test]
fn masks_broadcast_from_the_right_over_heads() {
    let attrs = heads("additive", &format!("scale={SCALE}, precision=f32"));
    let full =
        f32s(&[1, 2, 2, 3], &[0.0, -1.0, 0.5, 0.25, 0.0, -0.5, -0.75, 0.0, 0.0, 0.0, 1.0, 0.0]);
    let shared = f32s(&[1, 2, 3], &[0.0, -1.0, 0.5, 0.25, 0.0, -0.5]);

    let per_head = attention(&attrs, &[q(), k(), v(), full], &[Y]);
    let broadcast = attention(&attrs, &[q(), k(), v(), shared], &[Y]);

    assert_close(
        &per_head[0],
        &[1, 2, 2, 2],
        &[
            0.015_251_396_953_872_155,
            0.269_427_127_345_390_9,
            0.360_687_533_339_896_5,
            0.040_056_610_822_212_6,
            0.342_629_600_510_453_9,
            -0.034_868_326_422_855_06,
            0.480_982_469_021_171_14,
            -0.130_910_185_262_305_08,
        ],
    );
    assert_close(
        &broadcast[0],
        &[1, 2, 2, 2],
        &[
            0.015_251_396_953_872_155,
            0.269_427_127_345_390_9,
            0.360_687_533_339_896_5,
            0.040_056_610_822_212_6,
            0.085_719_305_019_672_26,
            0.221_039_481_997_205_16,
            0.310_943_889_270_120_6,
            0.090_615_179_561_376_81,
        ],
    );
}

#[test]
fn large_scores_do_not_overflow_and_a_zero_softcap_is_none() {
    let large = attention(&heads("none", "scale=4000.0, precision=f32"), &[q(), k(), v()], &[Y]);
    let uncapped = attention(
        &heads("none", &format!("scale={SCALE}, softcap=0.0, precision=f32")),
        &[q(), k(), v()],
        &[Y],
    );

    assert_close(&large[0], &[1, 2, 2, 2], &[-0.12, 0.25, 0.84, -0.49, 0.84, -0.49, 0.84, -0.49]);
    assert_close(
        &uncapped[0],
        &[1, 2, 2, 2],
        &[
            0.162_910_141_997_430_06,
            0.173_223_347_277_797_45,
            0.328_695_639_766_824_75,
            0.025_955_581_597_656_728,
            0.301_236_102_325_285_86,
            0.051_264_058_535_808_9,
            0.274_526_635_834_835_4,
            0.075_587_215_263_714_33,
        ],
    );
}
