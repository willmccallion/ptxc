//! `Resize`: modes, coordinate transforms, and sizes from scales or sizes.

use crate::builders::{
    FLOAT, assert_lines, f32_tensor, fixed, float, i64_tensor, import, int, ints, line, model,
    node, op_line, string,
};
use ptxc_onnx::ImportErrorKind;
use ptxc_onnx_proto::AttributeProto;

/// `x: f32[1, 1, 4, 6]` resized by `scales` on its last two axes.
fn by_scales(scales: &[f32], attrs: Vec<AttributeProto>) -> String {
    let constants = vec![f32_tensor("scales", &[4], &[1.0, 1.0, scales[0], scales[1]])];
    line(
        node("Resize", &["x", "", "scales"], &["y"], attrs),
        vec![fixed("x", FLOAT, &[1, 1, 4, 6])],
        constants,
    )
}

#[test]
fn modes_and_nearest_rounding() {
    let scaled = |attrs| by_scales(&[2.0, 2.0], attrs);

    assert_lines(&[
        (
            scaled(vec![]),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            scaled(vec![string("nearest_mode", "round_prefer_ceil")]),
            "resize<mode=nearest, nearest=round_prefer_ceil, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            scaled(vec![string("nearest_mode", "floor")]),
            "resize<mode=nearest, nearest=floor, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            scaled(vec![string("nearest_mode", "ceil")]),
            "resize<mode=nearest, nearest=ceil, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            scaled(vec![string("mode", "linear"), int("antialias", 1)]),
            "resize<mode=linear, coordinates=half_pixel, antialias=on, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            scaled(vec![string("mode", "cubic")]),
            "resize<mode=cubic, cubic_a=-0.75, exclude_outside=include, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            scaled(vec![
                string("mode", "cubic"),
                float("cubic_coeff_a", -0.5),
                int("exclude_outside", 1),
            ]),
            "resize<mode=cubic, cubic_a=-0.5, exclude_outside=exclude, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
    ]);
}

#[test]
fn coordinate_transforms() {
    let transform =
        |mode: &str| by_scales(&[2.0, 2.0], vec![string("coordinate_transformation_mode", mode)]);

    assert_lines(&[
        (
            transform("half_pixel_symmetric"),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=half_pixel_symmetric, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            transform("pytorch_half_pixel"),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=pytorch_half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            transform("align_corners"),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=align_corners, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
        (
            transform("asymmetric"),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=asymmetric, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [12, 2.0]]> %x : f32[1, 1, 8, 12]",
        ),
    ]);
}

#[test]
fn sizes_from_scales_are_floored_and_aspect_policies_share_one_scale() {
    let sizes = |policy: &str| {
        let attrs = vec![ints("axes", &[2, 3]), string("keep_aspect_ratio_policy", policy)];
        let constants = vec![i64_tensor("sizes", &[2], &[2, 2])];
        line(
            node("Resize", &["x", "", "", "sizes"], &["y"], attrs),
            vec![fixed("x", FLOAT, &[1, 1, 4, 6])],
            constants,
        )
    };

    assert_lines(&[
        (
            by_scales(&[0.6, 1.5], vec![]),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [2, 0.6], [9, 1.5]]> %x : f32[1, 1, 2, 9]",
        ),
        (
            sizes("stretch"),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [2, 0.5], [2, 0.33333334]]> %x : f32[1, 1, 2, 2]",
        ),
        (
            sizes("not_smaller"),
            "resize<mode=nearest, nearest=round_prefer_floor, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [2, 0.5], [3, 0.5]]> %x : f32[1, 1, 2, 3]",
        ),
    ]);
}

#[test]
fn scales_with_an_aspect_policy_are_invalid() {
    let constants = vec![f32_tensor("scales", &[4], &[1.0, 1.0, 2.0, 2.0])];
    let attrs = vec![string("keep_aspect_ratio_policy", "not_larger")];
    let model = model(
        vec![node("Resize", &["x", "", "scales"], &["y"], attrs)],
        vec![fixed("x", FLOAT, &[1, 1, 4, 6])],
        constants,
        vec![],
    );

    assert!(matches!(import(&model).unwrap_err().kind(), ImportErrorKind::Invalid { .. }));
}

#[test]
fn empty_scales_fall_back_to_sizes() {
    let constants = vec![f32_tensor("scales", &[0], &[]), i64_tensor("sizes", &[4], &[1, 1, 8, 3])];
    let resized = line(
        node("Resize", &["x", "", "scales", "sizes"], &["y"], vec![]),
        vec![fixed("x", FLOAT, &[1, 1, 4, 6])],
        constants,
    );

    assert_lines(&[(
        resized,
        "resize<mode=nearest, nearest=round_prefer_floor, coordinates=half_pixel, antialias=off, axes=[[1, 1.0], [1, 1.0], [8, 2.0], [3, 0.5]]> %x : f32[1, 1, 8, 3]",
    )]);
}

#[test]
fn resize_to_sizes_keeping_the_aspect_ratio_uses_one_scale() {
    let sizes = i64_tensor("sizes", &[2], &[2, 2]);
    let attrs = vec![ints("axes", &[2, 3]), string("keep_aspect_ratio_policy", "not_larger")];

    let line = op_line(&model(
        vec![node("Resize", &["x", "", "", "sizes"], &["y"], attrs)],
        vec![fixed("x", FLOAT, &[1, 1, 4, 6])],
        vec![sizes],
        vec![],
    ));

    assert!(line.contains("axes=[[1, 1.0], [1, 1.0], [1, 0.33333334], [2, 0.33333334]]"), "{line}");
}

#[test]
fn a_crop_region_with_axes_covers_only_those_axes() {
    let roi = f32_tensor("roi", &[4], &[0.0, 0.25, 1.0, 0.75]);
    let sizes = i64_tensor("sizes", &[2], &[2, 2]);
    let attrs =
        vec![ints("axes", &[2, 3]), string("coordinate_transformation_mode", "tf_crop_and_resize")];
    let node = node("Resize", &["x", "roi", "", "sizes"], &["y"], attrs);

    let line = op_line(&model(
        vec![node],
        vec![fixed("x", FLOAT, &[1, 1, 4, 4])],
        vec![roi, sizes],
        vec![],
    ));

    assert!(line.contains("roi=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.25, 0.75]]"), "{line}");
}

#[test]
fn a_crop_region_on_one_axis_has_one_start_and_one_end() {
    let constants = vec![f32_tensor("roi", &[2], &[0.25, 0.75]), i64_tensor("sizes", &[1], &[3])];
    let attrs =
        vec![ints("axes", &[3]), string("coordinate_transformation_mode", "tf_crop_and_resize")];

    let cropped = line(
        node("Resize", &["x", "roi", "", "sizes"], &["y"], attrs),
        vec![fixed("x", FLOAT, &[1, 1, 4, 6])],
        constants,
    );

    assert_lines(&[(
        cropped,
        "resize<mode=nearest, nearest=round_prefer_floor, coordinates=tf_crop_and_resize, roi=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.25, 0.75]], extrapolation=0.0, antialias=off, axes=[[1, 1.0], [1, 1.0], [4, 1.0], [3, 0.5]]> %x : f32[1, 1, 4, 3]",
    )]);
}
