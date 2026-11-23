//! Max, average and global pooling, on cases worked out by hand.

use crate::builders::{assert_close, assert_exact, f32s, op, tensor};

fn x() -> ptxc_core::HostTensor {
    f32s(&[1, 1, 5], &[1.0, 5.0, 2.0, 4.0, 3.0])
}

#[test]
fn max_pool_rounds_partial_windows_down_or_up() {
    assert_close(
        &op("max_pool<rounding=floor, axes=[[2, 2, 1, 0, 0]]>", "f32[1, 1, 2]", &[x()]),
        &[1, 1, 2],
        &[5.0, 4.0],
    );
    assert_close(
        &op("max_pool<rounding=ceil, axes=[[2, 2, 1, 0, 0]]>", "f32[1, 1, 3]", &[x()]),
        &[1, 1, 3],
        &[5.0, 4.0, 3.0],
    );
}

#[test]
fn max_pool_ignores_padding_and_dilates() {
    let padded = op("max_pool<rounding=floor, axes=[[3, 1, 1, 1, 1]]>", "f32[1, 1, 5]", &[x()]);
    let dilated = op("max_pool<rounding=floor, axes=[[2, 1, 2, 0, 0]]>", "f32[1, 1, 3]", &[x()]);
    let negative = tensor(&[1, 1, 3], vec![-3_i8, -1, -2]);
    let ints = op("max_pool<rounding=floor, axes=[[2, 1, 1, 1, 0]]>", "i8[1, 1, 3]", &[negative]);

    assert_close(&padded, &[1, 1, 5], &[5.0, 5.0, 5.0, 4.0, 4.0]);
    assert_close(&dilated, &[1, 1, 3], &[2.0, 5.0, 3.0]);
    assert_exact(&ints, &[1, 1, 3], &[-3, -1, -1]);
}

#[test]
fn max_pool_propagates_nan() {
    let with_nan = f32s(&[1, 1, 3], &[1.0, f32::NAN, 2.0]);

    assert_close(
        &op("max_pool<rounding=floor, axes=[[2, 1, 1, 0, 0]]>", "f32[1, 1, 2]", &[with_nan]),
        &[1, 1, 2],
        &[f64::NAN, f64::NAN],
    );
}

#[test]
fn average_pool_counts_padding_only_when_asked() {
    let x = f32s(&[1, 1, 4], &[1.0, 2.0, 3.0, 4.0]);

    let excluded = op(
        "average_pool<pads=exclude, rounding=floor, axes=[[3, 1, 1, 1, 1]]>",
        "f32[1, 1, 4]",
        std::slice::from_ref(&x),
    );
    let included = op(
        "average_pool<pads=include, rounding=floor, axes=[[3, 1, 1, 1, 1]]>",
        "f32[1, 1, 4]",
        &[x],
    );

    assert_close(&excluded, &[1, 1, 4], &[1.5, 2.0, 3.0, 3.5]);
    assert_close(&included, &[1, 1, 4], &[1.0, 2.0, 3.0, 7.0 / 3.0]);
}

#[test]
fn a_ceil_mode_window_past_the_padding_counts_only_what_it_covers() {
    let x = f32s(&[1, 1, 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);

    let included = op(
        "average_pool<pads=include, rounding=ceil, axes=[[2, 2, 1, 0, 0]]>",
        "f32[1, 1, 3]",
        &[x],
    );

    assert_close(&included, &[1, 1, 3], &[1.5, 3.5, 5.0]);
}

#[test]
fn global_pools_reduce_each_plane() {
    let x = || f32s(&[1, 2, 2], &[1.0, 2.0, 3.0, 9.0]);

    assert_close(&op("global_max_pool", "f32[1, 2, 1]", &[x()]), &[1, 2, 1], &[2.0, 9.0]);
    assert_close(&op("global_average_pool", "f32[1, 2, 1]", &[x()]), &[1, 2, 1], &[1.5, 6.0]);
}

#[test]
fn every_sample_and_channel_pools_separately() {
    let x = || f32s(&[2, 2, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    assert_close(
        &op("max_pool<rounding=floor, axes=[[2, 1, 1, 0, 0]]>", "f32[2, 2, 1]", &[x()]),
        &[2, 2, 1],
        &[2.0, 4.0, 6.0, 8.0],
    );
    assert_close(
        &op(
            "average_pool<pads=exclude, rounding=floor, axes=[[2, 1, 1, 0, 0]]>",
            "f32[2, 2, 1]",
            &[x()],
        ),
        &[2, 2, 1],
        &[1.5, 3.5, 5.5, 7.5],
    );
}
