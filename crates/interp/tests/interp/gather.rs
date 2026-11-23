//! `Gather`, `GatherElements` and `GatherND`, with negative and out-of-range
//! indices.

use crate::builders::{assert_close, assert_exact, f32s, i64s, op, tensor, try_run};
use ptxc_interp::InterpError;

/// `[[1, 2, 3], [4, 5, 6]]`.
fn data() -> ptxc_core::HostTensor {
    f32s(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

#[test]
fn gather_selects_slices_along_its_axis() {
    let rows = op("gather<axis=0>", "f32[2, 3]", &[data(), i64s(&[2], &[1, 0])]);
    let columns = op("gather<axis=1>", "f32[2, 2, 2]", &[data(), i64s(&[2, 2], &[0, -1, 2, 1])]);
    let scalar = op("gather<axis=1>", "f32[2]", &[data(), tensor(&[], vec![-2_i32])]);

    assert_close(&rows, &[2, 3], &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
    assert_close(&columns, &[2, 2, 2], &[1.0, 3.0, 3.0, 2.0, 4.0, 6.0, 6.0, 5.0]);
    assert_close(&scalar, &[2], &[2.0, 5.0]);
}

#[test]
fn gather_elements_replaces_one_coordinate() {
    let picked =
        op("gather_elements<axis=1>", "f32[2, 2]", &[data(), i64s(&[2, 2], &[2, 0, -1, 1])]);
    let ints = op(
        "gather_elements<axis=0>",
        "i64[1, 3]",
        &[i64s(&[2, 3], &[1, 2, 3, 4, 5, 6]), i64s(&[1, 3], &[1, 0, 1])],
    );

    assert_close(&picked, &[2, 2], &[3.0, 1.0, 6.0, 5.0]);
    assert_exact(&ints, &[1, 3], &[4, 2, 6]);
}

#[test]
fn an_index_outside_its_axis_is_an_error() {
    for index in [3, -4] {
        let text = "graph(%d: f32[2, 3], %i: i64[1]) {\n  %y = gather<axis=1> %d, %i : f32[2, 1]\n  return %y\n}\n";

        let error = try_run(text, &[data(), i64s(&[1], &[index])]).unwrap_err();

        assert!(
            matches!(error, InterpError::IndexOutOfRange { extent: 3, .. }),
            "{index}: {error}"
        );
    }
}

#[test]
fn gather_nd_selects_elements_or_slices_by_index_tuples() {
    let elements =
        op("gather_nd<batch_dims=0>", "f32[2]", &[data(), i64s(&[2, 2], &[0, 2, 1, -3])]);
    let rows = op("gather_nd<batch_dims=0>", "f32[2, 3]", &[data(), i64s(&[2, 1], &[1, -2])]);

    assert_close(&elements, &[2], &[3.0, 4.0]);
    assert_close(&rows, &[2, 3], &[4.0, 5.0, 6.0, 1.0, 2.0, 3.0]);
}

#[test]
fn gather_nd_indexes_each_batch_entry_separately() {
    let cube = f32s(&[2, 2, 2], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);

    let picked = op("gather_nd<batch_dims=1>", "f32[2, 2]", &[cube, i64s(&[2, 1], &[1, 0])]);

    assert_close(&picked, &[2, 2], &[2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn a_gather_nd_index_outside_its_axis_is_an_error() {
    let text = "graph(%d: f32[2, 3], %i: i64[1, 2]) {\n  %y = gather_nd<batch_dims=0> %d, %i : f32[1]\n  return %y\n}\n";

    let error = try_run(text, &[data(), i64s(&[1, 2], &[0, 3])]).unwrap_err();

    assert!(matches!(error, InterpError::IndexOutOfRange { index: 3, extent: 3 }), "{error}");
}
