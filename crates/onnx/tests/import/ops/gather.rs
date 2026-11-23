//! `Gather`, `GatherElements` and `GatherND`.

use crate::builders::{FLOAT, INT64, assert_lines, fixed, import, int, line, model, node};
use ptxc_onnx::ImportErrorKind;

#[test]
fn gathers_default_to_axis_zero_and_count_negative_axes_from_the_end() {
    let gather = |op: &str, attrs, indices: &[i64]| {
        let inputs = vec![fixed("x", FLOAT, &[2, 3]), fixed("i", INT64, indices)];
        line(node(op, &["x", "i"], &["y"], attrs), inputs, vec![])
    };

    assert_lines(&[
        (gather("Gather", vec![], &[4]), "gather<axis=0> %x, %i : f32[4, 3]"),
        (gather("Gather", vec![int("axis", -1)], &[4]), "gather<axis=1> %x, %i : f32[2, 4]"),
        (
            gather("GatherElements", vec![int("axis", 1)], &[2, 2]),
            "gather_elements<axis=1> %x, %i : f32[2, 2]",
        ),
    ]);
}

#[test]
fn gather_nd_reads_its_batch_dims_defaulting_to_none() {
    let gather_nd = |attrs, indices: &[i64]| {
        let inputs = vec![fixed("x", FLOAT, &[2, 3]), fixed("i", INT64, indices)];
        line(node("GatherND", &["x", "i"], &["y"], attrs), inputs, vec![])
    };

    assert_lines(&[
        (gather_nd(vec![], &[4, 1]), "gather_nd<batch_dims=0> %x, %i : f32[4, 3]"),
        (gather_nd(vec![int("batch_dims", 1)], &[2, 1]), "gather_nd<batch_dims=1> %x, %i : f32[2]"),
    ]);
}

#[test]
fn negative_gather_nd_batch_dims_are_invalid() {
    let inputs = vec![fixed("x", FLOAT, &[2, 3]), fixed("i", INT64, &[2, 1])];
    let gather_nd = node("GatherND", &["x", "i"], &["y"], vec![int("batch_dims", -1)]);

    let error = import(&model(vec![gather_nd], inputs, vec![], vec![])).unwrap_err();

    assert!(
        matches!(error.kind(), ImportErrorKind::Invalid { field: "batch_dims", .. }),
        "{error}"
    );
}
