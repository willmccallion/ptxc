//! `Gather` and `GatherElements`.

use crate::builders::{FLOAT, INT64, assert_lines, fixed, int, line, node};

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
