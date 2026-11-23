//! Reshape and the ops imported as it, transpose, tile, concat, slice,
//! split, pad and trilu.

use crate::builders::{
    FLOAT, INT32, assert_lines, f32_tensor, fixed, i64_tensor, input_names, int, ints, line,
    line_at, model, node, text,
};
use ptxc_onnx_proto::{AttributeProto, TensorProto};

fn shaped(
    op: &str,
    dims: &[i64],
    constants: Vec<TensorProto>,
    attrs: Vec<AttributeProto>,
) -> String {
    let names = input_names("x", &constants);
    line(node(op, &names, &["y"], attrs), vec![fixed("x", FLOAT, dims)], constants)
}

#[test]
fn reshape_copies_zeros_unless_allowed_and_infers_minus_one() {
    let target =
        |values: &[i64]| vec![i64_tensor("s", &[i64::try_from(values.len()).unwrap()], values)];

    assert_lines(&[
        (
            shaped("Reshape", &[2, 3, 4], target(&[0, -1]), vec![]),
            "reshape<shape=[2, 12]> %x : f32[2, 12]",
        ),
        (
            shaped("Reshape", &[2, 3, 4], target(&[-1, 6]), vec![]),
            "reshape<shape=[4, 6]> %x : f32[4, 6]",
        ),
        (
            shaped("Reshape", &[0, 3], target(&[3, 0]), vec![int("allowzero", 1)]),
            "reshape<shape=[3, 0]> %x : f32[3, 0]",
        ),
    ]);
}

#[test]
fn squeeze_unsqueeze_and_flatten_become_reshape() {
    assert_lines(&[
        (shaped("Squeeze", &[1, 3, 1], vec![], vec![]), "reshape<shape=[3]> %x : f32[3]"),
        (
            shaped("Squeeze", &[1, 3, 1], vec![i64_tensor("a", &[1], &[-1])], vec![]),
            "reshape<shape=[1, 3]> %x : f32[1, 3]",
        ),
        (
            shaped("Unsqueeze", &[3], vec![i64_tensor("a", &[2], &[0, -1])], vec![]),
            "reshape<shape=[1, 3, 1]> %x : f32[1, 3, 1]",
        ),
        (shaped("Flatten", &[2, 3, 4], vec![], vec![]), "reshape<shape=[2, 12]> %x : f32[2, 12]"),
        (
            shaped("Flatten", &[2, 3, 4], vec![], vec![int("axis", -1)]),
            "reshape<shape=[6, 4]> %x : f32[6, 4]",
        ),
        (
            shaped("Flatten", &[2, 3, 4], vec![], vec![int("axis", 0)]),
            "reshape<shape=[1, 24]> %x : f32[1, 24]",
        ),
    ]);
}

#[test]
fn expand_transpose_tile_and_concat() {
    let concat = line(
        node("Concat", &["x", "z"], &["y"], vec![int("axis", -1)]),
        vec![fixed("x", FLOAT, &[2, 1]), fixed("z", FLOAT, &[2, 3])],
        vec![],
    );

    assert_lines(&[
        (
            shaped("Expand", &[3, 1], vec![i64_tensor("s", &[3], &[2, 1, 4])], vec![]),
            "expand<shape=[2, 1, 4]> %x : f32[2, 3, 4]",
        ),
        (
            shaped("Transpose", &[2, 3, 4], vec![], vec![]),
            "transpose<permutation=[2, 1, 0]> %x : f32[4, 3, 2]",
        ),
        (
            shaped("Transpose", &[2, 3, 4], vec![], vec![ints("perm", &[2, 0, 1])]),
            "transpose<permutation=[2, 0, 1]> %x : f32[4, 2, 3]",
        ),
        (
            shaped("Tile", &[2, 3], vec![i64_tensor("r", &[2], &[3, 1])], vec![]),
            "tile<repeats=[3, 1]> %x : f32[6, 3]",
        ),
        (concat, "concat<axis=1> %x, %z : f32[2, 4]"),
    ]);
}

fn slice(dims: &[i64], starts: i64, ends: i64, steps: i64) -> String {
    let bounds = |name: &str, value: i64| i64_tensor(name, &[1], &[value]);
    let constants =
        vec![bounds("b", starts), bounds("e", ends), bounds("a", 0), bounds("t", steps)];
    shaped("Slice", dims, constants, vec![])
}

#[test]
fn slices_clamp_like_numpy_in_either_direction() {
    let int32_bounds = ["b", "e"].map(|name| TensorProto {
        name: Some(name.to_owned()),
        data_type: Some(INT32 as i32),
        dims: vec![1],
        int32_data: vec![if name == "b" { 1 } else { 3 }],
        ..TensorProto::default()
    });

    assert_lines(&[
        (slice(&[5], -1, -1000, -1), "slice<axes=[[0, 4, -1, 5]]> %x : f32[5]"),
        (slice(&[5], 1, 100, 1), "slice<axes=[[0, 1, 1, 4]]> %x : f32[4]"),
        (slice(&[5], -100, 2, 2), "slice<axes=[[0, 0, 2, 1]]> %x : f32[1]"),
        (slice(&[5], 3, 1, 1), "slice<axes=[[0, 0, 1, 0]]> %x : f32[0]"),
        (slice(&[5], 0, 5, 3), "slice<axes=[[0, 0, 3, 2]]> %x : f32[2]"),
        (slice(&[5], 100, 0, -2), "slice<axes=[[0, 4, -2, 2]]> %x : f32[2]"),
        (
            shaped("Slice", &[5], int32_bounds.to_vec(), vec![]),
            "slice<axes=[[0, 1, 1, 2]]> %x : f32[2]",
        ),
    ]);
}

#[test]
fn split_sizes_come_from_the_input_the_count_or_the_outputs() {
    let x = || vec![fixed("x", FLOAT, &[7])];
    let given = line(
        node("Split", &["x", "s"], &["a", "b"], vec![]),
        x(),
        vec![i64_tensor("s", &[2], &[2, 5])],
    );
    let counted = line_at(
        18,
        node("Split", &["x"], &["a", "b", "c"], vec![int("num_outputs", 3)]),
        x(),
        vec![],
    );
    let by_outputs = line(node("Split", &["x"], &["a", "b"], vec![int("axis", -1)]), x(), vec![]);

    assert_lines(&[
        (given, "split<axis=0, sizes=[2, 5]> %x : f32[2], f32[5]"),
        (counted, "split<axis=0, sizes=[3, 3, 1]> %x : f32[3], f32[3], f32[1]"),
        (by_outputs, "split<axis=0, sizes=[4, 3]> %x : f32[4], f32[3]"),
    ]);
}

#[test]
fn trilu_keeps_the_upper_or_lower_part_from_diagonal_k() {
    assert_lines(&[
        (shaped("Trilu", &[3, 3], vec![], vec![]), "trilu<part=upper, diagonal=0> %x : f32[3, 3]"),
        (
            shaped("Trilu", &[3, 3], vec![], vec![int("upper", 0)]),
            "trilu<part=lower, diagonal=0> %x : f32[3, 3]",
        ),
        (
            shaped("Trilu", &[3, 3], vec![i64_tensor("k", &[], &[-1])], vec![]),
            "trilu<part=upper, diagonal=-1> %x : f32[3, 3]",
        ),
    ]);
}

#[test]
fn constant_pad_gets_a_zero_value() {
    let pads = i64_tensor("p", &[4], &[1, 0, 0, 2]);
    let pad = text(&model(
        vec![node("Pad", &["x", "p"], &["y"], vec![])],
        vec![fixed("x", FLOAT, &[2, 2])],
        vec![pads],
        vec![],
    ));

    assert!(
        pad.contains("pad<mode=constant, pads=[[1, 0], [0, 2]]>")
            && pad.contains("const f32[] [0.0]"),
        "{pad}"
    );
}

#[test]
fn a_one_element_pad_value_becomes_a_scalar_even_when_computed() {
    let nodes = vec![
        node("CastLike", &["fill", "x"], &["fill_cast"], vec![]),
        node("Pad", &["x", "p", "fill_cast"], &["y"], vec![]),
    ];
    let constants = vec![i64_tensor("p", &[2], &[0, 1]), f32_tensor("fill", &[1], &[-1.0])];

    let graph = text(&model(nodes, vec![fixed("x", FLOAT, &[2])], constants, vec![]));

    assert!(graph.contains("const f32[] [-1.0]"), "{graph}");
}

fn pad_text(pads: &[i64], mode: &str) -> Result<String, ptxc_onnx::ImportError> {
    let model = model(
        vec![node("Pad", &["x", "p"], &["y"], vec![crate::builders::string("mode", mode)])],
        vec![fixed("x", FLOAT, &[4])],
        vec![i64_tensor("p", &[2], pads)],
        vec![],
    );
    crate::builders::import(&model).map(|graph| ptxc_graph::text::print(&graph))
}

#[test]
fn negative_pad_widths_alone_become_a_slice() {
    let graph = pad_text(&[-1, -2], "constant").unwrap();

    assert!(graph.contains("%y = slice<axes=[[0, 1, 1, 1]]> %x"), "{graph}");
    assert!(!graph.contains("pad<"), "{graph}");
}

#[test]
fn mixed_pad_widths_slice_then_pad() {
    let graph = pad_text(&[-1, 2], "edge").unwrap();

    let slice = graph.find("slice<axes=[[0, 1, 1, 3]]> %x").unwrap_or_else(|| panic!("{graph}"));
    let pad = graph.find("%y = pad<mode=edge, pads=[[0, 2]]>").unwrap_or_else(|| panic!("{graph}"));
    assert!(slice < pad, "{graph}");
}

#[test]
fn negative_widths_with_reflect_or_wrap_are_unsupported() {
    for mode in ["reflect", "wrap"] {
        let error = pad_text(&[-1, 1], mode).unwrap_err();

        assert_eq!(
            error.unsupported(),
            Some(&ptxc_onnx::Unsupported::Attribute {
                op: "Pad".to_owned(),
                attribute: "negative pads".to_owned()
            })
        );
    }
}

#[test]
fn removing_more_than_the_axis_is_invalid() {
    let error = pad_text(&[-3, -2], "constant").unwrap_err();

    assert!(matches!(error.kind(), ptxc_onnx::ImportErrorKind::Invalid { .. }), "{error}");
}

#[test]
fn crops_and_growth_on_opposite_ends_both_apply() {
    let end_crop = pad_text(&[0, -2], "constant").unwrap();
    let begin_growth = pad_text(&[2, -1], "edge").unwrap();

    assert!(end_crop.contains("%y = slice<axes=[[0, 0, 1, 2]]> %x"), "{end_crop}");
    assert!(begin_growth.contains("slice<axes=[[0, 0, 1, 3]]> %x"), "{begin_growth}");
    assert!(begin_growth.contains("%y = pad<mode=edge, pads=[[2, 0]]>"), "{begin_growth}");
}

#[test]
fn a_pad_value_with_several_elements_is_left_to_the_type_rule() {
    let nodes = vec![node("Pad", &["x", "p", "v"], &["y"], vec![])];
    let constants = vec![i64_tensor("p", &[2], &[0, 1]), f32_tensor("v", &[2], &[1.0, 2.0])];

    let error =
        crate::builders::import(&model(nodes, vec![fixed("x", FLOAT, &[2])], constants, vec![]))
            .unwrap_err();

    assert!(matches!(error.kind(), ptxc_onnx::ImportErrorKind::Node(_)), "{error}");
}

#[test]
fn a_minus_one_that_cannot_divide_the_rest_is_invalid() {
    let infer = |dims: &[i64], target: &[i64], attrs| {
        let nodes = vec![node("Reshape", &["x", "s"], &["y"], attrs)];
        let constants = vec![i64_tensor("s", &[i64::try_from(target.len()).unwrap()], target)];
        crate::builders::import(&model(nodes, vec![fixed("x", FLOAT, dims)], constants, vec![]))
            .unwrap_err()
            .to_string()
    };

    assert!(infer(&[2, 3, 4], &[-1, 5], vec![]).ends_with("Reshape cannot infer its -1"));
    assert!(
        infer(&[0, 3], &[0, -1], vec![int("allowzero", 1)])
            .ends_with("Reshape cannot infer its -1")
    );
}
